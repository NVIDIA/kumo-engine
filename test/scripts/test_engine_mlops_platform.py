# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Platform planning and readback guards; these do not qualify live services."""

import builtins
import errno
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

DIRECTORY = Path(__file__).resolve().parents[2] / 'packages/kumo-engine/test'
SPEC = importlib.util.spec_from_file_location(
    'platform_runner', DIRECTORY / 'wheel_mlops_platform.py')
assert SPEC is not None and SPEC.loader is not None
platform = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(platform)


def config(name='snowflake'):
    return json.loads(
        (DIRECTORY / 'mlops_platforms' / f'{name}.json').read_text())


@pytest.mark.parametrize('name', platform.PLATFORMS)
def test_templates_plan_without_importing_provider_code(name, monkeypatch,
                                                        capsys):
    original = builtins.__import__

    def guarded(module, *args, **kwargs):
        if module.split('.')[0] in ('kumo', 'snowflake', 'databricks', 'mlflow',
                                    'boto3'):
            raise AssertionError('Plan mode imported a service client')
        return original(module, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    assert platform.main(
        ['--config',
         str(DIRECTORY / 'mlops_platforms' / f'{name}.json')]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['platform'] == name
    assert len(result['mutations']) == 3


@pytest.mark.parametrize('change', [
    {
        'password': 'do-not-store'
    },
    {
        'database': 'DB;DROP SCHEMA PROD'
    },
    {
        'artifact_root': '@DB.SCHEMA.STAGE'
    },
    {
        'artifact_root': '@DB.SCHEMA.STAGE/../prod'
    },
    {
        'identity': 'active_python_session'
    },
    {
        'environment': 'anything'
    },
    {
        'accelerator': 'auto'
    },
    {
        'model_prefix': 'production/champion'
    },
    {
        'model_prefix': None
    },
])
def test_invalid_snowflake_profiles_are_rejected_before_execution(change):
    with pytest.raises(ValueError):
        platform.validate({**config(), **change})


@pytest.mark.parametrize('change', [
    {
        'identity': 'connection_toml',
        'execution_context': 'client'
    },
    {
        'identity': 'spcs',
        'execution_context': 'snowflake'
    },
    {
        'execution_context': 'client'
    },
    {
        'environment': 'venv'
    },
])
def test_snowflake_accepts_only_notebook_execution(change):
    with pytest.raises(ValueError, match='notebook'):
        platform.validate({**config(), **change})


def test_snowflake_requires_notebook_launch_before_connecting(monkeypatch):
    monkeypatch.delenv(platform.NOTEBOOK_CONTEXT, raising=False)
    instance = platform.Snowflake(config(), 'abcdef123456')
    # The guard must precede any provider import.
    with pytest.raises(platform.PlatformUnavailable, match='notebook'):
        instance.placement()


@pytest.mark.parametrize('name', platform.PLATFORMS)
def test_attempt_resources_are_scoped_and_reconstructible(name, tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config(name)))
    first, second = tmp_path / 'a', tmp_path / 'b'
    first.mkdir()
    second.mkdir()
    platform.prepare(first, path)
    platform.prepare(second, path)
    a, b = platform.backend(first), platform.backend(second)
    assert a.model != b.model and a.experiment != b.experiment
    assert a.artifact_root != b.artifact_root
    assert a.attempt in a.artifact_root and a.attempt.upper() in a.model.upper()
    restored = type(a)(a.config, a.attempt)
    assert restored.model == a.model and restored.artifact_root == a.artifact_root


class Row(dict):
    def as_dict(self):
        return dict(self)


def snowflake_run(monkeypatch, *, status='FINISHED', bundle='bundle',
                  steps=(1, 2)):
    profile = platform.Snowflake(config(), 'abcdef123456')
    calls = []

    def sql(command):
        calls.append(command)
        if command.startswith('SHOW RUNS'):
            values = [
                Row(name='FIT_RUN', metadata=json.dumps({'status': status}))
            ]
        elif command.startswith('SHOW RUN PARAMETERS'):
            values = [Row(name='kumo.bundle_id', value=bundle)]
        elif command.startswith('SHOW RUN METRICS'):
            values = [
                Row(name=name, step=step, value=0.5)
                for name in ('train/loss', 'val/auroc') for step in steps[-1:]
            ]
        else:
            raise AssertionError(command)
        return SimpleNamespace(collect=lambda: values)

    cfg = SimpleNamespace(tracking=SimpleNamespace(session=SimpleNamespace(
        sql=sql)))
    ref = SimpleNamespace(run_id='FIT_RUN')
    profile.history_calls = []

    class ExperimentTracking:
        def __init__(self, session, *, database_name, schema_name):
            assert session is cfg.tracking.session
            assert database_name == profile.config['database']
            assert schema_name == profile.config['schema']

        def set_experiment(self, experiment, **kwargs):
            assert experiment == profile.experiment
            assert kwargs == dict(database_name=profile.config['database'],
                                  schema_name=profile.config['schema'])

        def get_metric_history(self, *, run_name):
            profile.history_calls.append(run_name)
            return SimpleNamespace(collect=lambda: [
                Row(NAME=name, STEP=step, VALUE=0.5) for name in
                ('train/loss', 'val/auroc') for step in reversed(steps)
            ])

    import sys

    monkeypatch.setitem(sys.modules, 'snowflake.ml.experiment',
                        SimpleNamespace(ExperimentTracking=ExperimentTracking))
    profile.registered = lambda *args: None
    return profile, cfg, ref, calls


def test_snowflake_reads_public_history_and_json_status(monkeypatch):
    profile, cfg, ref, calls = snowflake_run(monkeypatch)
    record = profile.record(cfg, ref)
    assert record['params']['kumo.bundle_id'] == 'bundle'
    assert record['history']['val/auroc'] == [(1, 0.5), (2, 0.5)]
    assert record['url'] is None
    assert profile.history_calls == ['FIT_RUN']
    assert all('EXPERIMENT "KUMO_TEST_DB"."YOUR_TEST_SCHEMA".' in c
               for c in calls)
    assert not any('LIKE' in c for c in calls)


@pytest.mark.parametrize('change', [{
    'status': 'FAILED'
}, {
    'bundle': 'wrong'
}, {
    'steps': (2, )
}])
def test_failed_or_incomplete_snowflake_fit_cannot_pass(change, monkeypatch):
    profile, cfg, ref, _ = snowflake_run(monkeypatch, **change)
    result = SimpleNamespace(tracking=ref, id='bundle',
                             registered_model=SimpleNamespace(version='V1'))
    with pytest.raises(AssertionError):
        platform.fit_record(profile, cfg, result)


@pytest.mark.parametrize('cause,expected', [
    (TimeoutError('service timed out'), False),
    (FileNotFoundError('stage absent'), False),
    (PermissionError('wrapper only'), False),
    (PermissionError(errno.EACCES, 'denied'), True),
])
def test_probe_wrapper_does_not_turn_transport_errors_into_denial_proof(
        cause, expected):
    wrapped = PermissionError('MLOps cannot write under artifact_root')
    wrapped.__cause__ = cause
    assert platform.denied_access(wrapped) is expected


def test_provider_denial_codes_are_recognized():
    class SnowDenied(RuntimeError):
        sqlstate = '42501'

    class AwsDenied(RuntimeError):
        response = {'Error': {'Code': 'AccessDenied'}}

    error = SnowDenied('Snowflake denies access')
    assert platform.access_error(error) and platform.denied_access(error)
    error = AwsDenied('AWS denies access')
    assert platform.access_error(error) and platform.denied_access(error)
    wrapped = RuntimeError('Tracking unavailable')
    wrapped.__cause__ = error
    assert platform.access_error(wrapped)
    assert not platform.access_error(AssertionError('model identity wrong'))


def test_execute_needs_a_candidate_and_output_directory():
    with pytest.raises(SystemExit) as result:
        platform.main([
            '--config',
            str(DIRECTORY / 'mlops_platforms/databricks.json'), '--execute'
        ])
    assert result.value.code == 2


@pytest.mark.parametrize('phase', [None, 'negative'])
@pytest.mark.parametrize('notebook', [False, True])
@pytest.mark.parametrize('workspace', [None, 'workspace-123'])
def test_notebook_launch_requires_kernel_and_passes_only_identifiers(
        phase, notebook, workspace, tmp_path, monkeypatch):
    import sys

    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps(config()))
    inline_backend = 'module://matplotlib_inline.backend_inline'
    monkeypatch.setenv('MPLBACKEND', inline_backend)
    if workspace is None:
        monkeypatch.delenv('SNOWFLAKE_WORKSPACE_ID', raising=False)
    else:
        monkeypatch.setenv('SNOWFLAKE_WORKSPACE_ID', workspace)
    identity = ['ACCOUNT', 'USER', 'ROLE']
    session = SimpleNamespace(sql=lambda sql: SimpleNamespace(
        collect=lambda: [[123]] if 'CURRENT_SESSION' in sql else [identity]))
    extension = 'snowflake_notebook_utils.ipython_execution_context_monitor_extension'
    shell = SimpleNamespace(extension_manager=SimpleNamespace(
        loaded={extension} if notebook else set()))
    monkeypatch.setitem(sys.modules, 'IPython',
                        SimpleNamespace(get_ipython=lambda: shell))
    monkeypatch.setitem(sys.modules, 'snowflake.snowpark.context',
                        SimpleNamespace(get_active_session=lambda: session))
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if phase == 'negative':
            output = Path(command[command.index('--output-dir') + 1])
            assert (output / 'platform.json').is_file()
            (output / 'negative.json').write_text(
                json.dumps(dict(status='passed')))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(platform.subprocess, 'run', run)
    args = dict(config=profile, wheel=tmp_path / 'candidate.whl',
                expected_version='1.2.3', output_dir=tmp_path / 'evidence',
                phase=phase)
    if not notebook:
        with pytest.raises(platform.PlatformUnavailable,
                           match='notebook kernel'):
            platform.run_notebook(**args)
        assert not calls
        return
    result = platform.run_notebook(**args)
    assert result == dict(returncode=1,
                          summary=dict(status='passed') if phase else None)
    if phase:
        assert calls[0][0][-2:] == ['--phase', 'negative']
    command, options = calls[0]
    assert command[1] == '-I' and '--execute-platform' in command
    assert json.loads(options['env'][platform.NOTEBOOK_CONTEXT]) == dict(
        session='123', workspace=workspace, notebook_extension=True,
        identity=identity)
    assert platform.NOTEBOOK_CONTEXT not in platform.os.environ
    assert options['env']['MPLBACKEND'] == 'Agg'
    assert platform.os.environ['MPLBACKEND'] == inline_backend


@pytest.mark.parametrize(
    'identity',
    [['ACCOUNT', 'USER', 'ROLE'], ['ACCOUNT', 'SERVICE_USER', 'ROLE']])
def test_notebook_worker_identity_is_checked_before_mlops(
        identity, monkeypatch):
    import sys

    instance = platform.Snowflake(config(), 'abcdef123456')
    monkeypatch.setattr(
        instance, 'placement',
        lambda: dict(notebook=dict(identity=['ACCOUNT', 'USER', 'ROLE'])))
    connector = SimpleNamespace(session=lambda: SimpleNamespace(
        sql=lambda _: SimpleNamespace(collect=lambda: [identity])))
    monkeypatch.setitem(
        sys.modules, 'kumo.engine',
        SimpleNamespace(SnowflakeConnector=SimpleNamespace(
            from_spcs_env=lambda **_: connector)))
    monkeypatch.setitem(sys.modules, 'kumo',
                        SimpleNamespace(engine=sys.modules['kumo.engine']))
    if identity[1] == 'SERVICE_USER':
        with pytest.raises(platform.PlatformUnavailable, match='differs'):
            _ = instance.connector
    else:
        assert instance.connector is connector


def test_snowflake_cli_execution_is_rejected_with_complete_arguments(
        tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        platform.main([
            '--config',
            str(DIRECTORY / 'mlops_platforms/snowflake.json'), '--wheel',
            str(tmp_path / 'candidate.whl'), '--expected-version', '1.2.3',
            '--output-dir',
            str(tmp_path / 'evidence'), '--execute'
        ])
    assert error.value.code == 2
    assert 'requires run_notebook()' in capsys.readouterr().err
    assert not (tmp_path / 'evidence').exists()


@pytest.mark.parametrize('fault',
                         [None, 'model', 'previous', 'metric', 'missing'])
def test_drift_readback_matches_logged_previous_statistics(fault):
    record = dict(
        params={
            'kumo.prediction_id': 'prediction',
            'kumo.bundle_id': 'bundle',
            'kumo.drift_previous': 'previous'
        }, metrics={
            f'drift/{name}_vs_prev': 0.0
            for name in ('psi', 'ks', 'wasserstein')
        })
    if fault == 'model':
        record['params']['kumo.bundle_id'] = 'other'
    elif fault == 'previous':
        record['params']['kumo.drift_previous'] = 'other'
    elif fault == 'metric':
        record['metrics']['drift/psi_vs_prev'] = 0.1
    elif fault == 'missing':
        record['metrics'] = {'drift/psi': 0.0}
    profile = SimpleNamespace(record=lambda *_: record)
    run = SimpleNamespace(id='prediction', bundle_id='bundle')
    drift = SimpleNamespace(tracking='ref', previous_prediction_id='previous',
                            previous_psi=0.0, previous_ks=0.0,
                            previous_wasserstein=0.0)
    if fault:
        with pytest.raises(AssertionError):
            platform.drift_record(profile, None, run, drift)
    else:
        assert platform.drift_record(profile, None, run, drift) is record


@pytest.mark.parametrize('denied', [
    '@KUMO_TEST_DB.YOUR_TEST_SCHEMA.YOUR_TEST_STAGE/other-prefix',
    '@kumo_test_db.your_test_schema.your_test_stage/other-prefix',
])
def test_denied_snowflake_root_must_use_another_stage(denied):
    with pytest.raises(ValueError, match='separate Snowflake stage'):
        platform.validate({**config(), 'denied_artifact_root': denied})


@pytest.mark.parametrize(
    'case', ['denied', 'local-spool', 'missing', 'timeout', 'writable'])
def test_negative_requires_real_permission_denial(case, monkeypatch):
    import sys

    denied = '@KUMO_TEST_DB.YOUR_TEST_SCHEMA.DENIED_STAGE/mlops-validation'
    instance = platform.Snowflake({
        **config(), 'denied_artifact_root': denied
    }, 'abcdef123456')
    instance.connector = object()

    def mlops(root, **kwargs):
        assert root == (instance.connector, denied + '/abcdef123456')
        if case == 'writable':
            return object()
        error = PermissionError(errno.EACCES, 'write denied') if case in (
            'denied', 'local-spool') else PermissionError('Could not write')
        if case == 'denied':

            class ProviderDenied(RuntimeError):
                sqlstate = '42501'

            error.__cause__ = ProviderDenied('Snowflake stage write denied')
        elif case == 'missing':
            error.__cause__ = FileNotFoundError('stage missing')
        elif case == 'timeout':
            error.__cause__ = TimeoutError('provider timed out')
        raise error

    monkeypatch.setitem(sys.modules, 'kumo.engine',
                        SimpleNamespace(MLOps=mlops))
    if case == 'denied':
        assert instance.negative()['denied_artifact_access'] == 'passed'
    elif case == 'writable':
        with pytest.raises(AssertionError, match='was writable'):
            instance.negative()
    else:
        with pytest.raises(platform.PlatformUnavailable,
                           match='without permission-denial'):
            instance.negative()
