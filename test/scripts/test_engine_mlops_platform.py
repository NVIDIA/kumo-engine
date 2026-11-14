# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Notebook launch, identity, and provider permission boundaries."""

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
    return json.loads((DIRECTORY / 'mlops_platforms' / f'{name}.json').read_text())


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
    assert options['timeout'] == (300 if phase else None)
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


@pytest.mark.parametrize(
    'case', ['denied', 'local-spool', 'missing', 'timeout', 'writable'])
@pytest.mark.parametrize('name', platform.PLATFORMS)
def test_negative_requires_real_permission_denial(case, name, monkeypatch):
    import sys

    denied = {
        'snowflake': '@KUMO_TEST_DB.YOUR_TEST_SCHEMA.DENIED_STAGE/mlops-validation',
        'sagemaker': 's3://kumo-test-denied/mlops-validation',
    }[name]
    instance = platform.Backend({
        **config(name), 'denied_artifact_root': denied
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

            if name == 'snowflake':
                error.__cause__ = ProviderDenied('Snowflake stage write denied')
            else:
                error.__cause__ = RuntimeError('S3 write denied')
                error.__cause__.response = {
                    'Error': {'Code': 'AccessDenied'}}
        elif case == 'missing':
            error.__cause__ = FileNotFoundError('stage missing')
        elif case == 'timeout':
            error.__cause__ = TimeoutError('provider timed out')
        raise error

    engine = SimpleNamespace(MLOps=mlops)
    monkeypatch.setitem(sys.modules, 'kumo.engine', engine)
    monkeypatch.setitem(sys.modules, 'kumo', SimpleNamespace(engine=engine))
    if case == 'denied':
        assert instance.negative()['denied_artifact_access'] == 'passed'
    elif case == 'writable':
        with pytest.raises(AssertionError, match='was writable'):
            instance.negative()
    else:
        with pytest.raises(platform.PlatformUnavailable,
                           match='without permission-denial'):
            instance.negative()


@pytest.mark.parametrize('name', ['snowflake', 'sagemaker'])
def test_aws_denial_is_accepted_only_for_selected_aws_provider(name):
    error = PermissionError('wrapped provider failure')
    cause = RuntimeError('provider denied')
    cause.response = {'Error': {'Code': 'AccessDenied'}}
    error.__cause__ = cause
    assert platform.denied_access(error, provider_only=True,
                                  provider=name) is (name == 'sagemaker')
    assert not platform.denied_access(
        PermissionError(errno.EACCES, 'local spool'), provider_only=True,
        provider=name)
