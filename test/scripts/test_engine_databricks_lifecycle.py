# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline identity/evidence guards; live notebook execution is qualified separately."""
import base64
import importlib
import json
from pathlib import Path
from urllib.parse import quote

import pytest

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'packages/kumo-engine/test/databricks_notebook'


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    monkeypatch.syspath_prepend(str(DIRECTORY.parent))
    return importlib.import_module('databricks_notebook.lifecycle_notebook')


def passed_report():
    return dict(
        identity='test-principal', wheel_sha256='a' * 64,
        pip_check_returncode=0, phases={
            n:
            dict(
                status='passed', returncode=0,
                environment=dict(sha256='a' * 64), platform=dict(placement=dict(
                    context='databricks', user='test-principal')))
            for n in ('fit', 'score', 'evaluate', 'negative')
        }, required_gaps={'native_ui': dict(status='blocked')})


def test_engine_success_preserves_full_qualification_gap(runner):
    result = runner.engine_verdict(passed_report())
    assert result['engine_checks'] == 'passed'
    assert result['full_qualification'] == 'incomplete'
    assert result['remaining_requirements']['native_ui']['status'] == 'blocked'


@pytest.mark.parametrize('field', ['identity', 'wheel', 'context', 'pip'])
def test_passed_phase_requires_real_admission_evidence(runner, field):
    report = passed_report()
    phase = report['phases']['fit']
    if field == 'identity':
        phase['platform']['placement']['user'] = 'another-identity'
    elif field == 'context':
        phase['platform']['placement']['context'] = 'client'
    elif field == 'wheel':
        phase['environment']['sha256'] = 'b' * 64
    else:
        report['pip_check_returncode'] = 1
    with pytest.raises(ValueError):
        runner.engine_verdict(report)


def test_permission_block_does_not_qualify_all_engine_checks(runner):
    report = passed_report()
    report['phases']['negative'] = dict(status='blocked')
    result = runner.engine_verdict(report)
    assert result['positive_lifecycle'] == 'passed'
    assert result['engine_checks'] == 'incomplete'


@pytest.fixture
def evidence(runner, tmp_path):
    spec = dict(
        attempt='a' * 12, source_commit='b' * 40,
        candidate=dict(wheel_sha256='c' * 64), assets={}, test_sources={},
        expected_user='test-principal',
        output='/Volumes/kumo_test_catalogue/kumo_test/test_stage/test-only',
        notebook_path='/Shared/test-only')
    report = dict(status='blocked', attempt=spec['attempt'],
                  source_commit=spec['source_commit'],
                  wheel_sha256=spec['candidate']['wheel_sha256'], payload={},
                  test_sources={}, notebook_path=spec['notebook_path'],
                  identity='test-principal', execution_id='d' * 12)
    content = json.dumps(report).encode()
    receipt = dict(attempt=spec['attempt'], source_commit=spec['source_commit'],
                   output=spec['output'], execution_id=report['execution_id'],
                   files={'lifecycle-result.json': runner.digest(content)})
    submitted = runner.notebook(spec)
    commands = [
        dict(command=''.join(c['source']), state='finished',
             results=dict(data=[])) for c in submitted['cells']
    ]
    commands[-1]['results']['data'] = [
        dict(type='ansi',
             data='KUMO_DATABRICKS_LIFECYCLE_RECEIPT=' + json.dumps(receipt))
    ]
    return runner, spec, report, receipt, content, submitted, commands, tmp_path


def collect(evidence):
    runner, spec, report, receipt, content, submitted, commands, root = evidence
    encoded = base64.b64encode(
        quote(json.dumps(dict(commands=commands))).encode()).decode()
    export = dict(views=[
        dict(
            type='NOTEBOOK',
            content="<script>var __DATABRICKS_NOTEBOOK_MODEL = '" + encoded +
            "';</script>")
    ])

    class Client:
        def api(self, path, **kwargs):
            if '/runs/export?' in path:
                return export
            assert path == '/api/2.0/fs/files' + spec[
                'output'] + '/lifecycle-result.json'
            return content

    return runner.collect(
        Client(),
        dict(
            state=dict(result_state='SUCCESS'), tasks=[
                dict(task_key='graph_notebook', run_id=1,
                     state=dict(result_state='SUCCESS'))
            ]), spec, submitted, root)


def test_retained_lifecycle_is_bound_to_actual_export_and_hashes(evidence):
    assert collect(evidence)['execution_id'] == 'd' * 12


def test_lifecycle_download_cannot_substitute_another_report(evidence):
    mutated = list(evidence)
    mutated[4] = b'changed bytes'
    with pytest.raises(ValueError, match='checksum mismatch'):
        collect(mutated)


def test_unrelated_cell_source_is_rejected(evidence):
    evidence[6][-1]['command'] = 'print("unrelated")'
    with pytest.raises(ValueError, match='submitted execution'):
        collect(evidence)


def test_nonzero_worker_exit_cannot_be_qualified_from_a_passed_receipt(runner):
    report = passed_report()
    report['phases']['negative']['returncode'] = 1
    with pytest.raises(ValueError, match='worker exit code'):
        runner.engine_verdict(report)


def test_dirty_lifecycle_source_is_rejected_before_auth_or_cloud_calls(
        runner, tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(
        runner.subprocess, 'check_output', lambda command, **kwargs: 'a' * 40
        if 'rev-parse' in command else b'different committed source')
    monkeypatch.setattr(
        runner, 'Client',
        lambda config: pytest.fail('No cloud client should be constructed'))
    with pytest.raises(ValueError,
                       match='Commit the exact lifecycle test inputs'):
        runner.execute(SimpleNamespace(cluster_id='0101-010101-test'), tmp_path)


def test_notebook_keeps_committed_kernel_after_working_source_changes(
        runner, monkeypatch):
    def no_read(path, *args, **kwargs):
        raise AssertionError(
            'Frozen notebook construction must not reread working source')

    monkeypatch.setattr(Path, 'read_text', no_read)
    value = runner.notebook(dict(attempt='a' * 12),
                            kernel_source='COMMITTED_KERNEL_SENTINEL = True')
    code = ''.join(value['cells'][-1]['source'])
    assert 'COMMITTED_KERNEL_SENTINEL = True' in code


@pytest.mark.parametrize('http', [403, 404])
def test_denied_volume_setup_is_not_permission_denial_evidence(runner, http):
    class Client:
        def api(self, path):
            raise RuntimeError(f'Databricks API HTTP {http}')

    result = runner.denied_storage(Client(), runner.configuration())
    assert result['status'] == 'blocked'
    assert 'artifact_root' not in result
    assert result['reason']


def test_empty_readable_denied_volume_is_admitted_for_the_real_probe(runner):
    calls = []

    class Client:
        def api(self, path):
            calls.append(path)
            if '/unity-catalog/' in path:
                return dict(
                    full_name='kumo_test_denied.kumo_test.test_denied')
            return dict(contents=[])

    result = runner.denied_storage(Client(), runner.configuration())
    assert result['status'] == 'ready'
    assert calls[-1] == (
        '/api/2.0/fs/directories/Volumes/kumo_test_denied/kumo_test/test_denied'
    )
    assert result['artifact_root'].endswith(
        '/test_denied/mlops-validation/lifecycle')


def test_denied_volume_preflight_does_not_hide_service_failures(runner):
    class Client:
        def api(self, path):
            raise RuntimeError('Databricks API HTTP 500')

    with pytest.raises(RuntimeError, match='HTTP 500'):
        runner.denied_storage(Client(), runner.configuration())


def test_denied_volume_metadata_cannot_substitute_another_resource(runner):
    class Client:
        def api(self, path):
            return dict(full_name='another.schema.volume')

    with pytest.raises(ValueError, match='another Volume'):
        runner.denied_storage(Client(), runner.configuration())


def test_denied_storage_setup_cannot_be_substituted_in_collected_report(evidence):
    runner, spec, report, receipt, content, submitted, commands, root = evidence
    report['denied_storage'] = dict(status='ready', volume='/another/volume')
    content = json.dumps(report).encode()
    receipt['files']['lifecycle-result.json'] = runner.digest(content)
    commands[-1]['results']['data'][0]['data'] = (
        'KUMO_DATABRICKS_LIFECYCLE_RECEIPT=' + json.dumps(receipt))
    mutated = (runner, spec, report, receipt, content, submitted, commands, root)
    with pytest.raises(ValueError, match='identity differs'):
        collect(mutated)


def test_native_worker_crash_retains_and_redacts_phase_log(monkeypatch, tmp_path):
    import hashlib
    import subprocess
    import sys
    import tempfile
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(DIRECTORY))
    kernel = importlib.import_module('lifecycle_kernel')
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    monkeypatch.setattr(tempfile, 'mkdtemp', lambda **kwargs: str(runtime))
    monkeypatch.setenv('DATABRICKS_RUNTIME_VERSION', 'test-runtime')
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: None)
    workspace = SimpleNamespace(
        current_user=SimpleNamespace(me=lambda: SimpleNamespace(user_name='test-user')),
        config=SimpleNamespace(host='https://test.cloud.databricks.com',
                               authenticate=lambda: {'Authorization': 'Bearer test-runtime-token'}))
    monkeypatch.setitem(sys.modules, 'databricks.sdk',
                        SimpleNamespace(WorkspaceClient=lambda: workspace))
    context = SimpleNamespace(notebookPath=lambda: SimpleNamespace(get=lambda: '/Shared/test'))
    utils = SimpleNamespace(notebook=lambda: SimpleNamespace(getContext=lambda: context))
    dbutils = SimpleNamespace(notebook=SimpleNamespace(entry_point=SimpleNamespace(
        getDbutils=lambda: utils)))
    source = tmp_path / 'wheel_mlops_lifecycle.py'
    source.write_text('''from pathlib import Path

def execute(command, root, log, timeout, env=None):
    if '--execute-platform' in command:
        output = Path(command[command.index('--output-dir') + 1])
        output.mkdir()
        (output / 'fit.log').write_text('native worker crash: ' + env['DATABRICKS_TOKEN'])
        raise RuntimeError('Worker produced no receipt')
    return 0
''')
    wheel = tmp_path / 'candidate.whl'
    wheel.write_bytes(b'wheel')
    assets = {p.name: dict(path=str(p), size=p.stat().st_size,
                           sha256=hashlib.sha256(p.read_bytes()).hexdigest())
              for p in (source, wheel)}
    output = tmp_path / 'output'
    kernel.execute(dict(attempt='a' * 12, source_commit='b' * 40,
                        candidate=dict(wheel_file=wheel.name,
                                       wheel_sha256=assets[wheel.name]['sha256'],
                                       expected_version='1.0'),
                        notebook_path='/Shared/test', assets=assets, test_sources={},
                        expected_user='test-user', cluster_id='0101-010101-test',
                        artifact_root='/Volumes/test/models', experiment='test',
                        output=str(output)), dbutils)
    report = json.loads((output / 'lifecycle-result.json').read_text())
    assert report['status'] == 'failed'
    assert 'Worker produced no receipt' in report['error']
    assert (output / 'fit.log').read_text() == 'native worker crash: [REDACTED]'


def test_lifecycle_uses_mounted_files_but_keeps_native_tracking_registry(
        runner, monkeypatch):
    import sys
    from types import SimpleNamespace

    module = importlib.import_module('wheel_mlops_platform')
    config = json.loads((ROOT / 'packages/kumo-engine/test/mlops_platforms/databricks.json').read_text())
    profile = module.Mlflow(config, 'a' * 12)
    native = object()
    profile.connector = native
    monkeypatch.setattr(profile, 'placement', lambda: {})
    kumo = SimpleNamespace(MLOps=lambda root, **kwargs: dict(root=root, **kwargs),
                           ModelRegistryLocation=lambda name: name)
    monkeypatch.setitem(sys.modules, 'kumo.engine', kumo)
    monkeypatch.setitem(sys.modules, 'kumo', SimpleNamespace(engine=kumo))
    monkeypatch.setitem(sys.modules, 'mlflow', SimpleNamespace(
        set_tracking_uri=lambda uri: None, set_registry_uri=lambda uri: None))
    monkeypatch.setitem(sys.modules, 'kumo.engine.integrations.mlflow', SimpleNamespace(
        MlflowTracking=lambda experiment, **kwargs: kwargs['target']))
    cfg = profile.settings(register=True)
    assert cfg['root'] == profile.artifact_root
    assert isinstance(cfg['root'], str)
    assert cfg['tracking'] is native
    assert cfg['registry'] == profile.model
