# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for credential, resource and notebook-evidence boundaries."""
import base64
import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'packages/kumo-engine/test/databricks_notebook'


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    monkeypatch.syspath_prepend(str(DIRECTORY.parent))
    monkeypatch.setenv('KUMO_DATABRICKS_TEST_HOST',
                       'https://dbc-example.cloud.databricks.com')
    client = importlib.import_module('databricks_notebook.client')
    runner = importlib.import_module('databricks_notebook.run')
    return client, runner


@pytest.fixture
def config():
    return dict(host='https://dbc-example.cloud.databricks.com',
                volume='/unused-in-unit-tests', workspace_dir='/Shared/test',
                cluster_id=None)


def test_no_developer_pat_fallback(modules, config, monkeypatch):
    client, _ = modules
    monkeypatch.delenv('KUMO_DATABRICKS_TEST_PAT', raising=False)
    monkeypatch.setenv('DATABRICKS_TOKEN', 'developer-token-must-not-be-used')
    with pytest.raises(ValueError, match='KUMO_DATABRICKS_TEST_PAT'):
        client.Client(config)


def test_candidate_is_required_before_any_cloud_access(
        modules, monkeypatch, tmp_path):
    _, runner = modules
    monkeypatch.setattr(runner, 'Client', lambda *args: pytest.fail(
        'Candidate admission must precede cloud access'))
    with pytest.raises(ValueError, match='Supply --candidate'):
        runner.execute(SimpleNamespace(candidate=None), tmp_path)


@pytest.mark.parametrize('fault', [None, 'filename', 'checksum', 'version'])
def test_runtime_candidate_admission(modules, tmp_path, fault):
    _, runner = modules
    identity = dict(wheel_file='kumo_engine-1.0.0+abcdef0-cp311-abi3-manylinux_2_34_x86_64.whl',
                    wheel_sha256='a' * 64, expected_version='1.0.0+abcdef0')
    if fault:
        identity[dict(filename='wheel_file', checksum='wheel_sha256',
                      version='expected_version')[fault]] = '../unexpected'
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                     text=True).strip()
    path = tmp_path / 'candidate.json'
    path.write_text(json.dumps(
        dict(candidate=identity, build=dict(source_commit=source))))
    if fault:
        with pytest.raises(ValueError, match='Invalid candidate wheel identity'):
            runner.read_candidate(path)
    else:
        assert runner.read_candidate(path) == identity


def test_test_resources_are_fixed_and_need_no_config_secret(
        modules, monkeypatch):
    client, _ = modules
    monkeypatch.setenv('KUMO_DATABRICKS_TEST_CONFIG',
                       '{"host":"https://other.example"}')
    value = client.configuration()
    assert value['host'] == 'https://dbc-example.cloud.databricks.com'
    assert value[
        'volume'] == '/Volumes/kumo_test_catalogue/kumo_test/test_stage/mlops-validation/notebook-ui'
    assert value[
        'workspace_dir'] == '/Shared/kumo_test_catalogue/KUMO_TEST/notebook-ui'
    assert value['cluster_id'] is None


def test_cluster_id_cannot_inject_api_parameters(modules):
    client, _ = modules
    with pytest.raises(ValueError, match='cluster_id'):
        client.configuration('arbitrary-path?token=unexpected')


def test_redirects_never_forward_pat(modules):
    client, _ = modules
    assert client.NoRedirect().redirect_request(None, None, 302, '', {},
                                                'https://other.example') is None


class FakeClient:
    def __init__(self, responses):
        self.responses, self.calls = responses, []

    def api(self, path, method='GET', **kwargs):
        self.calls.append((path, method, kwargs))
        result = self.responses[path]
        if isinstance(result, Exception):
            raise result
        return result


def test_verified_asset_is_not_uploaded(modules):
    _, runner = modules
    content = b'unchanged candidate'
    remote = '/api/2.0/fs/files/volume/payload-cache/' + runner.digest(
        content) + '/candidate.whl'
    client = FakeClient({remote: content})
    asset, actual = runner.asset(client, '/volume', 'candidate.whl',
                                 runner.digest(content))
    assert actual == content and asset['size'] == len(content)
    assert [method for _, method, _ in client.calls] == ['GET']


def test_corrupt_cached_asset_is_not_installed_or_replaced(modules):
    _, runner = modules
    expected = runner.digest(b'expected')
    remote = '/api/2.0/fs/files/volume/payload-cache/' + expected + '/candidate.whl'
    client = FakeClient({remote: b'wrong bytes'})
    with pytest.raises(ValueError, match='checksum mismatch'):
        runner.asset(client, '/volume', 'candidate.whl', expected)
    assert len(client.calls) == 1


def test_customer_cluster_is_rejected(modules, config):
    _, runner = modules
    config['cluster_id'] = '0101-010101-example'
    client = FakeClient({
        '/api/2.0/clusters/get?cluster_id=' + config['cluster_id']:
        dict(cluster_name='customer-production', state='RUNNING',
             node_type_id='i3.xlarge', num_workers=0)
    })
    with pytest.raises(ValueError, match='dedicated'):
        runner.task_request(config, '/Shared/test', 'attempt', client)


def test_missing_x86_cluster_is_rejected_before_submission(modules, config):
    _, runner = modules
    with pytest.raises(ValueError, match='dedicated'):
        runner.task_request(config, '/Shared/test', 'attempt', None)


def successful_run():
    return dict(
        state=dict(result_state='SUCCESS'), tasks=[
            dict(task_key='graph_notebook', run_id=1,
                 state=dict(result_state='FAILED')),
            dict(task_key='graph_notebook', run_id=2,
                 state=dict(result_state='SUCCESS'))
        ])


def test_automatic_retry_uses_successful_attempt(modules):
    _, runner = modules
    assert runner.successful_task(successful_run()) == 2


def test_failed_parent_cannot_reuse_old_success(modules):
    _, runner = modules
    run = successful_run()
    run['state']['result_state'] = 'FAILED'
    with pytest.raises(ValueError, match='did not succeed'):
        runner.successful_task(run)


@pytest.fixture
def evidence(modules, tmp_path):
    _, runner = modules
    spec = dict(
        attempt='a' * 12, candidate=dict(wheel_file='candidate.whl',
                                         wheel_sha256='b' * 64),
        assets={'candidate.whl': dict(path='/volume/candidate.whl')},
        output='/volume/results/unique', source_commit='c' * 40,
        notebook_path='/Shared/test/unique')
    report = dict(status='passed', execution_id='d' * 12,
                  attempt=spec['attempt'],
                  wheel_sha256=spec['candidate']['wheel_sha256'],
                  source_commit=spec['source_commit'], payload=spec['assets'],
                  runtime=dict(notebook_path=spec['notebook_path']))
    files = {name: name.encode() for name in runner.FILES}
    files['graph-result.json'] = json.dumps(report).encode()
    receipt = dict(status='passed', attempt=spec['attempt'],
                   execution_id=report['execution_id'], output=spec['output'],
                   files={
                       k: runner.digest(v)
                       for k, v in files.items()
                   })
    commands = [
        dict(command=''.join(c['source']), state='finished',
             results=dict(data=[])) for c in runner.notebook(spec)['cells']
    ]
    commands[-1]['results']['data'].append(
        dict(type='ansi',
             data='KUMO_DATABRICKS_GRAPH_RECEIPT=' + json.dumps(receipt)))
    return runner, spec, files, commands, tmp_path, runner.notebook(spec)


def collect_fake(evidence):
    runner, spec, files, commands, root, submitted = evidence
    encoded = base64.b64encode(
        quote(json.dumps(dict(commands=commands))).encode()).decode()
    html = "<script>var __DATABRICKS_NOTEBOOK_MODEL = '" + encoded + "';</script>"
    responses = {
        '/api/2.2/jobs/runs/export?run_id=2&views_to_export=ALL':
        dict(views=[dict(type='NOTEBOOK', content=html)])
    }
    responses.update({
        '/api/2.0/fs/files' + spec['output'] + '/' + n: v
        for n, v in files.items()
    })
    return runner.collect(FakeClient(responses), successful_run(), spec, root,
                          submitted)


def test_export_receipt_binds_downloaded_report_to_execution(evidence):
    report = collect_fake(evidence)
    assert report['execution_id'] == 'd' * 12


def test_exported_code_cannot_be_substituted(evidence):
    evidence[3][-1]['command'] = 'print("unrelated notebook")'
    with pytest.raises(ValueError, match='submitted cells'):
        collect_fake(evidence)


def test_downloaded_bytes_must_match_visible_receipt(evidence):
    evidence[2]['graph.pdf'] = b'substituted file'
    with pytest.raises(ValueError, match='checksum mismatch'):
        collect_fake(evidence)


def test_extra_receipt_is_rejected(evidence):
    results = evidence[3][-1]['results']['data']
    results.append(results[0].copy())
    with pytest.raises(ValueError, match='exactly one'):
        collect_fake(evidence)


def test_workflow_secrets_are_available_only_after_native_environment_approval(
):
    workflow = yaml.safe_load(
        (ROOT /
         '.github/workflows/kumo-engine-databricks-notebook.yml').read_text())
    assert set(workflow[True]) == {'workflow_dispatch'}
    gate = next(
        step for step in workflow['jobs']['configuration']['steps']
        if step.get('name') == 'Verify maintainer-protected environment')
    job = workflow['jobs']['notebook-ui']
    assert job['environment'] == 'databricks-mlops-test'
    assert job['steps'][1] == gate
    assert 'secrets.' not in json.dumps(job['steps'][:2])
    assert job['permissions'] == {'contents': 'read', 'actions': 'read'}
    assert set(job['steps'][4]['env']) == {
        'KUMO_DATABRICKS_TEST_PAT', 'KUMO_DATABRICKS_TEST_HOST'
    }
    assert 'egu/release-v1.1-platform-tests' in job['if']
    assert 'DATABRICKS_TOKEN' not in json.dumps(workflow)


def test_both_scenarios_use_this_dispatch_wheel_and_runtime_manifest():
    workflow = yaml.safe_load(
        (ROOT / '.github/workflows/kumo-engine-databricks-notebook.yml').read_text())
    wheel = workflow['jobs']['wheel']
    assert wheel['needs'] == 'configuration'
    assert 'environment' not in wheel
    assert 'secrets.' not in json.dumps(wheel)
    assert wheel['uses'] == './.github/workflows/kumo-engine-pr.yml'
    assert wheel['with'] == {'build_only': True}
    for name in ('notebook-ui', 'engine-lifecycle'):
        job = workflow['jobs'][name]
        assert set(job['needs']) == {'configuration', 'wheel'}
        download = next(step for step in job['steps']
                        if step.get('name') == "Download this dispatch's wheel")
        assert download['with']['name'] == 'engine-pr-wheel-${{ github.sha }}'
        assert 'run-id' not in download['with']
        prepare = next(step for step in job['steps']
                       if step.get('name') == "Identify this checkout's wheel")
        assert '--source-ref "$GITHUB_SHA"' in prepare['run']
        assert '--github-run "$GITHUB_RUN_ID"' in prepare['run']
        execute = next(step for step in job['steps']
                       if '--execute' in step.get('run', ''))
        assert '--candidate "$DATABRICKS_TEST_CANDIDATE"' in execute['run']
    assert not (DIRECTORY / 'candidate.json').exists()


def test_collect_uses_the_frozen_submitted_notebook(evidence, monkeypatch):
    runner = evidence[0]

    def changed_source(config):
        raise AssertionError('Source should not be regenerated after execution')

    monkeypatch.setattr(runner, 'notebook', changed_source)
    assert collect_fake(evidence)['status'] == 'passed'


@pytest.mark.parametrize('value', ['0', '-1'])
def test_invalid_deadline_rejected_before_execution(modules, value):
    _, runner = modules
    with pytest.raises(Exception, match='positive'):
        runner.positive_seconds(value)


def test_invalid_pat_cannot_enter_request_error_output(modules, config,
                                                       monkeypatch):
    client, _ = modules
    token = 'test-token\ninvalid-header'
    monkeypatch.setenv('KUMO_DATABRICKS_TEST_PAT', token)
    with pytest.raises(ValueError) as error:
        client.Client(config)
    assert token not in str(error.value)


@pytest.mark.parametrize('job_name',
                         ['configuration', 'notebook-ui', 'engine-lifecycle'])
@pytest.mark.parametrize('event,branch,repository,expected', [
    ('push', 'egu/release-v1.1-platform-tests', 'NVIDIA/kumo-engine', False),
    ('workflow_dispatch', 'egu/release-v1.1-platform-tests', 'NVIDIA/kumo-engine',
     True),
    ('workflow_dispatch', 'main', 'NVIDIA/kumo-engine', True),
    ('push', 'main', 'NVIDIA/kumo-engine', False),
    ('pull_request', 'main', 'NVIDIA/kumo-engine', False),
    ('workflow_dispatch', 'untrusted', 'NVIDIA/kumo-engine', False),
    ('workflow_dispatch', 'egu/release-v1.1-platform-tests', 'other/repository',
     False),
])
def test_cloud_jobs_require_manual_main_or_feature_dispatch(
        job_name, event, branch, repository, expected):
    workflow = yaml.safe_load(
        (ROOT /
         '.github/workflows/kumo-engine-databricks-notebook.yml').read_text())
    condition = workflow['jobs'][job_name]['if']
    expression = condition.replace('&&', ' and ').replace('||', ' or ')
    for name, value in [
        ('inputs.mode',
         'engine' if job_name == 'engine-lifecycle' else 'graph-ui'),
        ('github.event_name', event), ('github.ref', 'refs/heads/' + branch),
        ('github.repository', repository)
    ]:
        expression = expression.replace(name, repr(value))
    assert eval(expression, {'__builtins__': {}}) is expected


def export_reply(commands):
    encoded = base64.b64encode(
        quote(json.dumps(dict(commands=commands))).encode()).decode()
    html = "<script>var __DATABRICKS_NOTEBOOK_MODEL = '" + encoded + "';</script>"
    return dict(views=[dict(type='NOTEBOOK', content=html)])


def test_export_waits_for_persisted_cell_outputs(modules, tmp_path,
                                                 monkeypatch):
    _, runner = modules
    pending = dict(position=4, state='finished',
                   resultDbfsStatus='WRITING_TO_DBFS', results=None)
    ready = dict(position=4, state='finished', resultDbfsStatus='FINISHED',
                 results=dict(data=[]))
    replies = iter([export_reply([pending]), export_reply([ready])])

    class ExportClient:
        def api(self, path):
            assert 'run_id=2&' in path
            return next(replies)

    monkeypatch.setattr(runner.time, 'sleep', lambda seconds: None)
    model = runner.notebook_export(ExportClient(), 2, tmp_path)
    assert model['commands'] == [ready]
    assert json.loads(
        (tmp_path / 'export-status.json').read_text())['attempts'] == 2


def test_unpersisted_outputs_cannot_pass_after_deadline(modules, tmp_path):
    _, runner = modules
    pending = dict(position=4, state='finished',
                   resultDbfsStatus='WRITING_TO_DBFS', results=None)
    path = '/api/2.2/jobs/runs/export?run_id=2&views_to_export=ALL'
    with pytest.raises(TimeoutError, match='still being persisted'):
        runner.notebook_export(FakeClient({path: export_reply([pending])}), 2,
                               tmp_path, timeout=0)


def test_all_cloud_jobs_get_pat_only_from_protected_environment_secret():
    path = ROOT / '.github/workflows/kumo-engine-databricks-notebook.yml'
    workflow = yaml.safe_load(path.read_text())
    for name in ('notebook-ui', 'engine-lifecycle'):
        job = workflow['jobs'][name]
        assert job['environment'] == 'databricks-mlops-test'
        found = False
        for step in job['steps']:
            if 'KUMO_DATABRICKS_TEST_PAT' in step.get('env', {}):
                assert step['env']['KUMO_DATABRICKS_TEST_HOST'] == (
                    '${{ secrets.KUMO_DATABRICKS_TEST_HOST }}')
                assert step['env']['KUMO_DATABRICKS_TEST_PAT'] == (
                    '${{ secrets.KUMO_DATABRICKS_TEST_PAT }}')
                found = True
            assert 'secrets.' not in step.get('run', '')
        assert found
    triggers = workflow.get('on', workflow.get(True))
    assert set(
        triggers['workflow_dispatch']['inputs']) == {'mode', 'cluster_id'}


@pytest.mark.parametrize('host', [
    '', 'http://dbc-example.cloud.databricks.com', 'https://example.com',
    'https://dbc-example.cloud.databricks.com/path',
    'https://dbc-example.cloud.databricks.com?token=x'
])
def test_live_client_rejects_missing_or_invalid_workspace_host(
        modules, monkeypatch, host):
    client, _ = modules
    monkeypatch.setenv('KUMO_DATABRICKS_TEST_PAT', 'synthetic-test-pat')
    monkeypatch.setenv('KUMO_DATABRICKS_TEST_HOST', host)
    with pytest.raises(ValueError, match='KUMO_DATABRICKS_TEST_HOST'):
        client.Client(client.configuration())


@pytest.mark.parametrize('same_contents', [True, False])
def test_concurrent_cache_creation_requires_matching_contents(
        modules, tmp_path, same_contents):
    _, runner = modules
    wheel = tmp_path / 'candidate.whl'
    wheel.write_bytes(b'candidate')
    expected = runner.digest(wheel.read_bytes())
    calls = []

    class Client:
        def api(self, path, method='GET', **kwargs):
            calls.append((path, method))
            if len(calls) == 1:
                raise RuntimeError('Databricks API HTTP 404')
            if method == 'PUT' and '?overwrite=false' in path:
                raise RuntimeError('Databricks API HTTP 409')
            return b'candidate' if same_contents else b'corrupt'

    if same_contents:
        _, content = runner.asset(Client(), '/volume', wheel.name, expected,
                                  wheel)
        assert content == b'candidate'
    else:
        with pytest.raises(ValueError, match='checksum mismatch'):
            runner.asset(Client(), '/volume', wheel.name, expected, wheel)
    assert calls[-1][1] == 'GET'
    assert sum(method == 'PUT' and '?overwrite=false' in path
               for path, method in calls) == 1


@pytest.mark.parametrize('state', ['RUNNING', 'PENDING', 'RESTARTING', 'TERMINATED'])
def test_existing_test_compute_is_admitted_without_starting_it(modules, state):
    client, _ = modules
    cluster_id = '0101-010101-example'
    fake = FakeClient({'/api/2.0/clusters/get?cluster_id=' + cluster_id:
                       dict(cluster_id=cluster_id, cluster_name='KUMO_TEST_NOTEBOOK',
                            node_type_id='i3.xlarge', num_workers=0, state=state)})
    result = client.admit_cluster(fake, cluster_id)
    assert result['state'] == state
    assert fake.calls == [('/api/2.0/clusters/get?cluster_id=' + cluster_id, 'GET', {})]


@pytest.mark.parametrize('change', [
    dict(cluster_id='0101-010101-other'), dict(cluster_name='customer-production'),
    dict(node_type_id='m7g.large'), dict(driver_node_type_id='m7g.large'),
    dict(driver_node_type_id=''), dict(num_workers=1), dict(state='TERMINATING'),
    dict(state='ERROR'), dict(state='RESIZING'), dict(state=None),
])
def test_cluster_admission_rejects_other_resources_and_states_immediately(
        modules, change):
    client, _ = modules
    cluster_id = '0101-010101-example'
    record = dict(cluster_id=cluster_id, cluster_name='KUMO_TEST_NOTEBOOK',
                  node_type_id='i3.xlarge', num_workers=0, state='PENDING')
    record.update(change)
    fake = FakeClient({'/api/2.0/clusters/get?cluster_id=' + cluster_id: record})
    with pytest.raises(ValueError):
        client.admit_cluster(fake, cluster_id)
    assert len(fake.calls) == 1


def test_job_startup_is_bounded_and_cancels_only_its_run(modules, monkeypatch, tmp_path):
    _, runner = modules
    cluster_id = '0101-010101-example'
    config = dict(workspace_dir='/Shared/test', cluster_id=cluster_id)

    class SubmitClient(FakeClient):
        def api(self, path, method='GET', body=None):
            self.calls.append((path, method, body))
            return self.responses[path]

    fake = SubmitClient({
        '/api/2.0/clusters/get?cluster_id=' + cluster_id:
            dict(cluster_id=cluster_id, cluster_name='KUMO_TEST_NOTEBOOK',
                 node_type_id='i3.xlarge', num_workers=0, state='TERMINATED'),
        '/api/2.0/workspace/mkdirs': {}, '/api/2.0/workspace/import': {},
        '/api/2.2/jobs/runs/submit': dict(run_id=123),
        '/api/2.2/jobs/runs/get?run_id=123': dict(state=dict(life_cycle_state='PENDING')),
        '/api/2.2/jobs/runs/cancel': {},
    })
    clock = iter([0, 61])
    monkeypatch.setattr(runner.time, 'monotonic', lambda: next(clock))
    spec = dict(notebook_path='/Shared/test/attempt', attempt='123456abcdef')
    with pytest.raises(TimeoutError, match='Notebook deadline'):
        runner.submit_and_wait(fake, config, spec, {}, tmp_path, 60)
    assert fake.calls[-1] == ('/api/2.2/jobs/runs/cancel', 'POST', dict(run_id=123))
    assert all(method == 'GET' for path, method, _ in fake.calls if '/clusters/' in path)
    submitted = next(body for path, method, body in fake.calls if path.endswith('/submit'))
    assert submitted['tasks'][0]['existing_cluster_id'] == cluster_id


def test_workflow_passes_required_x86_cluster_to_both_scenarios():
    workflow = yaml.safe_load((ROOT / '.github/workflows/kumo-engine-databricks-notebook.yml').read_text())
    triggers = workflow.get('on', workflow.get(True))
    assert triggers['workflow_dispatch']['inputs']['cluster_id']['required'] is True
    assert workflow['jobs']['notebook-ui']['timeout-minutes'] >= 100
    assert workflow['jobs']['engine-lifecycle']['timeout-minutes'] >= 120
    for name in ('notebook-ui', 'engine-lifecycle'):
        runs = [step for step in workflow['jobs'][name]['steps']
                if '--execute' in step.get('run', '')]
        assert len(runs) == 1
        assert '--cluster-id "$DATABRICKS_TEST_CLUSTER"' in runs[0]['run']
        assert runs[0]['env']['DATABRICKS_TEST_CLUSTER'] == '${{ inputs.cluster_id }}'


def test_graph_execute_requires_cluster_before_credentials_or_files(
        modules, monkeypatch, tmp_path):
    _, runner = modules
    monkeypatch.setattr(runner, 'Client', lambda *a: pytest.fail('credentials loaded'))
    output = tmp_path / 'not-created'
    monkeypatch.setattr(runner.sys, 'argv',
                        ['run.py', '--execute', '--output-dir', str(output)])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize('state', ['RUNNING', 'TERMINATED'])
def test_graph_task_uses_only_the_admitted_existing_cluster(modules, config, state):
    _, runner = modules
    cluster_id = '0101-010101-example'
    config['cluster_id'] = cluster_id
    fake = FakeClient({'/api/2.0/clusters/get?cluster_id=' + cluster_id:
                       dict(cluster_id=cluster_id, cluster_name='KUMO_TEST_NOTEBOOK',
                            node_type_id='i3.xlarge', num_workers=0, state=state)})
    request = runner.task_request(config, '/Shared/test', 'attempt', fake)
    assert request['tasks'][0]['existing_cluster_id'] == cluster_id
    assert 'environment_key' not in request['tasks'][0]
    assert 'environments' not in request


@pytest.mark.parametrize('failure', ['429', '503', 'read'])
def test_read_requests_retry_transient_failures(modules, monkeypatch, failure):
    import io
    import urllib.error
    client_module, _ = modules
    monkeypatch.setenv('KUMO_DATABRICKS_TEST_PAT', 'test-token')
    client = client_module.Client(client_module.configuration())
    monkeypatch.setattr(client_module.time, 'sleep', lambda _: None)
    calls = []

    class Response(io.BytesIO):
        def read(self, *args):
            if failure == 'read' and len(calls) == 1:
                raise ConnectionResetError('private provider text')
            return super().read(*args)

    def request(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1 and failure != 'read':
            raise urllib.error.HTTPError('private', int(failure), '', {}, None)
        return Response(b'{"ok":true}')

    monkeypatch.setattr(client.opener, 'open', request)
    assert client.api('/api/2.2/jobs/runs/get?run_id=1') == {'ok': True}
    assert len(calls) == 2


def test_submit_is_not_retried_after_ambiguous_failure(modules, monkeypatch):
    client_module, _ = modules
    monkeypatch.setenv('KUMO_DATABRICKS_TEST_PAT', 'test-token')
    client = client_module.Client(client_module.configuration())
    calls = []

    def request(*args, **kwargs):
        calls.append(1)
        raise ConnectionResetError('private provider text')

    monkeypatch.setattr(client.opener, 'open', request)
    with pytest.raises(RuntimeError, match='connection failed'):
        client.api('/api/2.2/jobs/runs/submit', 'POST', {})
    assert len(calls) == 1


def test_recovery_steps_always_run_before_publication():
    workflow = yaml.safe_load(
        (ROOT /
         '.github/workflows/kumo-engine-databricks-notebook.yml').read_text())
    for name in ('notebook-ui', 'engine-lifecycle'):
        steps = workflow['jobs'][name]['steps']
        recovery = next(
            i for i, step in enumerate(steps)
            if step.get('name') == 'Recover owned attempt resources')
        assert steps[recovery]['if'] == 'always()'
        assert steps[recovery]['timeout-minutes'] == 7
        assert recovery < next(
            i for i, step in enumerate(steps)
            if step.get('name') == 'Prepare public aggregate results')


def test_graph_provenance_rejects_dirty_kernel(modules, monkeypatch):
    _, runner = modules

    def committed(command, **kwargs):
        if command[1] == 'rev-parse':
            return 'a' * 40
        path = ROOT / command[2].split(':', 1)[1]
        return b'different committed kernel' if path.name == 'kernel.py' else path.read_bytes(
        )

    monkeypatch.setattr(runner.subprocess, 'check_output', committed)
    with pytest.raises(ValueError, match='exact graph test inputs'):
        runner.committed_graph_sources()


@pytest.mark.parametrize('status,rejected', [(403, True), (429, True),
                                             (408, False)])
def test_real_submit_rejection_cleans_up_without_hiding_provider_status(
        modules, monkeypatch, tmp_path, status, rejected):
    client_module, runner = modules
    cleanup = importlib.import_module('databricks_notebook.cleanup')
    cfg = client_module.configuration('1106-082402-test')
    attempt = 'a' * 12
    spec = dict(
        attempt=attempt,
        notebook_path=cfg['workspace_dir'] + '/kumo-test-graph-' + attempt,
        output=cfg['volume'] + '/runs/' + attempt + '/outputs')
    calls = []

    class Provider:
        def api(self, path, method='GET', body=None):
            calls.append((path, method))
            if method == 'GET':
                raise RuntimeError('Databricks API HTTP 404')
            if path.endswith('/runs/submit'):
                raise runner.APIError(status)
            return {}

    client = Provider()
    monkeypatch.setattr(runner, 'admit_cluster', lambda *args: None)
    with pytest.raises(runner.APIError, match=str(status)):
        with cleanup.AttemptResources(client, cfg, spec, tmp_path) as resources:
            runner.submit_and_wait(client, cfg, spec, {}, tmp_path, 60,
                                   resources=resources)
    ledger = json.loads((tmp_path / 'resources.json').read_text())
    result = json.loads((tmp_path / 'cleanup.json').read_text())
    assert ledger['submission_pending'] is not rejected
    assert (result['status'] == 'passed') is rejected
    assert any(path.endswith('/workspace/delete')
               for path, _ in calls) is rejected
    assert sum(path.endswith('/runs/submit') for path, _ in calls) == 1
    assert not any(path.endswith('/runs/cancel') for path, _ in calls)
