# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for credential, resource and notebook-evidence boundaries."""
import base64
import importlib
import json
from pathlib import Path
from urllib.parse import quote

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'packages/kumo-engine/test/databricks_notebook'


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    monkeypatch.syspath_prepend(str(DIRECTORY.parent))
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


def test_test_resources_are_fixed_and_need_no_config_secret(
        modules, monkeypatch):
    client, _ = modules
    monkeypatch.setenv('KUMO_DATABRICKS_TEST_CONFIG',
                       '{"host":"https://other.example"}')
    value = client.configuration()
    assert value['host'] == 'https://dbc-117b701d-939a.cloud.databricks.com'
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


def test_serverless_installs_only_after_architecture_check(modules, config):
    _, runner = modules
    payload = runner.task_request(config, '/Shared/test', 'attempt', None)
    assert payload['environments'][0]['spec']['dependencies'] == []
    assert payload['environments'][0]['spec']['environment_version'] == '2'


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
    assert set(workflow[True]) == {'workflow_dispatch', 'push'}
    assert workflow[True]['push']['branches'] == ['egu/databricks-ui-validation']
    gate = workflow['jobs']['configuration']['steps'][1]
    job = workflow['jobs']['notebook-ui']
    assert job['environment'] == 'databricks-mlops-test'
    assert job['steps'][1] == gate
    assert 'secrets.' not in json.dumps(job['steps'][:2])
    assert job['permissions'] == {'contents': 'read', 'actions': 'read'}
    assert set(job['steps'][4]['env']) == {'KUMO_DATABRICKS_TEST_PAT'}
    assert 'egu/databricks-ui-validation' in job['if']
    assert 'DATABRICKS_TOKEN' not in json.dumps(workflow)


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
