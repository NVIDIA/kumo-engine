# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline provider-state tests for recovery after losing every runner file."""
import importlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

import pytest
import yaml

DIRECTORY = Path(__file__).resolve(
).parents[2] / 'packages/kumo-engine/test/databricks_notebook'


class Provider:
    def __init__(self, config):
        self.config = config
        self.token = 'test-only-pat'
        self.user = config['expected_user']
        self.files, self.runs = {}, {}
        self.directories, self.notebooks = set(), set()
        self.calls = []
        self.fail_once = None
        self.cancel_stuck = False

    def mkdir(self, path):
        while path and path != '/':
            self.directories.add(path)
            path = path.rsplit('/', 1)[0]

    def api(self, path, method='GET', body=None, raw=None, binary=False,
            max_bytes=None):
        self.calls.append((path, method, body))
        parsed = urlsplit(path)
        target = unquote(parsed.path)
        if target == '/api/2.0/preview/scim/v2/Me':
            return dict(userName=self.user)
        if target.startswith('/api/2.0/fs/directories'):
            remote = target.removeprefix('/api/2.0/fs/directories')
            if method == 'PUT':
                self.mkdir(remote)
                return {}
            if method == 'DELETE':
                if self.fail_once == remote:
                    self.fail_once = None
                    raise RuntimeError('temporary provider error')
                assert not any(
                    p.rsplit('/', 1)[0] == remote for p in self.files)
                self.directories.discard(remote)
                return {}
            if remote not in self.directories:
                raise RuntimeError('Databricks API HTTP 404')
            contents = [
                dict(path=p + '/', is_directory=True)
                for p in sorted(self.directories)
                if p.rsplit('/', 1)[0] == remote
            ]
            contents += [
                dict(path=p, is_directory=False) for p in sorted(self.files)
                if p.rsplit('/', 1)[0] == remote
            ]
            return dict(contents=contents)
        if target.startswith('/api/2.0/fs/files'):
            remote = target.removeprefix('/api/2.0/fs/files')
            if method == 'PUT':
                if remote in self.files and parsed.query == 'overwrite=false':
                    raise RuntimeError('Databricks API HTTP 409')
                assert isinstance(raw, bytes)
                self.mkdir(remote.rsplit('/', 1)[0])
                self.files[remote] = raw
                return {}
            if method == 'DELETE':
                if self.fail_once == remote:
                    self.fail_once = None
                    raise RuntimeError('temporary provider error')
                self.files.pop(remote, None)
                return {}
            if remote not in self.files:
                raise RuntimeError('Databricks API HTTP 404')
            return self.files[remote]
        if target == '/api/2.0/workspace/delete':
            assert body['recursive'] is False
            if body['path'] not in self.notebooks:
                raise RuntimeError('Databricks API HTTP 404')
            self.notebooks.remove(body['path'])
            return {}
        if target == '/api/2.2/jobs/runs/get':
            run_id = int(parsed.query.split('=')[1])
            if run_id not in self.runs:
                raise RuntimeError('Databricks API HTTP 404')
            return self.runs[run_id]
        if target == '/api/2.2/jobs/runs/cancel':
            if not self.cancel_stuck:
                self.runs[
                    body['run_id']]['state']['life_cycle_state'] = 'TERMINATED'
            return {}
        if '/unity-catalog/models/' in target or '/mlflow/experiments/' in target:
            raise RuntimeError('Databricks API HTTP 404')
        raise AssertionError((path, method))


@pytest.fixture(autouse=True)
def isolated_launcher_helpers():
    # Each CLI has its own process and generic helper module names.
    absent = object()
    names = ('cleanup', 'retention_ledger')
    previous = {name: sys.modules.pop(name, absent) for name in names}
    try:
        yield
    finally:
        for name, module in previous.items():
            sys.modules.pop(name, None)
            if module is not absent:
                sys.modules[name] = module


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY.parent))
    monkeypatch.syspath_prepend(str(DIRECTORY))
    retention = importlib.import_module('retention_ledger')
    cleanup = importlib.import_module('cleanup')
    client = importlib.import_module('client')
    monkeypatch.setenv('KUMO_DATABRICKS_TEST_HOST',
                       'https://dbc-example.cloud.databricks.com')
    monkeypatch.setenv('KUMO_DATABRICKS_TEST_USER', 'test-principal')
    config = client.configuration()
    assert config['persistent_ledger'] is True
    return retention, cleanup, config, Provider(config)


def leaked(setup, root, *, attempt='a' * 12, submitted=True):
    retention, cleanup, config, cloud = setup
    root.mkdir()
    spec = dict(
        attempt=attempt,
        notebook_path=config['workspace_dir'] + '/kumo-test-graph-' + attempt,
        output=config['volume'] + '/runs/' + attempt + '/outputs')
    (root / 'scenario.json').write_text(json.dumps(spec))
    resources = cleanup.AttemptResources(cloud, config, spec, root).__enter__()
    prefix = spec['output'].removesuffix('/outputs')
    cloud.mkdir(prefix)
    cloud.files[prefix + '/wheel.whl'] = b'candidate'
    resources.importing()
    cloud.notebooks.add(spec['notebook_path'])
    resources.notebook_created()
    if submitted:
        resources.submitting()
        (root / 'submitted-run.json').write_text('{"run_id":17}')
        cloud.runs[17] = dict(
            run_id=17, creator_user_name=config['expected_user'],
            run_name='kumo-test-notebook-ui-' + attempt,
            state=dict(life_cycle_state='TERMINATED'), tasks=[
                dict(notebook_task=dict(notebook_path=spec['notebook_path']))
            ])
        resources.submitted()
    now = resources.ledger['started_ms'] + 25 * 3600000
    return resources, spec, now


def test_vm_loss_recovers_from_only_remote_signed_state(setup, tmp_path):
    retention, _, config, cloud = setup
    resource, spec, now = leaked(setup, tmp_path / 'lost-vm')
    path = resource.remote.path
    assert cloud.token.encode() not in cloud.files[path]
    shutil.rmtree(resource.root)
    result = retention.sweep(cloud, config, tmp_path / 'new-vm', execute=True,
                             now_ms=now)
    assert result['cleaned'] == 1
    assert path not in cloud.files
    assert spec['notebook_path'] not in cloud.notebooks
    assert not any('/runs/' in name for name in cloud.files)


def test_budget_exhaustion_preserves_signed_state_for_next_sweep(
        setup, tmp_path, monkeypatch):
    retention, _, config, cloud = setup
    resource, spec, now = leaked(setup, tmp_path / 'lost-vm')
    prefix = spec['output'].removesuffix('/outputs')
    child = prefix + '/child'
    cloud.mkdir(child)
    cloud.files[child + '/a'] = b'first'
    cloud.files[child + '/b'] = b'second'
    monkeypatch.setattr(retention, 'MAX_FILES', 4)
    result = retention.sweep(cloud, config, tmp_path / 'first-sweep',
                             execute=True, now_ms=now)
    assert result['cleaned'] == 0
    assert result['preserved'] == {'unproven_or_failed_cleanup': 1}
    assert resource.remote.path in cloud.files
    assert prefix + '/wheel.whl' in cloud.files
    assert child not in cloud.directories
    failed = tmp_path / 'first-sweep' / spec['attempt'] / 'cleanup.json'
    assert json.loads(failed.read_text())['failed_resources'] == ['storage']
    result = retention.sweep(cloud, config, tmp_path / 'second-sweep',
                             execute=True, now_ms=now)
    assert result['cleaned'] == 1
    assert resource.remote.path not in cloud.files
    assert not any('/runs/' in name for name in cloud.files)


@pytest.mark.parametrize('fault', [
    'tamper', 'unsigned', 'wrong_pat', 'foreign_user', 'foreign_scope', 'recent'
])
def test_unproven_or_recent_record_never_mutates_resources(
        setup, tmp_path, fault):
    retention, _, config, cloud = setup
    resource, _, now = leaked(setup, tmp_path / 'lost')
    path = resource.remote.path
    if fault in ('tamper', 'foreign_scope'):
        value = json.loads(cloud.files[path])
        value['payload']['config']['expected_user' if fault ==
                                   'tamper' else 'volume'] = 'foreign'
        if fault == 'foreign_scope':
            value['hmac_sha256'] = retention.signature(cloud, value['payload'])
        cloud.files[path] = json.dumps(value).encode()
    elif fault == 'unsigned':
        cloud.files[path] = json.dumps(
            json.loads(cloud.files[path])['payload']).encode()
    elif fault == 'wrong_pat':
        cloud.token = 'rotated-test-pat'
    elif fault == 'foreign_user':
        cloud.user = 'other-principal'
    else:
        now = resource.ledger['started_ms'] + 23 * 3600000
    cloud.calls.clear()
    if fault == 'foreign_user':
        with pytest.raises(ValueError, match='PAT identity'):
            retention.sweep(cloud, config, tmp_path / 'sweep', execute=True,
                            now_ms=now)
    else:
        result = retention.sweep(cloud, config, tmp_path / 'sweep',
                                 execute=True, now_ms=now)
        assert result['cleaned'] == 0
        assert sum(result['preserved'].values()) == 1
    assert all(method == 'GET' for _, method, _ in cloud.calls)
    assert path in cloud.files


@pytest.mark.parametrize(
    'pending', ['submission_pending', 'import_pending', 'experiment_pending'])
def test_unknown_creation_or_submission_is_preserved(setup, tmp_path, pending):
    retention, _, config, cloud = setup
    resource, _, now = leaked(setup, tmp_path / 'lost', submitted=False)
    resource.ledger[pending] = True
    resource.save()
    cloud.calls.clear()
    result = retention.sweep(cloud, config, tmp_path / 'sweep', execute=True,
                             now_ms=now)
    assert result['preserved'] == {'unknown_submission_or_creation': 1}
    assert all(method == 'GET' for _, method, _ in cloud.calls)


@pytest.mark.parametrize('foreign', [False, True])
def test_only_exact_owned_active_job_can_be_cancelled(setup, tmp_path, foreign):
    retention, _, config, cloud = setup
    resource, spec, now = leaked(setup, tmp_path / 'lost')
    cloud.runs[17]['state']['life_cycle_state'] = 'RUNNING'
    if foreign:
        cloud.runs[17]['creator_user_name'] = 'foreign-user'
    cloud.calls.clear()
    result = retention.sweep(cloud, config, tmp_path / 'sweep', execute=True,
                             now_ms=now)
    cancelled = [
        p for p, method, _ in cloud.calls if p.endswith('/runs/cancel')
    ]
    assert len(cancelled) == (0 if foreign else 1)
    assert result['cleaned'] == (0 if foreign else 1)
    assert (resource.remote.path in cloud.files) is foreign
    assert (spec['notebook_path'] in cloud.notebooks) is foreign


def test_partial_cleanup_checkpoint_survives_another_vm_loss(setup, tmp_path):
    retention, _, config, cloud = setup
    resource, spec, now = leaked(setup, tmp_path / 'lost')
    cloud.fail_once = spec['output'].removesuffix('/outputs') + '/wheel.whl'
    with pytest.raises(RuntimeError, match='cleanup failed'):
        resource.__exit__()
    assert retention.read(
        cloud, resource.remote.path)['ledger']['notebook_removed'] is True
    shutil.rmtree(resource.root)
    cloud.calls.clear()
    result = retention.sweep(cloud, config, tmp_path / 'replacement',
                             execute=True, now_ms=now)
    assert result['cleaned'] == 1
    assert not any(p.endswith('/workspace/delete') for p, _, _ in cloud.calls)


def test_plan_and_ttl_floor_cannot_delete_anything(setup, tmp_path):
    retention, _, config, cloud = setup
    leaked(setup, tmp_path / 'lost')
    cloud.calls.clear()
    result = retention.sweep(cloud, config, tmp_path / 'plan', now_ms=10**15)
    assert result['eligible'] == 1 and result['cleaned'] == 0
    assert all(method == 'GET' for _, method, _ in cloud.calls)
    with pytest.raises(ValueError, match='at least 24'):
        retention.sweep(cloud, config, tmp_path / 'bad', execute=True,
                        ttl_hours=23)


def test_shared_payload_cache_and_file_budget_are_preserved(
        setup, tmp_path, monkeypatch):
    retention, _, config, cloud = setup
    resource, _, now = leaked(setup, tmp_path / 'lost')
    cache = config['volume'] + '/payload-cache/hash/fixture.tar.gz'
    cloud.files[cache] = b'shared'
    monkeypatch.setattr(retention, 'MAX_FILES', 1)
    result = retention.sweep(cloud, config, tmp_path / 'small-budget',
                             execute=True, now_ms=now)
    assert result['cleaned'] == 0
    assert resource.remote.path in cloud.files and cloud.files[
        cache] == b'shared'


def test_actual_graph_entrypoint_persists_before_attempt_upload(
        setup, tmp_path, monkeypatch):
    retention, cleanup, config, cloud = setup
    runner = importlib.import_module('databricks_notebook.run')
    root = tmp_path / 'lost'
    root.mkdir()
    monkeypatch.setattr(
        runner, 'read_candidate',
        lambda _: dict(wheel_file='candidate.whl', wheel_sha256='a' * 64))
    monkeypatch.setattr(runner, 'committed_graph_sources', lambda: 'a' * 40)
    monkeypatch.setattr(runner, 'Client', lambda _: cloud)
    monkeypatch.setattr(runner, 'configuration', lambda _: config)
    monkeypatch.setattr(runner, 'admit_cluster', lambda *args: None)

    def upload(client, volume, name, expected, local=None):
        if '/runs/' in volume:
            controls = [p for p in cloud.files if p.endswith('/resources.json')]
            assert len(controls) == 1
            assert retention.read(cloud,
                                  controls[0])['ledger']['admitted'] is True
            raise SystemExit('VM was lost during attempt upload')
        return dict(path=volume + '/' + name, sha256=expected), b''

    monkeypatch.setattr(runner, 'asset', upload)
    # A lost VM cannot execute Python context-manager cleanup.
    monkeypatch.setattr(cleanup.AttemptResources, '__exit__',
                        lambda *args: False)
    with pytest.raises(SystemExit, match='VM was lost'):
        runner.execute(
            SimpleNamespace(candidate=None, cluster_id='0101-010101-test',
                            fixture=None, wheel=tmp_path / 'candidate.whl'),
            root)
    assert any(p.endswith('/resources.json') for p in cloud.files)


def test_manual_workflow_runs_one_sweep_before_notebooks():
    workflow = yaml.safe_load(
        (DIRECTORY.parents[3] /
         '.github/workflows/kumo-engine-databricks-notebook.yml').read_text())
    found = []
    for job in workflow['jobs'].values():
        steps = job.get('steps', [])
        sweep = [
            i for i, step in enumerate(steps)
            if 'retention_ledger.py' in step.get('run', '')
        ]
        if sweep:
            execution = next(i for i, step in enumerate(steps)
                             if '--execute --cluster-id' in step.get('run', ''))
            assert len(sweep) == 1 and sweep[0] < execution
            found.append(steps[sweep[0]].get('if'))
    assert found == [None, "inputs.mode == 'engine'"]


def test_duplicate_control_record_cannot_acquire_ownership(setup, tmp_path):
    retention, cleanup, config, cloud = setup
    resource, spec, _ = leaked(setup, tmp_path / 'first', submitted=False)
    # Remove attempt artifacts to isolate the independently checked ledger collision.
    prefix = spec['output'].removesuffix('/outputs')
    cloud.files.pop(prefix + '/wheel.whl')
    cloud.directories.remove(prefix)
    previous = cloud.files[resource.remote.path]
    second = tmp_path / 'second'
    second.mkdir()
    candidate = cleanup.AttemptResources(cloud, config, spec, second)
    with pytest.raises(RuntimeError, match='409'):
        candidate.__enter__()
    assert candidate.ledger['admitted'] is False
    assert cloud.files[resource.remote.path] == previous


def test_inventory_budget_and_repeated_pagination_fail_closed(
        setup, tmp_path, monkeypatch):
    retention, _, config, cloud = setup
    resource, _, now = leaked(setup, tmp_path / 'first')
    monkeypatch.setattr(retention, 'MAX_RECORDS', 0)
    cloud.calls.clear()
    with pytest.raises(ValueError, match='record budget'):
        retention.sweep(cloud, config, tmp_path / 'limited', execute=True,
                        now_ms=now)
    assert all(method == 'GET' for _, method, _ in cloud.calls)
    monkeypatch.setattr(retention, 'MAX_RECORDS', 200)
    original = cloud.api

    def repeated(path, method='GET', **kwargs):
        if '/retention?' in path:
            return dict(contents=[], next_page_token='same')
        return original(path, method, **kwargs)

    cloud.api = repeated
    with pytest.raises(ValueError, match='pagination is ambiguous'):
        retention.sweep(cloud, config, tmp_path / 'repeated', execute=True,
                        now_ms=now)
    assert resource.remote.path in cloud.files


def test_missing_job_history_preserves_all_resources(setup, tmp_path):
    retention, _, config, cloud = setup
    resource, spec, now = leaked(setup, tmp_path / 'lost')
    cloud.runs.clear()
    cloud.calls.clear()
    result = retention.sweep(cloud, config, tmp_path / 'recovery', execute=True,
                             now_ms=now)
    assert result['cleaned'] == 0
    assert all(method == 'GET' for _, method, _ in cloud.calls)
    assert resource.remote.path in cloud.files and spec[
        'notebook_path'] in cloud.notebooks


def test_control_record_network_read_is_bounded(setup, monkeypatch):
    import io
    retention, _, config, _ = setup
    client_module = importlib.import_module('client')
    monkeypatch.setenv('KUMO_DATABRICKS_TEST_PAT', 'test-token')
    client = client_module.Client(config)
    reads = []

    class Response(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    monkeypatch.setattr(client.opener, 'open',
                        lambda *args, **kwargs: Response(b'x' * (retention.MAX_BYTES * 2)))
    with pytest.raises(ValueError, match='response exceeds byte limit'):
        retention.read(client, retention.file_path(config, 'a' * 12))
    assert reads == [retention.MAX_BYTES + 1]


def test_empty_control_cleanup_is_retried_without_claiming_attempt_cleanup(
        setup, tmp_path):
    retention, _, config, cloud = setup
    resources, spec, now = leaked(setup, tmp_path / 'initial')
    folder = resources.remote.path.rsplit('/', 1)[0]
    cloud.fail_once = folder
    resources.__exit__()
    report = json.loads((resources.root / 'cleanup.json').read_text())
    assert report['status'] == 'passed'
    assert report['control_directory_cleanup'] == 'pending'
    assert folder in cloud.directories
    assert not cloud.files and spec['notebook_path'] not in cloud.notebooks
    result = retention.sweep(cloud, config, tmp_path / 'later', execute=True, now_ms=now)
    assert result['empty_controls_removed'] == 1
    assert result['cleaned'] == 0
    assert folder not in cloud.directories


def test_missing_record_does_not_authorize_nonempty_control_directory_deletion(
        setup, tmp_path):
    retention, _, config, cloud = setup
    folder = config['volume'] + '/retention/' + 'a' * 12
    cloud.mkdir(folder)
    cloud.files[folder + '/foreign.json'] = b'not a control record'
    result = retention.sweep(cloud, config, tmp_path / 'sweep', execute=True)
    assert result['empty_controls_removed'] == result['cleaned'] == 0
    assert cloud.files[folder + '/foreign.json']
    assert not any(method == 'DELETE' for _, method, _ in cloud.calls)
