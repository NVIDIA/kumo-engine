# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve(
).parents[2] / 'packages/kumo-engine/test/databricks_notebook/cleanup.py'
spec = importlib.util.spec_from_file_location('attempt_cleanup', SOURCE)
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


class Client:
    def __init__(self):
        self.calls = []
        self.present = False
        self.listing = []
        self.model = None
        self.run = 'TERMINATED'
        self.fail_delete = False

    def api(self, path, method='GET', body=None):
        self.calls.append((path, method, body))
        if method != 'GET':
            if self.fail_delete:
                raise RuntimeError('delete failed')
            return {}
        if '/jobs/runs/get?' in path:
            return dict(state=dict(life_cycle_state=self.run))
        if '/models/' in path and self.model is not None:
            return self.model
        if '/fs/directories' in path and self.present:
            return dict(contents=self.listing)
        raise RuntimeError('Databricks API HTTP 404')


@pytest.fixture
def attempt(tmp_path):
    client = Client()
    config = dict(workspace_dir='/Shared/test',
                  volume='/Volumes/test/notebook-ui',
                  expected_user='test-principal')
    attempt = 'a' * 12
    spec = dict(
        attempt=attempt,
        notebook_path=config['workspace_dir'] + '/kumo-test-graph-' + attempt,
        output=config['volume'] + '/runs/' + attempt + '/outputs')
    return client, config, spec, tmp_path


@pytest.mark.parametrize('body_fails', [False, True])
def test_cleanup_runs_after_success_and_failure(attempt, body_fails):
    client, config, spec, root = attempt
    with pytest.raises(ValueError) if body_fails else __import__(
            'contextlib').nullcontext():
        with cleanup.AttemptResources(client, config, spec, root) as resources:
            resources.notebook_created()
            client.present = True
            if body_fails:
                raise ValueError('scenario failed')
    deleted = [(p, m, b) for p, m, b in client.calls if m != 'GET']
    assert deleted == [
        ('/api/2.0/workspace/delete', 'POST',
         dict(path=spec['notebook_path'], recursive=False)),
        ('/api/2.0/fs/directories' + spec['output'].removesuffix('/outputs'),
         'DELETE', None)
    ]
    assert json.loads((root / 'cleanup.json').read_text())['status'] == 'passed'


def test_preexisting_prefix_never_acquires_cleanup_ownership(attempt):
    client, config, spec, root = attempt
    client.present = True
    with pytest.raises(ValueError, match='already exists'):
        with cleanup.AttemptResources(client, config, spec, root):
            pytest.fail('must not enter')
    assert all(method == 'GET' for _, method, _ in client.calls)


@pytest.mark.parametrize('case',
                         ['unknown_submission', 'running', 'query_failure'])
def test_unknown_or_active_job_preserves_resources(attempt, monkeypatch, case):
    client, config, spec, root = attempt
    with pytest.raises(RuntimeError):
        with cleanup.AttemptResources(client, config, spec, root) as resources:
            resources.notebook_created()
            resources.submitting()
            if case != 'unknown_submission':
                (root / 'submitted-run.json').write_text('{"run_id": 1}')
                if case == 'running':
                    client.run = 'RUNNING'
                    times = iter([0, 181])
                    monkeypatch.setattr(cleanup.time, 'monotonic',
                                        lambda: next(times))
                else:
                    client.api = lambda *args: (_ for _ in ()).throw(
                        RuntimeError('unreachable'))
    assert all(method == 'GET' for _, method, _ in client.calls)
    assert json.loads((root / 'cleanup.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('case', ['escaping_listing', 'delete_failure'])
def test_cleanup_failures_cannot_leave_passed_result(attempt, case):
    client, config, spec, root = attempt
    with pytest.raises(RuntimeError, match='cleanup failed'):
        with cleanup.AttemptResources(client, config, spec, root):
            client.present = True
            if case == 'escaping_listing':
                client.listing = [
                    dict(path='/Volumes/shared/valuable', is_directory=False)
                ]
            else:
                client.fail_delete = True
    assert json.loads((root / 'cleanup.json').read_text())['status'] == 'failed'
    if case == 'escaping_listing':
        assert all(method == 'GET' for _, method, _ in client.calls)


@pytest.mark.parametrize(
    'case', ['success', 'collision', 'changed_tag', 'ambiguous_create'])
def test_lifecycle_experiment_cleanup_requires_creation_receipt(attempt, case):
    client, config, spec, root = attempt
    folder = '/Shared/kumo_test_catalogue/KUMO_TEST/lifecycle'
    spec.update(
        notebook_path=folder + '/kumo-test-lifecycle-' + spec['attempt'],
        experiment=folder + '/KUMO_TEST_MLOPS', output=config['volume'] +
        '/lifecycle/runs/' + spec['attempt'] + '/outputs')
    original = client.api
    experiment = {}

    def api(path, method='GET', body=None):
        if '/mlflow/experiments/' not in path:
            return original(path, method, body)
        client.calls.append((path, method, body))
        if '/get-by-name?' in path:
            if case == 'collision':
                return dict(experiment=dict(experiment_id='existing'))
            raise RuntimeError('Databricks API HTTP 404')
        if path.endswith('/create'):
            if case == 'ambiguous_create':
                raise RuntimeError('response lost')
            experiment.update(experiment_id='123', name=body['name'],
                              creation_time=int(cleanup.time.time() * 1000),
                              tags=body['tags'])
            return dict(experiment_id='123')
        if '/get?' in path:
            if case == 'changed_tag':
                experiment['tags'] = []
            return dict(experiment=experiment)
        assert path.endswith('/delete')
        assert body == dict(experiment_id='123')
        return {}

    client.api = api
    context = __import__(
        'contextlib').nullcontext() if case == 'success' else pytest.raises(
            (ValueError, RuntimeError))
    with context:
        with cleanup.AttemptResources(client, config, spec, root,
                                      lifecycle=True) as resources:
            resources.prepare_experiment()
    deleted = [
        p for p, method, _ in client.calls if p.endswith('/experiments/delete')
    ]
    assert bool(deleted) == (case == 'success')
    assert json.loads((root / 'cleanup.json').read_text())['status'] == (
        'passed' if case in ('success', 'collision') else 'failed')


def test_ambiguous_import_is_reported_without_deleting_notebook(attempt):
    client, config, spec, root = attempt
    with pytest.raises(RuntimeError, match='cleanup failed'):
        with cleanup.AttemptResources(client, config, spec, root) as resources:
            resources.importing()
    assert not any(
        path.endswith('/workspace/delete') for path, _, _ in client.calls)
    assert json.loads((root / 'cleanup.json').read_text())['status'] == 'failed'


def test_directory_entries_with_provider_trailing_slash_are_cleaned(attempt):
    client, config, spec, root = attempt
    resource = cleanup.AttemptResources(client, config, spec, root)
    prefix = spec['output'].removesuffix('/outputs')
    names = ('default.html', 'sampled.html', 'graph-result.json',
             'graph.md', 'graph.pdf', 'graph.svg')
    responses = {
        prefix: [dict(path=prefix + '/outputs/', is_directory=True)],
        prefix + '/outputs': [
            dict(path=prefix + '/outputs/' + name, is_directory=False)
            for name in names
        ],
    }

    def api(path, method='GET', body=None):
        client.calls.append((path, method, body))
        if method == 'GET':
            directory = path.partition('?')[0].removeprefix('/api/2.0/fs/directories')
            return dict(contents=responses[directory])
        return {}

    client.api = api
    resource.clear_directory(prefix, [10000])
    deleted = [path for path, method, _ in client.calls if method == 'DELETE']
    assert deleted == [
        '/api/2.0/fs/files' + prefix + '/outputs/' + name for name in names
    ] + ['/api/2.0/fs/directories' + prefix + '/outputs',
         '/api/2.0/fs/directories' + prefix]


@pytest.mark.parametrize('suffix,directory', [
    ('/outputs//', True), ('//outputs/', True), ('/../other/', True),
    ('/outputs/../other/', True), ('/outputs/file/', False),
    ('/outputs/other', False),
])
def test_cleanup_rejects_ambiguous_or_escaping_entries_before_delete(
        attempt, suffix, directory):
    client, config, spec, root = attempt
    resource = cleanup.AttemptResources(client, config, spec, root)
    prefix = spec['output'].removesuffix('/outputs')
    client.present = True
    client.listing = [dict(path=prefix + suffix, is_directory=directory)]
    with pytest.raises(ValueError, match='escaped'):
        resource.clear_directory(prefix, [10000])
    assert all(method == 'GET' for _, method, _ in client.calls)


@pytest.fixture
def registered_model(attempt):
    import copy

    client, config, spec, root = attempt
    folder = '/Shared/kumo_test_catalogue/KUMO_TEST/lifecycle'
    spec.update(notebook_path=folder + '/kumo-test-lifecycle-' + spec['attempt'],
                experiment=folder + '/KUMO_TEST_MLOPS',
                output=config['volume'] + '/lifecycle/runs/' + spec['attempt'] + '/outputs')
    resource = cleanup.AttemptResources(client, config, spec, root, lifecycle=True)
    created = resource.ledger['started_ms'] + 100
    model = dict(id='model-id', full_name=resource.model.lower(),
                 owner=config['expected_user'], created_at=created)
    parent = model['full_name'].split('.')

    class ModelClient:
        def __init__(self):
            self.calls = []
            self.model = copy.deepcopy(model)
            self.versions = {
                str(number): dict(version=number, id='version-' + str(number),
                                  model_name=parent[2], catalog_name=parent[0],
                                  schema_name=parent[1], created_at=created + number,
                                  created_by=config['expected_user'], status='READY')
                for number in (1, 2)
            }
            self.pages = None
            self.after_delete = None

        def api(self, path, method='GET', body=None):
            self.calls.append((path, method))
            endpoint = '/api/2.1/unity-catalog/models/' + model['full_name']
            if path.startswith(endpoint + '/versions?'):
                if self.pages:
                    return self.pages(path)
                return dict(model_versions=copy.deepcopy(list(self.versions.values())),
                            next_page_token='')
            if path.startswith(endpoint + '/versions/'):
                assert method == 'DELETE'
                del self.versions[path.rsplit('/', 1)[1]]
                if self.after_delete:
                    self.after_delete()
                return {}
            assert path == endpoint
            if method == 'DELETE':
                if self.versions:
                    raise RuntimeError('Databricks API HTTP 400: model is not empty')
                self.model = None
                return {}
            if self.model is None:
                raise RuntimeError('Databricks API HTTP 404')
            return copy.deepcopy(self.model)

    provider = ModelClient()
    resource.client = provider
    return resource, provider, model


def test_registered_model_cleanup_removes_versions_before_parent(registered_model):
    resource, provider, model = registered_model
    resource.remove_model(model)
    endpoint = '/api/2.1/unity-catalog/models/' + model['full_name']
    assert [p for p, method in provider.calls if method == 'DELETE'] == [
        endpoint + '/versions/1', endpoint + '/versions/2', endpoint]
    assert provider.model is None and provider.versions == {}


@pytest.mark.parametrize('change', [
    dict(version=3), dict(created_by='another-principal'), dict(created_at=0),
    dict(id=''), dict(model_name='production-model'),
    dict(status='PENDING_REGISTRATION'), dict(version=2),
])
def test_all_model_versions_are_admitted_before_any_delete(registered_model, change):
    resource, provider, model = registered_model
    provider.versions['1'].update(change)
    with pytest.raises(ValueError):
        resource.remove_model(model)
    assert all(method == 'GET' for _, method in provider.calls)


@pytest.mark.parametrize('change', ['parent-id', 'version-id', 'new-version'])
def test_model_cleanup_stops_if_identity_changes_between_deletions(
        registered_model, change):
    resource, provider, model = registered_model

    def change_identity():
        if change == 'parent-id':
            provider.model['id'] = 'new-model-id'
        elif change == 'version-id':
            provider.versions['2']['id'] = 'new-version-id'
        else:
            provider.versions['3'] = dict(provider.versions['2'], version=3)

    provider.after_delete = change_identity
    with pytest.raises(ValueError):
        resource.remove_model(model)
    deleted = [path for path, method in provider.calls if method == 'DELETE']
    assert len(deleted) == 1 and deleted[0].endswith('/versions/1')
    assert provider.model is not None


def test_model_version_pagination_continues_after_empty_page(registered_model):
    import copy
    from urllib.parse import parse_qs, urlsplit

    resource, provider, model = registered_model

    def pages(path):
        token = parse_qs(urlsplit(path).query).get('page_token')
        return (dict(model_versions=copy.deepcopy(list(provider.versions.values())))
                if token else dict(model_versions=[], next_page_token='next'))

    provider.pages = pages
    resource.remove_model(model)
    assert provider.model is None


@pytest.mark.parametrize('case', ['repeat', 'invalid', 'excess'])
def test_ambiguous_model_version_pagination_never_deletes(registered_model, case):
    resource, provider, model = registered_model
    page = 0

    def pages(path):
        nonlocal page
        page += 1
        token = False if case == 'invalid' else 'same' if case == 'repeat' else str(page)
        return dict(model_versions=[], next_page_token=token)

    provider.pages = pages
    with pytest.raises(ValueError, match='pagination'):
        resource.remove_model(model)
    assert all(method == 'GET' for _, method in provider.calls)


@pytest.mark.parametrize('numbers', [(), ('1',), ('2',)])
def test_partial_attempt_model_cleanup_accepts_only_remaining_expected_versions(
        registered_model, numbers):
    resource, provider, model = registered_model
    provider.versions = {k: v for k, v in provider.versions.items() if k in numbers}
    resource.remove_model(model)
    deleted = [path for path, method in provider.calls if method == 'DELETE']
    assert len(deleted) == len(numbers) + 1
    assert provider.model is None


def test_recovery_reuses_ledger_without_reacquiring_ownership(attempt):
    client, config, spec, root = attempt
    resources = cleanup.AttemptResources(client, config, spec, root)
    resources.__enter__()
    resources.notebook_created()
    (root / 'scenario.json').write_text(json.dumps(spec))
    client.present = True
    cleanup.recover(root, client, config)
    assert json.loads((root / 'cleanup.json').read_text())['status'] == 'passed'
    assert any(method == 'DELETE' for _, method, _ in client.calls)


def test_recovery_refuses_changed_scope(attempt):
    client, config, spec, root = attempt
    resources = cleanup.AttemptResources(client, config, spec, root)
    resources.__enter__()
    changed = dict(
        spec, attempt='b' * 12,
        notebook_path=spec['notebook_path'].replace('a' * 12, 'b' * 12),
        output=spec['output'].replace('a' * 12, 'b' * 12))
    (root / 'scenario.json').write_text(json.dumps(changed))
    client.calls.clear()
    with pytest.raises(ValueError, match='scope differs'):
        cleanup.recover(root, client, config)
    assert client.calls == []


def test_unknown_submit_keeps_original_failure(attempt):
    client, config, spec, root = attempt
    with pytest.raises(RuntimeError, match='connection failed'):
        with cleanup.AttemptResources(client, config, spec, root) as resources:
            resources.submitting()
            raise RuntimeError('connection failed')
    assert all(method == 'GET' for _, method, _ in client.calls)


def test_definitive_rejected_submit_can_clean_up(attempt):
    client, config, spec, root = attempt
    with pytest.raises(RuntimeError, match='HTTP 403'):
        with cleanup.AttemptResources(client, config, spec, root) as resources:
            resources.notebook_created()
            resources.submitting()
            resources.submitted()
            raise RuntimeError('Databricks API HTTP 403')
    assert json.loads((root / 'cleanup.json').read_text())['status'] == 'passed'


@pytest.mark.parametrize('foreign', [False, True])
def test_recovery_cancels_only_proven_owned_run(attempt, foreign):
    client, config, spec, root = attempt
    resources = cleanup.AttemptResources(client, config, spec, root)
    resources.__enter__()
    resources.notebook_created()
    (root / 'scenario.json').write_text(json.dumps(spec))
    (root / 'submitted-run.json').write_text('{"run_id": 7}')
    original = client.api
    active = True

    def api(path, method='GET', body=None):
        nonlocal active
        if '/jobs/runs/get?' in path:
            return dict(
                run_id=7, creator_user_name='other'
                if foreign else config['expected_user'],
                run_name='kumo-test-notebook-ui-' + spec['attempt'], tasks=[
                    dict(notebook_task=dict(
                        notebook_path=spec['notebook_path']))
                ], state=dict(
                    life_cycle_state='RUNNING' if active else 'TERMINATED'))
        if path.endswith('/runs/cancel'):
            active = False
        return original(path, method, body)

    client.api = api
    if foreign:
        with pytest.raises(ValueError, match='ownership'):
            cleanup.recover(root, client, config)
        assert all(method == 'GET' for _, method, _ in client.calls)
    else:
        cleanup.recover(root, client, config)
        assert not active
        assert json.loads(
            (root / 'cleanup.json').read_text())['status'] == 'passed'


def test_cleanup_failure_does_not_hide_primary_submit_error(attempt):
    client, config, spec, root = attempt
    with pytest.raises(RuntimeError, match='HTTP 403'):
        with cleanup.AttemptResources(client, config, spec, root) as resources:
            resources.notebook_created()
            client.fail_delete = True
            raise RuntimeError('Databricks API HTTP 403')
    assert json.loads((root / 'cleanup.json').read_text())['status'] == 'failed'
