# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from scripts import seed_playwright_local_stack as seed

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_PATH = (_REPO_ROOT / 'test/rest/integ/test_cases' /
                 'test_e2e_fixtures.json')
_CONNECTOR_ARGS_PATH = (_REPO_ROOT / 'test/rest/integ/connector_args' /
                        'test_e2e_connector_args.json')
_PLAYWRIGHT_WORKFLOW_PATH = (_REPO_ROOT / '.github/workflows' /
                             'playwright_ui.yml')


def test_playwright_workflow_declares_checked_in_seed_inputs():
    workflow = _PLAYWRIGHT_WORKFLOW_PATH.read_text(encoding='utf-8')
    fixture_relative = _FIXTURE_PATH.relative_to(_REPO_ROOT).as_posix()
    connector_args_relative = _CONNECTOR_ARGS_PATH.relative_to(
        _REPO_ROOT).as_posix()

    assert json.loads(_FIXTURE_PATH.read_text(encoding='utf-8'))
    assert json.loads(_CONNECTOR_ARGS_PATH.read_text(encoding='utf-8'))
    assert (
        'PLAYWRIGHT_SEED_FIXTURE: '
        f'${{{{ github.workspace }}}}/{fixture_relative}' in workflow)
    assert (
        'PLAYWRIGHT_SEED_CONNECTORS: '
        f'${{{{ github.workspace }}}}/{connector_args_relative}' in workflow)
    # Each seed input must participate in both workflow change filters and be
    # passed explicitly to the seeder.  This prevents a fixture move from
    # silently deriving a nonexistent sibling path after backend startup.
    assert workflow.count(fixture_relative) == 3
    assert workflow.count(connector_args_relative) == 3


def _selected_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(seed, 'ORIGINAL_FIXTURE_FPIDS_PATH',
                        tmp_path / 'original.json')
    monkeypatch.setattr(seed, 'LOCALIZED_FIXTURE_FPIDS_PATH',
                        tmp_path / 'localized.json')
    raw = json.loads(_FIXTURE_PATH.read_text(encoding='utf-8'))
    return seed.filtered_fixture(raw)


def test_filtered_fixture_keeps_only_selected_query_graph(
        monkeypatch, tmp_path):
    fixture = _selected_fixture(monkeypatch, tmp_path)

    assert [query['pquery_name']
            for query in fixture['queries']] == [seed.PQUERY_NAME]
    assert len(fixture['graphs']) == 1
    assert set(fixture['graphs'][0]['tables']) == {'ACCOUNT', 'LOAN', 'TRANS'}
    assert len(fixture['tables']) == 3


def test_every_selected_fixture_table_uses_relative_source_contract(
        monkeypatch, tmp_path):
    fixture = _selected_fixture(monkeypatch, tmp_path)
    sources = [
        table['source_table']
        for graph in fixture['graphs']
        for table in graph['tables'].values()
    ] + [table['source_table'] for table in fixture['tables']]

    assert sources
    for source in sources:
        assert source['source']['mode'] == 'relative'
        assert seed.source_table_name(source) == source['source']['path']


def test_seed_fixture_threads_project_ownership(monkeypatch):
    calls = []
    monkeypatch.setattr(
        seed, 'request_json', lambda method, path, data: calls.append(
            (method, path, data)))

    seed.seed_fixture()

    assert calls == [(
        'POST',
        '/v0/fixtures/load_json',
        {
            'file_path': str(seed.LOCALIZED_FIXTURE_PATH),
            'project_id': seed.PROJECT_ID,
        },
    )]


def test_seed_project_uploads_remaps_every_fixture_table(monkeypatch, tmp_path):
    fixture = _selected_fixture(monkeypatch, tmp_path)
    connector_args = json.loads(
        _CONNECTOR_ARGS_PATH.read_text(encoding='utf-8'))
    for connector in connector_args:
        relative_root = connector['config']['root_dir'].removeprefix('./')
        connector['config']['root_dir'] = str(_REPO_ROOT / relative_root)

    calls = []

    def request_json(method, path, data=None, **_kwargs):
        calls.append((method, path, copy.deepcopy(data)))
        if method == 'GET':
            return {'connector_id': f'connector_{"a" * 32}'}
        if path.endswith('/uploads/start'):
            return {
                'temp_upload_path': f'/tmp/{data["source_table_name"]}',
                'upload_id': data['source_table_name'],
                'presigned_part_urls': {
                    '1': f'/{data["source_table_name"]}'
                },
            }
        return {}

    uploaded_parts = []

    def request_bytes(method, path, data):
        uploaded_parts.append((method, path, data))
        return {'etag': f'etag-{path}'}

    monkeypatch.setattr(seed, 'request_json', request_json)
    monkeypatch.setattr(seed, 'request_bytes', request_bytes)

    seed.seed_project_uploads(fixture, connector_args)

    canonical_id = f'connector_{"a" * 32}'
    graph_sources = [
        table['source_table']
        for table in fixture['graphs'][0]['tables'].values()
    ]
    assert {source['connector_id']
            for source in graph_sources} == {canonical_id}
    assert {
        table['source_table']['connector_id']
        for table in fixture['tables']
    } == {canonical_id}
    assert {path
            for _, path, _ in uploaded_parts} == {
                '/ACCOUNT',
                '/LOAN',
                '/TRANS',
            }
    assert all(data for _, _, data in uploaded_parts)
    assert len(
        [call for call in calls if call[1].endswith('/uploads/complete')]) == 3


def test_seed_ui_resources_creates_missing_aliases_without_update_semantics(
        monkeypatch):
    calls = []
    monkeypatch.setattr(seed, 'fixture_target', lambda _path: type(
        'Target', (), {
            'graph': {
                'tables': {
                    'ACCOUNT': {
                        'name': 'ACCOUNT'
                    }
                }
            }
        })())
    monkeypatch.setattr(seed, 'as_json_dict', lambda value: value)
    monkeypatch.setattr(seed, 'list_names', lambda _path: [])
    monkeypatch.setattr(seed, 'require_names', lambda _path, _names: None)
    monkeypatch.setattr(
        seed, 'request_json', lambda method, path, data: calls.append(
            (method, path, data)) or {})

    seed.seed_ui_resources()

    assert len(calls) == 2
    for method, path, _ in calls:
        query = parse_qs(urlparse(path).query)
        assert method == 'POST'
        assert query['project_id'] == [seed.PROJECT_ID]
        assert 'force_rename' not in query


def test_list_training_jobs_threads_project_ownership(monkeypatch):
    calls = []
    monkeypatch.setattr(
        seed, 'request_json',
        lambda method, path, data, **_kwargs: calls.append(
            (method, path, data)) or {'jobs': []})

    assert seed.list_training_jobs_by_tag('playwright') == []
    assert calls == [(
        'POST',
        '/v0/jobs/',
        {
            'project_id': seed.PROJECT_ID,
            'page': 0,
            'limit': 50,
            'job_type': ['TRAINING_JOB'],
            'search_keyword': 'playwright',
        },
    )]
