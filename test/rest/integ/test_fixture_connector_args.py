# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from pathlib import Path
from test.rest.integ import public_api_smoke_helpers
from test.rest.integ.fixture_connector_args import (
    _copy_fixture_source,
    _project_upload_fixture,
    duplicate_fixture_sources,
    load_fixture_connectors,
    provision_fixture_connectors,
    provision_project_upload_connector,
)
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from kumoapi.source_table import FileType, S3URISource, SourceTableListResponse

from io_utils.json import load_json
from kumo.testing.rest import get_and_expect_success, randomize_fixture_ids_v2
from rest.config.api_command import FixtureDefinition
from rest.config.connector import FileConnectorResourceConfig
from statedb.config.common import ConnectorID, ProjectID


def _local_fixture() -> FixtureDefinition:
    fixture = load_json(
        'test/rest/integ/test_cases/test_e2e_fixtures.json',
        FixtureDefinition,
    )
    fixture.queries = fixture.queries[:1]
    return randomize_fixture_ids_v2(fixture)


def _fixture_client(upload):
    services = SimpleNamespace(
        connectors=SimpleNamespace(
            register_project_upload_connector=AsyncMock(
                return_value=upload.id),
            get_internal=AsyncMock(return_value=upload),
        ))
    return SimpleNamespace(
        state_services=services,
        params=httpx.QueryParams(),
        get=lambda _path: SimpleNamespace(
            status_code=200,
            json=lambda: {'connector_id': str(upload.id)},
        ),
    )


def test_smoke_resource_loads_repository_fixture(monkeypatch) -> None:
    connector = SimpleNamespace()
    provision = AsyncMock(
        return_value=(ProjectID('default'), [connector]),
    )
    monkeypatch.setattr(
        public_api_smoke_helpers,
        'provision_fixture_connectors',
        provision,
    )

    fixture, loaded_connector, graph, pquery = (
        public_api_smoke_helpers.provision_smoke_resource(
            SimpleNamespace(),
        ))

    assert loaded_connector is connector
    assert set(graph['tables']) == {'ACCOUNT', 'LOAN', 'TRANS'}
    assert pquery['graph'] is graph
    assert pquery['name'].startswith(
        public_api_smoke_helpers.PQUERY_CASE,
    )
    assert fixture.queries[0].pquery_name == pquery['name']
    provision.assert_awaited_once()


def test_local_fixture_remaps_to_namespaced_project_upload() -> None:
    fixture = _local_fixture()
    source_connectors = load_fixture_connectors(fixture,
                                                ProjectID('default'))
    roots = {
        connector.id: connector.config.root_dir
        for connector in source_connectors
    }
    upload_id = ConnectorID(f'connector_{"a" * 32}')

    remapped = _project_upload_fixture(fixture, upload_id, roots)

    assert {
        table.get_connector_id()
        for graph in remapped.graphs
        for table in graph.tables.values()
    } == {upload_id}
    assert all(
        table.source_table.source.path.startswith('fixture-data/connector_')
        for graph in remapped.graphs
        for table in graph.tables.values()
    )
    assert remapped.graphs[0].fpid != fixture.graphs[0].fpid
    assert remapped.queries[0].graph_fpid == remapped.graphs[0].fpid


def test_local_fixture_rejects_full_s3_uri() -> None:
    fixture = _local_fixture()
    graph = fixture.graphs[0]
    table_name, table = next(iter(graph.tables.items()))
    uri_table = dataclasses.replace(
        table,
        source_table=dataclasses.replace(
            table.source_table,
            source=S3URISource(uri='s3://customer-bucket/table'),
        ),
    )
    fixture.graphs[0] = dataclasses.replace(
        graph,
        tables={**graph.tables, table_name: uri_table},
    )
    source_connectors = load_fixture_connectors(fixture,
                                                ProjectID('default'))

    with pytest.raises(ValueError, match='supported external S3 connector'):
        _project_upload_fixture(
            fixture,
            ConnectorID(f'connector_{"a" * 32}'),
            {
                connector.id: connector.config.root_dir
                for connector in source_connectors
            },
        )


@pytest.mark.asyncio
async def test_local_fixture_preserves_s3_upload_storage(monkeypatch) -> None:
    fixture = _local_fixture()
    upload_id = ConnectorID(f'connector_{"a" * 32}')
    upload = SimpleNamespace(
        id=upload_id,
        config=SimpleNamespace(
            root_dir='s3://diskgraph-test/uploaded-datasets/default'),
    )
    copy = AsyncMock()
    monkeypatch.setattr('test.rest.integ.fixture_connector_args.fsutil.copy',
                        SimpleNamespace(coroutine=copy))

    project_id, connectors = await provision_fixture_connectors(
        _fixture_client(upload), fixture, ProjectID('default'))

    assert project_id == ProjectID('default')
    assert connectors == [upload]
    assert copy.await_count == 3
    assert all(
        call.args[1].startswith(
            's3://diskgraph-test/uploaded-datasets/default/fixture-data/')
        for call in copy.await_args_list
    )
    assert {
        table.get_connector_id()
        for graph in fixture.graphs
        for table in graph.tables.values()
    } == {upload_id}


@pytest.mark.asyncio
async def test_public_e2e_duplicate_sources_preserve_table_basenames(
    tmp_path,
) -> None:
    fixture = _local_fixture()
    upload_id = ConnectorID(f'connector_{"a" * 32}')
    upload_root = tmp_path / 'uploaded-datasets' / 'default'
    upload = SimpleNamespace(
        id=upload_id,
        config=FileConnectorResourceConfig(
            name=str(upload_id),
            root_dir=str(upload_root),
        ),
    )
    project_id, connectors = await provision_fixture_connectors(
        _fixture_client(upload), fixture, ProjectID('default'))
    duplicate_paths = await duplicate_fixture_sources(connectors[0], fixture)

    assert project_id == ProjectID('default')
    graph_sources = {
        table.source_table.source.path
        for graph in fixture.graphs
        for table in graph.tables.values()
    }
    assert duplicate_paths.keys() == graph_sources
    account_path = next(path for path in graph_sources
                        if path.endswith('/ACCOUNT'))
    assert duplicate_paths[account_path].endswith('/ACCOUNT-duplicate')
    assert (upload_root / f'{duplicate_paths[account_path]}.parquet').is_file()
    assert not any(path.name.startswith('FINANCIAL-duplicate')
                   for path in upload_root.rglob('*'))

    file_connector = upload.config.as_connector()
    for graph in fixture.graphs:
        for table in graph.tables.values():
            source_table = table.source_table
            original = source_table.source.path
            duplicate = duplicate_paths[original]
            original_path, original_type = (
                await file_connector.get_path_and_type.coroutine(original))
            duplicate_path, duplicate_type = (
                await file_connector.get_path_and_type.coroutine(duplicate))
            assert duplicate_type == original_type
            assert duplicate_path.endswith(f'.{original_type}')
            assert int(Path(original_path).stat().st_mtime) != int(
                Path(duplicate_path).stat().st_mtime)

    # The extensionless fixture path contains both TRANS.parquet and
    # TRANS.csv. CompositeFileConnector intentionally chooses Parquet. That
    # distinction is semantic: its K_SYMBOL blanks are true nulls, while the
    # CSV ingestion path represents them as empty strings. The modified graph
    # must preserve the 3276 prediction candidates selected by ``IS NULL``.
    trans_path = next(path for path in graph_sources
                      if path.endswith('/TRANS'))
    original_trans, original_type = (
        await file_connector.get_path_and_type.coroutine(trans_path))
    duplicate_trans, duplicate_type = (
        await file_connector.get_path_and_type.coroutine(
            duplicate_paths[trans_path]))
    assert original_type == duplicate_type == 'parquet'
    original_df = pd.read_parquet(original_trans)
    duplicate_df = pd.read_parquet(duplicate_trans)
    assert original_df.equals(duplicate_df)
    assert duplicate_df['K_SYMBOL'].isna().sum() == 3276
    assert (duplicate_df['K_SYMBOL'] == '').sum() == 0


@pytest.mark.asyncio
async def test_public_e2e_duplicate_source_preserves_nested_parquet_path(
    tmp_path,
) -> None:
    fixture = _local_fixture()
    graph = fixture.graphs[0]
    table_name, table = next(iter(graph.tables.items()))
    nested_source = dataclasses.replace(
        table.source_table,
        source=dataclasses.replace(
            table.source_table.source,
            path='many/layers/events.parquet',
        ),
        file_type=FileType.PARQUET,
    )
    fixture.graphs[:] = [
        dataclasses.replace(
            graph,
            tables={
                table_name:
                dataclasses.replace(table, source_table=nested_source),
            },
        )
    ]
    upload_root = tmp_path / 'uploaded-datasets' / 'default'
    source_path = upload_root / 'many' / 'layers' / 'events.parquet'
    source_path.parent.mkdir(parents=True)
    pd.DataFrame({'id': [1, 2]}).to_parquet(source_path, index=False)
    upload = SimpleNamespace(
        config=FileConnectorResourceConfig(
            name=f'connector_{"b" * 32}',
            root_dir=str(upload_root),
        ))

    duplicate_paths = await duplicate_fixture_sources(upload, fixture)

    assert duplicate_paths == {
        'many/layers/events.parquet':
        'many/layers/events-duplicate.parquet',
    }
    duplicate_path = upload_root / 'many/layers/events-duplicate.parquet'
    assert duplicate_path.is_file()
    assert int(source_path.stat().st_mtime) != int(
        duplicate_path.stat().st_mtime)
    assert pd.read_parquet(duplicate_path).to_dict('list') == {'id': [1, 2]}


@pytest.mark.asyncio
async def test_fixture_copy_accepts_distinct_s3_data_version(monkeypatch) -> None:
    copy = AsyncMock()
    versions = {
        's3://bucket/source.csv': 100.25,
        's3://bucket/source-duplicate.csv': 101.25,
    }
    monkeypatch.setattr('test.rest.integ.fixture_connector_args.fsutil.copy',
                        SimpleNamespace(coroutine=copy))
    monkeypatch.setattr(
        'test.rest.integ.fixture_connector_args.fsutil.get_last_modified_time',
        lambda path: versions[path],
    )
    monkeypatch.setattr('test.rest.integ.fixture_connector_args.fsutil.is_local',
                        lambda _path: False)

    await _copy_fixture_source(
        's3://bucket/source.csv',
        's3://bucket/source-duplicate.csv',
    )

    copy.assert_awaited_once_with(
        's3://bucket/source.csv',
        's3://bucket/source-duplicate.csv',
        overwrite=True,
        raise_on_exists=False,
    )


@pytest.mark.parametrize(
    ('client_fixture', 'api_prefix', 'checks_source_tables'),
    (('client', '/v0', True), ('public_api_client', '/v1', False)),
)
@pytest.mark.asyncio
async def test_project_upload_fixture_is_hidden_on_private_and_public_routes(
    request,
    tmp_path,
    client_fixture,
    api_prefix,
    checks_source_tables,
) -> None:
    client: TestClient = request.getfixturevalue(client_fixture)
    source_root = tmp_path / f'{api_prefix[1:]}-fixture-data'
    source_root.mkdir()
    pd.DataFrame({'id': [1, 2]}).to_parquet(
        source_root / 'events.parquet', index=False)

    project_id, connector = await provision_project_upload_connector(
        client,
        source_root=str(source_root),
        api_prefix=api_prefix,
    )

    suffix = '/' if api_prefix == '/v0' else ''
    listed = client.get(
        f'{api_prefix}/projects/{project_id}/connectors{suffix}')
    assert listed.status_code == 200
    assert str(connector.id) not in {
        item['id'] for item in listed.json()
    }
    if checks_source_tables:
        source_tables = get_and_expect_success(
            client,
            (f'{api_prefix}/projects/{project_id}/connectors/{connector.id}/'
             'source-tables'),
            {},
            SourceTableListResponse,
        )
        assert 'events' in source_tables.table_names


@pytest.mark.asyncio
async def test_public_api_fixture_uploads_only_its_resolved_input_bytes(
        tmp_path, monkeypatch):
    from test.rest.integ import fixture_connector_args as uploads
    from test.rest.integ.fixture_connector_args import load_selected_fixture

    fixture = load_selected_fixture(
        'test/rest/integ/test_cases/test_e2e_fixtures.json',
        'saas_temporal_regression_base')
    assert len(fixture.queries) == len(fixture.graphs) == 1
    assert set(fixture.graphs[0].tables) == {'ACCOUNT', 'LOAN', 'TRANS'}
    assert len(fixture.tables) == 3
    connector = SimpleNamespace(
        id=ConnectorID(f'connector_{"b" * 32}'),
        config=SimpleNamespace(root_dir=str(tmp_path / 'uploads')))
    monkeypatch.setattr(uploads, 'ensure_test_project',
                        AsyncMock(return_value=ProjectID('default')))
    monkeypatch.setattr(uploads, '_canonical_project_upload_connector',
                        AsyncMock(return_value=connector))
    monkeypatch.setattr(uploads, 'bind_test_client_project', lambda *_: None)
    await uploads.provision_fixture_connectors(None, fixture,
                                               selected_sources_only=True)
    copied = sorted(path for path in (tmp_path / 'uploads').rglob('*')
                    if path.is_file())
    assert sorted(path.name for path in copied) == [
        'ACCOUNT.parquet', 'LOAN.parquet', 'TRANS.parquet'
    ]
    expected = {
        'ACCOUNT.parquet': Path('test/csv_data/FINANCIAL/ACCOUNT.parquet'),
        'TRANS.parquet': Path('test/csv_data/FINANCIAL/TRANS.parquet'),
        'LOAN.parquet': Path('test/csv_data/FINANCIAL_E2E/LOAN.parquet'),
    }
    for path in copied:
        assert path.read_bytes() == expected[path.name].read_bytes()
