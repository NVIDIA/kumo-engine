# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
from datetime import datetime
from hashlib import md5
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import kumoapi.data_snapshot as snapshot_interfaces
import pytest
from kumo_core.types import ConnectorType
from kumo_core.types.schema import FPID
from kumoapi.common import JobStatus, ValidationWarning
from kumoapi.data_source import DataSourceType
from kumoapi.typing import Dtype, Stype

from kumo.config.graph import GraphConfig, TableConfig
from kumo.connector.connector import ConnectorError
from kumo.data.backend_cache import (
    AsyncDataTaskStatus,
    TableDataTaskResult,
    TableRefreshStage,
)
from kumo.data.custom_sql import custom_sql_synthetic_name
from kumo.validation import ValidationResponse
from rest.backends.data_backend.data_backend import DataBackend
from rest.config.connector import FileConnectorResourceConfig
from rest.config.connector_db import ConnectorDBEntry, InternalConnectorDBEntry
from rest.config.data_backend import (
    ConnectorExecutionRef,
    GraphDataVersionInfo,
)
from rest.config.file_type import FileType
from rest.config.graph_v2 import (
    GraphDataComputeInput,
    GraphDBEntryV2,
    GraphDefinition,
    GraphSnapshotRequest,
)
from rest.config.table_v2 import (
    Column,
    CustomSQLSourceTable,
    S3SourceTable,
    TableDBEntryV2,
    TableDefinition,
)
from rest.driver.data_snapshot_api_driver import (
    DataSnapshotApiDriver,
    _hydrate_graph_config_source_from_graph_def,
    _merge_table_ingest_job_states,
    _PreparedTableRefresh,
)
from statedb.config.common import (
    ConnectorID,
    DataSnapshot,
    GraphIDv2,
    ProjectID,
    TableIDv2,
    TableName,
)
from statedb.config.table import TableDataVersion
from statedb.config.timestamp import ResourceTimestamps
from statedb.rest.connector_authorization import CONNECTOR_GRAPH_ACTIONS

_PROJECT_ID = ProjectID('project_' + 'a' * 32)


@pytest.fixture(autouse=True)
def auto_cleanup_per_func(monkeypatch):
    monkeypatch.setattr(DataSnapshotApiDriver, "require_project_access",
                        AsyncMock())
    yield


@pytest.fixture
def setup_rest_test_session() -> None:
    return None


def _connector_id(value) -> ConnectorID | None:
    if value is None:
        return None
    try:
        return ConnectorID(str(value))
    except ValueError:
        return ConnectorID('connector_' + md5(str(value).encode()).hexdigest())


def _make_s3_table(*, connector_id, source_table_name):
    return TableDefinition(
        cols=[Column(name='id', stype=Stype.ID, dtype=Dtype.int)],
        source_table=S3SourceTable(
            data_source_type=DataSourceType.S3,
            connector_id=_connector_id(connector_id),
            source={
                'mode': 'relative',
                'path': source_table_name
            },
            file_type=FileType.PARQUET,
        ),
    )


def _make_table_entry(table: TableDefinition,
                      alias: str | None = None) -> TableDBEntryV2:
    return TableDBEntryV2(
        id=table.fpid,
        project_id=_PROJECT_ID,
        alias=alias,
        config=table,
        col_dtypes={},
        timestamps=ResourceTimestamps(
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    )


def _make_file_connector_entry(connector_id: str,
                               root_dir: str) -> ConnectorDBEntry:
    normalized_id = _connector_id(connector_id)
    assert normalized_id is not None
    return InternalConnectorDBEntry(
        id=normalized_id,
        project_id=_PROJECT_ID,
        name=str(normalized_id),
        normalized_name=str(normalized_id).casefold(),
        config=FileConnectorResourceConfig(
            name=str(normalized_id),
            root_dir=root_dir,
        ),
        timestamps=ResourceTimestamps(
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
        description='',
    )


class _TableService:
    def __init__(self, entries):
        self.entries = entries
        self.get_calls = []
        self.get_definition_calls = []
        self.get_many_calls = []
        self.save_calls = []

    async def get_table_entry(self, project_id, fpid):
        self.get_calls.append((project_id, fpid))
        return self.entries.get(fpid)

    async def get_table_definition(self, project_id, fpid):
        self.get_definition_calls.append((project_id, fpid))
        entry = self.entries.get(fpid)
        return entry.config if entry is not None else None

    async def get_table_entries(self, project_id, fpids):
        fpids = list(fpids)
        self.get_many_calls.append((project_id, fpids))
        return {
            fpid: self.entries[fpid]
            for fpid in fpids if fpid in self.entries
        }

    async def save_table_definition(self, project_id, table_definition,
                                    **kwargs):
        self.save_calls.append((project_id, table_definition, kwargs))
        entry = _make_table_entry(table_definition)
        self.entries[entry.id] = entry
        return entry


class _ConnectorService:
    def __init__(self, connectors):
        self.connectors = {connector.id: connector for connector in connectors}
        self.list_calls = []
        self.get_calls = []
        self.get_many_calls = []
        self.get_required_calls = []
        self.get_spec_calls = []
        self.execution_ref_calls = []
        self.fresh_execution_ref_calls = []
        self._fresh_token_counter = 0

    async def list(self, **kwargs):
        self.list_calls.append(kwargs)
        connectors = list(self.connectors.values())
        root_dir = kwargs.get('root_dir')
        root_dirs = kwargs.get('root_dirs')
        if root_dir is not None:
            connectors = [
                connector for connector in connectors
                if getattr(connector.config, 'root_dir', None) == root_dir
            ]
        if root_dirs is not None:
            connectors = [
                connector for connector in connectors
                if getattr(connector.config, 'root_dir', None) in root_dirs
            ]
        return connectors

    async def get(self, connector_id):
        self.get_calls.append(connector_id)
        return self.connectors.get(connector_id)

    async def get_many(self, connector_ids):
        connector_ids = set(connector_ids)
        self.get_many_calls.append(connector_ids)
        return {
            connector_id: self.connectors[connector_id]
            for connector_id in connector_ids if connector_id in self.connectors
        }

    async def get_required(self, connector_id):
        self.get_required_calls.append(connector_id)
        return self.connectors[connector_id]

    async def get_connector_spec_from_id(self, connector_id):
        self.get_spec_calls.append(connector_id)
        return self.connectors[connector_id].as_connector_spec()

    async def execution_ref_for_project(self, principal, project_id,
                                        connector_id, actions):
        self.execution_ref_calls.append(
            (principal, project_id, connector_id, actions))
        return ConnectorExecutionRef(project_id=project_id,
                                     connector_id=connector_id)

    async def fresh_execution_refs_for_workflow(self, principal, references,
                                                actions):
        references = tuple(references)
        self.fresh_execution_ref_calls.append((principal, references, actions))
        fresh = []
        for reference in references:
            self._fresh_token_counter += 1
            fresh.append(
                dataclasses.replace(
                    reference,
                    authorization_token=f'fresh-{self._fresh_token_counter}',
                ))
        return tuple(fresh)


class _GraphService:
    def __init__(self, definitions, entries=None):
        self.definitions = definitions
        self.entries = entries or {}
        self.get_entry_calls = []
        self.get_definition_calls = []

    async def get_graph_entry(self, project_id, fpid):
        self.get_entry_calls.append((project_id, fpid))
        return self.entries.get(fpid)

    async def get_graph_definition(self, project_id, fpid):
        self.get_definition_calls.append((project_id, fpid))
        return self.definitions.get(fpid)

    async def list_graphs(self, project_id):
        return list(self.entries.values())

    async def get_graph_cache_input(self, graph_id, project_id):
        return await self.get_graph_data_compute_input(graph_id,
                                                       project_id=project_id)


def test_hydrate_graph_config_source_from_graph_def_backfills_missing_fields():
    graph_def = GraphDefinition(
        tables={
            'LOAN':
            _make_s3_table(
                connector_id='financial_conn',
                source_table_name='LOAN',
            ),
        },
        col_groups=[],
    )
    graph_config = GraphConfig(
        name='g',
        tables=[TableConfig(table_name='LOAN')],
    )

    hydrated = _hydrate_graph_config_source_from_graph_def(
        graph_config, graph_def)

    assert hydrated['LOAN'].data_source == str(_connector_id('financial_conn'))
    assert hydrated['LOAN'].source_name == 'LOAN'


def test_hydrate_graph_config_source_from_graph_def_preserves_existing_values():
    graph_def = GraphDefinition(
        tables={
            'LOAN':
            _make_s3_table(
                connector_id='financial_conn',
                source_table_name='LOAN',
            ),
        },
        col_groups=[],
    )
    graph_config = GraphConfig(
        name='g',
        tables=[
            TableConfig(
                table_name='LOAN',
                data_source='existing_conn',
                source_name='existing_source',
            )
        ],
    )

    hydrated = _hydrate_graph_config_source_from_graph_def(
        graph_config, graph_def)

    assert hydrated['LOAN'].data_source == 'existing_conn'
    assert hydrated['LOAN'].source_name == 'existing_source'


@pytest.mark.asyncio
@patch('rest.driver.data_snapshot_api_driver.TemporalClient')
async def test_get_table_definition_uses_state_services(mock_temporal_client):
    table = _make_s3_table(
        connector_id='connector_a',
        source_table_name='ACCOUNT',
    )
    entry = _make_table_entry(table)
    table_service = _TableService({entry.id: entry})
    data_backend_ctx = MagicMock()
    data_backend_ctx._config = MagicMock()
    driver = DataSnapshotApiDriver(data_backend_ctx)
    driver._configure_state_services(
        SimpleNamespace(tables=table_service, connectors=MagicMock()))

    got = await driver.get_table_definition(
        DataSnapshot.TABLE.format_id(entry.id, TableDataVersion('v1')),
        _PROJECT_ID,
    )

    assert got == table
    assert table_service.get_calls == [(_PROJECT_ID, entry.id)]


@pytest.mark.asyncio
@patch('rest.driver.data_snapshot_api_driver.TemporalClient')
async def test_get_graph_by_fpid_uses_state_services(mock_temporal_client):
    table = _make_s3_table(
        connector_id='connector_a',
        source_table_name='ACCOUNT',
    )
    graph_def = GraphDefinition(tables={'ACCOUNT': table}, col_groups=[])
    graph_service = _GraphService({graph_def.fpid: graph_def})
    data_backend_ctx = MagicMock()
    data_backend_ctx._config = MagicMock()
    driver = DataSnapshotApiDriver(data_backend_ctx)
    driver._configure_state_services(
        SimpleNamespace(
            graphs=graph_service,
            tables=MagicMock(),
            connectors=MagicMock(),
        ))

    graph_id, got = await driver._get_graph_by_fpid(_PROJECT_ID,
                                                    FPID(graph_def.fpid))

    assert graph_id == GraphIDv2.compose(fpid=graph_def.fpid)
    assert got == graph_def
    assert graph_service.get_definition_calls == [(_PROJECT_ID, graph_def.fpid)]


@pytest.mark.asyncio
@patch('rest.driver.data_snapshot_api_driver.TemporalClient')
@patch('rest.driver.data_snapshot_api_driver.get_graph_snapshot_info',
       new_callable=AsyncMock)
async def test_get_graph_snapshot_uses_snapshot_info(
    mock_get_graph_snapshot_info,
    mock_temporal_client,
):
    table = _make_s3_table(
        connector_id='connector_a',
        source_table_name='ACCOUNT',
    )
    graph_def = GraphDefinition(tables={TableName('ACCOUNT'): table},
                                col_groups=[])
    graph_id = GraphIDv2.compose(fpid=graph_def.fpid)
    graph_snapshot_id = DataSnapshot.GRAPH.format_id(graph_def.fpid, 'gv1')
    version_info = GraphDataVersionInfo(
        graph_data_version='gv1',
        table_data_versions={TableName('ACCOUNT'): TableDataVersion('tv1')},
        graph_name='Graph Alias',
    )
    data_backend_ctx = MagicMock()
    data_backend_ctx._config = MagicMock()
    driver = DataSnapshotApiDriver(data_backend_ctx)
    driver._configure_state_services(SimpleNamespace())
    driver._get_many_kumo_table_snapshots = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            snapshot_interfaces.TableSnapshotResource(
                id=DataSnapshot.TABLE.format_id(table.fpid,
                                                TableDataVersion('tv1')),
                table_definition=table,
                stages={
                    snapshot_interfaces.TableSnapshotStage.INGEST:
                    snapshot_interfaces.JobState(JobStatus.DONE),
                },
            )
        ])
    mock_get_graph_snapshot_info.return_value = (graph_id, version_info)

    response = await driver.get_graph_snapshot(
        GraphSnapshotRequest(
            graph_definiton=graph_def,
            graph_version='gv1',
        ),
        _PROJECT_ID,
    )

    assert response.snapshot_resource.id == graph_snapshot_id
    assert response.snapshot_resource.graph_name == 'Graph Alias'
    assert response.snapshot_resource.table_ids == {
        TableName('ACCOUNT'):
        DataSnapshot.TABLE.format_id(table.fpid, TableDataVersion('tv1'))
    }
    mock_get_graph_snapshot_info.assert_awaited_once_with(graph_snapshot_id)


@pytest.mark.asyncio
@patch('rest.driver.data_snapshot_api_driver.TemporalClient')
@patch('rest.driver.data_snapshot_api_driver.get_graph_snapshot_info',
       new_callable=AsyncMock)
@patch('rest.driver.data_snapshot_api_driver.DataBackend.get_edge_stats',
       new_callable=AsyncMock)
async def test_get_ui_edge_health_uses_state_services_without_opening_dao(
    mock_get_edge_stats,
    mock_get_graph_snapshot_info,
    mock_temporal_client,
):
    table = _make_s3_table(
        connector_id='connector_a',
        source_table_name='ACCOUNT',
    )
    entry = _make_table_entry(table)
    connector = _make_file_connector_entry('connector_a', 's3://bucket-a/path')
    graph_entry = GraphDBEntryV2(
        id=FPID('a' * 32),
        project_id=_PROJECT_ID,
        table_name_to_dbid={TableName('ACCOUNT'): entry.id},
        col_groups=[],
        alias='Graph Alias',
        timestamps=ResourceTimestamps(
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    )
    graph_id = GraphIDv2.compose(fpid=graph_entry.id)
    version_info = MagicMock()
    mock_get_edge_stats.return_value = None
    table_id = TableIDv2.compose(None, TableName('ACCOUNT'), entry.id)
    graph_input = GraphDataComputeInput(
        graph_display_name='Graph Alias',
        graph_key=GraphIDv2.compose(None, graph_entry.id),
        tables={table_id: table},
        base_tables={},
        connectors={
            connector.id:
            ConnectorExecutionRef(project_id=_PROJECT_ID,
                                  connector_id=connector.id)
        },
        graph_definition=GraphDefinition(
            tables={TableName('ACCOUNT'): table},
            col_groups=[],
        ),
        table_aliases={table_id: TableName('ACCOUNT')},
    )
    graph_service = _GraphService({}, {graph_entry.id: graph_entry})
    graph_service.get_graph_data_compute_input = AsyncMock(
        return_value=graph_input)
    table_service = _TableService({entry.id: entry})
    connector_service = _ConnectorService([connector])
    data_backend_ctx = MagicMock()
    data_backend_ctx._config = MagicMock()
    driver = DataSnapshotApiDriver(data_backend_ctx)
    driver._configure_state_services(
        SimpleNamespace(
            graphs=graph_service,
            tables=table_service,
            connectors=connector_service,
        ))
    mock_get_graph_snapshot_info.return_value = (graph_id, version_info)

    result = await driver.get_ui_edge_health(f'graph-{graph_entry.id}-v1',
                                             _PROJECT_ID)

    assert result.is_ready is False
    mock_get_graph_snapshot_info.assert_awaited_once_with(
        f'graph-{graph_entry.id}-v1')
    graph_service.get_graph_data_compute_input.assert_awaited_once_with(
        graph_id, project_id=_PROJECT_ID)
    assert graph_service.get_entry_calls == []
    assert table_service.get_many_calls == []
    assert connector_service.get_calls == []
    mock_get_edge_stats.assert_awaited_once()
    _, graph_input, got_version_info = mock_get_edge_stats.await_args.args
    assert got_version_info is version_info
    assert graph_input.graph_display_name == 'Graph Alias'
    assert graph_input.graph_key == GraphIDv2.compose(None, graph_entry.id)
    table_id = TableIDv2.compose(None, TableName('ACCOUNT'), entry.id)
    assert graph_input.tables == {table_id: table}
    assert graph_input.table_aliases == {table_id: TableName('ACCOUNT')}
    assert graph_input.graph_definition == GraphDefinition(
        tables={TableName('ACCOUNT'): table},
        col_groups=[],
    )


@pytest.mark.asyncio
@patch('rest.driver.data_snapshot_api_driver.TemporalClient')
@patch('rest.driver.data_snapshot_api_driver.DataBackend.start_refresh_graph',
       new_callable=AsyncMock)
@patch(
    'rest.driver.data_snapshot_api_driver.DataBackendActivities.save_graph_data_version',
    new_callable=AsyncMock)
@patch(
    'rest.driver.data_snapshot_api_driver.DataBackend.refresh_graph_data_version_in_activity',
    new_callable=AsyncMock)
async def test_create_graph_snapshot_uses_state_services_for_graph_metadata(
    mock_refresh_graph_data_version,
    mock_save_graph_data_version,
    mock_start_refresh_graph,
    mock_temporal_client,
):
    table = _make_s3_table(
        connector_id='connector_a',
        source_table_name='ACCOUNT',
    )
    entry = _make_table_entry(table)
    connector = _make_file_connector_entry('connector_a', 's3://bucket-a/path')
    graph_entry = GraphDBEntryV2(
        id=FPID('b' * 32),
        project_id=_PROJECT_ID,
        table_name_to_dbid={TableName('ACCOUNT'): entry.id},
        col_groups=[],
        alias='Graph Alias',
        timestamps=ResourceTimestamps(
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    )
    graph_def = GraphDefinition(tables={TableName('ACCOUNT'): table},
                                col_groups=[])
    table_id = TableIDv2.compose(None, TableName('ACCOUNT'), entry.id)
    graph_input = GraphDataComputeInput(
        graph_display_name='Graph Alias',
        graph_key=GraphIDv2.compose(None, graph_entry.id),
        tables={table_id: table},
        base_tables={},
        connectors={
            connector.id:
            ConnectorExecutionRef(project_id=_PROJECT_ID,
                                  connector_id=connector.id)
        },
        graph_definition=graph_def,
        table_aliases={table_id: TableName('ACCOUNT')},
    )
    version_info = SimpleNamespace(graph_data_version='gv1', graph_name=None)
    mock_refresh_graph_data_version.return_value = version_info

    graph_service = _GraphService({graph_entry.id: graph_def},
                                  {graph_entry.id: graph_entry})
    table_service = _TableService({entry.id: entry})
    connector_service = _ConnectorService([connector])
    graph_service.get_graph_data_compute_input = AsyncMock(
        return_value=graph_input)
    data_backend_ctx = MagicMock()
    data_backend_ctx._config = MagicMock()
    driver = DataSnapshotApiDriver(data_backend_ctx)
    driver._configure_state_services(
        SimpleNamespace(
            graphs=graph_service,
            tables=table_service,
            connectors=connector_service,
        ))
    driver._create_many_table_snapshots = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            DataSnapshot.TABLE.format_id(entry.id, TableDataVersion('v1'))
        ])

    snapshot_id = await driver.create_graph_snapshot(
        graph_entry.id,
        refresh_source=False,
        base_graph_name='Base Graph',
        project_id=_PROJECT_ID,
    )

    assert snapshot_id == DataSnapshot.GRAPH.format_id(graph_entry.id, 'gv1')
    assert graph_service.get_definition_calls == [(_PROJECT_ID, graph_entry.id)]
    assert graph_service.get_entry_calls == [(_PROJECT_ID, graph_entry.id)]
    assert table_service.get_many_calls == []
    driver._create_many_table_snapshots.assert_awaited_once_with(  # type: ignore[attr-defined]
        [table],
        refresh_source=False,
        project_id=_PROJECT_ID,
        defer_col_stats_validation=True,
    )
    assert graph_service.get_graph_data_compute_input.await_count == 1
    for graph_call in graph_service.get_graph_data_compute_input.await_args_list:
        assert graph_call.kwargs == {
            'project_id': _PROJECT_ID,
            'user_authorization': None,
        }
    assert connector_service.get_calls == []
    assert connector_service.list_calls == []
    assert version_info.graph_name == 'Graph Alias'
    assert len(connector_service.fresh_execution_ref_calls) == 1

    mock_refresh_graph_data_version.assert_awaited_once()
    assert mock_refresh_graph_data_version.await_args.args[
        0] is data_backend_ctx
    graph_input = mock_refresh_graph_data_version.await_args.args[1]
    table_id = TableIDv2.compose(None, TableName('ACCOUNT'), entry.id)
    assert graph_input.graph_definition == graph_def
    assert graph_input.pinned_data_versions == {
        table_id: TableDataVersion('v1')
    }
    mock_save_graph_data_version.assert_awaited_once_with(
        graph_input, version_info)
    mock_start_refresh_graph.assert_awaited_once()
    started_graph_input = mock_start_refresh_graph.await_args.args[1]
    assert next(iter(started_graph_input.connectors.values())
                ).authorization_token == 'fresh-1'


@pytest.mark.asyncio
@patch('rest.driver.data_snapshot_api_driver.TemporalClient')
@patch('rest.driver.data_snapshot_api_driver.DataBackend.start_refresh_graph',
       new_callable=AsyncMock)
@patch(
    'rest.driver.data_snapshot_api_driver.DataBackendActivities.save_graph_data_version',
    new_callable=AsyncMock)
@patch(
    'rest.driver.data_snapshot_api_driver.DataBackend.refresh_graph_data_version_in_activity',
    new_callable=AsyncMock)
async def test_create_graph_snapshot_missing_graph_entry_uses_base_graph_name_without_legacy_dao(
    mock_refresh_graph_data_version,
    mock_save_graph_data_version,
    mock_start_refresh_graph,
    mock_temporal_client,
):
    table = _make_s3_table(
        connector_id='connector_a',
        source_table_name='ACCOUNT',
    )
    entry = _make_table_entry(table)
    connector = _make_file_connector_entry('connector_a', 's3://bucket-a/path')
    graph_def = GraphDefinition(tables={TableName('ACCOUNT'): table},
                                col_groups=[])
    table_id = TableIDv2.compose(None, TableName('ACCOUNT'), entry.id)
    graph_input = GraphDataComputeInput(
        graph_display_name='Base Graph',
        graph_key=GraphIDv2.compose(None, graph_def.fpid),
        tables={table_id: table},
        base_tables={},
        connectors={
            connector.id:
            ConnectorExecutionRef(project_id=_PROJECT_ID,
                                  connector_id=connector.id)
        },
        graph_definition=graph_def,
        table_aliases={table_id: TableName('ACCOUNT')},
    )
    version_info = SimpleNamespace(graph_data_version='gv1', graph_name=None)
    mock_refresh_graph_data_version.return_value = version_info

    graph_service = _GraphService({graph_def.fpid: graph_def}, {})
    table_service = _TableService({entry.id: entry})
    connector_service = _ConnectorService([connector])
    graph_service.get_graph_data_compute_input = AsyncMock(
        return_value=graph_input)
    data_backend_ctx = MagicMock()
    data_backend_ctx._config = MagicMock()
    driver = DataSnapshotApiDriver(data_backend_ctx)
    driver._configure_state_services(
        SimpleNamespace(
            graphs=graph_service,
            tables=table_service,
            connectors=connector_service,
        ))
    driver._create_many_table_snapshots = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            DataSnapshot.TABLE.format_id(entry.id, TableDataVersion('v1'))
        ])

    snapshot_id = await driver.create_graph_snapshot(
        graph_def.fpid,
        refresh_source=False,
        base_graph_name='Base Graph',
        project_id=_PROJECT_ID,
    )

    assert snapshot_id == DataSnapshot.GRAPH.format_id(graph_def.fpid, 'gv1')
    assert graph_service.get_definition_calls == [(_PROJECT_ID, graph_def.fpid)]
    assert graph_service.get_entry_calls == [(_PROJECT_ID, graph_def.fpid)]
    driver._create_many_table_snapshots.assert_awaited_once_with(  # type: ignore[attr-defined]
        [table],
        refresh_source=False,
        project_id=_PROJECT_ID,
        defer_col_stats_validation=True,
    )
    assert version_info.graph_name == 'Base Graph (modified)'
    assert len(connector_service.fresh_execution_ref_calls) == 1
    mock_save_graph_data_version.assert_awaited_once_with(
        graph_input, version_info)
    mock_start_refresh_graph.assert_awaited_once()


@pytest.mark.asyncio
@patch('rest.driver.data_snapshot_api_driver.TemporalClient')
@patch('rest.driver.data_snapshot_api_driver.DataBackend.start_refresh_table',
       new_callable=AsyncMock)
@patch(
    'rest.driver.data_snapshot_api_driver.DataBackend.get_many_table_refresh_results',
    new_callable=AsyncMock)
@patch(
    'rest.driver.data_snapshot_api_driver.DataBackend.'
    'get_table_data_version_in_activity', new_callable=AsyncMock)
@patch('rest.driver.data_snapshot_api_driver.get_active_cache')
async def test_create_many_table_snapshots_uses_bulk_table_and_connector_reads(
    mock_get_active_cache,
    mock_get_table_data_version_in_activity,
    mock_get_many_table_refresh_results,
    mock_start_refresh_table,
    mock_temporal_client,
):
    connector = _make_file_connector_entry('connector_a',
                                           's3://bucket-bulk/path')
    table_a = _make_s3_table(
        connector_id='connector_a',
        source_table_name='ACCOUNT',
    )
    table_b = _make_s3_table(
        connector_id='connector_a',
        source_table_name='LOAN',
    )
    entry_a = _make_table_entry(table_a)
    entry_b = _make_table_entry(table_b)
    table_service = _TableService({
        entry_a.id: entry_a,
        entry_b.id: entry_b,
    })
    connector_service = _ConnectorService([connector])
    data_backend_ctx = MagicMock()
    data_backend_ctx._config = MagicMock()

    cache = MagicMock()
    cache.table_version_cache.get.coroutine = AsyncMock(return_value=None)
    mock_get_active_cache.return_value = cache
    mock_get_table_data_version_in_activity.side_effect = [
        'v-account', 'v-loan'
    ]
    mock_get_many_table_refresh_results.return_value = [None]

    driver = DataSnapshotApiDriver(data_backend_ctx)
    driver._configure_state_services(
        SimpleNamespace(tables=table_service, connectors=connector_service))

    snapshot_ids = await driver._create_many_table_snapshots(
        [table_a, table_b],
        refresh_source=True,
        project_id=_PROJECT_ID,
    )

    assert snapshot_ids == [
        DataSnapshot.TABLE.format_id(table_a.fpid,
                                     TableDataVersion('v-account')),
        DataSnapshot.TABLE.format_id(table_b.fpid, TableDataVersion('v-loan')),
    ]
    assert table_service.get_many_calls == [(_PROJECT_ID,
                                             [entry_a.id, entry_b.id])]
    assert table_service.get_calls == []
    assert connector_service.get_many_calls == []
    assert connector_service.get_required_calls == []
    assert connector_service.get_calls == []
    assert connector_service.execution_ref_calls == [
        (None, _PROJECT_ID, connector.id, CONNECTOR_GRAPH_ACTIONS),
        (None, _PROJECT_ID, connector.id, CONNECTOR_GRAPH_ACTIONS),
    ]
    assert cache.table_version_cache.get.coroutine.await_count == 2
    assert mock_get_table_data_version_in_activity.await_count == 2
    assert all(
        call.args[1] == ConnectorExecutionRef(project_id=_PROJECT_ID,
                                              connector_id=connector.id)
        for call in mock_get_table_data_version_in_activity.await_args_list)
    assert mock_get_many_table_refresh_results.await_count == 2
    assert mock_start_refresh_table.await_count == 2
    assert len(connector_service.fresh_execution_ref_calls) == 2
    assert [
        call.args[1].authorization_token
        for call in mock_start_refresh_table.await_args_list
    ] == ['fresh-1', 'fresh-2']
    assert [
        call.kwargs['defer_col_stats_validation']
        for call in mock_start_refresh_table.await_args_list
    ] == [False, False]


@pytest.mark.asyncio
async def test_table_snapshot_batch_can_defer_col_stats_for_graph_snapshot(
        monkeypatch) -> None:
    table = _make_s3_table(
        connector_id='connector_a',
        source_table_name='ACCOUNT',
    )
    entry = _make_table_entry(table)
    connector_id = table.get_connector_id()
    connector_service = SimpleNamespace(
        execution_ref_for_project=AsyncMock(return_value=ConnectorExecutionRef(
            project_id=_PROJECT_ID,
            connector_id=connector_id,
            authorization_token='probe-token',
        )), )
    data_backend_ctx = MagicMock()
    data_backend_ctx._config = MagicMock()
    driver = DataSnapshotApiDriver(data_backend_ctx)
    driver._configure_state_services(
        SimpleNamespace(connectors=connector_service))
    driver._prepare_state_services_table_refresh_with_connector = AsyncMock(  # type: ignore[method-assign]
        return_value=_PreparedTableRefresh(
            table_entry=entry,
            data_version=TableDataVersion('version-1'),
            execution_ref=ConnectorExecutionRef(
                project_id=_PROJECT_ID,
                connector_id=connector_id,
                authorization_token='workflow-token',
            ),
        ))
    start_refresh = AsyncMock(return_value='workflow-1')
    monkeypatch.setattr(DataBackend, 'start_refresh_table', start_refresh)

    data_versions = await driver._trigger_many_state_services_table_refresh(
        [entry],
        project_id=_PROJECT_ID,
        defer_col_stats_validation=True,
    )

    assert data_versions == [TableDataVersion('version-1')]
    start_refresh.assert_awaited_once()
    assert start_refresh.await_args.kwargs['defer_col_stats_validation'] is True


@pytest.mark.asyncio
@patch(
    'rest.driver.data_snapshot_api_driver.DataBackend.get_many_table_refresh_results',
    new_callable=AsyncMock)
@patch(
    'rest.driver.data_snapshot_api_driver.DataBackend.'
    'get_table_data_version_in_activity', new_callable=AsyncMock)
@patch('rest.driver.data_snapshot_api_driver.get_active_cache')
async def test_cache_miss_refresh_uses_fresh_workflow_capability(
    mock_get_active_cache,
    mock_get_table_data_version_in_activity,
    mock_get_many_table_refresh_results,
) -> None:
    """The version probe and table refresh cannot share one capability."""
    table = _make_s3_table(
        connector_id='connector_a',
        source_table_name='ACCOUNT',
    )
    entry = _make_table_entry(table)
    connector = _make_file_connector_entry('connector_a',
                                           's3://bucket-cache-miss/path')
    connector_service = _ConnectorService([connector])
    cache = MagicMock()
    cache.table_version_cache.get.coroutine = AsyncMock(return_value=None)
    mock_get_active_cache.return_value = cache
    mock_get_table_data_version_in_activity.return_value = 'version-1'
    mock_get_many_table_refresh_results.return_value = [None]
    data_backend_ctx = MagicMock()
    data_backend_ctx._config = MagicMock()
    driver = DataSnapshotApiDriver(data_backend_ctx)
    driver._configure_state_services(
        SimpleNamespace(connectors=connector_service))
    probe_ref = ConnectorExecutionRef(
        project_id=_PROJECT_ID,
        connector_id=connector.id,
        authorization_token='probe-token',
    )

    prepared = await driver._prepare_state_services_table_refresh_with_connector(
        entry,
        probe_ref,
        refresh_source=False,
    )

    mock_get_table_data_version_in_activity.assert_awaited_once_with(
        data_backend_ctx,
        probe_ref,
        'ACCOUNT',
        refresh_table_version=True,
        table_def=entry.config,
    )
    assert connector_service.fresh_execution_ref_calls == [(
        None,
        (probe_ref, ),
        CONNECTOR_GRAPH_ACTIONS,
    )]
    assert prepared.execution_ref is not None
    assert prepared.execution_ref.authorization_token == 'fresh-1'


@pytest.mark.asyncio
@patch(
    'rest.driver.data_snapshot_api_driver.DataBackend.get_many_table_refresh_results',
    new_callable=AsyncMock)
@patch(
    'rest.driver.data_snapshot_api_driver.DataBackend.'
    'get_table_data_version_in_activity', new_callable=AsyncMock)
@patch('rest.driver.data_snapshot_api_driver.get_active_cache')
async def test_cache_hit_without_refresh_reuses_unconsumed_capability(
    mock_get_active_cache,
    mock_get_table_data_version_in_activity,
    mock_get_many_table_refresh_results,
) -> None:
    table = _make_s3_table(
        connector_id='connector_a',
        source_table_name='ACCOUNT',
    )
    entry = _make_table_entry(table)
    connector = _make_file_connector_entry('connector_a',
                                           's3://bucket-cache-hit/path')
    connector_service = _ConnectorService([connector])
    cache = MagicMock()
    cache.table_version_cache.get.coroutine = AsyncMock(
        return_value='version-1')
    mock_get_active_cache.return_value = cache
    mock_get_many_table_refresh_results.return_value = [None]
    data_backend_ctx = MagicMock()
    data_backend_ctx._config = MagicMock()
    driver = DataSnapshotApiDriver(data_backend_ctx)
    driver._configure_state_services(
        SimpleNamespace(connectors=connector_service))
    unconsumed_ref = ConnectorExecutionRef(
        project_id=_PROJECT_ID,
        connector_id=connector.id,
        authorization_token='table-refresh-token',
    )

    prepared = await driver._prepare_state_services_table_refresh_with_connector(
        entry,
        unconsumed_ref,
        refresh_source=False,
    )

    mock_get_table_data_version_in_activity.assert_not_awaited()
    assert connector_service.fresh_execution_ref_calls == []
    assert prepared.execution_ref is unconsumed_ref


@pytest.mark.asyncio
async def test_table_refresh_batch_rolls_back_partial_workflow_starts(
    monkeypatch, ) -> None:
    tables = [
        _make_s3_table(connector_id='connector_a',
                       source_table_name=f'TABLE_{index}') for index in range(3)
    ]
    entries = [_make_table_entry(table) for table in tables]
    connector_id = tables[0].get_connector_id()
    connector_service = SimpleNamespace(
        execution_ref_for_project=AsyncMock(side_effect=[
            ConnectorExecutionRef(
                project_id=_PROJECT_ID,
                connector_id=connector_id,
                authorization_token=f'probe-{index}',
            ) for index in range(3)
        ]), )
    authorization_service = SimpleNamespace(revoke=AsyncMock())
    handles = {
        'workflow-0':
        SimpleNamespace(terminate=AsyncMock(
            side_effect=[RuntimeError('transient'), None])),
        'workflow-2':
        SimpleNamespace(terminate=AsyncMock()),
    }
    temporal_client = SimpleNamespace(get_workflow_handle=AsyncMock(
        side_effect=handles.__getitem__))
    data_backend_ctx = MagicMock()
    data_backend_ctx._config = MagicMock()
    data_backend_ctx.temporal_client.return_value = temporal_client
    driver = DataSnapshotApiDriver(data_backend_ctx)
    driver._configure_state_services(
        SimpleNamespace(
            connectors=connector_service,
            connector_authorizations=authorization_service,
        ))
    prepared = [
        _PreparedTableRefresh(
            table_entry=entry,
            data_version=TableDataVersion(f'version-{index}'),
            execution_ref=ConnectorExecutionRef(
                project_id=_PROJECT_ID,
                connector_id=connector_id,
                authorization_token=f'workflow-token-{index}',
            ),
        ) for index, entry in enumerate(entries)
    ]
    driver._prepare_state_services_table_refresh_with_connector = AsyncMock(  # type: ignore[method-assign]
        side_effect=prepared)
    start_refresh = AsyncMock(side_effect=[
        'workflow-0',
        RuntimeError('database is locked: secret.sqlite'),
        'workflow-2',
    ])
    monkeypatch.setattr(DataBackend, 'start_refresh_table', start_refresh)

    with pytest.raises(ConnectorError) as error:
        await driver._trigger_many_state_services_table_refresh(
            entries,
            project_id=_PROJECT_ID,
        )

    assert str(error.value) == (
        'Unable to start all table refreshes; partial starts were rolled back. '
        'Please retry.')
    assert 'locked' not in str(error.value)
    assert temporal_client.get_workflow_handle.await_args_list == [
        ((workflow_id, ), {}) for workflow_id in ('workflow-0', 'workflow-2')
    ]
    assert handles['workflow-0'].terminate.await_count == 2
    assert handles['workflow-2'].terminate.await_count == 1
    assert sorted(call.args[0]
                  for call in authorization_service.revoke.await_args_list) == [
                      'probe-0',
                      'probe-1',
                      'probe-2',
                      'workflow-token-0',
                      'workflow-token-1',
                      'workflow-token-2',
                  ]


@pytest.mark.asyncio
async def test_table_refresh_batch_revokes_all_refs_when_preparation_fails(
    monkeypatch, ) -> None:
    tables = [
        _make_s3_table(connector_id='connector_a',
                       source_table_name=f'TABLE_{index}') for index in range(3)
    ]
    entries = [_make_table_entry(table) for table in tables]
    connector_id = tables[0].get_connector_id()
    connector_service = SimpleNamespace(
        execution_ref_for_project=AsyncMock(side_effect=[
            ConnectorExecutionRef(
                project_id=_PROJECT_ID,
                connector_id=connector_id,
                authorization_token=f'probe-{index}',
            ) for index in range(3)
        ]), )
    authorization_service = SimpleNamespace(revoke=AsyncMock())
    data_backend_ctx = MagicMock()
    data_backend_ctx._config = MagicMock()
    driver = DataSnapshotApiDriver(data_backend_ctx)
    driver._configure_state_services(
        SimpleNamespace(
            connectors=connector_service,
            connector_authorizations=authorization_service,
        ))
    prepared = _PreparedTableRefresh(
        table_entry=entries[0],
        data_version=TableDataVersion('version-0'),
        execution_ref=ConnectorExecutionRef(
            project_id=_PROJECT_ID,
            connector_id=connector_id,
            authorization_token='workflow-token-0',
        ),
    )
    driver._prepare_state_services_table_refresh_with_connector = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            prepared,
            RuntimeError('database is locked: secret.sqlite'),
            dataclasses.replace(
                prepared,
                table_entry=entries[2],
                data_version=TableDataVersion('version-2'),
                execution_ref=None,
            ),
        ])
    start_refresh = AsyncMock()
    monkeypatch.setattr(DataBackend, 'start_refresh_table', start_refresh)

    with pytest.raises(ConnectorError) as error:
        await driver._trigger_many_state_services_table_refresh(
            entries,
            project_id=_PROJECT_ID,
        )

    assert str(
        error.value) == ('Unable to prepare all table refreshes; please retry.')
    assert 'locked' not in str(error.value)
    start_refresh.assert_not_awaited()
    assert sorted(call.args[0]
                  for call in authorization_service.revoke.await_args_list) == [
                      'probe-0',
                      'probe-1',
                      'probe-2',
                      'workflow-token-0',
                  ]


@pytest.mark.asyncio
async def test_table_refresh_cancellation_during_prepare_revokes_all_refs(
        monkeypatch) -> None:
    tables = [
        _make_s3_table(connector_id='connector_a',
                       source_table_name=f'TABLE_{index}') for index in range(3)
    ]
    entries = [_make_table_entry(table) for table in tables]
    connector_id = tables[0].get_connector_id()
    connector_service = SimpleNamespace(
        execution_ref_for_project=AsyncMock(side_effect=[
            ConnectorExecutionRef(
                project_id=_PROJECT_ID,
                connector_id=connector_id,
                authorization_token=f'probe-{index}',
            ) for index in range(3)
        ]), )

    async def revoke(token):
        if token == 'workflow-token-1':
            raise RuntimeError('cleanup-secret')

    authorization_service = SimpleNamespace(revoke=AsyncMock(
        side_effect=revoke))
    data_backend_ctx = MagicMock()
    driver = DataSnapshotApiDriver(data_backend_ctx)
    driver._configure_state_services(
        SimpleNamespace(
            connectors=connector_service,
            connector_authorizations=authorization_service,
        ))
    release = asyncio.Event()
    all_started = asyncio.Event()
    started = 0

    async def prepare(table_entry, execution_ref, *, refresh_source):
        nonlocal started
        index = entries.index(table_entry)
        started += 1
        if started == len(entries):
            all_started.set()
        await release.wait()
        return _PreparedTableRefresh(
            table_entry=table_entry,
            data_version=TableDataVersion(f'version-{index}'),
            execution_ref=dataclasses.replace(
                execution_ref,
                authorization_token=f'workflow-token-{index}',
            ),
        )

    driver._prepare_state_services_table_refresh_with_connector = prepare  # type: ignore[method-assign]
    start_refresh = AsyncMock()
    monkeypatch.setattr(DataBackend, 'start_refresh_table', start_refresh)

    task = asyncio.create_task(
        driver._trigger_many_state_services_table_refresh(
            entries,
            project_id=_PROJECT_ID,
        ))
    await all_started.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    start_refresh.assert_not_awaited()
    assert sorted(call.args[0]
                  for call in authorization_service.revoke.await_args_list) == [
                      'probe-0',
                      'probe-1',
                      'probe-2',
                      'workflow-token-0',
                      'workflow-token-1',
                      'workflow-token-2',
                  ]


@pytest.mark.asyncio
async def test_table_refresh_cancellation_during_start_rolls_back_new_only(
        monkeypatch) -> None:
    tables = [
        _make_s3_table(connector_id='connector_a',
                       source_table_name=f'TABLE_{index}') for index in range(3)
    ]
    entries = [_make_table_entry(table) for table in tables]
    connector_id = tables[0].get_connector_id()
    connector_service = SimpleNamespace(
        execution_ref_for_project=AsyncMock(side_effect=[
            ConnectorExecutionRef(
                project_id=_PROJECT_ID,
                connector_id=connector_id,
                authorization_token=f'probe-{index}',
            ) for index in range(3)
        ]), )

    async def revoke(token):
        if token == 'probe-1':
            raise RuntimeError('cleanup-secret')

    authorization_service = SimpleNamespace(revoke=AsyncMock(
        side_effect=revoke))
    lookup_attempts = {'workflow-0': 0, 'workflow-2': 0}
    handles = {
        'workflow-0':
        SimpleNamespace(terminate=AsyncMock()),
        'workflow-2':
        SimpleNamespace(terminate=AsyncMock(
            side_effect=RuntimeError('termination-secret'))),
    }

    async def get_handle(workflow_id):
        lookup_attempts[workflow_id] += 1
        if workflow_id == 'workflow-0' and lookup_attempts[workflow_id] == 1:
            raise RuntimeError('lookup-secret')
        return handles[workflow_id]

    temporal_client = SimpleNamespace(get_workflow_handle=AsyncMock(
        side_effect=get_handle))
    data_backend_ctx = MagicMock()
    data_backend_ctx.temporal_client.return_value = temporal_client
    driver = DataSnapshotApiDriver(data_backend_ctx)
    driver._configure_state_services(
        SimpleNamespace(
            connectors=connector_service,
            connector_authorizations=authorization_service,
        ))
    prepared = [
        _PreparedTableRefresh(
            table_entry=entry,
            data_version=TableDataVersion(f'version-{index}'),
            execution_ref=ConnectorExecutionRef(
                project_id=_PROJECT_ID,
                connector_id=connector_id,
                authorization_token=f'workflow-token-{index}',
            ),
        ) for index, entry in enumerate(entries)
    ]
    driver._prepare_state_services_table_refresh_with_connector = AsyncMock(  # type: ignore[method-assign]
        side_effect=prepared)
    release = asyncio.Event()
    all_started = asyncio.Event()
    started = 0

    async def start_refresh(_ctx, execution_ref, *_args, **_kwargs):
        nonlocal started
        index = int(execution_ref.authorization_token.rsplit('-', 1)[1])
        started += 1
        if started == len(entries):
            all_started.set()
        await release.wait()
        return None if index == 1 else f'workflow-{index}'

    monkeypatch.setattr(DataBackend, 'start_refresh_table', start_refresh)

    task = asyncio.create_task(
        driver._trigger_many_state_services_table_refresh(
            entries,
            project_id=_PROJECT_ID,
        ))
    await all_started.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    # workflow-1 represents an already-running workflow and must not be
    # terminated by rollback.  Confirmed starts are retried independently.
    assert set(lookup_attempts) == {'workflow-0', 'workflow-2'}
    assert lookup_attempts['workflow-0'] == 2
    assert lookup_attempts['workflow-2'] == 1
    handles['workflow-0'].terminate.assert_awaited_once()
    assert handles['workflow-2'].terminate.await_count == 3
    assert sorted(call.args[0]
                  for call in authorization_service.revoke.await_args_list) == [
                      'probe-0',
                      'probe-1',
                      'probe-2',
                      'workflow-token-0',
                      'workflow-token-1',
                      'workflow-token-2',
                  ]


@pytest.mark.asyncio
async def test_table_refresh_cancellation_cleanup_is_bounded(
        monkeypatch) -> None:
    table = _make_s3_table(connector_id='connector_a',
                           source_table_name='TABLE')
    entry = _make_table_entry(table)
    connector_id = table.get_connector_id()
    connector_service = SimpleNamespace(
        execution_ref_for_project=AsyncMock(return_value=ConnectorExecutionRef(
            project_id=_PROJECT_ID,
            connector_id=connector_id,
            authorization_token='probe',
        )), )
    authorization_service = SimpleNamespace(revoke=AsyncMock())
    never = asyncio.Event()
    handle = SimpleNamespace(terminate=AsyncMock(side_effect=never.wait))
    temporal_client = SimpleNamespace(get_workflow_handle=AsyncMock(
        return_value=handle))
    data_backend_ctx = MagicMock()
    data_backend_ctx.temporal_client.return_value = temporal_client
    driver = DataSnapshotApiDriver(data_backend_ctx)
    driver._configure_state_services(
        SimpleNamespace(
            connectors=connector_service,
            connector_authorizations=authorization_service,
        ))
    prepared = _PreparedTableRefresh(
        table_entry=entry,
        data_version=TableDataVersion('version'),
        execution_ref=ConnectorExecutionRef(
            project_id=_PROJECT_ID,
            connector_id=connector_id,
            authorization_token='workflow-token',
        ),
    )
    driver._prepare_state_services_table_refresh_with_connector = AsyncMock(  # type: ignore[method-assign]
        return_value=prepared)
    start_entered = asyncio.Event()
    release_start = asyncio.Event()

    async def start_refresh(*_args, **_kwargs):
        start_entered.set()
        await release_start.wait()
        return 'workflow'

    monkeypatch.setattr(DataBackend, 'start_refresh_table', start_refresh)
    monkeypatch.setattr(
        'rest.driver.data_snapshot_api_driver.'
        '_TABLE_REFRESH_BATCH_CLEANUP_TIMEOUT_SECONDS', 0.02)

    task = asyncio.create_task(
        driver._trigger_many_state_services_table_refresh(
            [entry],
            project_id=_PROJECT_ID,
        ))
    await start_entered.wait()
    started_at = asyncio.get_running_loop().time()
    task.cancel()
    release_start.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    elapsed = asyncio.get_running_loop().time() - started_at

    assert elapsed < 0.5
    handle.terminate.assert_awaited_once()
    assert sorted(call.args[0]
                  for call in authorization_service.revoke.await_args_list) == [
                      'probe', 'workflow-token'
                  ]


@pytest.mark.asyncio
async def test_create_table_snapshot_canonicalizes_custom_sql_name():
    """The snapshot-create path assigns the server-canonical synthetic name to a
    Custom SQL definition before persisting, so a SQL table snapshotted here
    dedupes onto the same fingerprint entry the table-create path produces
    rather than a divergent entry under the SDK's provisional name. A physical
    table is a no-op with no connector lookup."""
    cid = _connector_id('sql_conn')
    sql_def = TableDefinition(
        cols=[Column(name='id', stype=Stype.ID, dtype=Dtype.int)],
        source_table=CustomSQLSourceTable(connector_id=str(cid),
                                          connector_type='snowflake',
                                          sql='SELECT id FROM t',
                                          table='custom_sql_provisional',
                                          referenced_tables=['stale']),
        pkey='id',
    )
    connector = SimpleNamespace(config=SimpleNamespace(type=SimpleNamespace(
        value='snowflake')))
    connectors = SimpleNamespace(get_for_project=AsyncMock(
        return_value=connector))
    driver = DataSnapshotApiDriver(MagicMock())
    driver._configure_state_services(
        SimpleNamespace(tables=MagicMock(), connectors=connectors))

    canon = await driver._canonicalize_custom_sql_name(sql_def, _PROJECT_ID)
    assert canon.source_table.table == custom_sql_synthetic_name(
        'SELECT id FROM t', ConnectorType.SNOWFLAKE)
    # Referenced tables are recomputed server-side (the client 'stale' list is
    # overwritten with the query's real dependency).
    assert canon.source_table.referenced_tables == ['t']

    phys = _make_s3_table(connector_id='c', source_table_name='ACCOUNT')
    connectors.get_for_project.reset_mock()
    assert await driver._canonicalize_custom_sql_name(phys, _PROJECT_ID) is phys
    connectors.get_for_project.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_table_snapshot_persists_canonical_sql_fpid(monkeypatch):
    """End-to-end: create_table_snapshot canonicalizes BEFORE the entry
    lookup/persist and the returned snapshot id, so the entry saved and the id
    returned both carry the canonical fpid — never the SDK provisional one."""
    cid = _connector_id('sql_conn_e2e')
    provisional = TableDefinition(
        cols=[Column(name='id', stype=Stype.ID, dtype=Dtype.int)],
        source_table=CustomSQLSourceTable(connector_id=str(cid),
                                          connector_type='snowflake',
                                          sql='SELECT id FROM t',
                                          table='custom_sql_provisional',
                                          referenced_tables=[]),
        pkey='id',
    )
    canonical_name = custom_sql_synthetic_name('SELECT id FROM t',
                                               ConnectorType.SNOWFLAKE)
    canonical_fpid = dataclasses.replace(
        provisional,
        source_table=dataclasses.replace(provisional.source_table,
                                         table=canonical_name,
                                         referenced_tables=['t'])).fpid

    connector = SimpleNamespace(config=SimpleNamespace(type=SimpleNamespace(
        value='snowflake')))
    connectors = SimpleNamespace(get_for_project=AsyncMock(
        return_value=connector))
    driver = DataSnapshotApiDriver(MagicMock())
    driver._configure_state_services(
        SimpleNamespace(tables=MagicMock(), connectors=connectors))

    seen = {}

    async def _resolve(table_definition, project_id):
        seen['def'] = table_definition
        return _make_table_entry(table_definition)

    monkeypatch.setattr(driver, '_require_reachable_table_entry', _resolve)
    monkeypatch.setattr(driver, '_trigger_state_services_table_refresh',
                        AsyncMock(return_value=TableDataVersion('dv1')))

    snap_id = await driver.create_table_snapshot(provisional,
                                                 refresh_source=False,
                                                 project_id=_PROJECT_ID)

    # The definition handed to the entry lookup carries the canonical
    # name/fpid.
    assert seen['def'].source_table.table == canonical_name
    assert seen['def'].fpid == canonical_fpid
    # The returned snapshot id embeds the canonical fpid, not the provisional.
    assert str(canonical_fpid) in snap_id
    assert str(provisional.fpid) not in snap_id


@pytest.mark.asyncio
@pytest.mark.parametrize('stored, refusal', [
    ('absent', 'not saved in this project'),
    ('named', None),
    ('graph_referenced', None),
    ('nameless', 'stored without a name'),
])
async def test_create_table_snapshot_requires_a_reachable_table(
        stored, refusal):
    """A direct table snapshot resolves a definition the project already holds
    under a name or inside a saved graph, and refuses any other definition
    without persisting it."""
    table = _make_s3_table(connector_id='connector_a',
                           source_table_name='ACCOUNT')
    entry = _make_table_entry(table,
                              alias='account' if stored == 'named' else None)
    graph_entry = GraphDBEntryV2(
        id=FPID('b' * 32),
        project_id=_PROJECT_ID,
        table_name_to_dbid={TableName('ACCOUNT'): entry.id},
        col_groups=[],
        timestamps=ResourceTimestamps(
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    )
    table_service = _TableService({} if stored ==
                                  'absent' else {entry.id: entry})
    graph_service = _GraphService(
        {},
        {graph_entry.id: graph_entry} if stored == 'graph_referenced' else {})
    driver = DataSnapshotApiDriver(MagicMock())
    driver._configure_state_services(
        SimpleNamespace(
            tables=table_service,
            graphs=graph_service,
            connectors=MagicMock(),
        ))
    driver._trigger_state_services_table_refresh = AsyncMock(  # type: ignore[method-assign]
        return_value=TableDataVersion('dv1'))

    if refusal is None:
        snap_id = await driver.create_table_snapshot(table,
                                                     refresh_source=False,
                                                     project_id=_PROJECT_ID)
        assert snap_id == DataSnapshot.TABLE.format_id(table.fpid,
                                                       TableDataVersion('dv1'))
    else:
        with pytest.raises(ValueError, match=refusal):
            await driver.create_table_snapshot(table, refresh_source=False,
                                               project_id=_PROJECT_ID)
    assert table_service.save_calls == []


@pytest.mark.asyncio
async def test_graph_snapshot_persists_its_tables_without_an_alias(monkeypatch):
    """The graph snapshot path persists an embedded table the project does not
    hold yet, and gives it no project-local name."""
    table = _make_s3_table(connector_id='connector_a',
                           source_table_name='ACCOUNT')
    table_service = _TableService({})
    driver = DataSnapshotApiDriver(MagicMock())
    driver._configure_state_services(
        SimpleNamespace(tables=table_service, connectors=MagicMock()))
    monkeypatch.setattr(driver, '_validate_new_table_with_state_services',
                        AsyncMock())

    entry = await driver._ensure_table_entry_with_state_services(
        table, _PROJECT_ID)

    assert entry.alias is None
    assert table_service.save_calls == [(_PROJECT_ID, table, {})]


def _ingest_table_snapshot(
    snapshot_id: str,
    status: JobStatus,
    start_time: datetime,
    end_time: datetime,
    error: str | None = None,
) -> snapshot_interfaces.TableSnapshotResource:
    return snapshot_interfaces.TableSnapshotResource(
        id=snapshot_interfaces.TableSnapshotID(snapshot_id),
        stages={
            snapshot_interfaces.TableSnapshotStage.INGEST:
            snapshot_interfaces.JobState(
                status=status,
                start_time=start_time,
                end_time=end_time,
                error=error,
            ),
        },
    )


_EARLY_INGEST_START = datetime(2026, 7, 9, 14, 33, 41, 145444)
_EARLY_INGEST_END = datetime(2026, 7, 9, 14, 33, 41, 145446)
_LATE_INGEST_START = datetime(2026, 7, 9, 14, 33, 43, 151458)
_LATE_INGEST_END = datetime(2026, 7, 9, 14, 33, 43, 151460)


@pytest.mark.parametrize('early_first', [True, False])
def test_merge_table_ingest_job_states_spans_min_start_max_end(
        early_first: bool) -> None:
    early = _ingest_table_snapshot('table-early@1', JobStatus.DONE,
                                   _EARLY_INGEST_START, _EARLY_INGEST_END)
    late = _ingest_table_snapshot('table-late@1', JobStatus.DONE,
                                  _LATE_INGEST_START, _LATE_INGEST_END)
    snapshots = [early, late] if early_first else [late, early]

    merged = _merge_table_ingest_job_states(snapshots)

    # The merged stage must span from the earliest table start to the latest
    # table end, independent of table iteration order, so the reported
    # duration is the true (non-negative) span:
    assert merged.status == JobStatus.DONE
    assert merged.start_time == _EARLY_INGEST_START
    assert merged.end_time == _LATE_INGEST_END
    assert merged.end_time >= merged.start_time


@pytest.mark.parametrize('early_first', [True, False])
def test_merge_table_ingest_job_states_failed_keeps_true_span(
        early_first: bool) -> None:
    # The timing loop also runs when the merged status is FAILED; the span
    # and the aggregated per-table errors must both be preserved.
    early = _ingest_table_snapshot('table-early@1', JobStatus.FAILED,
                                   _EARLY_INGEST_START, _EARLY_INGEST_END,
                                   error='ingest failed for table-early')
    late = _ingest_table_snapshot('table-late@1', JobStatus.FAILED,
                                  _LATE_INGEST_START, _LATE_INGEST_END,
                                  error='ingest failed for table-late')
    snapshots = [early, late] if early_first else [late, early]

    merged = _merge_table_ingest_job_states(snapshots)

    assert merged.status == JobStatus.FAILED
    assert merged.start_time == _EARLY_INGEST_START
    assert merged.end_time == _LATE_INGEST_END
    assert merged.error is not None
    assert 'ingest failed for table-early' in merged.error
    assert 'ingest failed for table-late' in merged.error


def test_merge_table_ingest_job_states_single_table_passthrough() -> None:
    snapshot = _ingest_table_snapshot('table-solo@1', JobStatus.DONE,
                                      _EARLY_INGEST_START, _EARLY_INGEST_END)

    merged = _merge_table_ingest_job_states([snapshot])

    assert merged.status == JobStatus.DONE
    assert merged.start_time == _EARLY_INGEST_START
    assert merged.end_time == _EARLY_INGEST_END


def _refresh_result(
    *,
    stage: TableRefreshStage,
    warnings: list[ValidationWarning] | None = None,
    status: AsyncDataTaskStatus = AsyncDataTaskStatus.DONE,
    error: str | None = None,
) -> TableDataTaskResult:
    return TableDataTaskResult(
        status=status,
        validation=ValidationResponse(warnings=warnings or []),
        stage=stage,
        error=error,
        data_version='tv1',
        created_at=_EARLY_INGEST_START,
        updated_at=_EARLY_INGEST_END,
    )


@pytest.mark.asyncio
async def test_table_snapshot_cancels_full_stats_after_min_stats_failure(
        monkeypatch) -> None:
    table = _make_s3_table(connector_id='connector_a',
                           source_table_name='ACCOUNT')
    entry = _make_table_entry(table)
    failure = _refresh_result(
        stage=TableRefreshStage.COMPUTE_MIN_COL_STATS,
        status=AsyncDataTaskStatus.FAILED,
        error='wide integer is outside the signed 64-bit range',
    )
    driver = DataSnapshotApiDriver(MagicMock())
    driver._get_required_table_entry = AsyncMock(  # type: ignore[method-assign]
        return_value=entry)
    monkeypatch.setattr(
        'rest.driver.data_snapshot_api_driver.get_active_cache',
        lambda: SimpleNamespace(get_table_stats=lambda *_args: {}),
    )
    monkeypatch.setattr(
        DataBackend,
        'get_many_table_refresh_results',
        AsyncMock(side_effect=[
            [failure],
            [_refresh_result(stage=TableRefreshStage.INGEST_TABLE)],
            [failure],
        ]),
    )

    snapshot = await driver._get_kumo_table_snapshot(
        _PROJECT_ID,
        table.fpid,
        TableDataVersion('tv1'),
    )

    assert (snapshot.stages[snapshot_interfaces.TableSnapshotStage.
                            MIN_COL_STATS].status == JobStatus.FAILED)
    assert (snapshot.stages[snapshot_interfaces.TableSnapshotStage.
                            FULL_COL_STATS].status == JobStatus.CANCELLED)


@pytest.mark.asyncio
async def test_table_snapshot_surfaces_completed_validation_warnings(
        monkeypatch) -> None:
    table = _make_s3_table(connector_id='connector_a',
                           source_table_name='ACCOUNT')
    entry = _make_table_entry(table)
    warning = ValidationWarning(
        title='Duplicate primary keys',
        message='4,423 rows in Customer_Accounts are duplicated',
    )
    driver = DataSnapshotApiDriver(MagicMock())
    driver._get_required_table_entry = AsyncMock(  # type: ignore[method-assign]
        return_value=entry)
    monkeypatch.setattr(
        'rest.driver.data_snapshot_api_driver.get_active_cache',
        lambda: SimpleNamespace(get_table_stats=lambda *_args: {}),
    )
    monkeypatch.setattr(
        DataBackend,
        'get_many_table_refresh_results',
        AsyncMock(side_effect=[
            [
                _refresh_result(stage=TableRefreshStage.COMPLETED,
                                warnings=[warning])
            ],
            [_refresh_result(stage=TableRefreshStage.INGEST_TABLE)],
            [None],
        ]),
    )

    snapshot = await driver._get_kumo_table_snapshot(
        _PROJECT_ID,
        table.fpid,
        TableDataVersion('tv1'),
    )

    assert snapshot.stages[
        snapshot_interfaces.TableSnapshotStage.INGEST].warnings == [
            'Duplicate primary keys: '
            '4,423 rows in Customer_Accounts are duplicated'
        ]


@pytest.mark.asyncio
async def test_graph_table_snapshot_surfaces_completed_validation_warnings(
        monkeypatch) -> None:
    table = _make_s3_table(connector_id='connector_a',
                           source_table_name='ACCOUNT')
    entry = _make_table_entry(table)
    warning = ValidationWarning(
        title='Duplicate primary keys',
        message='4,423 rows in Customer_Accounts are duplicated',
    )
    driver = DataSnapshotApiDriver(MagicMock())
    driver._get_many_table_entries = AsyncMock(  # type: ignore[method-assign]
        return_value=[entry])
    monkeypatch.setattr(
        DataBackend,
        'get_many_table_refresh_results',
        AsyncMock(side_effect=[
            [
                _refresh_result(stage=TableRefreshStage.COMPLETED,
                                warnings=[warning])
            ],
            [_refresh_result(stage=TableRefreshStage.INGEST_TABLE)],
        ]),
    )

    snapshots = await driver._get_many_kumo_table_snapshots(
        _PROJECT_ID,
        {table.fpid: TableDataVersion('tv1')},
    )
    graph_ingest = _merge_table_ingest_job_states(snapshots)

    assert graph_ingest.warnings == [
        f'{snapshots[0].id} has warnings: Duplicate primary keys: '
        '4,423 rows in Customer_Accounts are duplicated'
    ]
