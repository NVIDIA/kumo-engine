# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kumo_core.types import ConnectorType
from kumo_core.types.schema import FPID
from kumoapi import graph as graph_api
from kumoapi.graph import TableName
from kumoapi.table import Column
from kumoapi.table import TableDefinition as ApiTableDefinition
from kumoapi.typing import Dtype, Stype

from kumo.data.custom_sql import (
    custom_sql_synthetic_name,
    referenced_tables_or_none,
)
from kumo.validation import (
    ValidationError,
    ValidationException,
    ValidationResponse,
)
from rest.config.data_backend import ConnectorExecutionRef, ConnectorSpec
from rest.config.file_type import FileType
from rest.config.graph import EdgeHealthStats
from rest.config.graph_v2 import (
    ColumnKey,
    ColumnKeyGroup,
    GraphDBEntryV2,
    GraphDefinition,
    ListResponse,
)
from rest.config.table import TableStats
from rest.config.table_v2 import Column as RestColumn
from rest.config.table_v2 import (
    CustomSQLSourceTable,
    S3SourceTable,
    TableDBEntryV2,
    TableDefinition,
    TableResourceResponse,
)
from rest.deps.authentication import UserAuthorization
from rest.deps.context import _current_user_ctx
from rest.driver.graph_driver import (
    _MAX_CONCURRENT_TABLE_VALIDATIONS,
    GraphDriver,
)
from rest.settings import DataScaleLimits
from statedb.config.common import ID, ConnectorID, GraphIDv2, ProjectID
from statedb.config.common import TableName as StateTableName
from statedb.config.timestamp import ResourceTimestamps
from statedb.db.identities import ConnectorService
from statedb.db.services import StateServices
from statedb.rest.connector_authorization import CONNECTOR_GRAPH_ACTIONS
from statedb.utils.errors import (
    GraphDefinitionAlreadyExists,
    GraphDoesNotExistError,
)

_TEST_PROJECT_ID = ProjectID("project_" + "1" * 32)
_TEST_CONNECTOR_ID = ConnectorID("connector_" + "1" * 32)
_SECOND_CONNECTOR_ID = ConnectorID("connector_" + "2" * 32)


@pytest.fixture(autouse=True)
def auto_cleanup_per_func(monkeypatch):
    monkeypatch.setattr(GraphDriver, "_require_project_access", AsyncMock())
    yield


@contextmanager
def _authenticated_principal():
    token = _current_user_ctx.set(UserAuthorization.disabled())
    try:
        yield _current_user_ctx.get()
    finally:
        _current_user_ctx.reset(token)


@pytest.fixture
def setup_rest_test_session() -> None:
    return None


def _create_graph_driver(
    mock_graphs_dao: AsyncMock,
    mock_tables_dao: AsyncMock,
    mock_connectors_dao: AsyncMock,
) -> GraphDriver:
    """Create a GraphDriver with neutral unit-test dependencies."""
    data_backend_ctx = AsyncMock()
    data_scale_limits = DataScaleLimits.parse_from_env()
    table_driver = AsyncMock()
    table_driver.validate_table = AsyncMock(return_value=ValidationResponse())

    return GraphDriver(
        data_backend_ctx=data_backend_ctx,
        data_scale_limits=data_scale_limits,
        table_driver=table_driver,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'lookup',
    [pytest.param('id', id='id'),
     pytest.param('name', id='name')],
)
async def test_resolve_project_graph_uses_state_services(lookup):
    project_id = "project_" + "a" * 32
    graph_def = _create_sample_graph_definition(["orders"])
    expected = _create_state_graph_entry(graph_def, "orders_graph",
                                         project_id=ProjectID(project_id))
    graph_service = _GraphService({expected.id: expected}, {})
    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(
        cast(StateServices, SimpleNamespace(graphs=graph_service)))
    graph_ref = (str(GraphIDv2.public(expected.id))
                 if lookup == 'id' else 'orders_graph')

    result = await graph_driver.resolve_project_graph(project_id, graph_ref)

    assert result == expected
    if lookup == 'id':
        assert graph_service.get_entry_calls == [(ProjectID(project_id),
                                                  expected.id)]
    else:
        assert graph_service.get_alias_calls == [(ProjectID(project_id),
                                                  'orders_graph')]


@pytest.mark.asyncio
async def test_list_project_graph_names_uses_state_services():
    project_id = "project_" + "a" * 32
    graph_def = _create_sample_graph_definition(["orders"])
    graph_entry = _create_state_graph_entry(graph_def, "orders_graph",
                                            project_id=ProjectID(project_id))
    graph_id = GraphIDv2.public(graph_entry.id)
    graph_service = _GraphService({graph_entry.id: graph_entry}, {})
    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(
        SimpleNamespace(graphs=graph_service))

    response = await graph_driver.list_project_graph_names(project_id)

    assert response == [
        ListResponse(
            name="orders_graph",
            id=graph_id,
            timestamps=graph_entry.timestamps,
        )
    ]
    assert graph_service.list_calls == [ProjectID(project_id)]


@pytest.mark.asyncio
async def test_validate_project_graph_name_uses_state_services():
    project_id = "project_" + "a" * 32
    graph_def = _create_sample_graph_definition(["orders"])
    graph_entry = _create_state_graph_entry(graph_def, "orders_graph",
                                            project_id=ProjectID(project_id))
    graph_service = _GraphService({graph_entry.id: graph_entry}, {})
    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(
        SimpleNamespace(graphs=graph_service))

    response = await graph_driver.validate_project_graph_name(
        project_id, "orders_graph")

    assert not response.is_valid
    assert response.error_msg == (
        'Graph name orders_graph is already used in this project.')
    assert graph_service.get_alias_calls == [(ProjectID(project_id),
                                              "orders_graph")]


@pytest.mark.asyncio
async def test_delete_graph_alias_uses_state_services():
    project_id = "project_" + "a" * 32
    graph_def = _create_sample_graph_definition(["orders"])
    graph_entry = _create_state_graph_entry(graph_def, "orders_graph",
                                            project_id=ProjectID(project_id))
    graph_service = _GraphService({graph_entry.id: graph_entry}, {})
    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(
        SimpleNamespace(graphs=graph_service))

    result = await graph_driver.delete_graph_alias(
        "orders_graph",
        project_id=project_id,
    )

    assert result == str(GraphIDv2.public(graph_entry.id))
    assert graph_service.clear_alias_calls == [(ProjectID(project_id),
                                                "orders_graph")]


@pytest.mark.asyncio
async def test_delete_graph_accepts_canonical_id():
    project_id = "project_" + "a" * 32
    entry = _create_state_graph_entry(
        _create_sample_graph_definition(["orders"]),
        "orders_graph",
        project_id=ProjectID(project_id),
    )
    graph_service = _GraphService({entry.id: entry}, {})
    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(
        cast(StateServices, SimpleNamespace(graphs=graph_service)))

    graph_id = str(GraphIDv2.public(entry.id))
    assert await graph_driver.delete_graph_alias(
        graph_id, project_id=project_id) == graph_id
    assert graph_service.clear_alias_by_id_calls == [(ProjectID(project_id),
                                                      entry.id)]


@pytest.mark.asyncio
async def test_refresh_graph_v2_missing_graph_uses_state_services():
    project_id = _TEST_PROJECT_ID
    graph_fpid = FPID("a" * 32)
    graph_id = GraphIDv2.public(graph_fpid)
    graph_service = _GraphService({}, {})
    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(
        SimpleNamespace(graphs=graph_service))

    with pytest.raises(GraphDoesNotExistError):
        await graph_driver.refresh_graph_v2(graph_id, project_id)

    assert graph_service.get_entry_calls == [(project_id, graph_fpid)]


@pytest.mark.asyncio
async def test_refresh_graph_v2_uses_state_services_for_workflow_input():
    project_id = _TEST_PROJECT_ID
    graph_def = _create_sample_graph_definition(["customers"])
    entry = _create_state_graph_entry(graph_def, "state_graph",
                                      project_id=project_id)
    graph_service = _GraphService({entry.id: entry}, {entry.id: graph_def})
    data_compute_input = MagicMock(name="data_compute_input")
    graph_service.get_graph_data_compute_input = AsyncMock(
        return_value=data_compute_input)
    state_services = SimpleNamespace(graphs=graph_service)
    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(state_services)

    with patch.object(
            graph_driver,
            "_start_graph_refresh",
            new_callable=AsyncMock,
    ) as start_graph_refresh:
        await graph_driver.refresh_graph_v2(GraphIDv2.public(entry.id),
                                            project_id)

    assert graph_service.get_entry_calls == [(project_id, entry.id)]
    graph_service.get_graph_data_compute_input.assert_awaited_once_with(
        GraphIDv2.public(entry.id),
        project_id=project_id,
        user_authorization=None,
    )
    start_graph_refresh.assert_awaited_once_with(data_compute_input)


def _create_sample_table_definition(
    table_name: str,
    connector_id: str | ConnectorID | None = _TEST_CONNECTOR_ID,
) -> ApiTableDefinition:
    """Helper to create a sample table definition for testing.

    Args:
        table_name: Name of the table
        fpid: FPID for the table
        connector_id: Optional connector ID

    Returns:
        TableDefinition object
    """
    cols = [
        Column(
            name=f"{table_name}_id",
            stype=Stype.ID,
            dtype=Dtype.int64,
        ),
        Column(
            name="name",
            stype=Stype.categorical,
            dtype=Dtype.string,
        ),
        Column(
            name="timestamp",
            stype=Stype.timestamp,
            dtype=Dtype.time,
        ),
    ]

    return TableDefinition(
        source_table=S3SourceTable(
            source={
                'mode': 'relative',
                'path': f"{table_name}.csv"
            },
            connector_id=connector_id,
            file_type=FileType.CSV,
        ),
        cols=cols,
        pkey=f"{table_name}_id",
        time_col="timestamp",
    )


def _create_sample_graph_definition(
    table_names: List[str],
    col_groups: List[ColumnKeyGroup] | None = None,
) -> GraphDefinition:
    """Helper to create a sample graph definition for testing.

    Args:
        fpid: FPID for the graph
        table_names: List of table names to include
        col_groups: Optional column key groups for edges

    Returns:
        GraphDefinition object
    """
    tables: Dict[TableName, ApiTableDefinition] = {}
    for name in table_names:
        # Create a proper 32-character hex FPID
        tables[TableName(name)] = _create_sample_table_definition(name)

    return GraphDefinition(
        tables=tables,
        col_groups=list(col_groups) if col_groups else [],
    )


def _create_linked_sample_graph_definition() -> GraphDefinition:
    graph_def = _create_sample_graph_definition(["customers", "orders"])
    orders = TableName("orders")
    graph_def.tables[orders] = dataclasses.replace(
        graph_def.tables[orders],
        cols=[
            *graph_def.tables[orders].cols,
            Column("customer_id", Stype.ID, Dtype.int64),
        ],
    )
    graph_def.col_groups.append(
        ColumnKeyGroup(columns=(
            ColumnKey(table_name="customers", col_name="customers_id"),
            ColumnKey(table_name="orders", col_name="customer_id"),
        )))
    return graph_def


_SQL_QUERY = "SELECT customers_id, name FROM customers"


def _create_sql_graph_definition(
    *,
    claimed_connector_type: str | None = None,
    table: str | None = None,
    referenced_tables: List[str] | None = None,
) -> GraphDefinition:
    """Graph with a single embedded Custom SQL table.

    ``connector_type`` is server-owned; a client sends it unset
    (``claimed_connector_type=None``). A test may pass a stale/incorrect value to
    prove the server derives + overrides it. ``table``/``referenced_tables``
    default to the server-canonical values for ``_SQL_QUERY`` under snowflake so
    the happy path is a canonicalization no-op.
    """
    if table is None:
        table = custom_sql_synthetic_name(_SQL_QUERY, ConnectorType.SNOWFLAKE)
    if referenced_tables is None:
        referenced_tables = referenced_tables_or_none(
            _SQL_QUERY, ConnectorType.SNOWFLAKE) or []
    sql_def = TableDefinition(
        cols=[
            RestColumn("customers_id", Stype.ID, Dtype.int64),
            RestColumn("name", Stype.categorical, Dtype.string),
        ],
        source_table=CustomSQLSourceTable(
            connector_id=_TEST_CONNECTOR_ID,
            connector_type=claimed_connector_type,
            sql=_SQL_QUERY,
            table=table,
            referenced_tables=referenced_tables,
        ),
        pkey="customers_id",
        time_col=None,
    )
    return GraphDefinition(tables={TableName("sql_table"): sql_def},
                           col_groups=[])


class _GraphService:
    def __init__(self, entries, definitions):
        self.entries = entries
        self.definitions = definitions
        self.get_entry_calls = []
        self.get_definition_calls = []
        self.get_alias_calls = []
        self.save_calls = []
        self.list_calls = []
        self.clear_alias_calls = []
        self.clear_alias_by_id_calls = []

    async def get_graph_entry(self, project_id, fpid):
        self.get_entry_calls.append((project_id, fpid))
        return self.entries.get(fpid)

    async def get_graph_entry_by_alias(self, project_id, alias):
        self.get_alias_calls.append((project_id, alias))
        for entry in self.entries.values():
            if entry.alias == alias:
                return entry
        return None

    async def get_graph_definition(self, project_id, fpid):
        self.get_definition_calls.append((project_id, fpid))
        return self.definitions.get(fpid)

    async def list_graphs(self, project_id):
        self.list_calls.append(project_id)
        return list(self.entries.values())

    async def list_named_graphs(self, project_id):
        self.list_calls.append(project_id)
        return [entry for entry in self.entries.values() if entry.alias]

    async def save_graph_definition(
        self,
        project_id,
        graph_definition,
        *,
        alias=None,
        force_rename=False,
        created_by="",
        created_by_name="",
    ):
        self.save_calls.append({
            "project_id": project_id,
            "graph_definition": graph_definition,
            "alias": alias,
            "force_rename": force_rename,
            "created_by": created_by,
            "created_by_name": created_by_name,
        })
        existing = self.entries.get(graph_definition.fpid)
        if existing is not None:
            if existing.alias and alias and existing.alias != alias:
                if not force_rename:
                    raise GraphDefinitionAlreadyExists("duplicate graph")
            existing.alias = alias if force_rename else alias or existing.alias
            if created_by_name and not existing.created_by_name:
                existing.created_by_name = created_by_name
            return existing

        entry = _create_state_graph_entry(
            graph_definition,
            alias,
            project_id=project_id,
            created_by_name=created_by_name,
        )
        self.entries[graph_definition.fpid] = entry
        self.definitions[graph_definition.fpid] = graph_definition
        return entry

    async def clear_graph_alias(self, project_id, alias):
        self.clear_alias_calls.append((project_id, alias))
        entry = await self.get_graph_entry_by_alias(project_id, alias)
        if entry is not None:
            entry.alias = None
        return entry

    async def clear_graph_alias_by_id(self, project_id, fpid):
        self.clear_alias_by_id_calls.append((project_id, fpid))
        entry = self.entries.get(fpid)
        if entry is not None:
            entry.alias = None
        return entry


class _ProjectService:
    def __init__(self):
        self.calls = []

    async def validate_project_for_creation(self, project_id,
                                            user_authorization):
        self.calls.append(
            ("validate_project_for_creation", project_id, user_authorization))


class _TableService:
    def __init__(self, entries):
        self.entries = entries
        self.get_entries_calls = []

    async def get_table_entries(self, project_id, fpids):
        fpids = list(fpids)
        self.get_entries_calls.append((project_id, fpids))
        return {fpid: self.entries[fpid] for fpid in fpids}


class _ConnectorEntry:
    def __init__(self, connector_id, connector_type="snowflake"):
        self.connector_id = ConnectorID(connector_id)
        # Credential-free connector metadata type the graph-create
        # canonicalization resolves the SQL ``connector_type`` against.
        self.connector_type = connector_type

    def as_connector_spec(self):
        return ConnectorSpec(
            connector_id=self.connector_id,
            connector_config={},
        )


class _ConnectorService:
    def __init__(self, entries):
        self.entries = entries
        self.get_calls = []
        self.get_many_calls = []
        self.execution_ref_calls = []
        self.get_for_project_calls = []

    async def get(self, connector_id):
        self.get_calls.append(connector_id)
        return self.entries.get(connector_id)

    async def get_many(self, connector_ids):
        connector_ids = {
            ConnectorID(str(connector_id))
            for connector_id in connector_ids
        }
        self.get_many_calls.append(connector_ids)
        return {
            connector_id: entry
            for connector_id, entry in self.entries.items()
            if connector_id in connector_ids
        }

    def connector_ids_for_graph(self, graph_definition):
        return ConnectorService.connector_ids_for_graph(graph_definition)

    async def execution_ref_for_project(self, principal, project_id,
                                        connector_id, actions):
        connector_id = ConnectorID(str(connector_id))
        self.execution_ref_calls.append(
            (principal, project_id, connector_id, actions))
        return ConnectorExecutionRef(project_id=ProjectID(str(project_id)),
                                     connector_id=connector_id)

    async def get_for_project(self, project_id, connector_id, principal):
        connector_id = ConnectorID(str(connector_id))
        self.get_for_project_calls.append((project_id, connector_id, principal))
        entry = self.entries[connector_id]
        return SimpleNamespace(config=SimpleNamespace(type=SimpleNamespace(
            value=entry.connector_type)))


@pytest.mark.asyncio
async def test_infer_graph_links_sends_only_execution_references():
    project_id = _TEST_PROJECT_ID
    graph_def = _create_sample_graph_definition(["customers", "orders"])
    first_id = _TEST_CONNECTOR_ID
    second_id = _SECOND_CONNECTOR_ID
    customers_table = graph_def.tables[TableName("customers")]
    orders_table = graph_def.tables[TableName("orders")]
    graph_def.tables[TableName("customers")] = dataclasses.replace(
        customers_table,
        source_table=dataclasses.replace(
            customers_table.source_table,
            connector_id=first_id,
        ),
    )
    graph_def.tables[TableName("orders")] = dataclasses.replace(
        orders_table,
        source_table=dataclasses.replace(
            orders_table.source_table,
            connector_id=second_id,
        ),
    )
    first_entry = _ConnectorEntry(first_id)
    second_entry = _ConnectorEntry(second_id)
    connector_service = _ConnectorService({
        first_id: first_entry,
        second_id: second_entry,
    })
    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(
        SimpleNamespace(connectors=connector_service))
    with patch(
            "rest.driver.graph_driver.DataBackend._execute_single_activity",
            new=AsyncMock(return_value=[]),
    ) as execute_activity:
        result = await graph_driver._infer_links(graph_def, project_id)

    assert result is graph_def
    assert {call[2]
            for call in connector_service.execution_ref_calls
            } == {first_id, second_id}
    activity_inputs = execute_activity.await_args.args[2]
    assert activity_inputs[0] is graph_def
    assert set(activity_inputs[1]) == {first_id, second_id}
    assert all(
        isinstance(reference, ConnectorExecutionRef)
        for reference in activity_inputs[1].values())


def _create_state_graph_entry(
    graph_def: GraphDefinition,
    alias: str | None,
    *,
    project_id: ProjectID = _TEST_PROJECT_ID,
    created_by_name: str = "",
) -> GraphDBEntryV2:
    return GraphDBEntryV2(
        timestamps=ResourceTimestamps(
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
        id=graph_def.fpid,
        project_id=project_id,
        table_name_to_dbid={
            StateTableName(table_name): table_def.fpid
            for table_name, table_def in graph_def.tables.items()
        },
        col_groups=graph_def.col_groups,
        alias=alias,
        created_by_name=created_by_name,
    )


def _create_state_table_entry(
    table_def: TableDefinition,
    alias: str | None = None,
    *,
    project_id: ProjectID = _TEST_PROJECT_ID,
) -> TableDBEntryV2:
    return TableDBEntryV2(
        timestamps=ResourceTimestamps(
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
        id=table_def.fpid,
        project_id=project_id,
        config=table_def,
        col_dtypes={col.name: col.dtype
                    for col in table_def.cols},
        alias=alias,
    )


@pytest.mark.asyncio
async def test_get_graph_v2_uses_state_services_when_configured():
    graph_def = _create_sample_graph_definition(["customers", "orders"])
    entry = _create_state_graph_entry(graph_def, "state_graph")
    graph_service = _GraphService({entry.id: entry}, {entry.id: graph_def})
    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(
        SimpleNamespace(graphs=graph_service))

    graph_resource, timestamps, table_mapping = await graph_driver.get_graph_v2(
        entry.id, _TEST_PROJECT_ID)

    assert graph_resource.id == GraphIDv2.public(entry.id)
    assert graph_resource.graph == graph_def
    assert graph_resource.name_alias == "state_graph"
    assert timestamps == entry.timestamps
    assert table_mapping == entry.table_name_to_dbid
    assert graph_service.get_entry_calls == [(_TEST_PROJECT_ID, entry.id)]
    assert graph_service.get_definition_calls == [(_TEST_PROJECT_ID, entry.id)]


@pytest.mark.asyncio
async def test_get_graph_v2_by_alias_uses_state_services_when_configured():
    graph_def = _create_sample_graph_definition(["customers"])
    entry = _create_state_graph_entry(graph_def, "state_graph_alias")
    graph_service = _GraphService({entry.id: entry}, {entry.id: graph_def})
    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(
        SimpleNamespace(graphs=graph_service))

    graph_resource, _, _ = await graph_driver.get_graph_v2_by_alias(
        ID("state_graph_alias"), _TEST_PROJECT_ID)

    assert graph_resource.id == GraphIDv2.public(entry.id)
    assert graph_resource.graph == graph_def
    assert graph_service.get_alias_calls == [(_TEST_PROJECT_ID,
                                              ID("state_graph_alias"))]
    assert graph_service.get_entry_calls == [(_TEST_PROJECT_ID, entry.id)]


@pytest.mark.asyncio
async def test_list_graph_names_v2_uses_state_services_when_configured():
    graph_def_1 = _create_sample_graph_definition(["customers"])
    graph_def_2 = _create_sample_graph_definition(["orders"])
    entry_1 = _create_state_graph_entry(graph_def_1, "graph_1",
                                        created_by_name="alice")
    entry_2 = _create_state_graph_entry(graph_def_2, None)
    graph_service = _GraphService(
        {
            entry_1.id: entry_1,
            entry_2.id: entry_2,
        },
        {
            entry_1.id: graph_def_1,
            entry_2.id: graph_def_2,
        },
    )
    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(
        SimpleNamespace(graphs=graph_service))

    response = await graph_driver.list_graph_names_v2(_TEST_PROJECT_ID)

    assert len(response) == 1
    assert response[0].id == f"graph-{entry_1.id}"
    assert response[0].name == "graph_1"
    assert response[0].num_tables == 1
    assert response[0].created_by_name == "alice"
    assert graph_service.list_calls == [_TEST_PROJECT_ID]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('operation', 'activity_result'),
    [
        pytest.param('infer_links', [], id='infer'),
        pytest.param('suggest_links_v2', [], id='suggest'),
        pytest.param('validate_graph', None, id='validate'),
    ],
)
async def test_graph_connector_operations_use_authorized_references(
    operation,
    activity_result,
):
    project_id = ProjectID("project_" + "1" * 32)
    graph_def = _create_sample_graph_definition(["customers"])
    connector_id = next(iter(graph_def.tables.values())).get_connector_id()
    connector_entry = _ConnectorEntry(connector_id)
    connector_service = _ConnectorService({connector_id: connector_entry})
    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(
        SimpleNamespace(connectors=connector_service))
    with patch(
            "rest.driver.graph_driver.DataBackend._execute_single_activity",
            new=AsyncMock(return_value=activity_result),
    ) as execute_activity:
        graph_arg = (graph_api.GraphDefinition(**dataclasses.asdict(graph_def))
                     if operation == 'validate_graph' else graph_def)
        result = await getattr(graph_driver, operation)(graph_arg, project_id)

    if operation == 'infer_links':
        assert result == graph_def
    elif operation == 'suggest_links_v2':
        assert result == []
    else:
        assert result is not None and not result.errors
    assert connector_service.execution_ref_calls == [
        (None, project_id, connector_id, CONNECTOR_GRAPH_ACTIONS)
    ]
    activity_inputs = execute_activity.await_args.args[2]
    assert activity_inputs[0].tables == graph_def.tables
    assert activity_inputs[1][connector_id] == ConnectorExecutionRef(
        project_id=project_id, connector_id=connector_id)


@pytest.mark.asyncio
async def test_create_full_graph_v2_force_rename_uses_state_services_directly():
    graph_def = _create_sample_graph_definition(table_names=["customers"])
    project_id = "project_" + "a" * 32
    project_service = _ProjectService()
    table_entries = {
        table_def.fpid: _create_state_table_entry(table_def)
        for table_def in graph_def.tables.values()
    }
    existing_graph = _create_state_graph_entry(graph_def, "old_alias",
                                               project_id=ProjectID(project_id))
    graph_service = _GraphService(
        entries={graph_def.fpid: existing_graph},
        definitions={graph_def.fpid: graph_def},
    )
    table_service = _TableService(table_entries)

    graph_driver = _create_graph_driver(
        AsyncMock(),
        AsyncMock(),
        AsyncMock(),
    )
    graph_driver._configure_state_services(
        SimpleNamespace(
            connectors=_ConnectorService({
                _TEST_CONNECTOR_ID:
                _ConnectorEntry(_TEST_CONNECTOR_ID),
            }),
            graphs=graph_service,
            tables=table_service,
            projects=project_service,
        ))
    graph_driver._table_driver._table_resource_responses_from_entries = (
        AsyncMock(return_value=[
            TableResourceResponse(
                id=table_entry.id,
                table=table_entry.config,
                name_alias=table_entry.alias,
                timestamps=table_entry.timestamps,
                stats=TableStats(size_bytes=200, num_rows=20),
            ) for table_entry in table_entries.values()
        ]))

    with _authenticated_principal() as principal:
        result = await graph_driver.create_full_graph_v2(
            graph_def,
            alias="new_alias",
            force_rename=True,
            skip_validation=True,
            project_id=project_id,
        )

    assert result == GraphIDv2.public(graph_def.fpid)
    assert project_service.calls == [("validate_project_for_creation",
                                      project_id, principal)]
    assert graph_service.entries[graph_def.fpid].alias == "new_alias"
    assert graph_service.save_calls[0]["force_rename"] is True


@pytest.mark.asyncio
async def test_create_full_graph_v2_persists_with_state_services():
    project_id = _TEST_PROJECT_ID
    graph_def = _create_sample_graph_definition(table_names=["customers"])
    table_entries = {
        table_def.fpid: _create_state_table_entry(table_def)
        for table_def in graph_def.tables.values()
    }
    graph_service = _GraphService(entries={}, definitions={})
    table_service = _TableService(table_entries)

    graph_driver = _create_graph_driver(
        AsyncMock(),
        AsyncMock(),
        AsyncMock(),
    )
    graph_driver._configure_state_services(
        SimpleNamespace(
            connectors=_ConnectorService({
                _TEST_CONNECTOR_ID:
                _ConnectorEntry(_TEST_CONNECTOR_ID),
            }),
            graphs=graph_service,
            tables=table_service,
            projects=_ProjectService(),
        ))

    table_resources = [
        TableResourceResponse(
            id=table_entry.id,
            table=table_entry.config,
            name_alias=table_entry.alias,
            timestamps=table_entry.timestamps,
            stats=TableStats(size_bytes=200, num_rows=20),
        ) for table_entry in table_entries.values()
    ]
    graph_driver._table_driver._table_resource_responses_from_entries = (
        AsyncMock(return_value=table_resources))

    with _authenticated_principal():
        result = await graph_driver.create_full_graph_v2(
            graph_def,
            alias="state_graph",
            skip_validation=True,
            project_id=str(project_id),
        )

    assert result == GraphIDv2.public(graph_def.fpid)
    assert graph_service.save_calls == [{
        "project_id": project_id,
        "graph_definition": graph_def,
        "alias": "state_graph",
        "force_rename": False,
        "created_by": "_AUTHORIZATION_DISABLED_",
        "created_by_name": "",
    }]
    assert graph_service.entries[graph_def.fpid].alias == "state_graph"
    assert graph_service.entries[graph_def.fpid].created_by_name == ""
    graph_driver._table_driver.validate_table.assert_not_awaited()
    graph_driver._table_driver._table_resource_responses_from_entries.assert_awaited_once_with(
        list(table_entries.values()), fetch_column_stats=False)


@pytest.mark.asyncio
async def test_create_full_graph_v2_rejects_non_warehouse_sql_connector():
    """An embedded Custom SQL table backed by a non-warehouse (file/blob)
    connector is rejected on graph create: a Custom SQL query has no SQL engine
    for such a connector. The server-derived ``connector_type`` gate is a cheap
    correctness/security check that runs even with ``skip_validation=True``, and
    nothing is persisted."""
    project_id = _TEST_PROJECT_ID
    graph_def = _create_sql_graph_definition()
    graph_service = _GraphService(entries={}, definitions={})
    connector_service = _ConnectorService({
        _TEST_CONNECTOR_ID:
        _ConnectorEntry(_TEST_CONNECTOR_ID, connector_type="s3"),
    })

    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(
        SimpleNamespace(
            connectors=connector_service,
            graphs=graph_service,
            tables=_TableService({}),
            projects=_ProjectService(),
        ))

    with _authenticated_principal(), pytest.raises(ValueError):
        await graph_driver.create_full_graph_v2(
            graph_def,
            alias="sql_graph",
            skip_validation=True,
            project_id=str(project_id),
        )
    assert graph_service.save_calls == []


@pytest.mark.asyncio
async def test_create_full_graph_v2_persists_valid_sql_connector_type():
    """Graph create stamps and persists the server-derived warehouse
    ``connector_type``; an (already-canonical) SQL table that already carries the
    derived type is a canonicalization no-op and is persisted with it."""
    project_id = _TEST_PROJECT_ID
    graph_def = _create_sql_graph_definition(claimed_connector_type="snowflake")
    table_entries = {
        table_def.fpid: _create_state_table_entry(table_def)
        for table_def in graph_def.tables.values()
    }
    graph_service = _GraphService(entries={}, definitions={})
    connector_service = _ConnectorService({
        _TEST_CONNECTOR_ID:
        _ConnectorEntry(_TEST_CONNECTOR_ID, connector_type="snowflake"),
    })

    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(
        SimpleNamespace(
            connectors=connector_service,
            graphs=graph_service,
            tables=_TableService(table_entries),
            projects=_ProjectService(),
        ))
    table_resources = [
        TableResourceResponse(
            id=table_entry.id,
            table=table_entry.config,
            name_alias=table_entry.alias,
            timestamps=table_entry.timestamps,
            stats=TableStats(size_bytes=200, num_rows=20),
        ) for table_entry in table_entries.values()
    ]
    graph_driver._table_driver._table_resource_responses_from_entries = (
        AsyncMock(return_value=table_resources))

    with _authenticated_principal():
        result = await graph_driver.create_full_graph_v2(
            graph_def,
            alias="sql_graph",
            skip_validation=True,
            project_id=str(project_id),
        )

    assert result == GraphIDv2.public(graph_def.fpid)
    assert connector_service.get_for_project_calls
    assert len(graph_service.save_calls) == 1
    persisted = graph_service.save_calls[0]["graph_definition"]
    persisted_sql = persisted.tables[TableName("sql_table")].source_table
    assert persisted_sql.connector_type == ConnectorType.SNOWFLAKE


@pytest.mark.asyncio
async def test_create_full_graph_v2_validates_tables_before_persisting():
    project_id = _TEST_PROJECT_ID
    graph_def = _create_sample_graph_definition(table_names=["customers"])
    table_entries = {
        table_def.fpid: _create_state_table_entry(table_def)
        for table_def in graph_def.tables.values()
    }
    graph_service = _GraphService(entries={}, definitions={})
    table_service = _TableService(table_entries)

    graph_driver = _create_graph_driver(
        AsyncMock(),
        AsyncMock(),
        AsyncMock(),
    )
    graph_driver._configure_state_services(
        SimpleNamespace(
            connectors=_ConnectorService({
                _TEST_CONNECTOR_ID:
                _ConnectorEntry(_TEST_CONNECTOR_ID),
            }),
            graphs=graph_service,
            tables=table_service,
            projects=_ProjectService(),
        ))
    graph_driver._table_driver.validate_table = AsyncMock(
        return_value=ValidationResponse(errors=[
            ValidationError(
                title="Table Validation Failure",
                message="source table does not exist",
            )
        ]))

    with _authenticated_principal(), pytest.raises(ValidationException,
                                                   match="does not exist"):
        await graph_driver.create_full_graph_v2(
            graph_def,
            alias="state_graph",
            skip_validation=False,
            project_id=str(project_id),
        )

    graph_driver._table_driver.validate_table.assert_awaited_once_with(
        next(iter(graph_def.tables.values())), project_id=str(project_id))
    assert graph_service.save_calls == []
    assert graph_service.entries == {}


@pytest.mark.asyncio
async def test_validate_graph_tables_runs_validations_concurrently_bounded():
    """Per-table validation must not run one table at a time: request latency
    would scale with table count and push wide graphs past upstream proxy
    deadlines (kumo-ai/kumo#30053). Concurrency stays bounded by
    ``_MAX_CONCURRENT_TABLE_VALIDATIONS``."""
    table_count = _MAX_CONCURRENT_TABLE_VALIDATIONS + 4
    graph_def = _create_sample_graph_definition(
        [f"table_{index:02d}" for index in range(table_count)])
    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(
        SimpleNamespace(connectors=_ConnectorService({
            _TEST_CONNECTOR_ID:
            _ConnectorEntry(_TEST_CONNECTOR_ID),
        })))

    in_flight = 0
    max_in_flight = 0
    release = asyncio.Event()

    async def tracked_validate(table_def, *, project_id):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        try:
            # Block until the test has observed saturation, so the overlap
            # assertion is deterministic; a sequential implementation blocks
            # its first call here and times out instead.
            await asyncio.wait_for(release.wait(), timeout=5)
        finally:
            in_flight -= 1
        return ValidationResponse()

    graph_driver._table_driver.validate_table = AsyncMock(
        side_effect=tracked_validate)

    validation_task = asyncio.create_task(
        graph_driver._validate_graph_tables_for_creation(
            graph_def, _TEST_PROJECT_ID, validate_target=True))

    async def saturated():
        while max_in_flight < _MAX_CONCURRENT_TABLE_VALIDATIONS:
            await asyncio.sleep(0)

    try:
        await asyncio.wait_for(saturated(), timeout=5)
    finally:
        release.set()
    await validation_task

    assert max_in_flight == _MAX_CONCURRENT_TABLE_VALIDATIONS
    assert (
        graph_driver._table_driver.validate_table.await_count == table_count)


@pytest.mark.asyncio
async def test_validate_graph_tables_reports_all_failing_tables_at_once():
    """Validation failures aggregate across tables (sorted by table name)
    instead of raising on the first failing table."""
    graph_def = _create_sample_graph_definition(
        ["orders", "customers", "events"])
    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(
        SimpleNamespace(connectors=_ConnectorService({
            _TEST_CONNECTOR_ID:
            _ConnectorEntry(_TEST_CONNECTOR_ID),
        })))
    failing_messages = {
        graph_def.tables[TableName("orders")].fpid:
        "orders source table does not exist",
        graph_def.tables[TableName("customers")].fpid:
        "customers permission denied",
    }

    async def validate(table_def, *, project_id):
        message = failing_messages.get(table_def.fpid)
        if message is None:
            return ValidationResponse()
        return ValidationResponse(errors=[
            ValidationError(
                title="Table Validation Failure",
                message=message,
            )
        ])

    graph_driver._table_driver.validate_table = AsyncMock(side_effect=validate)

    with pytest.raises(ValidationException) as exc_info:
        await graph_driver._validate_graph_tables_for_creation(
            graph_def, _TEST_PROJECT_ID, validate_target=True)

    message = str(exc_info.value)
    assert ("Table 'customers': Table Validation Failure. "
            "customers permission denied") in message
    assert ("Table 'orders': Table Validation Failure. "
            "orders source table does not exist") in message
    assert message.index("customers") < message.index("orders")
    assert graph_driver._table_driver.validate_table.await_count == 3


@pytest.mark.asyncio
async def test_validate_graph_tables_finishes_siblings_when_one_raises():
    """An exception escaping ``validate_table``'s containment propagates, but
    only after every sibling validation task finishes, so no warehouse probe
    keeps running behind an already-failed request."""
    graph_def = _create_sample_graph_definition(
        ["customers", "events", "orders"])
    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(
        SimpleNamespace(connectors=_ConnectorService({
            _TEST_CONNECTOR_ID:
            _ConnectorEntry(_TEST_CONNECTOR_ID),
        })))
    failing_fpid = graph_def.tables[TableName("customers")].fpid
    completed = []

    async def validate(table_def, *, project_id):
        if table_def.fpid == failing_fpid:
            raise RuntimeError("table service unavailable")
        # Take several event-loop turns so an implementation that propagates
        # the failure without awaiting its siblings leaves them unfinished.
        for _ in range(10):
            await asyncio.sleep(0)
        completed.append(table_def.fpid)
        return ValidationResponse()

    graph_driver._table_driver.validate_table = AsyncMock(side_effect=validate)

    with pytest.raises(RuntimeError, match="table service unavailable"):
        await graph_driver._validate_graph_tables_for_creation(
            graph_def, _TEST_PROJECT_ID, validate_target=True)

    assert graph_driver._table_driver.validate_table.await_count == 3
    assert sorted(completed) == sorted(
        table_def.fpid for name, table_def in graph_def.tables.items()
        if name != TableName("customers"))


@pytest.mark.asyncio
async def test_validate_graph_tables_authorizes_all_connectors_first():
    """A connector-authorization failure surfaces before any warehouse
    validation is launched, even when the denied connector belongs to a
    later table."""
    graph_def = _create_sample_graph_definition(["customers", "orders"])
    orders_table = graph_def.tables[TableName("orders")]
    graph_def.tables[TableName("orders")] = dataclasses.replace(
        orders_table,
        source_table=dataclasses.replace(
            orders_table.source_table,
            connector_id=_SECOND_CONNECTOR_ID,
        ),
    )

    class _DenyingConnectorService(_ConnectorService):
        async def execution_ref_for_project(self, principal, project_id,
                                            connector_id, actions):
            reference = await super().execution_ref_for_project(
                principal, project_id, connector_id, actions)
            if ConnectorID(str(connector_id)) == _SECOND_CONNECTOR_ID:
                raise PermissionError("connector access denied")
            return reference

    connector_service = _DenyingConnectorService({
        _TEST_CONNECTOR_ID:
        _ConnectorEntry(_TEST_CONNECTOR_ID),
        _SECOND_CONNECTOR_ID:
        _ConnectorEntry(_SECOND_CONNECTOR_ID),
    })
    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(
        SimpleNamespace(connectors=connector_service))

    with pytest.raises(PermissionError, match="connector access denied"):
        await graph_driver._validate_graph_tables_for_creation(
            graph_def, _TEST_PROJECT_ID, validate_target=True)

    assert len(connector_service.execution_ref_calls) == 2
    graph_driver._table_driver.validate_table.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_full_graph_v2_raises_on_conflict_without_reuse():
    project_id = _TEST_PROJECT_ID
    graph_def = _create_sample_graph_definition(table_names=["customers"])
    existing_entry = _create_state_graph_entry(graph_def, "old_alias")
    graph_service = _GraphService(
        entries={graph_def.fpid: existing_entry},
        definitions={graph_def.fpid: graph_def},
    )
    table_service = _TableService({
        table_def.fpid:
        _create_state_table_entry(table_def)
        for table_def in graph_def.tables.values()
    })

    graph_driver = _create_graph_driver(
        AsyncMock(),
        AsyncMock(),
        AsyncMock(),
    )
    graph_driver._configure_state_services(
        SimpleNamespace(
            connectors=_ConnectorService({
                _TEST_CONNECTOR_ID:
                _ConnectorEntry(_TEST_CONNECTOR_ID),
            }),
            graphs=graph_service,
            tables=table_service,
            projects=_ProjectService(),
        ))

    with _authenticated_principal(), pytest.raises(
            GraphDefinitionAlreadyExists):
        await graph_driver.create_full_graph_v2(
            graph_def,
            alias="new_alias",
            skip_validation=True,
            project_id=str(project_id),
        )


@pytest.mark.asyncio
async def test_get_edge_stats_uses_state_services_for_compute_input():
    project_id = _TEST_PROJECT_ID
    graph_def = _create_sample_graph_definition(["customers"])
    entry = _create_state_graph_entry(graph_def, "state_graph")
    table_def = graph_def.tables[TableName("customers")]
    connector_id = table_def.get_connector_id()
    connector_entry = _ConnectorEntry(connector_id)
    data_compute_input = SimpleNamespace(
        graph_display_name="state_graph",
        graph_definition=graph_def,
        connectors={connector_id: connector_entry.as_connector_spec()},
    )
    mock_graphs_dao = AsyncMock()
    mock_tables_dao = AsyncMock()
    mock_connectors_dao = AsyncMock()
    graph_driver = _create_graph_driver(mock_graphs_dao, mock_tables_dao,
                                        mock_connectors_dao)
    graph_service = SimpleNamespace(get_graph_data_compute_input=AsyncMock(
        return_value=data_compute_input))
    state_services = SimpleNamespace(graphs=graph_service)
    graph_driver._configure_state_services(state_services)

    with patch(
            "rest.driver.graph_driver.DataBackend.get_edge_stats",
            new_callable=AsyncMock,
            return_value=[],
    ) as mock_get_edge_stats:
        result = await graph_driver.get_edge_stats(GraphIDv2.public(entry.id),
                                                   project_id)

    assert result == EdgeHealthStats(is_ready=True, link_stats=[])
    mock_graphs_dao.get_graph_by_id.assert_not_awaited()
    mock_tables_dao.get_tables_by_ids.assert_not_awaited()
    mock_connectors_dao.get_connector_by_id.assert_not_awaited()
    graph_service.get_graph_data_compute_input.assert_awaited_once_with(
        GraphIDv2.public(entry.id),
        project_id=project_id,
        user_authorization=None,
    )
    data_compute_input = mock_get_edge_stats.await_args.args[1]
    assert data_compute_input.graph_display_name == "state_graph"
    assert data_compute_input.graph_definition == graph_def
    assert (data_compute_input.connectors[connector_id].connector_id ==
            connector_id)


@pytest.mark.asyncio
async def test_get_edge_stats_graph_not_found():
    """Test getting edge stats for a non-existent graph."""
    graph_fpid = FPID("a" * 32)
    graph_id = GraphIDv2.public(graph_fpid)
    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(
        SimpleNamespace(graphs=SimpleNamespace(
            get_graph_data_compute_input=AsyncMock(
                side_effect=GraphDoesNotExistError(graph_id)))))

    with pytest.raises(GraphDoesNotExistError):
        await graph_driver.get_edge_stats(graph_id,
                                          ProjectID("project_" + "1" * 32))


@pytest.mark.asyncio
async def test_graph_to_sdk_uses_state_services_when_configured():
    graph_def = _create_linked_sample_graph_definition()
    entry = _create_state_graph_entry(graph_def, "state_graph")
    table_entries = {
        table_fpid:
        SimpleNamespace(
            config=graph_def.tables[TableName(str(table_name))],
            alias=str(table_name),
        )
        for table_name, table_fpid in entry.table_name_to_dbid.items()
    }
    graph_service = _GraphService({entry.id: entry}, {entry.id: graph_def})
    table_service = _TableService(table_entries)
    connector_service = SimpleNamespace(
        source_table_connector_for_project=AsyncMock(
            return_value=SimpleNamespace(name="Shared graph source")))
    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(
        SimpleNamespace(
            graphs=graph_service,
            tables=table_service,
            connectors=connector_service,
        ))

    result = await graph_driver.graph_to_sdk(GraphIDv2.public(entry.id),
                                             _TEST_PROJECT_ID)

    assert "kumo" in result
    assert "Graph" in result
    assert f'project="{_TEST_PROJECT_ID}"' in result
    assert 'get_by_name("Shared graph source")' in result
    assert graph_service.get_entry_calls == [(_TEST_PROJECT_ID, entry.id)]
    assert table_service.get_entries_calls == [
        (_TEST_PROJECT_ID, list(entry.table_name_to_dbid.values()))
    ]


@pytest.mark.asyncio
async def test_get_graph_metadata_for_sdk_uses_state_services_when_configured():
    graph_def = _create_linked_sample_graph_definition()
    entry = _create_state_graph_entry(graph_def, "state_graph")
    table_entries = {
        table_fpid:
        SimpleNamespace(
            config=graph_def.tables[TableName(str(table_name))],
            alias=str(table_name),
        )
        for table_name, table_fpid in entry.table_name_to_dbid.items()
    }
    graph_service = _GraphService({entry.id: entry}, {entry.id: graph_def})
    table_service = _TableService(table_entries)
    connector_service = SimpleNamespace(
        source_table_connector_for_project=AsyncMock(
            return_value=SimpleNamespace(name="Shared graph source")))
    graph_driver = _create_graph_driver(AsyncMock(), AsyncMock(), AsyncMock())
    graph_driver._configure_state_services(
        SimpleNamespace(
            graphs=graph_service,
            tables=table_service,
            connectors=connector_service,
        ))

    connectors, tables, edges = await graph_driver.get_graph_metadata_for_sdk(
        entry.id, _TEST_PROJECT_ID)

    assert {connector.connector_id
            for connector in connectors} == {_TEST_CONNECTOR_ID}
    assert {
        connector.connector_config["lookup_name"]
        for connector in connectors
    } == {"Shared graph source"}
    assert {table.table_alias_in_graph
            for table in tables} == {
                "customers",
                "orders",
            }
    assert len(edges) == 1
    assert graph_service.get_entry_calls == [(_TEST_PROJECT_ID, entry.id)]
    assert table_service.get_entries_calls == [
        (_TEST_PROJECT_ID, list(entry.table_name_to_dbid.values()))
    ]


@pytest.mark.asyncio
async def test_get_graph_metadata_for_sdk_impl_requires_state_services():
    graph_def = _create_sample_graph_definition(
        ["customers", "orders"],
        col_groups=[
            ColumnKeyGroup(columns=(
                ColumnKey(table_name="customers", col_name="customers_id"),
                ColumnKey(table_name="orders", col_name="orders_id"),
            ))
        ],
    )
    mock_graphs_dao = AsyncMock()
    mock_tables_dao = AsyncMock()
    graph_driver = _create_graph_driver(mock_graphs_dao, mock_tables_dao,
                                        AsyncMock())

    with pytest.raises(RuntimeError,
                       match="GraphDriver requires StateServices"):
        await graph_driver._get_graph_metadata_for_sdk_impl(
            graph_def.fpid, _TEST_PROJECT_ID)

    mock_graphs_dao.get_graph_by_id.assert_not_awaited()
    mock_tables_dao.get_tables_by_ids.assert_not_awaited()
