# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import kumoapi.graph as graph_api
import pandas as pd
import pytest
from kumo_core.types import ConnectorType, TableStatType
from kumo_core.types.schema import FPID
from kumoapi.typing import Dtype, Stype

from kumo.config.graph import ForeignKeyConfig, GraphConfig, PrimaryKeyConfig
from kumo.config.graph import TableConfig as KumoTableConfig
from kumo.config.stats import StatsConfig
from kumo.data.backend_cache import ExactStats
from rest.config.connector import FileConnectorResourceConfig
from rest.config.data_backend import ConnectorSpec
from rest.config.file_type import FileType
from rest.config.graph_v2 import (
    ColumnKey,
    ColumnKeyGroup,
    GraphDBEntryV2,
    GraphDefinition,
)
from rest.config.table_v2 import (
    Column,
    CustomSQLSourceTable,
    GCSSourceTable,
    S3SourceTable,
    TableDefinition,
    TableReferenceMetadata,
)
from rest.driver.graph_logic import (
    GraphLogic,
    canonicalize_graph_custom_sql_tables,
)
from statedb.config.common import (
    ConnectorID,
    GraphIDv2,
    PredictiveQueryID,
    ProjectID,
)
from statedb.config.timestamp import ResourceTimestamps
from statedb.utils.errors import GraphDefinitionAlreadyExists

_TEST_PROJECT_ID = ProjectID("project_" + "1" * 32)
_TEST_CONNECTOR_ID = ConnectorID("connector_" + "1" * 32)


def test_partial_activity_graph_build_closes_every_created_connector(
        monkeypatch):
    from rest.driver import graph_logic
    first = MagicMock()
    second = MagicMock()
    connectors = iter([first, second])
    graph = MagicMock()
    graph.connect.side_effect = [None, RuntimeError('connect failed')]
    monkeypatch.setattr(graph_logic, 'Graph', lambda _name: graph)
    monkeypatch.setattr(graph_logic, 'get_connector_from_spec',
                        lambda _spec: next(connectors))
    specs = [
        SimpleNamespace(connector_id='first'),
        SimpleNamespace(connector_id='second'),
    ]

    with pytest.raises(RuntimeError, match='connect failed'):
        graph_logic._build_graph_from_definition_with_specs(
            SimpleNamespace(), 'graph', specs)

    first.close.assert_called_once_with()
    second.close.assert_called_once_with()


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.fixture
def setup_rest_test_session() -> None:
    return None


class _StateGraphService:
    def __init__(self, entry: GraphDBEntryV2, graph_def: GraphDefinition):
        self.entry = entry
        self.graph_def = graph_def
        self.get_entry_calls = []
        self.get_definition_calls = []
        self.get_entries_calls = []
        self.save_calls = []
        self.list_calls = []
        self.delete_calls = []

    async def get_graph_entry(self, project_id, fpid):
        self.get_entry_calls.append((project_id, fpid))
        return self.entry if fpid == self.entry.id else None

    async def get_graph_entries(self, project_id, fpids):
        fpids = set(fpids)
        self.get_entries_calls.append((project_id, fpids))
        return {self.entry.id: self.entry} if self.entry.id in fpids else {}

    async def get_graph_definition(self, project_id, fpid):
        self.get_definition_calls.append((project_id, fpid))
        return self.graph_def if fpid == self.entry.id else None

    async def list_graphs(self, project_id):
        self.list_calls.append(project_id)
        return [self.entry] if self.entry is not None else []

    async def list_named_graphs(self, project_id):
        self.list_calls.append(project_id)
        return ([self.entry]
                if self.entry is not None and self.entry.alias else [])

    async def decrement_or_delete_graph_definition(self, project_id, fpid):
        self.delete_calls.append((project_id, fpid))
        if self.entry is None or fpid != self.entry.id:
            return False
        self.entry = None
        return True

    async def save_graph_definition(
        self,
        project_id,
        graph_definition,
        *,
        alias=None,
        force_rename=False,
        created_by_name="",
    ):
        self.save_calls.append({
            "project_id": project_id,
            "graph_definition": graph_definition,
            "alias": alias,
            "force_rename": force_rename,
            "created_by_name": created_by_name,
        })
        if self.entry is not None and self.entry.id == graph_definition.fpid:
            if self.entry.alias and alias and self.entry.alias != alias:
                if not force_rename:
                    raise GraphDefinitionAlreadyExists("duplicate graph")
            return self.entry
        return self.entry


class _StatePQueryService:
    def __init__(self, pquery_id: PredictiveQueryID, graph_fpid):
        self.pquery_id = pquery_id
        self.graph_fpid = graph_fpid
        self.get_definition_by_id_calls = []
        self.get_definitions_by_ids_calls = []

    async def get_definition_by_id(self, pquery_id):
        self.get_definition_by_id_calls.append(pquery_id)
        if pquery_id != self.pquery_id:
            return None
        return SimpleNamespace(graph_fpid=self.graph_fpid)

    async def get_definitions_by_ids(self, pquery_ids):
        pquery_ids = list(pquery_ids)
        self.get_definitions_by_ids_calls.append(pquery_ids)
        return {
            pquery_id: SimpleNamespace(graph_fpid=self.graph_fpid)
            for pquery_id in pquery_ids if pquery_id == self.pquery_id
        }


def _create_state_graph_entry(graph_def: GraphDefinition) -> GraphDBEntryV2:
    return GraphDBEntryV2(
        timestamps=ResourceTimestamps(
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
        id=graph_def.fpid,
        project_id=_TEST_PROJECT_ID,
        table_name_to_dbid={
            table_name: table_def.fpid
            for table_name, table_def in graph_def.tables.items()
        },
        col_groups=graph_def.col_groups,
        alias="state_graph",
    )


def _create_state_services_for_graph(
    graph_def: GraphDefinition,
    pquery_id: PredictiveQueryID,
):
    entry = _create_state_graph_entry(graph_def)
    graph_service = _StateGraphService(entry, graph_def)
    pquery_service = _StatePQueryService(pquery_id, entry.id)

    return SimpleNamespace(
        graphs=graph_service,
        pqueries=pquery_service,
    )


def _create_state_services_for_connectors(
    connector_specs: Dict[ConnectorID, ConnectorSpec], ):
    def connector_ids_for_graph(_graph_def):
        return set(connector_specs)

    async def config_for_project(_principal, _project_id, connector_id):
        return connector_specs[connector_id]

    return SimpleNamespace(connectors=SimpleNamespace(
        connector_ids_for_graph=connector_ids_for_graph,
        config_for_project=config_for_project,
    ))


def _create_minimal_graph_definition() -> GraphDefinition:
    table_def = TableDefinition(
        source_table=S3SourceTable(
            connector_id=_TEST_CONNECTOR_ID,
            source={
                'mode': 'relative',
                'path': "customers.parquet"
            },
            file_type=FileType.PARQUET,
        ),
        cols=[
            Column(name="customer_id", stype=Stype.ID, dtype=Dtype.string),
        ],
        pkey="customer_id",
    )
    return GraphDefinition(
        tables={"customers": table_def},
        col_groups=[],
    )


def test_v2_foreign_keys_keep_hosted_order_and_destination_context():
    from rest.driver import graph_logic

    def table(name, columns, pkey=None):
        return TableDefinition(
            source_table=S3SourceTable(
                connector_id=_TEST_CONNECTOR_ID,
                source={
                    'mode': 'relative',
                    'path': f'{name}.parquet'
                },
                file_type=FileType.PARQUET,
            ),
            cols=[Column(column, Stype.ID, Dtype.string) for column in columns],
            pkey=pkey,
        )

    tables = {
        'users': table('users', ('id', 'legacy_id'), 'id'),
        'events': table('events', ('z_user_id', 'a_user_id')),
    }
    groups = [
        ColumnKeyGroup((
            ColumnKey('events', 'z_user_id'),
            ColumnKey('users', 'legacy_id'),
            ColumnKey('users', 'id'),
            ColumnKey('events', 'a_user_id'),
        ))
    ]

    links = graph_logic._make_fkeys_v2(graph_logic._make_col_index_v2(tables),
                                       groups, tables)

    assert links['users'] == []
    assert [link.name for link in links['events']] == [
        'a_user_id',
        'z_user_id',
    ]
    assert all(link.dst_table_name == 'users' for link in links['events'])
    assert all(link.dst_key == 'id' for link in links['events'])
    assert all(link.dst_data_source == _TEST_CONNECTOR_ID
               for link in links['events'])


def test_sdk_edge_export_uses_graph_aliases_and_shared_resolution():
    from rest.driver.graph_logic import get_edge_configs_for_sdk

    def table(source_name, graph_alias, columns, pkey=None):
        return TableReferenceMetadata(
            source_table_name=source_name,
            table_fpid=FPID('1' * 32),
            table_alias_in_graph=graph_alias,
            table_config={
                'cols':
                [Column(column, Stype.ID, Dtype.string) for column in columns],
                'pkey':
                pkey,
                'table_name':
                source_name,
            },
        )

    tables = [
        table('warehouse_users', 'users', ('id', 'legacy_id'), 'id'),
        table('warehouse_events', 'events', ('user_id', )),
    ]
    groups = [
        ColumnKeyGroup((
            ColumnKey('users', 'id'),
            ColumnKey('users', 'legacy_id'),
            ColumnKey('events', 'user_id'),
        ))
    ]

    assert get_edge_configs_for_sdk(groups, tables) == [
        graph_api.Edge('warehouse_events', 'user_id', 'warehouse_users'),
        graph_api.Edge('warehouse_users', 'legacy_id', 'warehouse_users'),
    ]


def test_fixture_export_preserves_column_group_fingerprint(monkeypatch):
    from rest.utils import graph as graph_utils

    users = KumoTableConfig(
        table_name='users',
        data_source='source',
        pkey=PrimaryKeyConfig(name='id', stype=Stype.ID, dtype=Dtype.string),
    )
    events = KumoTableConfig(
        table_name='events',
        data_source='source',
        pkey=None,
        fkeys=[
            ForeignKeyConfig(
                name='user_id',
                stype=Stype.ID,
                dtype=Dtype.string,
                dst_table='source.users',
                dst_key='id',
            )
        ],
    )

    def table_definition(table, _connector_id, _connector_config):
        columns = ('id', ) if table is users else ('user_id', )
        return TableDefinition(
            source_table=S3SourceTable(
                connector_id=_TEST_CONNECTOR_ID,
                source={
                    'mode': 'relative',
                    'path': f'{table.table_name}.parquet'
                },
                file_type=FileType.PARQUET,
            ),
            cols=[Column(column, Stype.ID, Dtype.string) for column in columns],
            pkey='id' if table is users else None,
        )

    monkeypatch.setattr(graph_utils, '_table_config_to_table_definition',
                        table_definition)
    graph = graph_utils.export_kumo_graph_to_fixture(
        GraphConfig(name='graph', tables=[users, events]),
        {
            'source': (
                _TEST_CONNECTOR_ID,
                FileConnectorResourceConfig(name='source', root_dir='.'),
            )
        },
    )

    assert len(graph.col_groups) == 1
    assert [(str(column.table_name), column.col_name)
            for column in graph.col_groups[0].columns] == [
                ('events', 'user_id'),
                ('users', 'id'),
            ]
    assert graph.col_groups[0].fpid == FPID('105e6af6f7cbcd4d5fa3d0db26b985c8')


def _create_infer_links_graph_definition(source_scheme: str) -> GraphDefinition:
    def source_table(table_name: str):
        source_name = f"hnm_tiny/{table_name}.parquet"
        if source_scheme == "s3":
            return S3SourceTable(
                connector_id=_TEST_CONNECTOR_ID,
                source={
                    'mode': 'relative',
                    'path': source_name
                },
                file_type=FileType.PARQUET,
            )
        return GCSSourceTable(
            connector_id=_TEST_CONNECTOR_ID,
            source_table_name=source_name,
            file_type=FileType.PARQUET,
        )

    return GraphDefinition(
        tables={
            "transactions":
            TableDefinition(
                cols=[
                    Column(name="t_dat", stype=Stype.timestamp,
                           dtype=Dtype.time),
                    Column(name="customer_id", stype=Stype.ID,
                           dtype=Dtype.string),
                    Column(name="article_id", stype=Stype.ID,
                           dtype=Dtype.string),
                    Column(name="price", stype=Stype.numerical,
                           dtype=Dtype.float),
                ],
                source_table=source_table("transactions"),
                pkey=None,
                time_col="t_dat",
            ),
            "customers":
            TableDefinition(
                cols=[
                    Column(name="customer_id", stype=Stype.ID,
                           dtype=Dtype.string),
                    Column(name="age", stype=Stype.numerical, dtype=Dtype.int),
                ],
                source_table=source_table("customers"),
                pkey="customer_id",
            ),
            "articles":
            TableDefinition(
                cols=[
                    Column(name="article_id", stype=Stype.ID,
                           dtype=Dtype.string),
                    Column(name="product_type_name", stype=Stype.categorical,
                           dtype=Dtype.string),
                ],
                source_table=source_table("articles"),
                pkey="article_id",
            ),
        },
        col_groups=[],
    )


def _num_rows_stat(val: Any) -> Dict[TableStatType, StatsConfig]:
    return {
        TableStatType.NUM_ROWS:
        StatsConfig(value=val, approx=False, data_version="none")
    }


def test_sum_num_rows_not_computed():
    stats = [
        ExactStats(table_stats=_num_rows_stat(10), col_stats={}),
    ]
    nones = [0]
    # Should return none on missing stats
    assert GraphLogic._sum_num_rows(stats, nones) is None


def test_sum_num_rows_invalid_stat():
    stats = [
        ExactStats(table_stats=_num_rows_stat("pie"), col_stats={}),
        ExactStats(table_stats={}, col_stats={}),
        ExactStats(table_stats=_num_rows_stat(11.5), col_stats={}),
    ]
    nones = []
    # Should return none on present but invalid stats
    assert GraphLogic._sum_num_rows(stats, nones) is None


def test_sum_num_rows_correct_stats():
    stats = [
        ExactStats(table_stats=_num_rows_stat(11), col_stats={}),
        ExactStats(table_stats=_num_rows_stat(12), col_stats={}),
    ]
    nones = []
    assert GraphLogic._sum_num_rows(stats, nones) == 23


@pytest.mark.asyncio
async def test_create_full_graph_def_v2_raises_definition_exists_on_alias_conflict(
):
    graph_fpid = "a" * 32
    # ``_tables`` is empty: the Custom SQL connector_type canonicalization that
    # runs before persistence is a no-op for a graph with no SQL tables, so this
    # alias-conflict path never touches the connector service.
    graph_def = SimpleNamespace(fpid=graph_fpid, _tables=lambda: {})
    graph_entry = SimpleNamespace(id=graph_fpid, alias="old_name")
    graph_service = _StateGraphService(graph_entry, graph_def)
    db = MagicMock()
    db.get_graph_v2_if_exists.side_effect = AssertionError(
        "StateDB path should not read legacy graph storage")
    graph_logic = GraphLogic(
        data_backend_ctx=AsyncMock(),
        state_services=SimpleNamespace(graphs=graph_service),
        project_id=_TEST_PROJECT_ID,
    )

    with pytest.raises(GraphDefinitionAlreadyExists):
        await graph_logic.create_full_graph_def_v2(
            graph_def,
            alias="new_name",
            force_rename=False,
        )
    db.get_graph_v2_if_exists.assert_not_called()


def _sql_graph_definition(
        claimed_connector_type: str | None = None) -> GraphDefinition:
    # ``connector_type`` is server-owned; a client sends it unset. Tests may pass
    # a stale/incorrect value to prove the server derives + overrides it.
    sql_def = TableDefinition(
        source_table=CustomSQLSourceTable(
            connector_id=_TEST_CONNECTOR_ID,
            connector_type=claimed_connector_type,
            sql="SELECT customer_id FROM customers",
            table="custom_sql_provisional",
        ),
        cols=[Column(name="customer_id", stype=Stype.ID, dtype=Dtype.string)],
        pkey="customer_id",
    )
    return GraphDefinition(tables={"sql_table": sql_def}, col_groups=[])


def _state_services_with_connector_type(connector_type: str) -> SimpleNamespace:
    return SimpleNamespace(connectors=SimpleNamespace(get_for_project=AsyncMock(
        return_value=SimpleNamespace(config=SimpleNamespace(
            type=SimpleNamespace(value=connector_type))))))


def test_add_tables_to_graph_v2_marks_custom_sql_tables_is_sql():
    """The validate/train activity builds its graph through
    ``_add_tables_to_graph_v2``. This is the actual path that must yield
    ``is_sql=True`` for a Custom SQL table so a downstream ``graph.config()``
    skips the physical ``get_data_version`` probe of the synthetic source name.
    A physical table stays ``is_sql=False``. The serialized ``TableConfig``
    carries only the boolean marker -- never the SQL text -- and it survives a
    ``Table.from_config`` round-trip (what the activity does before
    ``graph.config()``)."""
    from test.graph.test_custom_sql_infer import _FakeSQLConnector

    from kumo.graph import Graph, Table
    from rest.driver import graph_logic

    sql = "SELECT customer_id FROM customers"
    tables = {
        "sql_table":
        TableDefinition(
            source_table=CustomSQLSourceTable(
                connector_id=_TEST_CONNECTOR_ID,
                connector_type="snowflake",
                sql=sql,
                table="custom_sql_provisional",
            ),
            cols=[
                Column(name="customer_id", stype=Stype.ID, dtype=Dtype.string)
            ],
            pkey="customer_id",
        ),
        "phys_table":
        TableDefinition(
            source_table=S3SourceTable(
                connector_id=_TEST_CONNECTOR_ID,
                source={
                    'mode': 'relative',
                    'path': "customers.parquet"
                },
                file_type=FileType.PARQUET,
            ),
            cols=[
                Column(name="account_id", stype=Stype.ID, dtype=Dtype.string)
            ],
            pkey="account_id",
        ),
    }

    graph = Graph("NA")
    graph.connect(
        _FakeSQLConnector(str(_TEST_CONNECTOR_ID), query_cols=[],
                          sample_df=pd.DataFrame()))
    graph_logic._add_tables_to_graph_v2(graph, tables, col_groups=[])

    sql_table = graph.table("sql_table")
    assert sql_table.is_sql is True
    assert graph.table("phys_table").is_sql is False

    # Serialization carries only the boolean marker (no raw SQL) and survives a
    # Table.from_config round-trip -- exactly the activity's rebuild before
    # graph.config() resolves data_version.
    sql_config = sql_table.config()
    assert sql_config.is_sql is True
    assert not hasattr(sql_config, "custom_sql")
    assert Table.from_config(sql_config).is_sql is True


def test_make_graph_config_from_compute_input_sets_is_sql_marker():
    """The one-shot ``build_graph_config_activity`` builds a ``GraphConfig`` by
    constructing ``TableConfig`` directly from the definition (not from a
    ``Table``), then rebuilds a graph via ``Graph.from_config`` and calls
    ``graph.config()`` -- which resolves ``data_version``. Without the ``is_sql``
    marker carried here, a Custom SQL table would probe its synthetic source
    name as a physical table (a BigQuery 404), so confirm this construction path
    stamps the marker too (physical tables stay ``False``)."""
    from types import SimpleNamespace

    from rest.backends.data_backend.activities import graph_config_activity

    graph_def = GraphDefinition(
        tables={
            "sql_table":
            _sql_graph_definition("snowflake").tables["sql_table"],
            "phys_table":
            TableDefinition(
                source_table=S3SourceTable(
                    connector_id=_TEST_CONNECTOR_ID,
                    source={
                        'mode': 'relative',
                        'path': "customers.parquet"
                    },
                    file_type=FileType.PARQUET,
                ),
                cols=[
                    Column(name="account_id", stype=Stype.ID,
                           dtype=Dtype.string)
                ],
                pkey="account_id",
            ),
        },
        col_groups=[],
    )
    graph_input = SimpleNamespace(graph_definition=graph_def,
                                  graph_display_name="NA")

    config = graph_config_activity.make_graph_config_from_compute_input(
        graph_input, connector_configs={})
    by_name = {t.table_name: t for t in config.tables}
    assert by_name["sql_table"].is_sql is True
    assert by_name["phys_table"].is_sql is False
    # The raw SQL text is never serialized into the config.
    assert not hasattr(by_name["sql_table"], "custom_sql")


@pytest.mark.asyncio
async def test_canonicalize_graph_custom_sql_tables_ignores_client_type():
    """``connector_type`` is server-owned: the shared graph-persistence gate
    (used by both ``GraphDriver.create_full_graph_v2`` and the pquery-driven
    ``GraphLogic.create_full_graph_def_v2``) derives it from the resolved
    connector and overrides any value on the embedded def, so a stale/incorrect
    client-supplied type can never be persisted."""
    graph = _sql_graph_definition("bigquery")  # stale/incorrect client value
    state_services = _state_services_with_connector_type("snowflake")
    result = await canonicalize_graph_custom_sql_tables(graph, _TEST_PROJECT_ID,
                                                        state_services, None)
    src = result._tables()["sql_table"].source_table
    assert src.connector_type == ConnectorType.SNOWFLAKE


@pytest.mark.asyncio
async def test_canonicalize_graph_custom_sql_tables_stamps_resolved_type():
    """An embedded SQL table (client omits ``connector_type``) is stamped with
    the server-derived warehouse type and its provisional synthetic name is
    canonicalized from the normalized SQL."""
    graph = _sql_graph_definition()
    state_services = _state_services_with_connector_type("snowflake")
    result = await canonicalize_graph_custom_sql_tables(graph, _TEST_PROJECT_ID,
                                                        state_services, None)
    src = result._tables()["sql_table"].source_table
    assert src.connector_type == ConnectorType.SNOWFLAKE
    assert src.table != "custom_sql_provisional"


@pytest.mark.asyncio
async def test_canonicalize_graph_custom_sql_tables_noop_without_sql():
    """A graph with no Custom SQL tables is returned unchanged and never touches
    the connector service (so the check is free for physical-only graphs)."""
    graph = _create_minimal_graph_definition()
    connectors = SimpleNamespace(get_for_project=AsyncMock())
    result = await canonicalize_graph_custom_sql_tables(
        graph, _TEST_PROJECT_ID, SimpleNamespace(connectors=connectors), None)
    assert result is graph
    connectors.get_for_project.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_graph_def_resolves_pquery_only_id_with_state_services():
    graph_definition = _create_minimal_graph_definition()
    pquery_id = PredictiveQueryID("state_query")
    state_services = _create_state_services_for_graph(
        graph_definition,
        pquery_id,
    )
    db = MagicMock()
    db.get_graph_v2_if_exists.side_effect = AssertionError(
        "StateDB path should not read legacy graph storage")
    graph_logic = GraphLogic(
        data_backend_ctx=AsyncMock(),
        state_services=state_services,
        project_id=_TEST_PROJECT_ID,
    )

    result = await graph_logic.get_graph_def(GraphIDv2.compose(pquery_id))

    assert result == graph_definition
    assert state_services.pqueries.get_definition_by_id_calls == [pquery_id]
    assert state_services.graphs.get_definition_calls == [
        (_TEST_PROJECT_ID, graph_definition.fpid)
    ]
    db.get_graph_v2_if_exists.assert_not_called()


@pytest.mark.asyncio
async def test_async_state_graph_helpers_do_not_use_sync_bridge(monkeypatch):
    graph_definition = _create_minimal_graph_definition()
    pquery_id = PredictiveQueryID("state_query")
    state_services = _create_state_services_for_graph(
        graph_definition,
        pquery_id,
    )
    graph_logic = GraphLogic(
        data_backend_ctx=AsyncMock(),
        state_services=state_services,
        project_id=_TEST_PROJECT_ID,
    )

    def fail_sync_bridge(coro):
        coro.close()
        raise AssertionError("async graph helper used sync bridge")

    monkeypatch.setattr(graph_logic, "_run_state_service", fail_sync_bridge)

    graph_id = GraphIDv2.compose(pquery_id)

    raw_entry = await graph_logic.async_get_graph_entry_v2(graph_id)
    assert raw_entry == state_services.graphs.entry
    assert await graph_logic.async_exists(graph_id)

    canonical_entry = await graph_logic.async_get_graph_entry(graph_id)
    assert canonical_entry.display_name == "state_graph"

    entries = await graph_logic.async_get_many_graph_entries([graph_id])
    assert entries == [canonical_entry]

    graph_names = await graph_logic.async_list_graph_names_v2()
    assert [graph.name for graph in graph_names] == ["state_graph"]

    assert await graph_logic.async_delete_full_graph_def_v2(graph_id)
    assert state_services.graphs.delete_calls == [(_TEST_PROJECT_ID,
                                                   graph_definition.fpid)]


@pytest.mark.asyncio
async def test_infer_links():
    graph_def = _create_infer_links_graph_definition("s3")

    state_services = _create_state_services_for_connectors({
        _TEST_CONNECTOR_ID:
        ConnectorSpec(
            connector_id=_TEST_CONNECTOR_ID,
            connector_config=FileConnectorResourceConfig(
                str(_TEST_CONNECTOR_ID),
                "s3://kumo-public-datasets/hnm_tiny/",
            ).as_connector().config(),
        )
    })
    graph_logic = GraphLogic(
        data_backend_ctx=AsyncMock(),
        state_services=state_services,
        project_id=_TEST_PROJECT_ID,
    )
    new_graph = await graph_logic.infer_links(graph_def)
    assert len(new_graph.col_groups) == 2

    assert new_graph.col_groups[0].columns[0].table_name == 'customers'
    assert new_graph.col_groups[0].columns[0].col_name == 'customer_id'
    assert new_graph.col_groups[0].columns[1].table_name == 'transactions'
    assert new_graph.col_groups[0].columns[1].col_name == 'customer_id'

    assert new_graph.col_groups[1].columns[0].table_name == 'articles'
    assert new_graph.col_groups[1].columns[0].col_name == 'article_id'
    assert new_graph.col_groups[1].columns[1].table_name == 'transactions'
    assert new_graph.col_groups[1].columns[1].col_name == 'article_id'


@pytest.mark.asyncio
async def test_infer_links_with_gcs_source_table():
    """Test that infer_links works with GCS source tables."""
    graph_def = _create_infer_links_graph_definition("gs")

    state_services = _create_state_services_for_connectors({
        _TEST_CONNECTOR_ID:
        ConnectorSpec(
            connector_id=_TEST_CONNECTOR_ID,
            connector_config=FileConnectorResourceConfig(
                str(_TEST_CONNECTOR_ID),
                "gs://kumo-public-datasets/hnm_tiny/",
            ).as_connector().config(),
        )
    })
    graph_logic = GraphLogic(
        data_backend_ctx=AsyncMock(),
        state_services=state_services,
        project_id=_TEST_PROJECT_ID,
    )
    new_graph = await graph_logic.infer_links(graph_def)
    assert len(new_graph.col_groups) == 2

    assert new_graph.col_groups[0].columns[0].table_name == 'customers'
    assert new_graph.col_groups[0].columns[0].col_name == 'customer_id'
    assert new_graph.col_groups[0].columns[1].table_name == 'transactions'
    assert new_graph.col_groups[0].columns[1].col_name == 'customer_id'

    assert new_graph.col_groups[1].columns[0].table_name == 'articles'
    assert new_graph.col_groups[1].columns[0].col_name == 'article_id'
    assert new_graph.col_groups[1].columns[1].table_name == 'transactions'
    assert new_graph.col_groups[1].columns[1].col_name == 'article_id'
