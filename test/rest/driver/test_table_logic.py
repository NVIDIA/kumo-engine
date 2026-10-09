# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
import re
from datetime import datetime
from test.rest.config.test_table import _s3_named  # noqa: E402
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch

import pytest
from kumo_core.types import ConnectorType
from kumo_core.types.schema import FPID
from kumoapi.data_source import DataSourceType
from kumoapi.typing import Dtype, Stype

from kumo.connector.connector import ConnectorError
from kumo.data.custom_sql import (
    custom_sql_synthetic_name,
    referenced_tables_or_none,
)
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.graph_v2 import GraphDBEntryV2
from rest.config.table_v2 import (
    Column,
    CustomSQLSourceTable,
    FileType,
    GCSSourceTable,
    S3SourceTable,
    TableDBEntryV2,
    TableDefinition,
)
from rest.driver.error_constants import (
    TABLE_READ_ERROR_MESSAGE,
    TABLE_READ_PROVIDER_ERROR,
)
from rest.driver.table_logic import TableLogic
from rest.utils.routers import ValidationHTTPException
from statedb.config.common import (
    ConnectorID,
    PredictiveQueryID,
    ProjectID,
    TableIDv2,
    TableName,
)
from statedb.config.predictive_query import PQueryDefinition
from statedb.config.table import TableDataVersion
from statedb.config.timestamp import ResourceTimestamps
from statedb.rest.connector_authorization import CONNECTOR_TABLE_SOURCE_ACTIONS

_PROJECT_ID = ProjectID("project_" + "1" * 32)
_CONNECTOR_ID = ConnectorID("connector_" + "1" * 32)


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.fixture
def setup_rest_test_session() -> None:
    return None


@pytest.mark.asyncio
async def test_get_table_v2_for_dangling_v2_tables():
    definition = TableDefinition(
        cols=[Column(name='a', stype=Stype.ID, dtype=Dtype.int)],
        source_table=S3SourceTable(
            connector_id=_CONNECTOR_ID,
            source={
                'mode': 'relative',
                'path': 'path/file.parquet'
            },
            file_type=FileType.PARQUET,
            data_source_type=DataSourceType.S3,
        ),
    )
    fpid = definition.fpid
    table_entry = TableDBEntryV2(
        id=fpid,
        project_id=_PROJECT_ID,
        config=definition,
        col_dtypes={},
        timestamps=ResourceTimestamps(
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    )
    table_logic = TableLogic(
        MagicMock(),
        state_services=SimpleNamespace(
            tables=SimpleNamespace(
                get_table_entries_by_aliases=AsyncMock(
                    return_value={
                        TableName("customers"): table_entry,
                    }),
                get_table_entries=AsyncMock(return_value={fpid: table_entry}),
            ),
            pqueries=SimpleNamespace(get_definitions_by_ids=AsyncMock(
                return_value={})),
            graphs=SimpleNamespace(),
        ),
        project_id=_PROJECT_ID,
    )

    response = await table_logic._get_many_table_entries_v2_from_state_services(
        [TableIDv2('some_pq_name.customers-NA')])

    assert response is not None
    assert len(response) == 1
    assert response[0] is not None
    table_entry: TableDBEntryV2 = response[0]
    assert fpid == response[0].id


@pytest.mark.asyncio
async def test_get_table_v2_for_gcs_tables():
    """Test that GCS source tables are handled correctly."""
    definition = TableDefinition(
        cols=[Column(name='a', stype=Stype.ID, dtype=Dtype.int)],
        source_table=GCSSourceTable(
            connector_id=_CONNECTOR_ID,
            source_table_name='path/file.parquet',
            file_type=FileType.PARQUET,
            data_source_type=DataSourceType.GCS,
        ),
    )
    fpid = definition.fpid
    table_entry = TableDBEntryV2(
        id=fpid,
        project_id=_PROJECT_ID,
        config=definition,
        col_dtypes={},
        timestamps=ResourceTimestamps(
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    )
    table_logic = TableLogic(
        MagicMock(),
        state_services=SimpleNamespace(
            tables=SimpleNamespace(
                get_table_entries_by_aliases=AsyncMock(
                    return_value={
                        TableName("customers"): table_entry,
                    }),
                get_table_entries=AsyncMock(return_value={fpid: table_entry}),
            ),
            pqueries=SimpleNamespace(get_definitions_by_ids=AsyncMock(
                return_value={})),
            graphs=SimpleNamespace(),
        ),
        project_id=_PROJECT_ID,
    )

    response = await table_logic._get_many_table_entries_v2_from_state_services(
        [TableIDv2('some_pq_name.customers-NA')])

    assert response is not None
    assert len(response) == 1
    assert response[0] is not None
    table_entry: TableDBEntryV2 = response[0]
    assert fpid == response[0].id


@pytest.mark.asyncio
async def test_get_many_table_definitions_state_services_bulk_resolves_pquery_graph_ids(
):
    customers_def = _s3_named()
    orders_def = TableDefinition(
        cols=customers_def.cols,
        source_table=S3SourceTable(
            file_type=FileType.PARQUET,
            connector_id=_CONNECTOR_ID,
            source={
                'mode': 'relative',
                'path': 'orders'
            },
        ),
        pkey='id',
        time_col='ts',
    )
    customers_entry = TableDBEntryV2(
        id=customers_def.fpid,
        project_id=_PROJECT_ID,
        alias=None,
        config=customers_def,
        col_dtypes={},
        timestamps=ResourceTimestamps(
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    )
    orders_entry = TableDBEntryV2(
        id=orders_def.fpid,
        project_id=_PROJECT_ID,
        alias=None,
        config=orders_def,
        col_dtypes={},
        timestamps=ResourceTimestamps(
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    )
    graph_fpid = FPID("b" * 32)
    pquery_id = PredictiveQueryID("some_pq_name")
    pquery_def = PQueryDefinition(
        timestamps=ResourceTimestamps(
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
        pquery_name="some_pq_name",
        query_str="PREDICT COUNT(customers.*) FOR EACH customers.id",
        graph_fpid=graph_fpid,
    )
    graph_entry = GraphDBEntryV2(
        id=graph_fpid,
        project_id=_PROJECT_ID,
        table_name_to_dbid={
            TableName("customers"): customers_def.fpid,
            TableName("orders"): orders_def.fpid,
        },
        col_groups=[],
        timestamps=ResourceTimestamps(
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    )
    table_service = SimpleNamespace(
        get_table_entries=AsyncMock(return_value={
            customers_def.fpid: customers_entry,
            orders_def.fpid: orders_entry,
        }),
        get_table_entry_by_alias=AsyncMock(
            side_effect=AssertionError("scalar table alias fallback used")),
        get_table_entries_by_aliases=AsyncMock(return_value={}),
    )
    pquery_service = SimpleNamespace(
        get_definitions_by_ids=AsyncMock(return_value={pquery_id: pquery_def}),
        resolve_fpids=AsyncMock(
            side_effect=AssertionError("pquery fpid resolver used")),
        get_definitions=AsyncMock(
            side_effect=AssertionError("pquery fpid definitions lookup used")),
        get_alias=AsyncMock(
            side_effect=AssertionError("scalar pquery alias lookup used")),
        get_definition=AsyncMock(
            side_effect=AssertionError("scalar pquery definition lookup used")),
    )
    graph_service = SimpleNamespace(
        get_graph_entries=AsyncMock(return_value={graph_fpid: graph_entry}),
        get_graph_entry=AsyncMock(
            side_effect=AssertionError("scalar graph lookup used")),
    )
    table_logic = TableLogic(
        MagicMock(),
        state_services=SimpleNamespace(
            tables=table_service,
            pqueries=pquery_service,
            graphs=graph_service,
        ),
        project_id=_PROJECT_ID,
    )

    result = await table_logic.get_many_table_definitions([
        TableIDv2("some_pq_name.customers-NA"),
        TableIDv2("some_pq_name.orders-NA"),
    ])

    assert result == [customers_def, orders_def]
    pquery_service.get_definitions_by_ids.assert_awaited_once_with([pquery_id])
    pquery_service.resolve_fpids.assert_not_awaited()
    pquery_service.get_definitions.assert_not_awaited()
    pquery_service.get_alias.assert_not_awaited()
    pquery_service.get_definition.assert_not_awaited()
    graph_service.get_graph_entries.assert_awaited_once_with(
        _PROJECT_ID, {graph_fpid})
    graph_service.get_graph_entry.assert_not_awaited()
    table_service.get_table_entries.assert_awaited_once_with(
        _PROJECT_ID, {
            customers_def.fpid,
            orders_def.fpid,
        })
    table_service.get_table_entry_by_alias.assert_not_awaited()
    table_service.get_table_entries_by_aliases.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_many_table_definitions_state_services_bulk_resolves_aliases(
):
    customers_def = _s3_named()
    orders_def = TableDefinition(
        cols=customers_def.cols,
        source_table=S3SourceTable(
            file_type=FileType.PARQUET,
            connector_id=_CONNECTOR_ID,
            source={
                'mode': 'relative',
                'path': 'orders'
            },
        ),
        pkey='id',
        time_col='ts',
    )
    customers_entry = TableDBEntryV2(
        id=customers_def.fpid,
        project_id=_PROJECT_ID,
        alias="customers",
        config=customers_def,
        col_dtypes={},
        timestamps=ResourceTimestamps(
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    )
    orders_entry = TableDBEntryV2(
        id=orders_def.fpid,
        project_id=_PROJECT_ID,
        alias="orders",
        config=orders_def,
        col_dtypes={},
        timestamps=ResourceTimestamps(
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    )
    table_service = SimpleNamespace(
        get_table_entries=AsyncMock(return_value={
            customers_def.fpid: customers_entry,
            orders_def.fpid: orders_entry,
        }),
        get_table_entry_by_alias=AsyncMock(
            side_effect=AssertionError("scalar table alias lookup used")),
        get_table_entries_by_aliases=AsyncMock(
            return_value={
                TableName("customers"): customers_entry,
                TableName("orders"): orders_entry,
            }),
    )
    pquery_service = SimpleNamespace(
        get_definitions_by_ids=AsyncMock(return_value={}),
        resolve_fpids=AsyncMock(
            side_effect=AssertionError("pquery fpid resolver used")),
        get_alias=AsyncMock(
            side_effect=AssertionError("scalar pquery alias lookup used")),
        get_definition=AsyncMock(
            side_effect=AssertionError("pquery definition lookup used")),
    )
    table_logic = TableLogic(
        MagicMock(),
        state_services=SimpleNamespace(
            tables=table_service,
            pqueries=pquery_service,
            graphs=SimpleNamespace(get_graph_entries=AsyncMock(
                return_value={})),
        ),
        project_id=_PROJECT_ID,
    )

    result = await table_logic.get_many_table_definitions([
        TableIDv2("missing_pq.customers-NA"),
        TableIDv2("missing_pq.orders-NA"),
    ])

    assert result == [customers_def, orders_def]
    pquery_service.get_definitions_by_ids.assert_awaited_once_with(
        [PredictiveQueryID("missing_pq")])
    pquery_service.resolve_fpids.assert_not_awaited()
    pquery_service.get_alias.assert_not_awaited()
    pquery_service.get_definition.assert_not_awaited()
    table_service.get_table_entries_by_aliases.assert_awaited_once()
    aliases_arg = (
        table_service.get_table_entries_by_aliases.await_args.args[1])
    assert list(aliases_arg) == [TableName("customers"), TableName("orders")]
    table_service.get_table_entry_by_alias.assert_not_awaited()
    table_service.get_table_entries.assert_awaited_once_with(
        _PROJECT_ID, {
            customers_def.fpid,
            orders_def.fpid,
        })


@pytest.mark.asyncio
async def test_get_table_v2_state_services_resolves_alias():
    table_def = _s3_named()
    table_entry = TableDBEntryV2(
        id=table_def.fpid,
        project_id=_PROJECT_ID,
        alias="customers",
        config=table_def,
        col_dtypes={},
        timestamps=ResourceTimestamps(
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    )
    table_service = SimpleNamespace(
        get_table_entries=AsyncMock(return_value={table_def.fpid: table_entry}),
        get_table_entries_by_aliases=AsyncMock(
            return_value={
                TableName("customers"): table_entry,
            }),
    )
    table_logic = TableLogic(
        MagicMock(),
        state_services=SimpleNamespace(
            tables=table_service,
            pqueries=SimpleNamespace(),
            graphs=SimpleNamespace(),
        ),
        project_id=_PROJECT_ID,
    )

    result = await table_logic.get_table_v2(TableIDv2("NA.customers-NA"))

    assert result == table_def
    table_service.get_table_entries_by_aliases.assert_awaited_once()
    aliases_arg = table_service.get_table_entries_by_aliases.await_args.args[1]
    assert list(aliases_arg) == [TableName("customers")]
    table_service.get_table_entries.assert_awaited_once_with(
        _PROJECT_ID, {table_def.fpid})


@pytest.mark.asyncio
@patch.object(TableLogic, '_get_table_def_and_connector_ref')
@pytest.mark.parametrize('connector_exception_type',
                         [FileNotFoundError, ConnectorError])
async def test_get_col_stats_and_sample_rows_unreadable_sample_rows(
        mock_get_table_def_and_connector_ref_method, connector_exception_type):

    connector_exception_message = 'TEST_EXCEPTION'
    connector_exception = connector_exception_type(connector_exception_message)

    test_table_name = 'TEST_TABLE'
    test_connector_id = 'TEST_CONNECTOR'

    table_def_mock = Mock()
    table_def_mock.get_source_table_name.return_value = test_table_name
    table_def_mock.get_connector_id.return_value = test_connector_id
    table_def_mock.cols = []

    reference = ConnectorExecutionRef(project_id=_PROJECT_ID,
                                      connector_id=_CONNECTOR_ID)
    mock_get_table_def_and_connector_ref_method.return_value = (table_def_mock,
                                                                reference)

    table_logic = TableLogic(Mock(), project_id=_PROJECT_ID)

    error_message = TABLE_READ_ERROR_MESSAGE.format(test_table_name,
                                                    test_connector_id,
                                                    connector_exception)
    with patch(
            'rest.driver.table_logic.DataBackend._execute_single_activity',
            new=AsyncMock(side_effect=ValidationHTTPException(
                error_message, connector_exception)),
    ), pytest.raises(ValidationHTTPException, match=re.escape(error_message)):
        await table_logic.get_col_stats_and_sample_rows(Mock())


@pytest.mark.asyncio
async def test_get_col_stats_at_version_uses_fingerprint_connector_cache():
    table_def = _s3_named()
    version = TableDataVersion("test-version")
    cache = Mock()
    cache.get_min_col_stats.return_value = {}
    table_logic = TableLogic(Mock(), project_id=_PROJECT_ID)

    with patch.object(
            TableLogic,
            "_get_table_def_and_connector_ref",
            new=AsyncMock(return_value=(
                table_def,
                ConnectorExecutionRef(project_id=_PROJECT_ID,
                                      connector_id=_CONNECTOR_ID),
            )),
    ), patch(
            "rest.driver.table_logic.get_active_cache",
            return_value=cache,
    ):
        result = await table_logic.get_col_stats_at_version(
            TableIDv2("NA.transactions-NA"),
            version,
        )

    table_fingerprint = table_def.fingerprint()
    cache.load_min_col_stats_cache.assert_called_once_with(
        connector_name=table_fingerprint.connector_name,
        table_fingerprint=table_fingerprint,
        data_version=version,
    )
    cache.get_min_col_stats.assert_called_once()
    assert result.stats_dict == {}


@pytest.mark.asyncio
async def test_get_table_def_and_connector_ref_existing_connector_path():
    table_def = Mock()
    table_def.get_connector_id.return_value = _CONNECTOR_ID

    expected_ref = ConnectorExecutionRef(project_id=_PROJECT_ID,
                                         connector_id=_CONNECTOR_ID)
    execution_ref_for_project = AsyncMock(return_value=expected_ref)
    table_logic = TableLogic(
        Mock(),
        state_services=SimpleNamespace(connectors=SimpleNamespace(
            execution_ref_for_project=execution_ref_for_project)),
        project_id=_PROJECT_ID,
    )

    with patch.object(
            table_logic,
            "get_table_v2",
            new=AsyncMock(return_value=table_def),
    ):
        resolved_def, resolved_spec = (
            await table_logic._get_table_def_and_connector_ref(Mock()))

    assert resolved_def == table_def
    assert resolved_spec == expected_ref
    execution_ref_for_project.assert_awaited_once_with(
        ANY, _PROJECT_ID, table_def.get_connector_id.return_value,
        CONNECTOR_TABLE_SOURCE_ACTIONS)


@pytest.mark.asyncio
async def test_validate_table_named_connector_requires_state_services():
    db_mock = Mock()
    db_mock.get_connector_if_exists.side_effect = AssertionError(
        "legacy connector lookup used")
    table_logic = TableLogic(Mock(), project_id=_PROJECT_ID)

    with pytest.raises(RuntimeError, match="TableLogic requires StateServices"):
        await table_logic.validate_table(_s3_named())

    db_mock.get_connector_if_exists.assert_not_called()


# ---------------------------------------------------------------------------
# trigger_async_refresh_v2
# ---------------------------------------------------------------------------


def _make_table_logic_for_refresh_v2():
    """Build a `TableLogic` with the minimum mock surface needed to drive
    `trigger_async_refresh_v2` without hitting a real DB or temporal cluster.
    """
    data_backend_ctx_mock = Mock()
    data_backend_ctx_mock.temporal_client = Mock()
    execution_ref = ConnectorExecutionRef(
        project_id=_PROJECT_ID,
        connector_id=_CONNECTOR_ID,
        authorization_token='version-token',
    )
    refresh_ref = ConnectorExecutionRef(
        project_id=_PROJECT_ID,
        connector_id=_CONNECTOR_ID,
        authorization_token='refresh-token',
    )
    state_services = SimpleNamespace(
        tables=SimpleNamespace(get_table_entries=AsyncMock(return_value={})),
        connectors=SimpleNamespace(
            execution_ref_for_project=AsyncMock(return_value=execution_ref),
            fresh_execution_refs_for_workflow=AsyncMock(
                return_value=(refresh_ref, )),
        ),
    )
    return TableLogic(
        data_backend_ctx_mock,
        state_services=state_services,
        project_id=_PROJECT_ID,
    )


@pytest.mark.asyncio
async def test_trigger_async_refresh_v2_mints_a_fresh_workflow_reference():
    table_def = _s3_named()
    table_logic = _make_table_logic_for_refresh_v2()
    connectors = table_logic._require_state_services().connectors

    with patch('rest.driver.table_logic.get_active_cache') as mock_gac, \
            patch('rest.driver.table_logic.DataBackend.'
                  'get_table_data_version_in_activity',
                  new=AsyncMock(return_value='version-1')) as get_version, \
            patch('rest.driver.table_logic.DataBackend.'
                  'get_many_table_refresh_results',
                  new=AsyncMock(return_value=[None])), \
            patch('rest.driver.table_logic.DataBackend.start_refresh_table',
                  new=AsyncMock()) as start_refresh:
        mock_gac.return_value.table_version_cache.get.coroutine = AsyncMock(
            return_value=None)
        await table_logic.trigger_async_refresh_v2(
            table_def,
            TableIDv2('NA.tbl-abcdef123456abcdef123456abcdef12'),
        )

    version_reference = get_version.await_args.args[1]
    refresh_reference = start_refresh.await_args.args[1]
    assert version_reference.authorization_token == 'version-token'
    assert refresh_reference.authorization_token == 'refresh-token'
    assert version_reference.authorization_token != (
        refresh_reference.authorization_token)
    connectors.fresh_execution_refs_for_workflow.assert_awaited_once_with(
        ANY,
        (version_reference, ),
        CONNECTOR_TABLE_SOURCE_ACTIONS,
    )


@pytest.mark.asyncio
async def test_trigger_async_refresh_v2_sql_table_always_recomputes_version():
    """A Custom SQL table's version is a content hash the name-keyed
    table_version_cache cannot represent. Even on a cache hit with
    refresh_source=False, the version must be recomputed (folding in the SQL,
    output cols, and referenced-table versions) rather than served stale."""
    sql_def = TableDefinition(
        cols=[Column(name='item_key', stype=Stype.ID, dtype=Dtype.string)],
        source_table=CustomSQLSourceTable(
            connector_id=_CONNECTOR_ID,
            connector_type='snowflake',
            sql='SELECT item_key FROM sales',
            table='custom_sql_abc',
            referenced_tables=['sales'],
        ),
        pkey='item_key',
        time_col=None,
    )
    table_logic = _make_table_logic_for_refresh_v2()
    connectors = table_logic._require_state_services().connectors

    with patch('rest.driver.table_logic.get_active_cache') as mock_gac, \
            patch('rest.driver.table_logic.DataBackend.'
                  'get_table_data_version_in_activity',
                  new=AsyncMock(return_value='sql-hash-1')) as get_version, \
            patch('rest.driver.table_logic.DataBackend.'
                  'get_many_table_refresh_results',
                  new=AsyncMock(return_value=[None])), \
            patch('rest.driver.table_logic.DataBackend.start_refresh_table',
                  new=AsyncMock()):
        # A synthetic-name cache hit would let a physical table skip the probe;
        # the SQL table must ignore it and still recompute.
        mock_gac.return_value.table_version_cache.get.coroutine = AsyncMock(
            return_value=TableDataVersion('stale-cached'))
        await table_logic.trigger_async_refresh_v2(
            sql_def,
            TableIDv2('NA.tbl-abcdef123456abcdef123456abcdef12'),
            refresh_source=False,
        )

    get_version.assert_awaited_once()
    # Cache hit + refresh_source=False → recompute the content hash using cached
    # referenced-table versions (don't force-refresh them), threading table_def.
    assert get_version.await_args.kwargs['refresh_table_version'] is False
    assert get_version.await_args.kwargs['table_def'] is sql_def
    # The probe consumed the ref's capability, so a fresh workflow ref is minted.
    connectors.fresh_execution_refs_for_workflow.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('exc_type', [FileNotFoundError, ConnectorError])
async def test_trigger_async_refresh_v2_unreadable_source_raises(exc_type):
    """When the source-version probe raises, ``trigger_async_refresh_v2``
    surfaces it through ``_table_read_error_v2`` — i.e. wraps it in a
    ``ValidationHTTPException`` whose message uses the v1-equivalent
    formatting (verifies the v2 error helper is wired into this path).
    """
    table_def = _s3_named()
    expected_msg = re.escape(
        TABLE_READ_ERROR_MESSAGE.format(
            str(table_def.get_source_table_name()),
            table_def.get_connector_id(),
            TABLE_READ_PROVIDER_ERROR,
        ))
    table_logic = _make_table_logic_for_refresh_v2()

    with patch('rest.driver.table_logic.get_active_cache') as mock_gac, \
            patch('rest.driver.table_logic.DataBackend.'
                  'get_table_data_version_in_activity',
                  new=AsyncMock(side_effect=exc_type('boom'))):
        # Force the "no cached version → probe source" branch.
        mock_gac.return_value.table_version_cache.get \
            .coroutine = AsyncMock(return_value=None)

        with pytest.raises(ValidationHTTPException, match=expected_msg):
            await table_logic.trigger_async_refresh_v2(
                table_def, TableIDv2('NA.tbl-abcdef123456abcdef123456abcdef12'))


@pytest.mark.asyncio
async def test_trigger_async_refresh_v2_skips_when_already_completed():
    """If a prior refresh result is DONE/COMPLETED and the caller did not
    force a source refresh, the function returns the cached version
    without calling ``DataBackend.start_refresh_table``.
    """
    from kumo.data.backend_cache import AsyncDataTaskStatus, TableRefreshStage
    from rest.backends.data_backend.data_backend import DataBackend
    from rest.backends.data_backend.workflows import table_refresh_workflow
    from statedb.config.table import TableDataVersion

    table_def = _s3_named()
    cached_version = TableDataVersion('v0')
    table_logic = _make_table_logic_for_refresh_v2()

    completed_result = Mock()
    # `AsyncDataTaskStatus.completed` is a @property that returns True for
    # DONE/FAILED, so setting status=DONE already makes `.status.completed`
    # evaluate to True (which is what production checks).
    completed_result.status = AsyncDataTaskStatus.DONE
    completed_result.stage = TableRefreshStage.COMPLETED

    with patch('rest.driver.table_logic.get_active_cache') as mock_gac, \
            patch.object(DataBackend, 'get_many_table_refresh_results',
                         new=AsyncMock(return_value=[completed_result])), \
            patch.object(DataBackend, 'start_refresh_table',
                         new=AsyncMock()) as start_refresh_mock, \
            patch.object(table_refresh_workflow.TableRefreshWorkflow,
                         'compose_wf_id',
                         return_value='wf-id-stub'):
        # Cached source version present + refresh_source=False → the
        # "skip relaunch" arm fires.
        mock_gac.return_value.table_version_cache.get \
            .coroutine = AsyncMock(return_value=cached_version)

        result = await table_logic.trigger_async_refresh_v2(
            table_def, TableIDv2('NA.tbl-abcdef123456abcdef123456abcdef12'),
            refresh_source=False)

    assert result == TableDataVersion(cached_version)
    start_refresh_mock.assert_not_called()


@pytest.mark.asyncio
async def test_create_table_v2_canonicalizes_custom_sql_source():
    """`create_table_v2` must stamp a Custom SQL table's server-owned fields
    (derive `connector_type`, recompute `referenced_tables`, assign the
    canonical synthetic name) BEFORE it takes the fingerprint and persists — so
    this persist path cannot store a SQL table with an underived
    `connector_type`, a stale client-supplied reference list, or a provisional
    name. Mirrors the live `TableDriver.create_full_table` guard.
    """
    sql = 'SELECT item_key, total FROM sales'
    # Client shape: `connector_type` unset and a bogus `referenced_tables` list
    # (both server-owned), under the SDK's provisional synthetic name.
    table_def = TableDefinition(
        cols=[
            Column('item_key', Stype.ID, Dtype.string),
            Column('total', Stype.numerical, Dtype.float),
        ],
        source_table=CustomSQLSourceTable(
            connector_id=_CONNECTOR_ID,
            connector_type=None,
            sql=sql,
            table='custom_sql_provisional',
            referenced_tables=['WRONG'],
        ),
        pkey='item_key',
        time_col=None,
    )

    connector = SimpleNamespace(config=SimpleNamespace(type=SimpleNamespace(
        value='snowflake')))
    connector_service = SimpleNamespace(get_for_project=AsyncMock(
        return_value=connector))
    save_mock = AsyncMock()
    table_service = SimpleNamespace(
        # A truthy entry short-circuits the heavy `validate_table` path.
        get_table_entry=AsyncMock(return_value=object()),
        save_table_definition=save_mock,
    )
    table_logic = TableLogic(
        MagicMock(),
        state_services=SimpleNamespace(
            tables=table_service,
            connectors=connector_service,
        ),
        project_id=_PROJECT_ID,
    )

    await table_logic.create_table_v2(table_def)

    persisted = save_mock.await_args.args[1]
    src = persisted.source_table
    # `connector_type` derived from the resolved warehouse connector.
    assert src.connector_type == ConnectorType.SNOWFLAKE
    # `referenced_tables` recomputed server-side (client's 'WRONG' discarded).
    assert src.referenced_tables == referenced_tables_or_none(
        sql, ConnectorType.SNOWFLAKE)
    assert 'WRONG' not in src.referenced_tables
    # Canonical synthetic name derived from the normalized SQL.
    assert src.table == custom_sql_synthetic_name(sql, ConnectorType.SNOWFLAKE)
    # The fpid used for the dedupe lookup / persist is the canonical one, not
    # the provisional name's.
    table_service.get_table_entry.assert_awaited_once_with(
        _PROJECT_ID, persisted.fpid)
    assert persisted.fpid != table_def.fpid
    connector_service.get_for_project.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_table_v2_physical_source_is_not_canonicalized():
    """A physical (non-SQL) table must not trigger any connector-type lookup —
    the Custom SQL guard is a no-op for physical sources and persists the
    definition unchanged."""
    table_def = _s3_named()
    connector_service = SimpleNamespace(get_for_project=AsyncMock(
        side_effect=AssertionError("connector lookup for physical table")))
    save_mock = AsyncMock()
    table_service = SimpleNamespace(
        get_table_entry=AsyncMock(return_value=object()),
        save_table_definition=save_mock,
    )
    table_logic = TableLogic(
        MagicMock(),
        state_services=SimpleNamespace(
            tables=table_service,
            connectors=connector_service,
        ),
        project_id=_PROJECT_ID,
    )

    await table_logic.create_table_v2(table_def)

    connector_service.get_for_project.assert_not_awaited()
    assert save_mock.await_args.args[1] is table_def


@pytest.mark.asyncio
async def test_incomplete_direct_job_stats_report_unavailable_without_parsing(
        tmp_path):
    from kumo_core.types import ColStatType

    from kumo.config import BaseColumnConfig
    from rest.driver.table_logic import try_load_col_stats_from_per_job_dir

    stats_dir = tmp_path / 'outputs'
    stats_dir.mkdir()
    (stats_dir / 'col_stats').write_text('{}')
    score = BaseColumnConfig(
        name='SCORE',
        stype=Stype.numerical,
        dtype=Dtype.float64,
    )
    stats_set = MagicMock()
    stats_set.get_stats_types.return_value = {score: [ColStatType.QUANTILES]}
    table_def = MagicMock()
    table_def.get_source_table_name.return_value = 'predictions'

    with patch('rest.driver.table_logic._get_column_metadata',
               return_value={}), patch(
                   'rest.driver.table_logic._get_sample_rows',
                   new=AsyncMock(side_effect=AssertionError(
                       'direct output has no internal sample'))), patch(
                           'rest.driver.table_logic.'
                           '_build_column_summaries', return_value=[]):
        response = await try_load_col_stats_from_per_job_dir(
            table_def,
            MagicMock(),
            stats_set,
            str(tmp_path),
        )

    assert response is not None
    assert not response.statistics_available
    assert response.columns == []
    assert not response.approx
    assert response.rows_used == 0


@pytest.mark.asyncio
async def test_job_stats_preview_uses_only_selected_prediction_files(
        tmp_path):
    from types import SimpleNamespace

    from rest.driver.table_logic import try_load_col_stats_from_per_job_dir

    stats_dir = tmp_path / 'outputs'
    stats_dir.mkdir()
    (stats_dir / 'col_stats').write_text('{}')
    artifact_root = tmp_path / 'predictions.parquet'
    artifact_root.mkdir()
    selected = (str(artifact_root / 'part-00000.parquet'),
                str(artifact_root / 'part-00001.parquet'))
    for path in selected:
        pathlib.Path(path).write_bytes(b'selected')
    foreign = artifact_root / 'foreign.parquet'
    foreign.write_bytes(b'foreign')

    table_def = MagicMock()
    table_def.get_source_table_name.return_value = 'predictions'
    table_def.cols = ()
    stats_set = MagicMock()
    stats_set.get_stats_types.return_value = {}
    selected_sampler = AsyncMock(return_value={})
    resolved = SimpleNamespace(root=str(artifact_root), files=selected)

    with patch(
            'rest.driver.table_logic.resolve_prediction_artifact',
            new=SimpleNamespace(
                coroutine=AsyncMock(return_value=resolved))), patch(
                'rest.driver.table_logic._get_selected_sample_rows',
                new=selected_sampler), patch(
                    'rest.driver.table_logic._get_sample_rows',
                    new=AsyncMock(side_effect=AssertionError(
                        'selected preview must not use connector root'))), \
            patch('rest.driver.table_logic._get_column_metadata',
                  return_value={}), patch(
                      'rest.driver.table_logic._build_column_summaries',
                      return_value=[]):
        response = await try_load_col_stats_from_per_job_dir(
            table_def,
            MagicMock(),
            stats_set,
            str(tmp_path),
        )

    assert response is not None
    selected_sampler.assert_awaited_once_with(table_def, selected)
    assert str(foreign) not in selected
