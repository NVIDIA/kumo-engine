# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from kumo_core.types import ConnectorType
from kumo_core.types.schema import BaseColumnConfig
from kumoapi import table as table_api
from kumoapi.common import ValidationError as ApiValidationError
from kumoapi.source_table import \
    CustomSQLSourceTable as ApiCustomSQLSourceTable
from kumoapi.typing import Dtype, Stype

from kumo.connector.connector import ConnectorError
from kumo.data.custom_sql import (
    CustomSqlOutputSchemaError,
    custom_sql_synthetic_name,
    referenced_tables_or_none,
)
from kumo.graph.table import SQL_SAMPLE_MAX_SCAN_BYTES
from rest.backends.data_backend.context import DataBackendContext
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.file_type import FileType
from rest.config.table_v2 import Column as RestColumn
from rest.config.table_v2 import (
    CustomSQLSourceTable,
    DataSourceType,
    S3SourceTable,
    TableDBEntryV2,
    TableDefinition,
    TableKind,
    TableResourceResponse,
    derive_table_kind,
    table_args_to_kumo_table,
)
from rest.driver import table_driver as td
from rest.driver import table_logic as tl
from rest.driver.table_driver import TableDriver
from rest.settings import DataScaleLimits
from statedb.config.timestamp import ResourceTimestamps

_CID = 'connector_00000000000000000000000000000000'
_SQL = 'SELECT item_key, total FROM sales'
_PID = 'project_00000000000000000000000000000000'
_REF = ConnectorExecutionRef(connector_id=_CID, project_id=_PID,
                             authorization_token='tok')


def _rest_sql_def(sql: str = _SQL) -> TableDefinition:
    return TableDefinition(
        cols=[
            RestColumn('item_key', Stype.ID, Dtype.string),
            RestColumn('total', Stype.numerical, Dtype.float),
        ],
        source_table=CustomSQLSourceTable(connector_id=_CID,
                                          connector_type='snowflake', sql=sql,
                                          table='custom_sql_abc',
                                          referenced_tables=['sales']),
        pkey='item_key',
        time_col=None,
    )


def _api_sql_def(sql: str = _SQL, *, cols=None) -> table_api.TableDefinition:
    if cols is None:
        cols = [
            table_api.Column(name='item_key', stype=Stype.ID,
                             dtype=Dtype.string),
            table_api.Column(name='total', stype=Stype.numerical,
                             dtype=Dtype.float),
        ]
    return table_api.TableDefinition(
        cols=cols,
        source_table=ApiCustomSQLSourceTable(connector_id=_CID, sql=sql,
                                             table='custom_sql_abc'),
        pkey='item_key' if cols else None,
        time_col=None,
    )


def _sql_request(
    sql: str = _SQL,
    *,
    hints=None,
    pkey='item_key',
    preview_rows: int = 10,
):
    return table_api.SqlTableResolveRequest(
        source_table=ApiCustomSQLSourceTable(connector_id=_CID, sql=sql,
                                             table='pending'),
        column_hints=hints, pkey=pkey, preview_rows=preview_rows)


@pytest.mark.asyncio
async def test_resolve_custom_sql_pipeline_happy_path(monkeypatch):
    expected = table_api.SqlTableResolveResponse(
        table_definition=_api_sql_def(), preview_rows=[['k', '1.0']])
    monkeypatch.setattr(
        tl, 'resolve_connector_and_type_async',
        AsyncMock(return_value=(MagicMock(), ConnectorType.SNOWFLAKE)))
    monkeypatch.setattr(tl, '_validate_and_preview_sql_table',
                        AsyncMock(return_value=expected))

    resp = await tl._resolve_custom_sql(_REF, _sql_request())

    assert resp is expected
    assert resp.preview_rows == [['k', '1.0']]


@pytest.mark.asyncio
async def test_validate_custom_sql_activity_returns_typed_result(monkeypatch):
    expected = table_api.SqlTableResolveResponse(
        table_definition=_api_sql_def(), preview_rows=[['k', '1.0']])
    resolve = AsyncMock(return_value=expected)
    monkeypatch.setattr(tl, '_resolve_custom_sql', resolve)
    request = _sql_request()

    resp = await tl.validate_custom_sql(_REF, request)

    assert resp is expected
    resolve.assert_awaited_once_with(_REF, request, None)


@pytest.mark.asyncio
async def test_resolve_custom_sql_pipeline_invalid_returns_clean_message(
        monkeypatch):
    monkeypatch.setattr(
        tl, 'resolve_connector_and_type_async',
        AsyncMock(return_value=(MagicMock(), ConnectorType.SNOWFLAKE)))
    monkeypatch.setattr(
        tl, '_validate_and_preview_sql_table',
        AsyncMock(side_effect=ValueError('primary key must be an ID')))

    resp = await tl._resolve_custom_sql(_REF, _sql_request())

    assert resp.table_definition is None
    assert resp.errors == [
        ApiValidationError('Invalid Custom SQL table',
                           'primary key must be an ID')
    ]
    tl._validate_and_preview_sql_table.side_effect = RuntimeError(
        'secret provider detail')
    resp = await tl._resolve_custom_sql(_REF, _sql_request())
    assert resp.errors[0].title == 'Custom SQL resolution failed'
    assert 'secret provider detail' not in resp.errors[0].message
    tl._validate_and_preview_sql_table.side_effect = (
        tl.UnsupportedColumnTypeError('cast object_col to STRING'))
    resp = await tl._resolve_custom_sql(_REF, _sql_request())
    assert resp.errors[0].title == 'Unsupported column type'


def test_output_contract_error_is_invalid_schema():
    error = tl._sql_resolve_error(
        CustomSqlOutputSchemaError('computed output requires an alias'))
    assert error.title == 'Invalid output schema'


@pytest.mark.asyncio
async def test_snapshot_table_logic_uses_typed_sql_resolution(monkeypatch):
    logic = tl.TableLogic(AsyncMock(spec=DataBackendContext),
                          state_services=SimpleNamespace(), project_id=_PID)
    monkeypatch.setattr(logic, '_execution_connector',
                        AsyncMock(return_value=_REF))
    monkeypatch.setattr(tl, '_sql_reference_stype_hints',
                        AsyncMock(return_value=None))
    dispatch = AsyncMock(return_value=table_api.SqlTableResolveResponse(
        table_definition=_api_sql_def()))
    monkeypatch.setattr(tl.DataBackend, '_execute_single_activity', dispatch)

    await logic.validate_table(_rest_sql_def())

    request = dispatch.await_args.args[2][1]
    assert isinstance(request, table_api.SqlTableResolveRequest)
    assert request.column_hints


@pytest.mark.asyncio
async def test_resolver_rejects_implicit_output_before_describe():
    connector = _fake_sql_connector()
    request = table_api.TableMetadataRequest(
        source_table=_sql_request(sql='SELECT 1 + 2').source_table, cols=[])
    with pytest.raises(CustomSqlOutputSchemaError):
        await tl._resolve_table_metadata_with_connector(request, connector,
                                                        ConnectorType.BIGQUERY)
    assert connector.get_query_metadata_calls == []


@pytest.mark.asyncio
async def test_resolve_custom_sql_pipeline_forwards_preview_skipped_note(
        monkeypatch):
    monkeypatch.setattr(
        tl, 'resolve_connector_and_type_async',
        AsyncMock(return_value=(MagicMock(), ConnectorType.SNOWFLAKE)))
    monkeypatch.setattr(
        tl, '_validate_and_preview_sql_table',
        AsyncMock(return_value=table_api.SqlTableResolveResponse(
            table_definition=_api_sql_def(), preview_note=(
                'Preview skipped — query too large.'))))

    resp = await tl._resolve_custom_sql(_REF, _sql_request())

    assert resp.table_definition is not None
    assert resp.preview_rows == []
    assert resp.preview_note == 'Preview skipped — query too large.'


@pytest.mark.asyncio
async def test_resolve_custom_sql_pipeline_unsupported_dialect_no_escape(
        monkeypatch):
    """A connector whose type is not a supported SQL dialect (e.g. S3) returns a
    clean validation errors; the error-path `referenced_tables` extraction (which
    would also raise for that dialect) must not escape as a server error."""
    monkeypatch.setattr(tl, 'resolve_connector_and_type_async',
                        AsyncMock(return_value=(MagicMock(), ConnectorType.S3)))
    monkeypatch.setattr(
        tl, '_validate_and_preview_sql_table',
        AsyncMock(side_effect=ValueError(
            "Custom SQL Tables are not supported for 's3' connectors.")))

    resp = await tl._resolve_custom_sql(_REF, _sql_request())

    assert resp.table_definition is None
    assert 'not supported' in resp.errors[0].message


@pytest.mark.asyncio
async def test_resolve_custom_sql_pipeline_surfaces_warehouse_reason(
        monkeypatch):
    """A post-open query error keeps its safe warehouse reason."""
    monkeypatch.setattr(
        tl, 'resolve_connector_and_type_async',
        AsyncMock(return_value=(MagicMock(), ConnectorType.SNOWFLAKE)))
    monkeypatch.setattr(
        tl, '_validate_and_preview_sql_table',
        AsyncMock(side_effect=ConnectorError(
            'Snowflake failed to describe the custom SQL query')))

    resp = await tl._resolve_custom_sql(_REF, _sql_request())

    assert resp.table_definition is None
    message = resp.errors[0].message
    assert message.startswith('The warehouse could not run this query.')
    assert 'Snowflake failed to describe the custom SQL query' in message
    assert 'source access' not in message.lower()


@pytest.mark.asyncio
async def test_resolve_custom_sql_returns_typed_open_failure(monkeypatch):
    from contextlib import asynccontextmanager
    monkeypatch.setattr(
        tl, 'resolve_connector_and_type_async',
        AsyncMock(return_value=(MagicMock(), ConnectorType.SNOWFLAKE)))

    @asynccontextmanager
    async def failed_open(*_args, **_kwargs):
        raise ConnectorError('authentication failed')
        yield

    monkeypatch.setattr(tl, 'async_connector_lifecycle', failed_open)
    response = await tl._resolve_custom_sql(_REF, _sql_request())
    assert response.table_definition is None
    assert response.errors[0].title == 'Warehouse query failed'
    assert 'authentication failed' in response.errors[0].message


@pytest.mark.asyncio
async def test_resolve_custom_sql_derives_type_from_resolved_connector(
        monkeypatch):
    """Derive the warehouse type from the broker-resolved connector."""
    from contextlib import asynccontextmanager

    from kumo.connector import DatabricksConnector
    from rest.backends.data_backend import connector_broker
    from rest.config.data_backend import ResolvedConnectorSpec

    connector = DatabricksConnector(
        name='wh', host='https://workspace.cloud.databricks.com', cluster_id='',
        warehouse_id='sql-warehouse', catalog='main', schema_name='sales')
    # Mirror production: the broker hands back a spec whose connector_config is
    # the connector's NATIVE config (``.config()``), which carries no ``type``.
    resolved = ResolvedConnectorSpec(connector_id=_CID,
                                     connector_config=connector.config(),
                                     credentials=None)
    monkeypatch.setattr(connector_broker.connector_credential_broker,
                        'resolve_async', AsyncMock(return_value=resolved))

    # No-op lifecycle so no real warehouse connection is opened.
    @asynccontextmanager
    async def _lifecycle(conn, **_kwargs):
        yield conn

    monkeypatch.setattr(tl, 'async_connector_lifecycle', _lifecycle)

    captured = {}

    async def _fake_validate(request, conn, connector_type, **_kwargs):
        captured['connector'] = conn
        captured['connector_type'] = connector_type
        return table_api.SqlTableResolveResponse(
            table_definition=_api_sql_def())

    monkeypatch.setattr(tl, '_validate_and_preview_sql_table', _fake_validate)

    resp = await tl._resolve_custom_sql(_REF, _sql_request())

    assert resp.table_definition is not None
    # The real derivation ran: a Databricks native config resolves to DATABRICKS
    # and the connector object handed to the pipeline is the resolved instance.
    assert captured['connector_type'] == ConnectorType.DATABRICKS
    assert isinstance(captured['connector'], DatabricksConnector)


def _fake_sql_connector(scan_estimate_bytes=None):
    from test.graph.test_custom_sql_infer import _FakeSQLConnector

    query_cols = [
        BaseColumnConfig(name='item_key', dtype=Dtype.string),
        BaseColumnConfig(name='total', dtype=Dtype.float32),
    ]
    sample = pd.DataFrame({
        'item_key': [f'k{i}' for i in range(50)],
        'total': [float(i) for i in range(50)],
    })
    # Only a warehouse that gates is asked for an estimate.
    connector = _FakeSQLConnector(
        'wh', query_cols, sample, scan_estimate_bytes=scan_estimate_bytes,
        gates_query_scan=scan_estimate_bytes is not None)
    connector.database = 'ANALYTICS'
    connector.schema_name = 'PUBLIC'
    return connector


def test_sql_preview_rows_affordable_returns_bounded_aligned_rows():
    connector = _fake_sql_connector(scan_estimate_bytes=None)
    table = table_args_to_kumo_table(_rest_sql_def(), connector)
    columns = _rest_sql_def().cols

    rows = tl._sql_preview_rows(table._read_inference_sample(), columns, 3)

    assert rows is not None
    assert len(rows) == 3
    # Rows are aligned to ``columns`` order (item_key, total) and stringified.
    assert rows[0] == ['k0', '0.0']
    assert connector.sample_query_calls == [_SQL]


def test_sql_preview_rows_supports_arbitrary_binary_and_extreme_dates():
    sample = pd.DataFrame({
        'payload': [b'\xff\x00', None],
        'event_date': [date(1, 1, 1), date(9999, 12, 31)],
    })
    columns = [
        RestColumn('payload', Stype.categorical, Dtype.binary),
        RestColumn('event_date', Stype.categorical, Dtype.date),
    ]

    rows = tl._sql_preview_rows(sample, columns, 2)

    assert rows == [
        ['0xff00', '0001-01-01'],
        [None, '9999-12-31'],
    ]


def test_sql_preview_ignores_out_of_bounds_stats_for_extreme_dates(monkeypatch):
    connector = _fake_sql_connector()
    table = table_args_to_kumo_table(_rest_sql_def(), connector)
    monkeypatch.setattr(
        table,
        '_compute_stats_via_connector_sample',
        MagicMock(side_effect=pd.errors.OutOfBoundsDatetime('extreme date')),
    )
    request = {BaseColumnConfig(name='total', dtype=Dtype.float32): []}

    table.persist_sql_preview_col_stats('definition-version', request)


def test_sql_preview_rows_cost_gated_returns_none():
    connector = _fake_sql_connector(
        scan_estimate_bytes=SQL_SAMPLE_MAX_SCAN_BYTES + 1)
    table = table_args_to_kumo_table(_rest_sql_def(), connector)
    columns = _rest_sql_def().cols

    assert tl._sql_preview_rows(table._read_inference_sample(), columns, 10) \
        is None
    # The cost gate blocks the scan entirely.
    assert connector.sample_query_calls == []


# --- TableDriver SQL resolution dispatch -----------------------------------


def _make_driver() -> TableDriver:
    driver = TableDriver(
        data_backend_ctx=AsyncMock(spec=DataBackendContext),
        data_scale_limits=DataScaleLimits.parse_from_env(),
    )
    config = SimpleNamespace(
        type=SimpleNamespace(value=ConnectorType.SNOWFLAKE.value),
        database='ANALYTICS',
        schema_name='PUBLIC',
    )
    driver._configure_state_services(
        SimpleNamespace(connectors=SimpleNamespace(get_for_project=AsyncMock(
            return_value=SimpleNamespace(config=config)))))
    return driver


@pytest.mark.parametrize('surface',
                         ['resolve', 'create', 'persist', 'refresh', 'infer'])
@pytest.mark.asyncio
async def test_sql_write_is_rejected_at_every_predispatch_surface(
        monkeypatch, surface):
    """Every authoring and persisted-definition entry point applies the same
    guard before an older worker or warehouse can execute the query."""
    driver = _make_driver()
    ref = AsyncMock()
    dispatch = AsyncMock()
    save = AsyncMock()
    monkeypatch.setattr(driver, '_execution_ref', ref)
    monkeypatch.setattr(td.DataBackend, '_execute_single_activity', dispatch)
    driver._require_state_services().tables = SimpleNamespace(
        get_table_entry=AsyncMock(return_value=SimpleNamespace()),
        save_table_definition=save)
    sql = 'DELETE FROM sales'
    definition = _rest_sql_def(sql=sql)

    if surface == 'resolve':
        response = await driver.resolve_sql(_sql_request(sql=sql),
                                            project_id='default')
        assert 'read-only' in response.errors[0].message
    else:
        with pytest.raises(ValueError, match='read-only'):
            if surface == 'create':
                await driver._validate_new_table_for_state_service(
                    definition, project_id='default')
            elif surface == 'persist':
                await driver._persist_table_definition(definition,
                                                       alias='unsafe',
                                                       force_rename=False,
                                                       project_id='default')
            elif surface == 'refresh':
                await driver._trigger_async_refresh_for_table_entry(
                    SimpleNamespace(config=definition, alias='unsafe'),
                    project_id='default')
            else:
                await driver.infer_table_metadata(
                    table_api.TableMetadataRequest(
                        source_table=_sql_request(sql=sql).source_table,
                        cols=[]), 'default')

    ref.assert_not_awaited()
    dispatch.assert_not_awaited()
    save.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_sql_dispatches_activity_on_valid(monkeypatch):
    driver = _make_driver()
    ref = AsyncMock(return_value=_REF)
    resolved = _api_sql_def()
    dispatch = AsyncMock(return_value=table_api.SqlTableResolveResponse(
        table_definition=resolved, preview_rows=[['k', '1.0']]))
    monkeypatch.setattr(driver, '_execution_ref', ref)
    monkeypatch.setattr(td.DataBackend, '_execute_single_activity', dispatch)

    request = _sql_request()
    resp = await driver.resolve_sql(request, project_id='default')

    assert resp.table_definition is not None
    assert [col.name
            for col in resp.table_definition.cols] == ['item_key', 'total']
    assert resp.preview_rows == [['k', '1.0']]
    ref.assert_awaited_once()
    dispatch.assert_awaited_once()
    ctx, activity_fn, inputs, _prefix = dispatch.await_args.args
    assert activity_fn is td.validate_custom_sql
    assert inputs[0] is _REF
    assert inputs[1] is request


@pytest.mark.asyncio
async def test_create_path_validates_sql_via_activity_not_physical(monkeypatch):
    """The create-time validator routes a Custom SQL table to
    validate_custom_sql (never validate_table_definition_activity, which would
    probe the synthetic alias as a physical table)."""
    driver = _make_driver()
    dispatch = AsyncMock(return_value=table_api.SqlTableResolveResponse(
        table_definition=_api_sql_def()))
    monkeypatch.setattr(driver, '_execution_ref', AsyncMock(return_value=_REF))
    monkeypatch.setattr(td.DataBackend, '_execute_single_activity', dispatch)

    await driver._validate_new_table_for_state_service(_rest_sql_def(),
                                                       project_id='default')

    dispatch.assert_awaited_once()
    _ctx, activity_fn, inputs, _prefix = dispatch.await_args.args
    assert activity_fn is td.validate_custom_sql
    # 3rd arg is the Custom SQL stype-inheritance hints; None here since the
    # bare test state-services expose no sibling tables to inherit from.
    assert inputs[0] is _REF
    assert isinstance(inputs[1], table_api.SqlTableResolveRequest)
    assert inputs[1].column_hints


@pytest.mark.asyncio
async def test_create_path_raises_on_invalid_sql(monkeypatch):
    from kumo.validation import ValidationException

    driver = _make_driver()
    dispatch = AsyncMock(return_value=table_api.SqlTableResolveResponse(
        errors=[ApiValidationError('Invalid Custom SQL table', 'bad query')]))
    monkeypatch.setattr(driver, '_execution_ref', AsyncMock(return_value=_REF))
    monkeypatch.setattr(td.DataBackend, '_execute_single_activity', dispatch)

    with pytest.raises(ValidationException):
        await driver._validate_new_table_for_state_service(
            _rest_sql_def(), project_id='default')


@pytest.mark.asyncio
async def test_resolve_sql_unknown_dependencies_fail_before_activity_dispatch(
        monkeypatch):
    driver = _make_driver()
    dispatch = AsyncMock(return_value=table_api.SqlTableResolveResponse(
        table_definition=_api_sql_def()))
    monkeypatch.setattr(driver, '_execution_ref', AsyncMock(return_value=_REF))
    monkeypatch.setattr(td.DataBackend, '_execute_single_activity', dispatch)

    resp = await driver.resolve_sql(_sql_request(sql='not parseable ((('),
                                    project_id='default')

    assert 'cannot be safely verified' in resp.errors[0].message
    dispatch.assert_not_awaited()


# --- stype inheritance threaded through the live infer dispatch ------------


@pytest.mark.asyncio
async def test_infer_table_metadata_inherits_referenced_kumo_stypes(
        monkeypatch):
    from contextlib import asynccontextmanager
    from test.graph.test_custom_sql_infer import (
        _INHERIT_SQL,
        _inherit_connector,
    )

    connector = _inherit_connector()

    orders_cfg = SimpleNamespace(
        is_sql=False,
        get_source_table_name=lambda: 'orders',
        cols=[
            SimpleNamespace(name='user_id', stype=Stype.categorical),
            SimpleNamespace(name='amount', stype=Stype.categorical),
        ],
    )
    state_services = SimpleNamespace(
        tables=SimpleNamespace(list_tables=AsyncMock(
            return_value=[SimpleNamespace(config=orders_cfg)])),
        connectors=SimpleNamespace(get_for_project=AsyncMock(
            return_value=SimpleNamespace(config=SimpleNamespace(
                type=SimpleNamespace(value=ConnectorType.SNOWFLAKE.value),
                database='ANALYTICS', schema_name='PUBLIC')))),
    )
    driver = _make_driver()
    driver._configure_state_services(state_services)
    monkeypatch.setattr(driver, '_execution_ref', AsyncMock(return_value=_REF))

    async def _run_activity(_ctx, activity_fn, inputs, _prefix):
        return await activity_fn(*inputs)

    monkeypatch.setattr(td.DataBackend, '_execute_single_activity',
                        _run_activity)
    monkeypatch.setattr(tl, 'get_connector_from_spec_async',
                        AsyncMock(return_value=connector))

    @asynccontextmanager
    async def _lifecycle(conn, **_kwargs):
        yield conn

    monkeypatch.setattr(tl, 'async_connector_lifecycle', _lifecycle)

    request = table_api.TableMetadataRequest(
        source_table=ApiCustomSQLSourceTable(connector_id=_CID,
                                             connector_type='snowflake',
                                             sql=_INHERIT_SQL,
                                             table='custom_sql_inherit',
                                             referenced_tables=['orders']),
        cols=[],
        pkey='user_id',
        time_col=None,
    )

    resp = await driver.infer_table_metadata(request, _PID)

    stypes = {col.name: col.stype for col in resp.cols}
    assert stypes['user_id'] == Stype.ID
    assert stypes['spend'] == Stype.categorical
    assert stypes['total'] == Stype.numerical
    state_services.tables.list_tables.assert_awaited_once()


@pytest.mark.asyncio
async def test_infer_table_metadata_physical_preserves_two_arg_arity(
        monkeypatch):
    from kumoapi.source_table import SnowflakeSourceTable

    driver = _make_driver()
    monkeypatch.setattr(driver, '_execution_ref', AsyncMock(return_value=_REF))
    dispatch = AsyncMock(return_value=SimpleNamespace())
    monkeypatch.setattr(td.DataBackend, '_execute_single_activity', dispatch)

    physical_request = table_api.TableMetadataRequest(
        source_table=SnowflakeSourceTable(snowflake_connector_id=_CID,
                                          database='DB', schema_name='S',
                                          table='orders'),
        cols=[table_api.ColumnMetadataRequest(name='id')],
        pkey=None,
        time_col=None,
    )
    await driver.infer_table_metadata(physical_request, _PID)
    _ctx, activity_fn, inputs, _prefix = dispatch.await_args.args
    assert activity_fn is td.infer_table_metadata_activity
    assert inputs == [_REF, physical_request]

    monkeypatch.setattr(td, '_sql_reference_stype_hints',
                        AsyncMock(return_value=None))
    monkeypatch.setattr(driver, '_sql_dialect',
                        AsyncMock(return_value=ConnectorType.SNOWFLAKE))
    sql_request = table_api.TableMetadataRequest(
        source_table=ApiCustomSQLSourceTable(connector_id=_CID,
                                             connector_type='snowflake',
                                             sql=_SQL, table='custom_sql_abc',
                                             referenced_tables=['sales']),
        cols=[table_api.ColumnMetadataRequest(name='item_key')],
        pkey=None,
        time_col=None,
    )
    await driver.infer_table_metadata(sql_request, _PID)
    _ctx, activity_fn, inputs, _prefix = dispatch.await_args.args
    assert activity_fn is td.infer_table_metadata_activity
    assert len(inputs) == 3
    assert inputs[:2] == [_REF, sql_request]


@pytest.mark.asyncio
async def test_infer_table_metadata_rejects_non_warehouse_connector(
        monkeypatch):
    driver = _make_driver()
    ref = AsyncMock(return_value=_REF)
    monkeypatch.setattr(driver, '_execution_ref', ref)
    monkeypatch.setattr(
        driver, '_validate_custom_sql_request',
        AsyncMock(side_effect=ValueError(
            "Custom SQL Tables are not supported for 's3' connectors.")))
    dispatch = AsyncMock(return_value=SimpleNamespace())
    monkeypatch.setattr(td.DataBackend, '_execute_single_activity', dispatch)

    sql_request = table_api.TableMetadataRequest(
        source_table=ApiCustomSQLSourceTable(connector_id=_CID, sql=_SQL,
                                             table='custom_sql_abc'),
        cols=[table_api.ColumnMetadataRequest(name='item_key')],
        pkey=None,
        time_col=None,
    )
    with pytest.raises(ValueError, match="not supported for"):
        await driver.infer_table_metadata(sql_request, _PID)
    ref.assert_not_awaited()
    dispatch.assert_not_awaited()


# --- server-side synthetic-name canonicalization (Phase-1 regression) ------

# A whitespace / comment / keyword-case-only edit of ``_SQL``: normalizes to the
# same query, so it must resolve to the same synthetic name and fingerprint.
_SQL_FMT = 'select   item_key,\n       total\nFROM   sales  -- trailing note\n'
# A semantic edit (extra predicate): a new definition, a new identity.
_SQL_SEMANTIC = 'SELECT item_key, total FROM sales WHERE total > 0'


@pytest.mark.asyncio
async def test_canonicalize_custom_sql_name_stable_under_formatting_edits(
        monkeypatch):
    """Create-side canonicalization derives the synthetic ``table`` name from
    the normalized SQL, so a formatting-only edit keeps the same fingerprint
    (the create dedupes onto the existing entry and reuses its stats) while a
    semantic edit yields a new one. ``referenced_tables`` is recomputed
    server-side rather than trusting the client-supplied list."""
    driver = _make_driver()
    monkeypatch.setattr(driver, '_sql_dialect',
                        AsyncMock(return_value=ConnectorType.SNOWFLAKE))

    canon_base = await driver._canonicalize_custom_sql_name(
        _rest_sql_def(_SQL), 'p')
    canon_fmt = await driver._canonicalize_custom_sql_name(
        _rest_sql_def(_SQL_FMT), 'p')
    canon_semantic = await driver._canonicalize_custom_sql_name(
        _rest_sql_def(_SQL_SEMANTIC), 'p')

    expected = custom_sql_synthetic_name(_SQL, ConnectorType.SNOWFLAKE)
    assert canon_base.source_table.table == expected
    # Formatting-only edit → identical synthetic name AND fingerprint.
    assert canon_fmt.source_table.table == canon_base.source_table.table
    assert canon_fmt.fingerprint() == canon_base.fingerprint()
    # Semantic edit → new identity.
    assert canon_semantic.source_table.table != canon_base.source_table.table
    assert canon_semantic.fingerprint() != canon_base.fingerprint()

    # A bogus client-supplied referenced-tables list is overwritten with the
    # server-parsed dependency set.
    bogus = dataclasses.replace(
        _rest_sql_def(_SQL), source_table=dataclasses.replace(
            _rest_sql_def(_SQL).source_table, referenced_tables=['WRONG']))
    canon_bogus = await driver._canonicalize_custom_sql_name(bogus, 'p')
    assert (
        canon_bogus.source_table.referenced_tables == referenced_tables_or_none(
            _SQL, ConnectorType.SNOWFLAKE))


@pytest.mark.asyncio
async def test_canonicalize_custom_sql_name_physical_table_is_noop(monkeypatch):
    """A physical (non-SQL) table is returned unchanged without any dialect
    lookup — canonicalization only touches Custom SQL tables."""
    driver = _make_driver()
    dialect = AsyncMock()
    monkeypatch.setattr(driver, '_sql_dialect', dialect)

    phys_def = TableDefinition(
        cols=_rest_sql_def().cols,
        source_table=S3SourceTable(
            connector_id=_CID,
            source={
                'mode': 'relative',
                'path': 'raw'
            },
            file_type=FileType.PARQUET,  # pyright: ignore
            data_source_type=DataSourceType.S3),
        pkey='item_key',
        time_col=None)

    assert await driver._canonicalize_custom_sql_name(phys_def, 'p') is phys_def
    dialect.assert_not_awaited()


@pytest.mark.asyncio
async def test_canonicalize_custom_sql_name_fails_loud_when_undialectable(
        monkeypatch):
    """A SQL table whose warehouse dialect can't be resolved fails creation
    loudly rather than silently persisting the SDK's provisional (non-canonical)
    name — the authoritative ``_sql_dialect`` error propagates."""
    driver = _make_driver()
    monkeypatch.setattr(
        driver, '_sql_dialect',
        AsyncMock(side_effect=RuntimeError('connector unavailable')))

    with pytest.raises(RuntimeError):
        await driver._canonicalize_custom_sql_name(_rest_sql_def(_SQL), 'p')


@pytest.mark.asyncio
async def test_resolver_discovers_all_with_role_in_one_describe():
    connector = _fake_sql_connector()
    request = table_api.TableMetadataRequest(
        source_table=_sql_request().source_table, cols=[], pkey='item_key')
    response, described, _ = await tl._resolve_table_metadata_with_connector(
        request, connector, ConnectorType.SNOWFLAKE)
    assert [col.name for col in response.cols] == ['item_key', 'total']
    assert described is not None
    assert connector.get_query_metadata_calls == [_SQL]


@pytest.mark.asyncio
async def test_resolver_reuses_one_sample_for_inference_stats_and_preview():
    connector = _fake_sql_connector()

    response = await tl._validate_and_preview_sql_table(
        _sql_request(preview_rows=3),
        connector,
        ConnectorType.SNOWFLAKE,
    )

    assert response.table_definition is not None
    assert len(response.preview_rows) == 3
    assert connector.sample_query_calls == [_SQL]


@pytest.mark.asyncio
async def test_resolver_preserves_hint_then_validates_described_dtype():
    connector = _fake_sql_connector()
    hint = table_api.ColumnMetadataRequest('item_key', Stype.numerical,
                                           Dtype.int)
    request = table_api.TableMetadataRequest(
        source_table=_sql_request().source_table, cols=[hint])
    response, described, _ = await tl._resolve_table_metadata_with_connector(
        request, connector, ConnectorType.SNOWFLAKE, discover_sql_columns=False)
    definition = TableDefinition(cols=response.cols,
                                 source_table=_rest_sql_def().source_table)
    from kumo.validation import ValidationException
    with pytest.raises(ValidationException, match='should be String'):
        await tl._validate_table_metadata_v2.coroutine(definition, connector,
                                                       described)
    assert connector.get_query_metadata_calls == [_SQL]


@pytest.mark.asyncio
async def test_resolver_revalidates_native_list_hint_without_scalarizing():
    from test.graph.test_custom_sql_infer import _FakeSQLConnector

    sql = 'SELECT item_key, tags FROM sales'
    connector = _FakeSQLConnector(
        'wh',
        [
            BaseColumnConfig(name='item_key', dtype=Dtype.string),
            BaseColumnConfig(name='tags', dtype=Dtype.stringlist),
        ],
        pd.DataFrame({
            'item_key': ['a', 'b'],
            'tags': [['red'], ['blue', 'green']],
        }),
    )
    connector.database = 'ANALYTICS'
    connector.schema_name = 'PUBLIC'
    hints = [
        table_api.ColumnMetadataRequest(
            'item_key',
            Stype.ID,
            Dtype.string,
        ),
        table_api.ColumnMetadataRequest(
            'tags',
            Stype.multicategorical,
            Dtype.stringlist,
        ),
    ]

    response = await tl._validate_and_preview_sql_table(
        _sql_request(sql=sql, hints=hints),
        connector,
        ConnectorType.SNOWFLAKE,
    )

    tags = next(col for col in response.table_definition.cols
                if col.name == 'tags')
    assert tags.dtype == Dtype.stringlist
    assert tags.stype == Stype.multicategorical
    assert tags.multicategorical_delimiter is None


# --- table_kind stamping on TableResourceResponse -------------------------


def _rest_raw_def() -> TableDefinition:
    """A physical (non-SQL) table def, for the RAW side of the classification."""
    return TableDefinition(
        cols=[
            RestColumn('item_key', Stype.ID, Dtype.string),
            RestColumn('total', Stype.numerical, Dtype.float),
        ],
        source_table=S3SourceTable(
            source={
                'mode': 'relative',
                'path': 'sales.csv'
            },
            connector_id=_CID,
            file_type=FileType.CSV,
        ),
        pkey='item_key',
        time_col=None,
    )


def _entry(table_def: TableDefinition) -> TableDBEntryV2:
    return TableDBEntryV2(
        timestamps=ResourceTimestamps(created_at=datetime.now(),
                                      updated_at=datetime.now()),
        id=table_def.fpid,
        project_id=_PID,
        config=table_def,
        col_dtypes={col.name: col.dtype
                    for col in table_def.cols},
        alias=None,
    )


def test_table_kind_wire_values_match_ui_enum():
    # Locks the on-the-wire strings to the UI ``TableKind`` enum
    # (ui/src/types/tables.ts); a drift here silently misclassifies every table.
    assert TableKind.RAW.value == 'raw'
    assert TableKind.SQL.value == 'sql'


def test_derive_table_kind_across_rest_and_api_source_tables():
    # Robust across the REST-mirror and kumoapi CustomSQLSourceTable, and
    # against a raw physical source, using the string discriminator.
    assert derive_table_kind(_rest_sql_def()) == TableKind.SQL
    assert derive_table_kind(_api_sql_def()) == TableKind.SQL
    assert derive_table_kind(_rest_raw_def()) == TableKind.RAW


def test_table_resource_response_defaults_to_raw():
    # A response built without an explicit kind stays RAW (backward compatible).
    resp = TableResourceResponse(
        id=_rest_raw_def().fpid,
        table=_rest_raw_def(),
        name_alias=None,
        timestamps=ResourceTimestamps(created_at=datetime.now(),
                                      updated_at=datetime.now()),
    )
    assert resp.table_kind == TableKind.RAW


@pytest.mark.parametrize(('builder_type', 'method', 'module'), [
    (TableDriver, '_table_resource_responses_from_entries', td),
    (tl.TableLogic, '_get_many_table_response_v2', tl),
], ids=['single-get', 'list'])
@pytest.mark.asyncio
async def test_table_response_builders_stamp_table_kind(monkeypatch,
                                                        builder_type, method,
                                                        module):
    """Both the detail and list builders classify SQL tables on the wire."""
    fake_cache = SimpleNamespace(
        get_many_min_latest_exact_stats=SimpleNamespace(coroutine=AsyncMock(
            return_value=[None, None])))
    monkeypatch.setattr(module, 'get_active_cache', lambda: fake_cache)

    builder = builder_type.__new__(builder_type)
    responses = await getattr(
        builder, method)([_entry(_rest_sql_def()),
                          _entry(_rest_raw_def())])

    assert [r.table_kind for r in responses] == [TableKind.SQL, TableKind.RAW]


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
