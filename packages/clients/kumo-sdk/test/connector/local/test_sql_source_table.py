# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from datetime import datetime, timezone
from unittest import mock

import kumoapi.table as api
import pytest
import requests
from kumoapi.common import ValidationError
from kumoapi.data_source import (
    ConnectorResponse,
    ConnectorType,
    ResourceTimestamps,
    S3AnyURIConnectorTarget,
    S3ConnectorResourceConfig,
    SnowflakeConnectorResourceConfig,
)
from kumoapi.source_table import CustomSQLSourceTable
from kumoapi.typing import Dtype, Stype

from kumoai_studio import global_state
from kumoai_studio.client.table import TableAPI
from kumoai_studio.connector import (
    S3Connector,
    SnowflakeConnector,
    SQLSourceTable,
)
from kumoai_studio.graph import Table

SQL = "SELECT store_id, price * qty AS total FROM CST_SALES"

_TIMESTAMPS = ResourceTimestamps(
    created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
)


def _snowflake() -> SnowflakeConnector:
    # Bind the connector from a persisted response so construction is offline
    # (no ``create_if_not_exist`` round-trip).
    config = SnowflakeConnectorResourceConfig(warehouse='wh', database='db',
                                              schema_name='sch', role='r')
    response = ConnectorResponse(
        id='sf-connector-id',
        project_id='project-id',
        identity_id='identity-snowflake',
        name='sf',
        timestamps=_TIMESTAMPS,
        config=config,
    )
    return SnowflakeConnector(name='sf', warehouse='wh', database='db',
                              schema_name='sch', _response=response)


def _s3() -> S3Connector:
    config = S3ConnectorResourceConfig(target=S3AnyURIConnectorTarget())
    response = ConnectorResponse(
        id='s3-connector-id',
        project_id='project-id',
        identity_id='identity-s3',
        name='s3',
        timestamps=_TIMESTAMPS,
        config=config,
    )
    return S3Connector(name='s3', _response=response)


def _backend_of(state):
    """`SourceTable._from_api_table_definition` resolves a connector through
    the active backend rather than the REST client, so a definition can be
    rebuilt where there is no service; on the cloud backend the two are the
    same object.
    """
    backend = mock.MagicMock()
    backend.connector = state.client.connector_api
    return backend


def test_synthetic_name_stable_and_valid() -> None:
    name = _snowflake().sql(SQL).name
    # Deterministic in the query text:
    assert _snowflake().sql(SQL).name == name
    # Satisfies the Kumo `SourceTableName` charset (<= 256 chars):
    assert re.fullmatch(r'[a-zA-Z0-9_-]{1,256}', name)
    # A different query yields a different name:
    assert _snowflake().sql(SQL + ' ').name != name


def test_connector_sql_rejects_file_connector() -> None:
    with pytest.raises(ValueError, match='warehouse connectors'):
        _s3().sql(SQL)


def test_sql_source_table_head_uses_bounded_warehouse_preview() -> None:
    client = mock.MagicMock()
    client.table_api.resolve_sql.return_value = api.SqlTableResolveResponse(
        table_definition=api.TableDefinition(
            cols=[
                api.Column(
                    name='ROW_COUNT',
                    stype=Stype.numerical,
                    dtype=Dtype.int64,
                )
            ],
            source_table=CustomSQLSourceTable(
                connector_id='sf-connector-id',
                sql=SQL,
                table='custom_sql_preview',
            ),
        ),
        preview_rows=[['10000']],
    )
    global_state._url, global_state._api_key = 'http://kumo.ai', 'DISABLED'
    global_state.thread_local._client = client
    try:
        resolve_sql = client.table_api.resolve_sql
        source = _snowflake().sql(SQL)
        result = source.head(1)
        resolve_sql.return_value = api.SqlTableResolveResponse(
            table_definition=api.TableDefinition(
                cols=[
                    api.Column(
                        name='EARLIEST',
                        stype=Stype.timestamp,
                        dtype=Dtype.date,
                    ),
                    api.Column(
                        name='LATEST',
                        stype=Stype.timestamp,
                        dtype=Dtype.time,
                    ),
                ],
                source_table=CustomSQLSourceTable(
                    connector_id='sf-connector-id',
                    sql=SQL,
                    table='custom_sql_preview',
                ),
            ),
            preview_rows=[['0001-01-01', '9999-12-31 23:59:59']],
        )
        extreme_dates = source.head(1)
        resolve_sql.return_value = api.SqlTableResolveResponse(
            table_definition=api.TableDefinition(
                cols=[
                    api.Column(
                        name='ROW_COUNT',
                        stype=Stype.numerical,
                        dtype=Dtype.int64,
                    )
                ],
                source_table=CustomSQLSourceTable(
                    connector_id='sf-connector-id',
                    sql=SQL,
                    table='custom_sql_preview',
                ),
            ),
            preview_rows=[],
            preview_note='Preview skipped — query too large.',
        )
        with pytest.warns(UserWarning, match='query too large'):
            skipped_preview = source.head(1)
        client.table_api.resolve_sql.return_value = (
            api.SqlTableResolveResponse(
                errors=[ValidationError('Invalid SQL', 'bad query')]))
        with pytest.raises(ValueError, match='bad query'):
            source.head(1)
        with pytest.raises(ValueError, match='greater than 0'):
            source.head(0)
    finally:
        global_state.clear()

    assert result.to_dict(orient='records') == [{'ROW_COUNT': 10_000}]
    assert extreme_dates.to_dict(orient='records') == [{
        'EARLIEST':
        '0001-01-01',
        'LATEST':
        '9999-12-31 23:59:59',
    }]
    assert skipped_preview.empty
    request = client.table_api.resolve_sql.call_args_list[0].args[0]
    assert request.source_table.sql == SQL
    assert request.preview_rows == 1
    assert client.table_api.resolve_sql.call_args_list[0].kwargs == {
        'project_id': 'project-id',
    }


def _previewed(columns, rows):
    r"""The rows are assigned rather than passed in, because the response
    spells a preview cell as text and a warehouse that types its booleans does
    not: the point of the coercion under test is that it survives both.
    """
    response = api.SqlTableResolveResponse(
        table_definition=api.TableDefinition(
            cols=columns,
            source_table=CustomSQLSourceTable(connector_id='sf-connector-id',
                                              sql=SQL,
                                              table='custom_sql_preview'),
        ), )
    response.preview_rows = rows
    client = mock.MagicMock()
    client.table_api.resolve_sql.return_value = response
    global_state._url, global_state._api_key = 'http://kumo.ai', 'DISABLED'
    global_state.thread_local._client = client
    try:
        return _snowflake().sql(SQL).head(len(rows) or 1)
    finally:
        global_state.clear()


def test_a_preview_boolean_survives_whether_it_is_native_or_spelled() -> None:
    r"""A preview crosses as JSON, where a boolean is already one.

    Reading every value as text made a valid ``true`` from a warehouse that
    types its booleans raise on the way to a preview frame.
    """
    columns = [
        api.Column(name='ACTIVE', stype=Stype.categorical, dtype=Dtype.bool)
    ]

    frame = _previewed(columns, [[True], [False], ['TRUE'], ['false'], [None]])

    assert list(frame['ACTIVE']) == [True, False, True, False, None]


def test_a_zoned_preview_timestamp_is_bounded_in_utc() -> None:
    r"""Pandas' bounds are naive, and comparing across awareness raises."""
    columns = [
        api.Column(name='SEEN', stype=Stype.timestamp, dtype=Dtype.time)
    ]

    frame = _previewed(
        columns, [['2026-01-01T12:00:00Z'], ['2026-01-01T12:00:00+05:30']])

    seen = list(frame['SEEN'])
    assert [value.tz_convert('UTC').isoformat() for value in seen] == [
        '2026-01-01T12:00:00+00:00',
        '2026-01-01T06:30:00+00:00',
    ]


def test_from_api_definition_reconstructs_sql_source_table() -> None:
    # connector_type is server-owned and may be unset on a def; reconstruction
    # recovers the warehouse from the fetched connector configuration.
    api_st = CustomSQLSourceTable(connector_id='sf-connector-id', sql=SQL,
                                  table='custom_sql_fixed')
    assert api_st.connector_type is None
    definition = api.TableDefinition(
        cols=[api.Column(name='store_id', stype=Stype.ID, dtype=Dtype.int)],
        source_table=api_st,
        pkey='store_id',
    )

    state = mock.MagicMock()
    state.client.connector_api.get.return_value = ConnectorResponse(
        id='sf-connector-id',
        project_id='project-id',
        identity_id='identity-snowflake',
        name='sf',
        timestamps=_TIMESTAMPS,
        config=SnowflakeConnectorResourceConfig(warehouse='wh', database='db',
                                                schema_name='sch', role='r'),
    )
    with mock.patch('kumoai_studio.backend.cloud_backend',
                    return_value=_backend_of(state)):
        restored = Table._from_api_table_definition(definition)

    assert isinstance(restored.source_table, SQLSourceTable)
    assert isinstance(restored.source_table.connector, SnowflakeConnector)
    assert restored.source_table.sql == SQL
    # The backend-assigned synthetic name is preserved, not recomputed:
    assert restored.source_table.name == 'custom_sql_fixed'
    assert restored.source_table.connector.id == 'sf-connector-id'


def test_print_definition_uses_from_sql(capsys) -> None:
    Table.from_sql(_snowflake(), SQL,
                   primary_key='store_id').print_definition()
    out = capsys.readouterr().out
    assert 'Table.from_sql(' in out
    assert SQL in out
    assert 'columns=' not in out


def test_print_definition_escapes_sql_and_role_names(capsys) -> None:
    sql = 'SELECT \'a\\\\b\' AS "customer\'s id", \'"""\' AS marker'
    role = "customer's id"
    Table.from_sql(
        _snowflake(),
        sql,
        columns=[dict(name=role, stype='ID', dtype='string')],
        primary_key=role,
    ).print_definition()
    out = capsys.readouterr().out
    assert f'sql={sql!r}' in out
    assert f'primary_key={role!r}' in out


def test_infer_metadata_applies_ordered_authority_atomically() -> None:
    table = Table.from_sql(_snowflake(), SQL, primary_key='store_id')
    source = CustomSQLSourceTable(connector_id='sf-connector-id', sql=SQL,
                                  table='custom_sql_canonical',
                                  connector_type=ConnectorType.SNOWFLAKE,
                                  referenced_tables=['DB.SCH.CST_SALES'])
    result = api.SqlTableResolveResponse(
        table_definition=api.TableDefinition(
            cols=[
                api.Column(name='store_id', stype=Stype.ID, dtype=Dtype.int),
                api.Column(name='total', stype=Stype.numerical,
                           dtype=Dtype.float),
            ], source_table=source, pkey='store_id'),
        preview_rows=[['1', '4.0']])
    client = mock.MagicMock()
    client.table_api.resolve_sql.return_value = result

    global_state._url = 'http://kumo.ai'
    global_state._api_key = 'DISABLED'
    global_state.thread_local._client = client
    try:
        source_cols = table.source_table.columns
        resolved = table.infer_metadata(inplace=False)
    finally:
        global_state.clear()

    assert [col.name for col in resolved.columns] == ['store_id', 'total']
    assert [col.name for col in source_cols] == ['store_id', 'total']
    assert resolved.column('total').dtype == Dtype.float
    assert resolved.source_name == 'custom_sql_canonical'
    assert isinstance(resolved.source_table, SQLSourceTable)
    assert resolved.source_table.referenced_tables == ['DB.SCH.CST_SALES']
    assert [col.name for col in table.columns] == ['store_id']
    with pytest.raises(AttributeError):
        object.__setattr__(resolved.source_table, 'sql', 'SELECT 1 AS x')
    assert client.table_api.resolve_sql.call_args.kwargs[
        'project_id'] == 'project-id'
    request = client.table_api.resolve_sql.call_args.args[0]
    # Role setters add placeholder columns locally, but columns=None still asks
    # the server to discover the complete SQL output.
    assert request.column_hints is None
    client.table_api.infer_metadata.assert_not_called()
    client.table_api.validate_table.assert_not_called()


def test_infer_metadata_rejects_malformed_response_without_mutation() -> None:
    table = Table.from_sql(
        _snowflake(), SQL,
        columns=[dict(name='store_id', stype='ID',
                      dtype='int')], primary_key='store_id')
    result = api.SqlTableResolveResponse(table_definition=api.TableDefinition(
        cols=[
            api.Column(name='x', stype=Stype.numerical, dtype=Dtype.int),
            api.Column(name='x', stype=Stype.numerical, dtype=Dtype.int)
        ], source_table=CustomSQLSourceTable(
            connector_id='sf-connector-id', sql=SQL, table='canonical',
            connector_type=ConnectorType.SNOWFLAKE)))
    client = mock.MagicMock()

    def assert_rejected(response, match: str) -> None:
        client.table_api.resolve_sql.return_value = response
        global_state._url = 'http://kumo.ai'
        global_state._api_key = 'DISABLED'
        global_state.thread_local._client = client
        try:
            with pytest.raises(ValueError, match=match):
                table.infer_metadata()
        finally:
            global_state.clear()
        assert [col.name for col in table.columns] == ['store_id']
        assert table.primary_key is not None
        assert table.primary_key.name == 'store_id'

    assert_rejected(result, 'unique aliases')
    request = client.table_api.resolve_sql.call_args.args[0]
    assert request.column_hints is not None
    assert [hint.name for hint in request.column_hints] == ['store_id']

    assert_rejected(
        api.SqlTableResolveResponse(table_definition=api.TableDefinition(
            cols=[api.Column('x', Stype.numerical,
                             Dtype.int)], source_table=CustomSQLSourceTable(
                                 'sf-connector-id', SQL, 'canonical',
                                 ConnectorType.SNOWFLAKE), pkey='x')),
        'different explicit column selection')

    assert_rejected(
        api.SqlTableResolveResponse(table_definition=api.TableDefinition(
            cols=[api.Column('store_id', Stype.ID,
                             Dtype.int)], source_table=CustomSQLSourceTable(
                                 'sf-connector-id', SQL, 'canonical',
                                 ConnectorType.SNOWFLAKE))),
        'changed a requested column role')

    assert_rejected(
        api.SqlTableResolveResponse(table_definition=api.TableDefinition(
            cols=[api.Column('store_id', Stype.numerical,
                             Dtype.int)], source_table=CustomSQLSourceTable(
                                 'sf-connector-id', SQL, 'canonical',
                                 ConnectorType.SNOWFLAKE), pkey='store_id')),
        "primary key must be 'ID'")

    assert_rejected(
        api.SqlTableResolveResponse(table_definition=api.TableDefinition(
            cols=[api.Column('x', Stype.numerical,
                             Dtype.int)], source_table=CustomSQLSourceTable(
                                 'other-connector', SQL, 'canonical',
                                 ConnectorType.SNOWFLAKE))),
        'different source')


def test_validate_rejects_sql_error_without_mutation() -> None:
    table = Table.from_sql(_snowflake(), SQL, primary_key='store_id')
    result = api.SqlTableResolveResponse(
        errors=[ValidationError('Invalid SQL', 'bad query')])
    client = mock.MagicMock()
    client.table_api.resolve_sql.return_value = result
    global_state._url, global_state._api_key = 'http://kumo.ai', 'DISABLED'
    global_state.thread_local._client = client
    try:
        with pytest.raises(ValueError, match='bad query'):
            table.validate()
    finally:
        global_state.clear()
    assert [col.name for col in table.columns] == ['store_id']


def _multicategorical_response(source_table) -> api.SqlTableResolveResponse:
    # Resolution proves the delimiter used for scalar-string multicategorical
    # columns; without it the returned Column is invalid.
    return api.SqlTableResolveResponse(table_definition=api.TableDefinition(
        cols=[
            api.Column(name='store_id', stype=Stype.ID, dtype=Dtype.int),
            api.Column(name='tags', stype=Stype.multicategorical,
                       dtype=Dtype.string, multicategorical_delimiter='|'),
        ], source_table=source_table, pkey='store_id'))


def test_infer_metadata_preserves_multicategorical_delimiter() -> None:
    table = Table.from_sql(_snowflake(), SQL, primary_key='store_id')

    client = mock.MagicMock()
    client.table_api.resolve_sql.return_value = _multicategorical_response(
        CustomSQLSourceTable(connector_id='sf-connector-id', sql=SQL,
                             table='canonical',
                             connector_type=ConnectorType.SNOWFLAKE))

    global_state._url = 'http://kumo.ai'
    global_state._api_key = 'DISABLED'
    global_state.thread_local._client = client
    try:
        table.infer_metadata()
    finally:
        global_state.clear()

    # The delimiter survives the response conversion and local validation.
    assert table.column('tags').stype == Stype.multicategorical
    assert table.column('tags').multicategorical_delimiter == '|'
    table._validate_definition()


def test_sql_resolution_timeout_is_bounded_and_classified() -> None:
    client = mock.MagicMock()
    client._request.side_effect = requests.Timeout()
    request = api.SqlTableResolveRequest(
        source_table=CustomSQLSourceTable('sf-connector-id', SQL, 'pending'))
    with pytest.raises(TimeoutError, match='SQL validation timed out'):
        TableAPI(client).resolve_sql(request, timeout=(1, 2))
    assert client._request.call_args.kwargs['timeout'] == (1, 2)
