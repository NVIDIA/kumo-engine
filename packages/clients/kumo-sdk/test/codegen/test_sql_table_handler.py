# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
from typing import cast

import pytest
from kumoapi.data_source import (
    ConnectorResponse,
    ResourceTimestamps,
    SnowflakeConnectorResourceConfig,
)

from kumoai_studio.codegen.context import CodegenContext
from kumoai_studio.codegen.handlers.table import get_handlers
from kumoai_studio.codegen.identity import get_config_id
from kumoai_studio.connector import SnowflakeConnector
from kumoai_studio.graph import Table

SQL = "SELECT store_id, price * qty AS total FROM CST_SALES"

_TIMESTAMPS = ResourceTimestamps(
    created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
)


def _snowflake() -> SnowflakeConnector:
    config = SnowflakeConnectorResourceConfig(warehouse='w', database='d',
                                              schema_name='s', role='r')
    response = ConnectorResponse(
        id='sf-connector-id',
        project_id='project-id',
        identity_id='identity-snowflake',
        name='sf',
        timestamps=_TIMESTAMPS,
        config=config,
    )
    return SnowflakeConnector(name='sf', warehouse='w', database='d',
                              schema_name='s', _response=response)


def test_sql_table_emits_from_sql() -> None:
    conn = _snowflake()
    table = Table.from_sql(
        conn,
        SQL,
        columns=[
            dict(name='store_id', stype='ID', dtype='int'),
            dict(name='total', stype='numerical', dtype='float'),
        ],
        primary_key='store_id',
    )

    handler = get_handlers()[Table]
    ctx = CodegenContext()
    ctx.object_to_var = {get_config_id(conn): 'sf_conn'}
    code = "\n".join(handler.emit_lines(table, 'my_table', {}, ctx))

    # The custom-SQL table regenerates via Table.from_sql against the
    # connector, carrying the query and full output schema:
    assert 'Table.from_sql(' in code
    assert 'connector=sf_conn' in code
    assert f'sql={SQL!r}' in code
    assert "primary_key='store_id'" in code
    assert "dict(name='store_id', stype='ID', dtype='int')" in code
    assert "dict(name='total', stype='numerical', dtype='float')" in code

    # Never the physical, name-keyed source-table path (synthetic name leak):
    assert 'from_source_table' not in code
    assert 'custom_sql_' not in code

    lazy = Table.from_sql(conn, SQL, primary_key='store_id')
    assert 'columns=' not in '\n'.join(
        handler.emit_lines(lazy, 'lazy_table', {}, ctx))


def test_sql_table_emits_valid_python_for_tricky_alias() -> None:
    # A column alias may legally contain a single quote; the emitted literals
    # must be repr-escaped so the generated code parses.
    tricky = "customer's id"
    conn = _snowflake()
    table = Table.from_sql(
        conn,
        SQL,
        columns=[dict(name=tricky, stype='ID', dtype='int')],
        primary_key=tricky,
    )

    handler = get_handlers()[Table]
    ctx = CodegenContext()
    ctx.object_to_var = {get_config_id(conn): 'sf_conn'}
    code = "\n".join(handler.emit_lines(table, 'my_table', {}, ctx))

    # Raw single quotes would produce invalid Python; compile proves otherwise:
    compile(code, '<codegen>', 'exec')
    assert f"name={tricky!r}" in code
    assert f"primary_key={tricky!r}" in code


def test_sql_table_emits_multicategorical_delimiter() -> None:
    conn = _snowflake()
    table = Table.from_sql(
        conn,
        SQL,
        columns=[
            dict(name='tags', stype='multicategorical', dtype='string',
                 multicategorical_delimiter='|'),
        ],
    )

    handler = get_handlers()[Table]
    ctx = CodegenContext()
    ctx.object_to_var = {get_config_id(conn): 'sf_conn'}
    code = "\n".join(handler.emit_lines(table, 'my_table', {}, ctx))

    # The delimiter must be emitted; replaying a from_sql(...) that omits it
    # rebuilds a scalar multicategorical Column with no delimiter and raises
    # ValueError. Recreate the table from the generated code to prove it holds.
    compile(code, '<codegen>', 'exec')
    assert "multicategorical_delimiter='|'" in code
    scope = {'Table': Table, 'sf_conn': conn}
    exec(code, scope)
    regenerated = cast(Table, scope['my_table'])
    assert regenerated.column('tags').multicategorical_delimiter == '|'


def test_sql_codegen_preserves_format_and_rejects_empty_selection() -> None:
    conn = _snowflake()
    table = Table.from_sql(
        conn,
        SQL,
        columns=[
            dict(name='event_time', stype='timestamp', dtype='int',
                 timestamp_format='ms'),
        ],
        time_column='event_time',
    )
    handler = get_handlers()[Table]
    ctx = CodegenContext()
    ctx.object_to_var = {get_config_id(conn): 'sf_conn'}
    code = "\n".join(handler.emit_lines(table, 'my_table', {}, ctx))
    assert "timestamp_format='ms'" in code

    with pytest.raises(ValueError, match='omitted for discovery'):
        Table.from_sql(conn, SQL, columns=[])
