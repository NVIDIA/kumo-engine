# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from kumo_snowflake.sql import placeholder, qualified_name, quote_identifier
from kumo_snowflake.types import dtype_for
from kumoapi.typing import Dtype


def columns(cursor: Any, *, database: str, schema: str,
            table: str | None = None) -> dict[tuple[str, str], Dtype]:
    mark = placeholder(cursor)
    query = (
        'SELECT c.TABLE_NAME, c.COLUMN_NAME, c.DATA_TYPE, c.NUMERIC_SCALE '
        f'FROM {qualified_name(database, "INFORMATION_SCHEMA", "COLUMNS")} c '
        f'JOIN {qualified_name(database, "INFORMATION_SCHEMA", "TABLES")} t '
        'ON c.TABLE_CATALOG = t.TABLE_CATALOG '
        'AND c.TABLE_SCHEMA = t.TABLE_SCHEMA AND c.TABLE_NAME = t.TABLE_NAME '
        f'WHERE c.TABLE_SCHEMA = {mark}')
    parameters = [schema]
    if table is not None:
        query += f' AND c.TABLE_NAME = {mark}'
        parameters.append(table)
    query += ' ORDER BY c.TABLE_NAME, c.ORDINAL_POSITION'
    cursor.execute(query, parameters)
    return {
        (str(table_name), str(column_name)): dtype_for(str(data_type), scale)
        for table_name, column_name, data_type, scale in cursor.fetchall()
    }


def tables(cursor: Any, *, database: str, schema: str) -> list[str]:
    cursor.execute(
        f'SELECT TABLE_NAME FROM {qualified_name(database, "INFORMATION_SCHEMA", "TABLES")} '
        f'WHERE TABLE_SCHEMA = {placeholder(cursor)} ORDER BY TABLE_NAME',
        [schema])
    return [str(row[0]) for row in cursor.fetchall()]


def table_exists(cursor: Any, *, database: str, schema: str,
                 table: str) -> bool:
    mark = placeholder(cursor)
    cursor.execute(
        f'SELECT 1 FROM {qualified_name(database, "INFORMATION_SCHEMA", "TABLES")} '
        f'WHERE TABLE_SCHEMA = {mark} AND TABLE_NAME = {mark} LIMIT 1',
        [schema, table])
    return cursor.fetchone() is not None


def table_type(cursor: Any, *, database: str, schema: str,
               table: str) -> str | None:
    mark = placeholder(cursor)
    cursor.execute(
        f'select TABLE_TYPE from {quote_identifier(database)}.INFORMATION_SCHEMA.TABLES '
        f'where TABLE_SCHEMA = {mark} and TABLE_NAME = {mark}', (schema, table))
    rows = cursor.fetchall()
    return str(rows[0][0]) if rows else None
