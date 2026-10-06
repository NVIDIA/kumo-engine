# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from kumo_snowflake.catalog import columns
from kumoapi.typing import Dtype


@pytest.mark.parametrize('style, marker', [('pyformat', '%s'), ('qmark', '?')])
def test_catalog_binds_exact_names_and_joins_visible_tables(style, marker):
    cursor = Mock()
    cursor.connection._paramstyle = style
    cursor.fetchall.return_value = [
        ('Mixed.Table', 'Amount', 'NUMBER', 18),
        ('Mixed.Table', 'ID', 'NUMBER', 0),
        ('Mixed.Table', 'Array', 'ARRAY', None),
        ('Mixed.Table', 'Timestamp', 'TIMESTAMP_TZ', None),
    ]

    found = columns(cursor, database='Mixed"Db', schema="Schema's",
                    table='Mixed.Table')

    query, params = cursor.execute.call_args.args
    assert '"Mixed""Db"."INFORMATION_SCHEMA"."COLUMNS" c' in query
    assert '"Mixed""Db"."INFORMATION_SCHEMA"."TABLES" t' in query
    assert 'c.TABLE_CATALOG = t.TABLE_CATALOG' in query
    assert 'c.TABLE_SCHEMA = t.TABLE_SCHEMA' in query
    assert 'c.TABLE_NAME = t.TABLE_NAME' in query
    assert f'WHERE c.TABLE_SCHEMA = {marker} AND c.TABLE_NAME = {marker}' in query
    assert params == ["Schema's", 'Mixed.Table']
    assert 'TABLE_TYPE' not in query
    assert list(found.items()) == [
        (('Mixed.Table', 'Amount'), Dtype.float64),
        (('Mixed.Table', 'ID'), Dtype.int64),
        (('Mixed.Table', 'Array'), Dtype.string),
        (('Mixed.Table', 'Timestamp'), Dtype.time),
    ]
    cursor.close.assert_not_called()


def test_schema_catalog_queries_all_columns_in_catalog_order():
    cursor = Mock()
    cursor.connection._paramstyle = 'pyformat'
    cursor.fetchall.return_value = [(f'TABLE_{i // 2:02}', f'COLUMN_{i % 2:04}',
                                     'NUMBER', 0) for i in range(3)]

    found = columns(cursor, database='DB', schema='PUBLIC')

    query, params = cursor.execute.call_args.args
    assert 'SHOW ' not in query and 'LIMIT ' not in query
    assert query.endswith('ORDER BY c.TABLE_NAME, c.ORDINAL_POSITION')
    assert params == ['PUBLIC']
    assert len(found) == 3
    assert next(reversed(found)) == ('TABLE_01', 'COLUMN_0000')
