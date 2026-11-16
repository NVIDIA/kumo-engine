# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pandas as pd
from kumo_databricks.catalog import column_types, table_names


def frame(*rows):
    return pd.DataFrame(
        rows, columns=['TABLE_SCHEM', 'TABLE_NAME', 'COLUMN_NAME', 'TYPE_NAME'])


def test_catalog_listing_preserves_unrestricted_request_and_target_filter():
    cursor = Mock()
    cursor.columns.return_value.fetchall_arrow.return_value.to_pandas.return_value = frame(
        ('sales', 'orders', 'id', 'BIGINT'),
        ('Sales', 'orders', 'amount', 'DECIMAL(38,18)'),
        ('information_schema', 'tables', 'id', 'STRING'),
        ('', 'empty', 'id', 'STRING'), ('other', 'orders', 'id', 'BIGINT'))
    assert column_types(cursor, catalog='main', schema=None,
                        target_schema='sales') == {
                            ('orders', 'id'): 'BIGINT',
                            ('orders', 'amount'): 'DECIMAL(38,18)'
                        }
    cursor.columns.assert_called_once_with(catalog_name='main',
                                           schema_name=None)
    cursor.close.assert_not_called()


def test_table_listing_sorts_visible_names_in_target_schema():
    cursor = Mock()
    cursor.columns.return_value.fetchall_arrow.return_value.to_pandas.return_value = frame(
        ('sch', 'z_table', 'id', 'BIGINT'), ('SCH', 'a_table', 'id', 'BIGINT'),
        ('sch', 'a_table', 'value', 'STRING'),
        ('other', 'hidden', 'id', 'BIGINT'),
        ('information_schema', 'tables', 'id', 'BIGINT'))
    assert table_names(cursor, catalog='cat`alog', schema='sch',
                       target_schema='sch') == ['a_table', 'z_table']
    cursor.columns.assert_called_once_with(catalog_name='cat`alog',
                                           schema_name='sch')
    cursor.execute.assert_not_called()
    cursor.close.assert_not_called()
