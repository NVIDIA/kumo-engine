# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from kumoapi.typing import Dtype

from kumo.connector.bigquery_connector import BigQueryConnector
from kumo.connector.connector import ConnectorError
from kumo.connector.databricks_connector import DatabricksConnector
from kumo.connector.snowflake_connector import SnowflakeConnector


def test_databricks_identity_connector_never_falls_back_to_environment(
        monkeypatch) -> None:
    monkeypatch.setenv('DATABRICKS_TOKEN', 'environment-pat')
    monkeypatch.setenv('DATABRICKS_CLIENT_ID', 'environment-client')
    monkeypatch.setenv('DATABRICKS_CLIENT_SECRET', 'environment-secret')
    connector = DatabricksConnector(
        name='warehouse',
        host='https://workspace.cloud.databricks.com',
        cluster_id='',
        warehouse_id='sql-warehouse',
        catalog='main',
        schema_name='sales',
    )
    connector.require_explicit_credentials()

    with pytest.raises(ConnectorError, match='explicit OAuth'):
        connector._to_databricks_connection_info()


def test_snowflake_identity_connector_never_falls_back_to_environment(
        monkeypatch) -> None:
    monkeypatch.setenv('SNOWFLAKE_USER', 'environment-user')
    monkeypatch.setenv('SNOWFLAKE_PASSWORD', 'environment-password')
    connector = SnowflakeConnector(
        name='warehouse',
        account='account',
        warehouse='COMPUTE',
        database='DATA',
        schema_name='SALES',
    )
    connector.require_explicit_credentials()

    with pytest.raises(ConnectorError, match='explicit.*credentials'):
        connector._open(credentials=None)


def test_bigquery_identity_connector_never_falls_back_to_adc() -> None:
    connector = BigQueryConnector(
        name='warehouse',
        project_id='data-project',
        dataset_id='sales',
    )
    connector.require_explicit_credentials()

    with patch.object(connector, '_use_credentials') as fallback:
        with pytest.raises(ConnectorError, match='explicit.*credentials'):
            connector._open(credentials=None)
    fallback.assert_not_called()


def test_databricks_connector_schema_is_a_real_target_coordinate() -> None:
    connector = DatabricksConnector(
        name='warehouse',
        host='https://workspace.cloud.databricks.com',
        cluster_id='',
        warehouse_id='sql-warehouse',
        catalog='main',
        schema_name='sales',
    )

    connector._context = MagicMock()
    with patch('kumo_databricks.catalog.table_names', return_value=[]) as get:
        assert connector.list_tables() == []

    get.assert_called_once_with(
        connector._context.cursor.return_value.__enter__.return_value,
        catalog='main', schema='sales', target_schema='sales')
    assert connector.sql_table_name('orders') == '`sales`.`orders`'
    with pytest.raises(ConnectorError, match='connector schema'):
        connector.sql_table_name('finance.orders')
    with pytest.raises(ConnectorError, match='connector schema'):
        connector.list_tables('finance')


def test_fixed_schema_databricks_discovery_uses_bare_table_names() -> None:
    connector = DatabricksConnector(
        name='warehouse',
        host='https://workspace.cloud.databricks.com',
        cluster_id='',
        warehouse_id='sql-warehouse',
        catalog='main',
        schema_name='sales',
    )
    columns = {
        ('orders', 'order_id'): Dtype.int64,
        ('customers', 'customer_id'): Dtype.int64,
    }
    connector._context = MagicMock()

    with patch.object(connector, 'get_column_metadata', return_value=columns), \
            patch('kumo_databricks.catalog.table_names',
                  return_value=['customers', 'orders']):
        assert set(connector.list_tables()) == {'orders', 'customers'}
        assert connector.table_exists('orders')
        metadata = connector.get_table_metadata(
            table_names=['orders'],
            skip_cache=True,
            skip_inferral=True,
        )

    assert [table.table_name for table in metadata] == ['orders']


def test_fixed_schema_databricks_bare_name_resolves_provider_coordinate(
) -> None:
    connector = DatabricksConnector(
        name='warehouse',
        host='https://workspace.cloud.databricks.com',
        cluster_id='',
        warehouse_id='sql-warehouse',
        catalog='main',
        schema_name='sales',
    )
    context = MagicMock()
    cursor = context.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [('order_id', 'BIGINT', '')]

    result = connector.get_column_metadata(context, 'sales', ['orders'])

    assert result == {('orders', 'order_id'): Dtype.int64}
    cursor.execute.assert_called_once_with(
        'DESCRIBE EXTENDED `main`.`sales`.`orders`')
    cursor.columns.assert_not_called()
    with pytest.raises(ConnectorError, match='connector schema'):
        connector.get_column_metadata(
            context,
            'sales',
            ['finance.orders'],
        )


def test_fixed_schema_databricks_discards_catalog_wide_metadata_rows() -> None:
    connector = DatabricksConnector(
        name='warehouse',
        host='https://workspace.cloud.databricks.com',
        cluster_id='',
        warehouse_id='sql-warehouse',
        catalog='main',
        schema_name='sales',
    )
    provider_metadata = pd.DataFrame({
        'TABLE_SCHEM': ['sales', 'finance', 'information_schema'],
        'TABLE_NAME': ['orders', 'payroll', 'tables'],
        'COLUMN_NAME': ['order_id', 'employee_id', 'table_name'],
        'TYPE_NAME': ['BIGINT', 'BIGINT', 'STRING'],
    })
    context = MagicMock()
    cursor = context.cursor.return_value.__enter__.return_value
    cursor.columns.return_value.fetchall_arrow.return_value.to_pandas.return_value = provider_metadata

    result = connector.get_column_metadata(context, 'sales')

    assert result == {('orders', 'order_id'): Dtype.int64}
    cursor.columns.assert_called_once_with(
        catalog_name='main',
        schema_name='sales',
    )
