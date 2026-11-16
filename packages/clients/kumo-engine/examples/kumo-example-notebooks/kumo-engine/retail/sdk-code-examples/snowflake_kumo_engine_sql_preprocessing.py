# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preprocess customer-location entities with read-only Snowflake SQL.
Synthetic data only.
Generate synthetic example data first:
    python synthetic-data/engine_retail/generate.py --dataset stores --output-dir ./synthetic-stores
Use that directory as --source-root or --input-dir. Metrics are demonstration only.
Manually load these files into your own Snowflake schema, using uppercase table and column names.
For fashion, load transactions.parquet as TRANSACTIONS; customers/articles keep their names.
The generator writes local files only. Configure your own SPCS environment or account credentials.
Install: pip install "kumo-engine[snowflake]"
"""
import os

from kumo.engine import Edge, Graph, SnowflakeConnector, Table

source = SnowflakeConnector.from_connection_toml(
    connection_name=os.environ['SNOWFLAKE_CONNECTION_NAME'])
users = Table.from_source_table(
    source.sql(
        "SELECT TO_JSON(ARRAY_CONSTRUCT(user_id, location_id)) AS entity_id, user_id, location_id, created_time FROM USERS"
    ), primary_key='ENTITY_ID', time_column='CREATED_TIME')
orders = Table.from_source_table(
    source.sql(
        "SELECT TO_JSON(ARRAY_CONSTRUCT(user_id, location_id)) AS entity_id, store_id, order_time FROM ORDERS"
    ), time_column='ORDER_TIME')
stores = Table.from_source_table(source['STORES'], primary_key='STORE_ID',
                                 time_column='CREATED_TIME')
graph = Graph({
    'users': users,
    'orders': orders,
    'stores': stores
}, [Edge('orders', 'ENTITY_ID', 'users'),
    Edge('orders', 'STORE_ID', 'stores')])
graph.validate()
print('Validated graph:', graph.id)
