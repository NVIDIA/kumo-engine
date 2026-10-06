# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import uuid
from collections.abc import Callable

import pandas as pd
import pytest

from kumo.connector.csv_connector import CSVConnector
from kumo.graph.graph import Graph

# Node Prediction Graphs ######################################################


@pytest.fixture()
def create_node_pred_temporal_split_test_graph() -> Callable:
    return node_pred_temporal_split_test_graph


def node_pred_temporal_split_test_graph(
    name: str,
    root_dir: str,
    create_hypothetical_entity: bool = False,
) -> Graph:
    """Graph used to test temporal node prediction queries. Graph
    has tables `USERS`, `TRANSACTIONS`, `SESSIONS`, `ORDERS` and
    `BANKS`. Since this graph has only one timeframe it will generate
    an empty val table under most split method.

    Args:
        name: Name of graph.
        root_dir: Directory where csv files of tables will be saved.
        create_hypothetical_entity: If True, create a hypothetical entity
            `NEW_USERS` that is not present in the graph, but is saved in
            in the same directory as the other tables.
    """
    # Create tables:
    users = pd.DataFrame({'USER_ID': [0, 1, 2, 3]})
    new_users = pd.DataFrame({'USER_ID': [4, 5]})
    users = users.set_index('USER_ID')
    transactions = pd.DataFrame({
        'USER_ID': [0, 1, 2, 3, 1, 3, 3, 0, 1, 0, 0, 1, 2, 3, 0, 1, 2, 3],
        'AMOUNT': [
            0.0, 0.0, 0.0, 0.0, 150.0, 20.0, 30.0, 30.0, 150.0, 0.0, 30.0,
            30.0, 150.0, 0.0, 30.0, 30.0, 150.0, 0.0
        ],
        'BANK_ID': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        'DATE': [
            '1990-01-01', '1990-01-01', '1990-01-01', '1990-01-01',
            '1990-01-03', '1990-01-03', '1990-01-04', '1990-01-04',
            '1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07',
            '1990-01-07', '1990-01-07', '1990-01-08', '1990-01-08',
            '1990-01-08', '1990-01-09'
        ],
    })
    transactions['DATE'] = pd.to_datetime(transactions['DATE'])
    sessions = pd.DataFrame({
        'USER_ID': [1, 3, 1, 3],
        'DURATION': [1.0, 1.0, 1.0, 1.0],
        'DATE': ['1990-01-02', '1990-01-02', '1990-01-04', '1990-01-04'],
    })
    sessions['DATE'] = pd.to_datetime(sessions['DATE'])
    orders = pd.DataFrame({
        'USER_ID': [1, 2, 1, 2],
        'BALANCE': [1.0, 1.0, 1.0, 1.0],
        'DATE': ['1990-01-02', '1990-01-02', '1990-01-04', '1990-01-04'],
    })
    orders['DATE'] = pd.to_datetime(orders['DATE'])
    banks = pd.DataFrame({
        'BANK_ID': [0, 1],
        'BANK_TYPE': ['A', 'B'],
    })

    # Save tables to csv files:
    tables_dir = 'tables-' + uuid.uuid4().hex[:16]
    tables_path = os.path.join(root_dir, tables_dir)
    os.makedirs(tables_path)
    tables_names = ['USERS', 'TRANSACTIONS', 'SESSIONS', 'ORDERS', 'BANKS']
    tables = [users, transactions, sessions, orders, banks]
    for table_name, table in zip(tables_names, tables):
        table_path = os.path.join(tables_path, table_name + '.csv')
        table.to_csv(table_path,
                     index=True if table_name == 'USERS' else False)
    if create_hypothetical_entity:
        new_users_path = os.path.join(tables_path, 'NEW_USERS.csv')
        new_users.to_csv(new_users_path, index=False)

    # Create graph:
    graph = Graph(name=name)
    graph.connect(CSVConnector('csv',
                               root_dir=tables_path,
                               na_values='?'))
    graph.add('USERS', pkey='USER_ID')
    graph.add('TRANSACTIONS', cols=['AMOUNT'], time_col='DATE')
    graph.add('SESSIONS', cols=['DURATION'], time_col='DATE')
    graph.add('ORDERS', cols=['BALANCE'], time_col='DATE')
    graph.add('BANKS', cols=['BANK_TYPE'], pkey='BANK_ID')
    graph.link(src_table='TRANSACTIONS', src_key='USER_ID', dst_table='USERS')
    graph.link(src_table='SESSIONS', src_key='USER_ID', dst_table='USERS')
    graph.link(src_table='ORDERS', src_key='USER_ID', dst_table='USERS')
    graph.link(src_table='TRANSACTIONS', src_key='BANK_ID', dst_table='BANKS')
    return graph.infer_metadata().infer_encoders()
