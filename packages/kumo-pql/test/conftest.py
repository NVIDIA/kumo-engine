# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import pytest
from kumoapi.graph import ColumnKey, ColumnKeyGroup, GraphDefinition
from kumoapi.source_table import UnavailableSourceTable
from kumoapi.table import Column, TableDefinition
from kumoapi.typing import Dtype, Stype


@pytest.fixture()
def create_mock_graph() -> Callable[..., GraphDefinition]:
    return mock_graph


def mock_graph(add_time_cols: bool = True) -> GraphDefinition:
    graph = GraphDefinition(
        tables={
            "USERS":
            TableDefinition(
                cols=[
                    Column(name='USERS_ID', stype=Stype.ID, dtype=Dtype.int),
                    Column(name='AGE', stype=Stype.numerical, dtype=Dtype.int),
                    Column(name='TIME', stype=Stype.timestamp,
                           dtype=Dtype.time),
                    Column(name='MULTI', stype=Stype.multicategorical,
                           dtype=Dtype.string, multicategorical_delimiter=','),
                    Column(name='ARRAY', stype=Stype.multicategorical,
                           dtype=Dtype.intlist),
                    Column(name='DESC', stype=Stype.text, dtype=Dtype.string),
                    Column(name='LOCATION_ID', stype=Stype.ID, dtype=Dtype.int),
                    Column(name='EXTERNAL_ID', stype=Stype.ID, dtype=Dtype.int),
                ],
                source_table=UnavailableSourceTable(table='USERS'),
                pkey='USERS_ID',
            ),
            "TRANS":
            TableDefinition(
                cols=[
                    Column(name='TRANSACTION_ID', stype=Stype.ID,
                           dtype=Dtype.int),
                    Column(name='USER', stype=Stype.ID, dtype=Dtype.int),
                    Column(name='STORE', stype=Stype.ID, dtype=Dtype.int),
                    Column(name='TYPE', stype=Stype.categorical,
                           dtype=Dtype.string),
                    Column(name='CNT', stype=Stype.numerical, dtype=Dtype.int),
                    Column(name='FLAG', stype=Stype.categorical,
                           dtype=Dtype.bool),
                    Column(name='VALUE', stype=Stype.numerical,
                           dtype=Dtype.float),
                    Column(name='ARRAY', stype=Stype.multicategorical,
                           dtype=Dtype.intlist),
                ],
                source_table=UnavailableSourceTable(table='TRANS'),
                pkey='TRANSACTION_ID',
            ),
            "STORES":
            TableDefinition(
                cols=[
                    Column(name='STORES_ID', stype=Stype.ID, dtype=Dtype.int),
                    Column(name='TYPE', stype=Stype.categorical,
                           dtype=Dtype.string),
                ],
                source_table=UnavailableSourceTable(table='STORES'),
                pkey='STORES_ID',
            ),
            "RETURNS":
            TableDefinition(
                cols=[
                    Column(name='TRANSACTION_ID', stype=Stype.ID,
                           dtype=Dtype.int),
                ],
                source_table=UnavailableSourceTable(table='RETURNS'),
            ),
            "LOCATIONS":
            TableDefinition(
                cols=[
                    Column(name='LOCATION_ID', stype=Stype.ID, dtype=Dtype.int),
                    Column(name='NAME', stype=Stype.categorical,
                           dtype=Dtype.string),
                ],
                source_table=UnavailableSourceTable(table='LOCATIONS'),
                pkey='LOCATION_ID',
            ),
        },
        col_groups=[
            ColumnKeyGroup(columns=(
                ColumnKey('TRANS', 'USER'),
                ColumnKey('USERS', 'USERS_ID'),
            )),
            ColumnKeyGroup(columns=(
                ColumnKey('STORES', 'STORES_ID'),
                ColumnKey('TRANS', 'STORE'),
            )),
            ColumnKeyGroup(columns=(
                ColumnKey('RETURNS', 'TRANSACTION_ID'),
                ColumnKey('TRANS', 'TRANSACTION_ID'),
            )),
            ColumnKeyGroup(columns=(
                ColumnKey('LOCATIONS', 'LOCATION_ID'),
                ColumnKey('USERS', 'LOCATION_ID'),
            )),
        ],
    )
    if add_time_cols:
        graph.tables['TRANS'].cols.append(
            Column(name='TIME', dtype=Dtype.time, stype=Stype.timestamp))
        graph.tables['TRANS'].time_col = 'TIME'
        graph.tables['USERS'].cols.append(
            Column(name='TIME', dtype=Dtype.time, stype=Stype.timestamp))
        graph.tables['USERS'].time_col = 'TIME'
        graph.tables['RETURNS'].cols.append(
            Column(name='TIME', dtype=Dtype.time, stype=Stype.timestamp))
        graph.tables['RETURNS'].time_col = 'TIME'
    return graph
