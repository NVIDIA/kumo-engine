# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import kumoapi.graph as graph_api
import pytest
from kumoapi.typing import Dtype, Stype

from kumo.validation import ValidationException
from rest.config.graph_v2 import (
    ColumnKey,
    ColumnKeyGroup,
    GraphDefinition,
    build_graph_definition,
)
from rest.config.table_v2 import Column, DatabricksSourceTable, TableDefinition
from statedb.config.common import SourceTableName, TableIDv2, TableName


def test_validate_graph_args():
    tab_b = TableDefinition(
        cols=(Column('pk', Stype.ID, Dtype.string), ),
        source_table=DatabricksSourceTable(
            databricks_connector_id='dc_b',
            table=SourceTableName('src'),
        ),
        pkey='pk',
    )
    tab_c = TableDefinition(
        cols=(Column('pk', Stype.ID, Dtype.string), ),
        source_table=DatabricksSourceTable(
            databricks_connector_id='dc_c',
            table=SourceTableName('src'),
        ),
        pkey='pk',
    )
    col_group = ColumnKeyGroup(columns=(
        ColumnKey(table_name=TableName('b'), col_name='1'),
        ColumnKey(table_name=TableName('c'), col_name='2'),
    ))

    with pytest.raises(ValueError,
                       match=r"Table name is missing from the map: c"):
        col_group.as_column_group({
            TableName('b'):
            TableIDv2.compose(None, TableName('b'), tab_b.fpid),
        })

    name_to_tid = {
        TableName('b'): TableIDv2.compose(None, TableName('b'), tab_b.fpid),
        TableName('c'): TableIDv2.compose(None, TableName('c'), tab_c.fpid),
    }
    col_group.as_column_group(name_to_tid)

    GraphDefinition(
        tables={
            TableName('b'): tab_b,
            TableName('c'): tab_c,
        },
        col_groups=[col_group],
    )
    # TODO(zack): add tests for the remaining invarients


def test_build_graph_definition_converts_public_definition():
    graph = graph_api.GraphDefinition(tables={}, col_groups=[])

    result = build_graph_definition(graph)

    assert isinstance(result, GraphDefinition)


def test_build_graph_definition_sanitizes_validation_errors():
    column = graph_api.ColumnKey(table_name='customers', col_name='id')
    graph = graph_api.GraphDefinition(
        tables={},
        col_groups=[
            graph_api.ColumnKeyGroup(columns=(column, column)),
        ],
    )

    with pytest.raises(ValidationException) as exc_info:
        build_graph_definition(graph)

    message = exc_info.value.get_user_message()
    assert 'col_groups.0: All columns must be unique, found 1 dupes' in message
    assert 'input_value' not in message
    assert 'errors.pydantic.dev' not in message


def test_graph_edges_share_resolution_and_keep_legacy_orientation():
    users = TableDefinition(
        cols=(
            Column('id', Stype.ID, Dtype.string),
            Column('legacy_id', Stype.ID, Dtype.string),
        ),
        source_table=DatabricksSourceTable(
            databricks_connector_id='dc',
            table=SourceTableName('users'),
        ),
        pkey='id',
    )
    events = TableDefinition(
        cols=(Column('user_id', Stype.ID, Dtype.string), ),
        source_table=DatabricksSourceTable(
            databricks_connector_id='dc',
            table=SourceTableName('events'),
        ),
    )
    graph = GraphDefinition(
        tables={
            TableName('users'): users,
            TableName('events'): events,
        },
        col_groups=[
            ColumnKeyGroup(columns=(
                ColumnKey(TableName('users'), 'id'),
                ColumnKey(TableName('users'), 'legacy_id'),
                ColumnKey(TableName('events'), 'user_id'),
            ))
        ],
    )

    assert graph.edges == [
        graph_api.Edge('users', 'user_id', 'events'),
        graph_api.Edge('users', 'legacy_id', 'users'),
    ]
