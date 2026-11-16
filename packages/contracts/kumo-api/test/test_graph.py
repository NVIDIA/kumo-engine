# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from kumoapi.graph import (
    ColumnKey,
    ColumnKeyGroup,
    Edge,
    GraphSpecError,
    group_column_relationships,
    resolve_foreign_keys,
    validate_edge_endpoints,
)
from kumoapi.source_table import UnavailableSourceTable
from kumoapi.table import Column, TableDefinition
from kumoapi.typing import Dtype, Stype


def _table(name, columns, primary_key=None):
    return TableDefinition(
        cols=[Column(column, Stype.ID, Dtype.string) for column in columns],
        source_table=UnavailableSourceTable(name),
        pkey=primary_key,
    )


def test_column_relationships_group_by_destination_deterministically():
    users = ColumnKey("users", "id")
    products = ColumnKey("products", "id")

    groups = group_column_relationships([
        (ColumnKey("views", "user_id"), users),
        (ColumnKey("orders", "product_id"), products),
        (ColumnKey("orders", "user_id"), users),
    ])

    assert groups[0].columns == (
        ColumnKey("orders", "user_id"),
        users,
        ColumnKey("views", "user_id"),
    )
    assert groups[1].columns == (
        ColumnKey("orders", "product_id"),
        products,
    )


def test_column_relationships_leave_duplicates_for_validation():
    users = ColumnKey("users", "id")
    source = ColumnKey("orders", "user_id")

    groups = group_column_relationships([(source, users), (source, users)])

    assert groups[0].columns == (source, source, users)


@pytest.mark.parametrize(
    "edge, code, missing",
    [
        (Edge("missing", "user_id", "users"), "missing_source_table",
         "missing source table 'missing'"),
        (Edge("events", "user_id", "missing"), "missing_destination_table",
         "missing destination table 'missing'"),
    ],
)
def test_edge_endpoints_must_name_available_tables(edge, code, missing):
    tables = {"events": object(), "users": object()}

    with pytest.raises(GraphSpecError) as error:
        validate_edge_endpoints(tables, [edge])

    assert error.value.code == code
    assert error.value.edge == edge
    assert missing in str(error.value)
    assert "available tables: ['events', 'users']" in str(error.value)


def test_edge_endpoint_validation_accepts_valid_and_self_edges():
    validate_edge_endpoints(
        {
            "events": object(),
            "users": object()
        },
        [
            Edge("events", "user_id", "users"),
            Edge("users", "legacy_id", "users"),
        ],
    )


def test_foreign_keys_resolve_in_declared_order_with_self_edges():
    tables = {
        "users": _table("users", ("id", "legacy_id"), "id"),
        "events": _table("events", ("id", "user_id", "other_user_id"), "id"),
    }
    groups = [
        ColumnKeyGroup((
            ColumnKey("events", "other_user_id"),
            ColumnKey("users", "legacy_id"),
            ColumnKey("users", "id"),
        )),
        ColumnKeyGroup((
            ColumnKey("events", "user_id"),
            ColumnKey("users", "id"),
        )),
    ]

    assert resolve_foreign_keys(tables, groups) == (
        Edge("events", "other_user_id", "users"),
        Edge("users", "legacy_id", "users"),
        Edge("events", "user_id", "users"),
    )


def test_foreign_keys_require_one_primary_key_per_group():
    tables = {
        "users": _table("users", ("id", ), "id"),
        "events": _table("events", ("id", "user_id"), "id"),
    }

    with pytest.raises(GraphSpecError) as error:
        resolve_foreign_keys(tables, [
            ColumnKeyGroup((
                ColumnKey("users", "id"),
                ColumnKey("events", "id"),
            ))
        ])

    assert error.value.code == "multiple_primary_keys"


def test_foreign_keys_reject_a_group_without_a_primary_key():
    tables = {"events": _table("events", ("left", "right"))}

    with pytest.raises(GraphSpecError) as error:
        resolve_foreign_keys(tables, [
            ColumnKeyGroup((
                ColumnKey("events", "left"),
                ColumnKey("events", "right"),
            ))
        ])

    assert error.value.code == "no_primary_key"


def test_foreign_keys_reject_a_missing_source_column():
    tables = {
        "users": _table("users", ("id", ), "id"),
        "events": _table("events", ("user_id", )),
    }

    with pytest.raises(GraphSpecError) as error:
        resolve_foreign_keys(tables, [
            ColumnKeyGroup((
                ColumnKey("users", "id"),
                ColumnKey("events", "missing"),
            ))
        ])

    assert error.value.code == "missing_column"
    assert error.value.column == ColumnKey("events", "missing")


def test_foreign_keys_reject_a_missing_destination_column():
    tables = {
        "users": _table("users", ("legacy_id", ), "id"),
        "events": _table("events", ("user_id", )),
    }

    with pytest.raises(GraphSpecError) as error:
        resolve_foreign_keys(tables, [
            ColumnKeyGroup((
                ColumnKey("users", "id"),
                ColumnKey("events", "user_id"),
            ))
        ])

    assert error.value.code == "missing_column"
    assert error.value.column == ColumnKey("users", "id")


def test_graph_definition_uses_aliases_without_rewriting_sources():
    from kumoapi.graph import graph_definition_from_edges

    tables = {
        'markets': _table('markets.parquet', ['id'], 'id'),
        'events': _table('events.parquet', ['id', 'market_id'], 'id'),
    }
    definition = graph_definition_from_edges(
        tables, [Edge('events', 'market_id', 'markets')])
    assert definition.tables['markets'].source_table == tables[
        'markets'].source_table
    assert resolve_foreign_keys(definition.tables,
                                definition.col_groups) == (Edge(
                                    'events', 'market_id', 'markets'), )
    with pytest.raises(GraphSpecError, match='missing destination table'):
        graph_definition_from_edges(tables,
                                    [Edge('events', 'market_id', 'missing')])
    tables['markets'] = _table('markets.parquet', ['id'])
    with pytest.raises(ValueError, match='does not have a primary key'):
        graph_definition_from_edges(tables,
                                    [Edge('events', 'market_id', 'markets')])
