# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumoapi.common import ValidationResponse
from kumoapi.graph import ColumnKey, GraphDefinition
from kumoapi.pquery import ValidatedPredictiveQuery
from kumoapi.pquery.AST import Aggregation


def col_name(full_name: str) -> str:
    r""""Given a fully qualified column name in the format of
    `<table_name>`.`<col_name>`, returns the column name.

    Args:
        full_name: Fully qualified column name in the format of
            `<table_name>`.`<col_name>`.

    Returns:
        column name.
    """
    return full_name.split('.')[-1]


def table_name(full_name: str) -> str:
    r""""Given a fully qualified column name in the format of
    `<table_name>`.`<col_name>`, returns the table name.

    Args:
        full_name: Fully qualified column name in the format of
            `<table_name>`.`<col_name>`.

    Returns:
        table name.
    """
    if len(full_name.split('.')) < 2:
        raise ValueError(f'"{full_name}" object has no table name')
    return '.'.join(full_name.split('.')[:-1])


def fqn(table_name: str, col_name: str) -> str:
    r"""Given a table name and column name, returns a fully qualified column
    name as `<table_name>`.`<col_name>`.

    Args:
        table_name: A table name.
        col_name: A column name.

    Returns:
        Fully qualified column name in the format of
            `<table_name>`.`<col_name>`.
    """
    return f'{table_name}.{col_name}'


def merge(r1: ValidationResponse, r2: ValidationResponse) -> ValidationResponse:
    r"""Merges two validation responses.

    Args:
        r1: First validation response.
        r2: Second validation response.

    Returns:
        Merged validation response.
    """
    r1.warnings.extend(r2.warnings)
    r1.errors.extend(r2.errors)
    r1.info_items.extend(r2.info_items)
    return r1


def get_rhs_lp_table(
    query: ValidatedPredictiveQuery,
    graph: GraphDefinition,
) -> str:
    r"""Given a link prediction query, returns a fully qualified
    name `connector.table` for the RHS table.

    Args:
        query: The query.
        graph: Graph on which the query is defiend.
    """
    aggr = query.get_final_target_aggregation()
    assert isinstance(aggr, Aggregation)
    aggregated_col_name = aggr.get_target_column_name()
    edge_table = table_name(aggregated_col_name)
    edge_col = col_name(aggregated_col_name)
    for col_group in graph.col_groups:
        if ColumnKey(edge_table, edge_col) not in col_group.columns:
            continue
        # identify the primary key
        for col in col_group.columns:
            if graph.tables[col.table_name].pkey == col.col_name:
                return col.table_name
        raise AssertionError(f"col group {col_group} has no primary key")
    raise AssertionError(
        f"Foreign Key {aggregated_col_name} not part of any group")
