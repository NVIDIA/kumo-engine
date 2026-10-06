# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tempfile import _TemporaryFileWrapper
from typing import Callable

from kumoapi.pquery import ValidatedPredictiveQuery
from kumoapi.pquery.AST import (
    Aggregation,
    ASTNode,
    Column,
    Condition,
    DateOffsetRange,
    Filter,
    Join,
    LogicalOperation,
)
from kumoapi.pquery.AST.location_interval import ASTQueryLocationInterval
from kumoapi.typing import AggregationType, BoolOp, Dtype, RelOp, Stype
from kumopql.rewriting.filter_splitter import SplitFilter
from kumopql.rewriting.static_filter_splitter import StaticFilterSplitter

from kumo.graph.graph import Graph
from kumo.testing.pquery import get_validated_query

STATIC_JOIN_FILTER = Condition(
    target=Join(
        rhs_target=Column(
            fqn="LOCATIONS.NAME",
            dtype_maybe=Dtype.string,
            stype_maybe=Stype.categorical,
        ),
        rhs_key="LOCATIONS.LOCATION_ID",
        lhs_key="USERS.LOCATION_ID",
    ),
    op=RelOp.EQ,
    value="Pallet Town",
    dtype_maybe=Dtype.bool,
    stype_maybe=Stype.categorical,
)

TEMPORAL_COUNT_FILTER = Condition(
    target=Join(
        rhs_target=Aggregation(
            aggr=AggregationType.COUNT,
            target=Column(
                fqn="TRANSACTIONS.*",
                dtype_maybe=Dtype.int,
                stype_maybe=Stype.categorical,
            ),
            aggr_time_range=DateOffsetRange(
                start=None,  # None -> -inf
                end=0,
            ),
            dtype_maybe=Dtype.int,
            stype_maybe=Stype.numerical,
            group_by="TRANSACTIONS.USER_ID",
            time_col="TRANSACTIONS.DATE",
        ),
        rhs_key="TRANSACTIONS.USER_ID",
        lhs_key="USERS.USER_ID",
    ),
    op=RelOp.GT,
    value=0,
    dtype_maybe=Dtype.bool,
    stype_maybe=Stype.categorical,
)


def test_e2e_or(
    create_node_pred_graph: Callable[[str, str], Graph],
    tempdir: _TemporaryFileWrapper,
) -> None:
    graph = create_node_pred_graph('test_graph', tempdir.name)
    training_query = (
        "PREDICT LIST_DISTINCT(TRANSACTIONS.BANK_ID, 0, 1) RANK TOP 10 "
        "FOR EACH USERS.USER_ID "
        "WHERE COUNT(TRANSACTIONS.*, -inf, 0) > 0  "
        "OR LOCATIONS.NAME = \"Pallet Town\"")
    validated_training_query = get_validated_query(graph, training_query)

    splitter = StaticFilterSplitter()
    result = splitter.split(validated_training_query.entity_ast)
    assert not splitter.duplicates_inapplicable_nodes(result)
    assert isinstance(validated_training_query.entity_ast, Filter)
    expected = SplitFilter(
        applicable_filter=None,
        non_applicable_filter=LogicalOperation(
            left=TEMPORAL_COUNT_FILTER,
            bool_op=BoolOp.OR,
            right=STATIC_JOIN_FILTER,
            dtype_maybe=Dtype.bool,
            stype_maybe=Stype.categorical,
        ),
        rewritten=False,
        n_applicable_conditions=0,
        n_non_applicable_conditions=2,
        non_applicable_is_valid=True,
        original_filter=validated_training_query.entity_ast.condition,
    )
    _recursively_strip_location(result.applicable_filter)
    _recursively_strip_location(result.non_applicable_filter)
    _recursively_strip_location(expected.applicable_filter)
    _recursively_strip_location(expected.non_applicable_filter)
    assert result == expected


def test_e2e_temporal(
    create_node_pred_graph: Callable[[str, str], Graph],
    tempdir: _TemporaryFileWrapper,
) -> None:
    graph: Graph = create_node_pred_graph('test_graph', tempdir.name)
    training_query = (
        "PREDICT LIST_DISTINCT(TRANSACTIONS.BANK_ID, 0, 1) RANK TOP 10 "
        "FOR EACH USERS.USER_ID WHERE COUNT(TRANSACTIONS.*, -inf, 0) > 0")
    validated_training_query: ValidatedPredictiveQuery = (get_validated_query(
        graph, training_query))

    splitter = StaticFilterSplitter()
    result: SplitFilter = splitter.split(validated_training_query.entity_ast)
    assert not splitter.duplicates_inapplicable_nodes(result)
    assert isinstance(validated_training_query.entity_ast, Filter)
    expected = SplitFilter(
        applicable_filter=None,
        non_applicable_filter=TEMPORAL_COUNT_FILTER,
        rewritten=False,
        n_applicable_conditions=0,
        n_non_applicable_conditions=1,
        non_applicable_is_valid=True,
        original_filter=validated_training_query.entity_ast.condition,
    )
    _recursively_strip_location(result.applicable_filter)
    _recursively_strip_location(result.non_applicable_filter)
    _recursively_strip_location(expected.applicable_filter)
    _recursively_strip_location(expected.non_applicable_filter)
    assert result == expected


def test_e2e_mixed(
    create_node_pred_graph: Callable[[str, str], Graph],
    tempdir: _TemporaryFileWrapper,
) -> None:
    graph: Graph = create_node_pred_graph('test_graph', tempdir.name)
    training_query = (
        "PREDICT LIST_DISTINCT(TRANSACTIONS.BANK_ID, 0, 1) RANK TOP 10 "
        "FOR EACH USERS.USER_ID WHERE COUNT(TRANSACTIONS.*, -inf, 0) > 0 "
        "AND LOCATIONS.NAME = \"Pallet Town\"")
    validated_training_query: ValidatedPredictiveQuery = (get_validated_query(
        graph, training_query))

    splitter = StaticFilterSplitter()
    result: SplitFilter = splitter.split(validated_training_query.entity_ast)
    assert not splitter.duplicates_inapplicable_nodes(result)
    assert isinstance(validated_training_query.entity_ast, Filter)
    expected = SplitFilter(
        applicable_filter=STATIC_JOIN_FILTER,
        non_applicable_filter=TEMPORAL_COUNT_FILTER,
        rewritten=False,
        n_applicable_conditions=1,
        n_non_applicable_conditions=1,
        non_applicable_is_valid=True,
        original_filter=validated_training_query.entity_ast.condition,
    )
    _recursively_strip_location(result.applicable_filter)
    _recursively_strip_location(result.non_applicable_filter)
    _recursively_strip_location(expected.applicable_filter)
    _recursively_strip_location(expected.non_applicable_filter)
    assert result == expected


def test_e2e_static(
    create_node_pred_graph: Callable[[str, str], Graph],
    tempdir: _TemporaryFileWrapper,
) -> None:
    graph: Graph = create_node_pred_graph('test_graph', tempdir.name)
    training_query = (
        "PREDICT LIST_DISTINCT(TRANSACTIONS.BANK_ID, 0, 1) RANK TOP 10 "
        "FOR EACH USERS.USER_ID WHERE LOCATIONS.NAME = \"Pallet Town\"")
    validated_training_query: ValidatedPredictiveQuery = (get_validated_query(
        graph, training_query))

    splitter = StaticFilterSplitter()
    result: SplitFilter = splitter.split(validated_training_query.entity_ast)
    assert not splitter.duplicates_inapplicable_nodes(result)
    assert isinstance(validated_training_query.entity_ast, Filter)
    expected = SplitFilter(
        applicable_filter=STATIC_JOIN_FILTER,
        non_applicable_filter=None,
        rewritten=False,
        n_applicable_conditions=1,
        n_non_applicable_conditions=0,
        non_applicable_is_valid=True,
        original_filter=validated_training_query.entity_ast.condition,
    )
    _recursively_strip_location(result.applicable_filter)
    _recursively_strip_location(result.non_applicable_filter)
    _recursively_strip_location(expected.applicable_filter)
    _recursively_strip_location(expected.non_applicable_filter)
    assert result == expected


def _recursively_strip_location(ast: ASTNode | None) -> ASTNode | None:
    if ast is None:
        return None
    ast.location = ASTQueryLocationInterval(
        start_row=0,
        start_col=0,
        end_row=0,
        end_col=0,
        data_available=True,
    )
    for child in ast.children:
        _recursively_strip_location(child)
    return ast
