# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

from kumoapi.graph import GraphDefinition
from kumoapi.pquery import ValidatedPredictiveQuery
from kumoapi.pquery.AST import (
    Aggregation,
    Column,
    Condition,
    DateOffsetRange,
    Filter,
    LogicalOperation,
)
from kumoapi.task import TaskType
from kumoapi.typing import AggregationType, BoolOp, ProblemType, TimeUnit
from kumopql.parser import PQLParser
from kumopql.rewriting.bp_filter_transformer import (
    BPFilterTransformer,
    EditableQueryResponse,
)
from kumopql.validator import PredictiveQueryValidator


def get_validated_query(graph: GraphDefinition,
                        query: str) -> ValidatedPredictiveQuery:
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert response.ok, response
    assert validated_query is not None
    return validated_query


def test_bp_query_no_filters(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    query = ('PREDICT LIST_DISTINCT(TRANS.STORE, 0, 10) '
             'RANK TOP 10 FOR EACH USERS.USERS_ID')
    validated_query = get_validated_query(graph, query)
    transformer = BPFilterTransformer()
    transformed_query = transformer.generate_bp_query(
        validated_query, TaskType.TEMPORAL_LINK_PREDICTION, graph)
    assert str(transformed_query.target_ast) == (
        'LIST_DISTINCT(TRANS.STORE, 0, 10, days)')


def test_get_bp_editable_query_forecast(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    query = ('PREDICT SUM(TRANS.CNT, 0, 7) FORECAST 4 TIMEFRAMES '
             'FOR EACH USERS.USERS_ID')
    validated_query = get_validated_query(graph, query)
    assert validated_query.problem_type == ProblemType.FORECAST
    assert validated_query.num_forecasts == 4
    transformer = BPFilterTransformer()
    editable_query = transformer.get_bp_editable_query(validated_query,
                                                       TaskType.REGRESSION,
                                                       graph, '')
    assert editable_query.text == ('PREDICT SUM(TRANS.CNT, 0, 7, days)\n'
                                   'FORECAST 4 TIMEFRAMES\n'
                                   'FOR EACH USERS.USERS_ID')


def test_split_condition_edge() -> None:
    transformer = BPFilterTransformer()
    condition = Condition(target=Column(fqn='edge.col'), op='=', value=1)
    column = Column(fqn='edge.rhs_key')
    filter = Filter(target=column, condition=condition)
    split_filter = transformer.split_target_filter(filter, 'lhs_table',
                                                   'rhs_table')
    assert split_filter.applicable_filter is None
    assert str(split_filter.non_applicable_filter) == 'edge.col = 1'
    assert not split_filter.rewritten
    assert split_filter.n_applicable_conditions == 0
    assert split_filter.n_non_applicable_conditions == 1
    assert split_filter.non_applicable_is_valid


def test_split_condition_aggregation() -> None:
    transformer = BPFilterTransformer()
    condition = Condition(
        target=Aggregation(
            target=Column(fqn='rhs_table.col'), aggr=AggregationType.SUM,
            aggr_time_range=DateOffsetRange(0, 5, TimeUnit.DAYS)), op='>=',
        value=1)
    split_filter = transformer._recursively_split_filter(
        condition, condition, lhs_table='lhs_table', rhs_table='rhs_table')
    assert split_filter.applicable_filter is None
    assert str(split_filter.non_applicable_filter) == (
        'SUM(rhs_table.col, 0, 5, days) >= 1')
    assert not split_filter.rewritten
    assert split_filter.n_applicable_conditions == 0
    assert split_filter.n_non_applicable_conditions == 1
    assert split_filter.non_applicable_is_valid


def test_split_and_left_rhs() -> None:
    transformer = BPFilterTransformer()
    condition = LogicalOperation(
        left=Condition(target=Column(fqn='edge.col'), op='=', value=1),
        bool_op=BoolOp.AND, right=Condition(target=Column(fqn='rhs_table.col'),
                                            op='=', value=1))
    split_filter = transformer._recursively_split_filter(
        condition, condition, lhs_table='lhs_table', rhs_table='rhs_table')
    assert str(split_filter.applicable_filter) == 'rhs_table.col = 1'
    assert str(split_filter.non_applicable_filter) == 'edge.col = 1'
    assert not split_filter.rewritten
    assert split_filter.n_applicable_conditions == 1
    assert split_filter.n_non_applicable_conditions == 1
    assert split_filter.non_applicable_is_valid


def test_split_and_right_rhs() -> None:
    transformer = BPFilterTransformer()
    condition = LogicalOperation(
        left=Condition(target=Column(fqn='rhs_table.col'), op='=', value=1),
        bool_op=BoolOp.AND, right=Condition(target=Column(fqn='edge.col'),
                                            op='=', value=1))
    split_filter = transformer._recursively_split_filter(
        condition, condition, lhs_table='lhs_table', rhs_table='rhs_table')
    assert str(split_filter.applicable_filter) == 'rhs_table.col = 1'
    assert str(split_filter.non_applicable_filter) == 'edge.col = 1'
    assert not split_filter.rewritten
    assert split_filter.n_applicable_conditions == 1
    assert split_filter.n_non_applicable_conditions == 1
    assert split_filter.non_applicable_is_valid


def test_split_and_both_rhs() -> None:
    transformer = BPFilterTransformer()
    condition = LogicalOperation(
        left=Condition(target=Column(fqn='rhs_table.col'), op='=', value=1),
        bool_op=BoolOp.AND, right=Condition(target=Column(fqn='rhs_table.col'),
                                            op='=', value=1))
    split_filter = transformer._recursively_split_filter(
        condition, condition, lhs_table='lhs_table', rhs_table='rhs_table')
    assert str(split_filter.applicable_filter) == (
        '(rhs_table.col = 1) AND (rhs_table.col = 1)')
    assert split_filter.non_applicable_filter is None
    assert not split_filter.rewritten
    assert split_filter.n_applicable_conditions == 2
    assert split_filter.n_non_applicable_conditions == 0
    assert split_filter.non_applicable_is_valid


def test_split_and_neither_rhs() -> None:
    transformer = BPFilterTransformer()
    condition = LogicalOperation(
        left=Condition(target=Column(fqn='edge.col'), op='=', value=1),
        bool_op=BoolOp.AND, right=Condition(target=Column(fqn='edge.col'),
                                            op='=', value=1))
    split_filter = transformer._recursively_split_filter(
        condition, condition, lhs_table='lhs_table', rhs_table='rhs_table')
    assert split_filter.applicable_filter is None
    assert str(split_filter.non_applicable_filter) == (
        '(edge.col = 1) AND (edge.col = 1)')
    assert not split_filter.rewritten
    assert split_filter.n_applicable_conditions == 0
    assert split_filter.n_non_applicable_conditions == 2
    assert split_filter.non_applicable_is_valid


def test_split_or_left_rhs() -> None:
    transformer = BPFilterTransformer()
    condition = LogicalOperation(
        left=Condition(target=Column(fqn='rhs_table.col'), op='=', value=1),
        bool_op=BoolOp.OR, right=Condition(target=Column(fqn='edge.col'),
                                           op='=', value=1))
    split_filter = transformer._recursively_split_filter(
        condition, condition, lhs_table='lhs_table', rhs_table='rhs_table')
    assert split_filter.applicable_filter is None
    assert str(split_filter.non_applicable_filter) == (
        '(rhs_table.col = 1) OR (edge.col = 1)')
    assert not split_filter.rewritten
    assert split_filter.n_applicable_conditions == 0
    assert split_filter.n_non_applicable_conditions == 2
    assert split_filter.non_applicable_is_valid


def test_split_or_right_rhs() -> None:
    transformer = BPFilterTransformer()
    condition = LogicalOperation(
        left=Condition(target=Column(fqn='edge.col'), op='=', value=1),
        bool_op=BoolOp.OR, right=Condition(target=Column(fqn='rhs_table.col'),
                                           op='=', value=1))
    split_filter = transformer._recursively_split_filter(
        condition, condition, lhs_table='lhs_table', rhs_table='rhs_table')
    assert split_filter.applicable_filter is None
    assert str(split_filter.non_applicable_filter) == (
        '(edge.col = 1) OR (rhs_table.col = 1)')
    assert not split_filter.rewritten
    assert split_filter.n_applicable_conditions == 0
    assert split_filter.n_non_applicable_conditions == 2
    assert split_filter.non_applicable_is_valid


def test_split_or_both_rhs() -> None:
    transformer = BPFilterTransformer()
    condition = LogicalOperation(
        left=Condition(target=Column(fqn='rhs_table.col'), op='=', value=1),
        bool_op=BoolOp.OR, right=Condition(target=Column(fqn='rhs_table.col'),
                                           op='=', value=1))
    split_filter = transformer._recursively_split_filter(
        condition, condition, lhs_table='lhs_table', rhs_table='rhs_table')
    assert str(split_filter.applicable_filter) == (
        '(rhs_table.col = 1) OR (rhs_table.col = 1)')
    assert split_filter.non_applicable_filter is None
    assert not split_filter.rewritten
    assert split_filter.n_applicable_conditions == 2
    assert split_filter.n_non_applicable_conditions == 0
    assert split_filter.non_applicable_is_valid


def test_split_or_neither_rhs() -> None:
    transformer = BPFilterTransformer()
    condition = LogicalOperation(
        left=Condition(target=Column(fqn='edge.col'), op='=', value=1),
        bool_op=BoolOp.OR, right=Condition(target=Column(fqn='edge.col'),
                                           op='=', value=1))
    split_filter = transformer._recursively_split_filter(
        condition, condition, lhs_table='lhs_table', rhs_table='rhs_table')
    assert split_filter.applicable_filter is None
    assert str(split_filter.non_applicable_filter) == (
        '(edge.col = 1) OR (edge.col = 1)')
    assert not split_filter.rewritten
    assert split_filter.n_applicable_conditions == 0
    assert split_filter.n_non_applicable_conditions == 2
    assert split_filter.non_applicable_is_valid


def test_split_nested_not() -> None:
    transformer = BPFilterTransformer()
    condition = LogicalOperation(
        left=LogicalOperation(
            left=Condition(target=Column(fqn='rhs_table.col'), op='=',
                           value=1), bool_op=BoolOp.NOT), bool_op=BoolOp.NOT)
    split_filter = transformer._recursively_split_filter(
        condition, condition, lhs_table='lhs_table', rhs_table='rhs_table')
    assert str(split_filter.applicable_filter) == 'rhs_table.col = 1'
    assert split_filter.non_applicable_filter is None
    assert split_filter.rewritten
    assert split_filter.n_applicable_conditions == 1
    assert split_filter.n_non_applicable_conditions == 0
    assert split_filter.non_applicable_is_valid


def test_split_not_and_both_rhs() -> None:
    transformer = BPFilterTransformer()
    condition = LogicalOperation(
        left=LogicalOperation(
            left=Condition(target=Column(fqn='rhs_table.col'), op='=',
                           value=1), bool_op=BoolOp.AND,
            right=Condition(target=Column(fqn='rhs_table.col'), op='=',
                            value=1)), bool_op=BoolOp.NOT)
    split_filter = transformer._recursively_split_filter(
        condition, condition, lhs_table='lhs_table', rhs_table='rhs_table')
    assert str(split_filter.applicable_filter) == (
        '(NOT (rhs_table.col = 1)) OR (NOT (rhs_table.col = 1))')
    assert split_filter.non_applicable_filter is None
    assert split_filter.rewritten
    assert split_filter.n_applicable_conditions == 2
    assert split_filter.n_non_applicable_conditions == 0
    assert split_filter.non_applicable_is_valid


def test_split_not_and_single_rhs() -> None:
    transformer = BPFilterTransformer()
    condition = LogicalOperation(
        left=LogicalOperation(
            left=Condition(target=Column(fqn='edge.col'), op='=',
                           value=1), bool_op=BoolOp.AND,
            right=Condition(target=Column(fqn='rhs_table.col'), op='=',
                            value=1)), bool_op=BoolOp.NOT)
    split_filter = transformer._recursively_split_filter(
        condition, condition, lhs_table='lhs_table', rhs_table='rhs_table')
    assert split_filter.applicable_filter is None
    assert str(split_filter.non_applicable_filter) == (
        '(NOT (edge.col = 1)) OR (NOT (rhs_table.col = 1))')
    assert split_filter.rewritten
    assert split_filter.n_applicable_conditions == 0
    assert split_filter.n_non_applicable_conditions == 2
    assert split_filter.non_applicable_is_valid


def test_split_not_or_both_rhs() -> None:
    transformer = BPFilterTransformer()
    condition = LogicalOperation(
        left=LogicalOperation(
            left=Condition(target=Column(fqn='rhs_table.col'), op='=',
                           value=1), bool_op=BoolOp.OR,
            right=Condition(target=Column(fqn='rhs_table.col'), op='=',
                            value=1)), bool_op=BoolOp.NOT)
    split_filter = transformer._recursively_split_filter(
        condition, condition, lhs_table='lhs_table', rhs_table='rhs_table')
    assert str(split_filter.applicable_filter) == (
        '(NOT (rhs_table.col = 1)) AND (NOT (rhs_table.col = 1))')
    assert split_filter.non_applicable_filter is None
    assert split_filter.rewritten
    assert split_filter.n_applicable_conditions == 2
    assert split_filter.n_non_applicable_conditions == 0
    assert split_filter.non_applicable_is_valid


def test_split_not_or_single_rhs() -> None:
    transformer = BPFilterTransformer()
    condition = LogicalOperation(
        left=LogicalOperation(
            left=Condition(target=Column(fqn='edge.col'), op='=',
                           value=1), bool_op=BoolOp.OR,
            right=Condition(target=Column(fqn='rhs_table.col'), op='=',
                            value=1)), bool_op=BoolOp.NOT)
    split_filter = transformer._recursively_split_filter(
        condition, condition, lhs_table='lhs_table', rhs_table='rhs_table')
    assert str(split_filter.applicable_filter) == 'NOT (rhs_table.col = 1)'
    assert str(split_filter.non_applicable_filter) == 'NOT (edge.col = 1)'
    assert split_filter.rewritten
    assert split_filter.n_applicable_conditions == 1
    assert split_filter.n_non_applicable_conditions == 1
    assert split_filter.non_applicable_is_valid


def test_split_not_or_no_rhs() -> None:
    transformer = BPFilterTransformer()
    condition = LogicalOperation(
        left=LogicalOperation(
            left=Condition(target=Column(fqn='edge.col'), op='=', value=1),
            bool_op=BoolOp.OR, right=Condition(target=Column(fqn='edge.col'),
                                               op='=', value=1)),
        bool_op=BoolOp.NOT)
    split_filter = transformer._recursively_split_filter(
        condition, condition, lhs_table='lhs_table', rhs_table='rhs_table')
    assert split_filter.applicable_filter is None
    assert str(split_filter.non_applicable_filter) == (
        '(NOT (edge.col = 1)) AND (NOT (edge.col = 1))')
    assert split_filter.rewritten
    assert split_filter.n_applicable_conditions == 0
    assert split_filter.n_non_applicable_conditions == 2
    assert split_filter.non_applicable_is_valid


def test_split_complex_or() -> None:
    transformer = BPFilterTransformer()
    condition = LogicalOperation(
        left=LogicalOperation(
            left=Condition(target=Column(fqn='edge.col'), op='=',
                           value=1), bool_op=BoolOp.AND,
            right=Condition(target=Column(fqn='rhs_table.col'), op='=',
                            value=1)), bool_op=BoolOp.OR,
        right=LogicalOperation(
            left=Condition(target=Column(fqn='edge.col'), op='=',
                           value=2), bool_op=BoolOp.AND,
            right=Condition(target=Column(fqn='rhs_table.col'), op='=',
                            value=2)))
    split_filter = transformer._recursively_split_filter(
        condition, condition, lhs_table='lhs_table', rhs_table='rhs_table')
    assert str(split_filter.applicable_filter) == (
        '(rhs_table.col = 1) OR (rhs_table.col = 2)')
    assert str(split_filter.non_applicable_filter) == (
        '(((edge.col = 1) OR (edge.col = 2)) AND '
        '((edge.col = 1) OR (rhs_table.col = 2))) AND '
        '((rhs_table.col = 1) OR (edge.col = 2))')
    assert split_filter.rewritten
    assert split_filter.n_applicable_conditions == 2
    assert split_filter.n_non_applicable_conditions == 6
    assert split_filter.non_applicable_is_valid


def test_split_exponential_blowup() -> None:
    # condition of type "(x1 and y1) or ... or (xn and yn)" creates 2^n terms
    # when rewritten into DNF. But only one of them "x1 or .. or xn" is
    # applicable. The system must handle this correctly and without hiccups
    n_terms = 40
    transformer = BPFilterTransformer()
    expected_applicable_condition = 'rhs_table.col = 0'
    condition = LogicalOperation(
        left=Condition(target=Column(fqn='edge.col'), op='=', value=0),
        bool_op=BoolOp.AND, right=Condition(target=Column(fqn='rhs_table.col'),
                                            op='=', value=0))
    for i in range(1, n_terms):
        expected_applicable_condition = (
            f'({expected_applicable_condition}) OR (rhs_table.col = {i})')
        condition = LogicalOperation(
            left=condition, bool_op=BoolOp.OR, right=LogicalOperation(
                left=Condition(target=Column(fqn='edge.col'), op='=',
                               value=i), bool_op=BoolOp.AND,
                right=Condition(target=Column(fqn='rhs_table.col'), op='=',
                                value=i)))
    split_filter = transformer._recursively_split_filter(
        condition, condition, lhs_table='lhs_table', rhs_table='rhs_table')
    assert str(split_filter.applicable_filter) == expected_applicable_condition
    assert split_filter.non_applicable_filter is None
    assert split_filter.rewritten
    assert split_filter.n_applicable_conditions == n_terms
    # order of magnitude 10**12 - would be OOM if in memory
    assert split_filter.n_non_applicable_conditions > 6
    assert not split_filter.non_applicable_is_valid


def test_backtesting_node_prediction(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    training_query = (
        "PREDICT SUM(TRANS.VALUE "
        "WHERE TRANS.TYPE = 'A', 0, 7) "
        "FOR EACH USERS.USERS_ID WHERE LOCATIONS.NAME = \"Pallet Town\" "
        "ASSUMING COUNT(TRANS.*, 0, 3) > 0")
    validated_training_query = get_validated_query(graph, training_query)
    prediction_query = ("PREDICT SUM(TRANS.VALUE "
                        "WHERE TRANS.TYPE = \'A\', 0, 7) "
                        "FOR EACH USERS.USERS_ID WHERE USERS.AGE > 20")
    validated_prediction_query = get_validated_query(graph, prediction_query)
    transformer = BPFilterTransformer()
    backtesting_query = transformer.generate_backtesting_query(
        validated_training_query, validated_prediction_query,
        TaskType.REGRESSION, graph)
    assert str(backtesting_query.entity_ast) == (
        'USERS.USERS_ID WHERE USERS.AGE > 20')
    assert str(backtesting_query.target_ast) == (
        'SUM(TRANS.VALUE WHERE TRANS.TYPE = \'A\', 0, 7, days)')
    assert str(
        backtesting_query.whatif_ast) == ('COUNT(TRANS.*, 0, 3, days) > 0')


def test_backtesting_link_prediction_no_target_overrides(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    training_query = (
        "PREDICT LIST_DISTINCT(TRANS.STORE "
        "WHERE TRANS.TYPE = 'A' AND STORES.TYPE = 'A', 0, 7) "
        "RANK TOP 10 "
        "FOR EACH USERS.USERS_ID WHERE LOCATIONS.NAME = \"Pallet Town\" "
        "ASSUMING COUNT(TRANS.*, 0, 3) > 0")
    validated_training_query = get_validated_query(graph, training_query)
    prediction_query = ("PREDICT LIST_DISTINCT(TRANS.STORE "
                        "WHERE STORES.TYPE = 'A', 0, 7) RANK TOP 5 "
                        "FOR EACH USERS.USERS_ID WHERE USERS.AGE > 20")
    validated_prediction_query = get_validated_query(graph, prediction_query)
    transformer = BPFilterTransformer()
    backtesting_query = transformer.generate_backtesting_query(
        validated_training_query, validated_prediction_query,
        TaskType.TEMPORAL_LINK_PREDICTION, graph)
    assert str(backtesting_query.entity_ast) == (
        'USERS.USERS_ID WHERE USERS.AGE > 20')
    assert str(backtesting_query.target_ast) == (
        "LIST_DISTINCT(TRANS.STORE "
        "WHERE (TRANS.TYPE = 'A') AND (STORES.TYPE = 'A')"
        ", 0, 7, days)")
    assert str(
        backtesting_query.whatif_ast) == ('COUNT(TRANS.*, 0, 3, days) > 0')
    assert backtesting_query.top_k == 5
    assert backtesting_query.problem_type == ProblemType.RANK


def test_backtesting_link_prediction_deleted_target_filter(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    training_query = ("PREDICT LIST_DISTINCT(TRANS.STORE "
                      "WHERE STORES.TYPE = 'A', 0, 7) RANK TOP 10 "
                      "FOR EACH USERS.USERS_ID")
    validated_training_query = get_validated_query(graph, training_query)
    prediction_query = ("PREDICT LIST_DISTINCT(TRANS.STORE, 0, 7) "
                        "RANK TOP 10 FOR EACH USERS.USERS_ID")
    validated_prediction_query = get_validated_query(graph, prediction_query)
    transformer = BPFilterTransformer()
    backtesting_query = transformer.generate_backtesting_query(
        validated_training_query, validated_prediction_query,
        TaskType.TEMPORAL_LINK_PREDICTION, graph)
    assert str(backtesting_query.entity_ast) == 'USERS.USERS_ID'
    assert str(backtesting_query.target_ast) == (
        "LIST_DISTINCT(TRANS.STORE, 0, 7, days)")
    assert backtesting_query.whatif_ast is None


def test_backtesting_link_prediction_only_training_filter(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    training_query = ("PREDICT LIST_DISTINCT(TRANS.STORE "
                      "WHERE TRANS.TYPE = 'A', 0, 7) "
                      "RANK TOP 10 FOR EACH USERS.USERS_ID")
    validated_training_query = get_validated_query(graph, training_query)
    prediction_query = ("PREDICT LIST_DISTINCT(TRANS.STORE, 0, 7) "
                        "RANK TOP 10 FOR EACH USERS.USERS_ID")
    validated_prediction_query = get_validated_query(graph, prediction_query)
    transformer = BPFilterTransformer()
    backtesting_query = transformer.generate_backtesting_query(
        validated_training_query, validated_prediction_query,
        TaskType.TEMPORAL_LINK_PREDICTION, graph)
    assert str(backtesting_query.entity_ast) == 'USERS.USERS_ID'
    assert str(backtesting_query.target_ast) == (
        "LIST_DISTINCT(TRANS.STORE "
        "WHERE TRANS.TYPE = 'A', 0, 7, days)")
    assert backtesting_query.whatif_ast is None


def test_backtesting_link_prediction_only_prediction_filter(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    training_query = ("PREDICT LIST_DISTINCT(TRANS.STORE, 0, 7) "
                      "RANK TOP 10 FOR EACH USERS.USERS_ID")
    validated_training_query = get_validated_query(graph, training_query)
    prediction_query = ("PREDICT LIST_DISTINCT(TRANS.STORE "
                        "WHERE STORES.TYPE = 'A', 0, 7) "
                        "RANK TOP 10 FOR EACH USERS.USERS_ID")
    validated_prediction_query = get_validated_query(graph, prediction_query)
    transformer = BPFilterTransformer()
    backtesting_query = transformer.generate_backtesting_query(
        validated_training_query, validated_prediction_query,
        TaskType.TEMPORAL_LINK_PREDICTION, graph)
    assert str(backtesting_query.entity_ast) == 'USERS.USERS_ID'
    assert str(backtesting_query.target_ast) == (
        "LIST_DISTINCT(TRANS.STORE "
        "WHERE STORES.TYPE = 'A', 0, 7, days)")
    assert backtesting_query.whatif_ast is None


def test_backtesting_link_prediction_combine_filters(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    training_query = ("PREDICT LIST_DISTINCT(TRANS.STORE "
                      "WHERE STORES.TYPE = 'A' AND TRANS.TYPE = 'A', 0, 7) "
                      "RANK TOP 10 FOR EACH USERS.USERS_ID")
    validated_training_query = get_validated_query(graph, training_query)
    prediction_query = ("PREDICT LIST_DISTINCT(TRANS.STORE "
                        "WHERE STORES.TYPE = 'B', 0, 7) "
                        "RANK TOP 10 FOR EACH USERS.USERS_ID")
    validated_prediction_query = get_validated_query(graph, prediction_query)
    transformer = BPFilterTransformer()
    backtesting_query = transformer.generate_backtesting_query(
        validated_training_query, validated_prediction_query,
        TaskType.TEMPORAL_LINK_PREDICTION, graph)
    assert str(backtesting_query.entity_ast) == 'USERS.USERS_ID'
    assert str(backtesting_query.target_ast) == (
        "LIST_DISTINCT(TRANS.STORE "
        "WHERE (TRANS.TYPE = 'A') AND (STORES.TYPE = 'B')"
        ", 0, 7, days)")
    assert backtesting_query.whatif_ast is None


def get_substring(text: str, position: tuple[int, int]) -> str:
    return text[position[0]:position[1]]


def test_get_bp_editable_query_no_filter(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    query = ("PREDICT COUNT(TRANS.*, 0, 7) "
             "FOR EACH USERS.USERS_ID ")
    validated_query = get_validated_query(graph, query)
    transformer = BPFilterTransformer()
    editable_query = transformer.get_bp_editable_query(validated_query,
                                                       TaskType.REGRESSION,
                                                       graph, '')
    assert editable_query.text == ("PREDICT COUNT(TRANS.*, 0, 7, days) "
                                   "FOR EACH USERS.USERS_ID")
    entity_filter_field = editable_query.editable_fields[0]
    assert entity_filter_field.field_name == 'Entity filter'
    assert entity_filter_field.text == ''
    assert get_substring(editable_query.text,
                         entity_filter_field.position) == ''
    assert get_substring(editable_query.text,
                         entity_filter_field.bold_text_position) == ''
    assert entity_filter_field.full_filter_format_string == ' WHERE {0}'
    assert entity_filter_field.empty_filter_default == ''


def test_get_bp_editable_query_entity_filter(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    query = ("PREDICT COUNT(TRANS.* WHERE TRANS.TYPE = 'A', 0, 7) "
             "FOR EACH USERS.USERS_ID WHERE USERS.AGE > 20")
    validated_query = get_validated_query(graph, query)
    transformer = BPFilterTransformer()
    editable_query = transformer.get_bp_editable_query(validated_query,
                                                       TaskType.REGRESSION,
                                                       graph, '')
    assert editable_query.text == (
        "PREDICT COUNT(TRANS.* "
        "WHERE TRANS.TYPE = 'A', 0, 7, days) "
        "FOR EACH USERS.USERS_ID WHERE USERS.AGE > 20")
    entity_filter_field = editable_query.editable_fields[0]
    assert entity_filter_field.field_name == 'Entity filter'
    assert entity_filter_field.text == 'USERS.AGE > 20'
    assert get_substring(
        editable_query.text,
        entity_filter_field.position) == ' WHERE ' + entity_filter_field.text
    assert get_substring(
        editable_query.text,
        entity_filter_field.bold_text_position) == 'USERS.AGE > 20'
    assert entity_filter_field.full_filter_format_string == ' WHERE {0}'
    assert entity_filter_field.empty_filter_default == ''


def test_get_bp_editable_query_no_filter_lp(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    query = ("PREDICT LIST_DISTINCT(TRANS.STORE, 0, 7) "
             "RANK TOP 10 FOR EACH USERS.USERS_ID ")
    validated_query = get_validated_query(graph, query)
    transformer = BPFilterTransformer()
    editable_query = transformer.get_bp_editable_query(
        validated_query, TaskType.TEMPORAL_LINK_PREDICTION, graph, '')
    assert editable_query.text == (
        "PREDICT LIST_DISTINCT(TRANS.STORE, 0, 7, days) "
        "RANK TOP 10 FOR EACH USERS.USERS_ID")
    entity_filter_field = editable_query.editable_fields[1]
    assert entity_filter_field.field_name == 'Entity filter'
    assert entity_filter_field.text == ''
    assert get_substring(editable_query.text,
                         entity_filter_field.position) == ''
    assert get_substring(editable_query.text,
                         entity_filter_field.bold_text_position) == ''
    assert entity_filter_field.full_filter_format_string == ' WHERE {0}'
    assert entity_filter_field.empty_filter_default == ''

    target_filter_field = editable_query.editable_fields[0]
    assert target_filter_field.field_name == 'Target filter'
    assert target_filter_field.text == ''
    assert get_substring(editable_query.text,
                         target_filter_field.position) == ''
    assert get_substring(editable_query.text,
                         target_filter_field.bold_text_position) == ''
    assert target_filter_field.full_filter_format_string == (' WHERE {0}')
    assert target_filter_field.empty_filter_default == ''


def test_get_bp_editable_query_lp_complex_target_filter(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    query = ("PREDICT LIST_DISTINCT(TRANS.STORE "
             "WHERE NOT(STORES.TYPE = 'A' OR TRANS.TYPE = 'A'), 0, 7) "
             "RANK TOP 10 FOR EACH USERS.USERS_ID "
             "WHERE USERS.AGE > 20")
    validated_query = get_validated_query(graph, query)
    transformer = BPFilterTransformer()
    editable_query = transformer.get_bp_editable_query(
        validated_query, TaskType.TEMPORAL_LINK_PREDICTION, graph, '')
    assert editable_query.text == ("PREDICT LIST_DISTINCT(TRANS.STORE "
                                   "WHERE (NOT (TRANS.TYPE = 'A')) AND "
                                   "(NOT (STORES.TYPE = 'A')), 0, 7, days) "
                                   "RANK TOP 10 FOR EACH USERS.USERS_ID "
                                   "WHERE USERS.AGE > 20")
    assert ('bolded conditions are all met'
            in editable_query.validation_response.message())
    assert ('logical expression has been rewritten'
            in editable_query.validation_response.message())
    entity_filter_field = editable_query.editable_fields[1]
    assert entity_filter_field.field_name == 'Entity filter'
    assert entity_filter_field.text == 'USERS.AGE > 20'
    assert get_substring(
        editable_query.text,
        entity_filter_field.position) == (' WHERE ' + entity_filter_field.text)
    assert get_substring(
        editable_query.text,
        entity_filter_field.bold_text_position) == entity_filter_field.text
    assert entity_filter_field.full_filter_format_string == ' WHERE {0}'
    assert entity_filter_field.empty_filter_default == ''

    target_filter_field = editable_query.editable_fields[0]
    assert target_filter_field.field_name == 'Target filter'
    assert target_filter_field.text == 'NOT (STORES.TYPE = \'A\')'
    assert get_substring(editable_query.text,
                         target_filter_field.position) == (
                             ' WHERE (NOT (TRANS.TYPE = \'A\')) AND '
                             '(NOT (STORES.TYPE = \'A\'))')
    assert get_substring(
        editable_query.text,
        target_filter_field.bold_text_position) == target_filter_field.text
    assert target_filter_field.full_filter_format_string == (
        ' WHERE (NOT (TRANS.TYPE = \'A\')) AND ({0})')
    assert target_filter_field.empty_filter_default == (
        ' WHERE NOT (TRANS.TYPE = \'A\')')


def test_get_bp_editable_query_lp_only_applicable_target_filter(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    query = ("PREDICT LIST_DISTINCT(TRANS.STORE "
             "WHERE NOT(STORES.TYPE = 'A'), 0, 7) "
             "RANK TOP 10 FOR EACH USERS.USERS_ID")
    validated_query = get_validated_query(graph, query)
    transformer = BPFilterTransformer()
    editable_query = transformer.get_bp_editable_query(
        validated_query, TaskType.TEMPORAL_LINK_PREDICTION, graph, '')
    assert editable_query.text == (
        "PREDICT LIST_DISTINCT(TRANS.STORE "
        "WHERE NOT (STORES.TYPE = 'A'), 0, 7, days) "
        "RANK TOP 10 FOR EACH USERS.USERS_ID")
    assert ('bolded conditions are met'
            in editable_query.validation_response.message())
    entity_filter_field = editable_query.editable_fields[1]
    assert entity_filter_field.field_name == 'Entity filter'
    assert entity_filter_field.text == ''
    assert get_substring(editable_query.text,
                         entity_filter_field.position) == ''
    assert get_substring(editable_query.text,
                         entity_filter_field.bold_text_position) == ''
    assert entity_filter_field.full_filter_format_string == ' WHERE {0}'
    assert entity_filter_field.empty_filter_default == ''

    target_filter_field = editable_query.editable_fields[0]
    assert target_filter_field.field_name == 'Target filter'
    assert target_filter_field.text == 'NOT (STORES.TYPE = \'A\')'
    assert get_substring(
        editable_query.text,
        target_filter_field.position) == (' WHERE NOT (STORES.TYPE = \'A\')')
    assert get_substring(
        editable_query.text,
        target_filter_field.bold_text_position) == target_filter_field.text
    assert target_filter_field.full_filter_format_string == (' WHERE {0}')
    assert target_filter_field.empty_filter_default == ''


def test_get_bp_editable_query_lp_non_applicable_target_filter(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    query = ("PREDICT LIST_DISTINCT(TRANS.STORE "
             "WHERE NOT(TRANS.TYPE = 'A'), 0, 7) "
             "RANK TOP 10 FOR EACH USERS.USERS_ID")
    validated_query = get_validated_query(graph, query)
    transformer = BPFilterTransformer()
    editable_query = transformer.get_bp_editable_query(
        validated_query, TaskType.TEMPORAL_LINK_PREDICTION, graph, '')
    assert editable_query.text == ("PREDICT LIST_DISTINCT(TRANS.STORE "
                                   "WHERE NOT (TRANS.TYPE = 'A'), 0, 7, days) "
                                   "RANK TOP 10 FOR EACH USERS.USERS_ID")
    entity_filter_field = editable_query.editable_fields[1]
    assert entity_filter_field.field_name == 'Entity filter'
    assert entity_filter_field.text == ''
    assert get_substring(editable_query.text,
                         entity_filter_field.position) == ''
    assert get_substring(editable_query.text,
                         entity_filter_field.bold_text_position) == ''
    assert entity_filter_field.full_filter_format_string == ' WHERE {0}'
    assert entity_filter_field.empty_filter_default == ''

    target_filter_field = editable_query.editable_fields[0]
    assert target_filter_field.field_name == 'Target filter'
    assert target_filter_field.text == ''
    assert get_substring(
        editable_query.text,
        target_filter_field.position) == (' WHERE NOT (TRANS.TYPE = \'A\')')
    assert get_substring(editable_query.text,
                         target_filter_field.bold_text_position) == ''
    assert target_filter_field.full_filter_format_string == (
        ' WHERE (NOT (TRANS.TYPE = \'A\')) AND ({0})')
    assert target_filter_field.empty_filter_default == (
        ' WHERE NOT (TRANS.TYPE = \'A\')')


def test_get_bp_editable_query_lp_large_non_applicable_target_filter(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    query = ("PREDICT LIST_DISTINCT(TRANS.STORE "
             "WHERE (TRANS.TYPE = \'A\' AND STORES.TYPE = 'A') "
             "OR (TRANS.TYPE = 'A' AND STORES.TYPE = 'B') "
             "OR (TRANS.TYPE = 'B' AND STORES.TYPE = 'A'), 0, 7) "
             "RANK TOP 10 FOR EACH USERS.USERS_ID")
    validated_query = get_validated_query(graph, query)
    transformer = BPFilterTransformer()
    editable_query = transformer.get_bp_editable_query(
        validated_query, TaskType.TEMPORAL_LINK_PREDICTION, graph, '')
    # There should be no target filter
    entity_filter_field = editable_query.editable_fields[0]
    assert entity_filter_field.field_name == 'Entity filter'
    assert entity_filter_field.text == ''
    assert get_substring(editable_query.text,
                         entity_filter_field.position) == ''
    assert get_substring(editable_query.text,
                         entity_filter_field.bold_text_position) == ''
    assert entity_filter_field.full_filter_format_string == ' WHERE {0}'
    assert entity_filter_field.empty_filter_default == ''
    assert ('size of the target filter grew too large'
            in editable_query.validation_response.message())


def test_get_bp_edit_all_filters(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    query = ("PREDICT LIST_DISTINCT(TRANS.STORE "
             "WHERE NOT(STORES.TYPE = 'A' OR TRANS.TYPE = 'A'), 0, 7) "
             "RANK TOP 10 FOR EACH USERS.USERS_ID "
             "WHERE USERS.AGE > 20")
    validated_query = get_validated_query(graph, query)
    transformer = BPFilterTransformer()
    editable_query = transformer.get_bp_editable_query(
        validated_query, TaskType.TEMPORAL_LINK_PREDICTION, graph, '')
    user_edits = {
        'Entity filter': 'USERS.AGE < 21 AND USERS.AGE > 10',
        'Target filter': 'STORES.TYPE = "B"'
    }
    user_response = EditableQueryResponse(editable_query, user_edits)
    editable_query, applicable_query = transformer.insert_filter_overrides(
        user_response, graph, TaskType.TEMPORAL_LINK_PREDICTION)
    assert editable_query.text == ("PREDICT LIST_DISTINCT(TRANS.STORE "
                                   "WHERE (NOT (TRANS.TYPE = 'A')) AND "
                                   "(STORES.TYPE = \"B\"), 0, 7, days) "
                                   "RANK TOP 10 FOR EACH USERS.USERS_ID "
                                   "WHERE USERS.AGE < 21 AND USERS.AGE > 10")
    entity_filter_field = editable_query.editable_fields[1]
    assert entity_filter_field.field_name == 'Entity filter'
    assert entity_filter_field.text == 'USERS.AGE < 21 AND USERS.AGE > 10'
    assert get_substring(
        editable_query.text,
        entity_filter_field.position) == (' WHERE ' + entity_filter_field.text)
    assert get_substring(
        editable_query.text,
        entity_filter_field.bold_text_position) == entity_filter_field.text
    assert entity_filter_field.full_filter_format_string == ' WHERE {0}'
    assert entity_filter_field.empty_filter_default == ''

    target_filter_field = editable_query.editable_fields[0]
    assert target_filter_field.field_name == 'Target filter'
    assert target_filter_field.text == 'STORES.TYPE = "B"'
    assert get_substring(editable_query.text,
                         target_filter_field.position) == (
                             ' WHERE (NOT (TRANS.TYPE = \'A\')) AND '
                             '(STORES.TYPE = "B")')
    assert get_substring(
        editable_query.text,
        target_filter_field.bold_text_position) == target_filter_field.text
    assert target_filter_field.full_filter_format_string == (
        ' WHERE (NOT (TRANS.TYPE = \'A\')) AND ({0})')
    assert target_filter_field.empty_filter_default == (
        ' WHERE NOT (TRANS.TYPE = \'A\')')
    assert applicable_query is not None
    assert str(applicable_query.target_ast) == (
        "LIST_DISTINCT(TRANS.STORE "
        "WHERE STORES.TYPE = \"B\", 0, 7, days)")
    assert str(applicable_query.entity_ast) == (
        "USERS.USERS_ID WHERE (USERS.AGE < 21) AND (USERS.AGE > 10)")


def test_get_bp_drop_all_filters(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    query = ("PREDICT LIST_DISTINCT(TRANS.STORE "
             "WHERE NOT(STORES.TYPE = 'A' OR TRANS.TYPE = 'A'), 0, 7) "
             "RANK TOP 10 FOR EACH USERS.USERS_ID "
             "WHERE USERS.AGE > 20")
    validated_query = get_validated_query(graph, query)
    transformer = BPFilterTransformer()
    editable_query = transformer.get_bp_editable_query(
        validated_query, TaskType.TEMPORAL_LINK_PREDICTION, graph, '')
    user_edits = {
        'Entity filter': '',
        'Target filter': '',
    }
    user_response = EditableQueryResponse(editable_query, user_edits)
    editable_query, applicable_query = transformer.insert_filter_overrides(
        user_response, graph, TaskType.TEMPORAL_LINK_PREDICTION)
    assert editable_query.text == ("PREDICT LIST_DISTINCT(TRANS.STORE "
                                   "WHERE NOT (TRANS.TYPE = 'A'), 0, 7, days) "
                                   "RANK TOP 10 FOR EACH USERS.USERS_ID")
    entity_filter_field = editable_query.editable_fields[1]
    assert entity_filter_field.field_name == 'Entity filter'
    assert entity_filter_field.text == ''
    assert get_substring(editable_query.text,
                         entity_filter_field.position) == ''
    assert get_substring(
        editable_query.text,
        entity_filter_field.bold_text_position) == entity_filter_field.text
    assert entity_filter_field.full_filter_format_string == ' WHERE {0}'
    assert entity_filter_field.empty_filter_default == ''

    target_filter_field = editable_query.editable_fields[0]
    assert target_filter_field.field_name == 'Target filter'
    assert target_filter_field.text == ''
    assert get_substring(
        editable_query.text,
        target_filter_field.position) == (' WHERE NOT (TRANS.TYPE = \'A\')')
    assert get_substring(
        editable_query.text,
        target_filter_field.bold_text_position) == target_filter_field.text
    assert target_filter_field.full_filter_format_string == (
        ' WHERE (NOT (TRANS.TYPE = \'A\')) AND ({0})')
    assert target_filter_field.empty_filter_default == (
        ' WHERE NOT (TRANS.TYPE = \'A\')')
    assert applicable_query is not None
    assert str(applicable_query.target_ast) == (
        "LIST_DISTINCT(TRANS.STORE, 0, 7, days)")
    assert str(applicable_query.entity_ast) == ("USERS.USERS_ID")


def test_get_bp_edit_no_filters(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    query = ("PREDICT LIST_DISTINCT(TRANS.STORE "
             "WHERE NOT(STORES.TYPE = \"A\" OR TRANS.TYPE = 'A'),"
             " 0, 7) RANK TOP 10 FOR EACH USERS.USERS_ID "
             "WHERE USERS.AGE > 20")
    validated_query = get_validated_query(graph, query)
    transformer = BPFilterTransformer()
    editable_query = transformer.get_bp_editable_query(
        validated_query, TaskType.TEMPORAL_LINK_PREDICTION, graph, '')
    user_edits: dict[str, str] = {}
    user_response = EditableQueryResponse(editable_query, user_edits)
    editable_query, applicable_query = transformer.insert_filter_overrides(
        user_response, graph, TaskType.TEMPORAL_LINK_PREDICTION)
    assert editable_query.text == ("PREDICT LIST_DISTINCT(TRANS.STORE "
                                   "WHERE (NOT (TRANS.TYPE = 'A')) AND "
                                   "(NOT (STORES.TYPE = \"A\")), 0, 7, days) "
                                   "RANK TOP 10 FOR EACH USERS.USERS_ID "
                                   "WHERE USERS.AGE > 20")
    entity_filter_field = editable_query.editable_fields[1]
    assert entity_filter_field.field_name == 'Entity filter'
    assert entity_filter_field.text == 'USERS.AGE > 20'
    assert get_substring(
        editable_query.text,
        entity_filter_field.position) == (' WHERE ' + entity_filter_field.text)
    assert get_substring(
        editable_query.text,
        entity_filter_field.bold_text_position) == entity_filter_field.text
    assert entity_filter_field.full_filter_format_string == ' WHERE {0}'
    assert entity_filter_field.empty_filter_default == ''

    target_filter_field = editable_query.editable_fields[0]
    assert target_filter_field.field_name == 'Target filter'
    assert target_filter_field.text == 'NOT (STORES.TYPE = "A")'
    assert get_substring(editable_query.text,
                         target_filter_field.position) == (
                             ' WHERE (NOT (TRANS.TYPE = \'A\')) AND '
                             '(NOT (STORES.TYPE = "A"))')
    assert get_substring(
        editable_query.text,
        target_filter_field.bold_text_position) == target_filter_field.text
    assert target_filter_field.full_filter_format_string == (
        ' WHERE (NOT (TRANS.TYPE = \'A\')) AND ({0})')
    assert target_filter_field.empty_filter_default == (
        ' WHERE NOT (TRANS.TYPE = \'A\')')
    assert applicable_query is not None
    assert str(applicable_query.target_ast) == (
        "LIST_DISTINCT(TRANS.STORE "
        "WHERE NOT (STORES.TYPE = \"A\"), 0, 7, days)")
    assert str(
        applicable_query.entity_ast) == ("USERS.USERS_ID WHERE USERS.AGE > 20")


def test_get_bp_edit_node_pred_filters(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    query = ("PREDICT COUNT(TRANS.* "
             "WHERE NOT(STORES.TYPE = 'A' OR TRANS.TYPE = 'A'), 0, 7) "
             "FOR EACH USERS.USERS_ID WHERE USERS.AGE > 20")
    validated_query = get_validated_query(graph, query)
    transformer = BPFilterTransformer()
    editable_query = transformer.get_bp_editable_query(validated_query,
                                                       TaskType.REGRESSION,
                                                       graph, '')
    user_edits = {
        'Entity filter': 'USERS.AGE < 21',
    }
    user_response = EditableQueryResponse(editable_query, user_edits)
    editable_query, applicable_query = transformer.insert_filter_overrides(
        user_response, graph, TaskType.REGRESSION)
    assert editable_query.text == (
        "PREDICT COUNT(TRANS.* "
        "WHERE NOT ((STORES.TYPE = 'A') OR "
        "(TRANS.TYPE = 'A')), 0, 7, days) "
        "FOR EACH USERS.USERS_ID WHERE USERS.AGE < 21")
    entity_filter_field = editable_query.editable_fields[0]
    assert entity_filter_field.field_name == 'Entity filter'
    assert entity_filter_field.text == 'USERS.AGE < 21'
    assert get_substring(
        editable_query.text,
        entity_filter_field.position) == (' WHERE ' + entity_filter_field.text)
    assert get_substring(
        editable_query.text,
        entity_filter_field.bold_text_position) == entity_filter_field.text
    assert entity_filter_field.full_filter_format_string == ' WHERE {0}'
    assert entity_filter_field.empty_filter_default == ''
    assert applicable_query is not None


def test_get_bp_edit_incorrect_syntax(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    query = ("PREDICT COUNT(TRANS.* "
             "WHERE NOT(STORES.TYPE = 'A' OR TRANS.TYPE = 'A'), 0, 7) "
             "FOR EACH USERS.USERS_ID WHERE USERS.AGE > 20")
    validated_query = get_validated_query(graph, query)
    transformer = BPFilterTransformer()
    editable_query = transformer.get_bp_editable_query(validated_query,
                                                       TaskType.REGRESSION,
                                                       graph, '')
    user_edits = {
        'Entity filter': 'USERS.AG',
    }
    user_response = EditableQueryResponse(editable_query, user_edits)
    editable_query, applicable_query = transformer.insert_filter_overrides(
        user_response, graph, TaskType.REGRESSION)
    assert not editable_query.validation_response.ok
    assert ('Encountered the following issues during parsing:'
            in editable_query.validation_response.message())
    assert applicable_query is None
    assert editable_query.text == ("PREDICT COUNT(TRANS.* "
                                   "WHERE NOT ((STORES.TYPE = 'A') OR "
                                   "(TRANS.TYPE = 'A')), 0, 7, days) "
                                   "FOR EACH USERS.USERS_ID WHERE USERS.AG")
    entity_filter_field = editable_query.editable_fields[0]
    assert entity_filter_field.field_name == 'Entity filter'
    assert entity_filter_field.text == 'USERS.AG'
    assert get_substring(
        editable_query.text,
        entity_filter_field.position) == (' WHERE ' + entity_filter_field.text)
    assert get_substring(
        editable_query.text,
        entity_filter_field.bold_text_position) == entity_filter_field.text
    assert entity_filter_field.full_filter_format_string == ' WHERE {0}'
    assert entity_filter_field.empty_filter_default == ''


def test_get_bp_edit_invalid_semantics(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    query = ("PREDICT COUNT(TRANS.* "
             "WHERE NOT(STORES.TYPE = 'A' OR TRANS.TYPE = 'A'), 0, 7) "
             "FOR EACH USERS.USERS_ID WHERE USERS.AGE > 20")
    validated_query = get_validated_query(graph, query)
    transformer = BPFilterTransformer()
    editable_query = transformer.get_bp_editable_query(validated_query,
                                                       TaskType.REGRESSION,
                                                       graph, '')
    user_edits = {
        'Entity filter': 'USERS.INVALID_COL < 21',
    }
    user_response = EditableQueryResponse(editable_query, user_edits)
    editable_query, applicable_query = transformer.insert_filter_overrides(
        user_response, graph, TaskType.REGRESSION)
    assert not editable_query.validation_response.ok
    assert 'does not exist' in editable_query.validation_response.message()
    assert applicable_query is None
    assert editable_query.text == (
        "PREDICT COUNT(TRANS.* "
        "WHERE NOT ((STORES.TYPE = 'A') OR "
        "(TRANS.TYPE = 'A')), 0, 7, days) "
        "FOR EACH USERS.USERS_ID WHERE USERS.INVALID_COL < 21")
    entity_filter_field = editable_query.editable_fields[0]
    assert entity_filter_field.field_name == 'Entity filter'
    assert entity_filter_field.text == 'USERS.INVALID_COL < 21'
    assert get_substring(
        editable_query.text,
        entity_filter_field.position) == (' WHERE ' + entity_filter_field.text)
    assert get_substring(
        editable_query.text,
        entity_filter_field.bold_text_position) == entity_filter_field.text
    assert entity_filter_field.full_filter_format_string == ' WHERE {0}'
    assert entity_filter_field.empty_filter_default == ''


def test_get_bp_edit_only_target(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    query = ("PREDICT LIST_DISTINCT(TRANS.STORE "
             "WHERE NOT(STORES.TYPE = 'A' OR TRANS.TYPE = 'A'), 0, 7) "
             "RANK TOP 10 FOR EACH USERS.USERS_ID "
             "WHERE USERS.AGE > 20")
    validated_query = get_validated_query(graph, query)
    transformer = BPFilterTransformer()
    editable_query = transformer.get_bp_editable_query(
        validated_query, TaskType.TEMPORAL_LINK_PREDICTION, graph, '')
    user_edits = {'Target filter': 'STORES.TYPE = "B"'}
    user_response = EditableQueryResponse(editable_query, user_edits)
    editable_query, applicable_query = transformer.insert_filter_overrides(
        user_response, graph, TaskType.TEMPORAL_LINK_PREDICTION)
    assert editable_query.text == ("PREDICT LIST_DISTINCT(TRANS.STORE "
                                   "WHERE (NOT (TRANS.TYPE = 'A')) AND "
                                   "(STORES.TYPE = \"B\"), 0, 7, days) "
                                   "RANK TOP 10 FOR EACH USERS.USERS_ID "
                                   "WHERE USERS.AGE > 20")
    entity_filter_field = editable_query.editable_fields[1]
    assert entity_filter_field.field_name == 'Entity filter'
    assert entity_filter_field.text == 'USERS.AGE > 20'
    assert get_substring(
        editable_query.text,
        entity_filter_field.position) == (' WHERE ' + entity_filter_field.text)
    assert get_substring(
        editable_query.text,
        entity_filter_field.bold_text_position) == entity_filter_field.text
    assert entity_filter_field.full_filter_format_string == ' WHERE {0}'
    assert entity_filter_field.empty_filter_default == ''

    target_filter_field = editable_query.editable_fields[0]
    assert target_filter_field.field_name == 'Target filter'
    assert target_filter_field.text == 'STORES.TYPE = "B"'
    assert get_substring(editable_query.text,
                         target_filter_field.position) == (
                             ' WHERE (NOT (TRANS.TYPE = \'A\')) AND '
                             '(STORES.TYPE = "B")')
    assert get_substring(
        editable_query.text,
        target_filter_field.bold_text_position) == target_filter_field.text
    assert target_filter_field.full_filter_format_string == (
        ' WHERE (NOT (TRANS.TYPE = \'A\')) AND ({0})')
    assert target_filter_field.empty_filter_default == (
        ' WHERE NOT (TRANS.TYPE = \'A\')')
    assert applicable_query is not None
    assert str(applicable_query.target_ast) == (
        "LIST_DISTINCT(TRANS.STORE "
        "WHERE STORES.TYPE = \"B\", 0, 7, days)")
    assert str(
        applicable_query.entity_ast) == ("USERS.USERS_ID WHERE USERS.AGE > 20")


def test_get_bp_edit_only_entity(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    query = ("PREDICT LIST_DISTINCT(TRANS.STORE "
             "WHERE NOT(STORES.TYPE = 'A' OR TRANS.TYPE = 'A'), 0, 7) "
             "RANK TOP 10 FOR EACH USERS.USERS_ID "
             "WHERE USERS.AGE > 20")
    validated_query = get_validated_query(graph, query)
    transformer = BPFilterTransformer()
    editable_query = transformer.get_bp_editable_query(
        validated_query, TaskType.TEMPORAL_LINK_PREDICTION, graph, '')
    user_edits = {
        'Entity filter': 'USERS.AGE < 21',
    }
    user_response = EditableQueryResponse(editable_query, user_edits)
    editable_query, applicable_query = transformer.insert_filter_overrides(
        user_response, graph, TaskType.TEMPORAL_LINK_PREDICTION)
    assert editable_query.text == ("PREDICT LIST_DISTINCT(TRANS.STORE "
                                   "WHERE (NOT (TRANS.TYPE = 'A')) AND "
                                   "(NOT (STORES.TYPE = 'A')), 0, 7, days) "
                                   "RANK TOP 10 FOR EACH USERS.USERS_ID "
                                   "WHERE USERS.AGE < 21")
    entity_filter_field = editable_query.editable_fields[1]
    assert entity_filter_field.field_name == 'Entity filter'
    assert entity_filter_field.text == 'USERS.AGE < 21'
    assert get_substring(
        editable_query.text,
        entity_filter_field.position) == (' WHERE ' + entity_filter_field.text)
    assert get_substring(
        editable_query.text,
        entity_filter_field.bold_text_position) == entity_filter_field.text
    assert entity_filter_field.full_filter_format_string == ' WHERE {0}'
    assert entity_filter_field.empty_filter_default == ''

    target_filter_field = editable_query.editable_fields[0]
    assert target_filter_field.field_name == 'Target filter'
    assert target_filter_field.text == 'NOT (STORES.TYPE = \'A\')'
    assert get_substring(editable_query.text,
                         target_filter_field.position) == (
                             ' WHERE (NOT (TRANS.TYPE = \'A\')) AND '
                             '(NOT (STORES.TYPE = \'A\'))')
    assert get_substring(
        editable_query.text,
        target_filter_field.bold_text_position) == target_filter_field.text
    assert target_filter_field.full_filter_format_string == (
        ' WHERE (NOT (TRANS.TYPE = \'A\')) AND ({0})')
    assert target_filter_field.empty_filter_default == (
        ' WHERE NOT (TRANS.TYPE = \'A\')')
    assert applicable_query is not None
    assert str(applicable_query.target_ast) == (
        "LIST_DISTINCT(TRANS.STORE "
        "WHERE NOT (STORES.TYPE = 'A'), 0, 7, days)")
    assert str(
        applicable_query.entity_ast) == ("USERS.USERS_ID WHERE USERS.AGE < 21")


def test_get_bp_edit_edge_invalid_table(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    query = ("PREDICT LIST_DISTINCT(TRANS.STORE "
             "WHERE NOT(STORES.TYPE = 'A' OR TRANS.TYPE = 'A'), 0, 7) "
             "RANK TOP 10 FOR EACH USERS.USERS_ID WHERE USERS.AGE > 20")
    validated_query = get_validated_query(graph, query)
    transformer = BPFilterTransformer()
    editable_query = transformer.get_bp_editable_query(
        validated_query, TaskType.TEMPORAL_LINK_PREDICTION, graph, '')
    user_edits = {
        'Target filter': 'NOT (TRANS.VALUE > 10.0)',
    }
    user_response = EditableQueryResponse(editable_query, user_edits)
    editable_query, applicable_query = transformer.insert_filter_overrides(
        user_response, graph, TaskType.TEMPORAL_LINK_PREDICTION)
    assert not editable_query.validation_response.ok
    assert ('only conditions that refer to STORES'
            in editable_query.validation_response.message())
    assert applicable_query is None
    assert editable_query.text == (
        "PREDICT LIST_DISTINCT(TRANS.STORE "
        "WHERE (NOT (TRANS.TYPE = 'A')) AND "
        "(NOT (TRANS.VALUE > 10.0)), 0, 7, days) "
        "RANK TOP 10 FOR EACH USERS.USERS_ID WHERE USERS.AGE > 20")
    entity_filter_field = editable_query.editable_fields[1]
    assert entity_filter_field.field_name == 'Entity filter'
    assert entity_filter_field.text == 'USERS.AGE > 20'
    assert get_substring(
        editable_query.text,
        entity_filter_field.position) == (' WHERE ' + entity_filter_field.text)
    assert get_substring(
        editable_query.text,
        entity_filter_field.bold_text_position) == entity_filter_field.text
    assert entity_filter_field.full_filter_format_string == ' WHERE {0}'
    assert entity_filter_field.empty_filter_default == ''

    target_filter_field = editable_query.editable_fields[0]
    assert target_filter_field.field_name == 'Target filter'
    assert target_filter_field.text == 'NOT (TRANS.VALUE > 10.0)'
    assert get_substring(editable_query.text,
                         target_filter_field.position) == (
                             ' WHERE (NOT (TRANS.TYPE = \'A\')) AND '
                             '(NOT (TRANS.VALUE > 10.0))')
    assert get_substring(
        editable_query.text,
        target_filter_field.bold_text_position) == target_filter_field.text
    assert target_filter_field.full_filter_format_string == (
        ' WHERE (NOT (TRANS.TYPE = \'A\')) AND ({0})')
    assert target_filter_field.empty_filter_default == (
        ' WHERE NOT (TRANS.TYPE = \'A\')')
