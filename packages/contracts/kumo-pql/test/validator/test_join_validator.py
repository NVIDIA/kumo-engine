# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Callable

import pytest
from kumoapi.graph import ColumnKey, ColumnKeyGroup, GraphDefinition
from kumoapi.pquery.AST import Condition, Filter, Join, LogicalOperation
from kumoapi.table import Column
from kumoapi.typing import BoolOp, Dtype, Stype
from kumopql.parser import PQLParser
from kumopql.validator import JoinValidator, PredictiveQueryValidator


def test_hop_has_fkey(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = 'PREDICT TRANS.VALUE FOR EACH USERS.USERS_ID'
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert validated_query is None
    assert "we have found multiple" in response.message()
    assert ('To fix this, you have two options:' in response.message())


def test_event_column_in_entity_filter(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = ('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 14, DAYS) RANK TOP 10 '
             'FOR EACH USERS.USERS_ID WHERE TRANS.FLAG != false')
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)

    assert validated_query is None
    assert response.errors[0].title == 'Invalid entity filter'
    assert (
        "The entity filter condition 'TRANS.FLAG != false' references event "
        "column 'TRANS.FLAG' directly." in response.errors[0].message)
    assert ("'LIST_DISTINCT(TRANS.TYPE WHERE TRANS.FLAG != false, 0, 14, days)'"
            in response.errors[0].message)
    assert ("'COUNT(TRANS.* WHERE TRANS.FLAG != false, -INF, 0) > 0'"
            in response.errors[0].message)


def test_negated_event_column_in_entity_filter(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = ('PREDICT COUNT(TRANS.*, 0, 10) '
             'FOR EACH USERS.USERS_ID '
             'WHERE NOT (TRANS.FLAG = false)')
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)

    assert validated_query is None
    assert response.errors[0].title == 'Invalid entity filter'
    assert ("'COUNT(TRANS.* WHERE NOT (TRANS.FLAG = false), -INF, 0) > 0'"
            in response.errors[0].message)
    assert ("'COUNT(TRANS.* WHERE TRANS.FLAG = false, -INF, 0) > 0'"
            not in response.errors[0].message)


def test_static_event_column_filter_omits_temporal_fix(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = ('PREDICT USERS.AGE FOR EACH USERS.USERS_ID '
             'WHERE TRANS.FLAG = false')
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)

    assert validated_query is None
    assert response.errors[0].title == 'Invalid entity filter'
    assert 'COUNT(' not in response.errors[0].message
    assert 'inside that aggregation' not in response.errors[0].message


def test_mixed_table_entity_filter_omits_count_fix(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = ('PREDICT COUNT(TRANS.*, 0, 10) '
             'FOR EACH USERS.USERS_ID '
             'WHERE TRANS.FLAG = false AND USERS.AGE > 21')
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)

    assert validated_query is None
    assert response.errors[0].title == 'Invalid entity filter'
    assert 'COUNT(' not in response.errors[0].message


def test_aggregated_entity_filter_omits_nested_fix(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = ('PREDICT COUNT(TRANS.*, 0, 10) '
             'FOR EACH USERS.USERS_ID '
             'WHERE COUNT(TRANS.*, -10, 0) > 0 '
             'AND TRANS.FLAG = false')
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)

    assert validated_query is None
    assert response.errors[0].title == 'Invalid entity filter'
    assert 'To filter events used by the prediction target' not in (
        response.errors[0].message)
    assert 'To select entities that have matching events' not in (
        response.errors[0].message)


@pytest.mark.parametrize('query', [
    'PREDICT COUNT(TRANS.*, 0, 10) FOR EACH USERS.USERS_ID',
    'PREDICT USERS.AGE FOR EACH TRANS.TRANSACTION_ID',
])
def test_hop_has_unique_fkey(
        query: str, create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    # Add a cheeky extra foreign key
    graph.tables['TRANS'].cols.append(
        Column(name='USER2', dtype=Dtype.int, stype=Stype.ID))
    # The first column group corresponds to USERS.USERS_ID
    new_col_group = list(graph.col_groups[0].columns)
    new_col_group.insert(1, ColumnKey('TRANS', 'USER2'))
    graph.col_groups[0] = ColumnKeyGroup(columns=tuple(new_col_group))
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert validated_query is None
    assert re.search('The foreign key from table .* is not unique',
                     response.message()) is not None
    assert 'To unambiguously link each row in' in response.message()


@pytest.mark.parametrize('args', [
    {
        'query':
        'PREDICT COUNT(USERS.*, 0, 10) FOR EACH TRANS.TRANSACTION_ID',
        'validation_response':
        "There is no foreign key from table USERS to table TRANS"
    },
    {
        'query':
        'PREDICT COUNT(TRANS.*, 0, 10) FOR EACH USERS.USERS_ID '
        'WHERE COUNT(USERS.*, -10, 0) > 0',
        'validation_response':
        "There is no foreign key from table USERS to table USERS"
    },
    {
        'query': 'PREDICT COUNT(TRANS.*, 0, 10) FOR EACH USERS.USERS_ID '
        'WHERE SUM(USERS.AGE, -10, 0) > 0',
        'validation_response': "There is no foreign key from table USERS"
    },
])
def test_missing_fkey(
        args: dict, create_mock_graph: Callable[..., GraphDefinition]) -> None:
    """Test that a validation error is raised when a foreign key is
    misssing.
    """
    query = args['query']
    validation_response = args['validation_response']
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    if validation_response is not None:
        assert validation_response in response.message()
    else:
        assert response.ok


def test_select_cols_aggr(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    query = ('PREDICT COUNT(TRANS.* WHERE TRANS.VALUE > 0, 0, 10) '
             'FOR EACH USERS.USERS_ID')

    validator = PredictiveQueryValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert response.ok
    assert validated_query is not None

    assert isinstance(validated_query.target_ast, Join)


def test_select_cols_condition(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    query = ('PREDICT USERS.AGE '
             'FOR EACH TRANS.TRANSACTION_ID WHERE USERS.AGE > 21')

    validator = PredictiveQueryValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert response.ok
    assert validated_query is not None

    assert isinstance(validated_query.entity_ast, Filter)
    assert isinstance(validated_query.entity_ast.condition, Condition)
    assert isinstance(validated_query.entity_ast.condition.target, Join)


def test_select_cols_condition_with_logical(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    query = ('PREDICT USERS.AGE '
             'FOR EACH TRANS.TRANSACTION_ID WHERE USERS.AGE > 21 '
             'AND USERS.AGE <35')

    validator = PredictiveQueryValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert response.ok
    assert validated_query is not None

    assert isinstance(validated_query.entity_ast, Filter)
    assert isinstance(validated_query.entity_ast.condition, LogicalOperation)
    assert validated_query.entity_ast.condition.bool_op == BoolOp.AND
    assert isinstance(validated_query.entity_ast.condition.left, Condition)
    assert isinstance(validated_query.entity_ast.condition.right, Condition)
    assert validated_query.entity_ast.condition.left.target is not None
    assert validated_query.entity_ast.condition.right.target is not None
    assert isinstance(validated_query.entity_ast.condition.left.target, Join)
    assert isinstance(validated_query.entity_ast.condition.right.target, Join)


@pytest.mark.parametrize('query', [
    'PREDICT TRANS.VALUE FOR EACH USERS.USERS_ID',
    'PREDICT TRANS.VALUE = 0 FOR EACH USERS.USERS_ID',
    'PREDICT (TRANS.VALUE = 0) AND (TRANS.VALUE != 0) '
    'FOR EACH USERS.USERS_ID',
])
def test_target_entity_fkey_validation(
        query: str, create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = JoinValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(query)
    response = validator.infer_and_validate_joins(parsed_query)
    assert not response.ok
    assert re.search("To fix this, you have two options:",
                     response.message()) is not None
