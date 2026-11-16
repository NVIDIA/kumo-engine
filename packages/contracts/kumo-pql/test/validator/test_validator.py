# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Callable

import pytest
from kumoapi.graph import GraphDefinition
from kumoapi.json_serde import from_json, to_json
from kumoapi.pquery import ParsedPredictiveQuery, ValidatedPredictiveQuery
from kumoapi.typing import AggregationType
from kumopql.parser.parser import PQLParser
from kumopql.validator import PredictiveQueryValidator


def test_col_not_in_graph(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    # All columns used in the query must be in the graph.
    graph = create_mock_graph()
    parser = PQLParser()
    parsed_query = parser.to_parsed_predictive_query(
        'PREDICT TRANS.TYPE FOR EACH USERS.USERSS_ID')
    validator = PredictiveQueryValidator(graph)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert validated_query is None
    assert r"'USERSS_ID' does not exist in table 'USERS'" in response.message()
    assert "row 1, column 28" in response.message()


def test_table_not_in_graph(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    # All tables used in the query must be in the graph.
    graph = create_mock_graph()
    parser = PQLParser()
    parsed_query = parser.to_parsed_predictive_query(
        'PREDICT TRANS.TYPE FOR EACH USERSS.USERS_ID')
    validator = PredictiveQueryValidator(graph)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert validated_query is None
    assert "'USERSS' does not exist in the graph" in response.message()
    assert "row 1, column 28" in response.message()


def test_entity_pkey(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    # Entity column must be a primary key.
    graph = create_mock_graph()
    parser = PQLParser()
    parsed_query = parser.to_parsed_predictive_query(
        'PREDICT TRANS.TYPE FOR EACH USERS.AGE')
    validator = PredictiveQueryValidator(graph)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert validated_query is None
    assert re.search(r"Entity .* is not a primary key",
                     response.message()) is not None


@pytest.mark.parametrize('query', [
    'PREDICT SUM(TRANS.*, 0, 20) FOR EACH USERS.USERS_ID',
    'PREDICT TRANS.* FOR EACH USERS.USERS_ID',
    'PREDICT TRANS.* > 0 FOR EACH USERS.USERS_ID',
    'PREDICT COUNT(TRANS.* WHERE TRANS.* = "A", 0, 20) '
    'FOR EACH USERS.USERS_ID',
    'PREDICT TRANS.TYPE FOR EACH USERS.USERS_ID WHERE TRANS.* > 0',
    'PREDICT TRANS.TYPE FOR EACH USERS.USERS_ID '
    'ASSUMING SUM(TRANS.*, 0, 10) > 0',
])
def test_invalid_asterisk(
        query: str, create_mock_graph: Callable[..., GraphDefinition]) -> None:
    # Entity column must be a primary key.
    graph = create_mock_graph()
    parser = PQLParser()
    parsed_query = parser.to_parsed_predictive_query(query)
    validator = PredictiveQueryValidator(graph)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert validated_query is None
    assert re.search(r"Wildcard column .* appeared in an invalid position",
                     response.message()) is not None


@pytest.mark.parametrize('query', [
    'PREDICT COUNT(TRANS.*, 0, 20) FOR EACH USERS.USERS_ID',
    'PREDICT COUNT(TRANS.* WHERE TRANS.TYPE = "A", 0, 20) '
    'FOR EACH USERS.USERS_ID',
    'PREDICT LAST(TRANS.TYPE, 0, 10) FOR EACH USERS.USERS_ID '
    'WHERE COUNT(TRANS.*, -10, 0) > 0',
    'PREDICT LAST(TRANS.TYPE, 0, 10) FOR EACH USERS.USERS_ID '
    'ASSUMING COUNT(TRANS.*, 0, 10) > 0',
])
def test_valid_asterisk(
        query: str, create_mock_graph: Callable[..., GraphDefinition]) -> None:
    # Entity column must be a primary key.
    graph = create_mock_graph()
    parser = PQLParser()
    parsed_query = parser.to_parsed_predictive_query(query)
    validator = PredictiveQueryValidator(graph)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert validated_query is not None
    assert response.ok


def test_get_final_aggregation(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    # Entity column must be a primary key.
    query = 'PREDICT COUNT(TRANS.*, 0, 20) FOR EACH USERS.USERS_ID'
    graph = create_mock_graph()
    parser = PQLParser()
    parsed_query = parser.to_parsed_predictive_query(query)
    validator = PredictiveQueryValidator(graph)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert validated_query is not None
    assert response.ok
    aggr = validated_query.get_final_target_aggregation()
    assert aggr is not None
    assert aggr.aggr == AggregationType.COUNT
    col = validated_query.get_final_target_column()
    assert col is None


def test_get_final_column(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    # Entity column must be a primary key.
    query = 'PREDICT USERS.AGE FOR EACH TRANS.TRANSACTION_ID'
    graph = create_mock_graph()
    parser = PQLParser()
    parsed_query = parser.to_parsed_predictive_query(query)
    validator = PredictiveQueryValidator(graph)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert validated_query is not None
    assert response.ok
    aggr = validated_query.get_final_target_aggregation()
    assert aggr is None
    col = validated_query.get_final_target_column()
    assert col is not None
    assert col.fqn == 'USERS.AGE'


@pytest.mark.parametrize('query', [
    ('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) '
     'CLASSIFY FOR EACH USERS.USERS_ID'),
    ('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) '
     'RANK TOP 2 FOR EACH USERS.USERS_ID'),
    ('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) '
     'RANK TOP 10 FOR EACH USERS.USERS_ID'),
    ('PREDICT SUM(TRANS.CNT, 0, 20) FOR EACH USERS.USERS_ID '
     'WHERE COUNT(TRANS.*, -20, 0) > 0'),
    ('PREDICT SUM(TRANS.CNT, 0, 20) > 0 OR COUNT(TRANS.*, 0, 5) > 10 '
     'FOR EACH USERS.USERS_ID WHERE USERS.AGE > 21'),
])
def test_config_loading(
        query: str,
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(query)
    config = to_json(parsed_query)
    assert from_json(config, ParsedPredictiveQuery) == parsed_query

    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert response.ok
    assert validated_query is not None

    config = to_json(validated_query)
    reloaded_query = from_json(config, ValidatedPredictiveQuery)
    assert reloaded_query == validated_query


@pytest.mark.parametrize('query', [
    ('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) '
     'CLASSIFY FOR EACH USERS.USERS_ID'),
    ('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) '
     'RANK TOP 2 FOR EACH USERS.USERS_ID'),
    'PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) '
    'RANK TOP 10 FOR EACH USERS.USERS_ID',
])
def test_config_loading_array(
        query: str, create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()

    validator = PredictiveQueryValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(query)
    config = to_json(parsed_query)
    assert from_json(config, ParsedPredictiveQuery) == parsed_query

    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert response.ok
    assert validated_query is not None

    config = to_json(validated_query)
    reloaded_query = from_json(config, ValidatedPredictiveQuery)
    # Overwrite validation response -- false negative due to subclassing
    reloaded_query.validation_response = validated_query.validation_response
    assert reloaded_query == validated_query


def test_entity_table_in_target_filter_node_prediction(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = ('PREDICT COUNT(TRANS.* WHERE USERS.AGE > 18, 0, 2) '
             'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert validated_query is not None
    assert response.ok
    assert (
        response.warnings[0].title == 'Entity filter in the target definition')


def test_entity_table_in_target_filter_node_prediction_multi_filter(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = ('PREDICT COUNT(TRANS.* WHERE USERS.AGE > 18, 0, 2) > 0 '
             'OR SUM(TRANS.VALUE WHERE USERS.AGE < 30, 0, 2) < 100 '
             'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert validated_query is not None
    assert response.ok
    assert len(response.warnings) == 1
    assert (
        response.warnings[0].title == 'Entity filter in the target definition')


def test_entity_table_in_target_filter_node_prediction_valid(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = ('PREDICT COUNT(TRANS.* WHERE USERS.AGE > 18 '
             'AND TRANS.TYPE = "A", 0, 2) FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert validated_query is not None
    assert response.ok
    assert len(response.warnings) == 0


@pytest.mark.parametrize('aggr', ['LAST', 'FIRST'])
def test_last_first_fkey(
        aggr: str, create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = (f'PREDICT {aggr}(TRANS.STORE, 0, 2) '
             f'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert not response.ok
    assert validated_query is None
    assert response.errors[0].title == (
        'Aggregation not supported on foreign key columns')


def test_entity_table_in_target_filter_link_prediction(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = ('PREDICT LIST_DISTINCT(TRANS.STORE WHERE USERS.AGE > 18, 0, 2) '
             'RANK TOP 10 FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert not response.ok
    assert validated_query is None
    assert response.errors[0].title == 'Invalid target filter'


@pytest.mark.parametrize('rank', ['RANK TOP 10', 'CLASSIFY'])
def test_entity_col_in_target(
        rank: str, create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = (f'PREDICT LIST_DISTINCT(TRANS.USER, 0, 2) '
             f'{rank} FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert not response.ok
    assert validated_query is None
    assert response.errors[0].title == 'Nothing to predict'


@pytest.mark.parametrize('rank', ['RANK TOP 10', 'CLASSIFY'])
def test_pkey_in_target(
        rank: str, create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = (f'PREDICT LIST_DISTINCT(TRANS.TRANSACTION_ID, 0, 2) '
             f'{rank} FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert not response.ok
    assert validated_query is None
    assert (response.errors[0].title ==
            'Aggregation not supported on primary key columns')


def test_entity_col_in_target_valid(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = 'PREDICT COUNT(TRANS.USER, 0, 2) FOR EACH USERS.USERS_ID'
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert response.ok
    assert validated_query is not None


@pytest.mark.parametrize('query', [
    'PREDICT USERS.USERS_ID FOR EACH USERS.USERS_ID',
    'PREDICT USERS.USERS_ID > 0 FOR EACH USERS.USERS_ID',
    'PREDICT (USERS.USERS_ID > 0) OR (USERS.USERS_ID < 0) '
    'FOR EACH USERS.USERS_ID',
])
def test_entity_pkey_in_target(
        query: str, create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph(add_time_cols=False)
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert not response.ok
    assert validated_query is None
    assert "Predicting entity primary key column" in response.message()


def test_static_count_asterisk(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph(add_time_cols=False)
    query = 'PREDICT COUNT(TRANS.*) FOR EACH USERS.USERS_ID'
    parser = PQLParser()
    parsed_query = parser.to_parsed_predictive_query(query)
    validator = PredictiveQueryValidator(graph)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert validated_query is None
    assert response.errors[0].title == 'Aggregation missing a time range'


def test_single_target_fkey(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = ('PREDICT LAST(TRANS.TYPE, 0, 2) = "A" '
             'OR AVG(TRANS.VALUE, 0, 2) > 0 '
             'AND SUM(TRANS.CNT, 0, 2) < 2.0 '
             'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert response.ok
    assert validated_query is not None
    assert validated_query.get_single_target_fkey() == 'TRANS.USER'


def test_single_target_fkey_static(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph(add_time_cols=False)
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = ('PREDICT STORES.TYPE = "text" AND USERS.DESC CONTAINS "text" '
             'FOR EACH TRANS.TRANSACTION_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert response.ok
    assert validated_query is not None
    assert validated_query.get_single_target_fkey() == 'STORES.STORES_ID'
    assert validated_query.get_single_target_fkey(lhs=True) == 'TRANS.STORE'


def test_forecast_valid(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = ('PREDICT SUM(TRANS.CNT, 0, 20) FORECAST 4 TIMEFRAMES '
             'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert response.ok
    assert validated_query is not None
    assert validated_query.num_forecasts == 4


def test_forecast_zero(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = ('PREDICT SUM(TRANS.CNT, 0, 20) FORECAST 0 TIMEFRAMES '
             'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert validated_query is None
    assert 'FORECAST value must be at least 1' in response.message()


def test_forecast_negative(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = ('PREDICT SUM(TRANS.CNT, 0, 20) FORECAST -5 TIMEFRAMES '
             'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert validated_query is None
    assert 'FORECAST value must be at least 1' in response.message()


def test_forecast_too_large(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = ('PREDICT SUM(TRANS.CNT, 0, 20) FORECAST 10001 TIMEFRAMES '
             'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert validated_query is None
    assert 'FORECAST value too large' in response.message()


def test_forecast_static_target(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = ('PREDICT TRANS.TYPE FORECAST 4 TIMEFRAMES '
             'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert validated_query is None
    assert 'FORECAST requires a temporal target' in response.message()


def test_forecast_with_rank(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    parser = PQLParser()
    query = ('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) '
             'RANK TOP 10 FORECAST 4 TIMEFRAMES FOR EACH USERS.USERS_ID')
    with pytest.raises(ValueError):
        parser.to_parsed_predictive_query(query)


def test_forecast_with_classify(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    parser = PQLParser()
    query = ('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) '
             'CLASSIFY FORECAST 4 TIMEFRAMES FOR EACH USERS.USERS_ID')
    with pytest.raises(ValueError):
        parser.to_parsed_predictive_query(query)


def test_forecast_with_list_distinct(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    # LIST_DISTINCT + FORECAST is rejected by forecast validation.
    graph = create_mock_graph()
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    query = ('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) '
             'FORECAST 4 TIMEFRAMES FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert validated_query is None
    assert not response.ok
