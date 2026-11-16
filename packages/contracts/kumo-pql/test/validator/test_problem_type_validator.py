# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import pytest
from kumoapi.graph import GraphDefinition
from kumopql.parser.parser import PQLParser
from kumopql.validator import (
    PredictiveQueryValidator,
    ProblemTypeValidator,
    TypeValidator,
)


@pytest.mark.parametrize('query, is_valid', [
    ('PREDICT USERS.AGE RANK FOR EACH TRANS.TRANSACTION_ID', False),
    ('PREDICT USERS.AGE CLASSIFY FOR EACH TRANS.TRANSACTION_ID', False),
    ('PREDICT COUNT(TRANS.*, 0, 20) RANK FOR EACH USERS.USERS_ID', False),
    ('PREDICT COUNT(TRANS.*, 0, 20) CLASSIFY FOR EACH USERS.USERS_ID', False),
    (('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) '
      'CLASSIFY FOR EACH USERS.USERS_ID'), True),
    (('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) '
      'CLASSIFY FOR EACH USERS.USERS_ID'), True),
    (('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) '
      'RANK TOP 10 FOR EACH USERS.USERS_ID'), True),
])
def test_problem_type_validation(
        query: str, is_valid: bool,
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = ProblemTypeValidator()
    type_validator = TypeValidator(graph)

    parsed_query = parser.to_parsed_predictive_query(query)
    type_validator.infer_dtypes(parsed_query)
    type_validator.infer_stypes(parsed_query)
    response = validator.validate(parsed_query)
    if not is_valid:
        assert not response.ok
        assert ('is only supported for multicategorical targets.'
                in response.errors[0].title)
    else:
        assert response.ok


@pytest.mark.parametrize('query, is_valid', [
    ('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) FOR EACH USERS.USERS_ID',
     False),
    ('PREDICT LIST_DISTINCT(TRANS.STORE, 0, 20) FOR EACH USERS.USERS_ID',
     False),
    ('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) RANK TOP 10 '
     'FOR EACH USERS.USERS_ID', True),
    ('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) CLASSIFY '
     'FOR EACH USERS.USERS_ID', True),
])
def test_multicategorical_target_requires_problem_type(
        query: str, is_valid: bool,
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = ProblemTypeValidator()
    type_validator = TypeValidator(graph)

    parsed_query = parser.to_parsed_predictive_query(query)
    type_validator.infer_dtypes(parsed_query)
    type_validator.infer_stypes(parsed_query)
    response = validator.validate(parsed_query)
    if not is_valid:
        assert not response.ok
        assert ('Problem type is required for multicategorical targets.'
                in response.errors[0].title)
    else:
        assert response.ok


@pytest.mark.parametrize('query,is_valid', [
    ('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) RANK FOR EACH USERS.USERS_ID',
     False),
    ('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20)'
     'RANK TOP 2 FOR EACH USERS.USERS_ID', True),
])
def test_rank_requires_topk(
        query: str, is_valid: bool,
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = ProblemTypeValidator()
    type_validator = TypeValidator(graph)

    parsed_query = parser.to_parsed_predictive_query(query)
    type_validator.infer_dtypes(parsed_query)
    type_validator.infer_stypes(parsed_query)
    response = validator.validate(parsed_query)
    if not is_valid:
        assert not response.ok
        assert ("Problem type 'RANK' requires TOP K"
                in response.errors[0].title)
    else:
        assert response.ok


@pytest.mark.parametrize('problem_type', ['CLASSIFY', 'RANK'])
@pytest.mark.parametrize('top_k', [-1, 0, 1])
def test_top_k_must_be_positive(
        problem_type: str, top_k: int,
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    query = (f'PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) '
             f'{problem_type} TOP {top_k} FOR EACH USERS.USERS_ID')
    parsed = PQLParser().to_parsed_predictive_query(query)
    validator = TypeValidator(graph)
    validator.infer_dtypes(parsed)
    validator.infer_stypes(parsed)
    response = ProblemTypeValidator().validate(parsed)
    if top_k < 1:
        assert not response.ok
        assert f'got {top_k}' in response.errors[0].message
        assert 'at least 1' in response.errors[0].message
    else:
        assert response.ok


@pytest.mark.parametrize('query', [
    ('PREDICT SUM(TRANS.CNT, 0, 7) FORECAST 1 TIMEFRAMES '
     'FOR EACH USERS.USERS_ID'),
    ('PREDICT SUM(TRANS.CNT, 0, 7) FORECAST 5 TIMEFRAMES '
     'FOR EACH USERS.USERS_ID'),
    ('PREDICT SUM(TRANS.CNT, 0, 7) FORECAST 10000 TIMEFRAMES '
     'FOR EACH USERS.USERS_ID'),
    ('PREDICT AVG(TRANS.VALUE, 0, 30) FORECAST 12 TIMEFRAMES '
     'FOR EACH USERS.USERS_ID'),
    ('PREDICT COUNT(TRANS.*, 0, 7) FORECAST 4 TIMEFRAMES '
     'FOR EACH USERS.USERS_ID'),
    ('PREDICT SUM(TRANS.CNT, 0, 7, days) FORECAST 3 TIMEFRAMES '
     'FOR EACH USERS.USERS_ID'),
    ('PREDICT SUM(TRANS.CNT, 0, 7, months) FORECAST 6 TIMEFRAMES '
     'FOR EACH USERS.USERS_ID'),
])
def test_forecast_valid(
        query: str, create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = ProblemTypeValidator()
    type_validator = TypeValidator(graph)

    parsed_query = parser.to_parsed_predictive_query(query)
    type_validator.infer_dtypes(parsed_query)
    type_validator.infer_stypes(parsed_query)
    response = validator.validate(parsed_query)
    assert response.ok


@pytest.mark.parametrize('n', [0, -1])
def test_forecast_invalid_n_below_minimum(
        n: int, create_mock_graph: Callable[..., GraphDefinition]) -> None:
    """FORECAST N TIMEFRAMES is invalid when N < 1."""
    graph = create_mock_graph()
    parser = PQLParser()
    validator = ProblemTypeValidator()
    type_validator = TypeValidator(graph)

    query = (f'PREDICT SUM(TRANS.CNT, 0, 7) FORECAST {n} TIMEFRAMES '
             f'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    type_validator.infer_dtypes(parsed_query)
    type_validator.infer_stypes(parsed_query)
    response = validator.validate(parsed_query)
    assert not response.ok
    assert 'Invalid FORECAST value' in response.errors[0].title


def test_forecast_invalid_n_above_maximum(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    """FORECAST N TIMEFRAMES is invalid when N > MAX_TIMEFRAMES."""
    graph = create_mock_graph()
    parser = PQLParser()
    validator = ProblemTypeValidator()
    type_validator = TypeValidator(graph)

    query = ('PREDICT SUM(TRANS.CNT, 0, 7) FORECAST 10001 TIMEFRAMES '
             'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    type_validator.infer_dtypes(parsed_query)
    type_validator.infer_stypes(parsed_query)
    response = validator.validate(parsed_query)
    assert not response.ok
    assert 'FORECAST value too large' in response.errors[0].title


def test_forecast_requires_temporal_target(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    """FORECAST cannot be used with a static (non-temporal) target column."""
    graph = create_mock_graph()
    parser = PQLParser()
    validator = ProblemTypeValidator()
    type_validator = TypeValidator(graph)

    query = ('PREDICT USERS.AGE FORECAST 5 TIMEFRAMES '
             'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    type_validator.infer_dtypes(parsed_query)
    type_validator.infer_stypes(parsed_query)
    response = validator.validate(parsed_query)
    assert not response.ok
    assert 'FORECAST requires a temporal target' in response.errors[0].title


def test_forecast_requires_numerical_target(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    """FORECAST cannot be used with a categorical (non-numerical) target."""
    graph = create_mock_graph()
    parser = PQLParser()
    validator = ProblemTypeValidator()
    type_validator = TypeValidator(graph)

    # LAST of a categorical column has a time range but is not numerical
    query = ('PREDICT LAST(TRANS.TYPE, 0, 7) FORECAST 5 TIMEFRAMES '
             'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    type_validator.infer_dtypes(parsed_query)
    type_validator.infer_stypes(parsed_query)
    response = validator.validate(parsed_query)
    assert not response.ok
    assert 'FORECAST requires a numerical target' in response.errors[0].title


def test_static_list_distinct_classify(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    # LIST_DISTINCT w/o target aggregation and CLASSIFY is undefined now
    graph = create_mock_graph(add_time_cols=False)
    query = ('PREDICT LIST_DISTINCT(TRANS.STORE) CLASSIFY TOP 2 '
             'FOR EACH USERS.USERS_ID')
    parser = PQLParser()
    parsed_query = parser.to_parsed_predictive_query(query)
    validator = PredictiveQueryValidator(graph)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert validated_query is None
    assert response.errors[0].title == 'Aggregation missing a time range'
