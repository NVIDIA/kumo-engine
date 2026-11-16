# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Callable

import pytest
from kumoapi.graph import GraphDefinition
from kumoapi.table import Column
from kumoapi.typing import Dtype, Stype
from kumopql.parser import PQLParser
from kumopql.validator import PredictiveQueryValidator, TimeRangeValidator


def test_whatif_cannot_have_negative_dates(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TimeRangeValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(
        'PREDICT COUNT(TRANS.*, 0, 30) FOR EACH USERS.USERS_ID '
        'ASSUMING SUM(TRANS.VALUE, -10, 0) > 0')
    response = validator.validate_time_ranges(parsed_query)
    assert re.search(('ASSUMPTION cannot have an '
                      'aggregation in the past. Both start and end need '
                      'to be non negative.'), response.message()) is not None


def test_target_cannot_have_negative_dates(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TimeRangeValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(
        'PREDICT COUNT(TRANS.*, -10, 0) FOR EACH USERS.USERS_ID')
    response = validator.validate_time_ranges(parsed_query)
    assert re.search(('TARGET cannot have an '
                      'aggregation in the past. Both start and end need '
                      'to be non negative.'), response.message()) is not None


@pytest.mark.parametrize('add_time_cols', [False, True])
def test_static_target_entity_missing_time_col_warning(
        add_time_cols: bool,
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph(add_time_cols=add_time_cols)
    if not add_time_cols:
        # We add a time col to users to check if the validator correctly
        # complains about it
        graph.tables['USERS'].time_col = 'TIME'
        graph.tables['USERS'].cols.append(
            Column(name='TIME', stype=Stype.timestamp, dtype=Dtype.time))
    parser = PQLParser()
    query = 'PREDICT USERS.AGE FOR EACH TRANS.TRANSACTION_ID'
    validator = TimeRangeValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(query)
    response = validator.validate_time_ranges(parsed_query)
    if add_time_cols:
        assert re.search(('Entity table TRANS does not have a "Create date"'),
                         response.message()) is None
    else:
        assert re.search(('Entity table TRANS does not have a "Create date"'),
                         response.message()) is not None


def test_entity_cannot_have_positive_dates(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TimeRangeValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(
        'PREDICT COUNT(TRANS.*, 0, 10) FOR EACH USERS.USERS_ID '
        'WHERE COUNT(TRANS.*, 0, 10) > 0')
    response = validator.validate_time_ranges(parsed_query)
    assert re.search(
        ('.* ENTITY contains a future-looking time range. '
         'This is not allowed as it would fail during batch '
         'prediction due to unavailable future data. Consider '
         'using the ASSUMING clause for future scenarios instead.'),
        response.message()) is not None

    # Add an extra check to make sure that the error message is not
    # incorrect target column specification.
    assert re.search("incorrect target column specification.",
                     response.message()) is None


@pytest.mark.parametrize('query', [
    'PREDICT COUNT(TRANS.*) FOR EACH USERS.USERS_ID ',
    'PREDICT SUM(TRANS.VALUE) FOR EACH USERS.USERS_ID ',
    'PREDICT LIST_DISTINCT(TRANS.STORE) CONTAINS "1" FOR EACH USERS.USERS_ID ',
    'PREDICT LIST_DISTINCT(TRANS.TYPE) FOR EACH USERS.USERS_ID ',
    'PREDICT COUNT(TRANS.*, 0, 10) FOR EACH USERS.USERS_ID '
    'WHERE LIST_DISTINCT(TRANS.STORE) CONTAINS "1"',
])
@pytest.mark.parametrize('add_time_cols', [False, True])
def test_aggregation_has_dates(
        query: str, add_time_cols: bool,
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph(add_time_cols=add_time_cols)
    parser = PQLParser()
    validator = TimeRangeValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(query)
    response = validator.validate_time_ranges(parsed_query)
    assert not response.ok
    assert re.search('aggregation .* is missing a time range.',
                     response.message()) is not None


def test_aggregation_has_time_col(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    # a standard temporal aggregation without time column would result in
    # failure
    graph = create_mock_graph(add_time_cols=False)
    parser = PQLParser()
    validator = TimeRangeValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(
        'PREDICT COUNT(TRANS.*, 0, 10) FOR EACH USERS.USERS_ID')
    response = validator.validate_time_ranges(parsed_query)
    assert not response.ok
    assert re.search(
        'A \"Create date\" column is required to use a time range',
        response.message()) is not None

    # list distinct aggregation without time range with no time column is fine
    parser = PQLParser()
    validator = TimeRangeValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(
        'PREDICT LIST_DISTINCT(TRANS.STORE) FOR EACH USERS.USERS_ID')
    response = validator.validate_time_ranges(parsed_query)
    assert response.ok

    # list distinct aggregation with time range with no time column will result
    # in an error
    parser = PQLParser()
    validator = TimeRangeValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(
        'PREDICT LIST_DISTINCT(TRANS.TRANSACTION_ID, 0, 10) '
        'FOR EACH USERS.USERS_ID')
    response = validator.validate_time_ranges(parsed_query)
    assert not response.ok
    assert re.search(
        'A \"Create date\" column is required to use a time range',
        response.message()) is not None

    # a standard temporal aggregation with time column would not result in
    # failure
    graph = create_mock_graph(add_time_cols=True)
    parser = PQLParser()
    validator = TimeRangeValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(
        'PREDICT COUNT(TRANS.*, 0, 10) FOR EACH USERS.USERS_ID')
    response = validator.validate_time_ranges(parsed_query)
    assert response.ok


def test_whatif_static_query(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    query = ('PREDICT USERS.AGE FOR EACH USERS.USERS_ID '
             'ASSUMING COUNT(TRANS.*, 0, 10) > 5')
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TimeRangeValidator(graph)

    parsed_query = parser.to_parsed_predictive_query(query)
    response = validator.validate_time_ranges(parsed_query)
    assert not response.ok
    assert ('ASSUMING not allowed for static queries'
            in response.errors[0].title)


def test_temporal_entity_filter_in_static_query(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    query = ('PREDICT USERS.AGE FOR EACH USERS.USERS_ID '
             'WHERE COUNT(TRANS.*, -10, 0) > 5')
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TimeRangeValidator(graph)

    parsed_query = parser.to_parsed_predictive_query(query)
    response = validator.validate_time_ranges(parsed_query)
    assert not response.ok
    assert 'Temporal aggregation not allowed' in response.errors[0].title


def test_mixing_temporal_and_static_query(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    query = ('PREDICT USERS.AGE > 21 OR COUNT(TRANS.*, 0, 10) > 5 '
             'FOR EACH USERS.USERS_ID ')
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TimeRangeValidator(graph)

    parsed_query = parser.to_parsed_predictive_query(query)
    response = validator.validate_time_ranges(parsed_query)
    assert not response.ok
    assert 'Mixing static and temporal' in response.errors[0].title
    assert 'Mixing static and temporal' in response.errors[0].message


def test_static_whatif(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    query = ('PREDICT COUNT(TRANS.*, 0, 10) FOR EACH USERS.USERS_ID '
             'ASSUMING USERS.AGE > 21 AND COUNT(TRANS.*, 0, 5) > 5')
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TimeRangeValidator(graph)

    parsed_query = parser.to_parsed_predictive_query(query)
    response = validator.validate_time_ranges(parsed_query)
    assert not response.ok
    assert 'Static condition not allowed' in response.errors[0].title


def test_whatif_nested_static_filter(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    query = ('PREDICT COUNT(TRANS.*, 5, 10) FOR EACH USERS.USERS_ID '
             'ASSUMING COUNT(TRANS.* WHERE TRANS.VALUE > 10, 0, 5) > 0')
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TimeRangeValidator(graph)

    parsed_query = parser.to_parsed_predictive_query(query)
    response = validator.validate_time_ranges(parsed_query)
    assert response.ok


@pytest.mark.parametrize('query, expected_error', [
    ('PREDICT LIST_DISTINCT(TRANS.STORE) CLASSIFY FOR EACH USERS.USERS_ID',
     'LIST_DISTINCT is only possible for RANK problems'),
    ('PREDICT LIST_DISTINCT(TRANS.TYPE) CLASSIFY FOR EACH USERS.USERS_ID',
     'LIST_DISTINCT is only possible for RANK problems'),
    ('PREDICT LIST_DISTINCT(TRANS.TYPE) CLASSIFY '
     'TOP 10 FOR EACH USERS.USERS_ID',
     'LIST_DISTINCT is only possible for RANK problems'),
    ('PREDICT LIST_DISTINCT(TRANS.TYPE) RANK TOP 10 FOR EACH USERS.USERS_ID',
     'static usage of LIST_DISTINCT is only possible for the target clause '
     'with a foreign key'),
])
@pytest.mark.parametrize('add_time_cols', [True, False])
def test_list_distinct_temporal_target_static(
        query: str, expected_error: str, add_time_cols: bool,
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph(add_time_cols=add_time_cols)

    # Run validation
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert not response.ok
    assert validated_query is None
    assert expected_error in response.message()
