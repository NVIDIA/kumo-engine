# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumoapi.pquery.AST import Condition, Filter, LogicalOperation
from kumoapi.typing import AggregationType, ProblemType, RelOp, TimeUnit
from kumopql.parser.parser import PQLParser


def get_pq_time_units() -> list[TimeUnit]:
    return [
        TimeUnit.MINUTES,
        TimeUnit.HOURS,
        TimeUnit.DAYS,
        TimeUnit.MONTHS,
    ]


@pytest.fixture
def parser() -> PQLParser:
    return PQLParser()


def test_column_parsing(parser: PQLParser) -> None:
    query = 'predict table.value for each table.user'
    parsed = parser.to_parsed_predictive_query(query)
    assert parsed.entity_ast.to_string() == 'table.user'
    assert parsed.entity_column == 'table.user'
    assert parsed.target_ast.to_string() == 'table.value'


@pytest.mark.parametrize('problem_type',
                         ['RANK', 'rank', 'CLASSIFY', 'classify'])
def test_problem_type(parser: PQLParser, problem_type: str) -> None:
    query = f'PREDICT table.value {problem_type} FOR EACH table.user'
    assert parser.to_parsed_predictive_query(
        query).problem_type.value.upper() == problem_type.upper()


@pytest.mark.parametrize('top', ['TOP', 'top'])
@pytest.mark.parametrize('k', ['1', '25', '1000'])
def test_top_k(parser: PQLParser, top: str, k: str) -> None:
    query = f'PREDICT table.value {top} {k} FOR EACH table.user'
    assert parser.to_parsed_predictive_query(query).top_k == int(k)


@pytest.mark.parametrize('n', [1, 30, 120])
def test_forecast(parser: PQLParser, n: int) -> None:
    query = (f'PREDICT SUM(table.value, 0, 10) FORECAST {n} TIMEFRAMES '
             f'FOR EACH table.user')
    assert parser.to_parsed_predictive_query(query).num_forecasts == n


def test_forecast_default(parser: PQLParser) -> None:
    query = 'predict table.value for each table.user'
    assert parser.to_parsed_predictive_query(query).num_forecasts == 1


@pytest.mark.parametrize('n', [1, 30, 120])
def test_forecast_sets_problem_type(parser: PQLParser, n: int) -> None:
    """FORECAST clause always sets problem_type to FORECAST regardless of N."""
    query = (f'PREDICT SUM(table.value, 0, 10) FORECAST {n} TIMEFRAMES '
             f'FOR EACH table.user')
    parsed = parser.to_parsed_predictive_query(query)
    assert parsed.problem_type == ProblemType.FORECAST
    assert parsed.num_forecasts == n


def test_no_forecast_clause_leaves_problem_type_unset(
        parser: PQLParser) -> None:
    """Without a FORECAST clause, problem_type remains None for regression."""
    query = 'PREDICT SUM(table.value, 0, 10) FOR EACH table.user'
    parsed = parser.to_parsed_predictive_query(query)
    assert parsed.problem_type is None
    assert parsed.num_forecasts == 1


@pytest.mark.parametrize('aggr_type',
                         [aggr.value for aggr in AggregationType] +
                         [aggr.value.lower() for aggr in AggregationType])
@pytest.mark.parametrize(
    'time_unit',
    [time_unit.value.upper() for time_unit in get_pq_time_units()] +
    [time_unit.value for time_unit in get_pq_time_units()],
)
@pytest.mark.parametrize('infty', ['-INFINITY', '-INF', '-infinity', '-inf'])
def test_aggr(
    parser: PQLParser,
    aggr_type: str,
    time_unit: str,
    infty: str,
) -> None:
    query = f'PREDICT {aggr_type}(table.value, -30, -5) FOR EACH table.user'
    assert parser.to_parsed_predictive_query(query).target_ast.to_string() == (
        f'{aggr_type.upper()}(table.value, -30, -5, days)')

    query = (f'PREDICT {aggr_type}(table.value, -30, -5, {time_unit}) '
             f'FOR EACH table.user')
    assert parser.to_parsed_predictive_query(query).target_ast.to_string() == (
        f'{aggr_type.upper()}(table.value, -30, -5, {time_unit.lower()})')

    query = (f'PREDICT {aggr_type}(table.value, {infty}, -5, {time_unit}) '
             f'FOR EACH table.user')
    assert parser.to_parsed_predictive_query(query).target_ast.to_string() == (
        f'{aggr_type.upper()}(table.value, -INF, -5, {time_unit.lower()})')

    query = f'PREDICT {aggr_type}(table.value) FOR EACH table.user'
    assert parser.to_parsed_predictive_query(query).target_ast.to_string() == (
        f'{aggr_type.upper()}(table.value)')


@pytest.mark.parametrize('rel', [rel.value for rel in RelOp])
def test_numerical_relations(parser: PQLParser, rel: str) -> None:
    query = (f'PREDICT SUM(table.value, -30, -5) {rel} 0 '
             f'FOR EACH table.user')
    target = parser.to_parsed_predictive_query(query).target_ast
    assert isinstance(target, Condition)
    assert target.op.value == rel


def test_invalid_date(parser: PQLParser) -> None:
    with pytest.raises(TypeError, match='is not a valid instance'):
        parser.to_parsed_predictive_query('PREDICT table.date = 2020-02-30 '
                                          'FOR EACH table.user')


@pytest.mark.parametrize('where', ['WHERE', 'where'])
def test_with_entity_filter(parser: PQLParser, where: str) -> None:
    query = (
        f'PREDICT table.target FOR EACH table.user {where} table.col_a = 1')
    parsed = parser.to_parsed_predictive_query(query)
    assert isinstance(parsed.entity_ast, Filter)
    assert parsed.entity_column == 'table.user'
    assert parsed.entity_ast.to_string() == (
        'table.user WHERE table.col_a = 1')


@pytest.mark.parametrize('where', ['WHERE', 'where'])
def test_with_temporal_entity_filter(parser: PQLParser, where: str) -> None:
    query = (f'PREDICT table.target FOR EACH table.user {where} '
             f'SUM(table.col_a, 0, 20) > 1')
    parsed = parser.to_parsed_predictive_query(query)
    assert isinstance(parsed.entity_ast, Filter)
    assert parsed.entity_column == 'table.user'
    assert parsed.entity_ast.to_string() == (
        'table.user WHERE SUM(table.col_a, 0, 20, days) > 1')


@pytest.mark.parametrize('assuming', ['ASSUMING', 'assuming'])
def test_whatif(parser: PQLParser, assuming: str) -> None:
    query = (f'PREDICT AVG(table.target, 0, 10, DAYS) > 10 FOR EACH '
             f'table.user {assuming} SUM(table2.value, 0, 2, DAYS) < 5')
    parsed = parser.to_parsed_predictive_query(query)
    assert isinstance(parsed.whatif_ast, Condition)
    assert parsed.whatif_ast.to_string() == 'SUM(table2.value, 0, 2, days) < 5'


def test_pquerylang_all_together(parser: PQLParser) -> None:
    valid_query = ('PREDICT SUM(target.col WHERE target.value=1, 0, 8) > 0 OR '
                   'FIRST(target2.col2 WHERE target2.value=1, 0, 8) < 4\n'
                   'FOR EACH user.ID WHERE user.value = 1 OR '
                   'SUM(table2.value, -10, 0) > 10 # comment\n'
                   'ASSUMING SUM(table3.value, 0, 2) < 5')
    parsed = parser.to_parsed_predictive_query(valid_query)
    assert isinstance(parsed.whatif_ast, Condition)
    assert parsed.whatif_ast.to_string() == 'SUM(table3.value, 0, 2, days) < 5'
    assert (parsed.whatif_ast.get_location().message ==
            '(row 3, column 9: row 3, column 35)')
    assert isinstance(parsed.entity_ast, Filter)
    assert parsed.entity_ast.to_string() == (
        'user.ID WHERE (user.value = 1) OR '
        '(SUM(table2.value, -10, 0, days) > 10)')
    assert (parsed.entity_ast.get_location().message ==
            '(row 2, column 9: row 2, column 69)')
    assert parsed.entity_column == 'user.ID'
    assert isinstance(parsed.target_ast, LogicalOperation)
    assert parsed.target_ast.to_string() == (
        '(SUM(target.col WHERE target.value = 1, 0, 8, days) > 0) OR '
        '(FIRST(target2.col2 WHERE target2.value = 1, 0, 8, days) < 4)')
    assert parsed.target_ast.get_location().message == (
        '(row 1, column 8: row 1, column 108)')


@pytest.mark.parametrize('c', '\\\'\"/bfnrtu')
def test_valid_escaped_unicode(parser: PQLParser, c: str) -> None:
    query = f'PREDICT table.tag = "chr\\{c}1234567" FOR EACH table.user'
    assert parser.to_parsed_predictive_query(query)


@pytest.mark.parametrize('c', ['x', 'o', 'q', 'a', 'w', 's', '\\\\'])
def test_invalid_escaped_unicode(parser: PQLParser, c: str) -> None:
    query = f'PREDICT table.tag = "chr\\{c}1234567" FOR EACH table.user'
    with pytest.raises(ValueError, match='not a valid escape pattern'):
        parser.to_parsed_predictive_query(query)
