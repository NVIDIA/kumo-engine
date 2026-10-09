# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumoapi.typing import AggregationType, RelOp, TimeUnit
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


def test_comments(parser: PQLParser) -> None:
    assert parser.to_lisp_tree(
        'predict table.value for each table.user') == parser.to_lisp_tree(
            'predict table.value for each table.user#comment')
    assert parser.to_lisp_tree(
        'predict table.value > 2 for each table.user') == parser.to_lisp_tree(
            'predict table.value#comment one\n> 2 #comment two\n'
            'for each table.user')


@pytest.mark.parametrize('whitespace', [
    c for c in '\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0\u1680\u2000'
    '\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a'
    '\u2028\u2029\u202f\u205f\u3000'
])
def test_whitespace(parser: PQLParser, whitespace: str) -> None:
    assert parser.to_lisp_tree(
        f'predict table.value{whitespace} for each table.user'
    ) == parser.to_lisp_tree('predict table.value for each table.user')
    assert parser.to_lisp_tree(
        f'predict table.value{whitespace} > 2 for each table.user'
    ) == parser.to_lisp_tree('predict table.value > 2 for each table.user')


def test_column_parsing(parser: PQLParser) -> None:
    query = 'predict table.value for each table.user'
    expected = ('(prog predict (target (column table.value)) problem_spec '
                'for each (entity (column table.user)) whatif <EOF>)')
    assert parser.to_lisp_tree(query) == expected


@pytest.mark.parametrize('problem_type',
                         ['RANK', 'rank', 'CLASSIFY', 'classify'])
def test_problem_type(parser: PQLParser, problem_type: str) -> None:
    query = f'PREDICT table.value {problem_type} FOR EACH table.user'
    assert parser.to_lisp_tree(query) == (
        f'(prog PREDICT (target (column table.value)) '
        f'(problem_spec (problem_type {problem_type})) FOR EACH '
        f'(entity (column table.user)) whatif <EOF>)')


@pytest.mark.parametrize('top', ['TOP', 'top'])
@pytest.mark.parametrize('k', ['1', '25', '1000'])
def test_top_k(parser: PQLParser, top: str, k: str) -> None:
    query = f'PREDICT table.value {top} {k} FOR EACH table.user'
    assert parser.to_lisp_tree(query) == (
        f'(prog PREDICT (target (column table.value)) '
        f'(problem_spec (top_k {top} {k})) FOR EACH '
        f'(entity (column table.user)) whatif <EOF>)')


@pytest.mark.parametrize('forecast', ['FORECAST', 'forecast'])
@pytest.mark.parametrize('n', ['1', '30', '120'])
def test_forecast(parser: PQLParser, forecast: str, n: str) -> None:
    query = (f'PREDICT SUM(table.value, 0, 10) {forecast} {n} TIMEFRAMES '
             f'FOR EACH table.user')
    expected = (f'(prog PREDICT (target (aggregation SUM ( (column '
                f'table.value) , 0 , 10 ))) '
                f'(problem_spec (forecast {forecast} {n} TIMEFRAMES)) '
                f'FOR EACH (entity (column table.user)) whatif <EOF>)')
    assert parser.to_lisp_tree(query) == expected


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
    assert parser.to_lisp_tree(query) == (
        f'(prog PREDICT (target (aggregation {aggr_type} ( (column '
        f'table.value) , -30 , -5 ))) problem_spec '
        f'FOR EACH (entity (column table.user)) whatif <EOF>)')

    query = (f'PREDICT {aggr_type}(table.value, -30, -5, {time_unit}) '
             f'FOR EACH table.user')
    expected = (f'(prog PREDICT (target (aggregation {aggr_type} ( (column '
                f'table.value) , -30 , -5 , {time_unit} ))) problem_spec '
                f'FOR EACH (entity (column table.user)) whatif <EOF>)')
    assert parser.to_lisp_tree(query) == expected

    query = (f'PREDICT {aggr_type}(table.value, {infty}, -5, {time_unit}) '
             f'FOR EACH table.user')
    expected = (f'(prog PREDICT (target (aggregation {aggr_type} ( (column '
                f'table.value) , {infty} , -5 , {time_unit} ))) problem_spec '
                f'FOR EACH (entity (column table.user)) whatif <EOF>)')
    assert parser.to_lisp_tree(query) == expected

    query = f'PREDICT {aggr_type}(table.value) FOR EACH table.user'
    assert parser.to_lisp_tree(query) == (
        f'(prog PREDICT (target (aggregation {aggr_type} ( (column '
        f'table.value) ))) problem_spec FOR EACH '
        f'(entity (column table.user)) whatif <EOF>)')


@pytest.mark.parametrize('rel', [rel.value for rel in RelOp])
def test_numerical_relations(parser: PQLParser, rel: str) -> None:
    query = (f'PREDICT SUM(table.value, -30, -5) {rel} 0 '
             f'FOR EACH table.user')
    assert parser.to_lisp_tree(query) == (
        f'(prog PREDICT (target (condition (aggregation SUM ( (column '
        f'table.value) , -30 , -5 )) {rel} (constant 0))) problem_spec '
        f'FOR EACH (entity (column table.user)) whatif '
        f'<EOF>)')


@pytest.mark.parametrize('sep', ['.', '/', '-'])
def test_dates(parser: PQLParser, sep: str) -> None:
    assert parser.to_lisp_tree(
        f'PREDICT table.date = 2020{sep}01{sep}01 FOR EACH table.user'
    ) == (f'(prog PREDICT (target (condition (column table.date) = (constant '
          f'(datetime 2020{sep}01{sep}01)))) problem_spec '
          f'FOR EACH (entity (column table.user)) whatif <EOF>)')

    assert parser.to_lisp_tree(
        f'PREDICT table.date = 2020{sep}01{sep}01 06:06:06 FOR EACH table.user'
    ) == (f'(prog PREDICT (target (condition (column table.date) = (constant '
          f'(datetime 2020{sep}01{sep}01 06:06:06)))) problem_spec '
          f'FOR EACH (entity (column table.user)) whatif <EOF>)')


def test_strings(parser: PQLParser) -> None:
    assert parser.to_lisp_tree(
        'PREDICT table.str = "a" FOR EACH table.user'
    ) == ('(prog PREDICT (target (condition (column table.str) = (constant "a"'
          '))) problem_spec FOR EACH (entity (column '
          'table.user)) whatif <EOF>)')

    assert parser.to_lisp_tree(
        'PREDICT table.str = "SUM(table.value)>0" FOR EACH table.user') == (
            '(prog PREDICT (target (condition (column table.str) = (constant '
            '"SUM(table.value)>0"))) problem_spec FOR EACH '
            '(entity (column table.user)) whatif <EOF>)')

    assert parser.to_lisp_tree(
        'PREDICT table.str = "\\"" FOR EACH table.user'
    ) == (
        '(prog PREDICT (target (condition (column table.str) = (constant "\\""'
        '))) problem_spec FOR EACH (entity (column '
        'table.user)) whatif <EOF>)')

    assert parser.to_lisp_tree(
        'PREDICT table.str = "\'" FOR EACH table.user'
    ) == (
        '(prog PREDICT (target (condition (column table.str) = (constant "\'"'
        '))) problem_spec FOR EACH (entity (column '
        'table.user)) whatif <EOF>)')

    # Same as above, with single quotation marks
    assert parser.to_lisp_tree(
        'PREDICT table.str = \'a\' FOR EACH table.user'
    ) == (
        '(prog PREDICT (target (condition (column table.str) = (constant \'a\''
        '))) problem_spec FOR EACH (entity (column '
        'table.user)) whatif <EOF>)')

    assert parser.to_lisp_tree(
        'PREDICT table.str = \'SUM(table.value)>0\' FOR EACH table.user') == (
            '(prog PREDICT (target (condition (column table.str) = (constant '
            '\'SUM(table.value)>0\'))) problem_spec FOR EACH '
            '(entity (column table.user)) whatif <EOF>)')

    assert parser.to_lisp_tree(
        'PREDICT table.str = \'\\\'\' FOR EACH table.user') == (
            '(prog PREDICT (target (condition (column table.str) = (constant '
            '\'\\\'\'))) problem_spec FOR EACH (entity (column '
            'table.user)) whatif <EOF>)')

    assert parser.to_lisp_tree(
        'PREDICT table.str = \'\"\' FOR EACH table.user'
    ) == (
        '(prog PREDICT (target (condition (column table.str) = (constant \'"\''
        '))) problem_spec FOR EACH (entity (column '
        'table.user)) whatif <EOF>)')


@pytest.mark.parametrize('bool_format', [
    'true',
    'TRUE',
    'True',
    'false',
    'FALSE',
    'False',
])
def test_bool_is_correctly_formatted_by_parser(bool_format: str,
                                               parser: PQLParser) -> None:
    assert parser.to_lisp_tree(
        f'PREDICT table.value = {bool_format} FOR EACH table.user') == (
            f'(prog PREDICT (target (condition (column table.value) = '
            f'(constant {bool_format}))) problem_spec FOR EACH '
            f'(entity (column table.user)) whatif <EOF>)')


@pytest.mark.parametrize('dec', ['3e-5', '3.1E+15', '-3.2'])
def test_decimals(parser: PQLParser, dec: str) -> None:
    assert parser.to_lisp_tree(
        f'PREDICT table.value > {dec} FOR EACH table.user') == (
            f'(prog PREDICT (target (condition (column table.value) > '
            f'(constant {dec}))) problem_spec FOR EACH '
            f'(entity (column table.user)) whatif <EOF>)')


@pytest.mark.parametrize('rel', [
    'LIKE', 'CONTAINS', 'NOT CONTAINS', 'STARTS WITH', 'ENDS WITH', 'like',
    'contains', 'not contains', 'starts with', 'ends with'
])
def test_string_op(parser: PQLParser, rel: str) -> None:
    assert parser.to_lisp_tree(
        f'PREDICT table.value {rel} \'%string%\' FOR EACH table.user') == (
            f'(prog PREDICT (target (condition (column table.value) {rel} '
            f'(constant \'%string%\'))) problem_spec '
            f'FOR EACH (entity (column table.user)) whatif <EOF>)')


@pytest.mark.parametrize('null', ['NULL', 'null'])
@pytest.mark.parametrize('rel', ['IS', 'IS NOT', 'is', 'is not'])
def test_null_op(parser: PQLParser, null: str, rel: str) -> None:
    assert parser.to_lisp_tree(
        f'PREDICT table.value {rel} {null} FOR EACH table.user') == (
            f'(prog PREDICT (target (condition (column table.value) {rel} '
            f'(constant {null}))) problem_spec FOR EACH '
            f'(entity (column table.user)) whatif <EOF>)')


@pytest.mark.parametrize('and_op', ['AND', 'and'])
@pytest.mark.parametrize('or_op', ['OR', 'or'])
@pytest.mark.parametrize('not_op', ['NOT', 'not'])
def test_logical_conditions(
    parser: PQLParser,
    and_op: str,
    or_op: str,
    not_op: str,
) -> None:
    query = (f'PREDICT {not_op} table.a > 1 {and_op} table.b < 2 FOR EACH '
             f'table.user')
    assert parser.to_lisp_tree(query) == (
        f'(prog PREDICT (target (condition (condition {not_op} '
        f'(condition (column table.a) > (constant 1))) {and_op} '
        f'(condition (column table.b) < (constant 2)))) problem_spec '
        f'FOR EACH (entity (column table.user)) whatif '
        f'<EOF>)')

    assert parser.to_lisp_tree(
        f'PREDICT {not_op} table.a > 1 {and_op} table.b < 2{or_op} '
        f'table.c > 3 FOR EACH table.user') == (
            f'(prog PREDICT (target (condition (condition (condition '
            f'{not_op} (condition (column table.a) > (constant 1))) {and_op} '
            f'(condition (column table.b) < (constant 2))) {or_op} '
            f'(condition (column table.c) > (constant 3)))) problem_spec '
            f'FOR EACH (entity (column table.user)) '
            f'whatif <EOF>)')

    assert parser.to_lisp_tree(
        f'PREDICT {not_op} table.a > 1 {or_op} table.b < 2{and_op} '
        f'table.c > 3 FOR EACH table.user') == (
            f'(prog PREDICT (target (condition (condition {not_op} '
            f'(condition (column table.a) > (constant 1))) {or_op} '
            f'(condition (condition (column table.b) < (constant 2)) {and_op} '
            f'(condition (column table.c) > (constant 3))))) problem_spec '
            f'FOR EACH (entity (column table.user)) whatif <EOF>)')

    assert parser.to_lisp_tree(
        f'PREDICT (table.a > 1 {or_op} table.b < 2) {and_op} table.c '
        f'> 3 FOR EACH table.user') == (
            f'(prog PREDICT (target (condition (condition ( (condition '
            f'(condition (column table.a) > (constant 1)) {or_op} (condition '
            f'(column table.b) < (constant 2))) )) {and_op} (condition '
            f'(column table.c) > (constant 3)))) problem_spec '
            f'FOR EACH (entity (column table.user)) whatif <EOF>)')


@pytest.mark.parametrize('where', ['WHERE', 'where'])
def test_with_entity_filter(parser: PQLParser, where: str) -> None:
    query = (
        f'PREDICT table.target FOR EACH table.user {where} table.col_a = 1')
    expected = (f'(prog PREDICT (target (column table.target)) problem_spec '
                f'FOR EACH (entity (filtered_column (column table.user) '
                f'{where} (condition (column table.col_a) = (constant 1)))) '
                f'whatif <EOF>)')
    assert parser.to_lisp_tree(query) == expected


@pytest.mark.parametrize('where', ['WHERE', 'where'])
def test_with_temporal_entity_filter(parser: PQLParser, where: str) -> None:
    query = (f'PREDICT table.target FOR EACH table.user {where} '
             f'SUM(table.col_a, 0, 20) > 1')
    expected = (
        f'(prog PREDICT (target (column table.target)) problem_spec '
        f'FOR EACH (entity (filtered_column (column table.user) '
        f'{where} (condition (aggregation SUM ( (column table.col_a) , 0 , '
        f'20 )) > (constant 1)))) whatif <EOF>)')
    assert parser.to_lisp_tree(query) == expected


@pytest.mark.parametrize('where', ['WHERE', 'where'])
def test_with_target_filter(parser: PQLParser, where: str) -> None:
    query = (f'predict SUM(table.target {where} table.col_a > 10, 0, 10) '
             f'for each table.pkey')
    assert parser.to_lisp_tree(query) == (
        f'(prog predict (target (aggregation SUM ( (filtered_column (column '
        f'table.target) {where} (condition (column table.col_a) > '
        f'(constant 10))) , 0 , 10 ))) problem_spec '
        f'for each (entity (column table.pkey)) whatif <EOF>)')


def test_temporal_entity_filter_and_entity_filter(parser: PQLParser) -> None:
    query = ('PREDICT table.target FOR EACH table.user where '
             'SUM(table.col_a, 0, 20) > 1 and table.col >= 5')
    expected = (
        '(prog PREDICT (target (column table.target)) problem_spec '
        'FOR EACH (entity (filtered_column (column table.user) '
        'where (condition (condition (aggregation SUM ( (column table.col_a) '
        ', 0 , 20 )) > (constant 1)) and (condition (column table.col) >= '
        '(constant 5))))) whatif <EOF>)')
    assert parser.to_lisp_tree(query) == expected


@pytest.mark.parametrize('assuming', ['ASSUMING', 'assuming'])
def test_whatif(parser: PQLParser, assuming: str) -> None:
    query = (f'PREDICT AVG(table.target, 0, 10, DAYS) > 10 FOR EACH '
             f'table.user {assuming} SUM(table2.value, 0, 2, DAYS) < 5')
    assert parser.to_lisp_tree(query) == (
        f'(prog PREDICT (target (condition (aggregation AVG ( (column '
        f'table.target) , 0 , 10 , DAYS )) > (constant 10))) problem_spec '
        f'FOR EACH (entity (column table.user)) (whatif '
        f'{assuming} (condition (aggregation SUM ( (column table2.value) '
        f', 0 , 2 , DAYS )) < (constant 5))) <EOF>)')


def test_temporal_target_filter(parser: PQLParser) -> None:
    query = ('PREDICT COUNT(purchase.* WHERE COUNT(sessions.*, -10, 0) > 0, '
             '0, 10) FOR EACH user.user_id')
    assert parser.to_lisp_tree(query) == (
        '(prog PREDICT (target (aggregation COUNT ( (filtered_column (column '
        'purchase.*) WHERE (condition (aggregation COUNT ( (column '
        'sessions.*) , -10 , 0 )) > (constant 0))) , 0 , 10 ))) problem_spec '
        'FOR EACH (entity (column user.user_id)) whatif '
        '<EOF>)')


def test_nested_where(parser: PQLParser) -> None:
    query = ('PREDICT COUNT(purchase.*, 0, 10) FOR EACH user.user_id WHERE '
             'COUNT(purchase.* WHERE purchase.category="food", -10, 0) > 0')
    assert parser.to_lisp_tree(query) == (
        '(prog PREDICT (target (aggregation COUNT ( (column purchase.*) '
        ', 0 , 10 ))) problem_spec FOR EACH (entity '
        '(filtered_column (column user.user_id) WHERE (condition '
        '(aggregation COUNT ( (filtered_column (column purchase.*) WHERE '
        '(condition (column purchase.category) = (constant "food"))) , '
        '-10 , 0 )) > (constant 0)))) whatif <EOF>)')


@pytest.mark.parametrize('array', [
    "('a', 'b', 'c')",
    '(10.1, True, 100)',
])
def test_arrays(parser: PQLParser, array: str) -> None:
    parser.to_lisp_tree(
        f'PREDICT table.value FOR EACH table.user WHERE table.user IN {array}')


def test_pquerylang_all_together(parser: PQLParser) -> None:
    valid_query = ('PREDICT SUM(target.col WHERE target.value=1, 0, 8) > 0 OR '
                   'FIRST(target2.col2 WHERE target2.value=1, 0, 8) < 4\n'
                   'FOR EACH user.ID WHERE user.value = 1 OR '
                   'SUM(table2.value, -10, 0) > 10 # comment\n'
                   'ASSUMING SUM(table3.value, 0, 2) < 5')
    expected = (
        '(prog PREDICT (target (condition (condition (aggregation SUM '
        '( (filtered_column (column target.col) WHERE (condition (column '
        'target.value) = (constant 1))) , 0 , 8 )) > (constant 0)) OR '
        '(condition (aggregation FIRST ( (filtered_column (column '
        'target2.col2) WHERE (condition (column target2.value) = '
        '(constant 1))) , 0 , 8 )) < (constant 4)))) problem_spec '
        'FOR EACH (entity (filtered_column (column user.ID) '
        'WHERE (condition (condition (column user.value) = (constant 1)) '
        'OR (condition (aggregation SUM ( (column table2.value) , -10 , '
        '0 )) > (constant 10))))) (whatif ASSUMING (condition '
        '(aggregation SUM ( (column table3.value) , 0 , 2 )) < (constant '
        '5))) <EOF>)')
    assert parser.to_lisp_tree(valid_query) == expected
