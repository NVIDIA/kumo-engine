# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from antlr4.error.Errors import NoViableAltException
from kumoapi.typing import AggregationType, TimeUnit
from kumopql.parser.error_translator import Antlr4SyntaxError, ErrorTranslator
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
    with pytest.raises(
            ValueError, match="Line 1, col 13;  Syntax Error: token "
            "recognition error at: '.'"):
        parser.to_lisp_tree('predict table. for each table.user')
    with pytest.raises(
            ValueError, match="(?s)Line 1, col 21; Extraneous input, got "
            "'column', but expected one of FOR_EACH."):
        parser.to_lisp_tree('predict table.column.column for each table.user')
    with pytest.raises(
            ValueError, match="(?s)Line 1, col 13;  Syntax Error: token "
            "recognition error at: '.'"):
        parser.to_lisp_tree('predict table.123 for each table.user')


@pytest.mark.parametrize('aggr_type',
                         [aggr.value for aggr in AggregationType] +
                         [aggr.value.lower() for aggr in AggregationType])
def test_aggr(parser: PQLParser, aggr_type: str) -> None:
    with pytest.raises(
            ValueError, match=f'.*line 1 and col .* \'{aggr_type}'
            '[(]table.value,0[)]\'.* '):
        parser.to_lisp_tree(
            f'PREDICT {aggr_type}(table.value, 0) FOR EACH table.user')
    with pytest.raises(
            ValueError,
            match='.*Could not process the text: "200"; in line 1 and col.*'):
        parser.to_lisp_tree(f'PREDICT {aggr_type}(table.value, 0, -10, 200) '
                            f'FOR EACH table.user')
    with pytest.raises(
            ValueError,
            match='.*Could not process the text: "metres"; in line 1 and '
            'col.*'):
        parser.to_lisp_tree(
            f'PREDICT {aggr_type}(table.value, 0, -10, metres) '
            f'FOR EACH table.user')


@pytest.mark.parametrize('rel', ['==', '=>', '=<'])
def test_invalid_numerical_relations(parser: PQLParser, rel: str) -> None:
    with pytest.raises(
            ValueError, match='Line 1, col 35; Extraneous input, got .*, '
            'but expected one of '):
        parser.to_lisp_tree(f'PREDICT SUM(table.value, -30, -5) {rel} 0 '
                            f'FOR EACH table.user')


@pytest.mark.parametrize('sep', ['.', '/', '-'])
def test_dates(parser: PQLParser, sep: str) -> None:
    # Parser throws slightly different errors depending on the operator
    # Extraneous error/ No viable alternative error
    # So we just match that an error was thrown at the right place
    with pytest.raises(ValueError, match='Line 1, col 31; '):
        parser.to_lisp_tree(f'PREDICT table.date = 2020{sep}01{sep}01{sep}01 '
                            f'FOR EACH table.user')
    with pytest.raises(ValueError, match='Line 1, col 25; '):
        parser.to_lisp_tree(
            'PREDICT table.date = 2020{sep}01 FOR EACH table.user')


def test_strings(parser: PQLParser) -> None:
    with pytest.raises(
            ValueError,
            match='Line 1, col 22;  Syntax Error: token recognition error at:'
    ):
        parser.to_lisp_tree('PREDICT table.str = "\"" FOR EACH table.user')
    with pytest.raises(
            ValueError,
            match='Line 1, col 20;  Syntax Error: token recognition error at:'
    ):
        parser.to_lisp_tree('PREDICT table.str = " FOR EACH table.user')
    with pytest.raises(
            ValueError, match='Line 1, col 20;  Syntax Error: '
            'token recognition error at: .*'):
        parser.to_lisp_tree('PREDICT table.str = "\' FOR EACH table.user')

    # Same as above, with single quotation marks
    with pytest.raises(
            ValueError,
            match='Line 1, col 22;  Syntax Error: token recognition error at:'
    ):
        parser.to_lisp_tree('PREDICT table.str = \'\'\' FOR EACH table.user')
    with pytest.raises(
            ValueError,
            match='Line 1, col 20;  Syntax Error: token recognition error at:'
    ):
        parser.to_lisp_tree('PREDICT table.str = \' FOR EACH table.user')
    with pytest.raises(
            ValueError, match='Line 1, col 20;  Syntax Error: '
            'token recognition error at: .*'):
        parser.to_lisp_tree('PREDICT table.str = \'\" FOR EACH table.user')


def test_bool_incorrect_case_raises_error(parser: PQLParser) -> None:
    with pytest.raises(ValueError,
                       match='Line 1, col 22; Input Mismatch, got \'fAlSe\''):
        parser.to_lisp_tree('PREDICT table.value = fAlSe FOR EACH table.user')
    with pytest.raises(ValueError,
                       match='Line 1, col 22; Input Mismatch, got \'TrUe\''):
        parser.to_lisp_tree('PREDICT table.value = TrUe FOR EACH table.user')


def test_decimals(parser: PQLParser) -> None:
    with pytest.raises(
            ValueError, match='Line 1, col 23; Input Mismatch, got \',\', but '
            'expected one of'):
        parser.to_lisp_tree('PREDICT table.value > 3,3 FOR EACH table.user')


@pytest.mark.parametrize("array", [
    "('a',)",
    "('a')",
    "(10.1)",
    "(True)",
])
def test_arrays(parser: PQLParser, array: str) -> None:
    with pytest.raises(ValueError,
                       match='Array with single element is not supported'):
        parser.to_lisp_tree("PREDICT table.value FOR EACH "
                            f"table.user WHERE table.user IN {array}")


def test_errors(parser: PQLParser) -> None:
    with pytest.raises(ValueError, match='Input Mismatch, got \'<EOF>\''):
        parser.to_lisp_tree('PREDICT table.value FOR EACH')

    with pytest.raises(ValueError, match='Input Mismatch, got \'<EOF>\''):
        parser.to_lisp_tree('PREDICT table.value FOR EACH entity.id WHERE')

    for rule in ('condition', 'aggregation'):
        recognizer = Mock(ruleNames=[rule])
        error = NoViableAltException(recognizer)
        error.deadEndConfigs = [Mock(alt=0)]
        syntax_error = Antlr4SyntaxError(recognizer, Mock(text='invalid'), 1, 0,
                                         'invalid syntax', error, 0, [])
        message = ErrorTranslator()._handleNoViableAltException(syntax_error)[0]
        assert (
            'https://docs.nvidia.com/sdgm/kumo-engine/1.0.0/'
            'kumo-studio-guide-coming-soon/preliminary-documentation/'
            'train-model/predictive-query/pquery-structure'
        ) in message
        assert 'docs.kumo.ai' not in message


@pytest.mark.parametrize('query_string, error_regex', [
    (
        " ",
        r'Query does not start with the PREDICT keyword',
    ),
    (
        "not a predictive query",
        r'Query does not start with the PREDICT keyword',
    ),
    (
        "PREDICT ",
        r'The target \(PREDICT\) clause in this query is empty or invalid',
    ),
    (
        "PREDICT FOR EACH",
        r'The target \(PREDICT\) clause in this query is empty or invalid',
    ),
    (
        "PREDICT AGGR(T.C, 0, 7)",
        r'Query is missing the FOR EACH keyword',
    ),
    (
        "PREDICT LIST_DISTINCT(T.C, 0, 7) FOR EACH",
        r'The entity \(FOR EACH\) clause in this query is empty or invalid',
    ),
    (
        "PREDICT T FOR EACH T.C",
        r'The target \(PREDICT\) clause in this query is empty or invalid',
    ),
    (
        "PREDICT T. FOR EACH T.C",
        r'The target \(PREDICT\) clause in this query is empty or invalid',
    ),
    (
        "PREDICT T FOR EACH C",
        r'The target \(PREDICT\) clause in this query is empty or invalid',
    ),
    (
        "PREDICT T.C FOR EACH T.",
        r'The entity \(FOR EACH\) clause in this query is empty or invalid',
    ),
    (
        "PREDICT T.C FOR EACH T",
        r'The entity \(FOR EACH\) clause in this query is empty or invalid',
    ),
    (
        "PREDICT T FOR EACH T.C WHERE",
        r'The target \(PREDICT\) clause in this query is empty or invalid',
    ),
    (
        "PREDICT T. FOR EACH T.C WHERE",
        r'The target \(PREDICT\) clause in this query is empty or invalid',
    ),
    (
        "PREDICT T FOR EACH T WHERE",
        r'The target \(PREDICT\) clause in this query is empty or invalid',
    ),
    (
        "PREDICT T.C FOR EACH T. WHERE",
        r'The entity \(FOR EACH\) clause in this query is empty or invalid',
    ),
    (
        "PREDICT T.C FOR EACH T WHERE",
        r'The entity \(FOR EACH\) clause in this query is empty or invalid',
    ),
    (
        "PREDICT T.C FOR EACH T.C WHERE",
        r'The entity filter \(FOR EACH ... WHERE\) clause in this query '
        r'is empty or invalid',
    ),
    (
        "PREDICT COUNT(TRANS.* WHERE TRANS.PRICE > 1, 0, 30) FOR EACH USERS",
        r'The entity \(FOR EACH\) clause in this query is empty or invalid',
    ),
    (
        "PREDICT COUNT(TRANS.* WHERE TRANS.PRICE > 1, 0, 30) "
        "FOR EACH USERS.ID WHERE",
        r'The entity filter \(FOR EACH ... WHERE\) clause in this query '
        r'is empty or invalid',
    ),
    (
        "PREDICT T.C WHERE FOR EACH T.C",
        r'The target \(PREDICT\) clause in this query is invalid',
    ),
    (
        "PREDICT SUM(T.C WHERE) FOR EACH CUSTOMER.CUSTOMER_ID",
        r'The target filter \(PREDICT ... WHERE\) clause in this query '
        r'is empty or invalid',
    ),
    (
        "PREDICT SUM(T.C WHERE X) FOR EACH CUSTOMER.CUSTOMER_ID",
        r'The target filter \(PREDICT ... WHERE\) clause in this query '
        r'is empty or invalid',
    ),
    (
        "PREDICT SUM(T.C WHERE X.) FOR EACH CUSTOMER.CUSTOMER_ID",
        r'The target filter \(PREDICT ... WHERE\) clause in this query '
        r'is empty or invalid',
    ),
    (
        "PREDICT SUM(T.C WHERE X.Y) FOR EACH CUSTOMER.CUSTOMER_ID",
        r'The target filter \(PREDICT ... WHERE\) clause in this query '
        r'is empty or invalid',
    ),
    (
        "PREDICT T FOR EACH CUSTOMER.CUSTOMER_ID WHERE T.C > 0",
        r'The target \(PREDICT\) clause in this query is empty or invalid',
    ),
])
def test_lexer_required_field_syntax_errors(
    parser: PQLParser,
    query_string: str,
    error_regex: str,
) -> None:
    """This test makes sure that errors in parsing the query itself are shown
    in a nicer way to the user. This only tests that high-level summaries are
    shown (in addition to the detailed parser summary).
    """
    with pytest.raises(ValueError, match=error_regex):
        PQLParser().to_parsed_predictive_query(query_string)
