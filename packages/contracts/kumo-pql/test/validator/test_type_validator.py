# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Callable

import pytest
from kumoapi.graph import GraphDefinition
from kumoapi.pquery.AST.ast_node import ArrayDtype
from kumoapi.typing import Dtype, MemberOp, RelOp, StrOp, Stype
from kumopql.parser import PQLParser
from kumopql.validator import PredictiveQueryValidator, TypeValidator


@pytest.mark.parametrize('query, dtype', [
    ('PREDICT COUNT(TRANS.*, 0, 20) FOR EACH USERS.USERS_ID', Dtype.int),
    ('PREDICT COUNT_DISTINCT(TRANS.TYPE, 0, 20) FOR EACH USERS.USERS_ID',
     Dtype.int),
    ('PREDICT AVG(TRANS.CNT, 0, 20) FOR EACH USERS.USERS_ID', Dtype.float),
    ('PREDICT SUM(TRANS.CNT, 0, 20) FOR EACH USERS.USERS_ID', Dtype.int),
    ('PREDICT SUM(TRANS.VALUE, 0, 20) FOR EACH USERS.USERS_ID', Dtype.float),
    ('PREDICT MIN(TRANS.VALUE, 0, 20) FOR EACH USERS.USERS_ID', Dtype.float),
    ('PREDICT MAX(TRANS.VALUE, 0, 20) FOR EACH USERS.USERS_ID', Dtype.float),
    ('PREDICT LAST(TRANS.VALUE, 0, 20) FOR EACH USERS.USERS_ID', Dtype.float),
    ('PREDICT FIRST(TRANS.VALUE, 0, 20) FOR EACH USERS.USERS_ID', Dtype.float),
    ('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) FOR EACH USERS.USERS_ID',
     Dtype.string),
    ('PREDICT FIRST(TRANS.TYPE, 0, 1) FOR EACH USERS.USERS_ID', Dtype.string),
    ('PREDICT FIRST(TRANS.TYPE, 0, 1) = "A" FOR EACH USERS.USERS_ID',
     Dtype.bool),
    ('PREDICT FIRST(TRANS.TYPE, 0, 1) = "A" OR FIRST(TRANS.VALUE, 0, 1) > 0 '
     'FOR EACH USERS.USERS_ID', Dtype.bool),
    ('PREDICT SUM(TRANS.FLAG, 0, 1) FOR EACH USERS.USERS_ID', Dtype.int),
    ('PREDICT MIN(TRANS.FLAG, 0, 1) FOR EACH USERS.USERS_ID', Dtype.bool),
])
def test_dtype_is_correctly_inferred_from_target(
        query: str, dtype: Dtype,
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    parsed_query = parser.to_parsed_predictive_query(query)
    validator = TypeValidator(graph)
    validator.infer_dtypes(parsed_query)
    assert parsed_query.target_ast.dtype == dtype


@pytest.mark.parametrize('query, dtype', [
    ('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) FOR EACH USERS.USERS_ID',
     ArrayDtype(Dtype.string)),
    ('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) CONTAINS "X" '
     'FOR EACH USERS.USERS_ID', Dtype.bool),
    ('PREDICT LIST_DISTINCT(TRANS.STORE, 0, 20) FOR EACH USERS.USERS_ID',
     ArrayDtype(Dtype.int)),
    ('PREDICT USERS.ARRAY FOR EACH USERS.USERS_ID', ArrayDtype(Dtype.int)),
    ('PREDICT USERS.MULTI FOR EACH USERS.USERS_ID', ArrayDtype(Dtype.string)),
    ('PREDICT USERS.ARRAY CONTAINS 123 FOR EACH USERS.USERS_ID', Dtype.bool),
])
def test_dtype_is_correctly_inferred_from_target_array_targets(
        query: str, dtype: Dtype,
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    parsed_query = parser.to_parsed_predictive_query(query)
    validator = TypeValidator(graph, allow_array_targets=True)
    validator.infer_dtypes(parsed_query)
    assert parsed_query.target_ast.dtype == dtype
    assert validator.validate_dtypes(parsed_query).ok


@pytest.mark.parametrize('query', [
    ('PREDICT LIST_DISTINCT(TRANS.FLAG, 0, 20) FOR EACH USERS.USERS_ID'),
    ('PREDICT LIST_DISTINCT(TRANS.TIME, 0, 20) FOR EACH USERS.USERS_ID'),
])
def test_invalid_target_array_dtype_targets(
        query: str, create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    parsed_query = parser.to_parsed_predictive_query(query)
    validator = TypeValidator(graph, allow_array_targets=True)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert not response.ok
    assert 'Target label has data type ' in response.message()


def test_unsupported_target_array_dtype_targets(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    query = ('PREDICT LIST_DISTINCT(TRANS.TIME, 0, 20) CONTAINS 2020-01-01 '
             'FOR EACH USERS.USERS_ID')

    graph = create_mock_graph()
    parser = PQLParser()
    parsed_query = parser.to_parsed_predictive_query(query)
    validator = TypeValidator(graph, allow_array_targets=True,
                              allow_timestamp_arrays=False)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert not response.ok
    assert ('timestamp is not supported on Snowflake.' in response.message())

    query = ('PREDICT LIST_DISTINCT(TRANS.VALUE, 0, 20) CONTAINS 2020 '
             'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert response.ok


@pytest.mark.parametrize('query, stype', [
    ('PREDICT COUNT(TRANS.*, 0, 20) FOR EACH USERS.USERS_ID', Stype.numerical),
    ('PREDICT COUNT_DISTINCT(TRANS.TYPE, 0, 20) FOR EACH USERS.USERS_ID',
     Stype.numerical),
    ('PREDICT AVG(TRANS.CNT, 0, 20) FOR EACH USERS.USERS_ID', Stype.numerical),
    ('PREDICT SUM(TRANS.CNT, 0, 20) FOR EACH USERS.USERS_ID', Stype.numerical),
    ('PREDICT SUM(TRANS.VALUE, 0, 20) FOR EACH USERS.USERS_ID',
     Stype.numerical),
    ('PREDICT MIN(TRANS.VALUE, 0, 20) FOR EACH USERS.USERS_ID',
     Stype.numerical),
    ('PREDICT MAX(TRANS.VALUE, 0, 20) FOR EACH USERS.USERS_ID',
     Stype.numerical),
    ('PREDICT LAST(TRANS.VALUE, 0, 20) FOR EACH USERS.USERS_ID',
     Stype.numerical),
    ('PREDICT FIRST(TRANS.VALUE, 0, 20) FOR EACH USERS.USERS_ID',
     Stype.numerical),
    ('PREDICT LIST_DISTINCT(TRANS.TYPE, 0, 20) FOR EACH USERS.USERS_ID',
     Stype.multicategorical),
    ('PREDICT FIRST(TRANS.TYPE, 0, 1) FOR EACH USERS.USERS_ID',
     Stype.categorical),
    ('PREDICT FIRST(TRANS.TYPE, 0, 1) = "A" FOR EACH USERS.USERS_ID',
     Stype.categorical),
    ('PREDICT FIRST(TRANS.TYPE, 0, 1) = "A" OR FIRST(TRANS.VALUE, 0, 1) > 0 '
     'FOR EACH USERS.USERS_ID', Stype.categorical),
])
def test_stype_is_correctly_inferred_from_target(
        query: str, stype: Stype,
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    parsed_query = parser.to_parsed_predictive_query(query)
    validator = TypeValidator(graph)
    validator.infer_stypes(parsed_query)
    assert parsed_query.target_ast.stype == stype


@pytest.mark.parametrize('target',
                         ['TRANS.VALUE', 'TRANS.TIME', 'TRANS.STORE'])
@pytest.mark.parametrize('rel_op', [op.value for op in StrOp])
@pytest.mark.parametrize('const',
                         ['1', '6.5', '2020-01-01', '(\"text1\", \"text2\")'])
def test_strop_cannot_be_used_on_non_str_cols_and_targets(
    target: str,
    rel_op: str,
    const: str,
    create_mock_graph: Callable[..., GraphDefinition],
) -> None:
    col_err = f'Operator {rel_op} operates on strings but LAST({target},'
    const_err = f'Operator {rel_op} operates on strings but {const}'
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TypeValidator(graph)
    query = (f'PREDICT LAST({target}, 0, 2) {rel_op} {const} '
             f'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert not response.ok
    assert col_err in response.message()
    assert const_err in response.message()

    query = (f'PREDICT SUM(TRANS.VALUE, 0, 2) FOR EACH USERS.USERS_ID '
             f'WHERE LAST({target}, -1, 0) {rel_op} {const}')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert not response.ok
    assert col_err in response.message()
    assert const_err in response.message()

    query = (f'PREDICT SUM(TRANS.VALUE, 0, 2) FOR EACH USERS.USERS_ID '
             f'WHERE USERS.DESC {rel_op} {const}')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert not response.ok
    assert const_err in response.message()


@pytest.mark.parametrize('target, const, ok', [
    ('TRANS.VALUE', '6.5', True),
    ('TRANS.VALUE', '1', True),
    ('TRANS.VALUE', '2020-01-01', False),
    ('TRANS.VALUE', '(\'text1\', \'text2\')', False),
    ('TRANS.TIME', '6.5', False),
    ('TRANS.TIME', '1', False),
    ('TRANS.TIME', '2020-01-01', True),
    ('TRANS.TIME', '(\'text1\', \'text2\')', False),
    ('TRANS.STORE', '6.5', True),
    ('TRANS.STORE', '1', True),
    ('TRANS.STORE', '2020-01-01', False),
    ('TRANS.STORE', '(\'text1\', \'text2\')', False),
])
@pytest.mark.parametrize('rel_op', ['CONTAINS', 'NOT CONTAINS'])
def test_strop_cannot_be_used_on_mismatched_cols_and_targets_array_targets(
    target: str,
    const: str,
    ok: bool,
    rel_op: str,
    create_mock_graph: Callable[..., GraphDefinition],
) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TypeValidator(graph, allow_array_targets=True)
    query = (f'PREDICT LIST_DISTINCT({target}, 0, 2) {rel_op} {const} '
             f'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    if ok:
        assert response.ok
    else:
        assert not response.ok
        assert re.search(r'Expression .* has type .* but constant',
                         response.message()) is not None


@pytest.mark.parametrize('target',
                         ['TRANS.VALUE', 'TRANS.TIME', 'TRANS.STORE'])
@pytest.mark.parametrize('rel_op', ['STARTS WITH', 'ENDS WITH'])
@pytest.mark.parametrize('const',
                         ['1', '6.5', '2020-01-01', '(\"text1\", \"text2\")'])
def test_start_ends_with_cannot_be_used_on_arrays_array_target(
    target: str,
    rel_op: str,
    const: str,
    create_mock_graph: Callable[..., GraphDefinition],
) -> None:
    graph = create_mock_graph()
    col_err = (f'Operator {rel_op} operates on strings but '
               f'LIST_DISTINCT({target},')
    const_err = f'Operator {rel_op} operates on strings but {const}'
    parser = PQLParser()
    validator = TypeValidator(graph, allow_array_targets=True)
    query = (f'PREDICT LIST_DISTINCT({target}, 0, 2) {rel_op} {const} '
             f'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert not response.ok
    assert col_err in response.message()
    assert const_err in response.message()


@pytest.mark.parametrize('target, const', [
    ('TRANS.VALUE', '2020-01-01'),
    ('TRANS.TIME', '1.5'),
    ('TRANS.STORE', '"text"'),
    ('TRANS.VALUE', '"text"'),
    ('TRANS.TIME', '2'),
    ('TRANS.STORE', '2020-01-01'),
])
@pytest.mark.parametrize('rel_op', [op.value for op in RelOp])
def test_condition_validation_relop(
    target: str,
    const: str,
    rel_op: str,
    create_mock_graph: Callable[..., GraphDefinition],
) -> None:
    col_err = (f'Expression LAST[(]{target}, .*[)] has type '
               f'.* but constant {const}')
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TypeValidator(graph)
    query = (f'PREDICT LAST({target}, 0, 2) {rel_op} {const} '
             f'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert not response.ok
    assert re.search(col_err, response.message()) is not None

    query = (f'PREDICT SUM(TRANS.VALUE, 0, 2) FOR EACH USERS.USERS_ID '
             f'WHERE LAST({target}, -1, 0) {rel_op} {const}')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert not response.ok
    assert re.search(col_err, response.message()) is not None


@pytest.mark.parametrize('target, const', [
    ('TRANS.FLAG', 'False'),
    ('TRANS.FLAG', 'True'),
    ('TRANS.FLAG', 'NULL'),
])
@pytest.mark.parametrize(
    'rel_op', [op.value for op in RelOp if op not in [RelOp.NEQ, RelOp.EQ]])
def test_condition_validation_relop_bool_and_null_invalid(
    target: str,
    const: str,
    rel_op: str,
    create_mock_graph: Callable[..., GraphDefinition],
) -> None:
    col_err = (f'Expression LAST[(]{target}, .*[)] has type '
               f'.* but operator {rel_op}')
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TypeValidator(graph)
    query = (f'PREDICT LAST({target}, 0, 2) {rel_op} {const} '
             f'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert not response.ok
    assert re.search(col_err, response.message()) is not None

    query = (f'PREDICT SUM(TRANS.VALUE, 0, 2) FOR EACH USERS.USERS_ID '
             f'WHERE LAST({target}, -1, 0) {rel_op} {const}')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert not response.ok
    assert re.search(col_err, response.message()) is not None


@pytest.mark.parametrize('target, const', [
    ('TRANS.FLAG', 'False'),
    ('TRANS.FLAG', 'True'),
    ('TRANS.FLAG', 'NULL'),
])
@pytest.mark.parametrize('rel_op',
                         [RelOp.NEQ.value, RelOp.EQ.value, 'IS', 'IS NOT'])
def test_condition_validation_relop_bool_and_null_valid(
    target: str,
    const: str,
    rel_op: str,
    create_mock_graph: Callable[..., GraphDefinition],
) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TypeValidator(graph)
    query = (f'PREDICT LAST({target}, 0, 2) {rel_op} {const} '
             f'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert response.ok

    query = (f'PREDICT SUM(TRANS.VALUE, 0, 2) FOR EACH USERS.USERS_ID '
             f'WHERE LAST({target}, -1, 0) {rel_op} {const}')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert response.ok


@pytest.mark.parametrize('target, const', [
    ('TRANS.VALUE', '(1.0,2.0)'),
    ('TRANS.VALUE', '(1,2)'),
])
@pytest.mark.parametrize('rel_op', [op.value for op in RelOp])
def test_condition_validation_relop_array(
    target: str,
    const: str,
    rel_op: str,
    create_mock_graph: Callable[..., GraphDefinition],
) -> None:
    col_err = f'Operator {rel_op} does not support operations on arrays.'
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TypeValidator(graph)
    query = (f'PREDICT LAST({target}, 0, 2) {rel_op} {const} '
             f'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert not response.ok
    assert col_err in response.message()

    query = (f'PREDICT SUM(TRANS.VALUE, 0, 2) FOR EACH USERS.USERS_ID '
             f'WHERE LAST({target}, -1, 0) {rel_op} {const}')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert not response.ok
    assert col_err in response.message()


@pytest.mark.parametrize('const', ['1', '2.5', '"text"', '2020-01-01'])
@pytest.mark.parametrize('rel_op', [op.value for op in MemberOp])
def test_condition_validation_memberop_not_array(
    const: str,
    rel_op: str,
    create_mock_graph: Callable[..., GraphDefinition],
) -> None:
    target = 'TRANS.VALUE'
    col_err = f'Operator {rel_op} expects an array constant but'
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TypeValidator(graph)
    query = (f'PREDICT LAST({target}, 0, 2) {rel_op} {const} '
             f'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert not response.ok
    assert col_err in response.message()

    query = (f'PREDICT SUM(TRANS.VALUE, 0, 2) FOR EACH USERS.USERS_ID '
             f'WHERE LAST({target}, -1, 0) {rel_op} {const}')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert not response.ok
    assert col_err in response.message()


@pytest.mark.parametrize('target, const', [
    ('TRANS.VALUE', '(2020-01-01,2020-01-02)'),
    ('TRANS.TIME', '(1.0,2.0)'),
    ('TRANS.VALUE', '(1.0,"text")'),
    ('TRANS.TIME', '(2020-01-01, NULL)'),
])
@pytest.mark.parametrize('rel_op', [op.value for op in MemberOp])
def test_condition_validation_memberop_array(
    target: str,
    const: str,
    rel_op: str,
    create_mock_graph: Callable[..., GraphDefinition],
) -> None:
    col_err = ('Expression .* has type .* but array '
               'contains element')
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TypeValidator(graph)
    query = (f'PREDICT LAST({target}, 0, 2) {rel_op} {const} '
             f'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert not response.ok
    assert re.search(col_err, response.message()) is not None

    query = (f'PREDICT SUM(TRANS.VALUE, 0, 2) FOR EACH USERS.USERS_ID '
             f'WHERE LAST({target}, -1, 0) {rel_op} {const}')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert not response.ok
    assert re.search(col_err, response.message()) is not None


def test_condition_validation_is_in_deprecation(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    target = 'TRANS.VALUE'
    col_err = 'Operator IS IN is deprecated'
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TypeValidator(graph)
    query = (f'PREDICT LAST({target}, 0, 2) IS IN (1.0, 2.0) '
             f'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert response.ok
    assert col_err in response.message()


@pytest.mark.parametrize('aggr', ['SUM', 'AVG'])
@pytest.mark.parametrize('col', ['TRANS.TIME', 'TRANS.FLAG'])
def test_aggregation_validation(
    aggr: str,
    col: str,
    create_mock_graph: Callable[..., GraphDefinition],
) -> None:
    graph = create_mock_graph()
    col_err = f'Aggregation {aggr} can only operate'
    parser = PQLParser()
    validator = TypeValidator(graph)
    query = f'PREDICT {aggr}({col}, 0, 2) FOR EACH USERS.USERS_ID'
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert not response.ok
    assert col_err in response.message()


@pytest.mark.parametrize('aggr', ['MIN', 'MAX'])
def test_aggregation_bool_minmax_warning(
    aggr: str,
    create_mock_graph: Callable[..., GraphDefinition],
) -> None:
    graph = create_mock_graph()
    col_err = f'Aggregation {aggr} is not intended to operate on bool'
    parser = PQLParser()
    validator = TypeValidator(graph)
    query = (f'PREDICT {aggr}(TRANS.FLAG, 0, 2) '
             f'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert col_err in response.message()


def test_nested_aggregation_validation(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    col_err = 'Aggregation SUM can only operate'
    parser = PQLParser()
    validator = TypeValidator(graph)
    query = ('PREDICT SUM(TRANS.VALUE, 0, 2) FOR EACH USERS.USERS_ID '
             'WHERE SUM(TRANS.TIME, -1, 0) > 0 AND '
             'AVG(TRANS.TIME, -1, 0) > 0')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_dtypes(parsed_query)
    response = validator.validate_dtypes(parsed_query)
    assert not response.ok
    assert col_err in response.message()


@pytest.mark.parametrize('col', ['TYPE', 'STORE'])
@pytest.mark.parametrize('rel_op', ['>=', '>', '<', '<='])
def test_relop_stype_validation(
    col: str,
    rel_op: str,
    create_mock_graph: Callable[..., GraphDefinition],
) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TypeValidator(graph)
    query = (f'PREDICT SUM(TRANS.VALUE WHERE TRANS.{col} '
             f'{rel_op} 1, 0, 2) FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_stypes(parsed_query)
    response = validator.validate_stypes(parsed_query)
    assert re.search(".* has semantic type .* But you are comparing it*",
                     response.message()) is not None


def test_nested_stype_validation(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TypeValidator(graph)
    query = ('PREDICT SUM(TRANS.VALUE WHERE TRANS.TYPE '
             '> 1 OR TRANS.STORE <= 2, 0, 2) FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_stypes(parsed_query)
    response = validator.validate_stypes(parsed_query)
    assert len(response.warnings) == 2
    assert re.search(".* has semantic type .* But you are comparing it*",
                     response.message()) is not None


@pytest.mark.parametrize('aggr', ['SUM', 'AVG', 'MIN', 'MAX'])
def test_aggregation_stype_validation(
    aggr: str,
    create_mock_graph: Callable[..., GraphDefinition],
) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TypeValidator(graph)
    query = (f'PREDICT {aggr}(TRANS.TYPE, 0, 2)'
             f'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_stypes(parsed_query)
    response = validator.validate_stypes(parsed_query)
    assert re.search("Aggregation .* is intended to operate on numerical *",
                     response.message()) is not None


@pytest.mark.parametrize('aggr', ['SUM', 'AVG'])
def test_aggregation_stype_fkey_validation(
    aggr: str,
    create_mock_graph: Callable[..., GraphDefinition],
) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TypeValidator(graph)
    query = (f'PREDICT {aggr}(TRANS.STORE, 0, 2)'
             f'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_stypes(parsed_query)
    response = validator.validate_stypes(parsed_query)
    assert not response.ok
    assert re.search("Aggregation .* is not intended to operate on foreign *",
                     response.message()) is not None


def test_aggregation_stype_list_validation(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TypeValidator(graph)
    query = ('PREDICT LIST_DISTINCT(TRANS.VALUE, 0, 2)'
             'FOR EACH USERS.USERS_ID')
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_stypes(parsed_query)
    response = validator.validate_stypes(parsed_query)
    assert re.search("Aggregation .* is intended to operate on categorical *",
                     response.message()) is not None


@pytest.mark.parametrize('query', [
    'PREDICT LAST(TRANS.TIME, 0, 2) FOR EACH USERS.USERS_ID',
    'PREDICT USERS.DESC FOR EACH USERS.USERS_ID'
])
def test_invalid_target_stype_validation(
    query: str,
    create_mock_graph: Callable[..., GraphDefinition],
) -> None:
    graph = create_mock_graph()
    parser = PQLParser()
    validator = TypeValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(query)
    validator.infer_stypes(parsed_query)
    response = validator.validate_stypes(parsed_query)
    assert not response.ok
    assert re.search("Target label has semantic type .*, which Kumo *",
                     response.message()) is not None


@pytest.mark.parametrize(
    'query, is_datetime',
    [('PREDICT SUM(TRANS.VALUE, 0, 2000000000000000000000000) '
      'FOR EACH USERS.USERS_ID', True),
     ('PREDICT SUM(TRANS.VALUE, 0, 20) > 2000000000000000000000000 '
      'FOR EACH USERS.USERS_ID', False),
     ('PREDICT SUM(TRANS.VALUE, 0, 20) FOR EACH USERS.USERS_ID '
      'WHERE SUM(TRANS.VALUE, -20000000000000000000000000000, 0) > 0', True),
     ('PREDICT LIST_DISTINCT(orders.item_id, 0, 13123112819128, '
      'hours) FOR EACH users.user_id', True),
     ('PREDICT LIST_DISTINCT(orders.item_id, 0, 1312311281912800, '
      'minutes) FOR EACH users.user_id', True)])
def test_large_int_constants(
    query: str,
    is_datetime: bool,
    create_mock_graph: Callable[..., GraphDefinition],
) -> None:
    graph = create_mock_graph()
    parser = PQLParser()

    validator = PredictiveQueryValidator(graph)
    if is_datetime:
        with pytest.raises(ValueError, match=r"Unsupported constant"):
            parsed_query = parser.to_parsed_predictive_query(query)
    else:
        parsed_query = parser.to_parsed_predictive_query(query)
        validated_query, response = validator.validate_predictive_query(
            parsed_query)
        assert not response.ok
        assert validated_query is None
        assert "Unsupported constant" in response.message()
