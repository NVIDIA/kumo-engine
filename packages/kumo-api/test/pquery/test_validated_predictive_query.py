# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumoapi.json_serde import from_json, to_json
from kumoapi.pquery import ParsedPredictiveQuery, ValidatedPredictiveQuery
from kumoapi.pquery.AST import (
    Aggregation,
    Column,
    Condition,
    Constant,
    DateOffsetRange,
    Filter,
    Join,
    LogicalOperation,
)
from kumoapi.typing import AggregationType, BoolOp, Dtype, TimeUnit


def test_parsed_predictive_query_serialization():
    target = Aggregation(
        target=Column(fqn='LOAN.AMOUNT'),
        aggr=AggregationType.LIST_DISTINCT,
        aggr_time_range=DateOffsetRange(-30, 0, TimeUnit.DAYS),
        time_col='LOAN.DATE',
        dtype_maybe=Dtype.int,
        stype_maybe=None,
    )
    entity = Column(fqn='USER.USER_ID')
    parsed_query = ParsedPredictiveQuery(
        entity_ast=entity,
        target_ast=target,
        whatif_ast=None,
        top_k=10,
        problem_type='RANK',
        for_each='FOR EACH',
    )
    cfg = to_json(parsed_query)
    assert parsed_query == from_json(cfg, ParsedPredictiveQuery)
    assert parsed_query.to_string() == (
        "PREDICT LIST_DISTINCT(LOAN.AMOUNT, -30, 0, days) "
        "RANK TOP 10 FOR EACH USER.USER_ID")
    assert parsed_query.to_string(
        exclude_predict=True) == ("LIST_DISTINCT(LOAN.AMOUNT, -30, 0, days) "
                                  "RANK TOP 10 FOR EACH USER.USER_ID")
    assert parsed_query.to_string(
        rich=True) == ("[bold]PREDICT[/bold] [bold]LIST_DISTINCT[/bold]"
                       "(LOAN.AMOUNT, -30, 0, days) RANK [bold]TOP 10[/bold] "
                       "[bold]FOR EACH[/bold] USER.USER_ID")


def test_validated_predictive_query_serialization():
    target = Aggregation(
        target=Column(fqn='LOAN.AMOUNT'),
        aggr=AggregationType.LIST_DISTINCT,
        aggr_time_range=DateOffsetRange(-30, 0, TimeUnit.DAYS),
        time_col='LOAN.DATE',
        dtype_maybe=Dtype.int,
        stype_maybe=None,
    )
    entity = Column(fqn='USER.USER_ID')
    validated_query = ValidatedPredictiveQuery(
        entity_ast=entity,
        target_ast=target,
        whatif_ast=None,
        top_k=10,
        problem_type='RANK',
        for_each='FOR EACH',
    )
    cfg = to_json(validated_query)
    assert validated_query == from_json(cfg, ValidatedPredictiveQuery)


def test_exclude_cols_dict():
    target = LogicalOperation(
        left=Condition(
            target=Column(fqn='LOAN.AMOUNT'),
            op='>',
            value=Constant.from_value(1),
        ), bool_op=BoolOp.AND, right=LogicalOperation(
            left=Condition(
                target=Column(fqn='LOAN.TYPE'),
                op='=',
                value=Constant.from_value("dummy"),
            ),
            bool_op=BoolOp.OR,
            right=Condition(
                target=Join(rhs_target=Column(fqn='USER.SCORE')),
                op='>',
                value=Constant.from_value(10),
            ),
        ))
    entity = Column(fqn='USER.USER_ID')
    validated_query = ValidatedPredictiveQuery(
        entity_ast=entity,
        target_ast=target,
        whatif_ast=None,
        for_each='FOR EACH',
    )
    exclude_cols_dict = validated_query.get_exclude_cols_dict()
    for v in exclude_cols_dict.values():
        v.sort()
    assert exclude_cols_dict == {
        'LOAN': ['AMOUNT', 'TYPE'],
        'USER': ['SCORE'],
    }

    filtered = Filter(
        target=Column(fqn='LOAN.AMOUNT'),
        condition=Condition(
            target=Column(fqn='LOAN.TYPE'),
            op='=',
            value=Constant.from_value('active'),
        ),
    )
    # Filter is accepted by the runtime visitor but omitted from this field's
    # declared union.
    validated_query.target_ast = filtered  # ty: ignore[invalid-assignment]
    assert validated_query.get_exclude_cols_dict() == {
        'LOAN': ['AMOUNT'],
    }

    validated_query.target_ast = Aggregation(
        target=filtered,
        aggr=AggregationType.SUM,
        aggr_time_range=DateOffsetRange(-30, 0, TimeUnit.DAYS),
    )
    assert validated_query.get_exclude_cols_dict() == {}


def test_to_autoregressive_query():
    query = ParsedPredictiveQuery(
        entity_ast=Column(fqn='USER.USER_ID'),
        target_ast=Aggregation(
            target=Column(fqn='LOAN.AMOUNT'),
            aggr=AggregationType.SUM,
            aggr_time_range=DateOffsetRange(0, 7, TimeUnit.DAYS),
        ),
    )
    assert query.to_autoregressive_query().to_string() == (
        "PREDICT SUM(LOAN.AMOUNT, 0, 7, days) FOR EACH USER.USER_ID")

    query.target_ast = Condition(
        target=query.target_ast,  # type: ignore
        op='=',
        value=0,
    )
    assert query.to_autoregressive_query().to_string() == (
        "PREDICT SUM(LOAN.AMOUNT, 0, 7, days) FOR EACH USER.USER_ID")

    query.target_ast = LogicalOperation(
        left=query.target_ast,
        bool_op=BoolOp.NOT,
    )
    assert query.to_autoregressive_query().to_string() == (
        "PREDICT SUM(LOAN.AMOUNT, 0, 7, days) FOR EACH USER.USER_ID")

    query.entity_ast = Filter(
        target=query.entity_ast,  # type: ignore
        condition=Condition(
            target=Aggregation(
                target=Column(fqn='LOAN.AMOUNT'),
                aggr=AggregationType.SUM,
                aggr_time_range=DateOffsetRange(-7, 0, TimeUnit.DAYS),
            ),
            op='>',
            value=0,
        ),
    )
    assert query.to_autoregressive_query().to_string() == (
        "PREDICT SUM(LOAN.AMOUNT, 0, 7, days) FOR EACH USER.USER_ID")
