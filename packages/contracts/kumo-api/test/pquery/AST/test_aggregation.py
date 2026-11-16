# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from kumoapi.json_serde import from_json, to_json
from kumoapi.pquery.AST import Aggregation, Column, Condition, Filter
from kumoapi.pquery.AST.date_offset_range import DateOffsetRange
from kumoapi.typing import AggregationType, Dtype, RelOp, Stype, TimeUnit


def test_missing_target():
    with pytest.raises(ValueError, match='Class \'Aggregation\' is missing'):
        Aggregation()


def test_missing_aggr():
    with pytest.raises(ValueError, match='is not a valid AggregationType'):
        Aggregation(target=Column(fqn='table.col'))


@pytest.mark.parametrize('aggr', AggregationType)
@pytest.mark.parametrize('time_unit', TimeUnit)
@pytest.mark.parametrize('time_col', ['LOAN.DATE', None])
@pytest.mark.parametrize('dtype_maybe', [Dtype.float, None])
@pytest.mark.parametrize('stype_maybe', [Stype.numerical, None])
def test_aggregation_serialization(aggr, time_unit, time_col, dtype_maybe,
                                   stype_maybe):
    aggregation = Aggregation(
        target=Column(fqn='LOAN.AMOUNT'),
        aggr=aggr,
        aggr_time_range=DateOffsetRange(-30, 0, time_unit),
        time_col=time_col,
        dtype_maybe=dtype_maybe,
        stype_maybe=stype_maybe,
    )
    assert str(aggregation) == (f'{aggr.value}(LOAN.AMOUNT, -30, 0, '
                                f'{time_unit.value})')
    assert aggregation.to_string(
        rich=True) == (f'[bold]{aggr.value}[/bold](LOAN.AMOUNT, -30, 0, '
                       f'{time_unit.value})')
    cfg = to_json(aggregation)
    assert aggregation == from_json(cfg, Aggregation)

    condition = Condition(target=aggregation, op=RelOp.GT, value=0)
    assert str(condition) == (f'{aggr.value}(LOAN.AMOUNT, -30, 0, '
                              f'{time_unit.value}) > 0')
    assert condition.to_string(
        rich=True) == (f'[bold]{aggr.value}[/bold](LOAN.AMOUNT, -30, 0, '
                       f'{time_unit.value}) > 0')
    cfg = to_json(condition)
    assert condition == from_json(cfg, Condition)


def test_open_date_serialization():
    aggr = Aggregation(
        target=Column(fqn='LOAN.AMOUNT'),
        aggr=AggregationType.SUM,
        aggr_time_range=DateOffsetRange(None, 0, TimeUnit.DAYS),
    )
    assert str(aggr) == 'SUM(LOAN.AMOUNT, -INF, 0, days)'
    assert aggr.to_string(
        rich=True) == ('[bold]SUM[/bold](LOAN.AMOUNT, -INF, 0, days)')
    cfg = to_json(aggr)
    assert aggr == from_json(cfg, Aggregation)


def test_non_inf_date_offset_range():
    aggr = Aggregation(
        target=Column(fqn='LOAN.AMOUNT'),
        aggr=AggregationType.SUM,
        aggr_time_range=DateOffsetRange(None, 0, TimeUnit.DAYS),
    )
    assert aggr.non_inf_date_offset_range() is None
    aggr = Aggregation(
        target=Column(fqn='LOAN.AMOUNT'),
        aggr=AggregationType.SUM,
        aggr_time_range=DateOffsetRange(-30, 0, TimeUnit.DAYS),
    )
    assert aggr.non_inf_date_offset_range() == DateOffsetRange(
        -30, 0, TimeUnit.DAYS)


def test_is_static():
    aggr = Aggregation(
        target=Column(fqn='LOAN.AMOUNT'),
        aggr=AggregationType.SUM,
        aggr_time_range=DateOffsetRange(None, 0, TimeUnit.DAYS),
    )
    assert not aggr.is_static
    aggr = Aggregation(
        target=Column(fqn='LOAN.AMOUNT'),
        aggr=AggregationType.LIST_DISTINCT,
    )
    assert aggr.is_static


def test_get_target_column_name():
    aggr = Aggregation(
        target=Column(fqn='LOAN.AMOUNT'),
        aggr=AggregationType.SUM,
        aggr_time_range=DateOffsetRange(None, 0, TimeUnit.DAYS),
    )
    assert aggr.get_target_column_name() == 'LOAN.AMOUNT'
    aggr = Aggregation(
        target=Filter(
            target=Column(fqn='LOAN.AMOUNT'),
            condition=Condition(target=Column(fqn='blah.blah'), op='>',
                                value=20)),
        aggr=AggregationType.SUM,
        aggr_time_range=DateOffsetRange(-30, 0, TimeUnit.DAYS),
    )
    assert aggr.get_target_column_name() == 'LOAN.AMOUNT'


def test_all_time_columns():
    condition1 = Condition(
        target=Aggregation(
            target=Column(fqn='LOAN.AMOUNT'),
            aggr=AggregationType.SUM,
            aggr_time_range=DateOffsetRange(None, 0, TimeUnit.DAYS),
            time_col='table.time_column',
        ), op='>', value=20)
    condition2 = Condition(
        target=Aggregation(
            target=Column(fqn='LOAN.AMOUNT'),
            aggr=AggregationType.SUM,
            aggr_time_range=DateOffsetRange(None, 0, TimeUnit.DAYS),
            time_col='table2.time_column2',
        ), op='<', value=0)
    assert sorted((condition1 & condition2).all_time_columns) == [
        'table.time_column', 'table2.time_column2'
    ]
