# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from kumoapi.json_serde import from_json, to_json
from kumoapi.pquery.AST import Column, Condition, Constant
from kumoapi.typing import Dtype, MemberOp, RelOp, StrOp


def test_missing_target():
    with pytest.raises(ValueError, match='Class \'Condition\' is missing'):
        Condition()


def test_missing_op():
    with pytest.raises(ValueError, match='is not found in the relational'):
        Condition(target=Column(fqn='table.col'))


@pytest.mark.parametrize('rel_op', RelOp)
def test_relop_serialization(rel_op):
    condition = Condition(target=Column(fqn='table.age'), op=rel_op, value=20)
    cfg = condition.to_string()
    assert cfg == f"table.age {rel_op.value} 20"
    cfg = to_json(condition)
    assert condition == from_json(cfg, Condition)


@pytest.mark.parametrize('str_op', StrOp)
def test_strop_serialization(str_op):
    condition = Condition(target=Column(fqn='table.age'), op=str_op,
                          value=Constant.from_value("Text"))
    cfg = condition.to_string()
    assert cfg == f"table.age {str_op.value} \"Text\""
    cfg = to_json(condition)
    assert condition == from_json(cfg, Condition)


@pytest.mark.parametrize('rel_op, output', [(RelOp.EQ, "IS"),
                                            (RelOp.NEQ, "IS NOT")])
def test_null_serialization(rel_op, output):
    condition = Condition(target=Column(fqn='table.age'), op=rel_op,
                          value=Constant(value=None))
    cfg = condition.to_string()
    assert cfg == f"table.age {output} NULL"
    assert condition.to_string(
        rich=True) == (f'table.age [bold]{output}[/bold] [bold]NULL[/bold]')
    cfg = to_json(condition)
    assert condition == from_json(cfg, Condition)


@pytest.mark.parametrize('mem_op', MemberOp)
def test_memberop_serialization(mem_op):
    condition = Condition(target=Column(fqn='table.age'), op=mem_op,
                          value=Constant.from_value([1, 2]))
    cfg = condition.to_string()
    assert cfg == f"table.age {mem_op.value} (1, 2)"
    cfg = to_json(condition)
    assert condition == from_json(cfg, Condition)


def test_date_serialization():
    date = pd.to_datetime('2022-07-01')
    condition = Condition(
        target=Column(fqn='table.time'), op=RelOp.EQ,
        value=Constant(value='2022-07-01', dtype_maybe=Dtype.time))
    assert condition.get_typed_value() == date
    cfg = condition.to_string()
    assert cfg == 'table.time = 2022-07-01'
    cfg = to_json(condition)
    assert condition == from_json(cfg, Condition)


def test_like_serialization():
    condition = Condition(target=Column(fqn='table.age'), op='LIKE',
                          value=Constant.from_value("%Text"))
    cfg = condition.to_string()
    assert cfg == "table.age ENDS WITH \"Text\""
    cfg = to_json(condition)
    assert condition == from_json(cfg, Condition)

    condition = Condition(target=Column(fqn='table.age'), op='LIKE',
                          value=Constant.from_value("Text%"))
    cfg = condition.to_string()
    assert cfg == "table.age STARTS WITH \"Text\""
    cfg = to_json(condition)
    assert condition == from_json(cfg, Condition)

    condition = Condition(target=Column(fqn='table.age'), op='LIKE',
                          value=Constant.from_value("%Text%"))
    cfg = condition.to_string()
    assert cfg == "table.age CONTAINS \"Text\""
    cfg = to_json(condition)
    assert condition == from_json(cfg, Condition)
