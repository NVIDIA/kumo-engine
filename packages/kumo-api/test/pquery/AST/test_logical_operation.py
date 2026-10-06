# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

import pytest

from kumoapi.json_serde import from_json, to_json
from kumoapi.pquery.AST import Column, Condition, LogicalOperation
from kumoapi.typing import BoolOp


def test_missing_left():
    with pytest.raises(ValueError,
                       match='Class \'LogicalOperation\' is missing'):
        LogicalOperation()


def test_logical_operations():
    filter = LogicalOperation(
        left=LogicalOperation(
            left=LogicalOperation(
                left=Condition(target=Column(fqn='table.age'), op='>',
                               value=20), bool_op=BoolOp.AND,
                right=Condition(target=Column(fqn='table.type'), op='=',
                                value="X")), bool_op=BoolOp.OR,
            right=LogicalOperation(
                left=Condition(target=Column(fqn='table.age'), op='<',
                               value='X'), bool_op=BoolOp.AND,
                right=Condition(target=Column(fqn='table.name'), op='LIKE',
                                value="A%"))), bool_op=BoolOp.NOT)

    assert filter.left is not None
    assert isinstance(filter.left, LogicalOperation)
    assert filter.left.left is not None
    assert filter.left.right is not None
    assert isinstance(filter.left.left, LogicalOperation)
    assert isinstance(filter.left.right, LogicalOperation)
    assert filter.left.left.left is not None
    assert filter.left.left.right is not None
    assert filter.left.right.right is not None
    assert filter.right is None

    assert (filter.to_string() == 'NOT (((table.age > 20) AND '
            '(table.type = "X")) OR ((table.age < "X") AND (table.name '
            'STARTS WITH "A")))')
    assert (filter.to_string(
        rich=True) == '[bold]NOT[/bold] (((table.age > 20) [bold]AND[/bold] '
            '(table.type = "X")) [bold]OR[/bold] ((table.age < "X") '
            '[bold]AND[/bold] (table.name STARTS WITH "A")))')
    cfg = to_json(filter)
    assert filter == from_json(cfg, LogicalOperation)


def test_logical_operations_magic_methods():
    filter = ~(
        (Condition(target=Column(fqn='table.age'), op='>', value=20)
         & Condition(target=Column(fqn='table.type'), op='=', value="X")) |
        (Condition(target=Column(fqn='table.age'), op='<', value='X')
         & Condition(target=Column(fqn='table.name'), op='LIKE', value="A%")))

    assert filter.left is not None
    assert isinstance(filter.left, LogicalOperation)
    assert filter.left.left is not None
    assert filter.left.right is not None
    assert isinstance(filter.left.left, LogicalOperation)
    assert isinstance(filter.left.right, LogicalOperation)
    assert filter.left.left.left is not None
    assert filter.left.left.right is not None
    assert filter.left.right.right is not None
    assert filter.right is None

    assert (filter.to_string() == 'NOT (((table.age > 20) AND '
            '(table.type = "X")) OR ((table.age < "X") AND (table.name '
            'STARTS WITH "A")))')
    cfg = to_json(filter)
    assert filter == from_json(cfg, LogicalOperation)

    filter = (
        (~Condition(target=Column(fqn='table.age'), op='>', value=20)
         | ~Condition(target=Column(fqn='table.type'), op='=', value="X"))
        &
        (~Condition(target=Column(fqn='table.age'), op='<', value='X')
         | ~Condition(target=Column(fqn='table.name'), op='LIKE', value="A%")))

    assert (filter.to_string() == '((NOT (table.age > 20)) OR '
            '(NOT (table.type = "X"))) AND ((NOT (table.age < "X")) OR '
            '(NOT (table.name STARTS WITH "A")))')
    cfg = to_json(filter)
    assert filter == from_json(cfg, LogicalOperation)
