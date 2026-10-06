# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

from dataclasses import field
from typing import List, Optional, Union

import pydantic
from pydantic.dataclasses import dataclass

from kumoapi.pquery.AST.ast_node import ASTNode
from kumoapi.pquery.utils import maybe_bold
from kumoapi.typing import _FORBID_EXTRA_CONFIG, BoolOp


@dataclass(repr=False, config=_FORBID_EXTRA_CONFIG)
class LogicalOperation(ASTNode):
    r"""A `LogicalOperation` is the combination of :obj:`Condition` and
    boolean operators(:obj:`bool_op`) like `&`, `|` or `~`.

    .. code-block::python
        # conditions table.A>10 and table.B<100 are combined
        # using boolean operation &.
        c1 = Condition(target=Column(fqn='table.A'), op='>', value=10)
        c2 = Condition(target=Column(fqn='table.B'), op='<', value=100)
        combined = c1 & c2

    Args:
        left (condition or logical operation): AST corresponding to the
            left side of the expression.
        bool_op (BoolOp): Boolean operator.
        right (condition or logical operation, optional): AST corresponding
            to the right side of the expression. Should be :obj:`None`
            if and only if `bool_op` is :obj:`BoolOp.NOT`.
    """
    left: Union['Condition', 'LogicalOperation', None] = None
    bool_op: BoolOp = field(default=BoolOp.NOT)
    right: Optional[Union['Condition', 'LogicalOperation']] = None

    def __post_init__(self) -> None:
        if self.left is None:
            raise ValueError(f"Class '{self.__class__.__name__}' is missing a "
                             f"left-hand side (argument 'left').")
        if self.bool_op != BoolOp.NOT and self.right is None:
            raise ValueError(
                f"Nested '{self.__class__.__name__}' is missing a "
                f"right-hand side '{self.__class__.__name__}' since "
                "the boolean operator is 'AND' or 'OR'.")
        if self.bool_op == BoolOp.NOT and self.right is not None:
            raise ValueError(
                f"Nested '{self.__class__.__name__}' shouldn't have a "
                f"right-hand side '{self.__class__.__name__}' since "
                "boolean operator is 'NOT'.")
        super().__post_init__()

    @property
    def children(self) -> List['ASTNode']:
        assert self.left is not None
        if self.right is not None:
            return [self.left, self.right]
        else:
            return [self.left]

    def __and__(
            self, f2: Union['Condition',
                            'LogicalOperation']) -> 'LogicalOperation':
        return type(self)(left=self, bool_op=BoolOp.AND, right=f2)

    def __or__(
            self, f2: Union['Condition',
                            'LogicalOperation']) -> 'LogicalOperation':
        return type(self)(left=self, bool_op=BoolOp.OR, right=f2)

    def __invert__(self) -> 'LogicalOperation':
        return type(self)(left=self, bool_op=BoolOp.NOT)

    def to_string(self, rich: bool = False) -> str:
        r"""Creates a predictive query statement from the filter."""
        assert self.left is not None
        if self.bool_op == BoolOp.NOT:
            return (f'{maybe_bold("NOT", rich)} '
                    f'({self.left.to_string(rich=rich)})')

        assert self.right is not None
        return (f'({self.left.to_string(rich=rich)}) '
                f'{maybe_bold(self.bool_op.value, rich)} '
                f'({self.right.to_string(rich=rich)})')


from kumoapi.pquery.AST.condition import Condition  # noqa: E402

if pydantic.__version__.startswith('1.'):
    LogicalOperation.__pydantic_model__.update_forward_refs()
