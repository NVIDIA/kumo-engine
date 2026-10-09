# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, List, Optional, Tuple, Union

import pydantic
from pydantic.dataclasses import dataclass

from kumoapi.pquery.AST.ast_node import ASTNode
from kumoapi.pquery.AST.constant import Constant
from kumoapi.pquery.utils import maybe_bold
from kumoapi.typing import _FORBID_EXTRA_CONFIG, BoolOp, MemberOp, RelOp, StrOp

SQL_NULL = 'NULL'
SQL_IS = 'IS'
SQL_IS_NOT = 'IS NOT'
SQL_LIKE = 'LIKE'
SQL_NOT_LIKE = 'NOT LIKE'

logger = logging.getLogger(__name__)


@dataclass(repr=False, config=_FORBID_EXTRA_CONFIG)
class Condition(ASTNode):
    r"""Creates an atomic description of a condition on :obj:`target`.

    Args:
        target: `ASTNode` that computes a value on which the filter is applied.
        op: Relational or string operator for comparison.
        value: Value to compare against.
        input_op: Sometimes, `op` is changed as a result of rewriting, e.g.
            with `like_to_str_op`. If not-:obj:`None`, this is the original
            user input to be used, e.g. when reporting errors.
            (default: :obj:`None`)
    """
    target: Union['Aggregation', 'Column', 'Join', None] = None
    op: Union[RelOp, MemberOp, StrOp, str] = ''
    value: Union[Constant, int, float, str, bool, None] = None
    input_op: Optional[str] = None

    def __post_init__(self) -> None:
        if self.target is None:
            raise ValueError(f"Class '{self.__class__.__name__}' is missing a "
                             f"target.")
        # Backward compatibility with older configs
        if not isinstance(self.value, (Constant, dict)):
            self.value = Constant.from_value(self.value)
        if isinstance(self.op, str):
            if self.op in set(item.value for item in RelOp):
                self.op = RelOp(self.op)
            elif self.op in set(item.value for item in StrOp):
                self.op = StrOp(self.op)
            elif self.op in set(item.value for item in MemberOp):
                self.op = MemberOp(self.op)
            elif self.op == SQL_LIKE:
                self.input_op = self.op
                self.op, self.value.value = self.like_to_str_op()
            elif self.op == SQL_NOT_LIKE:
                self.input_op = self.op
                self.op, self.value.value = self.like_to_str_op(negate=True)
            else:
                raise ValueError(f"{self.op} is not found in the relational "
                                 f"nor the string operators.")
        super().__post_init__()

    @property
    def children(self) -> List[Union['ASTNode', int, float, str, bool]]:
        assert self.target is not None
        assert self.value is not None
        return [self.target, self.value]

    def __and__(
            self, f2: Union['Condition',
                            'LogicalOperation']) -> 'LogicalOperation':
        return LogicalOperation(
            left=self,
            bool_op=BoolOp.AND,
            right=f2,
        )

    def __or__(
            self, f2: Union['Condition',
                            'LogicalOperation']) -> 'LogicalOperation':
        return LogicalOperation(
            left=self,
            bool_op=BoolOp.OR,
            right=f2,
        )

    def __invert__(self) -> 'LogicalOperation':
        return LogicalOperation(left=self, bool_op=BoolOp.NOT)

    def get_typed_value(self) -> Any:
        assert isinstance(self.value, Constant)
        return self.value.typed_value()

    def to_string(self, rich: bool = False) -> str:
        r"""Creates a predictive query statement from the filter. """
        assert isinstance(self.op, (RelOp, MemberOp, StrOp))
        assert isinstance(self.value, Constant)
        op = self.op.value
        if self.value.dtype_maybe is None:
            value = maybe_bold(SQL_NULL, rich)
            op = SQL_IS if self.op == RelOp.EQ else op
            op = SQL_IS_NOT if self.op == RelOp.NEQ else op
            op = maybe_bold(op, rich)
        else:
            assert isinstance(self.value, Constant)
            value = self.value.to_string(rich=rich)
        assert self.target is not None
        return f'{self.target.to_string(rich=rich)} {op} {value}'

    def like_to_str_op(self, negate=False) -> Tuple[StrOp, str]:
        # Check the string whether starts with '%', ends with '%', or both
        str_op = None
        assert isinstance(self.value, Constant)
        value = self.value.typed_value()
        assert isinstance(value, str)
        if value.endswith('%') and value.startswith('%'):
            str_op = StrOp.CONTAINS
        elif value.endswith('%'):
            str_op = StrOp.STARTS_WITH
        elif value.startswith('%'):
            str_op = StrOp.ENDS_WITH

        if str_op is None:
            raise ValueError(
                'Condition must contain a % at the start ',
                'end, or both ends of the value being compared when using '
                'the LIKE operator')

        # Remove the '%' to get the original value
        assert isinstance(self.value.value, str)
        new_value = self.value.value.replace('%', '')

        # TODO: add support for not starts with, not ends with
        if negate:
            # raise exception if not strop.contains
            if str_op != StrOp.CONTAINS:
                raise ValueError("'NOT LIKE' only works for values that start "
                                 "with '%' and end with '%'.")
            str_op = StrOp.NOT_CONTAINS
        return str_op, new_value


from kumoapi.pquery.AST.aggregation import Aggregation  # noqa: E402
from kumoapi.pquery.AST.column import Column  # noqa: E402
from kumoapi.pquery.AST.join import Join  # noqa: E402
from kumoapi.pquery.AST.logical_operation import LogicalOperation  # noqa: E402

if pydantic.__version__.startswith('1.'):
    Condition.__pydantic_model__.update_forward_refs()
