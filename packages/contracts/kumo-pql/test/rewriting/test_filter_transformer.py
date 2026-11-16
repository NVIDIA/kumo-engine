# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from kumoapi.pquery.AST import Column, Condition, Filter, LogicalOperation
from kumoapi.typing import BoolOp, RelOp
from kumopql.rewriting.filter_splitter import FilterSplitter, SplitFilter


class TrivialFilterSplitter(FilterSplitter):
    def _condition_base_case(
        self,
        node: Condition,
        original_filter: Condition | LogicalOperation | None,
        **kwargs: Any,
    ) -> SplitFilter:
        """Base case for the recursive filter splitting. A condition is not
        applicable if its value is not 42. Otherwise, it should be applicable.

        Args:
            node: The condition to split.
            original_filter: Full input filter.
            **kwargs: Passed to children.

        Returns:
            SplitFilter: A split filter describing whether the input condition
            is applicable or not.
        """
        if node.value.typed_value() == 42:
            return SplitFilter(
                applicable_filter=node,
                non_applicable_filter=None,
                rewritten=False,
                n_applicable_conditions=1,
                n_non_applicable_conditions=0,
                non_applicable_is_valid=True,
                original_filter=original_filter,
            )
        else:
            return SplitFilter(
                applicable_filter=None,
                non_applicable_filter=node,
                rewritten=False,
                n_applicable_conditions=0,
                n_non_applicable_conditions=1,
                non_applicable_is_valid=True,
                original_filter=original_filter,
            )


def test_filter_splitter() -> None:
    # Test single fully static condition
    splitter = TrivialFilterSplitter()

    entity = Filter(
        target=Column(fqn="entity_table.entity_id"),
        condition=LogicalOperation(
            left=Condition(target=Column(fqn='entity_table.entity_id'),
                           op=RelOp.EQ, value=42), bool_op=BoolOp.AND,
            right=Condition(target=Column(fqn='entity_table.entity_id'),
                            op=RelOp.EQ, value=43)),
    )

    result = splitter._recursively_split_filter(entity.condition,
                                                entity.condition)
    assert isinstance(entity.condition, LogicalOperation)
    assert result == SplitFilter(
        applicable_filter=entity.condition.left,
        non_applicable_filter=entity.condition.right,
        rewritten=False,
        n_applicable_conditions=1,
        n_non_applicable_conditions=1,
        non_applicable_is_valid=True,
        original_filter=entity.condition,
    )
