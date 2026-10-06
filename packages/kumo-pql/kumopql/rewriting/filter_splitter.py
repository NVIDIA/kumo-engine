# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from kumoapi.pquery.AST import Condition, LogicalOperation
from kumoapi.typing import BoolOp, Dtype, Stype

# If the non-applicable part of the filter contains more than 6 conditions,
# we do not allow filter overrides. Instead, we instruct the user to report to
# Kumo. We can most definitely support more than 6 conditions in the backend
# but it'd look ugly in the UI. We can increase this as required.
MAX_N_CONDITIONS = 6


@dataclass
class SplitFilter:
    r"""Dataclass representing a filter split into two parts. One part is
    called applicable, the other non-applicable. Combining the two with AND
    should be equivalent to using the original filter.

    There are currently two use cases for this filter splitting:
        1. BP filter transformation, where we split the target filter into the
           subset that can be applied at batch prediction time and the subset
           that cannot.
        2. Static filter transformation, where we split the entity filter into
           the subset that is statically applicable at query time and the
           subset that contains conditions that may not be statically evaluated
           (e.g. because they include a temporal component).

    Args:
        applicable_filter (Optional[ASTNode]): If not :obj:`None`, the
            part of the filter that meets the applicability criteria.
        non_applicable_filter (Optional[ASTNode]): If not :obj:`None`, the
            part of the filter that does not meet the applicability criteria.
            Should be ignored with `non_applicable_is_valid` is :obj:`True`
            because it experienced an exponential blow-up during rewriting and
            was set to :obj:`None`.
        rewritten (bool): Indicates if any rewriting took place.
        n_applicable_conditions (int): Number of conditions in the applicable
            part of the query. Here to prevent an exponential blow-up.
        n_non_applicable_conditions (int): Number of conditions in the
            non-applicable part of the query. Here to prevent an exponential
            blow-up.
        non_applicable_is_valid (bool): :obj:`False` if the non-applicable part
            of the query had to be ommitted because of an exponential blow-up
            in its size. That happens when the total number of conditions in
            both applicable and non-applicable side exceed MAX_N_CONDITIONS.
            :obj:`True` otherwise. The applicable part of the filter does
            not grow during rewriting and is thus not tracked.
    """
    applicable_filter: Condition | LogicalOperation | None
    non_applicable_filter: Condition | LogicalOperation | None
    rewritten: bool
    n_applicable_conditions: int
    n_non_applicable_conditions: int
    non_applicable_is_valid: bool
    original_filter: Condition | LogicalOperation | None


class FilterSplitter:
    """Base class to generate a filter splitter that splits a Filter AST node
    into a subset of conditions that are applicable and a subset that is not
    based on the `_condition_base_case` method which is the only required
    method to implement for subclasses.

    If the `_condition_base_case` needs extra information, you can pass
    in additional kwargs from `_recursively_split_filter`.

    It is recommended but not required to implement a public entry method
    that calls `_recursively_split_filter` internally.

    Example Usage:

    .. code-block::python
        class TrivialFilterSplitter(FilterSplitter):
            def split(self, node: ASTNode) -> SplitFilter:
                if isinstance(node, Filter):
                    return self._recursively_split_filter(
                        node.condition, node.condition)
                if isinstance(node, Join):
                    return self.split(node.rhs_target)
                if isinstance(node, Aggregation):
                    return self.split(node.target)
                if (isinstance(node, Column)
                      or isinstance(node, LogicalOperation)):
                    return SplitFilter(
                        applicable_filter=None,
                        non_applicable_filter=None,
                        rewritten=False,
                        n_applicable_conditions=0,
                        n_non_applicable_conditions=0,
                        non_applicable_is_valid=True,
                        original_filter=original_filter,
                    )
                assert False, f"Unknown node type {type(node)}."

            def _condition_base_case(
                self,
                node: Condition,
                original_filter: Optional[Union[Condition, LogicalOperation]],
                **kwargs,
            ) -> SplitFilter:
                if node.value == 42:
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
    """
    def _condition_base_case(
        self,
        node: Condition,
        original_filter: Condition | LogicalOperation | None,
        **kwargs: Any,
    ) -> SplitFilter:
        """Base case for the recursive filter splitting. This function should
        return a SplitFilter for the base case with the node in the
        `applicable_filter` field if it is applicable or
        `non_applicable_filter` field if it is not, and
        `n_applicable_conditions`
        and `n_non_applicable_conditions` set to 1 and 0 respectively if the
        node is in the `applicable_filter` field or 0 and 1 respectively if it
        is in the `non_applicable_filter` field.

        The `non_applicable_is_valid` field should be set to :obj:`True` if
        the non-applicable filter is valid, i.e. if it did not experience an
        exponential blow-up during rewriting.

        Args:
            node (Condition): The leaf condition to evaluate.
            original_filter (Optional[Union[Condition, LogicalOperation]]): The
                original filter that we are splitting.
            **kwargs: Passed to children.

        Returns:
            A split filter describing whether this condition only
            is applicable or not.
        """
        raise NotImplementedError

    def _recursively_split_filter(
        self,
        node: Condition | LogicalOperation,
        original_filter: Condition | LogicalOperation | None,
        **kwargs: Any,
    ) -> SplitFilter:
        """Subcomponent of the filter transformer that takes a Filter Condition
        or Logical Operation (Filter.condition) and splits
        it into applicable and non-applicable parts based on the
        `_condition_base_case` method.

        Args:
            node: The condition or logical operation to split.
            original_filter: The original filter that we are splitting.
            **kwargs: Passed to children.

        Raises:
            ValueError: If the input node is invalid.

        Returns:
            The final split filter.
        """
        if isinstance(node, Condition):
            return self._condition_base_case(node, original_filter, **kwargs)
        assert isinstance(node, LogicalOperation)
        if node.bool_op == BoolOp.NOT:
            if isinstance(node.left, LogicalOperation):
                # we push down the negation to the leaves
                # Trivial case: two nested negations, we just drop them.
                if node.left.bool_op == BoolOp.NOT:
                    split_filter = self._recursively_split_filter(
                        node.left.left, original_filter, **kwargs)
                    split_filter.rewritten = True
                    return split_filter
                # For OR and AND, we use De-Morgan's laws
                assert node.left.right is not None
                opposite_bool_op = (BoolOp.OR if node.left.bool_op
                                    == BoolOp.AND else BoolOp.AND)
                negated_lhs = LogicalOperation(left=node.left.left,
                                               bool_op=BoolOp.NOT,
                                               dtype_maybe=Dtype.bool,
                                               stype_maybe=Stype.categorical)
                negated_rhs = LogicalOperation(left=node.left.right,
                                               bool_op=BoolOp.NOT,
                                               dtype_maybe=Dtype.bool,
                                               stype_maybe=Stype.categorical)
                inverted_bool_op_node = LogicalOperation(
                    left=negated_lhs, bool_op=opposite_bool_op,
                    right=negated_rhs, dtype_maybe=Dtype.bool,
                    stype_maybe=Stype.categorical)
                split_filter = self._recursively_split_filter(
                    inverted_bool_op_node, original_filter, **kwargs)
                split_filter.rewritten = True
                return split_filter
            else:
                split_filter = self._recursively_split_filter(
                    node.left, original_filter, **kwargs)
                new_condition = split_filter.applicable_filter
                if new_condition is None:
                    new_condition = split_filter.non_applicable_filter
                assert isinstance(new_condition, Condition)
                negated_condition = LogicalOperation(
                    left=new_condition, bool_op=BoolOp.NOT,
                    dtype_maybe=Dtype.bool, stype_maybe=Stype.categorical)
                if split_filter.applicable_filter is None:
                    return SplitFilter(
                        applicable_filter=None,
                        non_applicable_filter=negated_condition,
                        rewritten=split_filter.rewritten,
                        n_applicable_conditions=0,
                        n_non_applicable_conditions=1,
                        non_applicable_is_valid=True,
                        original_filter=original_filter,
                    )
                else:
                    return SplitFilter(
                        applicable_filter=negated_condition,
                        non_applicable_filter=None,
                        rewritten=split_filter.rewritten,
                        n_applicable_conditions=1,
                        n_non_applicable_conditions=0,
                        non_applicable_is_valid=True,
                        original_filter=original_filter,
                    )
        assert node.right is not None
        split_left = self._recursively_split_filter(node.left, original_filter,
                                                    **kwargs)
        split_right = self._recursively_split_filter(node.right,
                                                     original_filter, **kwargs)

        if node.bool_op == BoolOp.OR:
            # Split the expression into Conjunctive Normal Form (CNF)
            # (L_appl AND L_non_appl) OR (R_appl AND R_non_appl) =
            # (L_appl OR R_appl) AND (
            #      (L_appl OR R_non_appl) AND
            #      (L_non_appl OR R_appl) AND
            #      (L_non_appl OR R_non_appl)
            # )
            n_applicable_conditions = 0
            n_non_applicable_conditions = 0
            combined_applicable = None
            # (L_appl OR R_appl)
            if (split_left.applicable_filter is not None
                    and split_right.applicable_filter is not None):
                combined_applicable = LogicalOperation(
                    left=split_left.applicable_filter, bool_op=BoolOp.OR,
                    right=split_right.applicable_filter,
                    dtype_maybe=Dtype.bool, stype_maybe=Stype.categorical)
                n_applicable_conditions += (
                    split_left.n_applicable_conditions +
                    split_right.n_applicable_conditions)

            combined_non_applicable = None
            non_applicable_terms = []
            # (L_appl OR R_non_appl)
            if (split_left.applicable_filter is not None
                    and split_right.non_applicable_filter is not None):
                non_applicable_terms.append(
                    LogicalOperation(
                        left=deepcopy(split_left.applicable_filter),
                        bool_op=BoolOp.OR,
                        right=deepcopy(split_right.non_applicable_filter),
                        dtype_maybe=Dtype.bool, stype_maybe=Stype.categorical))
                n_non_applicable_conditions += (
                    split_left.n_applicable_conditions +
                    split_right.n_non_applicable_conditions)
            # (L_non_appl OR R_appl)
            if (split_left.non_applicable_filter is not None
                    and split_right.applicable_filter is not None):
                non_applicable_terms.append(
                    LogicalOperation(
                        left=deepcopy(split_left.non_applicable_filter),
                        bool_op=BoolOp.OR,
                        right=deepcopy(split_right.applicable_filter),
                        dtype_maybe=Dtype.bool, stype_maybe=Stype.categorical))
                n_non_applicable_conditions += (
                    split_left.n_non_applicable_conditions +
                    split_right.n_applicable_conditions)
            # (L_non_appl OR R_non_appl)
            if (split_left.non_applicable_filter is not None
                    and split_right.non_applicable_filter is not None):
                non_applicable_terms.append(
                    LogicalOperation(
                        left=deepcopy(split_left.non_applicable_filter),
                        bool_op=BoolOp.OR,
                        right=deepcopy(split_right.non_applicable_filter),
                        dtype_maybe=Dtype.bool, stype_maybe=Stype.categorical))
                n_non_applicable_conditions += (
                    split_left.n_non_applicable_conditions +
                    split_right.n_non_applicable_conditions)
            rewritten = (split_left.rewritten or split_right.rewritten
                         or len(non_applicable_terms) > 1
                         or len(non_applicable_terms) > 0
                         and combined_applicable is not None)

            # combine all non-applicable terms
            while len(non_applicable_terms) > 1:
                term1 = non_applicable_terms.pop()
                term2 = non_applicable_terms.pop()
                non_applicable_terms.append(
                    LogicalOperation(left=term1, bool_op=BoolOp.AND,
                                     right=term2, dtype_maybe=Dtype.bool,
                                     stype_maybe=Stype.categorical))
            if len(non_applicable_terms) > 0:
                combined_non_applicable = non_applicable_terms[0]
            non_applicable_is_valid = (split_left.non_applicable_is_valid
                                       and split_right.non_applicable_is_valid)

            # If split_left.bp_non_applicable_filter was None because the
            # expression was too long, its conditions weren't counted in
            # n_non_applicable_conditions yet
            if not split_left.non_applicable_is_valid:
                if split_right.applicable_filter is not None:
                    n_non_applicable_conditions += (
                        split_left.n_non_applicable_conditions +
                        split_right.n_applicable_conditions)
                if split_right.non_applicable_filter is not None:
                    n_non_applicable_conditions += (
                        split_left.n_non_applicable_conditions +
                        split_right.n_non_applicable_conditions)
            if not split_right.non_applicable_is_valid:
                if split_left.applicable_filter is not None:
                    n_non_applicable_conditions += (
                        split_left.n_applicable_conditions +
                        split_right.n_non_applicable_conditions)
                if split_left.non_applicable_filter is not None:
                    n_non_applicable_conditions += (
                        split_left.n_non_applicable_conditions +
                        split_right.n_non_applicable_conditions)
            if (not split_right.non_applicable_is_valid
                    and not split_left.non_applicable_is_valid):
                n_non_applicable_conditions += (
                    split_left.n_non_applicable_conditions +
                    split_right.n_non_applicable_conditions)

            if n_non_applicable_conditions > MAX_N_CONDITIONS:
                non_applicable_is_valid = False
                combined_non_applicable = None
            return SplitFilter(
                applicable_filter=combined_applicable,
                non_applicable_filter=combined_non_applicable,
                rewritten=rewritten,
                n_applicable_conditions=n_applicable_conditions,
                n_non_applicable_conditions=n_non_applicable_conditions,
                non_applicable_is_valid=non_applicable_is_valid,
                original_filter=original_filter,
            )

        elif node.bool_op == BoolOp.AND:
            # Split the expression into Conjunctive Normal Form (CNF)
            # (L_appl AND L_non_appl) AND (R_appl AND R_non_appl) =
            # (L_appl AND R_appl) AND (L_non_appl AND R_non_appl)
            combined_applicable = None
            if (split_left.applicable_filter is not None
                    and split_right.applicable_filter is not None):
                combined_applicable = LogicalOperation(
                    left=split_left.applicable_filter, bool_op=BoolOp.AND,
                    right=split_right.applicable_filter,
                    dtype_maybe=Dtype.bool, stype_maybe=Stype.categorical)
            elif split_left.applicable_filter is not None:
                combined_applicable = split_left.applicable_filter
            else:
                combined_applicable = split_right.applicable_filter
            combined_non_applicable = None
            if (split_left.non_applicable_filter is not None
                    and split_right.non_applicable_filter is not None):
                combined_non_applicable = LogicalOperation(
                    left=split_left.non_applicable_filter, bool_op=BoolOp.AND,
                    right=split_right.non_applicable_filter,
                    dtype_maybe=Dtype.bool, stype_maybe=Stype.categorical)
            elif split_left.non_applicable_filter is not None:
                combined_non_applicable = split_left.non_applicable_filter
            else:
                combined_non_applicable = split_right.non_applicable_filter
            non_applicable_is_valid = (split_left.non_applicable_is_valid
                                       and split_right.non_applicable_is_valid)
            if (split_left.n_non_applicable_conditions +
                    split_right.n_non_applicable_conditions
                    > MAX_N_CONDITIONS):
                combined_non_applicable = None
                non_applicable_is_valid = False
            return SplitFilter(
                applicable_filter=combined_applicable,
                non_applicable_filter=combined_non_applicable,
                rewritten=split_left.rewritten or split_right.rewritten,
                n_applicable_conditions=split_left.n_applicable_conditions +
                split_right.n_applicable_conditions,
                n_non_applicable_conditions=split_left.
                n_non_applicable_conditions +
                split_right.n_non_applicable_conditions,
                non_applicable_is_valid=non_applicable_is_valid,
                original_filter=original_filter,
            )
        else:
            raise ValueError(f'Unknown operation {node.bool_op}.')
