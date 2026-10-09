# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from kumoapi.pquery.AST import (
    Aggregation,
    ASTNode,
    Column,
    Condition,
    Filter,
    Join,
    LogicalOperation,
)
from kumopql.rewriting.filter_splitter import FilterSplitter, SplitFilter


class StaticFilterSplitter(FilterSplitter):
    """Transformer that splits filters into applicable and non-applicable parts
    based on whether the conditions are static or not.

    This filter transformer is used to split the entity and target ASTs into
    filters that can be applied before timeframe generation and filters that
    must be applied after timeframe generation.
    """
    def __init__(self) -> None:
        pass

    def duplicates_inapplicable_nodes(self, split_filter: SplitFilter) -> bool:
        """Check if there are duplicate inapplicable nodes in the filter
        tree.
        """
        if (split_filter.non_applicable_filter is None
                or split_filter.original_filter is None):
            return False
        old_num_aggregations = self._count_aggregations(
            split_filter.original_filter)
        new_num_aggregations = self._count_aggregations(
            split_filter.non_applicable_filter)
        return old_num_aggregations < new_num_aggregations

    def can_be_split_for_entity_optimization(
            self, split_filter: SplitFilter) -> bool:
        """Check if the filter can be split for entity optimization.

        A filter can be split for entity optimization if:
        - The applicable part of the filter is not None (which means that
          some splitting was done)
        - The non-applicable part of the filter is valid.
        - There are no duplicate inapplicable nodes in the filter tree.
        """
        return (split_filter.non_applicable_is_valid
                and split_filter.applicable_filter is not None
                and not self.duplicates_inapplicable_nodes(split_filter))

    def _count_aggregations(self, node: ASTNode) -> int:
        """Count the number of aggregations in the subtree."""
        if isinstance(node, Aggregation):
            return 1 + sum(
                self._count_aggregations(child) for child in node.children)
        return sum(self._count_aggregations(child) for child in node.children)

    def _check_if_duplicates_exist(
        self,
        node: ASTNode,
        seen_nodes: list[ASTNode],
    ) -> bool:
        """Check if there are duplicates in this node and its subtree.

        Args:
            node: The node to check.
            seen_nodes: The list of nodes that have already been seen.

        Returns:
            bool: Whether there are duplicates in this node and its subtree.
        """
        if node in seen_nodes:
            return True
        seen_nodes.append(node)
        for child in node.children:
            if self._check_if_duplicates_exist(child, seen_nodes):
                return True
        return False

    def split(
        self,
        node: ASTNode,
    ) -> SplitFilter:
        """Entry point for the filter splitter.

        Args:
            node: The node to split.

        Returns:
            A split filter describing whether the input node is
            applicable or not.
        """
        if isinstance(node, Filter):
            return self._recursively_split_filter(node.condition,
                                                  node.condition)
        if isinstance(node, Join):
            return self.split(node.rhs_target)
        if isinstance(node, Aggregation):
            return self.split(node.target)
        if isinstance(node, Column):
            return SplitFilter(
                applicable_filter=None,
                non_applicable_filter=None,
                rewritten=False,
                n_applicable_conditions=0,
                n_non_applicable_conditions=0,
                non_applicable_is_valid=True,
                original_filter=None,
            )
        raise AssertionError(f"Unknown node type {type(node)}.")

    def _condition_base_case(
        self,
        node: Condition,
        original_filter: Condition | LogicalOperation | None,
        **kwargs: Any,
    ) -> SplitFilter:
        """Base case for the recursive filter splitting. A condition is not
        applicable if its target is a non-static aggregation. Otherwise, it
        should be applicable.

        Args:
            node: The condition to split.
            original_filter: The full filter being split.
            kwargs: Any other args passed to children.

        Returns:
            A split filter describing whether the input condition
            is applicable or not.
        """
        assert isinstance(node, Condition)
        target = node.target
        if isinstance(target, Join):
            target = target.rhs_target
        if isinstance(target, Aggregation):
            # For entity filters this may not ever happen but it may for target
            # filters if we want to optimize those later.
            if target.is_static:
                # This aggregation is valid, but we need to make sure all sub
                # filters are valid.
                sub_filter = self.split(target.target)

                # No sub-filters, so this aggregation is valid.
                if (sub_filter.n_applicable_conditions == 0
                        and sub_filter.n_non_applicable_conditions == 0):
                    return SplitFilter(
                        applicable_filter=node,
                        non_applicable_filter=None,
                        rewritten=False,
                        n_applicable_conditions=1,
                        n_non_applicable_conditions=0,
                        non_applicable_is_valid=True,
                        original_filter=original_filter,
                    )

                # If all sub-filters are valid, this aggregation is valid.
                if sub_filter.n_non_applicable_conditions == 0:
                    assert sub_filter.n_applicable_conditions > 0
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
                    # If there are any invalid sub-filters, this aggregation
                    # is invalid.
                    assert sub_filter.n_applicable_conditions == 0
                    assert sub_filter.n_non_applicable_conditions > 0
                    return SplitFilter(
                        applicable_filter=None,
                        non_applicable_filter=node,
                        rewritten=False,
                        n_applicable_conditions=0,
                        n_non_applicable_conditions=1,
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
        assert isinstance(target, Column)
        return SplitFilter(
            applicable_filter=node,
            non_applicable_filter=None,
            rewritten=False,
            n_applicable_conditions=1,
            n_non_applicable_conditions=0,
            non_applicable_is_valid=True,
            original_filter=original_filter,
        )
