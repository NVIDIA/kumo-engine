# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

from pydantic.dataclasses import dataclass

from kumoapi.typing import _FORBID_EXTRA_CONFIG


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class ASTQueryLocationInterval:
    r"""A dataclass for location range within the input query that an
    :class:`ASTNode` corresponds to. ``(start_row, start_col)`` points to the
    first character in this node, while ``(end_row, end_col)`` points to the
    last character in the substring corresponding to the node.
    All rows are 1-indexed and all columns are 0-indexed. All of these follow
    conventions used by ANTLR4.

    Args:
        start_row: Row of the node start, 1-indexed.
        start_col: Column of the node start, 0-indexed.
        end_row: Row of the node end, 1-indexed.
        end_col: Column of the node end, 0-indexed.
        data_available: If :obj:`False`, this node is assumed
            to not carry information about its location. Added for
            compatibility with modified queries, e.g. after rewriting queries
            for optimization.
    """
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    data_available: bool = True

    @classmethod
    def merge(
        cls,
        interval_1: 'ASTQueryLocationInterval',
        interval_2: 'ASTQueryLocationInterval',
    ) -> 'ASTQueryLocationInterval':
        r"""Merges the two intervals by taking the earlier of the starts and
        the later of the two ends, returning a new class.

        Args:
            interval_1: First of the intervals.
            interval_2: Second of the intervals.
        """
        if not interval_1.data_available:
            return interval_2
        if not interval_2.data_available:
            return interval_1
        start_row = min(interval_1.start_row, interval_2.start_row)
        if interval_1.start_row == interval_2.start_row:
            start_col = min(interval_1.start_col, interval_2.start_col)
        elif interval_1.start_row == start_row:
            start_col = interval_1.start_col
        else:
            start_col = interval_2.start_col
        end_row = max(interval_1.end_row, interval_2.end_row)
        if interval_1.end_row == interval_2.end_row:
            end_col = max(interval_1.end_col, interval_2.end_col)
        elif interval_1.end_row == end_row:
            end_col = interval_1.end_col
        else:
            end_col = interval_2.end_col
        return cls(start_row, start_col, end_row, end_col)

    @property
    def message_start(self) -> str:
        if not self.data_available:
            return ''
        return f'row {self.start_row}, column {self.start_col}'

    @property
    def message_end(self) -> str:
        if not self.data_available:
            return ''
        return f'row {self.end_row}, column {self.end_col}'

    @property
    def message(self) -> str:
        if not self.data_available:
            return ''
        return f'({self.message_start}: {self.message_end})'
