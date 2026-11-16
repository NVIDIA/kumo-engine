# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable DiskGraph table-stat assembly with injected filesystem reads."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from ..types import TableStatType
from .types import TableStat


def compute_table_stats(
    source_path: str,
    *,
    is_directory: Callable[[str], bool],
    count_rows: Callable[[str], int],
    file_size: Callable[[str], int],
    directory_size: Callable[[str], int],
    index_path: str | None = None,
    data_version: str = "",
    stats: Iterable[TableStatType] | None = None,
) -> dict[TableStatType, TableStat]:
    wanted = set(stats) if stats is not None else set(TableStatType)
    result: dict[TableStatType, TableStat] = {}
    if TableStatType.NUM_ROWS in wanted:
        result[TableStatType.NUM_ROWS] = TableStat(count_rows(source_path),
                                                   data_version=data_version)
    if TableStatType.SOURCE_SIZE in wanted:
        size = (directory_size(source_path)
                if is_directory(source_path) else file_size(source_path))
        result[TableStatType.SOURCE_SIZE] = TableStat(size,
                                                      data_version=data_version)
    if TableStatType.MATERIALIZED_SIZE in wanted and index_path:
        result[TableStatType.MATERIALIZED_SIZE] = TableStat(
            directory_size(index_path), data_version=data_version)
    return result
