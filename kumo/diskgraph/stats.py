# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Table-stats computation for the DiskGraph path.

Reads row counts, byte sizes, and materialized sizes directly off parquet
files (via pyarrow) and the on-disk DG index (via ``fsutil``), producing
the :class:`TableStatType` subset consumers ask for.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from kumo_core.diskgraph.stats import compute_table_stats
from kumo_core.types import TableStatType

from io_utils import fsutil, parquet_util
from kumo.stats.stats import Stats

logger = logging.getLogger(__name__)


def _count_rows_for_path(path: str) -> int:
    """Return the row count for a parquet- or CSV-shaped path."""
    if fsutil.isdir(path):
        try:
            entries = list(fsutil.listdir(path))
        except Exception:  # pragma: no cover - defensive
            entries = []
        has_parquet = any(e.endswith('.parquet') for e in entries)
        has_csv = any(e.endswith('.csv') for e in entries)
        if has_parquet:
            return parquet_util.parquet_num_rows(path)
        if has_csv:
            return sum(
                fsutil.file_lines(e) for e in entries if e.endswith('.csv'))
        return parquet_util.parquet_num_rows(path)

    if path.endswith('.csv'):
        return fsutil.file_lines(path)
    return parquet_util.parquet_num_rows(path)


def compute_table_stats_native(
    parquet_path: str,
    index_path: Optional[str] = None,
    data_version: str = '',
    stats_list: Optional[list[TableStatType]] = None,
) -> Dict[TableStatType, Stats]:
    """Compute table stats from a parquet path + (optional) DG index path.

    ``MATERIALIZED_SIZE`` requires ``index_path``; omit it and the stat is
    skipped rather than raised.
    """
    portable = compute_table_stats(
        parquet_path,
        is_directory=fsutil.isdir,
        count_rows=_count_rows_for_path,
        file_size=lambda path: fsutil.filesize(path) or 0,
        directory_size=fsutil.dirsize,
        index_path=index_path,
        data_version=data_version,
        stats=stats_list,
    )
    return {
        key:
        Stats(value=value.value, approx=value.approx,
              data_version=value.data_version)
        for key, value in portable.items()
    }
