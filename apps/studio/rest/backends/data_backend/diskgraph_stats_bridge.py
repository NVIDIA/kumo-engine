# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temporal wrapper that copies DG-produced stats into the backend cache.

The pure body lives in :mod:`kumo.diskgraph.ingest_stats_cache`.
This module owns the Temporal registration; it delegates straight to the
service, working off the explicit ``index_path`` and backend cache.
"""

from temporalio import activity

from kumo.diskgraph import DiskgraphHandle, get_service
from rest.config.table_v2 import TableDefinition


@activity.defn
def diskgraph_ingest_stats_activity(
    index_path: str,
    graph_key: str,
    graph_data_version: str,
    tables: dict[str, TableDefinition],
    table_data_versions: dict[str, str],
    has_edges: bool,
    ingest_stats_path: str,
) -> dict[str, int] | None:
    """Copy Rust-produced col stats and edge stats from diskgraph ingest to cache.

    `tables` maps table alias -> v2 TableDefinition.

    Col stats are copied per-table from ``{index_path}/kumo_col_stats/``
    into the table cache directories.

    Delegates to :meth:`DiskgraphService.copy_ingest_stats_to_cache` and
    returns the small per-table duplicate-count map.
    """
    return get_service().copy_ingest_stats_to_cache(
        DiskgraphHandle(
            graph_key=graph_key,
            graph_data_version=graph_data_version,
        ),
        index_path,
        ingest_stats_path,
        tables=tables,
        table_data_versions=table_data_versions,
        has_edges=has_edges,
    )
