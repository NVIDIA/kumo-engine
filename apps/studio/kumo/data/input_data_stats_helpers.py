# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure helpers that read backend-cache stats into :class:`InputDataStats`.

These used to live in ``rest.backends.ml_backend.activities.common`` but
have no REST-specific dependency — they operate on a :class:`Graph`, a
:class:`BackendCacheManager`, and cached edge stats. Hosting them in
:mod:`kumo.data` lets the diskgraph-side caller
(``kumo.diskgraph.input_stats``) import them without crossing
the ``rest.backends`` layer boundary.
"""

from kumo_core.diskgraph.types import InputDataStats
from kumo_core.types import ColStatType, EdgeStatType

from io_utils import logging
from kumo.data.backend_cache import BackendCacheManager
from kumo.data.cache.graph_cache import GraphEdgeStats, SourceTableColumn
from kumo.diskgraph import MAX_EDGES_PER_EDGE_TYPE
from kumo.graph import Graph
from kumo.stats import is_missing_stat_value

logger = logging.getLogger(__name__)


def compute_time_ranges(
    graph: Graph,
    backend_cache: BackendCacheManager,
    result: InputDataStats,
) -> None:
    """Populate ``result.table_time_ranges`` from backend cache col stats."""
    for table in graph:
        if table.has_time_col():
            time_col_base = table.time_col.base_config()
            time_stats = backend_cache.get_col_stats(
                table.fingerprint(),
                {time_col_base: [ColStatType.MIN, ColStatType.MAX]},
                data_version=table.data_version, approx=False)
            if time_col_base in time_stats:
                min_stat = time_stats[time_col_base].get(ColStatType.MIN)
                max_stat = time_stats[time_col_base].get(ColStatType.MAX)
                min_value = (min_stat.value if min_stat
                             and not is_missing_stat_value(min_stat.value) else
                             None)
                max_value = (max_stat.value if max_stat
                             and not is_missing_stat_value(max_stat.value) else
                             None)
                result.table_time_ranges[table.name] = (min_value, max_value)
            else:
                logger.warning('Time column range missing for table %s',
                               table.name)


def compute_edge_counts(
    graph: Graph,
    graph_edge_stats: GraphEdgeStats,
    result: InputDataStats,
) -> None:
    """Populate edge counts on *result* from cached edge stats.

    Computes both ``graph_num_edges`` (raw) and
    ``graph_num_downsampled_edges`` (capped per edge type).
    Gracefully skips fkeys with missing destination metadata.
    """
    graph_num_edges = 0
    graph_num_downsampled_edges = 0
    if graph_edge_stats:
        for table in graph.tables:
            for fkey in table.fkeys:
                if not fkey.dst_data_source or not fkey.dst_table:
                    logger.warning(
                        "Skipping edge stats for fkey %s on %s: "
                        "dst_data_source=%s, dst_table=%s", fkey.name,
                        table.table_name, fkey.dst_data_source, fkey.dst_table)
                    continue
                src_key = SourceTableColumn(table.data_source,
                                            table.source_name, fkey.name)
                dst_key = SourceTableColumn(
                    fkey.dst_data_source,
                    graph.table(fkey.dst_table).source_name, fkey.dst_key)
                stats = graph_edge_stats.get_stats(src_key, dst_key)
                if not stats:
                    stats = graph_edge_stats.get_stats(dst_key, src_key)
                if stats:
                    edge_count = stats.stats[EdgeStatType.EDGE_COUNT].value
                    graph_num_edges += int(edge_count) * 2
                    graph_num_downsampled_edges += min(MAX_EDGES_PER_EDGE_TYPE,
                                                       int(edge_count)) * 2
    result.graph_num_edges = graph_num_edges
    result.graph_num_downsampled_edges = graph_num_downsampled_edges
