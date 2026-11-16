# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caller-side assembler for :class:`InputDataStats` from DG sources.

:class:`DiskgraphService` exposes :meth:`load_ingest_index_metadata` as
the primitive that reads ``index_metadata.json`` from the snapshot.
The remaining fields on ``InputDataStats`` come from the Kumo data-
backend cache (client-side resource on the REST worker's filesystem).
This function combines the two.
"""

from typing import Optional

from kumo_core.diskgraph.types import InputDataStats

from kumo.data import get_active_cache
from kumo.data.cache.graph_cache import VersionedGraphKey
from kumo.data.input_data_stats_helpers import (
    compute_edge_counts,
    compute_time_ranges,
)
from kumo.diskgraph import DiskgraphHandle, get_service
from kumo.graph import Graph


def assemble_input_data_stats(
    handle: DiskgraphHandle,
    *,
    graph: Graph,
    graph_key_ver: VersionedGraphKey,
    index_path: Optional[str] = None,
) -> InputDataStats:
    """Assemble :class:`InputDataStats` from DG metadata + backend cache.

    - ``num_nodes``: from ``index_metadata.json`` per node type.
    - edge counts / time ranges / table sizes / in-memory features:
      from the Kumo data-backend cache populated during ingest.
    """
    from kumo_core.types import TableStatType

    index_meta = get_service().load_ingest_index_metadata(
        handle, index_path=index_path)

    node_type_to_meta = {
        nt['name']: nt
        for nt in index_meta.get('node_types', [])
    }

    result = InputDataStats()

    graph_num_nodes = 0
    for table in graph:
        max_id = node_type_to_meta.get(table.table_name, {}).get('max_id')
        num_nodes = (max_id +
                     1) if isinstance(max_id, int) and max_id >= 0 else 0
        result.table_num_nodes[table.name] = num_nodes
        graph_num_nodes += num_nodes
    result.graph_num_nodes = graph_num_nodes

    backend_cache = get_active_cache()
    backend_cache.load_all_table_caches(graph.config())
    graph_edge_stats = backend_cache.graph_cache.get_graph_edge_stats(
        graph_key_ver)
    compute_edge_counts(graph, graph_edge_stats, result)

    compute_time_ranges(graph, backend_cache, result)

    # DiskGraph ingestion records SOURCE_SIZE in the backend cache; read it
    # directly instead of looking for separately materialized table data.
    total_table_size = 0
    for table in graph:
        size_stats = backend_cache.get_table_stats(
            table.fingerprint(),
            stats_list=[TableStatType.SOURCE_SIZE],
            data_version=table.data_version,
        )
        source_size_stat = size_stats.get(TableStatType.SOURCE_SIZE)
        table_size = int(source_size_stat.value) if source_size_stat else 0
        result.table_size_bytes[table.name] = table_size
        total_table_size += table_size
    result.graph_size_bytes = total_table_size

    num_in_memory_features = 0
    for table in graph:
        if table.has_time_col():
            table_stats = backend_cache.get_table_stats(
                table.fingerprint(),
                stats_list=[TableStatType.NUM_ROWS],
                data_version=table.data_version,
            )
            num_rows_stat = table_stats.get(TableStatType.NUM_ROWS)
            if num_rows_stat:
                num_in_memory_features += int(num_rows_stat.value)
    result.num_in_memory_features = num_in_memory_features

    return result
