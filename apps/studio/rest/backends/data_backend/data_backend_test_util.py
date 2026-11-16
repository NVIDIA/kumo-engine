# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict, List, Optional

from kumo_core.types import EdgeStatType

from async_utils.sync import blocking_func
from io_utils import fsutil, logging
from kumo.data import get_active_cache, get_persist_dir
from kumo.data.backend_cache import CACHE_DIR, BackendCacheManager
from kumo.data.cache.graph_cache import GraphKey, SourceTableColumn
from kumo.data.common import DATA_BACKEND_THREAD_POOL
from kumo.stats import Stats
from rest.backends.data_backend.context import DataBackendContext
from rest.config.data_backend import GraphVersion
from rest.config.graph import EdgeHealth
from rest.utils.driver_common import get_current_timestamps

log = logging.getLogger(__name__)


class DataBackendTestUtil:
    @staticmethod
    @blocking_func(DATA_BACKEND_THREAD_POOL)
    def clear_table_cache(ctx: DataBackendContext, connector_name: str,
                          source_table_name: str):
        # Clear out the data persist dir
        data_persist_dir = get_persist_dir()
        path = os.path.join(data_persist_dir, CACHE_DIR, 'tables',
                            connector_name, source_table_name or '')
        log.info('Clearing %s', path)
        fsutil.rm(path)

    @staticmethod
    @blocking_func(DATA_BACKEND_THREAD_POOL)
    def delete_graph_validation_result(graph_key: GraphKey):
        """Force deleting validation result for the graph at the latest data
        version.  This will clear any validation warning or error for the
        graph.

        This function is used for testing purpose. For production use, please
        consider implementing a "soft-delete" or "tombstone" approach.
        """
        cache: BackendCacheManager = get_active_cache()
        versioned_graph_key = cache.graph_cache.get_many_latest_graph_versions(
            [graph_key])[0]
        if not versioned_graph_key:
            log.warning(
                'Trying to delete validation for graph %s, but data version '
                'is missing.', graph_key)
            return
        graph_dir = cache.graph_cache.graph_dir(graph_key,
                                                versioned_graph_key.version)
        fsutil.rm(os.path.join(graph_dir, 'validation'))

    @staticmethod
    @blocking_func(DATA_BACKEND_THREAD_POOL)
    def set_mock_latest_edge_stats(
        graph_key: str,
        edges: List[EdgeHealth],
    ) -> str:
        """Save fake edge health stats to the backend for the unit testing.
        Creates and saves a new graph version with the current timestamp.
        """
        graph_version = GraphVersion(
            str(get_current_timestamps().updated_at.timestamp()))
        cache = get_active_cache()
        cache.graph_cache.save_latest_graph_version(graph_key, graph_version)
        for edge in edges:
            stats: Dict[EdgeStatType, Stats] = {}

            def set_stat_value(stat_type: EdgeStatType, value: Optional[float],
                               percent: bool = False) -> None:
                stat = Stats(value=value, approx=False,
                             data_version=graph_version)
                stats[stat_type] = stat
                if percent and value is not None:
                    stats[stat_type].value /= 100

            set_stat_value(EdgeStatType.NUM_SRC_NONZERO_DEGREE,
                           edge.matches.a_in_b)
            set_stat_value(EdgeStatType.NUM_DST_NONZERO_DEGREE,
                           edge.matches.b_in_a)
            set_stat_value(EdgeStatType.FRAC_SRC_NONZERO_DEGREE,
                           edge.percents.a_in_b, percent=True)
            set_stat_value(EdgeStatType.FRAC_DST_NONZERO_DEGREE,
                           edge.percents.b_in_a, percent=True)
            src = SourceTableColumn(edge.col_a.connector_id,
                                    edge.col_a.table_name,
                                    edge.col_a.column_name)
            dst = SourceTableColumn(edge.col_b.connector_id,
                                    edge.col_b.table_name,
                                    edge.col_b.column_name)
            cache.graph_cache.add_edge_stats(graph_key, src, dst, stats,
                                             graph_version)
        return graph_version
