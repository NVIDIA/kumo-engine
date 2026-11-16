# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os.path as osp
from typing import Dict, List, Optional, TypeAlias

from cachetools import TTLCache
from kumo_core.types import EdgeStatType
from wrapt import synchronized

from async_utils.sync import FutureDedupeOption, blocking_method
from io_utils import fsutil, json, kumo_dataclass
from kumo.data.cache.table_cache_dir import CACHE_TTL, MAX_CACHE_SIZE
from kumo.data.common import DATA_BACKEND_THREAD_POOL
from kumo.stats import Stats, StatsConfig
from kumo.stats.stats import ALL_EDGE_STAT_TYPES
from monitoring.tracing import TraceOptions

logger = logging.getLogger(__name__)


def graph_key(graph_name: str, data_version: str) -> str:
    return "__".join([graph_name, data_version])


# The data structure for storing edge stats in S3 JSON files.
GraphEdgeStatsDict = Dict[str, StatsConfig]


@kumo_dataclass
class SingleEdgeStats:
    """Contains all stats that were computed for a single edge.
    """
    stats: Dict[EdgeStatType, StatsConfig]


@kumo_dataclass
class SourceTableColumn:
    connector_name: str
    source_table_name: str
    col_name: Optional[str]


def edge_stat_key(
    src: SourceTableColumn,
    dst: SourceTableColumn,
    stat_type: EdgeStatType,
) -> str:
    # fkey_config.dst_key may be none so cast to str and index by it
    # anyway to avoid having dst key inference logic here
    return "__".join([
        src.source_table_name,
        src.connector_name,
        dst.source_table_name,
        dst.connector_name,
        str(src.col_name),
        str(dst.col_name),
        stat_type.value,
    ])


@kumo_dataclass
class GraphEdgeStats:
    """Wrapper around GraphEdgeStatsDict, providing helper functions for
    convenient lookup.
    """
    stats: GraphEdgeStatsDict

    def get_stats(self, src: SourceTableColumn,
                  dst: SourceTableColumn) -> Optional[SingleEdgeStats]:
        """Return all stats for the given edge. Note, the direction of the edge
        matters. Returns None if stats are not found for a particular edge.
        """
        found_stats: Dict[EdgeStatType, StatsConfig] = {}
        for typ in ALL_EDGE_STAT_TYPES:
            key = edge_stat_key(src, dst, typ)
            if key not in self.stats:
                continue
            found_stats[typ] = self.stats[key]

        if len(found_stats) == 0:
            return None

        return SingleEdgeStats(found_stats)


# Unique identifier for a graph in data backend. Corresponds to the ID for v1
# graphs, and the FPID for v2 graphs in the REST API.
GraphKey: TypeAlias = str
GraphVersion: TypeAlias = str


@kumo_dataclass
class VersionedGraphKey:
    """Unique identifier for graph stats for a single version of a graph.
    """
    key: GraphKey
    version: GraphVersion


class GraphCache():
    """The backend cache for everything related to a graph: version_info,
    edge_stats, ...
    """
    def __init__(self, cache_dir: str):
        self._cache_dir = cache_dir
        # 2 hours TTL cache.
        self._edge_stats_cache = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=CACHE_TTL)

    def reset(self, cache_dir: str | None = None):
        if cache_dir is not None:
            self._cache_dir = cache_dir

    @synchronized
    def get_missing_edge_stats(
        self,
        graph_name: str,
        src: SourceTableColumn,
        dst: SourceTableColumn,
        stats_list: List[EdgeStatType],
        data_version: str,
    ) -> List[EdgeStatType]:
        key = graph_key(graph_name, data_version)
        if key not in self._edge_stats_cache:
            self._load_edge_stats(graph_name, data_version)
        if key not in self._edge_stats_cache:
            return stats_list
        return_list = []
        for stat_type in stats_list:
            stat_key = edge_stat_key(src, dst, stat_type)
            if self._edge_stats_cache[key].get(stat_key) is None:
                return_list.append(stat_type)
        return return_list

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=True),
                     fut_dedupe_option=FutureDedupeOption.DISABLED)
    @synchronized
    def add_edge_stats(
        self,
        graph_name: str,
        src: SourceTableColumn,
        dst: SourceTableColumn,
        stats_dict: Dict[EdgeStatType, Stats],
        data_version: str,
    ):
        if not stats_dict:
            return
        key = graph_key(graph_name, data_version)

        if key not in self._edge_stats_cache:
            self._load_edge_stats(graph_name, data_version)
        if key not in self._edge_stats_cache:
            self._edge_stats_cache[key] = {}
        for stat_type, stat in stats_dict.items():
            if stat is not None:  # type: ignore
                stat_key = edge_stat_key(src, dst, stat_type)
                self._edge_stats_cache[key][stat_key] = stat.config()
        self._save_edge_stats(graph_name, data_version)
        logger.info(f"GraphCache size: {self.size()}")

    @synchronized
    def get_edge_stats(
        self,
        graph_name: str,
        src: SourceTableColumn,
        dst: SourceTableColumn,
        stats_list: List[EdgeStatType],
        data_version: str,
    ) -> Dict[EdgeStatType, Stats]:
        key = graph_key(graph_name, data_version)
        if key not in self._edge_stats_cache:
            self._load_edge_stats(graph_name, data_version)
        if not self._edge_stats_cache[key]:
            return {}

        return_dict = {}
        for stat_type in stats_list:
            stat_key = edge_stat_key(src, dst, stat_type)
            stat = self._edge_stats_cache[key].get(stat_key)
            if stat is not None:
                return_dict[stat_type] = Stats.from_config(stat)
        return return_dict

    def _save_edge_stats(self, graph_name: str, graph_version: str):
        key = graph_key(graph_name, graph_version)
        if key not in self._edge_stats_cache:
            logger.warning(
                'graph %s with version %s does not have cached edge stats.',
                graph_name, graph_version)
            return

        # Save the edge stats to a new file on S3
        stats_dict = self._edge_stats_cache[key]
        file_path = self.edge_stats_path(
            VersionedGraphKey(graph_name, graph_version))
        logger.info(f'Saving edge stats for {key} to {file_path}')
        json.save_json(stats_dict, file_path)

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=True))
    def _load_edge_stats(self, graph_name: str, graph_version: str) -> bool:
        file_path = self.edge_stats_path(
            VersionedGraphKey(graph_name, graph_version))

        value = json.load_json(file_path, Dict, default=None,
                               raise_on_missing=False)
        if value is None:
            logger.warning(f'No edge stats found at {file_path}')
            return False

        key = graph_key(graph_name, graph_version)
        logger.info(f'Loading edge stats for {key} from {file_path}')
        self._edge_stats_cache[key] = value
        logger.info(f"GraphCache size: {self.size()}")
        return True

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=True))
    def save_latest_graph_version(self, graph_name: str, graph_version: str):
        """Mark the given graph version as the "latest" graph version in the
        cache.
        """
        json.save_json(graph_version, self.latest_version_path(graph_name))

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=True))
    def get_many_latest_graph_versions(
            self,
            graph_names: List[GraphKey]) -> List[Optional[VersionedGraphKey]]:
        # Read the most recent graph versions from S3
        paths = [self.latest_version_path(name) for name in graph_names]
        files = fsutil.get_many_if_present(paths)

        # Parse the versions as json
        results: List[Optional[VersionedGraphKey]] = []
        for name, file in zip(graph_names, files):
            if file is None:
                results.append(None)
                continue
            version = json.fast_from_json(file, GraphVersion)
            results.append(VersionedGraphKey(name, version))

        return results

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=True))
    def get_graph_edge_stats(
            self, graph: VersionedGraphKey) -> Optional[GraphEdgeStats]:
        file_path = self.edge_stats_path(graph)
        r = json.load_json(file_path, GraphEdgeStatsDict,
                           raise_on_missing=False)
        if r is None:
            return r
        else:
            return GraphEdgeStats(r)

    def latest_version_path(self, graph_name: str) -> str:
        return osp.join(self._cache_dir, graph_name, 'latest_version')

    def version_info_path(self, graph_name: str, version: str) -> str:
        return osp.join(self._cache_dir, graph_name, version, 'version_info')

    def edge_stats_path(self, v: VersionedGraphKey) -> str:
        return osp.join(self._cache_dir, v.key, v.version, 'edge_stats')

    def graph_dir(self, graph_name: str, version: str) -> str:
        return osp.join(self._cache_dir, graph_name, version)

    def size(self) -> int:
        ret = 0
        for key in self._edge_stats_cache:
            ret += len(self._edge_stats_cache[key])
        return ret
