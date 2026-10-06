# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from unittest.mock import MagicMock

import pytest
from kumo_core.types import EdgeStatType, TableStatType, TaskType

from kumo.data import get_active_cache
from kumo.data.cache.graph_cache import SourceTableColumn, VersionedGraphKey
from kumo.stats import Stats
from kumo.task.task import Task
from rest.backends.ml_backend.activities import common as activity_common
from rest.settings import DataScaleLimits

ROOT_DIR = osp.dirname(osp.realpath(__file__))
AUTO_TRAIN_TESTDIR = osp.join(ROOT_DIR, 'test_results')


def test_estimate_input_stats_requires_resolved_table_versions():
    table = MagicMock()
    table.data_version = None
    table.name = 'unresolved'
    graph = MagicMock()
    graph.tables = [table]

    with pytest.raises(ValueError, match="missing version for 'unresolved'"):
        activity_common.estimate_input_data_stats(
            graph, VersionedGraphKey('graph', 'version'), None)


def test_estimate_input_stats(get_task, tmp_path, monkeypatch):
    task: Task = get_task(
        TaskType.MULTICLASS_CLASSIFICATION,
        temporal=False,
        materialize=False,
        create_new=True,
    )
    graph = task.graph
    graph_key = VersionedGraphKey(graph.name, 'dummy_version')

    backend_cache = get_active_cache()
    load_all_table_caches = MagicMock(
        side_effect=AssertionError('column stats must not be loaded'))
    monkeypatch.setattr(backend_cache, 'load_all_table_caches',
                        load_all_table_caches)
    monkeypatch.setattr(activity_common.time, 'sleep', lambda _: None)

    # test before creating stats to use the fallback code path:
    stats = activity_common.estimate_input_data_stats(graph, graph_key, None)

    assert stats.graph_size_bytes == DataScaleLimits.parse_from_env(
        None).max_graph_total_size_bytes

    for table in graph.tables:
        for fkey in table.fkeys:
            assert fkey.dst_data_source and fkey.dst_table
            src_key = SourceTableColumn(table.data_source, table.source_name,
                                        fkey.name)
            dst_key = SourceTableColumn(fkey.dst_data_source,
                                        graph.table(fkey.dst_table).source_name,
                                        fkey.dst_key)
            backend_cache.graph_cache.add_edge_stats(
                graph.name, src_key, dst_key,
                {EdgeStatType.EDGE_COUNT: Stats(100, False)}, 'dummy_version')
        backend_cache.add_table_stats(
            table.fingerprint(), {
                TableStatType.SOURCE_SIZE:
                Stats(100, False, data_version=table.data_version)
            })
    stats = activity_common.estimate_input_data_stats(graph, graph_key, None)
    assert stats is not None
    load_all_table_caches.assert_not_called()

    for table in graph.tables:
        backend_cache.add_table_stats(
            table.fingerprint(), {
                TableStatType.SOURCE_SIZE:
                Stats(100000000000, False, data_version=table.data_version)
            })
    stats = activity_common.estimate_input_data_stats(graph, graph_key, None)
    assert stats is not None

    for table in graph.tables:
        for fkey in table.fkeys:
            assert fkey.dst_data_source and fkey.dst_table
            src_key = SourceTableColumn(table.data_source, table.source_name,
                                        fkey.name)
            dst_key = SourceTableColumn(fkey.dst_data_source,
                                        graph.table(fkey.dst_table).source_name,
                                        fkey.dst_key)
            backend_cache.graph_cache.add_edge_stats(
                graph.name, src_key, dst_key,
                {EdgeStatType.EDGE_COUNT: Stats(1000000000, False)},
                'dummy_version')
        backend_cache.add_table_stats(
            table.fingerprint(), {
                TableStatType.SOURCE_SIZE:
                Stats(100, False, data_version=table.data_version)
            })
    stats = activity_common.estimate_input_data_stats(graph, graph_key, None)
    assert stats is not None


def test_estimate_input_stats_prefers_materialized_size(get_task, tmp_path):
    """MATERIALIZED_SIZE from a prior run should be preferred over
    SOURCE_SIZE, giving accurate estimates for repeat training jobs."""
    task: Task = get_task(
        TaskType.MULTICLASS_CLASSIFICATION,
        temporal=False,
        materialize=False,
        create_new=True,
    )
    graph = task.graph
    graph_key = VersionedGraphKey(graph.name, 'dummy_version')

    backend_cache = get_active_cache()

    # Simulate a scenario where SOURCE_SIZE is small (below threshold)
    # but MATERIALIZED_SIZE from a prior run is large (above threshold).
    # This is the Databricks views case: SOURCE_SIZE=117GB, actual=304GB.
    num_tables = len(list(graph.tables))
    small_source = int(50 * 10**9)  # 50 GB per table
    large_materialized = int(100 * 10**9)  # 100 GB per table

    for table in graph.tables:
        for fkey in table.fkeys:
            assert fkey.dst_data_source and fkey.dst_table
            src_key = SourceTableColumn(table.data_source, table.source_name,
                                        fkey.name)
            dst_key = SourceTableColumn(fkey.dst_data_source,
                                        graph.table(fkey.dst_table).source_name,
                                        fkey.dst_key)
            backend_cache.graph_cache.add_edge_stats(
                graph.name, src_key, dst_key,
                {EdgeStatType.EDGE_COUNT: Stats(100, False)}, 'dummy_version')
        backend_cache.add_table_stats(
            table.fingerprint(), {
                TableStatType.SOURCE_SIZE:
                Stats(small_source, False, data_version=table.data_version),
                TableStatType.MATERIALIZED_SIZE:
                Stats(large_materialized, False,
                      data_version=table.data_version),
            })

    stats = activity_common.estimate_input_data_stats(graph, graph_key, None)
    # Should use MATERIALIZED_SIZE (100GB * num_tables), not SOURCE_SIZE
    expected = large_materialized * num_tables
    assert stats.graph_size_bytes == expected


def test_estimate_input_stats_source_size_multiplier(get_task, tmp_path):
    """When only SOURCE_SIZE is available, a safety multiplier should be
    applied to account for materialization expansion."""
    task: Task = get_task(
        TaskType.MULTICLASS_CLASSIFICATION,
        temporal=False,
        materialize=False,
        create_new=True,
    )
    graph = task.graph
    graph_key = VersionedGraphKey(graph.name, 'dummy_version')

    backend_cache = get_active_cache()

    source_size_per_table = int(40 * 10**9)  # 40 GB per table

    for table in graph.tables:
        for fkey in table.fkeys:
            assert fkey.dst_data_source and fkey.dst_table
            src_key = SourceTableColumn(table.data_source, table.source_name,
                                        fkey.name)
            dst_key = SourceTableColumn(fkey.dst_data_source,
                                        graph.table(fkey.dst_table).source_name,
                                        fkey.dst_key)
            backend_cache.graph_cache.add_edge_stats(
                graph.name, src_key, dst_key,
                {EdgeStatType.EDGE_COUNT: Stats(100, False)}, 'dummy_version')
        backend_cache.add_table_stats(
            table.fingerprint(), {
                TableStatType.SOURCE_SIZE:
                Stats(source_size_per_table, False,
                      data_version=table.data_version),
            })

    stats = activity_common.estimate_input_data_stats(graph, graph_key, None)
    num_tables = len(list(graph.tables))
    raw_total = source_size_per_table * num_tables
    # 3x multiplier applied
    assert stats.graph_size_bytes == int(raw_total * 3.0)
