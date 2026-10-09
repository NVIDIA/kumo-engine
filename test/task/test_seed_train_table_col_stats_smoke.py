# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for the production train-table col-stats DataFrame seeder.

Exercises :meth:`kumo.task.task.Task.seed_train_table_col_stats_from_dataframe`
end-to-end: build a minimal :class:`Task` over a
:class:`PandasConnector` (which has no manifest sidecar and no
``_get_path`` resolver, so the existing
:meth:`Task.seed_train_table_col_stats_from_manifest` path raises),
seed the train-table col-stats cache from the in-memory train-table
DataFrame, and assert that
:meth:`Task._get_encoder_stats_from_cache_or_fail` resolves without
hitting the strict-path ``RuntimeError: diskgraph strict path requires
precomputed encoder stats; missing for ...`` raise at
``kumo/task/task.py``.

Without a DataFrame-friendly seed entry point, the only seeder
(:meth:`Task.seed_train_table_col_stats_from_manifest`) requires a
connector with ``_get_path`` and a Rust-producer manifest sidecar.
``PandasConnector``-backed test ``Task`` constructions therefore fall
off the strict path with the connector-resolver raise. This test
codifies that the sibling helper closes that gap.
"""
import numpy as np
import pandas as pd
from kumo_core.types import ColStatType
from kumoapi.typing import Dtype

from kumo.connector.pandas_connector import PandasConnector
from kumo.graph.graph import Graph
from kumo.task.task import Task


def test_seed_train_table_col_stats_from_dataframe_unblocks_strict_path(
    seed_col_stats, ):
    # 1. Build a minimal one-table graph with a PandasConnector. The
    #    ``seed_col_stats`` fixture covers entity-table cache seeding so
    #    ``graph.infer_metadata()`` can resolve. The new helper fills in
    #    the *train-table* cache namespace, which the entity-table
    #    helper does not cover.
    graph_df = pd.DataFrame({
        'STORE_ID': np.arange(20),
        'BRAND': ['PATAGONIA', 'KUMO'] * 10,
    })
    seed_col_stats(
        graph_df,
        table_name='STORES',
        data_version='seed-smoke-graph-v1',
        connector_name='pandas',
        primary_key='STORE_ID',
    )
    connector = PandasConnector({'STORES': graph_df})
    graph = Graph(name='seed_smoke_graph')
    graph.connect(connector)
    graph.add('STORES', pkey=dict(name='STORE_ID', dtype=Dtype.int))
    graph.infer_metadata()

    # 2. Build a Task with an in-memory train-table DataFrame. We pin
    # ``train_table_data_version`` so the cache key is stable; without
    # this the strict path raises a different (data_version-is-None)
    # error before reaching the missing-stat raise we care about here.
    train_df = pd.DataFrame({
        'STORE_ID': np.arange(20),
        'AMOUNT': np.linspace(1.0, 20.0, 20),
    })
    task = Task(
        graph,
        task_type='regression',
        train_table_name='TRAINING_AMOUNT',
        target_col=dict(name='AMOUNT', dtype=Dtype.float),
        fkeys=[{
            'name': 'STORE_ID',
            'dst_table': 'STORES',
        }],
        data_source=train_df,
        train_table_data_version='seed-smoke-v1',
        ignore_encoders=True,
        task_name='seed_smoke_task',
    )

    # Sanity: target_col is declared but not yet inferred — the cache
    # lookup will use ``target_col.base_config()``.
    assert task.train_table.target_col is not None
    assert task.train_table.target_col.name == 'AMOUNT'

    # 3. Seed the cache from the in-memory train-table DataFrame.
    seeded = task.seed_train_table_col_stats_from_dataframe(train_df)
    assert seeded is True

    # 4. Idempotency latch: a second call returns False without
    #    re-doing the work.
    assert task.seed_train_table_col_stats_from_dataframe(train_df) is False

    # 5. The strict-path encoder-stat lookup resolves. Without the
    #    seed, this would raise ``RuntimeError: diskgraph strict path
    #    requires precomputed encoder stats; missing for ...`` for the
    #    target column.
    target_stats = task._get_encoder_stats_from_cache_or_fail(
        task.train_table.target_col)

    # 6. Verify the seeded numeric stats round-trip the source data.
    assert ColStatType.MIN in target_stats
    assert ColStatType.MAX in target_stats
    assert target_stats[ColStatType.MIN].value == 1.0
    assert target_stats[ColStatType.MAX].value == 20.0
