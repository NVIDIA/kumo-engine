# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for the DG col-stats cache seeding helper.

Exercises :func:`test.util.cache_seeding.seed_col_stats_cache` end-to-end:
build a minimal single-table graph, seed the cache from the same
DataFrame, and assert that :meth:`Table.compute_col_stats` resolves
without raising :class:`UnsupportedInDiskgraphError`.

Without ``seed_col_stats`` this test would raise a
``Requested col stats missing`` error because there is no lazy compute
path for DG-served col stats.
"""
import os.path as osp
import pathlib

import numpy as np
import pandas as pd
from kumo_core.types import ColStatType
from kumoapi.typing import Dtype

from kumo.connector.parquet_connector import ParquetConnector
from kumo.graph.graph import Graph


def test_seed_col_stats_unblocks_compute_col_stats(
    seed_col_stats,
    tmp_path: pathlib.Path,
):
    # 1. Build a minimal one-table parquet graph. The autouse fixture in
    #    test/conftest.py has already wired up the DG-backed cache, so
    #    we can build the Graph and Connector freely.
    df = pd.DataFrame({
        'STORE_ID': np.arange(10),
        'BRAND': ['PATAGONIA', 'KUMO'] * 5,
        'AMOUNT': np.linspace(1.0, 10.0, 10),
    })
    root = tmp_path.resolve().as_posix()
    parquet_path = osp.join(root, 'STORES.parquet')
    df.to_parquet(parquet_path)

    connector = ParquetConnector(name='parquet', root_dir=root)
    data_version = connector.get_data_version('STORES')

    # 2. Seed the cache from the same DataFrame, using the same
    #    fingerprint shape Table.compute_col_stats will look up.
    seed_col_stats(
        df,
        table_name='STORES',
        data_version=data_version,
        connector_name='parquet',
        primary_key='STORE_ID',
    )

    # 3. Build the Graph + Table and call compute_col_stats. Without the
    #    seed call above this raises UnsupportedInDiskgraphError.
    graph = Graph(name='seed_smoke_graph')
    graph.connect(connector)
    graph.add('STORES', pkey=dict(name='STORE_ID', dtype=Dtype.int))
    graph.infer_metadata()

    table = graph['STORES']
    out = table.compute_col_stats(
        col_stat_types=[
            ColStatType.NA_FRACTION,
            ColStatType.MIN,
            ColStatType.MAX,
            ColStatType.MEAN,
            ColStatType.STD,
        ],
        approx=True,
    )

    # 4. Verify the cache-read produced the stats we seeded. The numeric
    #    AMOUNT column is the cleanest assertion target — its MIN/MAX must
    #    match the source DataFrame.
    assert 'AMOUNT' in out
    amount_stats = out['AMOUNT']
    assert ColStatType.MIN in amount_stats
    assert ColStatType.MAX in amount_stats
    assert amount_stats[ColStatType.MIN].value == 1.0
    assert amount_stats[ColStatType.MAX].value == 10.0
