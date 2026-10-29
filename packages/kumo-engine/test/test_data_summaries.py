# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from kumo_core.summaries import Dataset, Scalar, Table

import kumo.engine as kumo
from kumo.engine._data_summaries import snapshot_summaries, split_summaries
from kumo.engine._scratch import get_scratch_dir
from kumo.engine.pquery import PredictiveQuery


@pytest.fixture
def scratch(tmp_path):
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    yield tmp_path
    kumo.configure(scratch_dir=previous)


def test_summaries_match_the_snapshot_and_training_table(scratch):
    rng = np.random.default_rng(0)
    data = scratch / 'data'
    data.mkdir()
    users = pd.DataFrame({'USER_ID': np.arange(300),
                          'AGE': rng.normal(40, 10, 300)})
    users.loc[0, 'AGE'] = None
    orders = pd.DataFrame({
        'ORDER_ID': np.arange(6000),
        'USER_ID': rng.integers(0, 300, 6000),
        'ORDER_TIME': pd.Timestamp('2024-01-01') + pd.to_timedelta(
            rng.integers(0, 120, 6000), unit='D'),
    })
    users.to_parquet(data / 'users.parquet')
    orders.to_parquet(data / 'orders.parquet')
    connector = kumo.FileConnector(root_dir=str(data))
    graph = kumo.Graph(
        tables={
            'users': kumo.Table.from_source_table(
                connector['users'], primary_key='USER_ID'),
            'orders': kumo.Table.from_source_table(
                connector['orders'], primary_key='ORDER_ID',
                time_column='ORDER_TIME').with_column('USER_ID', stype='ID'),
        },
        edges=[kumo.Edge(src_table='orders', fkey='USER_ID',
                         dst_table='users')])
    snapshot = graph.snapshot()
    table = PredictiveQuery(
        graph, 'PREDICT COUNT(orders.*, 0, 7, days) > 0 FOR EACH users.USER_ID'
    ).generate_training_table(snapshot)

    splits = {summary.name: summary.value
              for summary in split_summaries(table.staging_dir)}
    for split in ('train', 'val', 'test'):
        seeds = pd.read_parquet(table.staging_dir / f'{split}_seeds')
        assert splits[f'data/{split}/rows'] == len(seeds)
        assert splits[f'data/{split}/positive_rate'] == pytest.approx(
            seeds['TARGET1'].mean())

    summaries = snapshot_summaries(Path(snapshot.index_path), 'training')
    assert {s.name: s.value for s in summaries if isinstance(s, Scalar)} == {
        'graph/rows/users': 300, 'graph/rows/orders': 6000,
        'graph/edges/orders_USER_ID_to_users_USER_ID': 6000,
        'graph/tables': 2, 'graph/edge_types': 1}
    stats, = (s for s in summaries if isinstance(s, Table))
    age = next(dict(zip(stats.columns, row)) for row in stats.rows
               if row[:2] == ('users', 'AGE'))
    assert (age['stype'], age['count'], age['null_rate'], age['min'],
            age['max']) == ('numerical', 300, 1 / 300, users['AGE'].min(),
                            users['AGE'].max())
    assert age['mean'] == pytest.approx(users['AGE'].mean())
    datasets = {s.name: s for s in summaries if isinstance(s, Dataset)}
    assert set(datasets) == {'users', 'orders'}
    assert datasets['orders'].schema == {
        'ORDER_ID': {'stype': 'id', 'dtype': 'int64'},
        'USER_ID': {'stype': 'id', 'dtype': 'int64'},
        'ORDER_TIME': {'stype': 'timestamp', 'dtype': 'date'}}
    assert datasets['orders'].profile['rows'] == 6000
    assert datasets['users'].source['location'] == str(
        (data / 'users.parquet').resolve())
    assert datasets['users'].context == 'training'
