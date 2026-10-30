# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from kumo_core.summaries import Dataset, Scalar, Table, Tag

import kumo.engine as kumo
from kumo.engine._data_summaries import (
    snapshot_summaries,
    split_summaries,
    table_details,
    target_edges,
)
from kumo.engine._manifest import artifact_summaries
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
        'USER_ID': rng.integers(0, 360, 6000),
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

    summaries = snapshot_summaries(Path(snapshot.index_path),
                                   snapshot.graph_config)
    matched = orders['USER_ID'] < 300
    relation = 'graph/edges/orders_USER_ID_to_users_USER_ID'
    scalars = {s.name: s.value for s in summaries if isinstance(s, Scalar)}
    assert scalars == {
        'graph/rows/users': 300, 'graph/rows/orders': 6000,
        f'{relation}/count': matched.sum(),
        f'{relation}/src_match_frac': pytest.approx(matched.mean()),
        f'{relation}/dst_match_frac': pytest.approx(
            orders.loc[matched, 'USER_ID'].nunique() / 300),
        f'{relation}/fk_lookup_failures': (~matched).sum(),
        **{f'graph/{table}/{name}': 0 for table in ('users', 'orders')
           for name in ('duplicate_pks', 'dropped_rows', 'null_time_dropped')},
        'graph/tables': 2, 'graph/edge_types': 1}
    assert not [s for s in summaries if isinstance(s, Tag)]
    stats, = (s for s in summaries if isinstance(s, Table))
    age = next(dict(zip(stats.columns, row)) for row in stats.rows
               if row[:2] == ('users', 'AGE'))
    assert (age['stype'], age['count'], age['null_rate'], age['min'],
            age['max']) == ('numerical', 300, 1 / 300, users['AGE'].min(),
                            users['AGE'].max())
    assert age['mean'] == pytest.approx(users['AGE'].mean())
    datasets = {s.name: s for s in artifact_summaries(
        snapshot.manifest, 'training',
        details=table_details(Path(snapshot.index_path)))
        if isinstance(s, Dataset)}
    assert set(datasets) == {'users', 'orders'}
    assert datasets['orders'].schema == {
        'ORDER_ID': {'stype': 'id', 'dtype': 'int64'},
        'USER_ID': {'stype': 'id', 'dtype': 'int64'},
        'ORDER_TIME': {'stype': 'timestamp', 'dtype': 'date'}}
    assert datasets['orders'].profile['rows'] == 6000
    assert datasets['users'].source['location'] == str(
        (data / 'users.parquet').resolve())
    assert datasets['users'].context == 'training'


def test_data_warnings_name_each_table_that_breaks_a_check(tmp_path):
    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value))

    write('ingest_stats.json', {'stats': {
        'node_counts': {'users': 9, 'orders': 100},
        'source_rows_by_table': {'users': 12, 'orders': 100},
        'edge_counts': {'orders_USER_ID_to_users_USER_ID': 10},
        'pk_duplicate_counts': {'users:shard-0': 2},
        'pk_null_timestamp_drops': {'orders:shard-0': 3},
        'dropped_rows_counts': {'users': 1, 'orders': 3},
        'key_failures_by_context': {'pkey:users.USER_ID': {
            'parse_failures': 1}},
    }})
    write('col_stats.json', {'orders': {
        'ORDER_TIME': {'semantic_type': 'timestamp', 'count': 97,
                       'null_count': 2, 'parse_fail_count': 0},
        'AMOUNT': {'semantic_type': 'numerical', 'count': 100,
                   'null_count': 0, 'parse_fail_count': 4,
                   'invalid_count': 6},
    }})
    write('edge_stats.json', {
        f'orders__c__users__c__USER_ID__USER_ID__{stat}': {'value': value}
        for stat, value in (('FRAC_SRC_NONZERO_DEGREE', 0.1),
                            ('FRAC_DST_NONZERO_DEGREE', 0.5))})
    config = {'tables': [
        {'table_name': 'users', 'pkey': {'name': 'USER_ID'}, 'fkeys': []},
        {'table_name': 'orders', 'pkey': {'name': 'ORDER_ID'},
         'time_col': {'name': 'ORDER_TIME'},
         'fkeys': [{'name': 'USER_ID', 'dst_table': 'users',
                    'dst_key': 'USER_ID'}]},
    ]}

    tag, = (s for s in snapshot_summaries(tmp_path, config)
            if isinstance(s, Tag))
    assert tag == Tag('kumo.data_warnings', '; '.join([
        'users: 1 missing or invalid primary keys',
        'users: 2 duplicate primary keys',
        'users: 9 rows',
        'orders: 5.0% missing or invalid time',
        'orders.AMOUNT: at least 10% invalid values',
        'orders.USER_ID -> users: 10.0% of orders rows match',
    ]))


def test_target_edges_are_even_bins_without_a_histogram(tmp_path):
    target = {'target_min': 0, 'target_percentile_25': 0, 'target_median': 0,
              'target_percentile_75': 25, 'target_max': 200}
    (tmp_path / 'split_stats.json').write_text(json.dumps(
        {'train_table_stats': {'target_distribution': target}}))

    assert target_edges(tmp_path) == tuple(
        sorted({*(4.0 * i for i in range(51)), 25.0}))
