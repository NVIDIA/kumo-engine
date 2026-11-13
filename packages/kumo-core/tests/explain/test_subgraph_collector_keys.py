# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from types import SimpleNamespace

import kumo_core.explain.subgraph_collector as collector_module
import numpy as np
import pandas as pd
import pytest
import torch
from kumo_core.explain.subgraph_collector import _set_column_values
from kumoapi.task import TaskType
from kumoapi.train import TrainingStage
from torch_geometric.data import HeteroData


@pytest.mark.parametrize('keys', [[297, 0], ['b', 'a']])
def test_object_keys_fill_a_numeric_column_without_warning(
        keys: list[object]) -> None:
    # DiskGraph attaches primary keys as an object array.
    df = pd.DataFrame({'movie_id': np.full(3, -1, dtype=np.int64)})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _set_column_values(df, 'movie_id', np.array([0, 2]),
                           np.array(keys, dtype=object))
    assert df['movie_id'].dtype == object
    assert df['movie_id'].tolist() == [keys[0], -1, keys[1]]


def test_numeric_keys_keep_a_numeric_column() -> None:
    df = pd.DataFrame({'movie_id': np.full(3, -1, dtype=np.int64)})
    _set_column_values(df, 'movie_id', np.array([1]), np.array([7]))
    assert df['movie_id'].dtype == np.int64
    assert df['movie_id'].tolist() == [-1, 7, -1]


@pytest.mark.parametrize('direction', ['forward', 'reverse', 'both'])
def test_foreign_keys_resolve_exact_relation_and_endpoint(
        tmp_path, monkeypatch, direction):
    """SupplyGraph group_id must not match subgroup_id or another endpoint."""
    batch = HeteroData()
    keys = {
        'products': ['P0', 'P1', 'P2'],
        'groups': ['G0', 'G1'],
        'subgroups': ['S0', 'S1', 'S2'],
        'other_groups': ['X0', 'X1']
    }
    for table, values in keys.items():
        batch[table].num_nodes = len(values)
        batch[table].pkey = np.array(values, dtype=object)
        batch[table].tf = SimpleNamespace()
    batch.input_type = 'products'
    batch['products'].batch_size = 3
    batch['products'].input_id = torch.arange(3)
    relations = [
        ('products', 'group_id', 'groups', [0, 0, 0]),
        ('products', 'subgroup_id', 'subgroups', [0, 1, 2]),
        ('products', 'backup_group_id', 'groups', [1, 1, 1]),
        ('products', 'group_id', 'other_groups', [1, 1, 1]),
    ]
    for src, relation, dst, destination in relations:
        edges = torch.tensor([[0, 1, 2], destination])
        if direction in ('forward', 'both'):
            batch[src, relation, dst].edge_index = edges
        if direction in ('reverse', 'both'):
            batch[dst, f'rev_{relation}', src].edge_index = edges.flip(0)
    store = SimpleNamespace(table_names=list(keys),
                            get_pkey_name=lambda table: 'id',
                            get_time_col_name=lambda table: None)
    schema = SimpleNamespace(
        get_foreign_keys=lambda table: [('group_id', 'groups'),
                                        ('subgroup_id', 'subgroups')]
        if table == 'products' else [])
    monkeypatch.setattr(
        collector_module, '_tf_to_df',
        lambda tf, table, store: pd.DataFrame(index=range(len(keys[table]))))
    collector = collector_module.SubgraphCollector(TaskType.REGRESSION,
                                                   str(tmp_path), ['csv'])
    captured = {}
    monkeypatch.setattr(
        collector, 'format_csv',
        lambda batch, seeds, frames, grads, stage: captured.update(frames))
    collector.explain_post_step(batch, SimpleNamespace(),
                                SimpleNamespace(data=[store, schema]), {},
                                TrainingStage.PRED, 0)
    assert captured['products']['group_id'].tolist() == ['G0'] * 3
    assert captured['products']['subgroup_id'].tolist() == ['S0', 'S1', 'S2']
