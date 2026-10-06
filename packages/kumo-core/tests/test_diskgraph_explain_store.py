# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
from kumo_core.diskgraph.runtime_adapter import _attach_graph_topology
from kumo_core.diskgraph.stores import DiskgraphFeatureStore
from kumo_core.encoders import TensorMapper
from kumo_core.encoders.tensor_frame_encoder import HeteroTensorFrameEncoder
from kumo_core.explain import SubgraphCollector
from kumo_core.explain.subgraph_collector import get_target_rhs_id
from kumo_core.features.stores import FeatureStoreTableSchema
from kumo_core.features.stores.feature import FeatureStoreTrainingTableSchema
from kumo_core.modeling.gnn.hetero import HeteroGNN
from kumo_core.modeling.heads.link_prediction import LinkPredHead
from kumo_core.train.shallow_rhs_link_gnn_module import (
    ShallowRHSLinkGNNLightningModule,
)
from kumo_core.types import TaskType
from kumoapi.encoder import Datetime
from kumoapi.train import TrainingStage
from kumoapi.typing import Stype
from torch_frame import TensorFrame
from torch_geometric.data import HeteroData


def test_collector_serializes_unencoded_temporal_columns(tmp_path):
    frame = pd.DataFrame({
        'id': ['event-1'],
        'created_at': pd.to_datetime(['2026-01-02']),
        'ended_at': pd.to_datetime(['2026-02-03']),
    })
    schema = FeatureStoreTableSchema(mapper_dict={}, pkey_name='id',
                                     time_col_name='created_at',
                                     end_time_col_name='ended_at')
    store = DiskgraphFeatureStore({'events': frame}, {'events': schema},
                                  pmm=False)
    batch = HeteroData()
    batch['events'].num_nodes = 1
    batch['events'].batch_size = 1
    batch['events'].input_id = torch.tensor([0])
    batch['events'].batch = torch.tensor([0])
    batch['events'].num_sampled_nodes = [1]
    collector = SubgraphCollector(TaskType.REGRESSION, str(tmp_path),
                                  formats=['nested_json'])
    collector.format_subgraph_json(
        batch, ['event-1'], {'events': frame}, {}, TrainingStage.TEST, store,
        SimpleNamespace(get_foreign_keys=lambda _: []), torch.nn.Identity())
    output = json.loads((tmp_path / 'test/nested_json/0.json').read_text())
    columns = {column['name']: column for column in output['cols']}
    assert columns['created_at']['stype'] == Stype.timestamp.value
    assert columns['ended_at']['stype'] == Stype.timestamp.value
    assert columns['created_at']['value'] == '2026-01-02T00:00:00'
    assert columns['ended_at']['value'] == '2026-02-03T00:00:00'
    assert (tmp_path / 'test/subgraph_json/0.json').stat().st_size > 0
    # Explanation metadata must not add new encoded model inputs.
    assert store._table_schema_dict['events'].mapper_dict == {}
    with pytest.raises(KeyError):
        store.get_tensor_mapper('events', 'missing_column')


def test_collector_tolerates_unreached_temporal_table(tmp_path):
    # The runtime adapter attaches ``time`` only to node types the sampler
    # reached; an unreached type is an empty block with ``pkey`` but no
    # ``time``, even though its table has a time column.
    schemas = {
        name: FeatureStoreTableSchema(mapper_dict={}, pkey_name='id',
                                      time_col_name='created_at')
        for name in ('lots', 'bids')
    }
    frame = pd.DataFrame({'id': [], 'created_at': pd.to_datetime([])})
    store = DiskgraphFeatureStore({name: frame
                                   for name in schemas}, schemas, pmm=False)
    batch = HeteroData()
    batch['lots'].pkey = np.array(['lot-1'], dtype=object)
    batch['lots'].tf = TensorFrame({}, {}, num_rows=1)
    batch['lots'].time = torch.tensor([1_767_312_000_000_000_000])
    batch['lots'].batch = torch.tensor([0])
    batch['lots'].batch_size = 1
    batch['lots'].input_id = torch.tensor([0])
    batch['bids'].n_id = torch.empty(0, dtype=torch.long)
    batch['bids'].pkey = np.array([], dtype=object)
    batch['bids'].tf = TensorFrame({}, {}, num_rows=0)
    batch['bids'].batch = torch.empty(0, dtype=torch.long)
    datamodule = SimpleNamespace(data=(
        store, SimpleNamespace(get_foreign_keys=lambda _: [])))

    collector = SubgraphCollector(TaskType.REGRESSION, str(tmp_path),
                                  formats=['csv'])
    collector.explain_post_step(batch, torch.nn.Identity(), datamodule, {},
                                TrainingStage.TEST, 0)

    lots = pd.read_csv(tmp_path / 'test/csv/0/lots.csv')
    assert lots.to_dict('records') == [{
        'id': 'lot-1',
        'created_at': '2026-01-02'
    }]
    bids = pd.read_csv(tmp_path / 'test/csv/0/bids.csv')
    assert list(bids.columns) == ['id', 'created_at'] and bids.empty


def test_explicit_temporal_mapper_is_preserved():
    mapper = TensorMapper.from_encoder(Datetime())
    schema = FeatureStoreTableSchema(mapper_dict={'created_at': mapper},
                                     time_col_name='created_at')
    store = DiskgraphFeatureStore(
        {'events': pd.DataFrame({'created_at': pd.to_datetime([])})},
        {'events': schema}, pmm=False)
    assert store.get_tensor_mapper('events', 'created_at') is mapper


# Relations: (src, rel, dst, per-hop counts, edges).
_LOT_TXN = 'transactions_SALE_LOT_KEY_to_lots_SALE_LOT_KEY'
_TXN_USER = 'transactions_PARTY_ID_MAIN_CLIENT_to_users_PARTY_ID_MAIN_CLIENT'
_LOT_PK = 'lots_TXN_KEY_to_transactions_TXN_KEY'


def _lp_explain_batch(relations, node_ids, batch_dict, per_hop_counts):
    channel = {
        'node_ids': {
            name: pa.array(ids, type=pa.uint64())
            for name, ids in node_ids.items()
        },
        'batch_dict': {
            name: pa.array(ids, type=pa.uint32())
            for name, ids in batch_dict.items()
        },
        'edge_index_dict': [{
            'edge_type': {
                'src': src,
                'rel': rel,
                'dst': dst
            },
            'src': pa.array(edges[0], type=pa.uint32()),
            'dst': pa.array(edges[1], type=pa.uint32()),
        } for src, rel, dst, _, edges in relations if edges[0]],
    }
    if per_hop_counts:
        channel['num_sampled_edges_dict'] = [{
            'edge_type': {
                'src': src,
                'rel': rel,
                'dst': dst
            },
            'counts': pa.array(counts, type=pa.uint64()),
        } for src, rel, dst, counts, _ in relations]
    batch = HeteroData()
    _attach_graph_topology(batch, channel)
    batch['lots'].batch_size = len(node_ids['lots'])
    return batch


def _format_rfm(tmp_path, batch, pred, y_idx, y_batch,
                entity_path=('transactions', 'SALE_LOT_KEY', 'lots')):
    batch['lots'].edge_label_index = torch.stack([y_batch, y_idx])
    module = ShallowRHSLinkGNNLightningModule(
        encoder=HeteroTensorFrameEncoder(table_encoder_dict={'users': {}},
                                         out_channels=4, time_col_dict={},
                                         exclude_cols_dict={}),
        model=HeteroGNN(channels=4, num_mp_layers=0,
                        metadata=(['lots', 'users'], [])),
        head=LinkPredHead(in_channels=4, embedding_dim=4,
                          use_rhs_projector=False, use_mlp_head=False),
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        train_table_name='TRAIN_TABLE',
        entity_table_names=('lots', 'users'),
        num_rhs_nodes=10,
        rhs_emb_mode='lookup',
    )
    store = DiskgraphFeatureStore(
        {
            'lots': pd.DataFrame({'id': ['lot-a', 'lot-b']}),
            'users': pd.DataFrame({'id': [7, 8, 9]}),
            'TRAIN_TABLE': pd.DataFrame({
                'ENTITY': ['lot-a'],
                'TARGET': [7],
                'train': [True],
                'val': [False],
            }),
        },
        {
            name: FeatureStoreTableSchema(mapper_dict={}, pkey_name='id')
            for name in ('lots', 'users')
        },
        train_table_schema=FeatureStoreTrainingTableSchema(
            table_name='TRAIN_TABLE', fkeys=[('ENTITY', 'lots'),
                                             ('TARGET', 'users')],
            train_mask_name='train', val_mask_name='val'), pmm=False)
    collector = SubgraphCollector(
        TaskType.TEMPORAL_LINK_PREDICTION, str(tmp_path), formats=['rfm'],
        target_to_entity_paths=[
            [entity_path],
            [('transactions', 'PARTY_ID_MAIN_CLIENT', 'users')],
        ])
    collector.format_rfm('lots', store, batch, ['lot-a', 'lot-b'],
                         TrainingStage.TEST, {'target': pred}, module)
    return [
        json.loads((tmp_path / f'test/rfm/{i}.json').read_text())
        for i in range(2)
    ]


def _counts(rfm):
    return (rfm['num_hist_items'], rfm['num_pred_hist_items'],
            rfm['num_true_items'], rfm['num_true_hist_items'])


def test_rfm_one_hop_plan(tmp_path):
    relations = [
        ('transactions', _LOT_TXN, 'lots', [0], ([], [])),
        ('lots', f'{_LOT_TXN}_reverse', 'transactions', [0], ([], [])),
        ('transactions', _TXN_USER, 'users', [0], ([], [])),
        ('users', f'{_TXN_USER}_reverse', 'transactions', [0], ([], [])),
    ]
    batch = _lp_explain_batch(relations, {'lots': [10, 11]},
                              {'lots': [0, 1]}, per_hop_counts=True)
    store = batch['users', 'rev_PARTY_ID_MAIN_CLIENT', 'transactions']
    assert store.num_sampled_edges == [0]

    rfms = _format_rfm(tmp_path, batch, pred=torch.tensor([[8, 5], [7, 9]]),
                       y_idx=torch.tensor([7, 9, 4]),
                       y_batch=torch.tensor([0, 1, 1]))
    assert [_counts(rfm) for rfm in rfms] == [(0, 0, 1, 0), (0, 0, 2, 0)]


def _two_hop_batch(per_hop_counts, txn_to_lot=_LOT_TXN,
                   lot_to_txn=f'{_LOT_TXN}_reverse', extra=(), totals=False):
    # Lot 0 -> users 7, 8; lot 1 -> user 9.
    relations = [
        ('transactions', txn_to_lot, 'lots', [3, 0], ([0, 1, 2], [0, 0, 1])),
        ('lots', lot_to_txn, 'transactions', [0, 0], ([], [])),
        ('transactions', _TXN_USER, 'users', [0, 0], ([], [])),
        ('users', f'{_TXN_USER}_reverse', 'transactions', [0, 3],
         ([0, 1, 2], [0, 1, 2])),
        *extra,
    ]
    if totals:  # Counts reported, but not split by hop.
        relations = [(*r[:3], [sum(r[3])], r[4]) for r in relations]
    return _lp_explain_batch(
        relations,
        {
            'lots': [10, 11],
            'transactions': [100, 101, 102],
            'users': [7, 8, 9]
        },
        {
            'lots': [0, 1],
            'transactions': [0, 0, 1],
            'users': [0, 0, 1]
        },
        per_hop_counts=per_hop_counts or totals,
    )


@pytest.mark.parametrize('per_hop_counts', [True, False])
def test_rfm_history(tmp_path, per_hop_counts):
    batch = _two_hop_batch(per_hop_counts)
    store = batch['users', 'rev_PARTY_ID_MAIN_CLIENT', 'transactions']
    assert store.num_sampled_edges == ([0, 3] if per_hop_counts else [3])

    rfms = _format_rfm(tmp_path, batch, pred=torch.tensor([[8, 5], [7, 9]]),
                       y_idx=torch.tensor([7, 9, 4]),
                       y_batch=torch.tensor([0, 1, 1]))
    assert [_counts(rfm) for rfm in rfms] == [(2, 1, 1, 1), (1, 1, 2, 1)]


@pytest.mark.parametrize('entity_path, txn_to_lot, lot_to_txn', [
    # Target PK is the key to the entity:
    (('lots', 'SALE_LOT_KEY', 'transactions'), _LOT_TXN,
     f'{_LOT_TXN}_reverse'),
    # Entity holds a key to the target PK:
    (('lots', 'TXN_KEY', 'transactions'), f'{_LOT_PK}_reverse', _LOT_PK),
])
def test_rfm_history_reversed_entity_path(tmp_path, entity_path, txn_to_lot,
                                          lot_to_txn):
    batch = _two_hop_batch(False, txn_to_lot, lot_to_txn)
    rfms = _format_rfm(tmp_path, batch, pred=torch.tensor([[8, 5], [7, 9]]),
                       y_idx=torch.tensor([7, 9, 4]),
                       y_batch=torch.tensor([0, 1, 1]),
                       entity_path=entity_path)
    assert [_counts(rfm) for rfm in rfms] == [(2, 1, 1, 1), (1, 1, 2, 1)]


def test_rfm_history_empty_forward_store(tmp_path):
    # Reciprocal key: an empty transactions -> lots TXN_KEY store must not
    # hide the populated rev_TXN_KEY one.
    reciprocal = 'transactions_TXN_KEY_to_lots_TXN_KEY'
    batch = _two_hop_batch(False, f'{_LOT_PK}_reverse', _LOT_PK, extra=[
        ('transactions', reciprocal, 'lots', [0, 0], ([], [])),
        ('lots', f'{reciprocal}_reverse', 'transactions', [0, 0], ([], [])),
    ], totals=True)
    assert batch['transactions', 'TXN_KEY', 'lots'].edge_index.numel() == 0
    rfms = _format_rfm(tmp_path, batch, pred=torch.tensor([[8, 5], [7, 9]]),
                       y_idx=torch.tensor([7, 9, 4]),
                       y_batch=torch.tensor([0, 1, 1]),
                       entity_path=('lots', 'TXN_KEY', 'transactions'))
    assert [_counts(rfm) for rfm in rfms] == [(2, 1, 1, 1), (1, 1, 2, 1)]


def test_historical_items_interleaved_seeds(tmp_path):
    # Second-hop edges arrive in sampler order, not grouped by seed.
    relations = [
        ('transactions', _LOT_TXN, 'lots', [3, 0], ([0, 1, 2], [1, 0, 0])),
        ('lots', f'{_LOT_TXN}_reverse', 'transactions', [0, 0], ([], [])),
        ('transactions', _TXN_USER, 'users', [0, 0], ([], [])),
        ('users', f'{_TXN_USER}_reverse', 'transactions', [0, 3],
         ([0, 1, 2], [0, 1, 2])),
    ]
    batch = _lp_explain_batch(
        relations,
        {
            'lots': [10, 11],
            'transactions': [100, 101, 102],
            'users': [7, 8, 9]
        },
        {
            'lots': [0, 1],
            'transactions': [1, 0, 0],
            'users': [1, 0, 0]
        },
        per_hop_counts=True,
    )
    collector = SubgraphCollector(
        TaskType.TEMPORAL_LINK_PREDICTION, str(tmp_path),
        formats=['historical_items'], target_to_entity_paths=[
            [('transactions', 'SALE_LOT_KEY', 'lots')],
            [('transactions', 'PARTY_ID_MAIN_CLIENT', 'users')],
        ])
    store = DiskgraphFeatureStore(
        {'users': pd.DataFrame({'id': []})},
        {'users': FeatureStoreTableSchema(mapper_dict={}, pkey_name='id')},
        pmm=False)
    collector.format_historical_items(batch, ['lot-a', 'lot-b'],
                                      {'users': pd.DataFrame({'id': [7, 8, 9]})},
                                      TrainingStage.TEST, store)
    items = [
        json.loads((tmp_path / f'test/historical_items/{i}.json').read_text())
        for i in range(2)
    ]
    assert [sorted(n['id'] for n in item['nodes'])
            for item in items] == [[8, 9], [7]]


@pytest.mark.parametrize('paths', [
    [],
    [[('orders', 'user_id', 'users')]],
    [[('orders', 'user_id', 'users'), ('users', 'region_id', 'regions')],
     [('orders', 'item_id', 'items')]],
    [[('orders', 'user_id', 'users')], [('orders', 'item_id', 'items')]],
])
def test_rfm_bad_paths_raise(paths):
    with pytest.raises(ValueError):
        get_target_rhs_id(paths, _two_hop_batch(True))
