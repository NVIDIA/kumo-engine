# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from kumo_core.explain.local_sensitivity import LocalSensitivity
from torch_geometric.data import HeteroData

from kumo.engine.evidence.bundle import EvidenceValidationError, _validate_graph


@pytest.mark.parametrize('ids', [[], [5, 9, 5]])
def test_unkeyed_walk_identity(tmp_path, ids):
    kinds = {'events': 'snapshot_row', 'customers': 'source_primary_key'}
    capture = LocalSensitivity(str(tmp_path), {'source_identity_kinds': kinds})
    batch = HeteroData()
    batch.input_type = 'customers'
    batch['customers'].n_id = torch.tensor([3])
    batch['customers'].pkey = ['customer-3']
    batch['events'].n_id = torch.tensor(ids, dtype=torch.long)
    module = SimpleNamespace(train_table_name='task')
    graph = capture._sample(batch, module)
    block = graph['nodes']['events']
    assert block['identity_kind'] == 'snapshot_row'
    assert block['node_ids'] == block['pkeys'] == ids
    for invalid_id in (None, '', '  ', 7):
        with pytest.raises(EvidenceValidationError, match='graph_snapshot_id'):
            _validate_graph(graph, kinds, invalid_id)
    _validate_graph(graph, kinds, 'test-snapshot')
    block['pkeys'] = [11]
    with pytest.raises(EvidenceValidationError):
        _validate_graph(graph, kinds, 'test-snapshot')
    del batch['customers'].pkey
    with pytest.raises(ValueError, match='source_primary_key identities for customers'):
        capture._sample(batch, module)


def test_removing_repeated_unkeyed_row_preserves_equal_valued_peer():
    from kumo_core.explain.perturbation import RecordRemovalPlan, remove_record

    events = {
        'num_nodes': 3, 'num_sampled_nodes': [0, 3],
        'identity_kind': 'snapshot_row', 'pkeys': [5, 9, 5],
        'node_ids': [5, 9, 5], 'feature_tensors': {},
        'numerical_features': {'amount': [100, 100, 100]},
    }
    relation = ('events', 'customer_id', 'customers')
    sample = {
        'seed_table': 'customers', 'entity_table_name': 'customers',
        'train_table_name': 'task', 'nodes': {'events': events},
        'edges': [{'type': relation, 'edge_index': [[0, 1, 2], [0, 0, 0]],
                   'num_sampled_edges': [3]}],
    }
    group = {'table': 'events', 'pkey': 5}
    plan = RecordRemovalPlan((group,), ('events',), (relation,))
    retained = remove_record(sample, group, plan)
    remaining = retained['nodes']['events']
    assert remaining['pkeys'] == remaining['node_ids'] == [9]
    assert remaining['numerical_features']['amount'] == [100]
    assert retained['edges'][0]['edge_index'] == [[0], [0]]


@pytest.mark.parametrize('policy', ['raise', 'record'])
def test_unkeyed_numerical_support_policy(tmp_path, policy):
    import json

    from kumo_core.encoders.fused.numerical import FusedNumericalEncoder
    from kumo_core.encoders.target.numerical import NumericalTargetEncoder
    from kumo_core.train.loss import MSELoss
    from kumoapi.encoder import NAStrategy, Numerical
    from kumoapi.task import TaskType

    numeric = FusedNumericalEncoder(
        {'amount': Numerical(na_strategy=NAStrategy.ZERO)}, [], 1, 'fuse')
    events = torch.nn.Module()
    events.stype_encoder_dict = torch.nn.ModuleDict({'numerical': numeric})
    encoder = torch.nn.Module()
    encoder.enc_dict = torch.nn.ModuleDict({'events': events})
    module = SimpleNamespace(
        encoder=encoder, train_table_name='task', device=torch.device('cpu'),
        task_type=TaskType.REGRESSION, loss=MSELoss(TaskType.REGRESSION),
        target_encoder=NumericalTargetEncoder(0, 1))
    dm = SimpleNamespace(source_identity_kinds={'events': 'snapshot_row'})
    capture = LocalSensitivity(str(tmp_path), {'run_id': 'test'},
                               walk_on_unsupported=policy)
    if policy == 'raise':
        with pytest.raises(ValueError, match='source primary key.*events'):
            capture.explain_start(module, dm, None)
        assert not list(tmp_path.iterdir())
        return

    capture.explain_start(module, dm, None)
    batch = HeteroData()
    batch.input_type = 'customers'
    batch['customers'].n_id = torch.tensor([0])
    batch['customers'].input_id = torch.tensor([0])
    batch['customers'].pkey = ['customer']
    batch['events'].n_id = torch.tensor([5])
    output = {'logits': torch.tensor([[1.]]), 'pkey': ['customer'], 'time': None}
    capture.explain_post_step(batch, module, dm, output, None, 0)
    row = json.loads((tmp_path / 'index.jsonl').read_text())
    assert row['status'] == 'unsupported' and 'events' in row['reason']
    assert row['graph_status'] == 'available'
    capture.explain_end(module, dm, {}, None)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['numerical_sensitivity_requested'] is True
    assert manifest['numerical_sensitivity'] is False
    assert manifest['numerical_sensitivity_unsupported_reason'] == row['reason']
    assert manifest['complete'] is False


def test_snapshot_row_without_primary_key_column(monkeypatch):
    from kumo_core.diskgraph.lightning_data import DiskgraphLightningDataModule

    dm = object.__new__(DiskgraphLightningDataModule)
    dm._pk_col_dict = None
    monkeypatch.setattr(dm, '_load_index_node_types', lambda: [
        {'name': 'events', 'primary_key_map': {'path': 'events.idx'}},
        {'name': 'customers', 'primary_key_map': {'column': 'id'}},
    ])
    assert dm.pk_col_dict == {'customers': 'id'}
    assert dm.source_identity_kinds == {
        'events': 'snapshot_row', 'customers': 'source_primary_key'}
