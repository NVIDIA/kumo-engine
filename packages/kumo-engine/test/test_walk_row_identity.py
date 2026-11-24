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
    _validate_graph(graph, kinds)
    block['pkeys'] = [11]
    with pytest.raises(EvidenceValidationError):
        _validate_graph(graph, kinds)
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
