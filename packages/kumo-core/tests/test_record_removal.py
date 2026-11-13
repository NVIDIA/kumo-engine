# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import pytest

from kumo_core.explain.perturbation import RecordRemovalPlan, reconstruct_batch, remove_record


def sample():
    node = {
        'pkeys': [10, 11, 10],
        'num_nodes': 3,
        'num_sampled_nodes': [0, 2, 1],
        'times_ns': [1, 2, 1],
        'node_ids': [0, 1, 0],
        'batch': [0, 0, 0],
        'feature_tensors': {
            'numerical': {
                'values': [[1], [2], [1]],
                'shape': [3, 1],
                'nonfinite': None
            }
        },
        'numerical_features': {
            'x': [1, 2, 1]
        }
    }
    return {
        'origin':
        'original_forward',
        'seed_table':
        'products',
        'train_table_name':
        '__task__',
        'entity_table_name':
        'products',
        'nodes': {
            'facts': node
        },
        'edges': [{
            'type': ['facts', 'product_id', 'products'],
            'edge_index': [[0, 1, 2], [0, 0, 0]],
            'num_sampled_edges': [2, 1]
        }, {
            'type': ['products', 'rev_product_id', 'facts'],
            'edge_index': [[0, 0], [0, 1]],
            'num_sampled_edges': [1, 1]
        }]
    }


def test_removes_all_occurrences_and_exact_incident_edges():
    graph = sample()
    before = copy.deepcopy(graph)
    plan = RecordRemovalPlan(({
        'table': 'facts',
        'pkey': 10
    }, ), ('facts', ), (('facts', 'product_id', 'products'), ))
    modified = remove_record(graph, plan.groups[0], plan)
    assert graph == before
    assert modified['nodes']['facts']['pkeys'] == [11]
    assert modified['nodes']['facts']['num_sampled_nodes'] == [0, 1, 0]
    assert modified['edges'][0]['edge_index'] == [[0], [0]]
    assert modified['edges'][1]['edge_index'] == [[0], [0]]
    assert modified['edges'][0]['num_sampled_edges'] == [1, 0]
    assert modified['edges'][1]['num_sampled_edges'] == [0, 1]


def test_rejects_parent_dependencies_and_seed_removal():
    graph = sample()
    plan = RecordRemovalPlan(
        ({
            'table': 'facts',
            'pkey': 10
        }, ), ('facts', ),
        (('facts', 'product_id', 'products'), ('child', 'fact_id', 'facts')))
    with pytest.raises(ValueError, match='inbound'):
        remove_record(graph, plan.groups[0], plan)
    with pytest.raises(ValueError, match='Seed'):
        remove_record(graph, {'table': 'products', 'pkey': 0}, plan)


def test_rejects_unknown_source_and_missing_schema():
    plan = RecordRemovalPlan((), ('facts', ), ())
    with pytest.raises(ValueError, match='schema'):
        remove_record(sample(), {'table': 'facts', 'pkey': 10}, plan)


def test_reconstruct_preserves_native_string_feature_keys():
    import torch
    graph = sample()
    graph['edges'] = []
    graph['seed_table'] = 'facts'
    feature = graph['nodes']['facts']['feature_tensors']['numerical']
    feature.update(dtype='torch.float32', columns=['quantity'])
    graph['nodes']['facts']['seed_time'] = [10]
    batch = reconstruct_batch(graph)
    assert batch['facts'].tf.feat_dict['numerical'].dtype == torch.float32
    assert batch['facts'].tf.col_names_dict['numerical'] == ['quantity']
    assert batch['facts'].seed_time.tolist() == [10]
