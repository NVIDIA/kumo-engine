# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import torch
from diskgraph import SamplingPlan
from kumo_core.diskgraph import lightning_data
from kumo_core.diskgraph.lightning_data import DiskgraphLightningDataModule
from kumo_core.features.stores.feature import FeatureStore
from kumo_core.io.json import save_json
from kumo_core.task_dataset import DiskgraphTaskManifest
from kumo_core.training.explain_selection import select_prediction_inputs


@pytest.fixture
def module(tmp_path):
    save_json({'node_types': [{'name': 'users'}, {'name': 'items'}]},
              str(tmp_path / 'index_metadata.json'))
    manifest = DiskgraphTaskManifest(
        'manifest', 'prediction', {},
        {'train_table_name': 'train', 'seed_node_type': 'users',
         'task_type': 'static_link_prediction'}, {},
        {'graph_index_path': str(tmp_path)})
    return DiskgraphLightningDataModule(
        manifest, {}, sampling_plan=SamplingPlan(), train_batch_size=2,
        eval_batch_size=2, base_seed=17,
        runtime_options={'disjoint': True, 'fused': True},
        resolve_entity_node_type='users', resolve_target_node_type='items')


def test_prediction_discovery_skips_model_and_features(monkeypatch, module):
    calls = []

    def batches(**kwargs):
        calls.append(kwargs)
        return [
            {'task': {'seed_node_ids': [7, 7], 'tt_stage_node_ids': [20, 5]}},
            {'task': {'seed_node_ids': [9, 8], 'tt_stage_node_ids': [11, 19]}},
        ]

    monkeypatch.setattr(lightning_data, 'DiskgraphTaskBatchIterable', batches)
    nodes, rows = module.prediction_inputs(3)
    assert nodes.tolist() == [7, 7, 9]
    assert rows.tolist() == [20, 5, 11]
    keys = MagicMock(spec=FeatureStore)
    keys.entity_table_names = ['users']
    keys.__getitem__.side_effect = lambda key: key[2] + 1000
    selected, mapping = select_prediction_inputs(keys, nodes, rows, 3)
    assert selected.tolist() == [5, 11, 20]
    assert mapping.mappings == {'first': ['2', '0', '1']}
    assert mapping.orig_mappings == {'first': [1007, 1007, 1009]}
    assert isinstance(calls[0]['sampling_plan'], SamplingPlan)
    assert calls[0]['sampling_plan'].layers == ()
    assert calls[0]['projections'] == {'users': [], 'items': []}
    assert calls[0]['max_batches'] == 2
    assert calls[0]['replay'] is False
    assert calls[0]['runtime_options'] == {
        'disjoint': True, 'entity_node_type': 'users', 'pseudo_seed_time': None,
        'disable_pseudo_time': True, 'fused': False,
    }


@pytest.mark.parametrize('stage', ['predict', 'test', 'val'])
def test_seed_resolution_matches_selected_replay(stage, monkeypatch, module):
    calls = []

    def batches(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(lightning_data, 'DiskgraphTaskBatchIterable', batches)
    module.explain_dataloader('pred' if stage == 'predict' else stage,
                              torch.tensor([20, 5]))
    options = calls[0]['runtime_options']
    assert options['selected_stage_node_ids'] == [20, 5]
    assert options['entity_node_type'] == 'users'
    if stage == 'predict':
        assert 'target_node_type' not in options
    else:
        assert options['target_node_type'] == 'items'


def test_prediction_discovery_rejects_misaligned_rows(monkeypatch, module):
    monkeypatch.setattr(
        lightning_data, 'DiskgraphTaskBatchIterable', lambda **kwargs: [
            {'task': {'seed_node_ids': [7], 'tt_stage_node_ids': [20, 5]}},
        ])
    with pytest.raises(RuntimeError, match='differ in length'):
        module.prediction_inputs(3)


def test_selected_prediction_rows_restore_on_failure(monkeypatch, module):
    calls = []

    def batches(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(lightning_data, 'DiskgraphTaskBatchIterable', batches)
    module.predict_dataloader()
    assert 'selected_stage_node_ids' not in calls[-1]['runtime_options']
    rows = torch.tensor([20, 5, 11])
    with pytest.raises(RuntimeError, match='forward failed'):
        with module.selected_prediction_rows(rows):
            module.predict_dataloader()
            assert calls[-1]['runtime_options']['selected_stage_node_ids'] == [
                20, 5, 11]
            raise RuntimeError('forward failed')
    module.predict_dataloader()
    assert 'selected_stage_node_ids' not in calls[-1]['runtime_options']
