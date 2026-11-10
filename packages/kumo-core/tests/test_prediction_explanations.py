# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import kumo_core.prediction_explanations as explanations
import pytest
import torch
from kumo_core.io import filesystem
from kumo_core.io.json import save_json
from kumo_core.task_dataset import TaskDatasetMaterializeResult
from kumo_core.training.config import FitMode, LinkPredModuleType
from kumo_core.training.explain_selection import select_input_nodes
from kumo_core.types.enums import TaskType
from kumoapi.train import TrainingStage


@pytest.mark.parametrize('task_type', [TaskType.REGRESSION,
                                       TaskType.STATIC_LINK_PREDICTION])
def test_prediction_replay_preserves_stage_ids(task_type, monkeypatch):
    """Collector ordinals follow selected stage rows, including temporal repeats."""
    from kumo_core.diskgraph import lightning_data
    from kumo_core.diskgraph.lightning_data import DiskgraphLightningDataModule

    class Store:
        entity_table_names = ['users']
        time_col_dict = {}

        def __getitem__(self, key):
            return key[2] + 100

    selected, mapping = select_input_nodes(
        task_type, Store(), torch.tensor([9, 7, 9]),
        torch.tensor([102, 100, 101]), {'pred': torch.ones(3)},
        TrainingStage.PRED)
    assert selected.tolist() == [100, 101, 102]
    assert mapping.mappings['first'] == ['2', '0', '1']
    assert mapping.orig_mappings['first'] == [109, 107, 109]

    module = DiskgraphLightningDataModule.__new__(DiskgraphLightningDataModule)
    module.runtime_options = {}
    module.eval_loader_kwargs = {'batch_size': 2}
    module.resolve_entity_node_type = 'users'
    module.resolve_target_node_type = 'items'
    module.task_manifest = SimpleNamespace(runtime_manifest_path='prediction')
    module.sampling_plan = {}
    module.base_seed = 0
    module._shuffle_state = None
    module.pin_memory = False
    module._task_type = task_type.value
    module._pseudo_time_options = Mock(return_value={})
    module._feature_projection = Mock(return_value={})
    module._to_hetero = lambda batch: batch
    iterable = Mock(return_value=[])
    monkeypatch.setattr(lightning_data, 'DiskgraphTaskBatchIterable', iterable)
    module.explain_dataloader('pred', selected)
    options = iterable.call_args.kwargs
    assert options['stage'] == 'predict'
    assert options['runtime_options']['selected_stage_node_ids'] == [100, 101, 102]
    assert options['runtime_options']['entity_node_type'] == 'users'
    assert 'target_node_type' not in options['runtime_options']
    assert options['dataloader_num_workers'] == 0


@pytest.mark.parametrize('task_type,module,expected', [
    (TaskType.REGRESSION, LinkPredModuleType.HYBRID, (True, True)),
    (TaskType.BINARY_CLASSIFICATION, LinkPredModuleType.HYBRID, (True, True)),
    (TaskType.MULTICLASS_CLASSIFICATION, LinkPredModuleType.HYBRID, (True, True)),
    (TaskType.MULTILABEL_CLASSIFICATION, LinkPredModuleType.HYBRID, (True, True)),
    (TaskType.STATIC_LINK_PREDICTION, LinkPredModuleType.HYBRID, (False, True)),
    (TaskType.TEMPORAL_LINK_PREDICTION, LinkPredModuleType.ID, (False, True)),
    (TaskType.STATIC_LINK_PREDICTION, LinkPredModuleType.TWO_TOWER, (False, False)),
    (TaskType.FORECASTING, LinkPredModuleType.HYBRID, (False, False)),
])
@pytest.mark.parametrize('threshold', [None, 0.73])
def test_prediction_capabilities(tmp_path, monkeypatch, task_type, module,
                                 expected, threshold):
    manifest = SimpleNamespace(task_type=task_type.value)
    monkeypatch.setattr(explanations.DiskgraphTaskManifest, 'load',
                        Mock(return_value=manifest))
    task = Mock()
    resolve = Mock(return_value=task)
    monkeypatch.setattr(explanations.ResolvedNumericalTask, 'from_manifest',
                        resolve)
    trainer = Mock(module_kwargs=SimpleNamespace(module=module))
    load = Mock(return_value=trainer)
    monkeypatch.setattr(explanations.DiskgraphNestedTrainer, 'load', load)

    def generate(*args, **kwargs):
        if expected[0]:
            save_json({}, str(tmp_path / 'xai/column_analysis/pred/column_analysis.json'))
        if expected[1]:
            save_json({'mappings': {'first': ['0']}}, str(tmp_path / 'xai/subgraph_collector/pred/strategy_mappings.json'))
            save_json({}, str(tmp_path / 'xai/subgraph_collector/pred/subgraph_json/0.json'))

    trainer.explain.side_effect = generate
    materialized = TaskDatasetMaterializeResult('manifest', 'runtime', 'bundle')
    encoders = Mock()
    options = {'unix_end_offset': 25}
    result = explanations.explain_prediction(
        materialized=materialized, model_dir='selected-model', run_id='run',
        output_dir=str(tmp_path / 'xai'), top_k=7, task_options=options,
        encoders=encoders, accelerator='cpu', binary_threshold=threshold)
    assert (result.cohorts, result.samples) == expected
    if any(expected):
        resolve.assert_called_once_with(manifest, name='run', **options)
        assert load.call_args.kwargs['encoders'] is encoders
        assert load.call_args.kwargs['binary_threshold'] == (
            0.5 if threshold is None else threshold)
        assert load.call_args.kwargs['diskgraph_task_dataset_bundle_dir'] == 'bundle'
        assert trainer._prediction_mode is True
        assert trainer.fit_mode is FitMode.fit
        assert trainer.explain.call_args.kwargs['top_k'] == 7
        assert trainer.explain.call_args.kwargs['collector_formats'] == ['nested_json']
        assert trainer.explain.call_args.kwargs['fit_mode_to_use'] is FitMode.fit
    else:
        trainer.explain.assert_not_called()


@pytest.mark.parametrize('partial', [False, True])
def test_missing_xai_preserves_predictions(tmp_path, monkeypatch, partial):
    predictions = str(tmp_path / 'prediction/_selected_artifacts.json')
    save_json({'selection': 'scored'}, predictions)
    monkeypatch.setattr(explanations.DiskgraphTaskManifest, 'load',
                        Mock(return_value=SimpleNamespace(task_type='regression')))
    monkeypatch.setattr(explanations.ResolvedNumericalTask, 'from_manifest', Mock())
    trainer = Mock()
    monkeypatch.setattr(explanations.DiskgraphNestedTrainer, 'load',
                        Mock(return_value=trainer))
    if partial:
        collector = tmp_path / 'xai/subgraph_collector/pred'
        save_json({'mappings': {'first': ['0', '1']}},
                  str(collector / 'strategy_mappings.json'))
        save_json({}, str(collector / 'subgraph_json/0.json'))
    with pytest.raises(RuntimeError, match='incomplete'):
        explanations.explain_prediction(
            materialized=TaskDatasetMaterializeResult('m', 'r', 'b'),
            model_dir='model', run_id='run', output_dir=str(tmp_path / 'xai'),
            encoders=Mock())
    assert filesystem.exists(predictions)
