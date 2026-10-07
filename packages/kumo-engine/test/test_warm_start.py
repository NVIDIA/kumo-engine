# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from kumo_core.automl.runtime import validate_warm_start_task_type
from kumo_core.io.filesystem import local_path
from kumo_core.io.json import load_json
from kumo_core.model_plan import ModelPlan as CoreModelPlan
from kumo_core.train.base_module import BaseLightningModule
from kumo_core.train.trainer import get_best_ckpt_path
from kumo_core.types import TaskType
from kumoapi.model_plan import ModelPlan
from kumoapi.plugin import ModelFetchingSink

from kumo.engine import FileConnector, Graph, PredictiveQuery, Table, Trainer


def _small_plan(refit):
    plan = ModelPlan()
    plan.training_job.num_experiments = 1
    plan.training_job.refit_trainval = refit is True
    plan.training_job.refit_full = refit == "full"
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 4
    plan.optimization.batch_size = [32]
    plan.model_architecture.channels = [16]
    return plan


def _hash_bundle(root):
    return {
        str(path.relative_to(root)):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob('*') if path.is_file()
    }


def test_unfitted_warm_start_rejected(snapshot, tmp_path, monkeypatch):
    from kumo_core.model_plan import TrainingTableGenerationPlan
    monkeypatch.setattr('kumo.engine._scratch._root', tmp_path)
    table = SimpleNamespace(generation_plan=TrainingTableGenerationPlan())
    with pytest.raises(ValueError, match='fitted or loaded'):
        Trainer(ModelPlan()).fit(table, snapshot=snapshot,
                                 warm_start=Trainer(ModelPlan()))


@pytest.mark.parametrize('current,source,message', [
    (TaskType.FORECASTING, TaskType.FORECASTING,
     'not supported for forecasting'),
    (TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION, 'different'),
])
def test_hosted_compatibility_policy(current, source, message):
    with pytest.raises(ValueError, match=message):
        validate_warm_start_task_type(current, source)


@pytest.mark.parametrize('refit', [False, True, 'full'])
def test_bundle_warm_start_fit_reload_and_predict(tmp_path, monkeypatch, refit,
                                                  request):
    """Real CPU fit and BP, with only the exported bundle retained as source."""
    monkeypatch.setattr('kumo.engine._scratch._root', tmp_path / 'scratch')
    thread_count = torch.get_num_threads()
    torch.set_num_threads(2)
    request.addfinalizer(lambda: torch.set_num_threads(thread_count))
    data = tmp_path / 'data'
    data.mkdir()
    pq.write_table(
        pa.table({
            'id': range(120),
            'value': [i * 0.1 for i in range(120)],
            'label': [i % 2 == 0 for i in range(100)] + [None] * 20
        }), data / 'users.parquet')
    graph = Graph({
        'users':
        Table.from_source_table(
            FileConnector(root_dir=str(data))['users'], primary_key='id')
    })
    snapshot = graph.snapshot()
    query = PredictiveQuery(graph, 'PREDICT users.label FOR EACH users.id')
    table = query.generate_training_table(snapshot)
    initial = Trainer(_small_plan(refit))
    first = initial.fit(table, snapshot=snapshot)
    # The bundle must retain the most-data checkpoint, not the last tested
    # fit mode. Check before deleting the original trainer directories.
    checkpoints = list(initial.scratch_dir.rglob('*.ckpt'))
    fit_suffix = ('_refit_full' if refit == 'full' else
                  '_refit_trainval' if refit is True else 'version_0')
    checkpoint_dirs = {
        path.parent
        for path in checkpoints if path.parent.parent.name.endswith(fit_suffix)
    }
    assert len(checkpoint_dirs) == 1
    selected, _ = get_best_ckpt_path(str(checkpoint_dirs.pop()))
    assert selected is not None
    assert Path(first.artifact_paths['selected_checkpoint']).read_bytes() == (
        local_path(selected).read_bytes())
    durable = tmp_path / 'retained-model'
    shutil.copytree(first.bundle_dir, durable)
    shutil.rmtree(initial.scratch_dir)
    if refit == 'full':
        sink = Mock(spec=ModelFetchingSink)
        sink.fetch_model.side_effect = lambda connector, target, **kwargs: shutil.copytree(
            durable, target)
        source = Trainer.load_from_connector(SimpleNamespace(sink_backend=sink),
                                             path='retained-model')
    else:
        source = Trainer.load(durable)
    if refit is False:
        wrong_query = PredictiveQuery(graph,
                                      'PREDICT users.value FOR EACH users.id')
        wrong_table = wrong_query.generate_training_table(snapshot)
        with pytest.raises(ValueError,
                           match='different from the current task type'):
            Trainer(_small_plan(False)).fit(wrong_table, snapshot=snapshot,
                                            warm_start=source)
    source_bundle = source._training_result.bundle_dir
    original_hashes = _hash_bundle(source_bundle)
    # Public property mutations must not change the persisted fitted structure.
    source.model_plan.model_architecture.channels = [64]
    destination = Trainer(_small_plan(False if refit == "full" else refit))
    destination.model_plan.model_architecture.channels = [32]
    destination.model_plan.optimization.batch_size = [16]
    configured_plan = copy.deepcopy(destination.model_plan)

    restored_states = []
    configured_states = []
    original_load = BaseLightningModule.load_checkpoint
    original_configure = BaseLightningModule.configure_optimizers

    def load(self, path, *args, **kwargs):
        result = original_load(self, path, *args, **kwargs)
        if str(path) == source._training_result.artifact_paths[
                'selected_checkpoint']:
            assert kwargs['load_optimizer_state'] is True
            checkpoint = torch.load(path, map_location='cpu',
                                    weights_only=False)
            for name, value in checkpoint['state_dict'].items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(self.state_dict()[name], value,
                                               rtol=0, atol=0)
            assert self._loaded_optimizer_states
            restored_states.append(copy.deepcopy(self._loaded_optimizer_states))
        return result

    def configure(self):
        states = copy.deepcopy(self._loaded_optimizer_states)
        result = original_configure(self)
        if states:
            for expected, config in zip(states, result):
                actual = config['optimizer'].state_dict()['state']
                assert actual.keys() == expected['state'].keys()
                for key, entry in expected['state'].items():
                    for name, value in entry.items():
                        if isinstance(value, torch.Tensor):
                            torch.testing.assert_close(actual[key][name], value,
                                                       rtol=0, atol=0)
            configured_states.append(states)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(BaseLightningModule, 'load_checkpoint', load)
        patch.setattr(BaseLightningModule, 'configure_optimizers', configure)
        second = destination.fit(table, snapshot=snapshot, warm_start=source)
    assert restored_states and configured_states
    assert destination.model_plan == configured_plan
    assert _hash_bundle(durable) == original_hashes
    assert _hash_bundle(source_bundle) == original_hashes
    assert (Path(second.artifact_paths['encoders']).read_bytes() == Path(
        source._training_result.artifact_paths['encoders']).read_bytes())
    effective = load_json(second.artifact_paths['resolved_model_plan'],
                          CoreModelPlan)
    source_plan = load_json(
        source._training_result.artifact_paths['resolved_model_plan'],
        CoreModelPlan)
    assert effective.model_architecture == source_plan.model_architecture
    assert effective.column_processing == source_plan.column_processing
    assert effective.neighbor_sampling == source_plan.neighbor_sampling
    assert effective.model_architecture.channels == [16]
    assert effective.optimization.batch_size == [16]
    restored = Trainer.load(second.bundle_dir)
    assert restored.model_plan.model_architecture.channels == [16]
    assert restored.model_plan.optimization.batch_size == [16]
    predictions = restored.predict(query.generate_prediction_table(snapshot),
                                   snapshot=snapshot,
                                   binary_classification_threshold=0.5)
    assert predictions.num_rows == 20
    assert len(predictions.frame) == 20
    if refit is False:
        third = destination.fit(table, snapshot=snapshot)
        ordinary_plan = load_json(third.artifact_paths['resolved_model_plan'],
                                  CoreModelPlan)
        assert ordinary_plan.model_architecture.channels == [32]
