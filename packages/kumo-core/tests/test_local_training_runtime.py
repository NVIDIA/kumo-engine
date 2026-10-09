# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from kumo_core.embedding_models import EmbeddingModelStore
from kumo_core.training.runtime import (
    LocalAutoTrainingRuntime,
    LocalExplainRuntime,
    local_training_bindings,
)


@pytest.mark.parametrize('use_input_id', [False, True])
def test_explain_selection_uses_hosted_policy_and_mapping(use_input_id):
    from kumo_core.training.explain_selection import select_input_nodes
    from kumo_core.types import TaskType
    from kumoapi.train import TrainingStage

    class FeatureStore:
        entity_table_names = ['users']
        time_col_dict = {}

        def __getitem__(self, key):
            return key[2] + 1000

    nodes = torch.tensor([9, 7, 9])
    input_id = torch.tensor([102, 100, 101]) if use_input_id else None
    args = (TaskType.REGRESSION, FeatureStore(), nodes, input_id, {
        'target': torch.tensor([0., 1., 4.]),
        'pred': torch.tensor([0., 3., 1.])
    }, TrainingStage.TEST)
    torch.manual_seed(42)
    expected_nodes, expected_mapping = select_input_nodes(*args)
    torch.manual_seed(42)
    selected, mapping = LocalExplainRuntime().select_input_nodes(*args)
    assert torch.equal(selected, expected_nodes)
    assert mapping == expected_mapping
    assert 'random' in mapping.mappings
    assert 'first' not in mapping.mappings
    assert mapping.orig_mappings['large_error'] == [1009, 1007, 1009]
    if use_input_id:
        assert selected.tolist() == [100, 101, 102]
        assert mapping.mappings['large_error'] == ['1', '0', '2']


def test_explain_uses_the_requested_stage_and_selected_seeds():
    calls = []
    nodes, ids = object(), object()
    datamodule = SimpleNamespace(
        stage_inputs=lambda stage: (calls.append(stage), nodes, None, ids),
        explain_dataloader=lambda stage, seeds: (stage, seeds),
    )
    runtime = LocalExplainRuntime()
    stage = SimpleNamespace(value='pred')

    assert runtime.get_explain_node_inputs(datamodule, stage) == (nodes, ids)
    assert runtime.get_eval_dataloader(datamodule, stage, ids) == ('pred', ids)
    assert calls == ['pred']


def test_local_runtime_binds_paths_and_refuses_a_different_task(tmp_path):
    task = SimpleNamespace(name='task', encoder_inputs={},
                           target_encoder_input=None)
    runtime = LocalAutoTrainingRuntime(home_dir=str(tmp_path),
                                       resolved_task=task, run_name='run')

    assert runtime.metrics_job_name() == 'run'
    assert runtime.default_home_dir() == str(tmp_path)
    assert runtime.artifact_runtime.local_model_root() == str(tmp_path)
    assert runtime.artifact_runtime.default_model_root() == str(tmp_path)
    assert runtime.load_task(task_name='task', home_dir=str(tmp_path)) is task
    assert runtime.load_task_from_home(str(tmp_path)) is task
    runtime.collect_encoders(task)
    with pytest.raises(ValueError, match='requested task'):
        runtime.load_task('different', str(tmp_path))
    with pytest.raises(ValueError, match='requested home'):
        runtime.load_task_from_home(str(tmp_path / 'other'))
    with pytest.raises(ValueError, match='encoder artifact'):
        runtime.collect_encoders(SimpleNamespace(name='other'))


def test_default_bindings_supply_shared_runtimes(tmp_path):
    training_input = SimpleNamespace(home_dir=str(tmp_path), task=object())

    store = EmbeddingModelStore()
    bindings = local_training_bindings(training_input, embedding_models=store)

    assert bindings.runtime.metrics_job_name() == 'kumo-engine'
    assert isinstance(bindings.runtime.explain_runtime, LocalExplainRuntime)
    kwargs = bindings.runtime.trainer_runtime_kwargs()
    assert kwargs['artifact_runtime'] is bindings.runtime.artifact_runtime
    assert kwargs['embedding_models'] is store
