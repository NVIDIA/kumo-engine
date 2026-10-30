# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kumo_core.model_plan import ModelPlan as CoreModelPlan
from kumo_core.model_plan import TrainingTableGenerationPlan
from kumo_core.resolved_task import ResolvedNumericalTask
from kumo_core.types import ExplainMethod
from kumoapi.model_plan import ModelPlan

from kumo.engine import Trainer, TrainingTable


@pytest.mark.parametrize('disabled,explain_fails,aggregate_fails', [
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (False, False, True),
])
@pytest.mark.parametrize('remote', [False, True])
@pytest.mark.parametrize('accelerator', ['cpu', 'gpu', 'mps', 'auto'])
def test_fit_runs_shared_explanations_after_training(tmp_path, snapshot,
                                                     monkeypatch, caplog,
                                                     staged_seed_dir, disabled,
                                                     explain_fails,
                                                     aggregate_fails,
                                                     accelerator, remote):
    plan = ModelPlan()
    plan.training_job.disable_explain = disabled
    endpoints = ['grpc://worker-a:8815', 'grpc://worker-b:8815']
    trainer = Trainer(
        plan, accelerator=accelerator,
        remote_sampling_endpoint=endpoints if remote else None,
        remote_sampling_num_threads=2,
    )
    monkeypatch.setattr(type(trainer), 'scratch_dir',
                        property(lambda _: tmp_path))
    table = TrainingTable(
        query='PREDICT users.label FOR EACH users.id',
        train_table_path=tmp_path / 'train.parquet',
        manifest_path=tmp_path / 'manifest.json',
        staging_dir=staged_seed_dir,
        required_stages=('train', 'val', 'test'),
        generation_plan=TrainingTableGenerationPlan(),
    )
    events = []

    def explain(**kwargs):
        events.append('explain')
        if explain_fails:
            raise RuntimeError('explanation failure')

    best_trainer = SimpleNamespace(
        explain=Mock(side_effect=explain),
        subgraph_collector_dir=lambda: str(tmp_path / 'subgraph_collector'),
        log_dir=str(tmp_path),
    )

    def aggregate_features(*args):
        events.append('aggregate')
        if aggregate_fails:
            raise RuntimeError('aggregation failure')

    aggregate = Mock(side_effect=aggregate_features)
    monkeypatch.setattr(
        'kumo.engine.trainer.trainer.aggregate_subgraph_features', aggregate)

    def train(training_input, **kwargs):
        assert kwargs['bindings'].runtime.trainer_runtime_kwargs()[
            'accelerator'] == accelerator
        assert kwargs['bindings'].runtime.diskgraph_runtime.sampling_options == (
            {'remote_sampling_endpoint': endpoints, 'num_threads': 2}
            if remote else {})
        events.append('train')
        return SimpleNamespace(
            metrics={'test_accuracy': 0.8},
            artifact_paths={'selected_checkpoint': '/model/selected.ckpt'},
            explain_disabled=training_input.model_plan.training_job.
            disable_explain,
            best_trainer=best_trainer,
        )

    core_plan = CoreModelPlan()
    core_plan.training_job.disable_explain = disabled
    monkeypatch.setattr(
        'kumo.engine.trainer.trainer.build_generated_training_task_dataset',
        lambda **kwargs: SimpleNamespace(feature_schema={'node_types': {}}))
    monkeypatch.setattr(
        'kumo_core.training.workflow.materialize_generated_training_task',
        lambda *args, **kwargs: SimpleNamespace(
            task=Mock(spec=ResolvedNumericalTask), model_plan=core_plan,
            bundle_dir=str(tmp_path)))
    monkeypatch.setattr(
        'kumo_core.training.orchestration.run_callable_training', train)
    monkeypatch.setattr('kumo.engine.trainer.trainer.write_training_spec',
                        lambda *args, **kwargs: events.append('finish'))

    result = trainer.fit(table, snapshot=snapshot)

    assert result.metrics == {'test_accuracy': 0.8}
    assert result.artifact_paths == {
        'selected_checkpoint': '/model/selected.ckpt'
    }
    assert trainer._training_result is result
    if disabled:
        assert events == ['train', 'finish']
        best_trainer.explain.assert_not_called()
    else:
        assert events == (['train', 'explain', 'finish'] if explain_fails else
                          ['train', 'explain', 'aggregate', 'finish'])
        # Leave fit-mode selection and supported-task checks to the shared
        # numerical Trainer.explain, just as the hosted training path does.
        best_trainer.explain.assert_called_once_with(
            explain_methods=[
                ExplainMethod.COLUMN_ANALYSIS, ExplainMethod.GRADIENT,
                ExplainMethod.SUBGRAPH_COLLECTOR
            ], stage='test')
    if disabled or explain_fails:
        aggregate.assert_not_called()
    else:
        from kumoapi.train import TrainingStage
        aggregate.assert_called_once_with(
            str(tmp_path / 'subgraph_collector' / 'test'), TrainingStage.TEST)
    if explain_fails or aggregate_fails:
        assert 'Training succeeded' in caplog.text
        assert ('explanation failure'
                if explain_fails else 'aggregation failure') in caplog.text
    else:
        assert 'post-training explanations failed' not in caplog.text
