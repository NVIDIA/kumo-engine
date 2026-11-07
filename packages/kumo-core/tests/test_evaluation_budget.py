# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from kumo_core.automl import auto_trainer
from kumo_core.automl.auto_trainer import AutoTrainer
from kumo_core.encoders.config import EncodersConfig
from kumo_core.instrumentation import NoOpTrainingInstrumentation
from kumo_core.model_plan import ModelPlan
from kumo_core.train.trainer import handle_gpu_oom
from kumo_core.types import TaskType
from kumoapi.model_plan import MissingType, RunMode
from kumoapi.pquery import QueryType


@pytest.mark.parametrize('steps,expected',
                         [(100, 204), (10_000, 20_313), (None, None),
                          (MissingType.VALUE, MissingType.VALUE)])
def test_automl_retry_preserves_evaluation_rows(steps, expected):
    trainer = AutoTrainer.__new__(AutoTrainer)
    trainer.num_experiments = 1
    trainer.max_tries, trainer.min_retry_batch_size = 3, 16
    cfg = ModelPlan()
    cfg.optimization.batch_size = [65]
    cfg.optimization.max_val_steps = cfg.optimization.max_test_steps = steps
    retried = trainer._handle_oom_retry(cfg, trial=0, try_idx=0)
    assert retried.optimization.batch_size == [32]
    assert retried.optimization.max_val_steps == expected
    assert retried.optimization.max_test_steps == expected


def test_retry_budget_reaches_the_trainer_constructor(tmp_path, monkeypatch):
    cfg = ModelPlan.from_default(task_type=TaskType.BINARY_CLASSIFICATION,
                                 query_type=QueryType.STATIC,
                                 has_train_table_time_col=False,
                                 run_mode=RunMode.DEBUG)
    cfg.optimization.batch_size = [64]
    cfg.optimization.max_val_steps, cfg.optimization.max_test_steps = 100, 200
    trainer = AutoTrainer.__new__(AutoTrainer)
    vars(trainer).update(
        root_dir=str(tmp_path), random_state_seed=0, num_experiments=1,
        max_tries=3, min_retry_batch_size=16, max_epochs=1,
        min_steps_per_epoch=1, max_steps_per_epoch=1,
        tune_metric='auroc', disable_compilation=True,
        _raise_fake_trial_error_msg=None, feature_store=None, graph_store=None,
        checkpoint_path=None, _encoders=EncodersConfig({}),
        diskgraph_task_dataset_bundle_dir='unused',
        instrumentation=NoOpTrainingInstrumentation(), task=SimpleNamespace(
            name='case', task_type=TaskType.BINARY_CLASSIFICATION,
            to_task_config=lambda: None),
        runtime=SimpleNamespace(trainer_runtime_kwargs=lambda: {}))
    for name in ('_get_encoder', '_get_model', '_get_head',
                 '_get_past_encoder'):
        setattr(trainer, name, lambda _: None)
    trainer._get_year_over_year_encoder = lambda: None
    cfg = trainer._handle_oom_retry(cfg, trial=0, try_idx=0)
    constructor = Mock(side_effect=RuntimeError('stop before initialization'))
    monkeypatch.setattr(auto_trainer, 'DiskgraphNestedTrainer', constructor)
    with pytest.raises(RuntimeError, match='stop before initialization'):
        trainer._train(cfg, metrics_client_job_name='test')
    received = constructor.call_args.kwargs
    assert (received['loader_kwargs']['batch_size'], received['max_val_steps'],
            received['max_test_steps']) == (32, 200, 400)


@pytest.mark.parametrize('stage,limit,caps', [
    ('test', 100, [100, 200, 400, 800]),
    ('test', None, [None, None, None, None]),
    ('predict', 100, [100, 100, 100, 100]),
])
@pytest.mark.parametrize('failure',
                         [None, torch.cuda.OutOfMemoryError, RuntimeError])
def test_evaluation_retry_preserves_budget_and_restores_state(
        stage, limit, caps, failure, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    module = SimpleNamespace(eval_loader_kwargs={'batch_size': 64},
                             max_test_steps=limit)
    trainer = SimpleNamespace(
        datamodule=module,
        task=SimpleNamespace(task_type=TaskType.BINARY_CLASSIFICATION),
        pl_model=SimpleNamespace(reset_embedding_hook=lambda: None))
    attempted = []

    def evaluate(self):
        batch = module.eval_loader_kwargs['batch_size']
        attempted.append((batch, module.max_test_steps))
        if batch > 16:
            raise torch.cuda.OutOfMemoryError('simulated')
        if failure is not None:
            raise failure('simulated')
        return 'complete'

    evaluate.__name__ = stage
    run = handle_gpu_oom(evaluate)
    if failure is not None:
        with pytest.raises(failure):
            run(trainer)
    else:
        assert run(trainer) == 'complete'
    expected = list(zip([64, 32, 16, 8], caps))
    assert attempted == expected[:4 if failure is torch.cuda.OutOfMemoryError
                                 else 3]
    if stage == 'test' or failure is None:
        assert module.eval_loader_kwargs == {'batch_size': 64}
    assert module.max_test_steps == limit
