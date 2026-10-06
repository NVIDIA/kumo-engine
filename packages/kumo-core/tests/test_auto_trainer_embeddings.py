# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import types

import pytest
from kumo_core import embedding_models
from kumo_core.automl import auto_trainer
from kumo_core.automl.auto_trainer import AutoTrainer
from kumo_core.embedding_models import (
    EmbeddingModelNotFoundError,
    EmbeddingModelStore,
)
from kumo_core.encoders.config import EncodersConfig
from kumo_core.instrumentation import NoOpTrainingInstrumentation
from kumo_core.model_plan import ModelPlan
from kumo_core.types import TaskType
from kumoapi.encoder import TextEmbedding
from kumoapi.model_plan import RunMode
from kumoapi.pquery import QueryType


def test_missing_model_fails_the_run_before_any_trial(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_models, "_default_store", None)
    monkeypatch.setattr(auto_trainer, "check_gpu_ready", lambda: None)
    trials = []

    def train(self, **kwargs):
        trials.append(kwargs)
        _ = self.encoders
        return {}

    monkeypatch.setattr(AutoTrainer, "_train", train)
    monkeypatch.setattr(
        AutoTrainer, "_resolve_encoders", lambda self: EncodersConfig(
            {"users": {
                "bio": TextEmbedding("missing")
            }}))

    monkeypatch.setattr(
        auto_trainer, "TPESearcher", lambda space, **kwargs: types.
        SimpleNamespace(suggest=lambda: copy.deepcopy(space)))
    monkeypatch.setattr(AutoTrainer, "_make_pruning_callback",
                        lambda self: None)
    monkeypatch.setattr(AutoTrainer, "log_dir", str(tmp_path / "log"))

    trainer = AutoTrainer.__new__(AutoTrainer)
    trainer.instrumentation = NoOpTrainingInstrumentation()
    trainer.runtime = types.SimpleNamespace(
        trainer_runtime_kwargs=lambda: {
            "embedding_models": EmbeddingModelStore()
        }, metrics_job_name=lambda: "job")
    trainer.home_dir = str(tmp_path)
    trainer.space = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.STATIC,
        has_train_table_time_col=False,
        run_mode=RunMode.DEBUG,
    )
    trainer.num_experiments = 1
    trainer.random_state_seed = 0
    trainer.tune_metric = trainer.space.training_job.tune_metric
    trainer.task = types.SimpleNamespace(
        task_type=TaskType.BINARY_CLASSIFICATION)
    trainer.max_tries = 1
    trainer.min_retry_batch_size = 16

    with pytest.raises(EmbeddingModelNotFoundError, match="'users.bio'"):
        trainer._run()
    assert trials == []
    assert list(tmp_path.iterdir()) == []
