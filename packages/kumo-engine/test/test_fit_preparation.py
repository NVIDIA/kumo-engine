# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trainer delegates preparation to the same core workflow used by workers."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from kumo_core.model_plan import TrainingTableGenerationPlan
from kumo_core.training import preparation
from kumo_core.training.table import GeneratedTrainTableResult
from kumo_core.training.workflow import TrainingWorkflowRequest
from kumoapi.model_plan import ModelPlan

import kumo.engine as kumo


def test_local_fit_forwards_rows_and_feature_validation_to_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local adapter forwards existing rows and checks the trainer's model store."""
    plan = ModelPlan()
    trainer = kumo.Trainer(plan, accelerator="cpu")
    request = Mock(spec=TrainingWorkflowRequest)
    table = kumo.TrainingTable(
        query="PREDICT users.label FOR EACH users.id",
        train_table_path=tmp_path / "tt.parquet",
        manifest_path=tmp_path / "manifest.json",
        staging_dir=tmp_path / "staging",
        required_stages=("train", "val", "test"),
        generation_plan=TrainingTableGenerationPlan(),
    )
    shared = Mock()
    check_features = Mock()
    monkeypatch.setattr(preparation, "prepare_training_task", shared)
    monkeypatch.setattr(
        "kumo.engine.trainer.trainer.require_embedding_models", check_features
    )

    result = trainer._prepare_training(request, table, plan)

    assert result is shared.return_value
    shared.assert_called_once_with(
        request,
        GeneratedTrainTableResult(
            str(table.train_table_path),
            str(table.manifest_path),
            str(table.staging_dir),
        ),
        table.required_stages,
        validate_features=shared.call_args.kwargs["validate_features"],
    )
    schema: dict[str, dict[str, object]] = {"node_types": {}}
    shared.call_args.kwargs["validate_features"](schema)
    check_features.assert_called_once_with(
        trainer._embedding_models,
        schema,
        encoder_overrides=plan.column_processing.encoder_overrides,
    )


def test_local_fit_reaches_training_only_after_shared_preparation(
    table_and_snapshot: tuple[kumo.TrainingTable, kumo.GraphSnapshot],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real local fit prepares once through core before entering training."""
    table, snapshot = table_and_snapshot
    trainer = kumo.Trainer(ModelPlan(), accelerator="cpu")
    shared = Mock(wraps=preparation.prepare_training_task)
    training = Mock(side_effect=RuntimeError("stop before training execution"))
    monkeypatch.setattr(preparation, "prepare_training_task", shared)
    monkeypatch.setattr(trainer, "_train", training)

    with pytest.raises(RuntimeError, match="stop before training execution"):
        trainer.fit(table, snapshot=snapshot)

    shared.assert_called_once()
    training.assert_called_once()
    assert training.call_args.args[0].train_table_extension_path
