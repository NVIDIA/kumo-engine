# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from kumo_core.callbacks import TrainerProgress, TrainingProgressLogger
from kumo_core.io.json import load_json, save_json
from kumoapi.jobs import EpochMetrics


def test_logger_initialization_does_not_overwrite_completed_progress(
        tmp_path) -> None:
    path = tmp_path / TrainingProgressLogger.FILE_NAME
    completed = TrainerProgress()
    completed.num_epochs_completed = 4
    completed.metrics = {
        epoch:
        EpochMetrics(
            time_taken=1.0,
            train_metrics={'train_loss': 0.1},
            validation_metrics={'val_ap': 0.2},
        )
        for epoch in range(completed.num_epochs_completed)
    }
    save_json(completed, str(path))
    original = path.read_bytes()

    trainer = SimpleNamespace(
        log_dir=str(tmp_path),
        max_epochs=5,
        tune_metric='ap',
    )
    progress_logger = TrainingProgressLogger(trainer)

    assert path.read_bytes() == original
    assert progress_logger.progress.num_epochs_completed == 0

    pl_module = SimpleNamespace(device=SimpleNamespace(type='cpu'))
    progress_logger.on_train_start(SimpleNamespace(), pl_module)

    restarted = load_json(str(path), cls=TrainerProgress)
    assert restarted.num_epochs_completed == 0
    assert restarted.start_time is not None
