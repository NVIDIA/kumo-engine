# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from kumo_core.automl import auto_trainer
from kumo_core.automl.auto_trainer import (
    AutoTrainer,
    _failed_trials_error_message,
)
from kumo_core.instrumentation import NoOpTrainingInstrumentation
from kumo_core.job_config import TrainStatus, TrialStatus
from kumo_core.types import TaskType


def test_failed_trials_error_includes_last_recorded_cause():
    status = TrainStatus(trials=[
        TrialStatus(
            trial=0,
            batch_size=64,
            stacktrace="Traceback:\nRuntimeError: missing validation metric",
        ),
        TrialStatus(
            trial=1,
            batch_size=32,
            stacktrace="Traceback:\nRuntimeError: CUDA out of memory",
        ),
    ])

    assert _failed_trials_error_message(status) == (
        "Failed to run any AutoML trial. "
        "Last trial error: RuntimeError: CUDA out of memory")


def test_failed_trials_error_handles_missing_tracebacks():
    status = TrainStatus(trials=[TrialStatus(trial=0, batch_size=64)])

    assert _failed_trials_error_message(
        status) == "Failed to run any AutoML trial"


def _run_trials(tmp_path, monkeypatch, outcomes):
    """Runs one trial per outcome: an exception or a validation result."""
    monkeypatch.setattr(auto_trainer, "check_gpu_ready", lambda: None)
    monkeypatch.setattr(
        auto_trainer, "TPESearcher", lambda *args, **kwargs: SimpleNamespace(
            sampled_params={}, update=lambda *args, **kwargs: None,
            suggest=lambda: SimpleNamespace(
                training_job=SimpleNamespace(),
                optimization=SimpleNamespace(batch_size=[64]))))
    monkeypatch.setattr(auto_trainer, "AutoTrainingProgressLogger",
                        lambda *args, **kwargs: None)
    outcomes = iter(outcomes)

    def train(**kwargs):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    trainer = SimpleNamespace(
        home_dir=str(tmp_path), log_dir=str(tmp_path), space=None,
        num_experiments=2, max_tries=1, random_state_seed=0, encoders=None,
        tune_metric="loss", val_tune_metric="val_loss",
        task=SimpleNamespace(task_type=TaskType.BINARY_CLASSIFICATION),
        instrumentation=NoOpTrainingInstrumentation(),
        runtime=SimpleNamespace(metrics_job_name=lambda: "job"),
        _emit=lambda *args, **kwargs: None,
        _make_pruning_callback=lambda: None,
        _handle_oom_retry=lambda *args, **kwargs: None, _train=train,
        get_best_trainer=lambda should_init: "best trainer")
    return AutoTrainer._run(trainer)


def test_partial_oom_run_saves_no_error_message(tmp_path, monkeypatch,
                                                caplog):
    outcomes = [RuntimeError("CUDA out of memory"), [0.5]]
    assert _run_trials(tmp_path, monkeypatch, outcomes) == "best trainer"

    status = TrainStatus.load(str(tmp_path))
    assert [trial.succeed for trial in status.trials] == [False, True]
    assert status.error_msg is None
    assert "Failed 1 AutoML trials" in caplog.text


def test_all_oom_run_saves_oom_message(tmp_path, monkeypatch):
    outcomes = [RuntimeError("CUDA out of memory") for _ in range(2)]
    with pytest.raises(RuntimeError, match="Failed to run any AutoML trial"):
        _run_trials(tmp_path, monkeypatch, outcomes)

    assert TrainStatus.load(str(tmp_path)).error_msg.startswith(
        "Out of 2 experiment(s), 2 experiment(s) failed due to a GPU "
        "out-of-memory issue.")
