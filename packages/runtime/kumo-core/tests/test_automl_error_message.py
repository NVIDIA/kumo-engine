# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.automl.auto_trainer import _failed_trials_error_message
from kumo_core.job_config import TrainStatus, TrialStatus


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
