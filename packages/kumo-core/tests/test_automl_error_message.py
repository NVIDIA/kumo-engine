# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumo_core.automl.auto_trainer import (
    _failed_trials_error_message,
    _raise_if_no_trial_succeeded,
)
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


def test_no_oom_message_when_a_trial_succeeded(tmp_path):
    oom = TrialStatus(trial=1, batch_size=64)
    status = TrainStatus(
        trials=[TrialStatus(trial=0, batch_size=64, succeed=True), oom])

    _raise_if_no_trial_succeeded(status, [oom], 2,
                                 TaskType.BINARY_CLASSIFICATION, str(tmp_path))

    assert status.error_msg is None


def test_oom_message_when_no_trial_succeeded(tmp_path):
    oom = TrialStatus(trial=0, batch_size=64)
    status = TrainStatus(trials=[oom])

    with pytest.raises(RuntimeError):
        _raise_if_no_trial_succeeded(status, [oom], 1,
                                     TaskType.BINARY_CLASSIFICATION,
                                     str(tmp_path))

    assert TrainStatus.load(str(tmp_path)).error_msg is not None
