# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for ML Job progress and terminal publication."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from kumo_core import runtime_progress
from kumo_core.callbacks import PredictionProgressLogger
from kumo_snowflake import _ml_job_runtime as runtime
from kumoapi.jobs import AutoTrainerProgress, TrainerProgress


class MemoryStore:
    def __init__(self) -> None:
        self.objects: dict[str, Any] = {}
        self.failures = 0
        self.lose_ack = False
        self.result_attempts = 0

    def put_json(self, location: str, payload: Any) -> None:
        if location == 'result.json':
            self.result_attempts += 1
            if self.failures:
                self.failures -= 1
                raise ConnectionError('transient result upload failure')
        self.objects[location] = copy.deepcopy(payload)
        if location == 'result.json' and self.lose_ack:
            self.lose_ack = False
            raise ConnectionError('lost terminal upload acknowledgement')

    def get_json(self, location: str, *, missing_ok: bool = False) -> Any:
        del missing_ok
        return self.objects.get(location)


@pytest.fixture
def store(monkeypatch: Any) -> MemoryStore:
    result = MemoryStore()
    monkeypatch.setattr(runtime, '_active_job_session', lambda: object())
    monkeypatch.setattr(runtime, 'StageStore', lambda *_: result)
    monkeypatch.setattr(runtime, '_RESULT_PUBLISH_RETRY_S', 0)
    return result


def execute(function: Any) -> dict[str, Any]:
    return runtime.execute_user_function(function, [], {}, {
        'schema_version': 1,
        'execution_id': 'review-regression',
        'stage_name': 'DB.PUBLIC.STAGE',
        'progress_location': 'progress.json',
        'result_location': 'result.json',
    })


@pytest.mark.parametrize('explicit_boundaries', [False, True])
def test_multiple_native_prediction_loggers_succeed(
    store: MemoryStore,
    explicit_boundaries: bool,
) -> None:
    def predict_partition(partition: int) -> None:
        callback = PredictionProgressLogger(None, partition, 2)
        callback.on_predict_start(None, None)
        for batch in range(2):
            callback.on_predict_batch_end(None, None, {}, None, batch)
        callback.on_predict_end(None, None)

    def predict_all() -> str:
        for partition in range(3):
            if explicit_boundaries:
                with runtime_progress.operation('prediction', str(partition),
                                                'Predicting'):
                    predict_partition(partition)
            else:
                predict_partition(partition)
        return 'three partitions predicted'

    result = execute(predict_all)
    assert result['status'] == 'SUCCEEDED'
    assert result['value'] == 'three partitions predicted'
    assert store.objects['result.json'] == result
    assert result['progress']['completed_iterations'] == 2


def test_late_trial_progress_does_not_reopen_completed_training(
    store: MemoryStore, ) -> None:
    def train() -> str:
        runtime_progress.progress_updated(
            'training', AutoTrainerProgress(total_trials=1, completed_trials=1))
        runtime_progress.progress_updated('training', TrainerProgress())
        return 'trained'

    result = execute(train)
    assert result['status'] == 'SUCCEEDED'
    assert result['progress']['completed_trials'] == 1
    assert [
        entry['stage_name'] for entry in result['status_report']['event_log']
    ].count('Training') == 1
    assert store.objects['result.json'] == result


@pytest.mark.parametrize('lose_ack', [False, True])
def test_failure_publication_retries_and_reconciles(
    store: MemoryStore,
    lose_ack: bool,
) -> None:
    store.failures = 0 if lose_ack else 2
    store.lose_ack = lose_ack
    original = ValueError('original workload failure')

    def fail() -> None:
        raise original

    with pytest.raises(ValueError) as caught:
        execute(fail)
    assert caught.value is original
    assert store.result_attempts == (1 if lose_ack else 3)
    assert store.objects['result.json']['status'] == 'FAILED'
    assert store.objects['result.json']['diagnostic']['message'] == str(
        original)


@pytest.mark.parametrize('signal', [SystemExit(143), KeyboardInterrupt()])
def test_cancellation_signals_write_cancelled_envelope(
    store: MemoryStore,
    signal: BaseException,
) -> None:
    def cancelled() -> None:
        raise signal

    with pytest.raises(type(signal)):
        execute(cancelled)
    result = store.objects['result.json']
    assert result['status'] == 'CANCELLED'
    assert result['status_report']['status'] == 'CANCELLED'
    assert store.objects['progress.json']['status_report'][
        'status'] == 'CANCELLED'


@pytest.mark.parametrize('fail_at', ['terminal', 'envelope'])
def test_success_finalization_errors_produce_durable_failure(
    store: MemoryStore,
    monkeypatch: Any,
    fail_at: str,
) -> None:
    if fail_at == 'terminal':
        original_terminal = runtime._RemoteObserver.terminal

        def broken_terminal(self: Any, succeeded: bool, *args: Any,
                            **kwargs: Any) -> Any:
            if succeeded:
                raise ValueError('terminal finalization failed')
            return original_terminal(self, succeeded, *args, **kwargs)

        monkeypatch.setattr(runtime._RemoteObserver, 'terminal',
                            broken_terminal)
    else:
        # Even permanently broken observation serialization must not prevent
        # recording the finalization failure in the durable envelope.
        def broken_fields(*args: Any) -> Any:
            raise ValueError('envelope finalization failed')

        monkeypatch.setattr(runtime, '_result_observation_fields',
                            broken_fields)

    with pytest.raises(ValueError, match='finalization failed'):
        execute(lambda: 'work completed')
    assert store.objects['result.json']['status'] == 'FAILED'
    assert 'finalization failed' in store.objects['result.json']['diagnostic'][
        'message']
