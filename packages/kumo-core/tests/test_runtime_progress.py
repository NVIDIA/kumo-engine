# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from typing import Any

import pytest
from kumo_core import runtime_progress


class _Observer:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def operation_started(self, kind: str, operation_id: str,
                          stage: str) -> None:
        self.calls.append(('start', kind, operation_id, stage))

    def stage_changed(self, stage: str, detail: str | None = None) -> None:
        self.calls.append(('stage', stage, detail))

    def progress_updated(self, kind: str, progress: Any) -> None:
        self.calls.append(('progress', kind, progress))

    def operation_finished(self, kind: str, operation_id: str,
                           error: BaseException | None) -> None:
        self.calls.append(('finish', kind, operation_id, error))


def test_no_observer_is_a_no_op() -> None:
    with runtime_progress.operation('training', 'job-1', 'Training'):
        runtime_progress.stage_changed('Predicting')
        runtime_progress.progress_updated('prediction', {'complete': 1})


def test_operation_and_native_progress_are_delivered_serially() -> None:
    observer = _Observer()
    token = runtime_progress.install(observer)
    try:
        with runtime_progress.operation('training', 'job-1', 'Training'):
            runtime_progress.progress_updated('training', {'trials': 2})
            runtime_progress.stage_changed('Publishing model')
    finally:
        runtime_progress.reset(token)
    assert observer.calls == [
        ('start', 'training', 'job-1', 'Training'),
        ('progress', 'training', {
            'trials': 2
        }),
        ('stage', 'Publishing model', None),
        ('finish', 'training', 'job-1', None),
    ]


def test_operation_reports_the_original_failure() -> None:
    observer = _Observer()
    token = runtime_progress.install(observer)
    error = RuntimeError('training failed')

    def fail_operation() -> None:
        with runtime_progress.operation('training', 'job-1', 'Training'):
            raise error

    try:
        with pytest.raises(RuntimeError, match='training failed') as caught:
            fail_operation()
    finally:
        runtime_progress.reset(token)
    assert caught.value is error
    assert observer.calls[-1] == ('finish', 'training', 'job-1', error)


def test_native_progress_from_a_worker_thread_reaches_the_observer() -> None:
    observer = _Observer()
    token = runtime_progress.install(observer)
    try:
        worker = threading.Thread(
            target=runtime_progress.progress_updated,
            args=('prediction', {
                'completed_iterations': 4
            }),
        )
        worker.start()
        worker.join()
    finally:
        runtime_progress.reset(token)
    assert observer.calls == [('progress', 'prediction', {
        'completed_iterations': 4
    })]


def test_observer_progress_failure_does_not_abort_the_workload() -> None:
    class _FailingObserver(_Observer):
        def progress_updated(self, kind: str, progress: Any) -> None:
            del kind, progress
            raise ConnectionError('progress sink unavailable')

    observer = _FailingObserver()
    token = runtime_progress.install(observer)
    try:
        runtime_progress.progress_updated('prediction',
                                          {'completed_iterations': 4})
    finally:
        runtime_progress.reset(token)


def test_observer_completion_failure_does_not_replace_workload_result() -> None:
    class _FailingObserver(_Observer):
        def operation_finished(self, kind: str, operation_id: str,
                               error: BaseException | None) -> None:
            del kind, operation_id, error
            raise ConnectionError('progress sink unavailable')

    observer = _FailingObserver()
    token = runtime_progress.install(observer)
    try:
        with runtime_progress.operation('prediction', 'bp-job-1', 'Predicting'):
            result = 'predicted'
    finally:
        runtime_progress.reset(token)

    assert result == 'predicted'


def test_reset_waits_for_an_in_flight_worker_callback() -> None:
    callback_started = threading.Event()
    release_callback = threading.Event()
    reset_finished = threading.Event()

    class _BlockingObserver(_Observer):
        def progress_updated(self, kind: str, progress: Any) -> None:
            callback_started.set()
            release_callback.wait()
            super().progress_updated(kind, progress)

    observer = _BlockingObserver()
    token = runtime_progress.install(observer)
    worker = threading.Thread(
        target=runtime_progress.progress_updated,
        args=('prediction', {
            'completed_iterations': 4
        }),
    )
    worker.start()
    assert callback_started.wait(timeout=1)

    def reset() -> None:
        runtime_progress.reset(token)
        reset_finished.set()

    resetter = threading.Thread(target=reset)
    resetter.start()
    assert not reset_finished.wait(timeout=0.05)
    release_callback.set()
    worker.join()
    resetter.join()
    assert reset_finished.is_set()


def test_a_second_process_observer_is_rejected() -> None:
    observer = _Observer()
    token = runtime_progress.install(observer)
    try:
        with pytest.raises(RuntimeError, match='observer is active'):
            runtime_progress.install(_Observer())
    finally:
        runtime_progress.reset(token)
