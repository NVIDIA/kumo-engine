# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the workflow-side pool coordinator's node-death-resilient runner.

These exercise ``run_pooled_activity``'s place → dispatch → health-poll →
re-route loop and its ``_await_worker_recovery`` decision logic by stubbing the
module-level ``workflow`` facade (``execute_activity`` / ``sleep``) — the only
Temporal primitives the runner touches — so no Temporal runtime is needed.

This module imports ``pool.wf_coordinator`` (→ ``pool.activities`` → the kumo
stack), so it needs the full kumo-stack test environment, like
``test_provision_terminal_nonretry.py``.
"""
import asyncio
from datetime import timedelta
from typing import Any, List

import pytest

from pool import timeouts, wf_coordinator
from pool.activities import (
    get_pool_stats_activity,
    place_snapshot_activity,
    release_snapshot_activity,
)
from pool.dataclasses import (
    PlacementResult,
    PlacementStatus,
    PoolStats,
    PoolWorkerStats,
    SnapshotActivity,
    WorkerStatus,
)
from pool.errors import HomeStaleError
from pool.task_queues import activity_queue


def test_placement_timeout_layers_cover_cold_k8s_path():
    internal_budget = timedelta(
        seconds=(timeouts.POOL_WORKER_STARTUP_GRACE_SEC +
                 timeouts.POOL_WORKER_PROVISION_TIMEOUT_SEC +
                 timeouts.ACTIVITY_POD_PROVISION_TIMEOUT_SEC +
                 timeouts.ACTIVITY_POLLER_READY_TIMEOUT_SEC +
                 timeouts.ACTIVITY_JOB_DELETE_TIMEOUT_SEC))

    assert timeouts.PLACE_SNAPSHOT_RPC_TIMEOUT > internal_budget
    assert timeouts.PLACE_START_TO_CLOSE > (timeouts.PLACE_SNAPSHOT_RPC_TIMEOUT)
    assert timeouts.PLACE_SCHEDULE_TO_CLOSE > timeouts.PLACE_START_TO_CLOSE


class _FakeWorkflow:
    """Minimal stand-in for ``temporalio.workflow`` covering the surface the
    runner uses: ``execute_activity``, ``sleep``, ``info``, ``logger``,
    ``upsert_memo``. ``execute_activity`` dispatches to a per-activity handler
    queue so a test can script success / failure per call."""
    def __init__(self, workflow_id: str = 'wf-1') -> None:
        self._workflow_id = workflow_id
        # activity_fn -> list of callables(call_index) returning result or
        # raising. Popped left-to-right per call.
        self.handlers: dict[Any, List[Any]] = {}
        self.sleeps: List[timedelta] = []
        self.dispatched: List[Any] = []
        self.memos: dict = {}

        class _Info:
            workflow_id = self._workflow_id

        class _Logger:
            def info(self, *a, **k):
                pass

            def warning(self, *a, **k):
                pass

            def exception(self, *a, **k):
                pass

        self._info = _Info()
        self.logger = _Logger()

    def info(self):
        return self._info

    def upsert_memo(self, memo: dict) -> None:
        self.memos.update(memo)

    async def sleep(self, duration) -> None:
        self.sleeps.append(
            duration if isinstance(duration, timedelta) else timedelta(
                seconds=duration))

    async def execute_activity(self, activity_fn, *args, **kwargs):
        self.dispatched.append(activity_fn)
        queue = self.handlers.get(activity_fn)
        if not queue:
            raise AssertionError(
                f'unexpected execute_activity for {activity_fn}')
        handler = queue.pop(0)
        return handler()


def _result(task_queue='tq-1', worker='worker-a',
            status=PlacementStatus.WARM) -> PlacementResult:
    return PlacementResult(
        snapshot_id='snap-1',
        task_queue=task_queue,
        status=status,
        tier_ssd_gib=0,
        pool_worker_uid=worker,
    )


def _stats(worker: str, health: WorkerStatus) -> PoolStats:
    return PoolStats(durable_workers=[
        PoolWorkerStats(
            pool_worker_uid=worker,
            health=health,
            pvc_free_bytes=0,
            pvc_total_bytes=0,
        )
    ])


def _activity_error(cause: BaseException | None = None) -> Exception:
    # Stand-in for temporalio.exceptions.ActivityError; the runner catches
    # that exact type, so use it directly.
    from temporalio.exceptions import ActivityError

    error = ActivityError(message='boom', scheduled_event_id=1,
                          started_event_id=1, identity='x',
                          activity_type='predict', activity_id='a',
                          retry_state=None)
    error.__cause__ = cause
    return error


def _timeout_activity_error(timeout_type) -> Exception:
    from temporalio.exceptions import TimeoutError

    return _activity_error(
        TimeoutError('activity timed out', type=timeout_type,
                     last_heartbeat_details=[]))


def _permanent_snapshot_miss_error() -> Exception:
    from temporalio.exceptions import ApplicationError

    from pool.activities import PERMANENT_SNAPSHOT_MISS_ERROR_TYPE

    return _activity_error(
        ApplicationError(
            'snapshot missing',
            type=PERMANENT_SNAPSHOT_MISS_ERROR_TYPE,
            non_retryable=True,
        ))


def _cancelled_activity_error() -> Exception:
    # The user cancels the job → Temporal delivers a CancelledError to the
    # activity, which surfaces at the coordinator as an ActivityError whose
    # ``.cause`` is that CancelledError. ``is_cancelled_exception`` keys off
    # exactly this shape (see temporalio_commons.error_utils).
    from temporalio.exceptions import ActivityError, CancelledError

    try:
        raise CancelledError('activity cancelled')
    except CancelledError as cancelled_err:
        try:
            raise ActivityError(
                message='activity cancelled',
                scheduled_event_id=1,
                started_event_id=1,
                identity='x',
                activity_type='train',
                activity_id='a',
                retry_state=None,
            ) from cancelled_err
        except ActivityError as activity_err:
            return activity_err


@pytest.fixture
def fake_workflow(monkeypatch):
    fw = _FakeWorkflow()
    monkeypatch.setattr(wf_coordinator, 'workflow', fw)
    return fw


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# _await_worker_recovery
# ---------------------------------------------------------------------------


def test_await_worker_recovery_ready_returns_retry(fake_workflow):
    fake_workflow.handlers[get_pool_stats_activity] = [
        lambda: _stats('worker-a', WorkerStatus.READY)
    ]
    decision = _run(wf_coordinator._await_worker_recovery('worker-a'))
    assert decision == 'retry'
    assert fake_workflow.sleeps == []


def test_await_worker_recovery_dead_returns_replace(fake_workflow):
    fake_workflow.handlers[get_pool_stats_activity] = [
        lambda: _stats('worker-a', WorkerStatus.DEAD)
    ]
    decision = _run(wf_coordinator._await_worker_recovery('worker-a'))
    assert decision == 'replace'


def test_await_worker_recovery_gone_returns_replace(fake_workflow):
    # Worker absent from stats → controller no longer knows it → replace.
    fake_workflow.handlers[get_pool_stats_activity] = [
        lambda: PoolStats(durable_workers=[])
    ]
    decision = _run(wf_coordinator._await_worker_recovery('worker-a'))
    assert decision == 'replace'


def test_await_worker_recovery_stale_then_ready_waits_then_retries(
        fake_workflow):
    fake_workflow.handlers[get_pool_stats_activity] = [
        lambda: _stats('worker-a', WorkerStatus.STALE),
        lambda: _stats('worker-a', WorkerStatus.READY),
    ]
    decision = _run(wf_coordinator._await_worker_recovery('worker-a'))
    assert decision == 'retry'
    # Slept once between the STALE poll and the recovering READY poll.
    assert fake_workflow.sleeps == [wf_coordinator._STALE_REPOLL_INTERVAL]


def test_await_worker_recovery_stale_then_dead_replaces(fake_workflow):
    fake_workflow.handlers[get_pool_stats_activity] = [
        lambda: _stats('worker-a', WorkerStatus.STALE),
        lambda: _stats('worker-a', WorkerStatus.DEAD),
    ]
    decision = _run(wf_coordinator._await_worker_recovery('worker-a'))
    assert decision == 'replace'
    assert fake_workflow.sleeps == [wf_coordinator._STALE_REPOLL_INTERVAL]


def test_await_worker_recovery_persistent_stale_bounded_then_replaces(
        fake_workflow):
    # Always STALE → bounded by _STALE_WAIT_CEILING, then replace.
    fake_workflow.handlers[get_pool_stats_activity] = [
        (lambda: _stats('worker-a', WorkerStatus.STALE)) for _ in range(100)
    ]
    decision = _run(wf_coordinator._await_worker_recovery('worker-a'))
    assert decision == 'replace'
    expected_sleeps = (wf_coordinator._STALE_WAIT_CEILING //
                       wf_coordinator._STALE_REPOLL_INTERVAL)
    assert len(fake_workflow.sleeps) == expected_sleeps


# ---------------------------------------------------------------------------
# run_pooled_activity loop
# ---------------------------------------------------------------------------


def _run_pooled(fake_workflow):
    return _run(
        wf_coordinator.run_pooled_activity(
            snapshot_id='snap-1',
            activity=SnapshotActivity.PREDICT,
            activity_fn='predict_fn',
            args=['a'],
            schedule_to_close_timeout=timedelta(hours=1),
            heartbeat_timeout=timedelta(minutes=5),
            max_attempts=3,
        ))


def test_run_pooled_activity_success_first_try(fake_workflow):
    fake_workflow.handlers[place_snapshot_activity] = [_result]
    fake_workflow.handlers['predict_fn'] = [lambda: 'ok']
    fake_workflow.handlers[release_snapshot_activity] = [lambda: None]

    assert _run_pooled(fake_workflow) == 'ok'
    # Placed once, released once.
    assert fake_workflow.dispatched.count(place_snapshot_activity) == 1
    assert fake_workflow.dispatched.count(release_snapshot_activity) == 1


def test_run_pooled_activity_stale_then_ready_retries_same_placement(
        fake_workflow):
    from temporalio.exceptions import TimeoutType

    fake_workflow.handlers[place_snapshot_activity] = [_result]
    err = _timeout_activity_error(TimeoutType.START_TO_CLOSE)
    fake_workflow.handlers['predict_fn'] = [
        (lambda: (_ for _ in ()).throw(err)),  # 1st dispatch fails
        lambda: 'ok',  # 2nd dispatch succeeds
    ]
    fake_workflow.handlers[get_pool_stats_activity] = [
        lambda: _stats('worker-a', WorkerStatus.STALE),
        lambda: _stats('worker-a', WorkerStatus.READY),
    ]
    fake_workflow.handlers[release_snapshot_activity] = [lambda: None]

    assert _run_pooled(fake_workflow) == 'ok'
    # Same placement reused (placed once), released once on exit.
    assert fake_workflow.dispatched.count(place_snapshot_activity) == 1
    assert fake_workflow.dispatched.count(release_snapshot_activity) == 1


def test_run_pooled_activity_dead_worker_re_places(fake_workflow):
    fake_workflow.handlers[place_snapshot_activity] = [
        lambda: _result(worker='worker-a'),
        lambda: _result(worker='worker-b'),
    ]
    err = _activity_error()
    fake_workflow.handlers['predict_fn'] = [
        (lambda: (_ for _ in ()).throw(err)),  # fails on worker-a
        lambda: 'ok',  # succeeds on worker-b
    ]
    fake_workflow.handlers[get_pool_stats_activity] = [
        lambda: _stats('worker-a', WorkerStatus.DEAD)
    ]
    fake_workflow.handlers[release_snapshot_activity] = [
        lambda: None, lambda: None
    ]

    assert _run_pooled(fake_workflow) == 'ok'
    # Re-placed (two place calls); the dropped placement + final placement each
    # release once.
    assert fake_workflow.dispatched.count(place_snapshot_activity) == 2
    assert fake_workflow.dispatched.count(release_snapshot_activity) == 2


def test_run_pooled_activity_exhausted_activity_pod_re_places_without_health(
        fake_workflow):
    fake_workflow.handlers[place_snapshot_activity] = [
        lambda: _result(task_queue=activity_queue('wf-1'), worker='worker-a'),
        lambda: _result(task_queue=activity_queue('wf-1'), worker='worker-a'),
    ]
    err = _activity_error()
    fake_workflow.handlers['predict_fn'] = [
        lambda: (_ for _ in ()).throw(err),
        lambda: 'ok',
    ]
    fake_workflow.handlers[release_snapshot_activity] = [
        lambda: None, lambda: None
    ]

    assert _run_pooled(fake_workflow) == 'ok'
    assert fake_workflow.dispatched.count(place_snapshot_activity) == 2
    assert fake_workflow.dispatched.count(release_snapshot_activity) == 2
    assert get_pool_stats_activity not in fake_workflow.dispatched


def test_run_pooled_activity_schedule_to_start_timeout_re_places_without_health(
        fake_workflow):
    from temporalio.exceptions import TimeoutType

    fake_workflow.handlers[place_snapshot_activity] = [
        lambda: _result(worker='worker-a'),
        lambda: _result(worker='worker-b'),
    ]
    timeout_error = _timeout_activity_error(TimeoutType.SCHEDULE_TO_START)
    fake_workflow.handlers['predict_fn'] = [
        lambda: (_ for _ in ()).throw(timeout_error),
        lambda: 'ok',
    ]
    fake_workflow.handlers[release_snapshot_activity] = [
        lambda: None, lambda: None
    ]

    assert _run_pooled(fake_workflow) == 'ok'
    assert fake_workflow.dispatched.count(place_snapshot_activity) == 2
    assert fake_workflow.dispatched.count(release_snapshot_activity) == 2
    assert get_pool_stats_activity not in fake_workflow.dispatched


def test_failed_release_prevents_poller_replacement(fake_workflow):
    from temporalio.exceptions import ApplicationError, TimeoutType

    fake_workflow.handlers[place_snapshot_activity] = [
        lambda: _result(worker='worker-a')
    ]
    timeout_error = _timeout_activity_error(TimeoutType.SCHEDULE_TO_START)
    fake_workflow.handlers['predict_fn'] = [
        lambda: (_ for _ in ()).throw(timeout_error)
    ]
    fake_workflow.handlers[release_snapshot_activity] = [
        lambda: (_ for _ in ()).throw(RuntimeError('cleanup failed'))
        for _ in range(2)
    ]

    with pytest.raises(ApplicationError) as exc_info:
        _run_pooled(fake_workflow)

    assert exc_info.value.type == 'PooledActivityNoPollerError'
    assert fake_workflow.dispatched.count(place_snapshot_activity) == 1
    assert fake_workflow.dispatched.count(release_snapshot_activity) == 2


def test_run_pooled_activity_snapshot_miss_re_places_without_health_poll(
        fake_workflow):
    fake_workflow.handlers[place_snapshot_activity] = [
        lambda: _result(worker='worker-a'),
        lambda: _result(worker='worker-a'),
    ]
    err = _permanent_snapshot_miss_error()
    fake_workflow.handlers['predict_fn'] = [
        (lambda: (_ for _ in ()).throw(err)),
        lambda: 'ok',
    ]
    fake_workflow.handlers[release_snapshot_activity] = [
        lambda: None, lambda: None
    ]

    assert _run_pooled(fake_workflow) == 'ok'
    assert fake_workflow.dispatched.count(place_snapshot_activity) == 2
    assert fake_workflow.dispatched.count(release_snapshot_activity) == 2
    assert get_pool_stats_activity not in fake_workflow.dispatched


def test_run_pooled_activity_exhausts_attempts_and_raises(fake_workflow):
    err = _activity_error()
    fake_workflow.handlers[place_snapshot_activity] = [
        lambda: _result(worker='worker-a') for _ in range(3)
    ]
    fake_workflow.handlers['predict_fn'] = [(lambda: (_ for _ in ()).throw(err))
                                            for _ in range(3)]
    # Worker stays READY → each failure re-dispatches on the same placement,
    # consuming an attempt; after max_attempts the last error is raised.
    fake_workflow.handlers[get_pool_stats_activity] = [
        (lambda: _stats('worker-a', WorkerStatus.READY)) for _ in range(3)
    ]
    fake_workflow.handlers[release_snapshot_activity] = [
        lambda: None for _ in range(3)
    ]

    with pytest.raises(Exception):
        _run_pooled(fake_workflow)
    # READY re-dispatch keeps the same placement → placed once, released once.
    assert fake_workflow.dispatched.count(place_snapshot_activity) == 1


def test_exhausted_poller_replacements_raise_nonretryable_error(fake_workflow):
    from temporalio.exceptions import ApplicationError, TimeoutType

    fake_workflow.handlers[place_snapshot_activity] = [
        lambda index=index: _result(worker=f'worker-{index}')
        for index in range(3)
    ]

    def raise_schedule_to_start_timeout():
        raise _timeout_activity_error(TimeoutType.SCHEDULE_TO_START)

    fake_workflow.handlers['predict_fn'] = [
        raise_schedule_to_start_timeout for _ in range(3)
    ]
    fake_workflow.handlers[release_snapshot_activity] = [
        lambda: None for _ in range(3)
    ]

    with pytest.raises(ApplicationError) as exc_info:
        _run_pooled(fake_workflow)

    assert exc_info.value.type == 'PooledActivityNoPollerError'
    assert exc_info.value.non_retryable is True
    assert 'after 3 placements' in str(exc_info.value)
    assert fake_workflow.dispatched.count(place_snapshot_activity) == 3
    assert fake_workflow.dispatched.count(release_snapshot_activity) == 3
    assert get_pool_stats_activity not in fake_workflow.dispatched


def test_run_pooled_activity_overflow_failure_re_places(fake_workflow):
    # Overflow placement (pool_worker_uid=None) → no health poll; drop +
    # re-place on failure.
    fake_workflow.handlers[place_snapshot_activity] = [
        lambda: _result(worker=None, status=PlacementStatus.COLD),
        lambda: _result(worker='worker-b'),
    ]
    err = _activity_error()
    fake_workflow.handlers['predict_fn'] = [
        (lambda: (_ for _ in ()).throw(err)),
        lambda: 'ok',
    ]
    fake_workflow.handlers[release_snapshot_activity] = [
        lambda: None, lambda: None
    ]

    assert _run_pooled(fake_workflow) == 'ok'
    assert fake_workflow.dispatched.count(place_snapshot_activity) == 2
    # No health poll for the overflow placement.
    assert get_pool_stats_activity not in fake_workflow.dispatched


def test_run_pooled_activity_cancellation_re_raises_without_re_dispatch(
        fake_workflow):
    # Regression for kumo#30191: a cancellation surfaces as an ActivityError
    # caused by CancelledError. It must NOT be treated as a recoverable worker
    # failure — the coordinator must re-raise it after exactly one train
    # dispatch, with zero worker-health polls and exactly one placement
    # release, so no second train run starts on the still-healthy worker.
    from temporalio.exceptions import ActivityError

    fake_workflow.handlers[place_snapshot_activity] = [_result]
    err = _cancelled_activity_error()
    fake_workflow.handlers['predict_fn'] = [
        (lambda: (_ for _ in ()).throw(err)),
    ]
    fake_workflow.handlers[release_snapshot_activity] = [lambda: None]

    with pytest.raises(ActivityError):
        _run_pooled(fake_workflow)

    # Exactly one dispatch of the activity — no re-dispatch after cancellation.
    assert fake_workflow.dispatched.count('predict_fn') == 1
    # No worker-health poll: cancellation is short-circuited before the poll.
    assert get_pool_stats_activity not in fake_workflow.dispatched
    # Placement placed once and released once (via the finally path).
    assert fake_workflow.dispatched.count(place_snapshot_activity) == 1
    assert fake_workflow.dispatched.count(release_snapshot_activity) == 1


def test_home_stale_error_is_runtime_error():
    # Sanity: HomeStaleError is a RuntimeError so existing broad handlers and
    # AutoRPC's default (retryable) classification treat it as transient.
    assert issubclass(HomeStaleError, RuntimeError)


# ---------------------------------------------------------------------------
# _place_snapshot: actionable error on a budget-exhausted placement (#29865)
# ---------------------------------------------------------------------------


def _place_request():
    from pool.dataclasses import PlacementRequest, ResourceHint
    return PlacementRequest(
        snapshot_id='snap-1',
        job_id='job-1',
        workflow_id='wf-1',
        activity=SnapshotActivity.GEN_TT,
        resource_hint=ResourceHint(),
    )


def _schedule_to_close_timeout():
    from temporalio.exceptions import TimeoutError, TimeoutType
    return TimeoutError('activity ScheduleToClose exceeded',
                        type=TimeoutType.SCHEDULE_TO_CLOSE,
                        last_heartbeat_details=[])


def _activity_error_from(cause: BaseException):
    """A realistic ``ActivityError`` wrapping ``cause`` - the shape
    ``workflow.execute_activity`` actually raises (it never surfaces a bare
    timeout / ApplicationError). Chained via ``from`` so ``cause`` is reachable
    on ``__cause__``, which ``_classify_placement_failure`` walks alongside
    ``.cause``; this exercises the dig-through-ActivityError the fix relies on."""
    from temporalio.exceptions import ActivityError
    try:
        raise ActivityError(message='place_snapshot_activity failed',
                            scheduled_event_id=1, started_event_id=1,
                            identity='x',
                            activity_type='place_snapshot_activity',
                            activity_id='a', retry_state=None) from cause
    except ActivityError as e:
        return e


def test_place_snapshot_timeout_without_mapped_cause_raises_pool_error(
        fake_workflow):
    """A placement that exhausts its budget with no mapped placement
    PublicException (pool controller unavailable / crash-looping, #29865)
    surfaces an actionable pool-availability error, not the generic
    "data too large" timeout fallback. Shape matches production: an
    ``ActivityError`` wrapping the ScheduleToClose ``TimeoutError``."""
    from temporalio.exceptions import ApplicationError

    from common_utils.error import PublicException

    err = _activity_error_from(_schedule_to_close_timeout())
    fake_workflow.handlers[place_snapshot_activity] = [
        lambda: (_ for _ in ()).throw(err),
    ]
    with pytest.raises(ApplicationError) as excinfo:
        _run(wf_coordinator._place_snapshot(_place_request()))

    app_err = excinfo.value
    assert app_err.type == 'PoolPlacementUnavailableError'
    assert app_err.non_retryable is True
    public_ex = PublicException.from_dict(app_err.details[0])
    msg = public_ex.message.lower()
    assert 'pool' in msg
    assert 'too large' not in msg  # not the misleading data-size message


def test_place_snapshot_mapped_placement_error_is_preserved(fake_workflow):
    """When the controller DID run and mapped a specific placement error, its
    PublicException must be preserved, not overwritten by the generic pool
    message. Anchored on ProvisionTerminalError: non-retryable, so it surfaces
    as ActivityError(cause=ApplicationError) with NO timeout and is preserved
    deterministically. (A ProvisionTimeoutError that only surfaces via the
    ScheduleToClose timeout is preserved best-effort - Temporal may not carry
    the mapped cause onto the timeout - but the generic message is a strict
    superset there, so the user still sees an actionable error either way.)"""
    from temporalio.exceptions import ApplicationError

    from common_utils.error import PublicException

    specific = ApplicationError(
        'compute pod failed to initialize',
        PublicException(message='bad image tag / missing secret').to_dict(),
        type='ProvisionTerminalError',
        non_retryable=True,
    )
    err = _activity_error_from(specific)
    fake_workflow.handlers[place_snapshot_activity] = [
        lambda: (_ for _ in ()).throw(err),
    ]
    with pytest.raises(BaseException) as excinfo:
        _run(wf_coordinator._place_snapshot(_place_request()))

    # The original error is re-raised unchanged - NOT replaced by ours.
    assert excinfo.value is err
    assert getattr(excinfo.value, 'type',
                   None) != 'PoolPlacementUnavailableError'
    # The specific mapped PublicException is still reachable on the chain.
    assert PublicException.from_dict(
        err.__cause__.details[0]).message == 'bad image tag / missing secret'


def test_place_snapshot_non_timeout_failure_is_not_translated(fake_workflow):
    """A non-timeout placement failure (e.g. an RBAC / spec error) is left
    exactly as-is: only budget-exhausted timeouts get the pool message."""
    err = _activity_error()  # ActivityError, no timeout, no PublicException
    fake_workflow.handlers[place_snapshot_activity] = [
        lambda: (_ for _ in ()).throw(err),
    ]
    with pytest.raises(BaseException) as excinfo:
        _run(wf_coordinator._place_snapshot(_place_request()))

    assert excinfo.value is err
    assert getattr(excinfo.value, 'type',
                   None) != 'PoolPlacementUnavailableError'
