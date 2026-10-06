# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: ProvisionTerminalError is non-retryable end-to-end.

Covers the full stack from AutoRPC server through wf_coordinator:

  fake PoolController (@rpc_handler raises ProvisionTerminalError)
    → AutoRPC is_retryable() == False  (not RuntimeError → not retried at RPC level)
    → wrapped in _AutoRpcNonRetriableError
    → RpcError(non_retryable=True) at place_snapshot_activity call site
    → ApplicationError(non_retryable=True)
    → _PLACE_RETRY_POLICY (maximum_attempts=8) aborts after 1 attempt

Two workers run simultaneously:
- AutoRPC server worker: fake PoolController task queue
- Temporal workflow worker: _DummySnapshotWorkflow + place_snapshot_activity

The test patches get_controller_client so place_snapshot_activity
talks to the fake PoolController.  Event history of the outer workflow
must show exactly 1 ACTIVITY_TASK_SCHEDULED for place_snapshot_activity.
"""
import asyncio
import uuid
from datetime import timedelta
from unittest import mock

import pytest  # pants: no-infer-dep
from temporalio import workflow
from temporalio.api.enums.v1 import EventType
from temporalio.client import WorkflowFailureError
from temporalio.common import RetryPolicy

from pool.activities import place_snapshot_activity, release_snapshot_activity
from pool.dataclasses import (
    PlacementRequest,
    PlacementResult,
    SnapshotActivity,
)
from pool.wf_coordinator import (
    _PLACE_RETRY_POLICY,
    snapshot_placement_session_ctx,
)
from pool.worker_manager.types import ProvisionTerminalError
from service_framework.deploy_env_config import DeployEnvConfig
from temporalio_commons import WorkerOptions
from temporalio_commons.auto_rpc import (
    AutoRpcServerWorkerConfig,
    auto_rpc_service,
    create_client,
    create_server_from_config,
    rpc_handler,
)
from temporalio_commons.auto_rpc_config import (
    AutoRpcClientConfig,
    AutoRpcTimeoutConfig,
)
from temporalio_commons.worker_registry import WorkerEnv

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def temporal_config() -> DeployEnvConfig:
    """Temporal config for this test.

    Reads Temporal Cloud connection details from the environment
    (e.g. DEPLOY_STAGE=test + AWS SM creds on devapp, or TEMPORAL_SERVER=cloud).
    No local Temporal server is started — uses the same cloud connection
    that other pool integration tests use on devapp.
    """
    return DeployEnvConfig.for_test()

# ---------------------------------------------------------------------------
# Fake PoolController AutoRPC service
# ---------------------------------------------------------------------------


@auto_rpc_service
class _FakePoolController:
    """Minimal PoolController stub for testing non-retryability.

    Raises a configurable exception from place_snapshot() and counts calls.
    """

    def __init__(self, exc: Exception | None = None) -> None:
        self.call_count = 0
        self._exc = exc or ProvisionTerminalError('injected by test')

    @rpc_handler
    async def place_snapshot(self, request: PlacementRequest) -> PlacementResult:
        self.call_count += 1
        raise self._exc

    @rpc_handler
    async def release_lease(self, snapshot_id: str, job_id: str) -> None:
        pass


# ---------------------------------------------------------------------------
# Dummy workflow using the placement context
# ---------------------------------------------------------------------------


@workflow.defn
class _DummySnapshotWorkflow:
    """Minimal workflow that acquires a pool lease."""

    @workflow.run
    async def run(self, snapshot_id: str) -> str:
        async with snapshot_placement_session_ctx(
            snapshot_id,
            SnapshotActivity.ENSURE_SNAPSHOT,
        ) as placement:
            return placement.task_queue


# ---------------------------------------------------------------------------
# Helper: count ACTIVITY_TASK_SCHEDULED events for a specific activity
# ---------------------------------------------------------------------------


async def _activity_schedule_count(wf_handle, activity_name: str) -> int:
    count = 0
    async for event in wf_handle.fetch_history_events():
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            if activity_name in event.activity_task_scheduled_event_attributes.activity_type.name:
                count += 1
    return count


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provision_terminal_error_no_retry_through_placement_context(
        temporal_config: DeployEnvConfig, monkeypatch: pytest.MonkeyPatch):
    """ProvisionTerminalError must not trigger retries at any layer.

    Verifies:
    1. fake_ctrl.call_count == 1: AutoRPC does NOT retry (is_retryable() == False)
    2. ACTIVITY_TASK_SCHEDULED count == 1: Temporal _PLACE_RETRY_POLICY does NOT
       retry place_snapshot_activity (non_retryable flag honored)

    Without the inheritance fix (ProvisionTerminalError inheriting RuntimeError),
    is_retryable() returns True, the error bypasses _AutoRpcNonRetriableError,
    arrives as non_retryable=False, and _PLACE_RETRY_POLICY fires 8× times.
    """
    # Shorten pool timeouts so the test doesn't wait minutes
    import pool.timeouts as _tmod
    monkeypatch.setattr(_tmod, 'PLACE_START_TO_CLOSE', timedelta(seconds=15))
    monkeypatch.setattr(_tmod, 'PLACE_SCHEDULE_TO_CLOSE', timedelta(seconds=30))

    fake_ctrl = _FakePoolController(
        exc=ProvisionTerminalError('CrashLoopBackOff: container exited immediately'))

    # AutoRPC client connecting to the fake controller on its task queue.
    # Single attempt at the AutoRPC level — the outer retry behavior is what
    # we're testing via _PLACE_RETRY_POLICY on the Temporal side.
    fake_client = create_client(
        _FakePoolController,
        AutoRpcClientConfig(
            temporal_config,
            timeout_config=AutoRpcTimeoutConfig(
                rpc_timeout=timedelta(seconds=10),
                retry_policy=RetryPolicy(maximum_attempts=1),
            ),
        ),
    )

    # Workflow worker: runs _DummySnapshotWorkflow + pool activities
    tq = f'test-provision-terminal-{uuid.uuid4().hex[:8]}'
    worker_env = WorkerEnv('test-worker', WorkerOptions(task_queue=tq), temporal_config)
    wf_worker = await worker_env.build_worker(
        activities=[place_snapshot_activity, release_snapshot_activity],
        workflows=[_DummySnapshotWorkflow],
    )

    # AutoRPC server worker: serves the fake PoolController
    server_cfg = AutoRpcServerWorkerConfig(temporal_config, max_executor_threads=2)
    autorpc_env, autorpc_worker = await create_server_from_config(fake_ctrl, server_cfg)
    autorpc_task = asyncio.create_task(autorpc_worker.run())

    try:
        async with wf_worker:
            client = await worker_env.client()

            # Redirect place_snapshot_activity → fake PoolController
            with mock.patch('pool.activities.get_controller_client',
                            return_value=fake_client), \
                 mock.patch('pool.activities.deploy_env'):

                wf_handle = await client.start_workflow(
                    _DummySnapshotWorkflow.run,
                    args=['NA-deadbeefdeadbeefdeadbeefdeadbeef@v1'],
                    id=f'test-provision-terminal-{uuid.uuid4().hex}',
                    task_queue=tq,
                )

                with pytest.raises(WorkflowFailureError):
                    await wf_handle.result()

        # --- Assertions ---

        # 1. AutoRPC level: fake_ctrl called exactly once (no RPC-level retries)
        assert fake_ctrl.call_count == 1, (
            f'place_snapshot was called {fake_ctrl.call_count}× at the AutoRPC level; '
            f'expected 1. ProvisionTerminalError is being retried by AutoRPC, '
            f'meaning is_retryable() returned True (RuntimeError in MRO).'
        )

        # 2. Temporal level: activity scheduled exactly once (_PLACE_RETRY_POLICY)
        schedules = await _activity_schedule_count(wf_handle, 'place_snapshot_activity')
        assert schedules == 1, (
            f'place_snapshot_activity was scheduled {schedules}× by Temporal; '
            f'expected 1. _PLACE_RETRY_POLICY has maximum_attempts='
            f'{_PLACE_RETRY_POLICY.maximum_attempts} but non_retryable=True '
            f'should stop all retries immediately.'
        )

    finally:
        await autorpc_worker.shutdown()
        autorpc_env.shutdown()
        autorpc_task.cancel()
        worker_env.shutdown()
