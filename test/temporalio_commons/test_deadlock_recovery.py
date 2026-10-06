# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Regression test for Temporal Python SDK PR #806 (deadlock recovery).

Verifies that when workflow tasks deadlock (block >2s), the worker recovers
and can still process new workflow tasks — i.e., deadlocked threads don't
permanently consume executor slots.

Prior to SDK v1.11.0, deadlocked threads leaked their slots, eventually
exhausting the workflow task executor and silently freezing the worker.
"""
import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
import pytest_asyncio
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

from service_framework.deploy_env_config import DeployEnvConfig
from temporalio_commons import terminate_open_workflows
from temporalio_commons.temporal_client import TemporalClient


@pytest_asyncio.fixture(autouse=True, scope='function')
async def terminate_workflows_after_test(temporal_config: DeployEnvConfig):
    yield
    await terminate_open_workflows.run(temporal_config, True)


@workflow.defn
class BlockingWorkflow:
    """Workflow that blocks the executor thread, triggering deadlock detection.

    Uses a busy-wait loop instead of time.sleep() because
    PyThreadState_SetAsyncExc (the SDK's thread interruption mechanism)
    can interrupt Python bytecode execution but NOT OS-level blocking
    calls like time.sleep() or threading.Event.wait().
    """
    @workflow.run
    async def run(self) -> str:
        # Busy-wait for 30s — exceeds the 2s deadlock detection threshold.
        # With UnsandboxedWorkflowRunner, this runs directly on the workflow
        # task executor thread. The SDK's deadlock detector will fire after
        # ~2s and (in v1.11.0+) interrupt this loop via PyThreadState_SetAsyncExc.
        end = time.monotonic() + 30
        while time.monotonic() < end:
            pass
        return 'blocked'


@workflow.defn
class SimpleWorkflow:
    """Workflow that returns immediately — used to verify recovery."""
    @workflow.run
    async def run(self) -> str:
        return 'ok'


@pytest.mark.asyncio
async def test_deadlock_recovery(temporal_config: DeployEnvConfig):
    """
    Deliberately exhaust a small workflow task executor with deadlocking
    workflows, then verify the worker recovers and can process new work.

    With the SDK fix (PR #806): deadlocked threads are interrupted via
    PyThreadState_SetAsyncExc, freeing executor slots for new tasks.
    Without the fix: threads are permanently consumed, SimpleWorkflow
    never gets a thread and times out.
    """
    deploy_config = temporal_config
    task_queue_raw = f'test-deadlock-recovery-{uuid.uuid4().hex[:8]}'
    # Apply the same task queue prefix the client interceptor adds,
    # so the Worker polls the same queue workflows are dispatched to.
    task_queue = deploy_config.task_queue(task_queue_raw)

    client = await TemporalClient(deploy_config).get()

    # Small thread pool to make exhaustion fast (3 threads).
    wf_task_executor = ThreadPoolExecutor(
        max_workers=3,
        thread_name_prefix="test_deadlock_wf_",
    )

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[BlockingWorkflow, SimpleWorkflow],
        activities=[],
        workflow_task_executor=wf_task_executor,
        workflow_runner=UnsandboxedWorkflowRunner(),
    )

    async with worker:
        # Start more blocking workflows than thread pool size to exhaust it.
        blocking_handles = []
        for _ in range(4):
            handle = await client.start_workflow(
                BlockingWorkflow.run,
                id=f'deadlock-test-blocking-{uuid.uuid4().hex[:8]}',
                task_queue=task_queue_raw,
                execution_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
            blocking_handles.append(handle)

        # Wait for deadlock detection + thread interruption to complete.
        # Deadlock fires at ~2s, interruption happens shortly after.
        await asyncio.sleep(6)

        # If the fix works, the worker should have recovered executor slots
        # and can process this simple workflow within a reasonable timeout.
        result = await client.execute_workflow(
            SimpleWorkflow.run,
            id=f'deadlock-test-simple-{uuid.uuid4().hex[:8]}',
            task_queue=task_queue_raw,
            execution_timeout=timedelta(seconds=15),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        assert result == 'ok'

    wf_task_executor.shutdown(wait=False, cancel_futures=True)
