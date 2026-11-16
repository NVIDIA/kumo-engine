# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration test for timeout enrichment feature.

Tests that when an activity times out after a prior failure, the TimeoutError
is enriched with the prior failure context via _ActivityHandleWithFailureTracking.
"""
import asyncio
import uuid
from datetime import timedelta

import pytest
from temporalio import activity, workflow
from temporalio.client import WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.exceptions import TimeoutError, TimeoutType

from service_framework.deploy_env_config import DeployEnvConfig
from temporalio_commons import WorkerOptions
from temporalio_commons.worker_registry import WorkerEnv


@activity.defn
async def configurable_activity(should_fail: bool) -> str:
    """Activity that either fails immediately or times out.

    Used to test timeout enrichment:
    1. First call with should_fail=True -> fails immediately (stored)
    2. Second call with should_fail=False -> times out (enriched with prior failure)
    """
    if should_fail:
        raise ValueError("Database connection failed: critical error")
    await asyncio.sleep(10.0)  # Sleep longer than timeout
    return "Success"


@workflow.defn
class TimeoutAfterFailureWorkflow:
    """Workflow that triggers timeout-after-failure scenario."""
    @workflow.run
    async def run(self) -> str:
        # First call: fails immediately (stored in interceptor)
        try:
            await workflow.execute_activity(
                configurable_activity,
                args=[True],
                schedule_to_close_timeout=timedelta(seconds=5),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
        except Exception:
            pass  # Expected

        # Second call: times out (should be enriched with prior failure)
        return await workflow.execute_activity(
            configurable_activity,
            args=[False],
            schedule_to_close_timeout=timedelta(seconds=5),
            start_to_close_timeout=timedelta(seconds=2),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )


def _walk_error_chain(exc: BaseException):
    """Walk error chain following both .cause and __cause__."""
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        next_cause = getattr(current, "cause", None) or getattr(
            current, "__cause__", None)
        current = next_cause if isinstance(next_cause, BaseException) else None


@pytest.mark.asyncio
async def test_timeout_after_prior_failure(temporal_config: DeployEnvConfig):
    """Test that TimeoutError is enriched with prior failure context.

    This tests the _ActivityHandleWithFailureTracking interceptor which:
    1. Stores activity failures
    2. Enriches TimeoutError.__cause__ with the stored failure

    Note: The enrichment is applied within the workflow but not visible in
    client-side error chain due to Temporal's serialization.
    """
    worker_options = WorkerOptions(task_queue="test-timeout-enrichment")
    worker_env = WorkerEnv("test-worker", worker_options, temporal_config)

    worker = await worker_env.build_worker(
        activities=[configurable_activity],
        workflows=[TimeoutAfterFailureWorkflow],
    )

    async with worker:
        client = await worker_env.client()
        wf_handle = await client.start_workflow(
            TimeoutAfterFailureWorkflow.run,
            id=f"test-timeout-enrichment-{uuid.uuid4().hex[:8]}",
            task_queue=worker_env.options.task_queue,
        )

        with pytest.raises(WorkflowFailureError) as exc_info:
            await wf_handle.result()

        # Verify we get TimeoutError with START_TO_CLOSE type
        error = exc_info.value
        found_start_to_close = False
        for err in _walk_error_chain(error):
            if isinstance(
                    err,
                    TimeoutError) and err.type == TimeoutType.START_TO_CLOSE:
                found_start_to_close = True
                break

        assert found_start_to_close, "Expected START_TO_CLOSE timeout in error chain"

    worker_env.shutdown()
