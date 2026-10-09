# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from asyncio import CancelledError
from datetime import timedelta
from time import sleep
from typing import AsyncGenerator, List, Optional

import pytest
import pytest_asyncio
from kumo_core.utils.time import now_utc
from temporalio import workflow
from temporalio.client import Client, WorkflowHandle
from temporalio.workflow import ActivityCancellationType

from io_utils import kumo_dataclass
from service_framework.deploy_env_config import DeployEnvConfig
from temporalio_commons.auto_rpc import NO_RETRY_POLICY
from temporalio_commons.process_pool_utils import process_pool_activity_defn
from temporalio_commons.worker_registry import WorkerEnv, WorkerOptions

_TEST_TASK_QUEUE = 'test-hearbeat-worker-task-queue'


@kumo_dataclass
class TestInput:
    activity_duration: timedelta
    cancel_activity_after: Optional[timedelta]
    heartbeat_timeout: timedelta = timedelta(seconds=6.0)


@process_pool_activity_defn(heartbeat_interval=timedelta(seconds=1.0))
def sync_activity(input: TestInput) -> str:
    start_ts = now_utc()
    try:
        elapsed = now_utc() - start_ts
        while elapsed < input.activity_duration:
            sleep(1.0)
            elapsed = now_utc() - start_ts
    except CancelledError:
        elapsed = now_utc() - start_ts
        print(f'sync_activity is canceled after {elapsed} sec!')
        return "canceled"
    return "succeeded"


@workflow.defn
class TestHeartbeatWorkflow:
    async def cancel_activity(self, delay_sec: float) -> bool:
        await asyncio.sleep(delay_sec)
        print(f'Cancelling activity after {delay_sec} sec')
        return self.activity_task.cancel()

    @workflow.run
    async def run(self, input: TestInput):
        self.activity_task = workflow.start_activity(
            sync_activity,
            input,
            schedule_to_close_timeout=timedelta(hours=1),
            heartbeat_timeout=input.heartbeat_timeout,
            retry_policy=NO_RETRY_POLICY,
            cancellation_type=ActivityCancellationType.
            WAIT_CANCELLATION_COMPLETED,
        )

        if input.cancel_activity_after is None:
            return await self.activity_task

        activity_result, _ = await asyncio.gather(
            self.activity_task,
            self.cancel_activity(input.cancel_activity_after.total_seconds()))
        return activity_result


# Setup worker and test client.
@pytest_asyncio.fixture
async def client(
        temporal_config: DeployEnvConfig) -> AsyncGenerator[Client, None]:
    worker_options = WorkerOptions(task_queue=_TEST_TASK_QUEUE,
                                   max_worker_processes=1)
    worker_env = WorkerEnv('test-heartbeat-worker', worker_options,
                           temporal_config)

    worker = await worker_env.build_worker(activities=[sync_activity],
                                           workflows=[TestHeartbeatWorkflow])
    async with worker:
        client = await worker_env.client()
        # We execute a dummy workflow/activity to make sure child process gets
        # initialized in the process pool before running the tests. Otherwise
        # initialization of child process may take longer than activity
        # heartbeat timeout, leading to unexpected workflow/activity failure.
        await client.execute_workflow(
            TestHeartbeatWorkflow.run,
            TestInput(activity_duration=timedelta(seconds=0.0),
                      cancel_activity_after=None,
                      heartbeat_timeout=timedelta(seconds=30)),
            id='test_heartbeat_wf_init',
            task_queue=_TEST_TASK_QUEUE,
            execution_timeout=timedelta(hours=1),
        )

        yield client

    worker_env.shutdown()


@pytest.mark.asyncio
async def test_activity_cancellation(client: Client):
    # Test case 1: cancel activity immediately (before it starts execution
    # in process pool)
    wf_result = await client.execute_workflow(
        TestHeartbeatWorkflow.run,
        TestInput(activity_duration=timedelta(seconds=60),
                  cancel_activity_after=timedelta(seconds=0.1)),
        id='test_heartbeat_wf_id1',
        task_queue=_TEST_TASK_QUEUE,
        execution_timeout=timedelta(hours=1),
    )
    assert wf_result == 'canceled'

    # Test case 2: cancel activity while it's running in process pool.
    wf_result = await client.execute_workflow(
        TestHeartbeatWorkflow.run,
        TestInput(activity_duration=timedelta(seconds=60),
                  cancel_activity_after=timedelta(seconds=5.0)),
        id='test_heartbeat_wf_id2',
        task_queue=_TEST_TASK_QUEUE,
        execution_timeout=timedelta(hours=1),
    )
    assert wf_result == 'canceled'


# Test case that simulates production scenario in which multiple runs of
# train/bp workflows are started and possibly canceled.
# In this test, we start 3 workflows and then cancel them; then launch the 4th
# workflow and execpt success. This verifies that the child worker
# process is freed up shortly after the cancellation of previous 3 workflows,
# otherwise the 6th workflow would see activity heartbeat timeout.
@pytest.mark.asyncio
async def test_cancellation_multiple_workflows(client: Client):
    wf_handles: List[WorkflowHandle] = []
    for i in range(0, 3):
        wf_handles.append(await client.start_workflow(
            TestHeartbeatWorkflow.run,
            TestInput(
                # Pretend this is a long running activity that takes 10min
                activity_duration=timedelta(minutes=10),
                cancel_activity_after=None,
                heartbeat_timeout=timedelta(seconds=30.0)),
            id=f'test_heartbeat_wf_to_cancel-{i}',
            task_queue=_TEST_TASK_QUEUE,
            execution_timeout=timedelta(hours=1),
        ))
    await asyncio.sleep(1.0)
    for wf_handle in wf_handles:
        await wf_handle.cancel()
    wf_result = await client.execute_workflow(
        TestHeartbeatWorkflow.run,
        TestInput(activity_duration=timedelta(seconds=2.0),
                  cancel_activity_after=None,
                  heartbeat_timeout=timedelta(seconds=30.0)),
        id='test_heartbeat_wf_success',
        task_queue=_TEST_TASK_QUEUE,
        execution_timeout=timedelta(hours=1),
    )
    assert wf_result == 'succeeded'
