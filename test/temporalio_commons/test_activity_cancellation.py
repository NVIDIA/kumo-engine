# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import random
from datetime import timedelta
from typing import List

import pytest
from temporalio import activity, workflow
from temporalio.client import WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, CancelledError

from rest.backends.ml_backend.workflows.mixins import (
    JobWorkflowCancellationMixin,
)
from service_framework.deploy_env_config import DeployEnvConfig
from temporalio_commons import WorkerOptions
from temporalio_commons.worker_registry import WorkerEnv


@activity.defn
async def dummy_activity(seconds_to_sleep: int, heartbeat: bool = False):
    for i in range(seconds_to_sleep):
        try:
            await asyncio.sleep(1.0)
        except BaseException as ex:
            print(f'received cancellation while sleeping: {type(ex)}')
            raise

        print(f'sleep {i+1} seconds')
        if heartbeat:
            try:
                activity.heartbeat()
            except BaseException as ex:
                print(f'received cancellation type={type(ex)} while heartbeat')
                raise
    return


async def run_activity_shield_cancel(act_task: asyncio.Task):
    try:
        return await asyncio.shield(act_task)
    except asyncio.CancelledError:
        print('Workflow receives cancellation signal!')
        act_task.cancel()
        await act_task
        print('Acitivty task is completed, re-raising cancellation exception.')
        raise


_TEST_HEARTBEAT_TIMEOUT = timedelta(seconds=10)


@workflow.defn
class DummyTestWorkflow:  # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, seconds_to_sleep: int, num_activities: int = 1,
                  enable_heartbeat: bool = False) -> str:
        for _ in range(num_activities):
            act_task = asyncio.create_task(
                workflow.execute_activity(
                    dummy_activity,
                    args=[seconds_to_sleep, enable_heartbeat],
                    schedule_to_close_timeout=(
                        timedelta(minutes=1) +
                        timedelta(seconds=seconds_to_sleep)),
                    retry_policy=RetryPolicy(maximum_attempts=1),
                    heartbeat_timeout=(None if not enable_heartbeat else
                                       _TEST_HEARTBEAT_TIMEOUT),
                    cancellation_type=workflow.ActivityCancellationType.
                    WAIT_CANCELLATION_COMPLETED,
                ))
            await run_activity_shield_cancel(act_task)

        return 'Done'


@pytest.mark.asyncio
async def test_activity_cancellation(temporal_config: DeployEnvConfig):
    test_worker_task_queue = 'test-worker-task-queue'
    worker_options = WorkerOptions(task_queue=test_worker_task_queue)
    worker_env = WorkerEnv('test-worker', worker_options, temporal_config)

    worker = await worker_env.build_worker(activities=[dummy_activity],
                                           workflows=[DummyTestWorkflow])
    async with worker:
        client = await worker_env.client()

        # Case 1: activity does not heartbeat and thus ignores cancellation.
        activity_duration_sec = 5
        num_activities = 2
        enable_heartbeat = False
        wf_handle = await client.start_workflow(
            DummyTestWorkflow.run,
            id='dummy_wf_id1',
            args=[activity_duration_sec, num_activities, enable_heartbeat],
            task_queue=test_worker_task_queue,
            retry_policy=RetryPolicy(maximum_attempts=1),  # no retry
            execution_timeout=timedelta(hours=1),
        )
        # Cancel wf in the middle of 1st activity execution.
        await asyncio.sleep(2.0)
        await wf_handle.cancel()

        with pytest.raises(WorkflowFailureError) as err_info:
            await wf_handle.result()  # failed
        cause = err_info.value.cause
        assert isinstance(cause, CancelledError)
        assert cause.message == 'Workflow cancelled'

        # Case 2: activity is long running, it heartbeats and receives
        # cancellation via heartbeat.
        activity_duration_sec = 600
        # if cancellation is ignored, it would take 600 * 10 seconds to finish
        # this workflow!!!
        num_activities = 10
        enable_heartbeat = True
        wf_handle = await client.start_workflow(
            DummyTestWorkflow.run,
            id='dummy_wf_id1',
            args=[activity_duration_sec, num_activities, enable_heartbeat],
            task_queue=test_worker_task_queue,
            retry_policy=RetryPolicy(maximum_attempts=1),  # no retry
            execution_timeout=timedelta(hours=1),
        )
        # Cancel wf in the middle of 1st activity execution.
        await asyncio.sleep(2.0)
        await wf_handle.cancel()

        with pytest.raises(WorkflowFailureError) as err_info:
            # While it would take the whole workflow 600 * 10 seconds to finish
            # we expect the activity to receive cancellation through
            # heartbeating at least every _TEST_HEARTBEAT_TIMEOUT seconds.
            await asyncio.wait_for(
                wf_handle.result(),
                timeout=_TEST_HEARTBEAT_TIMEOUT.total_seconds() * 2)
        cause = err_info.value.cause
        assert isinstance(cause, CancelledError)
        assert cause.message == 'Workflow cancelled'

    worker_env.shutdown()


@workflow.defn
class TestCancellationWorkflow(JobWorkflowCancellationMixin):
    @workflow.run
    async def run(self, activity_durations: List[float]) -> float:
        return await self.run_until_canceled(self._run(activity_durations))

    async def _run(self, activity_durations: List[float]) -> float:
        for i, duration in enumerate(activity_durations):
            print(f'Running activity {i} for {duration:.2f} sec')
            activity_err = None
            try:
                await workflow.execute_activity(
                    dummy_activity,
                    args=[duration, False],
                    schedule_to_close_timeout=timedelta(seconds=10),
                )
                print(f'Activity {i} is done')
                self.check_cancelled()
            except BaseException as err:
                print(f'TestCancellationWorkflow caught ex {type(err)}')
                activity_err = err
                raise
            finally:
                if activity_err:
                    print('Executing cleanup...')
                    await asyncio.sleep(1.0)
                    print('Cleanup is done.')

        return sum(activity_durations)


@pytest.mark.asyncio
async def test_workflow_cancellation(temporal_config: DeployEnvConfig):
    test_worker_task_queue = 'test-worker-task-queue'
    worker_options = WorkerOptions(task_queue=test_worker_task_queue)
    worker_env = WorkerEnv('test-worker', worker_options, temporal_config)

    worker = await worker_env.build_worker(activities=[dummy_activity],
                                           workflows=[TestCancellationWorkflow])
    async with worker:
        client = await worker_env.client()
        wf_handle1 = await client.start_workflow(
            TestCancellationWorkflow.run,
            # short activities with 10ms to 200ms
            [random.uniform(0.01, 0.2) for _ in range(100)],
            id='wf-test_workflow_cancellation-1',
            task_queue=test_worker_task_queue,
            execution_timeout=timedelta(hours=1),
        )
        wf_handle2 = await client.start_workflow(
            TestCancellationWorkflow.run,
            # short activities with 10ms to 200ms
            [random.uniform(0.01, 0.2) for _ in range(100)],
            id='wf-test_workflow_cancellation-2',
            task_queue=test_worker_task_queue,
            execution_timeout=timedelta(hours=1),
        )

        await asyncio.sleep(random.uniform(0.5, 1))
        # Here we test both ways to cancel workflow:
        # 1) native cancellation via cancel() API; 2) send a "cancel_wf" signal
        await wf_handle1.cancel()
        await wf_handle2.signal('cancel_wf')
        print('Requested cancellation to workflow!')

        with pytest.raises(WorkflowFailureError) as err_info:
            await wf_handle1.result()
        cause = err_info.value.cause
        assert isinstance(cause, CancelledError)
        assert cause.message == 'Workflow cancelled'

        with pytest.raises(WorkflowFailureError) as err_info:
            await wf_handle2.result()
        cause = err_info.value.cause
        # Slightly different from native cancellation via temporal client,
        # sending "cancel_wf" signal may end up with ActivityError with cause
        # of CancelledError.
        if isinstance(cause, ActivityError):
            cause = cause.cause
        assert isinstance(cause, CancelledError)
