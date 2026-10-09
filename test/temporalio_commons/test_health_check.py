# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import pathlib
import shlex
import sys
from datetime import timedelta
from time import sleep
from typing import List

import pytest
import pytest_asyncio
from temporalio import activity, workflow
from temporalio.client import WorkflowFailureError, WorkflowHandle
from temporalio.common import RetryPolicy

from io_utils import fsutil
from service_framework.deploy_env_config import DeployEnvConfig
from temporalio_commons import WorkerOptions, terminate_open_workflows
from temporalio_commons.health_check import health_check
from temporalio_commons.process_pool_utils import process_pool_activity_defn
from temporalio_commons.worker_registry import WorkerEnv


@pytest_asyncio.fixture(autouse=True, scope='function')
async def terminate_workflows_after_test(temporal_config: DeployEnvConfig):
    yield
    await terminate_open_workflows.run(temporal_config, True)


@activity.defn
async def hc_async_activity(duration_seconds: int, block_event_loop: bool):
    for i in range(duration_seconds):
        print(f'async activity sleep for {i} seconds...')
        activity.heartbeat('heartbeat from async activity...')
        if block_event_loop:
            sleep(1.0)
        else:
            await asyncio.sleep(1.0)

    return


def set_send_heartbeat(file_path: str, send_heartbeat: bool):
    data = {'send_heartbeat': send_heartbeat}
    with fsutil.get_handle(file_path, 'w') as f:
        f.write(json.dumps(data))


def get_send_heartbeat(file_path: str):
    with fsutil.get_handle(file_path, 'r') as f:
        data = f.read()
        return json.loads(data)['send_heartbeat']


@process_pool_activity_defn(
    # effectively disable auto heartbeater!
    heartbeat_interval=timedelta(hours=24))
def hc_sync_activity(
    duration_seconds: int,
    send_heartbeat_file_path: str,
    first_heartbeat_sent_file_path: str,
):
    """
    Because this dummy activity may be executed in separate process, we use
    `send_heartbeat_file_path` and `first_heartbeat_sent_file_path` to
    communicate status/command with pytest main loop.
    """
    first_heartbeat_sent = False
    for i in range(duration_seconds):
        heartbeat = get_send_heartbeat(send_heartbeat_file_path)
        print(f'sync activity sleep for {i} seconds, heartbeat: {heartbeat}')
        if heartbeat:
            activity.heartbeat('heartbeat from sync activity...')
            if not first_heartbeat_sent:
                first_heartbeat_sent = True
                fsutil.touch(first_heartbeat_sent_file_path)

        sleep(1.0)


@workflow.defn
class HealthCheckTestWorkflow_Async:  # pylint: disable=too-few-public-methods
    """
    This test workflow executes async activities with option to block either
    workflow task executor's event loop or block async activity event loop.
    """
    @workflow.run
    async def run(self, total_seconds: int, block_activity_loop: bool) -> str:
        await workflow.execute_activity(
            hc_async_activity,
            args=[total_seconds, block_activity_loop],
            schedule_to_close_timeout=timedelta(hours=1),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )

        return 'Done'


@workflow.defn
class HealthCheckTestWorkflow_Sync:  # pylint: disable=too-few-public-methods
    """
    This test workflow executes sync activities.
    """
    @workflow.run
    async def run(self, duration_seconds: int, num_activities: int,
                  send_heartbeat_file_path: str) -> str:
        base_dir = os.path.dirname(send_heartbeat_file_path)
        sync_activity_tasks = [
            workflow.start_activity(
                hc_sync_activity,
                args=[
                    duration_seconds, send_heartbeat_file_path,
                    os.path.join(base_dir, f'first_heartbeat_{i}')
                ],
                schedule_to_close_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(maximum_attempts=1),
                # set to None to disable auto heartbeater.
                heartbeat_timeout=None,
            ) for i in range(num_activities)
        ]
        # Also execute async activities and make sure that we detect stuck
        # executor for sync activities even when async activities are alive and
        # reporting heartbeats regularly.
        async_activity_tasks = [
            workflow.start_activity(
                hc_async_activity,
                args=[duration_seconds, False],
                schedule_to_close_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(maximum_attempts=1),
                # set to None to disable auto heartbeater.
                heartbeat_timeout=None,
            ) for _ in range(num_activities)
        ]

        await asyncio.wait(sync_activity_tasks)
        for async_activity_task in async_activity_tasks:
            async_activity_task.cancel()

        return 'Done'


@pytest.mark.asyncio
async def test_healthcheck_no_worker(temporal_config: DeployEnvConfig):
    """Fail health check if there is no worker running."""
    test_worker_task_queue = 'test-healthcheck-task-queue'

    # Health check failed when there is no worker running.
    with pytest.raises(WorkflowFailureError):
        await health_check(temporal_config, [test_worker_task_queue],
                           timeout=timedelta(seconds=2))

    worker_options = WorkerOptions(task_queue=test_worker_task_queue)
    worker_env = WorkerEnv('test-worker', worker_options, temporal_config)

    worker = await worker_env.build_worker(
        activities=[hc_async_activity],
        workflows=[HealthCheckTestWorkflow_Async])
    async with worker:
        # Now health check should pass after worker goes alive.
        await health_check(temporal_config, [test_worker_task_queue])


@pytest.mark.asyncio
@pytest.mark.parametrize("use_process_pool", [False, True])
async def test_healthcheck_stuck_executor(use_process_pool: bool,
                                          tmp_path: pathlib.Path,
                                          temporal_config: DeployEnvConfig):
    """Detect the case when activity executor executor is stuck."""
    test_worker_task_queue = 'test-healthcheck-task-queue'
    max_workers = 3
    worker_options = WorkerOptions(task_queue=test_worker_task_queue)
    if use_process_pool:
        pool_type = 'processpool'
        worker_options.max_worker_processes = max_workers
    else:
        pool_type = 'threadpool'
        worker_options.max_worker_threads = max_workers
    worker_env = WorkerEnv('test-worker', worker_options, temporal_config)

    worker = await worker_env.build_worker(
        activities=[hc_sync_activity, hc_async_activity],
        workflows=[HealthCheckTestWorkflow_Sync])

    # By default all activities should heartbeat.
    send_heartbeat_file_path = (tmp_path /
                                'send_heartbeat.json').absolute().as_posix()
    set_send_heartbeat(send_heartbeat_file_path, True)

    async with worker:
        # pass health check when no activity is running.
        await health_check(worker_env.deploy_config, [test_worker_task_queue],
                           unresponsive_threshold=timedelta(seconds=2))

        client = await worker_env.client()

        # Kick off a workflow with 2 activities that occupy all threads.
        wf_1 = await client.start_workflow(
            HealthCheckTestWorkflow_Sync.run,
            id=f'health_check_test_wf-{pool_type}-1',
            args=[120, max_workers, send_heartbeat_file_path],
            task_queue=test_worker_task_queue,
            retry_policy=RetryPolicy(maximum_attempts=1),  # no retry
            execution_timeout=timedelta(seconds=120),
        )

        print('Waiting for all activities to start sending first heartbeat.')
        while True:
            if all(
                    fsutil.exists(
                        os.path.join(tmp_path.absolute().as_posix(),
                                     f'first_heartbeat_{i}'))
                    for i in range(max_workers)):
                break
            await asyncio.sleep(0.1)
        print('All activities have started running.')

        # Passing health check as we see heartbeats from the worker.
        await health_check(worker_env.deploy_config, [test_worker_task_queue],
                           unresponsive_threshold=timedelta(seconds=2.0))

        # Disable heartbeat to simulate stuck worker!
        set_send_heartbeat(send_heartbeat_file_path, False)
        await asyncio.sleep(3.0)  # 3sec > 2sec (unresponsive threshold)
        with pytest.raises(RuntimeError):
            await health_check(worker_env.deploy_config,
                               [test_worker_task_queue],
                               unresponsive_threshold=timedelta(seconds=2))

        # Now we remove the heartbeat command file this would interrupt
        # execution of activities in the process pool due to FileNotFound.
        print('Delete send_heartbeat file to kill activities!')
        fsutil.rm(send_heartbeat_file_path)
        await wf_1.result()

    worker_env.shutdown()


@pytest.mark.asyncio
async def test_healthcheck_block_event_loop(temporal_config: DeployEnvConfig):
    """Detect the case when workflow task executor is event loop is stuck."""
    test_worker_task_queue = 'test-healthcheck-task-queue'
    worker_options = WorkerOptions(task_queue=test_worker_task_queue)
    worker_env = WorkerEnv('test-worker', worker_options, temporal_config)

    worker = await worker_env.build_worker(
        activities=[hc_async_activity],
        workflows=[HealthCheckTestWorkflow_Async])

    async with worker:
        client = await worker_env.client()

        block_event_loop = False
        wf_handles: List[WorkflowHandle] = []
        for i in range(3):
            wf_handle = await client.start_workflow(
                HealthCheckTestWorkflow_Async.run,
                id=f'health_check_test_wf-async-good-{i}',
                args=[5, block_event_loop],
                task_queue=test_worker_task_queue,
                retry_policy=RetryPolicy(maximum_attempts=1),  # no retry
                execution_timeout=timedelta(seconds=20),
            )
            wf_handles.append(wf_handle)
        # Expect to pass health check when execute async tasks normally.
        await health_check(worker_env.deploy_config, [test_worker_task_queue])
        await asyncio.gather(*[wf_handle.cancel() for wf_handle in wf_handles])

        # Now perform health check in a subprocess -- because we are going to
        # block the asyncio event loop in the current process.
        cmd = ('sleep 2; echo "starting health check"; '
               f'{shlex.quote(sys.executable)} '
               'temporalio_commons/health_check.py --timeout 2')
        subproc_env = os.environ.copy()
        subproc_env['WORKERS'] = 'test-worker'

        hc_proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, env=subproc_env,
            stderr=asyncio.subprocess.PIPE, shell=True)

        # Kick off a few async workflow that blocks workflow task executor's
        # event loop
        block_event_loop = True
        wf_handle = await client.start_workflow(
            HealthCheckTestWorkflow_Async.run,
            id='health_check_test_wf-async-bad',
            args=[10, block_event_loop],
            task_queue=test_worker_task_queue,
            retry_policy=RetryPolicy(maximum_attempts=1),  # no retry
            execution_timeout=timedelta(seconds=20),
        )

        stdout, stderr = await hc_proc.communicate()
        output = stdout.decode()
        error_output = stderr.decode()
        assert hc_proc.returncode != 0, error_output
        assert 'Health check failed' in output, error_output
        assert 'Workflow timed out' in output, error_output

    worker_env.shutdown()
