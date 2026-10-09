# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import multiprocessing
import os
import signal
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import timedelta
from time import monotonic, sleep
from unittest.mock import AsyncMock

import pytest
from temporalio import workflow
from temporalio.client import WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.service import RPCError, RPCStatusCode

from service_framework.deploy_env_config import DeployEnvConfig
from service_framework.deploy_stage import DeployStage
from temporalio_commons import WorkerOptions
from temporalio_commons.process_pool_utils import (
    HealingProcessPoolExecutor,
    process_pool_activity_defn,
)
from temporalio_commons.worker_registry import (
    RunSingleActivityWorkflow,
    WorkerEnv,
)


@pytest.mark.asyncio
async def test_single_activity_workflow_has_one_attempt_retry_boundary(
        monkeypatch):
    execute_activity = AsyncMock(return_value='result')
    monkeypatch.setattr(workflow, 'execute_activity', execute_activity)

    result = await RunSingleActivityWorkflow().run(
        'validation_activity', ['input'])

    assert result == 'result'
    retry_policy = execute_activity.await_args.kwargs['retry_policy']
    assert retry_policy.maximum_attempts == 1


@process_pool_activity_defn
def dummy_func(seconds_to_sleep: int = 60) -> str:
    for i in range(seconds_to_sleep):
        sleep(1.0)
        print(f'Sleep {i} second')

    return 'dummy_result'


@process_pool_activity_defn
def raise_ex_func():
    raise RPCError(message='dummy func raise ex',
                   status=RPCStatusCode.UNAVAILABLE,
                   raw_grpc_status=b'UNAVAILABLE')


@process_pool_activity_defn
def dummy_func2() -> str:
    return 'dummy_result2'


def read_deploy_env_config_json() -> str | None:
    return os.environ.get('DEPLOY_ENV_CONFIG_JSON')


@workflow.defn
class DummyTestWorkflow:  # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, raise_ex: bool = False, seconds_to_sleep: int = 1):
        if raise_ex:
            await workflow.execute_activity(
                raise_ex_func, schedule_to_close_timeout=timedelta(minutes=1),
                retry_policy=RetryPolicy(maximum_attempts=1))

        return await workflow.execute_activity(
            dummy_func, args=[seconds_to_sleep],
            schedule_to_close_timeout=(timedelta(minutes=1) +
                                       timedelta(seconds=seconds_to_sleep)),
            retry_policy=RetryPolicy(maximum_attempts=1))


@workflow.defn
class DummyTestWorkflow2:  # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, *args):
        return await workflow.execute_activity(
            dummy_func2, schedule_to_close_timeout=timedelta(minutes=1))


@pytest.mark.asyncio
async def test_process_pool(temporal_config: DeployEnvConfig):
    test_worker_task_queue = 'test-worker-task-queue'
    worker_options = WorkerOptions(task_queue=test_worker_task_queue,
                                   max_worker_processes=1)
    worker_env = WorkerEnv('test-worker', worker_options, temporal_config)

    worker = await worker_env.build_worker(
        activities=[dummy_func, raise_ex_func], workflows=[DummyTestWorkflow])
    async with worker:
        client = await worker_env.client()
        wf_handle = await client.start_workflow(
            DummyTestWorkflow.run,
            id='dummy_wf_id1',
            args=[False, 60],  # simulate acitivty that executes for 60sec
            task_queue=test_worker_task_queue,
            retry_policy=RetryPolicy(maximum_attempts=1),  # no retry
            execution_timeout=timedelta(minutes=1),
        )

        # make sure activity has started executing in child process
        await asyncio.sleep(2.0)

        # pylint: disable=protected-access
        assert worker_env._process_pool_executor is not None
        broken_pool = worker_env._process_pool_executor._pool
        child_processes = worker_env._process_pool_executor._pool._processes
        child_pid = list(child_processes.keys())[0]
        # Signal a SEGFAULT in child process
        os.kill(child_pid, signal.SIGSEGV)

        with pytest.raises(WorkflowFailureError) as err_info:
            wf_result = await wf_handle.result()  # failed
        activity_error = err_info.value.cause
        assert isinstance(activity_error, ActivityError)
        assert isinstance(activity_error.cause, ApplicationError)
        assert 'process was killed abruptly' in activity_error.cause.message

        # The original process pool has become broken/unusable
        with pytest.raises(BrokenProcessPool):
            broken_pool.submit(dummy_func).result()

        # Next activity execution should suceed as the healing process pool
        # should auto-heal and recreate a new pool.
        wf_result = await client.execute_workflow(
            DummyTestWorkflow.run,
            id='dummy_wf_id2',
            args=[False, 1],
            task_queue=test_worker_task_queue,
            execution_timeout=timedelta(minutes=1),
        )
        assert wf_result == 'dummy_result'
        new_pool = worker_env._process_pool_executor._pool
        # The underlying pool should have been re-created upon
        # BrokenProcessPool exception after the child process got killed.
        assert new_pool is not broken_pool

        # Verify that un-pickable exception gets wrapped as ApplicationFailure
        with pytest.raises(WorkflowFailureError) as err_info:
            await client.execute_workflow(
                DummyTestWorkflow.run, True, id='dummy_wf_id_ex',
                task_queue=test_worker_task_queue,
                execution_timeout=timedelta(minutes=1))
        activity_error = err_info.value.cause
        assert isinstance(activity_error, ActivityError)
        assert isinstance(activity_error.cause, ApplicationError)

    worker_env.shutdown()


def test_process_pool_refreshes_test_deploy_env_json(monkeypatch):
    old_config = '{"storage": {"data_plane_storage_root": "/tmp/old"}}'
    new_config = '{"storage": {"data_plane_storage_root": "/tmp/new"}}'
    monkeypatch.setenv('DEPLOY_ENV_CONFIG_JSON', old_config)
    executor = HealingProcessPoolExecutor(
        1,
        deploy_stage=DeployStage.TEST,
        env_vars={'DEPLOY_ENV_CONFIG_JSON': old_config},
    )

    try:
        assert executor.submit(read_deploy_env_config_json).result(
            timeout=30) == old_config
        old_pool = executor._pool  # pylint: disable=protected-access

        monkeypatch.setenv('DEPLOY_ENV_CONFIG_JSON', new_config)

        assert executor.submit(read_deploy_env_config_json).result(
            timeout=30) == new_config
        assert executor._pool is not old_pool  # pylint: disable=protected-access
    finally:
        executor.shutdown()


@pytest.mark.asyncio
async def test_multi_executors(temporal_config: DeployEnvConfig):
    test_worker_thread_queue = 'test-worker-thread-queue'
    test_worker_process_queue = 'test_worker_process_queue'
    with pytest.raises(ValueError):
        WorkerOptions(
            task_queue=test_worker_thread_queue,
            max_worker_processes=1,
            max_worker_threads=1,
        )
    with pytest.raises(ValueError):
        WorkerOptions(
            task_queue=test_worker_thread_queue,
            max_worker_processes=1,
            multi_executors=True,
        )
    worker_options = WorkerOptions(task_queue=test_worker_thread_queue,
                                   max_worker_threads=1, max_worker_processes=1,
                                   multi_executors=True)
    worker_env = WorkerEnv('test-worker', worker_options, temporal_config)
    thread_worker = await worker_env.build_worker(activities=[dummy_func],
                                                  workflows=[DummyTestWorkflow],
                                                  use_process_executor=False)
    process_worker = await worker_env.build_worker(
        activities=[dummy_func2], workflows=[DummyTestWorkflow2],
        alt_task_queue=test_worker_process_queue, worker_name_suffix='-process',
        use_process_executor=True)

    async with thread_worker, process_worker:
        client = await worker_env.client()
        assert worker_env._process_pool_executor is not None
        assert worker_env._thread_pool_executor is not None
        # No task has been submitted and the executors have 0 worker
        # thread/process
        # pylint: disable=protected-access
        assert len(worker_env._thread_pool_executor._threads) == 0
        assert len(worker_env._process_pool_executor._pool._processes) == 0
        wf_result = await client.execute_workflow(
            DummyTestWorkflow.run,
            id='dummy_wf_id1',
            task_queue=test_worker_thread_queue,
            execution_timeout=timedelta(minutes=1),
        )
        assert wf_result == 'dummy_result'

        client = await worker_env.client()
        wf_result = await client.execute_workflow(
            DummyTestWorkflow2.run,
            id='dummy_wf_id2',
            task_queue=test_worker_process_queue,
            execution_timeout=timedelta(minutes=1),
        )
        assert wf_result == 'dummy_result2'
        # Each executor has processed 1 task, so they should have 1
        # worker thread/process.
        assert len(worker_env._thread_pool_executor._threads) == 1
        assert len(worker_env._process_pool_executor._pool._processes) == 1

    worker_env.shutdown()


def _slow_child_initializer(sleep_seconds: float) -> None:
    # Simulates an expensive child-process import (e.g. `import kumo.graph`)
    # so that warm_up() cannot finish within its budget.
    sleep(sleep_seconds)


def _warmup_probe_task() -> str:
    return 'ok'


class _SlowInitProcessPool(HealingProcessPoolExecutor):
    """HealingProcessPoolExecutor whose child processes are slow to initialize,
    used to exercise warm_up()'s timeout bound."""

    def __init__(self, max_workers: int, *, init_sleep_seconds: float):
        self._init_sleep_seconds = init_sleep_seconds
        super().__init__(max_workers)

    def _create_pool(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_slow_child_initializer,
            initargs=(self._init_sleep_seconds,),
        )


def test_warm_up_is_bounded_by_timeout():
    # Children take ~5s to initialize. An unbounded warm_up() would block the
    # caller (worker build, before the worker starts polling) for that whole
    # time; the timeout must cap it so the worker can start polling promptly.
    init_sleep_seconds = 5.0
    executor = _SlowInitProcessPool(2, init_sleep_seconds=init_sleep_seconds)
    try:
        started = monotonic()
        # Must not raise even though children are not ready within the budget.
        executor.warm_up(timeout=1.0)
        elapsed = monotonic() - started
        assert elapsed < init_sleep_seconds - 1, (
            f'warm_up blocked for {elapsed:.1f}s; timeout was not enforced')

        # The pool is still usable: the unfinished children warm up lazily and
        # the submitted task completes once a child is ready.
        assert executor.submit(_warmup_probe_task).result(timeout=60) == 'ok'
    finally:
        executor.shutdown()
