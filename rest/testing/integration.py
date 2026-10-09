# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from builtins import BaseExceptionGroup
from collections.abc import Awaitable, Callable
from datetime import timedelta
from queue import Queue
from threading import Event, Thread
from typing import Any, Dict, Generator, List, Sequence, Tuple, Type

import temporalio.api.workflowservice.v1 as wsv1
from temporalio.client import WorkflowExecutionStatus, WorkflowFailureError
from temporalio.service import RPCError, RPCStatusCode
from temporalio.worker import Worker

from io_utils import logging
from rest.backends.data_backend.worker import DATA_BACKEND_WORKER_NAME

# isort: off
from rest.backends.ml_backend.activities.ml_backend_activities import (
    _bp_col_stat_workflow_id, )
# isort: on
from rest.backends.ml_backend.ml_job_backend import (
    _batch_prediction_workflow_id,
    _training_workflow_id,
)
from rest.backends.ml_backend.worker import ML_WORKFLOW_WORKER
from service_framework.deploy_env_config import DeployEnvConfig
from service_framework.deploy_stage import DeployStage
from statedb.config.common import BatchPredictionJobID, TrainingJobID
from temporalio_commons import auto_rpc_config
from temporalio_commons.temporal_client import TemporalClient
from temporalio_commons.worker_main import build_workers
from temporalio_commons.worker_registry import WorkerEnv

# Because we are running multiple workers asynchronously in separate
# threads, the AutoRPC server worker might not be responsive to incoming
# RPC requests when another worker is running compute-intensive workload.
# Therefore we set a large rpc timeout in the e2e test to avoid workflow
# timeout error.
TEST_RPC_TIMEOUT_CONFIG = auto_rpc_config.AutoRpcTimeoutConfig(
    rpc_timeout=timedelta(minutes=1))

log = logging.getLogger(__name__)

_TEST_WORKER_SHUTDOWN_TIMEOUT_SECONDS = 30
# Includes worker shutdown plus the worker-environment executor drain. The
# latter can take roughly 45 seconds on constrained CI runners.
_TEST_WORKER_THREAD_JOIN_TIMEOUT_SECONDS = 60
_TEST_WORKFLOW_CLEANUP_TIMEOUT_SECONDS = 15
_TEST_WORKFLOW_CLEANUP_POLL_SECONDS = 0.05


async def build_test_worker(
    worker_name: str,
    deploy_env_config: DeployEnvConfig,
    env_vars: Dict[str, Any] = {},
    activities_override: Sequence[Callable] = [],
    workflows_override: Sequence[Type] = [],
) -> Tuple[List[Worker], WorkerEnv]:
    workers, worker_envs = await build_workers(
        deploy_env_config, worker_names=[worker_name],
        env_var_test_override=env_vars,
        activities_override=activities_override,
        workflows_override=workflows_override)
    assert len(worker_envs) == 1
    return workers, worker_envs[0]


def run_server_until_stop(start_event: Event, stop_event: Event,
                          deploy_env_config: DeployEnvConfig,
                          exceptions: Queue[BaseException],
                          ml_backend_env_vars: Dict[str, Any] | None = None,
                          activities_override: Sequence[Callable] = [],
                          workflows_override: Sequence[Type] = []):
    ml_backend_env_vars = ml_backend_env_vars or {}
    worker_tasks: list[asyncio.Task] = []

    async def run_server_wait_stop():
        # Every worker that now builds an in-process MLJobBackend /
        # MLJobOutputStore needs the full ML backend env: the workflow worker
        # (in-process backend, now the single lifecycle owner) and the
        # data-backend worker (its spawned process-pool materialize activities
        # lazily build an in-process MLJobOutputStore via
        # MLBackendSettings.parse_from_env, which must see the test StateDB
        # path).
        ml_backend_full_env = {
            **ml_backend_env_vars,
        }
        ml_workflow_env_vars = ml_backend_full_env
        try:
            data_worker, data_worker_env = await build_test_worker(
                DATA_BACKEND_WORKER_NAME, deploy_env_config,
                ml_backend_full_env, activities_override, workflows_override)
            ml_wf_worker, ml_wf_worker_env = await build_test_worker(
                ML_WORKFLOW_WORKER, deploy_env_config, ml_workflow_env_vars,
                activities_override, workflows_override)

            workers = (data_worker + ml_wf_worker)
            worker_envs = [
                data_worker_env,
                ml_wf_worker_env,
            ]

            for worker in workers:
                task = asyncio.create_task(worker.run())
                worker_tasks.append(task)

        except BaseException as ex:
            start_event.set()
            exceptions.put(ex)
            raise

        start_event.set()  # Signal the main thread that workers are running.

        # Keep workers running until stop_event is set from main thread.
        await asyncio.get_event_loop().run_in_executor(
            None,  # use the default executor thread pool
            stop_event.wait)
        log.info('Received stop signal, shutting down workers...')

        async def _shutdown_worker(worker: Worker):
            log.info(
                'Start shutting down %s worker (%s)',
                worker.config().get('identity'),
                worker.task_queue,
            )
            try:
                await asyncio.wait_for(
                    worker.shutdown(),
                    timeout=_TEST_WORKER_SHUTDOWN_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"Worker {worker.config().get('identity')} did not "
                    'complete bounded test shutdown') from None
            log.info('Completed shutdown of %s worker (%s)',
                     worker.config().get('identity'), worker.task_queue)

        # After receiving the stop event, shutdown all workers before closing
        # their executors or the event loop. Retain and drain the run tasks so
        # an activity cannot keep heartbeating against a closed test server.
        shutdown_results = await asyncio.gather(
            *[_shutdown_worker(worker) for worker in workers],
            return_exceptions=True,
        )
        worker_errors = [
            result for result in shutdown_results
            if isinstance(result, BaseException)
            and not isinstance(result, asyncio.CancelledError)
        ]
        all_workers_stopped = all(worker.is_shutdown for worker in workers)

        if not all_workers_stopped:
            for task in worker_tasks:
                task.cancel()
        try:
            worker_results = await asyncio.wait_for(
                asyncio.gather(*worker_tasks, return_exceptions=True),
                timeout=5,
            )
        except asyncio.TimeoutError:
            worker_results = [
                RuntimeError('Temporal worker run tasks did not drain')]
        worker_errors.extend(
            result for result in worker_results
            if isinstance(result, BaseException)
            and not isinstance(result, asyncio.CancelledError))

        log.info('Start shutting down all worker environments...')
        try:
            for worker_env in worker_envs:
                log.info('Shutting down %s worker env.', worker_env.name)
                # A completed Temporal Worker has already waited for every
                # activity. Only then is executor wait=True safe and useful:
                # it proves no activity thread/process survives test teardown.
                worker_env.shutdown(
                    wait=all_workers_stopped,
                    cancel_futures=True,
                )
                log.info('Worker env %s shutdown is completed.',
                         worker_env.name)
        except BaseException as ex:
            worker_errors.append(ex)

        if worker_errors:
            raise BaseExceptionGroup(
                'Temporal test workers failed bounded shutdown',
                worker_errors,
            )
        log.info('Shutdown completed, exiting worker daemon thread...')

    try:
        asyncio.run(run_server_wait_stop())
    except BaseException as ex:
        start_event.set()
        if exceptions.empty():
            exceptions.put(ex)


async def terminate_and_confirm_no_open_workflows(
    deploy_env_config: DeployEnvConfig,
    *,
    timeout_seconds: float = _TEST_WORKFLOW_CLEANUP_TIMEOUT_SECONDS,
) -> None:
    """Terminate this test session's workflows before stopping its workers.

    The task-queue suffix is test-instance-specific. Refuse a broad suffix so
    cleanup can never terminate workflows from another test session. One
    monotonic deadline covers connection, list, and terminate RPCs as well as
    confirmation polling.
    """
    if deploy_env_config.deploy_mode != DeployStage.TEST:
        raise RuntimeError('Workflow cleanup is restricted to test deploys')
    if timeout_seconds <= 0:
        raise ValueError('Workflow cleanup timeout must be positive')

    task_queue_suffix = deploy_env_config.task_queue('')
    instance_id = deploy_env_config.instance_id
    if (not instance_id or not instance_id.startswith('pytest-')
            or task_queue_suffix != deploy_env_config.id_suffix
            or instance_id not in task_queue_suffix):
        raise RuntimeError(
            'Workflow cleanup requires a unique pytest task-queue suffix')

    deadline = time.monotonic() + timeout_seconds

    def remaining() -> float:
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError('Timed out cleaning up test workflows')
        return seconds

    async def bounded(awaitable_factory: Callable[[], Awaitable]):
        # Compute the deadline before creating the coroutine. If the shared
        # deadline has expired, this avoids leaking an un-awaited RPC coroutine.
        timeout = remaining()
        return await asyncio.wait_for(awaitable_factory(), timeout=timeout)

    temporal_client = TemporalClient(deploy_env_config)

    async def cleanup() -> None:
        client = await bounded(temporal_client.get)

        async def scoped_open_workflows():
            executions = []
            next_page_token = b''
            while True:
                async def list_open_page():
                    return await (
                        client.workflow_service.list_open_workflow_executions(
                            wsv1.ListOpenWorkflowExecutionsRequest(
                                namespace=(deploy_env_config.
                                           temporal_namespace),
                                next_page_token=next_page_token,
                            ),
                            retry=True,
                        )
                    )

                response = await bounded(list_open_page)
                executions.extend(
                    execution for execution in response.executions
                    if execution.task_queue.endswith(task_queue_suffix))
                next_page_token = response.next_page_token
                if not next_page_token:
                    return executions

        open_workflows = await scoped_open_workflows()
        for workflow_execution in open_workflows:
            async def terminate():
                return await (
                    client.workflow_service.terminate_workflow_execution(
                        wsv1.TerminateWorkflowExecutionRequest(
                            namespace=deploy_env_config.temporal_namespace,
                            workflow_execution=workflow_execution.execution,
                            reason=(
                                'Test session cleanup before worker shutdown'),
                        ),
                        retry=True,
                    )
                )

            try:
                await bounded(terminate)
            except RPCError as error:
                already_completed = ('workflow execution already completed'
                                     in str(error).lower())
                if (error.status == RPCStatusCode.NOT_FOUND
                        or already_completed):
                    continue
                raise

        while True:
            open_workflows = await scoped_open_workflows()
            if not open_workflows:
                return
            await asyncio.sleep(
                min(_TEST_WORKFLOW_CLEANUP_POLL_SECONDS, remaining()))

    cleanup_error: BaseException | None = None
    try:
        await cleanup()
    except BaseException as ex:
        cleanup_error = ex

    try:
        await asyncio.wait_for(temporal_client.close(), timeout=5)
    except BaseException as close_error:
        if cleanup_error is not None:
            raise BaseExceptionGroup(
                'Test workflow cleanup and Temporal client close both failed',
                [cleanup_error, close_error],
            )
        raise

    if cleanup_error is not None:
        raise cleanup_error


def run_temporal_workers(
    deploy_env_config: DeployEnvConfig,
    *,
    before_shutdown: Callable[[], Awaitable[None]] | None = None,
) -> Generator[bool, None, None]:
    start_event = Event()
    stop_event = Event()
    exceptions = Queue[BaseException]()
    worker_daemon_thread = Thread(
        name='TemporalWorkerDaemon',
        target=run_server_until_stop,
        args=[start_event, stop_event, deploy_env_config, exceptions],
        daemon=True,
    )
    worker_daemon_thread.start()
    start_event.wait()
    if not exceptions.empty():
        raise RuntimeError(
            'Temporal worker failed to start') from exceptions.get()

    cleanup_error: BaseException | None = None
    worker_error: BaseException | None = None
    try:
        yield True
    finally:
        if before_shutdown is not None:
            try:
                asyncio.run(before_shutdown())
            except BaseException as ex:
                cleanup_error = ex

        stop_event.set()
        worker_daemon_thread.join(
            timeout=_TEST_WORKER_THREAD_JOIN_TIMEOUT_SECONDS)
        if worker_daemon_thread.is_alive():
            worker_error = RuntimeError(
                'Temporal test worker thread did not stop within the '
                'bounded shutdown deadline')
        elif not exceptions.empty():
            worker_error = exceptions.get()

        if cleanup_error is not None and worker_error is not None:
            raise BaseExceptionGroup(
                'Test workflow and worker cleanup both failed',
                [cleanup_error, worker_error],
            )
        if cleanup_error is not None:
            raise cleanup_error
        if worker_error is not None:
            raise worker_error


async def wait_for_training_job_to_finish(training_job_id: TrainingJobID,
                                          temporal_client: TemporalClient):
    training_wf_id = _training_workflow_id(training_job_id)
    try:
        await temporal_client.result(training_wf_id)
    except WorkflowFailureError as e:
        log.error('Unexpected error while waiting for training job wf_id=%s',
                  training_wf_id)
        log.exception(e)


async def wait_for_training_workflow_to_finish(
    training_job_id: TrainingJobID,
    temporal_client: TemporalClient,
) -> None:
    """Wait for the parent workflow, including postprocessing, to finish.

    Unlike the legacy helper above, workflow failure is intentionally
    propagated. A DONE job row may be visible before detached postprocessing
    activities complete, so DiskGraph E2Es use this strict boundary before
    allowing session-scoped Temporal workers to shut down.
    """
    await temporal_client.result(_training_workflow_id(training_job_id))


async def wait_for_prediction_job_to_finish(
    predict_job_id: BatchPredictionJobID,
    temporal_client: TemporalClient,
    retry_count: int = 0,
):
    prediction_wf_id = _batch_prediction_workflow_id(predict_job_id)
    try:
        await temporal_client.result(prediction_wf_id)
    except WorkflowFailureError as e:
        log.error('Unexpected error while waiting for prediction job')
        log.exception(e)
    except RPCError as e:
        log.error('Transient RPC Error while waiting for job to finish')
        log.exception(e)
        if retry_count < 5:
            await wait_for_prediction_job_to_finish(
                predict_job_id,
                temporal_client,
                retry_count + 1,
            )
        else:
            raise e


async def wait_for_prediction_stats_to_finish(
        predict_job_id: BatchPredictionJobID, temporal_client: TemporalClient):
    prediction_stats_wf_id = _bp_col_stat_workflow_id(predict_job_id)
    try:
        await temporal_client.result(prediction_stats_wf_id)
    except WorkflowFailureError as e:
        log.error('Unexpected error while waiting for prediction stats job')
        log.exception(e)


async def wait_for_workflow_to_end(
    wf_id: str, temporal_client: TemporalClient,
    expected_end_status: WorkflowExecutionStatus = WorkflowExecutionStatus.
    COMPLETED):
    try:
        await temporal_client.result(wf_id)
    except WorkflowFailureError:
        # Allow workflow to fail and verify the end status via wf description.
        pass

    wf_desc = await temporal_client.get_workflow_desc(wf_id)
    log.info(f'wf_desc: {wf_desc}')
    assert wf_desc
    assert wf_desc.status == expected_end_status, (
        f'Workflow {wf_id} ended in status {wf_desc.status} but '
        f'{expected_end_status} is expected')
