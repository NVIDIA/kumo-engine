# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.api.common.v1 import WorkflowExecution
from temporalio.client import WorkflowFailureError
from temporalio.service import RPCError, RPCStatusCode

from rest.testing import integration
from rest.testing.integration import (
    terminate_and_confirm_no_open_workflows,
    wait_for_training_job_to_finish,
    wait_for_training_workflow_to_finish,
)
from service_framework.deploy_stage import DeployStage
from statedb.config.common import TrainingJobID


@pytest.fixture(scope="session", autouse=True)
def setup_rest_test_session():
    """Keep pure lifecycle-helper tests independent of REST/Temporal servers."""
    yield ""


@pytest.fixture(scope="session", autouse=True)
def temporal_workers_ready():
    yield True


@pytest.fixture(autouse=True)
def _restore_rest_data_backend():
    yield


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.mark.asyncio
async def test_wait_for_training_job_to_finish_swallows_workflow_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    temporal_client = AsyncMock()
    temporal_client.result.side_effect = WorkflowFailureError(
        cause=RuntimeError("boom"))

    monkeypatch.setattr(
        "rest.testing.integration._training_workflow_id",
        lambda training_job_id: "wf-test-123",
    )

    with caplog.at_level("ERROR", logger="rest.testing.integration"):
        await wait_for_training_job_to_finish(
            TrainingJobID("training-1"),
            temporal_client,
        )

    temporal_client.result.assert_awaited_once_with("wf-test-123")
    assert "wf-test-123" in caplog.text
    assert "Unexpected error while waiting for training job" in caplog.text


@pytest.mark.asyncio
async def test_wait_for_training_workflow_to_finish_propagates_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    temporal_client = AsyncMock()
    failure = WorkflowFailureError(cause=RuntimeError("boom"))
    temporal_client.result.side_effect = failure
    monkeypatch.setattr(
        "rest.testing.integration._training_workflow_id",
        lambda training_job_id: "wf-test-123",
    )

    with pytest.raises(WorkflowFailureError) as exc_info:
        await wait_for_training_workflow_to_finish(
            TrainingJobID("training-1"),
            temporal_client,
        )

    assert exc_info.value is failure
    temporal_client.result.assert_awaited_once_with("wf-test-123")


def _test_deploy_config() -> MagicMock:
    config = MagicMock()
    config.deploy_mode = DeployStage.TEST
    config.instance_id = "pytest-abc"
    config.id_suffix = "-pytest-abc"
    config.task_queue.return_value = "-pytest-abc"
    config.temporal_namespace = "pytest"
    return config


@pytest.mark.asyncio
async def test_workflow_cleanup_rejects_non_unique_task_queue_suffix():
    config = _test_deploy_config()
    config.task_queue.return_value = ""

    with pytest.raises(RuntimeError, match="unique pytest task-queue suffix"):
        await terminate_and_confirm_no_open_workflows(config)


@pytest.mark.asyncio
async def test_workflow_cleanup_closes_client_when_connect_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    wrapper = MagicMock()
    wrapper.get = AsyncMock(side_effect=RuntimeError("connect failed"))
    wrapper.close = AsyncMock()
    monkeypatch.setattr(integration, "TemporalClient", lambda config: wrapper)

    with pytest.raises(RuntimeError, match="connect failed"):
        await terminate_and_confirm_no_open_workflows(_test_deploy_config())

    wrapper.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_workflow_cleanup_closes_client_when_list_rpc_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    workflow_service = MagicMock()
    workflow_service.list_open_workflow_executions = AsyncMock(
        side_effect=RuntimeError("list failed"))
    client = SimpleNamespace(workflow_service=workflow_service)
    wrapper = MagicMock()
    wrapper.get = AsyncMock(return_value=client)
    wrapper.close = AsyncMock()
    monkeypatch.setattr(integration, "TemporalClient", lambda config: wrapper)

    with pytest.raises(RuntimeError, match="list failed"):
        await terminate_and_confirm_no_open_workflows(_test_deploy_config())

    wrapper.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_workflow_cleanup_only_terminates_session_task_queue(
    monkeypatch: pytest.MonkeyPatch,
):
    ours = SimpleNamespace(
        task_queue="data-backend-worker-pytest-abc",
        execution=WorkflowExecution(workflow_id="ours", run_id="run-ours"),
    )
    theirs = SimpleNamespace(
        task_queue="data-backend-worker-pytest-other",
        execution=WorkflowExecution(workflow_id="theirs", run_id="run-theirs"),
    )
    workflow_service = MagicMock()
    workflow_service.list_open_workflow_executions = AsyncMock(side_effect=[
        SimpleNamespace(executions=[ours, theirs], next_page_token=b""),
        SimpleNamespace(executions=[], next_page_token=b""),
    ])
    workflow_service.terminate_workflow_execution = AsyncMock()
    client = SimpleNamespace(workflow_service=workflow_service)
    wrapper = MagicMock()
    wrapper.get = AsyncMock(return_value=client)
    wrapper.close = AsyncMock()
    monkeypatch.setattr(integration, "TemporalClient", lambda config: wrapper)

    await terminate_and_confirm_no_open_workflows(_test_deploy_config())

    workflow_service.terminate_workflow_execution.assert_awaited_once()
    request = workflow_service.terminate_workflow_execution.await_args.args[0]
    assert request.workflow_execution.workflow_id == "ours"
    wrapper.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_workflow_cleanup_tolerates_terminate_not_found_race(
    monkeypatch: pytest.MonkeyPatch,
):
    ours = SimpleNamespace(
        task_queue="data-backend-worker-pytest-abc",
        execution=WorkflowExecution(workflow_id="ours", run_id="run-ours"),
    )
    workflow_service = MagicMock()
    workflow_service.list_open_workflow_executions = AsyncMock(side_effect=[
        SimpleNamespace(executions=[ours], next_page_token=b""),
        SimpleNamespace(executions=[], next_page_token=b""),
    ])
    not_found = RPCError("not found", RPCStatusCode.NOT_FOUND, b"")
    workflow_service.terminate_workflow_execution = AsyncMock(
        side_effect=not_found)
    client = SimpleNamespace(workflow_service=workflow_service)
    wrapper = MagicMock()
    wrapper.get = AsyncMock(return_value=client)
    wrapper.close = AsyncMock()
    monkeypatch.setattr(integration, "TemporalClient", lambda config: wrapper)

    await terminate_and_confirm_no_open_workflows(_test_deploy_config())

    workflow_service.terminate_workflow_execution.assert_awaited_once()
    wrapper.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_workflow_cleanup_tolerates_explicit_already_completed_race(
    monkeypatch: pytest.MonkeyPatch,
):
    ours = SimpleNamespace(
        task_queue="data-backend-worker-pytest-abc",
        execution=WorkflowExecution(workflow_id="ours", run_id="run-ours"),
    )
    workflow_service = MagicMock()
    workflow_service.list_open_workflow_executions = AsyncMock(side_effect=[
        SimpleNamespace(executions=[ours], next_page_token=b""),
        SimpleNamespace(executions=[], next_page_token=b""),
    ])
    already_completed = RPCError(
        "Workflow execution already completed",
        RPCStatusCode.FAILED_PRECONDITION,
        b"",
    )
    workflow_service.terminate_workflow_execution = AsyncMock(
        side_effect=already_completed)
    client = SimpleNamespace(workflow_service=workflow_service)
    wrapper = MagicMock()
    wrapper.get = AsyncMock(return_value=client)
    wrapper.close = AsyncMock()
    monkeypatch.setattr(integration, "TemporalClient", lambda config: wrapper)

    await terminate_and_confirm_no_open_workflows(_test_deploy_config())

    workflow_service.terminate_workflow_execution.assert_awaited_once()
    wrapper.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_workflow_cleanup_propagates_other_terminate_rpc_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    ours = SimpleNamespace(
        task_queue="data-backend-worker-pytest-abc",
        execution=WorkflowExecution(workflow_id="ours", run_id="run-ours"),
    )
    workflow_service = MagicMock()
    workflow_service.list_open_workflow_executions = AsyncMock(
        return_value=SimpleNamespace(executions=[ours], next_page_token=b""))
    unavailable = RPCError("unavailable", RPCStatusCode.UNAVAILABLE, b"")
    workflow_service.terminate_workflow_execution = AsyncMock(
        side_effect=unavailable)
    client = SimpleNamespace(workflow_service=workflow_service)
    wrapper = MagicMock()
    wrapper.get = AsyncMock(return_value=client)
    wrapper.close = AsyncMock()
    monkeypatch.setattr(integration, "TemporalClient", lambda config: wrapper)

    with pytest.raises(RPCError) as exc_info:
        await terminate_and_confirm_no_open_workflows(_test_deploy_config())

    assert exc_info.value is unavailable
    wrapper.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_workflow_cleanup_times_out_if_scoped_workflow_remains_open(
    monkeypatch: pytest.MonkeyPatch,
):
    ours = SimpleNamespace(
        task_queue="data-backend-worker-pytest-abc",
        execution=WorkflowExecution(workflow_id="ours", run_id="run-ours"),
    )
    response = SimpleNamespace(executions=[ours], next_page_token=b"")
    workflow_service = MagicMock()
    workflow_service.list_open_workflow_executions = AsyncMock(
        return_value=response)
    workflow_service.terminate_workflow_execution = AsyncMock()
    client = SimpleNamespace(workflow_service=workflow_service)
    wrapper = MagicMock()
    wrapper.get = AsyncMock(return_value=client)
    wrapper.close = AsyncMock()
    monkeypatch.setattr(integration, "TemporalClient", lambda config: wrapper)

    with pytest.raises(TimeoutError, match="Timed out cleaning up"):
        await terminate_and_confirm_no_open_workflows(
            _test_deploy_config(),
            timeout_seconds=0.001,
        )

    wrapper.close.assert_awaited_once_with()


def test_run_temporal_workers_cleans_workflows_before_stopping_workers(
    monkeypatch: pytest.MonkeyPatch,
):
    events = []

    class FakeThread:
        def __init__(self, *, target, args, daemon, **kwargs):
            del target, daemon, kwargs
            self.start_event, self.stop_event = args[:2]

        def start(self):
            events.append("start")
            self.start_event.set()

        def join(self, timeout):
            events.append(("join", timeout, self.stop_event.is_set()))

        def is_alive(self):
            return False

    async def before_shutdown():
        events.append(("cleanup", fake_thread.stop_event.is_set()))

    monkeypatch.setattr(integration, "Thread", FakeThread)
    workers = integration.run_temporal_workers(
        _test_deploy_config(),
        before_shutdown=before_shutdown,
    )
    assert next(workers) is True
    fake_thread = workers.gi_frame.f_locals["worker_daemon_thread"]

    with pytest.raises(StopIteration):
        next(workers)

    assert events == [
        "start",
        ("cleanup", False),
        ("join", 60, True),
    ]


def test_run_server_drains_worker_tasks_before_executor_shutdown(
    monkeypatch: pytest.MonkeyPatch,
):
    events = []
    workers = []

    class FakeWorker:
        def __init__(self, name):
            self.name = name
            self.task_queue = f"{name}-queue"
            self.is_shutdown = False
            self.finished = False
            self._stop = None
            workers.append(self)

        def config(self):
            return {"identity": self.name}

        async def run(self):
            self._stop = asyncio.Event()
            events.append((self.name, "run"))
            await self._stop.wait()
            await asyncio.sleep(0)
            self.finished = True
            events.append((self.name, "run-finished"))

        async def shutdown(self):
            events.append((self.name, "shutdown"))
            self.is_shutdown = True
            assert self._stop is not None
            self._stop.set()

    class FakeWorkerEnv:
        def __init__(self, name):
            self.name = name
            self.shutdown_args = None

        def shutdown(self, *, wait, cancel_futures):
            assert all(worker.finished for worker in workers)
            self.shutdown_args = (wait, cancel_futures)
            events.append((self.name, "env-shutdown"))

    data_worker = FakeWorker("data")
    ml_worker = FakeWorker("ml")
    data_env = FakeWorkerEnv("data-env")
    ml_env = FakeWorkerEnv("ml-env")
    worker_builds = iter([
        ([data_worker], data_env),
        ([ml_worker], ml_env),
    ])

    async def build_worker(*args, **kwargs):
        del args, kwargs
        return next(worker_builds)

    monkeypatch.setattr(integration, "build_test_worker", build_worker)

    def immediate_run_in_executor(loop, executor, func, *args):
        del executor

        async def wait_after_worker_tasks_start():
            # create_task(worker.run()) calls precede run_in_executor. Yield
            # once so both fake workers initialize before returning stop=True.
            await asyncio.sleep(0)
            return func(*args)

        return loop.create_task(wait_after_worker_tasks_start())

    monkeypatch.setattr(
        asyncio.BaseEventLoop,
        "run_in_executor",
        immediate_run_in_executor,
    )

    start_event = Event()
    stop_event = Event()
    stop_event.set()
    exceptions = Queue()
    config = _test_deploy_config()

    integration.run_server_until_stop(
        start_event,
        stop_event,
        config,
        exceptions,
    )

    assert start_event.is_set()
    assert exceptions.empty(), list(exceptions.queue)
    assert data_env.shutdown_args == (True, True)
    assert ml_env.shutdown_args == (True, True)
    last_worker_finish = max(
        events.index(("data", "run-finished")),
        events.index(("ml", "run-finished")),
    )
    first_env_shutdown = min(
        events.index(("data-env", "env-shutdown")),
        events.index(("ml-env", "env-shutdown")),
    )
    assert last_worker_finish < first_env_shutdown
