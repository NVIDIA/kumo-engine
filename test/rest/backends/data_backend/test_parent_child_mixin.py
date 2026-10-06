# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from datetime import datetime, timedelta
from typing import Dict

import pytest
import pytest_asyncio
from kumoapi.source_table import FileType, S3SourceTable
from temporalio import workflow
from temporalio.client import (
    WorkflowExecutionStatus,
    WorkflowFailureError,
    WorkflowHandle,
)
from temporalio.exceptions import CancelledError, ChildWorkflowError
from temporalio.service import RPCError, RPCStatusCode
from temporalio.workflow import ChildWorkflowCancellationType

from kumo.data.backend_cache import (
    AsyncDataTaskStatus,
    TableDataTaskResult,
    TableRefreshStage,
)
from kumo.validation.validation import ValidationError, ValidationResponse
from rest.backends.data_backend.activities import DataBackendActivities
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.workflows.mixins import (
    TableChildWorkflowMixin,
    TableParentWorkflowMixin,
)
from rest.config.table_v2 import TableDefinition
from rest.settings import DataBackendSettings
from service_framework.deploy_env_config import DeployEnvConfig
from temporalio_commons import terminate_open_workflows
from temporalio_commons.error_utils import extract_public_exception
from temporalio_commons.temporal_client import TemporalClient
from temporalio_commons.worker_registry import WorkerEnv, WorkerOptions

NUM_CHILD_WFS = 3


@workflow.defn
class DummyChildWorkflow(TableChildWorkflowMixin):
    def __init__(self) -> None:
        super().__init__()

    @workflow.signal
    async def complete_stage(self, stage: TableRefreshStage):
        await self.notify_watcher(
            TableDataTaskResult(status=AsyncDataTaskStatus.DONE,
                                validation=ValidationResponse(), stage=stage))

    def _completed(self) -> bool:
        for status_update in self._status_updates:
            if status_update.stage == TableRefreshStage.COMPLETED:
                return True
        return False

    @workflow.run
    async def run(self, table_def: TableDefinition) -> TableDataTaskResult:
        parent = workflow.info().parent
        if parent:
            await self.register_watcher(parent.workflow_id, parent.run_id)

        await workflow.wait_condition(self._completed)
        return TableDataTaskResult(status=AsyncDataTaskStatus.DONE,
                                   validation=ValidationResponse(),
                                   stage=TableRefreshStage.COMPLETED)


@workflow.defn
class DummyParentWorkflow(TableParentWorkflowMixin):
    def __init__(self) -> None:
        super(DummyParentWorkflow, self).__init__()
        self._stage_tasks: Dict[TableRefreshStage,
                                asyncio.Task[ValidationResponse]] = {}

    @workflow.query
    async def stage_completed(self, stage: TableRefreshStage) -> bool:
        task = self._stage_tasks.get(stage)
        return task is not None and task.done() and not task.exception()

    @workflow.run
    async def run(self, deploy_config: DeployEnvConfig) -> str:
        start_child_refresh_coros = []
        for i in range(NUM_CHILD_WFS):
            table_def = TableDefinition(
                cols=[],
                source_table=S3SourceTable(
                    connector_id=f'connector_{"0" * 32}',
                    source={'mode': 'relative', 'path': f'child_table_{i}'},
                    file_type=FileType.PARQUET,
                ),
                pkey=None,
                time_col=None,
            )
            start_child_refresh_coros.append(
                self.start_child_refresh(
                    DummyChildWorkflow.run,
                    DataBackendContext(deploy_config, DataBackendSettings()),
                    args=[table_def],
                    child_id=table_def.source_table_id,
                    child_wf_id=f'test-child-table-refresh-wf-{i}',
                    cancellation_type=ChildWorkflowCancellationType.
                    WAIT_CANCELLATION_COMPLETED,
                    execution_timeout=timedelta(minutes=1),
                ))
        await asyncio.gather(*start_child_refresh_coros)

        for stage in TableRefreshStage:
            self._stage_tasks[stage] = asyncio.ensure_future(
                self.wait_completion_at_stage(stage))

        await self._stage_tasks[TableRefreshStage.COMPLETED]
        return 'Done!'


@workflow.defn
class DummyFailingChildWorkflow(TableChildWorkflowMixin):
    def __init__(self) -> None:
        super().__init__()

    @workflow.run
    async def run(self, table_def: TableDefinition) -> TableDataTaskResult:
        parent = workflow.info().parent
        if parent:
            await self.register_watcher(parent.workflow_id, parent.run_id)

        validation = ValidationResponse(errors=[
            ValidationError(
                title='Column stats computation failed for table '
                '"child_table_0"',
                message='Reason: missing column id',
            ),
        ])
        await self.notify_watcher(
            TableDataTaskResult(
                status=AsyncDataTaskStatus.FAILED,
                validation=validation,
                stage=TableRefreshStage.COMPUTE_MIN_COL_STATS,
            ))
        return TableDataTaskResult(
            status=AsyncDataTaskStatus.FAILED,
            validation=validation,
            stage=TableRefreshStage.COMPUTE_MIN_COL_STATS,
        )


@workflow.defn
class DummyFailingParentWorkflow(TableParentWorkflowMixin):
    @workflow.run
    async def run(self, deploy_config: DeployEnvConfig) -> str:
        table_def = TableDefinition(
            cols=[],
            source_table=S3SourceTable(
                connector_id=f'connector_{"0" * 32}',
                source={'mode': 'relative', 'path': 'child_table_0'},
                file_type=FileType.PARQUET,
            ),
            pkey=None,
            time_col=None,
        )
        await self.start_child_refresh(
            DummyFailingChildWorkflow.run,
            DataBackendContext(deploy_config, DataBackendSettings()),
            args=[table_def],
            child_id=table_def.source_table_id,
            child_wf_id='test-failing-child-table-refresh-wf',
            cancellation_type=ChildWorkflowCancellationType.
            WAIT_CANCELLATION_COMPLETED,
            execution_timeout=timedelta(minutes=1),
        )
        await self.wait_completion_at_stage(
            TableRefreshStage.COMPUTE_MIN_COL_STATS)
        return 'Done!'


@pytest_asyncio.fixture(scope='function', autouse=True)
async def auto_cleanup_open_workflows():
    # Make sure to terminate open workflows after each test function.
    yield
    await terminate_open_workflows.run(DeployEnvConfig.for_test(), True)


@pytest.mark.asyncio
async def test_notification():
    test_worker_task_queue = 'test-worker-task-queue'
    worker_options = WorkerOptions(task_queue=test_worker_task_queue)
    deploy_config = DeployEnvConfig.for_test()
    worker_env = WorkerEnv('parent-child-test-worker', worker_options,
                           deploy_config)
    temporal_client = worker_env.temporal_client

    worker = await worker_env.build_worker(
        activities=[DataBackendActivities.wait_on_workflow],
        workflows=[DummyChildWorkflow, DummyParentWorkflow],
    )
    async with worker:
        client = await temporal_client.get()
        parent_wf_handle = await client.start_workflow(
            DummyParentWorkflow.run,
            id='dummy_parent_wf_id',
            args=[deploy_config],
            task_queue=test_worker_task_queue,
            execution_timeout=timedelta(minutes=1),
        )

        await wait_parent_stage(temporal_client, parent_wf_handle,
                                TableRefreshStage.INGEST_TABLE)

        await wait_parent_stage(temporal_client, parent_wf_handle,
                                TableRefreshStage.COMPLETED)
        result = await parent_wf_handle.result()
        assert result == 'Done!'


@pytest.mark.asyncio
async def test_cancellation_single_parent():
    test_worker_task_queue = 'test-worker-task-queue'
    worker_options = WorkerOptions(task_queue=test_worker_task_queue)
    deploy_config = DeployEnvConfig.for_test()
    worker_env = WorkerEnv('parent-child-test-worker', worker_options,
                           deploy_config)
    temporal_client = worker_env.temporal_client

    worker = await worker_env.build_worker(
        activities=[DataBackendActivities.wait_on_workflow],
        workflows=[DummyChildWorkflow, DummyParentWorkflow],
    )
    async with worker:
        client = await temporal_client.get()
        parent_wf_handle = await client.start_workflow(
            DummyParentWorkflow.run,
            id='dummy_parent_wf_id',
            args=[deploy_config],
            task_queue=test_worker_task_queue,
            execution_timeout=timedelta(minutes=1),
        )

        await wait_parent_stage(temporal_client, parent_wf_handle,
                                TableRefreshStage.INGEST_TABLE)

        await cancel_child(temporal_client, 0)
        # Because the child is real/direct child of the parent, the parent
        # is expected to FAIL without attempting to re-start the child.
        with pytest.raises(WorkflowFailureError) as ex:
            await parent_wf_handle.result()
        assert isinstance(ex.value.cause, ChildWorkflowError)
        assert isinstance(ex.value.cause.cause, CancelledError)


@pytest.mark.asyncio
async def test_cancellation_multi_parents():
    test_worker_task_queue = 'test-worker-task-queue'
    worker_options = WorkerOptions(task_queue=test_worker_task_queue)
    deploy_config = DeployEnvConfig.for_test()
    worker_env = WorkerEnv('parent-child-test-worker', worker_options,
                           deploy_config)
    temporal_client = worker_env.temporal_client

    worker = await worker_env.build_worker(
        activities=[DataBackendActivities.wait_on_workflow],
        workflows=[DummyChildWorkflow, DummyParentWorkflow],
    )
    async with worker:
        client = await temporal_client.get()
        parent_wf_handle = await client.start_workflow(
            DummyParentWorkflow.run,
            id='dummy_parent_wf_id',
            args=[deploy_config],
            task_queue=test_worker_task_queue,
            execution_timeout=timedelta(minutes=1),
        )

        await wait_parent_stage(temporal_client, parent_wf_handle,
                                TableRefreshStage.INGEST_TABLE)

        # Now we start another parent workflow
        parent_wf_handle_2 = await client.start_workflow(
            DummyParentWorkflow.run,
            id='dummy_parent_wf_id_2',
            args=[deploy_config],
            task_queue=test_worker_task_queue,
            execution_timeout=timedelta(minutes=1),
        )
        # Expect 2nd parent to receive notification of INGEST_TABLE immediately
        await wait_parent_stage(temporal_client, parent_wf_handle_2,
                                TableRefreshStage.INGEST_TABLE)

        # Parent wf #2 should still be running.
        parent2_desc = await parent_wf_handle_2.describe()
        assert parent2_desc.status == WorkflowExecutionStatus.RUNNING

        # Some time later (soon), parent #2 will see the canceled child wfs and
        # restart them.  Eventually parent #2 should complete successfully.
        await wait_parent_stage(temporal_client, parent_wf_handle_2,
                                TableRefreshStage.COMPLETED)
        result = await parent_wf_handle_2.result()
        assert result == 'Done!'


@pytest.mark.asyncio
async def test_wait_completion_raises_public_exception():
    test_worker_task_queue = 'test-worker-task-queue'
    worker_options = WorkerOptions(task_queue=test_worker_task_queue)
    deploy_config = DeployEnvConfig.for_test()
    worker_env = WorkerEnv('parent-child-test-worker', worker_options,
                           deploy_config)
    temporal_client = worker_env.temporal_client

    worker = await worker_env.build_worker(
        activities=[DataBackendActivities.wait_on_workflow],
        workflows=[DummyFailingChildWorkflow, DummyFailingParentWorkflow],
    )
    async with worker:
        client = await temporal_client.get()
        parent_wf_handle = await client.start_workflow(
            DummyFailingParentWorkflow.run,
            id='dummy_parent_wf_id_failure',
            args=[deploy_config],
            task_queue=test_worker_task_queue,
            execution_timeout=timedelta(minutes=1),
        )

        with pytest.raises(WorkflowFailureError) as ex:
            await parent_wf_handle.result()
        public_ex = extract_public_exception(ex.value)
        assert public_ex is not None
        msg = public_ex.get_user_message()
        # Verify the error message contains meaningful information about the failure
        assert 'Column stats computation failed' in msg
        assert 'Activity execution' not in msg

        # Now cancel the first parent, as result, all child wfs will be
        # canceled.
        await parent_wf_handle.cancel()
        with pytest.raises(WorkflowFailureError):
            # Wait until parent #1 status changes to failed/canceled.
            await parent_wf_handle.result()


async def cancel_child(temporal_client: TemporalClient, child_index):
    child_wf_id = f'test-child-table-refresh-wf-{child_index}'
    child_wf_hdl = await temporal_client.get_workflow_handle(child_wf_id)
    await child_wf_hdl.cancel()


async def wait_parent_stage(
        temporal_client: TemporalClient,
        parent_wf_handle: WorkflowHandle,
        stage: TableRefreshStage,
        *,
        poll_interval: float = 1.0,
        timeout=timedelta(seconds=30.0),
):
    start_time = datetime.now()
    while True:
        if datetime.now() - start_time > timeout:
            raise TimeoutError()
        stage_completed = await parent_wf_handle.query(
            DummyParentWorkflow.stage_completed, stage)
        if stage_completed:
            return
        await _tell_child_to_complete(temporal_client, stage)
        await asyncio.sleep(poll_interval)


async def _signal_child_wf(child_wf_handle: WorkflowHandle,
                           stage: TableRefreshStage):
    while True:
        try:
            return await child_wf_handle.signal(
                DummyChildWorkflow.complete_stage, stage)
        except RPCError as rpc_err:
            if rpc_err.status == RPCStatusCode.NOT_FOUND:
                await asyncio.sleep(1.0)
                continue
            raise


async def _tell_child_to_complete(temporal_client: TemporalClient,
                                  stage: TableRefreshStage):
    for i in range(NUM_CHILD_WFS):
        child_wf_id = f'test-child-table-refresh-wf-{i}'
        child_wf_hdl = await temporal_client.get_workflow_handle(child_wf_id)
        await _signal_child_wf(child_wf_hdl, stage)
