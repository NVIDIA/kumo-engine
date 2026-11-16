# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections import defaultdict
from datetime import timedelta
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
)

from temporalio import workflow
from temporalio.workflow import (
    ChildWorkflowCancellationType,
    ParentClosePolicy,
)

from io_utils import kumo_dataclass, logging
from kumo.data.backend_cache import TableDataTaskResult, TableRefreshStage
from kumo.validation.validation import ValidationResponse
from rest.backends import DATA_BACKEND_REFRESH_TIMEOUT_HOURS
from rest.backends.data_backend.context import DataBackendContext
from rest.constants import (
    RECEIVE_REFRESH_STATUS_UPDATE_SIGNAL,
    REGISTER_WATCHER_SIGNAL,
)
from statedb.config.common import TableID
from statedb.utils.errors import DataRefreshError
from temporalio_commons.temporal_client import ID_SUFFIX_MEMO_FIELD

from .workflow_utils import signal_workflow_if_running, start_child_data_wf

log = logging.getLogger(__name__)

WorkflowID = str


@kumo_dataclass(frozen=True, eq=True)
class WorkflowExecutionId:
    wf_id: WorkflowID
    run_id: str


class TableChildWorkflowMixin:
    def __init__(self) -> None:
        super(TableChildWorkflowMixin, self).__init__()
        self._watcher_lock = asyncio.Lock()
        # The following two are protected by above lock.
        self._watcher_wfs: Set[WorkflowExecutionId] = set()
        self._status_updates: List[TableDataTaskResult] = []
        self._wf_id = workflow.info().workflow_id

    @workflow.signal(name=REGISTER_WATCHER_SIGNAL)
    async def register_watcher(self, wf_id: str, run_id: str):
        """Allow an external workflow execution to register itself as "watcher"
        or "listener" to receive notification (via signal) about partial
        completion events (aka status updates), during execution of this
        TableRefreshWorkflow.

        The "watcher" is an external workflow execution identified by wf_id
        and run_id, see https://docs.temporal.io/workflows#workflow-execution

        "Callback" is expected to be workflow signal named as
        RECEIVE_REFRESH_STATUS_UPDATE_SIGNAL, with two arguments: this table
        refresh workflow's id and TableDataTaskResult.
        """
        watcher_id = WorkflowExecutionId(wf_id, run_id)
        async with self._watcher_lock:
            if watcher_id not in self._watcher_wfs:
                log.info('Register %s to watch on %s', watcher_id, self._wf_id)
                self._watcher_wfs.add(watcher_id)
                if self._status_updates:
                    # Send out historical status updates to the new watcher.
                    # py3.14 forbids passing bare coroutines to asyncio.wait;
                    # wrap each in a task.
                    await asyncio.wait([
                        asyncio.ensure_future(signal_workflow_if_running(
                            wf_id,
                            run_id,
                            RECEIVE_REFRESH_STATUS_UPDATE_SIGNAL,
                            args=[self._wf_id, status_update],
                        )) for status_update in self._status_updates
                    ])
            else:
                log.warning('%s is already watching %s', watcher_id,
                            self._wf_id)

    async def notify_watcher(self, status_update: TableDataTaskResult):
        async with self._watcher_lock:
            log.info('Notify parents about %s: %s', status_update,
                     self._watcher_wfs)
            self._status_updates.append(status_update)
            if self._watcher_wfs:
                # py3.14 forbids passing bare coroutines to asyncio.wait;
                # wrap each in a task.
                await asyncio.wait([
                    asyncio.ensure_future(signal_workflow_if_running(
                        watcher.wf_id,
                        watcher.run_id,
                        signal_name=RECEIVE_REFRESH_STATUS_UPDATE_SIGNAL,
                        args=[self._wf_id, status_update],
                    )) for watcher in self._watcher_wfs
                ])


class TableParentWorkflowMixin:
    def __init__(self) -> None:
        super(TableParentWorkflowMixin, self).__init__()
        self._child_wf_ids: Dict[TableID, WorkflowID] = {}
        self._child_status_updates: Dict[
            WorkflowID, List[TableDataTaskResult]] = defaultdict(list)
        self._child_wf_tasks: Dict[TableID,
                                   asyncio.Task[TableDataTaskResult]] = {}

    async def _custom_process_status_update(
        self,
        child_wf_id: WorkflowID,
        status_update: TableDataTaskResult,
    ) -> None:
        """
        Hook for subclass to implement any custom processing upon receiving
        status update from child.
        """

    @workflow.signal(name=RECEIVE_REFRESH_STATUS_UPDATE_SIGNAL)
    async def receive_refresh_status_update(self, child_wf_id: WorkflowID,
                                            status_update: TableDataTaskResult):
        child_wf_id = _strip_wf_id_suffix(child_wf_id)
        log.info('Parent wf received update for %s: %s', child_wf_id,
                 status_update)
        self._child_status_updates[child_wf_id].append(status_update)
        await self._custom_process_status_update(child_wf_id, status_update)

    async def start_child_refresh(
        self,
        workflow_fn: Callable[..., Awaitable[TableDataTaskResult]],
        ctx: DataBackendContext,
        *,
        child_id: TableID,
        child_wf_id: WorkflowID,
        args: Sequence[Any] = [],
        execution_timeout=timedelta(hours=DATA_BACKEND_REFRESH_TIMEOUT_HOURS),
        cancellation_type:
        ChildWorkflowCancellationType = ChildWorkflowCancellationType.
        TRY_CANCEL,
        parent_close_policy: ParentClosePolicy = ParentClosePolicy.
        REQUEST_CANCEL,
    ) -> asyncio.Task[TableDataTaskResult]:
        """
        Start a child table or view refresh workflow if it's not running,
        otherwise register as watcher and create a task to wait for the child
        refresh workflow to complete. In the latter case, if the child refresh
        workflow gets canceled (most likely by its real parent), this will
        attempt to re-start the child again (recursively).
        """

        if parent_close_policy == ParentClosePolicy.TERMINATE:
            log.warning('Parent must NOT TERMINATE child refresh upon close.')
            parent_close_policy = ParentClosePolicy.REQUEST_CANCEL

        assert child_id not in self._child_wf_ids, (
            f'start_child_refresh called twice for {child_id}')
        self._child_wf_ids[child_id] = child_wf_id

        child_wf_task = await start_child_data_wf(
            workflow_fn,
            ctx,
            wf_id=child_wf_id,
            args=args,
            execution_timeout=execution_timeout,
            parent_close_policy=parent_close_policy,
            cancellation_type=cancellation_type,
            signal_name=REGISTER_WATCHER_SIGNAL,
            signal_args=[
                workflow.info().workflow_id,
                workflow.info().run_id,
            ],
        )

        self._child_wf_tasks[child_id] = child_wf_task
        return child_wf_task

    async def wait_completion_at_stage(
            self, stage: TableRefreshStage) -> ValidationResponse:
        log.info(f"Waiting for the following table refresh workflows to "
                 f"complete {stage}:\n{self._child_wf_ids.values()}")
        table_results: List[TableDataTaskResult] = await asyncio.gather(*[
            self._wait_for_single_table_refresh_stage(table_id, stage)
            for table_id in self._child_wf_ids
        ])

        validation = ValidationResponse()
        critical_errors = []
        for _, task_result in zip(self._child_wf_ids.keys(), table_results):
            validation.merge(task_result.validation)
            if not task_result.ok:
                error_message = (task_result.error
                                 or task_result.validation.error_message()
                                 or 'Unknown Error.')
                critical_errors.append(error_message)

        if critical_errors:
            # Workflow must raise temporal failure to cause execution to fail
            # see: https://docs.temporal.io/kb/failures#throw-from-workflows
            if len(critical_errors) == 1:
                message = critical_errors[0]
            else:
                items = '\n'.join('  - ' + e.replace('\n', '\n    ')
                                  for e in critical_errors)
                message = (f"Kumo encountered errors in {len(critical_errors)} "
                           f"tables:\n{items}\n"
                           f"Please check your data sources and try again, "
                           f"or reach out to Kumo support for assistance.")
            raise DataRefreshError(message=message)

        log.info('All child refresh workflows have completed %s: %s', stage,
                 validation)
        return validation

    async def _wait_for_single_table_refresh_stage(
            self, child_id: TableID,
            stage: TableRefreshStage) -> TableDataTaskResult:
        assert child_id in self._child_wf_tasks

        # Check if the we have received status update from child about the
        # completion of the specific stage.
        def _table_stage_result() -> Optional[TableDataTaskResult]:
            child_wf_id = self._child_wf_ids[child_id]
            status_updates = self._child_status_updates[child_wf_id]
            for status_update in status_updates:
                if (status_update.stage == stage
                        and status_update.status.completed):
                    return status_update
            return None

        # DIAGNOSTIC INSTRUMENTATION (PR #28405 hang investigation).
        child_wf_id = self._child_wf_ids[child_id]
        log.info(
            "WAIT_BEGIN child_id=%s wf_id=%s stage=%s task_done=%s "
            "known_stages=%s",
            child_id,
            child_wf_id,
            stage,
            self._child_wf_tasks[child_id].done(),
            [
                s.stage.value
                for s in self._child_status_updates.get(child_wf_id, [])
            ],
        )

        # Wait for either completion of the specific stage, or completion of
        # the entire child workflow (could be early failure or error).
        # py3.14 forbids passing bare coroutines to asyncio.wait; wrap the
        # wait_condition coroutine in a task (the child task element is
        # already a Task).
        await asyncio.wait([
            self._child_wf_tasks[child_id],
            asyncio.ensure_future(
                workflow.wait_condition(
                    lambda: _table_stage_result() is not None)),
        ], return_when=asyncio.FIRST_COMPLETED)

        log.info(
            "WAIT_END child_id=%s wf_id=%s stage=%s task_done=%s "
            "stage_result_known=%s",
            child_id,
            child_wf_id,
            stage,
            self._child_wf_tasks[child_id].done(),
            _table_stage_result() is not None,
        )

        return _table_stage_result() or self._child_wf_tasks[child_id].result()


def _strip_wf_id_suffix(wf_id: str) -> str:
    id_suffix = workflow.memo_value(ID_SUFFIX_MEMO_FIELD,
                                    type_hint=bytes).decode()
    if wf_id.endswith(id_suffix):
        return wf_id[:-len(id_suffix)]
    return wf_id
