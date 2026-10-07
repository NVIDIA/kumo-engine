# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from datetime import timedelta
from typing import Any, Awaitable, Callable, Optional, Sequence, TypeVar

from temporalio import workflow
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    WorkflowAlreadyStartedError,
)
from temporalio.workflow import (
    ChildWorkflowCancellationType,
    ParentClosePolicy,
)

from io_utils import logging
from rest.backends.data_backend.activities import DataBackendActivities
from rest.backends.data_backend.context import DataBackendContext
from rest.constants import (
    CHILD_WF_WAITING_RETRY_POLICY,
    HEARTBEAT_TIMEOUT_ASYNC_ACTIVITY,
)
from temporalio_commons import is_cancelled_exception

log = logging.getLogger(__name__)
"""
All helper utilities in this file are meant to be called from workflow code
during workflow execution.
"""


async def signal_workflow_if_running(
    wf_id: str,
    run_id: str | None,
    signal_name: str,
    *,
    args: Sequence[Any] = [],
) -> bool:
    """
    Signal an external workflow execution if it's still running and return
    True; otherwise log warning message and return False.

    See https://docs.temporal.io/dev-guide/python/features#signals

    For more details on how "signal" works under the hood:
    1. The sender (client or workflow via SDK) sends a gRPC request to Temporal
    Cloud, the request contains wf_id (and optional run_id), signal name,
    signal args.
    2. Temporal Cloud tries to find a running wf execution for the given wf_id
    (and run_id if present); if the wf run exists, Temporal Cloud persists the
    signal request (into the wf task queue of the running wf) and responds OK;
    otherwise the gRPC raises ExternalWorkflowExecutionNotFound error to the
    sender.
    3. The recipient wf execution will poll the next WorkflowTask from wf task
    queue, and starts to execute the signal handler (based on signal name).
    Note that, if the requested signal name doesn't match a @workflow.signal
    handler, it's a noop, rather than error/exception.
    From the sender's perspective, sending out a signal is asynchronous, and
    the sender doesn't receive any response except the
    ExternalWorkflowExecutionNotFound when the workflow execution is not found
    for the wf_id + run_id (either because there is no such wf id, or the wf
    execution has already finished)

    Args:
        wf_id (str): workflow id of the wf execution to receive signal;
        run_id (str | None): run id of the wf execution to receive signal;

        signal_name (str): name of the signal function decorated by
        @workflow.signal in the recipient workflow;

        args (Sequence[Any], optional): input args to the signal function.
        Defaults to [].

    Returns:
        bool: True if workflow execution is found and signal request is
        successfully sent (recorded by Temporal Cluster).
    """
    wf_handle = workflow.get_external_workflow_handle(wf_id, run_id=run_id)
    try:
        await wf_handle.signal(signal=signal_name, args=args)
        log.info('Sent signal %s to target wf %s, run_id=%s', signal_name,
                 wf_id, run_id)
        return True
    except ApplicationError as err:
        if err.type == 'ExternalWorkflowExecutionNotFound':
            log.warning(
                'Failed to signal a workflow that is not running, '
                'wf_id=%s, run_id=%s, signal=%s, err=%s', wf_id, run_id,
                signal_name, err)
            return False
        raise


ChildReturnType = TypeVar('ChildReturnType')


async def start_child_data_wf(
    workflow_fn: Callable[..., Awaitable[ChildReturnType]],
    ctx: DataBackendContext,
    *,
    wf_id: str,
    args: Sequence[Any],
    execution_timeout: timedelta,
    cancellation_type:
    ChildWorkflowCancellationType = ChildWorkflowCancellationType.TRY_CANCEL,
    parent_close_policy: ParentClosePolicy = ParentClosePolicy.REQUEST_CANCEL,
    # Callers on a different queue than the child (e.g. ML workflows starting
    # a data-backend refresh) must set this; None inherits the caller's queue.
    task_queue: Optional[str] = None,
    signal_name: Optional[str] = None,
    signal_args: Sequence[Any] = [],
) -> asyncio.Task[ChildReturnType]:
    """
    Start a child data workflow (which is data-versionized and could be
    depended on by multiple "parent" workflows).  This will kick off a new run
    of child wf if it's not running; otherwise this will execute an activity to
    wait for (poll) the result of already-running child workflow.

    In the latter case, if the "child" workflow ends up getting canceled (by
    its original parent), this function will attempt to restart the child.
    """
    if parent_close_policy == ParentClosePolicy.TERMINATE:
        log.warning('Parent must NOT TERMINATE child refresh upon close.')
        parent_close_policy = ParentClosePolicy.REQUEST_CANCEL

    # Anonymous, recursively defined task wrapper that attempts to restart the
    # child when: 1) the task is execution of the activity "wait_on_workflow";
    # and 2) the activity failed due to the child workflow getting canceled.
    async def _auto_retry_canceled_child_refresh(
            child_task: asyncio.Task[ChildReturnType]) -> ChildReturnType:
        while True:
            try:
                return await child_task
            except BaseException as ex:
                if _child_refresh_canceled(ex):
                    log.warning(
                        'Child refresh workflow %s was canceled but '
                        'still needed by %s, restarting it...', wf_id,
                        workflow.info().workflow_id)
                    # Recursive call to restart the child.
                    child_task = await _create_child_wf_task()
                    continue
                raise

    async def _create_child_wf_task() -> asyncio.Task[ChildReturnType]:
        try:
            # Attempt to spawn the child workflow run. If successful, the
            # caller workflow is regarded as the "primary" or "real" parent,
            # and it won't auto-restart the child when it fails.
            log.info('start_child_data_wf: spawning child %s with args=%s',
                     wf_id, args)
            child_wf_task = await workflow.start_child_workflow(
                workflow_fn,
                args=args,
                id=wf_id,
                cancellation_type=cancellation_type,
                parent_close_policy=parent_close_policy,
                execution_timeout=execution_timeout,
                task_queue=task_queue,
            )
            if signal_name:
                try:
                    await child_wf_task.signal(signal_name, args=signal_args)
                except ApplicationError as err:
                    # The child can close before the signal lands; its
                    # outcome still arrives through child_wf_task.
                    if err.type != 'ExternalWorkflowExecutionNotFound':
                        raise
                    log.info('Child %s closed before signal %s', wf_id,
                             signal_name)
            log.info('Parent %s successfully spawned child %s',
                     workflow.info().workflow_id, wf_id)
            return child_wf_task
        except WorkflowAlreadyStartedError as err:
            # The "child" workflow is already running, likely launched by its
            # "primary" parent. The caller workflow is regarded as the "foster"
            # parent of the child.  Upon cancellation of the child, the
            # "foster" parent would attempt to re-start the child in a
            # recursive way by chaining a task wrapper of
            # _auto_retry_canceled_child_refresh() which recursively calls
            # _create_child_wf_task()
            log.info('Child workflow %s already exists...', wf_id)
            if signal_name:
                await signal_workflow_if_running(err.workflow_id, err.run_id,
                                                 signal_name, args=signal_args)
            child_wf_task = workflow.start_activity(
                DataBackendActivities.wait_on_workflow,
                args=[ctx, wf_id, err.run_id],
                # The activity is registered on the data-backend worker only,
                # so it must follow the child rather than inherit the caller's
                # queue.
                task_queue=task_queue,
                schedule_to_close_timeout=execution_timeout,
                heartbeat_timeout=HEARTBEAT_TIMEOUT_ASYNC_ACTIVITY,
                retry_policy=CHILD_WF_WAITING_RETRY_POLICY,
            )
            return asyncio.ensure_future(
                _auto_retry_canceled_child_refresh(child_wf_task))

    return await _create_child_wf_task()


# Decides if the exception indicates that the child table refresh workflow was
# canceled (by its real parent workflow).
def _child_refresh_canceled(ex: BaseException):
    if isinstance(ex, ActivityError) and isinstance(ex.cause, ApplicationError):
        cause = ex.cause.cause
        while cause:
            if is_cancelled_exception(cause):
                return True
            cause = getattr(cause, 'cause', None) or getattr(
                cause, '__cause__', None)
    return False
