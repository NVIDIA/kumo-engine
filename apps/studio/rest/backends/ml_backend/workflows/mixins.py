# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Awaitable, Optional, TypeVar

from temporalio import workflow
from temporalio.exceptions import CancelledError

from io_utils import logging
from temporalio_commons import is_cancelled_exception

log = logging.getLogger(__name__)

T = TypeVar('T')


class JobWorkflowCancellationMixin:
    def __init__(self) -> None:
        super(JobWorkflowCancellationMixin, self).__init__()
        self._cancelled: bool = False
        self._cancel_reason: Optional[str] = None

    @workflow.signal
    async def cancel_wf(self, reason: Optional[str] = None):
        log.warning('Workflow %s receives "cancel_wf" signal, reason=%s',
                    self.id, reason)
        self._cancel_reason = reason or self._cancel_reason
        self._cancelled = True

    @property
    def id(self):
        return workflow.info().workflow_id

    def check_cancelled(self):
        if self._cancelled:
            log.warning('Raising CancelledError for %s', self.id)
            raise CancelledError(self._cancel_reason or 'Workflow cancelled')

    async def run_until_canceled(self, coro: Awaitable[T]) -> T:
        task = asyncio.ensure_future(coro)
        wait_cancel_task = asyncio.ensure_future(
            workflow.wait_condition(lambda: self._cancelled))
        try:
            await workflow.wait([wait_cancel_task, task],
                                return_when='FIRST_COMPLETED')
        except BaseException as ex:
            log.warning('Workflow %s interrupted by %s(type=%s)', self.id, ex,
                        type(ex))
            # Record workflow cancellation state if applicable.
            self._cancelled = self._cancelled or is_cancelled_exception(ex)

        canceled = task.cancel(self._cancel_reason)  # noop if task is done
        log.warning('Main workflow task is canceled: %s', canceled)
        return await task
