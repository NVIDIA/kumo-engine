# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``start_child_data_wf`` joins a data-backend child that another job already
started. The join runs through an activity, so it must follow the child onto
the data-backend queue rather than inherit the caller's (kumo-ai/kumo#30363).
"""
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from temporalio.exceptions import WorkflowAlreadyStartedError

from rest.backends.data_backend.workflows import workflow_utils as mod
from rest.constants import DATA_BACKEND_TASK_QUEUE


async def _child_run(*_args: Any, **_kwargs: Any) -> Any:
    return None


@pytest.mark.parametrize('task_queue', [DATA_BACKEND_TASK_QUEUE, None])
@pytest.mark.asyncio
async def test_join_activity_follows_the_child_task_queue(
        monkeypatch, task_queue):
    """A caller off the data-backend queue (an ML workflow) must not strand
    the wait activity on a queue where it is not registered."""
    activity_kwargs: dict = {}

    async def _already_started(*_a: Any, **_k: Any) -> Any:
        raise WorkflowAlreadyStartedError('wf-1', 'GraphRefreshWorkflow',
                                          run_id='run-1')

    def _start_activity(_fn: Any, **kwargs: Any) -> Any:
        activity_kwargs.update(kwargs)

        async def _result() -> Any:
            return 'joined'

        return _result()

    monkeypatch.setattr(mod.workflow, 'start_child_workflow',
                        _already_started)
    monkeypatch.setattr(mod.workflow, 'start_activity', _start_activity)
    monkeypatch.setattr(mod.workflow, 'info',
                        lambda: SimpleNamespace(workflow_id='caller'))

    child_task = await mod.start_child_data_wf(
        _child_run,
        Any,  # type: ignore[arg-type]
        wf_id='wf-1',
        args=[],
        execution_timeout=timedelta(hours=1),
        task_queue=task_queue,
    )

    assert await child_task == 'joined'
    assert activity_kwargs['task_queue'] == task_queue
