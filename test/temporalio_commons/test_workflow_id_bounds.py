# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from service_framework.deploy_env_config import (
    MAX_WORKFLOW_ID_LENGTH,
    bounded_workflow_id,
)
from temporalio_commons.temporal_client import ID_MAX_LENGTH_MEMO_FIELD
from temporalio_commons.worker_interceptors import (
    KumoWorkflowOutboundInterceptor,
)

LIMIT_MEMO = {
    ID_MAX_LENGTH_MEMO_FIELD: SimpleNamespace(data=str(MAX_WORKFLOW_ID_LENGTH).encode())
}


async def _start_child(
    child_id: str, id_suffix: str | None, raw_memo=LIMIT_MEMO
) -> str:
    next_interceptor = SimpleNamespace(start_child_workflow=AsyncMock())
    interceptor = KumoWorkflowOutboundInterceptor(next_interceptor)
    interceptor.info = MagicMock(
        return_value=SimpleNamespace(
            search_attributes=None,
            task_timeout=timedelta(seconds=10),
            raw_memo=raw_memo,
        )
    )
    interceptor._get_id_suffix = MagicMock(return_value=id_suffix)
    child_input = SimpleNamespace(
        args=[],
        id=child_id,
        workflow='TableRefreshWorkflow',
        task_queue='task-queue',
        search_attributes=None,
        memo=None,
        task_timeout=None,
    )
    await interceptor.start_child_workflow(child_input)
    return child_input.id


@pytest.mark.asyncio
@pytest.mark.parametrize('id_suffix', ['-acme', None])
async def test_long_child_workflow_id_is_bounded(id_suffix) -> None:
    long_id = 'table-ingest-' + 'x' * 300
    started = await _start_child(long_id, id_suffix)

    suffix = id_suffix or ''
    assert len(started) == MAX_WORKFLOW_ID_LENGTH
    assert started == bounded_workflow_id(long_id, suffix) + suffix
    assert await _start_child(started, id_suffix) == started


@pytest.mark.asyncio
async def test_short_child_workflow_id_only_gains_suffix() -> None:
    assert await _start_child('child-wf', '-acme') == 'child-wf-acme'
    assert await _start_child('child-wf-acme', '-acme') == 'child-wf-acme'


@pytest.mark.asyncio
async def test_child_without_a_limit_keeps_long_workflow_id() -> None:
    long_id = 'table-ingest-' + 'x' * 300
    assert await _start_child(long_id, '-acme', raw_memo={}) == (long_id + '-acme')
