# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from kumo_core.diskgraph.runner import \
    DiskgraphCancelledError as CoreDiskgraphCancelledError
from kumo_core.diskgraph.runner import \
    DiskgraphDataplaneNoSpaceError as CoreDiskgraphDataplaneNoSpaceError
from kumo_core.diskgraph.runner import \
    DiskgraphNoSpaceError as CoreDiskgraphNoSpaceError
from kumo_core.diskgraph.runner import DiskgraphRetryExhaustedError
from kumo_core.diskgraph.runner import \
    NonRetryableDiskgraphError as CoreNonRetryableDiskgraphError
from temporalio import activity
from temporalio.common import Priority
from temporalio.exceptions import ApplicationError
from temporalio.worker._interceptor import ExecuteActivityInput

from common_utils.error import PublicException
from kumo.diskgraph.errors import (
    DiskgraphCancelledError,
    DiskgraphDataplaneNoSpaceError,
    DiskgraphNoSpaceError,
    NonRetryableDiskgraphError,
    as_platform_diskgraph_exception,
)
from temporalio_commons import worker_interceptors


@pytest.mark.parametrize(
    ('core_error', 'platform_type', 'non_retryable'),
    (
        (
            CoreDiskgraphCancelledError('cancel detail'),
            DiskgraphCancelledError,
            False,
        ),
        (
            CoreDiskgraphDataplaneNoSpaceError('dataplane detail'),
            DiskgraphDataplaneNoSpaceError,
            True,
        ),
        (
            CoreDiskgraphNoSpaceError('local SSD detail'),
            DiskgraphNoSpaceError,
            False,
        ),
        (
            CoreNonRetryableDiskgraphError('schema detail'),
            NonRetryableDiskgraphError,
            True,
        ),
        (
            DiskgraphRetryExhaustedError('retry detail'),
            NonRetryableDiskgraphError,
            True,
        ),
    ),
)
def test_platform_diskgraph_error_preserves_contract(
    core_error,
    platform_type,
    non_retryable,
):
    converted = as_platform_diskgraph_exception(core_error)

    assert type(converted) is platform_type
    assert isinstance(converted, PublicException)
    assert converted.get_user_message() == str(core_error)
    assert converted.non_retryable is non_retryable


def test_platform_diskgraph_error_keeps_private_diagnostics_out_of_message():
    core_error = CoreNonRetryableDiskgraphError(
        'safe public detail',
        private_output_tail='provider-secret-canary',
    )

    converted = as_platform_diskgraph_exception(core_error)

    assert converted.private_output_tail == 'provider-secret-canary'
    assert converted.output_fingerprint == core_error.output_fingerprint
    assert 'provider-secret-canary' not in converted.get_user_message()
    assert 'provider-secret-canary' not in str(converted.to_dict())


def _activity_info() -> activity.Info:
    now = datetime.now(timezone.utc)
    return activity.Info(
        activity_id='diskgraph-cancel-contract',
        activity_type='ensure_snapshot',
        attempt=1,
        current_attempt_scheduled_time=now,
        heartbeat_details=(),
        heartbeat_timeout=None,
        is_local=False,
        namespace='namespace',
        schedule_to_close_timeout=None,
        scheduled_time=now,
        start_to_close_timeout=None,
        started_time=now,
        task_queue='pool-worker',
        task_token=b'token',
        workflow_id='dg-ingest:test',
        workflow_namespace='namespace',
        workflow_run_id='run-id',
        workflow_type='DiskgraphIngestWorkflow',
        priority=Priority(),
        retry_policy=None,
    )


@pytest.mark.asyncio
async def test_cancelled_error_crosses_activity_boundary_as_retryable(
        monkeypatch):
    cancelled = as_platform_diskgraph_exception(
        CoreDiskgraphCancelledError(
            'cancel detail',
            private_output_tail='provider-secret-cancel-canary',
        ))
    next_interceptor = SimpleNamespace(
        execute_activity=AsyncMock(side_effect=cancelled))
    interceptor = worker_interceptors.KumoActivityInboundInterceptor(
        next_interceptor)
    interceptor._outbound = SimpleNamespace(track_last_heartbeat=False)
    monkeypatch.setattr(
        'temporalio_commons.worker_interceptors.activity.info',
        _activity_info,
    )
    monkeypatch.setattr(
        'temporalio_commons.worker_interceptors._kumo_record_event',
        lambda *_args, **_kwargs: None,
    )
    activity_input = ExecuteActivityInput(
        fn=lambda: None,
        args=(),
        executor=None,
        headers={},
    )

    with pytest.raises(ApplicationError) as caught:
        await interceptor.execute_activity(activity_input)

    assert caught.value.type == 'DiskgraphCancelledError'
    assert caught.value.non_retryable is False
    assert 'provider-secret-cancel-canary' not in str(caught.value)
    assert 'provider-secret-cancel-canary' not in str(caught.value.details)
