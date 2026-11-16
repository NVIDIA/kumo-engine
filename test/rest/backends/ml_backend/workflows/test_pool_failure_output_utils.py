# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from rest.backends.ml_backend.activities.pool_failure_output import (
    read_pool_failure_output,
)
from rest.backends.ml_backend.workflows import utils as module


@pytest.fixture
def setup_rest_test_session() -> None:
    return None


@pytest.mark.asyncio
async def test_read_saved_pool_failure_output_runs_the_read_activity(
        monkeypatch) -> None:
    execute = AsyncMock(return_value='ValueError: bad input')
    monkeypatch.setattr(module.workflow, 'execute_activity', execute)

    output = await module.read_saved_pool_failure_output(
        'trainingjob-1', ['/jobs/trainingjob-1.json.kumo-auto.failure.txt'])

    assert output == 'ValueError: bad input'
    call = execute.await_args
    assert call.args == (read_pool_failure_output, )
    assert call.kwargs['args'] == [
        'trainingjob-1', ['/jobs/trainingjob-1.json.kumo-auto.failure.txt']
    ]
    assert call.kwargs['schedule_to_close_timeout'] == timedelta(minutes=5)
    assert call.kwargs['retry_policy'].maximum_attempts == 3


@pytest.mark.asyncio
async def test_read_saved_pool_failure_output_never_fails_the_workflow(
        monkeypatch) -> None:
    monkeypatch.setattr(module.workflow, 'execute_activity',
                        AsyncMock(side_effect=RuntimeError('activity failed')))
    logger = Mock()
    monkeypatch.setattr(module.workflow, 'logger', logger)

    assert await module.read_saved_pool_failure_output('job-1', ['/x']) is None
    logger.warning.assert_called_once()
