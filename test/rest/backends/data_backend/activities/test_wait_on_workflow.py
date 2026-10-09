# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock

import pytest

from rest.backends.data_backend.activities import DataBackendActivities


@pytest.mark.asyncio
async def test_wait_on_workflow_uses_exact_run_id() -> None:
    ctx = Mock()
    temporal_client = Mock()
    temporal_client.result = AsyncMock(return_value='done')
    ctx.temporal_client.return_value = temporal_client

    result = await DataBackendActivities.wait_on_workflow(
        ctx, 'child-workflow', 'completed-run')

    assert result == 'done'
    temporal_client.result.assert_awaited_once_with('child-workflow',
                                                    run_id='completed-run')
