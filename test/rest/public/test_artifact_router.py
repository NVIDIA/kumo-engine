# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest
from kumoapi.common import JobStatus

from rest.public.routers.artifact import ArtifactRouterV1
from statedb.config.common import ProjectID

PROJECT_ID = ProjectID('project_' + 'a' * 32)


@pytest.mark.asyncio
async def test_artifact_status_and_cancel_forward_exact_project() -> None:
    driver = AsyncMock()
    driver.get_artifact_export_job_status.return_value = JobStatus.RUNNING
    driver.cancel_artifact_export_job.return_value = JobStatus.CANCELLED
    router = object.__new__(ArtifactRouterV1)
    router._driver = driver

    status = await router.get_artifact_export_job_status(
        'ae-job-status', str(PROJECT_ID))
    cancelled = await router.cancel_artifact_export_job(
        'ae-job-cancel', str(PROJECT_ID))

    assert status == JobStatus.RUNNING
    assert cancelled == JobStatus.CANCELLED
    driver.get_artifact_export_job_status.assert_awaited_once_with(
        'ae-job-status', PROJECT_ID)
    driver.cancel_artifact_export_job.assert_awaited_once_with(
        'ae-job-cancel', PROJECT_ID)
