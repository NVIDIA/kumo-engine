# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from statedb.config.common import ArtifactExportJobID, ProjectID, UserID
from statedb.db import StateDB
from statedb.db.artifact_exports import ArtifactExportJobService
from statedb.utils.errors import ArtifactExportJobNotFoundError

PROJECT_A = ProjectID('project_' + 'a' * 32)
PROJECT_B = ProjectID('project_' + 'b' * 32)
JOB_ID = ArtifactExportJobID('ae-job-' + 'd' * 32)
USER_ID = UserID('user-test')


@pytest_asyncio.fixture
async def artifact_jobs(tmp_path):
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    connectors = SimpleNamespace(require_project_access=AsyncMock())
    try:
        yield ArtifactExportJobService(db, connectors), connectors
    finally:
        await db.close()


async def _create(service, **overrides):
    args = dict(
        id=JOB_ID,
        project_id=PROJECT_A,
        created_by=USER_ID,
        request_id='00000000-0000-4000-8000-000000000001',
        request_digest='request-digest',
    )
    args.update(overrides)
    return await service.create(**args)


@pytest.mark.asyncio
async def test_request_identity_is_atomic_immutable_and_idempotent(
    artifact_jobs,
) -> None:
    service, _ = artifact_jobs
    first_result, duplicate_result = await asyncio.gather(
        _create(service), _create(service))
    first, first_inserted = first_result
    duplicate, duplicate_inserted = duplicate_result
    assert first == duplicate
    assert {first_inserted, duplicate_inserted} == {False, True}
    assert not first.dispatch_sealed
    assert not first.cancel_requested
    assert not hasattr(first, 'status')
    assert not hasattr(first, 'terminal_at')

    with pytest.raises(ValueError, match='different payload'):
        await _create(service, request_digest='different')

    seal_results = await asyncio.gather(
        service.seal_dispatch(JOB_ID, project_id=PROJECT_A),
        service.seal_dispatch(JOB_ID, project_id=PROJECT_A),
    )
    assert all(entry.dispatch_sealed for entry in seal_results)
    assert not any(entry.cancel_requested for entry in seal_results)
    assert (await _create(service))[0].dispatch_sealed

    cancel_job_id = ArtifactExportJobID('ae-job-' + 'e' * 32)
    await _create(service, id=cancel_job_id)
    await asyncio.gather(
        service.seal_dispatch(cancel_job_id, project_id=PROJECT_A),
        service.request_cancel(cancel_job_id, project_id=PROJECT_A),
    )
    cancelled = await service.request_cancel(
        cancel_job_id, project_id=PROJECT_A)
    assert cancelled.dispatch_sealed
    assert cancelled.cancel_requested
    assert (await service.seal_dispatch(
        cancel_job_id, project_id=PROJECT_A)).cancel_requested


@pytest.mark.asyncio
async def test_get_is_project_authorized_and_project_isolated(
        artifact_jobs) -> None:
    service, connectors = artifact_jobs
    principal = SimpleNamespace(user_id=USER_ID)
    await _create(service)
    loaded = await service.get_for_project(JOB_ID, PROJECT_A, principal)
    assert loaded.project_id == PROJECT_A
    connectors.require_project_access.assert_awaited_with(
        principal, PROJECT_A)

    with pytest.raises(ArtifactExportJobNotFoundError):
        await service.get_for_project(JOB_ID, PROJECT_B, principal)
