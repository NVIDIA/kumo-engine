# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi import Request, Response

from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.deps.authentication import RequestContext, UserAuthorization
from rest.driver.pquery_api_driver import PQueryApiDriver
from rest.public.routers import gen_pred_table
from rest.public.routers.gen_pred_table import GeneratePredictionTableRouter
from statedb.config.common import GeneratePredictionTableJobID, ProjectID

PROJECT_A = ProjectID("project_" + "a" * 32)


@pytest.mark.asyncio
@pytest.mark.parametrize(("is_admin", "project_ids", "expected_scope"), [
    (False, (PROJECT_A, ), {PROJECT_A}),
    (True, (PROJECT_A, ), None),
    (False, (), set()),
])
async def test_public_gen_pred_list_scopes_before_backend_limit(
        monkeypatch, is_admin, project_ids, expected_scope):
    job_id = GeneratePredictionTableJobID.make_random_id()
    backend = SimpleNamespace(
        get_generate_pred_table_jobs=AsyncMock(return_value=[]))
    router = object.__new__(GeneratePredictionTableRouter)
    router._ml_job_backend = cast(MLJobBackend, backend)
    router._driver = cast(PQueryApiDriver, SimpleNamespace())
    monkeypatch.setattr(
        gen_pred_table,
        "list_job_tags",
        AsyncMock(return_value={job_id: SimpleNamespace(tags={})}),
    )
    user_authorization = cast(
        UserAuthorization,
        SimpleNamespace(
            is_admin=is_admin,
            project_ids=tuple(str(id) for id in project_ids),
        ),
    )
    ctx = cast(
        RequestContext,
        SimpleNamespace(user_authorization=user_authorization),
    )

    request = cast(Request, SimpleNamespace())
    assert await router.list(request, Response(), limit=1, ctx=ctx) == []
    backend.get_generate_pred_table_jobs.assert_awaited_once_with(
        {job_id},
        project_ids=expected_scope,
        status=None,
        limit=1,
    )
