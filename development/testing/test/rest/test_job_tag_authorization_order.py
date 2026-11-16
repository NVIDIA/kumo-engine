# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from rest.public.routers.gen_pred_table import \
    GeneratePredictionTableRouter as PublicGeneratePredictionTableRouter
from rest.routers.gen_pred_table import \
    GeneratePredictionTableRouter as V0GeneratePredictionTableRouter
from rest.routers.gen_train_table import GenerateTrainTableRouter
from statedb.config.common import (
    GeneratePredictionTableJobID,
    GenerateTrainTableJobID,
    ProjectID,
)

PROJECT_A = ProjectID("project_" + "a" * 32)
PROJECT_B = ProjectID("project_" + "b" * 32)
_ROUTE_CASES = (
    ('public-gen-pred', PublicGeneratePredictionTableRouter,
     GeneratePredictionTableJobID, 'get_generate_pred_table_jobs', True),
    ('v0-gen-pred', V0GeneratePredictionTableRouter,
     GeneratePredictionTableJobID, 'get_generate_pred_table_jobs', False),
    ('v0-gen-train', GenerateTrainTableRouter, GenerateTrainTableJobID,
     'get_generate_train_table_jobs', False),
)


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(user_authorization=SimpleNamespace(
        is_admin=False,
        project_ids=(str(PROJECT_A), ),
        external_membership_unavailable=False,
    ))


@pytest.mark.asyncio
@pytest.mark.parametrize("row_state", ["missing", "foreign"])
@pytest.mark.parametrize(
    ('router_type', 'job_id_type', 'backend_method', 'is_public'),
    [pytest.param(*case[1:], id=case[0]) for case in _ROUTE_CASES],
)
async def test_job_routes_authorize_row_before_loading_tags(
    router_type,
    job_id_type,
    backend_method,
    is_public,
    row_state: str,
) -> None:
    foreign_rows = ([SimpleNamespace(
        project_id=PROJECT_B)] if row_state == "foreign" else [])
    tag_lookup = AsyncMock(
        side_effect=AssertionError("tag lookup preceded row authorization"))
    backend = SimpleNamespace()
    driver = SimpleNamespace(get_job_tags=tag_lookup)
    job_id = job_id_type.make_random_id()
    setattr(backend, backend_method, AsyncMock(return_value=foreign_rows))
    router = object.__new__(router_type)
    router._ml_job_backend = backend
    if is_public:
        router._driver = driver
        call = router.get(job_id, _ctx())
    else:
        backend.get_job_tags = tag_lookup
        call = router._get_visible_job(job_id, project_id=str(PROJECT_A),
                                       ctx=_ctx())

    with pytest.raises(HTTPException) as exc_info:
        await call
    assert exc_info.value.status_code == 404
    tag_lookup.assert_not_awaited()
