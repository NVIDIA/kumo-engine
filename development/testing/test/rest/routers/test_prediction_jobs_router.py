# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Tuple, cast
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from rest.config.status import EmptyResponse
from rest.deps.authentication import RequestContext
from rest.driver.pquery_api_driver import PQueryApiDriver
from rest.public.routers.prediction import PredictionRouterV1
from statedb.config.common import BatchPredictionJobID


@pytest.fixture(autouse=True, scope='session')
def setup_rest_test_session():
    yield


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


def _router(deleted: bool) -> Tuple[PredictionRouterV1, SimpleNamespace]:
    router = object.__new__(PredictionRouterV1)
    driver = SimpleNamespace(delete_batch_prediction=AsyncMock(
        return_value=deleted))
    router._driver = cast(PQueryApiDriver, driver)
    return router, driver


@pytest.mark.asyncio
async def test_delete_prediction_job_maps_nonexistent_to_404() -> None:
    """DELETE of a nonexistent batch-prediction job must return 404, matching
    the sibling ``DELETE /training_jobs/{id}`` contract, instead of reporting
    success (HTTP 200) while nothing was deleted.

    Existence is determined by the delete operation's own result, not a
    separate lookup, so a job removed concurrently still yields 404."""
    job_id = BatchPredictionJobID('pred-job-does-not-exist-000')
    router, driver = _router(deleted=False)
    ctx = cast(RequestContext, object())

    with pytest.raises(HTTPException) as exc_info:
        await router.delete_prediction_job(job_id, ctx=ctx)

    assert exc_info.value.status_code == 404
    assert str(job_id) in exc_info.value.detail
    driver.delete_batch_prediction.assert_awaited_once_with(ctx, job_id)


@pytest.mark.asyncio
async def test_delete_prediction_job_deletes_existing_job() -> None:
    job_id = BatchPredictionJobID('pred-job-exists-000')
    router, driver = _router(deleted=True)
    ctx = cast(RequestContext, object())

    response = await router.delete_prediction_job(job_id, ctx=ctx)

    assert isinstance(response, EmptyResponse)
    driver.delete_batch_prediction.assert_awaited_once_with(ctx, job_id)
