# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rest.public.routers.gen_pred_table import GeneratePredictionTableRouter
from statedb.config.common import GeneratePredictionTableJobID, JobType


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    return False


@pytest.fixture
def _rest_install_dg_col_stats_fallback():
    yield


@pytest.fixture(scope='session')
def setup_rest_test_session():
    yield


@pytest.fixture
def auto_cleanup_per_func():
    yield


@pytest.fixture(scope='session')
def force_local_cache_dir_for_public_tests():
    yield


def _auth() -> SimpleNamespace:
    return SimpleNamespace(authenticate_request_header=lambda: None, )


def _admin_ctx() -> SimpleNamespace:
    return SimpleNamespace(user_authorization=SimpleNamespace(
        is_admin=True,
        project_ids=(),
        external_membership_unavailable=False,
    ))


@pytest.mark.asyncio
async def test_get_generate_pred_table_job_allows_missing_tags():
    """An existing job whose tags row is missing returns ``tags == {}``.

    On the pre-fix handler the tag fetch went through the raising
    ``driver.get_job_tags``, which raises
    ``KeyError(f"{job_type} id={job_id} is not found in DB")`` when the tags
    row is absent, escaping the route unhandled as a 500. The driver mock
    below reproduces exactly that production behavior, so this test fails
    against the old handler and passes once the route uses the non-raising
    ``ml_job_backend.get_job_tags_if_exists`` with an empty-dict fallback,
    mirroring the gentraintable sibling.
    """
    job_id = GeneratePredictionTableJobID.make_random_id()
    resource = SimpleNamespace(tags=None, project_id='project_' + '1' * 32)
    driver = SimpleNamespace(
        get_job_tags=AsyncMock(side_effect=KeyError(
            f"{JobType.GENERATE_PREDICTION_TABLE_JOB} id={job_id} "
            "is not found in DB")), )
    ml_job_backend = SimpleNamespace(
        get_generate_pred_table_jobs=AsyncMock(return_value=[resource]),
        get_job_tags_if_exists=AsyncMock(return_value=None),
    )
    router = GeneratePredictionTableRouter(lambda: None, driver, _auth(),
                                           ml_job_backend)

    actual = await router.get(job_id, _admin_ctx())

    assert actual is resource
    assert actual.tags == {}
    ml_job_backend.get_generate_pred_table_jobs.assert_awaited_once_with(
        {job_id}, project_ids=None)
    ml_job_backend.get_job_tags_if_exists.assert_awaited_once_with(
        JobType.GENERATE_PREDICTION_TABLE_JOB, job_id)
    driver.get_job_tags.assert_not_called()


@pytest.mark.asyncio
async def test_get_generate_pred_table_job_returns_existing_tags():
    """When the tags row exists, its tags land on the returned resource."""
    job_id = GeneratePredictionTableJobID.make_random_id()
    resource = SimpleNamespace(tags=None, project_id='project_' + '1' * 32)
    driver = SimpleNamespace(get_job_tags=AsyncMock(), )
    ml_job_backend = SimpleNamespace(
        get_generate_pred_table_jobs=AsyncMock(return_value=[resource]),
        get_job_tags_if_exists=AsyncMock(return_value=SimpleNamespace(
            tags={'team': 'growth'})),
    )
    router = GeneratePredictionTableRouter(lambda: None, driver, _auth(),
                                           ml_job_backend)

    actual = await router.get(job_id, _admin_ctx())

    assert actual is resource
    assert actual.tags == {'team': 'growth'}
    driver.get_job_tags.assert_not_called()
