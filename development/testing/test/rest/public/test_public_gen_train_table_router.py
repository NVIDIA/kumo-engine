# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rest.public.routers.gen_train_table import GenerateTrainTableRouter
from statedb.config.common import GenerateTrainTableJobID, JobType


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


@pytest.mark.asyncio
async def test_get_generate_train_table_job_allows_missing_tags():
    job_id = GenerateTrainTableJobID.make_random_id()
    resource = SimpleNamespace(tags=None, project_id='project_' + '1' * 32)
    ctx = SimpleNamespace(user_authorization=SimpleNamespace(
        is_admin=True,
        project_ids=(),
        external_membership_unavailable=False,
    ))
    driver = SimpleNamespace(
        get_job_tags=AsyncMock(
            side_effect=AssertionError("legacy driver tags path called")), )
    ml_job_backend = SimpleNamespace(
        get_generate_train_table_jobs=AsyncMock(return_value=[resource]),
        get_job_tags_if_exists=AsyncMock(return_value=None),
    )
    router = GenerateTrainTableRouter(lambda: None, driver, _auth(),
                                      ml_job_backend)

    actual = await router.get(job_id, ctx)

    assert actual is resource
    assert actual.tags == {}
    ml_job_backend.get_generate_train_table_jobs.assert_awaited_once_with(
        {job_id}, project_ids=None)
    ml_job_backend.get_job_tags_if_exists.assert_awaited_once_with(
        JobType.GENERATE_TRAIN_TABLE_JOB, job_id)
    driver.get_job_tags.assert_not_called()
