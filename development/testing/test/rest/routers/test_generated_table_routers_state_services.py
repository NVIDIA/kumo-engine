# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rest.routers.gen_pred_table import GeneratePredictionTableRouter
from rest.routers.gen_train_table import GenerateTrainTableRouter
from statedb.config.common import (
    GeneratePredictionTableJobID,
    GenerateTrainTableJobID,
    JobType,
    ProjectID,
)
from statedb.config.job_tag import JobTagKey, JobTags, JobTagValue
from statedb.config.timestamp import get_current_timestamps

_PROJECT_ID = 'project_' + '1' * 32


def _auth() -> SimpleNamespace:
    return SimpleNamespace(authenticate_request_header=lambda: None, )


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(user_authorization=SimpleNamespace(
        is_admin=False,
        project_ids=(_PROJECT_ID, ),
        external_membership_unavailable=False,
    ))


def _tags() -> JobTags:
    return JobTags(
        timestamps=get_current_timestamps(),
        tags={JobTagKey("phase"): JobTagValue("one")},
    )


@pytest.mark.asyncio
async def test_generate_train_table_router_reads_tags_from_ml_backend():
    job_id = GenerateTrainTableJobID.make_random_id()
    resource = SimpleNamespace(tags=None, project_id=_PROJECT_ID)
    tags = _tags()
    driver = SimpleNamespace(
        get_job_tags=AsyncMock(
            side_effect=AssertionError("legacy driver tags path called")), )
    ml_job_backend = SimpleNamespace(
        get_generate_train_table_jobs=AsyncMock(return_value=[resource]),
        get_job_tags=AsyncMock(return_value=tags),
    )
    router = GenerateTrainTableRouter(_auth(), driver, ml_job_backend)

    actual = await router._get_visible_job(
        job_id,
        project_id=_PROJECT_ID,
        ctx=_ctx(),
    )

    assert actual is resource
    assert actual.tags == tags.tags
    ml_job_backend.get_generate_train_table_jobs.assert_awaited_once_with(
        {job_id}, project_ids={ProjectID(_PROJECT_ID)})
    ml_job_backend.get_job_tags.assert_awaited_once_with(
        JobType.GENERATE_TRAIN_TABLE_JOB, job_id)
    driver.get_job_tags.assert_not_called()


@pytest.mark.asyncio
async def test_generate_prediction_table_router_reads_tags_from_ml_backend():
    job_id = GeneratePredictionTableJobID.make_random_id()
    resource = SimpleNamespace(tags=None, project_id=_PROJECT_ID)
    tags = _tags()
    driver = SimpleNamespace(
        get_job_tags=AsyncMock(
            side_effect=AssertionError("legacy driver tags path called")), )
    ml_job_backend = SimpleNamespace(
        get_generate_pred_table_jobs=AsyncMock(return_value=[resource]),
        get_job_tags=AsyncMock(return_value=tags),
    )
    router = GeneratePredictionTableRouter(_auth(), driver, ml_job_backend)

    actual = await router._get_visible_job(
        job_id,
        project_id=_PROJECT_ID,
        ctx=_ctx(),
    )

    assert actual is resource
    assert actual.tags == tags.tags
    ml_job_backend.get_generate_pred_table_jobs.assert_awaited_once_with(
        {job_id}, project_ids={ProjectID(_PROJECT_ID)})
    ml_job_backend.get_job_tags.assert_awaited_once_with(
        JobType.GENERATE_PREDICTION_TABLE_JOB, job_id)
    driver.get_job_tags.assert_not_called()
