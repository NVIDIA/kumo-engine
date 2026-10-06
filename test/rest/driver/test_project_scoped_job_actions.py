# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.driver.pquery_api_driver import PQueryApiDriver
from statedb.config.common import (
    BatchPredictionJobID,
    ProjectID,
    TrainingJobID,
)
from statedb.utils.errors import (
    BatchPredictionNotInProjectError,
    TrainingJobNotInProjectError,
)

PROJECT_A = ProjectID('project_' + 'a' * 32)
PROJECT_B = ProjectID('project_' + 'b' * 32)
TRAINING_ID = TrainingJobID('trainingjob-project-scope')
PREDICTION_ID = BatchPredictionJobID('batchpredictionjob-project-scope')
_ACTION_CASES = (
    ('training-progress', TrainingJobNotInProjectError),
    ('training-cancel', TrainingJobNotInProjectError),
    ('prediction-progress', BatchPredictionNotInProjectError),
    ('prediction-cancel', BatchPredictionNotInProjectError),
)


def _driver(job_project: ProjectID):
    principal = SimpleNamespace(is_admin=True, project_ids=())
    projects = SimpleNamespace(get_project=AsyncMock())

    def get_batch_prediction_rows(ids, *, project_ids=None):
        if project_ids is not None and job_project not in project_ids:
            return []
        return [SimpleNamespace(project_id=job_project)]

    ml_jobs = SimpleNamespace(
        get_training_job=AsyncMock(
            return_value=SimpleNamespace(project_id=job_project)),
        get_batch_prediction_rows=AsyncMock(
            side_effect=get_batch_prediction_rows),
    )
    backend = SimpleNamespace(
        get_job_home_dir=AsyncMock(return_value='/kumo/training'),
        cancel_training_job=AsyncMock(return_value=True),
        cancel_batch_prediction_job=AsyncMock(return_value=True),
    )
    driver = object.__new__(PQueryApiDriver)
    driver._state_services = SimpleNamespace(projects=projects,
                                             ml_jobs=ml_jobs)
    driver._ml_job_backend = cast(MLJobBackend, backend)
    driver._load_training_progress_from_job_path = SimpleNamespace(
        coroutine=AsyncMock(return_value='training-progress'))
    driver._load_prediction_progress = SimpleNamespace(
        coroutine=AsyncMock(return_value='prediction-progress'))
    ctx = SimpleNamespace(user_authorization=principal)
    return driver, ctx, principal, projects, backend


async def _invoke(driver, ctx, action: str):
    if action == 'training-progress':
        return await driver.get_training_progress(ctx, TRAINING_ID, PROJECT_A)
    if action == 'training-cancel':
        return await driver.cancel_training_job(ctx, TRAINING_ID, PROJECT_A)
    if action == 'prediction-progress':
        return await driver.get_batch_prediction_progress(
            ctx, PREDICTION_ID, PROJECT_A)
    if action == 'prediction-cancel':
        return await driver.cancel_batch_prediction_job_by_id(
            ctx, PREDICTION_ID, PROJECT_A)
    raise AssertionError(action)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('job_project', 'allowed'),
    [pytest.param(PROJECT_A, True, id='same-project'),
     pytest.param(PROJECT_B, False, id='foreign-project')],
)
@pytest.mark.parametrize(
    ('action', 'error_type'),
    [pytest.param(action, error_type, id=action)
     for action, error_type in _ACTION_CASES],
)
async def test_project_scoped_job_action_authorization(
    action: str,
    error_type: type[Exception],
    job_project: ProjectID,
    allowed: bool,
) -> None:
    driver, ctx, principal, projects, backend = _driver(job_project)

    if allowed:
        result = await _invoke(driver, ctx, action)
        assert result in {'training-progress', 'prediction-progress', True}
    else:
        with pytest.raises(error_type):
            await _invoke(driver, ctx, action)

    projects.get_project.assert_awaited_once_with(PROJECT_A, principal)
    if action.startswith('prediction-'):
        state_services = driver._state_services
        assert state_services is not None
        prediction_rows = state_services.ml_jobs.get_batch_prediction_rows
        prediction_rows.assert_awaited_once_with([PREDICTION_ID],
                                                 project_ids=None)
    if allowed and action == 'training-cancel':
        backend.cancel_training_job.assert_awaited_once_with(TRAINING_ID)
    elif allowed and action == 'prediction-cancel':
        backend.cancel_batch_prediction_job.assert_awaited_once_with(
            PREDICTION_ID)
    elif not allowed:
        backend.get_job_home_dir.assert_not_awaited()
        backend.cancel_training_job.assert_not_awaited()
        backend.cancel_batch_prediction_job.assert_not_awaited()
