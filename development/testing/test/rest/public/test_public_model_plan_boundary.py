# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import kumoapi.model_plan as plan_api
import pytest
from kumo_core.model_plan import ModelPlan
from kumo_core.types import TaskType
from kumo_core.types.schema import FPID
from kumoapi.pquery import QueryType

from io_utils.json import to_json_dict
from rest.config.predictive_query import DefaultModelPlanInfo
from rest.config.training_job import TrainingJobOutput, TrainingJobResponse
from rest.public.routers.pquery import PQueryRouter
from rest.public.routers.training import TrainingRouterV1
from statedb.config.common import (
    GraphIDv2,
    JobEventLogEntry,
    JobStatus,
    ProjectID,
    TrainingJobID,
)
from statedb.config.job_tag import JobTags
from statedb.config.timestamp import get_current_timestamps

_PROJECT_ID = 'project_' + '1' * 32


def _internal_model_plan() -> ModelPlan:
    model_plan = ModelPlan()
    model_plan.training_table_generation.weight_col = 'events.weight'
    return model_plan


@pytest.mark.asyncio
async def test_default_model_plan_routes_return_public_contract() -> None:
    router = PQueryRouter.__new__(PQueryRouter)
    router._driver = AsyncMock()
    router._driver.get_pquery_default_model_plan.return_value = DefaultModelPlanInfo(
        model_plan=_internal_model_plan(),
        task_type=TaskType.REGRESSION,
        query_type=QueryType.STATIC,
    )
    graph_id = GraphIDv2.public(FPID('a' * 32))
    request = plan_api.SuggestModelPlanRequest(
        query_string='PREDICT events.label FOR EACH events.entity_id',
        graph_id=graph_id,
        run_mode=plan_api.RunMode.FAST,
    )

    response = await router.get_default_model_plan(request, _PROJECT_ID)

    assert isinstance(response, plan_api.ModelPlanInfo)
    assert 'training_table_generation' not in to_json_dict(response.model_plan)
    router._driver.get_pquery_default_model_plan.assert_awaited_once_with(
        request.query_string,
        GraphIDv2.from_public(graph_id),
        request.run_mode,
        ProjectID(_PROJECT_ID),
        False,
    )


def test_training_job_resource_returns_public_model_plan() -> None:
    router = TrainingRouterV1.__new__(TrainingRouterV1)
    job = TrainingJobResponse(
        id=TrainingJobID('trainingjob-model-plan-boundary'),
        execution=TrainingJobOutput(
            status=JobStatus.RUNNING,
            log=JobEventLogEntry.starting(),
        ),
        project_id=_PROJECT_ID,
        advanced_options=_internal_model_plan(),
        pquery_def_fpid=FPID('b' * 32),
    )

    resource = router._make_training_job_resource_v1(
        job, JobTags(timestamps=get_current_timestamps()))

    assert isinstance(resource.config.model_plan, plan_api.ModelPlan)
    assert 'training_table_generation' not in to_json_dict(
        resource.config.model_plan)
