# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import kumoapi.graph as graph_api
import kumoapi.source_table as source_table_api
import kumoapi.table as table_api
import pytest
from kumo_core.model_plan import ModelPlan
from kumo_core.types import TaskType
from kumoapi.pquery import QueryType
from kumoapi.typing import Dtype, Stype

from rest.config.predictive_query import DefaultModelPlanInfo
from rest.config.predictive_query_v2 import PQueryResource
from rest.routers.predictive_queries_v2 import PredictiveQueryV2Router
from statedb.config.common import PredictiveQueryID, ProjectID

_PROJECT_ID = 'project_0123456789abcdef0123456789abcdef'


def _pquery_resource() -> PQueryResource:
    return PQueryResource(
        query_string='PREDICT responses.value FOR EACH responses.id',
        graph=graph_api.GraphDefinition(
            tables={
                'responses':
                table_api.TableDefinition(
                    cols=[
                        table_api.Column(
                            name='id',
                            stype=Stype.ID,
                            dtype=Dtype.string,
                        ),
                        table_api.Column(
                            name='value',
                            stype=Stype.numerical,
                            dtype=Dtype.float,
                        ),
                    ],
                    pkey='id',
                    source_table=source_table_api.UnavailableSourceTable(
                        table='responses'),
                ),
            },
            col_groups=[],
        ),
        name='project_scoped_query',
        graph_display_name='Private graph display name',
        runtime_options={'list_distinct_target_is_array': False},
    )


@pytest.mark.asyncio
async def test_create_pquery_propagates_validated_project_to_driver():
    router = PredictiveQueryV2Router.__new__(PredictiveQueryV2Router)
    router._driver = AsyncMock()
    router._driver.create_predictive_query_v2.return_value = (
        PredictiveQueryID('project_scoped_query'))
    router._projects_driver = AsyncMock()
    request = _pquery_resource()

    response = await router.create_pquery_v2(
        request,
        project_id=_PROJECT_ID,
    )

    router._projects_driver._validate_project_for_creation.assert_awaited_once_with(
        _PROJECT_ID)
    _, kwargs = router._driver.create_predictive_query_v2.await_args
    assert kwargs == {'project_id': ProjectID(_PROJECT_ID)}
    assert router._driver.create_predictive_query_v2.await_args.args == (
        request, )
    assert response.id == 'project_scoped_query'


@pytest.mark.asyncio
async def test_validate_pquery_propagates_project_to_driver():
    router = PredictiveQueryV2Router.__new__(PredictiveQueryV2Router)
    router._driver = AsyncMock()
    validation = MagicMock()
    converted = MagicMock()
    router._driver.validate_predictive_query_v2.return_value = validation
    router._driver.convert_validation_response_to_error_details = MagicMock(
        return_value=converted)
    request = MagicMock()

    response = await router.validate_pquery(
        request,
        project_id=_PROJECT_ID,
    )

    router._driver.validate_predictive_query_v2.assert_awaited_once_with(
        request,
        check_unique_id=False,
        project_id=ProjectID(_PROJECT_ID),
    )
    assert response is converted


@pytest.mark.asyncio
async def test_model_plan_propagates_project_to_driver():
    router = PredictiveQueryV2Router.__new__(PredictiveQueryV2Router)
    router._driver = AsyncMock()
    plan_info = DefaultModelPlanInfo(
        model_plan=ModelPlan(),
        task_type=TaskType.REGRESSION,
        query_type=QueryType.STATIC,
    )
    router._driver.get_pquery_model_plan_with_graph_def.return_value = plan_info
    request = MagicMock()
    request.graph_definition = MagicMock()
    request.query_string = 'PREDICT responses.value FOR EACH responses.id'

    response = await router.get_default_model_plan(
        request,
        project_id=_PROJECT_ID,
    )

    router._driver.get_pquery_model_plan_with_graph_def.assert_awaited_once_with(
        request.query_string,
        request.graph_definition,
        request.run_mode,
        project_id=ProjectID(_PROJECT_ID),
    )
    assert response.suggested_model_plan is plan_info
