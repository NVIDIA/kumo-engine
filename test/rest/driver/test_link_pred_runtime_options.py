# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from kumo_core.types import TaskType
from kumo_core.types.schema import FPID
from kumoapi.pquery import QueryType

from kumo.validation import ValidationResponse
from rest.config.batch_prediction_job import BatchPredictionJobConfig
from rest.config.graph_v2 import GraphDefinition
from rest.config.predictive_query import PQueryRuntimeOptions
from rest.config.predictive_query_v2 import (
    PQueryResource,
    PQueryValidationArgs,
    pquery_resource_with_runtime_options,
)
from rest.driver import batch_prediction_job_logic, graph_logic, pquery_logic
from rest.driver.batch_prediction_job_logic import _make_bp_input_v2
from rest.driver.pquery_logic import PQueryLogic
from rest.settings import PredictiveQuerySettings
from statedb.config.common import (
    BatchPredictionJobID,
    GraphIDv2,
    PredictiveQueryID,
    ProjectID,
    TrainingJobID,
)

_PROJECT_ID = ProjectID('project_' + '1' * 32)


def _empty_graph_def() -> GraphDefinition:
    return GraphDefinition(tables={}, col_groups=[])


def _pquery_resource(
    *,
    list_distinct_target_is_array=False,
    lp_bp_layout='separate_lhs_rhs',
) -> PQueryResource:
    return PQueryResource(
        name='runtime_options_pquery',
        query_string='PREDICT LIST_DISTINCT(t.article_id) FOR EACH t.user_id',
        graph=_empty_graph_def(),
        runtime_options=PQueryRuntimeOptions(
            list_distinct_target_is_array=list_distinct_target_is_array,
            lp_bp_layout=lp_bp_layout,
        ),
    )


def test_pquery_resource_adapter_maps_top_level_raw_runtime_options():
    pquery_resource = pquery_resource_with_runtime_options(
        _pquery_resource(
            list_distinct_target_is_array=None,
            lp_bp_layout=None,
        ),
        {
            'list_distinct_target_is_array': False,
            'lp_bp_layout': 'separate_lhs_rhs',
        },
    )

    assert pquery_resource.list_distinct_target_is_array is False
    assert pquery_resource.lp_bp_layout == 'separate_lhs_rhs'


def test_pquery_resource_adapter_accepts_nested_runtime_options():
    pquery_resource = pquery_resource_with_runtime_options(
        _pquery_resource(
            list_distinct_target_is_array=None,
            lp_bp_layout=None,
        ),
        {
            'runtime_options': {
                'list_distinct_target_is_array': False,
                'lp_bp_layout': 'separate_lhs_rhs',
            },
        },
    )

    assert pquery_resource.list_distinct_target_is_array is False
    assert pquery_resource.lp_bp_layout == 'separate_lhs_rhs'


@pytest.mark.asyncio
async def test_make_predictive_config_v2_forwards_runtime_options(
    monkeypatch, ):
    make_config = AsyncMock(return_value='config')
    monkeypatch.setattr(
        pquery_logic,
        'make_predictive_config_with_graph',
        SimpleNamespace(coroutine=make_config),
    )
    monkeypatch.setattr(
        pquery_logic,
        'credential_free_graph_config_for_activity',
        MagicMock(return_value='credential-free-graph'),
    )

    logic = PQueryLogic(
        MagicMock(),
        state_services=SimpleNamespace(),
        project_id=_PROJECT_ID,
    )
    result = await logic.make_predictive_config_v2(
        PQueryValidationArgs(
            query_string=(
                'PREDICT LIST_DISTINCT(t.article_id) FOR EACH t.user_id'),
            graph=_empty_graph_def(),
            name='runtime_options_pquery',
            runtime_options=PQueryRuntimeOptions(
                list_distinct_target_is_array=False,
                lp_bp_layout='separate_lhs_rhs',
            ),
        ),
        graph=object(),
    )

    assert result == 'config'
    kwargs = make_config.await_args.kwargs
    assert kwargs['list_distinct_target_is_array'] is False
    assert kwargs['lp_bp_layout'] == 'separate_lhs_rhs'
    assert kwargs['graph_config'] == 'credential-free-graph'


@pytest.mark.asyncio
async def test_make_predictive_config_v2_preserves_omitted_runtime_options(
    monkeypatch, ):
    make_config = AsyncMock(return_value='config')
    monkeypatch.setattr(
        pquery_logic,
        'make_predictive_config_with_graph',
        SimpleNamespace(coroutine=make_config),
    )
    monkeypatch.setattr(
        pquery_logic,
        'credential_free_graph_config_for_activity',
        MagicMock(return_value='credential-free-graph'),
    )

    logic = PQueryLogic(
        MagicMock(),
        state_services=SimpleNamespace(),
        project_id=_PROJECT_ID,
    )
    result = await logic.make_predictive_config_v2(
        PQueryValidationArgs(
            query_string=(
                'PREDICT LIST_DISTINCT(t.article_id) FOR EACH t.user_id'),
            graph=_empty_graph_def(),
            name='runtime_options_pquery',
            runtime_options=PQueryRuntimeOptions(
                list_distinct_target_is_array=None,
                lp_bp_layout=None,
            ),
        ),
        graph=object(),
    )

    assert result == 'config'
    kwargs = make_config.await_args.kwargs
    assert kwargs['list_distinct_target_is_array'] is None
    assert kwargs['lp_bp_layout'] is None
    assert kwargs['graph_config'] == 'credential-free-graph'


@pytest.mark.asyncio
async def test_create_full_pquery_def_v2_persists_runtime_options(
    monkeypatch, ):
    class FakeGraphLogic:
        def __init__(self, data_backend_ctx, *, state_services=None,
                     project_id=None):
            pass

        async def create_full_graph_def_v2(self, graph_def):
            return object()

    monkeypatch.setattr(graph_logic, 'GraphLogic', FakeGraphLogic)

    pquery_service = SimpleNamespace(
        add_definition=AsyncMock(return_value=FPID('a' * 32)),
        set_alias=AsyncMock(),
    )
    logic = PQueryLogic(
        MagicMock(),
        state_services=SimpleNamespace(pqueries=pquery_service),
        project_id=_PROJECT_ID,
    )
    logic._validate_predictive_query_args = AsyncMock(return_value=MagicMock())
    pquery = SimpleNamespace(
        list_distinct_target_is_array=False,
        lp_bp_layout='separate_lhs_rhs',
        query_type=QueryType.TEMPORAL,
        get_task_type=lambda: TaskType.LINK_PREDICTION,
    )
    logic.parse_predictive_query_v2 = AsyncMock(return_value=(SimpleNamespace(
        config=SimpleNamespace(lp_bp_layout='separate_lhs_rhs'),
        predictive_query=pquery,
    ), ValidationResponse()))

    pq_def = await logic.create_full_pquery_def_v2(
        _pquery_resource(
            list_distinct_target_is_array=False,
            lp_bp_layout='separate_lhs_rhs',
        ),
        pquery_settings=PredictiveQuerySettings(),
    )

    assert pq_def.list_distinct_target_is_array is False
    assert pq_def.lp_bp_layout == 'separate_lhs_rhs'
    pquery_service.add_definition.assert_awaited_once()
    pquery_service.set_alias.assert_awaited_once_with(
        'runtime_options_pquery',
        FPID('a' * 32),
    )


@pytest.mark.asyncio
async def test_create_full_pquery_def_v2_preserves_omitted_runtime_options(
    monkeypatch, ):
    class FakeGraphLogic:
        def __init__(self, data_backend_ctx, *, state_services=None,
                     project_id=None):
            pass

        async def create_full_graph_def_v2(self, graph_def):
            return object()

    monkeypatch.setattr(graph_logic, 'GraphLogic', FakeGraphLogic)

    pquery_service = SimpleNamespace(
        add_definition=AsyncMock(return_value=FPID('a' * 32)),
        set_alias=AsyncMock(),
    )
    logic = PQueryLogic(
        MagicMock(),
        state_services=SimpleNamespace(pqueries=pquery_service),
        project_id=_PROJECT_ID,
    )
    logic._validate_predictive_query_args = AsyncMock(return_value=MagicMock())
    pquery = SimpleNamespace(
        list_distinct_target_is_array=False,
        lp_bp_layout=None,
        query_type=QueryType.TEMPORAL,
        get_task_type=lambda: TaskType.LINK_PREDICTION,
    )
    logic.parse_predictive_query_v2 = AsyncMock(return_value=(SimpleNamespace(
        config=SimpleNamespace(lp_bp_layout=None),
        predictive_query=pquery,
    ), ValidationResponse()))

    pq_def = await logic.create_full_pquery_def_v2(
        _pquery_resource(
            list_distinct_target_is_array=None,
            lp_bp_layout=None,
        ),
        pquery_settings=PredictiveQuerySettings(),
    )

    assert pq_def.list_distinct_target_is_array is None
    assert pq_def.lp_bp_layout is None
    pquery_service.add_definition.assert_awaited_once()
    pquery_service.set_alias.assert_awaited_once_with(
        'runtime_options_pquery',
        FPID('a' * 32),
    )


@pytest.mark.asyncio
async def test_make_bp_input_v2_rebuilds_config_with_runtime_options(
    monkeypatch, ):
    captured_args = []

    class FakePQueryLogic:
        def __init__(self, data_backend_ctx, *, state_services,
                     project_id=None):
            pass

        async def make_predictive_config_v2(self, val_args, graph=None):
            captured_args.append(val_args)
            return SimpleNamespace(graph=object())

    monkeypatch.setattr(
        batch_prediction_job_logic.pquery_logic,
        'PQueryLogic',
        FakePQueryLogic,
    )

    class FakeGraphLogic:
        def __init__(self, data_backend_ctx, *, state_services=None,
                     project_id=None):
            pass

        async def _make_graph_from_graph_def_v2(self, graph_def):
            return object()

    monkeypatch.setattr(
        batch_prediction_job_logic.graph_logic,
        'GraphLogic',
        FakeGraphLogic,
    )

    async def stop_after_config_rebuild(*_args, **_kwargs):
        raise RuntimeError('stop after config rebuild')

    monkeypatch.setattr(
        batch_prediction_job_logic,
        '_get_graph_data_compute_input_v2',
        stop_after_config_rebuild,
    )

    pq_resource = _pquery_resource(
        list_distinct_target_is_array=False,
        lp_bp_layout='separate_lhs_rhs',
    )
    job_config = BatchPredictionJobConfig(
        predictive_query_id=PredictiveQueryID('runtime_options_pquery'),
        training_job_id=TrainingJobID('training-job'),
        task_type=TaskType.LINK_PREDICTION,
        graph_id=GraphIDv2.compose(fpid=pq_resource.graph.fpid),
    )

    with pytest.raises(RuntimeError, match='stop after config rebuild'):
        await _make_bp_input_v2(
            pq_resource=pq_resource,
            pq_def_fpid=FPID('a' * 32),
            id=BatchPredictionJobID('bp-job'),
            job_config=job_config,
            data_backend_ctx=MagicMock(),
            state_services=SimpleNamespace(),
            project_id=None,
        )

    assert len(captured_args) == 1
    assert captured_args[0].list_distinct_target_is_array is False
    assert captured_args[0].lp_bp_layout == 'separate_lhs_rhs'
