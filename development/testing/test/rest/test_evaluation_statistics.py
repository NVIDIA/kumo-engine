# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from kumo_core.explain.schemas import (
    ColumnData,
    EntityOutput,
    HistoricalItems,
    NodeOutputData,
)
from kumo_core.job_config import BestTrainerConfig
from kumo_core.train.stats import (
    AllTargetRegressionStats,
    TargetHistogramAndStats,
)
from kumo_core.training.config import RefitConfig, TrainerConfig
from kumo_core.types import TaskType
from kumoapi.explain import ColumnAnalysisOutput, ColumnCohorts
from kumoapi.typing import Stype

from io_utils.json import save_json
from kumo.config.cohort import DimensionSpec
from kumo.config.explain import StrategyMappings
from kumo.explain.select import ClassificationSelectionStrategy
from kumo.home import HomeDirManager
from kumo.testing.rest import get_and_expect_error, get_and_expect_success
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.evaluation import XAIResponse
from rest.config.graph_v2 import GraphResourceResponse
from rest.config.status import RestStatusCode
from rest.driver.rest_api_driver import RestApiDriver
from rest.routers.evaluation_statistics import EvaluationStatisticsResource
from statedb.config.common import (
    JobStatus,
    PredictiveQueryID,
    ProjectID,
    TableName,
    TrainingJobID,
)


def test_binary_classification(client: TestClient, fake_training_job):
    # Make fake training job
    job_id = fake_training_job.training_job_id
    get_and_expect_error(
        client, f"/v0/training_jobs/{job_id}/eval/binary_classification", {},
        HTTPStatus.NOT_FOUND, RestStatusCode.EVALUATION_NOT_YET_COMPLETED.value)


@pytest.mark.asyncio
async def test_eval_training_job_lookup_uses_pquery_driver_helper():
    training_job_id = TrainingJobID("training-job-123")
    pquery_id = PredictiveQueryID("predictive_query_123")
    project_id = ProjectID("project_" + "1" * 32)
    expected_execution = object()
    ctx = object()
    pq_driver = SimpleNamespace(
        get_current_training_job_id=AsyncMock(return_value=training_job_id),
        _pquery_settings=None,
    )
    driver = SimpleNamespace(pquery_driver=SimpleNamespace(
        get_training_job=AsyncMock(return_value=SimpleNamespace(
            execution=expected_execution))))
    resource = EvaluationStatisticsResource(
        auth=SimpleNamespace(authenticate_request_header=lambda: None),
        driver=driver,
        pq_driver=pq_driver,
    )

    assert (await resource._get_completed_training_job_for_pq(
        pquery_id, project_id=project_id, ctx=ctx)) is expected_execution

    pq_driver.get_current_training_job_id.assert_awaited_once_with(
        ctx, pquery_id, project_id)
    driver.pquery_driver.get_training_job.assert_awaited_once_with(
        ctx, training_job_id)


@pytest.mark.asyncio
async def test_rest_home_dir_lookup_uses_pquery_driver_helper(tmp_path):
    training_job_id = TrainingJobID("training-job-123")
    pquery_id = PredictiveQueryID("predictive_query_123")
    project_id = ProjectID("project_00000000000000000000000000000001")
    user_authorization = object()
    ctx = SimpleNamespace(user_authorization=user_authorization)
    driver = object.__new__(RestApiDriver)
    driver._pquery_api_driver = SimpleNamespace(
        get_current_training_job_id=AsyncMock(return_value=training_job_id))
    driver._ml_job_backend = SimpleNamespace(
        get_training_job=AsyncMock(return_value=SimpleNamespace(
            execution=SimpleNamespace(
                status=JobStatus.DONE), project_id=project_id)),
        get_job_home_dir=AsyncMock(return_value=str(tmp_path)),
    )
    get_project = AsyncMock()
    driver._state_services = SimpleNamespace(projects=SimpleNamespace(
        get_project=get_project))

    home_dir, resolved_project_id, job = await driver._get_home_dir(
        id=pquery_id,
        job_id=None,
        ctx=ctx,
        project_id=project_id,
    )

    assert isinstance(home_dir, HomeDirManager)
    assert resolved_project_id == project_id
    assert job.project_id == project_id
    driver._pquery_api_driver.get_current_training_job_id.assert_awaited_once_with(
        ctx, pquery_id, project_id)
    driver._ml_job_backend.get_training_job.assert_awaited_once_with(
        ctx, training_job_id)
    driver._ml_job_backend.get_job_home_dir.assert_awaited_once_with(
        training_job_id)
    get_project.assert_awaited_once_with(project_id, user_authorization)


@pytest.mark.asyncio
async def test_get_xai(client: TestClient, ml_backend: MLJobBackend,
                       fake_graph_v2: GraphResourceResponse, fake_training_job):
    pquery = fake_training_job

    # after 120 seconds, the original job should be in the DONE state
    await ml_backend.mock_training_job_complete(
        fake_training_job.training_job_id)

    # Currently when column analysis is not ready,
    # an empty XAIResponse will be returned
    response = get_and_expect_success(
        client, f"/v0/training_jobs/{pquery.training_job_id}/eval/xai", {},
        XAIResponse)
    assert response == XAIResponse()  # empty

    col_cohorts = ColumnCohorts(
        stype=Stype.categorical,
        cohorts=['catA', 'catB'],
        populations=[0.4, 0.6],
        predictions=[0.3, 0.7],
        targets=[0.25, 0.65],
        prediction_score=0.1,
        target_score=0.1,
    )
    # Pick the first table from the v2 ``table_name_to_fpid_mapping``. The
    # job-centric XAI endpoint (``id=None``) routes to
    # ``pquery_logic.get_xai_training_job``, which keys ``XAIResponse.tables``
    # by the bare ``to_table_id(Table.name).table_name`` (connector_id is
    # stripped), so we look the response up by the unqualified table name.
    table_name_str, _ = next(
        iter(fake_graph_v2.table_name_to_fpid_mapping.items()))
    table_name = TableName(table_name_str)
    cohort_output = ColumnAnalysisOutput(task_type=fake_training_job.task_type)
    cohort_output[table_name, 0, 'FREQUENCY'] = col_cohorts

    # Upload mock file to the training dir
    assert pquery.training_job_id is not None
    home_dir_mgr = HomeDirManager(await ml_backend.get_job_home_dir(
        pquery.training_job_id))
    await save_json.coroutine(
        BestTrainerConfig(
            best_trainer=home_dir_mgr.automl_dir(pquery.pquery_id)),
        home_dir_mgr.best_trainer_config_path(pquery.pquery_id))
    best_trainer_dir = await home_dir_mgr.get_best_trainer_dir.coroutine(
        pquery.pquery_id)
    column_analysis_file_path = os.path.join(
        best_trainer_dir, 'column_analysis/test/column_analysis.json')
    await save_json.coroutine(cohort_output, column_analysis_file_path)

    # Test successful S3 load
    response = get_and_expect_success(
        client, f"/v0/training_jobs/{pquery.training_job_id}/eval/xai", {},
        XAIResponse)
    assert len(response.tables.keys()) == len(
        fake_graph_v2.table_name_to_fpid_mapping)
    col_xai = response.tables[table_name_str]['FREQUENCY (seed entities)']
    assert col_xai.cohort
    assert col_xai.cohort.dimensions == [
        DimensionSpec(
            name='Data Field', all_values=[
                'Column Value', 'Population Ratio', 'Target Average Score',
                'Predicted Average Score'
            ], value_type='str')
    ]
    assert col_xai.cohort.json_data == {
        'Target Average Score': col_cohorts.targets,
        'Predicted Average Score': col_cohorts.predictions,
        'Column Value': col_cohorts.cohorts,
        'Population Ratio': col_cohorts.populations,
    }

    # Multiclass cohort output
    col_cohorts_multicls = ColumnCohorts(
        stype=Stype.categorical,
        cohorts=['catA', 'catB'],
        populations=[0.4, 0.6],
        predictions={
            'ClassA': [0.3, 0.7],
            'ClassB': [0.2, 0.8]
        },
        targets={
            'ClassA': [0.25, 0.65],
            'ClassB': [0.22, 0.68]
        },
        prediction_score=0.1,
        target_score=0.1,
    )
    cohort_output[table_name, 1, 'FREQUENCY'] = col_cohorts_multicls
    await save_json.coroutine(cohort_output, column_analysis_file_path)
    response = get_and_expect_success(
        client, f"/v0/training_jobs/{pquery.training_job_id}/eval/xai", {},
        XAIResponse)
    col_xai = response.tables[table_name_str]['FREQUENCY (1 hop)']
    assert col_xai.cohort
    assert col_xai.cohort.dimensions == [
        DimensionSpec(name='Class Label', all_values=['ClassA', 'ClassB'],
                      value_type='str'),
        DimensionSpec(
            name='Data Field', all_values=[
                'Column Value', 'Population Ratio', 'Target Average Score',
                'Predicted Average Score'
            ], value_type='str')
    ]
    assert isinstance(col_cohorts_multicls.targets, dict)
    assert isinstance(col_cohorts_multicls.predictions, dict)
    assert col_xai.cohort.json_data['ClassA'] == {
        'Target Average Score': col_cohorts_multicls.targets['ClassA'],
        'Predicted Average Score': col_cohorts_multicls.predictions['ClassA'],
        'Column Value': col_cohorts_multicls.cohorts,
        'Population Ratio': col_cohorts_multicls.populations,
    }
    assert col_xai.cohort.json_data['ClassB'] == {
        'Target Average Score': col_cohorts_multicls.targets['ClassB'],
        'Predicted Average Score': col_cohorts_multicls.predictions['ClassB'],
        'Column Value': col_cohorts_multicls.cohorts,
        'Population Ratio': col_cohorts_multicls.populations,
    }

    # Test column analysis refit_trainval
    trainer_config_path = os.path.join(best_trainer_dir, 'config.json')
    trainer_config = TrainerConfig(
        None, None, None, refit_cfg=RefitConfig(refitted_trainval=True))
    await save_json.coroutine(trainer_config, trainer_config_path)
    column_analysis_refit_file_path = os.path.join(
        best_trainer_dir.rstrip('/') + '_refit_trainval',
        'column_analysis/test/column_analysis.json')
    await save_json.coroutine(cohort_output, column_analysis_refit_file_path)
    response = get_and_expect_success(
        client, f"/v0/training_jobs/{pquery.training_job_id}/eval/xai", {},
        XAIResponse)
    assert len(response.tables.keys()) == len(
        fake_graph_v2.table_name_to_fpid_mapping)


@pytest.mark.asyncio
async def test_get_entity_output(client: TestClient, ml_backend: MLJobBackend,
                                 fake_training_job):
    pquery = fake_training_job

    # The "safe" entity name saved in mappings and as file name
    entity_id = "0"

    # after 120 seconds, the original job should be in the DONE state
    await ml_backend.mock_training_job_complete(
        fake_training_job.training_job_id)

    # Currently when subgraph collection is not ready,
    # an empty entity output will be returned
    response = get_and_expect_success(
        client,
        f"/v0/training_jobs/{pquery.training_job_id}/eval/entity_output/{entity_id}",
        {},
        EntityOutput,
    )
    assert response == EntityOutput(task_type=None)  # empty

    output = EntityOutput(task_type=TaskType.BINARY_CLASSIFICATION,
                          prediction=0.0)

    # Upload mock file to the training dir
    assert pquery.training_job_id is not None
    home_dir_mgr = HomeDirManager(await ml_backend.get_job_home_dir(
        pquery.training_job_id))
    await save_json.coroutine(
        BestTrainerConfig(
            best_trainer=home_dir_mgr.automl_dir(pquery.pquery_id)),
        home_dir_mgr.best_trainer_config_path(pquery.pquery_id))
    best_trainer_dir = await home_dir_mgr.get_best_trainer_dir.coroutine(
        pquery.pquery_id)
    output_file_path = os.path.join(
        best_trainer_dir, f'subgraph_collector/test/output/{entity_id}.json')
    await save_json.coroutine(output, output_file_path)

    # Test successful S3 load
    response = get_and_expect_success(
        client,
        f"/v0/training_jobs/{pquery.training_job_id}/eval/entity_output/{entity_id}",
        {},
        EntityOutput,
    )
    assert response.prediction == 0.0

    # Test output refit_trainval
    trainer_config_path = os.path.join(best_trainer_dir, 'config.json')
    trainer_config = TrainerConfig(
        None, None, None, refit_cfg=RefitConfig(refitted_trainval=True))
    await save_json.coroutine(trainer_config, trainer_config_path)
    output_refit_file_path = os.path.join(
        best_trainer_dir.rstrip('/') + '_refit_trainval',
        f'subgraph_collector/test/output/{entity_id}.json')
    await save_json.coroutine(output, output_refit_file_path)
    response = get_and_expect_success(
        client,
        f"/v0/training_jobs/{pquery.training_job_id}/eval/entity_output/{entity_id}",
        {},
        EntityOutput,
    )
    assert response.prediction == 0.0


@pytest.mark.asyncio
async def test_get_entity_historical_items(client: TestClient,
                                           ml_backend: MLJobBackend,
                                           fake_training_job):
    pquery = fake_training_job

    entity_id: str = "0"

    # after 120 seconds, the original job should be in the DONE state
    await ml_backend.mock_training_job_complete(
        fake_training_job.training_job_id)

    # Currently when subgraph collection is not ready,
    # an empty historical items will be returned
    response = get_and_expect_success(
        client,
        f"/v0/training_jobs/{pquery.training_job_id}/eval/entity_historical_items/{entity_id}",
        {},
        HistoricalItems,
    )
    assert response == HistoricalItems()  # empty

    items = HistoricalItems(nodes=[
        NodeOutputData(
            id=entity_id,
            cols=[ColumnData(name="a", value=2, stype=Stype.numerical)],
            id_name='article_id',
        )
    ])

    # Upload mock file to the training dir
    assert pquery.training_job_id is not None
    home_dir_mgr = HomeDirManager(await ml_backend.get_job_home_dir(
        pquery.training_job_id))
    await save_json.coroutine(
        BestTrainerConfig(
            best_trainer=home_dir_mgr.automl_dir(pquery.pquery_id)),
        home_dir_mgr.best_trainer_config_path(pquery.pquery_id))
    best_trainer_dir = await home_dir_mgr.get_best_trainer_dir.coroutine(
        pquery.pquery_id)
    items_file_path = os.path.join(
        best_trainer_dir,
        f'subgraph_collector/test/historical_items/{entity_id}.json')
    await save_json.coroutine(items, items_file_path)

    # Test successful S3 load
    response = get_and_expect_success(
        client,
        f"/v0/training_jobs/{pquery.training_job_id}/eval/entity_historical_items/{entity_id}",
        {},
        HistoricalItems,
    )
    assert len(response.nodes[0].cols) > 0
    assert response.nodes[0].id_name == 'article_id'

    # Test historical items refit_trainval
    trainer_config_path = os.path.join(best_trainer_dir, 'config.json')
    trainer_config = TrainerConfig(
        None, None, None, refit_cfg=RefitConfig(refitted_trainval=True))
    await save_json.coroutine(trainer_config, trainer_config_path)
    items_refit_file_path = os.path.join(
        best_trainer_dir.rstrip('/') + '_refit_trainval',
        f'subgraph_collector/test/historical_items/{entity_id}.json')
    await save_json.coroutine(items, items_refit_file_path)
    response = get_and_expect_success(
        client,
        f"/v0/training_jobs/{pquery.training_job_id}/eval/entity_historical_items/{entity_id}",
        {},
        HistoricalItems,
    )
    assert len(response.nodes[0].cols) > 0
    assert response.nodes[0].id_name == 'article_id'


@pytest.mark.asyncio
async def test_get_available_xai_entities(client: TestClient,
                                          ml_backend: MLJobBackend,
                                          fake_training_job):
    pquery = fake_training_job

    # after 120 seconds, the original job should be in the DONE state
    await ml_backend.mock_training_job_complete(
        fake_training_job.training_job_id)

    # Currently when subgraph collection is not ready,
    # an empty StrategyMappings will be returned
    response = get_and_expect_success(
        client,
        f"/v0/training_jobs/{pquery.training_job_id}/eval/available_xai_entities",
        {},
        StrategyMappings,
    )
    assert response == StrategyMappings()  # empty

    strategy_mappings = StrategyMappings(mappings={
        ClassificationSelectionStrategy.UNSURE.value: ['0', '1', '2']
    })

    # Upload mock file to the training dir
    assert pquery.training_job_id is not None
    home_dir_mgr = HomeDirManager(await ml_backend.get_job_home_dir(
        pquery.training_job_id))
    await save_json.coroutine(
        BestTrainerConfig(
            best_trainer=home_dir_mgr.automl_dir(pquery.pquery_id)),
        home_dir_mgr.best_trainer_config_path(pquery.pquery_id))
    best_trainer_dir = await home_dir_mgr.get_best_trainer_dir.coroutine(
        pquery.pquery_id)
    mappings_file_path = os.path.join(
        best_trainer_dir, 'subgraph_collector/test/strategy_mappings.json')
    await save_json.coroutine(strategy_mappings, mappings_file_path)

    # Test successful S3 load
    response = get_and_expect_success(
        client,
        f"/v0/training_jobs/{pquery.training_job_id}/eval/available_xai_entities",
        {},
        StrategyMappings,
    )
    assert len(response.mappings) == 1
    assert response.orig_mappings is None

    strategy_mappings = StrategyMappings(
        mappings={
            ClassificationSelectionStrategy.UNSURE.value: ['0', '1', '2']
        }, orig_mappings={
            ClassificationSelectionStrategy.UNSURE.value: ['2', '1', '0']
        })
    await save_json.coroutine(strategy_mappings, mappings_file_path)
    response = get_and_expect_success(
        client,
        f"/v0/training_jobs/{pquery.training_job_id}/eval/available_xai_entities",
        {},
        StrategyMappings,
    )
    assert len(response.mappings) == 1
    assert response.orig_mappings is not None
    assert len(response.orig_mappings) == 1

    # Test available xai entities refit_trainval
    trainer_config_path = os.path.join(best_trainer_dir, 'config.json')
    trainer_config = TrainerConfig(
        None, None, None, refit_cfg=RefitConfig(refitted_trainval=True))
    await save_json.coroutine(trainer_config, trainer_config_path)

    # Currently when subgraph collection is not ready,
    # an empty StrategyMappings will be returned
    response = get_and_expect_success(
        client,
        f"/v0/training_jobs/{pquery.training_job_id}/eval/available_xai_entities",
        {},
        StrategyMappings,
    )
    assert response == StrategyMappings()  # empty

    mappings_refit_file_path = os.path.join(
        best_trainer_dir.rstrip('/') + '_refit_trainval',
        'subgraph_collector/test/strategy_mappings.json')
    await save_json.coroutine(strategy_mappings, mappings_refit_file_path)

    response = get_and_expect_success(
        client,
        f"/v0/training_jobs/{pquery.training_job_id}/eval/available_xai_entities",
        {},
        StrategyMappings,
    )
    assert len(response.mappings) == 1
    assert response.orig_mappings is not None
    assert len(response.orig_mappings) == 1


@pytest.mark.asyncio
async def test_get_regression_histogram(client: TestClient,
                                        ml_backend: MLJobBackend,
                                        fake_training_job):
    pquery = fake_training_job

    # after 120 seconds, the original job should be in the DONE state
    await ml_backend.mock_training_job_complete(
        fake_training_job.training_job_id)

    # Currently when histogram is not available, an empty dataclass is returned
    result1 = get_and_expect_success(
        client,
        f"/v0/training_jobs/{pquery.training_job_id}/eval/regression_histogram",
        {}, AllTargetRegressionStats)
    assert result1 == AllTargetRegressionStats.empty()

    # Upload mock file to the training dir
    assert pquery.training_job_id is not None
    path = HomeDirManager(await ml_backend.get_job_home_dir(
        pquery.training_job_id)).all_target_regression_stats

    stats = AllTargetRegressionStats.empty()
    stats.label_distribution.test_table_stats = TargetHistogramAndStats(
        target_mean=2)

    await save_json.coroutine(stats, path)

    # Test successful S3 load
    result2 = get_and_expect_success(
        client,
        f"/v0/training_jobs/{pquery.training_job_id}/eval/regression_histogram",
        {}, AllTargetRegressionStats)
    assert result2.label_distribution.test_table_stats is not None
    assert result2.label_distribution.test_table_stats.target_mean == 2
