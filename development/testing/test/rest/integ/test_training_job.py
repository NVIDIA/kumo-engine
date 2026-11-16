# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
from tempfile import TemporaryDirectory

import kumoapi.model_plan as plan_api
import pytest
from fastapi.testclient import TestClient
from kumo_core.job_config import JobConfig
from kumo_core.model_plan import ModelPlan
from kumoapi.graph import GraphDefinition as apiGraphDefinition
from kumoapi.pquery import ValidatedPredictiveQuery
from kumoapi.pquery.AST import Column

from io_utils.json import save_json
from kumo.config.predictive_query import GraphConfig, PredictiveQueryV2Config
from kumo.task.split import RandomSplit
from kumo.testing.rest import post_and_expect_success
from rest.backends.ml_backend.job_models import ListJobsQuery
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.backends.ml_backend.settings import MLJobSettings
from rest.backends.ml_backend.workflows import TrainingJobWorkflow
from rest.config.graph_v2 import GraphDataComputeInput, GraphDefinition
from rest.config.ml_backend import TrainingJobInput
from rest.config.predictive_query_v2 import ModelPlanResponse
from rest.config.status import IDResponse
from rest.config.training_job import TrainingJobOutput, TrainingSpec
from rest.config.training_job_v2 import TrainingJobRequest
from rest.constants import ML_WORKFLOW_TASK_QUEUE
from rest.deps.authentication import RequestContext
from rest.routers.jobs import ListJobsResponse
from service_framework.deploy_env_config import DeployEnvConfig
from statedb.config.common import (
    GenerateTrainTableJobID,
    GraphIDv1,
    JobStatus,
    PredictiveQueryID,
    TrainingJobID,
    now_utc,
)
from temporalio_commons.temporal_client import TemporalClient


def _training_spec() -> TrainingSpec:
    return TrainingSpec(
        predictive_query=PredictiveQueryV2Config(
            graph=GraphConfig(),
            name="name",
            validated_predictive_query=ValidatedPredictiveQuery(
                entity_ast=Column(fqn="entity.entity"),
                target_ast=Column(fqn="entity.target"),
            ),
        ),
        advanced_options=ModelPlan(),
        graph_id=GraphIDv1("graph_name"),
        predictive_query_id=PredictiveQueryID("name"),
        split=RandomSplit([0.8, 0.1, 0.1]).config(),
    )


@pytest.mark.asyncio
async def test_error_handling(temporal_workers_ready: bool, ctx: RequestContext,
                              deploy_env_config: DeployEnvConfig,
                              ml_backend: MLJobBackend,
                              temporal_client: TemporalClient,
                              default_project_id: str):
    # Test that an error during job execution will result in the job entering
    # the FAILED state, and capture the error message and stacktrace
    spec = _training_spec()
    job_settings = MLJobSettings(raise_fake_error=True,
                                 use_fake_job_backend=True)
    from rest.config.graph_v2 import GraphDefinition
    dummy_graph_compute_input = GraphDataComputeInput(
        'graph', 'graph', {}, {}, {}, GraphDefinition(tables={}, col_groups=[]))
    job_id = await ml_backend.start_training(
        None,
        dummy_graph_compute_input,
        spec,
        project_id=default_project_id,
        ml_job_settings=job_settings,
    )
    job_input = TrainingJobInput(
        id=job_id,
        generate_train_table_job_id=GenerateTrainTableJobID('gen-train-table'),
        graph_data_compute_input=dummy_graph_compute_input,
        spec=_training_spec(),
        start_time=now_utc(),
    )

    client = await temporal_client.get()
    with TemporaryDirectory() as dir:
        job_config_path = os.path.join(dir, 'job_config.json')
        await save_json.coroutine(JobConfig('home_dir', 'task_name'),
                                  job_config_path)
        await client.execute_workflow(
            TrainingJobWorkflow.run,
            id=TrainingJobWorkflow.generate_wf_id(job_id),
            args=[job_input, job_settings, job_config_path],
            task_queue=ML_WORKFLOW_TASK_QUEUE,
        )

    training_job = await ml_backend.get_training_job(ctx, job_id)
    assert training_job
    output: TrainingJobOutput = training_job.execution
    assert output.status == JobStatus.FAILED
    assert output.failure_info is not None
    assert output.failure_info.message == 'fake error'
    assert output.failure_info.stacktrace


def test_create_train_job(
    client: TestClient,
    graph_definition_v2: apiGraphDefinition,
    default_project_id: str,
    monkeypatch: pytest.MonkeyPatch,
):
    # The REST fixture authenticates with the native disabled principal. Keep
    # the activity-side authority resolver in that same mode.
    monkeypatch.setenv('DISABLE_AUTH', 'true')
    graph_definition_internal = GraphDefinition(
        **dataclasses.asdict(graph_definition_v2))

    # Even though the training job will create the graph, we need to create it
    # upfront so that model_plan request can work.
    graph_id = post_and_expect_success(
        client,
        (f'/v1/graphs/?project_id={default_project_id}'
         '&skip_validation=true'),
        graph_definition_v2,
        IDResponse,
    )

    pquery_str = (
        'PREDICT COUNT(transactions.*, 0, 30) FOR EACH customers.customer_id')
    payload = plan_api.SuggestModelPlanRequest(
        query_string=pquery_str,
        graph_id=graph_id.id,
        run_mode=plan_api.RunMode.FAST,
        graph_definition=graph_definition_internal,
    )

    model_plan = post_and_expect_success(
        client=client,
        path=(f'/v1/predictive_queries/model_plan?project_id='
              f'{default_project_id}'),
        payload=payload,
        response_type=ModelPlanResponse,
    )

    request = TrainingJobRequest(
        tags=['training_job', 'ltv'],
        predictive_query=pquery_str,
        graph_definition=graph_definition_internal,
        training_table_gen_plan=model_plan.suggested_model_plan.model_plan.
        training_table_generation,
        model_plan=model_plan.suggested_model_plan.model_plan,
        project_id=default_project_id,
    )

    response = post_and_expect_success(
        client,
        '/v0/training_jobs/train',
        request,
        TrainingJobID,
    )

    assert response is not None

    jobs_request = ListJobsQuery(project_id=default_project_id, limit=20,
                                 page=0)
    jobs_response = post_and_expect_success(
        client,
        '/v0/jobs/',
        jobs_request,
        ListJobsResponse,
    )

    assert jobs_response is not None
    # Note that list jobs doesn't return graph snapshot job
    assert len(jobs_response.jobs) == 2
    assert any(
        job.job_id.startswith('gen-traintable-job')
        for job in jobs_response.jobs)
    assert any(
        job.job_id.startswith('trainingjob') for job in jobs_response.jobs)
    assert jobs_response.jobs[0].source == 'UI'
    job_with_tags = [
        job for job in jobs_response.jobs
        if 'training_job' in job.metadata.tags and 'ltv' in job.metadata.tags
    ]
    assert job_with_tags is not None
    assert len(job_with_tags) == 2
    assert job_with_tags[0].metadata.tags['training_job'] == 'training_job'
    assert job_with_tags[0].metadata.tags['ltv'] == 'ltv'
