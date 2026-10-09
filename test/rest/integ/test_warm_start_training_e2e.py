# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from http import HTTPStatus

import kumoapi.jobs as jobs_api
import kumoapi.model_plan as plan_api
import kumoapi.pquery as pquery_api
import pytest
from fastapi.testclient import TestClient
from kumo_core.job_config import JobConfig
from kumoapi.graph import GraphDefinition
from kumoapi.model_plan import RunMode

from io_utils.json import load_json
from kumo.testing.rest import (
    get_and_expect_success,
    post_and_expect_error,
    post_and_expect_success,
)
from rest.backends.ml_backend.data_util import get_job_base_dir, get_job_dir
from rest.config.status import IDResponse
from statedb.config.common import JobStatus, TrainingJobID
from statedb.config.job_tag import TAG_KEY_WARM_START_JOB_ID

# This e2e drives the real DG-native server pipeline via the REST API
# (/v1/graphs -> /v1/predictive_queries -> /v1/gentraintable_jobs -> training ->
# warm-start retrain). The server-side gen-train-table / materialize activities
# resolve the DiskGraph index deterministically from the graph key + version
# (statedb/config/diskgraph.py:get_diskgraph_index_path_for_job) -- no fall-through to
# legacy generic materialization. That work runs on the diskgraph pool task
# queue, so bring up the pool worker for this module (same as
# test_diskgraph_e2e). The earlier "legacy generic materialization" skip reason
# was inaccurate.
#
# Pin ``storage_backend`` to ``local``: the s3 parametrization writes DiskGraph
# ingest artifacts to the shared ``s3://kumo-dev-dataplane-bucket/test/`` bucket
# and the gen-train-table step then reads them back, which requires live AWS
# dataplane-bucket access the local / default-CI env lacks (the warm-start
# gen-train-table child workflow fails with an S3 HeadObject 404). The local
# backend exercises the identical DG warm-start pipeline end to end, so pinning
# to local keeps the DG coverage without the external-infra dependency -- same
# scoping as ``test_e2e`` and ``test_e2e_public_api``.
pytestmark = [
    pytest.mark.usefixtures("_pool_subprocesses"),
    pytest.mark.parametrize("storage_backend", ["local"], indirect=True),
]


@pytest.mark.asyncio
@pytest.mark.parametrize('query,pquery_name', [
    (
        "PREDICT COUNT(transactions.*, 0, 20) FOR EACH customers.customer_id",
        "test_warm_start_regression",
    ),
    (
        "PREDICT customers.age > 30 FOR EACH customers.customer_id",
        "test_warm_start_binary_classification",
    ),
])
async def test_warm_start_training_e2e(
    public_api_client: TestClient,
    graph_definition_v2: GraphDefinition,
    deploy_env_config,
    default_project_id: str,
    query: str,
    pquery_name: str,
):

    # Step 1: Create a graph, pquery and traing table
    graph_id = post_and_expect_success(
        public_api_client,
        '/v1/graphs',
        graph_definition_v2,
        IDResponse,
        project_id=default_project_id,
    ).id
    pquery_request = pquery_api.PQueryResource(
        name=pquery_name,
        query_string=query,
        graph=graph_definition_v2,
    )
    pquery_id = post_and_expect_success(
        public_api_client,
        "/v1/predictive_queries/",
        pquery_request,
        IDResponse,
        project_id=default_project_id,
    ).id
    plan_request = plan_api.SuggestModelPlanRequest(
        query_string=query,
        graph_id=graph_id,
        run_mode=RunMode.FAST,
    )
    train_table_plan = post_and_expect_success(
        client=public_api_client,
        path='/v1/predictive_queries/train_table_generation_plan',
        payload=plan_request,
        response_type=jobs_api.TrainingTableGenerationPlan,
        project_id=default_project_id,
    )
    if pquery_name == "test_warm_start_binary_classification":
        train_table_plan.split = "RandomSplit([0.6, 0.2, 0.2], seed=0)"
    train_table_job_id = post_and_expect_success(
        public_api_client,
        '/v1/gentraintable_jobs',
        jobs_api.GenerateTrainTableRequest({}, pquery_id, train_table_plan,
                                           None),
        IDResponse,
        project_id=default_project_id,
    ).id
    status = get_and_expect_success(
        public_api_client,
        f'/v1/gentraintable_jobs/{train_table_job_id}',
        {
            'project_id': default_project_id
        },
        jobs_api.GenerateTrainTableJobResource,
    ).job_status_report.status
    while not status.is_terminal:
        await asyncio.sleep(10)
        status = get_and_expect_success(
            public_api_client,
            f'/v1/gentraintable_jobs/{train_table_job_id}',
            {
                'project_id': default_project_id
            },
            jobs_api.GenerateTrainTableJobResource,
        ).job_status_report.status
    assert status == JobStatus.DONE
    model_plan = post_and_expect_success(
        public_api_client,
        '/v1/predictive_queries/model_plan',
        plan_api.SuggestModelPlanRequest(
            query_string=query,
            graph_id=graph_id,
            run_mode=plan_api.RunMode.FAST,
        ),
        plan_api.ModelPlanInfo,
        project_id=default_project_id,
    ).model_plan
    model_plan.training_job.num_experiments = 1
    model_plan.optimization.max_epochs = 1
    model_plan.optimization.max_steps_per_epoch = 2
    model_plan.training_job.refit_trainval = True

    # Step 2: Start initial training job
    training_job_config = jobs_api.TrainingJobRequest(
        {},
        pquery_id,
        model_plan,
        None,
        train_table_job_id,
    )
    initial_train_job_id = post_and_expect_success(
        public_api_client,
        '/v1/training_jobs',
        training_job_config,
        IDResponse,
        project_id=default_project_id,
    ).id
    status = get_and_expect_success(
        public_api_client, f'/v1/training_jobs/{initial_train_job_id}', {
            'project_id': default_project_id
        }, jobs_api.TrainingJobResource).job_status_report.status
    while not status.is_terminal:
        await asyncio.sleep(10)
        status = get_and_expect_success(
            public_api_client, f'/v1/training_jobs/{initial_train_job_id}', {
                'project_id': default_project_id
            }, jobs_api.TrainingJobResource).job_status_report.status
    assert status == jobs_api.JobStatus.DONE

    # Check that warm start validation is working
    warm_start_training_job_config = jobs_api.TrainingJobRequest(
        {},
        pquery_id,
        model_plan,
        None,
        train_table_job_id,
        warm_start_job_id="non-existent-job-id",  # Set warm start job ID
    )
    post_and_expect_error(
        public_api_client,
        f'/v1/training_jobs?project_id={default_project_id}',
        warm_start_training_job_config,
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        error_pattern="Warm start training job.*not found",
    )

    # Step 3: Start second training job with warm_start_job_id set to the first job
    warm_start_training_job_config = jobs_api.TrainingJobRequest(
        {},
        pquery_id,
        model_plan,
        None,
        train_table_job_id,
        warm_start_job_id=initial_train_job_id,  # Set warm start job ID
    )
    warm_start_train_job_id = post_and_expect_success(
        public_api_client,
        '/v1/training_jobs',
        warm_start_training_job_config,
        IDResponse,
        project_id=default_project_id,
    ).id
    status = get_and_expect_success(
        public_api_client, f'/v1/training_jobs/{warm_start_train_job_id}', {
            'project_id': default_project_id
        }, jobs_api.TrainingJobResource).job_status_report.status
    while not status.is_terminal:
        await asyncio.sleep(10)
        status = get_and_expect_success(
            public_api_client, f'/v1/training_jobs/{warm_start_train_job_id}', {
                'project_id': default_project_id
            }, jobs_api.TrainingJobResource).job_status_report.status
    assert status == jobs_api.JobStatus.DONE

    # Step 4: Verify that warm start related fields are set correctly
    job_response = get_and_expect_success(
        public_api_client, f'/v1/training_jobs/{warm_start_train_job_id}',
        {'project_id': default_project_id}, jobs_api.TrainingJobResource)
    assert job_response.tags[TAG_KEY_WARM_START_JOB_ID] == initial_train_job_id
    assert job_response.config.warm_start_job_id == initial_train_job_id

    warm_start_job_dir = get_job_dir(get_job_base_dir(),
                                     TrainingJobID(warm_start_train_job_id))
    job_config_path = os.path.join(warm_start_job_dir, 'jobs',
                                   f'{warm_start_train_job_id}.json')
    job_config = load_json(job_config_path, JobConfig)
    assert job_config.warm_start_trainer_dir is not None
    assert initial_train_job_id in job_config.warm_start_trainer_dir
