# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import time
from http import HTTPStatus
from types import SimpleNamespace
from typing import Dict, Union

import kumoapi.model_plan as plan_api
import pytest
from fastapi.testclient import TestClient
from kumoapi.graph import GraphDefinition as apiGraphDefinition

from kumo.testing.rest import (
    get_and_expect_error,
    get_and_expect_success,
    post_and_expect_success,
)
from rest.backends.ml_backend.job_models import GenericJob, ListJobsQuery
from rest.config.graph_v2 import GraphDefinition
from rest.config.predictive_query import ModelPlansResponse
from rest.config.predictive_query_v2 import ModelPlanResponse
from rest.config.status import IDResponse
from rest.config.training_job_v2 import TrainingJobRequest
from rest.routers.jobs import ListJobsResponse
from statedb.config.common import JobID, JobStatus, TrainingJobID
from statedb.config.job_tag import JobTagRequest, JobTags
from statedb.db.identities import ConnectorService


def test_create_train_job(
    client: TestClient,
    graph_definition_v2: apiGraphDefinition,
    default_project_id: str,
):
    graph_definition_internal = GraphDefinition(
        **dataclasses.asdict(graph_definition_v2))

    # Even though the training job will create the graph, we need to create it
    # upfront so that model_plan request can work.
    graph_id = post_and_expect_success(
        client,
        f'/v1/graphs/?project_id={default_project_id}',
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
        predictive_query=pquery_str,
        graph_definition=graph_definition_internal,
        training_table_gen_plan=model_plan.suggested_model_plan.model_plan.
        training_table_generation,
        model_plan=model_plan.suggested_model_plan.model_plan,
        project_id=default_project_id,
    )

    response = post_and_expect_success(
        client,
        "/v0/training_jobs/train",
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


def test_get_currenty_running_job_or_job_status(
    client: TestClient,
    graph_definition_v2: apiGraphDefinition,
    default_project_id: str,
):
    # trigger endpoint without any running or queued jobs
    no_running_jobs = post_and_expect_success(
        client=client,
        path=f"/v0/jobs/get_active_jobs?project_id={default_project_id}",
        payload=None,
        response_type=Dict[JobID, Union[GenericJob, JobStatus]],
    )

    assert len(no_running_jobs) == 0

    graph_definition_internal = GraphDefinition(
        **dataclasses.asdict(graph_definition_v2))

    # Even though the training job will create the graph, we need to create it
    # upfront so that model_plan request can work.
    graph_id = post_and_expect_success(
        client,
        f'/v1/graphs/?project_id={default_project_id}',
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
        predictive_query=pquery_str,
        graph_definition=graph_definition_internal,
        training_table_gen_plan=model_plan.suggested_model_plan.model_plan.
        training_table_generation,
        model_plan=model_plan.suggested_model_plan.model_plan,
        project_id=default_project_id,
    )

    training_job_id = post_and_expect_success(
        client,
        "/v0/training_jobs/train",
        request,
        TrainingJobID,
    )

    assert training_job_id is not None

    # trigger the endpoint with running or queued jobs
    training_jobs = post_and_expect_success(
        client=client,
        path=f"/v0/jobs/get_active_jobs?project_id={default_project_id}",
        payload=None,
        response_type=Dict[JobID, Union[GenericJob, JobStatus]],
    )

    assert len(training_jobs) == 2

    # trigger the endpoint with a body in the post request
    training_jobs = post_and_expect_success(
        client=client,
        path=f"/v0/jobs/get_active_jobs?project_id={default_project_id}",
        payload=[training_job_id],
        response_type=Dict[JobID, Union[GenericJob, JobStatus]],
    )

    assert len(training_jobs) == 2
    # because the job id was supplied in the call's body,
    # we should be only return the status for this job
    assert training_jobs[training_job_id] in [
        JobStatus.RUNNING, JobStatus.QUEUED
    ]


def test_archive_job(
    client: TestClient,
    graph_definition_v2: apiGraphDefinition,
    default_project_id: str,
):
    """Test archiving and unarchiving a job, and ensure archived jobs are only
    included when explicitly filtered by the 'archived' tag.
    """
    graph_definition_internal = GraphDefinition(
        **dataclasses.asdict(graph_definition_v2))

    # Create graph and training job
    graph_id = post_and_expect_success(
        client,
        f'/v1/graphs/?project_id={default_project_id}',
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
        predictive_query=pquery_str,
        graph_definition=graph_definition_internal,
        training_table_gen_plan=model_plan.suggested_model_plan.model_plan.
        training_table_generation,
        model_plan=model_plan.suggested_model_plan.model_plan,
        project_id=default_project_id,
    )

    training_job_id = post_and_expect_success(
        client,
        "/v0/training_jobs/train",
        request,
        TrainingJobID,
    )

    assert training_job_id is not None

    # Get initial job list
    initial_jobs = post_and_expect_success(
        client,
        '/v0/jobs/',
        ListJobsQuery(project_id=default_project_id, limit=20, page=0),
        ListJobsResponse,
    )
    initial_job_ids = {job.job_id for job in initial_jobs.jobs}
    assert training_job_id in initial_job_ids

    # Archive the training job
    archive_result = post_and_expect_success(
        client,
        f"/v0/jobs/{training_job_id}/archive?archived=true&project_id={default_project_id}",
        None,
        bool,
    )
    assert archive_result is True

    # Verify archived job is not in default list
    jobs_after_archive = post_and_expect_success(
        client,
        '/v0/jobs/',
        ListJobsQuery(project_id=default_project_id, limit=20, page=0),
        ListJobsResponse,
    )
    job_ids_after = {job.job_id for job in jobs_after_archive.jobs}
    assert training_job_id not in job_ids_after

    # Verify archived job appears with 'archived' tag filter
    jobs_archived_filter = post_and_expect_success(
        client,
        '/v0/jobs/',
        ListJobsQuery(project_id=default_project_id, limit=20, page=0,
                      tags_filter=['archived']),
        ListJobsResponse,
    )
    archived_job_ids = {job.job_id for job in jobs_archived_filter.jobs}
    assert training_job_id in archived_job_ids

    # Archived jobs should not appear with other tags
    other_tag_filter_jobs = post_and_expect_success(
        client,
        '/v0/jobs/',
        ListJobsQuery(project_id=default_project_id, limit=20, page=0,
                      tags_filter=['some_non_archived_tag']),
        ListJobsResponse,
    )
    other_tag_ids = {job.job_id for job in other_tag_filter_jobs.jobs}
    assert training_job_id not in other_tag_ids, (
        f"Archived job {training_job_id} should not appear when filtering by unrelated tags"
    )

    # Unarchive the training job
    unarchive_result = post_and_expect_success(
        client,
        f"/v0/jobs/{training_job_id}/archive?archived=false&project_id={default_project_id}",
        None,
        bool,
    )
    assert unarchive_result is True
    time.sleep(0.1)

    # Verify the job is back in default list
    jobs_after_unarchive = post_and_expect_success(
        client,
        '/v0/jobs/',
        ListJobsQuery(project_id=default_project_id, limit=20, page=0),
        ListJobsResponse,
    )
    job_ids_unarchived = {job.job_id for job in jobs_after_unarchive.jobs}
    assert training_job_id in job_ids_unarchived

    # Verify the job does not appear in 'archived' filter
    jobs_archived_after_unarchive = post_and_expect_success(
        client,
        '/v0/jobs/',
        ListJobsQuery(project_id=default_project_id, limit=20, page=0,
                      tags_filter=['archived']),
        ListJobsResponse,
    )
    archived_ids_after = {
        job.job_id
        for job in jobs_archived_after_unarchive.jobs
    }
    assert training_job_id not in archived_ids_after


@pytest.mark.skip(reason="TODO fix in follow up")
def test_train_job_model_plans(
    client: TestClient,
    graph_definition_v2: apiGraphDefinition,
    default_project_id: str,
):
    """Test that the model plans are returned correctly for a binary classification job,
    with weight mode and majority sampling ratio.
    """
    graph_definition_internal = GraphDefinition(
        **dataclasses.asdict(graph_definition_v2))

    # Even though the training job will create the graph, we need to create it
    # upfront so that model_plan request can work.
    graph_id = post_and_expect_success(
        client,
        f'/v1/graphs/?project_id={default_project_id}',
        graph_definition_v2,
        IDResponse,
    )

    pquery_str = (
        'PREDICT COUNT(transactions.*, 0, 30)>0 FOR EACH customers.customer_id')
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
    model_plan.suggested_model_plan.model_plan.optimization.max_epochs = 1
    model_plan.suggested_model_plan.model_plan.training_job.num_experiments = 1
    model_plan.suggested_model_plan.model_plan.optimization.weight_mode = [
        plan_api.WeightMode.SAMPLE
    ]
    model_plan.suggested_model_plan.model_plan.optimization.majority_sampling_ratio = [
        1, 2
    ]

    request = TrainingJobRequest(
        predictive_query=pquery_str,
        graph_definition=graph_definition_internal,
        training_table_gen_plan=model_plan.suggested_model_plan.model_plan.
        training_table_generation,
        model_plan=model_plan.suggested_model_plan.model_plan,
        project_id=default_project_id,
    )

    training_job_id = post_and_expect_success(
        client,
        "/v0/training_jobs/train",
        request,
        TrainingJobID,
    )

    assert training_job_id is not None

    training_jobs = post_and_expect_success(
        client=client,
        path=f"/v0/jobs/get_active_jobs?project_id={default_project_id}",
        payload=[training_job_id],
        response_type=Dict[JobID, Union[GenericJob, JobStatus]],
    )

    while training_jobs[training_job_id] in [
            JobStatus.RUNNING, JobStatus.QUEUED
    ]:
        time.sleep(10)
        training_jobs = post_and_expect_success(
            client=client,
            path=f"/v0/jobs/get_active_jobs?project_id={default_project_id}",
            payload=[training_job_id],
            response_type=Dict[JobID, Union[GenericJob, JobStatus]],
        )
    model_plans = get_and_expect_success(
        client,
        f"/v0/training_jobs/{training_job_id}/model_plans",
        {},
        ModelPlansResponse,
    )

    assert len(model_plans.model_plans) == 1
    model_plan = model_plans.model_plans[0]
    assert model_plan.optimization.weight_mode == [plan_api.WeightMode.SAMPLE]
    assert model_plan.optimization.majority_sampling_ratio == [1, 2]


def test_training_to_sdk(
    client: TestClient,
    graph_definition_v2: apiGraphDefinition,
    default_project_id: str,
    monkeypatch: pytest.MonkeyPatch,
):
    empty_job_id = ""
    train_sdk_script = get_and_expect_error(
        client,
        f"/v0/training_jobs/{empty_job_id}/sdk",
        {},
        HTTPStatus.NOT_FOUND,
        "Not Found",
    )

    graph_definition_internal = GraphDefinition(
        **dataclasses.asdict(graph_definition_v2))

    # Even though the training job will create the graph, we need to create it
    # upfront so that model_plan request can work.
    graph_id = post_and_expect_success(
        client,
        f'/v1/graphs/?project_id={default_project_id}',
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
        predictive_query=pquery_str,
        graph_definition=graph_definition_internal,
        training_table_gen_plan=model_plan.suggested_model_plan.model_plan.
        training_table_generation,
        model_plan=model_plan.suggested_model_plan.model_plan,
        project_id=default_project_id,
    )

    training_job_id = post_and_expect_success(
        client,
        "/v0/training_jobs/train",
        request,
        TrainingJobID,
    )

    assert training_job_id is not None

    # trigger the endpoint with running or queued jobs
    training_jobs = post_and_expect_success(
        client=client,
        path=f"/v0/jobs/get_active_jobs?project_id={default_project_id}",
        payload=None,
        response_type=Dict[JobID, Union[GenericJob, JobStatus]],
    )

    # wait for job to get to state `Running`
    while training_jobs[training_job_id] in [JobStatus.QUEUED]:
        time.sleep(10)
        training_jobs = post_and_expect_success(
            client=client,
            path=f"/v0/jobs/get_active_jobs?project_id={default_project_id}",
            payload=[training_job_id],
            response_type=Dict[JobID, Union[GenericJob, JobStatus]],
        )

    # The shared HNM fixture is deliberately an internal local-file connector
    # so ingestion can run without cloud credentials. SDK replay only supports
    # public project connectors, so expose the fixture's safe name at that
    # boundary without exposing its internal root.
    async def visible_test_connector(self, project_id, connector_id,
                                     principal):
        connector = await self.get_internal(project_id, connector_id)
        return SimpleNamespace(id=connector.id, name=connector.name)

    monkeypatch.setattr(ConnectorService,
                        'source_table_connector_for_project',
                        visible_test_connector)

    train_sdk_script = get_and_expect_success(
        client,
        f"/v0/training_jobs/{training_job_id}/sdk",
        {'project_id': default_project_id},
        str,
    )
    train_sdk_script = json.loads(train_sdk_script)

    # Make sure all the necessary components are part of the generated script
    assert train_sdk_script != ""
    assert f'project="{default_project_id}"' in train_sdk_script
    assert 'Connector.get_by_name(' in train_sdk_script
    assert 'root_dir=' not in train_sdk_script
    components = [
        "pquery_config", "pquery = kumo.PredictiveQuery(", "train_table_config",
        "train_table", "model_plan = {'training_job':",
        "trainer = kumo.Trainer(model_plan)", "training_job = trainer.fit("
    ]

    for component in components:
        assert component in train_sdk_script


def test_add_and_delete_job_tag(
    client: TestClient,
    graph_definition_v2: apiGraphDefinition,
    default_project_id: str,
):
    graph_definition_internal = GraphDefinition(
        **dataclasses.asdict(graph_definition_v2))

    # Even though the training job will create the graph, we need to create it
    # upfront so that model_plan request can work.
    graph_id = post_and_expect_success(
        client,
        f'/v1/graphs/?project_id={default_project_id}',
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
        predictive_query=pquery_str,
        graph_definition=graph_definition_internal,
        training_table_gen_plan=model_plan.suggested_model_plan.model_plan.
        training_table_generation,
        model_plan=model_plan.suggested_model_plan.model_plan,
        project_id=default_project_id,
    )

    training_job_id = post_and_expect_success(
        client,
        "/v0/training_jobs/train",
        request,
        TrainingJobID,
    )

    train_job_tags = get_and_expect_success(
        client=client,
        path="/v0/jobs/training_tags",
        params={'project_id': default_project_id},
        response_type=Dict[JobID, JobTags],
    )

    # Make sure tag doesn't already exist in the tags
    new_tag = "test_tag"
    assert new_tag not in train_job_tags[training_job_id].tags
    initial_tags = dict(train_job_tags[training_job_id].tags)

    for protected_key in ("status", "pquery_name", "pquery_id"):
        protected_add_result = post_and_expect_success(
            client,
            "/v0/jobs/add_tag",
            JobTagRequest(
                job_id=training_job_id,
                project_id=default_project_id,
                key=protected_key,
                value="protected_value",
            ),
            bool,
        )
        assert not protected_add_result

    train_job_tags = get_and_expect_success(
        client=client,
        path="/v0/jobs/training_tags",
        params={'project_id': default_project_id},
        response_type=Dict[JobID, JobTags],
    )
    assert train_job_tags[training_job_id].tags == initial_tags

    job_tag_request = JobTagRequest(
        job_id=training_job_id,
        project_id=default_project_id,
        key=new_tag,
        value=None,
    )

    # Assure a tag can be added successfully
    successfully_added_tag = post_and_expect_success(
        client,
        "/v0/jobs/add_tag",
        job_tag_request,
        bool,
    )
    assert successfully_added_tag

    # Assure a tag can be removed successfully
    train_job_tags = get_and_expect_success(
        client=client,
        path="/v0/jobs/training_tags",
        params={'project_id': default_project_id},
        response_type=Dict[JobID, JobTags],
    )
    assert new_tag in train_job_tags[training_job_id].tags

    successfully_removed_tag = post_and_expect_success(
        client,
        "/v0/jobs/delete_tag",
        job_tag_request,
        bool,
    )
    assert successfully_removed_tag

    train_job_tags = get_and_expect_success(
        client=client,
        path="/v0/jobs/training_tags",
        params={'project_id': default_project_id},
        response_type=Dict[JobID, JobTags],
    )
    assert new_tag not in train_job_tags[training_job_id].tags

    # Assure removing a tag that doesn't exist for a job doesn't break things
    remove_non_existing_tag = post_and_expect_success(
        client,
        "/v0/jobs/delete_tag",
        job_tag_request,
        bool,
    )
    assert not remove_non_existing_tag

    # Assure an immutable tag (e.g. pquery_id) cannot be removed from a job
    job_tag_request = JobTagRequest(
        job_id=training_job_id,
        project_id=default_project_id,
        key="pquery_id",
        value=None,
    )
    fail_to_remove_immutable_tag = post_and_expect_success(
        client,
        "/v0/jobs/delete_tag",
        job_tag_request,
        bool,
    )
    assert not fail_to_remove_immutable_tag
