# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from kumo_core.types import TaskType
from kumo_core.types.schema import FPID
from kumoapi.graph import GraphDefinition as ApiGraphDefinition
from kumoapi.pquery import PQueryResource, QueryType

from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.graph_v2 import GraphDefinition
from rest.config.predictive_query import OutputPresignedUrlResponse
from rest.config.status import RestStatusCode
from rest.config.training_job import TrainingJobOutput, TrainingJobResponse
from rest.deps.authentication import Authentication, RequestContext, get_ctx
from rest.driver.jobs_driver import JobsDriver
from rest.driver.pquery_api_driver import (
    PQueryApiDriver,
    TrainingJobEncodersNotAvailableError,
)
from rest.driver.rest_api_driver import RestApiDriver
from rest.public.routers.training import TrainingRouterV1
from rest.routers import training_jobs
from rest.routers.training_jobs import TrainingJobsResource
from statedb.config.common import (
    ErrWarnResponse,
    GraphIDv2,
    JobEventLogEntry,
    JobStatus,
    PredictiveQueryID,
    ProjectID,
    TrainingJobID,
)
from statedb.utils.errors import TrainingJobDoesNotExistError

_PROJECT_ID = "project_" + "1" * 32
_OTHER_PROJECT_ID = "project_" + "2" * 32


@pytest.fixture(autouse=True, scope="session")
def setup_rest_test_session():
    yield


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


def _auth() -> Authentication:
    return cast(
        Authentication,
        SimpleNamespace(
            _customer_id='customer',
            authenticate_request_header=lambda: None,
        ),
    )


def _ctx() -> RequestContext:
    return cast(
        RequestContext,
        SimpleNamespace(user_authorization=SimpleNamespace(
            is_admin=False,
            project_ids=(_PROJECT_ID, ),
            external_membership_unavailable=False,
        )),
    )


@pytest.mark.parametrize(
    ("method", "path", "downstream_name", "downstream_result"),
    [
        (
            "POST",
            "/v0/training_jobs/{id}/warnings",
            "get_training_job_warnings",
            {
                "trainingjob-project-scope":
                ErrWarnResponse(errors=[], warnings=[], info_items=[]),
            },
        ),
        (
            "GET",
            "/v0/training_jobs/{id}/materialization_info",
            "get_training_job_materialization_info",
            None,
        ),
        (
            "GET",
            "/v0/training_jobs/{id}/get_train_table_split_stats",
            "get_train_table_split_stats",
            None,
        ),
        (
            "GET",
            "/v0/training_jobs/{id}/download_holdout",
            "generate_holdout_presigned_url",
            OutputPresignedUrlResponse(
                presigned_url="https://signed.example.test/holdout"),
        ),
    ],
)
def test_training_artifact_routes_gate_exact_project_before_downstream_access(
    monkeypatch,
    method: str,
    path: str,
    downstream_name: str,
    downstream_result,
) -> None:
    """Foreign and absent jobs are identical and never reach artifact code."""
    training_job_id = TrainingJobID("trainingjob-project-scope")
    jobs = {
        training_job_id: SimpleNamespace(project_id=_OTHER_PROJECT_ID),
    }

    async def get_training_job(_ctx, id, strip_expensive_fields=True):
        del strip_expensive_fields
        return jobs.get(id)

    get_training_job_mock = AsyncMock(side_effect=get_training_job)
    downstream = AsyncMock(return_value=downstream_result)
    driver = SimpleNamespace(
        get_training_job=get_training_job_mock,
        get_training_job_warnings=AsyncMock(),
        get_training_job_materialization_info=AsyncMock(),
        get_train_table_split_stats=AsyncMock(),
        generate_holdout_presigned_url=AsyncMock(),
    )
    setattr(driver, downstream_name, downstream)
    resource = TrainingJobsResource.__new__(TrainingJobsResource)
    resource._driver = driver
    monkeypatch.setattr(training_jobs, "DISABLE_UI_WARNINGS", False)

    app = FastAPI()
    app.dependency_overrides[get_ctx] = _ctx
    handler = {
        "get_training_job_warnings": resource.get_warnings,
        "get_training_job_materialization_info":
        resource.get_materialization_info,
        "get_train_table_split_stats": resource.get_train_table_split_stats,
        "generate_holdout_presigned_url":
        resource.generate_holdout_presigned_url,
    }[downstream_name]
    app.add_api_route(path, handler, methods=[method])

    with TestClient(app) as client:
        missing_project = client.request(
            method,
            path.format(id=training_job_id),
        )
        assert missing_project.status_code == 422
        downstream.assert_not_awaited()

        foreign = client.request(
            method,
            path.format(id=training_job_id),
            params={"project_id": _PROJECT_ID},
        )
        assert foreign.status_code == 404
        downstream.assert_not_awaited()

        jobs.clear()
        missing = client.request(
            method,
            path.format(id=training_job_id),
            params={"project_id": _PROJECT_ID},
        )
        assert missing.status_code == 404
        assert missing.json() == foreign.json()
        downstream.assert_not_awaited()

        jobs[training_job_id] = SimpleNamespace(project_id=_PROJECT_ID)
        same_project = client.request(
            method,
            path.format(id=training_job_id),
            params={"project_id": _PROJECT_ID},
        )
        assert same_project.status_code == 200, same_project.text
        downstream.assert_awaited_once()


@pytest.mark.asyncio
async def test_training_workflow_response_returns_v2_graph_id_for_statedb_job():
    graph = ApiGraphDefinition(tables={}, col_groups=[])
    graph_fpid = GraphDefinition(**dataclasses.asdict(graph)).fpid
    pquery_def_fpid = FPID("a" * 32)
    training_job_id = TrainingJobID("trainingjob-test")
    training_job = TrainingJobResponse(
        id=training_job_id,
        execution=TrainingJobOutput(
            status=JobStatus.RUNNING,
            log=JobEventLogEntry.starting(),
        ),
        pquery_def_fpid=pquery_def_fpid,
        graph_data_version="graph-version-test",
        project_id=_PROJECT_ID,
    )
    parsed_pq = SimpleNamespace(
        task_type=TaskType.REGRESSION,
        query_type=QueryType("static"),
    )
    pquery_resource = PQueryResource(
        query_string="PREDICT COUNT(transactions.*, 0, 30) "
        "FOR EACH customers.customer_id",
        graph=graph,
    )
    driver = SimpleNamespace(
        get_training_job=AsyncMock(return_value=training_job),
        get_parsed_pq_v2_by_fpid=AsyncMock(return_value=parsed_pq),
        get_predictive_query_v2=AsyncMock(return_value=pquery_resource),
    )
    router = TrainingJobsResource.__new__(TrainingJobsResource)
    router._driver = driver

    response = await router.get_generation_and_training_jobs(
        training_job_id,
        project_id=_PROJECT_ID,
        ctx=_ctx(),
    )

    assert response.legacy_graph_id == GraphIDv2.public(graph_fpid)
    assert response.graph_snapshot_id == (
        f"{GraphIDv2.public(graph_fpid)}@graph-version-test")
    driver.get_training_job.assert_awaited_once()
    driver.get_parsed_pq_v2_by_fpid.assert_awaited_once_with(
        pquery_def_fpid, ProjectID(_PROJECT_ID), skip_validation=True)
    driver.get_predictive_query_v2.assert_awaited_once_with(
        PredictiveQueryID.compose(fpid=pquery_def_fpid),
        ProjectID(_PROJECT_ID),
    )


def _encoder_job(status: JobStatus) -> TrainingJobResponse:
    return TrainingJobResponse(
        id=TrainingJobID("trainingjob-encoders"),
        execution=TrainingJobOutput(
            status=status,
            log=JobEventLogEntry.starting(),
        ),
        pquery_def_fpid=FPID("a" * 32),
        graph_data_version="graph-version-test",
        project_id=_PROJECT_ID,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [JobStatus.FAILED, JobStatus.CANCELLED])
async def test_terminal_job_without_encoders_raises_not_found(
        status: JobStatus):
    get_encoders = AsyncMock(return_value=None)
    driver = cast(
        PQueryApiDriver,
        SimpleNamespace(
            get_training_job=AsyncMock(return_value=_encoder_job(status)),
            _ml_job_backend=SimpleNamespace(get_encoders_for_job=get_encoders),
        ),
    )

    with pytest.raises(TrainingJobEncodersNotAvailableError) as error:
        await PQueryApiDriver.get_encoders_for_training_job(
            driver, _ctx(), TrainingJobID("trainingjob-encoders"))

    assert str(error.value) == RestStatusCode.ENCODERS_NOT_AVAILABLE.value
    get_encoders.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_training_job_does_not_fetch_encoders():
    get_encoders = AsyncMock()
    driver = cast(
        PQueryApiDriver,
        SimpleNamespace(
            get_training_job=AsyncMock(return_value=None),
            _ml_job_backend=SimpleNamespace(get_encoders_for_job=get_encoders),
        ),
    )

    with pytest.raises(TrainingJobDoesNotExistError):
        await PQueryApiDriver.get_encoders_for_training_job(
            driver, _ctx(), TrainingJobID("trainingjob-missing"))

    get_encoders.assert_not_awaited()


@pytest.mark.parametrize('surface', ['internal', 'public'])
def test_nonterminal_job_without_encoders_returns_null_through_http(surface):
    driver = cast(
        PQueryApiDriver,
        SimpleNamespace(
            _pquery_settings=SimpleNamespace(),
            get_encoders_for_training_job=AsyncMock(return_value=None),
        ),
    )
    if surface == 'internal':
        resource = TrainingJobsResource(
            _auth(),
            cast(RestApiDriver, SimpleNamespace()),
            driver,
            cast(JobsDriver, SimpleNamespace()),
            cast(MLJobBackend, SimpleNamespace()),
        )
        path = '/v0/training_jobs/trainingjob-encoders/encoders'
    else:
        resource = TrainingRouterV1(
            lambda: None,
            driver,
            _auth(),
            cast(MLJobBackend, SimpleNamespace()),
        )
        path = '/v1/training_jobs/trainingjob-encoders/encoders'
    app = FastAPI()
    app.dependency_overrides[get_ctx] = _ctx
    app.include_router(resource.router,
                       prefix='/v1' if surface == 'public' else '')

    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code == 200
    assert response.json() is None
    driver.get_encoders_for_training_job.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [JobStatus.RUNNING, JobStatus.FAILED])
async def test_persisted_encoders_are_returned_for_terminal_and_nonterminal_jobs(
        status: JobStatus):
    encoders = {"table.column": "Numerical()"}
    get_encoders = AsyncMock(return_value=encoders)
    driver = cast(
        PQueryApiDriver,
        SimpleNamespace(
            get_training_job=AsyncMock(return_value=_encoder_job(status)),
            _ml_job_backend=SimpleNamespace(get_encoders_for_job=get_encoders),
        ),
    )

    assert await PQueryApiDriver.get_encoders_for_training_job(
        driver, _ctx(), TrainingJobID("trainingjob-encoders")) == encoders


@pytest.mark.asyncio
async def test_encoder_backend_errors_are_not_masked():
    get_encoders = AsyncMock(side_effect=RuntimeError("backend unavailable"))
    driver = cast(
        PQueryApiDriver,
        SimpleNamespace(
            get_training_job=AsyncMock(
                return_value=_encoder_job(JobStatus.FAILED)),
            _ml_job_backend=SimpleNamespace(get_encoders_for_job=get_encoders),
        ),
    )

    with pytest.raises(RuntimeError, match="backend unavailable"):
        await PQueryApiDriver.get_encoders_for_training_job(
            driver, _ctx(), TrainingJobID("trainingjob-encoders"))


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_type", [
    TrainingJobsResource,
    TrainingRouterV1,
])
@pytest.mark.parametrize(
    ("driver_error", "detail"),
    [
        (
            TrainingJobDoesNotExistError(TrainingJobID("trainingjob-missing")),
            "Training Job with id='trainingjob-missing' does not exist",
        ),
        (
            TrainingJobEncodersNotAvailableError(),
            RestStatusCode.ENCODERS_NOT_AVAILABLE.value,
        ),
    ],
)
async def test_encoder_routes_map_missing_resources_to_404(
        resource_type, driver_error, detail):
    encoders = AsyncMock(side_effect=driver_error)
    resource = resource_type.__new__(resource_type)
    resource._driver = SimpleNamespace(get_encoders_for_training_job=encoders, )

    with pytest.raises(HTTPException) as error:
        await resource.get_encoders_for_training_job(
            TrainingJobID("trainingjob-missing"), _ctx())

    assert error.value.status_code == 404
    assert error.value.detail == detail
    encoders.assert_awaited_once()
