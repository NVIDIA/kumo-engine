# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
from http import HTTPStatus
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from common_utils.error import PublicException
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.deps.authentication import Authentication, RequestContext, get_ctx
from rest.driver.batch_prediction_job_logic import BatchPredictionJobLogic
from rest.driver.pquery_api_driver import PQueryApiDriver
from rest.public.app import _build_cors_app
from rest.public.routers.common import (
    APPLIED_TAG_FILTERS_HEADER,
    get_tag_filters,
    list_job_tags,
    resolve_requested_project,
)
from rest.public.routers.prediction import PredictionRouterV1
from rest.public.routers.training import TrainingRouterV1
from statedb.config.common import (
    BatchPredictionJobID,
    JobType,
    ProjectID,
    TrainingJobID,
)
from statedb.db.ml_jobs import MLJobService
from statedb.db.services import StateServices
from statedb.utils.errors import (
    BatchPredictionNotInProjectError,
    ProjectNotFoundError,
    TrainingJobNotInProjectError,
)

PROJECT_A = ProjectID('project_' + 'a' * 32)
PROJECT_B = ProjectID('project_' + 'b' * 32)


def _request(query: str) -> Request:
    return Request({
        'type': 'http',
        'query_string': query.encode(),
        'headers': [],
    })


def _ctx(
    *project_ids: ProjectID,
    admin: bool = False,
    external_membership_unavailable: bool = False,
) -> RequestContext:
    return cast(
        RequestContext,
        SimpleNamespace(user_authorization=SimpleNamespace(
            is_admin=admin,
            project_ids=tuple(str(project_id) for project_id in project_ids),
            external_membership_unavailable=external_membership_unavailable,
        )))


def test_bare_tag_filters_are_preserved_and_reported() -> None:
    response = Response()

    filters = get_tag_filters(
        _request('team=growth&pquery_id=pquery-deleted&limit=1'), response)

    assert filters == {
        'team': 'growth',
        'pquery_id': 'pquery-deleted',
    }
    assert response.headers[APPLIED_TAG_FILTERS_HEADER] == 'pquery_id,team'
    assert 'deprecation' not in response.headers
    assert 'warning' not in response.headers


def test_applied_tag_filters_header_is_cors_exposed() -> None:
    app = FastAPI()

    @app.get('/jobs')
    async def jobs(response: Response):  # pyright: ignore[reportUnusedFunction]
        response.headers[APPLIED_TAG_FILTERS_HEADER] = 'team'
        return []

    response = TestClient(_build_cors_app(app)).get(
        '/jobs', headers={'Origin': 'https://sdk.example'})

    assert response.headers['access-control-expose-headers'] == (
        APPLIED_TAG_FILTERS_HEADER)


@pytest.mark.asyncio
async def test_pquery_filter_does_not_probe_live_definitions() -> None:
    response = Response()
    list_job_tags_mock = AsyncMock(return_value={})
    require_predictive_query_exists = AsyncMock()
    driver = cast(
        PQueryApiDriver,
        SimpleNamespace(
            list_job_tags=list_job_tags_mock,
            require_predictive_query_exists=require_predictive_query_exists,
        ),
    )

    await list_job_tags(JobType.TRAINING_JOB,
                        _request('pquery_id=pquery-deleted'), driver, response)

    list_job_tags_mock.assert_awaited_once_with(JobType.TRAINING_JOB,
                                                {'pquery_id': 'pquery-deleted'})
    require_predictive_query_exists.assert_not_awaited()


@pytest.mark.parametrize(
    'query',
    ['Limit=1', 'Status=RUNNING', 'Project_id=x', 'Pquery_id=x', 'Model_id=x'],
)
def test_reserved_query_parameter_case_typo_is_rejected(query: str) -> None:
    with pytest.raises(HTTPException) as exc_info:
        get_tag_filters(_request(query), Response())

    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
    assert 'did you mean' in exc_info.value.detail


@pytest.mark.asyncio
async def test_requested_project_is_validated_with_request_identity() -> None:
    ctx = _ctx(PROJECT_A)
    driver = cast(
        PQueryApiDriver,
        SimpleNamespace(require_project_access=AsyncMock()),
    )

    assert await resolve_requested_project(ctx, driver, None) == (None, True)
    requested = await resolve_requested_project(ctx, driver, str(PROJECT_A))

    assert requested == (PROJECT_A, True)
    driver.require_project_access.assert_awaited_once_with(ctx, PROJECT_A)

    outage_ctx = _ctx(PROJECT_A, external_membership_unavailable=True)
    outage_driver = cast(
        PQueryApiDriver,
        SimpleNamespace(require_project_access=AsyncMock()),
    )
    with pytest.raises(HTTPException) as exc_info:
        await resolve_requested_project(outage_ctx, outage_driver,
                                        str(PROJECT_B))
    assert exc_info.value.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    outage_driver.require_project_access.assert_not_awaited()

    inaccessible_driver = cast(
        PQueryApiDriver,
        SimpleNamespace(require_project_access=AsyncMock()),
    )
    assert await resolve_requested_project(ctx, inaccessible_driver,
                                           str(PROJECT_B)) == (None, False)
    inaccessible_driver.require_project_access.assert_not_awaited()

    nonexistent_driver = cast(
        PQueryApiDriver,
        SimpleNamespace(require_project_access=AsyncMock(
            side_effect=ProjectNotFoundError('missing project'))),
    )
    assert await resolve_requested_project(ctx, nonexistent_driver,
                                           str(PROJECT_A)) == (None, False)
    nonexistent_driver.require_project_access.assert_awaited_once_with(
        ctx, PROJECT_A)


@pytest.mark.asyncio
async def test_prediction_list_scope_reaches_sql_before_limit() -> None:
    service = SimpleNamespace(list_batch_prediction_rows=AsyncMock(
        return_value=[]))
    logic = object.__new__(BatchPredictionJobLogic)
    logic._state_services = cast(StateServices,
                                 SimpleNamespace(ml_jobs=service))

    await logic._list_batch_prediction_job_entries(_ctx(admin=True),
                                                   project_id=PROJECT_A,
                                                   limit=1)

    kwargs = service.list_batch_prediction_rows.await_args.kwargs
    assert kwargs['project_ids'] == {PROJECT_A}
    assert kwargs['limit'] == 1


@pytest.mark.asyncio
async def test_training_list_scope_reaches_sql_before_limit() -> None:
    service = SimpleNamespace(list_training_job_rows=AsyncMock(return_value=[]))
    backend = object.__new__(MLJobBackend)
    backend._ml_job_service = cast(MLJobService, service)

    result = await backend.get_training_jobs(_ctx(admin=True), set(), limit=1,
                                             project_id=PROJECT_A)

    assert result == {}
    kwargs = service.list_training_job_rows.await_args.kwargs
    assert kwargs['project_ids'] == {PROJECT_A}


@pytest.mark.asyncio
@pytest.mark.parametrize('family', ['training', 'prediction'])
async def test_get_by_id_rejects_visible_project_mismatch(family: str) -> None:
    projects = SimpleNamespace(get_project=AsyncMock())
    ctx = _ctx(admin=True)
    driver = object.__new__(PQueryApiDriver)
    state_services = SimpleNamespace(projects=projects)
    driver._state_services = cast(StateServices, state_services)

    if family == 'training':
        driver._ml_job_backend = cast(
            MLJobBackend,
            SimpleNamespace(get_training_job=AsyncMock(
                return_value=SimpleNamespace(project_id=PROJECT_B))),
        )
        error_type = TrainingJobNotInProjectError
        call = driver.get_training_job(ctx, TrainingJobID.make_random_id(),
                                       project_id=PROJECT_A)
    else:
        state_services.ml_jobs = SimpleNamespace(
            get_batch_prediction_rows=AsyncMock(
                return_value=[SimpleNamespace(project_id=PROJECT_B)]))
        error_type = BatchPredictionNotInProjectError
        call = driver.get_batch_prediction_job(
            ctx,
            BatchPredictionJobID.make_random_id(),
            project_id=PROJECT_A,
        )

    with pytest.raises(error_type) as exc_info:
        await call
    assert exc_info.value.http_code == HTTPStatus.FORBIDDEN
    projects.get_project.assert_awaited_once_with(PROJECT_A,
                                                  ctx.user_authorization)


def test_prediction_read_returns_forbidden_for_visible_project_mismatch(
) -> None:
    job_id = BatchPredictionJobID.make_random_id()
    driver = SimpleNamespace(get_batch_prediction_job=AsyncMock(
        side_effect=BatchPredictionNotInProjectError('project mismatch')))
    router = PredictionRouterV1(
        lambda: None,
        cast(PQueryApiDriver, driver),
        cast(Authentication, SimpleNamespace()),
        cast(MLJobBackend, SimpleNamespace()),
    )
    app = FastAPI()
    app.add_exception_handler(
        PublicException,
        lambda _request, error: JSONResponse(
            status_code=error.http_code,
            content={'error_type': error.__class__.__name__},
        ),
    )
    app.include_router(router.router)
    app.dependency_overrides[get_ctx] = lambda: object()

    response = TestClient(app, raise_server_exceptions=False).get(
        f'/prediction_jobs/{job_id}?project_id={PROJECT_A}')

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json()['error_type'] == 'BatchPredictionNotInProjectError'


def test_progress_and_cancel_keep_required_project_scope() -> None:
    required_routes = (
        TrainingRouterV1.get_progress,
        TrainingRouterV1.cancel,
        PredictionRouterV1.get_progress,
        PredictionRouterV1.cancel_prediction_job,
    )
    optional_routes = (
        TrainingRouterV1.get,
        TrainingRouterV1.list,
        PredictionRouterV1.get_prediction_job,
        PredictionRouterV1.list_prediction_jobs,
    )

    for required_route in required_routes:
        parameter = inspect.signature(required_route).parameters['project_id']
        assert parameter.default is inspect.Parameter.empty
    for optional_route in optional_routes:
        parameter = inspect.signature(optional_route).parameters['project_id']
        assert parameter.default is None
