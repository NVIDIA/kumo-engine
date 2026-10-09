# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from common_utils.error import PublicException
from rest.driver.connector_api_driver import ConnectorApiDriver
from rest.driver.groups_driver import GroupsDriver
from rest.driver.projects_driver import ProjectsDriver
from rest.public.routers.projects import ProjectsRouter as PublicProjectsRouter
from rest.routers.groups import GroupsRouter
from rest.routers.identities import IdentitiesResource
from statedb.config.common import GroupID
from statedb.utils.errors import ResourceSlugConflictError


def _app_with_public_error_handler() -> FastAPI:
    app = FastAPI()

    async def handle_public_error(_request: Request,
                                  error: PublicException) -> JSONResponse:
        return JSONResponse(
            status_code=error.http_code,
            content={
                'error': error.get_user_message(),
                'error_type': error.__class__.__name__,
            },
        )

    app.add_exception_handler(PublicException, handle_public_error)
    return app


def _auth():
    return SimpleNamespace(
        authenticate_request_header=lambda: None,
        require_customer_admin=lambda: None,
    )


def _assert_safe_slug_conflict(response, resource: str, slug: str) -> None:
    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json() == {
        'error': f"{resource} slug '{slug}' is already in use.",
        'error_type': 'ResourceSlugConflictError',
    }


def test_project_create_slug_conflict_is_safe_http_409() -> None:
    slug = 'shared-project'
    driver = MagicMock(spec=ProjectsDriver)
    driver.create_project = AsyncMock(side_effect=ResourceSlugConflictError(
        f"Project slug '{slug}' is already in use."))
    app = _app_with_public_error_handler()
    app.include_router(
        PublicProjectsRouter(lambda: None, driver, _auth()).router,
        prefix='/v1',
    )

    with TestClient(app) as client:
        response = client.post(
            '/v1/projects', json={
                'name': 'Shared Project',
                'slug': slug,
                'group_id': str(GroupID.generate()),
            })

    _assert_safe_slug_conflict(response, 'Project', slug)


def test_group_create_slug_conflict_is_safe_http_409() -> None:
    slug = 'shared-group'
    driver = MagicMock(spec=GroupsDriver)
    driver._create_kumo_groups = AsyncMock(
        side_effect=ResourceSlugConflictError(
            f"Group slug '{slug}' is already in use."))
    app = _app_with_public_error_handler()
    app.include_router(GroupsRouter(_auth(), driver).router)

    with TestClient(app) as client:
        response = client.post(
            '/v0/groups', json={
                'display_name': 'Shared Group',
                'slug': slug,
            })

    _assert_safe_slug_conflict(response, 'Group', slug)


def test_identity_create_slug_conflict_is_safe_http_409() -> None:
    slug = 'shared-identity'
    service = SimpleNamespace(create=AsyncMock(
        side_effect=ResourceSlugConflictError(
            f"Identity slug '{slug}' is already in use.")))
    driver = MagicMock(spec=ConnectorApiDriver)
    driver._require_state_services.return_value = SimpleNamespace(
        identities=service)
    app = _app_with_public_error_handler()

    @app.middleware('http')
    async def add_principal(request, call_next):
        request.state.user_authorization = MagicMock()
        return await call_next(request)

    app.include_router(IdentitiesResource(_auth(), driver).router)

    with TestClient(app) as client:
        response = client.post(
            '/v0/identities/', json={
                'display_name': 'Shared Identity',
                'slug': slug,
                'config': {
                    'type': 's3',
                    'authentication': {
                        'mode':
                        'assume_role',
                        'partition':
                        'aws',
                        'role_arn':
                        ('arn:aws:iam::123456789012:role/shared-identity'),
                    },
                },
            })

    _assert_safe_slug_conflict(response, 'Identity', slug)
