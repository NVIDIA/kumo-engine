# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for public Projects router (GET/POST /v1/projects)."""
from datetime import datetime, timezone
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from kumoapi.data_source import DEFAULT_PROJECT_ID

from common_utils.error import PublicException
from rest.deps.authentication import Authentication
from rest.driver.projects_driver import ProjectsDriver
from rest.public.routers.projects import ProjectsRouter
from statedb.config.common import GroupID, ProjectID
from statedb.config.timestamp import ResourceTimestamps
from statedb.rest.project import ProjectDBEntry
from statedb.utils.errors import ProjectNotFoundError
from temporalio_commons import RpcError

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)
_TIMESTAMPS = ResourceTimestamps(created_at=_NOW, updated_at=_NOW)

_PROJECT_ID = ProjectID(f"project_{'a' * 32}")
_GROUP_ID = GroupID(f"group_{'b' * 32}")
_USER_ID = "test-user-123"


def _noop_auth():
    """No-op dependency override for api_key_auth."""


@pytest.fixture
def projects_driver():
    driver = MagicMock(spec=ProjectsDriver)
    driver.list_projects_in_group = AsyncMock()
    driver.create_project = AsyncMock()
    return driver


@pytest.fixture
def auth():
    a = MagicMock(spec=Authentication)
    return a


@pytest.fixture
def client(projects_driver, auth):
    app = FastAPI()
    app.add_exception_handler(
        PublicException,
        lambda _request, error: JSONResponse(
            status_code=error.http_code,
            content={
                'error': error.get_user_message(),
                'error_type': error.__class__.__name__,
            },
        ),
    )
    router = ProjectsRouter(
        api_key_auth=_noop_auth,
        projects_driver=projects_driver,
        auth=auth,
    )
    app.include_router(router.router, prefix="/v1")
    with TestClient(app) as c:
        yield c


def _make_project_entry(
    project_id=_PROJECT_ID,
    group_id=_GROUP_ID,
    name="my-project",
) -> ProjectDBEntry:
    return ProjectDBEntry(
        id=project_id,
        group_id=group_id,
        slug=name,
        display_name=name,
        created_by=_USER_ID,
        timestamps=_TIMESTAMPS,
    )


@pytest.mark.parametrize(
    'projects',
    [
        [],
        [
            _make_project_entry(name='proj-a'),
            _make_project_entry(project_id=ProjectID(f"project_{'c' * 32}"),
                                name='proj-b'),
        ],
    ],
    ids=('empty', 'populated'),
)
def test_list_projects_contract(client, projects_driver, projects):
    projects_driver.list_projects_in_group.return_value = projects

    response = client.get('/v1/projects', params={'group_id': str(_GROUP_ID)})

    assert response.status_code == HTTPStatus.OK
    assert [entry['display_name'] for entry in response.json()
            ] == [project.display_name for project in projects]
    assert all(
        set(entry) == {'id', 'slug', 'display_name', 'group_id'}
        for entry in response.json())
    assert _USER_ID not in response.text
    projects_driver.list_projects_in_group.assert_awaited_once_with(
        group_id=_GROUP_ID)


@pytest.mark.parametrize(
    ('route', 'driver_method', 'argument', 'project'),
    [
        ('/v1/projects/by-slug/my-project', 'get_project_by_slug', 'my-project',
         _make_project_entry()),
        ('/v1/projects/default', '_get_project', DEFAULT_PROJECT_ID,
         _make_project_entry(project_id=ProjectID(DEFAULT_PROJECT_ID),
                             name='default')),
    ],
    ids=('slug', 'default-id'),
)
def test_get_project_contract(client, projects_driver, route, driver_method,
                              argument, project):
    method = getattr(projects_driver, driver_method)
    method.return_value = project

    response = client.get(route)

    assert response.status_code == HTTPStatus.OK
    assert response.json()['id'] == str(project.id)
    assert response.json()['slug'] == project.slug
    method.assert_awaited_once_with(argument)


@pytest.mark.parametrize(
    ('route', 'driver_method'),
    [
        (f'/v1/projects/{_PROJECT_ID}', '_get_project'),
        ('/v1/projects/by-slug/hidden-project', 'get_project_by_slug'),
    ],
    ids=('id', 'slug'),
)
def test_hidden_or_missing_project_has_safe_404(client, projects_driver, route,
                                                driver_method):
    getattr(
        projects_driver,
        driver_method).side_effect = ProjectNotFoundError('Project not found')

    response = client.get(route)

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json() == {
        'error': 'Project not found',
        'error_type': 'ProjectNotFoundError',
    }


def test_create_project_contract(client, projects_driver):
    projects_driver.create_project.return_value = _make_project_entry()
    request = {
        'name': 'my-project',
        'group_id': str(_GROUP_ID),
        'slug': 'project-handle',
    }

    response = client.post('/v1/projects', json=request)

    assert response.status_code == HTTPStatus.OK
    assert response.json()['display_name'] == 'my-project'
    assert response.json()['group_id'] == str(_GROUP_ID)
    projects_driver.create_project.assert_awaited_once_with(
        **request, description='', is_private=False)


@pytest.mark.parametrize(
    ('method', 'route', 'driver_method', 'error_type', 'message', 'status'),
    [
        ('get', '/v1/projects?group_id=' + f"group_{'f' * 32}",
         'list_projects_in_group', 'KeyError', 'group not found',
         HTTPStatus.NOT_FOUND),
        ('post', '/v1/projects', 'create_project', 'KeyError',
         'Group not found', HTTPStatus.NOT_FOUND),
        ('post', '/v1/projects', 'create_project', 'ProjectForbiddenError',
         'Not a member of this group', HTTPStatus.FORBIDDEN),
    ],
)
def test_project_rpc_errors(client, projects_driver, method, route,
                            driver_method, error_type, message, status):
    getattr(projects_driver, driver_method).side_effect = RpcError(
        rpc_service='ProjectsService',
        rpc_method=driver_method,
        error_type=error_type,
        message=message,
        stack_trace='',
        non_retryable=True,
    )
    kwargs = ({
        'json': {
            'name': 'my-project',
            'group_id': str(_GROUP_ID)
        }
    } if method == 'post' else {})

    response = getattr(client, method)(route, **kwargs)

    assert response.status_code == status


@pytest.mark.parametrize(
    'payload',
    [
        {
            'group_id': str(_GROUP_ID)
        },
        {
            'name': 'my-project',
            'group_id': str(_GROUP_ID),
            'created_by': 'attacker-controlled',
        },
    ],
    ids=('missing-name', 'undeclared-field'),
)
def test_create_project_strict_request_schema(client, projects_driver, payload):
    response = client.post('/v1/projects', json=payload)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    projects_driver.create_project.assert_not_awaited()


def test_project_openapi_uses_exact_api_contract(client):
    schema = client.get('/openapi.json').json()
    create_schema = schema['components']['schemas']['CreateProjectArgs']
    response_schema = schema['components']['schemas']['ProjectResponse']
    assert create_schema['additionalProperties'] is False
    assert set(create_schema['properties']) == {
        'name', 'group_id', 'slug', 'description', 'is_private'
    }
    assert set(response_schema['properties']) == {
        'id', 'slug', 'display_name', 'group_id'
    }
