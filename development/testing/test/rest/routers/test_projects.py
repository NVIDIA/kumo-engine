# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from rest.driver.projects_driver import ProjectsDriver
from rest.routers.projects import ProjectsRouter
from rest.utils.routers import ValidateNameResponse
from statedb.utils.errors import ProjectNotFoundError


@pytest.fixture(scope='session', autouse=True)
def setup_rest_test_session():
    yield


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.fixture
def projects_router() -> tuple[ProjectsRouter, MagicMock]:
    auth = SimpleNamespace(authenticate_request_header=lambda: None)
    driver = MagicMock(spec=ProjectsDriver)
    driver.validate_project_name = AsyncMock(
        return_value=ValidateNameResponse(is_valid=True, error_msg=None))
    return ProjectsRouter(auth, driver), driver


@pytest.mark.asyncio
async def test_validate_project_name_is_group_neutral(projects_router) -> None:
    router, driver = projects_router

    response = await router.validate_project_name(name='Available')

    assert response == ValidateNameResponse(is_valid=True, error_msg=None)
    driver.validate_project_name.assert_awaited_once_with(name='Available')


def test_validate_project_name_openapi_has_no_group_parameter(
        projects_router) -> None:
    router, _ = projects_router
    app = FastAPI()
    app.include_router(router.router)

    operation = app.openapi(
    )['paths']['/v0/projects/utils/validate_name']['post']
    assert [(parameter['name'], parameter['in'])
            for parameter in operation['parameters']] == [('name', 'query')]


@pytest.mark.asyncio
async def test_internal_project_id_and_slug_lookup_share_safe_not_found(
        projects_router) -> None:
    router, driver = projects_router
    error = ProjectNotFoundError('Project not found')
    driver._get_project = AsyncMock(side_effect=error)
    driver.get_project_by_slug = AsyncMock(side_effect=error)

    with pytest.raises(ProjectNotFoundError, match='Project not found'):
        await router.get_project('project_' + 'f' * 32)
    with pytest.raises(ProjectNotFoundError, match='Project not found'):
        await router.get_project_by_slug('hidden-project')
