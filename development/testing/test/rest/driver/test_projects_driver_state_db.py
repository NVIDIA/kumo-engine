# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, sentinel

import pytest

from rest.deps.authentication import Permissions, UserAuthorization
from rest.deps.context import _current_user_ctx
from rest.driver.projects_driver import ProjectsDriver
from rest.utils.routers import ValidateNameResponse
from statedb.config.common import UserID

_USER = UserAuthorization(
    user_id=UserID('user_1'),
    access_token='token',
    permissions=[
        Permissions.READ.compose(),
        Permissions.WRITE.compose(),
    ],
    kumo_group_ids=('group_' + 'a' * 32, ),
    user_name='Test User',
)
_PROJECT_ID = 'project_' + 'a' * 32
_GROUP_ID = 'group_' + 'a' * 32
_VALID_NAME = ValidateNameResponse(is_valid=True, error_msg=None)


@pytest.mark.parametrize(
    ('invoke', 'service_method', 'expected_call', 'result'),
    [
        (
            lambda driver: driver.create_project(
                name='Project',
                group_id=_GROUP_ID,
                description='Description',
                is_private=True,
            ),
            'create_project',
            call(
                name='Project',
                group_id=_GROUP_ID,
                slug=None,
                description='Description',
                is_private=True,
                user_authorization=_USER,
            ),
            sentinel.project,
        ),
        (lambda driver: driver._list_visible_projects(),
         'list_visible_projects', call(_USER), [sentinel.project]),
        (lambda driver: driver.list_projects_in_group(_GROUP_ID),
         'list_projects_in_group', call(_GROUP_ID, _USER), [sentinel.project]),
        (lambda driver: driver._get_project(_PROJECT_ID), 'get_project',
         call(_PROJECT_ID, _USER), sentinel.project),
        (lambda driver: driver.get_project_by_slug('project'),
         'get_project_by_slug', call('project', _USER), sentinel.project),
        (lambda driver: driver._archive_project(_PROJECT_ID), 'archive_project',
         call(_PROJECT_ID, _USER), sentinel.project),
        (
            lambda driver: driver._update_project(
                _PROJECT_ID,
                name='Renamed',
                description='Updated',
                is_private=False,
            ),
            'update_project',
            call(
                project_id=_PROJECT_ID,
                name='Renamed',
                description='Updated',
                is_private=False,
                user_authorization=_USER,
            ),
            sentinel.project,
        ),
        (
            lambda driver: driver.validate_project_name(
                name='Renamed', exclude_project_id=_PROJECT_ID),
            'validate_project_name',
            call(
                name='Renamed',
                exclude_project_id=_PROJECT_ID,
                user_authorization=_USER,
            ),
            _VALID_NAME,
        ),
        (lambda driver: driver._validate_project_for_creation(_PROJECT_ID),
         'validate_project_for_creation', call(_PROJECT_ID,
                                               _USER), sentinel.project),
    ],
    ids=(
        'create',
        'list-visible',
        'list-group',
        'get-id',
        'get-slug',
        'archive',
        'update',
        'validate-name',
        'validate-creation',
    ),
)
@pytest.mark.asyncio
async def test_projects_driver_threads_principal_to_state_service(
    invoke,
    service_method,
    expected_call,
    result,
) -> None:
    service_call = AsyncMock(return_value=result)
    driver = ProjectsDriver()
    driver._configure_state_services(
        MagicMock(projects=SimpleNamespace(**{service_method: service_call})))
    token = _current_user_ctx.set(_USER)
    try:
        assert await invoke(driver) == result
    finally:
        _current_user_ctx.reset(token)

    assert service_call.await_args == expected_call
