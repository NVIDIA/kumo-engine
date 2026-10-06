# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from http import HTTPStatus

from fastapi.testclient import TestClient
from kumoapi.data_source import DEFAULT_GROUP_ID, DEFAULT_PROJECT_ID

from rest.deps.authentication import Permissions, UserAuthorization
from rest.driver.api_key_driver import APIKeyDriver
from statedb.config.common import UserID
from statedb.db.services import StateServices


def test_staged_user_key_can_discover_canonical_default_project(
    public_api_client: TestClient,
    state_services: StateServices,
    default_project_id: str,
) -> None:
    """The SDK bootstrap key is an ordinary authenticated user key."""
    assert default_project_id == DEFAULT_PROJECT_ID
    api_keys = APIKeyDriver()
    api_keys.configure_state_services(state_services)
    created = asyncio.run(
        api_keys.create_user_api_key(
            name='sdk-integration-bootstrap',
            user_id='phase1b-api-key-staging',
            user_name='phase1b-api-key-staging',
        ))

    headers = {'X-API-Key': created.value}
    listed = public_api_client.get('/v1/projects', headers=headers)
    assert listed.status_code == HTTPStatus.OK, listed.text
    assert DEFAULT_PROJECT_ID in {project['id'] for project in listed.json()}

    response = public_api_client.get(f'/v1/projects/{DEFAULT_PROJECT_ID}',
                                     headers=headers)
    assert response.status_code == HTTPStatus.OK, response.text
    assert response.json()['group_id'] == DEFAULT_GROUP_ID


def test_disabled_auth_request_sees_all_projects_including_private(
    unauthenticated_client: TestClient,
    public_api_client: TestClient,
    default_project_id: str,
) -> None:
    """Auth-disabled requests are the anonymous administrator, which has no
    identity boundary, so they see every project -- including
    private ones."""
    assert default_project_id == DEFAULT_PROJECT_ID
    state_services = unauthenticated_client.app_builder._state_services
    assert state_services is not None
    asyncio.run(state_services.groups.ensure_default_resources())
    admin = UserAuthorization(
        user_id=UserID('project-owner'),
        access_token='test-owner',
        permissions=[
            Permissions.READ.compose(),
            Permissions.WRITE.compose(),
            Permissions.ADMIN.compose(),
        ],
        kumo_group_ids=(DEFAULT_GROUP_ID, ),
    )
    private_project = asyncio.run(
        state_services.projects.create_project(
            name='Private disabled-auth boundary',
            group_id=DEFAULT_GROUP_ID,
            is_private=True,
            user_authorization=admin,
        ))

    private_default = unauthenticated_client.get(
        f'/v0/projects/{DEFAULT_PROJECT_ID}')
    public_default = public_api_client.get(f'/v1/projects/{DEFAULT_PROJECT_ID}')
    assert private_default.status_code == HTTPStatus.OK, private_default.text
    assert public_default.status_code == HTTPStatus.OK, public_default.text

    private_visible = unauthenticated_client.get(
        f'/v0/projects/{private_project.id}')
    public_visible = public_api_client.get(f'/v1/projects/{private_project.id}')
    assert private_visible.status_code == HTTPStatus.OK, private_visible.text
    assert public_visible.status_code == HTTPStatus.OK, public_visible.text
