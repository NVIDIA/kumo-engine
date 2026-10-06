# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from rest.config.group import (
    AddGroupMemberRequest,
    CreateExternalGroupMappingRequest,
    ReplaceExternalGroupMappingsRequest,
    ReplaceGroupMembersRequest,
)
from rest.driver.groups_driver import GroupsDriver
from rest.routers.groups import GroupsRouter
from statedb.config.common import GroupID
from statedb.utils.errors import (
    GroupInputError,
    GroupNotFoundError,
    GroupRelationConflictError,
)


@pytest.fixture(scope='session', autouse=True)
def setup_rest_test_session():
    yield


@pytest.fixture(scope='session', autouse=True)
def temporal_workers_ready():
    yield True


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.fixture
def groups_router() -> tuple[GroupsRouter, MagicMock]:
    auth = SimpleNamespace(
        authenticate_request_header=lambda: None,
        require_customer_admin=lambda: None,
    )
    driver = MagicMock(spec=GroupsDriver)
    return GroupsRouter(auth, driver), driver


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('error', 'status'),
    [
        (GroupInputError('principal_id cannot be empty'),
         HTTPStatus.BAD_REQUEST),
        (GroupRelationConflictError('membership changed concurrently'),
         HTTPStatus.CONFLICT),
    ],
)
async def test_add_member_maps_input_and_concurrency_errors(
        groups_router, error, status) -> None:
    router, driver = groups_router
    driver.add_member = AsyncMock(side_effect=error)

    with pytest.raises(HTTPException) as caught:
        await router.add_member(GroupID.generate(),
                                AddGroupMemberRequest(principal_id='user'))

    assert caught.value.status_code == status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('error', 'status'),
    [
        (GroupInputError('identity_provider_id cannot be empty'),
         HTTPStatus.BAD_REQUEST),
        (GroupRelationConflictError('external group already mapped'),
         HTTPStatus.CONFLICT),
        (GroupNotFoundError('group not found'), HTTPStatus.NOT_FOUND),
    ],
)
async def test_external_mapping_maps_input_and_concurrency_errors(
        groups_router, error, status) -> None:
    router, driver = groups_router
    driver.create_external_mapping = AsyncMock(side_effect=error)

    with pytest.raises(HTTPException) as caught:
        await router.create_external_mapping(
            GroupID.generate(),
            CreateExternalGroupMappingRequest(
                identity_provider_id='direct_connect',
                external_group_id='external',
            ),
        )

    assert caught.value.status_code == status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('error', 'status'),
    [
        (GroupInputError('external_group_id cannot be empty'),
         HTTPStatus.BAD_REQUEST),
        (GroupRelationConflictError('external group already mapped'),
         HTTPStatus.CONFLICT),
        (GroupNotFoundError('group not found'), HTTPStatus.NOT_FOUND),
    ],
)
async def test_replace_external_mappings_maps_atomic_service_errors(
        groups_router, error, status) -> None:
    router, driver = groups_router
    driver.replace_external_mappings = AsyncMock(side_effect=error)

    with pytest.raises(HTTPException) as caught:
        await router.replace_external_mappings(
            GroupID.generate(),
            ReplaceExternalGroupMappingsRequest(mappings=[]),
        )

    assert caught.value.status_code == status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('error', 'status'),
    [
        (GroupInputError('principal_id cannot be empty'),
         HTTPStatus.BAD_REQUEST),
        (GroupRelationConflictError('direct membership changed'),
         HTTPStatus.CONFLICT),
        (GroupNotFoundError('group not found'), HTTPStatus.NOT_FOUND),
    ],
)
async def test_replace_members_maps_atomic_service_errors(
        groups_router, error, status) -> None:
    router, driver = groups_router
    driver.replace_members = AsyncMock(side_effect=error)

    with pytest.raises(HTTPException) as caught:
        await router.replace_members(
            GroupID.generate(),
            ReplaceGroupMembersRequest(members=[]),
        )

    assert caught.value.status_code == status


def test_bulk_replacement_routes_are_admin_only() -> None:
    async def authenticate_request_header():
        return None

    async def require_customer_admin():
        return None

    auth = SimpleNamespace(
        authenticate_request_header=authenticate_request_header,
        require_customer_admin=require_customer_admin,
    )
    driver = MagicMock(spec=GroupsDriver)
    router = GroupsRouter(auth, driver).v1_router

    assert {
        (route.path, method)
        for route in router.routes for method in route.methods
    } >= {
        ('/v1/groups/{group_id}/members', 'PUT'),
        ('/v1/groups/{group_id}/external-mappings', 'PUT'),
    }
    assert [dependency.dependency for dependency in router.dependencies] == [
        authenticate_request_header,
        require_customer_admin,
    ]
