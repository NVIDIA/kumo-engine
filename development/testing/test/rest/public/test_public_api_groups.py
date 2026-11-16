# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for public Groups router (GET /v1/groups)."""
from datetime import datetime, timezone
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rest.config.group import ListUserKumoGroupsResponse
from rest.deps.authentication import Authentication
from rest.driver.groups_driver import GroupsDriver
from rest.public.routers.groups import GroupsRouter
from statedb.config.common import GroupID
from statedb.config.timestamp import ResourceTimestamps
from statedb.rest.group import KumoGroupDBEntry
from temporalio_commons import RpcError


def _noop_auth():
    """No-op dependency override for api_key_auth."""


@pytest.fixture
def groups_driver():
    driver = MagicMock(spec=GroupsDriver)
    driver.list_user_groups = AsyncMock()
    driver.get_kumo_group_by_slug = AsyncMock()
    return driver


@pytest.fixture
def auth():
    a = MagicMock(spec=Authentication)
    return a


@pytest.fixture
def client(groups_driver, auth):
    app = FastAPI()
    router = GroupsRouter(
        api_key_auth=_noop_auth,
        groups_driver=groups_driver,
        auth=auth,
    )
    app.include_router(router.router, prefix="/v1")
    with TestClient(app) as c:
        yield c


def test_list_user_groups_success(client, groups_driver):
    gid_1 = GroupID(f"group_{'a' * 32}")
    gid_2 = GroupID(f"group_{'b' * 32}")
    groups_driver.list_user_groups.return_value = [
        ListUserKumoGroupsResponse(group_id=gid_1, slug='team-a',
                                   display_name="team-a"),
        ListUserKumoGroupsResponse(group_id=gid_2, slug='team-b',
                                   display_name="team-b"),
    ]
    resp = client.get("/v1/groups")
    assert resp.status_code == HTTPStatus.OK
    data = resp.json()
    assert len(data) == 2
    assert data[0] == {
        "id": str(gid_1),
        "slug": "team-a",
        "display_name": "team-a",
    }
    assert data[0]["display_name"] == "team-a"
    assert data[1]["id"] == str(gid_2)


def test_get_group_by_slug_success(client, groups_driver):
    group_id = GroupID(f"group_{'a' * 32}")
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    groups_driver.get_kumo_group_by_slug.return_value = KumoGroupDBEntry(
        id=group_id,
        slug='team-alpha',
        display_name='Team Alpha',
        normalized_display_name='team alpha',
        timestamps=ResourceTimestamps(created_at=now, updated_at=now),
    )

    resp = client.get('/v1/groups/by-slug/team-alpha')

    assert resp.status_code == HTTPStatus.OK
    assert resp.json() == {
        'id': str(group_id),
        'slug': 'team-alpha',
        'display_name': 'Team Alpha',
    }
    groups_driver.get_kumo_group_by_slug.assert_awaited_once_with('team-alpha')


def test_list_user_groups_empty(client, groups_driver):
    groups_driver.list_user_groups.return_value = []
    resp = client.get("/v1/groups")
    assert resp.status_code == HTTPStatus.OK
    assert resp.json() == []


def test_list_user_groups_rpc_error_becomes_http_error(client, groups_driver):
    groups_driver.list_user_groups.side_effect = RpcError(
        rpc_service="GroupsService",
        rpc_method="list_user_groups",
        error_type="ValueError",
        message="something broke",
        stack_trace="",
        non_retryable=True,
    )
    resp = client.get("/v1/groups")
    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_list_user_groups_rpc_key_error_becomes_404(client, groups_driver):
    groups_driver.list_user_groups.side_effect = RpcError(
        rpc_service="GroupsService",
        rpc_method="list_user_groups",
        error_type="KeyError",
        message="not found",
        stack_trace="",
        non_retryable=True,
    )
    resp = client.get("/v1/groups")
    assert resp.status_code == HTTPStatus.NOT_FOUND
