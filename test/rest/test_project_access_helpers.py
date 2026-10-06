# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from http import HTTPStatus

import pytest
from fastapi import HTTPException

from rest.deps.authentication import RequestContext, UserAuthorization
from rest.public.routers.common import (
    filter_by_project,
    user_can_access_project,
)
from statedb.config.common import UserID


@pytest.fixture(autouse=True, scope="session")
def setup_rest_test_session():
    """These pure unit tests do not need the shared Temporal harness."""
    yield


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@dataclass(frozen=True)
class _Resource:
    project_id: str


def _context(*project_ids: str) -> RequestContext:
    return RequestContext(user_authorization=UserAuthorization(
        user_id=UserID("user"),
        access_token="token",
        permissions=("read", "write"),
        kumo_group_ids=("native-group", ),
        project_ids=tuple(project_ids),
        external_membership_unavailable=True,
    ))


def test_provider_outage_keeps_native_project_list_access() -> None:
    ctx = _context("native-project")
    resources = [_Resource("native-project"), _Resource("other-project")]

    assert filter_by_project(resources, ctx, lambda item: item.project_id) == [
        _Resource("native-project")
    ]
    assert user_can_access_project(ctx, "native-project")


def test_provider_outage_returns_503_only_for_unresolved_project_access(
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        user_can_access_project(_context("native-project"), "external-project")

    assert exc_info.value.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert "membership" in str(exc_info.value.detail).lower()
