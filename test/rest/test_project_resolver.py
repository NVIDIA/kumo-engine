# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest

from rest.deps.project_resolver import ProjectResolver


@pytest.fixture(autouse=True, scope="session")
def setup_rest_test_session():
    """This pure unit module does not need the shared Temporal harness."""
    yield


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@dataclass(frozen=True)
class _Principal:
    user_id: str = "user"
    kumo_group_ids: tuple[str, ...] = ("group", )


class _ProjectService:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def list_visible_project_ids(self, principal):
        self.calls.append(principal)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _state_services(projects):
    return type("_StateServices", (), {"projects": projects})()


@pytest.mark.asyncio
async def test_resolves_current_visible_projects_from_state_services():
    service = _ProjectService([["project-a"]])
    principal = _Principal()
    resolver = ProjectResolver(_state_services(service))

    assert await resolver.resolve_projects(principal) == ["project-a"]
    assert service.calls == [principal]


@pytest.mark.asyncio
async def test_project_state_is_not_cached_between_requests():
    service = _ProjectService([["project-a"], ["project-b"]])
    principal = _Principal()
    resolver = ProjectResolver(_state_services(service))

    assert await resolver.resolve_projects(principal) == ["project-a"]
    assert await resolver.resolve_projects(principal) == ["project-b"]
    assert len(service.calls) == 2


@pytest.mark.asyncio
async def test_configure_state_services_enables_late_binding():
    service = _ProjectService([[]])
    resolver = ProjectResolver()
    resolver.configure_state_services(_state_services(service))

    assert await resolver.resolve_projects(_Principal()) == []


@pytest.mark.asyncio
async def test_missing_service_and_service_failure_are_explicit():
    with pytest.raises(RuntimeError, match="not configured"):
        await ProjectResolver().resolve_projects(_Principal())

    resolver = ProjectResolver(
        _state_services(_ProjectService([RuntimeError("StateDB unavailable")])))
    with pytest.raises(RuntimeError, match="StateDB unavailable"):
        await resolver.resolve_projects(_Principal())
