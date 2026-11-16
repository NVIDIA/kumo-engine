# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cachetools
import pytest

from rest.deps.direct_connect_group_resolver import (
    _GROUP_CACHE_TTL_SECONDS,
    DirectConnectGroupResolver,
)
from rest.deps.group_membership import ResolvedGroupMembership


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.fixture
def setup_rest_test_session() -> None:
    """This pure unit module does not need the shared Temporal harness."""
    return None


@pytest.fixture(scope='session', autouse=True)
def temporal_workers_ready():
    yield True


class _GroupService:
    def __init__(self, mapping: dict[str, str], native=()) -> None:
        self.mapping = mapping
        self.native = list(native)
        self.map_calls = []

    async def resolve_native_memberships(self, *_args, **_kwargs):
        return list(self.native)

    async def map_idp_group_ids(self, external_ids, identity_provider_id):
        self.map_calls.append((list(external_ids), identity_provider_id))
        return [
            self.mapping[group_id] for group_id in external_ids
            if group_id in self.mapping
        ]


def _state_services(group_service):
    return type("_StateServices", (), {"groups": group_service})()


class _AuthStore:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def fetch_external_group_ids(self, user_id):
        self.calls.append(user_id)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.mark.asyncio
async def test_provider_results_are_cached_but_mapping_is_live():
    groups = _GroupService({"idp-a": "kumo-a"})
    store = _AuthStore([["idp-a"]])
    resolver = DirectConnectGroupResolver(
        state_services=_state_services(groups),
        auth_store=store,
        cache={},
    )

    first = await resolver.resolve_groups("user", {})
    groups.mapping = {"idp-a": "kumo-b"}
    second = await resolver.resolve_groups("user", {})

    assert first == ResolvedGroupMembership(("kumo-a", ))
    assert second == ResolvedGroupMembership(("kumo-b", ))
    assert store.calls == ["user"]
    assert groups.map_calls == [
        (["idp-a"], "direct_connect"),
        (["idp-a"], "direct_connect"),
    ]


@pytest.mark.asyncio
async def test_multiple_external_groups_mapping_to_one_kumo_group_are_deduped():
    groups = _GroupService({"idp-a": "kumo-a", "idp-b": "kumo-a"})
    resolver = DirectConnectGroupResolver(
        state_services=_state_services(groups),
        auth_store=_AuthStore([["idp-a", "idp-b"]]),
        cache={},
    )

    result = await resolver.resolve_groups("user", {})

    assert result == ResolvedGroupMembership(("kumo-a", ))
    assert groups.map_calls == [
        (["idp-a", "idp-b"], "direct_connect"),
    ]


@pytest.mark.asyncio
async def test_negative_provider_result_is_cached():
    groups = _GroupService({})
    store = _AuthStore([[]])
    resolver = DirectConnectGroupResolver(
        state_services=_state_services(groups),
        auth_store=store,
        cache={},
    )

    assert await resolver.resolve_groups("user", {}) == ResolvedGroupMembership(
        ())
    assert await resolver.resolve_groups("user", {}) == ResolvedGroupMembership(
        ())
    assert store.calls == ["user"]


@pytest.mark.asyncio
async def test_native_membership_survives_provider_outage():
    groups = _GroupService({}, native=("native", ))
    resolver = DirectConnectGroupResolver(
        state_services=_state_services(groups),
        auth_store=_AuthStore([RuntimeError("provider unavailable")]),
        cache={},
    )

    result = await resolver.resolve_groups("user", {})

    assert result == ResolvedGroupMembership(
        ("native", ), external_membership_unavailable=True)


@pytest.mark.asyncio
async def test_native_only_user_does_not_depend_on_provider():
    groups = _GroupService({}, native=("native", ))
    resolver = DirectConnectGroupResolver(
        state_services=_state_services(groups),
        auth_store=_AuthStore([None]),
        cache={},
    )

    result = await resolver.resolve_groups("password-user", {})

    assert result == ResolvedGroupMembership(("native", ))


@pytest.mark.asyncio
async def test_absent_external_adapter_is_a_valid_native_only_deployment():
    groups = _GroupService({}, native=("native", ))
    resolver = DirectConnectGroupResolver(
        state_services=_state_services(groups),
        auth_store=None,
        cache={},
    )

    assert await resolver.resolve_groups("user", {}) == ResolvedGroupMembership(
        ("native", ))


@pytest.mark.asyncio
async def test_direct_connect_positive_and_negative_results_expire_by_five_minutes(
):
    now = [0.0]
    cache = cachetools.TTLCache(
        maxsize=10,
        ttl=_GROUP_CACHE_TTL_SECONDS,
        timer=lambda: now[0],
    )
    groups = _GroupService({"idp-a": "mapped"})
    store = _AuthStore([["idp-a"], [], RuntimeError("provider unavailable")])
    resolver = DirectConnectGroupResolver(
        state_services=_state_services(groups),
        auth_store=store,
        cache=cache,
    )

    assert await resolver.resolve_groups("user", {}) == ResolvedGroupMembership(
        ("mapped", ))
    now[0] = _GROUP_CACHE_TTL_SECONDS
    assert await resolver.resolve_groups("user", {}) == ResolvedGroupMembership(
        ())
    now[0] = 2 * _GROUP_CACHE_TTL_SECONDS
    assert await resolver.resolve_groups("user", {}) == ResolvedGroupMembership(
        (), external_membership_unavailable=True)
    assert store.calls == ["user", "user", "user"]


@pytest.mark.asyncio
async def test_verified_login_seeds_only_transient_cache():
    groups = _GroupService({"idp-login": "mapped"})
    store = _AuthStore([RuntimeError("must not fetch")])
    resolver = DirectConnectGroupResolver(
        state_services=_state_services(groups),
        auth_store=store,
        cache={},
    )
    resolver.remember_external_groups("user",
                                      [" idp-login ", "idp-login", "   "])

    result = await resolver.resolve_groups("user", {})

    assert result == ResolvedGroupMembership(("mapped", ))
    assert store.calls == []
    assert groups.map_calls == [(["idp-login"], "direct_connect")]
