# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields

import pytest
from fastapi import HTTPException

from rest.config.api_key import APIKeyAuthIdentity, APIKeyFormat, ParsedAPIKey
from rest.config.status import RestStatusCode
from rest.deps import authentication as auth_deps
from rest.deps.authentication import AUTHORIZATION_DISABLED
from statedb.config.common import UserID
from statedb.config.direct_connect_auth import DirectConnectSessionRecord
from statedb.rest.api_key import APIKeyMetadataRecord


def test_owner_bound_api_key_round_trip() -> None:
    parsed = ParsedAPIKey.parse("deadbeef:named:secret")
    assert parsed.format == APIKeyFormat.OWNER_BOUND
    assert parsed.user_hash == "deadbeef"
    assert parsed.name == "named"
    assert parsed.secret == "secret"


def test_legacy_prefixed_api_key_round_trip() -> None:
    parsed = ParsedAPIKey.parse("legacy:deadbeef:named:secret")
    assert parsed.format == APIKeyFormat.LEGACY_PREFIXED
    assert parsed.user_hash == "deadbeef"
    assert parsed.name == "named"
    assert parsed.secret == "secret"
    assert not hasattr(parsed, "legacy_prefix")


@pytest.mark.parametrize(
    "api_key",
    [
        "", "secret", "named:secret", "bad-hash:named:secret",
        "deadbeef::secret", "prefix:deadbeef:named:secret:extra"
    ],
)
def test_invalid_api_key_formats_are_rejected(api_key: str) -> None:
    with pytest.raises(ValueError):
        ParsedAPIKey.parse(api_key)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "is_admin,expected_permissions",
    [(False, ["read", "write"]), (True, ["read", "write", "admin"])],
)
async def test_api_key_resolves_owner_authorization(
    is_admin: bool,
    expected_permissions: list[str],
) -> None:
    class FakeDriver:
        async def verify_api_key(self, api_key):
            assert api_key == "deadbeef:named:secret"
            return APIKeyAuthIdentity(
                user_id="direct_connect|owner1",
                is_admin=is_admin,
            )

    principal = await auth_deps.resolve_user_authorization_from_api_key(
        api_key="deadbeef:named:secret",
        api_key_driver=FakeDriver(),
    )
    assert principal.user_id == UserID("direct_connect|owner1")
    assert principal.permissions == expected_permissions


@pytest.mark.asyncio
async def test_missing_api_key_identity_is_rejected() -> None:
    class FakeDriver:
        async def verify_api_key(self, _api_key):
            return None

    with pytest.raises(HTTPException) as error:
        await auth_deps.resolve_user_authorization_from_api_key(
            api_key="deadbeef:named:secret",
            api_key_driver=FakeDriver(),
        )
    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_disabled_auth_falls_back_for_non_key_input() -> None:
    principal = await auth_deps.resolve_user_authorization_from_api_key(
        api_key="not-a-key",
        auth_disabled=True,
    )
    assert principal.user_id == UserID(AUTHORIZATION_DISABLED)
    assert principal.is_disabled
    assert principal.permissions == ["read", "write", "admin"]


@pytest.mark.asyncio
async def test_invalid_key_is_rejected_when_auth_is_enabled() -> None:
    with pytest.raises(HTTPException) as error:
        await auth_deps.resolve_user_authorization_from_api_key(
            api_key="not-a-key", )
    assert error.value.status_code == 401
    assert error.value.detail == RestStatusCode.AUTH_INVALID_API_KEY.value


def test_api_keys_and_sessions_never_persist_membership_snapshots() -> None:
    for record_type in (APIKeyMetadataRecord, DirectConnectSessionRecord):
        names = {field.name for field in fields(record_type)}
        assert "kumo_group_ids" not in names
        assert "external_group_ids" not in names
