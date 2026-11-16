# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import cachetools
import pytest
import pytest_asyncio

from rest.config.encryption import decrypt_secret_string, encrypt_secret_string
from rest.deps import direct_connect_oidc_client as oidc_module
from rest.deps.direct_connect_group_resolver import (
    _GROUP_CACHE_TTL_SECONDS,
    DirectConnectGroupResolver,
)
from rest.deps.direct_connect_oidc_client import DirectConnectOIDCGroupRefresh
from rest.driver.direct_connect_auth_storage import DirectConnectAuthStore
from rest.utils.driver_common import get_current_timestamps
from statedb.config.direct_connect_auth import DirectConnectAuthConfigRecord
from statedb.config.timestamp import now_utc
from statedb.db import StateDB, StateServices


@pytest_asyncio.fixture
async def store(tmp_path):
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    services = StateServices(db)
    result = DirectConnectAuthStore(services.direct_connect_auth)
    await result.set_config(
        DirectConnectAuthConfigRecord(
            timestamps=get_current_timestamps(),
            issuer="https://idp.example.com",
            client_id="kumo",
            encrypted_client_secret=encrypt_secret_string("client-secret"),
            groups_claim="groups",
            admin_emails=[],
        ))
    try:
        yield result, services
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_sessions_never_persist_provider_membership(store, monkeypatch):
    auth_store, services = store
    user_id, _ = await auth_store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="subject-1",
        email="user@example.com",
        display_name="User",
        refresh_token="initial-refresh-token",
        email_verified=True,
    )
    session_id = await auth_store.create_session(
        user_id=user_id,
        expires_time=now_utc(),
    )

    session = await auth_store.get_session(session_id)
    assert session is not None
    assert not hasattr(session, "idp_group_hashes")

    async with services.state_db.read_tx() as tx:
        linked = await tx.fetchrow(
            "SELECT payload FROM direct_connect_identity_links LIMIT 1")
        session_row = await tx.fetchrow(
            "SELECT id, payload FROM direct_connect_sessions LIMIT 1")
    assert linked is not None
    assert "initial-refresh-token" not in str(linked)
    assert session_row is not None
    assert session_row["id"] != session_id
    assert session_id not in session_row["payload"]
    assert "group" not in str(session_row).lower()

    class _FakeClient:
        def __init__(self, **_kwargs):
            self.refresh_groups = SimpleNamespace(
                coroutine=self._refresh_groups)

        async def _refresh_groups(self, *, refresh_token):
            assert refresh_token == "initial-refresh-token"
            return DirectConnectOIDCGroupRefresh(
                external_group_ids=["engineering"],
                refresh_token="rotated-refresh-token",
            )

    monkeypatch.setattr(oidc_module, "DirectConnectOIDCClient", _FakeClient)

    assert await auth_store.fetch_external_group_ids(user_id) == ["engineering"]
    async with services.direct_connect_auth.read() as reader:
        linked = await reader.get_identity_link_by_user_id(user_id)
    assert linked is not None
    assert decrypt_secret_string(
        linked[1].encrypted_refresh_token) == "rotated-refresh-token"


async def _assert_identity_rotation_replaces_old_link(
    first_store: DirectConnectAuthStore,
    services: StateServices,
    monkeypatch,
    *,
    second_store: DirectConnectAuthStore | None = None,
) -> None:
    user_id, _ = await first_store.upsert_user_from_idp(
        issuer="https://old-idp.example.com",
        subject="old-subject",
        email="rotated@example.com",
        display_name="Rotated User",
        refresh_token="old-refresh-token",
        email_verified=True,
    )
    config = await first_store.get_config()
    assert config is not None
    config.issuer = "https://new-idp.example.com"
    config.client_id = "rotated-client"
    config.encrypted_client_secret = encrypt_secret_string(
        "rotated-client-secret")
    await first_store.set_config(config)

    login_tokens = {"new-refresh-token"}
    if second_store is None:
        relinked_user_id, _ = await first_store.upsert_user_from_idp(
            issuer=config.issuer,
            subject="new-subject",
            email="rotated@example.com",
            display_name="Rotated User",
            refresh_token="new-refresh-token",
            email_verified=True,
        )
        assert relinked_user_id == user_id
    else:
        login_tokens.add("concurrent-refresh-token")
        relinked = await asyncio.gather(
            first_store.upsert_user_from_idp(
                issuer=config.issuer,
                subject="new-subject",
                email="rotated@example.com",
                display_name="Rotated User",
                refresh_token="new-refresh-token",
                email_verified=True,
            ),
            second_store.upsert_user_from_idp(
                issuer=config.issuer,
                subject="new-subject",
                email="rotated@example.com",
                display_name="Rotated User",
                refresh_token="concurrent-refresh-token",
                email_verified=True,
            ),
        )
        assert {item[0] for item in relinked} == {user_id}

    async with services.direct_connect_auth.read() as reader:
        assert await reader.get_user_id_by_identity(
            "https://old-idp.example.com", "old-subject") is None
        assert await reader.get_user_id_by_identity(
            config.issuer, "new-subject") == user_id
        linked = await reader.get_identity_link_by_user_id(user_id)
    assert linked is not None
    assert decrypt_secret_string(
        linked[1].encrypted_refresh_token) in login_tokens

    async with services.state_db.read_tx() as tx:
        rows = await tx.fetch(
            "SELECT payload FROM direct_connect_identity_links")
    assert len(rows) == 1
    persisted_payload = str(rows[0])
    for token in (*login_tokens, "old-refresh-token"):
        assert token not in persisted_payload

    class _RotatedClient:
        def __init__(self, **kwargs):
            assert kwargs == {
                "issuer": "https://new-idp.example.com",
                "client_id": "rotated-client",
                "client_secret": "rotated-client-secret",
                "groups_claim": "groups",
            }
            self.refresh_groups = SimpleNamespace(
                coroutine=self._refresh_groups)

        async def _refresh_groups(self, *, refresh_token):
            assert refresh_token in login_tokens
            return DirectConnectOIDCGroupRefresh(
                external_group_ids=["new-idp-group"],
                refresh_token="post-refresh-token",
            )

    class _GroupService:
        async def resolve_native_memberships(self, *_args, **_kwargs):
            return []

        async def map_idp_group_ids(self, external_ids, identity_provider_id):
            assert identity_provider_id == "direct_connect"
            return list(external_ids)

    monkeypatch.setattr(oidc_module, "DirectConnectOIDCClient",
                        _RotatedClient)
    now = [0.0]
    cache = cachetools.TTLCache(
        maxsize=10,
        ttl=_GROUP_CACHE_TTL_SECONDS,
        timer=lambda: now[0],
    )
    resolver = DirectConnectGroupResolver(
        state_services=SimpleNamespace(groups=_GroupService()),
        auth_store=first_store,
        cache=cache,
    )
    resolver.remember_external_groups(user_id, ["old-idp-group"])
    assert (await resolver.resolve_groups(user_id, {})).group_ids == (
        "old-idp-group", )

    now[0] = _GROUP_CACHE_TTL_SECONDS
    assert (await resolver.resolve_groups(user_id, {})).group_ids == (
        "new-idp-group", )
    async with services.direct_connect_auth.read() as reader:
        linked = await reader.get_identity_link_by_user_id(user_id)
    assert linked is not None
    assert decrypt_secret_string(
        linked[1].encrypted_refresh_token) == "post-refresh-token"
    async with services.state_db.read_tx() as tx:
        rows = await tx.fetch(
            "SELECT payload FROM direct_connect_identity_links")
    assert len(rows) == 1
    persisted_payload = str(rows[0])
    for token in (*login_tokens, "old-refresh-token", "post-refresh-token"):
        assert token not in persisted_payload


@pytest.mark.asyncio
async def test_sqlite_identity_rotation_replaces_old_link_and_refreshes(
        store, monkeypatch):
    auth_store, services = store
    await _assert_identity_rotation_replaces_old_link(
        auth_store,
        services,
        monkeypatch,
    )


async def _assert_refresh_is_serialized_across_stores(
    first_store: DirectConnectAuthStore,
    second_store: DirectConnectAuthStore,
    services: StateServices,
    monkeypatch,
) -> None:
    user_id, _ = await first_store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="concurrent-subject",
        email="concurrent@example.com",
        display_name="Concurrent User",
        refresh_token="initial-refresh-token",
        email_verified=True,
    )
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    refresh_tokens: list[str] = []
    active_calls = 0
    max_active_calls = 0

    class _RotatingClient:
        def __init__(self, **_kwargs):
            self.refresh_groups = SimpleNamespace(
                coroutine=self._refresh_groups)

        async def _refresh_groups(self, *, refresh_token):
            nonlocal active_calls, max_active_calls
            call_number = len(refresh_tokens) + 1
            refresh_tokens.append(refresh_token)
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
            try:
                if call_number == 1:
                    first_started.set()
                    await asyncio.wait_for(release_first.wait(), timeout=5)
                return DirectConnectOIDCGroupRefresh(
                    external_group_ids=[f"group-{call_number}"],
                    refresh_token=f"rotated-refresh-token-{call_number}",
                )
            finally:
                active_calls -= 1

    monkeypatch.setattr(oidc_module, "DirectConnectOIDCClient",
                        _RotatingClient)
    first_task = asyncio.create_task(
        first_store.fetch_external_group_ids(user_id))
    await asyncio.wait_for(first_started.wait(), timeout=5)
    second_task = asyncio.create_task(
        second_store.fetch_external_group_ids(user_id))
    await asyncio.sleep(0.05)

    assert refresh_tokens == ["initial-refresh-token"]
    release_first.set()
    results = await asyncio.wait_for(
        asyncio.gather(first_task, second_task), timeout=5)

    assert results == [["group-1"], ["group-2"]]
    assert refresh_tokens == [
        "initial-refresh-token",
        "rotated-refresh-token-1",
    ]
    assert max_active_calls == 1
    async with services.direct_connect_auth.read() as reader:
        linked = await reader.get_identity_link_by_user_id(user_id)
    assert linked is not None
    assert decrypt_secret_string(linked[1].encrypted_refresh_token) == (
        "rotated-refresh-token-2")


@pytest.mark.asyncio
async def test_sqlite_refresh_is_serialized_across_service_instances(
        store, monkeypatch):
    first_store, first_services = store
    second_db = await StateDB.open_sqlite(first_services.state_db.target)
    second_services = StateServices(second_db)
    second_store = DirectConnectAuthStore(second_services.direct_connect_auth)
    try:
        await _assert_refresh_is_serialized_across_stores(
            first_store,
            second_store,
            first_services,
            monkeypatch,
        )
    finally:
        await second_db.close()


@pytest.mark.asyncio
async def test_postgres_refresh_is_serialized_across_service_instances(
    postgres_dsn,
    postgres_connection,
    monkeypatch,
):
    await postgres_connection.execute("DROP SCHEMA public CASCADE")
    await postgres_connection.execute("CREATE SCHEMA public")
    first_db = await StateDB.open_postgres(postgres_dsn)
    second_db = await StateDB.open_postgres(postgres_dsn)
    first_services = StateServices(first_db)
    second_services = StateServices(second_db)
    first_store = DirectConnectAuthStore(first_services.direct_connect_auth)
    second_store = DirectConnectAuthStore(second_services.direct_connect_auth)
    await first_store.set_config(
        DirectConnectAuthConfigRecord(
            timestamps=get_current_timestamps(),
            issuer="https://idp.example.com",
            client_id="kumo",
            encrypted_client_secret=encrypt_secret_string("client-secret"),
            groups_claim="groups",
            admin_emails=[],
        ))
    try:
        await _assert_refresh_is_serialized_across_stores(
            first_store,
            second_store,
            first_services,
            monkeypatch,
        )
    finally:
        await first_db.close()
        await second_db.close()
        await postgres_connection.execute("DROP SCHEMA public CASCADE")
        await postgres_connection.execute("CREATE SCHEMA public")


@pytest.mark.asyncio
async def test_postgres_concurrent_identity_rotation_replaces_old_link(
    postgres_dsn,
    postgres_connection,
    monkeypatch,
):
    await postgres_connection.execute("DROP SCHEMA public CASCADE")
    await postgres_connection.execute("CREATE SCHEMA public")
    first_db = await StateDB.open_postgres(postgres_dsn)
    second_db = await StateDB.open_postgres(postgres_dsn)
    first_services = StateServices(first_db)
    second_services = StateServices(second_db)
    first_store = DirectConnectAuthStore(first_services.direct_connect_auth)
    second_store = DirectConnectAuthStore(second_services.direct_connect_auth)
    await first_store.set_config(
        DirectConnectAuthConfigRecord(
            timestamps=get_current_timestamps(),
            issuer="https://old-idp.example.com",
            client_id="kumo",
            encrypted_client_secret=encrypt_secret_string("client-secret"),
            groups_claim="groups",
            admin_emails=[],
        ))
    try:
        await _assert_identity_rotation_replaces_old_link(
            first_store,
            first_services,
            monkeypatch,
            second_store=second_store,
        )
    finally:
        await first_db.close()
        await second_db.close()
        await postgres_connection.execute("DROP SCHEMA public CASCADE")
        await postgres_connection.execute("CREATE SCHEMA public")
