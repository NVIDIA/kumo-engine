# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import threading
from contextlib import asynccontextmanager
from datetime import timedelta
from http import HTTPStatus

import pytest
import pytest_asyncio
from cryptography.fernet import InvalidToken
from fastapi import HTTPException

from rest.driver import direct_connect_auth_storage as storage
from rest.driver.direct_connect_auth_storage import (
    DirectConnectAuthStore,
    direct_connect_user_id_for_identity,
)
from rest.utils.driver_common import get_current_timestamps
from statedb.config.direct_connect_auth import (
    DirectConnectAuthConfigRecord,
    DirectConnectPasswordCredentialRecord,
    DirectConnectSessionRecord,
    DirectConnectUserRecord,
)
from statedb.config.timestamp import now_utc
from statedb.db import StateDB
from statedb.db.direct_connect_auth import DirectConnectAuthService


def test_password_reset_token_key_uses_keyed_cmac_lookup(monkeypatch):
    monkeypatch.setattr(
        storage,
        "_get_password_reset_token_pepper",
        lambda: "test-password-reset-pepper",
    )
    token = "reset-token-value"

    token_key = storage._password_reset_token_key(token)

    legacy_sha256_key = (
        "86e01be8f3cbb0abf8b06ae31da0fd5669ba13d80856e588941fbc543056247a")
    assert token_key == "v2:d1RPAOcK256MmWyEGs5Lqw"
    assert token_key != legacy_sha256_key
    assert token_key != token
    assert storage._password_reset_token_key(token) == token_key


def test_password_reset_token_key_depends_on_token_and_pepper(monkeypatch):
    monkeypatch.setattr(
        storage,
        "_get_password_reset_token_pepper",
        lambda: "test-password-reset-pepper",
    )
    token_key = storage._password_reset_token_key("reset-token-value")
    different_token_key = storage._password_reset_token_key(
        "different-reset-token-value")

    storage._clear_password_reset_token_lookup_key_cache_for_test()
    monkeypatch.setattr(
        storage,
        "_get_password_reset_token_pepper",
        lambda: "different-password-reset-pepper",
    )
    different_pepper_key = storage._password_reset_token_key(
        "reset-token-value")

    assert different_token_key != token_key
    assert different_pepper_key != token_key


def test_password_reset_token_lookup_key_is_cached(monkeypatch):
    calls = 0

    def _pepper():
        nonlocal calls
        calls += 1
        return f"test-password-reset-pepper-{calls}"

    monkeypatch.setattr(storage, "_get_password_reset_token_pepper", _pepper)

    first = storage._password_reset_token_key("reset-token-value")
    second = storage._password_reset_token_key("reset-token-value")

    assert second == first
    assert calls == 1

    storage._clear_password_reset_token_lookup_key_cache_for_test()
    third = storage._password_reset_token_key("reset-token-value")

    assert third != first
    assert calls == 2


def test_password_reset_token_key_accepts_long_pepper(monkeypatch):
    monkeypatch.setattr(
        storage,
        "_get_password_reset_token_pepper",
        lambda: "p" * 100,
    )
    token = "reset-token-value"

    token_key = storage._password_reset_token_key(token)

    assert token_key == storage._password_reset_token_key(token)
    assert token_key.startswith("v2:")
    assert len(token_key) == 25
    assert token_key != token


class _FakeDirectConnectAuthService:
    def __init__(self) -> None:
        self.config = None
        self.users = {}
        self.sessions = {}
        self.init_calls = 0
        self.close_calls = 0
        self.read_calls = 0
        self.write_calls = 0
        self.get_config_calls = 0
        self.set_config_calls = []
        self.list_users_calls = 0
        self.get_user_calls = []
        self.get_user_by_email_calls = []
        self.get_session_calls = []
        self.pop_session_calls = []

    async def init_once(self) -> None:
        self.init_calls += 1

    async def close(self) -> None:
        self.close_calls += 1

    @asynccontextmanager
    async def read(self):
        self.read_calls += 1
        yield self

    @asynccontextmanager
    async def write(self):
        self.write_calls += 1
        yield self

    async def get_config(self):
        self.get_config_calls += 1
        return self.config

    async def set_config(self, config):
        self.set_config_calls.append(config)
        self.config = config

    async def list_users(self):
        self.list_users_calls += 1
        return list(self.users.items())

    async def get_user(self, user_id):
        self.get_user_calls.append(user_id)
        return self.users.get(user_id)

    async def get_user_by_email(self, email):
        self.get_user_by_email_calls.append(email)
        normalized_email = email.strip().lower()
        for user_id, user in self.users.items():
            if user.email.strip().lower() == normalized_email:
                return user_id, user
        return None

    async def get_session(self, session_id):
        self.get_session_calls.append(session_id)
        return self.sessions.get(session_id)

    async def pop_session(self, session_id):
        self.pop_session_calls.append(session_id)
        return self.sessions.pop(session_id, None)


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    storage._clear_password_reset_token_lookup_key_cache_for_test()
    yield
    storage._clear_password_reset_token_lookup_key_cache_for_test()


@pytest_asyncio.fixture
async def auth_store(tmp_path, monkeypatch) -> DirectConnectAuthStore:
    monkeypatch.setattr(
        "rest.driver.direct_connect_auth_storage.encrypt_secret_string",
        lambda value: value,
    )
    monkeypatch.setattr(
        "rest.driver.direct_connect_auth_storage.decrypt_secret_string",
        lambda value: value,
    )
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    store = DirectConnectAuthStore(DirectConnectAuthService(state_db))
    await store.set_config(
        DirectConnectAuthConfigRecord(
            timestamps=get_current_timestamps(),
            issuer="https://idp.example.com",
            client_id="kumo",
            encrypted_client_secret="secret",
            groups_claim="groups",
            admin_emails=["admin@example.com"],
            enabled=True,
        ))
    return store


async def _empty_auth_store(tmp_path, monkeypatch) -> DirectConnectAuthStore:
    monkeypatch.setattr(
        "rest.driver.direct_connect_auth_storage.encrypt_secret_string",
        lambda value: value,
    )
    monkeypatch.setattr(
        "rest.driver.direct_connect_auth_storage.decrypt_secret_string",
        lambda value: value,
    )
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    return DirectConnectAuthStore(DirectConnectAuthService(state_db))


@pytest.mark.asyncio
async def test_store_uses_configured_direct_connect_auth_service():
    store = DirectConnectAuthStore()
    service = _FakeDirectConnectAuthService()

    with pytest.raises(RuntimeError):
        await store.get_config()

    store.configure_auth_service(service)
    config = DirectConnectAuthConfigRecord(
        timestamps=get_current_timestamps(),
        issuer="https://idp.example.com",
        client_id="kumo",
        encrypted_client_secret="secret",
        groups_claim="groups",
        admin_emails=["admin@example.com"],
        enabled=True,
    )

    await store.init_once()
    await store.set_config(config)
    assert await store.get_config() == config
    await store.close()

    assert service.init_calls == 1
    assert service.set_config_calls == [config]
    assert service.get_config_calls == 1
    assert service.write_calls == 0
    assert service.read_calls == 0
    assert service.close_calls == 1


@pytest.mark.asyncio
async def test_store_uses_explicit_direct_connect_service_methods():
    service = _FakeDirectConnectAuthService()
    store = DirectConnectAuthStore(service)
    timestamps = get_current_timestamps()
    user = DirectConnectUserRecord(
        timestamps=timestamps,
        email="admin@example.com",
        display_name="Admin",
        role="admin",
    )
    session = DirectConnectSessionRecord(
        timestamps=timestamps,
        user_id="user_1",
        expires_time=now_utc() + timedelta(hours=1),
    )
    service.users["user_1"] = user
    service.sessions["session_1"] = session

    assert [user_response.email for user_response in await store.list_users()
            ] == ["admin@example.com"]
    assert await store.get_user("user_1") == user
    assert await store.get_user_by_email("ADMIN@example.com") == (
        "user_1",
        user,
    )
    assert await store.get_session("session_1") == session
    await store.delete_session("session_1")

    assert service.list_users_calls == 1
    assert service.get_user_calls == ["user_1"]
    assert service.get_user_by_email_calls == ["ADMIN@example.com"]
    assert service.get_session_calls == ["session_1"]
    assert service.pop_session_calls == ["session_1"]
    assert service.sessions == {}
    assert service.read_calls == 0
    assert service.write_calls == 0


@pytest.mark.asyncio
async def test_direct_connect_auth_service_explicit_methods():
    service = DirectConnectAuthService.__new__(DirectConnectAuthService)
    timestamps = get_current_timestamps()
    config = DirectConnectAuthConfigRecord(
        timestamps=timestamps,
        issuer="https://idp.example.com",
        client_id="kumo",
        encrypted_client_secret="secret",
        groups_claim="groups",
        admin_emails=["admin@example.com"],
        enabled=True,
    )
    user = DirectConnectUserRecord(
        timestamps=timestamps,
        email="admin@example.com",
        display_name="Admin",
        role="admin",
    )
    session = DirectConnectSessionRecord(
        timestamps=timestamps,
        user_id="user_1",
        expires_time=now_utc() + timedelta(hours=1),
    )
    calls = []

    class _Reader:
        async def get_config(self):
            calls.append(("get_config", ))
            return config

        async def list_users(self):
            calls.append(("list_users", ))
            return [("user_1", user)]

        async def get_user(self, user_id):
            calls.append(("get_user", user_id))
            return user

        async def get_user_id_by_email(self, email):
            calls.append(("get_user_id_by_email", email))
            return "user_1"

        async def get_session(self, session_id):
            calls.append(("get_session", session_id))
            return session

    class _Writer(_Reader):
        async def set_config(self, got_config):
            calls.append(("set_config", got_config))

        async def pop_session(self, session_id):
            calls.append(("pop_session", session_id))
            return session

    @asynccontextmanager
    async def read():
        calls.append(("read", ))
        yield _Reader()

    @asynccontextmanager
    async def write():
        calls.append(("write", ))
        yield _Writer()

    service.read = read
    service.write = write

    assert await service.get_config() == config
    await service.set_config(config)
    assert await service.list_users() == [("user_1", user)]
    assert await service.get_user("user_1") == user
    assert await service.get_user_by_email("ADMIN@example.com") == (
        "user_1",
        user,
    )
    assert await service.get_session("session_1") == session
    assert await service.pop_session("session_1") == session
    assert calls == [
        ("read", ),
        ("get_config", ),
        ("write", ),
        ("set_config", config),
        ("read", ),
        ("list_users", ),
        ("read", ),
        ("get_user", "user_1"),
        ("read", ),
        ("get_user_id_by_email", "ADMIN@example.com"),
        ("get_user", "user_1"),
        ("read", ),
        ("get_session", "session_1"),
        ("write", ),
        ("pop_session", "session_1"),
    ]


@pytest.mark.asyncio
async def test_state_db_direct_connect_auth_store_persists_through_tables(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        "rest.driver.direct_connect_auth_storage.encrypt_secret_string",
        lambda value: value,
    )
    monkeypatch.setattr(
        "rest.driver.direct_connect_auth_storage.decrypt_secret_string",
        lambda value: value,
    )
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    store = DirectConnectAuthStore(DirectConnectAuthService(state_db))
    await store.seed_config_from_env(
        issuer="https://idp.example.com/",
        client_id="kumo",
        client_secret="client-secret",
        groups_claim="groups",
        admin_emails=["Admin@Example.com"],
    )
    user_id, user = await store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="subject-1",
        email="USER@example.com",
        display_name="User Example",
        refresh_token="refresh-token",
    )
    session_id = await store.create_session(
        user_id=user_id,
        expires_time=now_utc() + timedelta(hours=1),
    )

    reloaded = DirectConnectAuthStore(DirectConnectAuthService(state_db))

    config = await reloaded.get_config()
    assert config is not None
    assert config.issuer == "https://idp.example.com"
    assert config.admin_emails == ["admin@example.com"]
    assert await reloaded.get_client_secret() == "client-secret"

    reloaded_user = await reloaded.get_user(user_id)
    assert reloaded_user is not None
    assert reloaded_user.email == "user@example.com"
    assert reloaded_user.display_name == user.display_name

    email_lookup = await reloaded.get_user_by_email("USER@example.com")
    assert email_lookup is not None
    assert email_lookup[0] == user_id

    session = await reloaded.get_session(session_id)
    assert session is not None
    assert session.user_id == user_id
    assert not hasattr(session, "idp_group_hashes")

    async with state_db.read_tx() as tx:
        direct_connect_table_counts = {
            table_name: (await
                         tx.fetchrow(f"SELECT COUNT(*) FROM {table_name}"))[0]
            for table_name in (
                "direct_connect_config",
                "direct_connect_users",
                "direct_connect_identity_links",
                "direct_connect_sessions",
            )
        }
    assert direct_connect_table_counts == {
        "direct_connect_config": 1,
        "direct_connect_users": 1,
        "direct_connect_identity_links": 1,
        "direct_connect_sessions": 1,
    }


@pytest.mark.asyncio
async def test_state_db_direct_connect_auth_store_deletes_removed_rows(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        "rest.driver.direct_connect_auth_storage.encrypt_secret_string",
        lambda value: value,
    )
    monkeypatch.setattr(
        "rest.driver.direct_connect_auth_storage.decrypt_secret_string",
        lambda value: value,
    )
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    store = DirectConnectAuthStore(DirectConnectAuthService(state_db))
    user = await store.create_local_user(email="local@example.com")
    await store.set_temporary_password(user.user_id, "SeedPass12345")
    session_id = await store.create_session(
        user_id=user.user_id,
        expires_time=now_utc() + timedelta(hours=12),
    )
    _status, reset_token, _user_id, _record = await store.password_login(
        email="local@example.com",
        password="SeedPass12345",
        reset_expires_time=now_utc() + timedelta(minutes=10),
    )

    await store.complete_password_reset(
        reset_token=reset_token,
        new_password="FinalPass67890",
    )
    assert await store.get_session(session_id) is None

    async with state_db.read_tx() as tx:
        session_count = (
            await
            tx.fetchrow("SELECT COUNT(*) FROM direct_connect_sessions"))[0]
        reset_token_count = (await tx.fetchrow(
            "SELECT COUNT(*) FROM direct_connect_password_reset_tokens"))[0]

    assert session_count == 0
    assert reset_token_count == 1


@pytest.mark.asyncio
async def test_first_matching_admin_email_bootstraps_admin(auth_store):
    user_id, user = await auth_store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="admin-subject",
        email="ADMIN@example.com",
        display_name="Admin User",
        refresh_token="refresh-token",
        email_verified=True,
    )

    assert user_id == direct_connect_user_id_for_identity(
        "https://idp.example.com", "admin-subject")
    assert user.email == "admin@example.com"
    assert user.role == "admin"


@pytest.mark.asyncio
async def test_new_identity_reuses_existing_email_user(auth_store):
    admin_user_id, admin_user = await auth_store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="admin-subject",
        email="admin@example.com",
        display_name="Admin User",
        refresh_token="refresh-token",
        email_verified=True,
    )
    other_user_id, other_user = await auth_store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="other-subject",
        email="admin@example.com",
        display_name="Other User",
        refresh_token="refresh-token",
        email_verified=True,
    )

    assert admin_user.role == "admin"
    assert other_user_id == admin_user_id
    assert other_user.role == "admin"
    assert other_user.display_name == "Other User"

    reloaded_admin = await auth_store.get_user(admin_user_id)
    assert reloaded_admin is not None
    assert reloaded_admin.role == "admin"
    assert reloaded_admin.display_name == "Other User"

    repeated_user_id, repeated_user = await auth_store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="other-subject",
        email="admin@example.com",
        display_name="Repeated User",
        refresh_token="refresh-token",
        email_verified=True,
    )

    assert repeated_user_id == admin_user_id
    assert repeated_user.display_name == "Repeated User"


@pytest.mark.asyncio
async def test_new_identity_links_admin_created_local_user(auth_store):
    local_user = await auth_store.create_local_user(
        email="local@example.com",
        display_name="Local User",
    )

    user_id, user = await auth_store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="local-subject",
        email="LOCAL@example.com",
        display_name="IdP User",
        refresh_token="refresh-token",
    )

    assert user_id == local_user.user_id
    assert user.email == "local@example.com"
    assert user.display_name == "IdP User"

    repeated_user_id, repeated_user = await auth_store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="local-subject",
        email="local@example.com",
        display_name="IdP User Again",
        refresh_token="refresh-token",
    )

    assert repeated_user_id == local_user.user_id
    assert repeated_user.display_name == "IdP User Again"


@pytest.mark.asyncio
async def test_existing_identity_rejects_conflicting_email_user(auth_store):
    idp_user_id, _ = await auth_store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="idp-subject",
        email="idp@example.com",
        display_name="IdP User",
        refresh_token="refresh-token",
    )
    local_user = await auth_store.create_local_user(
        email="local@example.com",
        display_name="Local User",
    )

    with pytest.raises(HTTPException) as exc_info:
        await auth_store.upsert_user_from_idp(
            issuer="https://idp.example.com",
            subject="idp-subject",
            email="local@example.com",
            display_name="Conflicting User",
            refresh_token="refresh-token",
        )

    assert exc_info.value.status_code == HTTPStatus.CONFLICT
    assert await auth_store.get_user(idp_user_id) is not None
    assert await auth_store.get_user(local_user.user_id) is not None


@pytest.mark.asyncio
async def test_email_index_removes_old_email_when_user_email_changes(
        auth_store):
    user_id, _ = await auth_store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="user-subject",
        email="old@example.com",
        display_name="User",
        refresh_token="refresh-token",
    )

    assert await auth_store.get_user_by_email("old@example.com") is not None

    updated_user_id, updated_user = await auth_store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="user-subject",
        email="new@example.com",
        display_name="User",
        refresh_token="refresh-token",
    )

    assert updated_user_id == user_id
    assert updated_user.email == "new@example.com"
    assert await auth_store.get_user_by_email("old@example.com") is None

    new_email_lookup = await auth_store.get_user_by_email("new@example.com")
    assert new_email_lookup is not None
    assert new_email_lookup[0] == user_id


@pytest.mark.asyncio
async def test_admin_email_with_unverified_email_does_not_bootstrap_admin(
        auth_store):
    _user_id, user = await auth_store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="unverified-admin-subject",
        email="admin@example.com",
        display_name="Unverified Admin",
        refresh_token="refresh-token",
        email_verified=False,
    )

    assert user.role == "user"


@pytest.mark.asyncio
async def test_create_session_never_persists_provider_membership(auth_store):
    user_id, _ = await auth_store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="session-hash-subject",
        email="session-hash.com",
        display_name="Session Hash",
        refresh_token="refresh-token",
    )
    session_id = await auth_store.create_session(
        user_id=user_id,
        expires_time=now_utc() + timedelta(minutes=5),
    )

    session = await auth_store.get_session(session_id)
    assert session is not None
    assert not hasattr(session, "idp_group_hashes")


@pytest.mark.asyncio
async def test_cleanup_expired_sessions_is_bounded(auth_store):
    user_id, _ = await auth_store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="session-user-subject",
        email="session-user@example.com",
        display_name="Session User",
        refresh_token="refresh-token",
    )
    expired_ids = [
        await auth_store.create_session(
            user_id=user_id,
            expires_time=now_utc() - timedelta(minutes=1),
        ) for _ in range(3)
    ]
    active_id = await auth_store.create_session(
        user_id=user_id,
        expires_time=now_utc() + timedelta(minutes=5),
    )

    assert await auth_store.cleanup_expired_sessions(limit=2) == 2

    remaining_expired = [
        session_id for session_id in expired_ids
        if await auth_store.get_session(session_id) is not None
    ]
    assert len(remaining_expired) == 1
    assert await auth_store.get_session(active_id) is not None

    assert await auth_store.cleanup_expired_sessions(limit=2) == 1
    assert all([
        await auth_store.get_session(session_id) is None
        for session_id in expired_ids
    ])
    assert await auth_store.get_session(active_id) is not None


@pytest.mark.asyncio
async def test_seed_config_from_env_does_not_overwrite_existing_config(
        auth_store):
    await auth_store.seed_config_from_env(
        issuer="https://new-idp.example.com",
        client_id="new-client",
        client_secret="new-secret",
        groups_claim="new-groups",
        admin_emails=["new-admin@example.com"],
    )

    config = await auth_store.get_config()
    assert config is not None
    assert config.issuer == "https://idp.example.com"
    assert config.client_id == "kumo"
    assert config.groups_claim == "groups"
    assert config.admin_emails == ["admin@example.com"]


@pytest.mark.asyncio
async def test_seed_config_from_env_requires_core_oidc_values(
        monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "rest.driver.direct_connect_auth_storage.encrypt_secret_string",
        lambda value: calls.append(value) or value,
    )
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    store = DirectConnectAuthStore(DirectConnectAuthService(state_db))

    await store.seed_config_from_env(
        issuer="https://idp.example.com",
        client_id="",
        client_secret="secret",
        groups_claim="groups",
        admin_emails=["admin@example.com"],
    )

    assert await store.get_config() is None
    assert calls == []


@pytest.mark.asyncio
async def test_seed_config_persists_google_groups_settings(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        "rest.driver.direct_connect_auth_storage.encrypt_secret_string",
        lambda value: f"enc:{value}",
    )
    monkeypatch.setattr(
        "rest.driver.direct_connect_auth_storage.decrypt_secret_string",
        lambda value: value.removeprefix("enc:"),
    )
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    store = DirectConnectAuthStore(DirectConnectAuthService(state_db))

    await store.seed_config(
        issuer="https://accounts.google.com",
        client_id="kumo",
        client_secret="secret",
        groups_claim="groups",
        admin_emails=["admin@example.com"],
        google_sa_key='{"type": "service_account"}',
        google_admin_email="workspace-admin@example.com",
    )

    config = await store.get_config()
    assert config is not None
    assert config.admin_emails == ["admin@example.com"]
    assert config.encrypted_google_sa_key == 'enc:{"type": "service_account"}'
    assert config.google_admin_email == "workspace-admin@example.com"
    assert await store.get_google_groups_settings() == (
        '{"type": "service_account"}',
        "workspace-admin@example.com",
    )


@pytest.mark.asyncio
async def test_seed_or_reconcile_oidc_config_creates_missing_config(
        monkeypatch, tmp_path):
    store = await _empty_auth_store(tmp_path, monkeypatch)

    await store.seed_or_reconcile_oidc_config(
        issuer="https://idp.example.com/",
        client_id="kumo",
        client_secret="secret",
        groups_claim="",
        admin_emails=[
            "Admin@Example.com",
            "admin@example.com",
            "  ",
            "Second-Admin@Example.com",
        ],
    )

    config = await store.get_config()
    assert config is not None
    assert config.issuer == "https://idp.example.com"
    assert config.client_id == "kumo"
    assert config.groups_claim == "groups"
    assert config.admin_emails == [
        "admin@example.com",
        "second-admin@example.com",
    ]
    assert await store.get_client_secret() == "secret"


@pytest.mark.asyncio
async def test_seed_or_reconcile_oidc_config_repairs_admin_emails_before_admin(
        auth_store):
    _user_id, user = await auth_store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="new-admin-subject",
        email="new-admin@example.com",
        display_name="New Admin",
        refresh_token="refresh-token",
        email_verified=True,
    )
    assert user.role == "user"

    await auth_store.seed_or_reconcile_oidc_config(
        issuer="https://idp.example.com/",
        client_id="kumo",
        client_secret="secret",
        groups_claim="groups",
        admin_emails=["NEW-ADMIN@example.com"],
    )

    config = await auth_store.get_config()
    assert config is not None
    assert config.admin_emails == ["new-admin@example.com"]

    _user_id, user = await auth_store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="new-admin-subject",
        email="new-admin@example.com",
        display_name="New Admin",
        refresh_token="refresh-token",
        email_verified=True,
    )
    assert user.role == "admin"


@pytest.mark.asyncio
async def test_seed_or_reconcile_oidc_config_keeps_admin_emails_with_admin(
        auth_store):
    _user_id, user = await auth_store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="admin-subject",
        email="admin@example.com",
        display_name="Admin",
        refresh_token="refresh-token",
        email_verified=True,
    )
    assert user.role == "admin"

    await auth_store.seed_or_reconcile_oidc_config(
        issuer="https://idp.example.com",
        client_id="kumo",
        client_secret="secret",
        groups_claim="groups",
        admin_emails=["new-admin@example.com"],
    )

    config = await auth_store.get_config()
    assert config is not None
    assert config.admin_emails == ["admin@example.com"]


@pytest.mark.asyncio
async def test_seed_or_reconcile_oidc_config_allows_missing_deploy_values_with_admin(
        auth_store):
    _user_id, user = await auth_store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="admin-subject",
        email="admin@example.com",
        display_name="Admin",
        refresh_token="refresh-token",
        email_verified=True,
    )
    assert user.role == "admin"

    await auth_store.seed_or_reconcile_oidc_config(
        issuer="",
        client_id="",
        client_secret="",
        groups_claim="groups",
        admin_emails=[],
    )

    config = await auth_store.get_config()
    assert config is not None
    assert config.issuer == "https://idp.example.com"
    assert config.client_id == "kumo"
    assert config.admin_emails == ["admin@example.com"]
    assert await auth_store.get_client_secret() == "secret"


@pytest.mark.asyncio
async def test_seed_or_reconcile_oidc_config_keeps_stored_admins_when_deploy_empty(
        auth_store):
    await auth_store.seed_or_reconcile_oidc_config(
        issuer="https://idp.example.com",
        client_id="kumo",
        client_secret="secret",
        groups_claim="groups",
        admin_emails=[],
    )

    config = await auth_store.get_config()
    assert config is not None
    assert config.admin_emails == ["admin@example.com"]


@pytest.mark.parametrize(("overrides", "field"), [
    ({
        "issuer": "https://other-idp.example.com"
    }, "issuer"),
    ({
        "client_id": "other-client"
    }, "client_id"),
    ({
        "client_secret": "other-secret"
    }, "client_secret"),
])
@pytest.mark.asyncio
async def test_seed_or_reconcile_oidc_config_applies_env_drift(
        auth_store, caplog, overrides, field):
    args = {
        "issuer": "https://idp.example.com",
        "client_id": "kumo",
        "client_secret": "secret",
        "groups_claim": "groups",
        "admin_emails": ["new-admin@example.com"],
    }
    args.update(overrides)

    with caplog.at_level(logging.WARNING):
        await auth_store.seed_or_reconcile_oidc_config(**args)

    config = await auth_store.get_config()
    assert config is not None
    assert config.issuer == args["issuer"]
    assert config.client_id == args["client_id"]
    assert config.admin_emails == ["new-admin@example.com"]
    assert await auth_store.get_client_secret() == args["client_secret"]
    assert field in caplog.text
    assert "other-secret" not in caplog.text


@pytest.mark.asyncio
async def test_seed_or_reconcile_oidc_config_reseeds_undecryptable_secret(
        auth_store, monkeypatch):
    def _raise_invalid_token(_value):
        raise InvalidToken()

    monkeypatch.setattr(
        "rest.driver.direct_connect_auth_storage.decrypt_secret_string",
        _raise_invalid_token,
    )

    await auth_store.seed_or_reconcile_oidc_config(
        issuer="https://idp.example.com",
        client_id="kumo",
        client_secret="rotated-secret",
        groups_claim="groups",
        admin_emails=["admin@example.com"],
    )

    monkeypatch.setattr(
        "rest.driver.direct_connect_auth_storage.decrypt_secret_string",
        lambda value: value,
    )
    assert await auth_store.get_client_secret() == "rotated-secret"


@pytest.mark.asyncio
async def test_seed_or_reconcile_oidc_config_applies_rotated_idp_registration(
        auth_store):
    await auth_store.seed_or_reconcile_oidc_config(
        issuer="https://new-idp.example.com/",
        client_id="new-client",
        client_secret="new-secret",
        groups_claim="groups",
        admin_emails=["admin@example.com"],
    )

    config = await auth_store.get_config()
    assert config is not None
    assert config.issuer == "https://new-idp.example.com"
    assert config.client_id == "new-client"
    assert config.enabled is True
    assert await auth_store.get_client_secret() == "new-secret"


@pytest.mark.asyncio
async def test_seed_or_reconcile_oidc_config_updates_groups_claim(auth_store):
    await auth_store.set_config(
        DirectConnectAuthConfigRecord(
            timestamps=get_current_timestamps(),
            issuer="https://idp.example.com",
            client_id="kumo",
            encrypted_client_secret="secret",
            groups_claim="myGroups",
            admin_emails=["admin@example.com"],
            enabled=True,
        ))

    await auth_store.seed_or_reconcile_oidc_config(
        issuer="https://idp.example.com",
        client_id="kumo",
        client_secret="secret",
        groups_claim="roles",
        admin_emails=["admin@example.com"],
        google_sa_key='{"type": "service_account"}',
        google_admin_email="workspace-admin@example.com",
    )

    config = await auth_store.get_config()
    assert config is not None
    assert config.encrypted_google_sa_key == '{"type": "service_account"}'
    assert config.google_admin_email == "workspace-admin@example.com"
    assert await auth_store.get_google_groups_settings() == (
        '{"type": "service_account"}',
        "workspace-admin@example.com",
    )


@pytest.mark.asyncio
async def test_seed_or_reconcile_oidc_config_keeps_groups_claim_when_deploy_empty(
        auth_store):
    await auth_store.set_config(
        DirectConnectAuthConfigRecord(
            timestamps=get_current_timestamps(),
            issuer="https://idp.example.com",
            client_id="kumo",
            encrypted_client_secret="secret",
            groups_claim="myGroups",
            admin_emails=["admin@example.com"],
            enabled=True,
        ))

    await auth_store.seed_or_reconcile_oidc_config(
        issuer="",
        client_id="",
        client_secret="",
        groups_claim="",
        admin_emails=[],
    )

    config = await auth_store.get_config()
    assert config is not None
    assert config.groups_claim == "myGroups"


@pytest.mark.asyncio
async def test_seed_or_reconcile_oidc_config_rejects_disabled_existing_config(
        auth_store):
    config = await auth_store.get_config()
    assert config is not None
    config.enabled = False
    await auth_store.set_config(config)

    with pytest.raises(ValueError, match="disabled"):
        await auth_store.seed_or_reconcile_oidc_config(
            issuer="https://idp.example.com",
            client_id="kumo",
            client_secret="secret",
            groups_claim="groups",
            admin_emails=["admin@example.com"],
        )

    config = await auth_store.get_config()
    assert config is not None
    assert config.enabled is False


@pytest.mark.asyncio
async def test_seed_or_reconcile_oidc_config_requires_deploy_admins(
        monkeypatch, tmp_path):
    store = await _empty_auth_store(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="admin_emails"):
        await store.seed_or_reconcile_oidc_config(
            issuer="https://idp.example.com",
            client_id="kumo",
            client_secret="secret",
            groups_claim="groups",
            admin_emails=[],
        )

    assert await store.get_config() is None


@pytest.mark.asyncio
async def test_seed_or_reconcile_oidc_config_requires_core_oidc_values(
        monkeypatch, tmp_path):
    store = await _empty_auth_store(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="client_id"):
        await store.seed_or_reconcile_oidc_config(
            issuer="https://idp.example.com",
            client_id="",
            client_secret="secret",
            groups_claim="groups",
            admin_emails=["admin@example.com"],
        )

    assert await store.get_config() is None


@pytest.mark.asyncio
async def test_google_groups_settings_empty_when_not_configured(auth_store):
    assert await auth_store.get_google_groups_settings() == ("", "")


@pytest.mark.asyncio
async def test_update_user_rejects_disabling_last_active_admin(auth_store):
    admin_user_id, _ = await auth_store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="admin-subject",
        email="admin@example.com",
        display_name="Admin User",
        refresh_token="refresh-token",
        email_verified=True,
    )

    with pytest.raises(Exception) as exc_info:
        await auth_store.update_user(admin_user_id, disabled=True)

    assert getattr(exc_info.value, "status_code", None) == 400
    assert (getattr(exc_info.value, "detail",
                    "") == "Cannot disable or demote last admin")


@pytest.mark.asyncio
async def test_update_user_allows_reenabling_user(auth_store):
    user_id, _ = await auth_store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="user-subject",
        email="user@example.com",
        display_name="User",
        refresh_token="refresh-token",
    )
    await auth_store.update_user(user_id, disabled=True)

    updated = await auth_store.update_user(user_id, disabled=False)

    assert updated.disabled is False


@pytest.mark.asyncio
async def test_local_password_login_requires_first_login_reset(auth_store):
    user = await auth_store.create_local_user(
        email="local@example.com",
        display_name="Local User",
    )
    await auth_store.set_temporary_password(user.user_id, "SeedPass12345")

    status, reset_token, user_id, record = await auth_store.password_login(
        email="LOCAL@example.com",
        password="SeedPass12345",
        reset_expires_time=now_utc() + timedelta(minutes=10),
    )

    assert status == "password_reset_required"
    assert reset_token
    assert user_id == user.user_id
    assert record.email == "local@example.com"

    async with auth_store._service._state_db.read_tx() as tx:
        rows = await tx.fetch(
            "SELECT id FROM direct_connect_password_reset_tokens")
    assert [row["id"] for row in rows
            ] == [storage._password_reset_token_key(reset_token)]
    assert reset_token not in [row["id"] for row in rows]


@pytest.mark.asyncio
async def test_password_login_verification_runs_off_event_loop(
        auth_store, monkeypatch):
    user = await auth_store.create_local_user(email="local@example.com")
    await auth_store.set_temporary_password(user.user_id, "SeedPass12345")
    event_loop_thread_id = threading.get_ident()
    verify_thread_ids = []
    original_verify = storage._verify_password

    def _verify_on_worker(*args, **kwargs):
        verify_thread_ids.append(threading.get_ident())
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(storage, "_verify_password", _verify_on_worker)

    status, reset_token, _user_id, _record = await auth_store.password_login(
        email="local@example.com",
        password="SeedPass12345",
        reset_expires_time=now_utc() + timedelta(minutes=5),
    )

    assert status == "password_reset_required"
    assert reset_token
    assert verify_thread_ids
    assert all(thread_id != event_loop_thread_id
               for thread_id in verify_thread_ids)


@pytest.mark.asyncio
async def test_complete_password_reset_allows_normal_password_login(auth_store):
    user = await auth_store.create_local_user(email="local@example.com")
    await auth_store.set_temporary_password(user.user_id, "SeedPass12345")
    _status, reset_token, _user_id, _record = await auth_store.password_login(
        email="local@example.com",
        password="SeedPass12345",
        reset_expires_time=now_utc() + timedelta(minutes=10),
    )

    reset_user_id, reset_user = await auth_store.complete_password_reset(
        reset_token=reset_token,
        new_password="FinalPass67890",
    )
    status, second_reset_token, login_user_id, login_user = await auth_store.password_login(
        email="local@example.com",
        password="FinalPass67890",
        reset_expires_time=now_utc() + timedelta(minutes=10),
    )

    assert reset_user_id == user.user_id
    assert reset_user.email == "local@example.com"
    assert status == "ok"
    assert second_reset_token is None
    assert login_user_id == user.user_id
    assert login_user.email == "local@example.com"


async def _get_credential(auth_store, user_id):
    async with auth_store._service.read() as db:
        return await db.get_password_credential(user_id)


@pytest.mark.asyncio
async def test_new_password_credential_persists_current_scrypt_params(
        auth_store):
    user = await auth_store.create_local_user(
        email="local@example.com",
        password="SeedPass12345",
    )

    credential = await _get_credential(auth_store, user.user_id)

    assert credential is not None
    assert credential.hash_params == storage._PASSWORD_SCRYPT_PARAMS


@pytest.mark.asyncio
async def test_complete_password_reset_rejects_stale_credential(auth_store):
    user = await auth_store.create_local_user(email="local@example.com")
    await auth_store.set_temporary_password(user.user_id, "SeedPass12345")
    _status, reset_token, _user_id, _record = await auth_store.password_login(
        email="local@example.com",
        password="SeedPass12345",
        reset_expires_time=now_utc() + timedelta(minutes=10),
    )

    await auth_store.set_temporary_password(user.user_id, "RotatedPass67890")

    with pytest.raises(Exception) as stale_token:
        await auth_store.complete_password_reset(
            reset_token=reset_token,
            new_password="FinalPass67890",
        )
    assert getattr(stale_token.value, "status_code", None) == 401

    status, rotated_reset_token, login_user_id, _record = await auth_store.password_login(
        email="local@example.com",
        password="RotatedPass67890",
        reset_expires_time=now_utc() + timedelta(minutes=10),
    )
    assert status == "password_reset_required"
    assert rotated_reset_token
    assert login_user_id == user.user_id


@pytest.mark.asyncio
async def test_create_local_user_with_password_requires_first_login_reset(
        auth_store):
    user = await auth_store.create_local_user(
        email="local@example.com",
        password="SeedPass12345",
    )

    status, reset_token, login_user_id, record = await auth_store.password_login(
        email="local@example.com",
        password="SeedPass12345",
        reset_expires_time=now_utc() + timedelta(minutes=10),
    )

    assert status == "password_reset_required"
    assert reset_token
    assert login_user_id == user.user_id
    assert record.email == "local@example.com"


@pytest.mark.asyncio
async def test_create_local_user_rejects_weak_password_without_creating_user(
        auth_store):
    with pytest.raises(Exception) as weak_password:
        await auth_store.create_local_user(
            email="local@example.com",
            password="short1",
        )

    assert getattr(weak_password.value, "status_code", None) == 400
    assert await auth_store.get_user_by_email("local@example.com") is None


@pytest.mark.asyncio
async def test_temporary_password_enforces_password_policy(auth_store):
    user = await auth_store.create_local_user(email="local@example.com")

    with pytest.raises(Exception) as exc_info:
        await auth_store.set_temporary_password(user.user_id, "short1")

    assert getattr(exc_info.value, "status_code", None) == 400
    assert (getattr(exc_info.value, "detail",
                    "") == "Password must be at least 12 characters")


@pytest.mark.asyncio
async def test_complete_password_reset_rejects_weak_or_reused_password(
        auth_store):
    user = await auth_store.create_local_user(email="local@example.com")
    await auth_store.set_temporary_password(user.user_id, "SeedPass12345")
    _status, reset_token, _user_id, _record = await auth_store.password_login(
        email="local@example.com",
        password="SeedPass12345",
        reset_expires_time=now_utc() + timedelta(minutes=10),
    )

    with pytest.raises(Exception) as weak_password:
        await auth_store.complete_password_reset(
            reset_token=reset_token,
            new_password="short1",
        )
    assert getattr(weak_password.value, "status_code", None) == 400
    assert (getattr(weak_password.value, "detail",
                    "") == "Password must be at least 12 characters")

    with pytest.raises(Exception) as reused_password:
        await auth_store.complete_password_reset(
            reset_token=reset_token,
            new_password="SeedPass12345",
        )
    assert getattr(reused_password.value, "status_code", None) == 400
    assert (getattr(
        reused_password.value, "detail",
        "") == "New password must differ from the temporary password")


@pytest.mark.asyncio
async def test_complete_password_reset_invalidates_existing_user_sessions(
        auth_store):
    user = await auth_store.create_local_user(email="local@example.com")
    other_user = await auth_store.create_local_user(email="other@example.com")
    await auth_store.set_temporary_password(user.user_id, "SeedPass12345")
    old_session_id = await auth_store.create_session(
        user_id=user.user_id,
        expires_time=now_utc() + timedelta(hours=12),
    )
    other_session_id = await auth_store.create_session(
        user_id=other_user.user_id,
        expires_time=now_utc() + timedelta(hours=12),
    )
    _status, reset_token, _user_id, _record = await auth_store.password_login(
        email="local@example.com",
        password="SeedPass12345",
        reset_expires_time=now_utc() + timedelta(minutes=10),
    )

    await auth_store.complete_password_reset(
        reset_token=reset_token,
        new_password="FinalPass67890",
    )

    assert await auth_store.get_session(old_session_id) is None
    assert await auth_store.get_session(other_session_id) is not None


@pytest.mark.asyncio
async def test_password_login_runs_dummy_hash_for_unknown_email(
        auth_store, monkeypatch):
    dummy_hash_calls = 0

    def count_dummy_hash(_password: str) -> None:
        nonlocal dummy_hash_calls
        dummy_hash_calls += 1

    monkeypatch.setattr(storage, "_verify_dummy_password", count_dummy_hash)

    with pytest.raises(Exception) as wrong_email:
        await auth_store.password_login(
            email="missing@example.com",
            password="SeedPass12345",
            reset_expires_time=now_utc() + timedelta(minutes=10),
        )

    assert getattr(wrong_email.value, "status_code", None) == 401
    assert dummy_hash_calls == 1


@pytest.mark.asyncio
async def test_password_login_rejects_wrong_password_and_disabled_user(
        auth_store):
    user = await auth_store.create_local_user(email="local@example.com")
    await auth_store.set_temporary_password(user.user_id, "SeedPass12345")

    with pytest.raises(Exception) as wrong_password:
        await auth_store.password_login(
            email="local@example.com",
            password="wrong-password",
            reset_expires_time=now_utc() + timedelta(minutes=10),
        )
    assert getattr(wrong_password.value, "status_code", None) == 401

    await auth_store.update_user(user.user_id, disabled=True)
    with pytest.raises(Exception) as disabled_user:
        await auth_store.password_login(
            email="local@example.com",
            password="SeedPass12345",
            reset_expires_time=now_utc() + timedelta(minutes=10),
        )
    assert getattr(disabled_user.value, "status_code", None) == 401


@pytest.mark.asyncio
async def test_password_login_rate_limits_failed_email(auth_store):
    email = "local@example.com"
    source_ip = "203.0.113.10"

    for _ in range(storage._PASSWORD_LOGIN_EMAIL_LIMIT):
        await auth_store.check_password_login_allowed(email=email,
                                                      source_ip=source_ip)
        await auth_store.record_password_login_failure(email=email,
                                                       source_ip=source_ip)

    with pytest.raises(Exception) as locked:
        await auth_store.check_password_login_allowed(email=email,
                                                      source_ip="203.0.113.11")
    assert getattr(locked.value, "status_code", None) == 429

    await auth_store.record_password_login_success(email=email)
    await auth_store.check_password_login_allowed(email=email,
                                                  source_ip="203.0.113.11")


@pytest.mark.asyncio
async def test_password_login_rate_limits_failed_source_ip(auth_store):
    source_ip = "203.0.113.10"

    for index in range(storage._PASSWORD_LOGIN_IP_LIMIT):
        email = f"user-{index}@example.com"
        await auth_store.check_password_login_allowed(email=email,
                                                      source_ip=source_ip)
        await auth_store.record_password_login_failure(email=email,
                                                       source_ip=source_ip)

    with pytest.raises(Exception) as locked:
        await auth_store.check_password_login_allowed(email="other@example.com",
                                                      source_ip=source_ip)
    assert getattr(locked.value, "status_code", None) == 429


def test_password_login_source_ip_none_is_distinct_from_unknown_peer():
    email = "local@example.com"
    email_key = (storage._password_login_attempt_key("email", email),
                 storage._PASSWORD_LOGIN_EMAIL_LIMIT)
    unknown_ip_key = (storage._password_login_attempt_key(
        "ip", storage._PASSWORD_LOGIN_UNKNOWN_IP),
                      storage._PASSWORD_LOGIN_IP_LIMIT)

    assert storage._password_login_attempt_keys(email, None) == [email_key]
    assert storage._password_login_attempt_keys(email,
                                                "") == [email_key,
                                                        unknown_ip_key]


@pytest.mark.asyncio
async def test_password_login_without_trusted_source_ip_uses_email_limit_only(
        auth_store):
    for index in range(storage._PASSWORD_LOGIN_IP_LIMIT + 1):
        email = f"user-{index}@example.com"
        await auth_store.check_password_login_allowed(email=email,
                                                      source_ip=None)
        await auth_store.record_password_login_failure(email=email,
                                                       source_ip=None)

    # Distinct users are not collapsed into a shared "unknown" IP bucket.
    await auth_store.check_password_login_allowed(email="other@example.com",
                                                  source_ip=None)

    email = "limited@example.com"
    for _ in range(storage._PASSWORD_LOGIN_EMAIL_LIMIT):
        await auth_store.check_password_login_allowed(email=email,
                                                      source_ip=None)
        await auth_store.record_password_login_failure(email=email,
                                                       source_ip=None)

    with pytest.raises(Exception) as locked:
        await auth_store.check_password_login_allowed(email=email,
                                                      source_ip=None)
    assert getattr(locked.value, "status_code", None) == 429


@pytest.mark.asyncio
async def test_password_hash_is_not_plaintext(auth_store):
    user = await auth_store.create_local_user(email="local@example.com")
    await auth_store.set_temporary_password(user.user_id, "SeedPass12345")

    async with auth_store._service.read() as db:
        credential = await db.get_password_credential(user.user_id)

    assert credential is not None
    assert credential.hash_algorithm == "scrypt"
    assert credential.password_hash_b64 != "SeedPass12345"
    assert credential.salt_b64


@pytest.mark.asyncio
async def test_seed_local_admins_creates_first_temp_password_admin(auth_store):
    await auth_store.seed_local_admins(
        admin_emails=["LOCAL@example.com"],
        temporary_password="SeedPass12345",
    )

    lookup = await auth_store.get_user_by_email("local@example.com")
    assert lookup is not None
    user_id, user = lookup
    assert user.role == "admin"
    assert user.disabled is False

    status, reset_token, login_user_id, _record = await auth_store.password_login(
        email="local@example.com",
        password="SeedPass12345",
        reset_expires_time=now_utc() + timedelta(minutes=10),
    )
    assert status == "password_reset_required"
    assert reset_token
    assert login_user_id == user_id


@pytest.mark.asyncio
async def test_seed_local_admins_repairs_configured_admin_without_password(
        auth_store):
    existing_admin = await auth_store.create_local_user(
        email="admin@example.com",
        role="admin",
    )

    await auth_store.seed_local_admins(
        admin_emails=["admin@example.com"],
        temporary_password="SeedPass12345",
    )

    status, reset_token, login_user_id, _record = await auth_store.password_login(
        email="admin@example.com",
        password="SeedPass12345",
        reset_expires_time=now_utc() + timedelta(minutes=10),
    )
    assert status == "password_reset_required"
    assert reset_token
    assert login_user_id == existing_admin.user_id


@pytest.mark.asyncio
async def test_seed_local_admins_repairs_configured_admin_with_empty_password(
        auth_store):
    existing_admin = await auth_store.create_local_user(
        email="admin@example.com",
        role="admin",
    )
    async with auth_store._service.write() as db:
        await db.set_password_credential(
            existing_admin.user_id,
            DirectConnectPasswordCredentialRecord(
                timestamps=get_current_timestamps(),
                password_hash_b64="",
                salt_b64="",
                hash_algorithm="scrypt",
                hash_params={
                    "n": 16384,
                    "r": 8,
                    "p": 1,
                    "dklen": 32
                },
                must_reset_password=True,
            ),
        )

    await auth_store.seed_local_admins(
        admin_emails=["admin@example.com"],
        temporary_password="SeedPass12345",
    )

    status, reset_token, login_user_id, _record = await auth_store.password_login(
        email="admin@example.com",
        password="SeedPass12345",
        reset_expires_time=now_utc() + timedelta(minutes=10),
    )
    assert status == "password_reset_required"
    assert reset_token
    assert login_user_id == existing_admin.user_id


@pytest.mark.asyncio
async def test_seed_local_admins_does_not_overwrite_configured_admin_password(
        auth_store):
    existing_admin = await auth_store.create_local_user(
        email="admin@example.com",
        role="admin",
    )
    await auth_store.set_temporary_password(
        existing_admin.user_id,
        "ExistingPass12345",
    )

    await auth_store.seed_local_admins(
        admin_emails=["admin@example.com"],
        temporary_password="FinalPass67890",
    )

    status, _reset_token, login_user_id, _record = await auth_store.password_login(
        email="admin@example.com",
        password="ExistingPass12345",
        reset_expires_time=now_utc() + timedelta(minutes=10),
    )
    assert status == "password_reset_required"
    assert login_user_id == existing_admin.user_id


@pytest.mark.asyncio
async def test_seed_local_admins_does_not_overwrite_active_admin(auth_store):
    existing_admin = await auth_store.create_local_user(
        email="admin@example.com",
        role="admin",
    )
    await auth_store.set_temporary_password(
        existing_admin.user_id,
        "ExistingPass12345",
    )

    await auth_store.seed_local_admins(
        admin_emails=["new-admin@example.com"],
        temporary_password="FinalPass67890",
    )

    assert await auth_store.get_user_by_email("new-admin@example.com") is None
    status, _reset_token, _user_id, _record = await auth_store.password_login(
        email="admin@example.com",
        password="ExistingPass12345",
        reset_expires_time=now_utc() + timedelta(minutes=10),
    )
    assert status == "password_reset_required"
