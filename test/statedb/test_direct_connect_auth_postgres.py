# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta

import asyncpg
import pytest

from rest.driver.direct_connect_auth_storage import DirectConnectAuthStore
from statedb.config.timestamp import now_utc
from statedb.db import StateDB
from statedb.db.direct_connect_auth import DirectConnectAuthService

_DIRECT_CONNECT_TABLES = (
    "direct_connect_config",
    "direct_connect_users",
    "direct_connect_identity_links",
    "direct_connect_sessions",
    "direct_connect_password_credentials",
    "direct_connect_password_reset_tokens",
    "direct_connect_password_login_attempts",
)


async def _reset_direct_connect_tables(
    postgres_connection: asyncpg.Connection, ) -> None:
    await postgres_connection.execute(
        f"TRUNCATE {', '.join(_DIRECT_CONNECT_TABLES)}")


async def _build_store(
    postgres_dsn: str,
    postgres_connection: asyncpg.Connection,
    *,
    reset: bool = False,
) -> DirectConnectAuthStore:
    state_db = await StateDB.open_postgres(postgres_dsn)
    if reset:
        await _reset_direct_connect_tables(postgres_connection)
    return DirectConnectAuthStore(DirectConnectAuthService(state_db))


@pytest.mark.asyncio
async def test_direct_connect_store_persists_through_postgres_reload(
    postgres_dsn: str,
    postgres_connection: asyncpg.Connection,
):
    store = await _build_store(postgres_dsn, postgres_connection, reset=True)
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

    reloaded = await _build_store(postgres_dsn, postgres_connection)

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


@pytest.mark.asyncio
async def test_direct_connect_postgres_stores_encrypted_normalized_values(
    postgres_dsn: str,
    postgres_connection: asyncpg.Connection,
):
    store = await _build_store(postgres_dsn, postgres_connection, reset=True)

    await store.seed_config_from_env(
        issuer="https://idp.example.com",
        client_id="kumo",
        client_secret="plain-client-secret",
        groups_claim="groups",
        admin_emails=[],
    )
    user_id, _ = await store.upsert_user_from_idp(
        issuer="https://idp.example.com",
        subject="subject-1",
        email="user@example.com",
        display_name="User Example",
        refresh_token="refresh-token",
    )

    await store.create_session(
        user_id=user_id,
        expires_time=now_utc() + timedelta(hours=1),
    )

    config_row = await postgres_connection.fetchrow(
        """
        SELECT
            pg_typeof(payload)::text AS payload_type,
            payload::text AS payload_text,
            payload->>'encrypted_client_secret' AS encrypted_secret
        FROM direct_connect_config
        WHERE id = 'default'
        """, )
    session_row = await postgres_connection.fetchrow(
        """
        SELECT payload::text AS payload_text
        FROM direct_connect_sessions
        LIMIT 1
        """, )

    assert config_row is not None
    assert config_row["payload_type"] == "jsonb"
    assert config_row["encrypted_secret"]
    assert config_row["encrypted_secret"] != "plain-client-secret"
    assert "plain-client-secret" not in config_row["payload_text"]
    assert session_row is not None
    assert "secret-idp-group" not in session_row["payload_text"]
    assert await store.get_client_secret() == "plain-client-secret"
