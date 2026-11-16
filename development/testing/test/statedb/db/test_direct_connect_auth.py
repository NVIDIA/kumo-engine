# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import SecretStr

import statedb.db.direct_connect_auth as direct_connect_auth
from io_utils import kumo_dataclass
from io_utils.json import from_json_dict, to_json_dict
from rest.utils.driver_common import get_current_timestamps
from statedb.config.direct_connect_auth import (
    DirectConnectAuthConfigRecord,
    DirectConnectPasswordCredentialRecord,
    DirectConnectPasswordLoginAttemptRecord,
    DirectConnectPasswordResetRecord,
    DirectConnectSessionRecord,
    DirectConnectUserRecord,
    direct_connect_password_credential_tag,
)
from statedb.db import StateDB
from statedb.db.repositories import JsonRepository


def _ts():
    return get_current_timestamps()


class _FakePostgresStateTx:
    backend = "postgres"

    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []

    async def execute(self, sql: str, *args: object) -> str:
        self.execute_calls.append((sql, args))
        return "SELECT 0"

    async def fetchrow(self, sql: str, *args: object) -> None:
        raise AssertionError("fetchrow should not be used")

    async def fetch(self, sql: str, *args: object) -> list[object]:
        self.fetch_calls.append((sql, args))
        return []


class _FakePostgresStateDB:
    def __init__(self, tx: _FakePostgresStateTx) -> None:
        self._tx = tx

    @asynccontextmanager
    async def write_tx(self):
        yield self._tx


def test_direct_connect_secret_records_do_not_use_secretstr_for_state_json(
) -> None:
    @kumo_dataclass
    class SecretStrRecord:
        secret: SecretStr

    with pytest.raises(ValueError, match="Secure serialization of SecretStr"):
        to_json_dict(SecretStrRecord(secret="stored-secret-value"))


def test_direct_connect_secret_records_repr_and_log_dict_redact_values(
) -> None:
    config = DirectConnectAuthConfigRecord(
        timestamps=_ts(),
        issuer="https://idp.example.com",
        client_id="kumo",
        encrypted_client_secret="encrypted-client-secret-value",
    )
    credential = DirectConnectPasswordCredentialRecord(
        timestamps=_ts(),
        password_hash_b64="password-hash-value",
        salt_b64="password-salt-value",
        hash_params={
            "n": 16384,
            "r": 8,
            "p": 1,
            "dklen": 32
        },
    )
    secret_values = (
        "encrypted-client-secret-value",
        "password-hash-value",
        "password-salt-value",
    )
    for rendered in (
            repr(config),
            str(config),
            repr(config.to_log_dict()),
            repr(credential),
            str(credential),
            repr(credential.to_log_dict()),
    ):
        for secret_value in secret_values:
            assert secret_value not in rendered
        assert "<redacted>" in rendered


def test_direct_connect_secret_records_json_persistence_keeps_values() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    config = DirectConnectAuthConfigRecord(
        timestamps=_ts(),
        issuer="https://idp.example.com",
        client_id="kumo",
        encrypted_client_secret="encrypted-client-secret-value",
    )
    session = DirectConnectSessionRecord(
        timestamps=_ts(),
        user_id="user-1",
        expires_time=now,
    )
    credential = DirectConnectPasswordCredentialRecord(
        timestamps=_ts(),
        password_hash_b64="password-hash-value",
        salt_b64="password-salt-value",
        hash_params={
            "n": 16384,
            "r": 8,
            "p": 1,
            "dklen": 32
        },
    )
    config_payload = to_json_dict(config)
    session_payload = to_json_dict(session)
    credential_payload = to_json_dict(credential)
    restored_config = from_json_dict(config_payload,
                                     DirectConnectAuthConfigRecord)
    restored_credential = from_json_dict(
        credential_payload, DirectConnectPasswordCredentialRecord)

    assert (config_payload["encrypted_client_secret"] ==
            "encrypted-client-secret-value")
    assert "idp_group_hashes" not in session_payload
    assert credential_payload["password_hash_b64"] == "password-hash-value"
    assert credential_payload["salt_b64"] == "password-salt-value"
    assert (restored_config.encrypted_client_secret ==
            "encrypted-client-secret-value")
    assert restored_credential.password_hash_b64 == "password-hash-value"


@pytest.mark.asyncio
async def test_direct_connect_session_bearer_is_one_way_at_rest(
        tmp_path, caplog) -> None:
    db_path = tmp_path / "state.db"
    state_db = await StateDB.open_sqlite(str(db_path))
    service = direct_connect_auth.DirectConnectAuthService(state_db)
    raw_token = "raw-session-token-never-persist-or-log"
    expires_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    async with service.write() as writer:
        await writer.set_session(
            raw_token,
            DirectConnectSessionRecord(
                timestamps=_ts(),
                user_id="user-1",
                expires_time=expires_time,
            ),
        )

    async with state_db.read_tx() as tx:
        row = await tx.fetchrow(
            "SELECT id, payload FROM direct_connect_sessions LIMIT 1")
    assert row is not None
    stored_key = row["id"]
    exported_row = json.dumps({
        "id": stored_key,
        "payload": row["payload"],
    })
    assert stored_key == direct_connect_auth._session_lookup_key(raw_token)
    assert stored_key.startswith("v1:")
    assert raw_token not in exported_row
    assert raw_token not in caplog.text

    async with service.read() as reader:
        session = await reader.get_session(raw_token)
        # A disclosed StateDB row key is not itself a replayable cookie: the
        # request boundary treats it as an input token and hashes it again.
        assert await reader.get_session(stored_key) is None
    assert session is not None
    assert session.user_id == "user-1"
    assert session.expires_time == expires_time

    async with service.write() as writer:
        assert await writer.pop_session(stored_key) is None
        assert await writer.get_session(raw_token) is not None
        deleted = await writer.pop_session(raw_token)
        assert deleted is not None
        assert await writer.get_session(raw_token) is None

    await state_db.close()
    assert raw_token.encode("utf-8") not in db_path.read_bytes()


@pytest.mark.asyncio
async def test_direct_connect_auth_uses_sql_backed_email_lookup_and_cleanup(
        tmp_path) -> None:
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    service = direct_connect_auth.DirectConnectAuthService(state_db)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stale_before = now - timedelta(hours=1)

    async with service.write() as writer:
        await writer.set_user(
            "user-1",
            DirectConnectUserRecord(
                timestamps=_ts(),
                email="User@Example.COM ",
                display_name="User",
            ),
        )
        await writer.set_session(
            "expired-older",
            DirectConnectSessionRecord(
                timestamps=_ts(),
                user_id="user-1",
                expires_time=now - timedelta(hours=2),
            ),
        )
        await writer.set_session(
            "expired-newer",
            DirectConnectSessionRecord(
                timestamps=_ts(),
                user_id="user-1",
                expires_time=now - timedelta(hours=1),
            ),
        )
        await writer.set_session(
            "future",
            DirectConnectSessionRecord(
                timestamps=_ts(),
                user_id="user-1",
                expires_time=now + timedelta(hours=1),
            ),
        )
        await writer.set_password_reset_token(
            "expired-token",
            DirectConnectPasswordResetRecord(
                timestamps=_ts(),
                user_id="user-1",
                expires_time=now - timedelta(minutes=1),
            ),
        )
        await writer.set_password_reset_token(
            "used-token",
            DirectConnectPasswordResetRecord(
                timestamps=_ts(),
                user_id="user-1",
                expires_time=now + timedelta(hours=1),
                used=True,
            ),
        )
        await writer.set_password_reset_token(
            "future-token",
            DirectConnectPasswordResetRecord(
                timestamps=_ts(),
                user_id="user-1",
                expires_time=now + timedelta(hours=1),
            ),
        )
        await writer.set_password_login_attempt(
            "stale-unlocked",
            DirectConnectPasswordLoginAttemptRecord(
                timestamps=_ts(),
                first_failed_time=stale_before - timedelta(minutes=1),
            ),
        )
        await writer.set_password_login_attempt(
            "stale-locked",
            DirectConnectPasswordLoginAttemptRecord(
                timestamps=_ts(),
                first_failed_time=stale_before - timedelta(minutes=1),
                locked_until=now + timedelta(minutes=5),
            ),
        )
        await writer.set_password_login_attempt(
            "fresh-unlocked",
            DirectConnectPasswordLoginAttemptRecord(
                timestamps=_ts(),
                first_failed_time=now,
            ),
        )

    async with state_db.read_tx() as tx:
        row = await tx.fetchrow(
            "SELECT payload FROM direct_connect_users WHERE id = ?",
            "user-1",
        )
    assert row is not None
    assert json.loads(row["payload"])["normalized_email"] == "user@example.com"

    async with service.read() as reader:
        assert await reader.get_user_id_by_email(" user@example.com"
                                                 ) == "user-1"

    async with service.write() as writer:
        assert await writer.delete_expired_sessions(now=now, limit=1) == 1
        assert await writer.get_session("expired-older") is None
        assert await writer.get_session("expired-newer") is not None
        assert await writer.get_session("future") is not None

        assert await writer.delete_sessions_for_user("user-1") == 2
        assert await writer.get_session("expired-newer") is None
        assert await writer.get_session("future") is None

        assert (await writer.delete_expired_password_reset_tokens(
            now=now,
            limit=10,
        ) == 2)
        assert await writer.get_password_reset_token("expired-token") is None
        assert await writer.get_password_reset_token("used-token") is None
        assert await writer.get_password_reset_token("future-token") is not None

        assert (await writer.delete_expired_password_login_attempts(
            stale_before=stale_before,
            now=now,
            limit=10,
        ) == 1)
        assert await writer.get_password_login_attempt("stale-unlocked") is None
        assert await writer.get_password_login_attempt("stale-locked"
                                                       ) is not None
        assert await writer.get_password_login_attempt("fresh-unlocked"
                                                       ) is not None


@pytest.mark.asyncio
async def test_delete_expired_password_login_attempts_rechecks_stale_predicate(
        tmp_path, monkeypatch) -> None:
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    service = direct_connect_auth.DirectConnectAuthService(state_db)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stale_before = now - timedelta(hours=1)

    async with service.write() as writer:
        await writer.set_password_login_attempt(
            "stale-unlocked",
            DirectConnectPasswordLoginAttemptRecord(
                timestamps=_ts(),
                first_failed_time=stale_before - timedelta(minutes=1),
            ),
        )
        await writer.set_password_login_attempt(
            "refreshed-window",
            DirectConnectPasswordLoginAttemptRecord(
                timestamps=_ts(),
                first_failed_time=now,
            ),
        )
        await writer.set_password_login_attempt(
            "refreshed-lock",
            DirectConnectPasswordLoginAttemptRecord(
                timestamps=_ts(),
                first_failed_time=stale_before - timedelta(minutes=1),
                locked_until=now + timedelta(minutes=5),
            ),
        )

    async def select_stale_ids(*args: object, **kwargs: object) -> list[str]:
        return ["stale-unlocked", "refreshed-window", "refreshed-lock"]

    monkeypatch.setattr(direct_connect_auth, "_select_ids", select_stale_ids)

    async with service.write() as writer:
        assert (await writer.delete_expired_password_login_attempts(
            stale_before=stale_before,
            now=now,
            limit=10,
        ) == 1)
        assert await writer.get_password_login_attempt("stale-unlocked") is None
        assert await writer.get_password_login_attempt("refreshed-window"
                                                       ) is not None
        assert await writer.get_password_login_attempt("refreshed-lock"
                                                       ) is not None


@pytest.mark.asyncio
async def test_direct_connect_auth_records_login_failures_conditionally(
        tmp_path) -> None:
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    service = direct_connect_auth.DirectConnectAuthService(state_db)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    window = timedelta(minutes=10)
    lockout = timedelta(minutes=15)

    await service.record_password_login_failures(
        [("attempt-key", 2)],
        now=now,
        window=window,
        lockout=lockout,
    )
    await service.record_password_login_failures(
        [("attempt-key", 2)],
        now=now + timedelta(minutes=1),
        window=window,
        lockout=lockout,
    )

    async with service.read() as reader:
        locked = await reader.get_password_login_attempt("attempt-key")
    assert locked is not None
    assert locked.first_failed_time == now
    assert locked.failure_count == 2
    assert locked.locked_until == now + timedelta(minutes=1) + lockout

    await service.record_password_login_failures(
        [("attempt-key", 2)],
        now=now + timedelta(minutes=30),
        window=window,
        lockout=lockout,
    )

    async with service.read() as reader:
        reset = await reader.get_password_login_attempt("attempt-key")
    assert reset is not None
    assert reset.first_failed_time == now + timedelta(minutes=30)
    assert reset.failure_count == 1
    assert reset.locked_until is None


@pytest.mark.asyncio
async def test_direct_connect_auth_records_duplicate_login_failure_keys(
        tmp_path) -> None:
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    service = direct_connect_auth.DirectConnectAuthService(state_db)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    window = timedelta(minutes=10)
    lockout = timedelta(minutes=15)

    await service.record_password_login_failures(
        [("duplicate-key", 3), ("duplicate-key", 3)],
        now=now,
        window=window,
        lockout=lockout,
    )

    async with service.read() as reader:
        unlocked = await reader.get_password_login_attempt("duplicate-key")
    assert unlocked is not None
    assert unlocked.first_failed_time == now
    assert unlocked.failure_count == 2
    assert unlocked.locked_until is None

    second_now = now + timedelta(minutes=1)
    await service.record_password_login_failures(
        [("duplicate-key", 3), ("duplicate-key", 3)],
        now=second_now,
        window=window,
        lockout=lockout,
    )

    async with service.read() as reader:
        locked = await reader.get_password_login_attempt("duplicate-key")
    assert locked is not None
    assert locked.first_failed_time == now
    assert locked.failure_count == 4
    assert locked.locked_until == second_now + lockout


@pytest.mark.asyncio
async def test_direct_connect_auth_bulk_reads_login_failures(
        tmp_path, monkeypatch) -> None:
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    service = direct_connect_auth.DirectConnectAuthService(state_db)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    window = timedelta(minutes=10)
    lockout = timedelta(minutes=15)

    async with service.write() as writer:
        await writer.set_password_login_attempt(
            "existing-key",
            DirectConnectPasswordLoginAttemptRecord(
                timestamps=_ts(),
                first_failed_time=now - timedelta(minutes=1),
                failure_count=1,
            ),
        )

    original_get = JsonRepository.get
    original_list_by_ids = JsonRepository.list_by_ids
    bulk_read_ids: list[tuple[str, ...]] = []

    async def fail_attempt_scalar_get(self, tx, id):
        if self._table_name == "direct_connect_password_login_attempts":
            raise AssertionError("password login attempts must be bulk read")
        return await original_get(self, tx, id)

    async def spy_list_by_ids(self, tx, ids, **kwargs):
        if self._table_name == "direct_connect_password_login_attempts":
            bulk_read_ids.append(tuple(ids))
        return await original_list_by_ids(self, tx, ids, **kwargs)

    monkeypatch.setattr(JsonRepository, "get", fail_attempt_scalar_get)
    monkeypatch.setattr(JsonRepository, "list_by_ids", spy_list_by_ids)

    await service.record_password_login_failures(
        [("existing-key", 3), ("new-key", 2), ("second-new-key", 2)],
        now=now,
        window=window,
        lockout=lockout,
    )

    assert bulk_read_ids == [("existing-key", "new-key", "second-new-key")]

    monkeypatch.setattr(JsonRepository, "get", original_get)
    async with service.read() as reader:
        existing = await reader.get_password_login_attempt("existing-key")
        new = await reader.get_password_login_attempt("new-key")
        second_new = await reader.get_password_login_attempt("second-new-key")

    assert existing is not None
    assert existing.failure_count == 2
    assert existing.locked_until is None
    assert new is not None
    assert new.failure_count == 1
    assert second_new is not None
    assert second_new.failure_count == 1


@pytest.mark.asyncio
async def test_direct_connect_auth_locks_postgres_login_failure_keys_before_bulk_read(
) -> None:
    tx = _FakePostgresStateTx()
    service = direct_connect_auth.DirectConnectAuthService(
        _FakePostgresStateDB(tx))
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    await service.record_password_login_failures(
        [("key-b", 3), ("key-a", 2), ("key-b", 3)],
        now=now,
        window=timedelta(minutes=10),
        lockout=timedelta(minutes=15),
    )

    assert tx.execute_calls
    lock_sql, lock_args = tx.execute_calls[0]
    assert "pg_advisory_xact_lock" in lock_sql
    assert "direct_connect_password_login_attempts" in lock_sql
    assert lock_args == (["key-a", "key-b"], )

    assert tx.fetch_calls
    bulk_sql, bulk_args = tx.fetch_calls[0]
    assert "FROM direct_connect_password_login_attempts" in bulk_sql
    assert bulk_args == ("key-b", "key-a")


@pytest.mark.asyncio
async def test_direct_connect_auth_handles_tagged_and_untagged_password_reset(
        tmp_path) -> None:
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    service = direct_connect_auth.DirectConnectAuthService(state_db)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    old_credential = DirectConnectPasswordCredentialRecord(
        timestamps=_ts(),
        password_hash_b64="old-hash",
        salt_b64="old-salt",
        hash_params={
            "n": 16384,
            "r": 8,
            "p": 1,
            "dklen": 32
        },
        must_reset_password=True,
    )

    async with service.write() as writer:
        await writer.set_user(
            "user-1",
            DirectConnectUserRecord(
                timestamps=_ts(),
                email="user@example.com",
                display_name="User",
            ),
        )
        await writer.set_password_credential("user-1", old_credential)
        await writer.set_password_reset_token(
            "reset-key",
            DirectConnectPasswordResetRecord(
                timestamps=_ts(),
                user_id="user-1",
                expires_time=now + timedelta(minutes=10),
                credential_tag=direct_connect_password_credential_tag(
                    old_credential),
            ),
        )
        await writer.set_session(
            "session-1",
            DirectConnectSessionRecord(
                timestamps=_ts(),
                user_id="user-1",
                expires_time=now + timedelta(hours=1),
            ),
        )

    credential = DirectConnectPasswordCredentialRecord(
        timestamps=_ts(),
        password_hash_b64="new-hash",
        salt_b64="new-salt",
        hash_params={
            "n": 16384,
            "r": 8,
            "p": 1,
            "dklen": 32
        },
        must_reset_password=False,
    )
    status, user_id, user = await service.complete_password_reset(
        token_key="reset-key",
        now=now,
        expected_credential=old_credential,
        credential=credential,
    )
    second_status, second_user_id, second_user = (
        await service.complete_password_reset(
            token_key="reset-key",
            now=now,
            expected_credential=old_credential,
            credential=credential,
        ))

    assert (status, user_id) == ("ok", "user-1")
    assert user is not None
    assert user.email == "user@example.com"
    assert (second_status, second_user_id, second_user) == (
        "invalid_token",
        None,
        None,
    )
    async with service.read() as reader:
        reset_record = await reader.get_password_reset_token("reset-key")
        stored_credential = await reader.get_password_credential("user-1")
        session = await reader.get_session("session-1")
    assert reset_record is not None
    assert reset_record.used is True
    assert stored_credential is not None
    assert stored_credential.password_hash_b64 == "new-hash"
    assert stored_credential.must_reset_password is False
    assert session is None

    async with service.write() as writer:
        await writer.set_password_reset_token(
            "untagged-reset-key",
            DirectConnectPasswordResetRecord(
                timestamps=_ts(),
                user_id="user-1",
                expires_time=now + timedelta(minutes=10),
            ),
        )
    untagged_status, untagged_user_id, untagged_user = (
        await service.complete_password_reset(
            token_key="untagged-reset-key",
            now=now,
            expected_credential=credential,
            credential=old_credential,
        ))

    assert (untagged_status, untagged_user_id, untagged_user) == (
        "invalid_token",
        None,
        None,
    )
    async with service.read() as reader:
        untagged_reset_record = await reader.get_password_reset_token(
            "untagged-reset-key")
        stored_credential = await reader.get_password_credential("user-1")
    assert untagged_reset_record is not None
    assert untagged_reset_record.used is False
    assert stored_credential is not None
    assert stored_credential.password_hash_b64 == "new-hash"


@pytest.mark.asyncio
async def test_direct_connect_auth_update_user_preserves_last_admin(
        tmp_path) -> None:
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    service = direct_connect_auth.DirectConnectAuthService(state_db)

    async with service.write() as writer:
        await writer.set_user(
            "admin-1",
            DirectConnectUserRecord(
                timestamps=_ts(),
                email="admin-1@example.com",
                display_name="Admin 1",
                role="admin",
            ),
        )

    status, record = await service.update_user_if_not_last_admin(
        "admin-1",
        disabled=True,
    )
    assert status == "last_admin"
    assert record is not None
    assert record.disabled is False

    async with service.write() as writer:
        await writer.set_user(
            "admin-2",
            DirectConnectUserRecord(
                timestamps=_ts(),
                email="admin-2@example.com",
                display_name="Admin 2",
                role="admin",
            ),
        )

    status, record = await service.update_user_if_not_last_admin(
        "admin-1",
        disabled=True,
    )
    assert status == "updated"
    assert record is not None
    assert record.disabled is True

    status, record = await service.update_user_if_not_last_admin(
        "admin-2",
        role="user",
    )
    assert status == "last_admin"
    assert record is not None
    assert record.role == "admin"


@pytest.mark.asyncio
async def test_direct_connect_auth_update_user_invalidates_stale_sessions(
        tmp_path) -> None:
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    service = direct_connect_auth.DirectConnectAuthService(state_db)
    expires_time = datetime(2030, 1, 1, tzinfo=timezone.utc)

    async def _add_session(token: str, user_id: str) -> None:
        async with service.write() as writer:
            await writer.set_session(
                token,
                DirectConnectSessionRecord(
                    timestamps=_ts(),
                    user_id=user_id,
                    expires_time=expires_time,
                ),
            )

    async def _has_session(token: str) -> bool:
        async with service.read() as reader:
            return await reader.get_session(token) is not None

    async with service.write() as writer:
        for user_id, role in (
            ("admin-1", "admin"),
            ("admin-2", "admin"),
            ("user-3", "user"),
        ):
            await writer.set_user(
                user_id,
                DirectConnectUserRecord(
                    timestamps=_ts(),
                    email=f"{user_id}@example.com",
                    display_name=user_id,
                    role=role,
                ),
            )
    await _add_session("admin-1-a", "admin-1")
    await _add_session("admin-1-b", "admin-1")
    await _add_session("admin-2-a", "admin-2")
    await _add_session("user-3-a", "user-3")

    # Demotion deletes every session of the demoted user, nobody else's.
    status, record = await service.update_user_if_not_last_admin(
        "admin-1",
        role="user",
    )
    assert status == "updated"
    assert record is not None and record.role == "user"
    assert await _has_session("admin-1-a") is False
    assert await _has_session("admin-1-b") is False
    assert await _has_session("admin-2-a") is True
    assert await _has_session("user-3-a") is True

    # A no-op update (same role, same disabled) keeps sessions intact.
    await _add_session("admin-1-c", "admin-1")
    status, record = await service.update_user_if_not_last_admin(
        "admin-1",
        role="user",
    )
    assert status == "updated"
    assert await _has_session("admin-1-c") is True

    # Promotion is a role change too and forces re-authentication.
    status, record = await service.update_user_if_not_last_admin(
        "user-3",
        role="admin",
    )
    assert status == "updated"
    assert await _has_session("user-3-a") is False

    # Disabling deletes sessions (guarded path: admin-2 still active).
    await _add_session("user-3-b", "user-3")
    status, record = await service.update_user_if_not_last_admin(
        "user-3",
        disabled=True,
    )
    assert status == "updated"
    assert record is not None and record.disabled is True
    assert await _has_session("user-3-b") is False

    # A rejected last-admin demotion must leave the sessions untouched
    # (admin-2 is the only active admin while user-3 is disabled).
    status, record = await service.update_user_if_not_last_admin(
        "admin-2",
        role="user",
    )
    assert status == "last_admin"
    assert await _has_session("admin-2-a") is True

    # Re-enabling is not a privilege reduction; sessions are kept.
    await _add_session("user-3-c", "user-3")
    status, record = await service.update_user_if_not_last_admin(
        "user-3",
        disabled=False,
    )
    assert status == "updated"
    assert record is not None and record.disabled is False
    assert await _has_session("user-3-c") is True
