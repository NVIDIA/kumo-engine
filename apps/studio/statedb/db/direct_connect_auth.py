# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Optional

from rest.utils.driver_common import get_current_timestamps, update_timestamps
from statedb.config.direct_connect_auth import (
    DirectConnectAuthConfigRecord,
    DirectConnectIdentityLinkRecord,
    DirectConnectPasswordCredentialRecord,
    DirectConnectPasswordLoginAttemptRecord,
    DirectConnectPasswordResetRecord,
    DirectConnectSessionRecord,
    DirectConnectUserRecord,
    DirectConnectUserRole,
    direct_connect_password_credential_tag,
)
from statedb.config.timestamp import now_utc
from statedb.db.json_storage import (
    encode_timestamp,
    payload_dict_for_storage,
    payload_json_for_storage,
    utcnow,
)
from statedb.db.repositories import (
    JsonRepository,
    _encode_json_value,
    _in_predicate,
    _json_value,
    _placeholder,
)
from statedb.db.row_parsing import row_str
from statedb.db.runtime import StateDB, StateTx

_CONFIG_ROW_ID = "default"


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _identity_key(issuer: str, subject: str) -> str:
    return hashlib.sha256(f"{issuer}|{subject}".encode("utf-8")).hexdigest()


def _session_lookup_key(session_token: str) -> str:
    """Return the one-way StateDB key for a Direct Connect bearer token.

    Session tokens carry 256 bits of server-generated entropy, so a
    domain-separated SHA-256 digest is sufficient for lookup without storing
    a credential that a read-only database disclosure could replay.  The
    version prefix makes the stored value self-describing; the raw token is
    retained only by the caller/browser.
    """
    digest = hashlib.sha256(
        b"kumo-direct-connect-session-v1\0" + session_token.encode("utf-8"),
    ).hexdigest()
    return f"v1:{digest}"


class DirectConnectAuthStateReader:
    def __init__(self, tx: StateTx) -> None:
        self._tx = tx
        self._config = JsonRepository("direct_connect_config",
                                      DirectConnectAuthConfigRecord)
        self._users = JsonRepository("direct_connect_users",
                                     DirectConnectUserRecord)
        self._identity_links = JsonRepository(
            "direct_connect_identity_links",
            DirectConnectIdentityLinkRecord,
        )
        self._sessions = JsonRepository("direct_connect_sessions",
                                        DirectConnectSessionRecord)
        self._password_credentials = JsonRepository(
            "direct_connect_password_credentials",
            DirectConnectPasswordCredentialRecord,
        )
        self._password_reset_tokens = JsonRepository(
            "direct_connect_password_reset_tokens",
            DirectConnectPasswordResetRecord,
        )
        self._password_login_attempts = JsonRepository(
            "direct_connect_password_login_attempts",
            DirectConnectPasswordLoginAttemptRecord,
        )

    async def get_config(self) -> Optional[DirectConnectAuthConfigRecord]:
        return deepcopy(await self._config.get_payload(self._tx,
                                                       _CONFIG_ROW_ID))

    async def get_user(
        self,
        user_id: str,
    ) -> Optional[DirectConnectUserRecord]:
        return deepcopy(await self._users.get_payload(self._tx, user_id))

    async def list_users(self) -> list[tuple[str, DirectConnectUserRecord]]:
        return [(row.id, deepcopy(row.payload))
                for row in await self._users.list_all(self._tx)]

    async def get_user_id_by_email(self, email: str) -> Optional[str]:
        normalized_email = _normalize_email(email)
        row = await self._tx.fetchrow(
            f"""
            SELECT id
            FROM direct_connect_users
            WHERE {_json_value(self._tx, 'normalized_email')} = {_placeholder(self._tx, 1)}
               OR ({_json_value(self._tx, 'normalized_email')} IS NULL
                   AND {_normalized_email_sql(self._tx, 'email')} = {_placeholder(self._tx, 2)})
            ORDER BY id
            LIMIT 1
            """,
            normalized_email,
            normalized_email,
        )
        return _row_id(row) if row is not None else None

    async def get_user_id_by_identity(
        self,
        issuer: str,
        subject: str,
    ) -> Optional[str]:
        link = await self._identity_links.get_payload(
            self._tx,
            _identity_key(issuer, subject),
        )
        return link.user_id if link is not None else None

    async def get_identity_link_by_user_id(
        self,
        user_id: str,
    ) -> Optional[tuple[str, DirectConnectIdentityLinkRecord]]:
        row = await self._identity_links.find_one_by_fields(
            self._tx,
            {"user_id": user_id},
        )
        if row is None:
            return None
        return row.id, deepcopy(row.payload)

    async def get_session(
        self,
        session_id: str,
    ) -> Optional[DirectConnectSessionRecord]:
        return deepcopy(await self._sessions.get_payload(
            self._tx, _session_lookup_key(session_id)))

    async def get_password_credential(
        self,
        user_id: str,
    ) -> Optional[DirectConnectPasswordCredentialRecord]:
        return deepcopy(await self._password_credentials.get_payload(
            self._tx, user_id))

    async def get_password_reset_token(
        self,
        token_key: str,
    ) -> Optional[DirectConnectPasswordResetRecord]:
        return deepcopy(await self._password_reset_tokens.get_payload(
            self._tx, token_key))

    async def get_password_login_attempt(
        self,
        key: str,
    ) -> Optional[DirectConnectPasswordLoginAttemptRecord]:
        return deepcopy(await self._password_login_attempts.get_payload(
            self._tx, key))

    async def list_password_login_attempts_by_key(
        self,
        keys: list[str],
    ) -> dict[str, DirectConnectPasswordLoginAttemptRecord]:
        rows = await self._password_login_attempts.list_by_ids(self._tx, keys)
        ordered = JsonRepository.rows_in_id_order(rows, keys)
        return {row.id: deepcopy(row.payload) for row in ordered}


class DirectConnectAuthStateWriter(DirectConnectAuthStateReader):
    async def set_config(self, config: DirectConnectAuthConfigRecord) -> None:
        config = deepcopy(config)
        config.timestamps = update_timestamps(config.timestamps)
        await self._config.upsert(self._tx, _CONFIG_ROW_ID, config)

    async def set_user(
        self,
        user_id: str,
        record: DirectConnectUserRecord,
    ) -> None:
        record = deepcopy(record)
        record.timestamps = update_timestamps(record.timestamps)
        await self._users.upsert(
            self._tx,
            user_id,
            record,
            extra_fields={"normalized_email": _normalize_email(record.email)},
        )

    async def set_identity_user_id(
        self,
        issuer: str,
        subject: str,
        user_id: str,
        encrypted_refresh_token: str,
    ) -> None:
        identity_link_id = _identity_key(issuer, subject)
        await self.lock_identity_refresh(user_id)
        await self.lock_identity_link(identity_link_id)
        previous_link_ids = await _select_ids(
            self._tx,
            "direct_connect_identity_links",
            (f"{_json_value(self._tx, 'user_id')} = "
             f"{_placeholder(self._tx, 1)} AND id <> "
             f"{_placeholder(self._tx, 2)}"),
            [user_id, identity_link_id],
            "id",
        )
        await _delete_ids(
            self._tx,
            "direct_connect_identity_links",
            previous_link_ids,
        )
        await self._identity_links.upsert(
            self._tx,
            identity_link_id,
            DirectConnectIdentityLinkRecord(
                timestamps=get_current_timestamps(),
                user_id=user_id,
                encrypted_refresh_token=encrypted_refresh_token,
            ),
        )

    async def lock_identity_refresh(self, user_id: str) -> None:
        if self._tx.backend != "postgres":
            return
        await self._tx.fetchrow(
            """
            SELECT pg_advisory_xact_lock(
                hashtext('direct_connect_identity_refresh'),
                hashtext($1)
            )
            """,
            user_id,
        )

    async def lock_identity_link(self, identity_link_id: str) -> None:
        if self._tx.backend != "postgres":
            return
        await self._tx.fetchrow(
            """
            SELECT pg_advisory_xact_lock(
                hashtext('direct_connect_identity_link'),
                hashtext($1)
            )
            """,
            identity_link_id,
        )

    async def get_identity_link_for_refresh(
        self,
        user_id: str,
    ) -> Optional[tuple[str, DirectConnectIdentityLinkRecord]]:
        await self.lock_identity_refresh(user_id)
        for_update = "FOR UPDATE" if self._tx.backend == "postgres" else ""
        rows = await self._tx.fetch(
            f"""
            SELECT id, payload, created_at, updated_at
            FROM direct_connect_identity_links
            WHERE {_json_value(self._tx, 'user_id')} = {_placeholder(self._tx, 1)}
            ORDER BY id
            {for_update}
            """,
            user_id,
        )
        if len(rows) > 1:
            raise ValueError(
                f"Expected at most one identity link for user {user_id!r}")
        if not rows:
            return None
        row = self._identity_links._row_from_list_row(rows[0])
        return row.id, deepcopy(row.payload)

    async def set_identity_refresh_token(
        self,
        identity_link_id: str,
        record: DirectConnectIdentityLinkRecord,
        encrypted_refresh_token: str,
    ) -> None:
        record = deepcopy(record)
        record.encrypted_refresh_token = encrypted_refresh_token
        record.timestamps = update_timestamps(record.timestamps)
        await self._identity_links.upsert(self._tx, identity_link_id, record)

    async def set_session(
        self,
        session_id: str,
        record: DirectConnectSessionRecord,
    ) -> None:
        record = deepcopy(record)
        record.timestamps = update_timestamps(record.timestamps)
        await self._sessions.upsert(
            self._tx, _session_lookup_key(session_id), record)

    async def pop_session(
        self,
        session_id: str,
    ) -> Optional[DirectConnectSessionRecord]:
        existing = await self.get_session(session_id)
        if existing is not None:
            await self._sessions.delete(
                self._tx, _session_lookup_key(session_id))
        return existing

    async def delete_sessions_for_user(self, user_id: str) -> int:
        session_ids = await _select_ids(
            self._tx,
            "direct_connect_sessions",
            f"{_json_value(self._tx, 'user_id')} = {_placeholder(self._tx, 1)}",
            [user_id],
            "id",
        )
        await _delete_ids(self._tx, "direct_connect_sessions", session_ids)
        return len(session_ids)

    async def delete_expired_sessions(
        self,
        now: Optional[datetime] = None,
        limit: int = 100,
    ) -> int:
        now = now or now_utc()
        expired = await _select_ids(
            self._tx,
            "direct_connect_sessions",
            f"{_json_value(self._tx, 'expires_time')} <= {_placeholder(self._tx, 1)}",
            [_datetime_json_value(now)],
            f"{_json_value(self._tx, 'expires_time')}, id",
            limit=limit,
        )
        await _delete_ids(self._tx, "direct_connect_sessions", expired)
        return len(expired)

    async def set_password_credential(
        self,
        user_id: str,
        record: DirectConnectPasswordCredentialRecord,
    ) -> None:
        record = deepcopy(record)
        record.timestamps = update_timestamps(record.timestamps)
        await self._password_credentials.upsert(self._tx, user_id, record)

    async def set_password_reset_token(
        self,
        token_key: str,
        record: DirectConnectPasswordResetRecord,
    ) -> None:
        record = deepcopy(record)
        record.timestamps = update_timestamps(record.timestamps)
        await self._password_reset_tokens.upsert(self._tx, token_key, record)

    async def delete_expired_password_reset_tokens(
        self,
        now: Optional[datetime] = None,
        limit: int = 100,
    ) -> int:
        now = now or now_utc()
        expired = await _select_ids(
            self._tx,
            "direct_connect_password_reset_tokens",
            f"{_json_value(self._tx, 'used')} = {_placeholder(self._tx, 1)} "
            f"OR {_json_value(self._tx, 'expires_time')} <= {_placeholder(self._tx, 2)}",
            [_encode_json_value(self._tx, True),
             _datetime_json_value(now)],
            f"{_json_value(self._tx, 'expires_time')}, id",
            limit=limit,
        )
        await _delete_ids(self._tx, "direct_connect_password_reset_tokens",
                          expired)
        return len(expired)

    async def set_password_login_attempt(
        self,
        key: str,
        record: DirectConnectPasswordLoginAttemptRecord,
    ) -> None:
        record = deepcopy(record)
        record.timestamps = update_timestamps(record.timestamps)
        await self._password_login_attempts.upsert(self._tx, key, record)

    async def pop_password_login_attempt(
        self,
        key: str,
    ) -> Optional[DirectConnectPasswordLoginAttemptRecord]:
        existing = await self.get_password_login_attempt(key)
        if existing is not None:
            await self._password_login_attempts.delete(self._tx, key)
        return existing

    async def lock_password_login_attempt_keys(self, keys: list[str]) -> None:
        if self._tx.backend != "postgres":
            return
        lock_keys = sorted(set(keys))
        if not lock_keys:
            return
        # Lock the logical keys before reading so concurrent failures cannot
        # race on rows that do not exist yet; SELECT FOR UPDATE only protects
        # existing rows.
        await self._tx.execute(
            """
            WITH ordered_keys AS (
                SELECT attempt_key
                FROM unnest($1::text[]) AS keys(attempt_key)
                ORDER BY attempt_key
            )
            SELECT pg_advisory_xact_lock(
                hashtext('direct_connect_password_login_attempts'),
                hashtext(attempt_key)
            )
            FROM ordered_keys
            """,
            lock_keys,
        )

    async def delete_expired_password_login_attempts(
        self,
        *,
        stale_before: datetime,
        now: Optional[datetime] = None,
        limit: int = 100,
    ) -> int:
        now = now or now_utc()
        stale_where_sql = (
            f"({_json_value(self._tx, 'locked_until')} IS NULL "
            f"OR {_json_value(self._tx, 'locked_until')} <= {_placeholder(self._tx, 1)}) "
            f"AND {_json_value(self._tx, 'first_failed_time')} <= {_placeholder(self._tx, 2)}"
        )
        stale_args = [
            _datetime_json_value(now),
            _datetime_json_value(stale_before),
        ]
        expired = await _select_ids(
            self._tx,
            "direct_connect_password_login_attempts",
            stale_where_sql,
            list(stale_args),
            f"{_json_value(self._tx, 'first_failed_time')}, id",
            limit=limit,
        )
        return await _delete_ids_where(
            self._tx,
            "direct_connect_password_login_attempts",
            expired,
            stale_where_sql,
            stale_args,
        )


def _row_id(row: object) -> str:
    return row_str(row, "id", 0)


def _datetime_json_value(value: datetime) -> str:
    return value.isoformat()


def _normalized_email_sql(tx: StateTx, field_name: str) -> str:
    value = _json_value(tx, field_name)
    return (f"lower(btrim({value}))"
            if tx.backend == "postgres" else f"lower(trim({value}))")


def _aliased_json_value(tx: StateTx, table_alias: str, field_name: str) -> str:
    if tx.backend == "postgres":
        return f"{table_alias}.payload->>'{field_name}'"
    return f"json_extract({table_alias}.payload, '$.{field_name}')"


async def _select_ids(
    tx: StateTx,
    table_name: str,
    where_sql: str,
    args: list[object],
    order_by_sql: str,
    *,
    limit: int | None = None,
) -> list[str]:
    limit_sql = ""
    if limit is not None:
        args.append(limit)
        limit_sql = f"LIMIT {_placeholder(tx, len(args))}"
    rows = await tx.fetch(
        f"""
        SELECT id
        FROM {table_name}
        WHERE {where_sql}
        ORDER BY {order_by_sql}
        {limit_sql}
        """,
        *args,
    )
    return [_row_id(row) for row in rows]


async def _delete_ids(tx: StateTx, table_name: str, row_ids: list[str]) -> None:
    if not row_ids:
        return
    args: list[object] = []
    predicate = _in_predicate(tx, "id", row_ids, args)
    await tx.execute(
        f"""
        DELETE FROM {table_name}
        WHERE {predicate}
        """,
        *args,
    )


async def _delete_ids_where(
    tx: StateTx,
    table_name: str,
    row_ids: list[str],
    where_sql: str,
    args: list[object],
) -> int:
    if not row_ids:
        return 0
    delete_args = list(args)
    id_predicate = _in_predicate(tx, "id", row_ids, delete_args)
    return await _execute_count(
        tx,
        f"""
        DELETE FROM {table_name}
        WHERE ({where_sql})
          AND {id_predicate}
        """,
        delete_args,
    )


async def _execute_count(tx: StateTx, sql: str, args: list[object]) -> int:
    result = await tx.execute(sql, *args)
    if tx.backend == "postgres":
        try:
            return int(str(result).split()[-1])
        except (IndexError, ValueError):
            return 0
    row = await tx.fetchrow("SELECT changes()")
    return int(row[0] if row is not None else 0)


async def _conditional_update_payload(
    tx: StateTx,
    table_name: str,
    row_id: str,
    record: object,
    *,
    extra_where_sql: str,
    extra_args: list[object],
    search_fields: dict[str, object] | None = None,
) -> int:
    record = deepcopy(record)
    if hasattr(record, "timestamps"):
        record.timestamps = update_timestamps(record.timestamps)
    payload_json = payload_json_for_storage(
        payload_dict_for_storage(record, extra_fields=search_fields))
    updated_at = encode_timestamp(tx, utcnow())
    payload_expr = (f"{_placeholder(tx, 1)}::jsonb"
                    if tx.backend == "postgres" else _placeholder(tx, 1))
    return await _execute_count(
        tx,
        f"""
        UPDATE {table_name}
        SET payload = {payload_expr},
            updated_at = {_placeholder(tx, 2)}
        WHERE id = {_placeholder(tx, 3)}
          AND {extra_where_sql}
        """,
        [payload_json, updated_at, row_id, *extra_args],
    )


class DirectConnectAuthService:
    def __init__(self, state_db: StateDB) -> None:
        self._state_db = state_db

    async def init_once(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def get_config(self) -> Optional[DirectConnectAuthConfigRecord]:
        async with self.read() as reader:
            return await reader.get_config()

    async def set_config(self, config: DirectConnectAuthConfigRecord) -> None:
        async with self.write() as writer:
            await writer.set_config(config)

    async def list_users(self, ) -> list[tuple[str, DirectConnectUserRecord]]:
        async with self.read() as reader:
            return await reader.list_users()

    async def get_user(
        self,
        user_id: str,
    ) -> Optional[DirectConnectUserRecord]:
        async with self.read() as reader:
            return await reader.get_user(user_id)

    async def get_user_by_email(
        self,
        email: str,
    ) -> Optional[tuple[str, DirectConnectUserRecord]]:
        async with self.read() as reader:
            user_id = await reader.get_user_id_by_email(email)
            if not user_id:
                return None
            record = await reader.get_user(user_id)
            if record is None:
                return None
            return user_id, record

    @asynccontextmanager
    async def serialized_identity_refresh(
        self,
        user_id: str,
    ) -> AsyncGenerator[
            tuple[DirectConnectAuthStateWriter,
                  Optional[tuple[str, DirectConnectIdentityLinkRecord]]], None]:
        """Hold the cross-replica identity lock through one IdP refresh."""
        async with self._state_db.write_tx() as tx:
            writer = DirectConnectAuthStateWriter(tx)
            linked = await writer.get_identity_link_for_refresh(user_id)
            yield writer, linked

    async def get_session(
        self,
        session_id: str,
    ) -> Optional[DirectConnectSessionRecord]:
        async with self.read() as reader:
            return await reader.get_session(session_id)

    async def pop_session(
        self,
        session_id: str,
    ) -> Optional[DirectConnectSessionRecord]:
        async with self.write() as writer:
            return await writer.pop_session(session_id)

    async def record_password_login_failures(
        self,
        attempts: list[tuple[str, int]],
        *,
        now: datetime,
        window: timedelta,
        lockout: timedelta,
        stale_before: datetime | None = None,
        cleanup_limit: int = 100,
    ) -> None:
        async with self.write() as writer:
            keys = list(dict.fromkeys(key for key, _ in attempts))
            await writer.lock_password_login_attempt_keys(keys)
            if stale_before is not None:
                await writer.delete_expired_password_login_attempts(
                    stale_before=stale_before,
                    now=now,
                    limit=cleanup_limit,
                )
            records = await writer.list_password_login_attempts_by_key(keys)
            for key, limit in attempts:
                record = records.get(key)
                if (record is None or record.first_failed_time <= now - window):
                    record = DirectConnectPasswordLoginAttemptRecord(
                        timestamps=get_current_timestamps(),
                        first_failed_time=now,
                        failure_count=0,
                    )
                    records[key] = record
                record.failure_count += 1
                if record.failure_count >= limit:
                    record.locked_until = now + lockout
                await writer.set_password_login_attempt(key, record)

    async def complete_password_reset(
        self,
        *,
        token_key: str,
        now: datetime,
        expected_credential: DirectConnectPasswordCredentialRecord | None,
        credential: DirectConnectPasswordCredentialRecord,
    ) -> tuple[str, str | None, DirectConnectUserRecord | None]:
        async with self._state_db.write_tx() as tx:
            writer = DirectConnectAuthStateWriter(tx)
            row = await tx.fetchrow(
                f"""
                SELECT payload, created_at, updated_at
                FROM direct_connect_password_reset_tokens
                WHERE id = {_placeholder(tx, 1)}
                  AND {_json_value(tx, 'used')} = {_placeholder(tx, 2)}
                  AND {_json_value(tx, 'expires_time')} > {_placeholder(tx, 3)}
                LIMIT 1
                """,
                token_key,
                _encode_json_value(tx, False),
                _datetime_json_value(now),
            )
            if row is None:
                return "invalid_token", None, None

            reset_record = (writer._password_reset_tokens._row_from_db_row(
                token_key,
                row,
            ).payload)
            user = await writer.get_user(reset_record.user_id)
            if user is None:
                return "invalid_token", None, None
            if user.disabled:
                return "disabled", reset_record.user_id, user
            current_credential = await writer.get_password_credential(
                reset_record.user_id)
            current_credential_tag = direct_connect_password_credential_tag(
                current_credential)
            if current_credential_tag != direct_connect_password_credential_tag(
                    expected_credential):
                return "invalid_token", None, None
            if reset_record.credential_tag is None:
                return "invalid_token", None, None
            if reset_record.credential_tag != current_credential_tag:
                return "invalid_token", None, None

            reset_record.used = True
            updated = await _conditional_update_payload(
                tx,
                "direct_connect_password_reset_tokens",
                token_key,
                reset_record,
                extra_where_sql=(
                    f"{_json_value(tx, 'used')} = {_placeholder(tx, 4)} "
                    f"AND {_json_value(tx, 'expires_time')} > "
                    f"{_placeholder(tx, 5)}"),
                extra_args=[
                    _encode_json_value(tx, False),
                    _datetime_json_value(now),
                ],
            )
            if updated != 1:
                return "invalid_token", None, None

            await writer.set_password_credential(
                reset_record.user_id,
                credential,
            )
            await writer.delete_sessions_for_user(reset_record.user_id)
            return "ok", reset_record.user_id, user

    async def update_user_if_not_last_admin(
        self,
        user_id: str,
        *,
        role: DirectConnectUserRole | None = None,
        disabled: bool | None = None,
    ) -> tuple[str, DirectConnectUserRecord | None]:
        async with self._state_db.write_tx() as tx:
            writer = DirectConnectAuthStateWriter(tx)
            record = await writer.get_user(user_id)
            if record is None:
                return "not_found", None

            proposed_role = role or record.role
            proposed_disabled = (record.disabled
                                 if disabled is None else disabled)
            demoting_last_admin = (record.role == "admin"
                                   and not record.disabled
                                   and (proposed_role != "admin"
                                        or proposed_disabled))
            # A role change or a disable must end the user's live sessions
            # in the same transaction (like password reset above), so the
            # user re-authenticates under the new privileges instead of
            # keeping a stale session; no-op updates keep sessions intact.
            invalidate_sessions = (proposed_role != record.role
                                   or (proposed_disabled
                                       and not record.disabled))

            record.role = proposed_role
            record.disabled = proposed_disabled
            if not demoting_last_admin:
                await writer.set_user(user_id, record)
                if invalidate_sessions:
                    await writer.delete_sessions_for_user(user_id)
                return "updated", record

            if tx.backend == "postgres":
                await tx.fetch(
                    f"""
                    SELECT id
                    FROM direct_connect_users
                    WHERE {_json_value(tx, 'role')} = {_placeholder(tx, 1)}
                      AND ({_json_value(tx, 'disabled')} = {_placeholder(tx, 2)}
                           OR {_json_value(tx, 'disabled')} IS NULL)
                    FOR UPDATE
                    """,
                    "admin",
                    _encode_json_value(tx, False),
                )

            other_role_expr = _aliased_json_value(tx, "other", "role")
            other_disabled_expr = _aliased_json_value(tx, "other", "disabled")
            updated = await _conditional_update_payload(
                tx,
                "direct_connect_users",
                user_id,
                record,
                extra_where_sql=(
                    "EXISTS ("
                    "SELECT 1 FROM direct_connect_users other "
                    f"WHERE other.id <> {_placeholder(tx, 4)} "
                    f"AND {other_role_expr} = {_placeholder(tx, 5)} "
                    f"AND ({other_disabled_expr} = {_placeholder(tx, 6)} "
                    f"OR {other_disabled_expr} IS NULL))"),
                extra_args=[
                    user_id,
                    "admin",
                    _encode_json_value(tx, False),
                ],
                search_fields={
                    "normalized_email": _normalize_email(record.email)
                },
            )
            if updated == 1:
                if invalidate_sessions:
                    await writer.delete_sessions_for_user(user_id)
                return "updated", record

            existing = await writer.get_user(user_id)
            return "last_admin", existing

    @asynccontextmanager
    async def read(
        self, ) -> AsyncGenerator[DirectConnectAuthStateReader, None]:
        async with self._state_db.read_tx() as tx:
            yield DirectConnectAuthStateReader(tx)

    @asynccontextmanager
    async def write(
        self, ) -> AsyncGenerator[DirectConnectAuthStateWriter, None]:
        async with self._state_db.write_tx() as tx:
            yield DirectConnectAuthStateWriter(tx)
