# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

import asyncpg

from statedb.db.schema import (
    SCHEMA_ROW_ID,
    SCHEMA_VERSION,
    STATE_DB_TABLES,
    BackendName,
    IndexSpec,
    TableSpec,
    legacy_schema_identity,
    schema_identity,
    validate_identifier,
)


class StateDBSchemaMismatchError(RuntimeError):
    pass


class StateDBBootstrapBusyError(RuntimeError):
    """SQLite schema bootstrap could not obtain its bounded write lease."""


_POSTGRES_BOOTSTRAP_ADVISORY_LOCK_ID = 0x5354415445444201
_SQLITE_BOOTSTRAP_BUSY_TIMEOUT_MS = 250
_SQLITE_BOOTSTRAP_MAX_ATTEMPTS = 8
_SQLITE_BOOTSTRAP_INITIAL_BACKOFF_SECONDS = 0.025
_SQLITE_BOOTSTRAP_MAX_BACKOFF_SECONDS = 0.2
_SQLITE_BOOTSTRAP_LOCKS_GUARD = threading.Lock()
_SQLITE_BOOTSTRAP_LOCKS_PID = os.getpid()
_SQLITE_BOOTSTRAP_LOCKS: dict[str, threading.Lock] = {}
PostgresSchemaMigration = Callable[[asyncpg.Connection], Awaitable[None]]
log = logging.getLogger(__name__)


def _reset_sqlite_bootstrap_locks_after_fork() -> None:
    """Replace process-local lock state inherited by a forked child."""
    global _SQLITE_BOOTSTRAP_LOCKS_GUARD
    global _SQLITE_BOOTSTRAP_LOCKS_PID
    global _SQLITE_BOOTSTRAP_LOCKS
    _SQLITE_BOOTSTRAP_LOCKS_GUARD = threading.Lock()
    _SQLITE_BOOTSTRAP_LOCKS_PID = os.getpid()
    _SQLITE_BOOTSTRAP_LOCKS = {}


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_reset_sqlite_bootstrap_locks_after_fork)


@dataclass(frozen=True)
class BootstrapResult:
    backend: BackendName
    schema_hash: str
    table_count: int
    index_count: int
    created_schema_row: bool
    applied_migrations: tuple[int, ...] = ()
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class _ExistingIndex:
    name: str
    table_name: str
    definition: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _schema_payload(schema_hash: str, schema_version: int) -> dict[str, Any]:
    return {"schema_hash": schema_hash, "schema_version": schema_version}


def _schema_version_from_payload(payload_text: str | None) -> int | None:
    if payload_text is None:
        return None
    version = _decode_schema_payload(payload_text).get("schema_version")
    return version if type(version) is int else None


def _schema_hash_from_payload(payload_text: str | None) -> str | None:
    if payload_text is None:
        return None
    schema_hash = _decode_schema_payload(payload_text).get("schema_hash")
    return schema_hash if isinstance(schema_hash, str) else None


def _sqlite_table_ddl(table_name: str) -> str:
    validate_identifier(table_name)
    return f"""
        CREATE TABLE {table_name} (
            id TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """


def _postgres_table_ddl(table_name: str) -> str:
    validate_identifier(table_name)
    return f"""
        CREATE TABLE {table_name} (
            id TEXT PRIMARY KEY,
            payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
    """


def _schema_mismatch_message(found: str | None, expected: str) -> str:
    return f"State DB schema hash mismatch: found={found!r}, expected={expected!r}"


def _expected_table_names(tables: tuple[TableSpec, ...]) -> set[str]:
    return {table.name for table in tables}


def _expected_index_specs(
    tables: tuple[TableSpec, ...], ) -> dict[str, tuple[str, IndexSpec]]:
    indexes: dict[str, tuple[str, IndexSpec]] = {}
    for table in tables:
        for index in table.indexes:
            if index.name in indexes:
                raise ValueError(f"Duplicate State DB index: {index.name!r}")
            indexes[index.name] = (table.name, index)
    return indexes


def _schema_exists(
    existing_tables: set[str],
    existing_indexes: dict[str, _ExistingIndex],
) -> bool:
    return bool(existing_tables or existing_indexes)


def _decode_schema_payload(payload_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise StateDBSchemaMismatchError(
            "State DB schema metadata is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise StateDBSchemaMismatchError(
            "State DB schema metadata must be a JSON object")
    return payload


def _normalize_sqlite_ddl(sql: str) -> str:
    normalized = " ".join(sql.strip().rstrip(";").split())
    index_ddl = _normalize_sqlite_index_ddl(normalized)
    return index_ddl if index_ddl is not None else normalized


def _sqlite_identifier_pattern() -> str:
    return r'(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)'


def _normalize_sqlite_identifier(identifier: str) -> str:
    if identifier.startswith('"') and identifier.endswith('"'):
        identifier = identifier[1:-1]
    validate_identifier(identifier)
    return identifier


def _matching_paren_position(sql: str, open_position: int) -> int:
    depth = 0
    quote: str | None = None
    for position in range(open_position, len(sql)):
        char = sql[position]
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == "(":
            depth += 1
            continue
        if char == ")":
            depth -= 1
            if depth == 0:
                return position
    return -1


def _outer_parentheses_wrap_expression(expression: str) -> bool:
    if not expression.startswith("(") or not expression.endswith(")"):
        return False
    return _matching_paren_position(expression, 0) == len(expression) - 1


def _strip_redundant_outer_parentheses(expression: str) -> str:
    expression = expression.strip()
    while _outer_parentheses_wrap_expression(expression):
        expression = expression[1:-1].strip()
    return expression


def _normalize_sqlite_expression(expression: str) -> str:
    expression = " ".join(expression.split())
    expression = _strip_redundant_outer_parentheses(expression)
    expression = re.sub(r"\s*([(),])\s*", r"\1", expression)
    expression = re.sub(r"\bis\s+not\s+null\b", "IS NOT NULL", expression,
                        flags=re.IGNORECASE)
    return expression


def _normalize_sqlite_index_ddl(sql: str) -> str | None:
    identifier = _sqlite_identifier_pattern()
    prefix = re.match(
        rf"^CREATE\s+(?P<unique>UNIQUE\s+)?INDEX\s+"
        rf"(?P<index>{identifier})\s+ON\s+(?P<table>{identifier})\s*",
        sql,
        flags=re.IGNORECASE,
    )
    if prefix is None:
        return None
    expression_open = prefix.end()
    if expression_open >= len(sql) or sql[expression_open] != "(":
        return None
    expression_close = _matching_paren_position(sql, expression_open)
    if expression_close < 0:
        return None

    index_name = _normalize_sqlite_identifier(prefix.group("index"))
    table_name = _normalize_sqlite_identifier(prefix.group("table"))
    expression = _normalize_sqlite_expression(sql[expression_open +
                                                  1:expression_close])

    suffix = sql[expression_close + 1:].strip()
    where_sql = ""
    if suffix:
        where_match = re.match(r"^WHERE\s+(.+)$", suffix, flags=re.IGNORECASE)
        if where_match is None:
            return None
        where_sql = " WHERE " + _normalize_sqlite_expression(
            where_match.group(1))

    unique = "UNIQUE " if prefix.group("unique") else ""
    return (f"CREATE {unique}INDEX {index_name} "
            f"ON {table_name} ({expression}){where_sql}")


def _normalize_postgres_index_ddl(sql: str) -> str:
    normalized = " ".join(sql.strip().rstrip(";").lower().split())
    normalized = normalized.replace(" if not exists", "")
    normalized = normalized.replace(" using btree", "")
    normalized = re.sub(r"\bon\s+[a-z_][a-z0-9_]*\.", "on ", normalized)
    normalized = normalized.replace("::text[]", "")
    normalized = normalized.replace("::text", "")
    normalized = re.sub(r"\s*(->>|#>>)\s*", r"\1", normalized)
    where_match = re.search(r"\s+where\s+", normalized)
    if where_match is None:
        ddl = normalized
        predicate = None
    else:
        ddl = normalized[:where_match.start()]
        predicate = normalized[where_match.end():]

    ddl = _strip_postgres_json_expression_parentheses(ddl)
    ddl = re.sub(r"\s*([(),])\s*", r"\1", ddl)
    if predicate is not None:
        predicate = _strip_postgres_json_expression_parentheses(predicate)
        predicate = _strip_balanced_outer_parentheses(predicate)
        predicate = re.sub(r"\s*([(),])\s*", r"\1", predicate)
        predicate = re.sub(r"\s*(<>|=)\s*", r"\1", predicate)
        return f"{ddl} where {predicate}"
    return ddl


def _strip_postgres_json_expression_parentheses(value: str) -> str:
    """Canonicalize pg_get_indexdef's redundant JSON-expression wrappers."""
    pattern = re.compile(r"\((payload(?:->>|#>>)'[^']*')\)")
    while True:
        normalized = pattern.sub(r"\1", value)
        if normalized == value:
            return value
        value = normalized


def _strip_balanced_outer_parentheses(value: str) -> str:
    """Remove only parentheses enclosing an entire SQL expression.

    ``pg_get_indexdef`` wraps partial-index predicates, including predicates
    containing calls such as ``COALESCE(...)``.  A flat regular expression
    cannot distinguish that wrapper from the function's own parentheses.
    """
    value = value.strip()
    while value.startswith("(") and value.endswith(")"):
        depth = 0
        encloses_entire_value = True
        for position, character in enumerate(value):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth < 0:
                    return value
                if depth == 0 and position != len(value) - 1:
                    encloses_entire_value = False
                    break
        if depth != 0 or not encloses_entire_value:
            break
        value = value[1:-1].strip()
    return value


def _index_matches(
    existing_index: _ExistingIndex,
    table_name: str,
    index: IndexSpec,
    backend: BackendName,
) -> bool:
    if existing_index.table_name != table_name:
        return False
    expected_ddl = index.ddl(table_name, backend)
    if backend == "sqlite":
        return (_normalize_sqlite_ddl(
            existing_index.definition) == _normalize_sqlite_ddl(expected_ddl))
    return (_normalize_postgres_index_ddl(existing_index.definition) ==
            _normalize_postgres_index_ddl(expected_ddl))


def _validate_existing_schema(
    *,
    backend: BackendName,
    tables: tuple[TableSpec, ...],
    expected_hash: str,
    existing_tables: set[str],
    existing_indexes: dict[str, _ExistingIndex],
    payload_text: str | None,
    compatible_hashes: tuple[str, ...] = (),
) -> None:
    expected_tables = _expected_table_names(tables)
    expected_indexes = _expected_index_specs(tables)

    extra_tables = sorted(existing_tables - expected_tables)
    if extra_tables:
        raise StateDBSchemaMismatchError(
            "State DB schema has unexpected tables: "
            f"{extra_tables}")

    missing_tables = sorted(expected_tables - existing_tables)
    if missing_tables:
        raise StateDBSchemaMismatchError(
            "State DB schema is missing State DB tables: "
            f"{missing_tables}")

    missing_indexes = sorted(set(expected_indexes) - set(existing_indexes))
    if missing_indexes:
        raise StateDBSchemaMismatchError(
            "State DB schema is missing State DB indexes: "
            f"{missing_indexes}")

    extra_indexes = sorted(set(existing_indexes) - set(expected_indexes))
    if extra_indexes:
        raise StateDBSchemaMismatchError(
            "State DB schema has unexpected State DB indexes: "
            f"{extra_indexes}")

    wrong_indexes = sorted(index_name
                           for index_name, (table_name,
                                            index) in expected_indexes.items()
                           if not _index_matches(existing_indexes[index_name],
                                                 table_name, index, backend))
    if wrong_indexes:
        raise StateDBSchemaMismatchError(
            "State DB schema has wrong State DB indexes: "
            f"{wrong_indexes}")

    payload = (_decode_schema_payload(payload_text)
               if payload_text is not None else None)
    if payload is None:
        raise StateDBSchemaMismatchError(
            "State DB schema metadata row is missing")

    found_hash = payload.get("schema_hash")
    if found_hash != expected_hash and found_hash not in compatible_hashes:
        raise StateDBSchemaMismatchError(
            _schema_mismatch_message(found_hash, expected_hash))


def _fetch_sqlite_tables(connection: sqlite3.Connection) -> set[str]:
    cursor = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'")
    try:
        return {
            row[0]
            for row in cursor.fetchall()
            if not str(row[0]).startswith("sqlite_")
        }
    finally:
        cursor.close()


def _fetch_sqlite_indexes(
    connection: sqlite3.Connection, ) -> dict[str, _ExistingIndex]:
    cursor = connection.execute("""
        SELECT name, tbl_name, sql
        FROM sqlite_master
        WHERE type = 'index' AND sql IS NOT NULL
        """)
    try:
        return {
            row[0]:
            _ExistingIndex(
                name=row[0],
                table_name=row[1],
                definition=row[2],
            )
            for row in cursor.fetchall()
        }
    finally:
        cursor.close()


def _fetch_sqlite_schema_payload(
    connection: sqlite3.Connection,
    existing_tables: set[str],
) -> str | None:
    if "statedb_schema" not in existing_tables:
        return None
    try:
        cursor = connection.execute(
            "SELECT payload FROM statedb_schema WHERE id = ?",
            (SCHEMA_ROW_ID, ),
        )
    except sqlite3.DatabaseError as exc:
        raise StateDBSchemaMismatchError(
            "State DB schema metadata cannot be read") from exc
    try:
        row = cursor.fetchone()
    finally:
        cursor.close()
    return row[0] if row is not None else None


def _sqlite_bootstrap_lock(sqlite_path: str) -> threading.Lock:
    global _SQLITE_BOOTSTRAP_LOCKS_PID
    global _SQLITE_BOOTSTRAP_LOCKS
    canonical_path = os.path.realpath(sqlite_path)
    with _SQLITE_BOOTSTRAP_LOCKS_GUARD:
        # Defensive fallback for platforms or embedders that do not run the
        # registered hook.  The hook is still required to replace a guard that
        # was locked by a vanished parent thread.
        pid = os.getpid()
        if pid != _SQLITE_BOOTSTRAP_LOCKS_PID:
            _SQLITE_BOOTSTRAP_LOCKS = {}
            _SQLITE_BOOTSTRAP_LOCKS_PID = pid
        return _SQLITE_BOOTSTRAP_LOCKS.setdefault(canonical_path,
                                                  threading.Lock())


async def _acquire_sqlite_bootstrap_lock(lock: threading.Lock) -> None:
    """Acquire a process lock without binding it to one event loop/thread."""
    while not lock.acquire(blocking=False):
        await asyncio.sleep(0.001)


def _sqlite_is_busy(error: sqlite3.Error) -> bool:
    code = getattr(error, 'sqlite_errorcode', None)
    if code in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
        return True
    message = str(error).casefold()
    return 'locked' in message or 'busy' in message


def _sqlite_bootstrap_result(
    *,
    tables: tuple[TableSpec, ...],
    expected_hash: str,
    schema_version: int,
    created_schema_row: bool,
) -> BootstrapResult:
    return BootstrapResult(
        backend='sqlite',
        schema_hash=expected_hash,
        table_count=len(tables),
        index_count=sum(len(table.indexes) for table in tables),
        created_schema_row=created_schema_row,
        schema_version=schema_version,
    )


def _bootstrap_sqlite_read_only_fast_path(
    sqlite_path: str,
    *,
    tables: tuple[TableSpec, ...],
    expected_hash: str,
    schema_version: int,
) -> BootstrapResult | None:
    """Validate a current schema without requesting any SQLite write lock."""
    if not os.path.isfile(sqlite_path):
        return None
    uri = f'{Path(sqlite_path).resolve().as_uri()}?mode=ro'
    try:
        connection = sqlite3.connect(
            uri,
            uri=True,
            isolation_level=None,
            timeout=0,
        )
    except sqlite3.OperationalError as error:
        if _sqlite_is_busy(error) or not os.path.isfile(sqlite_path):
            return None
        raise
    try:
        existing_tables = _fetch_sqlite_tables(connection)
        existing_indexes = _fetch_sqlite_indexes(connection)
        if not _schema_exists(existing_tables, existing_indexes):
            return None
        payload_text = _fetch_sqlite_schema_payload(connection, existing_tables)
        try:
            _validate_existing_schema(
                backend='sqlite',
                tables=tables,
                expected_hash=expected_hash,
                existing_tables=existing_tables,
                existing_indexes=existing_indexes,
                payload_text=payload_text,
            )
        except StateDBSchemaMismatchError:
            # Recheck under the serialized writer lease. Another process may
            # currently be committing the first complete schema.
            return None
        if (_schema_version_from_payload(payload_text) != schema_version
                or _schema_hash_from_payload(payload_text) != expected_hash):
            return None
        return _sqlite_bootstrap_result(
            tables=tables,
            expected_hash=expected_hash,
            schema_version=schema_version,
            created_schema_row=False,
        )
    except sqlite3.OperationalError as error:
        if _sqlite_is_busy(error):
            return None
        raise
    finally:
        connection.close()


def _bootstrap_sqlite_write_once(
    sqlite_path: str,
    *,
    tables: tuple[TableSpec, ...],
    expected_hash: str,
    schema_version: int,
    payload_json: str,
) -> BootstrapResult:
    connection = sqlite3.connect(
        sqlite_path,
        isolation_level=None,
        timeout=_SQLITE_BOOTSTRAP_BUSY_TIMEOUT_MS / 1000,
    )
    try:
        connection.execute(
            f'PRAGMA busy_timeout={_SQLITE_BOOTSTRAP_BUSY_TIMEOUT_MS}')
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('BEGIN IMMEDIATE')
        try:
            existing_tables = _fetch_sqlite_tables(connection)
            existing_indexes = _fetch_sqlite_indexes(connection)
            if _schema_exists(existing_tables, existing_indexes):
                payload_text = _fetch_sqlite_schema_payload(
                    connection, existing_tables)
                _validate_existing_schema(
                    backend='sqlite',
                    tables=tables,
                    expected_hash=expected_hash,
                    existing_tables=existing_tables,
                    existing_indexes=existing_indexes,
                    payload_text=payload_text,
                    compatible_hashes=(legacy_schema_identity(tables), ),
                )
                payload_version = _schema_version_from_payload(payload_text)
                payload_hash = _schema_hash_from_payload(payload_text)
                if (payload_version != schema_version
                        or payload_hash != expected_hash):
                    now = _now_iso()
                    connection.execute(
                        """
                        UPDATE statedb_schema
                        SET payload = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (payload_json, now, SCHEMA_ROW_ID),
                    )
                created_schema_row = False
            else:
                for table in tables:
                    connection.execute(_sqlite_table_ddl(table.name))
                for table in tables:
                    for index in table.indexes:
                        connection.execute(index.ddl(table.name, 'sqlite'))
                now = _now_iso()
                connection.execute(
                    """
                    INSERT INTO statedb_schema
                        (id, payload, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (SCHEMA_ROW_ID, payload_json, now, now),
                )
                created_schema_row = True
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
    finally:
        connection.close()
    return _sqlite_bootstrap_result(
        tables=tables,
        expected_hash=expected_hash,
        schema_version=schema_version,
        created_schema_row=created_schema_row,
    )


async def _fetch_postgres_tables(connection: asyncpg.Connection, ) -> set[str]:
    rows = await connection.fetch("""
        SELECT tablename
        FROM pg_catalog.pg_tables
        WHERE schemaname = current_schema()
        """)
    return {row["tablename"] for row in rows}


async def _fetch_postgres_indexes(
    connection: asyncpg.Connection, ) -> dict[str, _ExistingIndex]:
    rows = await connection.fetch("""
        SELECT
            index_class.relname AS indexname,
            table_class.relname AS tablename,
            pg_catalog.pg_get_indexdef(index_class.oid) AS indexdef
        FROM pg_catalog.pg_index index_info
        JOIN pg_catalog.pg_class index_class
            ON index_class.oid = index_info.indexrelid
        JOIN pg_catalog.pg_class table_class
            ON table_class.oid = index_info.indrelid
        JOIN pg_catalog.pg_namespace table_namespace
            ON table_namespace.oid = table_class.relnamespace
        WHERE table_namespace.nspname = current_schema()
            AND NOT index_info.indisprimary
        """)
    return {
        row["indexname"]:
        _ExistingIndex(
            name=row["indexname"],
            table_name=row["tablename"],
            definition=row["indexdef"],
        )
        for row in rows
    }


async def _fetch_postgres_schema_payload(
    connection: asyncpg.Connection,
    existing_tables: set[str],
) -> str | None:
    if "statedb_schema" not in existing_tables:
        return None
    try:
        row = await connection.fetchrow(
            """
            SELECT payload::text AS payload
            FROM statedb_schema
            WHERE id = $1
            """,
            SCHEMA_ROW_ID,
        )
    except asyncpg.PostgresError as exc:
        raise StateDBSchemaMismatchError(
            "State DB schema metadata cannot be read") from exc
    return row["payload"] if row is not None else None


async def bootstrap_sqlite(
    sqlite_path: str,
    *,
    tables: tuple[TableSpec, ...] = STATE_DB_TABLES,
    schema_version: int = SCHEMA_VERSION,
) -> BootstrapResult:
    os.makedirs(os.path.dirname(os.path.abspath(sqlite_path)), exist_ok=True)
    expected_hash = schema_identity('sqlite', tables)
    payload_json = json.dumps(
        _schema_payload(expected_hash, schema_version),
        sort_keys=True,
    )

    fast_result = _bootstrap_sqlite_read_only_fast_path(
        sqlite_path,
        tables=tables,
        expected_hash=expected_hash,
        schema_version=schema_version,
    )
    if fast_result is not None:
        return fast_result

    process_lock = _sqlite_bootstrap_lock(sqlite_path)
    await _acquire_sqlite_bootstrap_lock(process_lock)
    try:
        # A concurrent in-process opener may have completed while this caller
        # waited for the per-path lease.
        fast_result = _bootstrap_sqlite_read_only_fast_path(
            sqlite_path,
            tables=tables,
            expected_hash=expected_hash,
            schema_version=schema_version,
        )
        if fast_result is not None:
            return fast_result

        backoff = _SQLITE_BOOTSTRAP_INITIAL_BACKOFF_SECONDS
        for attempt in range(_SQLITE_BOOTSTRAP_MAX_ATTEMPTS):
            try:
                return _bootstrap_sqlite_write_once(
                    sqlite_path,
                    tables=tables,
                    expected_hash=expected_hash,
                    schema_version=schema_version,
                    payload_json=payload_json,
                )
            except sqlite3.OperationalError as error:
                if not _sqlite_is_busy(error):
                    raise
                if attempt + 1 >= _SQLITE_BOOTSTRAP_MAX_ATTEMPTS:
                    raise StateDBBootstrapBusyError(
                        'State DB schema initialization is temporarily busy; '
                        'please retry.') from None
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2,
                              _SQLITE_BOOTSTRAP_MAX_BACKOFF_SECONDS)
    finally:
        process_lock.release()


async def bootstrap_postgres(
    dsn: str,
    *,
    password: Callable[[], Awaitable[str]] | None = None,
    tables: tuple[TableSpec, ...] = STATE_DB_TABLES,
    schema_version: int = SCHEMA_VERSION,
    migrations: Mapping[int, PostgresSchemaMigration] | None = None,
) -> BootstrapResult:
    expected_hash = schema_identity('postgres', tables)
    payload_json = json.dumps(
        _schema_payload(expected_hash, schema_version),
        sort_keys=True,
    )
    applied_migrations: tuple[int, ...] = ()
    connect_options = {"password": password} if password is not None else {}
    connection = await asyncpg.connect(dsn, **connect_options)
    try:
        async with connection.transaction():
            await connection.execute(
                "SELECT pg_advisory_xact_lock($1)",
                _POSTGRES_BOOTSTRAP_ADVISORY_LOCK_ID,
            )
            existing_tables = await _fetch_postgres_tables(connection)
            existing_indexes = await _fetch_postgres_indexes(connection)
            if _schema_exists(existing_tables, existing_indexes):
                payload_text = await _fetch_postgres_schema_payload(
                    connection, existing_tables)
                try:
                    _validate_existing_schema(
                        backend="postgres",
                        tables=tables,
                        expected_hash=expected_hash,
                        existing_tables=existing_tables,
                        existing_indexes=existing_indexes,
                        payload_text=payload_text,
                        compatible_hashes=(legacy_schema_identity(tables), ),
                    )
                except StateDBSchemaMismatchError as exc:
                    found_version = _schema_version_from_payload(payload_text)
                    if found_version == schema_version:
                        raise
                    if found_version is None or found_version > schema_version:
                        raise StateDBSchemaMismatchError(
                            "No StateDB migration path from schema version "
                            f"{found_version!r} to {schema_version!r}") from exc
                    migration_steps = []
                    next_version = found_version
                    while next_version < schema_version:
                        migration = (migrations or {}).get(next_version)
                        if migration is None:
                            raise StateDBSchemaMismatchError(
                                "No StateDB migration path from schema version "
                                f"{found_version!r} to {schema_version!r}; "
                                f"missing migration from {next_version!r} "
                                f"to {next_version + 1!r}") from exc
                        migration_steps.append((next_version, migration))
                        next_version += 1
                    for next_version, migration in migration_steps:
                        try:
                            await migration(connection)
                        except Exception:
                            log.exception(
                                "StateDB Postgres schema migration %s -> %s "
                                "failed; rolling back to schema version %s",
                                next_version,
                                next_version + 1,
                                found_version,
                            )
                            raise
                    now = datetime.now(timezone.utc)
                    await connection.execute(
                        """
                        UPDATE statedb_schema
                        SET payload = $1::jsonb, updated_at = $2
                        WHERE id = $3
                        """,
                        payload_json,
                        now,
                        SCHEMA_ROW_ID,
                    )
                    existing_tables = await _fetch_postgres_tables(connection)
                    existing_indexes = await _fetch_postgres_indexes(connection)
                    payload_text = await _fetch_postgres_schema_payload(
                        connection, existing_tables)
                    _validate_existing_schema(
                        backend="postgres",
                        tables=tables,
                        expected_hash=expected_hash,
                        existing_tables=existing_tables,
                        existing_indexes=existing_indexes,
                        payload_text=payload_text,
                    )
                    applied_migrations = tuple(
                        version for version, _ in migration_steps)
                else:
                    found_version = _schema_version_from_payload(payload_text)
                    found_hash = _schema_hash_from_payload(payload_text)
                    if (found_version != schema_version
                            or found_hash != expected_hash):
                        now = datetime.now(timezone.utc)
                        await connection.execute(
                            """
                            UPDATE statedb_schema
                            SET payload = $1::jsonb, updated_at = $2
                            WHERE id = $3
                            """,
                            payload_json,
                            now,
                            SCHEMA_ROW_ID,
                        )
                created_schema_row = False
            else:
                for table in tables:
                    await connection.execute(_postgres_table_ddl(table.name))
                for table in tables:
                    for index in table.indexes:
                        await connection.execute(
                            index.ddl(table.name, "postgres"))
                now = datetime.now(timezone.utc)
                await connection.execute(
                    """
                    INSERT INTO statedb_schema
                        (id, payload, created_at, updated_at)
                    VALUES ($1, $2::jsonb, $3, $4)
                    """,
                    SCHEMA_ROW_ID,
                    payload_json,
                    now,
                    now,
                )
                created_schema_row = True
    finally:
        await connection.close()

    return BootstrapResult(
        backend="postgres",
        schema_hash=expected_hash,
        table_count=len(tables),
        index_count=sum(len(table.indexes) for table in tables),
        created_schema_row=created_schema_row,
        applied_migrations=applied_migrations,
        schema_version=schema_version,
    )
