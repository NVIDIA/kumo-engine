# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import multiprocessing as mp
import queue
import sqlite3
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import asyncpg
import pytest

from statedb.db import (
    SCHEMA_ROW_ID,
    SCHEMA_VERSION,
    STATE_DB_TABLE_NAMES,
    STATE_DB_TABLES,
    StateDB,
    StateDBBootstrapBusyError,
    StateDBSchemaMismatchError,
    TableSpec,
)
from statedb.db import bootstrap as bootstrap_mod
from statedb.db import bootstrap_postgres, bootstrap_sqlite, schema_identity
from statedb.db.bootstrap import (
    _ExistingIndex,
    _index_matches,
    _postgres_table_ddl,
    _sqlite_table_ddl,
)
from statedb.db.migrations import POSTGRES_SCHEMA_MIGRATIONS
from statedb.db.schema import IndexSpec, legacy_schema_identity


@pytest.fixture
def setup_rest_test_session() -> None:
    """Schema bootstrap tests do not require the shared Temporal harness."""
    return None


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


def _execute_sqlite(sqlite_path, sql: str, *args) -> None:
    connection = sqlite3.connect(sqlite_path)
    try:
        connection.execute(sql, args)
        connection.commit()
    finally:
        connection.close()


def _fetch_sqlite(sqlite_path, sql: str, *args):
    connection = sqlite3.connect(sqlite_path)
    try:
        cursor = connection.execute(sql, args)
        try:
            return cursor.fetchall()
        finally:
            cursor.close()
    finally:
        connection.close()


async def _reset_postgres_schema(connection) -> None:
    await connection.execute("DROP SCHEMA public CASCADE")
    await connection.execute("CREATE SCHEMA public")


async def _fetch_postgres_table_names(connection) -> set[str]:
    rows = await connection.fetch("""
        SELECT tablename
        FROM pg_catalog.pg_tables
        WHERE schemaname = current_schema()
        """)
    return {row["tablename"] for row in rows}


async def _fetch_postgres_schema_metadata(connection) -> dict:
    row = await connection.fetchrow(
        "SELECT payload FROM statedb_schema WHERE id = $1",
        SCHEMA_ROW_ID,
    )
    assert row is not None
    return json.loads(row["payload"])


def _schema_metadata_payload(schema_hash: str) -> str:
    return json.dumps(
        {
            "schema_hash":
            schema_hash,
            "schema_version":
            SCHEMA_VERSION,
            "table_names": [table.name for table in STATE_DB_TABLES],
            "index_names": [
                index.name for table in STATE_DB_TABLES
                for index in table.indexes
            ],
        },
        sort_keys=True,
    )


class _FakePostgresTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePostgresConnection:
    def __init__(self):
        self.execute_calls = []
        self.closed = False

    def transaction(self):
        return _FakePostgresTransaction()

    async def execute(self, sql: str, *args):
        self.execute_calls.append((sql, args))

    async def fetch(self, sql: str, *args):
        return []

    async def fetchrow(self, sql: str, *args):
        return None

    async def close(self):
        self.closed = True


class _MigratingFakePostgresConnection(_FakePostgresConnection):
    def __init__(
        self,
        *,
        table_names: set[str],
        schema_hash: str,
        schema_version: int,
    ):
        super().__init__()
        self.table_names = table_names
        self.schema_hash = schema_hash
        self.schema_version = schema_version

    async def execute(self, sql: str, *args):
        await super().execute(sql, *args)
        if "UPDATE statedb_schema" in sql:
            payload = json.loads(args[0])
            self.schema_hash = payload["schema_hash"]
            self.schema_version = payload["schema_version"]

    async def fetch(self, sql: str, *args):
        if "FROM pg_catalog.pg_tables" in sql:
            return [{"tablename": name} for name in sorted(self.table_names)]
        if "FROM pg_catalog.pg_index" in sql:
            return []
        return []

    async def fetchrow(self, sql: str, *args):
        if "FROM statedb_schema" in sql and "statedb_schema" in self.table_names:
            return {
                "payload":
                json.dumps({
                    "schema_hash": self.schema_hash,
                    "schema_version": self.schema_version,
                })
            }
        return None


def test_state_db_schema_inventory_includes_rest_and_ml_tables() -> None:
    table_names = set(STATE_DB_TABLE_NAMES)

    assert "connectors" in table_names
    assert "tables" in table_names
    assert "graphs" in table_names
    assert "predictive_queries" in table_names
    assert "api_keys" in table_names
    assert "api_key_secrets" in table_names
    assert "training_jobs" in table_names
    assert "batch_prediction_jobs" in table_names
    assert "baseline_jobs" not in table_names
    assert "online_serving_jobs" not in table_names
    assert "job_tags" in table_names
    assert "job_resources" not in table_names
    assert "direct_connect_users" in table_names
    assert "direct_connect_password_login_attempts" in table_names


def test_state_db_schema_identity_is_stable() -> None:
    assert schema_identity("sqlite") == schema_identity("sqlite")
    assert schema_identity("postgres") == schema_identity("postgres")
    assert len(schema_identity("sqlite")) == 64


def test_state_db_schema_identity_ignores_table_purpose() -> None:
    first = (TableSpec("records", "first description"), )
    second = (TableSpec("records", "updated description"), )

    assert schema_identity("sqlite", first) == schema_identity("sqlite", second)
    assert schema_identity("postgres",
                           first) == schema_identity("postgres", second)


def test_state_db_schema_identity_only_uses_active_backend_ddl() -> None:
    original = IndexSpec(
        name="records_value_idx",
        postgres_expression="(payload->>'value')",
        sqlite_expression="json_extract(payload, '$.value')",
    )
    changed_sqlite = IndexSpec(
        name=original.name,
        postgres_expression=original.postgres_expression,
        sqlite_expression="json_extract(payload, '$.other_value')",
    )
    original_tables = (TableSpec("records", "records", (original, )), )
    changed_tables = (TableSpec("records", "records", (changed_sqlite, )), )

    assert schema_identity("postgres", original_tables) == schema_identity(
        "postgres", changed_tables)
    assert schema_identity("sqlite", original_tables) != schema_identity(
        "sqlite", changed_tables)


def test_project_group_and_identity_names_and_slugs_have_global_unique_indexes(
) -> None:
    projects = next(table for table in STATE_DB_TABLES
                    if table.name == "projects")
    project_name_index = next(
        index for index in projects.indexes
        if index.name == "projects_display_name_unique_idx")
    groups = next(table for table in STATE_DB_TABLES if table.name == "groups")
    group_name_index = next(index for index in groups.indexes
                            if index.name == "groups_display_name_unique_idx")
    identities = next(table for table in STATE_DB_TABLES
                      if table.name == "identities")

    slug_indexes = {
        table.name:
        next(index for index in table.indexes
             if index.name == f"{table.name}_slug_unique_idx")
        for table in (projects, groups, identities)
    }

    assert SCHEMA_VERSION == 17
    assert project_name_index.unique is True
    assert project_name_index.postgres_expression == (
        "(payload->>'normalized_display_name')")
    assert project_name_index.sqlite_expression == (
        "json_extract(payload, '$.normalized_display_name')")
    assert "group_id" not in project_name_index.postgres_expression
    assert "group_id" not in project_name_index.sqlite_expression
    assert group_name_index.unique is True
    for slug_index in slug_indexes.values():
        assert slug_index.unique is True
        assert slug_index.postgres_expression == "(payload->>'slug')"
        assert slug_index.sqlite_expression == (
            "json_extract(payload, '$.slug')")


def test_identity_unique_indexes_only_cover_live_identities() -> None:
    identities = next(table for table in STATE_DB_TABLES
                      if table.name == "identities")
    unique_indexes = [index for index in identities.indexes if index.unique]
    assert len(unique_indexes) == 3
    for index in unique_indexes:
        assert index.postgres_where == "payload->>'tombstoned_at' IS NULL"
        assert index.sqlite_where == (
            "json_extract(payload, '$.tombstoned_at') IS NULL")


def test_connector_indexes_only_cover_live_name_and_identity_lookups() -> None:
    connectors = next(table for table in STATE_DB_TABLES
                      if table.name == "connectors")
    name_index = next(index for index in connectors.indexes
                      if index.name == "connectors_project_name_unique_idx")
    identity_index = next(index for index in connectors.indexes
                          if index.name == "connectors_live_identity_idx")

    assert name_index.unique is True
    assert name_index.postgres_where == ("payload->>'tombstoned_at' IS NULL")
    assert name_index.sqlite_where == (
        "json_extract(payload, '$.tombstoned_at') IS NULL")
    assert "identity_id" in identity_index.postgres_expression
    assert "tombstoned_at" in identity_index.postgres_where
    assert "identity_id" in identity_index.sqlite_expression
    assert "tombstoned_at" in identity_index.sqlite_where


def test_state_db_schema_ddl_is_not_masked() -> None:
    table = next(table for table in STATE_DB_TABLES if table.indexes)
    index = table.indexes[0]

    assert "IF NOT EXISTS" not in _sqlite_table_ddl(table.name).upper()
    assert "IF NOT EXISTS" not in _postgres_table_ddl(table.name).upper()
    assert "IF NOT EXISTS" not in index.ddl(table.name, "sqlite").upper()
    assert "IF NOT EXISTS" not in index.ddl(table.name, "postgres").upper()


def test_postgres_index_match_accepts_pg_catalog_where_parentheses() -> None:
    table = next(table for table in STATE_DB_TABLES
                 if table.name == "batch_prediction_jobs")
    index = next(index for index in table.indexes
                 if index.name == "batch_prediction_jobs_workflow_id_idx")
    existing = _ExistingIndex(
        name=index.name,
        table_name=table.name,
        definition=("CREATE INDEX batch_prediction_jobs_workflow_id_idx "
                    "ON public.batch_prediction_jobs USING btree "
                    "(((payload ->> 'batch_prediction_workflow_id'::text))) "
                    "WHERE ((payload ->> 'batch_prediction_workflow_id'::text) "
                    "IS NOT NULL)"),
    )

    assert _index_matches(existing, table.name, index, "postgres")


def test_postgres_index_match_accepts_nested_partial_predicate_parentheses(
) -> None:
    table = next(table for table in STATE_DB_TABLES if table.name == "tables")
    index = next(index for index in table.indexes
                 if index.name == "tables_project_alias_unique_idx")
    existing = _ExistingIndex(
        name=index.name,
        table_name=table.name,
        definition=("CREATE UNIQUE INDEX tables_project_alias_unique_idx "
                    "ON public.tables USING btree "
                    "(((payload ->> 'project_id'::text)), "
                    "((payload ->> 'normalized_alias'::text))) "
                    "WHERE (COALESCE((payload ->> 'normalized_alias'::text), "
                    "''::text) <> ''::text)"),
    )

    assert _index_matches(existing, table.name, index, "postgres")


def test_sqlite_index_match_accepts_equivalent_workflow_index_ddl() -> None:
    table = next(table for table in STATE_DB_TABLES
                 if table.name == "batch_prediction_jobs")
    index = next(index for index in table.indexes
                 if index.name == "batch_prediction_jobs_workflow_id_idx")
    existing = _ExistingIndex(
        name=index.name,
        table_name=table.name,
        definition=(
            'CREATE INDEX "batch_prediction_jobs_workflow_id_idx" '
            'ON "batch_prediction_jobs" '
            "((json_extract(payload, '$.batch_prediction_workflow_id'))) "
            "WHERE (json_extract(payload, '$.batch_prediction_workflow_id') "
            "is not null)"),
    )

    assert _index_matches(existing, table.name, index, "sqlite")


@pytest.mark.asyncio
async def test_bootstrap_sqlite_creates_schema_for_empty_target(
    tmp_path, ) -> None:
    sqlite_path = tmp_path / "state.db"

    result = await bootstrap_sqlite(str(sqlite_path))

    assert result.backend == "sqlite"
    assert result.schema_hash == schema_identity("sqlite")
    assert result.table_count == len(STATE_DB_TABLE_NAMES)
    assert result.created_schema_row is True

    table_names = {
        row[0]
        for row in _fetch_sqlite(
            sqlite_path, "SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    rows = _fetch_sqlite(
        sqlite_path,
        "SELECT payload FROM statedb_schema WHERE id = ?",
        SCHEMA_ROW_ID,
    )
    row = rows[0] if rows else None

    assert table_names == set(STATE_DB_TABLE_NAMES)
    assert row is not None
    payload = json.loads(row[0])
    assert payload["schema_hash"] == schema_identity("sqlite")
    assert payload["schema_version"] == SCHEMA_VERSION

    second_result = await bootstrap_sqlite(str(sqlite_path))
    assert second_result.created_schema_row is False


@pytest.mark.asyncio
async def test_bootstrap_sqlite_upgrades_legacy_schema_identity(
    tmp_path, ) -> None:
    sqlite_path = tmp_path / "state.db"
    await bootstrap_sqlite(str(sqlite_path))
    legacy_hash = legacy_schema_identity()
    _execute_sqlite(
        sqlite_path,
        "UPDATE statedb_schema SET payload = ? WHERE id = ?",
        json.dumps({
            "schema_hash": legacy_hash,
            "schema_version": SCHEMA_VERSION,
        }),
        SCHEMA_ROW_ID,
    )

    result = await bootstrap_sqlite(str(sqlite_path))

    payload = json.loads(
        _fetch_sqlite(
            sqlite_path,
            "SELECT payload FROM statedb_schema WHERE id = ?",
            SCHEMA_ROW_ID,
        )[0][0])
    assert result.created_schema_row is False
    assert result.schema_hash == schema_identity("sqlite")
    assert payload == {
        "schema_hash": schema_identity("sqlite"),
        "schema_version": SCHEMA_VERSION,
    }


@pytest.mark.asyncio
async def test_bootstrap_sqlite_current_schema_uses_read_only_fast_path(
    tmp_path, ) -> None:
    sqlite_path = tmp_path / 'state.db'
    await bootstrap_sqlite(str(sqlite_path))
    writer = sqlite3.connect(sqlite_path, isolation_level=None)
    try:
        writer.execute('BEGIN IMMEDIATE')
        started = time.monotonic()
        result = await bootstrap_sqlite(str(sqlite_path))
        elapsed = time.monotonic() - started
    finally:
        writer.rollback()
        writer.close()

    assert result.created_schema_row is False
    assert elapsed < 0.1


@pytest.mark.asyncio
async def test_bootstrap_sqlite_retries_busy_writer_until_release(
        tmp_path, monkeypatch) -> None:
    sqlite_path = tmp_path / 'state.db'
    writer = sqlite3.connect(
        sqlite_path,
        isolation_level=None,
        check_same_thread=False,
    )
    writer.execute('PRAGMA journal_mode=WAL')
    writer.execute('BEGIN IMMEDIATE')
    attempts = 0
    original = bootstrap_mod._bootstrap_sqlite_write_once

    def counted_write(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(bootstrap_mod, '_bootstrap_sqlite_write_once',
                        counted_write)
    monkeypatch.setattr(bootstrap_mod, '_SQLITE_BOOTSTRAP_BUSY_TIMEOUT_MS', 5)
    monkeypatch.setattr(bootstrap_mod, '_SQLITE_BOOTSTRAP_MAX_ATTEMPTS', 20)
    monkeypatch.setattr(bootstrap_mod,
                        '_SQLITE_BOOTSTRAP_INITIAL_BACKOFF_SECONDS', 0.005)
    monkeypatch.setattr(bootstrap_mod, '_SQLITE_BOOTSTRAP_MAX_BACKOFF_SECONDS',
                        0.02)

    def release_writer() -> None:
        time.sleep(0.04)
        writer.rollback()
        writer.close()

    release = threading.Thread(target=release_writer)
    release.start()
    try:
        result = await bootstrap_sqlite(str(sqlite_path))
    finally:
        release.join(timeout=1)

    assert result.created_schema_row is True
    assert attempts > 1


@pytest.mark.asyncio
async def test_bootstrap_sqlite_busy_exhaustion_is_bounded_and_sanitized(
        tmp_path, monkeypatch) -> None:
    sqlite_path = tmp_path / 'state.db'
    writer = sqlite3.connect(sqlite_path, isolation_level=None)
    writer.execute('PRAGMA journal_mode=WAL')
    writer.execute('BEGIN IMMEDIATE')
    monkeypatch.setattr(bootstrap_mod, '_SQLITE_BOOTSTRAP_BUSY_TIMEOUT_MS', 1)
    monkeypatch.setattr(bootstrap_mod, '_SQLITE_BOOTSTRAP_MAX_ATTEMPTS', 2)
    monkeypatch.setattr(bootstrap_mod,
                        '_SQLITE_BOOTSTRAP_INITIAL_BACKOFF_SECONDS', 0)
    try:
        started = time.monotonic()
        with pytest.raises(StateDBBootstrapBusyError) as caught:
            await bootstrap_sqlite(str(sqlite_path))
        elapsed = time.monotonic() - started
    finally:
        writer.rollback()
        writer.close()

    assert elapsed < 0.2
    assert 'temporarily busy' in str(caught.value)
    assert 'locked' not in str(caught.value).casefold()


def test_simultaneous_fresh_sqlite_services_bootstrap_once(tmp_path) -> None:
    sqlite_path = str(tmp_path / 'state.db')
    barrier = threading.Barrier(8)
    created: list[bool] = []
    errors: list[BaseException] = []

    def open_and_close() -> bool:
        async def run() -> bool:
            state_db = await StateDB.open_sqlite(sqlite_path, pool_size=1)
            try:
                return state_db.bootstrap_result.created_schema_row
            finally:
                await state_db.close()

        return asyncio.run(run())

    def run() -> None:
        barrier.wait()
        try:
            created.append(open_and_close())
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive(), 'concurrent SQLite bootstrap thread hung'

    assert errors == []
    assert created.count(True) == 1
    assert created.count(False) == 7


def test_simultaneous_fresh_sqlite_process_bootstraps(tmp_path) -> None:
    if 'fork' not in mp.get_all_start_methods():
        pytest.skip('fork start method is required')
    sqlite_path = str(tmp_path / 'state.db')
    context = mp.get_context('fork')
    barrier = context.Barrier(4)
    results: mp.Queue = context.Queue()

    def run() -> None:
        barrier.wait()
        try:
            result = asyncio.run(bootstrap_sqlite(sqlite_path))
        except BaseException as error:
            results.put(('error', type(error).__name__, str(error)))
        else:
            results.put(('ok', result.created_schema_row, ''))

    processes = [context.Process(target=run) for _ in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail('concurrent SQLite bootstrap process hung')

    reported = []
    for _ in processes:
        try:
            reported.append(results.get(timeout=1))
        except queue.Empty:
            pytest.fail('concurrent SQLite bootstrap did not report a result')

    assert all(process.exitcode == 0 for process in processes)
    assert all(result[0] == 'ok' for result in reported), reported
    assert sum(bool(result[1]) for result in reported) == 1


def test_sqlite_bootstrap_fork_replaces_locked_parent_guards(tmp_path) -> None:
    if 'fork' not in mp.get_all_start_methods():
        pytest.skip('fork start method is required')
    sqlite_path = str(tmp_path / 'forked-state.db')
    parent_path_lock = bootstrap_mod._sqlite_bootstrap_lock(sqlite_path)
    parent_path_lock.acquire()
    parent_guard = bootstrap_mod._SQLITE_BOOTSTRAP_LOCKS_GUARD
    parent_guard.acquire()
    context = mp.get_context('fork')
    receive, send = context.Pipe(duplex=False)

    def bootstrap_in_child() -> None:
        try:
            result = asyncio.run(bootstrap_sqlite(sqlite_path))
            send.send(('ok', result.created_schema_row,
                       bootstrap_mod._SQLITE_BOOTSTRAP_LOCKS_PID))
        except BaseException as error:
            send.send(('error', type(error).__name__, str(error)))
        finally:
            send.close()

    process = context.Process(target=bootstrap_in_child)
    try:
        process.start()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail('forked SQLite bootstrap inherited a locked guard')
    finally:
        parent_guard.release()
        parent_path_lock.release()
        send.close()

    assert process.exitcode == 0
    assert receive.poll(1)
    status, created, child_pid = receive.recv()
    receive.close()
    assert status == 'ok'
    assert created is True
    assert child_pid == process.pid


@pytest.mark.asyncio
async def test_bootstrap_sqlite_rejects_unexpected_existing_table_before_repair(
    tmp_path, ) -> None:
    sqlite_path = tmp_path / "state.db"
    _execute_sqlite(sqlite_path,
                    "CREATE TABLE old_rest_metadata (id TEXT PRIMARY KEY)")

    with pytest.raises(StateDBSchemaMismatchError, match="unexpected tables"):
        await bootstrap_sqlite(str(sqlite_path))

    table_names = {
        row[0]
        for row in _fetch_sqlite(
            sqlite_path, "SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert table_names == {"old_rest_metadata"}


@pytest.mark.asyncio
async def test_bootstrap_sqlite_rejects_partial_state_db_footprint_before_repair(
    tmp_path, ) -> None:
    sqlite_path = tmp_path / "state.db"
    _execute_sqlite(sqlite_path, "CREATE TABLE projects (id TEXT PRIMARY KEY)")

    with pytest.raises(StateDBSchemaMismatchError,
                       match="missing State DB tables"):
        await bootstrap_sqlite(str(sqlite_path))

    table_names = {
        row[0]
        for row in _fetch_sqlite(
            sqlite_path, "SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert table_names == {"projects"}


@pytest.mark.asyncio
async def test_bootstrap_sqlite_rejects_missing_index_before_repair(
    tmp_path, ) -> None:
    sqlite_path = tmp_path / "state.db"
    await bootstrap_sqlite(str(sqlite_path))
    _execute_sqlite(sqlite_path, "DROP INDEX projects_status_group_private_idx")

    with pytest.raises(StateDBSchemaMismatchError,
                       match="missing State DB indexes"):
        await bootstrap_sqlite(str(sqlite_path))

    index_names = {
        row[0]
        for row in _fetch_sqlite(
            sqlite_path, "SELECT name FROM sqlite_master WHERE type = 'index'")
    }
    assert "projects_status_group_private_idx" not in index_names


@pytest.mark.asyncio
async def test_bootstrap_sqlite_rejects_wrong_index(tmp_path) -> None:
    sqlite_path = tmp_path / "state.db"
    await bootstrap_sqlite(str(sqlite_path))
    _execute_sqlite(sqlite_path, "DROP INDEX projects_status_group_private_idx")
    _execute_sqlite(
        sqlite_path,
        "CREATE INDEX projects_status_group_private_idx ON projects (id)")

    with pytest.raises(StateDBSchemaMismatchError,
                       match="wrong State DB indexes"):
        await bootstrap_sqlite(str(sqlite_path))


@pytest.mark.asyncio
async def test_bootstrap_sqlite_rejects_extra_index(tmp_path) -> None:
    sqlite_path = tmp_path / "state.db"
    await bootstrap_sqlite(str(sqlite_path))
    _execute_sqlite(sqlite_path,
                    "CREATE INDEX projects_extra_idx ON projects (id)")

    with pytest.raises(StateDBSchemaMismatchError,
                       match="unexpected State DB indexes"):
        await bootstrap_sqlite(str(sqlite_path))


@pytest.mark.asyncio
async def test_bootstrap_sqlite_accepts_equivalent_workflow_index_ddl(
    tmp_path, ) -> None:
    sqlite_path = tmp_path / "state.db"
    await bootstrap_sqlite(str(sqlite_path))
    _execute_sqlite(sqlite_path,
                    "DROP INDEX batch_prediction_jobs_workflow_id_idx")
    _execute_sqlite(
        sqlite_path,
        """
        CREATE INDEX "batch_prediction_jobs_workflow_id_idx"
        ON "batch_prediction_jobs"
            ((json_extract(payload, '$.batch_prediction_workflow_id')))
        WHERE (
            json_extract(payload, '$.batch_prediction_workflow_id') is not null
        )
        """,
    )

    result = await bootstrap_sqlite(str(sqlite_path))

    assert result.created_schema_row is False


@pytest.mark.asyncio
async def test_bootstrap_sqlite_rejects_extra_state_db_table(tmp_path) -> None:
    sqlite_path = tmp_path / "state.db"
    extra_tables = STATE_DB_TABLES + (TableSpec("statedb_extra",
                                                "old StateDB table"), )
    await bootstrap_sqlite(str(sqlite_path), tables=extra_tables)

    with pytest.raises(StateDBSchemaMismatchError, match="unexpected tables"):
        await bootstrap_sqlite(str(sqlite_path))


@pytest.mark.asyncio
async def test_bootstrap_sqlite_rejects_schema_hash_mismatch(tmp_path) -> None:
    sqlite_path = tmp_path / "state.db"
    await bootstrap_sqlite(str(sqlite_path))
    _execute_sqlite(
        sqlite_path,
        "UPDATE statedb_schema SET payload = ? WHERE id = ?",
        _schema_metadata_payload("wrong"),
        SCHEMA_ROW_ID,
    )

    with pytest.raises(StateDBSchemaMismatchError, match="schema hash"):
        await bootstrap_sqlite(str(sqlite_path))


@pytest.mark.asyncio
async def test_bootstrap_postgres_takes_advisory_lock_before_ddl(
    monkeypatch, ) -> None:
    connection = _FakePostgresConnection()

    async def password() -> str:
        return "token"

    async def fake_connect(
        dsn: str,
        **kwargs: Any,
    ) -> _FakePostgresConnection:
        assert dsn == "postgresql://state"
        assert kwargs == {"password": password}
        return connection

    monkeypatch.setattr("statedb.db.bootstrap.asyncpg.connect", fake_connect)

    result = await bootstrap_postgres("postgresql://state", password=password)

    assert result.backend == "postgres"
    assert connection.closed is True
    first_sql, first_args = connection.execute_calls[0]
    assert "pg_advisory_xact_lock" in first_sql
    assert first_args
    assert "CREATE TABLE" not in first_sql


@pytest.mark.asyncio
async def test_bootstrap_postgres_applies_registered_schema_migration(
    monkeypatch, ) -> None:
    tables = (
        TableSpec("statedb_schema", "schema metadata"),
        TableSpec("projects", "project records"),
    )
    old_version = 1
    new_version = 2
    connection = _MigratingFakePostgresConnection(
        table_names={"statedb_schema"},
        schema_hash=schema_identity("postgres", (tables[0], )),
        schema_version=old_version,
    )

    async def fake_connect(dsn: str):
        assert dsn == "postgresql://state"
        return connection

    async def add_projects(conn):
        conn.table_names.add("projects")

    monkeypatch.setattr("statedb.db.bootstrap.asyncpg.connect", fake_connect)

    result = await bootstrap_postgres(
        "postgresql://state",
        tables=tables,
        schema_version=new_version,
        migrations={old_version: add_projects},
    )

    assert result.created_schema_row is False
    assert result.applied_migrations == (old_version, )
    assert connection.schema_hash == schema_identity("postgres", tables)
    assert connection.schema_version == new_version


@pytest.mark.asyncio
async def test_postgres_12_to_13_migration_replaces_api_key_indexes() -> None:
    connection = _FakePostgresConnection()

    await POSTGRES_SCHEMA_MIGRATIONS[12](connection)

    api_keys = next(table for table in STATE_DB_TABLES
                    if table.name == "api_keys")
    assert connection.execute_calls == [
        ("DROP INDEX IF EXISTS api_keys_customer_type_owner_idx", ()),
        ("DROP INDEX IF EXISTS api_keys_verification_lookup_idx", ()),
        ("DROP INDEX IF EXISTS api_keys_owner_name_unique_idx", ()),
        *((index.ddl("api_keys", "postgres"), ())
          for index in api_keys.indexes),
    ]


@pytest.mark.asyncio
async def test_postgres_13_to_14_migration_drops_job_resources() -> None:
    connection = _FakePostgresConnection()

    await POSTGRES_SCHEMA_MIGRATIONS[13](connection)

    assert connection.execute_calls == [
        ("DROP TABLE IF EXISTS job_resources", ()),
    ]


@pytest.mark.asyncio
async def test_postgres_14_to_15_migration_drops_baseline_jobs() -> None:
    connection = _FakePostgresConnection()

    await POSTGRES_SCHEMA_MIGRATIONS[14](connection)

    assert connection.execute_calls == [
        ("DROP TABLE IF EXISTS baseline_jobs", ()),
    ]


@pytest.mark.asyncio
async def test_postgres_15_to_16_migration_drops_online_serving_jobs() -> None:
    connection = _FakePostgresConnection()

    await POSTGRES_SCHEMA_MIGRATIONS[15](connection)

    assert connection.execute_calls == [
        ("DROP TABLE IF EXISTS online_serving_jobs", ()),
    ]


@pytest.mark.asyncio
async def test_bootstrap_postgres_upgrades_legacy_schema_identity(
    monkeypatch, ) -> None:
    tables = (
        TableSpec("statedb_schema", "schema metadata"),
        TableSpec("projects", "project records"),
    )
    connection = _MigratingFakePostgresConnection(
        table_names={table.name
                     for table in tables},
        schema_hash=legacy_schema_identity(tables),
        schema_version=SCHEMA_VERSION,
    )

    async def fake_connect(dsn: str):
        assert dsn == "postgresql://state"
        return connection

    monkeypatch.setattr("statedb.db.bootstrap.asyncpg.connect", fake_connect)

    result = await bootstrap_postgres("postgresql://state", tables=tables)

    assert result.created_schema_row is False
    assert result.applied_migrations == ()
    assert connection.schema_hash == schema_identity("postgres", tables)
    assert connection.schema_version == SCHEMA_VERSION


@pytest.mark.asyncio
async def test_bootstrap_postgres_applies_schema_migrations_in_order(
    monkeypatch, ) -> None:
    tables = (
        TableSpec("statedb_schema", "schema metadata"),
        TableSpec("projects", "project records"),
        TableSpec("jobs", "job records"),
    )
    connection = _MigratingFakePostgresConnection(
        table_names={"statedb_schema"},
        schema_hash=schema_identity("postgres", (tables[0], )),
        schema_version=1,
    )
    applied = []

    async def fake_connect(dsn: str):
        assert dsn == "postgresql://state"
        return connection

    async def add_projects(conn):
        applied.append(1)
        conn.table_names.add("projects")

    async def add_jobs(conn):
        applied.append(2)
        conn.table_names.add("jobs")

    monkeypatch.setattr("statedb.db.bootstrap.asyncpg.connect", fake_connect)

    result = await bootstrap_postgres(
        "postgresql://state",
        tables=tables,
        schema_version=3,
        migrations={
            1: add_projects,
            2: add_jobs,
        },
    )

    assert applied == [1, 2]
    assert result.applied_migrations == (1, 2)
    assert connection.schema_hash == schema_identity("postgres", tables)
    assert connection.schema_version == 3


@pytest.mark.asyncio
async def test_bootstrap_postgres_rejects_missing_intermediate_migration(
    monkeypatch, ) -> None:
    tables = (
        TableSpec("statedb_schema", "schema metadata"),
        TableSpec("projects", "project records"),
        TableSpec("jobs", "job records"),
    )
    old_hash = schema_identity("postgres", (tables[0], ))
    connection = _MigratingFakePostgresConnection(
        table_names={"statedb_schema"},
        schema_hash=old_hash,
        schema_version=1,
    )
    applied = []

    async def fake_connect(dsn: str):
        assert dsn == "postgresql://state"
        return connection

    async def add_projects(conn):
        applied.append(1)
        conn.table_names.add("projects")

    monkeypatch.setattr("statedb.db.bootstrap.asyncpg.connect", fake_connect)

    with pytest.raises(StateDBSchemaMismatchError,
                       match="missing migration from 2 to 3"):
        await bootstrap_postgres(
            "postgresql://state",
            tables=tables,
            schema_version=3,
            migrations={1: add_projects},
        )

    assert applied == []
    assert connection.table_names == {"statedb_schema"}
    assert connection.schema_hash == old_hash
    assert connection.schema_version == 1


@pytest.mark.asyncio
async def test_bootstrap_postgres_migrations_use_real_postgres_transaction(
    postgres_dsn: str,
    postgres_connection,
) -> None:
    old_tables = (TableSpec("statedb_schema", "schema metadata"), )
    target_tables = old_tables + (
        TableSpec("projects", "project records"),
        TableSpec("jobs", "job records"),
    )
    old_hash = schema_identity("postgres", old_tables)

    await _reset_postgres_schema(postgres_connection)
    try:
        await bootstrap_postgres(
            postgres_dsn,
            tables=old_tables,
            schema_version=1,
        )

        async def add_projects(conn):
            await conn.execute(_postgres_table_ddl("projects"))

        async def add_jobs_then_fail(conn):
            await conn.execute(_postgres_table_ddl("jobs"))
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await bootstrap_postgres(
                postgres_dsn,
                tables=target_tables,
                schema_version=3,
                migrations={
                    1: add_projects,
                    2: add_jobs_then_fail,
                },
            )

        assert await _fetch_postgres_table_names(postgres_connection) == {
            "statedb_schema"
        }
        payload = await _fetch_postgres_schema_metadata(postgres_connection)
        assert payload == {
            "schema_hash": old_hash,
            "schema_version": 1,
        }

        async def add_jobs(conn):
            await conn.execute(_postgres_table_ddl("jobs"))

        result = await bootstrap_postgres(
            postgres_dsn,
            tables=target_tables,
            schema_version=3,
            migrations={
                1: add_projects,
                2: add_jobs,
            },
        )

        assert result.applied_migrations == (1, 2)
        assert await _fetch_postgres_table_names(postgres_connection) == {
            "jobs",
            "projects",
            "statedb_schema",
        }
        payload = await _fetch_postgres_schema_metadata(postgres_connection)
        assert payload == {
            "schema_hash": schema_identity("postgres", target_tables),
            "schema_version": 3,
        }
    finally:
        await _reset_postgres_schema(postgres_connection)


def _oldest_supported_postgres_tables() -> tuple[TableSpec, ...]:
    legacy_api_key_indexes = (
        IndexSpec(
            name="api_keys_customer_type_owner_idx",
            postgres_expression=("(payload->>'customer_id'), "
                                 "(payload->>'type'), "
                                 "(payload->>'owner_user_id')"),
            sqlite_expression=("json_extract(payload, '$.customer_id'), "
                               "json_extract(payload, '$.type'), "
                               "json_extract(payload, '$.owner_user_id')"),
        ),
        IndexSpec(
            name="api_keys_verification_lookup_idx",
            postgres_expression=("(payload->>'customer_id'), "
                                 "(payload->>'name'), "
                                 "(payload->>'owner_user_hash')"),
            sqlite_expression=("json_extract(payload, '$.customer_id'), "
                               "json_extract(payload, '$.name'), "
                               "json_extract(payload, '$.owner_user_hash')"),
        ),
        IndexSpec(
            name="api_keys_owner_name_unique_idx",
            postgres_expression=("(payload->>'customer_id'), "
                                 "(payload->>'owner_user_id'), "
                                 "(payload->>'name')"),
            sqlite_expression=("json_extract(payload, '$.customer_id'), "
                               "json_extract(payload, '$.owner_user_id'), "
                               "json_extract(payload, '$.name')"),
            unique=True,
        ),
    )

    def at_oldest_version(table: TableSpec) -> TableSpec:
        if table.name == "identities":
            return replace(table, indexes=tuple(
                replace(index, postgres_where=None, sqlite_where=None)
                for index in table.indexes))
        if table.name == "api_keys":
            return TableSpec(table.name, table.purpose, legacy_api_key_indexes)
        return table

    tables = tuple(at_oldest_version(table) for table in STATE_DB_TABLES)
    return tables + (
        TableSpec("job_resources", "retired ML job resource analytics"),
        TableSpec("baseline_jobs", "retired baseline job metadata"),
        TableSpec("online_serving_jobs", "retired online-serving job metadata"),
    )


@pytest.mark.asyncio
async def test_postgres_identity_recreation_after_schema_migration(
    postgres_dsn: str,
    postgres_connection,
) -> None:
    old_tables = tuple(
        replace(table, indexes=tuple(
            replace(index, postgres_where=None, sqlite_where=None)
            for index in table.indexes))
        if table.name == "identities" else table
        for table in STATE_DB_TABLES)
    payload = {
        "identity_key": "databricks:workspace:client",
        "slug": "identity",
        "normalized_display_name": "identity",
        "tombstoned_at": "2026-01-01T00:00:00Z",
    }
    insert = ("INSERT INTO identities (id, payload, created_at, updated_at) "
              "VALUES ($1, $2::jsonb, now(), now())")
    await _reset_postgres_schema(postgres_connection)
    try:
        await bootstrap_postgres(
            postgres_dsn, tables=old_tables, schema_version=16)
        await postgres_connection.execute(insert, "deleted", json.dumps(payload))
        result = await bootstrap_postgres(
            postgres_dsn, migrations=POSTGRES_SCHEMA_MIGRATIONS)
        assert result.applied_migrations == (16, )
        payload["tombstoned_at"] = None
        await postgres_connection.execute(insert, "new", json.dumps(payload))
        with pytest.raises(asyncpg.UniqueViolationError):
            await postgres_connection.execute(
                insert, "duplicate", json.dumps(payload))
        assert await postgres_connection.fetchval(
            "SELECT count(*) FROM identities") == 2
    finally:
        await _reset_postgres_schema(postgres_connection)


@pytest.mark.asyncio
async def test_all_registered_postgres_migrations_run_in_order(
    postgres_dsn: str,
    postgres_connection,
) -> None:
    oldest_version = min(POSTGRES_SCHEMA_MIGRATIONS)
    expected_migrations = tuple(range(oldest_version, SCHEMA_VERSION))
    assert tuple(POSTGRES_SCHEMA_MIGRATIONS) == expected_migrations
    old_tables = _oldest_supported_postgres_tables()

    await _reset_postgres_schema(postgres_connection)
    try:
        await bootstrap_postgres(
            postgres_dsn,
            tables=old_tables,
            schema_version=oldest_version,
        )
        old_table_names = await _fetch_postgres_table_names(postgres_connection)
        assert {
            "job_resources",
            "baseline_jobs",
            "online_serving_jobs",
        } <= old_table_names

        result = await bootstrap_postgres(
            postgres_dsn,
            migrations=POSTGRES_SCHEMA_MIGRATIONS,
        )

        assert result.applied_migrations == expected_migrations
        assert await _fetch_postgres_table_names(postgres_connection) == set(
            STATE_DB_TABLE_NAMES)
        assert await _fetch_postgres_schema_metadata(postgres_connection) == {
            "schema_hash": schema_identity("postgres"),
            "schema_version": SCHEMA_VERSION,
        }
    finally:
        await _reset_postgres_schema(postgres_connection)


@pytest.mark.asyncio
async def test_real_postgres_enforces_global_project_and_group_names(
    postgres_dsn: str,
    postgres_connection,
) -> None:
    await _reset_postgres_schema(postgres_connection)
    try:
        await bootstrap_postgres(postgres_dsn)
        now = datetime(2026, 7, 5, tzinfo=timezone.utc)

        await postgres_connection.execute(
            "INSERT INTO groups (id, payload, created_at, updated_at) "
            "VALUES ($1, $2::jsonb, $3, $3)",
            "group_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            json.dumps({"normalized_display_name": "engineering"}),
            now,
        )
        with pytest.raises(asyncpg.UniqueViolationError):
            await postgres_connection.execute(
                "INSERT INTO groups (id, payload, created_at, updated_at) "
                "VALUES ($1, $2::jsonb, $3, $3)",
                "group_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                json.dumps({"normalized_display_name": "engineering"}),
                now,
            )

        await postgres_connection.execute(
            "INSERT INTO projects (id, payload, created_at, updated_at) "
            "VALUES ($1, $2::jsonb, $3, $3)",
            "project_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            json.dumps({
                "group_id": "group_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "normalized_display_name": "analytics",
            }),
            now,
        )
        with pytest.raises(asyncpg.UniqueViolationError):
            await postgres_connection.execute(
                "INSERT INTO projects (id, payload, created_at, updated_at) "
                "VALUES ($1, $2::jsonb, $3, $3)",
                "project_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                json.dumps({
                    "group_id": "group_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    "normalized_display_name": "analytics",
                }),
                now,
            )
    finally:
        await _reset_postgres_schema(postgres_connection)
