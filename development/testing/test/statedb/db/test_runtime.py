# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import multiprocessing as mp
import os
import queue
import sqlite3
import threading
from contextlib import suppress
from typing import Any

import pytest

import statedb.db.runtime as runtime_mod
from common_utils.error_diagnostics import (
    exception_message_should_be_logged_at_error,
    safe_exception_message_for_log,
)
from statedb.db import (
    SCHEMA_VERSION,
    STATE_DB_TABLE_NAMES,
    StateDB,
    schema_identity,
)
from statedb.db.bootstrap import BootstrapResult


class _FakePostgresTransaction:
    def __init__(self, connection: "_FakePostgresConnection",
                 readonly: bool) -> None:
        self._connection = connection
        self._readonly = readonly

    async def __aenter__(self) -> "_FakePostgresTransaction":
        self._connection.transaction_readonly_modes.append(self._readonly)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class _FakePostgresConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.transaction_readonly_modes: list[bool] = []
        self.close_count = 0

    def transaction(self, *,
                    readonly: bool = False) -> _FakePostgresTransaction:
        return _FakePostgresTransaction(self, readonly)

    async def execute(self, sql: str, *args: Any) -> str:
        self.executed.append((sql, args))
        return "EXECUTE 1"

    async def close(self) -> None:
        self.close_count += 1


class _FakePostgresPoolAcquire:
    def __init__(self, pool: "_FakePostgresPool") -> None:
        self._pool = pool

    async def __aenter__(self) -> _FakePostgresConnection:
        self._pool.acquire_count += 1
        return self._pool.connection

    async def __aexit__(self, *exc_info: object) -> None:
        self._pool.release_count += 1


class _FakePostgresPool:
    def __init__(self) -> None:
        self.connection = _FakePostgresConnection()
        self.acquire_count = 0
        self.release_count = 0
        self.closed = False

    def acquire(self) -> _FakePostgresPoolAcquire:
        return _FakePostgresPoolAcquire(self)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_postgres_network_failure_is_marked_at_connect_boundary(
        monkeypatch) -> None:
    error = ConnectionRefusedError('private-db.internal:5432')

    async def create_pool(**_kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(runtime_mod.asyncpg, 'create_pool', create_pool)
    pool = runtime_mod._PostgresConnectionPool(  # pylint: disable=protected-access
        'postgresql://state-db', min_pool_size=1, max_pool_size=10)

    with pytest.raises(ConnectionRefusedError) as raised:
        await pool._open_pool()  # pylint: disable=protected-access

    assert raised.value is error
    assert not exception_message_should_be_logged_at_error(error)
    assert safe_exception_message_for_log(error) == (
        'StateDB PostgreSQL connection failed')


def _execute_sqlite(sqlite_path, sql: str, *args) -> None:
    connection = sqlite3.connect(sqlite_path)
    try:
        connection.execute(sql, args)
        connection.commit()
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_state_db_sqlite_write_and_read_tx(tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    try:
        async with db.write_tx() as tx:
            await tx.execute(
                """
                INSERT INTO projects (id, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                "project_1",
                json.dumps({"display_name": "Project One"}),
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            )

        async with db.read_tx() as tx:
            row = await tx.fetchrow("SELECT payload FROM projects WHERE id = ?",
                                    "project_1")
    finally:
        await db.close()

    assert row is not None
    assert json.loads(row[0]) == {"display_name": "Project One"}


@pytest.mark.asyncio
async def test_state_db_sqlite_write_tx_rolls_back_on_error(tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    try:
        with pytest.raises(RuntimeError, match="abort"):
            async with db.write_tx() as tx:
                await tx.execute(
                    """
                    INSERT INTO projects (id, payload, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    "project_1",
                    json.dumps({"display_name": "Project One"}),
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                )
                raise RuntimeError("abort")

        async with db.read_tx() as tx:
            row = await tx.fetchrow("SELECT payload FROM projects WHERE id = ?",
                                    "project_1")
    finally:
        await db.close()

    assert row is None


@pytest.mark.asyncio
async def test_state_db_sqlite_write_tx_cancellation_does_not_leak_writer_lock(
        tmp_path) -> None:
    """A cancelled queued BEGIN must be rolled back before its pool is reused."""
    sqlite_path = str(tmp_path / 'state.db')
    holder = await StateDB.open_sqlite(sqlite_path, pool_size=1)
    cancelled = await StateDB.open_sqlite(sqlite_path, pool_size=1)
    verifier = await StateDB.open_sqlite(sqlite_path, pool_size=1)

    async def set_busy_timeout(db: StateDB, milliseconds: int) -> None:
        pool = db._sqlite_pool  # pylint: disable=protected-access
        assert pool is not None
        assert len(pool._available) == 1  # pylint: disable=protected-access
        connection = pool._available[0]  # pylint: disable=protected-access
        cursor = await runtime_mod._await_aiosqlite(  # pylint: disable=protected-access
            connection.execute(f'PRAGMA busy_timeout={milliseconds}'))
        await runtime_mod._await_aiosqlite(cursor.close())  # pylint: disable=protected-access

    holder_context = holder.write_tx()
    await holder_context.__aenter__()
    try:
        blocked = asyncio.create_task(cancelled.write_tx().__aenter__())
        await asyncio.sleep(0.05)
        blocked.cancel()

        # Releasing this first lets the already-queued BEGIN complete. Before
        # the fix it completed after cancellation and left its connection in
        # an unowned write transaction.
        await holder_context.__aexit__(None, None, None)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(blocked, timeout=2)

        await set_busy_timeout(verifier, 100)
        async with verifier.write_tx() as tx:
            await tx.execute('SELECT 1')
    finally:
        await holder.close()
        await cancelled.close()
        await verifier.close()


@pytest.mark.asyncio
async def test_state_db_schema_status_reports_healthy_sqlite(tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    try:
        status = await db.get_schema_status()
        healthy = await db.is_healthy()
    finally:
        await db.close()

    assert status.ok is True
    assert status.backend == "sqlite"
    assert status.expected_schema_version == SCHEMA_VERSION
    assert status.stored_schema_version == SCHEMA_VERSION
    assert status.expected_schema_hash == schema_identity("sqlite")
    assert status.stored_schema_hash == schema_identity("sqlite")
    assert status.expected_table_count == len(STATE_DB_TABLE_NAMES)
    assert status.existing_state_table_count == len(STATE_DB_TABLE_NAMES)
    assert status.missing_table_names == ()
    assert status.error_type is None
    assert status.error_message is None
    assert healthy is True


@pytest.mark.asyncio
async def test_state_db_schema_status_reports_missing_table(tmp_path, ) -> None:
    sqlite_path = tmp_path / "state.db"
    db = await StateDB.open_sqlite(str(sqlite_path))
    try:
        _execute_sqlite(sqlite_path, "DROP TABLE projects")

        status = await db.get_schema_status()
        healthy = await db.is_healthy()
    finally:
        await db.close()

    assert status.ok is False
    assert status.stored_schema_version == SCHEMA_VERSION
    assert status.stored_schema_hash == schema_identity("sqlite")
    assert status.missing_table_names == ("projects", )
    assert status.existing_state_table_count == len(STATE_DB_TABLE_NAMES) - 1
    assert healthy is False


@pytest.mark.asyncio
async def test_state_db_sqlite_uses_owned_async_connections(
    tmp_path,
    monkeypatch,
) -> None:
    sqlite_connect_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    real_connect = runtime_mod.aiosqlite.connect

    def connect(*args: Any, **kwargs: Any) -> Any:
        sqlite_connect_calls.append((args, kwargs))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(runtime_mod.aiosqlite, "connect", connect)

    db = await StateDB.open_sqlite(str(tmp_path / "state.db"), pool_size=2)
    try:
        assert len(sqlite_connect_calls) == 2

        async with db.write_tx() as tx:
            await tx.execute(
                """
                INSERT INTO projects (id, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                "project_1",
                json.dumps({"display_name": "Project One"}),
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            )
        async with db.read_tx() as tx:
            row = await tx.fetchrow("SELECT payload FROM projects WHERE id = ?",
                                    "project_1")

        assert row is not None
        assert len(sqlite_connect_calls) == 2
    finally:
        await db.close()

    with pytest.raises(RuntimeError, match="sqlite backend is closed"):
        async with db.read_tx():
            pass


def test_state_db_sqlite_close_is_not_tied_to_opening_loop(tmp_path) -> None:
    async def open_db() -> StateDB:
        db = await StateDB.open_sqlite(str(tmp_path / "state.db"), pool_size=1)
        async with db.read_tx():
            pass
        return db

    db = asyncio.run(open_db())
    asyncio.run(db.close())


def test_state_db_sqlite_transactions_can_run_on_new_thread(tmp_path) -> None:
    async def open_db() -> StateDB:
        return await StateDB.open_sqlite(str(tmp_path / "state.db"),
                                         pool_size=1)

    async def insert_from_thread(db: StateDB) -> None:
        async with db.write_tx() as tx:
            await tx.execute(
                """
                INSERT INTO projects (id, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                "project_from_thread",
                json.dumps({"display_name": "Project From Thread"}),
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            )

    async def read_from_parent(db: StateDB) -> Any:
        async with db.read_tx() as tx:
            return await tx.fetchrow(
                "SELECT payload FROM projects WHERE id = ?",
                "project_from_thread",
            )

    db = asyncio.run(open_db())
    errors: list[Exception] = []

    def run_in_thread() -> None:
        try:
            asyncio.run(insert_from_thread(db))
        except Exception as exc:
            errors.append(exc)

    try:
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()
        if errors:
            raise errors[0]

        row = asyncio.run(read_from_parent(db))
    finally:
        asyncio.run(db.close())

    assert row is not None
    assert json.loads(row[0]) == {"display_name": "Project From Thread"}


def test_state_db_sqlite_transactions_do_not_cross_process(tmp_path) -> None:
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("fork start method is required for inherited connections")

    async def open_db() -> StateDB:
        return await StateDB.open_sqlite(str(tmp_path / "state.db"),
                                         pool_size=1)

    async def read_in_child(db: StateDB) -> None:
        async with db.read_tx() as tx:
            await tx.fetchrow("SELECT COUNT(*) FROM projects")

    db = asyncio.run(open_db())
    context = mp.get_context("fork")
    result_queue: mp.Queue = context.Queue()

    def run_in_child() -> None:
        original_await_aiosqlite = runtime_mod._await_aiosqlite  # pylint: disable=protected-access

        async def fail_fast_await_aiosqlite(
            awaitable: Any,
            *,
            timeout_seconds: float = 60.0,
        ) -> Any:
            del timeout_seconds
            return await original_await_aiosqlite(
                awaitable,
                timeout_seconds=0.05,
            )

        runtime_mod._await_aiosqlite = fail_fast_await_aiosqlite  # pylint: disable=protected-access
        try:
            asyncio.run(read_in_child(db))
        except Exception as exc:
            result_queue.put(("error", type(exc).__name__, str(exc)))
        else:
            result_queue.put(("ok", "", ""))

    process = context.Process(target=run_in_child)
    try:
        process.start()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("child process hung while reusing a sqlite connection")
        try:
            status, exc_type, exc_message = result_queue.get(timeout=1)
        except queue.Empty:
            pytest.fail("child process exited without reporting sqlite result")
    finally:
        asyncio.run(db.close())

    assert process.exitcode == 0
    assert status == "error"
    assert exc_type == "TimeoutError"
    assert "timed out" in exc_message


@pytest.mark.asyncio
async def test_state_db_postgres_transactions_lease_from_pool(
    monkeypatch, ) -> None:
    bootstrap_calls: list[str] = []
    create_pool_calls: list[dict[str, Any]] = []
    fake_pool = _FakePostgresPool()

    async def password() -> str:
        return "token"

    async def bootstrap_postgres(dsn: str, **kwargs: Any) -> BootstrapResult:
        bootstrap_calls.append(dsn)
        assert kwargs["password"] is password
        return BootstrapResult(
            backend="postgres",
            schema_hash="hash",
            table_count=0,
            index_count=0,
            created_schema_row=False,
        )

    async def create_pool(**kwargs: Any) -> _FakePostgresPool:
        create_pool_calls.append(kwargs)
        return fake_pool

    async def connect(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("StateDB transactions must use the owned pool")

    monkeypatch.setattr(runtime_mod, "bootstrap_postgres", bootstrap_postgres)
    monkeypatch.setattr(runtime_mod.asyncpg, "create_pool", create_pool)
    monkeypatch.setattr(runtime_mod.asyncpg, "connect", connect)

    db = await StateDB.open_postgres(
        "postgresql://state-db", password=password)
    try:
        async with db.write_tx() as tx:
            await tx.execute("SELECT 1")
        async with db.read_tx() as tx:
            await tx.execute("SELECT 2")
    finally:
        await db.close()

    assert bootstrap_calls == ["postgresql://state-db"]
    assert create_pool_calls == [{
        "dsn": "postgresql://state-db",
        "password": password,
        "min_size": 1,
        "max_size": 10,
    }]
    assert fake_pool.acquire_count == 2
    assert fake_pool.release_count == 2
    assert fake_pool.connection.transaction_readonly_modes == [False, True]
    assert fake_pool.connection.executed == [
        ("SELECT 1", ()),
        ("SELECT 2", ()),
    ]
    assert fake_pool.closed is True


@pytest.mark.asyncio
async def test_state_db_postgres_reuses_pool_from_non_owner_loop(
    monkeypatch, ) -> None:
    fake_pool = _FakePostgresPool()
    create_pool_calls: list[dict[str, Any]] = []

    async def bootstrap_postgres(dsn: str, **kwargs: Any) -> BootstrapResult:
        del kwargs
        return BootstrapResult(
            backend="postgres",
            schema_hash="hash",
            table_count=0,
            index_count=0,
            created_schema_row=False,
        )

    async def create_pool(**kwargs: Any) -> _FakePostgresPool:
        create_pool_calls.append(kwargs)
        return fake_pool

    async def connect(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("StateDB transactions must use postgres pools")

    monkeypatch.setattr(runtime_mod, "bootstrap_postgres", bootstrap_postgres)
    monkeypatch.setattr(runtime_mod.asyncpg, "create_pool", create_pool)
    monkeypatch.setattr(runtime_mod.asyncpg, "connect", connect)

    db = await StateDB.open_postgres("postgresql://state-db")

    async def run_tx_on_new_loop() -> None:
        async with db.write_tx() as tx:
            await tx.execute("SELECT 3")

    errors: list[Exception] = []

    def run_in_thread() -> None:
        try:
            asyncio.run(run_tx_on_new_loop())
        except Exception as exc:
            errors.append(exc)

    try:
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()
        if errors:
            raise errors[0]
    finally:
        await db.close()

    assert create_pool_calls == [{
        "dsn": "postgresql://state-db",
        "min_size": 1,
        "max_size": 10,
    }]
    assert fake_pool.acquire_count == 1
    assert fake_pool.release_count == 1
    assert fake_pool.connection.transaction_readonly_modes == [False]
    assert fake_pool.connection.executed == [("SELECT 3", ())]
    assert fake_pool.closed is True


@pytest.mark.asyncio
async def test_postgres_pool_close_drains_in_flight_open(monkeypatch, ) -> None:
    create_pool_started = threading.Event()
    finish_create_pool = threading.Event()
    fake_pool = _FakePostgresPool()

    async def create_pool(**kwargs: Any) -> _FakePostgresPool:
        del kwargs
        create_pool_started.set()
        await asyncio.to_thread(finish_create_pool.wait)
        return fake_pool

    monkeypatch.setattr(runtime_mod.asyncpg, "create_pool", create_pool)
    pool = runtime_mod._PostgresConnectionPool(  # pylint: disable=protected-access
        "postgresql://state-db",
        min_pool_size=1,
        max_pool_size=10,
    )

    open_task = asyncio.create_task(pool.open())
    close_task = None
    try:
        assert await asyncio.to_thread(create_pool_started.wait, 1)
        close_task = asyncio.create_task(pool.close())
        await asyncio.sleep(0)

        assert open_task.done() is False
        assert close_task.done() is False

        finish_create_pool.set()
        await asyncio.wait_for(open_task, timeout=1)
        await asyncio.wait_for(close_task, timeout=1)
        open_done_after_close = open_task.done()
    finally:
        finish_create_pool.set()
        if not open_task.done():
            await asyncio.wait_for(open_task, timeout=1)
        if close_task is not None and not close_task.done():
            await asyncio.wait_for(close_task, timeout=1)

    assert open_done_after_close is True
    assert fake_pool.closed is True


@pytest.mark.asyncio
async def test_postgres_pool_close_allows_active_transaction_to_release(
    monkeypatch, ) -> None:
    close_started = threading.Event()
    release_started = threading.Event()

    class BlockingClosePostgresPoolAcquire(_FakePostgresPoolAcquire):
        async def __aexit__(self, *exc_info: object) -> None:
            release_started.set()
            await super().__aexit__(*exc_info)

    class BlockingClosePostgresPool(_FakePostgresPool):
        def acquire(self) -> BlockingClosePostgresPoolAcquire:
            return BlockingClosePostgresPoolAcquire(self)

        async def close(self) -> None:
            close_started.set()
            await asyncio.to_thread(release_started.wait)
            await super().close()

    async def bootstrap_postgres(dsn: str, **kwargs: Any) -> BootstrapResult:
        del dsn, kwargs
        return BootstrapResult(
            backend="postgres",
            schema_hash="hash",
            table_count=0,
            index_count=0,
            created_schema_row=False,
        )

    fake_pool = BlockingClosePostgresPool()

    async def create_pool(**kwargs: Any) -> BlockingClosePostgresPool:
        del kwargs
        return fake_pool

    monkeypatch.setattr(runtime_mod, "bootstrap_postgres", bootstrap_postgres)
    monkeypatch.setattr(runtime_mod.asyncpg, "create_pool", create_pool)

    db = await StateDB.open_postgres("postgresql://state-db")
    tx_context = db.write_tx()
    await tx_context.__aenter__()
    close_task = asyncio.create_task(db.close())
    try:
        assert await asyncio.to_thread(close_started.wait, 1)

        release_error = None
        try:
            await tx_context.__aexit__(None, None, None)
        except Exception as exc:
            release_error = exc
        finally:
            release_started.set()

        await asyncio.wait_for(close_task, timeout=1)
    finally:
        release_started.set()
        if not close_task.done():
            await asyncio.wait_for(close_task, timeout=1)

    assert release_error is None
    assert fake_pool.release_count == 1
    assert fake_pool.closed is True


@pytest.mark.asyncio
async def test_postgres_process_pool_retries_after_late_open_failure(
    monkeypatch, ) -> None:
    create_pool_calls: list[dict[str, Any]] = []
    fake_pool = _FakePostgresPool()

    async def create_pool(**kwargs: Any) -> _FakePostgresPool:
        create_pool_calls.append(kwargs)
        if len(create_pool_calls) == 1:
            raise RuntimeError("open failed")
        return fake_pool

    monkeypatch.setattr(runtime_mod.asyncpg, "create_pool", create_pool)
    pools = runtime_mod._PostgresConnectionPools(  # pylint: disable=protected-access
        "postgresql://state-db",
        min_pool_size=1,
        max_pool_size=10,
    )

    with pytest.raises(RuntimeError, match="open failed"):
        async with pools.transaction("read"):
            pass

    async with pools.transaction("write") as tx:
        await tx.execute("SELECT 5")

    await pools.close()

    expected_pool_kwargs = {
        "dsn": "postgresql://state-db",
        "min_size": 1,
        "max_size": 10,
    }
    assert create_pool_calls == [expected_pool_kwargs, expected_pool_kwargs]
    assert fake_pool.acquire_count == 1
    assert fake_pool.release_count == 1
    assert fake_pool.connection.executed == [("SELECT 5", ())]
    assert fake_pool.closed is True


@pytest.mark.asyncio
async def test_postgres_pool_cancelled_open_waiter_does_not_poison_pool(
    monkeypatch, ) -> None:
    create_pool_started = threading.Event()
    finish_create_pool = threading.Event()
    fake_pool = _FakePostgresPool()

    async def create_pool(**kwargs: Any) -> _FakePostgresPool:
        del kwargs
        create_pool_started.set()
        await asyncio.to_thread(finish_create_pool.wait)
        return fake_pool

    monkeypatch.setattr(runtime_mod.asyncpg, "create_pool", create_pool)
    pool = runtime_mod._PostgresConnectionPool(  # pylint: disable=protected-access
        "postgresql://state-db",
        min_pool_size=1,
        max_pool_size=10,
    )

    open_task = asyncio.create_task(pool.open())
    try:
        assert await asyncio.to_thread(create_pool_started.wait, 1)
        open_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await open_task

        finish_create_pool.set()
        await pool.open()
    finally:
        finish_create_pool.set()
        await pool.close()

    assert fake_pool.closed is True


@pytest.mark.asyncio
async def test_postgres_pool_cancelled_acquire_releases_completed_lease(
    monkeypatch, ) -> None:
    transaction_started = threading.Event()
    finish_transaction = threading.Event()

    class BlockingTransaction(_FakePostgresTransaction):
        async def __aenter__(self) -> "BlockingTransaction":
            transaction_started.set()
            await asyncio.to_thread(finish_transaction.wait)
            await super().__aenter__()
            return self

    class BlockingConnection(_FakePostgresConnection):
        def transaction(self, *, readonly: bool = False) -> BlockingTransaction:
            return BlockingTransaction(self, readonly)

    fake_pool = _FakePostgresPool()
    fake_pool.connection = BlockingConnection()

    async def create_pool(**kwargs: Any) -> _FakePostgresPool:
        del kwargs
        return fake_pool

    monkeypatch.setattr(runtime_mod.asyncpg, "create_pool", create_pool)
    pool = runtime_mod._PostgresConnectionPool(  # pylint: disable=protected-access
        "postgresql://state-db",
        min_pool_size=1,
        max_pool_size=10,
    )

    acquire_task = asyncio.create_task(pool.acquire("write"))
    try:
        assert await asyncio.to_thread(transaction_started.wait, 1)
        acquire_task.cancel()
        finish_transaction.set()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(acquire_task, timeout=1)
    finally:
        finish_transaction.set()
        await pool.close()

    assert fake_pool.acquire_count == 1
    assert fake_pool.release_count == 1
    assert fake_pool.connection.transaction_readonly_modes == [False]


@pytest.mark.asyncio
async def test_postgres_pool_cancelled_pending_acquire_does_not_hang(
    monkeypatch, ) -> None:
    acquire_started = threading.Event()
    acquire_cancelled = threading.Event()
    finish_acquire = threading.Event()

    class BlockingAcquireContext(_FakePostgresPoolAcquire):
        async def __aenter__(self) -> _FakePostgresConnection:
            acquire_started.set()
            try:
                await asyncio.to_thread(finish_acquire.wait)
            except asyncio.CancelledError:
                acquire_cancelled.set()
                raise
            return await super().__aenter__()

    class BlockingAcquirePool(_FakePostgresPool):
        def acquire(self) -> BlockingAcquireContext:
            return BlockingAcquireContext(self)

    fake_pool = BlockingAcquirePool()

    async def create_pool(**kwargs: Any) -> BlockingAcquirePool:
        del kwargs
        return fake_pool

    monkeypatch.setattr(runtime_mod.asyncpg, "create_pool", create_pool)
    pool = runtime_mod._PostgresConnectionPool(  # pylint: disable=protected-access
        "postgresql://state-db",
        min_pool_size=1,
        max_pool_size=10,
    )

    acquire_task = asyncio.create_task(pool.acquire("write"))
    try:
        assert await asyncio.to_thread(acquire_started.wait, 1)
        acquire_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(acquire_task, timeout=0.1)
        assert await asyncio.to_thread(acquire_cancelled.wait, 1)
    finally:
        finish_acquire.set()
        with suppress(asyncio.CancelledError):
            await acquire_task
        await pool.close()

    assert fake_pool.acquire_count == 0
    assert fake_pool.release_count == 0


@pytest.mark.asyncio
async def test_postgres_pool_release_after_owner_loop_stops_fails_fast(
    monkeypatch, ) -> None:
    fake_pool = _FakePostgresPool()

    async def create_pool(**kwargs: Any) -> _FakePostgresPool:
        del kwargs
        return fake_pool

    monkeypatch.setattr(runtime_mod.asyncpg, "create_pool", create_pool)
    pool = runtime_mod._PostgresConnectionPool(  # pylint: disable=protected-access
        "postgresql://state-db",
        min_pool_size=1,
        max_pool_size=10,
    )

    lease = await pool.acquire("write")
    await pool.close()

    with pytest.raises(RuntimeError, match="postgres backend is closed"):
        await asyncio.wait_for(
            pool.release(lease, None, None, None),
            timeout=1,
        )


def test_state_db_postgres_uses_process_local_pool_after_fork(
    monkeypatch, ) -> None:
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("fork start method is required for inherited connections")

    connect_calls: list[dict[str, Any]] = []
    create_pool_calls: list[dict[str, Any]] = []
    created_pools: list[_FakePostgresPool] = []

    async def bootstrap_postgres(dsn: str, **kwargs: Any) -> BootstrapResult:
        del kwargs
        return BootstrapResult(
            backend="postgres",
            schema_hash="hash",
            table_count=0,
            index_count=0,
            created_schema_row=False,
        )

    async def create_pool(**kwargs: Any) -> _FakePostgresPool:
        pool = _FakePostgresPool()
        create_pool_calls.append(kwargs)
        created_pools.append(pool)
        return pool

    async def connect(**kwargs: Any) -> None:
        connect_calls.append(kwargs)
        raise AssertionError("StateDB transactions must use postgres pools")

    monkeypatch.setattr(runtime_mod, "bootstrap_postgres", bootstrap_postgres)
    monkeypatch.setattr(runtime_mod.asyncpg, "create_pool", create_pool)
    monkeypatch.setattr(runtime_mod.asyncpg, "connect", connect)

    db = asyncio.run(StateDB.open_postgres("postgresql://state-db"))
    context = mp.get_context("fork")
    result_queue: mp.Queue = context.Queue()

    def run_in_child() -> None:
        async def run_tx() -> None:
            async with db.write_tx() as tx:
                await tx.execute("SELECT 4")

        try:
            asyncio.run(run_tx())
        except Exception as exc:
            result_queue.put({
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })
            return

        child_pool = created_pools[-1]
        result_queue.put({
            "status": "ok",
            "connect_calls": connect_calls,
            "create_pool_calls": create_pool_calls,
            "pool_acquire_count": child_pool.acquire_count,
            "pool_release_count": child_pool.release_count,
            "executed": child_pool.connection.executed,
            "closed_before_child_close": child_pool.closed,
        })
        asyncio.run(db.close())
        result_queue.put({
            "status": "closed",
            "child_pool_closed": child_pool.closed,
        })

    process = context.Process(target=run_in_child)
    try:
        process.start()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("child process hung while using postgres StateDB")
        try:
            result = result_queue.get(timeout=1)
        except queue.Empty:
            pytest.fail(
                "child process exited without reporting postgres result")
        try:
            close_result = result_queue.get(timeout=1)
        except queue.Empty:
            pytest.fail("child process exited without closing postgres pool")
    finally:
        asyncio.run(db.close())

    expected_pool_kwargs = {
        "dsn": "postgresql://state-db",
        "min_size": 1,
        "max_size": 10,
    }
    assert process.exitcode == 0
    assert result == {
        "status": "ok",
        "connect_calls": [],
        "create_pool_calls": [expected_pool_kwargs, expected_pool_kwargs],
        "pool_acquire_count": 1,
        "pool_release_count": 1,
        "executed": [("SELECT 4", ())],
        "closed_before_child_close": False,
    }
    assert close_result == {
        "status": "closed",
        "child_pool_closed": True,
    }
    assert connect_calls == []
    assert create_pool_calls == [expected_pool_kwargs]
    assert created_pools[0].closed is True


def test_state_db_postgres_child_close_discards_inherited_pools(
    monkeypatch, ) -> None:
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("fork start method is required for inherited connections")

    async def bootstrap_postgres(dsn: str, **kwargs: Any) -> BootstrapResult:
        del dsn, kwargs
        return BootstrapResult(
            backend="postgres",
            schema_hash="hash",
            table_count=0,
            index_count=0,
            created_schema_row=False,
        )

    async def create_pool(**kwargs: Any) -> _FakePostgresPool:
        del kwargs
        return _FakePostgresPool()

    monkeypatch.setattr(runtime_mod, "bootstrap_postgres", bootstrap_postgres)
    monkeypatch.setattr(runtime_mod.asyncpg, "create_pool", create_pool)

    db = asyncio.run(StateDB.open_postgres("postgresql://state-db"))
    assert db._postgres_pools is not None  # pylint: disable=protected-access
    parent_pid = os.getpid()
    context = mp.get_context("fork")
    result_queue: mp.Queue = context.Queue()

    def run_in_child() -> None:
        assert db._postgres_pools is not None  # pylint: disable=protected-access
        pools = db._postgres_pools  # pylint: disable=protected-access
        keys_before_close = tuple(pools._pools.keys())  # pylint: disable=protected-access
        asyncio.run(db.close())
        keys_after_close = tuple(pools._pools.keys())  # pylint: disable=protected-access
        result_queue.put((keys_before_close, keys_after_close))

    process = context.Process(target=run_in_child)
    try:
        process.start()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("child process hung while closing postgres StateDB")
        try:
            keys_before_close, keys_after_close = result_queue.get(timeout=1)
        except queue.Empty:
            pytest.fail("child process exited without reporting pool keys")
    finally:
        asyncio.run(db.close())

    assert process.exitcode == 0
    assert keys_before_close == (parent_pid, )
    assert keys_after_close == ()
