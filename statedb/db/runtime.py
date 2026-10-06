# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import aiosqlite
import asyncpg
from asyncpg.pool import PoolConnectionProxy

from io_utils.errors import restricted_diagnostic_boundary
from statedb.db.bootstrap import (
    BootstrapResult,
    bootstrap_postgres,
    bootstrap_sqlite,
)
from statedb.db.migrations import POSTGRES_SCHEMA_MIGRATIONS
from statedb.db.schema import (
    SCHEMA_ROW_ID,
    SCHEMA_VERSION,
    STATE_DB_TABLE_NAMES,
    BackendName,
    schema_identity,
)


def _sqlite_md5(value: object) -> str:
    return hashlib.md5(str(value).encode("utf-8")).hexdigest()


TxMode = Literal["read", "write"]
PostgresConnection = asyncpg.Connection | PoolConnectionProxy
_SQLITE_POOL_WAIT_SECONDS = 0.001


async def _await_aiosqlite(
    awaitable: Any,
    *,
    timeout_seconds: float = 60.0,
) -> Any:
    """Await aiosqlite worker-thread work without relying on one wakeup."""
    task = asyncio.ensure_future(awaitable)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while not task.done():
        if loop.time() >= deadline:
            task.cancel()
            raise TimeoutError("StateDB sqlite operation timed out")
        await asyncio.sleep(0.001)
    return task.result()


@runtime_checkable
class StateTx(Protocol):
    backend: BackendName

    async def execute(self, sql: str, *args: Any) -> Any:
        raise NotImplementedError()

    async def fetchrow(self, sql: str, *args: Any) -> Any | None:
        raise NotImplementedError()

    async def fetch(self, sql: str, *args: Any) -> Sequence[Any]:
        raise NotImplementedError()


@dataclass
class _SqliteStateTx:
    connection: aiosqlite.Connection
    backend: BackendName = "sqlite"

    async def execute(self, sql: str, *args: Any) -> Any:
        cursor = await _await_aiosqlite(self.connection.execute(sql, args))
        await _await_aiosqlite(cursor.close())

    async def fetchrow(self, sql: str, *args: Any) -> Any | None:
        cursor = await _await_aiosqlite(self.connection.execute(sql, args))
        try:
            return await _await_aiosqlite(cursor.fetchone())
        finally:
            await _await_aiosqlite(cursor.close())

    async def fetch(self, sql: str, *args: Any) -> Sequence[Any]:
        cursor = await _await_aiosqlite(self.connection.execute(sql, args))
        try:
            return await _await_aiosqlite(cursor.fetchall())
        finally:
            await _await_aiosqlite(cursor.close())


@dataclass
class _PostgresStateTx:
    pool: "_PostgresConnectionPool"
    connection: PostgresConnection
    backend: BackendName = "postgres"

    async def execute(self, sql: str, *args: Any) -> str:
        return await self.pool.execute(self.connection, sql, *args)

    async def fetchrow(self, sql: str, *args: Any) -> asyncpg.Record | None:
        return await self.pool.fetchrow(self.connection, sql, *args)

    async def fetch(self, sql: str, *args: Any) -> Sequence[asyncpg.Record]:
        return await self.pool.fetch(self.connection, sql, *args)


@dataclass(frozen=True)
class _PostgresConnectionLease:
    connection: PostgresConnection
    acquire_context: Any
    transaction_context: Any


@dataclass
class _PostgresAcquireState:
    acquire_context: Any | None = None
    connection: PostgresConnection | None = None


class _PostgresConnectionPool:
    """Own a process-local asyncpg pool on a dedicated event-loop thread.

    asyncpg pools and pooled connections are event-loop affine, while StateDB is
    used from sync fixtures, worker threads, and request loops. This wrapper
    routes all asyncpg operations to its owner loop and exposes loop-agnostic
    transaction helpers to callers in the same process. It must not be reused
    after fork; _PostgresConnectionPools creates a fresh wrapper per child PID.
    """

    def __init__(
        self,
        dsn: str,
        *,
        password: Callable[[], Awaitable[str]] | None = None,
        min_pool_size: int,
        max_pool_size: int,
    ) -> None:
        self._dsn = dsn
        self._password = password
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._pool: asyncpg.Pool | None = None
        self._open_future: Any = None
        self._cancelled_acquire_futures: set[Any] = set()
        self._cancelled_acquire_releases: set[Any] = set()
        self._closed = False
        self._loop_ready = threading.Event()

    async def open(self) -> None:
        await self._ensure_open()

    async def acquire(self, mode: TxMode) -> _PostgresConnectionLease:
        await self._ensure_open()
        state = _PostgresAcquireState()
        future = self._submit(self._acquire(mode, state))
        try:
            return await asyncio.shield(asyncio.wrap_future(future))
        except asyncio.CancelledError as exc:
            self._release_cancelled_acquire(future, state, exc)
            raise

    async def release(
        self,
        lease: _PostgresConnectionLease,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        await self._ensure_owner_process()
        await self._run(self._release(lease, exc_type, exc, traceback))

    async def execute(
        self,
        connection: PostgresConnection,
        sql: str,
        *args: Any,
    ) -> str:
        await self._ensure_open()
        with restricted_diagnostic_boundary(
                safe_message='StateDB PostgreSQL execute failed',
                exception_types=asyncpg.PostgresError):
            return await self._run(connection.execute(sql, *args))

    async def fetchrow(
        self,
        connection: PostgresConnection,
        sql: str,
        *args: Any,
    ) -> asyncpg.Record | None:
        await self._ensure_open()
        with restricted_diagnostic_boundary(
                safe_message='StateDB PostgreSQL fetch failed',
                exception_types=asyncpg.PostgresError):
            return await self._run(connection.fetchrow(sql, *args))

    async def fetch(
        self,
        connection: PostgresConnection,
        sql: str,
        *args: Any,
    ) -> Sequence[asyncpg.Record]:
        await self._ensure_open()
        with restricted_diagnostic_boundary(
                safe_message='StateDB PostgreSQL fetch failed',
                exception_types=asyncpg.PostgresError):
            return await self._run(connection.fetch(sql, *args))

    async def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop = self._loop
            thread = self._thread
            open_future = self._open_future
        if loop is None:
            return

        if open_future is not None:
            with suppress(asyncio.CancelledError, Exception):
                await asyncio.shield(asyncio.wrap_future(open_future))

        await self._drain_cancelled_acquires()
        await self._drain_cancelled_acquire_releases()
        future = asyncio.run_coroutine_threadsafe(self._close_pool(), loop)
        await asyncio.wrap_future(future)
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            await asyncio.to_thread(thread.join)

    async def _ensure_open(self) -> None:
        await self._ensure_owner_process()
        while True:
            with self._lock:
                if self._closed:
                    raise RuntimeError("StateDB postgres backend is closed")
                if (self._open_future is not None
                        and self._open_future.done()
                        and self._open_future_failed(self._open_future)):
                    self._open_future = None
                if self._open_future is None:
                    self._start_loop()
                    assert self._loop is not None
                    self._open_future = asyncio.run_coroutine_threadsafe(
                        self._open_pool(),
                        self._loop,
                    )
                open_future = self._open_future
            try:
                await asyncio.shield(asyncio.wrap_future(open_future))
            except asyncio.CancelledError:
                if open_future.cancelled():
                    self._clear_open_future(open_future)
                raise
            except Exception:
                self._clear_open_future(open_future)
                raise
            else:
                return

    async def _ensure_owner_process(self) -> None:
        if self._pid != os.getpid():
            raise RuntimeError("StateDB postgres pool belongs to another process")

    def _open_future_failed(self, open_future: Any) -> bool:
        if open_future.cancelled():
            return True
        try:
            open_future.result()
        except Exception:
            return True
        return False

    def _clear_open_future(self, open_future: Any) -> None:
        with self._lock:
            if self._open_future is open_future:
                self._open_future = None

    def _start_loop(self) -> None:
        if self._loop is not None:
            return
        self._thread = threading.Thread(
            target=self._run_loop,
            name="statedb-postgres-pool",
            daemon=True,
        )
        self._thread.start()
        self._loop_ready.wait()

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._loop_ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    async def _open_pool(self) -> None:
        options: dict[str, Any] = {
            "dsn": self._dsn,
            "min_size": self._min_pool_size,
            "max_size": self._max_pool_size,
        }
        if self._password is not None:
            options["password"] = self._password
        with restricted_diagnostic_boundary(
                safe_message='StateDB PostgreSQL connection failed',
                exception_types=(asyncpg.PostgresError, OSError)):
            self._pool = await asyncpg.create_pool(**options)

    async def _acquire(
        self,
        mode: TxMode,
        state: _PostgresAcquireState,
    ) -> _PostgresConnectionLease:
        if self._pool is None:
            raise RuntimeError("StateDB postgres backend is not open")
        acquire_context = self._pool.acquire()
        state.acquire_context = acquire_context
        with restricted_diagnostic_boundary(
                safe_message='StateDB PostgreSQL acquire failed',
                exception_types=asyncpg.PostgresError):
            connection = await acquire_context.__aenter__()
        state.connection = connection
        transaction_context = connection.transaction(readonly=(mode == "read"))
        try:
            with restricted_diagnostic_boundary(
                    safe_message='StateDB PostgreSQL transaction failed',
                    exception_types=asyncpg.PostgresError):
                await transaction_context.__aenter__()
        except BaseException:
            await acquire_context.__aexit__(None, None, None)
            raise
        return _PostgresConnectionLease(
            connection=connection,
            acquire_context=acquire_context,
            transaction_context=transaction_context,
        )

    async def _release(
        self,
        lease: _PostgresConnectionLease,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        try:
            with restricted_diagnostic_boundary(
                    safe_message='StateDB PostgreSQL transaction failed',
                    exception_types=asyncpg.PostgresError):
                await lease.transaction_context.__aexit__(
                    exc_type, exc, traceback)
        finally:
            with restricted_diagnostic_boundary(
                    safe_message='StateDB PostgreSQL release failed',
                    exception_types=asyncpg.PostgresError):
                await lease.acquire_context.__aexit__(exc_type, exc, traceback)

    async def _close_pool(self) -> None:
        if self._pool is not None:
            with restricted_diagnostic_boundary(
                    safe_message='StateDB PostgreSQL close failed',
                    exception_types=asyncpg.PostgresError):
                await self._pool.close()
            self._pool = None

    def _release_cancelled_acquire(
        self,
        future: Any,
        state: _PostgresAcquireState,
        cancellation: asyncio.CancelledError,
    ) -> None:
        with self._lock:
            self._cancelled_acquire_futures.add(future)

        def cleanup(done_future: Any) -> None:
            with self._lock:
                self._cancelled_acquire_futures.discard(done_future)
            try:
                lease = done_future.result()
            except Exception:
                return
            try:
                release_future = self._submit(
                    self._release(
                        lease,
                        type(cancellation),
                        cancellation,
                        cancellation.__traceback__,
                    ))
            except RuntimeError:
                return
            with self._lock:
                self._cancelled_acquire_releases.add(release_future)
            release_future.add_done_callback(self._discard_cancelled_release)

        future.add_done_callback(cleanup)
        if state.connection is None:
            future.cancel()

    def _discard_cancelled_release(self, release_future: Any) -> None:
        with self._lock:
            self._cancelled_acquire_releases.discard(release_future)

    async def _drain_cancelled_acquires(self) -> None:
        while True:
            with self._lock:
                futures = tuple(self._cancelled_acquire_futures)
            if not futures:
                return
            await asyncio.gather(
                *(asyncio.wrap_future(future) for future in futures),
                return_exceptions=True,
            )

    async def _drain_cancelled_acquire_releases(self) -> None:
        while True:
            with self._lock:
                futures = tuple(self._cancelled_acquire_releases)
            if not futures:
                return
            await asyncio.gather(
                *(asyncio.wrap_future(future) for future in futures),
                return_exceptions=True,
            )

    def _submit(self, awaitable: Any) -> Any:
        loop = self._loop
        if loop is None or loop.is_closed() or not loop.is_running():
            if asyncio.iscoroutine(awaitable):
                awaitable.close()
            raise RuntimeError("StateDB postgres backend is closed")
        return asyncio.run_coroutine_threadsafe(awaitable, loop)

    async def _run(self, awaitable: Any) -> Any:
        future = self._submit(awaitable)
        return await asyncio.wrap_future(future)


class _PostgresConnectionPools:
    """Manage one Postgres pool wrapper per process.

    StateDB objects can be inherited across fork, but asyncpg pool state,
    sockets, and owner threads cannot. The manager keys live pools by PID and
    discards inherited entries when a child process first touches the manager;
    the child then lazily opens its own dedicated-loop pool.
    """

    def __init__(
        self,
        dsn: str,
        *,
        password: Callable[[], Awaitable[str]] | None = None,
        min_pool_size: int,
        max_pool_size: int,
    ) -> None:
        self._dsn = dsn
        self._password = password
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._pools: dict[int, _PostgresConnectionPool] = {}
        self._closed = False

    @classmethod
    async def open(
        cls,
        dsn: str,
        *,
        password: Callable[[], Awaitable[str]] | None = None,
        min_pool_size: int,
        max_pool_size: int,
    ) -> "_PostgresConnectionPools":
        pools = cls(
            dsn,
            password=password,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
        )
        try:
            await pools._get_current_process_pool()
        except BaseException:
            await pools.close()
            raise
        return pools

    @asynccontextmanager
    async def transaction(
        self,
        mode: TxMode,
    ) -> AsyncGenerator[_PostgresStateTx, None]:
        pool = await self._get_current_process_pool()
        lease = await pool.acquire(mode)
        try:
            yield _PostgresStateTx(pool=pool, connection=lease.connection)
        except BaseException as exc:
            await pool.release(lease, type(exc), exc, exc.__traceback__)
            raise
        else:
            await pool.release(lease, None, None, None)

    async def close(self) -> None:
        with self._lock:
            pid = self._discard_inherited_pools_locked()
            self._closed = True
            pool = self._pools.pop(pid, None)
        if pool is not None:
            await pool.close()

    async def _get_current_process_pool(self) -> _PostgresConnectionPool:
        with self._lock:
            pid = self._discard_inherited_pools_locked()
            if self._closed:
                raise RuntimeError("StateDB postgres backend is closed")
            pool = self._pools.get(pid)
            if pool is None:
                pool = _PostgresConnectionPool(
                    self._dsn,
                    password=self._password,
                    min_pool_size=self._min_pool_size,
                    max_pool_size=self._max_pool_size,
                )
                self._pools[pid] = pool
        try:
            await pool.open()
        except Exception:
            with self._lock:
                if self._pools.get(pid) is pool:
                    self._pools.pop(pid, None)
            await pool.close()
            raise
        return pool

    def _discard_inherited_pools_locked(self) -> int:
        pid = os.getpid()
        if self._pid != pid:
            self._pools.clear()
            self._pid = pid
        return pid


class _SqliteConnectionPool:
    def __init__(self, connections: Sequence[aiosqlite.Connection]) -> None:
        if not connections:
            raise ValueError("StateDB sqlite pool must own at least one connection")
        self._available = list(connections)
        self._leased: set[aiosqlite.Connection] = set()
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    async def open(
        cls,
        sqlite_path: str,
        *,
        pool_size: int,
    ) -> "_SqliteConnectionPool":
        if pool_size < 1:
            raise ValueError("sqlite pool_size must be at least 1")
        connections: list[aiosqlite.Connection] = []
        try:
            for _ in range(pool_size):
                connection = await _open_aiosqlite_connection(
                    sqlite_path,
                    isolation_level=None,
                )
                connection.row_factory = aiosqlite.Row
                await _await_aiosqlite(
                    connection.create_function(
                        "md5",
                        1,
                        _sqlite_md5,
                        deterministic=True,
                    ))
                cursor = await _await_aiosqlite(
                    connection.execute("PRAGMA busy_timeout=30000"))
                await _await_aiosqlite(cursor.close())
                connections.append(connection)
        except BaseException:
            for connection in connections:
                await _close_aiosqlite_connection(connection)
            raise
        return cls(connections)

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        while True:
            with self._lock:
                if self._closed:
                    raise RuntimeError("StateDB sqlite backend is closed")
                if self._available:
                    connection = self._available.pop()
                    self._leased.add(connection)
                    break
            await asyncio.sleep(_SQLITE_POOL_WAIT_SECONDS)

        should_close = False
        try:
            yield connection
        finally:
            with self._lock:
                if self._closed:
                    should_close = True
                else:
                    self._leased.discard(connection)
                    self._available.append(connection)
            if should_close:
                try:
                    await _close_aiosqlite_connection(connection)
                finally:
                    with self._lock:
                        self._leased.discard(connection)

    async def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            available = tuple(self._available)
            self._available.clear()

        for connection in available:
            await _close_aiosqlite_connection(connection)

        while True:
            with self._lock:
                leased_count = len(self._leased)
            if leased_count == 0:
                break
            await asyncio.sleep(_SQLITE_POOL_WAIT_SECONDS)


async def _open_aiosqlite_connection(
    sqlite_path: str,
    **kwargs: Any,
) -> aiosqlite.Connection:
    connection_or_awaitable = aiosqlite.connect(sqlite_path, **kwargs)
    if (hasattr(connection_or_awaitable, "start")
            and hasattr(connection_or_awaitable, "_connect")):
        connection = connection_or_awaitable
        # aiosqlite 0.17 wraps this same path in Connection.__await__.
        # Owning it here keeps startup consistent with the pooled lifecycle.
        connection.start()
        return await _await_aiosqlite(connection._connect())
    return await _await_aiosqlite(connection_or_awaitable)


async def _close_aiosqlite_connection(
    connection: aiosqlite.Connection,
) -> None:
    try:
        await _await_aiosqlite(connection.close())
    except ValueError as exc:
        if "Connection closed" not in str(exc):
            raise


@dataclass(frozen=True)
class StateDBSchemaStatus:
    backend: BackendName
    expected_schema_version: int
    stored_schema_version: int | None
    expected_schema_hash: str
    stored_schema_hash: str | None
    expected_table_count: int
    existing_state_table_count: int
    missing_table_names: tuple[str, ...]
    ok: bool
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class StateDB:
    backend: BackendName
    target: str = field(repr=False)
    bootstrap_result: BootstrapResult
    _sqlite_pool: _SqliteConnectionPool | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _postgres_pools: _PostgresConnectionPools | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @classmethod
    async def open_sqlite(
        cls,
        sqlite_path: str,
        *,
        pool_size: int = 4,
    ) -> "StateDB":
        result = await bootstrap_sqlite(sqlite_path)
        sqlite_pool = await _SqliteConnectionPool.open(
            sqlite_path,
            pool_size=pool_size,
        )
        return cls(
            backend="sqlite",
            target=sqlite_path,
            bootstrap_result=result,
            _sqlite_pool=sqlite_pool,
        )

    @classmethod
    async def open_postgres(
        cls,
        dsn: str,
        *,
        password: Callable[[], Awaitable[str]] | None = None,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> "StateDB":
        if password is None:
            result = await bootstrap_postgres(
                dsn,
                migrations=POSTGRES_SCHEMA_MIGRATIONS,
            )
        else:
            result = await bootstrap_postgres(
                dsn,
                password=password,
                migrations=POSTGRES_SCHEMA_MIGRATIONS,
            )
        postgres_pools = await _PostgresConnectionPools.open(
            dsn,
            password=password,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
        )
        return cls(
            backend="postgres",
            target=dsn,
            bootstrap_result=result,
            _postgres_pools=postgres_pools,
        )

    async def get_schema_status(self) -> StateDBSchemaStatus:
        expected_hash = schema_identity(self.backend)
        try:
            async with self.read_tx() as tx:
                if self.backend == "sqlite":
                    table_rows = await tx.fetch(
                        "SELECT name FROM sqlite_master WHERE type = 'table'")
                    existing_tables = {row[0] for row in table_rows}
                    schema_row = await tx.fetchrow(
                        "SELECT payload FROM statedb_schema WHERE id = ?",
                        SCHEMA_ROW_ID,
                    )
                    payload_text = schema_row[0] if schema_row else None
                else:
                    table_rows = await tx.fetch(
                        """
                        SELECT tablename
                        FROM pg_catalog.pg_tables
                        WHERE schemaname = current_schema()
                        """)
                    existing_tables = {row["tablename"] for row in table_rows}
                    schema_row = await tx.fetchrow(
                        """
                        SELECT payload::text AS payload
                        FROM statedb_schema
                        WHERE id = $1
                        """,
                        SCHEMA_ROW_ID,
                    )
                    payload_text = schema_row["payload"] if schema_row else None

            missing_tables = tuple(
                sorted(
                    table for table in STATE_DB_TABLE_NAMES
                    if table not in existing_tables))
            payload = json.loads(payload_text) if payload_text else {}
            stored_hash = payload.get("schema_hash")
            stored_version = payload.get("schema_version")
            if not isinstance(stored_hash, str):
                stored_hash = None
            if type(stored_version) is not int:
                stored_version = None
            ok = (stored_hash == expected_hash and
                  stored_version == SCHEMA_VERSION and not missing_tables)
            return StateDBSchemaStatus(
                backend=self.backend,
                expected_schema_version=SCHEMA_VERSION,
                stored_schema_version=stored_version,
                expected_schema_hash=expected_hash,
                stored_schema_hash=stored_hash,
                expected_table_count=len(STATE_DB_TABLE_NAMES),
                existing_state_table_count=len(
                    set(STATE_DB_TABLE_NAMES).intersection(existing_tables)),
                missing_table_names=missing_tables,
                ok=ok,
            )
        except Exception as exc:
            return StateDBSchemaStatus(
                backend=self.backend,
                expected_schema_version=SCHEMA_VERSION,
                stored_schema_version=None,
                expected_schema_hash=expected_hash,
                stored_schema_hash=None,
                expected_table_count=len(STATE_DB_TABLE_NAMES),
                existing_state_table_count=0,
                missing_table_names=STATE_DB_TABLE_NAMES,
                ok=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    async def is_healthy(self) -> bool:
        return (await self.get_schema_status()).ok

    async def close(self) -> None:
        if self.backend == "sqlite":
            if self._sqlite_pool is not None:
                await self._sqlite_pool.close()
            return
        if self._postgres_pools is not None:
            await self._postgres_pools.close()

    async def __aenter__(self) -> "StateDB":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    @asynccontextmanager
    async def read_tx(self) -> AsyncGenerator[StateTx, None]:
        async with self._tx("read") as tx:
            yield tx

    @asynccontextmanager
    async def write_tx(self) -> AsyncGenerator[StateTx, None]:
        async with self._tx("write") as tx:
            yield tx

    @asynccontextmanager
    async def _tx(self, mode: TxMode) -> AsyncGenerator[StateTx, None]:
        if self.backend == "sqlite":
            async with self._sqlite_tx(mode) as tx:
                yield tx
            return
        async with self._postgres_tx(mode) as tx:
            yield tx

    @asynccontextmanager
    async def _sqlite_tx(self, mode: TxMode) -> AsyncGenerator[StateTx, None]:
        if self._sqlite_pool is None:
            raise RuntimeError("StateDB sqlite backend is not open")
        async with self._sqlite_pool.acquire() as connection:
            # BEGIN stays inside the handler: aiosqlite runs SQLite work on a
            # per-connection thread, so a cancelled BEGIN can still execute.
            # Rolling back on every failure drains it before the connection
            # returns to the pool, so a late BEGIN can't leak the writer lock.
            try:
                cursor = await _await_aiosqlite(
                    connection.execute(
                        "BEGIN IMMEDIATE" if mode == "write" else "BEGIN"))
                await _await_aiosqlite(cursor.close())
                yield _SqliteStateTx(connection)
                if mode == "write":
                    await _await_aiosqlite(connection.commit())
                else:
                    await _await_aiosqlite(connection.rollback())
            except BaseException:
                await _await_aiosqlite(connection.rollback())
                raise

    @asynccontextmanager
    async def _postgres_tx(self, mode: TxMode) -> AsyncGenerator[StateTx, None]:
        if self._postgres_pools is None:
            raise RuntimeError("StateDB postgres backend is not open")
        async with self._postgres_pools.transaction(mode) as tx:
            yield tx
