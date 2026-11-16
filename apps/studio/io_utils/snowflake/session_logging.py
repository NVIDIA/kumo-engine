# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import re
from collections.abc import Generator
from contextlib import contextmanager
from enum import Enum
from time import perf_counter
from typing import Any, Callable

import wrapt
from snowflake.connector.connection import SnowflakeConnection
from snowflake.connector.cursor import SnowflakeCursor
from snowflake.snowpark import Session
from snowflake.snowpark._internal.server_connection import ServerConnection
from snowflake.snowpark.exceptions import SnowparkSessionException

from io_utils import logging
from io_utils.retry import retry

logger = logging.getLogger(__name__)

_LONG_RUNNING_QUERY_THRESHOLD_MS = 10_000
_QUERY_RETRY_TRIES = 3
_QUERY_RETRY_DELAY_SECONDS = 0.5
_QUERY_RETRY_BACKOFF = 1.0
_QUERY_RETRY_JITTER_SECONDS = 0.0


class QueryRetryScope(Enum):
    IDEMPOTENT = "idempotent"
    ALL = "all"


_QUERY_RETRY_SCOPE: QueryRetryScope = QueryRetryScope.IDEMPOTENT

_IDEMPOTENT_SQL_PREFIXES = frozenset({
    "SELECT",
    "SHOW",
    "DESCRIBE",
    "DESC",
    "LIST",
    "EXPLAIN",
})
_RETRYABLE_SQLSTATE_PREFIXES = ("08", "40")
_RETRYABLE_ERRNOS = frozenset({
    250001,  # failed to connect to DB
    250002,  # failed to get response
    250003,  # failed to get response (retryable transport issue)
    250006,  # failed to connect after retry
    390111,  # session no longer exists
})
_RETRYABLE_MESSAGE_PATTERNS = (
    "session no longer exists",
    "session has been closed",
    "session was closed",
    "connection reset",
    "connection aborted",
    "connection refused",
    "broken pipe",
    "temporarily unavailable",
    "transient",
    "timed out",
    "timeout",
)
_NON_RETRYABLE_MESSAGE_PATTERNS = (
    "syntax error",
    "compilation error",
    "insufficient privileges",
    "not authorized",
    "does not exist",
    "invalid identifier",
)
_MULTI_STATEMENT_ERROR_PATTERNS = (
    "actual statement count",
    "execution of multiple statements failed on statement",
)
_SQL_PREFIX_RE = re.compile(r"^\s*([A-Za-z]+)")
_SQLSTATE_RE = re.compile(r"^[0-9A-Z]{5}$")
_QUERY_ID_RE = re.compile(
    r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$")

# These wrappers intentionally use `wrapt.ObjectProxy` instead of direct
# subclassing of Snowflake/Snowpark runtime classes. Those objects are
# constructed internally with private state and lifecycle behavior that we do
# not control. Proxying lets us preserve runtime compatibility while injecting
# logging and wiring behavior across cursor/connection/server/session layers.


def _sql_prefix(command: str) -> str:
    match = _SQL_PREFIX_RE.match(command)
    if match is None:
        return ""
    return match.group(1).upper()


def _is_retry_scope_allowed(command: str) -> bool:
    if _QUERY_RETRY_SCOPE == QueryRetryScope.ALL:
        return True
    return _sql_prefix(command) in _IDEMPOTENT_SQL_PREFIXES


def _is_multi_statement_error(error: BaseException) -> bool:
    message = str(error).lower()
    return any(pattern in message
               for pattern in _MULTI_STATEMENT_ERROR_PATTERNS)


def _extract_sqlstate(error: BaseException) -> str | None:
    sqlstate = getattr(error, "sqlstate", None)
    if isinstance(sqlstate, str) and _SQLSTATE_RE.fullmatch(sqlstate):
        return sqlstate
    return None


def safe_query_id_for_logging(value: object) -> str:
    if isinstance(value, str) and _QUERY_ID_RE.fullmatch(value):
        return value
    return "unknown"


def _extract_errno(error: BaseException) -> int | None:
    errno = getattr(error, "errno", None)
    if isinstance(errno, int):
        return errno
    try:
        return int(errno) if errno is not None else None
    except (TypeError, ValueError):
        return None


def _safe_error_fields(error: BaseException) -> tuple[str, str, str]:
    """Return allowlisted Snowflake failure fields for structured logging."""
    sqlstate = _extract_sqlstate(error) or "unknown"
    errno = _extract_errno(error)
    return type(error).__name__, sqlstate, str(errno or "unknown")


def _is_retryable_error(error: BaseException) -> bool:
    message = str(error).lower()
    if any(pattern in message for pattern in _NON_RETRYABLE_MESSAGE_PATTERNS):
        return False

    errno = _extract_errno(error)
    if errno in _RETRYABLE_ERRNOS:
        return True

    sqlstate = _extract_sqlstate(error)
    if sqlstate and sqlstate.startswith(_RETRYABLE_SQLSTATE_PREFIXES):
        return True

    if isinstance(error, (SnowparkSessionException, TimeoutError,
                          ConnectionError, BrokenPipeError, BlockingIOError)):
        return True

    return any(pattern in message for pattern in _RETRYABLE_MESSAGE_PATTERNS)


def retry_policy(handler: Callable[[Exception], None]):
    def _retry_handler(error: Exception) -> None:
        if not _is_retryable_error(error):
            raise error
        handler(error)

    return retry(
        exceptions=Exception,
        callbacks={lambda _error: True: _retry_handler},
        tries=_QUERY_RETRY_TRIES,
        delay=_QUERY_RETRY_DELAY_SECONDS,
        backoff=_QUERY_RETRY_BACKOFF,
        jitter=_QUERY_RETRY_JITTER_SECONDS,
        logger=logger,
    )


class _LoggedSnowflakeCursor(wrapt.ObjectProxy):
    """Proxy around `SnowflakeCursor` that adds query/fetch logging."""
    def __enter__(self) -> "_LoggedSnowflakeCursor":
        entered = self.__wrapped__.__enter__()
        return ensure_logged_connector_cursor(entered)

    def execute(self, command: str, *args, **kwargs):
        start = perf_counter()
        logger.debug(
            "snowflake operation=%s status=start",
            "cursor.execute",
        )
        try:
            result = self._execute_with_retry(command, args, kwargs)
            duration_ms = int((perf_counter() - start) * 1000)
            logger.info(
                ("snowflake operation=%s status=success"
                 " duration_ms=%d query_id=%s"),
                "cursor.execute",
                duration_ms,
                safe_query_id_for_logging(self.__wrapped__.sfqid),
            )
            self._log_long_running_query_report(
                query_id=self.__wrapped__.sfqid,
                duration_ms=duration_ms,
            )
            return result
        except Exception as error:
            error_type, sqlstate, errno = _safe_error_fields(error)
            logger.error(
                ("snowflake operation=%s status=error duration_ms=%d "
                 "error_type=%s sqlstate=%s errno=%s query_id=%s"),
                "cursor.execute",
                int((perf_counter() - start) * 1000),
                error_type,
                sqlstate,
                errno,
                safe_query_id_for_logging(self.__wrapped__.sfqid),
            )
            raise

    def _log_long_running_query_report(
        self,
        query_id: str | None,
        duration_ms: int,
    ) -> None:
        """Log operator stats report for long-running queries."""
        safe_query_id = safe_query_id_for_logging(query_id)
        if (safe_query_id == "unknown"
                or duration_ms < _LONG_RUNNING_QUERY_THRESHOLD_MS):
            return

        stats_sql = ("SELECT * FROM TABLE(GET_QUERY_OPERATOR_STATS("
                     f"'{safe_query_id}'))")
        stats_query_id = None
        try:
            # Use a separate cursor so we do not mutate the original cursor's
            # result-set metadata (description/rows) for the caller query.
            with self.__wrapped__.connection.cursor() as stats_cursor:
                stats_cursor.execute(stats_sql, timeout=5)
                rows = stats_cursor.fetchall()
                stats_query_id = stats_cursor.sfqid
            logger.info(
                ("snowflake operation=%s status=success query_id=%s "
                 "duration_ms=%d operator_count=%d"),
                "query_operator_stats",
                safe_query_id,
                duration_ms,
                len(rows),
            )
        except Exception as error:
            error_type, sqlstate, errno = _safe_error_fields(error)
            logger.error(
                ("snowflake operation=%s status=error query_id=%s "
                 "error_type=%s sqlstate=%s errno=%s"),
                "query_operator_stats",
                safe_query_id_for_logging(stats_query_id),
                error_type,
                sqlstate,
                errno,
            )
            return

    def _refresh_cursor_for_retry(self) -> None:
        connection = getattr(self.__wrapped__, "connection", None)
        if connection is None:
            return
        try:
            self.__wrapped__.close()
        except Exception:
            pass
        try:
            self.__wrapped__ = connection.cursor(
                cursor_class=type(self.__wrapped__))
        except Exception:
            self.__wrapped__ = connection.cursor()

    def _execute_with_retry(self, command: str, args: tuple[Any, ...],
                            kwargs: dict[str, Any]):
        if not _is_retry_scope_allowed(command):
            return self.__wrapped__.execute(command, *args, **kwargs)

        def _handle_retry(error: Exception) -> None:
            if _is_multi_statement_error(error):
                logger.info(
                    "snowflake operation=%s status=skip_retry reason=%s",
                    "cursor.execute",
                    "multi_statement",
                )
                raise error
            self._refresh_cursor_for_retry()

        @retry_policy(_handle_retry)
        def _run_execute():
            return self.__wrapped__.execute(command, *args, **kwargs)

        return _run_execute()

    def fetchone(self):
        with _log_connector_call("cursor.fetchone"):
            return self.__wrapped__.fetchone()

    def fetchmany(self, size: int | None = None):
        with _log_connector_call("cursor.fetchmany"):
            return self.__wrapped__.fetchmany(size=size)

    def fetchall(self):
        with _log_connector_call("cursor.fetchall"):
            return self.__wrapped__.fetchall()


class _LoggedSnowflakeConnection(wrapt.ObjectProxy):
    """Proxy around `SnowflakeConnection` that returns logged cursors."""
    def cursor(
        self,
        cursor_class: type[SnowflakeCursor] = SnowflakeCursor,
    ) -> _LoggedSnowflakeCursor:
        cursor = self.__wrapped__.cursor(cursor_class=cursor_class)
        return ensure_logged_connector_cursor(cursor)

    def commit(self) -> None:
        with _log_connector_call("connection.commit"):
            self.__wrapped__.commit()

    def rollback(self) -> None:
        with _log_connector_call("connection.rollback"):
            self.__wrapped__.rollback()

    def close(self, retry: bool = True) -> None:
        with _log_connector_call("connection.close"):
            self.__wrapped__.close(retry=retry)


class _LoggedServerConnection(wrapt.ObjectProxy):
    """Proxy for `ServerConnection` that wires logged connector internals."""
    def __init__(self, wrapped: ServerConnection):
        super().__init__(wrapped)
        self.__wrapped__._conn = ensure_logged_connector_connection(
            self.__wrapped__._conn)
        # Newer snowflake-connector versions expose ``_cursor`` as a read-only
        # thread-local property derived lazily from ``self._conn.cursor()``.
        # Because ``_conn`` is now a logged connection (whose ``cursor()``
        # returns a logged cursor), ``_cursor`` already yields a logged cursor
        # and the property cannot be assigned. Only seed it directly on older
        # versions where ``_cursor`` is a writable instance attribute.
        if not isinstance(
                inspect.getattr_static(type(self.__wrapped__), '_cursor',
                                       None), property):
            self.__wrapped__._cursor = ensure_logged_connector_cursor(
                self.__wrapped__._cursor)


class _LoggedSnowparkSession(wrapt.ObjectProxy):
    """Session-level proxy that routes Snowpark execution through wrappers."""
    def __init__(self, wrapped: Session):
        super().__init__(wrapped)
        if isinstance(self.__wrapped__._conn, ServerConnection):
            self.__wrapped__._conn = ensure_logged_server_connection(
                self.__wrapped__._conn)

    @property
    def connection(
            self) -> _LoggedSnowflakeConnection:  # type: ignore[override]
        return ensure_logged_connector_connection(self.__wrapped__.connection)


@contextmanager
def _log_connector_call(operation: str) -> Generator[None]:
    start = perf_counter()
    try:
        yield
        logger.debug(
            "snowflake operation=%s status=success duration_ms=%d",
            operation,
            int((perf_counter() - start) * 1000),
        )
    except Exception as error:
        error_type, sqlstate, errno = _safe_error_fields(error)
        logger.error(
            ("snowflake operation=%s status=error duration_ms=%d "
             "error_type=%s sqlstate=%s errno=%s"),
            operation,
            int((perf_counter() - start) * 1000),
            error_type,
            sqlstate,
            errno,
        )
        raise


def ensure_logged_connector_cursor(
    cursor: SnowflakeCursor | _LoggedSnowflakeCursor,
) -> _LoggedSnowflakeCursor:
    if isinstance(cursor, _LoggedSnowflakeCursor):
        return cursor
    return _LoggedSnowflakeCursor(cursor)


def ensure_logged_connector_connection(
    connection: SnowflakeConnection | _LoggedSnowflakeConnection,
) -> _LoggedSnowflakeConnection:
    if isinstance(connection, _LoggedSnowflakeConnection):
        return connection
    return _LoggedSnowflakeConnection(connection)


def ensure_logged_server_connection(
    connection: ServerConnection | _LoggedServerConnection,
) -> _LoggedServerConnection:
    if isinstance(connection, _LoggedServerConnection):
        return connection
    return _LoggedServerConnection(connection)


def ensure_logged_snowpark_session(
    session: Session | _LoggedSnowparkSession, ) -> Session:
    if isinstance(session, _LoggedSnowparkSession):
        return session  # type: ignore[return-value]
    return _LoggedSnowparkSession(session)  # type: ignore[return-value]
