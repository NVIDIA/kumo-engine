# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prometheus metrics for Kumo-owned Postgres interactions."""
import time
from contextlib import contextmanager
from enum import Enum
from typing import Iterator

from prometheus_client import Counter, Histogram


class PostgresMetricDb(str, Enum):
    STATE = 'state'


class PostgresMetricOperation(str, Enum):
    CONNECT = 'connect'
    ENSURE_TABLE = 'ensure_table'
    GET = 'get'
    ITEMS = 'items'
    SET = 'set'
    DELETE = 'delete'
    EXECUTE = 'execute'


class PostgresMetricErrorType(str, Enum):
    AUTHENTICATION = 'authentication'
    AUTHORIZATION = 'authorization'
    CONNECTION = 'connection'
    DATA = 'data'
    DATABASE = 'database'
    INTEGRITY = 'integrity'
    INTERFACE = 'interface'
    PROGRAMMING = 'programming'
    QUERY_CANCELED = 'query_canceled'
    RESOURCE = 'resource'
    TIMEOUT = 'timeout'
    TLS = 'tls'
    TRANSACTION = 'transaction'
    UNKNOWN = 'unknown'


POSTGRES_DB_STATE = PostgresMetricDb.STATE.value

POSTGRES_OPERATION_CONNECT = PostgresMetricOperation.CONNECT.value
POSTGRES_OPERATION_ENSURE_TABLE = PostgresMetricOperation.ENSURE_TABLE.value
POSTGRES_OPERATION_GET = PostgresMetricOperation.GET.value
POSTGRES_OPERATION_ITEMS = PostgresMetricOperation.ITEMS.value
POSTGRES_OPERATION_SET = PostgresMetricOperation.SET.value
POSTGRES_OPERATION_DELETE = PostgresMetricOperation.DELETE.value
POSTGRES_OPERATION_EXECUTE = PostgresMetricOperation.EXECUTE.value

_VALID_DBS = frozenset(item.value for item in PostgresMetricDb)
_VALID_OPERATIONS = frozenset(item.value for item in PostgresMetricOperation)
_ERROR = PostgresMetricErrorType

_SQLSTATE_EXACT = {
    # insufficient_privilege
    '42501': _ERROR.AUTHORIZATION.value,
    # query_canceled
    '57014': _ERROR.QUERY_CANCELED.value,
    # too_many_connections
    '53300': _ERROR.CONNECTION.value,
    # idle_in_transaction_session_timeout
    '25P03': _ERROR.TIMEOUT.value,
    # lock_not_available
    '55P03': _ERROR.TIMEOUT.value,
    # admin_shutdown, crash_shutdown, cannot_connect_now
    '57P01': _ERROR.CONNECTION.value,
    '57P02': _ERROR.CONNECTION.value,
    '57P03': _ERROR.CONNECTION.value,
    # idle_session_timeout
    '57P05': _ERROR.TIMEOUT.value,
}
# Exception class names (lowercased) → metric error type. Lookup walks the
# exception's MRO, so registering a base class also classifies all subclasses
# (e.g. asyncpg UniqueViolationError → IntegrityConstraintViolationError).
_CLIENT_EXCEPTION_NAMES = {
    # stdlib
    'sslerror': _ERROR.TLS.value,
    'timeouterror': _ERROR.TIMEOUT.value,
    'permissionerror': _ERROR.AUTHORIZATION.value,
    'connectionerror': _ERROR.CONNECTION.value,
    'brokenpipeerror': _ERROR.CONNECTION.value,
    'oserror': _ERROR.CONNECTION.value,
    # DBAPI bases (psycopg2, psycopg3, SQLAlchemy share these names)
    'interfaceerror': _ERROR.INTERFACE.value,
    'databaseerror': _ERROR.DATABASE.value,
    'internalerror': _ERROR.DATABASE.value,
    'dataerror': _ERROR.DATA.value,
    'operationalerror': _ERROR.CONNECTION.value,
    'integrityerror': _ERROR.INTEGRITY.value,
    'programmingerror': _ERROR.PROGRAMMING.value,
    'notsupportederror': _ERROR.PROGRAMMING.value,
    'disconnectionerror': _ERROR.CONNECTION.value,
    'invalidcachedstatementerror': _ERROR.PROGRAMMING.value,
    # asyncpg.exceptions
    'connectiondoesnotexisterror': _ERROR.CONNECTION.value,
    'connectionfailureerror': _ERROR.CONNECTION.value,
    'toomanyconnectionserror': _ERROR.CONNECTION.value,
    'cannotconnectnowerror': _ERROR.CONNECTION.value,
    'querycancelederror': _ERROR.QUERY_CANCELED.value,
    'integrityconstraintviolationerror': _ERROR.INTEGRITY.value,
    'invalidauthorizationspecificationerror': _ERROR.AUTHENTICATION.value,
    'invalidpassworderror': _ERROR.AUTHENTICATION.value,
    'insufficientprivilegeerror': _ERROR.AUTHORIZATION.value,
    'transactionrollbackerror': _ERROR.TRANSACTION.value,
    'insufficientresourceserror': _ERROR.RESOURCE.value,
}

_SQLSTATE_CLASS = {
    '08': _ERROR.CONNECTION.value,
    '22': _ERROR.DATA.value,
    '23': _ERROR.INTEGRITY.value,
    '25': _ERROR.TRANSACTION.value,
    '28': _ERROR.AUTHENTICATION.value,
    '3D': _ERROR.DATABASE.value,
    '3F': _ERROR.DATABASE.value,
    '40': _ERROR.TRANSACTION.value,
    '42': _ERROR.PROGRAMMING.value,
    '53': _ERROR.RESOURCE.value,
    '57': _ERROR.RESOURCE.value,
    '58': _ERROR.DATABASE.value,
}

POSTGRES_OPERATION_DURATION = Histogram(
    'kumo_postgres_operation_duration_seconds',
    'Wall time for Kumo Postgres operations.',
    ['db', 'operation'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
POSTGRES_OPERATION_ERRORS = Counter(
    'kumo_postgres_operation_errors',
    'Kumo Postgres operation failures.',
    ['db', 'operation', 'error_type'],
)


def observe_postgres_duration(db: str, operation: str,
                              duration_seconds: float) -> None:
    _validate_labels(db, operation)
    POSTGRES_OPERATION_DURATION.labels(
        db=db, operation=operation).observe(duration_seconds)


def record_postgres_error(db: str, operation: str, exc: BaseException) -> None:
    _validate_labels(db, operation)
    POSTGRES_OPERATION_ERRORS.labels(
        db=db,
        operation=operation,
        error_type=classify_postgres_error(exc),
    ).inc()


@contextmanager
def observe_postgres_operation(db: str, operation: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        record_postgres_error(db, operation, exc)
        raise
    finally:
        observe_postgres_duration(db, operation, time.perf_counter() - start)


def classify_postgres_error(exc: BaseException) -> str:
    # Some Postgres clients wrap the driver error under .orig/cause/context.
    # Walk the chain so both direct asyncpg and SQLAlchemy-style errors are
    # classified consistently.
    chain = _exception_chain(exc)
    for current in chain:
        category = _classify_sqlstate(current)
        if category != PostgresMetricErrorType.UNKNOWN.value:
            return category
    for current in chain:
        category = _classify_client_exception(current)
        if category != PostgresMetricErrorType.UNKNOWN.value:
            return category
    return PostgresMetricErrorType.UNKNOWN.value


def _classify_sqlstate(exc: BaseException) -> str:
    sqlstate = getattr(exc, 'sqlstate', None) or getattr(exc, 'pgcode', None)
    if not isinstance(sqlstate, str):
        return PostgresMetricErrorType.UNKNOWN.value
    if sqlstate in _SQLSTATE_EXACT:
        return _SQLSTATE_EXACT[sqlstate]
    return _SQLSTATE_CLASS.get(sqlstate[:2],
                               PostgresMetricErrorType.UNKNOWN.value)


def _classify_client_exception(exc: BaseException) -> str:
    for cls in type(exc).__mro__:
        category = _CLIENT_EXCEPTION_NAMES.get(cls.__name__.lower())
        if category is not None:
            return category
    return PostgresMetricErrorType.UNKNOWN.value


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = _next_wrapped_exception(current)
    return chain


def _next_wrapped_exception(exc: BaseException) -> BaseException | None:
    for attr in ('orig', '__cause__', '__context__'):
        wrapped = getattr(exc, attr, None)
        if isinstance(wrapped, BaseException):
            return wrapped
    return None


def _validate_labels(db: str, operation: str) -> None:
    if db not in _VALID_DBS:
        raise ValueError(f'Invalid postgres metric db label: {db!r}')
    if operation not in _VALID_OPERATIONS:
        raise ValueError(
            f'Invalid postgres metric operation label: {operation!r}')
