# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ssl

import pytest

from service_framework.postgres_metrics import (
    POSTGRES_DB_STATE,
    POSTGRES_OPERATION_EXECUTE,
    PostgresMetricErrorType,
    classify_postgres_error,
    observe_postgres_duration,
)


class _SqlstateError(Exception):
    def __init__(self, sqlstate: str) -> None:
        super().__init__(sqlstate)
        self.sqlstate = sqlstate


class _PgcodeError(Exception):
    def __init__(self, pgcode: str) -> None:
        super().__init__(pgcode)
        self.pgcode = pgcode


class _WrappedError(Exception):
    def __init__(self, orig: BaseException) -> None:
        super().__init__(str(orig))
        self.orig = orig


@pytest.mark.parametrize(("exc", "expected"), [
    (_SqlstateError("08006"), PostgresMetricErrorType.CONNECTION.value),
    (_SqlstateError("22001"), PostgresMetricErrorType.DATA.value),
    (_SqlstateError("23505"), PostgresMetricErrorType.INTEGRITY.value),
    (_SqlstateError("25006"), PostgresMetricErrorType.TRANSACTION.value),
    (_SqlstateError("25P03"), PostgresMetricErrorType.TIMEOUT.value),
    (_SqlstateError("28P01"), PostgresMetricErrorType.AUTHENTICATION.value),
    (_SqlstateError("3D000"), PostgresMetricErrorType.DATABASE.value),
    (_SqlstateError("40001"), PostgresMetricErrorType.TRANSACTION.value),
    (_SqlstateError("42501"), PostgresMetricErrorType.AUTHORIZATION.value),
    (_SqlstateError("42601"), PostgresMetricErrorType.PROGRAMMING.value),
    (_SqlstateError("53300"), PostgresMetricErrorType.CONNECTION.value),
    (_SqlstateError("55P03"), PostgresMetricErrorType.TIMEOUT.value),
    (_SqlstateError("57014"), PostgresMetricErrorType.QUERY_CANCELED.value),
    (_SqlstateError("57P03"), PostgresMetricErrorType.CONNECTION.value),
    (_SqlstateError("57P05"), PostgresMetricErrorType.TIMEOUT.value),
    (_SqlstateError("58000"), PostgresMetricErrorType.DATABASE.value),
    (_PgcodeError("23503"), PostgresMetricErrorType.INTEGRITY.value),
])
def test_classify_postgres_error_uses_sqlstate(exc, expected):
    assert classify_postgres_error(exc) == expected


@pytest.mark.parametrize(("exc", "expected"), [
    (ssl.SSLError("bad certificate"), PostgresMetricErrorType.TLS.value),
    (TimeoutError("timed out"), PostgresMetricErrorType.TIMEOUT.value),
    (PermissionError("permission denied"),
     PostgresMetricErrorType.AUTHORIZATION.value),
    (OSError("connection refused"), PostgresMetricErrorType.CONNECTION.value),
])
def test_classify_postgres_error_handles_client_side_errors(exc, expected):
    assert classify_postgres_error(exc) == expected


def test_classify_postgres_error_prefers_wrapped_sqlstate():
    wrapped = _WrappedError(_SqlstateError("28P01"))
    assert classify_postgres_error(
        wrapped) == PostgresMetricErrorType.AUTHENTICATION.value


def test_observe_postgres_duration_rejects_unknown_labels():
    with pytest.raises(ValueError, match="operation"):
        observe_postgres_duration(POSTGRES_DB_STATE, "select_user", 1.0)
    with pytest.raises(ValueError, match="db"):
        observe_postgres_duration("tenant_db", POSTGRES_OPERATION_EXECUTE, 1.0)
