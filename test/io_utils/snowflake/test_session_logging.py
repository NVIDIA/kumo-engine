# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

import pytest
from snowflake.snowpark import Session

from io_utils import logging
from io_utils.snowflake import session_logging
from io_utils.snowflake.session_logging import (
    _LoggedSnowflakeConnection,
    _LoggedSnowflakeCursor,
    ensure_logged_connector_connection,
    ensure_logged_snowpark_session,
)
from kumo.testing.snowflake import setup_full_snowflake_session


def test_logged_connector_cursor_execute_logs_and_query_id(tempdir, caplog):
    session = setup_full_snowflake_session(tempdir)
    wrapped = ensure_logged_snowpark_session(session)

    caplog.set_level(logging.DEBUG)
    caplog.clear()

    with wrapped.connection.cursor() as cur:
        result = cur.execute("SELECT 1")

    assert result is not None
    assert "operation=cursor.execute status=start" in caplog.text
    assert "operation=cursor.execute status=success" in caplog.text
    assert "query_id=" in caplog.text


def test_logged_connector_cursor_execute_logs_error(tempdir, caplog):
    session = setup_full_snowflake_session(tempdir)
    wrapped = ensure_logged_snowpark_session(session)

    caplog.set_level(logging.DEBUG)
    caplog.clear()

    with wrapped.connection.cursor() as cur:
        with pytest.raises(Exception):
            cur.execute("SELECT * FROM definitely_missing_table")

    assert "operation=cursor.execute status=error" in caplog.text
    assert "SELECT * FROM definitely_missing_table" not in caplog.text


def test_long_running_query_operator_stats_report_uses_threshold_override(
        tempdir, caplog, monkeypatch):
    session = setup_full_snowflake_session(tempdir)
    wrapped = ensure_logged_snowpark_session(session)
    monkeypatch.setattr(
        session_logging,
        "_LONG_RUNNING_QUERY_THRESHOLD_MS",
        0,
    )

    caplog.set_level(logging.INFO)
    caplog.clear()

    with wrapped.connection.cursor() as cur:
        cur.execute("SELECT 1")

    assert any("operation=query_operator_stats status=success" in rec.message
               for rec in caplog.records)
    assert any("operator_count=" in rec.message for rec in caplog.records)


def test_logged_connector_connection_wraps_commit_rollback_close(
        tempdir, caplog):
    session = setup_full_snowflake_session(tempdir)
    wrapped = ensure_logged_snowpark_session(session)
    conn = ensure_logged_connector_connection(wrapped.connection)

    caplog.set_level(logging.DEBUG)
    caplog.clear()

    conn.commit()
    conn.rollback()
    conn.close()

    assert "operation=connection.commit status=success" in caplog.text
    assert "operation=connection.rollback status=success" in caplog.text
    assert "operation=connection.close status=success" in caplog.text


def test_ensure_logged_connector_connection_idempotent(tempdir):
    session = setup_full_snowflake_session(tempdir)
    wrapped_session = ensure_logged_snowpark_session(session)
    conn = wrapped_session.connection
    wrapped = ensure_logged_connector_connection(conn)
    wrapped_again = ensure_logged_connector_connection(wrapped)
    assert wrapped is wrapped_again


def test_logged_snowpark_session_connection_is_wrapped(tempdir):
    session = setup_full_snowflake_session(tempdir)
    wrapped = ensure_logged_snowpark_session(session)

    conn_1 = wrapped.connection
    conn_2 = wrapped.connection

    assert isinstance(conn_1, _LoggedSnowflakeConnection)
    assert isinstance(conn_2, _LoggedSnowflakeConnection)


def test_wrapped_session_compatible_with_snowpark_dataframe(tempdir):
    session = setup_full_snowflake_session(tempdir)
    wrapped = ensure_logged_snowpark_session(session)
    assert isinstance(wrapped, Session)

    df = wrapped.sql("SELECT 1 AS A")
    rows = df.collect()

    assert rows[0][0] == 1
    assert isinstance(df.session, Session)


def test_wrapped_session_logs_query_id_for_dataframe_collect(tempdir, caplog):
    session = setup_full_snowflake_session(tempdir)
    wrapped = ensure_logged_snowpark_session(session)

    caplog.set_level(logging.INFO)
    caplog.clear()

    _ = wrapped.sql("SELECT 1 AS A").collect()
    assert any("query_id=" in rec.message for rec in caplog.records)


def test_wrapped_session_file_put_logs_cursor_execute(tempdir, tmp_path,
                                                      caplog):
    session = setup_full_snowflake_session(tempdir)
    wrapped = ensure_logged_snowpark_session(session)

    local_file = tmp_path / "session_logging_put_test.txt"
    local_file.write_text("hello world", encoding="utf-8")

    caplog.set_level(logging.DEBUG)
    caplog.clear()

    wrapped.file.put(str(local_file), "@~/session_logging_test")
    assert any("query: [PUT" in rec.message for rec in caplog.records)


class _FakeTransientError(Exception):
    errno = 390111
    sqlstate = "08001"


class _FakeFatalError(Exception):
    pass


class _RetryConnection:
    def __init__(self):
        self.cursor_refreshes = 0

    def cursor(self, cursor_class=None):
        self.cursor_refreshes += 1
        return _RetryCursor(self, fail_count=0)


class _RetryCursor:
    def __init__(
        self,
        connection: _RetryConnection,
        *,
        fail_count: int,
        error_factory=None,
    ):
        self.connection = connection
        self._fail_count = fail_count
        self._error_factory = error_factory or _FakeTransientError
        self.calls = 0
        self.closed = False
        self.sfqid = "01234567-89ab-cdef-0123-456789abcdef"

    def execute(self, _command, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        if self.calls <= self._fail_count:
            raise self._error_factory("Session no longer exists")
        return "ok"

    def close(self):
        self.closed = True


def test_logged_cursor_retries_retryable_select(monkeypatch):
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_TRIES", 3)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_BACKOFF", 1)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_JITTER_SECONDS", 0)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_SCOPE",
                        session_logging.QueryRetryScope.IDEMPOTENT)

    connection = _RetryConnection()
    cursor = _RetryCursor(connection, fail_count=1)
    logged_cursor = _LoggedSnowflakeCursor(cursor)

    result = logged_cursor.execute("SELECT 1")

    assert result == "ok"
    assert cursor.calls == 1
    assert connection.cursor_refreshes == 1


def test_logged_cursor_redacts_provider_error_and_inline_credentials(
        monkeypatch, caplog):
    provider_body = 'snowflake-provider-body secret=driver-canary'
    credentials = {
        'access': 'AKIA_CANARY_ACCESS',
        'secret': 'snowflake-secret-canary',
        'token': 'snowflake-session-token-canary',
    }
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_TRIES", 2)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_BACKOFF", 1)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_JITTER_SECONDS", 0)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_SCOPE",
                        session_logging.QueryRetryScope.ALL)

    connection = _RetryConnection()
    cursor = _RetryCursor(
        connection,
        fail_count=2,
        error_factory=lambda _msg: _FakeTransientError(provider_body),
    )
    command = (
        "COPY INTO 's3://internal/output/' FROM source "
        "CREDENTIALS = ("
        f"AWS_KEY_ID = '{credentials['access']}' "
        f"AWS_SECRET_KEY = '{credentials['secret']}' "
        f"AWS_TOKEN = '{credentials['token']}')")

    caplog.set_level(logging.DEBUG)
    logged_cursor = _LoggedSnowflakeCursor(cursor)
    # Keep the same failing fake across retries. Cursor replacement is tested
    # separately; this test is about every log sink on exhausted retries.
    monkeypatch.setattr(logged_cursor, '_refresh_cursor_for_retry', lambda: None)
    with pytest.raises(_FakeTransientError):
        logged_cursor.execute(command)

    assert provider_body not in caplog.text
    for secret in credentials.values():
        assert secret not in caplog.text
    assert command not in caplog.text
    assert 'COPY INTO' not in caplog.text
    assert 'error_type=_FakeTransientError' in caplog.text
    assert 'sqlstate=08001' in caplog.text
    assert 'errno=390111' in caplog.text
    assert 'Traceback' not in caplog.text


def test_logged_cursor_rejects_poisoned_driver_fields(monkeypatch, caplog):
    poisoned_sqlstate = 'sqlstate-secret-canary\nforged-log-line'
    poisoned_query_id = 'query-id-secret-canary\nforged-log-line'

    def error_factory(_message):
        error = _FakeTransientError('provider body')
        error.sqlstate = poisoned_sqlstate
        return error

    monkeypatch.setattr(session_logging, "_QUERY_RETRY_TRIES", 1)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_SCOPE",
                        session_logging.QueryRetryScope.ALL)
    cursor = _RetryCursor(
        _RetryConnection(),
        fail_count=1,
        error_factory=error_factory,
    )
    cursor.sfqid = poisoned_query_id

    caplog.set_level(logging.DEBUG)
    with pytest.raises(_FakeTransientError):
        _LoggedSnowflakeCursor(cursor).execute('SELECT 1')

    assert poisoned_sqlstate not in caplog.text
    assert poisoned_query_id not in caplog.text
    assert 'secret-canary' not in caplog.text
    assert 'sqlstate=unknown' in caplog.text
    assert 'query_id=unknown' in caplog.text


def test_operator_stats_log_omits_provider_rows(caplog):
    canary = 'operator expression password=stats-secret-canary'
    query_id = '01234567-89ab-cdef-0123-456789abcdef'

    class StatsCursor:
        sfqid = query_id
        description = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, *_args, **_kwargs):
            return self

        def fetchall(self):
            return [(canary, )]

    class StatsConnection:
        def cursor(self):
            return StatsCursor()

    wrapped = type('Cursor', (), {
        'connection': StatsConnection(),
        'sfqid': query_id,
    })()
    caplog.set_level(logging.INFO)

    _LoggedSnowflakeCursor(wrapped)._log_long_running_query_report(
        query_id=query_id,
        duration_ms=10_001,
    )

    assert 'operator_count=1' in caplog.text
    assert canary not in caplog.text
    assert 'stats-secret-canary' not in caplog.text


def test_logged_cursor_ineligible_for_retry_after_multi_statement(
        tempdir, monkeypatch, caplog):
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_TRIES", 3)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_BACKOFF", 1)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_JITTER_SECONDS", 0)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_SCOPE",
                        session_logging.QueryRetryScope.IDEMPOTENT)
    monkeypatch.setattr(session_logging, "_is_retryable_error",
                        lambda _error: True)
    caplog.set_level(logging.INFO)
    caplog.clear()

    session = setup_full_snowflake_session(tempdir)
    wrapped = ensure_logged_snowpark_session(session)

    with wrapped.connection.cursor() as cur:
        logged_cur = cast(_LoggedSnowflakeCursor, cur)

        refresh_calls = 0
        original_refresh = logged_cur._refresh_cursor_for_retry

        def _count_refresh() -> None:
            nonlocal refresh_calls
            refresh_calls += 1
            original_refresh()

        monkeypatch.setattr(logged_cur, "_refresh_cursor_for_retry",
                            _count_refresh)

        with pytest.raises(Exception):
            logged_cur.execute(
                "SELECT 1; SELECT * FROM definitely_missing_table")

        # Retry should be disabled because the failing request is multi-statement.
        assert refresh_calls == 0
        assert any("status=skip_retry" in rec.message
                   and "reason=multi_statement" in rec.message
                   for rec in caplog.records)


def test_logged_cursor_does_not_retry_non_retryable_error(monkeypatch):
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_TRIES", 3)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_BACKOFF", 1)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_JITTER_SECONDS", 0)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_SCOPE",
                        session_logging.QueryRetryScope.IDEMPOTENT)

    connection = _RetryConnection()
    cursor = _RetryCursor(
        connection,
        fail_count=1,
        error_factory=lambda msg: _FakeFatalError(f"syntax error: {msg}"),
    )
    logged_cursor = _LoggedSnowflakeCursor(cursor)

    with pytest.raises(_FakeFatalError):
        logged_cursor.execute("SELECT 1")

    assert cursor.calls == 1
    assert connection.cursor_refreshes == 0


def test_logged_cursor_does_not_retry_non_idempotent_by_default(monkeypatch):
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_TRIES", 3)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_BACKOFF", 1)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_JITTER_SECONDS", 0)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_SCOPE",
                        session_logging.QueryRetryScope.IDEMPOTENT)

    connection = _RetryConnection()
    cursor = _RetryCursor(connection, fail_count=1)
    logged_cursor = _LoggedSnowflakeCursor(cursor)

    with pytest.raises(_FakeTransientError):
        logged_cursor.execute("INSERT INTO foo VALUES (1)")

    assert cursor.calls == 1
    assert connection.cursor_refreshes == 0


def test_logged_cursor_retries_non_idempotent_with_all_scope(monkeypatch):
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_TRIES", 3)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_BACKOFF", 1)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_JITTER_SECONDS", 0)
    monkeypatch.setattr(session_logging, "_QUERY_RETRY_SCOPE",
                        session_logging.QueryRetryScope.ALL)

    connection = _RetryConnection()
    cursor = _RetryCursor(connection, fail_count=1)
    logged_cursor = _LoggedSnowflakeCursor(cursor)

    result = logged_cursor.execute("INSERT INTO foo VALUES (1)")

    assert result == "ok"
    assert cursor.calls == 1
    assert connection.cursor_refreshes == 1
