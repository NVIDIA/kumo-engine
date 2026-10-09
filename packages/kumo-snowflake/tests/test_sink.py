# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recovery behavior at the local Snowflake artifact-export boundary."""

from contextlib import contextmanager
from types import SimpleNamespace

from kumo_snowflake import session, sink
from kumo_snowflake.sink import SnowflakeSinkBackend
from kumo_snowflake.write import write_artifacts_to_table
from kumoapi.jobs import WriteMode


class _Session:
    def __init__(self):
        self.connection = SimpleNamespace(is_closed=lambda: False)
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def test_a_forced_session_refresh_discards_the_cached_connection(monkeypatch):
    opened = []
    cache = session._SessionCache()
    principal = SimpleNamespace(name="oauth")

    def connect(*_):
        value = _Session()
        opened.append(value)
        return value

    monkeypatch.setattr(session, "_connect", connect)

    first = cache.get(principal, None)
    second = cache.get(principal, None, fresh=True)

    assert first is opened[0]
    assert second is opened[1]
    assert first.close_calls == 1


def test_an_owned_session_leaves_the_cached_connection_open(monkeypatch):
    opened = []
    cache = session._SessionCache()
    principal = SimpleNamespace(name="oauth")

    def connect(*_):
        value = _Session()
        opened.append(value)
        return value

    monkeypatch.setattr(session, "_connect", connect)
    cached = cache.get(principal, None)

    with session._owned_session(principal, None) as retry:
        assert retry is opened[1]

    assert cached.close_calls == 0
    assert retry.close_calls == 1
    assert cache.get(principal, None) is cached


class _ExpiredTokenError(Exception):
    errno = 390114
    sqlstate = "08001"


def test_an_overwrite_retries_once_with_a_fresh_session_after_token_expiry(
        monkeypatch):
    backend = SnowflakeSinkBackend()
    cached_sessions = []
    retry_sessions = []
    exports = []

    @contextmanager
    def open_session(_connector):
        cached_sessions.append(1)
        yield _Session()

    @contextmanager
    def open_retry_session(_principal, _connector):
        value = _Session()
        retry_sessions.append(value)
        try:
            yield value
        finally:
            value.close()

    def export(*_args, **_kwargs):
        exports.append(1)
        if len(exports) == 1:

            def upload(*_args, **_kwargs) -> None:
                raise _ExpiredTokenError()

            write_artifacts_to_table(SimpleNamespace(context=object()),
                                     "PREDICTIONS", "/tmp/predictions.parquet",
                                     replace=True, upload_local=upload)

    monkeypatch.setattr(backend, "_session", open_session)
    monkeypatch.setattr(backend, "_principal", lambda _connector: object())
    monkeypatch.setattr(sink, "_owned_session", open_retry_session)
    monkeypatch.setattr(backend, "_qualified", lambda *_: '"PREDICTIONS"')
    monkeypatch.setattr(sink, "_require_snowpark", lambda _session: None)
    monkeypatch.setattr(
        "kumo_core.artifact_staging.selected_local_parquet_files",
        lambda _files: ["/tmp/predictions.parquet"])
    monkeypatch.setattr("kumo_snowflake.write.export_table_files", export)

    assert backend.write_files(
        SimpleNamespace(), "PREDICTIONS", ["ignored"],
        write_mode=WriteMode.OVERWRITE) == '"PREDICTIONS"'
    assert cached_sessions == [1]
    assert len(retry_sessions) == 1
    assert retry_sessions[0].close_calls == 1
    assert len(exports) == 2


def test_an_append_does_not_retry_after_token_expiry(monkeypatch):
    backend = SnowflakeSinkBackend()

    @contextmanager
    def open_session(_connector):
        yield _Session()

    monkeypatch.setattr(backend, "_session", open_session)
    monkeypatch.setattr(sink, "_require_snowpark", lambda _session: None)
    monkeypatch.setattr(
        "kumo_core.artifact_staging.selected_local_parquet_files",
        lambda _files: ["/tmp/predictions.parquet"])
    monkeypatch.setattr(
        "kumo_snowflake.write.export_table_files", lambda *_args, **_kwargs:
        (_ for _ in ()).throw(_ExpiredTokenError()))

    try:
        backend.write_files(SimpleNamespace(), "PREDICTIONS", ["ignored"],
                            write_mode=WriteMode.APPEND)
    except _ExpiredTokenError:
        pass
    else:
        raise AssertionError("append must not be retried after token expiry")
