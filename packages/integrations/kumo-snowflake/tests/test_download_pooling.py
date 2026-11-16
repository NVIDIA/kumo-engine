# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for staged-file connection reuse and retry safety."""

import logging
import threading
import traceback

import pytest
import requests
from kumo_core.sources import transfer
from kumo_core.sources.errors import PermanentSourceError, TransientSourceError
from kumo_core.sources.transfer import StagedFile
from kumo_snowflake import staging


class Response:
    def __init__(self, body=b"PAR1", error=None, status=200):
        self.body = body
        self.error = error
        self.status = status
        self.exited = False

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.exited = True
        return False

    def raise_for_status(self):
        if self.status >= 400:
            response = requests.Response()
            response.status_code = self.status
            response.url = "https://stage/file?X-Amz-Signature=never-log"
            raise requests.HTTPError("signed request failed", response=response)

    def iter_content(self, chunk_size=1):
        assert chunk_size == staging.DOWNLOAD_CHUNK
        yield self.body
        if self.error is not None:
            raise self.error


class SessionFactory:
    def __init__(self, responder, close_error=None):
        self.responder = responder
        self.close_error = close_error
        self.opened = []

    def __call__(self):
        factory = self

        class Session:
            def __init__(self):
                self.adapters = {}
                self.calls = []
                self.close_calls = 0

            def mount(self, prefix, adapter):
                self.adapters[prefix] = adapter

            def get(self, url, **kwargs):
                self.calls.append((url, kwargs, threading.get_ident()))
                return factory.responder(self, url, **kwargs)

            def close(self):
                self.close_calls += 1
                if factory.close_error is not None:
                    raise factory.close_error

        session = Session()
        self.opened.append(session)
        return session


def install(monkeypatch, responder, close_error=None):
    factory = SessionFactory(responder, close_error)
    monkeypatch.setattr(requests, "Session", factory)
    return factory


def objects(count):
    files = tuple(
        StagedFile(f"table/{index}.parquet", 4) for index in range(count))
    urls = {
        staged.path: f"https://stage/{index}?sig=secret"
        for index, staged in enumerate(files)
    }
    return files, urls


def test_sixteen_workers_each_reuse_one_zero_retry_session(
        monkeypatch, tmp_path):
    monkeypatch.setattr(transfer, "download_workers",
                        lambda worker_ceiling: worker_ceiling)
    barrier = threading.Barrier(staging.DOWNLOAD_WORKERS)
    first_calls = set()
    lock = threading.Lock()

    def respond(session, url, **kwargs):
        del url, kwargs
        with lock:
            first = id(session) not in first_calls
            first_calls.add(id(session))
        if first:
            barrier.wait(timeout=10)
        return Response()

    sessions = install(monkeypatch, respond)
    files, urls = objects(2 * staging.DOWNLOAD_WORKERS)

    paths = staging.download(files, urls, str(tmp_path))

    assert len(paths) == len(files)
    assert len(sessions.opened) == staging.DOWNLOAD_WORKERS
    thread_sessions = {}
    for session in sessions.opened:
        assert session.close_calls == 1
        assert set(session.adapters) == {"http://", "https://"}
        for adapter in session.adapters.values():
            assert adapter.max_retries.total == 0
            assert adapter._pool_connections == 8
            assert adapter._pool_maxsize == 1
            assert adapter._pool_block is True
        for _url, kwargs, thread_id in session.calls:
            assert kwargs == {
                "stream": True,
                "timeout": staging.DOWNLOAD_TIMEOUT,
            }
            thread_sessions.setdefault(thread_id, set()).add(id(session))
    assert len(thread_sessions) == staging.DOWNLOAD_WORKERS
    assert all(len(owned) == 1 for owned in thread_sessions.values())


def test_transient_stream_failure_retries_cleanly(monkeypatch, tmp_path,
                                                  caplog):
    attempts = 0
    responses = []
    target = tmp_path / "part.parquet"

    def respond(session, url, **kwargs):
        nonlocal attempts
        del session, url, kwargs
        attempts += 1
        if attempts == 2:
            assert not target.exists()
        response = (Response(
            b"PAR1partial",
            requests.exceptions.ChunkedEncodingError("signed url"))
                    if attempts == 1 else Response())
        responses.append(response)
        return response

    sessions = install(monkeypatch, respond)
    sleeps = []
    monkeypatch.setattr(staging.time, "sleep", sleeps.append)
    caplog.set_level(logging.WARNING, logger=staging.__name__)
    file = StagedFile("table/part.parquet", 4)

    assert staging.download([file], {file.path: "https://stage?sig=secret"},
                            str(tmp_path)) == (str(target), )
    assert attempts == 2
    assert sleeps == [1.0]
    assert all(response.exited for response in responses)
    assert sessions.opened[0].close_calls == 1
    assert "table/part.parquet" in caplog.text
    assert "sig=secret" not in caplog.text


@pytest.mark.parametrize("error_type", [
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
])
def test_transient_retry_budget_and_backoff_are_bounded(monkeypatch, tmp_path,
                                                        caplog, error_type):
    secret = "transport-secret"

    def respond(session, url, **kwargs):
        del session, url, kwargs
        return Response(
            b"PAR1partial",
            error_type(f"failed at https://stage?signature={secret}"))

    sessions = install(monkeypatch, respond)
    sleeps = []
    monkeypatch.setattr(staging.time, "sleep", sleeps.append)
    caplog.set_level(logging.WARNING, logger=staging.__name__)
    file = StagedFile("table/part.parquet", 4)

    with pytest.raises(TransientSourceError) as caught:
        staging.download([file], {file.path: "https://stage?sig=secret"},
                         str(tmp_path))

    rendered = "".join(
        traceback.format_exception(caught.type, caught.value, caught.tb))
    assert len(sessions.opened[0].calls) == staging.DOWNLOAD_ATTEMPTS
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0] + [60.0] * 5
    assert not (tmp_path / "part.parquet").exists()
    assert secret not in rendered
    assert secret not in caplog.text
    assert "During handling of the above exception" not in rendered
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("status,attempts", [(503, 2), (403, 1)])
def test_only_transient_http_statuses_retry(monkeypatch, tmp_path, status,
                                            attempts):
    calls = 0

    def respond(session, url, **kwargs):
        nonlocal calls
        del session, url, kwargs
        calls += 1
        return Response(status=status if calls == 1 else 200)

    install(monkeypatch, respond)
    monkeypatch.setattr(staging.time, "sleep", lambda seconds: None)
    file = StagedFile("table/part.parquet", 4)
    if status == 503:
        staging.download([file], {file.path: "https://stage"}, str(tmp_path))
    else:
        with pytest.raises(PermanentSourceError, match="status=403"):
            staging.download([file], {file.path: "https://stage"},
                             str(tmp_path))
    assert calls == attempts


def test_stage_size_is_attested_after_bounded_retries(monkeypatch, tmp_path):
    sessions = install(monkeypatch,
                       lambda session, url, **kwargs: Response(b"PAR1short"))
    sleeps = []
    monkeypatch.setattr(staging.time, "sleep", sleeps.append)
    file = StagedFile("table/part.parquet", 12)

    with pytest.raises(TransientSourceError, match="downloaded 9 of 12"):
        staging.download([file], {file.path: "https://stage"}, str(tmp_path))

    assert len(sessions.opened[0].calls) == staging.DOWNLOAD_ATTEMPTS
    assert len(sleeps) == staging.DOWNLOAD_ATTEMPTS - 1


def test_real_http_error_traceback_never_contains_signed_url(
        monkeypatch, tmp_path):
    secret = "do-not-disclose"

    def respond(session, url, **kwargs):
        del session, url, kwargs
        response = requests.Response()
        response.status_code = 403
        response.url = f"https://stage/part?X-Amz-Signature={secret}"
        response._content = b""
        response._content_consumed = True
        return response

    sessions = install(monkeypatch, respond)
    file = StagedFile("table/part.parquet", 4)
    with pytest.raises(PermanentSourceError) as caught:
        staging.download([file], {file.path: "https://stage"}, str(tmp_path))

    rendered = "".join(
        traceback.format_exception(caught.type, caught.value, caught.tb))
    assert secret not in rendered
    assert "During handling of the above exception" not in rendered
    assert caught.value.__cause__ is None
    assert sessions.opened[0].close_calls == 1


def test_close_waits_for_an_inflight_response():
    entered = threading.Event()
    release = threading.Event()

    class BlockingResponse(Response):
        def __enter__(self):
            entered.set()
            return super().__enter__()

    sessions = SessionFactory(lambda session, url, **kwargs: BlockingResponse())
    owner = staging._WorkerSessions(sessions)

    def use_response():
        with owner.get("https://stage", stream=True) as response:
            assert response is not None
            assert release.wait(timeout=10)

    worker = threading.Thread(target=use_response)
    worker.start()
    assert entered.wait(timeout=10)
    owner.close()
    assert sessions.opened[0].close_calls == 0
    release.set()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert sessions.opened[0].close_calls == 1


def test_a_late_straggler_never_reuses_a_session_close_already_reclaimed():
    sessions = SessionFactory(lambda session, url, **kwargs: Response())
    owner = staging._WorkerSessions(sessions)
    with owner.get("https://stage"):
        pass
    owner.close()
    assert sessions.opened[0].close_calls == 1

    with owner.get("https://stage"):
        assert sessions.opened[1].close_calls == 0
    assert [session.close_calls for session in sessions.opened] == [1, 1]


def test_close_failure_does_not_mask_a_download_failure(monkeypatch, tmp_path,
                                                        caplog):
    secret = "never-render-close-error"
    sessions = install(
        monkeypatch, lambda session, url, **kwargs:
        (_ for _ in ()).throw(ValueError("original download failure")),
        RuntimeError(f"closing https://stage?signature={secret}"))
    file = StagedFile("table/part.parquet", 4)
    caplog.set_level(logging.WARNING, logger=staging.__name__)

    with pytest.raises(ValueError, match="original download failure"):
        staging.download([file], {file.path: "https://stage"}, str(tmp_path))

    assert sessions.opened[0].close_calls == 1
    assert secret not in caplog.text


def test_close_failure_does_not_discard_a_completed_download(
        monkeypatch, tmp_path, caplog):
    secret = "never-render-close-error"
    sessions = install(
        monkeypatch, lambda session, url, **kwargs: Response(),
        RuntimeError(f"closing https://stage?signature={secret}"))
    file = StagedFile("table/part.parquet", 4)
    caplog.set_level(logging.WARNING, logger=staging.__name__)

    assert staging.download([file], {file.path: "https://stage"},
                            str(tmp_path)) == (str(tmp_path / "part.parquet"), )

    assert sessions.opened[0].close_calls == 1
    assert "RuntimeError" in caplog.text
    assert secret not in caplog.text


def test_every_adapter_is_cleared_when_one_cleanup_fails(caplog):
    secret = "never-render-adapter-error"

    class Adapter:
        def __init__(self, error=None):
            self.error = error
            self.closed = False

        def close(self):
            self.closed = True
            if self.error is not None:
                raise self.error

    failed = Adapter(RuntimeError(f"pool for https://stage?sig={secret}"))
    healthy = Adapter()
    session = type("Session", (), {"adapters": {"a": failed, "b": healthy}})()
    caplog.set_level(logging.WARNING, logger=staging.__name__)

    staging._discard_pooled_connections(session)

    assert failed.closed and healthy.closed
    assert secret not in caplog.text
