# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from functools import partial
from types import TracebackType
from typing import Callable, Iterator, Mapping, Sequence

import requests
from kumo_core.sources.diagnostics import safe_summary
from kumo_core.sources.errors import PermanentSourceError, TransientSourceError
from kumo_core.sources.transfer import StagedFile, fetch_all
from kumo_snowflake.spcs import SPCS_TOKEN_PATH
from requests.adapters import HTTPAdapter
from snowflake.connector.cursor import SnowflakeCursor

MAX_FILE_SIZE = 128 * 1024 * 1024
PRESIGNED_BATCH = 1000

#: Every link is minted before the first byte is fetched and nothing re-mints
#: an expired one, so this covers the largest table at the slowest plausible
#: link: over 250 Mbps an hour is 112 GB and a day is 2.7 TB. The links
#: address a `TEMPORARY` stage Snowflake drops with the session, so the longer
#: window outlives nothing.
URL_LIFETIME_SECONDS = 24 * 60 * 60
DOWNLOAD_TIMEOUT = 300
# Large enough to amortize Python iteration without holding more than 128 MiB
# across the measured-optimal sixteen active transfers.
DOWNLOAD_CHUNK = 8 * 1024 * 1024
# A 32/64-worker sweep increased contention; sixteen won on all eleven tables.
DOWNLOAD_WORKERS = 16
# The retry budget lets a multi-hour ingest ride out an object-store incident.
DOWNLOAD_ATTEMPTS = 12
DOWNLOAD_RETRY_BACKOFF_SECONDS = 1.0
DOWNLOAD_RETRY_MAX_BACKOFF_SECONDS = 60.0
TRANSIENT_DOWNLOAD_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
NOT_PLAINTEXT = "; the stage was created without SNOWFLAKE_SSE encryption"

logger = logging.getLogger(__name__)

CREATE_STAGE = ("CREATE TEMPORARY STAGE IF NOT EXISTS {stage} "
                "ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')")
REMOVE = "REMOVE '{location}'"
UNLOAD = ("COPY INTO '{location}' FROM (SELECT {projection} FROM {source}) "
          "FILE_FORMAT=(TYPE=PARQUET) MAX_FILE_SIZE={max_file_size} "
          "HEADER=TRUE OVERWRITE=TRUE")
LIST = "LIST '{location}'"
PRESIGN = ("SELECT path, GET_PRESIGNED_URL('@{stage}', path, {lifetime}) "
           "FROM (VALUES {values}) AS s(path)")
PRESIGN_ONE = "SELECT GET_PRESIGNED_URL('@{stage}', {path}, {lifetime})"


@dataclass(frozen=True)
class UnloadResult:
    rows_unloaded: int
    input_bytes: int


def _location(stage: str, prefix: str = "") -> str:
    return f"@{stage}/{prefix}"


def create_stage(cursor: SnowflakeCursor, stage: str) -> None:
    """A temporary stage needs no grant a reader does not already hold, and
    server-side encryption is what makes a presigned fetch return plaintext."""
    cursor.execute(CREATE_STAGE.format(stage=stage))


def clear_prefix(cursor: SnowflakeCursor, stage: str, prefix: str) -> None:
    cursor.execute(REMOVE.format(location=_location(stage, prefix)))


def unload(
    cursor: SnowflakeCursor,
    *,
    stage: str,
    prefix: str,
    projection: str,
    source: str,
) -> UnloadResult:
    cursor.execute(
        UNLOAD.format(
            location=_location(stage, prefix),
            projection=projection,
            source=source,
            max_file_size=MAX_FILE_SIZE,
        ))
    rows = cursor.fetchall()
    if not rows:
        raise PermanentSourceError(
            f"snowflake unloaded {source} without reporting a row count")
    return UnloadResult(rows_unloaded=int(rows[0][0]),
                        input_bytes=int(rows[0][1]))


def list_prefix(cursor: SnowflakeCursor, stage: str,
                prefix: str) -> tuple[StagedFile, ...]:
    cursor.execute(LIST.format(location=_location(stage, prefix)))
    files = []
    for row in cursor.fetchall():
        _, _, relative = str(row[0]).partition("/")
        if not relative.startswith(prefix):
            raise PermanentSourceError(
                f"snowflake listed {row[0]!r} outside the prefix {prefix!r}")
        files.append(StagedFile(path=relative, size=int(row[1])))
    if not files:
        raise PermanentSourceError(
            f"snowflake unloaded nothing to the stage prefix {prefix!r}")
    return tuple(files)


def _literal(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _batches(paths: Sequence[str]) -> Iterator[Sequence[str]]:
    for start in range(0, len(paths), PRESIGNED_BATCH):
        yield paths[start:start + PRESIGNED_BATCH]


def presigned_urls(cursor: SnowflakeCursor, stage: str,
                   paths: Sequence[str]) -> dict[str, str]:
    """The paths are interpolated because a bound parameter reaches
    ``GET_PRESIGNED_URL`` unfilled and the statement is rejected; minting a
    thousand costs 0.25s against minutes of downloading, so it stays serial."""
    urls: dict[str, str] = {}
    if os.path.exists(SPCS_TOKEN_PATH):
        urls = _presigned_one_at_a_time(cursor, stage, paths)
    else:
        for batch in _batches(list(paths)):
            values = ",".join(f"({_literal(path)})" for path in batch)
            cursor.execute(
                PRESIGN.format(stage=stage, lifetime=URL_LIFETIME_SECONDS,
                               values=values))
            urls.update({str(row[0]): str(row[1]) for row in cursor.fetchall()})
    missing = [path for path in paths if path not in urls]
    if missing:
        raise PermanentSourceError(
            f"snowflake minted no download url for {missing[0]!r}")
    return urls


def _presigned_one_at_a_time(cursor: SnowflakeCursor, stage: str,
                             paths: Sequence[str]) -> dict[str, str]:
    """Inside Snowpark Container Services the path has to reach
    ``GET_PRESIGNED_URL`` as a literal rather than as a column reference, so
    the batched form above mints urls that fetch nothing."""
    urls = {}
    for path in paths:
        cursor.execute(
            PRESIGN_ONE.format(stage=stage, path=_literal(path),
                               lifetime=URL_LIFETIME_SECONDS))
        rows = cursor.fetchall()
        if rows and rows[0][0]:
            urls[path] = str(rows[0][0])
    return urls


def download(
    files: Sequence[StagedFile],
    urls: Mapping[str, str],
    dest: str,
    *,
    worker_ceiling: int | None = None,
) -> tuple[str, ...]:
    """Fetch one table with one persistent HTTP session per active worker.

    Snowflake's staged files share an object-store endpoint, so retaining the
    worker's connection avoids a TLS handshake for every file. Sixteen is an
    evidence-backed connector limit rather than a target: a lower caller or
    CPU ceiling still wins.
    """
    sessions = _WorkerSessions(_download_session)
    try:
        ceiling = (DOWNLOAD_WORKERS if worker_ceiling is None else min(
            DOWNLOAD_WORKERS, worker_ceiling))
        result = fetch_all(files, dest, fetch=partial(_fetch, sessions, urls),
                           diagnosis=NOT_PLAINTEXT, worker_ceiling=ceiling)
        return result
    finally:
        sessions.close()


@dataclass
class _SessionState:
    session: requests.Session
    active: int = 0
    closed: bool = False


class _SessionLocal(threading.local):
    state: _SessionState | None = None


class _TrackedResponse:
    """Keeps an in-flight response alive if another worker fails first."""
    def __init__(self, response: requests.Response,
                 release: Callable[[], None]) -> None:
        self._response = response
        self._release = release
        self._entered = False

    def __enter__(self) -> requests.Response:
        succeeded = False
        try:
            entered = self._response.__enter__()
            succeeded = True
            self._entered = True
            return entered
        finally:
            if not succeeded:
                self._release()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self._response.__exit__(exc_type, exc_value, traceback)
        finally:
            if self._entered:
                self._entered = False
                self._release()


class _WorkerSessions:
    """Invocation-local sessions, reused by a worker and safely reclaimed.

    Core transfer intentionally reports the first failure without waiting for
    running peers. Closing an ordinary shared session at that point races
    those peers. This owner closes idle sessions immediately and defers an
    active session until its response context exits.
    """
    def __init__(self, factory: Callable[[], requests.Session]) -> None:
        self._factory = factory
        self._local = _SessionLocal()
        self._states: list[_SessionState] = []
        self._lock = threading.Lock()
        self._closing = False

    def get(self, url: str, *, stream: bool = False,
            timeout: float | None = None) -> _TrackedResponse:
        state = self._local.state
        with self._lock:
            if state is not None and not state.closed:
                state.active += 1
            else:
                state = None
        if state is None:
            state = _SessionState(self._factory())
            self._local.state = state
            with self._lock:
                self._states.append(state)
                state.active += 1
        succeeded = False
        try:
            response = state.session.get(url, stream=stream, timeout=timeout)
            succeeded = True
            return _TrackedResponse(response, partial(self._release, state))
        finally:
            if not succeeded:
                self._release(state)

    def discard_current(self) -> None:
        state = self._local.state
        if state is not None:
            _discard_pooled_connections(state.session)

    def close(self) -> None:
        ready: list[requests.Session] = []
        with self._lock:
            self._closing = True
            for state in self._states:
                if state.active == 0 and not state.closed:
                    state.closed = True
                    ready.append(state.session)
        error = _close_sessions(ready)
        if error is not None:
            _warn_cleanup("closing staged download sessions", error)

    def _release(self, state: _SessionState) -> None:
        session: requests.Session | None = None
        with self._lock:
            state.active -= 1
            if self._closing and state.active == 0 and not state.closed:
                state.closed = True
                session = state.session
        if session is not None:
            error = _close_sessions([session])
            if error is not None:
                _warn_cleanup("closing an in-flight download session", error)


def _download_session() -> requests.Session:
    """Create one worker session with hidden urllib3 retries disabled."""
    session = requests.Session()
    try:
        for prefix in ("http://", "https://"):
            session.mount(
                prefix,
                HTTPAdapter(pool_connections=8, pool_maxsize=1, max_retries=0,
                            pool_block=True))
    except Exception:
        try:
            session.close()
        except Exception as error:
            _warn_cleanup("closing a partially configured session", error)
        raise
    return session


def _close_sessions(sessions: Sequence[requests.Session]) -> Exception | None:
    first: Exception | None = None
    for session in sessions:
        try:
            session.close()
        except Exception as error:
            if first is None:
                first = error
    return first


def _warn_cleanup(action: str, error: Exception) -> None:
    """Never render a third-party exception that may retain a signed URL."""
    logger.warning("%s raised %s; continuing with the primary outcome", action,
                   type(error).__name__)


def _discard_pooled_connections(session: requests.Session) -> None:
    for adapter in set(session.adapters.values()):
        try:
            adapter.close()
        except Exception as error:
            _warn_cleanup("discarding a failed connection pool", error)


def _fetch(session: _WorkerSessions, urls: Mapping[str, str],
           staged: StagedFile, target: str) -> None:
    """``requests`` names the url it failed on, and that url is a usable
    signature for the life of the link, so only type and status survive."""
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        _remove_partial(target)
        try:
            with session.get(urls[staged.path], stream=True,
                             timeout=DOWNLOAD_TIMEOUT) as response:
                response.raise_for_status()
                with open(target, "wb") as handle:
                    for chunk in response.iter_content(DOWNLOAD_CHUNK):
                        handle.write(chunk)
        except Exception as error:
            _remove_partial(target, error)
            if (not isinstance(error, requests.RequestException)
                    or attempt == DOWNLOAD_ATTEMPTS
                    or not _transient_download_error(error)):
                if isinstance(error, requests.HTTPError):
                    status = getattr(getattr(error, "response", None),
                                     "status_code", None)
                    failure = (TransientSourceError
                               if status in TRANSIENT_DOWNLOAD_STATUS else
                               PermanentSourceError)
                    raise failure(
                        f"snowflake download of {staged.path} failed on "
                        f"attempt {attempt}/{DOWNLOAD_ATTEMPTS}: "
                        f"{safe_summary(error, status=status)}") from None
                if isinstance(error, requests.RequestException):
                    raise TransientSourceError(
                        f"snowflake download of {staged.path} failed on "
                        f"attempt {attempt}/{DOWNLOAD_ATTEMPTS}: "
                        f"{safe_summary(error)}") from None
                raise
            reason = type(error).__name__
        else:
            written = os.path.getsize(target)
            if staged.size is None or written == staged.size:
                return
            mismatch = TransientSourceError(
                f"downloaded {written} of {staged.size} bytes for "
                f"{staged.path}")
            _remove_partial(target, mismatch)
            if attempt == DOWNLOAD_ATTEMPTS:
                raise mismatch
            reason = "stage-size mismatch"

        session.discard_current()
        _download_backoff(staged.path, attempt, reason)

    raise AssertionError("the bounded download loop did not terminate")


def _transient_download_error(error: requests.RequestException) -> bool:
    if isinstance(error, (requests.ConnectionError, requests.Timeout,
                          requests.exceptions.ChunkedEncodingError)):
        return True
    if isinstance(error, requests.HTTPError):
        status = getattr(error.response, "status_code", None)
        return status in TRANSIENT_DOWNLOAD_STATUS
    return False


def _remove_partial(target: str,
                    original_error: Exception | None = None) -> None:
    try:
        os.remove(target)
    except FileNotFoundError:
        pass
    except OSError as error:
        if original_error is None:
            raise
        _warn_cleanup("removing a partial staged file", error)


def _download_backoff(relative: str, attempt: int, reason: str) -> None:
    delay = min(DOWNLOAD_RETRY_BACKOFF_SECONDS * (2**(attempt - 1)),
                DOWNLOAD_RETRY_MAX_BACKOFF_SECONDS)
    logger.warning(
        "retrying staged file %r after attempt %d/%d in %.1f seconds (%s)",
        relative, attempt, DOWNLOAD_ATTEMPTS, delay, reason)
    time.sleep(delay)
