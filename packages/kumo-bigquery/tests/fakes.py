# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stand-ins for the BigQuery clients, so unit tests need no network."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from zlib import crc32

MODIFIED = datetime(2016, 3, 12, 1, 30, 13, tzinfo=timezone.utc)
SCHEMA = ("state", "gender", "year", "name", "number")
PROBE_MARKER = "SELECT 1 AS probe"


@dataclass(frozen=True)
class FakeField:
    name: str


@dataclass
class FakeTable:
    """The metadata ``get_table`` returns for one object."""

    table_type: str = "TABLE"
    modified: datetime | None = MODIFIED
    schema: Sequence[FakeField] = field(
        default_factory=lambda: tuple(FakeField(n) for n in SCHEMA))


@dataclass(frozen=True)
class FakeTableReference:
    project: str
    dataset_id: str
    table_id: str


@dataclass(frozen=True)
class FakeSessionInfo:
    session_id: str | None = "session-0f1e"


class FakeQueryJob:
    def __init__(self, destination: Any, bytes_processed: int,
                 schema: Sequence[FakeField] = (),
                 statement_type: str = "SELECT", session_info: Any = None,
                 ddl_target_table: Any = None,
                 result_timeouts: list[float | None] | None = None) -> None:
        self.destination = destination
        self.total_bytes_processed = bytes_processed
        self.schema = schema
        self.statement_type = statement_type
        self.session_info = session_info
        self.ddl_target_table = ddl_target_table
        self.result_timeouts = result_timeouts

    def result(self, timeout: float | None = None) -> FakeQueryJob:
        if self.result_timeouts is not None:
            self.result_timeouts.append(timeout)
        return self


DESTINATION = FakeTableReference("example-project", "_anon", "anon0f1e")


class FakeBigQueryClient:
    """Records every statement and replays canned table metadata.

    Probes are recorded apart from the statements a read issues, so a test
    asserting what a read ran does not have to filter them out.
    """
    def __init__(self, tables: Mapping[str, FakeTable] | None = None,
                 destination: Any = DESTINATION,
                 bytes_processed: int = 66629424,
                 raises: Mapping[str, BaseException] | None = None,
                 probe_raises: BaseException | None = None,
                 table_raises: BaseException | None = None,
                 missing: bool = False, project: str = "example-project",
                 on_export: Any = None, statement_type: str = "SELECT",
                 query_schema: Sequence[FakeField] | None = None,
                 session_id: str | None = "session-0f1e") -> None:
        self.statement_type = statement_type
        self.session_id = session_id
        self.query_schema = (tuple(
            FakeField(n)
            for n in SCHEMA) if query_schema is None else query_schema)
        self.dry_runs: list[str] = []
        self.on_export = on_export
        self.project = project
        self.tables = dict(tables or {})
        self.destination = destination
        self.bytes_processed = bytes_processed
        self.raises = dict(raises or {})
        self.probe_raises = probe_raises
        self.table_raises = table_raises
        self.missing = missing
        self.statements: list[str] = []
        self.probes: list[str] = []
        self.job_configs: list[Any] = []
        self.result_timeouts: list[float | None] = []

    def get_table(self, name: str) -> FakeTable:
        if self.table_raises is not None:
            raise self.table_raises
        if self.missing:
            raise KeyError(name)
        return self.tables.get(name, FakeTable())

    def query(self, sql: str, **kwargs: Any) -> FakeQueryJob:
        config = kwargs.get("job_config")
        self.job_configs.append(config)
        if getattr(kwargs.get("job_config"), "dry_run", False):
            self.dry_runs.append(sql)
            return FakeQueryJob(self.destination, self.bytes_processed,
                                self.query_schema, self.statement_type,
                                result_timeouts=self.result_timeouts)
        if PROBE_MARKER in sql:
            self.probes.append(sql)
            if self.probe_raises is not None:
                raise self.probe_raises
            return FakeQueryJob(self.destination, self.bytes_processed,
                                result_timeouts=self.result_timeouts)
        self.statements.append(sql)
        for marker, error in self.raises.items():
            if marker in sql:
                raise error
        if self.on_export is not None and sql.startswith("EXPORT DATA"):
            self.on_export()
        creates_session_table = sql.startswith("CREATE TEMP TABLE")
        return FakeQueryJob(
            self.destination,
            self.bytes_processed,
            session_info=(FakeSessionInfo(self.session_id) if getattr(
                config, "create_session", False) else None),
            ddl_target_table=(self.destination
                              if creates_session_table else None),
            result_timeouts=self.result_timeouts,
        )

    def matching(self, marker: str) -> list[str]:
        return [
            statement for statement in self.statements if marker in statement
        ]


class FakeBlob:
    """One object in the staging bucket, checksummed as the client does."""
    def __init__(self, name: str, body: bytes, owner: FakeStorageClient,
                 stale: bool = False) -> None:
        self.name = name
        self.body = body
        self.size = len(body)
        self.owner = owner
        self.stale = stale

    def download_to_filename(self, target: str) -> None:
        """Stands in for the end-to-end checksum ``download_to_filename``
        runs."""
        if self.stale:
            from google.api_core.exceptions import NotFound
            raise NotFound(f"No such object: {self.name}")
        self.owner.fetched.append(self.name)
        if self.owner.arrive is not None:
            self.owner.arrive(self.name)
        for marker, error in self.owner.download_raises.items():
            if marker in self.name:
                raise error
        landed = self.owner.corrupt(self.body)
        if crc32(landed) != crc32(self.body):
            from google.cloud.storage.exceptions import DataCorruption
            raise DataCorruption(None, f"checksum mismatch for {self.name}")
        with open(target, "wb") as handle:
            handle.write(landed)

    def delete(self) -> None:
        batch = getattr(self.owner.open_batch, "names", None)
        if batch is not None:
            batch.append(self.name)
        else:
            self.owner.remove(self.name)


class FakeBatch:
    def __init__(self, owner: FakeStorageClient) -> None:
        self.owner = owner

    def __enter__(self) -> FakeBatch:
        self.owner.open_batch.names = []
        return self

    def __exit__(self, *raised: Any) -> None:
        names = self.owner.open_batch.names
        self.owner.open_batch.names = None
        if raised[0] is not None:
            return
        self.owner.batches.append(names)
        if self.owner.on_batch is not None:
            self.owner.on_batch(names)
        failures = []
        for name in names:
            try:
                self.owner.remove(name)
            except Exception as failure:
                failures.append(failure)
        if failures:
            raise failures[0]


class FakeStorageClient:
    """An in-memory bucket, listed by prefix.

    ``download_raises`` is keyed by a name marker, so one object can fail
    while the rest are held at ``arrive``, which every download calls before
    it can fail.
    """
    def __init__(self, bodies: Mapping[str, bytes] | None = None,
                 stale_listing: bool = False,
                 download_raises: Mapping[str, BaseException] | None = None,
                 list_raises: BaseException | None = None,
                 corrupt: Callable[[bytes], bytes] = lambda body: body,
                 arrive: Callable[[str], Any] | None = None,
                 delete_raises: Mapping[str, BaseException] | None = None,
                 on_batch: Callable[[list[str]], Any] | None = None) -> None:
        self.bodies = dict(bodies or {})
        self.delete_raises = dict(delete_raises or {})
        self.on_batch = on_batch
        self.open_batch = threading.local()
        self.batches: list[list[str]] = []
        self.stale_listing = stale_listing
        self.download_raises = dict(download_raises or {})
        self.list_raises = list_raises
        self.corrupt = corrupt
        self.arrive = arrive
        self.listed: list[str] = []
        self.fetched: list[str] = []

    def list_blobs(self, bucket: str,
                   prefix: str | None = None) -> list[FakeBlob]:
        self.listed.append(prefix or "")
        if self.list_raises is not None:
            raise self.list_raises
        return [
            FakeBlob(name, body, self, stale=self.stale_listing)
            for name, body in sorted(self.bodies.items())
            if name.startswith(prefix or "")
        ]

    def bucket(self, name: str) -> FakeBucket:
        return FakeBucket(name, self)

    def batch(self) -> FakeBatch:
        return FakeBatch(self)

    def remove(self, name: str) -> None:
        from google.api_core.exceptions import NotFound
        if name in self.delete_raises:
            raise self.delete_raises[name]
        if self.bodies.pop(name, None) is None:
            raise NotFound(f"No such object: {name}")


class FakeBucket:
    """Hands out live object handles, unpinned by any listing."""
    def __init__(self, name: str, owner: FakeStorageClient) -> None:
        self.name = name
        self.owner = owner

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(name, self.owner.bodies.get(name, b""), self.owner)


@dataclass(frozen=True)
class FakeStream:
    name: str


@dataclass(frozen=True)
class FakeMessage:
    """One ``ReadRowsResponse``, carrying an Arrow IPC message."""

    arrow_record_batch: Any


@dataclass(frozen=True)
class _Serialized:
    serialized_record_batch: bytes


@dataclass(frozen=True)
class _ArrowSchema:
    serialized_schema: bytes


@dataclass(frozen=True)
class FakeReadSession:
    streams: Sequence[FakeStream]
    arrow_schema: Any


class FakeTransport:
    """The private attribute a real client exposes its credential through."""
    def __init__(self, credentials: Any) -> None:
        self._credentials = credentials

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class FakeCredentials:
    """Names the fake client it was taken from, so that the workers a pool
    builds from it are fakes replaying the same pages."""

    client: Any = None


def fake_workers(count: int, *, credentials: Any) -> tuple[Any, ...]:
    """Stands in for building real gRPC clients, which test_streaming covers
    against the real transport."""
    return (credentials.client, ) * count


class FakeReadClient:
    """Records every session it opens and replays canned record batches."""
    def __init__(self, pages: Mapping[str, Sequence[Any]] | None = None,
                 schema: Any = None, raises: BaseException | None = None,
                 arrive: Callable[[], Any] | None = None,
                 credentials: Any = None) -> None:
        self.transport = FakeTransport(credentials)
        self.pages = dict(pages or {})
        self.schema = schema
        self.raises = raises
        self.arrive = arrive
        self.sessions: list[Any] = []
        self.parents: list[str] = []
        self.stream_counts: list[int] = []

    def create_read_session(self, parent: str = "", read_session: Any = None,
                            max_stream_count: int = -1) -> FakeReadSession:
        self.parents.append(parent)
        self.sessions.append(read_session)
        self.stream_counts.append(max_stream_count)
        if self.raises is not None:
            raise self.raises
        return FakeReadSession(
            tuple(FakeStream(name) for name in self.pages),
            _ArrowSchema(self.schema.serialize().to_pybytes()))

    def read_rows(self, name: str) -> list[FakeMessage]:
        if self.arrive is not None:
            self.arrive()
        return [
            FakeMessage(_Serialized(batch.serialize().to_pybytes()))
            for batch in self.pages[name]
        ]
