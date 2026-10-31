# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stand-ins for a Databricks cursor and workspace client, so units need no network."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.parse import unquote

import pyarrow as pa
from databricks.sdk.errors import BadRequest, NotFound
from databricks.sql.exc import DataError
from sources.recording import RecordingConnection, RecordingCursor

FakeConnection = RecordingConnection

EXECUTE_ROWS = 10_000
EXECUTE_BYTES = 1 << 20


@dataclass
class FakeLink:
    """One CloudFetch file as the server advertises it, before it is fetched."""

    startRowOffset: int
    rowCount: int
    bytesNum: int


class FakeChunk:
    """One page's result queue: what its links promised, and what arrived."""
    def __init__(self, advertised: pa.Table, delivered: pa.Table, offset: int,
                 links: bool) -> None:
        self.result_links = [
            FakeLink(offset, advertised.num_rows, advertised.nbytes)
        ] if links else []
        self.delivered = delivered
        self.closed = False

    def remaining_rows(self) -> pa.Table:
        return self.delivered

    def close(self) -> None:
        self.closed = True


class FakeWarehouse:
    """A server that honours a byte bound and refuses a client that has fallen
    behind, the way ``ThriftDatabricksClient.fetch_results`` does. ``delivers``
    returns fewer rows than were advertised, which Databricks is documented to
    do while still reporting success."""
    def __init__(self, table: pa.Table, delivers: int | None = None,
                 links: bool = True) -> None:
        self.table = table
        self.delivers = delivers
        self.links = links
        self.position = 0
        self.calls: list[dict] = []
        self.chunks: list[FakeChunk] = []

    def fetch_results(self, **asked: Any) -> tuple:
        self.calls.append(asked)
        if asked["expected_row_start_offset"] < self.position:
            raise DataError("some result batches must have been skipped")
        return self.page(asked["max_rows"], asked["max_bytes"])

    def page(self, max_rows: int, max_bytes: int) -> tuple:
        rows = self._within(max_rows, max_bytes)
        advertised = self.table.slice(self.position, rows)
        delivered = (advertised if self.delivers is None else advertised.slice(
            0, self.delivers))
        chunk = FakeChunk(advertised, delivered, self.position, self.links)
        self.chunks.append(chunk)
        self.position += rows
        return (chunk, self.position
                < self.table.num_rows, len(chunk.result_links))

    def _within(self, max_rows: int, max_bytes: int) -> int:
        remaining = self.table.num_rows - self.position
        if remaining <= 0:
            return 0
        rows = min(max_rows, remaining)
        while rows > 1 and self.table.slice(self.position,
                                            rows).nbytes > max_bytes:
            rows //= 2
        return rows


class FakeResultSet:
    """The executed statement: its first page, and the Arrow schema bytes the
    driver keeps private to the result set."""
    def __init__(self, chunk: FakeChunk, more: bool, links: int) -> None:
        self.results = chunk
        self.has_more_rows = more
        self.num_chunks = links
        self.lz4_compressed = True
        self._arrow_schema_bytes = b"arrow-schema"


class FakeCursor(RecordingCursor):
    """Records every statement, and pages an Arrow result as the driver does."""
    def __init__(self, responses=None, raises=None,
                 arrow: pa.Table | None = None, delivers: int | None = None,
                 links: bool = True) -> None:
        super().__init__(responses, raises)
        self.buffer_size_bytes = EXECUTE_BYTES
        self.arraysize = EXECUTE_ROWS
        self.arrow = arrow
        self.delivers = delivers
        self.links = links
        self.backend: FakeWarehouse | None = None
        self.active_result_set: FakeResultSet | None = None
        self.active_command_id: str | None = None
        self.description: list | None = None

    def execute(self, sql: str, params: Any = None) -> "FakeCursor":
        result = super().execute(sql, params)
        if self.arrow is not None and sql.lstrip().startswith("SELECT"):
            self._begin()
        return result

    def _begin(self) -> None:
        assert self.arrow is not None
        self.backend = FakeWarehouse(self.arrow, self.delivers, self.links)
        self.active_command_id = "command"
        self.description = [(field.name, str(field.type), None)
                            for field in self.arrow.schema]
        self.active_result_set = FakeResultSet(
            *self.backend.page(self.arraysize, self.buffer_size_bytes))


@dataclass
class FakeEntry:
    path: str
    file_size: int
    is_directory: bool = False


class TrackedStream(io.BytesIO):
    """A stream that remembers whether the reader closed it."""
    def __init__(self, body: bytes,
                 raises: BaseException | None = None) -> None:
        super().__init__(body)
        self.raises = raises
        self.closed_by_reader = False

    def read(self, size: int = -1) -> bytes:
        if self.raises is not None:
            raise self.raises
        return super().read(size)

    def close(self) -> None:
        self.closed_by_reader = True
        super().close()


@dataclass
class FakeDownload:
    contents: Any


class FakeFiles:
    """An in-memory volume directory."""
    def __init__(self, objects: Mapping[str, bytes] | None = None,
                 read_raises: BaseException | None = None,
                 on_delete: Callable[[str], Any] | None = None) -> None:
        self.objects = dict(objects or {})
        self.on_delete = on_delete
        self.read_raises = read_raises
        self.streams: list[TrackedStream] = []

    def list_directory_contents(self, path: str) -> list[FakeEntry]:
        prefix = path.rstrip("/") + "/"
        entries = [
            FakeEntry(name, len(body))
            for name, body in sorted(self.objects.items())
            if name.startswith(prefix)
        ]
        if not entries:
            raise NotFound("The directory being accessed is not found.")
        return entries

    def download(self, path: str) -> FakeDownload:
        stream = TrackedStream(self.objects[path], self.read_raises)
        self.streams.append(stream)
        return FakeDownload(contents=stream)

    def delete(self, path: str) -> None:
        if self.on_delete is not None:
            self.on_delete(path)
        if self.objects.pop(path, None) is None:
            raise NotFound(f"No such file: {path}")

    def delete_directory(self, path: str) -> None:
        return None


class FakeVolume(FakeFiles):
    """A volume with real directories, uploads and ranged downloads."""
    def __init__(self, objects: Mapping[str, bytes] | None = None) -> None:
        super().__init__(objects)
        self.directories: set[str] = set()
        self.ignore_range = False

    def _is_directory(self, path: str) -> bool:
        return path in self.directories or any(
            name.startswith(path + "/") for name in self.objects)

    def list_directory_contents(self, path: str) -> list[FakeEntry]:
        path = path.rstrip("/")
        if not self._is_directory(path):
            raise NotFound(f"No such directory: {path}")
        children: dict[str, FakeEntry] = {}
        for name in sorted(self.objects) + sorted(self.directories):
            if not name.startswith(path + "/"):
                continue
            child = name[len(path) + 1:].split("/")[0]
            full = f"{path}/{child}"
            children[full] = (FakeEntry(full, len(self.objects[full]))
                              if full in self.objects else FakeEntry(
                                  full + "/", 0, is_directory=True))
        return list(children.values())

    def upload(self, path: str, contents: Any, *, overwrite: bool) -> None:
        assert overwrite
        self.objects[path] = contents.read()

    def get_metadata(self, path: str) -> Any:
        if path.endswith("/"):
            raise BadRequest("Bad Request")
        if path not in self.objects:
            raise NotFound(f"No such file: {path}")
        return FakeMetadata(len(self.objects[path]))

    def get_directory_metadata(self, path: str) -> None:
        if not self._is_directory(path):
            raise NotFound(f"No such directory: {path}")

    def create_directory(self, path: str) -> None:
        self.directories.add(path)

    def delete_directory(self, path: str) -> None:
        self.directories.discard(path.rstrip("/"))

    def do(self, method: str, path: str, *, raw: bool,
           response_headers: list[str], headers: Mapping[str, str]) -> dict:
        assert method == "GET" and raw
        remote = unquote(path.removeprefix("/api/2.0/fs/files"))
        if remote not in self.objects:
            raise NotFound(f"No such file: {remote}")
        if self.ignore_range:
            return {"contents": FakeStreamingResponse(self.objects[remote])}
        first, last = headers["Range"].removeprefix("bytes=").split("-")
        size = len(self.objects[remote])
        last = last or str(size - 1)
        stream = FakeStreamingResponse(
            self.objects[remote][int(first):int(last) + 1])
        self.streams.append(stream)
        return {
            "contents": stream,
            "content-range": f"bytes {first}-{last}/{size}"
        }


@dataclass
class FakeMetadata:
    content_length: int


class FakeStreamingResponse(TrackedStream):
    def set_chunk_size(self, size: int) -> None:
        pass


class FakeWorkspaceClient:
    """Exposes only the files API the reader uses."""
    def __init__(self, files: FakeFiles) -> None:
        self.files = files
        self.api_client = files
