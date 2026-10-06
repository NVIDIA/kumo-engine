# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import threading
from typing import Any, Iterable, Iterator

import pyarrow as pa
from databricks.sql.client import Cursor
from kumo_core.sources.arrow_parquet import (
    WriteResult,
    materialize_arrow_streams,
)
from kumo_core.sources.cpu import encode_workers
from kumo_core.sources.errors import (
    PermanentSourceError,
    RowCountMismatchError,
    SchemaChangedError,
)

PAGE_BYTES = 32 * 1024 * 1024
PAGE_ROWS = 1_000_000

#: How many bytes of *queued* pages this holds, and so -- divided by the page
#: size -- how many encoders it runs. Not the reader's footprint: each encoder
#: also holds the page it is encoding and a shard accumulator, so at the
#: defaults the reader costs around 1 GiB, not 256 MiB.
QUEUED_PAGE_BYTES = 256 * 1024 * 1024

HANDOVER_POLL_SECONDS = 0.05


def arrow_pages(
    cursor: Cursor,
    query: str,
    *,
    max_bytes: int = PAGE_BYTES,
    max_rows: int = PAGE_ROWS,
) -> Iterator[pa.Table]:
    """``maxRows`` is advisory and only ``maxBytes`` is honored, so the row
    bound is a coarse ceiling. The first page comes from ``execute``, bounded
    by the cursor's own 100MB buffer and not by anything passed here, so the
    cursor is set before the statement runs.
    """
    cursor.buffer_size_bytes = max_bytes
    cursor.arraysize = max_rows
    cursor.execute(query)
    result = cursor.active_result_set
    if result is None or cursor.active_command_id is None:
        raise PermanentSourceError("databricks returned no result set to page")
    projected = tuple(column[0] for column in cursor.description or ())
    chunk, more = result.results, result.has_more_rows
    chunk_id, received = result.num_chunks, 0
    arriving: tuple[str, ...] | None = None
    while True:
        drained = _drain(chunk)
        arriving = _same_shape(drained, arriving)
        page = _projected(drained, projected)
        received += page.num_rows
        yield page
        if not more:
            return
        chunk, more, links = cursor.backend.fetch_results(
            command_id=cursor.active_command_id,
            max_rows=max_rows,
            max_bytes=max_bytes,
            expected_row_start_offset=received,
            lz4_compressed=result.lz4_compressed,
            arrow_schema_bytes=_schema_bytes(result),
            description=cursor.description,
            chunk_id=chunk_id,
            use_cloud_fetch=True,
        )
        chunk_id += links


def _same_shape(page: pa.Table,
                arriving: tuple[str, ...] | None) -> tuple[str, ...]:
    """Every page of one result must arrive under the same Arrow schema.

    `_projected` renames by position, which is only sound while the column
    order is fixed for the whole result.
    """
    names = tuple(page.schema.names)
    if arriving is not None and names != arriving:
        raise SchemaChangedError(
            f"databricks changed the result schema mid-read: pages arrived as "
            f"{list(arriving)} and then {list(names)}. Columns are matched by "
            f"position, so continuing would relabel them.")
    return names


def _projected(page: pa.Table, projected: tuple[str, ...]) -> pa.Table:
    """Databricks' CloudFetch Arrow schema reports the *inner* column name
    where a projection reads from a derived table, while the DB-API
    description reports the alias the statement asked for. Matched by
    position, which `_same_shape` is what makes safe.
    """
    if not projected or len(projected) != page.num_columns:
        return page
    if tuple(page.schema.names) == projected:
        return page
    return page.rename_columns(list(projected))


def stream_to_parquet(
    cursor: Cursor,
    query: str,
    dest: str,
    *,
    worker_ceiling: int | None = None,
    max_bytes: int = PAGE_BYTES,
) -> WriteResult:
    count = _encoder_count(max_bytes, worker_ceiling)
    relay = PageRelay(count)
    pages = arrow_pages(cursor, query, max_bytes=max_bytes)
    producer = threading.Thread(target=relay.feed, args=(pages, ), daemon=True)
    producer.start()
    try:
        return materialize_arrow_streams([relay.stream() for _ in range(count)],
                                         dest, worker_ceiling=count)
    finally:
        relay.abandon()
        producer.join()


class PageRelay:
    """The queue is bounded because an unbounded one accumulates the whole
    table, which is what paging by bytes exists to prevent."""
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.pages: queue.Queue = queue.Queue(maxsize=capacity)
        self._drained = threading.Event()
        self._abandoned = threading.Event()
        self._failure: BaseException | None = None

    def feed(self, pages: Iterable[pa.Table]) -> None:
        try:
            for page in pages:
                if not self._offer(page):
                    return
        except BaseException as error:
            # Wider than `Exception` deliberately: a death this thread did
            # not record -- an interrupt, a cancellation -- leaves `stream`
            # waiting on a queue nothing will ever fill.
            self._failure = error
        finally:
            self._drained.set()

    def stream(self) -> Iterator[pa.Table]:
        while True:
            try:
                yield self.pages.get(timeout=HANDOVER_POLL_SECONDS)
            except queue.Empty:
                if self._drained.is_set() and self.pages.empty():
                    break
        if self._failure is not None:
            raise self._failure

    def abandon(self) -> None:
        """Release a producer parked on a queue no consumer will drain."""
        self._abandoned.set()

    def _offer(self, page: pa.Table) -> bool:
        while not self._abandoned.is_set():
            try:
                self.pages.put(page, timeout=HANDOVER_POLL_SECONDS)
                return True
            except queue.Full:
                continue
        return False


def _encoder_count(max_bytes: int, worker_ceiling: int | None) -> int:
    """Every encoder holds a page and so does every queue slot, so more
    encoders than the queue can feed only costs memory."""
    return max(
        1, min(encode_workers(worker_ceiling), QUEUED_PAGE_BYTES // max_bytes))


def _drain(chunk: Any) -> pa.Table:
    """Databricks is documented to report success while returning part of a
    result, so a page is reconciled against the rows its links advertised.
    Each page owns a CloudFetch pool that leaks unless it is closed.
    """
    advertised = _advertised_rows(chunk)
    try:
        page = chunk.remaining_rows()
    finally:
        chunk.close()
    if not isinstance(page, pa.Table):
        raise PermanentSourceError(
            "databricks answered with a column-based result set; a streamed "
            "read needs arrow")
    if advertised is not None and page.num_rows < advertised:
        raise RowCountMismatchError(expected=advertised, actual=page.num_rows)
    return page


def _advertised_rows(chunk: Any) -> int | None:
    """CloudFetch states each file's rows before downloading it; an inline
    page states nothing, and can only be counted from what arrived."""
    links = getattr(chunk, "result_links", None)
    if not links:
        return None
    return sum(link.rowCount for link in links)


def _schema_bytes(result: Any) -> bytes | None:
    """A page carries Arrow data without its schema, and the driver keeps the
    statement's schema private to the result set."""
    return getattr(result, "_arrow_schema_bytes", None)
