# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paging a Databricks result by bytes, and encoding the pages in parallel."""

import threading
import time
from itertools import accumulate
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from databricks.sql.exc import DataError
from fakes import FakeCursor
from kumo_core.sources import cpu
from kumo_core.sources.arrow_parquet import materialize_arrow_streams
from kumo_core.sources.errors import RowCountMismatchError
from kumo_databricks.streaming import (
    PAGE_BYTES,
    QUEUED_PAGE_BYTES,
    PageRelay,
    _encoder_count,
    arrow_pages,
    stream_to_parquet,
)

QUERY = "SELECT `c_custkey`, `c_name` FROM `samples`.`tpch`.`customer`"
SCHEMA = pa.schema([("c_custkey", pa.int64()), ("c_name", pa.string())])
ROWS = 20_000
SMALL, LARGE = 1 << 12, 1 << 16


def table(rows):
    return pa.table(
        {
            "c_custkey": list(range(rows)),
            "c_name": [f"n{index}" for index in range(rows)]
        }, schema=SCHEMA)


def cursor(rows=ROWS, **shape):
    return FakeCursor(arrow=table(rows), **shape)


def test_a_page_is_bounded_by_bytes_not_by_the_advisory_row_count():
    narrow = list(arrow_pages(cursor(), QUERY, max_bytes=SMALL))
    wide = list(arrow_pages(cursor(), QUERY, max_bytes=LARGE))
    assert max(page.nbytes for page in narrow) <= SMALL
    assert max(page.nbytes for page in wide) > SMALL
    assert len(narrow) > len(wide)
    assert sum(page.num_rows for page in narrow) == ROWS
    assert narrow[0].nbytes <= SMALL


def test_each_page_is_asked_for_from_where_the_last_one_ended():
    """The offset handed over is the driver's own contiguity check."""
    reader = cursor()
    pages = list(arrow_pages(reader, QUERY, max_bytes=SMALL))
    assert [call["expected_row_start_offset"] for call in reader.backend.calls
            ] == list(accumulate(page.num_rows for page in pages))[:-1]


@pytest.mark.parametrize("links,reported", [
    (True, RowCountMismatchError),
    (False, DataError),
])
def test_a_page_shorter_than_it_advertised_is_refused(links, reported):
    """Links are what a page is reconciled against; without them a shortfall
    surfaces only later, as the offset the driver checks."""
    with pytest.raises(reported):
        list(
            arrow_pages(cursor(delivers=3, links=links), QUERY,
                        max_bytes=SMALL))


def test_every_page_queue_is_closed_because_each_owns_a_download_pool():
    reader = cursor()
    list(arrow_pages(reader, QUERY, max_bytes=SMALL))
    assert len(reader.backend.chunks) > 1
    assert all(chunk.closed for chunk in reader.backend.chunks)


def out(tmp_path) -> Path:
    """``run_prepare`` empties the destination; a read only fills it."""
    destination = tmp_path / "out"
    destination.mkdir()
    return destination


def test_streaming_writes_parquet_holding_every_row_it_read(tmp_path):
    result = stream_to_parquet(cursor(), QUERY, str(out(tmp_path)),
                               max_bytes=SMALL)
    assert result.rows == ROWS
    assert len(set(result.paths)) == len(result.paths)
    assert sum(pq.ParquetFile(path).metadata.num_rows
               for path in result.paths) == ROWS


def test_an_empty_source_still_writes_one_file_carrying_its_schema(tmp_path):
    result = stream_to_parquet(cursor(0), QUERY, str(out(tmp_path)))
    assert result.rows == 0
    assert pq.ParquetFile(result.paths[0]).schema_arrow.names == SCHEMA.names


def test_a_failure_while_paging_reaches_the_writer_and_leaves_nothing(tmp_path):
    dest = out(tmp_path)
    with pytest.raises(RowCountMismatchError):
        stream_to_parquet(cursor(delivers=3), QUERY, str(dest))
    assert not dest.exists()


def test_pages_reach_a_parquet_part_for_every_encoder(tmp_path, monkeypatch):
    """One shared writer serialises every encoder behind a single encode; the
    gate holds each stream at its first page, so all three must be fed. The
    budget is pinned because a ceiling only caps: a narrower runner would
    otherwise start fewer encoders than the gate waits for."""
    monkeypatch.setattr(cpu, "cpu_budget", lambda: 3)
    relay = PageRelay(3)
    gate = threading.Barrier(3, timeout=10)
    producer = threading.Thread(target=relay.feed,
                                args=(iter([table(10)] * 3), ), daemon=True)
    producer.start()
    try:
        result = materialize_arrow_streams(
            [_gated(relay.stream(), gate) for _ in range(3)],
            str(out(tmp_path)), worker_ceiling=3)
    finally:
        relay.abandon()
        producer.join(timeout=5)
    assert result.rows == 30
    assert {Path(path).name.split("-")[1]
            for path in result.paths} == {"00000", "00001", "00002"}


def test_the_producer_stalls_rather_than_buffering_the_whole_result():
    relay = PageRelay(2)
    produced = []

    def pages():
        for index in range(50):
            produced.append(index)
            yield table(1)

    producer = threading.Thread(target=relay.feed, args=(pages(), ),
                                daemon=True)
    producer.start()
    try:
        assert _settles(produced, relay.capacity + 1)
        next(relay.stream())
        assert _settles(produced, relay.capacity + 2)
    finally:
        relay.abandon()
        producer.join(timeout=5)
    assert not producer.is_alive()


def _gated(stream, gate):
    for page in stream:
        gate.wait()
        yield page


def _settles(produced, expected, seconds=2.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and len(produced) != expected:
        time.sleep(0.01)
    time.sleep(0.2)
    return len(produced) == expected


def test_the_queued_page_bound_outranks_the_worker_ceiling(monkeypatch):
    # A ceiling caps the available CPUs; pin the budget independently of the
    # runner's quota so this test exercises the queue and worker limits.
    monkeypatch.setattr(cpu, "cpu_budget", lambda: 8)
    assert _encoder_count(QUEUED_PAGE_BYTES, 1000) == 1
    assert _encoder_count(PAGE_BYTES, 2) == 2
    assert _encoder_count(PAGE_BYTES, 0) == 1
    assert _encoder_count(PAGE_BYTES, None) == _encoder_count(PAGE_BYTES, 1000)


def test_a_page_that_changes_column_order_is_refused_not_relabelled():
    import pyarrow as pa
    from kumo_core.sources.errors import SchemaChangedError
    from kumo_databricks.streaming import _same_shape

    first = pa.table({'a': pa.array([1]), 'b': pa.array([2])})
    reordered = pa.table({'b': pa.array([2]), 'a': pa.array([1])})

    arriving = _same_shape(first, None)
    assert arriving == ('a', 'b')
    assert _same_shape(first, arriving) == ('a', 'b')
    with pytest.raises(SchemaChangedError, match='changed the result schema'):
        _same_shape(reordered, arriving)
