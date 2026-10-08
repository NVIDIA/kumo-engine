# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The object-store reader, exercised over a local filesystem.

Injecting the filesystem is what makes this tier possible:
``pyarrow.fs.LocalFileSystem`` answers the same ``get_file_info`` and
``open_input_file`` calls as ``S3FileSystem``, so everything but the network
runs here.
"""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.sources.errors import (
    PermanentSourceError,
    ReconnectRequiredError,
    RowCountMismatchError,
    SourceAccessDeniedError,
    SourceNotFoundError,
    TransientSourceError,
    UnsupportedReadModeError,
)
from kumo_core.sources.object_store import (
    ObjectStoreSourceReader,
    column_dtypes,
)
from kumo_core.sources.prepare import read_table
from kumo_core.sources.types import Attestation, ReadMode
from kumo_core.types.enums import ConnectorType
from kumoapi.typing import Dtype
from pyarrow.fs import LocalFileSystem

from tests.sources.conformance import SourceReaderConformance, parquet_columns

ROWS = 400
COLUMNS = {"id": Dtype.int64, "amount": Dtype.float64}
COLUMNS_ALL = {**COLUMNS, "label": Dtype.string}


def frame(rows: int = ROWS) -> pa.Table:
    return pa.table({
        "id":
        pa.array(range(rows), pa.int64()),
        "amount":
        pa.array([float(index) for index in range(rows)], pa.float64()),
        "label":
        pa.array(["x"] * rows, pa.string()),
    })


def reader_over(root) -> ObjectStoreSourceReader:
    return ObjectStoreSourceReader(LocalFileSystem(), root=str(root))


@pytest.mark.parametrize("sql", ["SELECT id FROM events", ""])
def test_sql_is_rejected_before_source_access_or_destination_cleanup(
        tmp_path, sql):
    reader = reader_over(tmp_path / "missing")
    destination = tmp_path / "retained"
    destination.mkdir()
    sentinel = destination / "existing.parquet"
    sentinel.write_bytes(b"existing")
    with pytest.raises(UnsupportedReadModeError, match="cannot run SQL"):
        reader.data_version("events", sql=sql, columns=COLUMNS)
    with pytest.raises(UnsupportedReadModeError, match="cannot run SQL"):
        read_table(reader, "events", str(destination), columns=COLUMNS, sql=sql,
                   version="supplied")
    assert sentinel.read_bytes() == b"existing"


def staged_frame(table) -> pa.Table:
    return pq.read_table(sorted(table.paths))


def write_parts(directory: Path, *tables: pa.Table) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index, table in enumerate(tables):
        pq.write_table(table, directory / f"part-{index}.parquet")
    return directory


@pytest.fixture
def root(tmp_path) -> Path:
    half = ROWS // 2
    parts = write_parts(tmp_path / "events",
                        frame().slice(0, half),
                        frame().slice(half))
    (parts / "_SUCCESS").write_text("", encoding="utf-8")
    pq.write_table(frame(), tmp_path / "single.parquet")
    for day in ("2026-01-01", "2026-01-02"):
        write_parts(tmp_path / "hive" / f"dt={day}", frame(2))
    (tmp_path / "zips.csv").write_text("zip,count\n01234,5\n00987,7\n01111,9\n",
                                       encoding="utf-8")
    return tmp_path


def test_the_source_type_is_fixed_by_the_class(root):
    assert reader_over(root).source_type is ConnectorType.S3


def test_only_a_streamed_read_is_offered(root):
    assert reader_over(root).read_modes("events") == (ReadMode.STREAMED, )


def test_the_version_changes_when_a_part_is_rewritten(root):
    reader = reader_over(root)
    before = reader.data_version("events")
    os.utime(root / "events" / "part-1.parquet", (2_000_000_000, 2_000_000_000))
    assert reader.data_version("events") != before


def test_version_tracks_removal_of_older_direct_file(root):
    reader = reader_over(root)
    older, newer = (root / "events" / "part-0.parquet",
                    root / "events" / "part-1.parquet")
    os.utime(newer, (2_000_000_000, 2_000_000_000))
    os.utime(older, (1_000_000_000, 1_000_000_000))
    before = reader.data_version("events")
    older.unlink()
    assert reader.data_version("events") != before


@pytest.mark.parametrize("table",
                         ["single", "single.parquet", "events", "hive"])
def test_version_uses_only_metadata_for_recursive_sources(root, table):
    from pyarrow.fs import FileSelector

    class MetadataOnlyFilesystem:
        def get_file_info(self, target):
            if isinstance(target, FileSelector):
                assert target.recursive
            return LocalFileSystem().get_file_info(target)

    reader = ObjectStoreSourceReader(MetadataOnlyFilesystem(), root=str(root))
    version = reader.data_version(table)
    if table in ("events", "hive"):
        assert len(version) == 64
    else:
        assert int(version) > 0


def test_version_of_exact_root_does_not_probe_other_suffixes(root):
    reader = reader_over(root / "single")
    with pytest.raises(SourceNotFoundError):
        reader.data_version("")


def test_the_version_uses_the_listing_before_files_are_read(root, tmp_path):
    """An object landing after the listing must not be stamped onto output
    staged without it: a later run recomputing that version would reuse this
    output and never read the object it missed."""
    reader = reader_over(root)
    before = reader.data_version("events")
    real = reader._objects

    def list_then_a_late_write(table, path, max_entries=None,
                               include_hidden=False):
        found = real(table, path, max_entries, include_hidden)
        late = root / "events" / "part-late.parquet"
        pq.write_table(frame(2), late)
        os.utime(late, (2_000_000_000, 2_000_000_000))
        return found

    reader._objects = list_then_a_late_write
    staged = reader.prepare("events", str(tmp_path / "out"), columns=COLUMNS,
                            mode=ReadMode.STREAMED)
    assert staged.version == before
    # The late object still moves the version, so the next run re-reads it
    # rather than hitting cache on this output.
    assert reader.data_version("events") != before


def test_preparation_uses_same_metadata_version(root, tmp_path):
    reader = reader_over(root)
    staged = reader.prepare("events", str(tmp_path / "out"), columns=COLUMNS,
                            mode=ReadMode.STREAMED)
    assert staged.version == reader.data_version("events")


def test_the_projection_is_pushed_down_in_order(root, tmp_path):
    table = reader_over(root).prepare(
        "events", str(tmp_path / "out"), columns={
            "amount": Dtype.float64,
            "id": Dtype.int64
        }, mode=ReadMode.STREAMED)
    assert parquet_columns(table.directory) == ["amount", "id"]


def test_a_declared_dtype_is_cast_on_the_way_out(root, tmp_path):
    table = reader_over(root).prepare("events", str(tmp_path / "out"),
                                      columns={"id": Dtype.string},
                                      mode=ReadMode.STREAMED)
    assert staged_frame(table).schema.field("id").type == pa.string()


def test_a_column_the_table_lacks_is_reported(root, tmp_path):
    with pytest.raises(SourceNotFoundError, match="absent"):
        reader_over(root).prepare("events", str(tmp_path / "out"),
                                  columns={"absent":
                                           Dtype.int64}, mode=ReadMode.STREAMED)


def test_every_part_is_read(root, tmp_path):
    table = reader_over(root).prepare("events", str(tmp_path / "out"),
                                      columns=COLUMNS, mode=ReadMode.STREAMED)
    assert table.num_rows == ROWS
    assert table.rows_attested_by is Attestation.LOCAL


def test_identical_parquet_is_copied_rather_than_re_encoded(root, tmp_path):
    table = reader_over(root).prepare(
        "events", str(tmp_path / "out"), columns={
            "id": Dtype.int64,
            "amount": Dtype.float64,
            "label": Dtype.string
        }, mode=ReadMode.STREAMED)
    assert sorted(os.path.basename(path) for path in table.paths) == [
        "part-0.parquet", "part-1.parquet"
    ]
    assert table.num_rows == ROWS


def test_a_matching_table_is_written_once_by_the_injected_transfer(
        root, tmp_path):
    import shutil

    fetched = []

    def fetch(source, target):
        fetched.append(os.path.basename(source))
        shutil.copyfile(source, target)

    reader = ObjectStoreSourceReader(LocalFileSystem(), root=str(root),
                                     fetch=fetch)
    staged = reader.prepare("events", str(tmp_path / "out"),
                            columns=COLUMNS_ALL, mode=ReadMode.STREAMED)
    assert sorted(fetched) == ["part-0.parquet", "part-1.parquet"]
    assert [Path(path).read_bytes() for path in sorted(staged.paths)] == [
        (root / "events" / name).read_bytes() for name in sorted(fetched)
    ]
    assert not list(tmp_path.glob("kumo-raw-source-*"))


def test_a_projection_is_read_in_place_and_staged_compactly(root, tmp_path):
    from unittest.mock import Mock

    fetch = Mock()
    reader = ObjectStoreSourceReader(LocalFileSystem(), root=str(root),
                                     fetch=fetch)
    staged = reader.prepare("events", str(tmp_path / "out"), columns=COLUMNS,
                            mode=ReadMode.STREAMED)
    fetch.assert_not_called()
    assert not list(tmp_path.glob("kumo-raw-source-*"))
    assert staged_frame(staged).to_pydict() == frame().select(
        list(COLUMNS)).to_pydict()
    column = pq.ParquetFile(staged.paths[0]).metadata.row_group(0).column(0)
    assert column.compression == "ZSTD"
    assert column.dictionary_page_offset is not None


def test_io_threads_are_raised_while_any_read_needs_them():
    from kumo_core.sources.object_store import _io_threads

    before = pa.io_thread_count()
    try:
        with _io_threads(before + 8):
            with _io_threads(before + 4):
                assert pa.io_thread_count() == before + 8
            assert pa.io_thread_count() == before + 8
        assert pa.io_thread_count() == before
    finally:
        pa.set_io_thread_count(before)


def test_concurrent_stagings_split_the_free_memory(tmp_path, monkeypatch):
    import threading

    from kumo_core.sources import cpu, object_store
    from pyarrow.fs import SubTreeFileSystem

    monkeypatch.setattr(cpu, "_CGROUP_ROOT", tmp_path / "cgroup")
    monkeypatch.setattr(cpu, "_MEMINFO", tmp_path / "meminfo")
    monkeypatch.setattr(cpu, "cpu_budget", lambda: 64)
    (tmp_path / "meminfo").write_text(f"MemAvailable: {8 << 20} kB\n")
    first_inside, second_sized = threading.Event(), threading.Event()
    sized = {}
    real = object_store.materialize_arrow_streams

    def observed(streams, destination, **kwargs):
        sized[Path(destination).name] = kwargs["worker_ceiling"]
        if not first_inside.is_set():
            first_inside.set()
            assert second_sized.wait(5)
        else:
            second_sized.set()
        return real(streams, destination, **kwargs)

    monkeypatch.setattr(object_store, "materialize_arrow_streams", observed)
    for table in ("a", "b"):
        write_parts(tmp_path / "data" / table,
                    *[frame(4) for _ in range(64)])
    reader = ObjectStoreSourceReader(
        SubTreeFileSystem(str(tmp_path), LocalFileSystem()), root="data")

    def stage(table):
        return reader.prepare(table, str(tmp_path / table), columns=COLUMNS,
                              mode=ReadMode.STREAMED)

    errors = []

    def stage_first():
        try:
            stage("a")
        except BaseException as error:
            errors.append(error)
            first_inside.set()

    first = threading.Thread(target=stage_first)
    first.start()
    try:
        assert first_inside.wait(5)
        stage("b")
    finally:
        first.join(10)
    assert not errors
    # Half of 8 GiB at a little over 256 MiB a stream is 15; the second
    # staging halves what those 15 left.
    assert sized == {"a": 15, "b": 8}
    assert cpu._reserved == 0


# The fast path may only trust what every object says ######################


def read_all(root, name, tmp_path):
    reader = reader_over(root)
    return reader.prepare(name, str(tmp_path / "out"),
                          columns=reader.dtypes(name), mode=ReadMode.STREAMED)


def test_a_column_only_a_later_part_has_is_not_copied_into_staging(tmp_path):
    """A dataset takes its schema from one fragment, so a column the catalog
    never reported would otherwise be byte-copied to disk and counted."""
    write_parts(
        tmp_path / "t", pa.table({"id": pa.array([1, 2], pa.int64())}),
        pa.table({
            "id": pa.array([3, 4], pa.int64()),
            "extra": pa.array(["x", "y"])
        }))
    staged = read_all(tmp_path, "t", tmp_path)
    assert staged.num_rows == 4
    assert parquet_columns(staged.directory) == ["id"]


def test_a_column_a_later_part_is_missing_is_filled_rather_than_refused(
        tmp_path):
    """Column-drop evolution is ordinary in a lake; the re-encode path fills
    the gap with nulls, while a byte copy would hand ingest a file whose
    schema is missing a declared column."""
    write_parts(
        tmp_path / "t",
        pa.table({
            "id": pa.array([1, 2], pa.int64()),
            "tier": pa.array(["a", "b"])
        }), pa.table({"id": pa.array([3, 4], pa.int64())}))
    staged = read_all(tmp_path, "t", tmp_path)
    written = staged_frame(staged)
    assert written.num_rows == 4
    assert written.column("tier").to_pylist().count(None) == 2


def test_parts_that_disagree_on_a_type_are_never_staged_side_by_side(tmp_path):
    """A byte copy would leave one staged file calling ``id`` an integer and
    the next calling it a string; reconciling to the declared type is what the
    re-encode path is for."""
    write_parts(tmp_path / "t", pa.table({"id": pa.array([1, 2], pa.int64())}),
                pa.table({"id": pa.array(["3", "4"], pa.string())}))
    staged = read_all(tmp_path, "t", tmp_path)
    types = {pq.read_schema(path).field("id").type for path in staged.paths}
    assert types == {pa.int64()}
    assert sorted(staged_frame(staged).column("id").to_pylist()) == [1, 2, 3, 4]


def test_parts_in_a_different_column_order_are_re_encoded(tmp_path):
    write_parts(
        tmp_path / "t",
        pa.table({
            "a": pa.array([1, 2], pa.int64()),
            "b": pa.array(["p", "q"])
        }),
        pa.table({
            "b": pa.array(["r", "s"]),
            "a": pa.array([3, 4], pa.int64())
        }))
    staged = read_all(tmp_path, "t", tmp_path)
    assert staged.num_rows == 4
    assert parquet_columns(staged.directory) == ["a", "b"]
    assert not any(
        os.path.basename(path).startswith("part-0.parquet")
        for path in staged.paths)


@pytest.mark.parametrize("stored,declared", [
    (pa.int8(), Dtype.int32),
    (pa.int16(), Dtype.int32),
    (pa.list_(pa.int32()), Dtype.intlist),
    (pa.list_(pa.float64()), Dtype.floatlist),
])
def test_a_type_the_catalog_widens_is_widened_on_disk_too(
        stored, declared, tmp_path):
    """The catalog reports ``int8`` as ``int32``; a byte copy would leave
    staging holding the narrower type the graph never declared."""
    values = {
        pa.int8(): [1, 2],
        pa.int16(): [1, 2],
        pa.list_(pa.int32()): [[1], [2]],
        pa.list_(pa.float64()): [[1.5], [2.5]],
    }[stored]
    write_parts(tmp_path / "t", pa.table({"c": pa.array(values, stored)}))
    reader = reader_over(tmp_path)
    assert reader.dtypes("t") == {"c": declared}
    staged = read_all(tmp_path, "t", tmp_path)
    assert staged_frame(staged).schema.field("c").type != stored


def test_a_dictionary_column_is_decoded_rather_than_copied(tmp_path):
    write_parts(tmp_path / "t",
                pa.table({"c": pa.array(["a", "b", "a"]).dictionary_encode()}))
    reader = reader_over(tmp_path)
    assert reader.dtypes("t") == {"c": Dtype.string}
    staged = read_all(tmp_path, "t", tmp_path)
    assert staged_frame(staged).schema.field("c").type == pa.string()


def test_a_timestamp_is_copied_because_neither_path_would_cast_it(tmp_path):
    """``Dtype.date`` has no Arrow spelling to cast to, so the re-encode path
    writes what it read; copying is then the same table, faster."""
    write_parts(tmp_path / "t",
                pa.table({"ts": pa.array([1, 2], pa.timestamp("s"))}))
    staged = read_all(tmp_path, "t", tmp_path)
    assert [os.path.basename(path)
            for path in staged.paths] == ["part-0.parquet"]


def test_a_decimal_is_rewritten_rather_than_passed_through_as_it_lies(tmp_path):
    """``DECIMAL(38,0)`` is reported as ``int64`` only by an unsafe cast, so
    the bytes cannot be copied while the catalog claims an integer."""
    write_parts(
        tmp_path / "t",
        pa.table({
            "id": pa.array([Decimal(7)] * 3, pa.decimal128(38, 0)),
            "price": pa.array([Decimal("1.25")] * 3, pa.decimal128(18, 2)),
        }))
    reader = reader_over(tmp_path)
    assert reader.dtypes("t") == {"id": Dtype.int64, "price": Dtype.float32}
    staged = read_all(tmp_path, "t", tmp_path)
    written = staged_frame(staged)
    assert written.schema.field("id").type == pa.int64()
    assert written.column("id").to_pylist() == [7, 7, 7]


# Formats and failures #####################################################


def test_a_hive_partition_key_becomes_a_real_string_column(root, tmp_path):
    """The hosted connector reads a partitioned table through
    ``pq.ParquetDataset``, which dictionary-encodes the keys, so ``dt`` is a
    string in the cloud and has to be one here."""
    reader = reader_over(root)
    assert reader.dtypes("hive")["dt"] is Dtype.string
    table = reader.prepare("hive", str(tmp_path / "out"), columns={
        "id": Dtype.int64,
        "dt": Dtype.string
    }, mode=ReadMode.STREAMED)
    written = staged_frame(table)
    assert written.column_names == ["id", "dt"]
    assert sorted(set(
        written.column("dt").to_pylist())) == ["2026-01-01", "2026-01-02"]


def test_a_csv_table_keeps_the_leading_zeros_of_a_zip_code(root):
    """The one narrowing everything else is built on: Arrow left to itself
    reads ``01234`` as 1234, and the cloud keeps the column as text."""
    reader = reader_over(root)
    assert reader.resolve("zips")[1] == "csv"
    assert reader.dtypes("zips") == {
        "zip": Dtype.string,
        "count": Dtype.int64,
    }


def test_an_unreadable_object_is_a_source_error_not_a_bare_arrow_one(tmp_path):
    directory = tmp_path / "t"
    directory.mkdir()
    (directory / "part-0.parquet").write_bytes(b"")
    with pytest.raises(PermanentSourceError, match="not readable Parquet"):
        reader_over(tmp_path).dtypes("t")


def test_a_failed_read_leaves_no_parquet_behind(root, tmp_path):
    broken = root / "broken"
    write_parts(broken, frame())
    (broken / "part-1.parquet").write_bytes(b"PAR1 not really parquet")
    destination = tmp_path / "out"
    with pytest.raises(PermanentSourceError):
        reader_over(root).prepare("broken", str(destination),
                                  columns={"id": Dtype.string},
                                  mode=ReadMode.STREAMED)
    assert list(destination.glob("*.parquet")) == []


def test_a_root_that_is_not_there_is_told_apart_from_an_empty_one(tmp_path):
    from kumo_core.sources import layout

    with pytest.raises(SourceNotFoundError, match="Nothing exists at"):
        layout.scan(LocalFileSystem(), str(tmp_path / "typo"))
    assert layout.scan(LocalFileSystem(), str(tmp_path)) == []


def test_a_head_reports_the_same_columns_the_catalog_does(tmp_path):
    """A sample naming a column the catalog does not is one nothing
    downstream can place: the pandas index column is dropped from the
    reported schema, so it has to be absent from the rows as well."""
    written = pa.table({
        "__index_level_0__": pa.array([0, 1], pa.int64()),
        "id": pa.array([7, 8], pa.int64()),
    })
    write_parts(tmp_path / "indexed", written)
    reader = reader_over(tmp_path)
    assert reader.head("indexed",
                       2).column_names == list(reader.dtypes("indexed"))


def test_a_column_type_kumo_cannot_read_is_refused_rather_than_copied(tmp_path):
    """The byte-copy path takes a plain multi-part Parquet table without
    looking at what is in it, which made it the one way a struct column could
    reach a graph. The hosted connector refuses these at its metadata read."""
    written = pa.table({
        "id":
        pa.array([1, 2], pa.int64()),
        "payload":
        pa.array([{
            "a": 1
        }, {
            "a": 2
        }], pa.struct([("a", pa.int64())])),
    })
    write_parts(tmp_path / "nested", written)
    reader = reader_over(tmp_path)
    with pytest.raises(PermanentSourceError, match="payload"):
        reader.prepare(
            "nested", str(tmp_path / "out"), columns={
                "id": Dtype.int64,
                "payload": Dtype.unsupported
            }, mode=ReadMode.STREAMED)


@pytest.mark.parametrize("name", ["../secrets", "a/../../secrets", "a\\b"])
def test_a_table_name_that_walks_out_of_the_root_is_refused(name, root):
    """The connector root is the only fence a local read has -- it runs under
    the developer's own AWS credentials, and no filesystem here confines
    anything. The hosted connector rejects exactly these components."""
    with pytest.raises(PermanentSourceError, match="below the connector root"):
        reader_over(root).resolve(name)


def test_a_verbatim_copy_reports_a_vendor_failure_as_one(root, tmp_path):
    """The copy is the default path for a plain multi-part Parquet table and
    sat outside the translation, so a 503 mid-download left the reader as a
    bare ``OSError`` that nothing above could classify or retry."""
    class _FailingDownload(LocalFileSystem):
        def open_input_stream(self, *args, **kwargs):
            raise pa.ArrowIOError("AWS Error SLOW_DOWN during GetObject")

    reader = ObjectStoreSourceReader(_FailingDownload(), root=str(root))
    with pytest.raises(TransientSourceError):
        reader.prepare("events", str(tmp_path / "out"), columns=COLUMNS_ALL,
                       mode=ReadMode.STREAMED)


def test_pandas_index_columns_are_dropped_from_the_reported_schema():
    schema = pa.schema([("id", pa.int64()), ("__index_level_0__", pa.int64())])
    assert column_dtypes(schema) == {"id": Dtype.int64}


def test_the_scan_of_a_part_does_not_start_before_it_is_reached(
        tmp_path, monkeypatch):
    """Building every scanner up front starts every one of them, and the
    buffers that allocates grow with the table rather than with the pool
    consuming them: 12 parts held 17MB before a row was read, and the reviewed
    shape held 2124MB.

    Asserted on whether the scans have begun rather than on bytes, because the
    Arrow pool is process-global and every other test in this file moves it.
    """
    import inspect

    from kumo_core.sources import object_store

    # Two columns and one declared, so the read re-encodes rather than taking
    # the byte-for-byte copy, which streams nothing.
    wide = pa.table({
        "c": pa.array(["x" * 16] * 2_000),
        "d": pa.array(range(2_000), pa.int64()),
    })
    write_parts(tmp_path / "wide", *([wide] * 12))

    handed = {}
    real = object_store.materialize_arrow_streams

    def measured(streams, destination, **kwargs):
        streams = list(streams)
        handed["states"] = [
            inspect.getgeneratorstate(stream)
            if inspect.isgenerator(stream) else "STARTED" for stream in streams
        ]
        return real(streams, destination, **kwargs)

    monkeypatch.setattr(object_store, "materialize_arrow_streams", measured)
    table = reader_over(tmp_path).prepare("wide", str(tmp_path / "out"),
                                          columns={"c": Dtype.string},
                                          mode=ReadMode.STREAMED)

    assert table.num_rows == 24_000
    assert handed["states"] == ["GEN_CREATED"] * 12, (
        f'scans already under way at hand-over: {handed["states"]}')


class _Failing:
    """A filesystem that answers every call with one vendor failure."""
    def __init__(self, error):
        self._error = error

    def get_file_info(self, *args, **kwargs):
        raise self._error

    def open_input_file(self, *args, **kwargs):
        raise self._error

    def open_input_stream(self, *args, **kwargs):
        raise self._error


@pytest.mark.parametrize(
    "error,expected",
    [
        (pa.ArrowIOError("AccessDenied: not authorized"),
         SourceAccessDeniedError),
        (pa.ArrowIOError("ExpiredToken: the token has expired"),
         ReconnectRequiredError),
        (pa.ArrowIOError("503 SlowDown, please reduce your request rate"),
         TransientSourceError),
        (pa.ArrowIOError("NoSuchBucket: the bucket is not there"),
         SourceNotFoundError),
        (PermissionError("the prefix is not readable"),
         SourceAccessDeniedError),
        (pa.ArrowInvalid("not parseable"), PermanentSourceError),
        # AWS phrases a rejected key as an absence, and a key is free to be
        # spelled like a status.
        (pa.ArrowIOError("The AWS Access Key Id you provided does not exist in "
                         "our records"), ReconnectRequiredError),
        (pa.ArrowIOError("When getting information for key "
                         "'bucket/reports/404/part.parquet': AWS Error "
                         "ACCESS_DENIED"), SourceAccessDeniedError),
        (pa.ArrowIOError(
            "Could not open 'bucket/not-found/x.parquet': AWS Error "
            "SLOW_DOWN during HeadObject"), TransientSourceError),
        (pa.ArrowIOError(
            "Could not open 'forbidden-data/events/x.parquet': AWS "
            "Error NO_SUCH_KEY"), SourceNotFoundError),
    ])
def test_a_filesystem_failure_is_reported_as_the_kind_it_is(error, expected):
    """Nothing above this reader retries what it cannot classify, so a 503 has
    to arrive as transient and an expired token as needing a reconnect.
    ``ArrowIOError`` is an ``OSError`` and not an ``ArrowException``, so it
    reaches here only if the wrapper catches both.
    """
    reader = ObjectStoreSourceReader(_Failing(error), root="bucket")
    with pytest.raises(expected) as raised:
        reader.resolve("events")
    assert type(raised.value) is expected


def test_acquisition_lists_source_objects_once(root, tmp_path):
    listings = []
    reader = reader_over(root)
    real = reader._objects

    def counted(table, path, max_entries=None, include_hidden=False):
        listings.append(path)
        return real(table, path, max_entries, include_hidden)

    reader._objects = counted
    reader.prepare("events", str(tmp_path / "out"), columns=COLUMNS,
                   mode=ReadMode.STREAMED)
    assert listings == [f"{root}/events/"]


def test_a_csv_table_of_more_parts_than_the_cloud_takes_is_refused(tmp_path):
    """A graph the cloud refuses must not be buildable here: the ceiling is
    `layout`'s, and the hosted CSV connector enforces the same one."""
    from kumo_core.sources import layout

    directory = tmp_path / "wide"
    directory.mkdir()
    for index in range(layout.MAX_CSV_PARTITIONS + 1):
        (directory / f"part-{index}.csv").write_text("id\n1\n",
                                                     encoding="utf-8")
    with pytest.raises(PermanentSourceError, match="partitions"):
        reader_over(tmp_path).dtypes("wide")


def test_a_csv_table_larger_than_the_cloud_takes_is_refused(
        tmp_path, monkeypatch):
    from kumo_core.sources import layout

    monkeypatch.setattr(layout, "MAX_CSV_DATASIZE", 4)
    (tmp_path / "big.csv").write_text("id\n1\n2\n", encoding="utf-8")
    with pytest.raises(PermanentSourceError, match="bytes"):
        reader_over(tmp_path).dtypes("big")


def test_a_listing_failure_is_classified_rather_than_raised_as_an_oserror(
        tmp_path):
    """`tables` and the CSV catalog listed outside the translation, so an
    unreachable bucket reached the caller as a bare `OSError`."""
    reader = ObjectStoreSourceReader(
        _Failing(pa.ArrowIOError("AWS Error ACCESS_DENIED")), root="bucket")
    with pytest.raises(SourceAccessDeniedError):
        reader.tables()


def test_the_tables_under_a_root_are_named_by_the_reader(root):
    assert reader_over(root).tables() == ["events", "hive", "single", "zips"]


class TestObjectStoreConformance(SourceReaderConformance):
    """The shared contract, over a directory of parts on local disk."""
    _conformance_root: Path

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def conformance_source(cls, tmp_path_factory) -> None:
        cls._conformance_root = tmp_path_factory.mktemp("object-store-source")
        write_parts(cls._conformance_root / "events",
                    frame().slice(0, 100),
                    frame().slice(100))

    def reader(self) -> ObjectStoreSourceReader:
        return reader_over(str(self._conformance_root))

    def table(self) -> str:
        return "events"

    def columns(self) -> Mapping[str, Dtype]:
        return COLUMNS

    def truth_rows(self) -> int:
        return ROWS


def test_unprojected_parquet_preserves_bytes_without_random_access(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    original = source / 'part.parquet'
    pq.write_table(pa.table({
        'id': [1, 2],
        'extra': ['a', 'b']
    }), original, compression='gzip')
    filesystem = LocalFileSystem()

    class SequentialOnly:
        def get_file_info(self, path):
            return filesystem.get_file_info(path)

        def open_input_stream(self, path):
            return filesystem.open_input_stream(path)

        def open_input_file(self, path):
            raise AssertionError(
                'Raw preservation must not read remote footers')

    reader = ObjectStoreSourceReader(SequentialOnly(), root=str(tmp_path))
    staged = reader.prepare('source', str(tmp_path / 'output'), columns=None,
                            mode=ReadMode.STREAMED, version='supplied')
    assert len(staged.paths) == 1
    assert Path(staged.paths[0]).read_bytes() == original.read_bytes()
    assert staged.version == 'supplied'
    assert staged.source_size == original.stat().st_size
    assert staged.num_rows is None


@pytest.mark.parametrize('nested', ['country=US', 'nested'])
def test_unprojected_nested_parquet_materializes_local_columns(
        tmp_path, nested):
    source = tmp_path / 'source' / nested
    source.mkdir(parents=True)
    pq.write_table(pa.table({'id': [1, 2]}), source / 'part.parquet')
    reader = reader_over(tmp_path)
    reader.file_format = 'parquet'
    staged = reader.prepare('source', str(tmp_path / 'output'), columns=None,
                            mode=ReadMode.STREAMED, version='supplied')
    written = pq.read_table(list(staged.paths))
    assert written.column('id').to_pylist() == [1, 2]
    assert all(
        Path(path).parent == tmp_path / 'output' for path in staged.paths)
    if nested.startswith('country='):
        assert written.column('country').to_pylist() == ['US', 'US']
    assert staged.num_rows == 2


@pytest.mark.parametrize('expected_rows', [2, 3])
def test_unprojected_row_attestation_counts_only_the_local_copy(
        tmp_path, monkeypatch, expected_rows):
    from kumo_core.sources import object_store

    source = tmp_path / 'table.parquet'
    output = tmp_path / 'output'
    pq.write_table(pa.table({'id': [1, 2]}), source)
    counts = []
    count_rows = object_store.count_rows

    def count_local(paths):
        assert all(Path(path).parent == output for path in paths)
        counts.append(list(paths))
        return count_rows(paths)

    monkeypatch.setattr(object_store, 'count_rows', count_local)
    reader = reader_over(tmp_path)
    if expected_rows == 2:
        staged = reader.prepare('table', str(output), columns=None,
                                mode=ReadMode.STREAMED,
                                expected_rows=expected_rows)
        assert staged.num_rows == 2
    else:
        with pytest.raises(RowCountMismatchError):
            reader.prepare('table', str(output), columns=None,
                           mode=ReadMode.STREAMED, expected_rows=expected_rows)
    assert len(counts) == 1


def test_explicit_format_does_not_relax_default_discovery(tmp_path):
    nested = tmp_path / 'table' / 'ordinary_directory'
    nested.mkdir(parents=True)
    pq.write_table(pa.table({'id': [1]}), nested / 'part.parquet')
    with pytest.raises(PermanentSourceError, match='not a Parquet Hive table'):
        reader_over(tmp_path).prepare('table', str(tmp_path / 'output'),
                                      columns=None, mode=ReadMode.STREAMED)


def test_unprojected_parquet_keeps_unknown_input_size_separate(tmp_path):
    from types import SimpleNamespace

    source = tmp_path / 'table.parquet'
    pq.write_table(pa.table({'id': [1, 2]}), source)
    filesystem = LocalFileSystem()

    class UnknownSize:
        def get_file_info(self, path):
            info = filesystem.get_file_info(path)
            return SimpleNamespace(path=info.path, type=info.type, size=None,
                                   mtime=info.mtime, mtime_ns=info.mtime_ns)

        def open_input_stream(self, path):
            return filesystem.open_input_stream(path)

    reader = ObjectStoreSourceReader(UnknownSize(), root=str(tmp_path))
    staged = reader.prepare('table', str(tmp_path / 'output'), columns=None,
                            mode=ReadMode.STREAMED)
    assert staged.scanned_bytes is None
    assert staged.source_size == source.stat().st_size


@pytest.mark.parametrize('partition,values', [
    ('year', [2025, 2026]),
    ('country', ['CA', 'US']),
])
def test_partitions_with_duplicate_basenames_are_read_in_place(
        tmp_path, monkeypatch, partition, values):
    from unittest.mock import Mock

    monkeypatch.chdir(tmp_path)
    for value, key in enumerate(values, start=1):
        part = tmp_path / 'source' / f'{partition}={key}' / 'part.parquet'
        part.parent.mkdir(parents=True)
        pq.write_table(pa.table({'id': [value]}), part)
    fetch = Mock()
    reader = ObjectStoreSourceReader(LocalFileSystem(), root=str(tmp_path),
                                     file_format='parquet', fetch=fetch)
    staged = reader.prepare('source', 'output', columns=None,
                            mode=ReadMode.STREAMED)
    result = pq.read_table(list(staged.paths)).sort_by('id')
    assert result.to_pydict() == {'id': [1, 2], partition: values}
    assert not pa.types.is_dictionary(result.schema.field(partition).type)
    fetch.assert_not_called()
    assert not list(tmp_path.glob('kumo-raw-source-*'))


@pytest.mark.parametrize('kind', ['csv', 'parquet'])
def test_injected_transfer_drains_siblings_before_cleanup(tmp_path, kind):
    import shutil
    import threading
    from concurrent.futures import ThreadPoolExecutor

    source = tmp_path / 'source'
    source.mkdir()
    for name in ['a', 'b']:
        part = source / f'{name}.{kind}'
        if kind == 'csv':
            part.write_text('id\n1\n')
        else:
            pq.write_table(pa.table({'id': [1]}), part)
    started = threading.Event()
    failed = threading.Event()
    release = threading.Event()
    targets = []

    def fetch(source_path, target):
        targets.append(target)
        if Path(source_path).stem == 'a':
            assert started.wait(5)
            failed.set()
            raise RuntimeError('download failed')
        started.set()
        assert release.wait(5)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)

    reader = ObjectStoreSourceReader(LocalFileSystem(), root=str(tmp_path),
                                     file_format=kind, fetch=fetch)
    with ThreadPoolExecutor(max_workers=1) as caller:
        pending = caller.submit(reader.prepare, 'source',
                                str(tmp_path / 'output'), columns=None,
                                mode=ReadMode.STREAMED, worker_ceiling=2)
        try:
            assert failed.wait(5)
            assert not pending.done()
        finally:
            release.set()
        with pytest.raises(RuntimeError, match='download failed'):
            pending.result(timeout=5)
    assert all(not Path(target).exists() for target in targets)
    assert not list(tmp_path.glob('kumo-raw-source-*'))


@pytest.mark.parametrize('suffix', ['csv', 'parquet'])
def test_untransformed_read_preserves_nested_files(tmp_path, suffix):
    root = tmp_path / 'source'
    part = root / 'table' / 'country=US' / f'part.{suffix}'
    part.parent.mkdir(parents=True)
    if suffix == 'parquet':
        pq.write_table(pa.table({'id': [1, 2]}), part)
    else:
        part.write_text('id\n1\n2\n')
    reader = ObjectStoreSourceReader(LocalFileSystem(), root=str(root),
                                     file_format=suffix)
    staged = reader.prepare('table', str(tmp_path / 'prepared'), columns=None,
                            mode=ReadMode.STREAMED, transform=False)
    assert len(staged.paths) == 1
    copied = Path(staged.paths[0])
    assert copied.relative_to(
        staged.directory).as_posix() == f'country=US/part.{suffix}'
    assert copied.read_bytes() == part.read_bytes()


@pytest.mark.parametrize('entry_path,size,max_entries,max_bytes', [
    ('bucket/root//evil.parquet', 1, 200, None),
    ('bucket/other/part.parquet', 1, 200, None),
    ('bucket/root/part.parquet', 2, 200, 1),
    ('bucket/root/part.parquet', None, 200, 1),
    ('bucket/root/part.parquet', 1, 0, None),
])
def test_read_limits_refuse_before_download(tmp_path, entry_path, size,
                                            max_entries, max_bytes):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from pyarrow.fs import FileSelector, FileType
    filesystem = Mock()
    # ``mtime``/``path`` mirror a real ``FileInfo``: the version is sampled
    # before the listing, so the stub is asked for them ahead of the refusal.
    filesystem.get_file_info.side_effect = lambda request: (
        [
            SimpleNamespace(path=entry_path, size=size, type=FileType.File,
                            mtime=None)
        ] if isinstance(request, FileSelector) else SimpleNamespace(
            path='bucket/root', type=FileType.Directory, mtime=None))
    fetch = Mock()
    reader = ObjectStoreSourceReader(filesystem, root='bucket',
                                     file_format='parquet', fetch=fetch)
    with pytest.raises(PermanentSourceError):
        reader.prepare('root', str(tmp_path / 'prepared'), columns=None,
                       mode=ReadMode.STREAMED, transform=False,
                       max_entries=max_entries, max_bytes=max_bytes)
    fetch.assert_not_called()


def test_untransformed_empty_source_reports_zero_rows(tmp_path):
    (tmp_path / 'empty').mkdir()
    reader = ObjectStoreSourceReader(LocalFileSystem(), root=str(tmp_path),
                                     file_format='parquet')
    staged = reader.prepare('empty', str(tmp_path / 'prepared'), columns=None,
                            mode=ReadMode.STREAMED, transform=False)
    assert staged.num_rows == 0
    assert staged.paths == ()
    assert staged.source_size == 0


def test_untransformed_read_keeps_hidden_parquet(tmp_path):
    source = tmp_path / 'table'
    source.mkdir()
    for name in ['visible.parquet', '_hidden.parquet']:
        pq.write_table(pa.table({'id': [1]}), source / name)
    reader = ObjectStoreSourceReader(LocalFileSystem(), root=str(tmp_path),
                                     file_format='parquet')
    raw = reader.prepare('table', str(tmp_path / 'raw'), columns=None,
                         mode=ReadMode.STREAMED, transform=False)
    graph = reader.prepare('table', str(tmp_path / 'graph'), columns=None,
                           mode=ReadMode.STREAMED)
    assert {Path(path).name
            for path in raw.paths} == {'visible.parquet', '_hidden.parquet'}
    assert {Path(path).name for path in graph.paths} == {'visible.parquet'}


@pytest.mark.parametrize("missing_first", [False, True])
def test_graph_ingest_rejects_a_declared_column_missing_from_any_part(
        tmp_path, missing_first):
    parts = [frame(2), frame(2).drop(["amount"])]
    if missing_first:
        parts.reverse()
    write_parts(tmp_path / "events", *parts)
    with pytest.raises(PermanentSourceError,
                       match="missing declared columns: amount"):
        reader_over(tmp_path).validate_ingest_columns("events", COLUMNS)


def test_graph_ingest_accepts_hive_partition_columns(root):
    reader_over(root).validate_ingest_columns("hive", {
        **COLUMNS, "dt": Dtype.string
    })


def test_graph_ingest_allows_declared_type_conversion_and_column_reorder(
        tmp_path):
    write_parts(tmp_path / "events", frame(2),
                pa.table({
                    "amount": [2, 3],
                    "id": ["2", "3"]
                }))
    reader_over(tmp_path).validate_ingest_columns("events", {
        "id": Dtype.string,
        "amount": Dtype.float64
    })


@pytest.mark.parametrize("second_partition,physical_dt,missing", [
    ("region=US", False, True),
    ("region=US", True, False),
    ("dt=__HIVE_DEFAULT_PARTITION__", False, False),
    ("d%74=day-b", False, False),
])
def test_graph_ingest_checks_partition_columns_per_file(tmp_path,
                                                        second_partition,
                                                        physical_dt, missing):
    first = tmp_path / "events" / "dt=day-a"
    second = tmp_path / "events" / second_partition
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    pq.write_table(pa.table({"id": [1]}), first / "part-0.parquet")
    values = {"id": [2]}
    if physical_dt:
        values["dt"] = ["day-b"]
    pq.write_table(pa.table(values), second / "part-1.parquet")
    reader = reader_over(tmp_path)
    columns = {"id": Dtype.int64, "dt": Dtype.string}
    if missing:
        with pytest.raises(PermanentSourceError,
                           match="missing declared columns: dt"):
            reader.validate_ingest_columns("events", columns)
    else:
        reader.validate_ingest_columns("events", columns)


# A re-encoded part streams below its row groups (kumo#31439) ###############


def _arrow_scan(parts: list[Path], projection: Mapping[str, pa.DataType],
                **kwargs) -> pa.Table:
    """What the re-encode path wrote before kumo#31439: Arrow's own scan."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    dataset = ds.dataset([str(part) for part in parts], **kwargs)
    return dataset.to_table(columns={
        name: pc.field(name).cast(dtype)
        for name, dtype in projection.items()
    })


def test_a_column_subset_is_re_encoded_exactly_as_a_scan_would(tmp_path):
    rows = 150_000
    first = pa.table({
        "id": pa.array(range(rows), pa.int64()),
        "amount": pa.array([index / 3 for index in range(rows)]),
        "tier": pa.array(["gold", None, "tin"] * (rows // 3)).dictionary_encode(),
        "tags": pa.array([[index, None] if index % 5 else None
                          for index in range(rows)], pa.list_(pa.int32())),
        "unread": pa.array(["x"] * rows),
    })
    second = pa.table({
        "tags": pa.array([[7]] * 10, pa.list_(pa.int32())),
        "unread": pa.array(["y"] * 10),
        "amount": pa.array(range(10), pa.int32()),
        "id": pa.array(range(10), pa.int64()),
    })
    directory = tmp_path / "t"
    directory.mkdir()
    parts = [directory / "part-0.parquet", directory / "part-1.parquet"]
    pq.write_table(first, parts[0], row_group_size=100_000)
    pq.write_table(second, parts[1])
    columns = {
        "tier": Dtype.string,
        "id": Dtype.int64,
        "tags": Dtype.intlist,
        "amount": Dtype.float64,
    }

    staged = reader_over(tmp_path).prepare("t", str(tmp_path / "out"),
                                           columns=columns,
                                           mode=ReadMode.STREAMED)

    expected = _arrow_scan(
        parts, {
            "tier": pa.string(),
            "id": pa.int64(),
            "tags": pa.list_(pa.int64()),
            "amount": pa.float64()
        }, schema=pq.read_schema(parts[0]))
    assert staged.num_rows == rows + 10
    assert staged_frame(staged).equals(expected)


def test_a_hive_partition_is_re_encoded_exactly_as_a_scan_would(tmp_path):
    import pyarrow.dataset as ds

    partitions = {"dt=a": [1, 2], "dt=__HIVE_DEFAULT_PARTITION__": [3]}
    for directory, ids in partitions.items():
        write_parts(tmp_path / "t" / directory,
                    pa.table({"id": ids, "unread": ["x"] * len(ids)}))
    staged = reader_over(tmp_path).prepare("t", str(tmp_path / "out"),
                                           columns={
                                               "dt": Dtype.string,
                                               "id": Dtype.int64
                                           }, mode=ReadMode.STREAMED)
    expected = _arrow_scan(
        sorted((tmp_path / "t").rglob("*.parquet")), {
            "dt": pa.string(),
            "id": pa.int64()
        }, partitioning=ds.HivePartitioning.discover(infer_dictionary=True),
        partition_base_dir=str(tmp_path / "t"))
    assert staged_frame(staged).sort_by("id").equals(expected.sort_by("id"))


def _one_row_group(rows: int, column: pa.Array) -> pa.Table:
    return pa.table({"c": column, "unread": pa.nulls(rows, pa.int64())})


_SHAPES = {
    "narrow": lambda: _one_row_group(
        4_500_000, pa.array(np.arange(4_500_000))),
    "wide strings": lambda: _one_row_group(
        768, pa.array([f"{index:065536d}" for index in range(768)])),
    "wide lists": lambda: _one_row_group(
        6_000,
        pa.ListArray.from_arrays(
            pa.array(np.arange(0, 6_000_001, 1_000, dtype=np.int32)),
            pa.array(np.arange(6_000_000)))),
    "dictionary-encoded strings": lambda: _one_row_group(
        500_000, pa.array(["a" * 100, "b" * 100] * 250_000)),
    "one repeated wide string": lambda: _one_row_group(
        3_000, pa.array(["x" * 16_384] * 3_000)),
}


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_a_parquet_part_is_read_in_batches_bounded_by_bytes(
        tmp_path, monkeypatch, shape):
    """A Spark part's row groups run to 5.7M rows; scanning decoded each one
    whole, 3.4GiB per worker and 220GiB across a 64-core machine."""
    import pyarrow.dataset as ds
    from kumo_core.sources import object_store

    monkeypatch.setattr(object_store, "_PARQUET_BATCH_BYTES", 1 << 20)
    monkeypatch.setattr(object_store, "_PARQUET_PROBE_ROWS", 64)
    table = _SHAPES[shape]()
    # Pages of ~1MiB, as Spark writes them: a page is decoded whole.
    row_group = table.select(["c"]).nbytes
    pq.write_table(table, tmp_path / "part.parquet",
                   row_group_size=table.num_rows,
                   write_batch_size=max(1, (1 << 20) * table.num_rows //
                                        table.nbytes))
    dataset = ds.dataset(str(tmp_path / "part.parquet"))
    del table

    peak = rows = 0
    baseline = pa.total_allocated_bytes()
    for batch in object_store._fragment_batches(
            next(dataset.get_fragments()), dataset.schema,
            {"c": ds.field("c")}, ["c"]):
        peak = max(peak, pa.total_allocated_bytes() - baseline)
        rows += batch.num_rows

    assert rows == dataset.count_rows()
    assert row_group >= 32 << 20
    assert peak < 8 << 20, f"{peak:,} bytes held beside {row_group:,}"


def test_an_empty_part_is_staged_as_its_schema(tmp_path):
    write_parts(tmp_path / "t", frame(0))
    staged = reader_over(tmp_path).prepare("t", str(tmp_path / "out"),
                                           columns={"id": Dtype.int64},
                                           mode=ReadMode.STREAMED)
    assert staged.num_rows == 0
    assert staged_frame(staged).schema.names == ["id"]


@pytest.mark.parametrize("name", ["blob", "a.b"])
def test_large_remote_row_groups_leave_room_for_fewer_streams(
        tmp_path, monkeypatch, name):
    """A remote part pre-buffers its row group's selected chunks whole."""
    from kumo_core.sources import cpu
    from pyarrow.fs import SubTreeFileSystem

    monkeypatch.setattr(cpu, "_CGROUP_ROOT", tmp_path / "cgroup")
    monkeypatch.setattr(cpu, "_MEMINFO", tmp_path / "meminfo")
    (tmp_path / "meminfo").write_text(f"MemAvailable: {9 << 18} kB\n")
    ceilings = []
    real = cpu.memory_ceiling

    def observed(per_worker, worker_ceiling=None):
        ceilings.append(real(per_worker, worker_ceiling))
        return ceilings[-1]

    monkeypatch.setattr(cpu, "memory_ceiling", observed)
    blob = pa.array([os.urandom(1 << 20) for _ in range(64)])
    for table, rows in (("small", 1), ("large", 64)):
        write_parts(tmp_path / "data" / table,
                    pa.table({name: blob.slice(0, rows), "unread": range(rows)}))
        staged = ObjectStoreSourceReader(
            SubTreeFileSystem(str(tmp_path), LocalFileSystem()),
            root="data").prepare(table, str(tmp_path / table),
                                 columns={name: Dtype.binary},
                                 mode=ReadMode.STREAMED)
        assert staged.num_rows == rows

    assert ceilings == [4, 3]


@pytest.mark.parametrize('change', [
    'rewrite_older', 'same_second', 'resize', 'add', 'remove', 'rename'
])
def test_partitioned_version_tracks_every_part(root, change):
    reader = reader_over(root)
    older = root / 'hive' / 'dt=2026-01-01' / 'part-0.parquet'
    newer = root / 'hive' / 'dt=2026-01-02' / 'part-0.parquet'
    os.utime(older, ns=(1_000_000_000_000_000_000,) * 2)
    os.utime(newer, ns=(2_000_000_000_000_000_000,) * 2)
    before = reader.data_version('hive')
    assert reader.data_version('hive') == before
    if change == 'rewrite_older':
        pq.write_table(frame(3), older)
        os.utime(older, ns=(1_000_000_001_000_000_000,) * 2)
    elif change == 'same_second':
        os.utime(older, ns=(1_000_000_000_000_000_001,) * 2)
    elif change == 'resize':
        pq.write_table(frame(3), older)
        os.utime(older, ns=(1_000_000_000_000_000_000,) * 2)
    elif change == 'add':
        added = root / 'hive' / 'dt=2026-01-03' / 'part-0.parquet'
        added.parent.mkdir()
        pq.write_table(frame(2), added)
        os.utime(added, ns=(1_000_000_000_000_000_000,) * 2)
    elif change == 'remove':
        older.unlink()
    else:
        older.rename(older.with_name('renamed.parquet'))
    assert reader.data_version('hive') != before


def test_version_ignores_non_data_markers(root):
    reader = reader_over(root)
    before = reader.data_version('events')
    (root / 'events' / '_SUCCESS').write_text('updated')
    assert reader.data_version('events') == before


def test_a_row_group_too_large_for_free_memory_is_refused_up_front(
        tmp_path, monkeypatch):
    from unittest.mock import Mock

    from kumo_core.sources import cpu, object_store
    from pyarrow.fs import SubTreeFileSystem

    monkeypatch.setattr(cpu, "_CGROUP_ROOT", tmp_path / "cgroup")
    monkeypatch.setattr(cpu, "_MEMINFO", tmp_path / "meminfo")
    (tmp_path / "meminfo").write_text(f"MemAvailable: {200 << 10} kB\n")
    monkeypatch.setattr(object_store, "materialize_arrow_streams",
                        Mock(side_effect=AssertionError("read started")))
    write_parts(tmp_path / "data" / "wide", frame())
    reader = ObjectStoreSourceReader(
        SubTreeFileSystem(str(tmp_path), LocalFileSystem()), root="data")
    with pytest.raises(PermanentSourceError,
                       match=r"row group 0 of .*part-0\.parquet needs 256 MiB"
                       r", but only 200 MiB is free"):
        reader.prepare("wide", str(tmp_path / "out"), columns=COLUMNS,
                       mode=ReadMode.STREAMED)


def test_a_staging_waits_for_memory_another_staging_holds(
        tmp_path, monkeypatch):
    import threading

    from kumo_core.sources import cpu, object_store
    from pyarrow.fs import SubTreeFileSystem

    monkeypatch.setattr(cpu, "_CGROUP_ROOT", tmp_path / "cgroup")
    monkeypatch.setattr(cpu, "_MEMINFO", tmp_path / "meminfo")
    (tmp_path / "meminfo").write_text(f"MemAvailable: {700 << 10} kB\n")
    started = threading.Event()
    real = object_store.materialize_arrow_streams

    def observed(streams, destination, **kwargs):
        started.set()
        return real(streams, destination, **kwargs)

    monkeypatch.setattr(object_store, "materialize_arrow_streams", observed)
    write_parts(tmp_path / "data" / "events", frame())
    reader = ObjectStoreSourceReader(
        SubTreeFileSystem(str(tmp_path), LocalFileSystem()), root="data")
    staged = []
    with cpu.memory_share(400 << 20, lambda ceiling: 1):
        waiting = threading.Thread(target=lambda: staged.append(
            reader.prepare("events", str(tmp_path / "out"), columns=COLUMNS,
                           mode=ReadMode.STREAMED)))
        waiting.start()
        assert not started.wait(0.5)
    waiting.join(10)
    assert staged and staged[0].num_rows == ROWS


def test_a_part_rewritten_during_a_projected_read_is_retried(
        tmp_path, monkeypatch):
    from kumo_core.sources import object_store

    parts = write_parts(tmp_path / "events", frame())
    real = object_store.materialize_arrow_streams

    def rewritten(streams, destination, **kwargs):
        written = real(streams, destination, **kwargs)
        pq.write_table(frame(ROWS + 1), parts / "part-0.parquet")
        return written

    monkeypatch.setattr(object_store, "materialize_arrow_streams", rewritten)
    with pytest.raises(TransientSourceError, match="changed while it was read"):
        reader_over(tmp_path).prepare("events", str(tmp_path / "out"),
                                      columns=COLUMNS, mode=ReadMode.STREAMED)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("stage", ["footer", "data", "copy", "failed-data"])
def test_provider_identity_catches_same_size_same_mtime_replacement(
        tmp_path, monkeypatch, stage):
    parts = write_parts(tmp_path / "events", frame())
    path = parts / "part-0.parquet"
    before = path.stat()
    identity = ["original"]
    reader = reader_over(tmp_path)
    reader.object_identity = lambda path: (identity[0],)
    destination = tmp_path / "out"
    method = {"footer": "_dataset", "copy": "_copy"}.get(stage, "_stream")
    original = getattr(reader, method)

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        identity[0] = "replacement"
        # Size and timestamp alone cannot see this provider-side replacement.
        assert path.stat().st_size == before.st_size
        assert path.stat().st_mtime_ns == before.st_mtime_ns
        if stage == "failed-data":
            raise ValueError("mixed data failed decoding")
        return result

    monkeypatch.setattr(reader, method, changed)
    with pytest.raises(TransientSourceError, match="changed while it was read"):
        reader.prepare("events", str(destination),
                       columns=None if stage == "copy" else COLUMNS,
                       mode=ReadMode.STREAMED)
    assert not destination.exists()


@pytest.mark.parametrize("failure, expected", [
    (SourceAccessDeniedError, SourceAccessDeniedError),
    (SourceNotFoundError, TransientSourceError),
    (RuntimeError, RuntimeError),
])
def test_failed_identity_recheck_discards_staged_output(
        tmp_path, failure, expected):
    parts = write_parts(tmp_path / "events", frame())
    reader = reader_over(tmp_path)
    calls = []

    def identity(path):
        calls.append(path)
        if len(calls) > 1:
            raise failure("verification failed")
        return ("original",)

    reader.object_identity = identity
    destination = tmp_path / "out"
    with pytest.raises(expected):
        reader.prepare("events", str(destination), columns=COLUMNS,
                       mode=ReadMode.STREAMED)
    assert not destination.exists()
    assert (parts / "part-0.parquet").exists()
