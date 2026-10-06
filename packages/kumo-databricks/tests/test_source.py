# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the Databricks reader does that no other reader does."""

import io
from contextlib import nullcontext

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from databricks.sdk.errors import PermissionDenied, Unauthenticated
from databricks.sql.exc import (
    DataError,
    OperationalError,
    ServerOperationError,
)
from fakes import FakeConnection, FakeCursor, FakeFiles, FakeWorkspaceClient
from kumo_core.sources.errors import (
    PermanentSourceError,
    ReconnectRequiredError,
    RowCountMismatchError,
    SourceAccessDeniedError,
    SourceError,
    TransientSourceError,
)
from kumo_core.sources.types import Attestation, ReadMode
from kumo_databricks import DatabricksSourceReader
from kumoapi.typing import Dtype

CATALOG, SCHEMA, TABLE = "samples", "tpch", "customer"
SOURCE = "`samples`.`tpch`.`customer`"
VOLUME = "/Volumes/kumo_test_catalogue/amazon_10k/vol/out"
ROOT = f"{VOLUME}/{TABLE}"
#: An attempt other than the one under test: an overlapping read, or one
#: interrupted before it could clear up after itself.
ELSEWHERE = f"{ROOT}/{'0' * 32}"
ROWS = 7_500
COLUMNS = {"c_custkey": Dtype.int64, "c_acctbal": Dtype.float}
PROJECTION = "`c_custkey`, CAST(`c_acctbal` AS DOUBLE) AS `c_acctbal`"
DESCRIPTION = [
    ("c_custkey", "bigint", None),
    ("c_acctbal", "decimal(38,2)", None),
    ("", "", ""),
    ("# Detailed Table Information", "", ""),
    ("Type", "MANAGED", ""),
]
HISTORY = [(947, "2026-08-27T15:07:20", "WRITE")]


def arrow(rows=ROWS):
    return pa.table({
        "c_custkey": list(range(rows)),
        "c_acctbal": [float(index) for index in range(rows)]
    })


def parquet_bytes(rows):
    sink = io.BytesIO()
    pq.write_table(arrow(rows), sink)
    return sink.getvalue()


def directory_in(statement):
    """The directory an unload writes to; named per attempt, so it cannot be
    known before the statement that carries it."""
    return statement.split("DIRECTORY '", 1)[1].split("'", 1)[0]


class StagingCursor(FakeCursor):
    """A warehouse that materialises its unload into the fake volume, at
    whatever directory the statement names.

    ``during`` runs once, with the unload landed and nothing downloaded yet,
    which is where a second read of the same table overlaps this one.
    """

    files = None

    def __init__(self, *args, during=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.during = during
        self.locations = []

    def execute(self, sql):
        result = super().execute(sql)
        if "INSERT OVERWRITE" in sql and self.files is not None:
            location = directory_in(sql)
            self.locations.append(location)
            self.files.objects[f"{location}/_SUCCESS"] = b""
            self.files.objects[f"{location}/part-00000-tid-1.c000"
                               f".snappy.parquet"] = parquet_bytes(ROWS)
            during, self.during = self.during, None
            if during is not None:
                during()
        return result


def volume_files(stale=False):
    objects = ({
        f"{ELSEWHERE}/part-09999-tid-0.c000.snappy.parquet":
        parquet_bytes(11)
    } if stale else {})
    return FakeFiles(objects)


def warehouse(raises=None, during=None):
    return StagingCursor(
        {
            "DESCRIBE EXTENDED": DESCRIPTION,
            "DESCRIBE HISTORY": HISTORY
        }, raises=raises, arrow=arrow(), during=during)


def build(cursor=None, files=None, **extra):
    cursor = warehouse() if cursor is None else cursor
    if files is not None:
        cursor.files = files
    return DatabricksSourceReader(
        FakeConnection(cursor),
        catalog=CATALOG,
        schema=SCHEMA,
        workspace_client=FakeWorkspaceClient(files) if files else None,
        staging_volume_path=VOLUME if files else None,
        **extra,
    )


def read(tmp_path, mode, cursor=None, files=None, **kwargs):
    dest = str(tmp_path / mode.value)
    return build(cursor, files).prepare(TABLE, dest, columns=COLUMNS, mode=mode,
                                        **kwargs)


def test_streaming_is_always_offered_and_staging_leads_where_a_volume_is():
    cursor = warehouse()
    assert build(cursor).read_modes(TABLE) == (ReadMode.STREAMED, )
    assert cursor.sql() == []

    assert build(files=volume_files()).read_modes(TABLE) == (ReadMode.STAGED,
                                                             ReadMode.STREAMED)


@pytest.mark.parametrize("mode", list(ReadMode))
def test_neither_mode_reports_a_count_the_warehouse_stands_behind(
        tmp_path, mode):
    """``INSERT OVERWRITE DIRECTORY`` returns no row count, so staging attests
    nothing that streaming does not."""
    staged = read(tmp_path, mode, files=volume_files())
    assert staged.num_rows == ROWS
    assert staged.rows_attested_by is Attestation.LOCAL


def test_a_streamed_read_selects_the_projection_and_nothing_else(tmp_path):
    cursor = warehouse()
    read(tmp_path, ReadMode.STREAMED, cursor)
    assert cursor.matching("SELECT") == [f"SELECT {PROJECTION} FROM {SOURCE}"]


def test_a_staged_read_unloads_the_projection_to_the_volume_directory(tmp_path):
    cursor = warehouse()
    read(tmp_path, ReadMode.STAGED, cursor, volume_files())
    location, = cursor.locations
    assert cursor.matching("INSERT OVERWRITE DIRECTORY") == [
        f"INSERT OVERWRITE DIRECTORY '{location}' USING PARQUET "
        f"SELECT {PROJECTION} FROM {SOURCE}"
    ]


def test_the_unload_names_a_directory_of_its_own_under_the_tables_one(tmp_path):
    first, second = warehouse(), warehouse()
    read(tmp_path / "first", ReadMode.STAGED, first, volume_files())
    read(tmp_path / "second", ReadMode.STAGED, second, volume_files())
    location, = first.locations
    assert location.startswith(f"{ROOT}/")
    attempt = location.removeprefix(f"{ROOT}/")
    assert attempt.isalnum() and len(attempt) == 32
    assert second.locations != first.locations


@pytest.mark.parametrize("mode", list(ReadMode))
def test_an_alias_renames_the_column_the_warehouse_returns(tmp_path, mode):
    cursor = warehouse()
    read(tmp_path, mode, cursor, volume_files(),
         aliases={"c_custkey": "customer"})
    assert any("`c_custkey` AS `customer`" in statement
               for statement in cursor.sql())


def test_another_attempts_parquet_is_neither_downloaded_nor_deleted(tmp_path):
    """``INSERT OVERWRITE DIRECTORY`` has no atomic commit, so an interrupted
    write leaves its parquet behind."""
    files = volume_files(stale=True)
    staged = read(tmp_path, ReadMode.STAGED, files=files)
    assert staged.num_rows == ROWS
    assert sum(pq.ParquetFile(path).metadata.num_rows
               for path in staged.paths) == ROWS
    assert list(files.objects) == list(volume_files(stale=True).objects)


def test_the_staging_directory_is_left_empty_by_a_read_that_worked(tmp_path):
    files = volume_files()
    read(tmp_path, ReadMode.STAGED, files=files)
    assert files.objects == {}


@pytest.mark.parametrize("asked,unloading,reported", [
    ({
        "expected_rows": ROWS + 1
    }, None, RowCountMismatchError),
    ({}, OperationalError("lost"), TransientSourceError),
])
def test_a_staged_read_that_failed_still_clears_the_staging_directory(
        tmp_path, asked, unloading, reported):
    files = volume_files(stale=True)
    raises = {"INSERT OVERWRITE": unloading} if unloading else None
    with pytest.raises(reported):
        read(tmp_path, ReadMode.STAGED, warehouse(raises=raises), files,
             **asked)
    assert list(files.objects) == list(volume_files(stale=True).objects)


@pytest.mark.parametrize("read_raises,outcome", [
    (None, nullcontext()),
    (OSError("reset"), pytest.raises(TransientSourceError)),
])
def test_every_downloaded_stream_is_closed(tmp_path, read_raises, outcome):
    files = volume_files()
    files.read_raises = read_raises
    with outcome:
        read(tmp_path, ReadMode.STAGED, files=files)
    assert files.streams
    assert all(stream.closed_by_reader for stream in files.streams)


def test_a_driver_failure_is_reported_by_its_code_not_its_text(tmp_path):
    """Driver messages carry the statement that failed and the data in it."""
    cursor = warehouse(
        raises={
            "SELECT":
            ServerOperationError("[UNRESOLVED_COLUMN] cannot resolve "
                                 "`token` in SELECT `token`='hunter2'")
        })
    with pytest.raises(PermanentSourceError) as failure:
        read(tmp_path, ReadMode.STREAMED, cursor)
    reported = str(failure.value)
    assert "hunter2" not in reported
    assert "ServerOperationError code=UNRESOLVED_COLUMN" in reported
    assert failure.value.table == TABLE
    assert "streamed" in reported
    assert failure.value.__cause__ is None


@pytest.mark.parametrize("raised,reported", [
    (DataError("some result batches must have been skipped"),
     PermanentSourceError),
    (ServerOperationError("TABLE_OR_VIEW_NOT_FOUND"), PermanentSourceError),
    (Unauthenticated("token expired"), ReconnectRequiredError),
    (ServerOperationError("Invalid access token"), ReconnectRequiredError),
    (OperationalError("connection reset"), TransientSourceError),
])
def test_a_driver_failure_is_classified_by_what_the_caller_should_do(
        tmp_path, raised, reported):
    with pytest.raises(reported):
        read(tmp_path, ReadMode.STREAMED, warehouse(raises={"SELECT": raised}))


@pytest.mark.parametrize("failing,denied", [
    (PermissionDenied("no WRITE VOLUME on the staging volume"), True),
    (ServerOperationError("[INSUFFICIENT_PERMISSIONS] on `out`"), True),
    (ServerOperationError("[INSUFFICIENT_PRIVILEGES] to write"), True),
    (ServerOperationError("[TABLE_OR_VIEW_NOT_FOUND] no such table"), False),
    (OperationalError("connection reset"), False),
])
def test_only_a_missing_grant_is_what_falls_back_to_streaming(
        tmp_path, failing, denied):
    with pytest.raises(SourceError) as failure:
        read(tmp_path, ReadMode.STAGED,
             warehouse(raises={"INSERT OVERWRITE": failing}), volume_files())
    assert isinstance(failure.value, SourceAccessDeniedError) is denied


def test_the_worker_ceiling_is_what_reaches_the_volume_download_pool(
        tmp_path, monkeypatch):
    from kumo_databricks import staging
    seen = []
    fetch_all = staging.fetch_all

    def record(*args, worker_ceiling=None, **kwargs):
        seen.append(worker_ceiling)
        return fetch_all(*args, worker_ceiling=worker_ceiling, **kwargs)

    monkeypatch.setattr(staging, "fetch_all", record)
    for ceiling in (None, 2):
        reader = build(files=volume_files())
        reader.prepare(TABLE, str(tmp_path / f"{ceiling}"), columns=COLUMNS,
                       mode=ReadMode.STAGED, worker_ceiling=ceiling)
    assert seen == [None, 2]


def test_two_overlapping_reads_of_one_table_never_stage_over_each_other(
        tmp_path):
    """One volume serves every read, so two reads of the same table -- a retry
    that overlaps its predecessor, or two graphs built at once -- meet in it."""
    files = volume_files()
    finished = []

    def second_read():
        cursor = warehouse()
        finished.append(
            (cursor, read(tmp_path / "second", ReadMode.STAGED, cursor, files)))

    first = warehouse(during=second_read)
    landed = read(tmp_path / "first", ReadMode.STAGED, first, files)
    second, overlapped = finished[0]

    assert set(first.locations).isdisjoint(second.locations)
    assert landed.num_rows == overlapped.num_rows == ROWS
    assert len(landed.paths) == len(overlapped.paths) == 1
    assert files.objects == {}
