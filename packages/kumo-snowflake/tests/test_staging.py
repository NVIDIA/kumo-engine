# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The statements the unload runs, and the objects it leaves to be fetched."""

import os
import threading

import pytest
from fakes import FakeCursor, FakeSessions
from kumo_core.sources import cpu
from kumo_core.sources.errors import PermanentSourceError
from kumo_snowflake.sql import build_projection, quote_identifier
from kumo_snowflake.staging import (
    URL_LIFETIME_SECONDS,
    StagedFile,
    create_stage,
    download,
    list_prefix,
    presigned_urls,
    unload,
)
from kumoapi.typing import Dtype

STAGE = '"SYNTHETIC_DB"."SYNTHETIC_SCHEMA"."KUMO_INGEST"'
PREFIX = "EVENTS/"
SOURCE = '"SYNTHETIC_DB"."SYNTHETIC_SCHEMA"."EVENTS"'
COLUMNS = {"ENTITY_ID": Dtype.string, "VALUE": Dtype.float}
WIDE_ENOUGH = 5
NEVER_ENOUGH = 0.5
LISTING = [
    ("kumo_ingest/EVENTS/data_0_0_0.snappy.parquet", 12, "md5", "date"),
    ("kumo_ingest/EVENTS/data_0_1_0.snappy.parquet", 8, "md5", "date"),
]


@pytest.fixture(autouse=True)
def outside_spcs(monkeypatch, tmp_path):
    """These tests exercise batched presigning without a runtime token mount."""
    monkeypatch.setattr("kumo_snowflake.staging.SPCS_TOKEN_PATH",
                        str(tmp_path / "missing-spcs-token"))


@pytest.mark.parametrize("declared,expression", [
    ("TIMESTAMP_TZ",
     'CAST(CONVERT_TIMEZONE(\'UTC\', "EVENT_TIME") AS TIMESTAMP_NTZ)'),
    ("TIMESTAMP_LTZ",
     'CAST(CONVERT_TIMEZONE(\'UTC\', "EVENT_TIME") AS TIMESTAMP_NTZ)'),
    ("TIMESTAMP_NTZ", 'CAST("EVENT_TIME" AS TIMESTAMP_NTZ)'),
    (None, 'CAST("EVENT_TIME" AS TIMESTAMP_NTZ)'),
])
def test_only_a_timestamp_carrying_an_instant_is_converted_to_utc(
        declared, expression):
    types = {"EVENT_TIME": declared} if declared else {}
    assert build_projection({"EVENT_TIME": Dtype.time}, {"EVENT_TIME": "ts"},
                            types) == f'{expression} AS "ts"'
    assert "CAST" not in build_projection(COLUMNS)


def test_a_legacy_date_dtype_does_not_leave_ltz_unloadable():
    assert build_projection({"EVENT_TIME": Dtype.date}, source_types={
        "EVENT_TIME": "TIMESTAMP_LTZ"
    }) == ('CAST(CONVERT_TIMEZONE(\'UTC\', "EVENT_TIME") AS TIMESTAMP_NTZ) '
           'AS "EVENT_TIME"')


def test_an_identifier_longer_than_snowflake_allows_is_refused():
    assert quote_identifier("A" * 255) == '"%s"' % ("A" * 255)
    with pytest.raises(ValueError):
        quote_identifier("A" * 256)


def test_the_stage_is_temporary_and_server_side_encrypted():
    """Encryption at rest is what makes a presigned fetch return plaintext."""
    cursor = FakeCursor()
    create_stage(cursor, STAGE)
    assert cursor.sql() == [
        'CREATE TEMPORARY STAGE IF NOT EXISTS '
        '"SYNTHETIC_DB"."SYNTHETIC_SCHEMA"."KUMO_INGEST" '
        "ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')"
    ]


def test_an_unload_that_reports_nothing_is_refused():
    with pytest.raises(PermanentSourceError):
        unload(FakeCursor({"COPY INTO": []}), stage=STAGE, prefix=PREFIX,
               projection='"VALUE"', source=SOURCE)


def test_listing_returns_stage_relative_paths_and_their_sizes():
    files = list_prefix(FakeCursor({"LIST ": LISTING}), STAGE, PREFIX)
    assert [(f.path, f.size) for f in files] == [
        ("EVENTS/data_0_0_0.snappy.parquet", 12),
        ("EVENTS/data_0_1_0.snappy.parquet", 8),
    ]


@pytest.mark.parametrize("listing", [
    [("kumo_ingest/OTHER/x.parquet", 1, "m", "d")],
    [],
])
def test_a_listing_that_is_empty_or_foreign_is_refused(listing):
    with pytest.raises(PermanentSourceError):
        list_prefix(FakeCursor({"LIST ": listing}), STAGE, PREFIX)


def test_presigned_paths_are_interpolated_not_bound():
    """A bound parameter reaches GET_PRESIGNED_URL unfilled and is rejected."""
    cursor = FakeCursor({"GET_PRESIGNED_URL": [("a.parquet", "https://x")]})
    presigned_urls(cursor, STAGE, ["a.parquet"])
    statement, params = cursor.statements[0]
    assert params is None
    assert "('a.parquet')" in statement
    assert (f"GET_PRESIGNED_URL('@{STAGE}', path, "
            f"{URL_LIFETIME_SECONDS})") in statement


def test_presigned_requests_are_batched():
    paths = [f"p{index}.parquet" for index in range(2500)]
    cursor = FakeCursor(
        {"GET_PRESIGNED_URL": [(path, f"https://{path}") for path in paths]})
    urls = presigned_urls(cursor, STAGE, paths)
    assert len(cursor.matching("GET_PRESIGNED_URL")) == 3
    assert len(urls) == 2500


def test_a_missing_presigned_url_is_refused():
    cursor = FakeCursor({"GET_PRESIGNED_URL": [("a.parquet", "https://x")]})
    with pytest.raises(PermanentSourceError):
        presigned_urls(cursor, STAGE, ["a.parquet", "b.parquet"])


def files_and_urls():
    files = (StagedFile("EVENTS/one.parquet",
                        12), StagedFile("EVENTS/two.parquet", 8))
    return files, {
        "EVENTS/one.parquet": "https://one",
        "EVENTS/two.parquet": "https://two",
    }


def test_every_object_is_fetched_from_the_url_minted_for_it(
        tmp_path, monkeypatch):
    files, urls = files_and_urls()
    sessions = FakeSessions({
        "https://one": b"PAR1" + b"a" * 8,
        "https://two": b"PAR1" + b"b" * 4
    })
    monkeypatch.setattr("requests.Session", sessions)
    paths = download(files, urls, str(tmp_path))
    assert [os.path.basename(path)
            for path in paths] == ["one.parquet", "two.parquet"]
    assert [os.path.getsize(path) for path in paths] == [12, 8]


def test_the_download_pool_is_as_wide_as_the_cpu_budget_allows(
        tmp_path, monkeypatch):
    """Sized from the CPU this process may use rather than from a constant, so
    a cgroup-limited worker does not open the pool a bare host would: eight
    objects are fetched at once on a budget of two and only four on one."""
    files = tuple(StagedFile(f"p{index}.parquet", 4) for index in range(8))
    urls = {staged.path: f"https://{staged.path}" for staged in files}
    bodies = {url: b"PAR1" for url in urls.values()}

    def fetch(budget, wait):
        monkeypatch.setattr(cpu, "cpu_budget", lambda: budget)
        arrivals = threading.Barrier(len(files), timeout=wait)
        sessions = FakeSessions(bodies, arrivals=arrivals)
        monkeypatch.setattr("requests.Session", sessions)
        return download(files, urls, str(tmp_path))

    assert len(fetch(2, WIDE_ENOUGH)) == len(files)
    with pytest.raises(threading.BrokenBarrierError):
        fetch(1, NEVER_ENOUGH)
