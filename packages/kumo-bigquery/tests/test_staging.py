# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The export BigQuery runs, the probe that asks whether it may, and the
objects it leaves in the bucket."""

import os
import threading

import pytest
from fakes import FakeBigQueryClient, FakeStorageClient
from google.api_core import exceptions
from google.cloud.storage.exceptions import DataCorruption
from kumo_bigquery.errors import translated
from kumo_bigquery.sql import (
    build_projection,
    order_by_schema,
    output_columns,
    quote_identifier,
    table_path,
)
from kumo_bigquery.staging import (
    DELETE_BATCH,
    StagingLocation,
    clear_prefix,
    download,
    probe,
)
from kumo_core.sources.cpu import MAX_DOWNLOAD_WORKERS
from kumo_core.sources.errors import (
    PermanentSourceError,
    SourceAccessDeniedError,
    TransientSourceError,
)
from kumo_core.sources.types import Capability, ReadMode
from kumoapi.typing import Dtype

PROJECT, DATASET, TABLE = "bigquery-public-data", "usa_names", "usa_1910_2013"
SOURCE = "`bigquery-public-data`.`usa_names`.`usa_1910_2013`"
SCHEMA = ("state", "gender", "year", "name", "number")
COLUMNS = {"state": Dtype.string, "year": Dtype.int64, "number": Dtype.int64}
BUCKET = "bigquery-connector-testing"
URI = f"gs://{BUCKET}/ingest"
OBJECT_URI = f"gs://{BUCKET}/ingest/usa_1910_2013/part-*.parquet"
PARQUET = b"PAR1" + b"payload"
PATIENCE = 30
BRIEFLY = 1


def location():
    return StagingLocation.parse(URI).child(TABLE)


def probed(client=None, storage=None):
    return probe(client or FakeBigQueryClient(), storage or FakeStorageClient(),
                 location=StagingLocation.parse(URI))


@pytest.mark.parametrize("name",
                         ["a`, (SELECT 1) AS `b", "a\\b", "a\nb", "a\rb"])
def test_an_identifier_that_could_break_out_of_its_quotes_is_refused(name):
    """BigQuery escapes none of these inside a quoted identifier, and the
    name reaching here came from the source's own schema."""
    with pytest.raises(ValueError):
        quote_identifier(name)


def test_a_table_path_names_the_storage_api_resource():
    assert table_path(PROJECT, DATASET, TABLE) == (
        "projects/bigquery-public-data/datasets/usa_names/tables/usa_1910_2013")


def test_no_dtype_this_reader_carries_needs_a_cast():
    assert build_projection(COLUMNS) == "`state`, `year`, `number`"
    assert output_columns(COLUMNS, {"year": "yr"}) == ("state", "yr", "number")


def test_the_projection_is_reordered_into_the_sources_own_column_order():
    asked = {"number": Dtype.int64, "state": Dtype.string}
    assert order_by_schema(asked, SCHEMA) == {
        "state": Dtype.string,
        "number": Dtype.int64
    }
    assert order_by_schema(COLUMNS, SCHEMA) == COLUMNS


def test_a_column_absent_from_the_source_is_refused_before_any_scan():
    with pytest.raises(ValueError) as failure:
        order_by_schema({"stat": Dtype.string}, SCHEMA)
    assert "stat" in str(failure.value)


def test_a_staging_uri_splits_into_a_bucket_and_a_prefix():
    parsed = StagingLocation.parse(URI)
    assert (parsed.bucket, parsed.prefix) == (BUCKET, "ingest")
    assert StagingLocation.parse("gs://bucket").prefix == ""
    with pytest.raises(ValueError):
        StagingLocation.parse("s3://bucket/prefix")


def test_each_table_exports_to_its_own_prefix_under_the_mandatory_wildcard():
    assert location().prefix == "ingest/usa_1910_2013"
    assert location().object_uri() == OBJECT_URI


def test_a_probe_writes_one_row_where_a_real_export_would_and_reads_none():
    client = FakeBigQueryClient()
    assert probed(client) is Capability.ALLOWED
    assert client.statements == []
    assert client.probes[0].startswith(
        f"EXPORT DATA OPTIONS(uri='gs://{BUCKET}/ingest/_probe/")
    assert client.probes[0].endswith(
        "/probe-*.parquet', format='PARQUET', overwrite=true) "
        "AS SELECT 1 AS probe")


@pytest.mark.parametrize("failure,answer", [
    (exceptions.Forbidden("access denied"), Capability.DENIED),
    (exceptions.ServiceUnavailable("backend error"), Capability.UNKNOWN),
    (TimeoutError("the job did not finish"), Capability.UNKNOWN),
])
def test_a_probe_reports_a_denial_but_never_mistakes_silence_for_one(
        failure, answer):
    assert probed(FakeBigQueryClient(probe_raises=failure)) is answer


@pytest.mark.parametrize("raises,answer", [
    (None, Capability.ALLOWED),
    (exceptions.Forbidden("access denied"), Capability.DENIED),
])
def test_a_probe_clears_the_prefix_it_wrote_to_whatever_it_found(
        raises, answer):
    """What it wrote is in the customer's own bucket."""
    client, storage = FakeBigQueryClient(
        probe_raises=raises), FakeStorageClient()
    assert probed(client, storage) is answer
    assert client.probes[0].startswith(
        f"EXPORT DATA OPTIONS(uri='gs://{BUCKET}/{storage.listed[0]}"
        f"probe-*.parquet'")


def test_a_probe_that_could_not_tidy_up_still_reports_what_it_learned():
    """The grant that matters was demonstrated by the export succeeding."""
    storage = FakeStorageClient(list_raises=exceptions.Forbidden("no listing"))
    assert probed(storage=storage) is Capability.ALLOWED


def test_clearing_a_prefix_deletes_only_that_prefix():
    other = "ingest/other_table/part-000.parquet"
    storage = FakeStorageClient({
        "ingest/usa_1910_2013/part-000.parquet": PARQUET,
        other: PARQUET
    })
    clear_prefix(storage, location())
    assert storage.bodies == {other: PARQUET}
    clear_prefix(FakeStorageClient(), location())


def staged(count):
    return {
        f"ingest/usa_1910_2013/part-{index:012d}.parquet": PARQUET
        for index in range(count)
    }


def test_clear_batches_deletes_in_parallel():
    started = threading.Barrier(2, timeout=PATIENCE)
    storage = FakeStorageClient(staged(DELETE_BATCH + 1),
                                on_batch=lambda names: started.wait())
    clear_prefix(storage, location())
    assert storage.bodies == {}
    assert sorted(map(len, storage.batches)) == [1, DELETE_BATCH]


def test_clear_tolerates_missing_object():
    objects = staged(3)
    storage = FakeStorageClient(
        objects, on_batch=lambda names: storage.bodies.pop(min(objects)))
    clear_prefix(storage, location())
    assert storage.bodies == {}


def test_clear_surfaces_403_hidden_behind_404():
    gone, kept = sorted(staged(2))
    storage = FakeStorageClient(
        staged(2), delete_raises={kept: exceptions.Forbidden("no grant")},
        on_batch=lambda names: storage.bodies.pop(gone))
    with pytest.raises(SourceAccessDeniedError), translated(
            "usa_1910_2013", ReadMode.STAGED):
        clear_prefix(storage, location())
    assert list(storage.bodies) == [kept]


def test_clear_unexplained_survivor_is_transient():
    late = "ingest/usa_1910_2013/part-late.parquet"
    storage = FakeStorageClient(
        staged(1), on_batch=lambda names: storage.bodies.update({late: PARQUET}))
    with pytest.raises(TransientSourceError):
        clear_prefix(storage, location())
    assert storage.bodies == {}


def test_a_download_takes_every_object_under_the_prefix_and_no_other_at_once(
        tmp_path):
    """One object at a time leaves the bandwidth the export was sized for
    idle, and a live run stages under a unique prefix, so only a unit test
    can see the neighbour that must be left alone."""
    started = threading.Barrier(2, timeout=PATIENCE)
    storage = FakeStorageClient(
        {
            "ingest/usa_1910_2013/part-000.parquet": PARQUET,
            "ingest/usa_1910_2013/part-001.parquet": PARQUET + b"more",
            "ingest/other_table/part-000.parquet": PARQUET,
        }, arrive=lambda name: started.wait())
    paths = download(storage, location(), str(tmp_path))
    assert sorted(os.path.basename(path) for path in paths) == [
        "part-000.parquet", "part-001.parquet"
    ]
    assert sorted(os.path.getsize(path)
                  for path in paths) == [len(PARQUET),
                                         len(PARQUET) + 4]


def test_an_object_corrupted_to_its_own_length_is_still_refused(tmp_path):
    storage = FakeStorageClient({"ingest/usa_1910_2013/a.parquet": PARQUET},
                                corrupt=lambda body: body[:-1] + b"X")
    with pytest.raises(DataCorruption):
        download(storage, location(), str(tmp_path))


def test_an_export_that_wrote_nothing_is_refused(tmp_path):
    with pytest.raises(PermanentSourceError):
        download(FakeStorageClient(), location(), str(tmp_path))


def test_a_stale_listing_handle_does_not_defeat_the_download(tmp_path):
    storage = FakeStorageClient(
        {
            "ingest/usa_1910_2013/part-000.parquet": PARQUET,
            "ingest/usa_1910_2013/part-001.parquet": PARQUET,
        }, stale_listing=True)
    assert len(download(storage, location(), str(tmp_path))) == 2


def test_one_object_that_could_not_be_fetched_cancels_the_rest(tmp_path):
    """A pool that only stopped submitting would still drain its queue, so
    every remaining object would be paid for to reach a failure already
    known. The error is reported as the client raised it.

    Every download but the failing one is held, so what a worker picked up
    before the failure landed is bounded by the pool's width however the
    scheduler ran it, and the rest are still queued to be cancelled.
    """
    failing = "ingest/usa_1910_2013/part-000.parquet"
    failure = OSError("connection reset")
    objects = MAX_DOWNLOAD_WORKERS * 2
    held = threading.Event()

    def hold(name):
        if name != failing:
            held.wait(BRIEFLY)

    storage = FakeStorageClient(
        {
            f"ingest/usa_1910_2013/part-{index:03d}.parquet": PARQUET
            for index in range(objects)
        }, download_raises={failing: failure}, arrive=hold)
    with pytest.raises(OSError) as raised:
        download(storage, location(), str(tmp_path))
    assert raised.value is failure
    assert len(storage.fetched) < objects
