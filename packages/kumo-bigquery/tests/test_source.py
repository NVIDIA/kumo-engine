# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the BigQuery reader does that no other reader does."""

import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fakes import (
    FakeBigQueryClient,
    FakeCredentials,
    FakeReadClient,
    FakeStorageClient,
    FakeTable,
    fake_workers,
)
from google.api_core import exceptions
from kumo_bigquery import BigQuerySourceReader, streaming
from kumo_core.sources import cpu
from kumo_core.sources.errors import (
    PermanentSourceError,
    ReconnectRequiredError,
    SourceAccessDeniedError,
    SourceNotFoundError,
    TransientSourceError,
    UnsupportedReadModeError,
)
from kumo_core.sources.types import ReadMode
from kumoapi.typing import Dtype

PROJECT, DATASET, TABLE = "bigquery-public-data", "usa_names", "usa_1910_2013"
SOURCE = "`bigquery-public-data`.`usa_names`.`usa_1910_2013`"
PATH = "projects/bigquery-public-data/datasets/usa_names/tables/usa_1910_2013"
STAGING = "gs://test-staging-bucket/ingest"
COLUMNS = {"state": Dtype.string, "number": Dtype.int64}
ROWS = 3
SCHEMA = pa.schema([("state", pa.string()), ("number", pa.int64())])
EXPORT = ("EXPORT DATA OPTIONS(uri='gs://test-staging-bucket/ingest/"
          "usa_1910_2013/part-*.parquet', format='PARQUET', overwrite=true) "
          "AS SELECT `state`, `number` FROM " + SOURCE)
VIEW = {f"{PROJECT}.{DATASET}.{TABLE}": FakeTable(table_type="VIEW")}


@pytest.fixture(autouse=True)
def workers_are_fakes(monkeypatch):
    monkeypatch.setattr(streaming, "read_clients", fake_workers)


def batch(states, numbers):
    return pa.RecordBatch.from_arrays(
        [pa.array(states, pa.string()),
         pa.array(numbers, pa.int64())], schema=SCHEMA)


def parquet_bytes(states, numbers):
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_batches([batch(states, numbers)]), sink)
    return sink.getvalue().to_pybytes()


def read_client():
    return FakeReadClient(
        {
            "stream/0": [batch(["CA"], [7]),
                         batch(["NY"], [8])],
            "stream/1": [batch(["TX"], [9])],
        }, schema=SCHEMA)


STALE = "ingest/usa_1910_2013/part-000000000009.parquet"
EXPORTED = {
    "ingest/usa_1910_2013/part-000000000000.parquet":
    parquet_bytes(["CA", "NY"], [7, 8]),
    "ingest/usa_1910_2013/part-000000000001.parquet":
    parquet_bytes(["TX"], [9]),
}


def storage_client():
    """A prefix left dirty by a half-finished earlier call."""
    return FakeStorageClient({STALE: parquet_bytes(["ZZ"], [99])})


def build(client=None, staging=None, storage=None, reads=None, **extra):
    reads = reads or read_client()
    return BigQuerySourceReader(
        client or FakeBigQueryClient(),
        project=PROJECT,
        dataset=DATASET,
        read_client=reads,
        storage_client=(storage or storage_client()) if staging else None,
        staging_uri=staging,
        credentials=FakeCredentials(reads),
        **extra,
    )


def staged_reader(client=None, storage=None):
    client = client or FakeBigQueryClient()
    storage = storage or storage_client()
    client.on_export = lambda: storage.bodies.update(EXPORTED)
    return build(client, staging=STAGING, storage=storage)


def read(reader, tmp_path, mode=ReadMode.STREAMED, **kwargs):
    dest = str(tmp_path / mode.value)
    return reader.prepare(TABLE, dest, columns=COLUMNS, mode=mode, **kwargs)


def test_staging_leads_but_only_where_a_probe_found_it_writable(tmp_path):
    """A user with neither a bucket nor a grant gets a streamed read rather
    than a failure."""
    assert build().read_modes(TABLE) == (ReadMode.STREAMED, )
    assert staged_reader().read_modes(TABLE) == (ReadMode.STAGED,
                                                 ReadMode.STREAMED)

    denied = staged_reader(
        FakeBigQueryClient(probe_raises=exceptions.Forbidden("no such grant")))
    assert denied.read_modes(TABLE) == (ReadMode.STREAMED, )
    with pytest.raises(UnsupportedReadModeError):
        read(denied, tmp_path, ReadMode.STAGED)


def test_the_probe_is_answered_once_unless_it_could_not_be_answered():
    reader = staged_reader()
    for _ in range(3):
        reader.read_modes(TABLE)
    assert len(reader.client.probes) == 1

    unanswered = staged_reader(
        FakeBigQueryClient(probe_raises=exceptions.ServiceUnavailable("blip")))
    for _ in range(3):
        assert unanswered.read_modes(TABLE) == (ReadMode.STREAMED, )
    assert len(unanswered.client.probes) == 3


def test_one_read_probes_once_even_where_the_probe_goes_unanswered(tmp_path):
    """What ``read_source`` does: ask for the modes, then read in the first.
    The guard inside the read must not ask the warehouse the same question,
    since an unanswered probe is deliberately not remembered."""
    reader = staged_reader(
        FakeBigQueryClient(probe_raises=exceptions.ServiceUnavailable("blip")))
    read(reader, tmp_path, reader.read_modes(TABLE)[0])
    assert len(reader.client.probes) == 1


def test_a_staging_bucket_without_a_client_to_fetch_it_is_refused():
    with pytest.raises(ValueError):
        BigQuerySourceReader(FakeBigQueryClient(), project=PROJECT,
                             dataset=DATASET, read_client=read_client(),
                             staging_uri=STAGING)


def test_a_physical_table_is_streamed_from_the_table_itself(tmp_path):
    """Each of the session's streams lands in parts of its own, because one
    shared writer serialises every stream behind a single encode."""
    client, reads = FakeBigQueryClient(), read_client()
    table = read(build(client, reads=reads), tmp_path)
    assert client.statements == []
    assert reads.sessions[0].table == PATH
    assert list(
        reads.sessions[0].read_options.selected_fields) == ["state", "number"]
    assert table.num_rows == ROWS
    assert all(os.path.exists(path) for path in table.paths)
    assert len(
        {os.path.basename(path).rsplit("-", 1)[0]
         for path in table.paths}) == len(reads.pages)


@pytest.mark.parametrize("tables,aliases", [(VIEW, None),
                                            (None, {
                                                "state": "region"
                                            })])
def test_a_view_and_a_rename_are_both_streamed_through_a_query_job(
        tmp_path, tables, aliases):
    client, reads = FakeBigQueryClient(tables), read_client()
    read(build(client, reads=reads), tmp_path, aliases=aliases)
    assert reads.sessions[0].table.endswith("/tables/anon0f1e")
    assert list(reads.sessions[0].read_options.selected_fields) == (
        ["region", "number"] if aliases else ["state", "number"])
    assert client.statements[-1] == "DROP TABLE _SESSION.kumo_read_target"


def test_an_alias_is_applied_in_the_sql_of_either_mode(tmp_path):
    client = FakeBigQueryClient()
    read(build(client), tmp_path, aliases={"state": "region"})
    staged = FakeBigQueryClient()
    read(staged_reader(staged), tmp_path, ReadMode.STAGED,
         aliases={"state": "region"})
    assert client.statements == [
        ("CREATE TEMP TABLE _SESSION.kumo_read_target AS "
         f"SELECT `state` AS `region`, `number` FROM {SOURCE}"),
        "DROP TABLE _SESSION.kumo_read_target",
    ]
    assert (f"AS SELECT `state` AS `region`, `number` FROM {SOURCE}"
            in staged.statements[0])


def test_a_failed_session_read_still_drops_its_temporary_table(tmp_path):
    client = FakeBigQueryClient(VIEW)
    reads = FakeReadClient(raises=RuntimeError("read failed"))
    with pytest.raises(RuntimeError, match="read failed"):
        read(build(client, reads=reads), tmp_path)
    assert client.statements[-1] == "DROP TABLE _SESSION.kumo_read_target"


def test_a_cleanup_failure_does_not_hide_the_read_failure(tmp_path, caplog):
    client = FakeBigQueryClient(
        VIEW, raises={"DROP TABLE": RuntimeError("sensitive provider text")})
    reads = FakeReadClient(raises=RuntimeError("read failed"))
    with pytest.raises(RuntimeError, match="read failed"):
        read(build(client, reads=reads), tmp_path)
    assert "RuntimeError" in caplog.text
    assert "sensitive provider text" not in caplog.text


def test_a_cleanup_failure_does_not_discard_a_completed_read(tmp_path, caplog):
    client = FakeBigQueryClient(
        VIEW, raises={"DROP TABLE": RuntimeError("sensitive provider text")})

    table = read(build(client), tmp_path)

    assert table.num_rows == ROWS
    assert "RuntimeError" in caplog.text
    assert "sensitive provider text" not in caplog.text


def test_the_projection_is_reordered_into_the_sources_own_column_order(
        tmp_path):
    client = FakeBigQueryClient()
    staged_reader(client).prepare(
        TABLE, str(tmp_path / "ordered"), columns={
            "number": Dtype.int64,
            "state": Dtype.string
        }, mode=ReadMode.STAGED)
    assert client.statements == [EXPORT]

    with pytest.raises(ValueError):
        build().prepare(TABLE, str(tmp_path / "absent"),
                        columns={"stat": Dtype.string}, mode=ReadMode.STREAMED)


def test_the_scan_is_reported_only_where_a_job_measured_one(tmp_path):
    """A direct read session runs no query job, so nothing reported a scan."""
    assert read(build(), tmp_path).scanned_bytes is None
    assert read(build(), tmp_path, aliases={
        "state": "region"
    }).scanned_bytes == 66629424
    assert read(staged_reader(), tmp_path,
                ReadMode.STAGED).scanned_bytes == 66629424


def test_the_row_count_is_reconciled_without_counting_the_source(tmp_path):
    """BigQuery charges for a COUNT(*) over a view by scanning it whole."""
    client = FakeBigQueryClient()
    read(build(client), tmp_path, expected_rows=ROWS)
    assert client.matching("COUNT(") == []


def test_the_prefix_is_cleared_before_the_export_and_again_after(tmp_path):
    """A retried half-finished call would download stale objects beside its own."""
    storage = storage_client()
    table = read(staged_reader(storage=storage), tmp_path, ReadMode.STAGED)
    assert sorted(os.path.basename(path) for path in table.paths) == [
        "part-000000000000.parquet", "part-000000000001.parquet"
    ]
    assert table.num_rows == ROWS
    assert storage.bodies == {}


def test_a_failed_export_still_clears_the_prefix(tmp_path):
    client = FakeBigQueryClient(
        raises={"EXPORT DATA": exceptions.BadRequest("no such column")})
    storage = storage_client()
    with pytest.raises(PermanentSourceError):
        read(staged_reader(client, storage), tmp_path, ReadMode.STAGED)
    assert storage.bodies == {}


def test_a_client_failure_is_reported_by_its_status_not_its_text(tmp_path):
    """API messages quote the statement that failed and the rows in it."""
    refusal = exceptions.BadRequest("Invalid value at 'token': 'hunter2'")
    client = FakeBigQueryClient(raises={"EXPORT DATA": refusal})
    with pytest.raises(PermanentSourceError) as failure:
        read(staged_reader(client), tmp_path, ReadMode.STAGED)
    reported = str(failure.value)
    assert "hunter2" not in reported
    assert "BadRequest code=400" in reported
    assert failure.value.table == TABLE
    assert "staged" in reported
    assert failure.value.__cause__ is None


@pytest.mark.parametrize(
    "raised,reported",
    [
        (exceptions.Unauthorized("invalid credentials"),
         ReconnectRequiredError),
        (exceptions.ServiceUnavailable("backend error"), TransientSourceError),
        # A staged read resolves the table before it exports, so a `NotFound`
        # from the export itself is about the destination, never the source.
        (exceptions.NotFound("no such bucket"), PermanentSourceError),
        (exceptions.Forbidden("no storage.objects.create"),
         SourceAccessDeniedError),
        (exceptions.Forbidden("query result exceeded the anonymous-table limit",
                              errors=[{
                                  "reason": "responseTooLarge"
                              }]), PermanentSourceError),
    ])
def test_a_client_failure_is_classified_by_what_the_caller_should_do(
        tmp_path, raised, reported):
    client = FakeBigQueryClient(raises={"EXPORT DATA": raised})
    with pytest.raises(reported) as failure:
        read(staged_reader(client), tmp_path, ReadMode.STAGED)
    # A denial is what lets a staged read fall back, so nothing else may
    # arrive as one.
    denied = isinstance(failure.value, SourceAccessDeniedError)
    assert denied is (reported is SourceAccessDeniedError)


def test_a_table_the_catalog_cannot_resolve_is_reported_as_missing(tmp_path):
    """The one failure a caller can act on without asking anyone for a grant:
    the name was wrong, and no mode and no retry changes that."""
    client = FakeBigQueryClient(
        table_raises=exceptions.NotFound("Not found: Table usa_1910_2013"))
    with pytest.raises(SourceNotFoundError) as failure:
        read(build(client), tmp_path)
    assert failure.value.table == TABLE

    with pytest.raises(SourceNotFoundError):
        build(client).data_version(TABLE)


def test_a_staged_object_that_vanished_is_not_a_missing_table(tmp_path):
    storage = FakeStorageClient(
        download_raises={"part": exceptions.NotFound("No such object")})
    client = FakeBigQueryClient()
    client.on_export = lambda: storage.bodies.update(EXPORTED)

    with pytest.raises(PermanentSourceError) as failure:
        read(build(client, staging=STAGING, storage=storage), tmp_path,
             ReadMode.STAGED)
    assert not isinstance(failure.value, SourceNotFoundError)


def test_a_failure_descending_only_from_exception_is_still_transient(tmp_path):
    from google.auth.exceptions import TransportError
    from google.resumable_media.common import DataCorruption
    client = FakeBigQueryClient(
        raises={"EXPORT DATA": TransportError("connection reset")})
    with pytest.raises(TransientSourceError):
        read(staged_reader(client), tmp_path, ReadMode.STAGED)

    storage = FakeStorageClient(download_raises={"part": DataCorruption(None)})
    client = FakeBigQueryClient()
    client.on_export = lambda: storage.bodies.update(EXPORTED)
    with pytest.raises(TransientSourceError):
        read(build(client, staging=STAGING, storage=storage), tmp_path,
             ReadMode.STAGED)


@pytest.mark.parametrize("ceiling,extra_clients", [(None, 1), (8, 1), (1, 0)])
def test_the_worker_ceiling_never_opens_more_readers_than_there_are_streams(
        tmp_path, monkeypatch, ceiling, extra_clients):
    """Eight is a ceiling; two streams is all the session planned, and one
    worker shares no connection so it builds none."""
    # Keep the CPU budget above the stream count so this tests the stream cap
    # even on a one-CPU host.
    monkeypatch.setattr(cpu, "cpu_budget", lambda: 8)
    built = []

    def record(count, *, credentials):
        built.append(count)
        return fake_workers(count, credentials=credentials)

    monkeypatch.setattr(streaming, "read_clients", record)
    reads = read_client()
    read(build(reads=reads), tmp_path, worker_ceiling=ceiling)
    assert built == ([extra_clients] if extra_clients else [])
    assert len(reads.pages) == 2
