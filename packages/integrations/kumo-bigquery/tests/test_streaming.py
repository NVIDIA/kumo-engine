# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opening a read session, consuming its streams, and standing in for one."""

import os
import threading

import pyarrow as pa
import pytest
from fakes import DESTINATION, FakeBigQueryClient, FakeReadClient
from google.auth.credentials import AnonymousCredentials
from google.cloud.bigquery_storage import types
from kumo_bigquery.streaming import (
    DROP_TEMP,
    TEMP_TABLE_CLEANUP_TIMEOUT_SECONDS,
    ReadTarget,
    client_pool,
    credentials_of,
    open_session,
    query_target,
    read_clients,
    stream_batches,
)
from kumo_core.sources import cpu
from kumo_core.sources.arrow_parquet import materialize_arrow_streams
from kumo_core.sources.errors import PermanentSourceError

BILLING = "example-project"
PATH = "projects/bigquery-public-data/datasets/usa_names/tables/usa_1910_2013"
SOURCE = "`bigquery-public-data`.`usa_names`.`usa_1910_2013`"
FIELDS = ("state", "number")
SCHEMA = pa.schema([("state", pa.string()), ("number", pa.int64())])
STREAMS = 3
ROWS = 4
PATIENCE = 30


def batch(states, numbers):
    return pa.RecordBatch.from_arrays(
        [pa.array(states, pa.string()),
         pa.array(numbers, pa.int64())], schema=SCHEMA)


def read_client(pages=None, arrive=None):
    return FakeReadClient({
        "stream/0": [batch(["CA"], [7]), batch(["NY"], [8])],
        "stream/1": [batch(["TX"], [9])],
        "stream/2": [batch(["WA"], [10])],
    } if pages is None else pages, schema=SCHEMA, arrive=arrive)


def session(client, fields=FIELDS):
    return open_session(client, ReadTarget(path=PATH, fields=fields),
                        billing_project=BILLING)


def test_the_session_reads_the_named_fields_as_arrow_compressed_on_the_wire():
    client = read_client()
    session(client, fields=("region", ))
    options = client.sessions[0].read_options
    assert client.sessions[0].table == PATH
    assert client.sessions[0].data_format == types.DataFormat.ARROW
    assert list(options.selected_fields) == ["region"]
    assert options.arrow_serialization_options.buffer_compression == (
        types.ArrowSerializationOptions.CompressionCodec.ZSTD)
    assert not options.response_compression_codec
    assert client.parents == ["projects/example-project"]
    assert client.stream_counts == [0]


def test_every_stream_is_read_at_once_and_writes_a_part_of_its_own(
        tmp_path, monkeypatch):
    """The budget is pinned because a ceiling only caps: a narrower runner
    would otherwise open fewer readers than there are streams."""
    monkeypatch.setattr(cpu, "cpu_budget", lambda: STREAMS)
    started = threading.Barrier(STREAMS, timeout=PATIENCE)
    client = read_client(arrive=started.wait)
    written = materialize_arrow_streams(
        stream_batches(session(client), (client, ) * STREAMS), str(tmp_path),
        worker_ceiling=STREAMS)
    assert written.rows == ROWS
    assert len(
        {os.path.basename(path).rsplit("-", 1)[0]
         for path in written.paths}) == STREAMS


def test_a_session_with_no_streams_still_carries_the_schema():
    client = read_client(pages={})
    produced = [
        part for stream in stream_batches(session(client), (client, ))
        for part in stream
    ]
    assert [part.schema for part in produced] == [SCHEMA]
    assert sum(part.num_rows for part in produced) == 0


def test_each_worker_gets_a_client_of_its_own_built_from_one_credential():
    credentials = AnonymousCredentials()
    clients = read_clients(STREAMS, credentials=credentials)
    try:
        assert len({id(c.transport.grpc_channel) for c in clients}) == STREAMS
        assert all(credentials_of(client) is credentials for client in clients)
    finally:
        for client in clients:
            client.transport.close()


@pytest.mark.parametrize("client", [object(), FakeReadClient()])
def test_a_client_exposing_no_credential_is_refused_not_serialised(client):
    with pytest.raises(PermanentSourceError) as failure:
        credentials_of(client)
    assert "_credentials" in str(failure.value)


def test_a_lone_worker_needs_no_credential_because_it_shares_nothing():
    """A single-stream table would otherwise fail for want of a channel it
    was never going to build."""
    client = read_client()
    with client_pool(client, 1, credentials=None) as clients:
        assert clients == (client, )


def test_a_query_target_reads_and_releases_a_session_temporary_table():
    client = FakeBigQueryClient(bytes_processed=66629424)
    resolved = query_target(client,
                            sql=f"SELECT `state` AS `region` FROM {SOURCE}",
                            fields=("region", ), job_config=None)
    assert client.statements == [
        ("CREATE TEMP TABLE _SESSION.kumo_read_target AS "
         f"SELECT `state` AS `region` FROM {SOURCE}")
    ]
    assert resolved.path == (f"projects/{DESTINATION.project}/datasets/"
                             f"{DESTINATION.dataset_id}/tables/"
                             f"{DESTINATION.table_id}")
    assert resolved.fields == ("region", )
    assert resolved.bytes_scanned == 66629424
    assert resolved.release is not None
    resolved.release()
    assert client.statements[-1] == ("DROP TABLE _SESSION.kumo_read_target")
    assert client.job_configs[0].create_session is True
    properties = client.job_configs[-1].connection_properties
    assert [(item.key, item.value)
            for item in properties] == [("session_id", "session-0f1e")]
    assert client.result_timeouts == [None, TEMP_TABLE_CLEANUP_TIMEOUT_SECONDS]


def test_a_query_without_a_session_id_is_refused():
    client = FakeBigQueryClient(session_id=None)
    with pytest.raises(PermanentSourceError,
                       match="created no BigQuery session"):
        query_target(client, sql=f"SELECT `state` FROM {SOURCE}", fields=FIELDS,
                     job_config=None)
    assert client.statements == [
        "CREATE TEMP TABLE _SESSION.kumo_read_target AS "
        f"SELECT `state` FROM {SOURCE}"
    ]


def test_a_query_that_wrote_no_destination_is_refused():
    client = FakeBigQueryClient(destination=None)
    with pytest.raises(PermanentSourceError):
        query_target(client, sql=f"SELECT `state` FROM {SOURCE}", fields=FIELDS,
                     job_config=None)
    assert client.statements[-1] == "DROP TABLE _SESSION.kumo_read_target"


def test_a_missing_destination_error_survives_cleanup_failure(caplog):
    client = FakeBigQueryClient(
        destination=None,
        raises={DROP_TEMP: RuntimeError("private cleanup detail")},
    )

    with pytest.raises(PermanentSourceError, match="created no session"):
        query_target(client, sql=f"SELECT `state` FROM {SOURCE}", fields=FIELDS,
                     job_config=None)

    assert "produced no destination" in caplog.text
    assert "private cleanup detail" not in caplog.text


@pytest.mark.parametrize('phase', ['session', 'read', 'writer', 'cancel'])
def test_target_release_covers_acquisition_and_consumer_failures(phase):
    import asyncio
    from unittest.mock import Mock

    from kumo_bigquery.streaming import target_streams

    failure = (asyncio.CancelledError()
               if phase == 'cancel' else RuntimeError(phase))
    client = read_client()
    if phase == 'session':
        client.create_read_session = Mock(side_effect=failure)
    elif phase == 'read':
        client.read_rows = Mock(side_effect=failure)
    release = Mock()
    target = ReadTarget(PATH, FIELDS, release=release)

    with pytest.raises(type(failure)):
        with target_streams(client, target, billing_project=BILLING,
                            worker_ceiling=1) as (streams, workers):
            assert workers == 1
            next(streams[0])
            raise failure

    release.assert_called_once_with()
