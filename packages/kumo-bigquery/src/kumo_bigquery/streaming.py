# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
import queue
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
from google.auth.credentials import Credentials
from kumo_bigquery.errors import summary
from kumo_core.sources.cpu import encode_workers
from kumo_core.sources.errors import PermanentSourceError

TEMP_TABLE = "kumo_read_target"
CREATE_TEMP = f"CREATE TEMP TABLE _SESSION.{TEMP_TABLE} AS {{sql}}"
DROP_TEMP = f"DROP TABLE _SESSION.{TEMP_TABLE}"
SERVER_CHOSEN_STREAMS = 0
OWN_CONNECTION = ("grpc.use_local_subchannel_pool", 1)
TEMP_TABLE_CLEANUP_TIMEOUT_SECONDS = 60

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReadTarget:
    path: str
    fields: tuple[str, ...]
    bytes_scanned: int | None = None
    release: Callable[[], None] | None = field(default=None, repr=False,
                                               compare=False)


def query_target(
    client: Any,
    *,
    sql: str,
    fields: tuple[str, ...],
    job_config: Any,
) -> ReadTarget:
    """Session tables avoid the anonymous result-size limit and are dropped
    after reading."""
    from google.cloud import bigquery

    create_config = (copy.deepcopy(job_config)
                     if job_config is not None else bigquery.QueryJobConfig())
    create_config.create_session = True
    job = client.query(CREATE_TEMP.format(sql=sql), job_config=create_config)
    job.result()
    session = job.session_info
    session_id = session.session_id if session is not None else None
    if not session_id:
        raise PermanentSourceError("query created no BigQuery session")

    def drop_table() -> None:
        drop_config = copy.deepcopy(create_config)
        drop_config.create_session = False
        drop_config.connection_properties = [
            bigquery.ConnectionProperty(key="session_id", value=session_id)
        ]
        client.query(DROP_TEMP, job_config=drop_config).result(
            timeout=TEMP_TABLE_CLEANUP_TIMEOUT_SECONDS)

    destination = job.ddl_target_table
    if destination is None:
        _release(drop_table, "query produced no destination")
        raise PermanentSourceError("query created no session temporary table")
    return ReadTarget(
        path=(f"projects/{destination.project}/datasets/"
              f"{destination.dataset_id}/tables/{destination.table_id}"),
        fields=fields,
        bytes_scanned=int(job.total_bytes_processed or 0),
        release=drop_table,
    )


def _release(release: Callable[[], None], reason: str) -> None:
    try:
        release()
    except Exception as error:  # noqa: BLE001
        logger.error("Could not drop BigQuery temporary result (%s): %s",
                     reason, summary(error))


@contextmanager
def target_streams(
    read_client: Any,
    target: ReadTarget,
    *,
    billing_project: str,
    credentials: Credentials | None = None,
    worker_ceiling: int | None = None,
) -> Iterator[tuple[tuple[Iterator[Any], ...], int]]:
    try:
        session = open_session(read_client, target,
                               billing_project=billing_project)
        workers = min(encode_workers(worker_ceiling),
                      max(len(session.streams), 1))
        with client_pool(read_client, workers,
                         credentials=credentials) as clients:
            yield stream_batches(session, clients), workers
    finally:
        if target.release is not None:
            _release(target.release, "read completed or failed")


def open_session(
    read_client: Any,
    target: ReadTarget,
    *,
    billing_project: str,
) -> Any:
    """Compression is asked for on the session because the same option on the
    request is documented as not yet available. ``max_stream_count`` stays
    server-chosen: the win is in consuming the streams it planned."""
    from google.cloud.bigquery_storage import types

    return read_client.create_read_session(
        parent=f"projects/{billing_project}",
        read_session=types.ReadSession(
            table=target.path, data_format=types.DataFormat.ARROW,
            read_options=types.ReadSession.TableReadOptions(
                selected_fields=list(target.fields),
                arrow_serialization_options=types.ArrowSerializationOptions(
                    buffer_compression=(
                        types.ArrowSerializationOptions.CompressionCodec.ZSTD)),
            )),
        max_stream_count=SERVER_CHOSEN_STREAMS,
    )


def stream_batches(
    session: Any,
    clients: Sequence[Any],
) -> tuple[Iterator[Any], ...]:
    """One iterator per stream, each holding a client for the whole of its
    read so that no two share a connection. A session with no streams still
    carries the schema an empty source writes Parquet from."""
    schema = pa.ipc.read_schema(
        pa.py_buffer(session.arrow_schema.serialized_schema))
    if not session.streams:
        return (iter((schema.empty_table(), )), )
    idle: queue.SimpleQueue = queue.SimpleQueue()
    for client in clients:
        idle.put(client)
    return tuple(
        _batches(idle, stream.name, schema) for stream in session.streams)


def read_clients(count: int, *, credentials: Credentials) -> tuple[Any, ...]:
    """gRPC pools subchannels globally, so channels built from identical
    arguments collapse onto one connection however many there are."""
    from google.cloud.bigquery_storage import BigQueryReadClient
    from google.cloud.bigquery_storage_v1.services import big_query_read

    transport = big_query_read.transports.BigQueryReadGrpcTransport

    def channel(host: str, **kwargs: Any) -> Any:
        options = [*(kwargs.pop("options", None) or ()), OWN_CONNECTION]
        return transport.create_channel(host, options=options, **kwargs)

    return tuple(
        BigQueryReadClient(
            transport=transport(credentials=credentials, channel=channel))
        for _ in range(count))


def credentials_of(read_client: Any) -> Any:
    """Read off a private attribute whose absence is raised rather than
    tolerated: workers without it would silently share one channel."""
    credentials = getattr(getattr(read_client, "transport", None),
                          "_credentials", None)
    if credentials is None:
        raise PermanentSourceError(
            f"{type(read_client).__name__}.transport._credentials is absent "
            "or None, so no per-worker bigquery read client can be built; "
            "pass credentials to BigQuerySourceReader")
    return credentials


@contextmanager
def client_pool(
    read_client: Any,
    count: int,
    *,
    credentials: Credentials | None,
) -> Iterator[tuple[Any, ...]]:
    """The client that opened the session is worker zero; a lone worker
    shares nothing, so it needs no credential at all."""
    extra: tuple[Any, ...] = ()
    if count > 1:
        extra = read_clients(
            count - 1, credentials=(credentials if credentials is not None else
                                    credentials_of(read_client)))
    try:
        yield (read_client, *extra)
    finally:
        for client in extra:
            client.transport.close()


@contextmanager
def _borrowed(idle: queue.SimpleQueue) -> Iterator[Any]:
    client = idle.get()
    try:
        yield client
    finally:
        idle.put(client)


def _batches(
    idle: queue.SimpleQueue,
    name: str,
    schema: pa.Schema,
) -> Iterator[pa.RecordBatch]:
    """Decoded here rather than by ``page.to_arrow``, which is pending
    deprecation and switches implementation when ``pandas_gbq`` is
    importable."""
    with _borrowed(idle) as client:
        for message in client.read_rows(name):
            payload = message.arrow_record_batch.serialized_record_batch
            if payload:
                yield pa.ipc.read_record_batch(pa.py_buffer(payload), schema)
