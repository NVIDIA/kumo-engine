# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
from google.auth.credentials import Credentials
from kumo_bigquery.errors import rules, summary
from kumo_core.sources.cpu import encode_workers
from kumo_core.sources.diagnostics import transient
from kumo_core.sources.errors import PermanentSourceError

TEMP_TABLE = "kumo_read_target"
CREATE_TEMP = f"CREATE TEMP TABLE _SESSION.{TEMP_TABLE} AS {{sql}}"
DROP_TEMP = f"DROP TABLE _SESSION.{TEMP_TABLE}"
SERVER_CHOSEN_STREAMS = 0
OWN_CONNECTION = ("grpc.use_local_subchannel_pool", 1)
TEMP_TABLE_CLEANUP_TIMEOUT_SECONDS = 60
# AppendRows requests are capped at 10 MB.
MAX_APPEND_BYTES = 8 << 20
MAX_IN_FLIGHT = 16
MAX_APPEND_RETRIES = 5

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
    from google.cloud.bigquery_storage import BigQueryReadClient
    from google.cloud.bigquery_storage_v1.services import big_query_read

    return _own_clients(BigQueryReadClient,
                        big_query_read.transports.BigQueryReadGrpcTransport,
                        count, credentials)


def write_clients(count: int, *, credentials: Credentials) -> tuple[Any, ...]:
    from google.cloud.bigquery_storage import BigQueryWriteClient
    from google.cloud.bigquery_storage_v1.services import big_query_write

    return _own_clients(BigQueryWriteClient,
                        big_query_write.transports.BigQueryWriteGrpcTransport,
                        count, credentials)


def _own_clients(client_type: Any, transport: Any, count: int,
                 credentials: Credentials) -> tuple[Any, ...]:
    """gRPC pools subchannels globally, so channels built from identical
    arguments collapse onto one connection however many there are."""
    def channel(host: str, **kwargs: Any) -> Any:
        options = [*(kwargs.pop("options", None) or ()), OWN_CONNECTION]
        return transport.create_channel(host, options=options, **kwargs)

    return tuple(
        client_type(
            transport=transport(credentials=credentials, channel=channel))
        for _ in range(count))


def credentials_of(client: Any) -> Any:
    """Read off a private attribute whose absence is raised rather than
    tolerated: workers without it would silently share one channel."""
    credentials = getattr(getattr(client, "transport", None), "_credentials",
                          None)
    if credentials is None:
        raise PermanentSourceError(
            f"{type(client).__name__}.transport._credentials is absent "
            "or None, so no per-worker bigquery client can be built; "
            "pass credentials explicitly")
    return credentials


@contextmanager
def client_pool(
    client: Any,
    count: int,
    *,
    credentials: Credentials | None,
    build: Callable[..., tuple[Any, ...]] | None = None,
) -> Iterator[tuple[Any, ...]]:
    """The client that opened the session is worker zero; a lone worker
    shares nothing, so it needs no credential at all."""
    extra: tuple[Any, ...] = ()
    if count > 1:
        extra = (build or read_clients)(
            count - 1, credentials=(credentials if credentials is not None else
                                    credentials_of(client)))
    try:
        yield (client, *extra)
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


# Writing: the Storage Write API, the reverse of the read session above. ####


def write_streams(
    write_client: Any,
    parent: str,
    schema: pa.Schema,
    sources: Sequence[Callable[[], Iterable[pa.RecordBatch]]],
    *,
    credentials: Credentials | None = None,
    worker_ceiling: int | None = None,
) -> list[str]:
    """Appends every source's batches to pending streams on ``parent``, one
    stream and one connection per worker, and returns the finalized streams.
    Nothing is visible until :func:`commit_streams`; an uncommitted pending
    stream is discarded by BigQuery."""
    workers = min(encode_workers(worker_ceiling), max(len(sources), 1))
    groups = [sources[index::workers] for index in range(workers)]
    stop = threading.Event()

    def write(client: Any, group: Sequence[Any]) -> str:
        try:
            return _write_stream(client, parent, schema, group, stop)
        except BaseException:
            stop.set()
            raise

    with client_pool(write_client, workers, credentials=credentials,
                     build=write_clients) as clients:
        # Waits for all, so no stream is still appending when this returns.
        with ThreadPoolExecutor(workers) as pool:
            futures = [
                pool.submit(write, client, group)
                for client, group in zip(clients, groups)
            ]
    failures = [
        error for future in futures if (error := future.exception()) is not None
        and not isinstance(error, _Stopped)
    ]
    if failures:
        raise min(failures, key=lambda error: transient(error, rules()))
    return [future.result() for future in futures]


def commit_streams(write_client: Any, parent: str, names: list[str]) -> None:
    """All of ``names`` become visible at once, or none do."""
    from google.cloud.bigquery_storage_v1 import types

    response = write_client.batch_commit_write_streams(
        types.BatchCommitWriteStreamsRequest(parent=parent,
                                             write_streams=names))
    # A retried commit that already landed reports its streams as committed.
    already = types.StorageError.StorageErrorCode.STREAM_ALREADY_COMMITTED
    errors = [
        error for error in response.stream_errors if error.code != already
    ]
    if errors:
        raise PermanentSourceError("storage write api commit failed: " +
                                   "; ".join(error.error_message
                                             for error in errors))


def probe_write_stream(write_client: Any, parent: str) -> None:
    """Opens and finalizes one empty pending stream on ``parent``, which is
    what a streamed write needs beyond table access: the Storage API enabled
    and permission to write through it."""
    from google.cloud.bigquery_storage_v1 import types

    stream = write_client.create_write_stream(
        parent=parent,
        write_stream=types.WriteStream(type_=types.WriteStream.Type.PENDING))
    write_client.finalize_write_stream(name=stream.name)


class _Stopped(Exception):
    """Another stream of the same write failed."""


def _write_stream(client: Any, parent: str, schema: pa.Schema,
                  sources: Sequence[Callable[[], Iterable[pa.RecordBatch]]],
                  stop: threading.Event) -> str:
    from google.cloud.bigquery_storage_v1 import types

    stream = client.create_write_stream(
        parent=parent,
        write_stream=types.WriteStream(type_=types.WriteStream.Type.PENDING))
    template = types.AppendRowsRequest(
        write_stream=stream.name, arrow_rows=types.AppendRowsRequest.ArrowData(
            writer_schema=types.ArrowSchema(
                serialized_schema=schema.serialize().to_pybytes())))
    appender = _Appender(client, template)
    offset = 0
    try:
        for source in sources:
            for batch in source():
                for rows, count in append_payloads(batch):
                    if stop.is_set():
                        raise _Stopped()
                    appender.send(offset, rows)
                    offset += count
        appender.flush()
    finally:
        appender.close()
    written = client.finalize_write_stream(name=stream.name).row_count
    if written != offset:
        raise PermanentSourceError(
            f"storage write stream holds {written} rows, not the {offset} "
            f"appended")
    return stream.name


def append_payloads(
        batch: pa.RecordBatch,
        limit: int = MAX_APPEND_BYTES) -> Iterator[tuple[bytes, int]]:
    """Serialized slices of ``batch`` that each fit one request, with their
    row counts."""
    data = batch.serialize().to_pybytes()
    if len(data) <= limit:
        if batch.num_rows:
            yield data, batch.num_rows
        return
    if batch.num_rows == 1:
        raise PermanentSourceError(
            f"one row serializes to {len(data)} bytes, over the storage write "
            f"api request limit")
    half = batch.num_rows // 2
    yield from append_payloads(batch.slice(0, half), limit)
    yield from append_payloads(batch.slice(half), limit)


class _Appender:
    """A bounded window of offset-tagged requests in flight on one stream.
    A transient failure reconnects and resends every unconfirmed request; one
    whose offset already landed is refused with ``AlreadyExists``, so a
    retry never duplicates rows."""
    def __init__(self, client: Any, template: Any) -> None:
        self._client = client
        self._template = template
        self._stream: Any = None
        self._pending: deque = deque()

    def send(self, offset: int, rows: bytes) -> None:
        entry = [_append_request(offset, rows), None]
        self._pending.append(entry)
        self._send(entry)
        while len(self._pending) > MAX_IN_FLIGHT:
            self._confirm_oldest()

    def flush(self) -> None:
        while self._pending:
            self._confirm_oldest()

    def close(self) -> None:
        if self._stream is not None:
            with suppress(Exception):
                self._stream.close()
            self._stream = None

    def _send(self, entry: list) -> None:
        """A failure to send is kept as the entry's outcome, so it is
        retried where every other failure is."""
        try:
            entry[1] = self._open().send(entry[0])
        except Exception as error:
            entry[1] = _Failed(error)

    def _open(self) -> Any:
        from google.cloud.bigquery_storage_v1 import writer
        if self._stream is None:
            self._stream = writer.AppendRowsStream(self._client, self._template)
        return self._stream

    def _confirm_oldest(self) -> None:
        from google.api_core.exceptions import AlreadyExists

        for attempt in range(MAX_APPEND_RETRIES + 1):
            try:
                self._pending[0][1].result()
            except AlreadyExists:
                # A resent request whose rows already landed.
                pass
            except Exception as error:
                if attempt == MAX_APPEND_RETRIES or not _retryable(error):
                    raise _row_errors(error) from None
                logger.warning(
                    "storage write api append failed (%s); "
                    "resending", summary(error))
                time.sleep(min(2**attempt, 30))
                self.close()
                for entry in self._pending:
                    self._send(entry)
                continue
            self._pending.popleft()
            return


class _Failed:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    def result(self) -> Any:
        raise self._error


def _row_errors(error: BaseException) -> BaseException:
    """The reason a row was refused is on the response, not the message."""
    rows = getattr(getattr(error, "response", None), "row_errors", None)
    if not rows:
        return error
    return PermanentSourceError("storage write api refused rows: " +
                                "; ".join(row.message for row in rows[:5]))


def _append_request(offset: int, rows: bytes) -> Any:
    from google.cloud.bigquery_storage_v1 import types
    return types.AppendRowsRequest(
        offset=offset, arrow_rows=types.AppendRowsRequest.ArrowData(
            rows=types.ArrowRecordBatch(serialized_record_batch=rows)))


def _retryable(error: BaseException) -> bool:
    """As Google's own append retries: ``ABORTED`` and ``CANCELLED`` too,
    while ``FAILED_PRECONDITION`` (e.g. a wrong offset) stays fatal."""
    from google.api_core.exceptions import Aborted, Cancelled
    from google.cloud.bigquery_storage_v1.exceptions import StreamClosedError
    return (isinstance(error, (StreamClosedError, Aborted, Cancelled))
            or transient(error, rules()))
