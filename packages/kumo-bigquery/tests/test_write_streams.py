# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Storage Write API batching, retries, and stream distribution."""

import random
from types import SimpleNamespace

import pyarrow as pa
import pytest
from google.api_core import exceptions
from google.cloud.bigquery_storage import types
from google.cloud.bigquery_storage_v1.exceptions import StreamClosedError
from kumo_bigquery import streaming
from kumo_bigquery.streaming import _Appender, append_payloads, write_streams
from kumo_core.sources.errors import PermanentSourceError

CREDENTIALS = object()
SCHEMA = pa.schema([('id', pa.int64()), ('name', pa.string())])


def batch(count, start=0):
    return pa.RecordBatch.from_pylist([{
        'id': start + i,
        'name': f'row-{start + i}'
    } for i in range(count)], schema=SCHEMA)


def decode(payload):
    return pa.ipc.read_record_batch(pa.py_buffer(payload), SCHEMA)


@pytest.mark.parametrize('limit', [400, 700])
def test_append_payloads_fit_the_limit_and_preserve_rows(limit):
    original = batch(40)

    payloads = list(append_payloads(original, limit))

    assert len(payloads) > 1
    assert all(len(data) <= limit for data, _ in payloads)
    decoded = [decode(data) for data, _ in payloads]
    assert [b.num_rows for b in decoded] == [count for _, count in payloads]
    assert pa.Table.from_batches(decoded).equals(
        pa.Table.from_batches([original]))


def test_append_payloads_rejects_a_row_over_the_limit():
    with pytest.raises(PermanentSourceError, match='one row'):
        list(append_payloads(batch(3), 10))


def test_append_payloads_yields_nothing_for_an_empty_batch():
    assert list(append_payloads(batch(0))) == []


class Wire:
    """In-memory stand-in for the service behind ``AppendRowsStream``."""
    def __init__(self, fail_acks=()):
        self.accepted = {}
        self.rows = {}
        self.rejected = 0
        self.fail_acks = set(fail_acks)
        self.sends = 0

    def append(self, stream, request):
        offsets = self.accepted.setdefault(stream, [])
        self.sends += 1
        if request.offset in offsets:
            self.rejected += 1
            return Ack(exceptions.AlreadyExists('offset exists'))
        offsets.append(request.offset)
        payload = request.arrow_rows.rows.serialized_record_batch
        if len(payload) > 1:
            count = pa.ipc.read_record_batch(pa.py_buffer(payload),
                                             SCHEMA).num_rows
            self.rows[stream] = self.rows.get(stream, 0) + count
        if self.sends in self.fail_acks:
            return Ack(exceptions.ServiceUnavailable('ack lost'))
        return Ack(None)


class Ack:
    def __init__(self, error):
        self.error = error

    def result(self):
        if self.error is not None:
            raise self.error


@pytest.fixture
def wire(monkeypatch):
    state = Wire()

    class FakeAppendRowsStream:
        def __init__(self, client, template):
            self.name = template.write_stream

        def send(self, request):
            return state.append(self.name, request)

        def close(self):
            pass

    monkeypatch.setattr(
        'google.cloud.bigquery_storage_v1.writer.'
        'AppendRowsStream', FakeAppendRowsStream)
    monkeypatch.setattr(streaming.time, 'sleep', lambda _: None)
    return state


def test_appender_resends_without_duplicating_rows(wire):
    wire.fail_acks = {3}
    appender = _Appender(None, types.AppendRowsRequest(write_stream='s'))

    for offset in range(5):
        appender.send(offset, b'x')
    appender.flush()

    assert sorted(wire.accepted['s']) == [0, 1, 2, 3, 4]
    assert wire.rejected > 0


def test_appender_raises_non_transient_errors(wire):
    appender = _Appender(None, types.AppendRowsRequest(write_stream='s'))
    wire.append = lambda stream, request: Ack(exceptions.BadRequest('bad'))
    appender.send(0, b'x')

    with pytest.raises(exceptions.BadRequest):
        appender.flush()


def test_appender_does_not_retry_failed_precondition(wire):
    appender = _Appender(None, types.AppendRowsRequest(write_stream='s'))
    sends = []
    wire.append = lambda stream, request: (sends.append(
        request.offset), Ack(exceptions.FailedPrecondition('offset')))[1]
    appender.send(0, b'x')

    with pytest.raises(exceptions.FailedPrecondition):
        appender.flush()
    assert sends == [0]


FAULTS = (exceptions.ServiceUnavailable, exceptions.Aborted,
          exceptions.Cancelled, StreamClosedError)
REQUESTS = 200
SEEDS = 50


def run_faulty_stream(monkeypatch, seed):
    """Every send or ack fails 10% of the time, and the request may or may
    not have landed first. A failure breaks the connection, as it does on the
    service, so later requests on it fail until the appender reconnects. The
    service refuses a gap in offsets and acknowledges a landed one with
    ``AlreadyExists``."""
    rng = random.Random(seed)
    landed = []

    class Acked:
        def __init__(self, error=None):
            self.error = error

        def result(self):
            if self.error is not None:
                raise self.error

    class FaultyStream:
        def __init__(self, client, template):
            self.broken = False

        def send(self, request):
            if self.broken:
                return Acked(StreamClosedError('connection broken'))
            offset = request.offset
            if offset < len(landed):
                return Acked(exceptions.AlreadyExists('offset exists'))
            if offset > len(landed):
                return Acked(exceptions.OutOfRange(f'gap at {offset}'))
            if rng.random() >= 0.1:
                landed.append(offset)
                return Acked()
            self.broken = True
            if rng.random() < 0.5:
                landed.append(offset)
            error = rng.choice(FAULTS)('injected')
            if rng.random() < 0.5:
                raise error
            return Acked(error)

        def close(self):
            pass

    monkeypatch.setattr(
        'google.cloud.bigquery_storage_v1.writer.AppendRowsStream',
        FaultyStream)
    monkeypatch.setattr(streaming.time, 'sleep', lambda _: None)
    appender = _Appender(None, types.AppendRowsRequest(write_stream='s'))
    for offset in range(REQUESTS):
        appender.send(offset, b'x')
    appender.flush()
    return landed


def test_appender_lands_every_offset_once_under_random_faults(monkeypatch):
    failures = {}
    for seed in range(SEEDS):
        try:
            landed = run_faulty_stream(monkeypatch, seed)
            assert landed == list(range(REQUESTS))
        except Exception as error:
            failures[seed] = repr(error)

    print(f'fault harness: {len(failures)}/{SEEDS} seeds failed')
    assert not failures


class FakeWriteClient:
    def __init__(self):
        self.created = []
        self.finalized = []
        self.transport = SimpleNamespace(close=lambda: None)

    def create_write_stream(self, parent, write_stream):
        name = f'{parent}/streams/{len(self.created)}'
        self.created.append(name)
        return SimpleNamespace(name=name)

    def finalize_write_stream(self, name):
        self.finalized.append(name)
        return SimpleNamespace(row_count=self.wire.rows.get(name, 0))


@pytest.fixture
def clients(monkeypatch, wire):
    monkeypatch.setattr(streaming, 'encode_workers',
                        lambda ceiling: ceiling or 1)
    shared = FakeWriteClient()
    shared.wire = wire
    monkeypatch.setattr(streaming, 'write_clients', lambda count, credentials:
                        (shared, ) * count)
    return shared


def source(start, count=2):
    return lambda: [batch(count, start)]


def test_write_streams_spreads_sources_over_worker_streams(clients, wire):
    sources = [source(0), source(10), source(20)]

    names = write_streams(clients, 'p', SCHEMA, sources,
                          credentials=CREDENTIALS, worker_ceiling=2)

    assert len(names) == 2 == len(set(names))
    assert sorted(clients.finalized) == sorted(names)
    assert sorted(len(wire.accepted[name]) for name in names) == [1, 2]


def test_write_streams_uses_no_more_streams_than_sources(clients, wire):
    names = write_streams(clients, 'p', SCHEMA, [source(0)],
                          credentials=CREDENTIALS, worker_ceiling=2)

    assert len(names) == 1


def test_write_streams_raises_the_failing_source_error(clients, wire):
    def broken():
        raise exceptions.BadRequest('bad source')

    def slow():
        return [batch(2)] * 50

    with pytest.raises(exceptions.BadRequest, match='bad source'):
        write_streams(clients, 'p', SCHEMA, [slow, broken],
                      credentials=CREDENTIALS, worker_ceiling=2)
