# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Engine sink and its transports, against fakes of the Google clients."""

import io
import logging
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fakes import FakeStorageClient
from google.api_core import exceptions
from google.cloud import bigquery
from google.cloud.bigquery_storage import types
from kumo_bigquery import session
from kumo_bigquery.sink import BigQuerySinkBackend
from kumo_bigquery.streaming import commit_streams
from kumo_core.sources.errors import PermanentSourceError, SourceNotFoundError
from kumo_core.sources.types import ReadMode
from kumoapi.data_source import BigQueryConnectorResourceConfig
from kumoapi.jobs import WriteMode

CONNECTOR = SimpleNamespace(
    id='c', identity=None,
    config=BigQueryConnectorResourceConfig(project_id='proj-test',
                                           dataset_id='d', location='US'))
REF = 'proj-test.d.out'
ALREADY = types.StorageError.StorageErrorCode.STREAM_ALREADY_COMMITTED


class FakeJob:
    job_type = 'query'
    job_id = 'job-1'
    errors = None

    def result(self, **_):
        return self


class FakeWarehouse:
    """Records what the sink does to BigQuery, in order."""
    def __init__(self, tables=None, dataset_location='US', missing=False):
        self.tables = dict(tables or {})
        self.dataset_location = dataset_location
        self.missing = missing
        self.events = []
        self.loads = []
        self.queries = []
        self.created = []
        self.seen_at_load = []
        self.storage = None

    def get_dataset(self, name):
        if self.missing:
            raise exceptions.NotFound(name)
        return SimpleNamespace(location=self.dataset_location)

    def get_table(self, ref):
        if ref not in self.tables:
            raise exceptions.NotFound(ref)
        return self.tables[ref]

    def create_table(self, table):
        self.created.append(table)
        self.events.append(('create', table.table_id))

    def delete_table(self, table, not_found_ok=False):
        self.events.append(('delete', str(getattr(table, 'table_id',
                                                  table)).rsplit('.', 1)[-1]))

    def test_iam_permissions(self, ref, permissions):
        return {'permissions': list(permissions)}

    def query(self, sql, job_config=None):
        self.queries.append((sql, job_config))
        self.events.append(('query', sql))
        return FakeJob()

    def load_table_from_uri(self, uris, ref, job_config=None):
        self.loads.append((uris, ref, job_config))
        self.seen_at_load = sorted(self.storage.bodies)
        self.events.append(('load', uris))
        return FakeJob()

    def update_table(self, table, fields):
        self.events.append(('update', fields))


def existing_table(*fields):
    return SimpleNamespace(table_type='TABLE',
                           schema=[bigquery.SchemaField(*f)
                                   for f in fields], time_partitioning=None,
                           range_partitioning=None, clustering_fields=None)


class FakeWriteClient:
    def __init__(self, extra_rows=0, stream_errors=()):
        self.parents = {}
        self.committed = []
        self.rows = {}
        self.extra_rows = extra_rows
        self.stream_errors = list(stream_errors)

    @staticmethod
    def table_path(project, dataset, table):
        return f'projects/{project}/datasets/{dataset}/tables/{table}'

    def create_write_stream(self, parent, write_stream):
        name = f'{parent}/streams/{len(self.parents)}'
        self.parents[name] = parent
        return SimpleNamespace(name=name)

    def finalize_write_stream(self, name):
        return SimpleNamespace(row_count=self.rows.get(name, 0) +
                               self.extra_rows)

    def batch_commit_write_streams(self, request):
        self.committed.append((request.parent, list(request.write_streams)))
        return SimpleNamespace(stream_errors=self.stream_errors)


@pytest.fixture
def make_sink(monkeypatch):
    def make(warehouse, storage=None, write_client=None, **options):
        warehouse.storage = storage
        write_client = write_client or FakeWriteClient()

        class Ack:
            def result(self):
                return None

        class AppendRowsStream:
            def __init__(self, client, template):
                self.name = template.write_stream
                self.schema = pa.ipc.read_schema(
                    pa.py_buffer(
                        template.arrow_rows.writer_schema.serialized_schema))

            def send(self, request):
                rows = pa.ipc.read_record_batch(
                    pa.py_buffer(
                        request.arrow_rows.rows.serialized_record_batch),
                    self.schema).num_rows
                write_client.rows[self.name] = write_client.rows.get(
                    self.name, 0) + rows
                return Ack()

            def close(self):
                pass

        monkeypatch.setattr(
            'google.cloud.bigquery_storage_v1.writer.AppendRowsStream',
            AppendRowsStream)
        monkeypatch.setattr(BigQuerySinkBackend, '_client',
                            lambda self, connector: warehouse)
        monkeypatch.setattr(BigQuerySinkBackend, '_storage',
                            lambda self, connector: storage)
        monkeypatch.setattr(BigQuerySinkBackend, '_principal',
                            lambda self, connector: SimpleNamespace(name='n'))
        monkeypatch.setattr(BigQuerySinkBackend, '_credentials',
                            lambda self, connector: object())
        monkeypatch.setattr(session._clients, 'write_client',
                            lambda principal: write_client)
        return BigQuerySinkBackend(**options), write_client

    return make


def shard(tmp_path, name='part-0.parquet', **columns):
    path = tmp_path / name
    pq.write_table(pa.table(columns or {'id': pa.array([1, 2], pa.int32())}),
                   path)
    return str(path)


def uploaded_tables(storage):
    return [
        pq.read_table(io.BytesIO(body))
        for _, body in sorted(storage.bodies.items())
    ]


def test_preflight_names_a_missing_dataset(make_sink):
    sink, _ = make_sink(FakeWarehouse(missing=True))

    with pytest.raises(SourceNotFoundError, match='proj-test.d does not exist'):
        sink.preflight_write_files(CONNECTOR, 'out')


def test_preflight_refuses_a_dataset_elsewhere(make_sink):
    sink, _ = make_sink(FakeWarehouse(dataset_location='EU'))

    with pytest.raises(PermanentSourceError, match="location='EU'"):
        sink.preflight_write_files(CONNECTOR, 'out')


def test_streamed_overwrite_preflight_probes_a_stream_and_drops_the_probe(
        make_sink):
    warehouse = FakeWarehouse()
    sink, write_client = make_sink(warehouse)

    sink.preflight_write_files(CONNECTOR, 'out')

    assert len(write_client.parents) == 1
    assert [kind for kind, _ in warehouse.events].count('create') == 1 == [
        kind for kind, _ in warehouse.events
    ].count('delete')


def test_a_bucket_outside_the_dataset_location_falls_back_to_streamed(
        make_sink, caplog):
    storage = FakeStorageClient(location='EU')
    sink, write_client = make_sink(FakeWarehouse(), storage,
                                   staging_uri='gs://bkt/pre')

    with caplog.at_level(logging.WARNING, logger='kumo_bigquery.sink'):
        assert sink.write_modes(CONNECTOR) == (ReadMode.STREAMED, )
        sink.preflight_write_files(CONNECTOR, 'out')

    assert 'Staging bucket gs://bkt is in EU' in caplog.text
    assert write_client.parents


def test_a_colocated_bucket_stages(make_sink):
    sink, _ = make_sink(FakeWarehouse(), FakeStorageClient(location='US-EAST1'),
                        staging_uri='gs://bkt/pre')

    assert sink.write_modes(CONNECTOR) == (ReadMode.STAGED, ReadMode.STREAMED)


def test_an_explicit_staged_mode_is_refused_for_a_distant_bucket(make_sink):
    sink, _ = make_sink(FakeWarehouse(), FakeStorageClient(location='EU'),
                        staging_uri='gs://bkt/pre', mode=ReadMode.STAGED)

    with pytest.raises(Exception, match='cannot read') as caught:
        sink.preflight_write_files(CONNECTOR, 'out')
    assert type(caught.value).__name__ == 'UnsupportedReadModeError'


def test_staged_write_uploads_conformed_shards_and_loads_once(
        make_sink, tmp_path):
    storage = FakeStorageClient()
    warehouse = FakeWarehouse({
        REF:
        existing_table(('id', 'INTEGER', 'NULLABLE'),
                       ('run', 'STRING', 'NULLABLE'))
    })
    sink, _ = make_sink(warehouse, storage, staging_uri='gs://bkt/pre')
    files = [shard(tmp_path, 'a.parquet'), shard(tmp_path, 'b.parquet')]
    written = sink.write_files(CONNECTOR, 'out', files, WriteMode.APPEND,
                               extra_columns={'run': 'r1'})

    assert written == REF
    (uri, ref, config), = warehouse.loads
    prefix = uri.rsplit('/', 1)[0]
    assert uri.startswith('gs://bkt/pre/_kumo.export/')
    assert uri.endswith('/part-*.parquet')
    assert ref == REF
    assert config.write_disposition == bigquery.WriteDisposition.WRITE_APPEND
    assert [f.name for f in config.schema] == ['id', 'run']
    assert len(warehouse.seen_at_load) == 2
    assert [f'gs://bkt/{n}' for n in warehouse.seen_at_load] == [
        f'{prefix}/part-00000000.parquet', f'{prefix}/part-00000001.parquet'
    ]
    assert storage.bodies == {}


def test_staged_upload_bodies_are_conformed(make_sink, tmp_path):
    storage = FakeStorageClient()
    warehouse = FakeWarehouse()
    sink, _ = make_sink(warehouse, storage, staging_uri='gs://bkt/pre')
    seen = []
    warehouse.load_table_from_uri = lambda uris, ref, job_config=None: (
        seen.extend(uploaded_tables(storage)), FakeJob())[1]

    sink.write_files(CONNECTOR, 'out', [shard(tmp_path)],
                     extra_columns={'run': 'r1'})

    assert seen[0].schema.types == [pa.int64(), pa.string()]
    assert seen[0].to_pydict() == {'id': [1, 2], 'run': ['r1', 'r1']}
    assert storage.bodies == {}


def test_streamed_append_commits_into_the_destination(make_sink, tmp_path):
    warehouse = FakeWarehouse(
        {REF: existing_table(('id', 'INTEGER', 'NULLABLE'))})
    sink, write_client = make_sink(warehouse)

    sink.write_files(CONNECTOR, 'out', [shard(tmp_path)], WriteMode.APPEND)

    (parent, streams), = write_client.committed
    assert parent == 'projects/proj-test/datasets/d/tables/out'
    assert [write_client.rows[s] for s in streams] == [2]
    assert warehouse.events == []


def test_streamed_overwrite_commits_into_a_temp_table_then_truncates(
        make_sink, tmp_path):
    warehouse = FakeWarehouse()
    sink, write_client = make_sink(warehouse)

    sink.write_files(CONNECTOR, 'out', [shard(tmp_path)], WriteMode.OVERWRITE)

    (parent, _), = write_client.committed
    temp = parent.rsplit('/', 1)[1]
    assert temp.startswith('_kumo_export_')
    assert [kind
            for kind, _ in warehouse.events] == ['create', 'query', 'delete']
    assert warehouse.events[0][1] == temp == warehouse.events[2][1]
    sql, config = warehouse.queries[0]
    assert sql == f'SELECT * FROM `proj-test.d.{temp}`'
    assert config.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE
    assert str(config.destination) == 'proj-test.d.out'
    assert warehouse.created[0].expires is not None


def test_a_stream_holding_the_wrong_row_count_fails_before_any_replace(
        make_sink, tmp_path):
    warehouse = FakeWarehouse()
    sink, write_client = make_sink(warehouse,
                                   write_client=FakeWriteClient(extra_rows=1))

    with pytest.raises(PermanentSourceError, match='holds 3 rows, not the 2'):
        sink.write_files(CONNECTOR, 'out', [shard(tmp_path)],
                         WriteMode.OVERWRITE)

    assert write_client.committed == []
    assert [kind for kind, _ in warehouse.events] == ['create', 'delete']


def test_a_failed_commit_leaves_the_destination_alone(make_sink, tmp_path):
    error = types.StorageError(
        code=types.StorageError.StorageErrorCode.STREAM_NOT_FOUND,
        error_message='gone')
    warehouse = FakeWarehouse()
    sink, _ = make_sink(warehouse,
                        write_client=FakeWriteClient(stream_errors=[error]))

    with pytest.raises(PermanentSourceError, match='gone'):
        sink.write_files(CONNECTOR, 'out', [shard(tmp_path)],
                         WriteMode.OVERWRITE)

    assert warehouse.queries == []
    assert [kind for kind, _ in warehouse.events] == ['create', 'delete']


def test_commit_streams_ignores_already_committed_streams():
    done = types.StorageError(code=ALREADY, error_message='committed')
    client = FakeWriteClient(stream_errors=[done])

    commit_streams(client, 'parent', ['s0'])

    assert client.committed == [('parent', ['s0'])]


def test_commit_streams_raises_other_stream_errors():
    done = types.StorageError(code=ALREADY, error_message='committed')
    other = types.StorageError(
        code=types.StorageError.StorageErrorCode.STREAM_FINALIZED,
        error_message='not finalized')

    with pytest.raises(PermanentSourceError, match='not finalized'):
        commit_streams(FakeWriteClient(stream_errors=[done, other]), 'parent',
                       ['s0'])
