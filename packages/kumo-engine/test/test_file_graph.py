# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
import pytest
from kumo_core.diskgraph.snapshot import DiskgraphIngestResult
from kumo_core.sources.object_store_access import ObjectStoreAccess
from kumoapi.typing import Dtype, Stype
from pyarrow import fs

from kumo.engine import Graph, Table
from kumo.engine.connector.gcs import GCSConnector
from kumo.engine.connector.identity import GCSIdentity


@pytest.mark.parametrize('extension', ['csv', 'parquet'])
def test_source_table_inference_and_graph_acquisition(extension, tmp_path,
                                                      monkeypatch):
    objects = tmp_path / 'objects'
    dataset = objects / 'bucket' / 'events'
    dataset.mkdir(parents=True)
    expected = pa.table({
        'id':
        range(100),
        'value':
        pa.array([1, 2, 3, 4] + [None] * 96, type=pa.int64()),
    })
    source = dataset / f'part.{extension}'
    if extension == 'csv':
        csv.write_csv(expected, source)
    else:
        pq.write_table(expected, source)
    filesystem = fs.SubTreeFileSystem(str(objects), fs.LocalFileSystem())
    email = 'reader@test-project.iam.gserviceaccount.com'
    identity = GCSIdentity(workload_service_account_email=email)
    credential = object()
    bindings = []

    def session(principal):
        assert principal.config == identity._config
        return credential

    def access(*, credentials, project, requester_pays_project):
        assert credentials is credential
        bindings.append(requester_pays_project)
        return ObjectStoreAccess(filesystem)

    monkeypatch.setattr('kumo_gcs.backend._gcs_session_for', session)
    monkeypatch.setattr('kumo_gcs.source_access.source_access', access)
    monkeypatch.setattr('kumo.engine._scratch._root', tmp_path / 'scratch')
    connector = GCSConnector(root_dir='gs://bucket', identity=identity,
                             requester_pays_project='billing-project')
    table = Table.from_source_table(connector[f'events/part.{extension}'],
                                    primary_key='id')
    columns = {column.name: column for column in table.definition.cols}
    assert columns['id'].stype == Stype.ID
    assert columns['value'].dtype == Dtype.int64
    assert columns['value'].stype == Stype.numerical
    ingested = []

    def ingest(graph_config, spec):
        assert set(spec.source_paths) == {table.table_name}
        local = Path(spec.source_paths[table.table_name])
        assert local.is_relative_to(tmp_path / 'scratch')
        actual = pq.ParquetDataset(local).read()
        assert actual.to_pydict() == expected.to_pydict()
        assert actual.schema.field('value').type == pa.int64()
        ingested.append(graph_config)
        return DiskgraphIngestResult.for_index(spec.output_dir)

    monkeypatch.setattr('kumo_core.diskgraph.ingest.execute_diskgraph_ingest',
                        ingest)
    graph = Graph(tables=[table])
    snapshot = graph.snapshot()
    assert len(ingested) == 1
    assert snapshot.graph_config == ingested[0]
    assert snapshot.id.startswith(graph.id + '@')
    assert bindings and set(bindings) == {'billing-project'}


@pytest.mark.parametrize('suffix', ['', '.parquet'])
def test_graph_aliases_preserve_object_source_acquisition(
        suffix, tmp_path, monkeypatch):
    from kumo_s3.backend import S3SourceBackend
    from kumoapi.graph import Edge

    from kumo.engine.connector.s3 import S3Connector

    objects = tmp_path / 'objects' / 'bucket'
    objects.mkdir(parents=True)
    frames = {
        'markets': pa.table({'id': [1, 2]}),
        'events': pa.table({
            'id': [10, 11],
            'market_id': [1, 2]
        }),
    }
    for name, frame in frames.items():
        pq.write_table(frame, objects / f'{name}.parquet')
    access = ObjectStoreAccess(
        fs.SubTreeFileSystem(str(objects.parent), fs.LocalFileSystem()))
    monkeypatch.setattr(S3SourceBackend, '_source_access', lambda *args: access)
    monkeypatch.setattr('kumo.engine._scratch._root', tmp_path / 'scratch')
    connector = S3Connector(root_dir='s3://bucket')
    tables = {
        alias: Table.from_source_table(connector[alias + suffix],
                                       primary_key='id')
        for alias in frames
    }
    ingested = []

    def ingest(config, spec):
        assert set(spec.source_paths) == set(frames)
        for alias, frame in frames.items():
            assert pq.read_table(
                spec.source_paths[alias]).to_pydict() == frame.to_pydict()
        ingested.append(config)
        return DiskgraphIngestResult.for_index(spec.output_dir)

    monkeypatch.setattr('kumo_core.diskgraph.ingest.execute_diskgraph_ingest',
                        ingest)
    graph = Graph(tables, [Edge('events', 'market_id', 'markets')])
    graph.validate()
    assert {table.name for table in graph.native_graph.tables} == set(frames)
    snapshot = graph.snapshot()
    assert snapshot.graph_config == ingested[0]
    assert tables['markets'].table_name == 'markets' + suffix
    assert tables[
        'markets'].definition.source_table.source.path == 'markets' + suffix
    assert graph.id == Graph(dict(reversed(list(tables.items()))),
                             graph.edges).id
    assert graph.id != Graph({'other': tables['markets']}).id


@pytest.mark.parametrize('override', [False, True])
def test_identifier_inference_survives_real_snapshot_ingestion(
        tmp_path, monkeypatch, override):
    from itertools import combinations

    from kumo.engine import FileConnector, configure

    monkeypatch.setattr('kumo.engine._scratch._root', tmp_path / 'scratch')
    configure(scratch_dir=str(tmp_path / 'scratch'))
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    pq.write_table(
        pa.table({
            'row_id':
            range(100),
            'user_id':
            range(1000, 1100),
            'customer_id': [f'customer-{i}' for i in range(100)],
            'metric_id': [float(i) for i in range(100)],
            'age':
            range(100),
            'tags': ['|'.join(pair) for pair in combinations('abcde', 2)] * 10,
        }), source_dir / 'events.parquet')
    table = Table.from_source_table(
        FileConnector(root_dir=str(source_dir))['events.parquet'],
        primary_key='row_id')
    expected = Stype.numerical if override else Stype.ID
    if override:
        table = table.with_column('user_id', stype=Stype.numerical)
    columns = {column.name: column for column in table.definition.cols}
    assert columns['user_id'].stype == expected
    assert columns['customer_id'].stype == Stype.ID
    assert columns['metric_id'].stype == Stype.numerical
    assert columns['age'].stype == Stype.numerical
    assert columns['tags'].stype == Stype.multicategorical
    assert columns['tags'].multicategorical_delimiter == '|'

    snapshot = Graph({'events': table}).snapshot()
    config = snapshot.graph_config['tables'][0]
    assert config['pkey']['name'] == 'row_id'
    assert config['fkeys'] == []
    ingested_columns = {column['name']: column for column in config['cols']}
    assert ingested_columns['user_id']['stype'] == expected
    assert ingested_columns['customer_id']['stype'] == Stype.ID
    assert snapshot.index_path.is_dir()
    assert (snapshot.index_path / 'ingest_stats.json').is_file()
    assert (snapshot.index_path / 'col_stats.json').is_file()
    native_config = json.loads(
        (snapshot.index_path / 'diskgraph_config.json').read_text())
    native_columns = {
        column['name']: column
        for column in native_config['tables'][0]['columns']
    }
    assert native_columns['user_id']['semantic_type'] == ('numerical'
                                                          if override else 'id')
    assert native_columns['customer_id']['semantic_type'] == 'id'
    assert native_columns['tags']['semantic_type'] == 'multicategorical'
    assert native_columns['tags']['multicategorical_delimiter'] == '|'
    stats = json.loads(
        (snapshot.index_path / 'ingest_stats.json').read_text())['stats']
    assert stats['total_nodes'] == 100
    assert stats['source_rows_by_table'] == {'events': 100}
    assert stats['key_parse_failures'] == 0
    assert stats['key_lookup_failures'] == 0
    assert stats['key_membership_failures'] == 0


def test_numerical_names_survive_real_snapshot_ingestion(tmp_path):
    from kumo.engine import FileConnector, configure

    configure(scratch_dir=str(tmp_path / 'scratch'))
    root = tmp_path / 'source'
    root.mkdir()
    pq.write_table(
        pa.table({
            'row_id': range(100),
            'num_items': [0, 1] * 50,
            'price': [1.0, 2.0] * 50,
            'score': [1, 2] * 50,
            'num_users_id': range(100),
            'measurement': [0, 1] * 50,
        }), root / 'events.parquet')
    table = Table.from_source_table(
        FileConnector(root_dir=str(root))['events.parquet'],
        primary_key='row_id')
    columns = {col.name: col for col in table.definition.cols}
    for name in ('num_items', 'price', 'score'):
        assert columns[name].stype == Stype.numerical
    assert columns['num_users_id'].stype == Stype.ID
    assert columns['measurement'].stype == Stype.categorical
    table = table.with_column('score', stype=Stype.categorical)
    snapshot = Graph({'events': table}).snapshot()
    native = json.loads(
        (snapshot.index_path / 'diskgraph_config.json').read_text())
    native_columns = {
        col['name']: col['semantic_type']
        for col in native['tables'][0]['columns']
    }
    assert native_columns['num_items'] == 'numerical'
    assert native_columns['price'] == 'numerical'
    assert native_columns['score'] == 'categorical'
    assert native_columns['num_users_id'] == 'id'
    stats = json.loads((snapshot.index_path / 'ingest_stats.json').read_text())
    assert stats['stats']['total_nodes'] == 100


def test_declared_foreign_key_type_survives_real_snapshot(
        tmp_path, monkeypatch):
    from kumoapi.graph import Edge

    from kumo.engine import FileConnector

    monkeypatch.setattr('kumo.engine._scratch._root', tmp_path / 'scratch')
    source = tmp_path / 'source'
    source.mkdir()
    pq.write_table(pa.table({'pk': range(100)}), source / 'users.parquet')
    pq.write_table(pa.table({
        'pk': range(100),
        'customer': range(100)
    }), source / 'orders.parquet')
    connector = FileConnector(root_dir=str(source))
    users = Table.from_source_table(connector['users'], primary_key='pk')
    orders = Table.from_source_table(connector['orders'], primary_key='pk')
    assert next(col for col in orders.definition.cols
                if col.name == 'customer').stype == Stype.numerical
    snapshot = Graph({
        'users': users,
        'orders': orders
    }, [Edge('orders', 'customer', 'users')]).snapshot()
    projected = next(table for table in snapshot.graph_config['tables']
                     if table['table_name'] == 'orders')
    assert projected['fkeys'][0]['stype'] == 'ID'
    native = json.loads(
        (snapshot.index_path / 'diskgraph_config.json').read_text())
    table = next(table for table in native['tables']
                 if table['name'] == 'orders')
    assert next(col for col in table['columns']
                if col['name'] == 'customer')['semantic_type'] == 'id'
    stats = json.loads(
        (snapshot.index_path / 'ingest_stats.json').read_text())['stats']
    assert stats['total_nodes'] == 200
    # DiskGraph counts the forward relation and its reverse.
    assert stats['total_edges'] == 200
    assert stats['key_lookup_failures'] == 0


def test_omitted_self_relation_preserves_native_type_and_query_task(tmp_path):
    from kumoapi.graph import Edge

    from kumo.engine import FileConnector, configure
    from kumo.engine.pquery import PredictiveQuery

    configure(scratch_dir=str(tmp_path / 'scratch'))
    root = tmp_path / 'source'
    root.mkdir()
    pq.write_table(pa.table({
        'pk': range(100),
        'customer': range(100),
    }), root / 'orders.parquet')
    orders = Table.from_source_table(
        FileConnector(root_dir=str(root))['orders'], primary_key='pk')
    orders = orders.with_column('customer', stype=Stype.numerical)
    graph = Graph({'orders': orders}, [Edge('orders', 'customer', 'orders')])
    with pytest.warns(UserWarning, match='Self-relation .* omitted'):
        graph.validate()
    query = PredictiveQuery(graph, 'PREDICT orders.customer FOR EACH orders.pk')
    assert query.get_task_type().value == 'regression'
    snapshot = graph.snapshot()
    assert snapshot.graph_config['tables'][0]['fkeys'] == []
    native = json.loads(
        (snapshot.index_path / 'diskgraph_config.json').read_text())
    columns = {col['name']: col for col in native['tables'][0]['columns']}
    assert columns['customer']['semantic_type'] == 'numerical'
    stats = json.loads((snapshot.index_path / 'ingest_stats.json').read_text())
    assert stats['stats']['total_nodes'] == 100
    assert stats['stats']['total_edges'] == 0
