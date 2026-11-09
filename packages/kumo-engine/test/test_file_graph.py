# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
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


_PAIRS = {
    'i8': (pa.array([-128, 127, None], pa.int8()), Dtype.int32, Stype.numerical,
           [-128, 127, None], pa.int8()),
    'i16': (pa.array([-32768, 32767, None], pa.int16()), Dtype.int32,
            Stype.numerical, [-32768, 32767, None], pa.int16()),
    'u8': (pa.array([0, 255, None], pa.uint8()), Dtype.int32,
           Stype.categorical, [0, 255, None], pa.uint8()),
    'u16': (pa.array([0, 65535, None], pa.uint16()), Dtype.int32,
            Stype.categorical, [0, 65535, None], pa.uint16()),
    'u32': (pa.array([0, 2**32 - 1, None], pa.uint32()), Dtype.int64,
            Stype.categorical, [0, 2**32 - 1, None], pa.uint32()),
    'u64': (pa.array([5, 2**64 - 1, None], pa.uint64()), Dtype.int64,
            Stype.categorical, [5, None, None], pa.uint64()),
    'decimal': (pa.array([Decimal(-7), Decimal(2**62), None],
                         pa.decimal128(38, 0)), Dtype.int64,
                Stype.categorical, [-7, 2**62, None], pa.decimal128(38, 0)),
    'fraction': (pa.array([Decimal('1.25'), Decimal('-2.50'), None],
                          pa.decimal128(10, 2)), Dtype.float32,
                 Stype.numerical, [1.25, -2.5, None], pa.decimal128(10, 2)),
    'dictionary': (pa.array(['a', 'b', None]).dictionary_encode(),
                   Dtype.string, Stype.categorical, ['a', 'b', None],
                   pa.dictionary(pa.int32(), pa.string())),
    'large_string': (pa.array(['a', 'b', None], pa.large_string()),
                     Dtype.string, Stype.categorical, ['a', 'b', None],
                     pa.large_string()),
    'text': (pa.array(['7', 'x', None]), Dtype.int64, Stype.categorical,
             [7, None, None], pa.string()),
    'list': (pa.array([[1, 2], [], None], pa.large_list(pa.int32())),
             Dtype.intlist, Stype.multicategorical, [[1, 2], [], None],
             pa.large_list(pa.field('element', pa.int32()))),
    'timestamp': (pa.array([1_000, 86_400_000, None],
                           pa.timestamp('ms', tz='America/New_York')),
                  Dtype.date, Stype.timestamp, [
                      datetime(1970, 1, 1, 0, 0, 1),
                      datetime(1970, 1, 2), None
                  ], pa.timestamp('ms', tz='America/New_York')),
    'dictionary_int8': (pa.DictionaryArray.from_arrays(
        pa.array([0, 1, None], pa.int8()), pa.array(['a b', 'c'])),
                        Dtype.string, Stype.text, ['a b', 'c', None],
                        pa.dictionary(pa.int8(), pa.string())),
    'i64_int32': (pa.array([-5, 2**31, None]), Dtype.int32, Stype.numerical,
                  [-5, None, None], pa.int64()),
    'u32_int32': (pa.array([7, 2**32 - 1, None], pa.uint32()), Dtype.int32,
                  Stype.categorical, [7, None, None], pa.uint32()),
    'u64_int32': (pa.array([7, 2**63, None], pa.uint64()), Dtype.int32,
                  Stype.categorical, [7, None, None], pa.uint64()),
    'f64_int64': (pa.array([2.0, 1.5, float('nan')]), Dtype.int64,
                  Stype.numerical, [2, None, None], pa.float64()),
    'f64_int64_big': (pa.array([float('inf'), 1e30, -3.0]), Dtype.int64,
                      Stype.numerical, [None, None, -3], pa.float64()),
    'f32_int32': (pa.array([7.0, 3e9, None], pa.float32()), Dtype.int32,
                  Stype.numerical, [7, None, None], pa.float32()),
    'dec_int64': (pa.array([Decimal('12.91'), Decimal('7.00'), None],
                           pa.decimal128(10, 2)), Dtype.int64, Stype.numerical,
                  [None, 7, None], pa.decimal128(10, 2)),
    'f16': (pa.array(np.array([1.5, -2.0, 65504.0], np.float16)),
            Dtype.float32, Stype.numerical, [1.5, -2.0, 65504.0],
            pa.float16()),
    'list_u32': (pa.array([[1, 2**32 - 1], [], None], pa.list_(pa.uint32())),
                 Dtype.intlist, Stype.multicategorical, [[1, 2**32 - 1], [],
                                                         None],
                 pa.list_(pa.field('element', pa.uint32()))),
    'list_i64_floatlist': (pa.array([[1, -2], [], None], pa.list_(pa.int64())),
                           Dtype.floatlist, Stype.sequence, [[1.0, -2.0], [],
                                                             None],
                           pa.list_(pa.field('element', pa.int64()))),
    'list_f32_intlist': (pa.array([[1.0, 2.0], [1.5], None],
                                  pa.list_(pa.float32())), Dtype.intlist,
                         Stype.multicategorical, [[1, 2], None, None],
                         pa.list_(pa.field('element', pa.float32()))),
    'f64_string': (pa.array([1e70, 1.5, None]), Dtype.string,
                   Stype.categorical, ['1e70', '1.5', None], pa.float64()),
    'i64_multicat': (pa.array([1, 22, None]), Dtype.string,
                     Stype.multicategorical, [['1'], ['22'], None],
                     pa.int64()),
    'bin_multicat': (pa.array([b'a|b', b'c', None]), Dtype.string,
                     Stype.multicategorical, [['a', 'b'], ['c'], None],
                     pa.binary()),
    'ts_string': (pa.array([1_000, 86_400_007, None],
                           pa.timestamp('ms', tz='UTC')), Dtype.string,
                  Stype.categorical, ['1', '86401', None],
                  pa.timestamp('ms', tz='UTC')),
    'str_bool': (pa.array(['true', 'x', 'maybe']), Dtype.bool,
                 Stype.categorical, [True, True, None], pa.string()),
}


def test_staged_source_types_ingest_as_their_declared_dtypes(
        tmp_path, monkeypatch):
    """Staging hands DiskGraph the source's own types, as the hosted path's
    verbatim copy does; DiskGraph converts them to the declared dtypes and
    nulls, with a warning, a value the dtype cannot hold."""
    import kumo_core.diskgraph.ingest as dg_ingest
    from diskgraph import SamplerEngine

    from kumo.engine import FileConnector, configure

    configure(scratch_dir=str(tmp_path / 'scratch'))
    root = tmp_path / 'source'
    root.mkdir()
    tables = {}
    for name, (column, dtype, stype, _, _) in _PAIRS.items():
        pq.write_table(
            pa.table({
                'id': range(3),
                'c': column,
                'unused': range(3)
            }), root / f'{name}.parquet')
        tables[name] = Table.from_source_table(
            FileConnector(root_dir=str(root))[name], ['c', 'id'],
            primary_key='id').with_column(
                'c', dtype=dtype, stype=stype,
                multicategorical_delimiter='|' if stype
                == Stype.multicategorical and dtype == Dtype.string else None)
    staged = {}
    ingest = dg_ingest.execute_diskgraph_ingest

    def capture(config, spec, **kwargs):
        for name, path in spec.source_paths.items():
            staged[name] = {
                pq.read_schema(part).field('c').type
                for part in Path(path).rglob('*.parquet')
            }
        return ingest(config, spec, **kwargs)

    monkeypatch.setattr(dg_ingest, 'execute_diskgraph_ingest', capture)
    with pytest.warns(UserWarning) as caught:
        snapshot = Graph(tables).snapshot()
    invalid = {
        str(warning.message).split(' in ')[1].split('.')[0]
        for warning in caught if 'invalid values found' in str(warning.message)
    }
    assert invalid == {
        'u64', 'text', 'i64_int32', 'u32_int32', 'u64_int32', 'f64_int64',
        'f64_int64_big', 'f32_int32', 'dec_int64', 'list_f32_intlist',
        'str_bool'
    }

    engine = SamplerEngine(str(snapshot.index_path))
    for name, (_, _, _, expected, stored) in _PAIRS.items():
        assert staged[name] == {stored}, name
        batches = [
            event['batch'] for event in engine.fetch_features([{
                'node_type': name,
                'node_ids': [0, 1, 2],
                'feature_names': ['id', 'c'],
            }]) if event['type'] == 'feature'
        ]
        fetched = pa.Table.from_batches(batches).sort_by('id')
        assert fetched.column('c').to_pylist() == expected, name


def test_parts_of_different_types_ingest_as_their_declared_dtypes(tmp_path):
    from diskgraph import SamplerEngine

    from kumo.engine import FileConnector, configure

    configure(scratch_dir=str(tmp_path / 'scratch'))
    root = tmp_path / 'source'
    parts = {
        'dates': ([
            pa.array([0], pa.date32()),
            pa.array([86_400_000 + 3_723_000], pa.timestamp('ms'))
        ], Dtype.date, Stype.timestamp,
                  [datetime(1970, 1, 1),
                   datetime(1970, 1, 2, 1, 2, 3)]),
        'ints': ([pa.array([2**31 - 1], pa.int32()),
                  pa.array([-2], pa.int16())], Dtype.int32,
                 Stype.numerical, [2**31 - 1, -2]),
    }
    tables = {}
    for name, (arrays, dtype, stype, _) in parts.items():
        (root / name).mkdir(parents=True)
        for index, array in enumerate(arrays):
            pq.write_table(
                pa.table({
                    'id': [index],
                    'c': array,
                    'unused': ['x']
                }), root / name / f'part-{index}.parquet')
        tables[name] = Table.from_source_table(
            FileConnector(root_dir=str(root))[name],
            ['c', 'id'] if name == 'dates' else None,
            primary_key='id').with_column('c', dtype=dtype, stype=stype)
    engine = SamplerEngine(str(Graph(tables).snapshot().index_path))
    for name, (_, _, _, expected) in parts.items():
        batches = [
            event['batch'] for event in engine.fetch_features([{
                'node_type': name,
                'node_ids': [0, 1],
                'feature_names': ['id', 'c'],
            }]) if event['type'] == 'feature'
        ]
        fetched = pa.Table.from_batches(batches).sort_by('id')
        assert fetched.column('c').to_pylist() == expected, name


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


def test_unsigned_parquet_columns_snapshot_with_exact_values(tmp_path):
    from kumo.engine import FileConnector, configure

    configure(scratch_dir=str(tmp_path / 'scratch'))
    root = tmp_path / 'source'
    root.mkdir()
    data = pa.table({
        'pk': range(4),
        'u8': pa.array([0, 255, None, 7], pa.uint8()),
        'u16': pa.array([65535, None, 0, 3], pa.uint16()),
        'u32': pa.array([None, 4294967295, 2147483648, 0], pa.uint32()),
    })
    pq.write_table(data, root / 'events.parquet')
    table = Table.from_source_table(
        FileConnector(root_dir=str(root))['events'], primary_key='pk')
    assert {col.name: col.dtype for col in table.definition.cols} == {
        'pk': Dtype.int64,
        'u8': Dtype.int32,
        'u16': Dtype.int32,
        'u32': Dtype.int64,
    }
    for name in ('u8', 'u16', 'u32'):
        table = table.with_column(name, stype=Stype.categorical)
    snapshot = Graph({'events': table}).snapshot()
    stats = json.loads(
        (snapshot.index_path / 'col_stats.json').read_text())['events']
    for name in ('u8', 'u16', 'u32'):
        values = [value for value in data[name].to_pylist() if value is not None]
        assert dict(stats[name]['category_counts']) == {
            value: 1 for value in values
        }
        assert stats[name]['null_count'] == 1


@pytest.mark.parametrize('column, source_type', [
    (pa.nulls(3), 'null'),
    (pa.array([1, 2, 2**64 - 1], pa.uint64()), 'uint64'),
    (pa.array([[True], [False], []]), 'list<element: bool>'),
])
def test_an_undeclared_unreadable_column_is_refused_before_ingest(
        column, source_type, tmp_path):
    from kumo.engine import FileConnector, configure

    configure(scratch_dir=str(tmp_path / 'scratch'))
    root = tmp_path / 'source'
    root.mkdir()
    pq.write_table(pa.table({'pk': range(3), 'bad': column}),
                   root / 'events.parquet')
    table = Table.from_source_table(
        FileConnector(root_dir=str(root))['events'], primary_key='pk')
    with pytest.raises(
            ValueError,
            match=re.escape(f"Table 'events' has columns whose source type "
                            f"Kumo cannot read: 'bad' ({source_type})")):
        Graph({'events': table}).snapshot()
    assert not (tmp_path / 'scratch' / 'snapshots').exists()


def test_a_declared_dtype_resolves_an_unreadable_column(tmp_path):
    from kumo.engine import FileConnector, configure

    configure(scratch_dir=str(tmp_path / 'scratch'))
    root = tmp_path / 'source'
    root.mkdir()
    pq.write_table(
        pa.table({
            'pk': pa.array(range(3), pa.uint64()),
            'big': pa.array([0, 2**63 - 1, None], pa.uint64()),
        }), root / 'events.parquet')
    source = FileConnector(root_dir=str(root))['events']
    declared = Table.from_source_table(source, primary_key='pk',
                                       dtypes={
                                           'pk': Dtype.int64,
                                           'big': Dtype.int64
                                       })
    chained = Table.from_source_table(source, primary_key='pk').with_column(
        'pk', dtype=Dtype.int64).with_column('big', dtype=Dtype.int64)
    assert declared.definition.cols == chained.definition.cols
    columns = {col.name: col for col in declared.definition.cols}
    assert columns['pk'].stype == Stype.ID
    assert columns['big'].dtype == Dtype.int64
    assert columns['big'].stype not in (None, Stype.unsupported)

    snapshot = Graph({
        'events': declared.with_column('big', stype=Stype.categorical)
    }).snapshot()
    stats = json.loads(
        (snapshot.index_path / 'col_stats.json').read_text())['events']['big']
    assert dict(stats['category_counts']) == {0: 1, 2**63 - 1: 1}
    assert stats['null_count'] == 1


def _float_key_graph(root, connector_cls, users, events):
    from kumoapi.graph import Edge

    root.mkdir()
    pq.write_table(pa.table(users), root / 'users.parquet')
    pq.write_table(pa.table(events), root / 'events.parquet')
    connector = connector_cls(root_dir=str(root))
    return Graph(
        {
            'users':
            Table.from_source_table(connector['users'], primary_key='id'),
            'events':
            Table.from_source_table(connector['events'],
                                    primary_key='event_id'),
        }, [Edge('events', 'user_id', 'users')])


def test_float_key_columns_ingest_as_integer_keys(tmp_path, monkeypatch):
    from diskgraph import SamplerEngine

    from kumo.engine import FileConnector, configure
    from kumo.engine import graph as graph_module

    configure(scratch_dir=str(tmp_path / 'scratch'))
    graph = _float_key_graph(
        tmp_path / 'source', FileConnector, {
            'id': pa.array([1.0, 2.0, 3.0]),
            'name': ['a', 'b', 'c']
        }, {
            'event_id': range(4),
            'user_id': pa.array([1.0, 2.0, None, 3.0]),
            'amount': [1, 2, 3, 4],
        })
    for table, key in (('users', 'id'), ('events', 'user_id')):
        columns = graph.tables[table].definition.cols
        assert next(c.dtype for c in columns if c.name == key) == Dtype.float64

    with pytest.warns(UserWarning) as caught:
        snapshot = graph.snapshot()
    messages = [str(warning.message) for warning in caught]
    assert [m for m in messages if 'float64' in m] == [
        'Key column users.id is stored as float64; its whole-number values '
        'are converted to int64 at snapshot time. Store keys as integers or '
        'strings at the source to avoid this.',
        'Key column events.user_id is stored as float64; its whole-number '
        'values are converted to int64 at snapshot time. Store keys as '
        'integers or strings at the source to avoid this.',
    ]
    assert not [m for m in messages if 'invalid values' in m]

    stats = json.loads(
        (snapshot.index_path / 'ingest_stats.json').read_text())['stats']
    assert stats['total_nodes'] == 7
    assert stats['edge_counts']['events_user_id_to_users_id'] == 3
    assert stats['key_failures_by_context'] == {
        'fkey:events.user_id': {
            'parse_failures': 1,
            'non_null_parse_failures': 0,
            'lookup_failures': 0,
            'membership_failures': 0,
        }
    }
    native = json.loads(
        (snapshot.index_path / 'diskgraph_config.json').read_text())
    native_columns = {
        (table['name'], column['name']): column
        for table in native['tables'] for column in table['columns']
    }
    assert native_columns['users', 'id']['ingested_dtype'] == 'int64'
    assert native_columns['events', 'user_id']['ingested_dtype'] == 'int64'

    engine = SamplerEngine(str(snapshot.index_path))
    batches = [
        event['batch'] for event in engine.fetch_features([{
            'node_type': 'events',
            'node_ids': [0, 1, 2, 3],
            'feature_names': ['event_id', 'user_id'],
        }]) if event['type'] == 'feature'
    ]
    fetched = pa.Table.from_batches(batches).sort_by('event_id')
    assert fetched.column('user_id').type == pa.int64()
    assert fetched.column('user_id').to_pylist() == [1, 2, None, 3]

    def unexpected(*args, **kwargs):
        raise AssertionError('a cached snapshot must not scan its keys')

    monkeypatch.setattr(graph_module, '_count_inexact_keys', unexpected)
    assert graph.snapshot().id == snapshot.id


@pytest.mark.parametrize('extension, table_name, key, values', [
    ('parquet', 'users', 'id', [1.0, 2.5, 3.0]),
    ('csv', 'events', 'user_id', [1.0, 2.5, None, 3.0]),
])
def test_float_key_with_fractional_values_is_refused(
        tmp_path, extension, table_name, key, values):
    from kumoapi.graph import Edge

    from kumo.engine import FileConnector, configure

    configure(scratch_dir=str(tmp_path / 'scratch'))
    root = tmp_path / 'source'
    root.mkdir()
    columns = {
        'users': {
            'id': [1, 2, 3],
            'name': ['a', 'b', 'c']
        },
        'events': {
            'event_id': list(range(4)),
            'user_id': [1, 2, None, 3],
        },
    }
    columns[table_name][key] = pa.array(values, pa.float64())
    for name, data in columns.items():
        if extension == 'csv':
            csv.write_csv(pa.table(data), root / f'{name}.csv')
        else:
            pq.write_table(pa.table(data), root / f'{name}.parquet')
    connector = FileConnector(root_dir=str(root))
    graph = Graph(
        {
            'users':
            Table.from_source_table(connector['users'], primary_key='id'),
            'events':
            Table.from_source_table(connector['events'],
                                    primary_key='event_id'),
        }, [Edge('events', 'user_id', 'users')])

    message = (rf"Key column {table_name}\.{key} is stored as float64 and 1 "
               r"of its values cannot be converted to an integer exactly")
    with pytest.warns(UserWarning, match='stored as float64'):
        with pytest.raises(ValueError, match=message):
            graph.snapshot()


@pytest.mark.parametrize('target, parts, refused', [
    ('int64', [pa.array([-2.0**53 + 1], pa.float64()),
               pa.array([0.0, 2.0**53 - 1], pa.float64())], 0),
    ('int64', [pa.array([2.0**53, -2.0**53], pa.float64()),
               pa.array([2.0**53 + 2], pa.float64())], 3),
    ('int64', [pa.array([2.0**53 + 2, -2.0**53 - 2], pa.float64()),
               pa.array([1.0, float('inf'), float('nan'), None],
                        pa.float64())], 3),
    ('int64', [pa.array([float('inf'), float('-inf'), float('nan'), 1.0],
                        pa.float64())], 2),
    ('int64', [pa.array([float('inf'), 1.0], pa.float32())], 1),
    ('int64', [pa.array(['1.0'], pa.string()),
               pa.array(['2.5'], pa.string())], 1),
    ('int64', [pa.array([' 7 ', '+8', '-9.00', '0001', '0' * 30 + '1', None],
                        pa.string())], 0),
    ('int64', [pa.array(['8.', '1e3', '', 'inf', '5.0.0', '.0', '٣'],
                        pa.string())], 7),
    ('int64', [pa.array(['9223372036854775807', '9223372036854775808',
                         '-9223372036854775808', '-9223372036854775809'],
                        pa.string())], 2),
    ('int64', [pa.array(['2.5', '3.0'], pa.large_string())], 1),
    ('int64', [pa.array(['1.0', '2.5', None, '1.0']).dictionary_encode()], 1),
    ('byte', [pa.array(['255', '256', '-1', '0'], pa.string())], 2),
    ('byte', [pa.array(['-0', '-0.0', '+0', '0'], pa.string())], 2),
    ('int64', [pa.array([1.0], pa.float32()),
               pa.array([16777216.5], pa.float64())], 1),
    ('int64', [pa.array(['x'], pa.string()), pa.array([2.5], pa.float64())],
     2),
    ('int64', [pa.array([16777215.0, 16777216.0], pa.float32()),
               pa.array([16777220.0], pa.float32())], 2),
    ('int64', [pa.array(np.array([2047.0, 2048.0, 2050.0], np.float16))],
     2),
    ('int64', [pa.array([Decimal('7.00'), None], pa.decimal128(10, 2))], 0),
    ('int64', [pa.array([Decimal('2.50'), Decimal('7.00')],
                        pa.decimal128(10, 2))], 1),
    ('int64', [pa.array([Decimal('9223372036854775807.00'),
                         Decimal('-9223372036854775809.00')],
                        pa.decimal128(38, 2))], 1),
    ('int64', [pa.array([Decimal('0E-10'), Decimal('0.0000000001')],
                        pa.decimal128(10, 10))], 1),
    *([('int64', [pa.array([Decimal('7.00')], pa.decimal32(9, 2))], 0)]
      if hasattr(pa, 'decimal32') else []),
    ('int64', [pa.array([Decimal('1.' + '0' * 58),
                         Decimal('1.' + '0' * 57 + '1')],
                        pa.decimal256(76, 58))], 1),
    ('int64', [pa.array([Decimal(0), Decimal('1E-76'), None],
                        pa.decimal256(76, 76))], 1),
    ('int64', [pa.array([None, None], pa.float64())], 0),
    ('int32', [pa.array([1.0], pa.float64()),
               pa.array([2**40, 1, None], pa.int64())], 1),
    ('int64', [pa.array([2**63, 1], pa.uint64())], 1),
    ('byte', [pa.array([255, 256, -1], pa.int64())], 2),
    ('byte', [pa.array([200.0, 0.0, None], pa.float64())], 0),
    ('byte', [pa.array([-1.0, 256.0], pa.float64())], 2),
    ('byte', [pa.array([float('nan'), float('-inf'), 1.0], pa.float64())],
     2),
    ('byte', [pa.array([Decimal('-1.' + '0' * 74), Decimal('1.' + '0' * 74)],
                       pa.decimal256(76, 74))], 1),
])
def test_fractional_key_check_reads_each_staged_part(tmp_path, target, parts,
                                                     refused):
    from kumoapi.graph import Edge
    from kumoapi.source_table import FileType, S3RelativeSource, S3SourceTable
    from kumoapi.table import Column, TableDefinition

    from kumo.engine import S3Connector

    connector = S3Connector(root_dir='s3://schema-only-fixture')

    def table(name, columns, primary_key):
        source = S3SourceTable(connector_id=connector.id,
                               file_type=FileType.PARQUET,
                               source=S3RelativeSource(path=name))
        definition = TableDefinition(cols=columns, source_table=source,
                                     pkey=primary_key)
        return Table(connector, name, definition)

    graph = Graph(
        {
            'users':
            table('users', [Column('id', Stype.ID, Dtype(target))], 'id'),
            'events':
            table('events', [
                Column('event_id', Stype.ID, Dtype.int64),
                Column('user_id', Stype.ID, Dtype.float64),
            ], 'event_id'),
        }, [Edge('events', 'user_id', 'users')])
    for part, chunk in enumerate(parts):
        pq.write_table(pa.table({'user_id': chunk}),
                       tmp_path / f'part-{part}.parquet')
    paths = [str(path) for path in sorted(tmp_path.glob('*.parquet'))]

    with pytest.warns(UserWarning, match=f'stored as float64.*{target}'):
        graph._ingest_tables
    if refused:
        with pytest.raises(ValueError, match=f'{refused} of its values'):
            graph._refuse_fractional_keys('events', paths)
    else:
        graph._refuse_fractional_keys('events', paths)


def test_staged_parts_of_other_formats_are_left_to_diskgraph(tmp_path):
    from kumo.engine.graph import _staged_parquet_parts

    staged = tmp_path / 'staged'
    staged.mkdir()
    table = pa.table({'id': pa.array([1.0, 2.0])})
    pq.write_table(table, staged / 'part-0.parquet')
    (staged / 'part-1.csv').write_text('id\n1.5\n')
    pq.write_table(table, tmp_path / 'extra.parquet')
    (tmp_path / 'extra.csv').write_text('id\n1.5\n')

    parts = _staged_parquet_parts([
        str(staged),
        str(tmp_path / 'extra.csv'),
        str(tmp_path / 'extra.parquet'),
    ])

    assert parts == [staged / 'part-0.parquet', tmp_path / 'extra.parquet']
