# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from datetime import datetime, timezone
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


def _time_snapshot(tmp_path, extension, values):
    from kumo.engine import FileConnector, configure

    configure(scratch_dir=str(tmp_path / 'scratch'))
    root = tmp_path / 'source'
    root.mkdir(exist_ok=True)
    table = pa.table({
        'id': pa.array(range(len(values)), pa.int64()),
        'ts': values
    })
    path = root / f'events.{extension}'
    if extension == 'csv':
        csv.write_csv(table, path)
    else:
        pq.write_table(table, path)
    source = Table.from_source_table(
        FileConnector(root_dir=str(root))[f'events.{extension}'],
        primary_key='id', time_column='ts')
    snapshot = Graph({'events': source}).snapshot()
    assert 'timestamp_format' not in snapshot.graph_config['tables'][0][
        'time_col']
    native = json.loads(
        (snapshot.index_path / 'diskgraph_config.json').read_text())
    column = next(column for column in native['tables'][0]['columns']
                  if column['name'] == 'ts')
    stats = json.loads(
        (snapshot.index_path / 'col_stats.json').read_text())['events']['ts']
    ingest = json.loads(
        (snapshot.index_path / 'ingest_stats.json').read_text())['stats']
    return column, stats, ingest


def _secs(text):
    return int(
        datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp())


def _user_warnings(caught):
    return [str(w.message) for w in caught if w.category is UserWarning]


@pytest.mark.parametrize('extension', ['csv', 'parquet'])
@pytest.mark.parametrize('values, hint, reading, first', [
    ([20240115, 20240116, 20240117],
     ('timestamp_format', '%Y%m%d'), 'YYYYMMDD dates', '2024-01-15 00:00:00'),
    ([20240115103000, 20240116103000], ('timestamp_format', '%Y%m%d%H%M%S'),
     'YYYYMMDDHHMMSS date-times', '2024-01-15 10:30:00'),
    ([1705276800, 1705363200],
     ('timestamp_unit', 'seconds'), 'epoch seconds', '2024-01-15 00:00:00'),
    ([1705276800000, 1705363200000],
     ('timestamp_unit', 'millis'), 'epoch milliseconds', '2024-01-15 00:00:00'),
    ([1705276800000000, 1705363200000000],
     ('timestamp_unit', 'micros'), 'epoch microseconds', '2024-01-15 00:00:00'),
    ([1705276800000000000, 1705363200000000000],
     ('timestamp_unit', 'nanos'), 'epoch nanoseconds', '2024-01-15 00:00:00'),
])
def test_integer_time_columns_take_the_one_reading_every_value_allows(
        tmp_path, extension, values, hint, reading, first):
    with pytest.warns(
            UserWarning, match=re.escape(f'Time column events.ts holds whole '
                                         f'numbers; its {len(values)} values '
                                         f'are read as {reading} ({values[0]} '
                                         f'is {first})')) as caught:
        column, stats, ingest = _time_snapshot(tmp_path, extension,
                                               pa.array(values, pa.int64()))
    assert len(_user_warnings(caught)) == 1
    assert column[hint[0]] == hint[1]
    assert ingest['total_nodes'] == len(values)
    assert stats['min_secs'] == _secs(first)


@pytest.mark.parametrize('extension', ['csv', 'parquet'])
@pytest.mark.parametrize('values, message', [
    ([86400, 172800], '2 are too small or too large to be a date '
     '(e.g. 86400)'),
    ([2023, 2024, 2025], '3 are too small or too large to be a date '
     '(e.g. 2023)'),
    ([202301, 202302], '2 are too small or too large to be a date '
     '(e.g. 202301)'),
    ([20240115, 1705276800, 86400], '1 is a YYYYMMDD date (e.g. 20240115), '
     '1 is epoch seconds (e.g. 1705276800), 1 is too small or too large to '
     'be a date (e.g. 86400)'),
    ([20241301, 20240115], '1 is a YYYYMMDD date (e.g. 20240115), 1 is too '
     'small or too large to be a date (e.g. 20241301)'),
])
def test_integer_time_columns_without_one_reading_are_refused(
        tmp_path, extension, values, message):
    with pytest.raises(
            ValueError, match=re.escape(
                f'Time column events.ts holds whole numbers with no single '
                f'reading: {message}. Set timestamp_format on the column to '
                f"'s', 'ms', 'us', 'ns', '%Y%m%d' or '%Y%m%d%H%M%S', or "
                f'convert it to a timestamp at the source.')):
        _time_snapshot(tmp_path, extension, pa.array(values, pa.int64()))
    assert not list((tmp_path / 'scratch').rglob('ingest_stats.json'))


@pytest.mark.parametrize('extension', ['csv', 'parquet'])
@pytest.mark.parametrize('values, impossible, far, last', [
    ([
        '2024-01-15', '2024-02-30', '0000-00-00', '9999-12-31', '2024-01-16',
        '2024-02-29'
    ], '2 values in time column events.ts are not dates on the calendar '
     '(e.g. 2024-02-30, 0000-00-00)',
     '1 value in time column events.ts falls outside 1677-09-21 to '
     '2262-04-11 (e.g. 9999-12-31)', '2024-02-29 00:00:00'),
    ([
        '2024-01-15 10:00:00', '2024-01-15 25:00:00', '2024-01-15 10:61:00',
        '1600-01-01 00:00:00', '2024-01-15 23:59:59'
    ], '2 values in time column events.ts are not dates on the calendar '
     '(e.g. 2024-01-15 25:00:00, 2024-01-15 10:61:00)',
     '1 value in time column events.ts falls outside 1677-09-21 to '
     '2262-04-11 (e.g. 1600-01-01 00:00:00)', '2024-01-15 23:59:59'),
])
def test_impossible_and_far_dates_in_the_time_column_drop_their_rows(
        tmp_path, extension, values, impossible, far, last):
    with pytest.warns(UserWarning) as caught:
        column, stats, ingest = _time_snapshot(tmp_path, extension,
                                               pa.array(values, pa.string()))
    assert _user_warnings(caught) == [
        f'{impossible}; they are treated as missing and their rows are '
        'dropped like any row without a time. Correct them at the source to '
        'keep the rows.',
        f'{far}; it is treated as missing and its row is dropped like any '
        'row without a time. Replace placeholder dates with nulls at the '
        'source to keep the row.',
    ]
    assert ingest['total_nodes'] == len(values) - 3
    assert ingest['dropped_rows_counts'] == {'events': 3}
    assert stats['max_secs'] == _secs(last)
    assert stats['min_secs'] == _secs(values[0])


@pytest.mark.parametrize('extension', ['csv', 'parquet'])
@pytest.mark.parametrize('values, fmt, order, settled, first, last', [
    (['05/03/1990', '13/01/1990', '06/01/1990'], '%d/%m/%Y', 'day first',
     '1 of 3 values (e.g. 13/01/1990)', '1990-01-06', '1990-03-05'),
    (['05/03/1990', '01/13/1990', '12/25/1990'], '%m/%d/%Y', 'month first',
     '2 of 3 values (e.g. 01/13/1990)', '1990-01-13', '1990-12-25'),
    (['05/03/1990 10:30:00', '13/01/1990 08:00:00'], '%d/%m/%Y %H:%M:%S',
     'day first', '1 of 2 values (e.g. 13/01/1990 08:00:00)',
     '1990-01-13 08:00:00', '1990-03-05 10:30:00'),
])
def test_slash_dates_take_the_one_order_every_value_allows(
        tmp_path, extension, values, fmt, order, settled, first, last):
    with pytest.warns(
            UserWarning,
            match=re.escape(f'Time column events.ts holds slash dates read '
                            f'{order} ({fmt}) because {settled} read only that '
                            'way. Set timestamp_format on the column to '
                            'override.')) as caught:
        column, stats, ingest = _time_snapshot(tmp_path, extension,
                                               pa.array(values, pa.string()))
    assert len(_user_warnings(caught)) == 1
    assert column['timestamp_format'] == fmt
    assert ingest['total_nodes'] == len(values)
    assert (stats['min_secs'], stats['max_secs']) == (_secs(first), _secs(last))


@pytest.mark.parametrize('extension', ['csv', 'parquet'])
@pytest.mark.parametrize('values, message', [
    (['05/03/1990', '06/01/1990', '12/01/1990'],
     'holds slash dates whose day and month order the data does not settle: '
     'every one of its 3 values (e.g. 05/03/1990) reads both ways. Set '
     "timestamp_format on the column to '%d/%m/%Y' or '%m/%d/%Y'."),
    (['13/01/1990', '01/13/1990', '05/03/1990'],
     'holds slash dates in both orders: 1 value (e.g. 13/01/1990) reads only '
     'day first and 1 (e.g. 01/13/1990) only month first. Fix the column at '
     'the source, or set timestamp_format on the column to the order you '
     'want and the other values become missing.'),
])
def test_slash_dates_without_one_order_are_refused(tmp_path, extension, values,
                                                   message):
    with pytest.raises(ValueError,
                       match=re.escape(f'Time column events.ts {message}')):
        _time_snapshot(tmp_path, extension, pa.array(values, pa.string()))


@pytest.mark.parametrize('extension', ['csv', 'parquet'])
@pytest.mark.parametrize('values', [
    ['2024-01-16 10:00:00', '2024-01-15 10:00:00+02:00', '2024-01-17 10:00:00Z'],
    [
        '2024-01-16T10:00:00', '2024-01-15T10:00:00.250+0200',
        '2024-01-17T10:00:00Z'
    ],
    ['2024-01-16T10:00:00', '2024-01-15 10:00:00+02:00', '2024-01-17T10:00:00Z'],
    ['2024-01-16', '2024-01-15 10:00:00+02:00', '2024-01-17 10:00'],
])
def test_mixed_timezones_keep_every_value_and_warn(tmp_path, extension, values):
    zoned = [v for v in values if v.endswith('Z') or '+' in v]
    bare = [v for v in values if v not in zoned]
    with pytest.warns(
            UserWarning,
            match=re.escape(f'Time column events.ts mixes {len(zoned)} value'
                            f"{'s' if len(zoned) != 1 else ''} with a "
                            f'timezone, converted to UTC, and {len(bare)} '
                            f'without one, read as UTC (e.g. {zoned[0]} and '
                            f'{bare[0]}). Declare one timezone at the source '
                            'so the events line up.')) as caught:
        column, stats, ingest = _time_snapshot(tmp_path, extension,
                                               pa.array(values, pa.string()))
    assert len(_user_warnings(caught)) == 1
    assert column['timestamp_format'] == '%Y-%m-%d %H:%M:%S%.f%#z'
    assert ingest['total_nodes'] == 3
    assert ingest['dropped_rows_counts'] == {}
    assert (stats['min_secs'],
            stats['max_secs']) == (_secs('2024-01-15 08:00:00'),
                                   _secs('2024-01-17 10:00:00'))
