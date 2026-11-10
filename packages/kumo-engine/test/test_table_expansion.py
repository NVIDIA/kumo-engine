# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additive table expansion preserves parent graph files and ordinary snapshots."""
import hashlib
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
import pytest
from diskgraph import SamplerEngine, verify_snapshot_integrity
from kumoapi.graph import Edge

from kumo.engine import FileConnector, Graph, GraphSnapshot, Table, configure


def hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob('*') if p.is_file()
    }


def features(snapshot, name):
    engine = SamplerEngine(str(snapshot.index_path), num_threads=1)
    meta = json.loads((snapshot.index_path / 'index_metadata.json').read_text())
    node = next(n for n in meta['node_types'] if n['name'] == name)
    ids = [i for s in node['shards'] for i in range(s['min_id'], s['max_id'] + 1)]
    rows = [row for event in engine.fetch_features([
        {'node_type': name, 'node_ids': ids, 'feature_names': []}])
            if event.get('batch') is not None for row in event['batch'].to_pylist()]
    return sorted(rows, key=lambda row: row['id'])


@pytest.fixture(params=['csv', 'parquet'])
def fixture(tmp_path, request):
    data = tmp_path / 'data'
    data.mkdir()
    configure(scratch_dir=tmp_path / 'scratch')

    def write(name, table):
        path = data / f'{name}.{request.param}'
        if name == 'events':
            (data / name).mkdir(exist_ok=True)
            path = data / name / f'0.{request.param}'
        if request.param == 'csv':
            csv.write_csv(table, path)
        else:
            pq.write_table(table, path, row_group_size=2)
        return path

    users = pa.table({'id': ['001', '002', '003'], 'score': [1., 2., 3.],
                      'label': ['a', 'b', 'c']})
    events = pa.table({'id': [1, 2, 3], 'user_id': ['001', '002', '003'],
                       'time': ['2026-01-01T00:00:00Z'] * 3, 'value': [10., 20., 30.]})
    write('users', users)
    write('events', events)
    connector = FileConnector(root_dir=str(data))
    graph = Graph(tables={
        'users': Table.from_source_table(connector['users'], primary_key='id').with_column('id', dtype='string'),
        'events': Table.from_source_table(connector['events'], primary_key='id', time_column='time').with_column('user_id', dtype='string'),
    }, edges=[Edge(src_table='events', fkey='user_id', dst_table='users')])
    return graph, write, users, events


def sample_edges(snapshot, table, key, cutoff=None, fused=True):
    from diskgraph import SamplerEngine, TimeUnit
    engine = SamplerEngine(str(snapshot.index_path), random_seed=42,
                           num_threads=1)
    metadata = json.loads(
        (snapshot.index_path / 'index_metadata.json').read_text())
    node = next(n for n in metadata['node_types'] if n['name'] == table)
    ids = [
        i for shard in node['shards']
        for i in range(shard['min_id'], shard['max_id'] + 1)
    ]
    mask = engine.canonical_node_mask(table, ids, pa.array([key] * len(ids)))
    seed = [i for i, match in zip(ids, mask) if match]
    assert len(seed) == 1
    plan = {
        'layers': [{
            'fanouts': {},
            'default_fanout': {
                'fanout': 100,
                'mode': 'temporal_most_recent'
                if cutoff is not None else 'deterministic_uniform',
                'absent_key_policy': 'uniform_fallback'
            }
        }]
    }
    options = {} if cutoff is None else {
        'seed_timestamps': {
            table: [cutoff]
        },
        'seed_timestamp_unit': TimeUnit.Seconds
    }
    events = list(
        engine.sample({table: seed}, json.dumps(plan), fused=fused, **options))
    mapping = json.loads(engine.get_type_mapping())
    sampled = {mapping['node_types'][event['type_id']]: event['batch'].column('id_b').to_pylist()
               for event in events if event['type'] == 'node'}
    keys = {}
    for name, node_ids in sampled.items():
        keys[name] = {}
        for row in features(snapshot, name):
            matches = engine.canonical_node_mask(name, node_ids, pa.array([row['id']] * len(node_ids)))
            keys[name].update({node_id: row['id'] for node_id, match in zip(node_ids, matches) if match})
    edges = []
    for event in events:
        if event['type'] != 'edge':
            continue
        relation = mapping['edge_relations'][event['rel_id']]
        src, dst = relation['src_type'], relation['dst_type']
        for row in event['batch'].to_pylist():
            edges.append((relation['name'], keys[src][sampled[src][row['id_a']]],
                          keys[dst][sampled[dst][row['id_b']]]))
    return seed[0], sorted(edges)


@pytest.mark.parametrize('fused', [False, True])
@pytest.mark.parametrize('links', ['isolated', 'incoming', 'outgoing', 'both', 'missing'])
def test_expansion_matches_full_and_preserves_parent(fixture, links, fused, monkeypatch):
    graph, write, _, _ = fixture
    parent = graph.snapshot(enable_expansion=True)
    before = hashes(parent.index_path)
    # Existing users.id and events.id also serve as FKs into devices.
    write('devices', pa.table({'id': [1, 2, 3], 'user_id': ['001', None, '999' if links == 'missing' else '003'],
                               'time': ['2025-12-31T00:00:00Z', '2026-01-02T00:00:00Z', '2026-01-03T00:00:00Z'],
                               'score': [2., 3., 4.]}))
    connector = graph._named_tables['users'].connector
    device = Table.from_source_table(connector['devices'], primary_key='id', time_column='time').with_column('user_id', dtype='string')
    edges = list(graph.edges)
    if links in ('outgoing', 'both', 'missing'):
        edges.append(Edge(src_table='devices', fkey='user_id', dst_table='users'))
    if links in ('incoming', 'both'):
        edges.append(Edge(src_table='events', fkey='id', dst_table='devices'))
    expanded = Graph(tables={**graph._named_tables, 'devices': device}, edges=edges)
    from kumo.engine.connector.file import _FileSourceBackend
    prepare = _FileSourceBackend.prepare_ingest
    staged = []

    def track(self, connector, table, *args, **kwargs):
        staged.append(table)
        return prepare(self, connector, table, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(_FileSourceBackend, 'prepare_ingest', track)
        result = expanded.snapshot(previous_snapshot=parent)
    assert (result.index_path / '_expansion/lineage.json').is_file()
    verify_snapshot_integrity(str(result.index_path))
    manifest = json.loads((result.index_path / '_MANIFEST.json').read_text())
    assert {f['rel_path'] for f in manifest['files']} == set(hashes(result.index_path)) - {'_SUCCESS', '_MANIFEST.json'}
    assert staged == ['devices']
    report = json.loads((result.index_path / 'ingest_stats.json').read_text())
    assert report['source_config']['sha256'] == hashlib.sha256((result.index_path / 'diskgraph_config.json').read_bytes()).hexdigest()
    assert report['stats']['output'] == str(result.index_path)
    for name in graph._named_tables:
        assert hashes(parent.index_path / 'nodes' / name) == hashes(result.index_path / 'nodes' / name)
    full = expanded.snapshot(enable_expansion=True)
    for name in expanded._named_tables:
        assert features(result, name) == features(full, name)
        for row in features(result, name):
            for cutoff in (None, 1767312000, 1767484800):
                assert sample_edges(result, name, row['id'], cutoff=cutoff, fused=fused)[1] == sample_edges(full, name, row['id'], cutoff=cutoff, fused=fused)[1]
    stats = 'edge_stats.json'
    assert json.loads((result.index_path / stats).read_text()) == json.loads((full.index_path / stats).read_text())
    assert hashes(parent.index_path) == before


@pytest.mark.parametrize('change', ['append', 'feature', 'schema', 'remove_link', 'duplicate_new_key'])
def test_expansion_falls_back_for_incompatible_changes(fixture, change):
    graph, write, users, _ = fixture
    parent = graph.snapshot(enable_expansion=True)
    tables = dict(graph._named_tables)
    edges = list(graph.edges)
    if change == 'append':
        write('users', pa.concat_tables([users, pa.table({'id': ['004'], 'score': [4.], 'label': ['d']})]))
    elif change == 'feature':
        write('users', users.set_column(1, 'score', pa.array([9., 2., 3.])))
    elif change == 'schema':
        tables['users'] = tables['users'].with_column('score', stype='categorical')
    elif change == 'remove_link':
        edges = []
    write('devices', pa.table({'id': [1, 1 if change == 'duplicate_new_key' else 2], 'user_id': ['001', '002']}))
    connector = tables['users'].connector
    tables['devices'] = Table.from_source_table(connector['devices'], primary_key='id').with_column('user_id', dtype='string')
    graph = Graph(tables=tables, edges=edges)
    result = graph.snapshot(previous_snapshot=parent)
    assert not (result.index_path / '_expansion/lineage.json').exists()
    full = graph.snapshot()
    for name in graph._named_tables:
        assert features(result, name) == features(full, name)


def test_new_foreign_key_reorders_converter_columns(fixture):
    graph, write, _, events = fixture
    write('events', events.append_column('device_id', pa.array([1, 2, 3])))
    connector = graph._named_tables['events'].connector
    tables = dict(graph._named_tables)
    tables['events'] = Table.from_source_table(
        connector['events'], primary_key='id', time_column='time'
    ).with_column('user_id', dtype='string').with_column('device_id', stype='ID')
    parent = Graph(tables=tables, edges=graph.edges).snapshot(enable_expansion=True)
    before = hashes(parent.index_path)
    write('devices', pa.table({'id': [1, 2, 3]}))
    tables['devices'] = Table.from_source_table(connector['devices'], primary_key='id')
    expanded = Graph(tables=tables, edges=[Edge('events', 'device_id', 'devices'), *graph.edges])
    result = expanded.snapshot(previous_snapshot=parent)
    full = expanded.snapshot()
    assert (result.index_path / '_expansion/lineage.json').is_file()
    assert hashes(parent.index_path) == before
    for name in tables:
        assert features(result, name) == features(full, name)
        for row in features(result, name):
            for fused in (False, True):
                assert sample_edges(result, name, row['id'], fused=fused)[1] == sample_edges(full, name, row['id'], fused=fused)[1]


def test_multiple_and_repeated_expansions(fixture, tmp_path):
    from kumo_core.diskgraph.snapshot import (
        DiskgraphLocalConfig,
        DiskgraphSnapshotHandle,
        LocalSnapshotStore,
    )
    from kumo_core.diskgraph.stores import diskgraph_index_size_bytes

    graph, write, _, _ = fixture
    parent = graph.snapshot(enable_expansion=True)
    tables = dict(graph._named_tables)
    connector = tables['users'].connector
    edges = list(graph.edges)
    for names in [('devices', 'sessions'), ('locations',)]:
        before = hashes(parent.index_path)
        for name in names:
            write(name, pa.table({'id': [1, 2], 'user_id': ['001', '002']}))
            tables[name] = Table.from_source_table(connector[name], primary_key='id').with_column('user_id', dtype='string')
            edges.append(Edge(src_table=name, fkey='user_id', dst_table='users'))
        if 'sessions' in names:
            edges.append(Edge(src_table='sessions', fkey='id', dst_table='devices'))
        expanded = Graph(tables=tables, edges=edges)
        result = expanded.snapshot(previous_snapshot=parent)
        assert (result.index_path / '_expansion/lineage.json').is_file()
        verify_snapshot_integrity(str(result.index_path))
        files = set(hashes(result.index_path)) - {'_SUCCESS', '_MANIFEST.json'}
        manifest = json.loads((result.index_path / '_MANIFEST.json').read_text())
        assert {f['rel_path'] for f in manifest['files']} == files
        assert diskgraph_index_size_bytes(str(result.index_path)) == sum(
            (result.index_path / name).stat().st_size for name in files)
        store = LocalSnapshotStore(DiskgraphLocalConfig(str(tmp_path / 'restored')))
        handle = DiskgraphSnapshotHandle(result.index_path.parent.name, result.index_path.name)
        durable = str(tmp_path / 'durable')
        shutil.copytree(result.index_path, store.durable_path(handle, durable))
        assert store.restore(handle, durable, exists=os.path.exists, copy_tree=shutil.copytree) is not None
        assert hashes(Path(store.path(handle))) == hashes(result.index_path)
        full = expanded.snapshot(enable_expansion=True)
        for name in tables:
            assert features(result, name) == features(full, name)
            for row in features(result, name):
                assert sample_edges(result, name, row['id'])[1] == sample_edges(full, name, row['id'])[1]
        assert hashes(parent.index_path) == before
        parent = result


@pytest.mark.parametrize('entry_point', ['sdk', 'native'])
def test_case_colliding_table_preserves_parent(fixture, tmp_path, entry_point):
    from diskgraph import DiskgraphSystemError, expand_snapshot

    graph, write, _, _ = fixture
    parent = graph.snapshot(enable_expansion=True)
    before = hashes(parent.index_path)
    children = set(parent.index_path.parent.iterdir())
    output = tmp_path / 'rejected-expansion'
    if entry_point == 'sdk':
        write('new_users', pa.table({'id': ['004'], 'score': [99.]}))
        connector = graph._named_tables['users'].connector
        expanded = Graph(tables={**graph._named_tables, 'USERS': Table.from_source_table(
            connector['new_users'], primary_key='id')}, edges=graph.edges)
        with pytest.raises(DiskgraphSystemError, match='collide under lowercase normalization'):
            expanded.snapshot(previous_snapshot=parent)
    else:
        config = json.loads((parent.index_path / 'diskgraph_config.json').read_text())
        for table in config['tables']:
            table['source'] = {'type': 'parquet', 'paths': [str(p) for p in (
                parent.index_path / '_expansion/sources' / table['name']).glob('*.parquet')]}
        users = next(t for t in config['tables'] if t['name'] == 'users')
        config['tables'].append({**users, 'name': 'USERS'})
        with pytest.raises(DiskgraphSystemError, match='collide under lowercase normalization'):
            expand_snapshot(json.dumps({'parent': str(parent.index_path),
                                        'output': str(output), 'expansion': config}))
    assert not output.exists()
    assert set(parent.index_path.parent.iterdir()) == children
    assert hashes(parent.index_path) == before


@pytest.mark.parametrize('semantic', ['multicategorical', 'latlong'])
def test_native_expansion_normalizes_columns(fixture, tmp_path, semantic):
    from types import SimpleNamespace

    from diskgraph import expand_snapshot

    graph, _, _, _ = fixture
    parent = graph.snapshot(enable_expansion=True)
    before = hashes(parent.index_path)
    config = json.loads((parent.index_path / 'diskgraph_config.json').read_text())
    for table in config['tables']:
        table['source'] = {'type': 'parquet', 'paths': [str(p) for p in (
            parent.index_path / '_expansion/sources' / table['name']).glob('*.parquet')]}
    values = ['a,b', 'c'] if semantic == 'multicategorical' else [[1., 2.], [3., 4.]]
    path = tmp_path / 'extra.parquet'
    pq.write_table(pa.table({'id': [1, 2], 'value': values}), path)
    column = {'name': 'value', 'semantic_type': semantic,
              'ingested_dtype': 'string' if semantic == 'multicategorical' else 'float64_array'}
    if semantic == 'multicategorical':
        column['multicategorical_delimiter'] = ','
    config['tables'].append({'name': 'extra', 'primary_key': 'id',
                             'data_version': hashlib.sha256(path.read_bytes()).hexdigest(),
                             'source': {'type': 'parquet', 'paths': [str(path)]},
                             'columns': [{'name': 'id', 'ingested_dtype': 'int64'}, column]})
    output = tmp_path / 'normalized-expansion'
    expand_snapshot(json.dumps({'parent': str(parent.index_path),
                                'output': str(output), 'expansion': config}))
    expected = [['a', 'b'], ['c']] if semantic == 'multicategorical' else values
    assert [row['value'] for row in features(SimpleNamespace(index_path=output), 'extra')] == expected
    assert hashes(parent.index_path) == before


def test_expansion_rejects_source_changes_during_build(fixture, monkeypatch):
    import diskgraph
    graph, write, users, _ = fixture
    parent = graph.snapshot(enable_expansion=True)
    before = hashes(parent.index_path)
    write('devices', pa.table({'id': [1, 2]}))
    connector = graph._named_tables['users'].connector
    expanded = Graph(tables={**graph._named_tables, 'devices': Table.from_source_table(connector['devices'], primary_key='id')}, edges=graph.edges)
    expand = diskgraph.expand_snapshot

    def mutate(request):
        expand(request)
        write('users', users.set_column(1, 'score', pa.array([9., 2., 3.])))
    monkeypatch.setattr(diskgraph, 'expand_snapshot', mutate)
    with pytest.raises(RuntimeError, match='Source changed while expanding graph'):
        expanded.snapshot(previous_snapshot=parent)
    assert hashes(parent.index_path) == before


def test_removed_update_api_and_ordinary_snapshot(fixture):
    graph, write, users, _ = fixture
    first = graph.snapshot()
    original = hashes(first.index_path)
    assert not (first.index_path / '_expansion').exists()
    assert not hasattr(GraphSnapshot, 'update')
    assert not hasattr(GraphSnapshot, 'compact')
    path = write('users', users.set_column(1, 'score', pa.array([9., 2., 3.])))
    # Ordinary single-file source versions have whole-second timestamps.
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    current = graph.snapshot()
    assert features(current, 'users')[0]['score'] == 9.
    assert hashes(first.index_path) == original
    with pytest.raises(ValueError, match='enable_expansion=True'):
        graph.snapshot(previous_snapshot=first)


def test_exact_hash_detects_same_size_edit_with_restored_mtime(fixture):
    from kumo.engine.expansion import source_fingerprints

    graph, _, _, _ = fixture
    table = graph._named_tables['users']
    reader, name = table.connector.source_backend._reader(table.connector, table.table_name)
    path = Path(reader.object_paths(name)[0])
    before = source_fingerprints(graph)
    stat = path.stat()
    raw = path.read_bytes()
    # Byte identity is independent of mtime/size, even if the edit is not valid data.
    changed = bytearray(raw)
    changed[len(changed) // 2] ^= 1
    path.write_bytes(changed)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert source_fingerprints(graph) != before


@pytest.mark.parametrize('error', [RuntimeError, KeyboardInterrupt])
def test_expansion_failure_keeps_parent_and_removes_child(fixture, monkeypatch, error):
    import diskgraph

    graph, write, _, _ = fixture
    parent = graph.snapshot(enable_expansion=True)
    original = hashes(parent.index_path)
    write('devices', pa.table({'id': [1, 2]}))
    connector = graph._named_tables['users'].connector
    expanded = Graph(tables={**graph._named_tables, 'devices': Table.from_source_table(
        connector['devices'], primary_key='id')}, edges=graph.edges)
    outputs = []

    def fail(request):
        output = Path(json.loads(request)['output'])
        output.mkdir()
        (output / 'partial').write_bytes(b'partial')
        outputs.append(output)
        raise error('injected expansion failure')

    with monkeypatch.context() as patch:
        patch.setattr(diskgraph, 'expand_snapshot', fail)
        with pytest.raises(error, match='injected expansion failure'):
            expanded.snapshot(previous_snapshot=parent)
    assert outputs and not outputs[0].exists()
    result = expanded.snapshot(previous_snapshot=parent)
    verify_snapshot_integrity(str(result.index_path))
    full = expanded.snapshot()
    for name in expanded._named_tables:
        assert features(result, name) == features(full, name)
    assert hashes(parent.index_path) == original


def test_consolidation_matches_full_and_supports_further_expansion(fixture, monkeypatch):
    from kumo.engine.connector.file import _FileSourceBackend

    graph, write, users, _ = fixture
    parent = graph.snapshot(enable_expansion=True)
    tables, edges = dict(graph._named_tables), list(graph.edges)
    connector = tables['users'].connector
    for name in ('devices', 'sessions'):
        write(name, pa.table({'id': [1, 2], 'user_id': ['001', '002']}))
        tables[name] = Table.from_source_table(connector[name], primary_key='id').with_column('user_id', dtype='string')
        edges.append(Edge(src_table=name, fkey='user_id', dst_table='users'))
        graph = Graph(tables=tables, edges=edges)
        parent = graph.snapshot(previous_snapshot=parent)
    full = graph.snapshot(enable_expansion=True)
    verify_snapshot_integrity(str(parent.index_path))
    original = hashes(parent.index_path)
    write('users', users.set_column(1, 'score', pa.array([9., 2., 3.])))

    def no_source_access(*args, **kwargs):
        pytest.fail('Consolidation must use retained data, not connectors')

    with monkeypatch.context() as patch:
        patch.setattr(_FileSourceBackend, 'prepare_ingest', no_source_access)
        result = parent.consolidate()
    assert result.id != parent.id
    assert result.graph_config == parent.graph_config
    assert not (result.index_path / '_expansion/relations.json').exists()
    assert not list((result.index_path / '_expansion').glob('edges-*.bin'))
    assert result.consolidate() is result
    for name in tables:
        assert features(result, name) == features(full, name) == features(parent, name)
        for row in features(result, name):
            for fused in (False, True):
                for cutoff in (None, 1767312000):
                    assert sample_edges(result, name, row['id'], cutoff, fused)[1] == sample_edges(parent, name, row['id'], cutoff, fused)[1]
    for path in ('col_stats.json', 'edge_stats.json'):
        assert json.loads((result.index_path / path).read_text()) == json.loads((full.index_path / path).read_text())
    assert hashes(parent.index_path) == original
    shutil.rmtree(parent.index_path)
    verify_snapshot_integrity(str(result.index_path))
    manifest = json.loads((result.index_path / '_MANIFEST.json').read_text())
    assert {f['rel_path'] for f in manifest['files']} == set(hashes(result.index_path)) - {'_SUCCESS', '_MANIFEST.json'}
    write('users', users)
    write('locations', pa.table({'id': [1, 2], 'user_id': ['001', '002']}))
    tables['locations'] = Table.from_source_table(connector['locations'], primary_key='id').with_column('user_id', dtype='string')
    graph = Graph(tables=tables, edges=[*edges, Edge(src_table='locations', fkey='user_id', dst_table='users')])
    next_snapshot = graph.snapshot(previous_snapshot=result)
    assert (next_snapshot.index_path / '_expansion/lineage.json').is_file()
    reference = graph.snapshot(enable_expansion=True)
    for name in tables:
        assert features(next_snapshot, name) == features(reference, name)
        for row in features(reference, name):
            assert sample_edges(next_snapshot, name, row['id'])[1] == sample_edges(reference, name, row['id'])[1]


@pytest.mark.parametrize('failure', ['retain', 'ingest', 'missing_source'])
def test_consolidation_failure_preserves_parent(fixture, monkeypatch, failure):
    import kumo_core.diskgraph.ingest as ingest

    import kumo.engine.expansion as expansion

    graph, write, _, _ = fixture
    parent = graph.snapshot(enable_expansion=True)
    write('devices', pa.table({'id': [1, 2]}))
    connector = graph._named_tables['users'].connector
    graph = Graph(tables={**graph._named_tables, 'devices': Table.from_source_table(
        connector['devices'], primary_key='id')}, edges=graph.edges)
    parent = graph.snapshot(previous_snapshot=parent)
    if failure == 'missing_source':
        for path in (parent.index_path / '_expansion/sources/devices').glob('*.parquet'):
            path.unlink()
    original = hashes(parent.index_path)
    siblings = set(parent.index_path.parent.iterdir())

    def fail(*args, **kwargs):
        raise RuntimeError('injected consolidation failure')

    if failure == 'retain':
        monkeypatch.setattr(expansion, 'retain_sources', fail)
    elif failure == 'ingest':
        monkeypatch.setattr(ingest, 'execute_diskgraph_ingest', fail)
    with pytest.raises((ValueError, RuntimeError), match='retained sources|injected consolidation failure'):
        parent.consolidate()
    assert hashes(parent.index_path) == original
    assert set(parent.index_path.parent.iterdir()) == siblings


@pytest.mark.parametrize('enabled', [False, True])
def test_consolidation_without_additions_is_noop(fixture, enabled):
    graph, _, _, _ = fixture
    snapshot = graph.snapshot(enable_expansion=enabled)
    original = hashes(snapshot.index_path)
    assert snapshot.consolidate() is snapshot
    assert hashes(snapshot.index_path) == original
