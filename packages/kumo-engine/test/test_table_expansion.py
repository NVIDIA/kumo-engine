# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additive table expansion preserves parent graph files and ordinary snapshots."""
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
import pytest
from diskgraph import SamplerEngine
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


def test_multiple_and_repeated_expansions(fixture):
    graph, write, _, _ = fixture
    parent = graph.snapshot(enable_expansion=True)
    tables = dict(graph._named_tables)
    connector = tables['users'].connector
    edges = list(graph.edges)
    for names in [('devices', 'sessions'), ('locations',)]:
        for name in names:
            write(name, pa.table({'id': [1, 2], 'user_id': ['001', '002']}))
            tables[name] = Table.from_source_table(connector[name], primary_key='id').with_column('user_id', dtype='string')
            edges.append(Edge(src_table=name, fkey='user_id', dst_table='users'))
        if 'sessions' in names:
            edges.append(Edge(src_table='sessions', fkey='id', dst_table='devices'))
        expanded = Graph(tables=tables, edges=edges)
        result = expanded.snapshot(previous_snapshot=parent)
        assert (result.index_path / '_expansion/lineage.json').is_file()
        full = expanded.snapshot(enable_expansion=True)
        for name in tables:
            assert features(result, name) == features(full, name)
            for row in features(result, name):
                assert sample_edges(result, name, row['id'])[1] == sample_edges(full, name, row['id'])[1]
        parent = result


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
    write('users', users.set_column(1, 'score', pa.array([9., 2., 3.])))
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


def test_expansion_failure_keeps_parent_and_removes_child(fixture, monkeypatch):
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
        raise RuntimeError('injected expansion failure')

    monkeypatch.setattr(diskgraph, 'expand_snapshot', fail)
    with pytest.raises(RuntimeError, match='injected expansion failure'):
        expanded.snapshot(previous_snapshot=parent)
    assert hashes(parent.index_path) == original
    assert outputs and not outputs[0].exists()
