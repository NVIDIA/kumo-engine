# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumoapi.graph import Edge
from test_feature_updates import fixture  # noqa: F401 (pytest fixture)
from test_feature_updates import features
from test_incremental_snapshot import hashes, sample_edges, update_info

from kumo.engine import Graph, Table


@pytest.mark.parametrize('links', ['isolated', 'incoming', 'outgoing', 'both', 'missing'])
def test_expansion_matches_full_and_preserves_parent(fixture, links, monkeypatch):
    graph, write, _, _ = fixture
    parent = graph.snapshot(enable_updates=True)
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
    assert update_info(result)['creation_mode'] == 'expansion'
    assert staged == ['devices']
    assert result.tables_rebuilt == []
    report = json.loads((result.index_path / 'ingest_stats.json').read_text())
    assert report['source_config']['sha256'] == hashlib.sha256((result.index_path / 'diskgraph_config.json').read_bytes()).hexdigest()
    assert report['stats']['output'] == str(result.index_path)
    for name in graph._named_tables:
        assert hashes(parent.index_path / 'nodes' / name) == hashes(result.index_path / 'nodes' / name)
    full = result.update(mode='full')
    assert result.validate() == full.validate()
    for name in expanded._named_tables:
        assert features(result, name) == features(full, name)
        for row in features(result, name):
            for cutoff in (None, 1767312000, 1767484800):
                assert sample_edges(result, name, row['id'], cutoff=cutoff)[1] == sample_edges(full, name, row['id'], cutoff=cutoff)[1]
    stats = 'edge_stats.json'
    assert json.loads((result.index_path / stats).read_text()) == json.loads((full.index_path / stats).read_text())
    compacted = result.compact()
    assert compacted.validate() == result.validate()
    for name in expanded._named_tables:
        assert features(compacted, name) == features(result, name)
    assert hashes(parent.index_path) == before


def test_expansion_between_appends_and_feature_edits(fixture, tmp_path):
    graph, write, users, events = fixture
    parent = graph.snapshot(enable_updates=True)
    events = pa.concat_tables([events, events.slice(0, 1).set_column(0, 'id', pa.array([4]))])
    write('events', events)
    path = tmp_path / 'data/events/0.parquet'
    if path.exists():
        with pq.ParquetWriter(path, events.schema) as writer:
            for start, size in [(0, 2), (2, 1), (3, 1)]:
                writer.write_table(events.slice(start, size))
    parent = parent.update(mode='append')
    users = users.set_column(1, 'score', pa.array([9., 2., 3.]))
    write('users', users)
    parent = parent.update()
    before = hashes(parent.index_path)
    devices = pa.table({'id': [1, 2, 3, 4], 'user_id': ['001', '002', '003', '001'],
                        'time': ['2026-01-01T00:00:00Z'] * 4, 'score': [1., 2., 3., 4.]})
    write('devices', devices)
    connector = graph._named_tables['users'].connector
    expanded = Graph(tables={**graph._named_tables, 'devices': Table.from_source_table(
        connector['devices'], primary_key='id', time_column='time').with_column('user_id', dtype='string')},
        edges=[*graph.edges, Edge(src_table='events', fkey='id', dst_table='devices'),
               Edge(src_table='devices', fkey='user_id', dst_table='users')])
    result = expanded.snapshot(previous_snapshot=parent)
    assert update_info(result)['creation_mode'] == 'expansion'
    assert features(result, 'users')[0]['score'] == 9.
    for current in (result, result.compact()):
        devices = devices.set_column(3, 'score', pa.array([11., 2., 3., 4.]))
        write('devices', devices)
        edited = current.update()
        assert update_info(edited)['creation_mode'] == 'feature_update'
        assert json.loads((edited.index_path / '_incremental/deltas.json').read_text())['version'] == 3
        devices = pa.concat_tables([devices, devices.slice(0, 1).set_column(0, 'id', pa.array([5]))])
        write('devices', devices)
        appended = edited.update(mode='append')
        full = appended.update(mode='full')
        assert appended.validate() == full.validate()
        for name in expanded._named_tables:
            assert features(appended, name) == features(full, name)
            for row in features(appended, name):
                assert sample_edges(appended, name, row['id'], cutoff=1767312000)[1] == sample_edges(full, name, row['id'], cutoff=1767312000)[1]
        devices = devices.slice(0, 4).set_column(3, 'score', pa.array([1., 2., 3., 4.]))
        write('devices', devices)
    write('events', events.set_column(1, 'user_id', pa.array(['002', '002', '003', '001'])))
    rebuilt = result.update()
    assert rebuilt.validate() == rebuilt.update(mode='full').validate()
    assert hashes(parent.index_path) == before


@pytest.mark.parametrize('change', ['feature', 'schema', 'remove_link', 'duplicate_new_key'])
def test_expansion_falls_back_for_incompatible_changes(fixture, change):
    graph, write, users, _ = fixture
    parent = graph.snapshot(enable_updates=True)
    tables = dict(graph._named_tables)
    edges = list(graph.edges)
    if change == 'feature':
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
    assert update_info(result)['creation_mode'] != 'expansion'
    assert result.validate() == result.update(mode='full').validate()


def test_multiple_and_repeated_expansions(fixture):
    graph, write, _, _ = fixture
    parent = graph.snapshot(enable_updates=True)
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
        result = Graph(tables=tables, edges=edges).snapshot(previous_snapshot=parent)
        assert update_info(result)['creation_mode'] == 'expansion'
        assert result.validate() == result.update(mode='full').validate()
        parent = result


def test_expansion_rejects_source_changes_during_build(fixture, monkeypatch):
    import diskgraph
    graph, write, users, _ = fixture
    parent = graph.snapshot(enable_updates=True)
    before = hashes(parent.index_path)
    write('devices', pa.table({'id': [1, 2]}))
    connector = graph._named_tables['users'].connector
    expanded = Graph(tables={**graph._named_tables, 'devices': Table.from_source_table(connector['devices'], primary_key='id')}, edges=graph.edges)
    append = diskgraph.append_snapshot

    def mutate(request):
        append(request)
        write('users', users.set_column(1, 'score', pa.array([9., 2., 3.])))
    monkeypatch.setattr(diskgraph, 'append_snapshot', mutate)
    with pytest.raises(RuntimeError, match='Source changed while expanding users'):
        expanded.snapshot(previous_snapshot=parent)
    assert hashes(parent.index_path) == before
