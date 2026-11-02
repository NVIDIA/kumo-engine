# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
import pytest
from diskgraph import SamplerEngine
from kumoapi.graph import Edge
from test_incremental_snapshot import hashes, sample_edges, update_info

from kumo.engine import FileConnector, Graph, Table, configure


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


@pytest.mark.parametrize('api', ['update', 'snapshot'])
def test_feature_replacements_preserve_structure_and_compact(fixture, api):
    graph, write, users, _ = fixture
    first = graph.snapshot(enable_updates=True)
    original = hashes(first.index_path)
    write('users', users.set_column(1, 'score', pa.array([None, 8., 3.])))
    updated = first.update() if api == 'update' else graph.snapshot(previous_snapshot=first)
    info = update_info(updated)
    assert info['creation_mode'] == 'feature_update'
    assert json.loads((updated.index_path / '_incremental/deltas.json').read_text())['version'] == 2
    assert info['replaced_rows'] == {'users': 2}
    assert updated.tables_rebuilt == []
    assert hashes(first.index_path / 'nodes') == hashes(updated.index_path / 'nodes')
    assert features(updated, 'users')[0]['score'] is None
    assert features(updated, 'users')[1]['score'] == 8.
    assert features(first, 'users')[1]['score'] == 2.
    assert sample_edges(first, 'users', '001') == sample_edges(updated, 'users', '001')
    full = updated.update(mode='full')
    assert features(updated, 'users') == features(full, 'users')
    assert updated.validate() == full.validate()
    compacted = updated.compact()
    assert compacted.index_path != updated.index_path
    assert features(compacted, 'users') == features(updated, 'users')
    assert compacted.validate() == updated.validate()
    assert not json.loads((compacted.index_path / '_incremental/deltas.json').read_text())['tables']['users']['feature_file']
    assert hashes(first.index_path) == original


@pytest.mark.parametrize('change', ['key', 'foreign_key', 'time', 'delete', 'reorder'])
def test_structural_edits_fall_back(fixture, change):
    graph, write, _, events = fixture
    first = graph.snapshot(enable_updates=True)
    if change == 'key':
        events = events.set_column(0, 'id', pa.array([4, 2, 3]))
    elif change == 'foreign_key':
        events = events.set_column(1, 'user_id', pa.array(['002', '002', '003']))
    elif change == 'time':
        events = events.set_column(2, 'time', pa.array(['2026-01-02T00:00:00Z'] * 3))
    elif change == 'delete':
        events = events.slice(0, 2)
    else:
        events = events.take(pa.array([2, 1, 0]))
    write('events', events)
    updated = first.update()
    assert updated.tables_rebuilt == ['events']
    full = updated.update(mode='full')
    assert features(updated, 'events') == features(full, 'events')
    assert updated.validate() == full.validate()


def test_edit_append_edit_add_table(fixture, tmp_path):
    graph, write, users, events = fixture
    first = graph.snapshot(enable_updates=True)
    original = hashes(first.index_path)
    users = users.set_column(1, 'score', pa.array([9., 2., 3.]))
    write('users', users)
    edited = first.update()
    assert update_info(edited)['creation_mode'] == 'feature_update'
    root = tmp_path / 'data'
    directory = root / 'events'
    source = next(directory.glob('0.*'))
    added = events.slice(0, 1).set_column(0, 'id', pa.array([4]))
    if source.suffix == '.csv':
        csv.write_csv(added, directory / '1.csv')
    else:
        pq.write_table(added, directory / '1.parquet')
    appended = edited.update(mode='append')
    assert features(appended, 'users') == features(edited, 'users')
    assert appended.rows_appended == 1
    # Edit a row whose ID was introduced by append, including after compaction.
    edited_event = added.set_column(3, 'value', pa.array([77.]))
    if source.suffix == '.csv':
        csv.write_csv(edited_event, directory / '1.csv')
    else:
        pq.write_table(edited_event, directory / '1.parquet')
    appended = appended.update()
    assert update_info(appended)['creation_mode'] == 'feature_update'
    assert features(appended, 'events')[-1]['value'] == 77.
    users = users.set_column(1, 'score', pa.array([9., 7., 3.]))
    write('users', users)
    replaced = appended.update()
    assert update_info(replaced)['creation_mode'] == 'feature_update'
    assert replaced.validate()['nodes']['events'] == 4
    assert features(replaced, 'users')[1]['score'] == 7.
    write('devices', pa.table({'id': [1, 2], 'user_id': ['001', '002']}))
    connector = graph._named_tables['users'].connector
    expanded = Graph(tables={**graph._named_tables, 'devices': Table.from_source_table(
        connector['devices'], primary_key='id').with_column('user_id', dtype='string')},
        edges=[*graph.edges, Edge(src_table='devices', fkey='user_id', dst_table='users')])
    final = expanded.snapshot(previous_snapshot=replaced)
    assert features(final, 'users') == features(replaced, 'users')
    assert final.validate()['nodes']['devices'] == 2
    assert hashes(first.index_path) == original


def test_comparison_handles_repartitioning_lists_and_nulls(tmp_path):
    from kumo.engine.feature_updates import _changed_rows

    original = pa.table({'id': range(5), 'tags': [['a'], None, [], ['b'], ['a', 'b']]})
    pq.write_table(original, tmp_path / 'old.parquet', row_group_size=3)
    current = original.set_column(1, 'tags', pa.array([['a'], ['c'], [], ['b'], ['a', 'b']]))
    pq.write_table(current.slice(0, 2), tmp_path / 'new0.parquet', row_group_size=1)
    pq.write_table(current.slice(2), tmp_path / 'new1.parquet', row_group_size=2)
    target = tmp_path / 'patch.parquet'
    assert _changed_rows([tmp_path / 'old.parquet'],
                         [tmp_path / 'new0.parquet', tmp_path / 'new1.parquet'], ['id'], target) == 1
    assert pq.read_table(target).to_pylist() == [{'id': 1, 'tags': ['c']}]


def test_repeated_variable_length_text_replacements(fixture):
    graph, write, users, _ = fixture
    tables = dict(graph._named_tables)
    tables['users'] = tables['users'].with_column('label', stype='text')
    graph = Graph(tables=tables, edges=graph.edges)
    first = graph.snapshot(enable_updates=True)
    original = hashes(first.index_path)
    parent = first
    for value in ['long text ' * 200, None, 'a']:
        write('users', users.set_column(2, 'label', pa.array([value, 'b', 'c'])))
        updated = parent.update()
        assert update_info(updated)['creation_mode'] == 'feature_update'
        full = updated.update(mode='full')
        assert features(updated, 'users') == features(full, 'users')
        if value is not None:
            assert features(updated, 'users')[0]['label'] == value
        assert hashes(updated.index_path / 'nodes') == hashes(first.index_path / 'nodes')
        parent = updated
    assert features(parent, 'users') == features(first, 'users')
    assert features(parent.compact(), 'users') == features(first, 'users')
    assert hashes(first.index_path) == original


def test_source_change_during_feature_comparison_is_rejected(fixture, monkeypatch):
    from kumo.engine import feature_updates

    graph, write, users, _ = fixture
    first = graph.snapshot(enable_updates=True)
    original = hashes(first.index_path)
    compare = feature_updates._changed_rows

    def mutate(*args):
        result = compare(*args)
        write('users', users.set_column(1, 'score', pa.array([99., 2., 3.])))
        return result

    monkeypatch.setattr(feature_updates, '_changed_rows', mutate)
    write('users', users.set_column(1, 'score', pa.array([9., 2., 3.])))
    with pytest.raises(RuntimeError, match='Source changed while staging users'):
        first.update()
    assert hashes(first.index_path) == original
