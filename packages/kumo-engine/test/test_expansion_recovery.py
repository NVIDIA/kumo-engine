# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expansion failures and recovery across independent Python processes."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pyarrow as pa
import pytest
from diskgraph import DiskgraphSystemError, verify_snapshot_integrity
from kumo_core.sources.errors import SourceNotFoundError
from kumoapi.graph import Edge
from test_table_expansion import fixture  # noqa: F401
from test_table_expansion import features, hashes, sample_edges

from kumo.engine import FileConnector, Graph, GraphSnapshot, Table, configure


def add_table(graph, write, name='devices'):
    write(name, pa.table({'id': [1, 2], 'user_id': ['001', '002']}))
    connector = graph._named_tables['users'].connector
    return Graph(
        tables={
            **graph._named_tables, name:
            Table.from_source_table(connector[name],
                                    primary_key='id').with_column(
                                        'user_id', dtype='string')
        }, edges=[
            *graph.edges,
            Edge(src_table=name, fkey='user_id', dst_table='users')
        ])


def record(snapshot):
    return dict(id=snapshot.id, index_path=str(snapshot.index_path),
                graph_config=snapshot.graph_config)


def reload_snapshot(value):
    return GraphSnapshot(value['id'], Path(value['index_path']),
                         value['graph_config'])


def assert_equivalent(actual, expected, graph):
    verify_snapshot_integrity(str(actual.index_path))
    for name in graph._named_tables:
        assert features(actual, name) == features(expected, name)
        for row in features(actual, name):
            for fused in (False, True):
                assert sample_edges(actual, name, row['id'],
                                    fused=fused)[1] == sample_edges(
                                        expected, name, row['id'],
                                        fused=fused)[1]


@pytest.mark.parametrize('dropped', ['users', 'events'])
def test_dropped_table_rebuilds(fixture, dropped):
    graph, write, _, _ = fixture
    parent = graph.snapshot(enable_expansion=True)
    before = hashes(parent.index_path)
    expanded = add_table(graph, write)
    reduced = Graph(
        tables={
            n: t
            for n, t in expanded._named_tables.items() if n != dropped
        }, edges=[
            e for e in expanded.edges
            if dropped not in (e.src_table, e.dst_table)
        ])
    result = reduced.snapshot(previous_snapshot=parent)
    assert not (result.index_path / '_expansion/lineage.json').exists()
    assert {
        n['name']
        for n in json.loads((result.index_path /
                             'index_metadata.json').read_text())['node_types']
    } == set(reduced._named_tables)
    assert_equivalent(result, reduced.snapshot(), reduced)
    assert hashes(parent.index_path) == before


@pytest.mark.parametrize('missing', ['users', 'devices'])
def test_missing_original_source_fails_safely(fixture, missing):
    graph, write, _, _ = fixture
    parent = graph.snapshot(enable_expansion=True)
    expanded = add_table(graph, write)
    before = hashes(parent.index_path)
    siblings = set(parent.index_path.parent.iterdir())
    table = expanded._named_tables[missing]
    reader, name = table.connector.source_backend._reader(
        table.connector, table.table_name)
    for path in reader.object_paths(name):
        Path(path).unlink()
    with pytest.raises(SourceNotFoundError):
        expanded.snapshot(previous_snapshot=parent)
    assert set(parent.index_path.parent.iterdir()) == siblings
    assert hashes(parent.index_path) == before


def test_missing_retained_source_falls_back(fixture):
    graph, write, _, _ = fixture
    parent = graph.snapshot(enable_expansion=True)
    expanded = add_table(graph, write)
    next((parent.index_path /
          '_expansion/sources/users').glob('*.parquet')).unlink()
    before = hashes(parent.index_path)
    result = expanded.snapshot(previous_snapshot=parent)
    assert not (result.index_path / '_expansion/lineage.json').exists()
    assert_equivalent(result, expanded.snapshot(), expanded)
    assert hashes(parent.index_path) == before


@pytest.mark.parametrize('missing', [
    '_SUCCESS', '_MANIFEST.json', '_expansion/state.json',
    '_expansion/source_fingerprints.json', 'ingest_stats.json',
    'diskgraph_config.json', 'index_metadata.json'
])
def test_missing_metadata_rejects_parent(fixture, missing):
    graph, write, _, _ = fixture
    parent = graph.snapshot(enable_expansion=True)
    expanded = add_table(graph, write)
    (parent.index_path / missing).unlink()
    before = hashes(parent.index_path)
    siblings = set(parent.index_path.parent.iterdir())
    with pytest.raises(
        (FileNotFoundError, ValueError, RuntimeError, DiskgraphSystemError)):
        expanded.snapshot(previous_snapshot=parent)
    assert hashes(parent.index_path) == before
    assert set(parent.index_path.parent.iterdir()) == siblings


def worker(payload):
    """Recreate graph definitions and snapshot handles without inherited objects."""
    connector = FileConnector(root_dir=payload['data'])
    configure(scratch_dir=Path(payload['scratch']))
    tables = {
        'users':
        Table.from_source_table(connector['users'],
                                primary_key='id').with_column(
                                    'id', dtype='string'),
        'events':
        Table.from_source_table(connector['events'], primary_key='id',
                                time_column='time').with_column(
                                    'user_id', dtype='string'),
    }
    edges = [Edge(src_table='events', fkey='user_id', dst_table='users')]
    for name in payload['added']:
        tables[name] = Table.from_source_table(connector[name],
                                               primary_key='id').with_column(
                                                   'user_id', dtype='string')
        edges.append(Edge(src_table=name, fkey='user_id', dst_table='users'))
    if payload.get('interrupt'):
        import diskgraph
        expand = diskgraph.expand_snapshot

        def pause(request):
            expand(request)
            Path(payload['ready']).write_text(json.loads(request)['output'])
            time.sleep(120)

        diskgraph.expand_snapshot = pause
    graph = Graph(tables=tables, edges=edges)
    result = graph.snapshot(
        previous_snapshot=reload_snapshot(payload['parent']))
    assert_equivalent(result, graph.snapshot(), graph)
    Path(payload['result']).write_text(json.dumps(record(result)))


@pytest.mark.parametrize('killed', [False, True])
def test_restart_and_further_expansion(fixture, tmp_path, killed):
    graph, write, _, _ = fixture
    parent = graph.snapshot(enable_expansion=True)
    expanded = add_table(graph, write)
    child = expanded.snapshot(previous_snapshot=parent)
    before = {s.id: hashes(s.index_path) for s in (parent, child)}
    add_table(expanded, write, 'sessions')
    payload = dict(data=str(tmp_path / 'data'),
                   scratch=str(tmp_path / 'scratch'), parent=record(child),
                   added=['devices', 'sessions'], ready=str(tmp_path / 'ready'),
                   result=str(tmp_path / 'result.json'))
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(map(str, sys.path))}
    command = [
        sys.executable, '-c',
        'import json,sys; from test_expansion_recovery import worker; worker(json.loads(sys.argv[1]))'
    ]
    if killed:
        log = tmp_path / 'interrupted.log'
        with log.open('w') as stream:
            process = subprocess.Popen(
                command + [json.dumps({
                    **payload, 'interrupt': True
                })], env=env, stdout=stream, stderr=stream)
            try:
                deadline = time.monotonic() + 90
                while not Path(payload['ready']).exists():
                    assert process.poll() is None, log.read_text()
                    assert time.monotonic() < deadline, log.read_text()
                    time.sleep(.1)
            finally:
                process.kill()
                process.wait(timeout=10)
        incomplete = Path(Path(payload['ready']).read_text())
        assert not (incomplete / '_SUCCESS').exists()
        # A killed process cannot clean its directory; it must never be reused.
        payload['parent'] = record(child)
    result = subprocess.run(command + [json.dumps(payload)], env=env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    restored = reload_snapshot(json.loads(Path(payload['result']).read_text()))
    assert (restored.index_path / '_expansion/lineage.json').is_file()
    if killed:
        assert restored.index_path != incomplete
        with pytest.raises((ValueError, RuntimeError)):
            add_table(expanded, write,
                      'sessions').snapshot(previous_snapshot=GraphSnapshot(
                          'incomplete', incomplete, restored.graph_config))
    for snapshot in (parent, child):
        assert hashes(snapshot.index_path) == before[snapshot.id]
