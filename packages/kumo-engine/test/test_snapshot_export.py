# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import threading
import warnings
import zlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from diskgraph import format_versions
from kumoapi.data_source import DataSourceType
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem

import kumo.engine as kumo
import kumo.engine._snapshot_io as snapshot_io
from kumo.engine import GraphSnapshot
from kumo.engine._scratch import get_scratch_dir


@pytest.fixture
def scratch(tmp_path):
    previous = get_scratch_dir()

    def use(name):
        kumo.configure(scratch_dir=tmp_path / name)

    use('cpu')
    yield use
    kumo.configure(scratch_dir=previous)


def _index_files(path):
    return {
        str(file.relative_to(path)): file.read_bytes()
        for file in Path(path).rglob('*') if file.is_file()
        and file.name not in ('_SUCCESS', 'snapshot-manifest.json')
    }


def _exported_files(snapshot):
    return {
        name: data
        for name, data in _index_files(snapshot.index_path).items()
        if not name.startswith('train_tables/')
    }


@pytest.fixture
def exports(tmp_path):
    return kumo.FileConnector(root_dir=str(tmp_path / 'exports'))


def _export(snapshot, exports, tmp_path):
    location = snapshot.export(exports, 'one')
    return location, tmp_path / 'exports' / location


def _edit_manifest(exported, **changes):
    path = exported / 'kumo_snapshot.json'
    path.write_text(json.dumps({**json.loads(path.read_text()), **changes}))


@pytest.fixture
def workers(monkeypatch):
    @contextmanager
    def four(found, worker_ceiling=None):
        yield 4

    monkeypatch.setattr('kumo_core.sources.object_store.copies', four)


def _no_ingest(monkeypatch):
    monkeypatch.setattr(
        'kumo_core.diskgraph.ingest.execute_diskgraph_ingest',
        Mock(side_effect=AssertionError('ingest must not run')))


def test_export_load_round_trip_relocates_and_is_reused(
        tmp_path, scratch, graph, exports, monkeypatch):
    monkeypatch.setattr(snapshot_io, '_CHUNK', 1024)
    monkeypatch.setattr(snapshot_io, '_RANGE_BYTES', 4096)
    snapshot = graph.snapshot()
    location = snapshot.export(exports, 'one/')
    graph_id, data_version = snapshot.id.split('@')
    assert location == f'one/{graph_id}/{data_version}'
    assert snapshot.export(exports, 'one') == location
    exported = tmp_path / 'exports' / location
    assert _index_files(exported / 'index') == _exported_files(snapshot)
    manifest = json.loads((exported / 'kumo_snapshot.json').read_text())
    assert manifest['snapshot_id'] == snapshot.id
    assert manifest['snapshot'] == snapshot.manifest

    shutil.rmtree(tmp_path / 'cpu')
    scratch('gpu')
    loaded = GraphSnapshot.load_from_connector(exports, path=location,
                                               graph=graph)
    assert loaded.id == snapshot.id
    assert loaded.graph_config == snapshot.graph_config
    assert loaded.index_path.is_relative_to(tmp_path / 'gpu')
    assert _index_files(loaded.index_path) == _index_files(exported / 'index')
    marker = json.loads((loaded.index_path / '_SUCCESS').read_text())
    assert marker['index_path'] == str(loaded.index_path)
    recorded = loaded.manifest
    assert recorded.pop('loaded') == {
        'formats': format_versions(),
        'environment': manifest['environment'],
    }
    assert recorded == manifest['snapshot']

    _no_ingest(monkeypatch)
    monkeypatch.setattr(snapshot_io, 'load_into',
                        Mock(side_effect=AssertionError('already loaded')))
    assert GraphSnapshot.load_from_connector(exports, path=location) == loaded
    assert graph.snapshot() == loaded


@pytest.mark.parametrize('damage', [False, True])
def test_in_place_index_reuse_is_unchanged(scratch, graph, monkeypatch,
                                           damage):
    built = graph.snapshot()
    marker = (built.index_path / '_SUCCESS').read_bytes()
    if damage:
        stats = built.index_path / 'col_stats.json'
        stats.write_bytes(stats.read_bytes()[:-1])
        (built.index_path / 'extra.bin').write_bytes(b'x')
    _no_ingest(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert graph.snapshot() == built
    assert (built.index_path / '_SUCCESS').read_bytes() == marker


def test_moved_index_is_rebuilt_as_before(tmp_path, scratch, graph):
    built = graph.snapshot()
    shutil.move(tmp_path / 'cpu', tmp_path / 'moved')
    scratch('moved')
    rebuilt = graph.snapshot()
    assert rebuilt.id == built.id
    marker = json.loads((rebuilt.index_path / '_SUCCESS').read_text())
    assert marker['index_path'] == str(rebuilt.index_path)


def test_load_replaces_a_damaged_local_copy(
        tmp_path, scratch, graph, exports):
    snapshot = graph.snapshot()
    location, _ = _export(snapshot, exports, tmp_path)
    scratch('gpu')
    loaded = GraphSnapshot.load_from_connector(exports, path=location)
    stats = loaded.index_path / 'col_stats.json'
    stats.write_bytes(stats.read_bytes()[:-1])
    with pytest.warns(UserWarning, match='Reloading the damaged'):
        again = GraphSnapshot.load_from_connector(exports, path=location)
    assert _index_files(again.index_path) == _exported_files(snapshot)


def _interrupt_export(snapshot, exports, monkeypatch):
    stream = snapshot_io._stream
    finished = threading.Event()

    def interrupted(source, size, stop, write=None):
        if write is not None and source.name == 'diskgraph_config.json':
            finished.wait(10)
            raise OSError('connection reset')
        crcs = stream(source, size, stop, write)
        if write is not None and source.name == 'col_stats.json':
            finished.set()
        return crcs

    monkeypatch.setattr(snapshot_io, '_stream', interrupted)
    monkeypatch.setattr(snapshot_io, '_ATTEMPTS', 1)
    with pytest.raises(OSError, match='connection reset'):
        snapshot.export(exports, 'one')
    monkeypatch.setattr(snapshot_io, '_stream', stream)
    monkeypatch.setattr(snapshot_io, '_ATTEMPTS', 3)


@pytest.mark.parametrize('change', [False, True])
def test_interrupted_export_is_not_loadable_and_resumes(
        tmp_path, scratch, graph, exports, monkeypatch, workers, change):
    snapshot = graph.snapshot()
    _interrupt_export(snapshot, exports, monkeypatch)
    graph_id, data_version = snapshot.id.split('@')
    location = f'one/{graph_id}/{data_version}'
    with pytest.raises(FileNotFoundError, match='kumo_snapshot.json'):
        GraphSnapshot.load_from_connector(exports, path=location)
    if change:
        stats = snapshot.index_path / 'col_stats.json'
        data = stats.read_bytes()
        stats.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        index_manifest = snapshot.index_path / '_MANIFEST.json'
        index_manifest.write_bytes(index_manifest.read_bytes() + b' ')

    assert snapshot.export(exports, 'one') == location
    scratch('gpu')
    loaded = GraphSnapshot.load_from_connector(exports, path=location)
    assert (loaded.index_path / 'col_stats.json').read_bytes() == (
        snapshot.index_path / 'col_stats.json').read_bytes()


def test_concurrent_identical_exports_both_succeed(tmp_path, scratch, graph,
                                                   exports, monkeypatch):
    snapshot = graph.snapshot()
    parallel = snapshot_io._parallel
    calls = []

    def racing(*args):
        calls.append(args)
        if len(calls) == 1:
            assert snapshot.export(exports, 'one') == location
        return parallel(*args)

    graph_id, data_version = snapshot.id.split('@')
    location = f'one/{graph_id}/{data_version}'
    monkeypatch.setattr(snapshot_io, '_parallel', racing)
    assert snapshot.export(exports, 'one') == location
    assert len(calls) == 2
    scratch('gpu')
    loaded = GraphSnapshot.load_from_connector(exports, path=location)
    assert _index_files(loaded.index_path) == _exported_files(snapshot)


def test_export_committed_by_a_late_identical_export_succeeds(
        tmp_path, scratch, graph, exports, monkeypatch):
    snapshot = graph.snapshot()
    write = snapshot_io._write_json
    raced = []

    def racing(filesystem, path, value):
        if path.endswith('kumo_snapshot.json') and not raced:
            raced.append(None)
            raced[0] = snapshot.export(exports, 'one')
        return write(filesystem, path, value)

    monkeypatch.setattr(snapshot_io, '_write_json', racing)
    location = snapshot.export(exports, 'one')
    assert raced == [location]
    assert not (tmp_path / 'exports' / location / '_EXPORTING').exists()


def test_rerunning_export_repairs_a_committed_export(
        tmp_path, scratch, graph, exports):
    snapshot = graph.snapshot()
    location, exported = _export(snapshot, exports, tmp_path)
    stats = exported / 'index/col_stats.json'
    original = stats.read_bytes()
    stats.write_bytes(original[:3])
    (exported / 'index/index_metadata.json').unlink()
    assert snapshot.export(exports, 'one') == location
    assert stats.read_bytes() == original
    scratch('gpu')
    loaded = GraphSnapshot.load_from_connector(exports, path=location)
    assert _index_files(loaded.index_path) == _exported_files(snapshot)


def test_failed_load_leaves_nothing_and_restarts_cleanly(
        tmp_path, scratch, graph, exports):
    snapshot = graph.snapshot()
    location, exported = _export(snapshot, exports, tmp_path)
    stats = exported / 'index/col_stats.json'
    original = stats.read_bytes()
    stats.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))

    scratch('gpu')
    with pytest.raises(ValueError, match='checksum'):
        GraphSnapshot.load_from_connector(exports, path=location)
    graph_dir = tmp_path / 'gpu/snapshots' / snapshot.index_path.parent.name
    assert not [path for path in graph_dir.iterdir() if path.suffix != '.lock']

    stats.write_bytes(original)
    (graph_dir / f'{snapshot.index_path.name}.loading').mkdir()
    (graph_dir / f'{snapshot.index_path.name}.loading/stale').write_text('x')
    loaded = GraphSnapshot.load_from_connector(exports, path=location)
    assert _index_files(loaded.index_path) == _exported_files(snapshot)
    assert [
        path.name for path in graph_dir.iterdir() if path.suffix != '.lock'
    ] == [snapshot.index_path.name]


def test_load_needs_the_full_size_while_an_old_copy_remains(
        tmp_path, scratch, graph, exports, monkeypatch):
    snapshot = graph.snapshot()
    location, exported = _export(snapshot, exports, tmp_path)
    total = sum(entry['size'] for entry in json.loads(
        (exported / 'kumo_snapshot.json').read_text())['files'])
    scratch('gpu')
    target = (tmp_path / 'gpu/snapshots' / snapshot.index_path.parent.name /
              snapshot.index_path.name)
    target.mkdir(parents=True)
    (target / 'stale').write_bytes(b'x' * total)
    monkeypatch.setattr(
        snapshot_io.os, 'statvfs',
        lambda path: SimpleNamespace(f_frsize=1, f_bavail=total - 1))
    with pytest.raises(OSError, match=f'needs {total} bytes'):
        GraphSnapshot.load_from_connector(exports, path=location)
    assert (target / 'stale').is_file()


@pytest.mark.parametrize('field, value', [
    ('data_version', '../../victim'),
    ('data_version', '/tmp/victim'),
    ('graph_key', '..'),
    ('files', [{
        'path': '../../victim/x',
        'size': 1,
        'crc32': [0]
    }]),
])
def test_load_rejects_a_hostile_manifest(tmp_path, scratch, graph, exports,
                                         field, value):
    snapshot = graph.snapshot()
    location, exported = _export(snapshot, exports, tmp_path)
    victim = tmp_path / 'victim'
    victim.mkdir()
    (victim / 'keep').write_text('x')
    _edit_manifest(exported, **{field: value})
    scratch('gpu/a/b')
    with pytest.raises(ValueError):
        GraphSnapshot.load_from_connector(exports, path=location)
    assert (victim / 'keep').read_text() == 'x'


@pytest.mark.parametrize('edit', [
    lambda manifest: manifest.pop('files'),
    lambda manifest: manifest.pop('range_bytes'),
    lambda manifest: manifest.pop('graph_config'),
    lambda manifest: manifest.update(snapshot=[1]),
    lambda manifest: manifest['files'].append('index_metadata.json'),
    lambda manifest: manifest['files'][0].pop('crc32'),
    lambda manifest: manifest['files'].append(manifest['files'][0]),
])
def test_load_rejects_a_malformed_manifest_with_value_error(
        tmp_path, scratch, graph, exports, edit):
    snapshot = graph.snapshot()
    location, exported = _export(snapshot, exports, tmp_path)
    path = exported / 'kumo_snapshot.json'
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest))
    scratch('gpu')
    with pytest.raises(ValueError, match='export is corrupt'):
        GraphSnapshot.load_from_connector(exports, path=location)
    path.write_text('[]')
    with pytest.raises(ValueError, match='not a JSON object'):
        GraphSnapshot.load_from_connector(exports, path=location)


def test_load_records_differences_instead_of_refusing(
        tmp_path, scratch, graph, exports, monkeypatch):
    exporter_formats = {**format_versions(), 'layout': -1}
    monkeypatch.setattr('diskgraph.format_versions', lambda: exporter_formats)
    snapshot = graph.snapshot()
    monkeypatch.undo()
    location, exported = _export(snapshot, exports, tmp_path)
    _edit_manifest(exported, format=2)
    other = kumo.Graph(tables={'users': graph.tables['users']})
    scratch('gpu')
    with pytest.warns(UserWarning) as caught:
        loaded = GraphSnapshot.load_from_connector(exports, path=location,
                                                   graph=other)
    messages = ' '.join(str(warning.message) for warning in caught)
    assert 'layout format 2' in messages and 'was taken of graph' in messages
    assert loaded.id == snapshot.id
    assert loaded.manifest['formats'] == exporter_formats
    assert loaded.manifest['loaded']['formats'] == format_versions()


def test_export_refuses_a_path_holding_other_data(tmp_path, scratch, graph,
                                                  exports):
    snapshot = graph.snapshot()
    graph_id, data_version = snapshot.id.split('@')
    occupied = tmp_path / 'exports/occupied' / graph_id / data_version
    occupied.mkdir(parents=True)
    (occupied / 'other.parquet').write_bytes(b'x')
    with pytest.raises(ValueError, match='not empty'):
        snapshot.export(exports, 'occupied')

    _, exported = _export(snapshot, exports, tmp_path)
    _edit_manifest(exported, index_manifest='other')
    with pytest.raises(ValueError, match='differs from this one'):
        snapshot.export(exports, 'one')
    assert (exported / 'index/col_stats.json').is_file()


@pytest.mark.parametrize('kind, reason', [
    ('SNOWFLAKE', 'Snowflake stages'),
    ('DATABRICKS', 'Unity Catalog Volume'),
    ('BIGQUERY', 'BigQuery'),
])
def test_warehouse_connectors_are_not_supported(scratch, graph, kind, reason):
    connector = type(f'{kind.title()}Connector', (), {
        'source_type': DataSourceType[kind],
    })()
    with pytest.raises(NotImplementedError, match=reason):
        graph.snapshot().export(connector, 'x')
    with pytest.raises(NotImplementedError, match=reason):
        GraphSnapshot.load_from_connector(connector, path='x')


def test_interrupted_load_joins_workers_before_cleaning_up(
        tmp_path, scratch, graph, exports, monkeypatch, workers):
    snapshot = graph.snapshot()
    location, _ = _export(snapshot, exports, tmp_path)
    download = snapshot_io._download
    running = threading.Event()

    def interrupted(root, staging, client, item, stop):
        if item[0] == 'col_stats.json':
            running.wait(10)
            raise KeyboardInterrupt
        running.set()
        stop.wait(1)
        download(root, staging, client, item, stop)

    monkeypatch.setattr(snapshot_io, '_download', interrupted)
    scratch('gpu')
    with pytest.raises(KeyboardInterrupt):
        GraphSnapshot.load_from_connector(exports, path=location)
    graph_dir = tmp_path / 'gpu/snapshots' / snapshot.index_path.parent.name
    assert not [path for path in graph_dir.iterdir() if path.suffix != '.lock']


def test_load_explains_an_index_the_installed_diskgraph_cannot_open(
        tmp_path, scratch, graph, exports):
    snapshot = graph.snapshot()
    location, exported = _export(snapshot, exports, tmp_path)
    metadata = exported / 'index/index_metadata.json'
    garbage = b'{' * metadata.stat().st_size
    metadata.write_bytes(garbage)
    manifest = json.loads((exported / 'kumo_snapshot.json').read_text())
    for entry in manifest['files']:
        if entry['path'] == 'index_metadata.json':
            entry['crc32'] = [zlib.crc32(garbage)]
    manifest['snapshot']['formats'] = {'layout': -1}
    manifest['snapshot']['environment']['kumo_engine'] = '0.0.1'
    manifest['format'] = 2
    (exported / 'kumo_snapshot.json').write_text(json.dumps(manifest))

    scratch('gpu')
    with pytest.raises(RuntimeError, match=r"formats \{'layout': -1\} by "
                       r'kumo-engine 0\.0\.1; installed are formats .*'
                       r'export layout format 2; this engine reads 1\)\. '
                       'The on-disk formats differ: rebuild'):
        GraphSnapshot.load_from_connector(exports, path=location)


def test_load_ignores_unknown_keys_and_a_missing_snapshot_manifest(
        tmp_path, scratch, graph, exports):
    snapshot = graph.snapshot()
    location, exported = _export(snapshot, exports, tmp_path)
    path = exported / 'kumo_snapshot.json'
    manifest = json.loads(path.read_text())
    manifest['added_later'] = {'x': 1}
    manifest['snapshot'] = None
    for entry in manifest['files']:
        entry['added_later'] = 1
    path.write_text(json.dumps(manifest))
    scratch('gpu')
    loaded = GraphSnapshot.load_from_connector(exports, path=location)
    assert loaded.id == snapshot.id
    assert loaded.manifest is None


@pytest.mark.parametrize('scheme', ['s3', 'gs'])
def test_object_store_round_trip(tmp_path, scratch, graph, monkeypatch,
                                 scheme):
    storage = tmp_path / 'storage'
    storage.mkdir()
    bucket = SubTreeFileSystem(str(storage), LocalFileSystem())
    options = []

    def s3(bucket_name, *, options=None, **kwargs):
        options_seen.append(options)
        return bucket

    options_seen = options
    monkeypatch.setattr('kumo_s3.filesystem', s3)

    class Gcs(SubTreeFileSystem):
        def __init__(self):
            super().__init__(str(storage), LocalFileSystem())

    monkeypatch.setattr('kumo_gcs.backend.fs.GcsFileSystem', Gcs)

    snapshot = graph.snapshot()
    connector = kumo.S3Connector() if scheme == 's3' else kumo.GCSConnector()
    location = snapshot.export(connector, f'{scheme}://bucket/snapshots/')
    graph_id, data_version = snapshot.id.split('@')
    assert location == f'{scheme}://bucket/snapshots/{graph_id}/{data_version}'
    assert (storage / 'bucket/snapshots' / graph_id / data_version /
            'kumo_snapshot.json').is_file()
    scratch('gpu')
    loaded = GraphSnapshot.load_from_connector(connector, path=location,
                                               graph=graph)
    assert loaded.id == snapshot.id
    assert _index_files(loaded.index_path) == _exported_files(snapshot)
    if scheme == 's3':
        assert options and all(
            option['allow_delayed_open'] and not option['background_writes']
            for option in options)


def test_transfer_renews_an_expired_client(tmp_path, scratch, graph, exports,
                                           monkeypatch):
    snapshot = graph.snapshot()
    location, _ = _export(snapshot, exports, tmp_path)
    download = snapshot_io._download
    failed = []

    def expiring(root, staging, client, item, stop):
        if item[0] == 'col_stats.json':
            if not failed:
                failed.append(client)
                raise OSError('AWS Error ACCESS_DENIED: ExpiredToken')
            assert client is not failed[0]
        return download(root, staging, client, item, stop)

    monkeypatch.setattr(snapshot_io, '_download', expiring)
    scratch('gpu')
    GraphSnapshot.load_from_connector(exports, path=location)
    assert failed


def _set_mtime(path, seconds):
    os.utime(path, (seconds, seconds))


def _rewrite_events(data):
    frame = pd.read_parquet(data / 'events.parquet')
    frame.iloc[::-1].to_parquet(data / 'events.parquet')
    _set_mtime(data / 'events.parquet', 1_700_009_123)


@pytest.fixture
def dated(tmp_path, graph):
    data = tmp_path / 'data'
    for name in ('users', 'events'):
        _set_mtime(data / f'{name}.parquet', 1_700_000_123)
    return data


def _forbid_staging(monkeypatch):
    from kumo_core.sources.object_store import ObjectStoreSourceReader

    from kumo.engine.connector.file import _FileSourceBackend

    fail = Mock(side_effect=AssertionError('must not stage'))
    monkeypatch.setattr('kumo_core.sources.prepare.read_table', fail)
    monkeypatch.setattr(ObjectStoreSourceReader, 'prepare', fail)
    monkeypatch.setattr(_FileSourceBackend, 'prepare_ingest', fail)


def test_source_changes_reads_stamps_without_staging(
        scratch, graph, dated, monkeypatch):
    snapshot = graph.snapshot()
    _forbid_staging(monkeypatch)
    unchanged = {'status': 'unchanged', 'recorded': '1700000123',
                 'current': '1700000123'}
    assert snapshot.source_changes(graph) == {'users': unchanged,
                                              'events': unchanged}
    _rewrite_events(dated)
    assert snapshot.source_changes(graph) == {
        'users': unchanged,
        'events': {'status': 'changed', 'recorded': '1700000123',
                   'current': '1700009123'},
    }


def test_source_changes_unknown_for_hour_stamps_and_read_errors(
        scratch, graph, dated, monkeypatch):
    from kumo.engine.connector.file import _FileSourceBackend

    _set_mtime(dated / 'users.parquet', 1_700_002_800)
    snapshot = graph.snapshot()
    stamp = _FileSourceBackend.get_data_version

    def failing(self, connector, table_name, project_id=None):
        if table_name == 'events.parquet':
            raise OSError('listing failed')
        return stamp(self, connector, table_name)

    monkeypatch.setattr(_FileSourceBackend, 'get_data_version', failing)
    assert snapshot.source_changes(graph) == {
        'users': {'status': 'unknown', 'recorded': '1700002800',
                  'current': '1700002800'},
        'events': {'status': 'unknown', 'recorded': '1700000123',
                   'current': None},
    }


@pytest.mark.parametrize('edit', ['absent', 'reconstructed', 'missing',
                                  'malformed'])
def test_source_changes_unknown_without_a_recorded_version(
        scratch, graph, dated, edit):
    snapshot = graph.snapshot()
    path = snapshot.index_path / 'snapshot-manifest.json'
    manifest = json.loads(path.read_text())
    if edit == 'absent':
        path.unlink()
    elif edit == 'reconstructed':
        path.write_text(json.dumps({**manifest, 'reconstructed': True}))
    elif edit == 'missing':
        del manifest['parents']['source_tables']['users']
        path.write_text(json.dumps(manifest))
    else:
        path.write_text(json.dumps({**manifest, 'parents': []}))
    changes = snapshot.source_changes(graph)
    assert changes['users'] == {'status': 'unknown', 'recorded': None,
                                'current': '1700000123'}
    assert changes['events']['status'] == (
        'unchanged' if edit == 'missing' else 'unknown')


def test_source_changes_unknown_for_a_moved_table(scratch, graph, dated):
    snapshot = graph.snapshot()
    path = snapshot.index_path / 'snapshot-manifest.json'
    manifest = json.loads(path.read_text())
    manifest['parents']['source_tables']['users']['location'] += '.old'
    path.write_text(json.dumps(manifest))
    changes = snapshot.source_changes(graph)
    assert changes['users']['status'] == 'unknown'
    assert changes['events']['status'] == 'unchanged'


def test_load_warns_only_with_graph_and_only_for_changed_tables(
        tmp_path, scratch, graph, dated, exports):
    location, _ = _export(graph.snapshot(), exports, tmp_path)
    scratch('gpu')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        GraphSnapshot.load_from_connector(exports, path=location, graph=graph)
        _rewrite_events(dated)
        GraphSnapshot.load_from_connector(exports, path=location)
    with pytest.warns(UserWarning) as caught:
        GraphSnapshot.load_from_connector(exports, path=location, graph=graph)
    [message] = [str(warning.message) for warning in caught]
    assert 'source table(s) events;' in message
