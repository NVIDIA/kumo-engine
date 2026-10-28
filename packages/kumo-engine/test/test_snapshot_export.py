# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import re
import shutil
import threading
import warnings
import zlib
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem

import kumo.engine as kumo
import kumo.engine._snapshot_io as snapshot_io
from kumo.engine import GraphSnapshot, SnapshotOutputConfig
from kumo.engine._scratch import get_scratch_dir


@pytest.fixture
def scratch(tmp_path):
    previous = get_scratch_dir()

    def use(name):
        kumo.configure(scratch_dir=tmp_path / name)

    use('cpu')
    yield use
    kumo.configure(scratch_dir=previous)


@pytest.fixture
def exports(tmp_path):
    return kumo.FileConnector(root_dir=str(tmp_path / 'exports'))


def _index_files(path):
    return {
        str(file.relative_to(path)): file.read_bytes()
        for file in Path(path).rglob('*') if file.is_file()
        and file.name not in ('_SUCCESS', '_provenance.json', '.kumo_export')
        and not str(file.relative_to(path)).startswith('train_tables/')
    }


def _export(snapshot, exports, tmp_path):
    location = snapshot.export(SnapshotOutputConfig(exports, 'one'))
    return location, tmp_path / 'exports' / location


def _generation(exported):
    marker = json.loads((exported / 'kumo_snapshot.json').read_text())
    return exported / marker['generation']


def _edit_manifest(exported, edit):
    path = exported / 'kumo_snapshot.json'
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest))


def _edit_index_manifest(exported, edit):
    path = _generation(exported) / 'index/_MANIFEST.json'
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest))
    _edit_manifest(
        exported, lambda m: m.update(
            index_manifest=hashlib.sha256(path.read_bytes()).hexdigest()))


def _load(exports, location, **kwargs):
    return GraphSnapshot.load_from_connector(exports, path=location, **kwargs)


def test_export_load_round_trip_relocates_and_is_reused(
        tmp_path, scratch, graph, exports, monkeypatch):
    monkeypatch.setattr(snapshot_io, '_CHUNK', 1024)
    monkeypatch.setattr(snapshot_io, '_RANGE_BYTES', 4096)
    snapshot = graph.snapshot()
    location = snapshot.export(SnapshotOutputConfig(exports, 'one/'))
    graph_id, data_version = snapshot.id.split('@')
    assert location == f'one/{graph_id}/{data_version}'
    assert snapshot.export(SnapshotOutputConfig(exports, 'one')) == location
    exported = tmp_path / 'exports' / location
    recorded = (snapshot.index_path / '_provenance.json').read_bytes()
    assert (_generation(exported) / '_provenance.json').read_bytes() == recorded

    shutil.rmtree(tmp_path / 'cpu')
    scratch('gpu')
    loaded = _load(exports, location, graph=graph)
    assert (loaded.id, loaded.graph_config) == (snapshot.id,
                                                snapshot.graph_config)
    assert loaded.index_path.is_relative_to(tmp_path / 'gpu')
    assert _index_files(loaded.index_path) == _index_files(
        _generation(exported) / 'index')
    assert (loaded.index_path /
            '_provenance.json').read_bytes() == recorded
    reexported = tmp_path / 'exports' / loaded.export(
        SnapshotOutputConfig(exports, 'two'))
    assert (_generation(reexported) /
            '_provenance.json').read_bytes() == recorded

    monkeypatch.setattr(
        'kumo_core.diskgraph.ingest.execute_diskgraph_ingest',
        Mock(side_effect=AssertionError('ingest must not run')))
    monkeypatch.setattr(snapshot_io, 'load_into',
                        Mock(side_effect=AssertionError('already loaded')))
    assert _load(exports, location) == loaded
    assert graph.snapshot() == loaded


def test_damaged_index_is_reused_by_snapshot_but_reloaded_by_load(
        tmp_path, scratch, graph, exports, monkeypatch):
    snapshot = graph.snapshot()
    location, _ = _export(snapshot, exports, tmp_path)
    expected = _index_files(snapshot.index_path)
    stats = snapshot.index_path / 'col_stats.json'
    stats.write_bytes(stats.read_bytes()[:-1])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert graph.snapshot() == snapshot

    with pytest.warns(UserWarning, match='Reloading the damaged'):
        loaded = _load(exports, location)
    assert _index_files(loaded.index_path) == expected


def test_an_interrupted_reexport_leaves_the_last_export_loadable(
        tmp_path, scratch, graph, exports, monkeypatch):
    @contextmanager
    def four(found, worker_ceiling=None):
        yield 4

    monkeypatch.setattr('kumo_core.sources.object_store.copies', four)
    snapshot = graph.snapshot()
    location, exported = _export(snapshot, exports, tmp_path)
    first = _generation(exported)
    open_output, finished = snapshot_io._open_output, threading.Event()

    def interrupted(client, path, size):
        if path.endswith('diskgraph_config.json'):
            finished.wait(10)
            raise OSError('connection reset')
        if path.endswith('col_stats.json'):
            finished.set()
        return open_output(client, path, size)

    rebuilt = _rebuilt(snapshot, tmp_path, 'rebuilt')
    with monkeypatch.context() as patch:
        patch.setattr(snapshot_io, '_open_output', interrupted)
        patch.setattr(snapshot_io, '_ATTEMPTS', 1)
        with pytest.raises(OSError, match='connection reset'):
            rebuilt.export(SnapshotOutputConfig(exports, 'one'))
    assert _generation(exported) == first
    scratch('gpu')
    assert _index_files(_load(exports, location).index_path) == _index_files(
        snapshot.index_path)

    assert rebuilt.export(SnapshotOutputConfig(exports, 'one')) == location
    assert _generation(exported) != first and not first.exists()


def test_reexporting_an_identical_snapshot_reuses_the_export(
        tmp_path, scratch, graph, exports, monkeypatch, caplog):
    snapshot = graph.snapshot()
    location, exported = _export(snapshot, exports, tmp_path)
    first = _generation(exported)
    monkeypatch.setattr(snapshot_io, '_transfer',
                        Mock(side_effect=AssertionError('uploaded again')))
    with caplog.at_level('INFO', logger=snapshot_io.__name__):
        assert snapshot.export(SnapshotOutputConfig(exports, 'one')) == location
    assert _generation(exported) == first and 'reused it' in caplog.text

    (first / '_DONE').unlink()
    with pytest.raises(AssertionError, match='uploaded again'):
        snapshot.export(SnapshotOutputConfig(exports, 'one'))


def test_concurrent_exports_of_one_snapshot_both_succeed(
        tmp_path, scratch, graph, exports, monkeypatch):
    snapshot = graph.snapshot()
    transfer, raced = snapshot_io._transfer, []

    def racing(*args):
        if not raced:
            raced.append(None)
            raced[0] = snapshot.export(SnapshotOutputConfig(exports, 'one'))
        return transfer(*args)

    monkeypatch.setattr(snapshot_io, '_transfer', racing)
    assert raced == [snapshot.export(SnapshotOutputConfig(exports, 'one'))]
    assert _load(exports, raced[0]).id == snapshot.id


def _rebuilt(snapshot, tmp_path, name):
    r"""Another build of the same logical snapshot: same id and file sizes,
    one digit different in ``col_stats.json``."""
    index = (tmp_path / name / snapshot.index_path.parent.name /
             snapshot.index_path.name)
    shutil.copytree(snapshot.index_path, index)
    (index / '.kumo_export').unlink(missing_ok=True)
    stats = index / 'col_stats.json'
    data = stats.read_bytes()
    position = re.search(rb'[0-9]([0-9])', data).start(1)
    digit = b'1' if data[position:position + 1] != b'1' else b'2'
    stats.write_bytes(data[:position] + digit + data[position + 1:])
    return replace(snapshot, index_path=index)


def test_interleaved_exports_of_different_builds_stay_consistent(
        tmp_path, scratch, graph, exports, monkeypatch):
    first = graph.snapshot()
    second = _rebuilt(first, tmp_path, 'second')
    commit, paused = snapshot_io._commit, threading.Event()
    resumed = threading.Event()

    def held(*args):
        if threading.current_thread() is not threading.main_thread():
            paused.set()
            resumed.wait(30)
        return commit(*args)

    monkeypatch.setattr(snapshot_io, '_commit', held)
    errors = []

    def export_first():
        try:
            first.export(SnapshotOutputConfig(exports, 'one'))
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=export_first)
    worker.start()
    assert paused.wait(30)
    location = second.export(SnapshotOutputConfig(exports, 'one'))
    resumed.set()
    worker.join(30)
    assert not errors

    scratch('gpu')
    assert _index_files(_load(exports, location).index_path) == _index_files(
        first.index_path)
    generations = [path for path in (tmp_path / 'exports' / location).iterdir()
                   if path.is_dir()]
    assert generations == [_generation(tmp_path / 'exports' / location)]


def test_a_load_follows_an_export_that_commits_while_it_downloads(
        tmp_path, scratch, graph, exports, monkeypatch):
    first = graph.snapshot()
    second = _rebuilt(first, tmp_path, 'second')
    location = first.export(SnapshotOutputConfig(exports, 'one'))
    load_into, calls = snapshot_io.load_into, []

    def overtaken(*args):
        calls.append(args[2])
        if len(calls) == 1:
            second.export(SnapshotOutputConfig(exports, 'one'))
        return load_into(*args)

    monkeypatch.setattr(snapshot_io, 'load_into', overtaken)
    scratch('gpu')
    loaded = _load(exports, location)
    assert len(calls) == 2 and calls[0] != calls[1]
    assert _index_files(loaded.index_path) == _index_files(second.index_path)


def test_a_cached_load_is_reused_only_for_the_same_export(
        tmp_path, scratch, graph, exports):
    first = graph.snapshot()
    second = _rebuilt(first, tmp_path, 'second')
    one = first.export(SnapshotOutputConfig(exports, 'one'))
    two = second.export(SnapshotOutputConfig(exports, 'two'))
    assert _index_files(first.index_path) != _index_files(second.index_path)

    scratch('gpu')
    for location, build in ((one, first), (two, second), (one, first)):
        loaded = _load(exports, location)
        assert _index_files(loaded.index_path) == _index_files(
            build.index_path)


def test_overlapping_marker_commits_leave_one_whole_marker(
        tmp_path, monkeypatch):
    root = tmp_path / 'location'
    markers = {
        name * 32: json.dumps({'generation': name * 32, 'crc': crc}).encode()
        for name, crc in (('a', 1), ('b', 1234567))
    }
    for generation in markers:
        (root / generation).mkdir(parents=True)
    write, halfway = snapshot_io._write, threading.Barrier(2)

    def interleaved(filesystem, path, payload):
        if not path.endswith('/kumo_snapshot.json'):
            return write(filesystem, path, payload)
        with filesystem.open_output_stream(path) as writer:
            writer.write(payload[:5])
            halfway.wait(10)
            writer.write(payload[5:])

    monkeypatch.setattr(snapshot_io, '_write', interleaved)
    threads = [
        threading.Thread(target=snapshot_io._commit,
                         args=(LocalFileSystem(), str(root), generation,
                               marker))
        for generation, marker in markers.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    committed = (root / 'kumo_snapshot.json').read_bytes()
    assert committed in markers.values()


def test_an_unreadable_marker_is_corrupt_on_load_and_reexported_by_a_store(
        tmp_path, scratch, graph):
    snapshot = graph.snapshot()
    settings = kumo.MLOps(artifact_root=str(tmp_path / 'root'))
    snapshot_io.store(snapshot, settings)
    location = snapshot_io.export_location(snapshot, 'snapshots')
    marker = tmp_path / 'root' / location / 'kumo_snapshot.json'
    marker.write_bytes(b'{"generation": "a"}34567}')
    connector = kumo.FileConnector(root_dir=str(tmp_path / 'root'))
    with pytest.raises(ValueError, match='corrupt.*unreadable'):
        _load(connector, location)

    snapshot_io.store(snapshot, settings)
    assert _load(connector, location).id == snapshot.id


def test_loading_an_export_on_the_exporting_machine_skips_the_checksum_pass(
        tmp_path, scratch, graph, exports, monkeypatch):
    snapshot = graph.snapshot()
    location, _ = _export(snapshot, exports, tmp_path)
    monkeypatch.setattr(
        'diskgraph.snapshot_range_checksums',
        Mock(side_effect=AssertionError('the local index was re-read')))
    assert _load(exports, location) == snapshot


def test_failed_upload_is_retried_with_a_fresh_client(
        scratch, graph, exports, monkeypatch):
    open_output, clients = snapshot_io._open_output, []

    def expiring(client, path, size):
        if path.endswith('col_stats.json'):
            clients.append(client)
            if len(clients) == 1:
                raise OSError('ExpiredToken')
        return open_output(client, path, size)

    monkeypatch.setattr(snapshot_io, '_open_output', expiring)
    graph.snapshot().export(SnapshotOutputConfig(exports, 'one'))
    assert len(clients) == 2 and clients[0] is not clients[1]


def test_reexport_after_redaction_publishes_the_redacted_manifest(
        tmp_path, scratch, graph, exports):
    full = graph.snapshot()
    location, exported = _export(full, exports, tmp_path)
    first = _generation(exported)
    redacted = graph.snapshot(record_sources=False)
    assert redacted.id == full.id
    assert redacted.export(SnapshotOutputConfig(exports, 'one')) == location
    committed = _generation(exported)
    assert committed != first
    assert (committed / '_provenance.json').read_bytes() == (
        redacted.index_path / '_provenance.json').read_bytes()
    assert json.loads((committed / '_provenance.json').read_text())[
        'record_sources'] is False


def test_reexport_replaces_an_export_missing_a_remote_file(
        tmp_path, scratch, graph, exports):
    snapshot = graph.snapshot()
    location, exported = _export(snapshot, exports, tmp_path)
    first = _generation(exported)
    (first / 'index' / 'col_stats.json').unlink()
    assert snapshot.export(SnapshotOutputConfig(exports, 'one')) == location
    assert _generation(exported) != first
    scratch('empty')
    assert _index_files(_load(exports, location).index_path) == _index_files(
        snapshot.index_path)


def test_reexport_supersedes_a_damaged_export(tmp_path, scratch, graph,
                                              exports):
    snapshot = graph.snapshot()
    location, exported = _export(snapshot, exports, tmp_path)
    damaged = _generation(exported)
    stats = damaged / 'index/col_stats.json'
    stats.write_bytes(stats.read_bytes()[:3])
    (damaged / 'index/index_metadata.json').unlink()
    _edit_manifest(exported, lambda m: m.update(snapshot_id='other'))
    assert snapshot.export(SnapshotOutputConfig(exports, 'one')) == location
    assert not damaged.exists()
    scratch('gpu')
    assert _load(exports, location).id == snapshot.id


def test_failed_load_leaves_nothing_and_restarts_cleanly(
        tmp_path, scratch, graph, exports):
    snapshot = graph.snapshot()
    location, exported = _export(snapshot, exports, tmp_path)
    stats = _generation(exported) / 'index/col_stats.json'
    original = stats.read_bytes()
    stats.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))

    scratch('gpu')
    with pytest.raises(ValueError, match='corrupt.*checksum'):
        _load(exports, location)
    graph_dir = tmp_path / 'gpu/snapshots' / snapshot.index_path.parent.name
    assert not [path for path in graph_dir.iterdir() if path.suffix != '.lock']

    stats.write_bytes(original)
    (graph_dir / f'{snapshot.index_path.name}.loading').mkdir()
    loaded = _load(exports, location)
    assert _index_files(loaded.index_path) == _index_files(snapshot.index_path)
    assert [
        path.name for path in graph_dir.iterdir() if path.suffix != '.lock'
    ] == [snapshot.index_path.name]


def test_load_refuses_before_downloading_when_disk_is_too_small(
        tmp_path, scratch, graph, exports, monkeypatch):
    location, _ = _export(graph.snapshot(), exports, tmp_path)
    scratch('gpu')
    monkeypatch.setattr(snapshot_io.os, 'statvfs',
                        lambda path: SimpleNamespace(f_frsize=1, f_bavail=0))
    monkeypatch.setattr(snapshot_io, '_download',
                        Mock(side_effect=AssertionError('downloaded')))
    with pytest.raises(OSError, match='free bytes'):
        _load(exports, location)


@pytest.mark.parametrize('index, edit', [
    (False, lambda manifest: manifest.update(data_version='../../victim')),
    (False, lambda manifest: manifest.pop('crc32')),
    (True, lambda manifest: manifest['files'][0].update(
        rel_path='../../victim/x')),
    (True, lambda manifest: manifest['files'].append(manifest['files'][0])),
])
def test_load_rejects_a_hostile_or_malformed_manifest(tmp_path, scratch, graph,
                                                      exports, index, edit):
    location, exported = _export(graph.snapshot(), exports, tmp_path)
    victim = tmp_path / 'victim'
    victim.mkdir()
    (victim / 'keep').write_text('x')
    (_edit_index_manifest if index else _edit_manifest)(exported, edit)
    scratch('gpu/a/b')
    with pytest.raises(ValueError):
        _load(exports, location)
    assert (victim / 'keep').read_text() == 'x'


def test_load_rejects_unknown_formats_and_accepts_unknown_fields(
        tmp_path, scratch, graph, exports):
    snapshot = graph.snapshot()
    location, exported = _export(snapshot, exports, tmp_path)
    scratch('gpu')

    _edit_manifest(exported, lambda m: m.update(format=2))
    with pytest.raises(ValueError, match='format 2'):
        _load(exports, location)

    _edit_manifest(exported, lambda m: m.update(format=1, added_later={'x': 1}))
    other = kumo.Graph(tables={'users': graph.tables['users']})
    with pytest.warns(UserWarning, match=r'another graph\. Schema changes: '
                      r'table events added\.$'):
        loaded = _load(exports, location, graph=other)
    assert loaded.id == snapshot.id


def test_load_warns_about_a_graph_that_moved_with_the_same_schema(
        tmp_path, scratch, graph, exports):
    location, _ = _export(graph.snapshot(), exports, tmp_path)
    shutil.copytree(tmp_path / 'data', tmp_path / 'prod')
    prod = kumo.FileConnector(root_dir=str(tmp_path / 'prod'))
    moved = kumo.Graph(
        tables={
            name: replace(table, connector=prod)
            for name, table in graph.tables.items()
        }, edges=graph.edges)
    assert moved.id != graph.id

    with pytest.warns(UserWarning, match=r'another graph\. The schema is '
                      r'the same; only table connectors or locations differ'):
        _load(exports, location, graph=moved)


@pytest.mark.parametrize('recorded', ['newer', 'missing'])
def test_snapshot_manifest_travels_unchanged_or_is_reconstructed(
        tmp_path, scratch, graph, exports, recorded):
    snapshot = graph.snapshot()
    path = snapshot.index_path / '_provenance.json'
    if recorded == 'newer':
        path.write_bytes(b'{"schema_version": 99, "future": [1]}')
    else:
        path.unlink()
    location, exported = _export(snapshot, exports, tmp_path)
    scratch('gpu')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        loaded = _load(exports, location)
    shipped = (_generation(exported) / '_provenance.json').read_bytes()
    assert (loaded.index_path / '_provenance.json').read_bytes() == shipped
    if recorded == 'newer':
        assert shipped == b'{"schema_version": 99, "future": [1]}'
    else:
        manifest = loaded.manifest
        assert (manifest.id, manifest.reconstructed) == (snapshot.id, True)
        assert manifest.parents.source_tables == {}


def test_warehouse_connectors_are_not_supported(scratch, graph):
    connector = SimpleNamespace(sink_backend=object())
    with pytest.raises(NotImplementedError, match='FileConnector'):
        graph.snapshot().export(SnapshotOutputConfig(connector, 'x'))
    with pytest.raises(NotImplementedError, match='FileConnector'):
        _load(connector, 'x')


def test_load_explains_an_index_the_installed_diskgraph_cannot_open(
        tmp_path, scratch, graph, exports):
    location, exported = _export(graph.snapshot(), exports, tmp_path)
    metadata = _generation(exported) / 'index/index_metadata.json'
    garbage = b'{' * metadata.stat().st_size
    metadata.write_bytes(garbage)

    def unreadable(manifest):
        manifest['crc32']['index_metadata.json'] = [zlib.crc32(garbage)]
        manifest['environment']['kumo_engine'] = '0.0.1'

    _edit_manifest(exported, unreadable)
    scratch('gpu')
    with pytest.raises(RuntimeError, match='exported by kumo-engine 0.0.1'):
        _load(exports, location)


@pytest.mark.parametrize('scheme', ['s3', 'gs'])
def test_object_store_round_trip(tmp_path, scratch, graph, monkeypatch,
                                 scheme):
    storage = tmp_path / 'storage'
    storage.mkdir()

    class Bucket(SubTreeFileSystem):
        def __init__(self, *args, **kwargs):
            super().__init__(str(storage), LocalFileSystem())

    monkeypatch.setattr('kumo_s3.filesystem', Bucket)
    monkeypatch.setattr('kumo_gcs.backend.fs.GcsFileSystem', Bucket)
    snapshot = graph.snapshot()
    connector = kumo.S3Connector() if scheme == 's3' else kumo.GCSConnector()
    location = snapshot.export(
        SnapshotOutputConfig(connector, f'{scheme}://bucket/snapshots/'))
    graph_id, data_version = snapshot.id.split('@')
    assert location == f'{scheme}://bucket/snapshots/{graph_id}/{data_version}'
    scratch('gpu')
    loaded = _load(connector, location, graph=graph)
    assert _index_files(loaded.index_path) == _index_files(snapshot.index_path)


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


@pytest.mark.parametrize('record_sources', [True, False])
def test_source_changes_reads_stamps_without_staging(
        scratch, graph, dated, monkeypatch, record_sources):
    from kumo_core.sources.object_store import ObjectStoreSourceReader

    from kumo.engine.connector.file import _FileSourceBackend

    _set_mtime(dated / 'users.parquet', 1_700_002_800)
    snapshot = graph.snapshot(record_sources=record_sources)
    fail = Mock(side_effect=AssertionError('must not stage'))
    monkeypatch.setattr('kumo_core.sources.prepare.read_table', fail)
    monkeypatch.setattr(ObjectStoreSourceReader, 'prepare', fail)
    monkeypatch.setattr(_FileSourceBackend, 'prepare_ingest', fail)
    assert snapshot.source_changes(graph) == {
        'users': {'status': 'unchanged', 'recorded': '1700002800',
                  'current': '1700002800'},
        'events': {'status': 'unchanged', 'recorded': '1700000123',
                   'current': '1700000123'},
    }
    _rewrite_events(dated)
    assert snapshot.source_changes(graph)['events'] == {
        'status': 'changed', 'recorded': '1700000123', 'current': '1700009123'}

    monkeypatch.setattr(_FileSourceBackend, 'get_data_version',
                        Mock(side_effect=OSError('listing failed')))
    assert snapshot.source_changes(graph)['events'] == {
        'status': 'unknown', 'recorded': '1700000123', 'current': None}


def test_source_changes_compares_directory_listings(tmp_path, scratch):
    parts = tmp_path / 'dir' / 'users'
    parts.mkdir(parents=True)
    pd.DataFrame({'ID': range(5)}).to_parquet(parts / 'part-0.parquet')
    connector = kumo.FileConnector(root_dir=str(tmp_path / 'dir'))
    graph = kumo.Graph(tables={
        'users': kumo.Table.from_source_table(connector['users'],
                                              primary_key='ID')
    }, edges=[])
    snapshot = graph.snapshot()
    assert snapshot.source_changes(graph)['users']['status'] == 'unchanged'
    pd.DataFrame({'ID': range(5, 9)}).to_parquet(parts / 'part-1.parquet')
    assert snapshot.source_changes(graph)['users']['status'] == 'changed'


@pytest.mark.parametrize('record_sources', [True, False])
def test_source_changes_notices_a_rebound_source(tmp_path, scratch, graph,
                                                 dated, record_sources):
    import shutil

    from kumoapi.graph import Edge
    from kumoapi.typing import Stype

    snapshot = graph.snapshot(record_sources=record_sources)
    shutil.copytree(dated, tmp_path / 'other')
    connector = kumo.FileConnector(root_dir=str(tmp_path / 'other'))
    rebound = kumo.Graph(
        tables={
            'users':
            kumo.Table.from_source_table(connector['users.parquet'],
                                         primary_key='ID'),
            'events':
            kumo.Table.from_source_table(
                connector['events.parquet'], primary_key='EID',
                time_column='TS').with_column('USER_ID', stype=Stype.ID),
        }, edges=[Edge(src_table='events', fkey='USER_ID', dst_table='users')])

    changes = snapshot.source_changes(rebound)
    assert {change['status'] for change in changes.values()} == {'unknown'}
    assert changes['events']['recorded'] == changes['events']['current']


@pytest.mark.parametrize('edit', ['absent', 'reconstructed', 'hour_bucket'])
def test_source_changes_unknown_without_a_comparable_version(
        scratch, graph, dated, edit):
    snapshot = graph.snapshot()
    path = snapshot.index_path / '_provenance.json'
    manifest = json.loads(path.read_text())
    if edit == 'absent':
        path.unlink()
    elif edit == 'reconstructed':
        path.write_text(json.dumps({**manifest, 'reconstructed': True}))
    else:
        for entry in manifest['parents']['source_tables'].values():
            entry['version_kind'] = 'hour_bucket'
        path.write_text(json.dumps(manifest))
    assert {change['status'] for change in snapshot.source_changes(
        graph).values()} == {'unknown'}


def test_load_warns_only_with_graph_and_only_for_changed_tables(
        tmp_path, scratch, graph, dated, exports):
    location, _ = _export(graph.snapshot(), exports, tmp_path)
    scratch('gpu')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _load(exports, location, graph=graph)
        _rewrite_events(dated)
        _load(exports, location)
    with pytest.warns(UserWarning) as caught:
        _load(exports, location, graph=graph)
    [message] = [str(warning.message) for warning in caught]
    assert 'source table(s) events;' in message


class _Bytes:
    def __init__(self, data):
        self.data = data

    def read_at(self, offset, length):
        return self.data[offset:offset + length]

    def close(self):
        pass


def test_a_sink_with_only_a_filesystem_holds_exports(tmp_path, scratch,
                                                     graph):
    from kumo_core.chunked_fs import ChunkedFileHandler
    from pyarrow.fs import FileSelector, FileType, PyFileSystem

    class Store(ChunkedFileHandler):
        part_bytes = 1000

        def __init__(self):
            self.objects = {}

        def get_type_name(self):
            return 'memory'

        def _size(self, path):
            data = self.objects.get(path)
            return None if data is None else len(data)

        def _list(self, directory):
            return {p: len(d) for p, d in self.objects.items()
                    if p.startswith(directory + '/')}

        def _put(self, local, path):
            self.objects[path] = Path(local).read_bytes()

        def _part(self, path, size):
            return _Bytes(self.objects[path])

        def _remove(self, path, *, tree):
            for name in list(self.objects):
                if name.startswith(path + '/') if tree else name == path:
                    del self.objects[name]

    store = Store()
    sink = SimpleNamespace(
        holds_files=lambda connector, path: path.startswith('@'),
        filesystem=lambda connector, path: (PyFileSystem(store),
                                            path.rstrip('/')))
    connector = SimpleNamespace(sink_backend=sink)
    snapshot = graph.snapshot()
    location = snapshot.export(SnapshotOutputConfig(connector, '@stage/s'))
    assert any('.chunks/' in name for name in store.objects)
    scratch('gpu')
    loaded = _load(connector, location)
    assert loaded.id == snapshot.id
    filesystem, root = sink.filesystem(connector, location)
    with filesystem.open_input_stream(f'{root}/kumo_snapshot.json') as reader:
        generation = json.loads(reader.readall())['generation']
    filesystem, root = sink.filesystem(connector,
                                       f'{location}/{generation}/index')
    for info in filesystem.get_file_info(FileSelector(root, recursive=True)):
        if info.type is FileType.File:
            with filesystem.open_input_file(info.path) as reader:
                assert (loaded.index_path / info.path[len(root) + 1:]
                        ).read_bytes() == reader.read()
    with pytest.raises(NotImplementedError, match='file destination'):
        snapshot.export(SnapshotOutputConfig(connector, 'DB.SCHEMA'))


def test_a_mlops_block_stores_each_snapshot_once(tmp_path, scratch, graph,
                                                 monkeypatch):
    root = tmp_path / 'artifacts'
    cfg = kumo.MLOps(artifact_root=str(root))
    with kumo.mlops(cfg):
        snapshot = graph.snapshot()
        monkeypatch.setattr(snapshot_io, 'export',
                            Mock(side_effect=AssertionError('re-exported')))
        assert graph.snapshot() == snapshot
    graph_id, data_version = snapshot.id.split('@')
    stored = root / 'snapshots' / graph_id / data_version
    assert (stored / 'kumo_snapshot.json').is_file()
    assert (_generation(stored) / '_provenance.json').is_file()
    assert _load(kumo.FileConnector(root_dir=str(root)),
                 f'snapshots/{graph_id}/{data_version}') == snapshot
    assert GraphSnapshot.load_from_connector(cfg, id=snapshot.id) == snapshot
    with pytest.raises(TypeError, match='kumo.MLOps'):
        GraphSnapshot.load_from_connector(kumo.FileConnector(root_dir='.'),
                                          id=snapshot.id)
    graph.snapshot(mlops=kumo.MLOps(artifact_root=str(tmp_path / 'off'),
                                    store={'model'}))
    assert not (tmp_path / 'off').exists()


def test_loading_a_missing_export_names_its_full_path(tmp_path, exports):
    missing = tmp_path / 'exports' / 'snapshots' / 'graph' / 'version'
    with pytest.raises(FileNotFoundError,
                       match=f'No complete snapshot export at {missing}: '):
        _load(exports, 'snapshots/graph/version')


@pytest.mark.parametrize('on_error', ['warn', 'raise'])
def test_a_failed_store_follows_on_error(tmp_path, scratch, graph,
                                         monkeypatch, on_error):
    monkeypatch.setattr(snapshot_io, 'export',
                        Mock(side_effect=OSError('ACCESS_DENIED')))
    cfg = kumo.MLOps(str(tmp_path / 'root'), on_error=on_error)
    if on_error == 'raise':
        with pytest.raises(RuntimeError, match='Could not store snapshot'):
            graph.snapshot(mlops=cfg)
    else:
        with pytest.warns(UserWarning, match='ACCESS_DENIED'):
            graph.snapshot(mlops=cfg)
