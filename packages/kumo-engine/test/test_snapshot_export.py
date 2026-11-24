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
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from kumoapi.graph import Edge
from kumoapi.typing import Stype
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


def _put(snapshot, connector, path='one'):
    return snapshot.export(SnapshotOutputConfig(connector, path))


def _export(snapshot, exports, tmp_path):
    location = _put(snapshot, exports)
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


@pytest.mark.parametrize('portable', [False, True])
def test_export_load_round_trip_relocates_and_is_reused(
        tmp_path, scratch, graph, exports, monkeypatch, portable):
    if portable:
        # Linux's disk_usage uses statvfs internally; Windows uses its native
        # disk-space API. Keep that stdlib boundary available in this simulation.
        disk = shutil.disk_usage(tmp_path)
        monkeypatch.setattr(snapshot_io.shutil, 'disk_usage', lambda path: disk)
        for name in ('statvfs', 'pwrite', 'posix_fallocate'):
            monkeypatch.delattr(os, name, raising=False)
    monkeypatch.setattr(snapshot_io, '_CHUNK', 1024)
    monkeypatch.setattr(snapshot_io, '_RANGE_BYTES', 4096)
    snapshot = graph.snapshot()
    location = _put(snapshot, exports, 'one/')
    graph_id, data_version = snapshot.id.split('@')
    assert location == f'one/{graph_id}/{data_version}'
    assert _put(snapshot, exports) == location
    exported = tmp_path / 'exports' / location
    recorded = (snapshot.index_path / '_provenance.json').read_bytes()
    assert (_generation(exported) / '_provenance.json').read_bytes() == recorded

    shutil.rmtree(tmp_path / 'cpu')
    scratch('gpu')
    loaded = _load(exports, location, graph=graph)
    assert (loaded.id, loaded.graph_config) == (snapshot.id,
                                                snapshot.graph_config)
    assert loaded.index_path.is_relative_to(tmp_path / 'gpu')
    assert loaded.index_path == (tmp_path / 'gpu/indexes' /
                                 snapshot.index_path.name)
    assert _index_files(loaded.index_path) == _index_files(
        _generation(exported) / 'index')
    assert (loaded.index_path /
            '_provenance.json').read_bytes() == recorded
    reexported = tmp_path / 'exports' / _put(loaded, exports, 'two')
    assert (_generation(reexported) /
            '_provenance.json').read_bytes() == recorded

    monkeypatch.setattr(
        'kumo_core.diskgraph.ingest.execute_diskgraph_ingest',
        Mock(side_effect=AssertionError('ingest must not run')))
    monkeypatch.setattr(snapshot_io, 'load_into',
                        Mock(side_effect=AssertionError('already loaded')))
    assert _load(exports, location) == loaded
    assert graph.snapshot() == loaded


@pytest.mark.parametrize('operation', ['snapshot', 'import'])
def test_damaged_index_is_rebuilt_or_reloaded(
        tmp_path, scratch, graph, exports, operation):
    snapshot = graph.snapshot()
    location, _ = _export(snapshot, exports, tmp_path)
    expected = _index_files(snapshot.index_path)
    stats = snapshot.index_path / 'col_stats.json'
    stats.write_bytes(stats.read_bytes()[:-1])
    damaged = stats.read_bytes()
    from kumo.engine._monitoring import _local_index
    assert _local_index(snapshot.id) is None
    if operation == 'snapshot':
        with pytest.warns(UserWarning, match='Reloading the damaged'):
            rebuilt = graph.snapshot()
        assert rebuilt.id == snapshot.id
        assert rebuilt.index_path != snapshot.index_path
        assert (rebuilt.index_path / 'col_stats.json').read_bytes() == expected['col_stats.json']
        assert stats.read_bytes() == damaged
        assert _local_index(rebuilt.id) == rebuilt.index_path
        return

    with pytest.warns(UserWarning, match='Reloading the damaged'):
        loaded = _load(exports, location)
    assert loaded.index_path == snapshot.index_path
    assert _index_files(loaded.index_path) == expected
    quarantined = list((tmp_path / 'cpu/snapshots' / snapshot._graph_key /
                        '.damaged').glob('*/col_stats.json'))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == damaged
    from kumo.engine._monitoring import _local_index
    assert _local_index(loaded.id) == loaded.index_path


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
            _put(rebuilt, exports)
    assert _generation(exported) == first
    scratch('gpu')
    assert _index_files(_load(exports, location).index_path) == _index_files(
        snapshot.index_path)

    assert _put(rebuilt, exports) == location
    assert _generation(exported) != first and not first.exists()


def test_reexporting_an_identical_snapshot_reuses_the_export(
        tmp_path, scratch, graph, exports, monkeypatch, caplog):
    snapshot = graph.snapshot()
    location, exported = _export(snapshot, exports, tmp_path)
    first = _generation(exported)
    monkeypatch.setattr(snapshot_io, '_transfer',
                        Mock(side_effect=AssertionError('uploaded again')))
    with caplog.at_level('INFO', logger=snapshot_io.__name__):
        assert _put(snapshot, exports) == location
    assert _generation(exported) == first and 'reused it' in caplog.text

    (first / '_DONE').unlink()
    with pytest.raises(AssertionError, match='uploaded again'):
        _put(snapshot, exports)


def test_concurrent_exports_of_one_snapshot_both_succeed(
        tmp_path, scratch, graph, exports, monkeypatch):
    snapshot = graph.snapshot()
    transfer, raced = snapshot_io._transfer, []

    def racing(*args):
        if not raced:
            raced.append(None)
            raced[0] = _put(snapshot, exports)
        return transfer(*args)

    monkeypatch.setattr(snapshot_io, '_transfer', racing)
    assert raced == [_put(snapshot, exports)]
    assert _load(exports, raced[0]).id == snapshot.id


def _rebuilt(snapshot, tmp_path, name, *, retain_build_id=False):
    r"""Same logical id and sizes, other bytes and physical build identity."""
    from diskgraph import new_index_build_id

    index = (tmp_path / name / snapshot.index_path.parent.name /
             snapshot.index_path.name)
    shutil.copytree(snapshot.index_path, index)
    (index / '.kumo_export').unlink(missing_ok=True)
    stats = index / 'col_stats.json'
    data = stats.read_bytes()
    position = re.search(rb'[0-9]([0-9])', data).start(1)
    digit = b'1' if data[position:position + 1] != b'1' else b'2'
    stats.write_bytes(data[:position] + digit + data[position + 1:])
    if not retain_build_id:
        metadata = index / 'index_metadata.json'
        content = metadata.read_text()
        previous = json.loads(content)['index_build_id']
        metadata.write_text(content.replace(previous, new_index_build_id()))
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
            _put(first, exports)
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=export_first)
    worker.start()
    assert paused.wait(30)
    location = _put(second, exports)
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
    location = _put(first, exports)
    load_into, calls = snapshot_io.load_into, []

    def overtaken(*args):
        calls.append(args[2])
        if len(calls) == 1:
            _put(second, exports)
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
    one = _put(first, exports)
    two = _put(second, exports, 'two')
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
    _put(graph.snapshot(), exports)
    assert len(clients) == 2 and clients[0] is not clients[1]


def test_reexport_after_redaction_publishes_the_redacted_manifest(
        tmp_path, scratch, graph, exports):
    full = graph.snapshot()
    location, exported = _export(full, exports, tmp_path)
    first = _generation(exported)
    redacted = graph.snapshot(record_sources=False)
    assert redacted.id == full.id
    assert _put(redacted, exports) == location
    committed = _generation(exported)
    assert committed != first
    assert (committed / '_provenance.json').read_bytes() == (
        redacted.index_path / '_provenance.json').read_bytes()
    assert json.loads((committed / '_provenance.json').read_text())[
        'record_sources'] is False


@pytest.mark.parametrize('damage', ['missing_file', 'damaged'])
def test_reexport_replaces_an_export_that_is_not_an_intact_copy(
        tmp_path, scratch, graph, exports, damage):
    snapshot = graph.snapshot()
    location, exported = _export(snapshot, exports, tmp_path)
    first = _generation(exported)
    stats = first / 'index/col_stats.json'
    if damage == 'missing_file':
        stats.unlink()
    else:
        stats.write_bytes(stats.read_bytes()[:3])
        (first / 'index/index_metadata.json').unlink()
        _edit_manifest(exported, lambda m: m.update(snapshot_id='other'))
    assert _put(snapshot, exports) == location
    assert not first.exists()
    scratch('empty')
    assert _index_files(_load(exports, location).index_path) == _index_files(
        snapshot.index_path)


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
    graph_dir = tmp_path / 'gpu/snapshots' / snapshot._graph_key
    assert not [path for path in graph_dir.iterdir() if path.suffix != '.lock']

    stats.write_bytes(original)
    (graph_dir / f'{snapshot._data_version}.loading').mkdir()
    loaded = _load(exports, location)
    assert _index_files(loaded.index_path) == _index_files(snapshot.index_path)
    assert [
        path.name for path in graph_dir.iterdir() if path.suffix != '.lock'
    ] == [f'{snapshot._data_version}.index.json']


def test_load_refuses_before_downloading_when_disk_is_too_small(
        tmp_path, scratch, graph, exports, monkeypatch):
    location, _ = _export(graph.snapshot(), exports, tmp_path)
    scratch('gpu')
    monkeypatch.setattr(snapshot_io.shutil, 'disk_usage',
                        lambda path: SimpleNamespace(free=0))
    monkeypatch.setattr(snapshot_io, '_download',
                        Mock(side_effect=AssertionError('downloaded')))
    with pytest.raises(OSError, match='free bytes'):
        _load(exports, location)


@pytest.mark.parametrize('staging_name', [
    'staging',
    pytest.param('staging::literal', marks=pytest.mark.skipif(
        os.name != 'posix', reason='Literal colons require POSIX filenames')),
])
def test_parallel_ranges_preserve_binary_bytes(
        tmp_path, monkeypatch, staging_name):
    source = tmp_path / 'source/index'
    source.mkdir(parents=True)
    payload = bytes(range(256)) * 97 + b'\r\n\x1a\n'
    (source / 'data').write_bytes(payload)
    staging = tmp_path / staging_name
    staging.mkdir()
    (staging / 'data').write_bytes(b'\x00' * len(payload))
    monkeypatch.delattr(os, 'pwrite', raising=False)
    monkeypatch.setattr(snapshot_io, '_CHUNK', 257)
    stop = threading.Event()
    ranges = [('data', offset, min(1021, len(payload) - offset))
              for offset in range(0, len(payload), 1021)]

    def download(item):
        snapshot_io._download(str(source.parent), staging, LocalFileSystem(),
                              item, stop)

    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(download, reversed(ranges)))

    assert (staging / 'data').read_bytes() == payload


@pytest.mark.parametrize('free', [0, 1024])
def test_portable_allocation_checks_free_space(tmp_path, monkeypatch, free):
    for name in ('statvfs', 'posix_fallocate'):
        monkeypatch.delattr(os, name, raising=False)
    monkeypatch.setattr(snapshot_io.shutil, 'disk_usage',
                        lambda path: SimpleNamespace(free=free))
    staging = tmp_path / 'staging'
    files = [('nested/data', 1024), ('empty', 0)]

    if free < 1024:
        with pytest.raises(OSError, match='1024 free bytes'):
            snapshot_io._allocate(staging, files)
        assert list(staging.iterdir()) == []
    else:
        snapshot_io._allocate(staging, files)
        assert (staging / 'nested/data').stat().st_size == 1024
        assert (staging / 'empty').stat().st_size == 0


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
    with pytest.raises(NotImplementedError, match='file connector'):
        _put(graph.snapshot(), connector, 'x')
    with pytest.raises(NotImplementedError, match='file connector'):
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
    location = _put(snapshot, connector, f'{scheme}://bucket/snapshots/')
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
    location = _put(snapshot, connector, '@stage/s')
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
    with pytest.raises(NotImplementedError, match='S3, GCS or file connector'):
        _put(snapshot, connector, 'DB.SCHEMA')


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
                       match=f'No snapshot export at {missing} '):
        _load(exports, 'snapshots/graph/version')


@pytest.mark.parametrize('on_error, outcome', [
    ('warn', partial(pytest.warns, UserWarning, match='ACCESS_DENIED')),
    ('raise',
     partial(pytest.raises, RuntimeError, match='Could not store snapshot')),
])
def test_a_failed_store_follows_on_error(tmp_path, scratch, graph,
                                         monkeypatch, on_error, outcome):
    monkeypatch.setattr(snapshot_io, 'export',
                        Mock(side_effect=OSError('ACCESS_DENIED')))
    with outcome():
        graph.snapshot(mlops=kumo.MLOps(str(tmp_path / 'root'),
                                        on_error=on_error))


def test_managed_snapshot_reuse_and_legacy_compatibility(
        tmp_path, scratch, graph, exports):
    snapshot = graph.snapshot()
    build_id = json.loads(
        (snapshot.index_path / 'index_metadata.json').read_text())['index_build_id']
    assert snapshot.index_path == tmp_path / 'cpu/indexes' / build_id
    assert graph.snapshot().index_path == snapshot.index_path
    reference = (tmp_path / 'cpu/snapshots' / snapshot._graph_key /
                 f'{snapshot._data_version}.index.json')
    assert json.loads(reference.read_text()) == {'index_build_id': build_id}
    # Simulate an existing snapshot from before the layout migration.
    legacy = reference.with_name(snapshot._data_version)
    snapshot.index_path.rename(legacy)
    from kumo_core.diskgraph.snapshot import (
        DiskgraphIngestResult,
        write_diskgraph_success_marker,
    )
    write_diskgraph_success_marker(str(legacy), DiskgraphIngestResult.for_index(legacy))
    reference.unlink()
    reused = graph.snapshot()
    assert reused.id == snapshot.id
    assert reused.index_path == legacy
    location = _put(reused, exports)
    scratch('restored')
    restored = _load(exports, location)
    assert restored.id == snapshot.id
    assert restored._data_version == snapshot._data_version
    assert _index_files(restored.index_path) == _index_files(legacy)


def test_import_does_not_replace_a_managed_base(tmp_path, scratch, graph, exports):
    first = graph.snapshot()
    first_bytes = _index_files(first.index_path)
    reference = (tmp_path / 'cpu/snapshots' / first._graph_key /
                 f'{first._data_version}.index.json')
    reference.unlink()
    second = graph.snapshot()
    assert second.id == first.id
    assert second.index_path != first.index_path
    second_bytes = _index_files(second.index_path)
    location = _put(first, exports)
    restored = _load(exports, location)
    assert restored.index_path == first.index_path
    assert _index_files(first.index_path) == first_bytes
    assert _index_files(second.index_path) == second_bytes
    assert _index_files(restored.index_path) == first_bytes
    assert graph.snapshot() == restored
    assert json.loads(reference.read_text()) == {
        'index_build_id': first.index_path.name}
    assert set((tmp_path / 'cpu/indexes').iterdir()) == {
        first.index_path, second.index_path}
    assert not reference.with_name(first._data_version).exists()


def test_import_reuses_an_unselected_build_without_downloading(
        tmp_path, scratch, graph, exports, monkeypatch):
    first = graph.snapshot()
    one = _put(first, exports)
    second = _rebuilt(first, tmp_path, 'second')
    two = _put(second, exports, 'two')
    scratch('imported')
    imported_first = _load(exports, one)
    imported_second = _load(exports, two)
    before = {p: _index_files(p) for p in
              (imported_first.index_path, imported_second.index_path)}
    monkeypatch.setattr(snapshot_io, 'load_into',
                        Mock(side_effect=AssertionError('downloaded again')))
    assert _load(exports, one).index_path == imported_first.index_path
    assert graph.snapshot().index_path == imported_first.index_path
    assert set((tmp_path / 'imported/indexes').iterdir()) == set(before)
    assert {p: _index_files(p) for p in before} == before
    graph_dir = tmp_path / 'imported/snapshots' / first._graph_key
    assert {p.name for p in graph_dir.iterdir()} == {
        'source_tables.lock', f'{first._data_version}.index.json'}


def test_import_rejects_conflicting_contents_for_an_existing_build(
        tmp_path, scratch, graph, exports):
    first = graph.snapshot()
    before = _index_files(first.index_path)
    reference = (tmp_path / 'cpu/snapshots' / first._graph_key /
                 f'{first._data_version}.index.json')
    selected = reference.read_bytes()
    conflicting = _rebuilt(first, tmp_path, 'conflicting', retain_build_id=True)
    location = _put(conflicting, exports)
    with pytest.raises(ValueError, match='Conflicting contents for index build'):
        _load(exports, location)
    assert _index_files(first.index_path) == before
    assert reference.read_bytes() == selected


def test_failed_import_preserves_the_selected_managed_build(
        tmp_path, scratch, graph, exports, monkeypatch):
    first = graph.snapshot()
    before = _index_files(first.index_path)
    reference = (tmp_path / 'cpu/snapshots' / first._graph_key /
                 f'{first._data_version}.index.json')
    selected = reference.read_bytes()
    other = _rebuilt(first, tmp_path, 'other')
    location = _put(other, exports)
    monkeypatch.setattr(snapshot_io, 'load_into',
                        Mock(side_effect=OSError('transfer failed')))
    with pytest.raises(OSError, match='transfer failed'):
        _load(exports, location)
    assert reference.read_bytes() == selected
    assert _index_files(first.index_path) == before
    assert graph.snapshot() == first


def test_idless_legacy_export_remains_importable(
        tmp_path, scratch, graph, exports):
    snapshot = graph.snapshot()
    metadata_path = snapshot.index_path / 'index_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.pop('index_build_id')
    metadata_path.write_text(json.dumps(metadata))
    manifest_path = snapshot.index_path / '_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest['files']:
        if entry['rel_path'] == metadata_path.name:
            entry['size_bytes'] = metadata_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest))
    location = _put(snapshot, exports)
    scratch('legacy-import')
    loaded = _load(exports, location)
    assert loaded.index_path == (tmp_path / 'legacy-import/snapshots' /
                                 snapshot._graph_key / snapshot._data_version)
    assert _index_files(loaded.index_path) == _index_files(snapshot.index_path)
    assert not (tmp_path / 'legacy-import/indexes').exists()


@pytest.mark.parametrize('operation', ['snapshot', 'import'])
@pytest.mark.parametrize('damage', [
    'reference_json', 'reference_shape', 'reference_id', 'reference_traversal',
    'metadata_missing', 'metadata_json', 'metadata_directory', 'marker_missing',
    'artifact_missing',
    'legacy_metadata_missing',
])
def test_damaged_snapshot_state_is_a_cache_miss(
        tmp_path, scratch, graph, exports, operation, damage):
    from kumo_core.diskgraph.snapshot import (
        DiskgraphIngestResult,
        write_diskgraph_success_marker,
    )

    from kumo.engine._monitoring import _local_index

    snapshot = graph.snapshot()
    location = _put(snapshot, exports)
    original = snapshot.index_path
    metadata = json.loads((original / 'index_metadata.json').read_text())
    reference = (tmp_path / 'cpu/snapshots' / snapshot._graph_key /
                 f'{snapshot._data_version}.index.json')
    if damage.startswith('reference_'):
        reference.write_text({
            'reference_json': '{broken',
            'reference_shape': '[]',
            'reference_id': '{"index_build_id": 1}',
            'reference_traversal': '{"index_build_id": "../outside"}',
        }[damage])
    else:
        if damage.startswith('legacy_'):
            legacy = reference.with_name(snapshot._data_version)
            original.rename(legacy)
            original = legacy
            reference.unlink()
            write_diskgraph_success_marker(
                str(legacy), DiskgraphIngestResult.for_index(legacy))
        if damage.endswith('metadata_missing'):
            (original / 'index_metadata.json').unlink()
        elif damage == 'metadata_json':
            (original / 'index_metadata.json').write_text('{broken')
        elif damage == 'metadata_directory':
            (original / 'index_metadata.json').unlink()
            (original / 'index_metadata.json').mkdir()
        elif damage == 'marker_missing':
            (original / '_SUCCESS').unlink()
        elif damage == 'artifact_missing':
            (original / 'col_stats.json').unlink()
    assert _local_index(snapshot.id) is None
    retained = _index_files(original)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        recovered = (graph.snapshot() if operation == 'snapshot' else
                     _load(exports, location))
    assert recovered.id == snapshot.id
    build_id = json.loads(
        (recovered.index_path / 'index_metadata.json').read_text())['index_build_id']
    assert recovered.index_path == tmp_path / 'cpu/indexes' / build_id
    assert json.loads(reference.read_text()) == {'index_build_id': build_id}
    assert _local_index(recovered.id) == recovered.index_path
    assert graph.snapshot() == recovered
    if operation == 'snapshot':
        assert build_id != metadata['index_build_id']
        if not damage.startswith('legacy_'):
            assert _index_files(original) == retained
    else:
        assert build_id == metadata['index_build_id']
    assert not reference.with_name(snapshot._data_version).exists()


@pytest.mark.parametrize('operation', ['snapshot', 'import', 'monitoring'])
def test_snapshot_cache_io_errors_are_not_treated_as_damage(
        scratch, graph, exports, monkeypatch, operation):
    import kumo_core.diskgraph.snapshot as snapshots

    from kumo.engine._monitoring import _local_index

    snapshot = graph.snapshot()
    location = _put(snapshot, exports)

    def denied(*args):
        raise PermissionError('snapshot permission denied')

    monkeypatch.setattr(snapshots, '_read_json_object', denied)
    monkeypatch.setattr(snapshot_io, 'load_into',
                        Mock(side_effect=AssertionError('must not restore')))
    monkeypatch.setattr('kumo_core.diskgraph.ingest.execute_diskgraph_ingest',
                        Mock(side_effect=AssertionError('must not ingest')))
    with pytest.raises(PermissionError, match='snapshot permission denied'):
        if operation == 'snapshot':
            graph.snapshot()
        elif operation == 'import':
            _load(exports, location)
        else:
            _local_index(snapshot.id)


@pytest.mark.parametrize('operation', ['snapshot', 'import', 'monitoring'])
def test_native_verification_io_errors_are_not_treated_as_damage(
        scratch, graph, exports, monkeypatch, operation):
    from diskgraph import DiskgraphCorruptionError

    from kumo.engine._monitoring import _local_index

    snapshot = graph.snapshot()
    location = _put(snapshot, exports)

    def denied(path):
        raise DiskgraphCorruptionError(f'io error verifying {path}: Permission denied')

    monkeypatch.setattr('diskgraph.verify_snapshot_integrity', denied)
    monkeypatch.setattr(snapshot_io, 'load_into',
                        Mock(side_effect=AssertionError('must not restore')))
    monkeypatch.setattr('kumo_core.diskgraph.ingest.execute_diskgraph_ingest',
                        Mock(side_effect=AssertionError('must not ingest')))
    with pytest.raises(OSError, match='io error verifying'):
        if operation == 'snapshot':
            graph.snapshot()
        elif operation == 'import':
            _load(exports, location)
        else:
            _local_index(snapshot.id)


@pytest.mark.parametrize('reference_state', [
    'valid', 'absent', 'json', 'shape', 'id', 'traversal', 'target_missing',
    'metadata_missing', 'metadata_json', 'metadata_directory', 'metadata_mismatch',
    'marker_missing', 'artifact_missing',
])
@pytest.mark.parametrize('legacy_state', ['healthy', 'absent', 'metadata'])
@pytest.mark.parametrize('operation', ['snapshot', 'import'])
def test_completed_snapshot_fallback_and_recovery_combinations(
        tmp_path, scratch, graph, exports, monkeypatch,
        reference_state, legacy_state, operation):
    from kumo_core.diskgraph.snapshot import (
        DiskgraphIngestResult,
        DiskgraphSnapshotHandle,
        write_diskgraph_success_marker,
    )

    from kumo.engine._monitoring import _local_index
    from kumo.engine.graph import _store

    snapshot = graph.snapshot()
    location = _put(snapshot, exports)
    store = _store()
    handle = DiskgraphSnapshotHandle(snapshot._graph_key, snapshot._data_version)
    managed = snapshot.index_path
    legacy = Path(store.staging_path(handle))
    if legacy_state != 'absent':
        shutil.copytree(managed, legacy)
        write_diskgraph_success_marker(str(legacy), DiskgraphIngestResult.for_index(legacy))
        if legacy_state == 'metadata':
            (legacy / 'index_metadata.json').write_text('{broken')
    reference = store._reference(handle)
    if reference_state == 'absent':
        reference.unlink()
    elif reference_state in {'json', 'shape', 'id', 'traversal'}:
        reference.write_text({
            'json': '{broken', 'shape': '[]', 'id': '{"index_build_id": 1}',
            'traversal': '{"index_build_id": "../outside"}',
        }[reference_state])
    elif reference_state == 'target_missing':
        shutil.rmtree(managed)
    elif reference_state == 'metadata_missing':
        (managed / 'index_metadata.json').unlink()
    elif reference_state == 'metadata_json':
        (managed / 'index_metadata.json').write_text('{broken')
    elif reference_state == 'metadata_directory':
        (managed / 'index_metadata.json').unlink()
        (managed / 'index_metadata.json').mkdir()
    elif reference_state == 'metadata_mismatch':
        (managed / 'index_metadata.json').write_text('{"index_build_id":"other"}')
    elif reference_state == 'marker_missing':
        (managed / '_SUCCESS').unlink()
    elif reference_state == 'artifact_missing':
        (managed / 'col_stats.json').unlink()
    expected = (managed if reference_state == 'valid' else
                legacy if legacy_state == 'healthy' else None)
    root = tmp_path / 'cpu'
    before = _index_files(root)
    inodes = {name: (root / name).stat().st_ino for name in before}
    reference_before = reference.read_bytes() if reference.is_file() else None

    assert _local_index(snapshot.id) == expected
    assert store.list_ready() == ([handle] if expected is not None else [])
    if expected is not None:
        assert store.path(handle) == str(expected)
        assert store.verify(handle).index_path == str(expected)
        # Victsm's healthy-legacy/broken-reference case must reuse rather than
        # deleting the retained copy or restoring/re-ingesting another one.
        monkeypatch.setattr('kumo_core.diskgraph.ingest.execute_diskgraph_ingest',
                            Mock(side_effect=AssertionError('must reuse completed snapshot')))
        monkeypatch.setattr(snapshot_io, 'load_into',
                            Mock(side_effect=AssertionError('must not restore completed snapshot')))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        recovered = graph.snapshot() if operation == 'snapshot' else _load(exports, location)
    assert recovered.id == snapshot.id
    assert _local_index(snapshot.id) == recovered.index_path
    assert graph.snapshot() == recovered
    if expected is not None:
        assert recovered.index_path == expected
        assert _index_files(root) == before
        assert {name: (root / name).stat().st_ino for name in before} == inodes
        assert (reference.read_bytes() if reference.is_file() else None) == reference_before
    else:
        build = json.loads((recovered.index_path / 'index_metadata.json').read_text())[
            'index_build_id']
        assert recovered.index_path == root / 'indexes' / build
        assert (build != managed.name) == (operation == 'snapshot')


@pytest.mark.parametrize('requested_copy', ['existing', 'download', 'failed_download'])
def test_importing_another_build_preserves_a_healthy_legacy_fallback(
        tmp_path, scratch, graph, exports, monkeypatch, requested_copy):
    from kumo_core.diskgraph.snapshot import (
        DiskgraphIngestResult,
        DiskgraphSnapshotHandle,
        write_diskgraph_success_marker,
    )

    from kumo.engine._monitoring import _local_index
    from kumo.engine.graph import _store

    requested = graph.snapshot()
    location = _put(requested, exports)
    store = _store()
    handle = DiskgraphSnapshotHandle(requested._graph_key, requested._data_version)
    reference = store._reference(handle)
    reference.unlink()
    other = graph.snapshot()
    legacy = Path(store.staging_path(handle))
    other.index_path.rename(legacy)
    write_diskgraph_success_marker(str(legacy), DiskgraphIngestResult.for_index(legacy))
    reference.write_text('{broken')
    if requested_copy != 'existing':
        shutil.rmtree(requested.index_path)
    before = _index_files(legacy)
    inodes = {name: (legacy / name).stat().st_ino for name in before}
    retained_reference = reference.read_bytes()
    real_load, copied = snapshot_io.load_into, []

    def load(connector, path, root, manifest, target):
        assert target.is_relative_to(tmp_path / 'cpu/snapshots/.imports')
        assert target != legacy
        copied.append(target)
        if requested_copy == 'failed_download':
            target.mkdir(parents=True)
            (target / 'partial').write_bytes(b'partial transfer')
            raise OSError('transfer failed')
        return real_load(connector, path, root, manifest, target)

    monkeypatch.setattr(snapshot_io, 'load_into', load)
    monkeypatch.setattr('kumo_core.diskgraph.ingest.execute_diskgraph_ingest',
                        Mock(side_effect=AssertionError('must not re-ingest')))
    if requested_copy == 'failed_download':
        with pytest.raises(OSError, match='transfer failed'):
            _load(exports, location)
        assert reference.read_bytes() == retained_reference
        assert _local_index(requested.id) == legacy
        assert graph.snapshot().index_path == legacy
    else:
        loaded = _load(exports, location)
        assert loaded.index_path == requested.index_path
        assert graph.snapshot() == loaded
        assert _local_index(loaded.id) == loaded.index_path
    assert bool(copied) == (requested_copy != 'existing')
    assert _index_files(legacy) == before
    assert {name: (legacy / name).stat().st_ino for name in before} == inodes
    assert store.list_ready() == [handle]
    assert not list((tmp_path / 'cpu/snapshots/.imports').rglob('_SUCCESS'))
    assert all(not target.exists() for target in copied)


def test_a_different_idless_export_cannot_replace_a_healthy_legacy_fallback(
        tmp_path, scratch, graph, exports, monkeypatch):
    from kumo_core.diskgraph.snapshot import (
        DiskgraphIngestResult,
        DiskgraphSnapshotHandle,
        write_diskgraph_success_marker,
    )

    from kumo.engine._monitoring import _local_index
    from kumo.engine.graph import _store

    retained = graph.snapshot()
    other = _rebuilt(retained, tmp_path, 'idless')
    metadata = other.index_path / 'index_metadata.json'
    payload = json.loads(metadata.read_text())
    payload.pop('index_build_id')
    metadata.write_text(json.dumps(payload))
    manifest = other.index_path / '_MANIFEST.json'
    payload = json.loads(manifest.read_text())
    for entry in payload['files']:
        entry['size_bytes'] = (other.index_path / entry['rel_path']).stat().st_size
    manifest.write_text(json.dumps(payload))
    location = _put(other, exports)
    store = _store()
    handle = DiskgraphSnapshotHandle(retained._graph_key, retained._data_version)
    legacy = Path(store.staging_path(handle))
    retained.index_path.rename(legacy)
    write_diskgraph_success_marker(str(legacy), DiskgraphIngestResult.for_index(legacy))
    reference = store._reference(handle)
    reference.write_text('{broken')
    before = _index_files(legacy), reference.read_bytes()
    inodes = {name: (legacy / name).stat().st_ino for name in before[0]}
    monkeypatch.setattr(snapshot_io, 'load_into',
                        Mock(side_effect=AssertionError('must not overwrite legacy')))
    monkeypatch.setattr('kumo_core.diskgraph.ingest.execute_diskgraph_ingest',
                        Mock(side_effect=AssertionError('must reuse legacy')))

    with pytest.raises(ValueError, match='ID-less export cannot replace'):
        _load(exports, location)
    assert (_index_files(legacy), reference.read_bytes()) == before
    assert {name: (legacy / name).stat().st_ino for name in before[0]} == inodes
    assert _local_index(retained.id) == legacy
    assert graph.snapshot().index_path == legacy
