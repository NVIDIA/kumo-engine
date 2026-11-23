# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from kumo_core.diskgraph.snapshot import (
    DiskgraphIngestResult,
    DiskgraphLocalConfig,
    DiskgraphSnapshotError,
    DiskgraphSnapshotHandle,
    LocalSnapshotStore,
    check_diskgraph_output,
    rebased,
    validate_cached_source_times,
)


def _manifest(path):
    (path / '_MANIFEST.json').write_text(json.dumps({
        'schema_version': 1,
        'files': [{'rel_path': child.name, 'size_bytes': child.stat().st_size}
                  for child in path.iterdir()
                  if child.is_file() and child.name not in ('_SUCCESS', '_MANIFEST.json')],
    }))


@pytest.fixture
def staged(tmp_path):
    store = LocalSnapshotStore(DiskgraphLocalConfig(
        str(tmp_path / 'snapshots'), managed_root=str(tmp_path)))
    handle = DiskgraphSnapshotHandle('graph', 'version')
    path = Path(store.staging_path(handle))
    path.mkdir(parents=True)
    result = DiskgraphIngestResult.for_index(path)
    (path / 'index_metadata.json').write_text(json.dumps({'index_build_id': 'build'}))
    (path / '_SUCCESS').write_text(result.to_json())
    (path / 'col_stats.json').write_text('{}')
    (path / 'ingest_stats.json').write_text('{}')
    (path / 'kumo_col_stats').mkdir()
    _manifest(path)
    return store, handle, result, path


def test_publish_moves_bytes_and_rebases_results(staged, tmp_path):
    store, handle, result, source = staged
    inode = (source / 'col_stats.json').stat().st_ino
    published = store.publish(handle, result)
    target = tmp_path / 'indexes' / 'build'
    assert published.index_path == str(target)
    assert published.missing_paths() == []
    assert (target / 'col_stats.json').stat().st_ino == inode
    assert not source.exists()
    reopened = LocalSnapshotStore(store.config)
    assert reopened.path(handle) == str(target)
    assert reopened.predicted_result(handle) == published


def _durable_copy(store, handle, source, tmp_path):
    root = tmp_path / 'durable'
    destination = Path(store.durable_path(handle, str(root)))
    shutil.copytree(source, destination)
    return str(root), destination


def _files(path):
    return {str(p.relative_to(path)): p.read_bytes()
            for p in path.rglob('*') if p.is_file()}


def test_restore_reuses_a_published_build_without_copying(staged, tmp_path):
    store, handle, result, source = staged
    durable_root, _ = _durable_copy(store, handle, source, tmp_path)
    published = Path(store.publish(handle, result).index_path)
    before = _files(published), store._reference(handle).read_bytes()

    def copy_tree(*args):
        pytest.fail('A verified published build must not be copied over')

    restored = store.restore(handle, durable_root, exists=lambda p: Path(p).exists(),
                             copy_tree=copy_tree)
    assert restored.index_path == str(published)
    assert (_files(published), store._reference(handle).read_bytes()) == before


@pytest.mark.parametrize('identified', [False, True])
@pytest.mark.parametrize('stale_staging', [False, True])
def test_restore_stages_and_publishes_new_managed_copies(
        staged, tmp_path, identified, stale_staging):
    store, handle, _, source = staged
    if not identified:
        (source / 'index_metadata.json').write_text('{}')
        _manifest(source)
    durable_root, durable = _durable_copy(store, handle, source, tmp_path)
    expected = (durable / 'col_stats.json').read_bytes()
    shutil.rmtree(source)  # The copied marker now names a vanished location.
    if stale_staging:
        source.mkdir()
        (source / '_SUCCESS').write_text('interrupted old copy')
    destinations = []

    def copy_tree(origin, destination):
        destinations.append(Path(destination))
        assert Path(destination) == source
        assert not Path(destination).is_relative_to(tmp_path / 'indexes')
        shutil.copytree(origin, destination)

    restored = store.restore(handle, durable_root, exists=lambda p: Path(p).exists(),
                             copy_tree=copy_tree)
    expected_path = tmp_path / 'indexes/build' if identified else source
    assert restored.index_path == str(expected_path)
    assert destinations == [source]
    assert store.path(handle) == str(expected_path)
    assert store.verify(handle) == restored
    assert (expected_path / 'col_stats.json').read_bytes() == expected
    assert store._reference(handle).is_file() == identified


def test_restore_does_not_overwrite_a_valid_legacy_copy_hidden_by_a_bad_reference(
        staged, tmp_path):
    store, handle, _, source = staged
    durable_root, _ = _durable_copy(store, handle, source, tmp_path)
    store._reference(handle).write_text('damaged reference')
    before = _files(source), store._reference(handle).read_bytes()

    def copy_tree(*args):
        pytest.fail('A valid retained legacy snapshot must not be overwritten')

    with pytest.raises(FileExistsError, match='completed snapshot'):
        store.restore(handle, durable_root, exists=lambda p: Path(p).exists(),
                      copy_tree=copy_tree)
    assert (_files(source), store._reference(handle).read_bytes()) == before


@pytest.mark.parametrize('failure', ['copy', 'integrity'])
def test_failed_restore_preserves_published_bytes_and_reference(
        staged, tmp_path, failure):
    store, handle, result, source = staged
    durable_root, _ = _durable_copy(store, handle, source, tmp_path)
    published = Path(store.publish(handle, result).index_path)
    (published / 'index_metadata.json').unlink()
    before = _files(published), store._reference(handle).read_bytes()
    destinations = []

    def copy_tree(origin, destination):
        destinations.append(Path(destination))
        assert Path(destination) == source
        assert not Path(destination).is_relative_to(tmp_path / 'indexes')
        shutil.copytree(origin, destination)
        if failure == 'copy':
            raise OSError('interrupted copy')
        (Path(destination) / 'col_stats.json').write_text('truncated')

    from diskgraph import DiskgraphCorruptionError

    with pytest.raises(OSError if failure == 'copy' else DiskgraphCorruptionError):
        store.restore(handle, durable_root, exists=lambda p: Path(p).exists(),
                      copy_tree=copy_tree)
    assert destinations == [source]
    assert (_files(published), store._reference(handle).read_bytes()) == before
    assert not source.exists()


def test_restore_repairs_a_damaged_managed_build_after_staging(staged, tmp_path):
    store, handle, result, source = staged
    durable_root, _ = _durable_copy(store, handle, source, tmp_path)
    published = Path(store.publish(handle, result).index_path)
    (published / 'index_metadata.json').unlink()
    damaged = _files(published)

    def copy_tree(origin, destination):
        assert Path(destination) == source
        assert _files(published) == damaged
        shutil.copytree(origin, destination)

    restored = store.restore(handle, durable_root, exists=lambda p: Path(p).exists(),
                             copy_tree=copy_tree)
    assert restored.index_path == str(published)
    assert store.verify(handle) == restored
    assert not source.exists()
    quarantined = list((source.parent / '.damaged').iterdir())
    assert len(quarantined) == 1 and _files(quarantined[0]) == damaged


def test_crash_after_rename_preserves_a_recoverable_build(
        staged, tmp_path, monkeypatch):
    import kumo_core.diskgraph.snapshot as snapshots

    store, handle, result, source = staged
    target = tmp_path / 'indexes' / 'build'
    marker = (source / '_SUCCESS').read_bytes()
    metadata = (source / 'index_metadata.json').read_bytes()
    data = source / 'node-data.bin'
    data.write_bytes(b'completed node data')
    inode = data.stat().st_ino
    retained = tmp_path / 'indexes' / 'other-build' / 'node-data.bin'
    retained.parent.mkdir(parents=True)
    retained.write_bytes(b'other completed build')
    write_marker = snapshots.write_diskgraph_success_marker

    class Interrupted(BaseException):
        pass

    def interrupt_before_marker_rewrite(output_dir, published):
        assert Path(output_dir) == target
        assert not source.exists()
        assert target.is_dir()
        assert published.index_path == str(target)
        raise Interrupted

    monkeypatch.setattr(snapshots, 'write_diskgraph_success_marker',
                        interrupt_before_marker_rewrite)
    with pytest.raises(Interrupted):
        store.publish(handle, result)

    assert not source.exists()
    assert not store._reference(handle).exists()
    assert (target / '_SUCCESS').read_bytes() == marker
    assert (target / 'index_metadata.json').read_bytes() == metadata
    assert check_diskgraph_output(str(target)) is None
    assert result.missing_paths()

    # Recovery can address the physical build even while its logical reference
    # is absent. Automatic discovery and reattachment are tracked in #486.
    physical = SimpleNamespace(path=lambda _: str(target))
    recovered = rebased(physical, handle)
    assert recovered == DiskgraphIngestResult.for_index(target)
    assert recovered.missing_paths() == []
    write_marker(str(target), recovered)
    assert check_diskgraph_output(str(target)) == recovered
    assert (target / 'node-data.bin').read_bytes() == b'completed node data'
    assert (target / 'node-data.bin').stat().st_ino == inode
    assert retained.read_bytes() == b'other completed build'
    assert not store._reference(handle).exists()


def test_legacy_location_remains_readable(staged):
    store, handle, _, source = staged
    assert store.path(handle) == str(source)
    assert not store._reference(handle).exists()


def test_publish_does_not_replace_an_existing_build(staged, tmp_path):
    store, handle, result, source = staged
    target = tmp_path / 'indexes' / 'build'
    target.mkdir(parents=True)
    (target / 'sentinel').write_text('keep')
    with pytest.raises(FileExistsError):
        store.publish(handle, result)
    assert source.exists()
    assert (target / 'sentinel').read_text() == 'keep'
    assert not store._reference(handle).exists()

    # The normal cache lookup can still reuse the retained completed staging.
    assert store.path(handle) == str(source)
    assert check_diskgraph_output(store.path(handle)) == result
    retained = rebased(store, handle)
    assert retained == result
    with pytest.raises(FileExistsError):
        store.publish(handle, retained)
    assert check_diskgraph_output(str(source)) == result
    assert (target / 'sentinel').read_text() == 'keep'
    assert not store._reference(handle).exists()


@pytest.mark.parametrize('damage', ['missing_id', 'incomplete'])
def test_invalid_build_is_not_published(staged, damage):
    store, handle, result, source = staged
    if damage == 'missing_id':
        (source / 'index_metadata.json').write_text('{}')
    else:
        (source / '_SUCCESS').unlink()
    with pytest.raises(DiskgraphSnapshotError):
        store.publish(handle, result)
    assert source.exists()
    assert not store._reference(handle).exists()


def test_reference_checks_physical_identity(staged, tmp_path):
    store, handle, result, _ = staged
    store.publish(handle, result)
    (tmp_path / 'indexes/build/index_metadata.json').write_text(
        json.dumps({'index_build_id': 'wrong'}))
    with pytest.raises(ValueError, match='mismatch'):
        store.path(handle)


def test_evicted_build_falls_back_to_staging(staged, tmp_path):
    import shutil
    store, handle, result, source = staged
    store.publish(handle, result)
    shutil.rmtree(tmp_path / 'indexes/build')
    assert store.path(handle) == str(source)


def test_listing_includes_managed_and_idless_legacy_snapshots(staged, tmp_path):
    store, handle, result, _ = staged
    store.publish(handle, result)
    assert store.list_ready() == [handle]
    legacy = DiskgraphSnapshotHandle('old', 'v0')
    path = Path(store.staging_path(legacy))
    path.mkdir(parents=True)
    (path / 'index_metadata.json').write_text('{}')
    (path / 'col_stats.json').write_text('{}')
    (path / 'ingest_stats.json').write_text('{}')
    (path / 'kumo_col_stats').mkdir()
    (path / '_SUCCESS').write_text(DiskgraphIngestResult.for_index(path).to_json())
    _manifest(path)
    assert set(store.list_ready()) == {handle, legacy}
    assert store.path(legacy) == str(path)
    store._reference(handle).write_text('{invalid')
    assert store.list_ready() == [legacy]


def _completed_legacy(store, handle):
    path = Path(store.staging_path(handle))
    path.mkdir(parents=True)
    (path / 'index_metadata.json').write_text('{}')
    (path / 'col_stats.json').write_text('{}')
    (path / 'ingest_stats.json').write_text('{}')
    (path / 'kumo_col_stats').mkdir()
    (path / '_SUCCESS').write_text(DiskgraphIngestResult.for_index(path).to_json())
    (path / '_MANIFEST.json').write_text(json.dumps({
        'schema_version': 1,
        'files': [{'rel_path': name, 'size_bytes': (path / name).stat().st_size}
                  for name in ('index_metadata.json', 'col_stats.json',
                               'ingest_stats.json')],
    }))
    return path


def test_legacy_directory_with_reference_suffix_remains_readable(staged):
    store, _, _, _ = staged
    legacy = DiskgraphSnapshotHandle('graph', 'old.index.json')
    path = _completed_legacy(store, legacy)

    assert store.path(legacy) == str(path)
    assert legacy in store.list_ready()
    assert DiskgraphSnapshotHandle('graph', 'old') not in store.list_ready()
    assert store.resolve_version_prefix('graph', 'old') == str(path)


def test_reserved_version_publication_is_rejected_before_mutation(staged):
    store, _, result, source = staged
    reserved = DiskgraphSnapshotHandle('graph', 'version.index.json')
    marker = (source / '_SUCCESS').read_bytes()

    with pytest.raises(ValueError, match='must not end'):
        store.publish(reserved, result)

    assert source.is_dir()
    assert (source / '_SUCCESS').read_bytes() == marker
    assert not (source.parents[2] / 'indexes').exists()
    assert not store._reference(reserved).exists()


def test_reference_publication_does_not_replace_a_legacy_directory(staged):
    store, handle, result, source = staged
    legacy = DiskgraphSnapshotHandle(
        handle.graph_key, handle.graph_data_version + '.index.json')
    legacy_path = _completed_legacy(store, legacy)
    marker = (legacy_path / '_SUCCESS').read_bytes()

    with pytest.raises(FileExistsError, match='reference is a directory'):
        store.publish(handle, result)

    assert source.is_dir()
    assert (legacy_path / '_SUCCESS').read_bytes() == marker
    assert not (source.parents[2] / 'indexes').exists()
    assert store.path(legacy) == str(legacy_path)
    assert set(store.list_ready()) == {handle, legacy}


def test_listing_deduplicates_handles_and_has_stable_order(staged):
    store, handle, result, _ = staged
    store.publish(handle, result)
    _completed_legacy(store, handle)
    first = DiskgraphSnapshotHandle(handle.graph_key, 'a-version')
    _completed_legacy(store, first)

    assert store.list_ready() == [first, handle]
    assert store.list_ready() == [first, handle]


@pytest.mark.parametrize('payload', [
    [], None, 1, 'reference', {}, {'index_build_id': None},
    {'index_build_id': []}, {'index_build_id': 3}, {'index_build_id': ''},
])
def test_invalid_reference_shapes_do_not_break_listing(staged, payload):
    store, handle, result, _ = staged
    store.publish(handle, result)
    legacy = DiskgraphSnapshotHandle('other', 'legacy')
    _completed_legacy(store, legacy)
    store._reference(handle).write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        store.path(handle)
    assert store.list_ready() == [legacy]


@pytest.mark.parametrize('payload', [[], None, 1, 'metadata', {},
                                     {'index_build_id': 3}])
def test_invalid_managed_metadata_does_not_break_listing(
        staged, tmp_path, payload):
    store, handle, result, _ = staged
    store.publish(handle, result)
    legacy = DiskgraphSnapshotHandle('other', 'legacy')
    _completed_legacy(store, legacy)
    (tmp_path / 'indexes/build/index_metadata.json').write_text(
        json.dumps(payload))

    with pytest.raises(ValueError):
        store.path(handle)
    assert store.list_ready() == [legacy]


@pytest.mark.parametrize('payload', [[], None, 1, 'metadata'])
def test_invalid_legacy_metadata_does_not_break_listing(staged, payload):
    store, _, _, source = staged
    legacy = DiskgraphSnapshotHandle('other', 'legacy')
    _completed_legacy(store, legacy)
    (source / 'index_metadata.json').write_text(json.dumps(payload))

    assert store.list_ready() == [legacy]


def test_existing_managed_reference_for_a_reserved_version_remains_readable(staged):
    store, handle, result, _ = staged
    published = store.publish(handle, result)
    old_version = DiskgraphSnapshotHandle(handle.graph_key,
                                          handle.graph_data_version + '.index.json')
    store._reference(handle).rename(store._reference(old_version))

    assert store.path(old_version) == published.index_path
    assert store.list_ready() == [old_version]
    assert store.resolve_version_prefix(handle.graph_key, old_version.graph_data_version) == published.index_path


@pytest.mark.parametrize('complete', [False, True])
def test_selecting_legacy_checks_the_restore_before_removing_its_reference(
        staged, complete):
    from diskgraph import DiskgraphCorruptionError

    store, handle, result, source = staged
    (source / '_MANIFEST.json').write_text(json.dumps({
        'schema_version': 1,
        'files': [{'rel_path': name, 'size_bytes': (source / name).stat().st_size}
                  for name in ('index_metadata.json', 'col_stats.json',
                               'ingest_stats.json')],
    }))
    published = Path(store.publish(handle, result).index_path)
    retained = {p.name: p.read_bytes() for p in published.iterdir() if p.is_file()}
    reference = store._reference(handle).read_bytes()
    restored = _completed_legacy(store, handle)
    if not complete:
        (restored / '_SUCCESS').unlink()
        with pytest.raises(DiskgraphCorruptionError):
            store.use_legacy(handle)
        assert store._reference(handle).read_bytes() == reference
        assert store.verify(handle).index_path == str(published)
    else:
        store.use_legacy(handle)
        assert not store._reference(handle).exists()
        assert store.verify(handle).index_path == str(restored)
    assert {p.name: p.read_bytes() for p in published.iterdir()
            if p.is_file()} == retained


def test_selecting_legacy_restore_preserves_suffix_named_directory(staged):
    store, original, _, _ = staged
    restored = DiskgraphSnapshotHandle('graph', 'restore')
    suffix_named = DiskgraphSnapshotHandle('graph', 'restore.index.json')
    restored_path = _completed_legacy(store, restored)
    other_path = _completed_legacy(store, suffix_named)
    marker = (other_path / '_SUCCESS').read_bytes()

    store.use_legacy(restored)

    assert store.verify(restored).index_path == str(restored_path)
    assert (other_path / '_SUCCESS').read_bytes() == marker
    assert store.path(suffix_named) == str(other_path)
    assert set(store.list_ready()) == {original, restored, suffix_named}


@pytest.mark.parametrize('collision', ['reserved_version', 'reference_directory'])
def test_import_publication_preserves_builds_on_reference_name_collision(
        staged, collision):
    store, handle, _, _ = staged
    imported = DiskgraphSnapshotHandle('graph', 'import')
    source = _completed_legacy(store, imported)
    metadata = source / 'index_metadata.json'
    metadata.write_text(json.dumps({'index_build_id': 'import-build'}))
    manifest = source / '_MANIFEST.json'
    content = json.loads(manifest.read_text())
    for entry in content['files']:
        if entry['rel_path'] == metadata.name:
            entry['size_bytes'] = metadata.stat().st_size
    manifest.write_text(json.dumps(content))
    published = store.publish_import(imported, DiskgraphIngestResult.for_index(source))
    retained = Path(published.index_path)
    before = {p.name: p.read_bytes() for p in retained.iterdir() if p.is_file()}
    shutil.copytree(retained, source)
    (source / '_SUCCESS').write_text(DiskgraphIngestResult.for_index(source).to_json())
    if collision == 'reserved_version':
        bad = DiskgraphSnapshotHandle('graph', 'import.index.json')
        expected = ValueError
    else:
        bad = handle
        store._reference(bad).mkdir()
        expected = FileExistsError
    with pytest.raises(expected):
        store.publish_import(bad, DiskgraphIngestResult.for_index(source))
    assert source.is_dir()
    assert {p.name: p.read_bytes() for p in retained.iterdir()
            if p.is_file()} == before


@pytest.mark.parametrize('payload', [[], None, 1, 'metadata'])
def test_nonobject_ingest_metadata_is_rejected_before_publication(
        staged, payload):
    store, handle, result, source = staged
    (source / 'index_metadata.json').write_text(json.dumps(payload))
    marker = (source / '_SUCCESS').read_bytes()

    with pytest.raises(ValueError, match='must contain a JSON object'):
        store.publish(handle, result)

    assert source.is_dir()
    assert (source / '_SUCCESS').read_bytes() == marker
    assert not (source.parents[2] / 'indexes').exists()
    assert not store._reference(handle).exists()


@pytest.mark.parametrize('managed', [False, True])
def test_version_prefix_resolves_completed_snapshot(staged, managed):
    store, handle, result, source = staged
    index = store.publish(handle, result).index_path if managed else str(source)

    assert store.resolve_version_prefix(handle.graph_key, 'ver') == index
    assert store.resolve_version_prefix(handle.graph_key, 'version') == index
    assert store.resolve_version_prefix(handle.graph_key, 'missing') is None
    assert store.resolve_version_prefix('missing', 'ver') is None


def test_version_prefix_rejects_ambiguity_within_one_graph(staged):
    store, handle, result, _ = staged
    store.publish(handle, result)
    other = DiskgraphSnapshotHandle(handle.graph_key, 'version-other')
    other_path = _completed_legacy(store, other)
    _completed_legacy(store, DiskgraphSnapshotHandle('another-graph', 'version'))

    assert store.resolve_version_prefix(handle.graph_key, 'ver') is None
    assert store.resolve_version_prefix(handle.graph_key, 'version-other') == str(other_path)
    assert store.resolve_version_prefix('another-graph', 'ver') is not None


def test_version_prefix_deduplicates_the_same_physical_path(staged):
    store, handle, result, _ = staged
    published = store.publish(handle, result)
    alias = DiskgraphSnapshotHandle(handle.graph_key, 'version-alias')
    store._reference(alias).write_bytes(store._reference(handle).read_bytes())

    assert store.resolve_version_prefix(handle.graph_key, 'ver') == published.index_path


def test_version_prefix_keeps_legacy_fallback_for_a_broken_reference(staged):
    store, handle, result, _ = staged
    store.publish(handle, result)
    legacy = _completed_legacy(store, handle)
    store._reference(handle).write_text('[]')

    assert store.resolve_version_prefix(handle.graph_key, 'ver') == str(legacy)


@pytest.mark.parametrize('damage', ['statistics_only', 'missing_marker',
                                    'invalid_reference', 'invalid_metadata'])
def test_version_prefix_excludes_incomplete_or_invalid_candidates(staged, damage):
    store, handle, result, source = staged
    if damage == 'statistics_only':
        for name in ('_SUCCESS', 'index_metadata.json', 'ingest_stats.json'):
            (source / name).unlink()
        (source / 'kumo_col_stats').rmdir()
    elif damage == 'missing_marker':
        (source / '_SUCCESS').unlink()
    else:
        index = Path(store.publish(handle, result).index_path)
        if damage == 'invalid_reference':
            store._reference(handle).write_text('[]')
        else:
            (index / 'index_metadata.json').write_text('[]')

    assert store.resolve_version_prefix(handle.graph_key, 'ver') is None


def test_version_prefix_treats_glob_metacharacters_literally(staged):
    store, handle, _, _ = staged
    literal = DiskgraphSnapshotHandle(handle.graph_key, 'literal[1]-version')
    path = _completed_legacy(store, literal)

    assert store.resolve_version_prefix(handle.graph_key, 'literal[1]') == str(path)
    assert store.resolve_version_prefix(handle.graph_key, 'literal*') is None


@pytest.mark.parametrize(('graph_key', 'prefix'), [
    ('', 'ver'), ('graph', ''), ('..', 'ver'), ('../graph', 'ver'),
    ('/graph', 'ver'), ('graph\\other', 'ver'), ('C:graph', 'ver'),
    ('graph', '../ver'), ('graph', 'ver/other'), ('graph', 'ver\\other'),
])
def test_version_prefix_refuses_path_components(staged, graph_key, prefix):
    store, _, _, _ = staged
    assert store.resolve_version_prefix(graph_key, prefix) is None


@pytest.mark.parametrize('metadata', [
    {'node_types': None},
    {'node_types': {}},
    {'node_types': 'events'},
    {'node_types': [None]},
    {'node_types': [[]]},
    {'node_types': [{'create_time_col': 1}]},
    {'node_types': [{'create_time_col': []}]},
    {'node_types': [{'name': None, 'create_time_col': 'ts'}]},
    {'node_types': [{'name': 'events', 'create_time_col': 'ts', 'shards': None}]},
    {'node_types': [{'name': 'events', 'create_time_col': 'ts', 'shards': {}}]},
    {'node_types': [{'name': 'events', 'create_time_col': 'ts', 'shards': [None]}]},
    {'node_types': [{'name': 'events', 'create_time_col': 'ts', 'shards': [{}]}]},
    {'node_types': [{'name': 'events', 'create_time_col': 'ts',
                     'shards': [{'min_id': None, 'max_id': 0}]}]},
    {'node_types': [{'name': 'events', 'create_time_col': 'ts',
                     'shards': [{'min_id': 0, 'max_id': '1'}]}]},
    {'node_types': [{'name': 'events', 'create_time_col': 'ts',
                     'shards': [{'min_id': False, 'max_id': 1}]}]},
    {'node_types': [{'name': 'events', 'create_time_col': 'ts',
                     'shards': [{'min_id': 0, 'max_id': 1}, None]}]},
])
def test_invalid_cached_time_metadata_does_not_hide_a_healthy_prefix_match(
        staged, metadata):
    store, handle, _, source = staged
    (source / 'index_metadata.json').write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match='cached'):
        validate_cached_source_times(str(source))
    assert store.resolve_version_prefix(handle.graph_key, 'ver') is None
    assert store.list_ready() == []

    healthy = DiskgraphSnapshotHandle(handle.graph_key, 'version-healthy')
    healthy_path = _completed_legacy(store, healthy)
    assert store.resolve_version_prefix(handle.graph_key, 'ver') == str(healthy_path)
    assert store.list_ready() == [healthy]


def test_custom_snapshot_root_is_preserved_with_managed_publication(tmp_path):
    store = LocalSnapshotStore(DiskgraphLocalConfig(
        str(tmp_path / 'old-snapshot-location'), managed_root=str(tmp_path / 'managed')))
    handle = DiskgraphSnapshotHandle('graph', 'version')
    source = _completed_legacy(store, handle)
    assert source == tmp_path / 'old-snapshot-location/graph/version'
    metadata = source / 'index_metadata.json'
    metadata.write_text(json.dumps({'index_build_id': 'build'}))
    # This case checks path policy; native manifest sizes must match the new ID.
    manifest = json.loads((source / '_MANIFEST.json').read_text())
    for entry in manifest['files']:
        entry['size_bytes'] = (source / entry['rel_path']).stat().st_size
    (source / '_MANIFEST.json').write_text(json.dumps(manifest))
    published = store.publish(handle, DiskgraphIngestResult.for_index(source))
    assert published.index_path == str(tmp_path / 'managed/indexes/build')
    reference = tmp_path / 'old-snapshot-location/graph/version.index.json'
    assert json.loads(reference.read_text()) == {'index_build_id': 'build'}
    assert store.path(handle) == published.index_path
    assert store.list_ready() == [handle]
    assert not (tmp_path / 'managed/snapshots').exists()
