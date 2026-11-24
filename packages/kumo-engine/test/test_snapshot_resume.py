# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path

import pytest

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir


@pytest.fixture
def scratch(tmp_path):
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    try:
        yield tmp_path / 'scratch'
    finally:
        kumo.configure(scratch_dir=previous)


def _interrupted(graph, monkeypatch):
    """Fail actual native publication after Index has persisted its checkpoint."""
    import kumo_core.diskgraph.ingest as ingest
    from kumo_core.diskgraph.runner import DiskgraphIngestError

    execute = ingest.execute_diskgraph_ingest
    paths = []

    def fail_publication(config, spec):
        paths.append(Path(spec.output_dir))
        # Native finalization rejects this payload before _SUCCESS is written.
        # Unlike killing a fast fixture, this preserves real checkpoints
        # deterministically at the Index/publication boundary.
        failed = replace(spec, execution=replace(
            spec.execution, success_marker_content='{'))
        return execute(config, failed)

    with monkeypatch.context() as patch:
        patch.setattr(ingest, 'execute_diskgraph_ingest', fail_publication)
        with pytest.raises(DiskgraphIngestError):
            graph.snapshot()
    assert len(paths) == 1
    assert not (paths[0] / '_SUCCESS').exists()
    assert (paths[0] / '_tmp/_phase_complete_index').is_file()
    return paths[0]


def _files(path):
    return {str(p.relative_to(path)): p.read_bytes()
            for p in path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('reference', ['absent', 'malformed'])
@pytest.mark.parametrize('metadata', ['valid', 'missing', 'truncated'])
def test_snapshot_resumes_native_checkpoint_without_reencoding(
        scratch, graph, monkeypatch, metadata, reference):
    import kumo_core.diskgraph.runner as runner
    from diskgraph import has_resumable_ingest

    staging = _interrupted(graph, monkeypatch)
    state = staging / '_tmp/encode_state.json'
    encoded = state.read_bytes()
    build = json.loads(encoded)['index_build_id']
    inode = state.stat().st_ino
    sentinel = staging / 'retained-checkpoint'
    sentinel.write_bytes(b'not a fresh ingest')
    sentinel_inode = sentinel.stat().st_ino
    if reference == 'malformed':
        staging.with_name(staging.name + '.index.json').write_text('{broken')
    metadata_path = staging / 'index_metadata.json'
    if metadata == 'missing':
        metadata_path.unlink()
    elif metadata == 'truncated':
        metadata_path.write_text('{"dataset_name":')
    assert has_resumable_ingest(str(staging))
    commands = []
    build_command = runner._build_ingest_cmd

    def record(*args, **kwargs):
        assert state.read_bytes() == encoded
        assert state.stat().st_ino == inode
        command = build_command(*args, **kwargs)
        commands.append(command)
        return command

    monkeypatch.setattr(runner, '_build_ingest_cmd', record)
    snapshot = graph.snapshot()
    assert snapshot.index_path == scratch / 'indexes' / build
    assert not staging.exists()
    assert json.loads((snapshot.index_path / 'index_metadata.json').read_text())[
        'index_build_id'] == build
    retained = snapshot.index_path / sentinel.name
    assert retained.read_bytes() == b'not a fresh ingest'
    assert retained.stat().st_ino == sentinel_inode
    assert not (snapshot.index_path / '_tmp').exists()
    assert len(commands) == 1
    assert commands[0][commands[0].index('--resume-from-phase') + 1] == 'index'
    monkeypatch.setattr(runner, '_build_ingest_cmd',
                        lambda *a, **k: pytest.fail('completed snapshot re-ingested'))
    assert graph.snapshot() == snapshot


@pytest.mark.parametrize('damage', [
    'no_checkpoint', 'checkpoint_path_file', 'partial_encode', 'invalid_chain',
    'published_bad_marker', 'published_missing_metadata',
])
def test_nonresumable_staging_rebuilds_with_a_new_identity(
        scratch, graph, monkeypatch, damage):
    import shutil

    from diskgraph import has_resumable_ingest
    from kumo_core.diskgraph.snapshot import DiskgraphIngestResult

    staging = _interrupted(graph, monkeypatch)
    old_id = json.loads((staging / '_tmp/encode_state.json').read_text())[
        'index_build_id']
    sentinel = staging / 'stale-sentinel'
    sentinel.write_bytes(b'must be discarded')
    if damage in ('no_checkpoint', 'checkpoint_path_file'):
        shutil.rmtree(staging / '_tmp')
        if damage == 'checkpoint_path_file':
            (staging / '_tmp').write_bytes(b'not a checkpoint directory')
    elif damage == 'partial_encode':
        for marker in (staging / '_tmp').glob('_phase_complete_*'):
            marker.unlink()
    elif damage == 'invalid_chain':
        (staging / '_tmp/_phase_complete_pk_map').unlink()
    else:
        (staging / '_SUCCESS').write_text(
            '{bad' if damage == 'published_bad_marker' else
            DiskgraphIngestResult.for_index(staging).to_json())
        if damage == 'published_missing_metadata':
            (staging / 'index_metadata.json').unlink()
    if damage != 'invalid_chain':
        assert not has_resumable_ingest(str(staging))
    snapshot = graph.snapshot()
    assert snapshot.index_path.name != old_id
    assert not staging.exists()
    assert not (snapshot.index_path / sentinel.name).exists()
    assert graph.snapshot() == snapshot


@pytest.mark.parametrize('damage, message', [
    ('metadata_identity', 'metadata/state mismatch'),
    ('encode_state', 'key must be a string'),
    ('source_identity', 'resume source identity'),
])
def test_unsafe_checkpoint_is_rejected_without_deleting_it(
        scratch, graph, monkeypatch, damage, message):
    import kumo_core.diskgraph.runner as runner
    from kumo_core.diskgraph.runner import DiskgraphIngestError

    staging = _interrupted(graph, monkeypatch)
    if damage == 'metadata_identity':
        path = staging / 'index_metadata.json'
        payload = json.loads(path.read_text())
        payload['index_build_id'] = 'another-build'
        path.write_text(json.dumps(payload))
    elif damage == 'encode_state':
        (staging / '_tmp/encode_state.json').write_text('{broken')
    else:
        path = staging / '_tmp/_phase_complete_encode'
        payload = json.loads(path.read_text())
        payload['dataset_name'] = 'another-source'
        path.write_text(json.dumps(payload))
    paths = [staging / '_tmp/encode_state.json',
             staging / '_tmp/_phase_complete_encode',
             staging / 'index_metadata.json']
    before = [(p.read_bytes(), p.stat().st_ino) for p in paths]
    outputs = []
    run = runner._run_with_heartbeat

    def record(*args, **kwargs):
        result = run(*args, **kwargs)
        outputs.append(result[1])
        return result

    monkeypatch.setattr(runner, '_run_with_heartbeat', record)
    with pytest.raises(DiskgraphIngestError):
        graph.snapshot()
    assert message in '\n'.join(outputs)
    assert [(p.read_bytes(), p.stat().st_ino) for p in paths] == before
    assert not (staging / '_SUCCESS').exists()


def test_healthy_managed_snapshot_wins_without_touching_a_checkpoint(
        scratch, graph, monkeypatch):
    import kumo_core.diskgraph.ingest as ingest
    from kumo_core.diskgraph.snapshot import DiskgraphSnapshotHandle

    from kumo.engine.graph import _store

    published = graph.snapshot()
    store = _store()
    handle = DiskgraphSnapshotHandle(published._graph_key, published._data_version)
    reference = store._reference(handle)
    reference.unlink()
    staging = _interrupted(graph, monkeypatch)
    reference.write_text(json.dumps({'index_build_id': published.index_path.name}))
    before = _files(staging)
    inodes = {name: (staging / name).stat().st_ino for name in before}
    monkeypatch.setattr(ingest, 'execute_diskgraph_ingest',
                        lambda *a, **k: pytest.fail('healthy managed build must be reused'))
    assert graph.snapshot() == published
    assert _files(staging) == before
    assert {name: (staging / name).stat().st_ino for name in before} == inodes


@pytest.mark.parametrize('error', [
    PermissionError('permission denied'), OSError('resource unavailable'),
])
def test_resume_eligibility_io_errors_do_not_authorize_cleanup(
        scratch, graph, monkeypatch, error):
    import diskgraph
    import kumo_core.diskgraph.ingest as ingest

    staging = _interrupted(graph, monkeypatch)
    before = _files(staging)

    def fail(*args):
        raise error

    monkeypatch.setattr(diskgraph, 'has_resumable_ingest', fail)
    monkeypatch.setattr(ingest, 'execute_diskgraph_ingest',
                        lambda *a, **k: pytest.fail('I/O error must not trigger ingest'))
    with pytest.raises(type(error), match=str(error)):
        graph.snapshot()
    assert _files(staging) == before


def _export_for_checkpoint(graph, tmp_path, *, idless=False):
    import shutil

    snapshot = graph.snapshot()
    exported = snapshot
    if idless:
        index = tmp_path / 'idless-export'
        shutil.copytree(snapshot.index_path, index)
        metadata = index / 'index_metadata.json'
        payload = json.loads(metadata.read_text())
        payload.pop('index_build_id')
        metadata.write_text(json.dumps(payload))
        manifest = index / '_MANIFEST.json'
        payload = json.loads(manifest.read_text())
        for entry in payload['files']:
            entry['size_bytes'] = (index / entry['rel_path']).stat().st_size
        manifest.write_text(json.dumps(payload))
        exported = replace(snapshot, index_path=index)
    exports = kumo.FileConnector(root_dir=str(tmp_path / 'exports'))
    location = exported.export(kumo.SnapshotOutputConfig(exports, 'one'))
    from kumo_core.diskgraph.snapshot import DiskgraphSnapshotHandle

    from kumo.engine.graph import _store

    store = _store()
    handle = DiskgraphSnapshotHandle(snapshot._graph_key, snapshot._data_version)
    store._reference(handle).unlink()
    shutil.rmtree(snapshot.index_path)
    return snapshot, store, handle, exports, location


@pytest.mark.parametrize('reference', ['absent', 'malformed'])
@pytest.mark.parametrize('outcome', ['success', 'failed_download', 'idless'])
def test_import_preserves_native_ingest_checkpoints(
        scratch, graph, tmp_path, monkeypatch, reference, outcome):
    import kumo.engine._snapshot_io as snapshot_io

    snapshot, store, handle, exports, location = _export_for_checkpoint(
        graph, tmp_path, idless=outcome == 'idless')
    staging = _interrupted(graph, monkeypatch)
    (staging / 'index_metadata.json').write_text('{interrupted')
    ref = store._reference(handle)
    if reference == 'malformed':
        ref.write_text('{broken')
    before = _files(staging)
    inodes = {name: (staging / name).stat().st_ino for name in before}
    original_ref = ref.read_bytes() if ref.exists() else None
    load = snapshot_io.load_into
    targets = []

    def record(connector, path, root, manifest, target):
        assert target != staging
        targets.append(target)
        if outcome == 'failed_download':
            target.mkdir(parents=True)
            (target / 'partial').write_bytes(b'unfinished download')
            raise OSError('download failed')
        return load(connector, path, root, manifest, target)

    monkeypatch.setattr(snapshot_io, 'load_into', record)
    if outcome == 'success':
        loaded = kumo.GraphSnapshot.load_from_connector(exports, path=location)
        assert loaded.index_path == snapshot.index_path
        assert graph.snapshot() == loaded
    else:
        error, message = ((OSError, 'download failed') if outcome == 'failed_download'
                          else (ValueError, 'ID-less export cannot replace'))
        with pytest.raises(error, match=message):
            kumo.GraphSnapshot.load_from_connector(exports, path=location)
        assert (ref.read_bytes() if ref.exists() else None) == original_ref
    assert _files(staging) == before
    assert {name: (staging / name).stat().st_ino for name in before} == inodes
    assert all(not path.exists() for path in targets)
    assert not list((scratch / 'snapshots/.imports').rglob('_SUCCESS'))


@pytest.mark.parametrize('outcome', ['success', 'failed_copy', 'idless'])
def test_generic_restore_preserves_native_ingest_checkpoints(
        scratch, graph, tmp_path, monkeypatch, outcome):
    import shutil

    from kumo_core.diskgraph.snapshot import DiskgraphSnapshotHandle

    from kumo.engine.graph import _lock, _store

    snapshot = graph.snapshot()
    store = _store()
    handle = DiskgraphSnapshotHandle(snapshot._graph_key, snapshot._data_version)
    durable_root = tmp_path / 'durable'
    durable = Path(store.durable_path(handle, str(durable_root)))
    shutil.copytree(snapshot.index_path, durable)
    if outcome == 'idless':
        metadata = durable / 'index_metadata.json'
        payload = json.loads(metadata.read_text())
        payload.pop('index_build_id')
        metadata.write_text(json.dumps(payload))
        manifest = durable / '_MANIFEST.json'
        payload = json.loads(manifest.read_text())
        for entry in payload['files']:
            entry['size_bytes'] = (durable / entry['rel_path']).stat().st_size
        manifest.write_text(json.dumps(payload))
    store._reference(handle).unlink()
    shutil.rmtree(snapshot.index_path)
    staging = _interrupted(graph, monkeypatch)
    before = _files(staging)
    inodes = {name: (staging / name).stat().st_ino for name in before}
    targets = []

    def copy(source, target):
        assert Path(target) != staging
        targets.append(Path(target))
        shutil.copytree(source, target)
        if outcome == 'failed_copy':
            raise OSError('copy failed')

    def call():
        with _lock(handle.graph_key):
            return store.restore(handle, str(durable_root),
                                 exists=lambda p: Path(p).exists(), copy_tree=copy)
    if outcome == 'success':
        assert call().index_path == str(snapshot.index_path)
    else:
        error, message = ((OSError, 'copy failed') if outcome == 'failed_copy' else
                          (ValueError, 'cannot replace a completed legacy snapshot'))
        with pytest.raises(error, match=message):
            call()
    assert _files(staging) == before
    assert {name: (staging / name).stat().st_ino for name in before} == inodes
    assert all(not path.exists() for path in targets)


@pytest.mark.parametrize('operation', ['eligibility', 'snapshot', 'import', 'restore'])
@pytest.mark.parametrize('mode', [0, 0o400], ids=['unreadable', 'unsearchable'])
def test_unreadable_native_checkpoint_never_authorizes_partial_cleanup(
        scratch, graph, tmp_path, monkeypatch, operation, mode):
    import stat

    import diskgraph
    import kumo_core.diskgraph.ingest as ingest

    import kumo.engine._snapshot_io as snapshot_io
    from kumo.engine.graph import _lock

    _, store, handle, exports, location = _export_for_checkpoint(graph, tmp_path)
    staging = _interrupted(graph, monkeypatch)
    (staging / 'retained-checkpoint').write_bytes(b'must survive a failed access probe')
    before = _files(staging)
    inodes = {name: (staging / name).stat().st_ino for name in before}
    temporary = staging / '_tmp'
    permissions = stat.S_IMODE(temporary.stat().st_mode)

    def forbidden(*args, **kwargs):
        pytest.fail('denied checkpoint access must fail before ingest or copying')

    monkeypatch.setattr(ingest, 'execute_diskgraph_ingest', forbidden)
    monkeypatch.setattr(snapshot_io, 'load_into', forbidden)
    temporary.chmod(mode)
    try:
        try:
            (temporary / '_phase_complete_encode').read_bytes()
        except PermissionError:
            pass
        else:
            pytest.skip('this user can bypass filesystem permission bits')
        # This is the actual native behavior that previously authorized
        # rmtree: exists() silently treats the inaccessible markers as absent.
        assert not diskgraph.has_resumable_ingest(str(staging))
        with pytest.raises(PermissionError):
            if operation == 'eligibility':
                store.has_resumable_staging(handle)
            elif operation == 'snapshot':
                graph.snapshot()
            elif operation == 'import':
                kumo.GraphSnapshot.load_from_connector(exports, path=location)
            else:
                with _lock(handle.graph_key):
                    store.restore(handle, str(tmp_path / 'durable'),
                                  exists=lambda path: True, copy_tree=forbidden)
    finally:
        temporary.chmod(permissions)
    assert _files(staging) == before
    assert {name: (staging / name).stat().st_ino for name in before} == inodes
