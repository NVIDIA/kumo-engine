# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from kumo_core.diskgraph.snapshot import (
    DiskgraphIngestResult,
    DiskgraphLocalConfig,
    DiskgraphSnapshotError,
    DiskgraphSnapshotHandle,
    LocalSnapshotStore,
)


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
    assert set(store.list_ready()) == {handle, legacy}
    assert store.path(legacy) == str(path)
    store._reference(handle).write_text('{invalid')
    assert store.list_ready() == [legacy]
