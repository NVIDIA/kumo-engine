# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``pool.persistence.PoolStatePersistence``."""
import asyncio

import pytest

from io_utils.json import save_json
from pool.dataclasses import (
    DataplaneBackup,
    K8sPlacementState,
    PoolState,
    SnapshotPlacement,
    SnapshotState,
)
from pool.persistence import PoolStatePersistence


def _k8s_factory() -> PoolState:
    return PoolState(placement_state=K8sPlacementState())


def _make_state() -> PoolState:
    return PoolState(
        placement_state=K8sPlacementState(),
        snapshots={
            'snap-1':
            SnapshotPlacement(worker='worker-a', state=SnapshotState.READY,
                              size_bytes=123),
        },
        dataplane_backups={
            'snap-1':
            DataplaneBackup(snapshot_id='snap-1',
                            dataplane_path='s3://b/snap-1'),
        },
    )


def test_load_returns_empty_state_when_missing(tmp_path):
    # First-boot self-heal: missing file -> empty state, eagerly persisted.
    path = tmp_path / 'missing.json'
    persistence = PoolStatePersistence(str(path), _k8s_factory)
    state = asyncio.run(persistence.load())
    assert state.snapshots == {}
    assert state.placement_state.type == 'k8s'
    assert state.placement_state.pvcs == {}
    assert state.dataplane_backups == {}
    # Persisted compressed at <path>.gz (not the plain .json).
    assert (tmp_path / 'missing.json.gz').exists()
    assert not path.exists()
    reloaded = asyncio.run(persistence.load())
    assert reloaded == state


def test_save_load_round_trip(tmp_path):
    persistence = PoolStatePersistence(str(tmp_path / 'pool_state.json'),
                                       _k8s_factory)
    original = _make_state()
    asyncio.run(persistence.save(original))
    loaded = asyncio.run(persistence.load())
    assert loaded == original


def test_save_writes_gzip_suffix(tmp_path):
    path = tmp_path / 'pool_state.json'
    persistence = PoolStatePersistence(str(path), _k8s_factory)
    asyncio.run(persistence.save(_make_state()))
    # Compressed bytes live at .json.gz; the plain .json is not written.
    gz_head = (tmp_path / 'pool_state.json.gz').read_bytes()[:2]
    assert gz_head == b'\x1f\x8b'
    assert not path.exists()


def test_load_reads_legacy_uncompressed_and_migrates(tmp_path):
    # Backward compat: a pool_state.json written before compression was enabled
    # is plain JSON at the bare path. load() must read it and migrate in place —
    # write .json.gz and remove the stale legacy .json.
    path = tmp_path / 'pool_state.json'
    gz = tmp_path / 'pool_state.json.gz'
    original = _make_state()
    save_json(original, str(path))  # compressed=False (legacy plain JSON)
    legacy_head = path.read_bytes()[:2]
    assert legacy_head != b'\x1f\x8b'

    persistence = PoolStatePersistence(str(path), _k8s_factory)
    loaded = asyncio.run(persistence.load())
    assert loaded == original

    # load() migrated in place: .json.gz written, legacy .json removed.
    gz_head = gz.read_bytes()[:2]
    assert gz_head == b'\x1f\x8b'
    assert not path.exists()
    assert asyncio.run(persistence.load()) == original


def test_load_rejects_invalid_json(tmp_path):
    path = str(tmp_path / 'pool_state.json')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('{not valid json')
    persistence = PoolStatePersistence(path, _k8s_factory)
    with pytest.raises(ValueError, match='Failed to parse'):
        asyncio.run(persistence.load())


def test_concurrent_saves_do_not_clobber(tmp_path):
    # Without the global write lock, interleaved writes could produce
    # invalid JSON and ``load`` would raise before the final assertion.
    persistence = PoolStatePersistence(str(tmp_path / 'pool_state.json'),
                                       _k8s_factory)

    async def _drive() -> PoolState:
        state_a = _make_state()
        state_b = _make_state()
        state_b.dataplane_backups['snap-1'] = DataplaneBackup(
            snapshot_id='snap-1', dataplane_path='s3://b/snap-1-updated')
        await asyncio.gather(persistence.save(state_a),
                             persistence.save(state_b))
        return await persistence.load()

    loaded = asyncio.run(_drive())
    assert loaded.dataplane_backups['snap-1'].dataplane_path in (
        's3://b/snap-1', 's3://b/snap-1-updated')
