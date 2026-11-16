# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for diskgraph disk space management."""
import json
import os
import time

import pytest
from kumo_core.diskgraph.snapshot import (
    DiskgraphIngestResult,
    check_diskgraph_output,
    write_diskgraph_success_marker,
)

# ---------------------------------------------------------------------------
# check_diskgraph_output: LRU mtime touch
# ---------------------------------------------------------------------------


def test_check_diskgraph_output_touches_success_mtime(tmp_path):
    """Cache hit should update _SUCCESS mtime for LRU accuracy."""
    index_path = str(tmp_path)
    col_stats = tmp_path / 'col_stats.json'
    ingest_stats = tmp_path / 'ingest_stats.json'
    kumo_stats = tmp_path / 'kumo_col_stats'
    col_stats.write_text('{}')
    ingest_stats.write_text('{}')
    kumo_stats.mkdir()

    result = DiskgraphIngestResult(
        index_path=index_path,
        col_stats_path=str(col_stats),
        ingest_stats_path=str(ingest_stats),
        kumo_stats_dir=str(kumo_stats),
    )
    write_diskgraph_success_marker(str(tmp_path), result)

    success_path = tmp_path / '_SUCCESS'

    # Backdate the _SUCCESS mtime
    old_time = time.time() - 3600
    os.utime(success_path, (old_time, old_time))
    mtime_before = success_path.stat().st_mtime

    # Cache hit should touch the mtime
    cached = check_diskgraph_output(str(tmp_path))
    assert cached is not None

    mtime_after = success_path.stat().st_mtime
    assert mtime_after > mtime_before, (
        f"Expected _SUCCESS mtime to be updated on cache hit: "
        f"before={mtime_before}, after={mtime_after}")


def test_check_diskgraph_output_no_success_no_touch(tmp_path):
    """No _SUCCESS file -> returns None, no crash from utime."""
    assert check_diskgraph_output(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# DiskManager pybind
# ---------------------------------------------------------------------------


@pytest.fixture
def data_root(tmp_path):
    """Create a data root with some fake indexes."""
    root = tmp_path / 'diskgraph-data'
    root.mkdir()

    # Create graph-A/v1 (valid, old)
    a_v1 = root / 'graph-A' / 'v1'
    a_v1.mkdir(parents=True)
    (a_v1 / 'index_metadata.json').write_text('{}')
    (a_v1 / 'data.bin').write_bytes(b'\x00' * 1000)
    success = a_v1 / '_SUCCESS'
    success.write_text('{}')
    old_time = time.time() - 7200
    os.utime(success, (old_time, old_time))

    # Create graph-A/v2 (valid, newer)
    a_v2 = root / 'graph-A' / 'v2'
    a_v2.mkdir(parents=True)
    (a_v2 / 'index_metadata.json').write_text('{}')
    (a_v2 / 'data.bin').write_bytes(b'\x00' * 500)
    success2 = a_v2 / '_SUCCESS'
    success2.write_text('{}')

    # Create graph-B/v1 (orphan, no _SUCCESS)
    b_v1 = root / 'graph-B' / 'v1'
    b_v1.mkdir(parents=True)
    (b_v1 / 'partial.bin').write_bytes(b'\x00' * 200)

    return root


def test_disk_manager_scan_indexes(data_root):
    """DiskManager.scan_indexes() finds valid and orphan indexes."""
    import diskgraph
    dm = diskgraph.DiskManager(str(data_root))
    result = json.loads(dm.scan_indexes())

    assert len(result) == 3
    keys = {(r['graph_key'], r['graph_version']) for r in result}
    assert ('graph-A', 'v1') in keys
    assert ('graph-A', 'v2') in keys
    assert ('graph-B', 'v1') in keys

    orphans = [r for r in result if not r['has_success_marker']]
    assert len(orphans) == 1
    assert orphans[0]['graph_key'] == 'graph-B'


def test_disk_manager_ensure_space_enough(data_root):
    """When enough space is available, no eviction happens."""
    import diskgraph
    dm = diskgraph.DiskManager(str(data_root))
    result = json.loads(dm.ensure_space(100, 'graph-C', 'v1'))
    assert result['evicted'] == []
    assert result['freed_bytes'] == 0


def test_disk_manager_disk_usage(data_root):
    """disk_usage() returns non-zero values."""
    import diskgraph
    dm = diskgraph.DiskManager(str(data_root))
    result = json.loads(dm.disk_usage())
    assert result['total_bytes'] > 0
    assert result['available_bytes'] > 0


def test_estimate_ingest_size_bad_json():
    """estimate_ingest_size with invalid JSON raises config error."""
    import diskgraph
    with pytest.raises(diskgraph.DiskgraphConfigError):
        diskgraph.estimate_ingest_size('not valid json')


# ---------------------------------------------------------------------------
# check_diskgraph_output: failure paths
# ---------------------------------------------------------------------------


def test_check_diskgraph_output_corrupted_success(tmp_path):
    """Corrupted _SUCCESS JSON -> returns None (re-ingest)."""
    success = tmp_path / '_SUCCESS'
    success.write_text('not valid json {{{')
    assert check_diskgraph_output(str(tmp_path)) is None


def test_check_diskgraph_output_missing_referenced_file(tmp_path):
    """Valid _SUCCESS but referenced col_stats file missing -> None."""
    index_path = str(tmp_path)
    ingest_stats = tmp_path / 'ingest_stats.json'
    kumo_stats = tmp_path / 'kumo_col_stats'
    ingest_stats.write_text('{}')
    kumo_stats.mkdir()

    result = DiskgraphIngestResult(
        index_path=index_path,
        col_stats_path=str(tmp_path / 'col_stats.json'),  # does NOT exist
        ingest_stats_path=str(ingest_stats),
        kumo_stats_dir=str(kumo_stats),
    )
    write_diskgraph_success_marker(str(tmp_path), result)
    assert check_diskgraph_output(str(tmp_path)) is None


def test_check_diskgraph_output_missing_referenced_dir(tmp_path):
    """Valid _SUCCESS but referenced kumo_stats dir missing -> None."""
    index_path = str(tmp_path)
    col_stats = tmp_path / 'col_stats.json'
    ingest_stats = tmp_path / 'ingest_stats.json'
    col_stats.write_text('{}')
    ingest_stats.write_text('{}')
    # kumo_col_stats dir NOT created

    result = DiskgraphIngestResult(
        index_path=index_path,
        col_stats_path=str(col_stats),
        ingest_stats_path=str(ingest_stats),
        kumo_stats_dir=str(tmp_path / 'kumo_col_stats'),
    )
    write_diskgraph_success_marker(str(tmp_path), result)
    assert check_diskgraph_output(str(tmp_path)) is None


def test_check_diskgraph_output_touch_failure_logs_warning(
    tmp_path,
    monkeypatch,
):
    """touch failure should log warning, not raise."""
    index_path = str(tmp_path)
    col_stats = tmp_path / 'col_stats.json'
    ingest_stats = tmp_path / 'ingest_stats.json'
    kumo_stats = tmp_path / 'kumo_col_stats'
    col_stats.write_text('{}')
    ingest_stats.write_text('{}')
    kumo_stats.mkdir()

    result = DiskgraphIngestResult(
        index_path=index_path,
        col_stats_path=str(col_stats),
        ingest_stats_path=str(ingest_stats),
        kumo_stats_dir=str(kumo_stats),
    )
    write_diskgraph_success_marker(str(tmp_path), result)

    import diskgraph

    monkeypatch.setattr(
        diskgraph,
        'touch_success_marker',
        lambda *_args, **_kwargs:
        (_ for _ in ()).throw(PermissionError('mocked permission denied')),
    )

    # Should NOT raise — returns the cached result with a warning
    cached = check_diskgraph_output(str(tmp_path))
    assert cached is not None
    assert cached.index_path == index_path


def test_write_read_success_marker_roundtrip(tmp_path):
    """write + check round-trip preserves all fields."""
    col_stats = tmp_path / 'col_stats.json'
    ingest_stats = tmp_path / 'ingest_stats.json'
    kumo_stats = tmp_path / 'kumo_col_stats'
    col_stats.write_text('{}')
    ingest_stats.write_text('{}')
    kumo_stats.mkdir()

    original = DiskgraphIngestResult(
        index_path=str(tmp_path),
        col_stats_path=str(col_stats),
        ingest_stats_path=str(ingest_stats),
        kumo_stats_dir=str(kumo_stats),
    )
    write_diskgraph_success_marker(str(tmp_path), original)
    cached = check_diskgraph_output(str(tmp_path))

    assert cached is not None
    assert cached.index_path == original.index_path
    assert cached.col_stats_path == original.col_stats_path
    assert cached.ingest_stats_path == original.ingest_stats_path
    assert cached.kumo_stats_dir == original.kumo_stats_dir
