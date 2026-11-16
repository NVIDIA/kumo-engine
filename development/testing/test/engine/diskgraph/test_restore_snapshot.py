# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the dataplane restore path in
``DiskgraphService.restore_snapshot`` (issue #30984).

A plain local directory stands in for the dataplane mirror throughout --
no cloud access needed. ``verify_snapshot_integrity`` / ``read_success_marker``
are the real native calls; fixtures build a ``_SUCCESS`` payload that points
at the deterministic local ``output_dir`` so the post-restore check passes
for real, exactly as it does in production (index paths are stable per
handle across workers). Fixtures also write a real ``_MANIFEST.json`` --
that's the producer's contract with ``verify_snapshot_integrity``, which
re-reads and re-validates it from the local copy; it has nothing to do with
how ``restore_snapshot`` itself plans the transfer.
"""
import json
import os
import threading
import time

import pytest

from kumo.diskgraph._backend import DiskgraphHandle
from kumo.diskgraph.service import _RESTORE_SUCCESS_FILENAME, DiskgraphService
from service_framework._raw_deploy_config import DiskgraphConfig

_MANIFEST_FILENAME = '_MANIFEST.json'


def _build_service(tmp_path):
    return DiskgraphService(
        DiskgraphConfig(data_root=str(tmp_path / 'local')))


def _default_files():
    return {
        'col_stats.json': b'{}',
        'ingest_stats.json': b'{}',
        'kumo_col_stats/marker.json': b'{"k": "v"}',
        'data/part-00000.bin': os.urandom(37),
        'data/part-00001.bin': os.urandom(53),
        'data/nested/part-00002.bin': os.urandom(11),
    }


def _write_dataplane_mirror(dataplane_dir, output_dir, files):
    os.makedirs(dataplane_dir, exist_ok=True)
    for rel_path, content in files.items():
        path = os.path.join(dataplane_dir, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as fh:
            fh.write(content)

    manifest_payload = {
        'schema_version': 1,
        'files': [{
            'rel_path': rel_path,
            'size_bytes': len(content)
        } for rel_path, content in sorted(files.items())],
    }
    manifest_path = os.path.join(dataplane_dir, _MANIFEST_FILENAME)
    with open(manifest_path, 'w') as fh:
        json.dump(manifest_payload, fh)

    success_payload = {
        'index_path': output_dir,
        'col_stats_path': os.path.join(output_dir, 'col_stats.json'),
        'ingest_stats_path': os.path.join(output_dir, 'ingest_stats.json'),
        'kumo_stats_dir': os.path.join(output_dir, 'kumo_col_stats'),
    }
    success_path = os.path.join(dataplane_dir, _RESTORE_SUCCESS_FILENAME)
    with open(success_path, 'w') as fh:
        json.dump(success_payload, fh)


def test_restore_snapshot_happy_path(tmp_path):
    service = _build_service(tmp_path)
    handle = DiskgraphHandle.for_test()
    dataplane_root = str(tmp_path / 'dataplane-root')
    output_dir = service.local_snapshot_path(handle)
    dataplane_dir = service.dataplane_snapshot_path(handle, dataplane_root)
    files = _default_files()
    _write_dataplane_mirror(dataplane_dir, output_dir, files)

    result = service.restore_snapshot(handle, dataplane_root)

    assert result is not None
    assert result.index_path == output_dir
    for rel_path, content in files.items():
        with open(os.path.join(output_dir, rel_path), 'rb') as fh:
            assert fh.read() == content
    assert os.path.exists(os.path.join(output_dir, _MANIFEST_FILENAME))
    assert os.path.exists(os.path.join(output_dir, _RESTORE_SUCCESS_FILENAME))


def test_restore_snapshot_success_written_last(tmp_path, monkeypatch):
    """``_SUCCESS`` is copied on its own, after the rest of the mirror.
    If that final copy fails, the rest of the tree must already be in
    place locally, but ``_SUCCESS`` itself must not be -- that is the
    invariant Rust's ingest-side ordering (everything, then ``_SUCCESS``)
    exists to preserve.
    """
    from io_utils import fsutil

    service = _build_service(tmp_path)
    handle = DiskgraphHandle.for_test()
    dataplane_root = str(tmp_path / 'dataplane-root')
    output_dir = service.local_snapshot_path(handle)
    dataplane_dir = service.dataplane_snapshot_path(handle, dataplane_root)
    files = _default_files()
    _write_dataplane_mirror(dataplane_dir, output_dir, files)

    real_copy_file = fsutil.copy_file

    def _flaky_copy_file(src, dst, *args, **kwargs):
        if os.path.basename(src) == _RESTORE_SUCCESS_FILENAME:
            raise RuntimeError('simulated crash before success marker lands')
        return real_copy_file(src, dst, *args, **kwargs)

    monkeypatch.setattr(fsutil, 'copy_file', _flaky_copy_file)

    with pytest.raises(RuntimeError, match='simulated crash'):
        service.restore_snapshot(handle, dataplane_root)

    for rel_path in files:
        assert os.path.exists(os.path.join(output_dir, rel_path))
    assert os.path.exists(os.path.join(output_dir, _MANIFEST_FILENAME))
    assert not os.path.exists(
        os.path.join(output_dir, _RESTORE_SUCCESS_FILENAME))


def test_restore_snapshot_removes_stale_success_before_copy(
        tmp_path, monkeypatch):
    """A ``_SUCCESS`` left behind by an earlier attempt must be gone before
    the bulk copy starts -- otherwise it marks a half-restored tree ready
    for the whole duration of the copy.
    """
    from io_utils import fsutil

    service = _build_service(tmp_path)
    handle = DiskgraphHandle.for_test()
    dataplane_root = str(tmp_path / 'dataplane-root')
    output_dir = service.local_snapshot_path(handle)
    dataplane_dir = service.dataplane_snapshot_path(handle, dataplane_root)
    files = _default_files()
    _write_dataplane_mirror(dataplane_dir, output_dir, files)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, _RESTORE_SUCCESS_FILENAME), 'w') as fh:
        fh.write('stale')

    real_copy = fsutil.copy

    def _checking_copy(*args, **kwargs):
        assert not os.path.exists(
            os.path.join(output_dir, _RESTORE_SUCCESS_FILENAME))
        return real_copy(*args, **kwargs)

    monkeypatch.setattr(fsutil, 'copy', _checking_copy)

    result = service.restore_snapshot(handle, dataplane_root)

    assert result is not None
    assert os.path.exists(os.path.join(output_dir, _RESTORE_SUCCESS_FILENAME))


def test_restore_snapshot_heartbeats_before_first_batch(tmp_path):
    service = _build_service(tmp_path)
    handle = DiskgraphHandle.for_test()
    dataplane_root = str(tmp_path / 'dataplane-root')
    output_dir = service.local_snapshot_path(handle)
    dataplane_dir = service.dataplane_snapshot_path(handle, dataplane_root)
    files = _default_files()
    _write_dataplane_mirror(dataplane_dir, output_dir, files)

    calls = []

    def _heartbeat():
        calls.append(len(calls))
        if len(calls) == 1:
            # Fires before any byte of the transfer has moved.
            assert not any(
                os.path.exists(os.path.join(output_dir, rel))
                for rel in files)

    result = service.restore_snapshot(handle, dataplane_root,
                                      heartbeat=_heartbeat)

    assert result is not None
    # A tiny/fast transfer completes well inside one 30s ticker interval,
    # so only the guaranteed pre-transfer heartbeat is expected here; the
    # ticker firing repeatedly during a slow transfer is covered by
    # test_restore_snapshot_heartbeat_ticker_fires_during_slow_transfer.
    assert len(calls) >= 1


def test_restore_snapshot_heartbeat_ticker_fires_during_slow_transfer(
        tmp_path, monkeypatch):
    """A single ``fsutil.copy`` call has no intra-file progress signal, so
    the only way to heartbeat during one slow/retrying object is a ticker
    running concurrently with the transfer.
    """
    from io_utils import fsutil
    from kumo.diskgraph import service as service_module

    monkeypatch.setattr(service_module, '_RESTORE_HEARTBEAT_INTERVAL_SECS',
                        0.02)

    service = _build_service(tmp_path)
    handle = DiskgraphHandle.for_test()
    dataplane_root = str(tmp_path / 'dataplane-root')
    output_dir = service.local_snapshot_path(handle)
    dataplane_dir = service.dataplane_snapshot_path(handle, dataplane_root)
    files = _default_files()
    _write_dataplane_mirror(dataplane_dir, output_dir, files)

    real_copy = fsutil.copy

    def _slow_copy(*args, **kwargs):
        # Several ticks' worth of sleep at the shrunken interval above,
        # standing in for one very slow (or deeply retrying) object.
        time.sleep(0.3)
        return real_copy(*args, **kwargs)

    monkeypatch.setattr(fsutil, 'copy', _slow_copy)

    heartbeats = []
    result = service.restore_snapshot(handle, dataplane_root,
                                      heartbeat=lambda: heartbeats.append(1))

    assert result is not None
    # 1 pre-transfer heartbeat plus several ticks during the 0.3s sleep.
    assert len(heartbeats) > 2


def test_restore_snapshot_heartbeat_ticker_stops_after_transfer(
        tmp_path, monkeypatch):
    """The ticker must be joined in a ``finally`` -- it can't outlive the
    transfer or fire after ``restore_snapshot`` has already returned.
    """
    from kumo.diskgraph import service as service_module

    monkeypatch.setattr(service_module, '_RESTORE_HEARTBEAT_INTERVAL_SECS',
                        0.02)

    service = _build_service(tmp_path)
    handle = DiskgraphHandle.for_test()
    dataplane_root = str(tmp_path / 'dataplane-root')
    output_dir = service.local_snapshot_path(handle)
    dataplane_dir = service.dataplane_snapshot_path(handle, dataplane_root)
    files = _default_files()
    _write_dataplane_mirror(dataplane_dir, output_dir, files)

    heartbeats = []
    result = service.restore_snapshot(handle, dataplane_root,
                                      heartbeat=lambda: heartbeats.append(1))

    assert result is not None
    assert not any(t.name == 'diskgraph-restore-heartbeat'
                   for t in threading.enumerate())

    count_at_return = len(heartbeats)
    time.sleep(0.2)
    assert len(heartbeats) == count_at_return


def test_restore_snapshot_no_mirror_returns_none(tmp_path):
    service = _build_service(tmp_path)
    handle = DiskgraphHandle.for_test()
    dataplane_root = str(tmp_path / 'dataplane-root')

    assert service.restore_snapshot(handle, dataplane_root) is None
