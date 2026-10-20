# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Snapshot export and load through real S3 and GCS; each store is skipped
unless its writable root is set."""

import contextlib
import os
import uuid

import pytest
from pyarrow.fs import FileSelector

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir
from kumo.engine._snapshot_io import _location

ROOTS = {
    's3': os.environ.get('KUMO_SNAPSHOT_LIVE_S3_ROOT'),
    'gs': os.environ.get('KUMO_SNAPSHOT_LIVE_GCS_ROOT'),
}


@pytest.mark.parametrize('scheme', ['s3', 'gs'])
def test_snapshot_round_trip_through_object_store(tmp_path, graph, scheme):
    root = ROOTS[scheme]
    if not root:
        pytest.skip(f'LIVE SNAPSHOT {scheme.upper()} NOT RUN: set '
                    f"KUMO_SNAPSHOT_LIVE_{'S3' if scheme == 's3' else 'GCS'}"
                    '_ROOT to a writable prefix')
    connector = kumo.S3Connector() if scheme == 's3' else kumo.GCSConnector()
    path = f"{root.rstrip('/')}/engine-snapshot-{uuid.uuid4().hex[:12]}"
    previous = get_scratch_dir()
    try:
        kumo.configure(scratch_dir=tmp_path / 'cpu')
        snapshot = graph.snapshot()
        location = snapshot.export(connector, path)
        kumo.configure(scratch_dir=tmp_path / 'gpu')
        loaded = kumo.GraphSnapshot.load_from_connector(
            connector, path=location, graph=graph)
    finally:
        kumo.configure(scratch_dir=previous)
        filesystem, prefix = _location(connector, path)
        filesystem.delete_dir_contents(prefix, missing_dir_ok=True)
        with contextlib.suppress(OSError):
            filesystem.delete_dir(prefix)
        assert not filesystem.get_file_info(
            FileSelector(prefix, allow_not_found=True))
    assert loaded.id == snapshot.id
    for file in snapshot.index_path.rglob('*'):
        relative = file.relative_to(snapshot.index_path)
        if (file.is_file()
                and file.name not in ('_SUCCESS', 'snapshot-manifest.json')
                and relative.parts[0] != 'train_tables'):
            assert (loaded.index_path / relative).read_bytes() == \
                file.read_bytes()
