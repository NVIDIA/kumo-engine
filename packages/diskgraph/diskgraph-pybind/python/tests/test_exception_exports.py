# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import diskgraph
import pytest


def test_public_corruption_error_catches_snapshot_verifier(tmp_path):
    with pytest.raises(diskgraph.DiskgraphCorruptionError):
        diskgraph.verify_snapshot_integrity(str(tmp_path))


def test_build_identity_is_exported():
    assert diskgraph.__version__
    assert diskgraph.BUILD_COMMIT


def test_snapshot_range_checksums_match_zlib(tmp_path):
    import json
    import zlib

    payload = bytes(range(256)) * 5
    (tmp_path / "a.bin").write_bytes(payload)
    (tmp_path / "_MANIFEST.json").write_text(json.dumps({
        "schema_version": 1,
        "files": [{"rel_path": "a.bin", "size_bytes": len(payload)}],
    }))
    crcs = diskgraph.snapshot_range_checksums(str(tmp_path), 512)
    assert crcs["a.bin"] == [zlib.crc32(payload[i:i + 512])
                             for i in range(0, len(payload), 512)]
    assert crcs["_MANIFEST.json"] == [
        zlib.crc32((tmp_path / "_MANIFEST.json").read_bytes())]
    with pytest.raises(diskgraph.DiskgraphCorruptionError):
        diskgraph.snapshot_range_checksums(str(tmp_path / "missing"), 512)
