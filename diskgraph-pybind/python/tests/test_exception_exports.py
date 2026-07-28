from __future__ import annotations

import diskgraph
import pytest


def test_public_corruption_error_catches_snapshot_verifier(tmp_path):
    with pytest.raises(diskgraph.DiskgraphCorruptionError):
        diskgraph.verify_snapshot_integrity(str(tmp_path))
