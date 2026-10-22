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
