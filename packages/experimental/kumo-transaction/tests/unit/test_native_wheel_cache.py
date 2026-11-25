# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI must never reuse a mixed-source or corrupted native wheel set."""

import json
from pathlib import Path

from scripts.build_kumo_loader_wheels import PACKAGES, digest, source_identity, valid_cache

from tfm.extras.kumo_pytorch import SOURCE_REVISION


def test_dependency_pins_agree():
    url, revision = source_identity()
    assert url == "https://github.com/NVIDIA/kumo-engine.git"
    assert revision == SOURCE_REVISION


def test_cache_rejects_modified_bytes_or_revision(tmp_path: Path):
    identity = {"revision": SOURCE_REVISION}
    wheels = {}
    for package in PACKAGES:
        path = tmp_path / f"{package}-1.0-py3-none-any.whl"
        path.write_bytes(package.encode())
        wheels[path.name] = digest(path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"identity": identity, "wheels": wheels}))
    assert valid_cache(tmp_path, identity) == wheels
    assert valid_cache(tmp_path, {"revision": "different"}) is None
    victim = tmp_path / next(iter(wheels))
    victim.write_bytes(b"corrupt")
    assert valid_cache(tmp_path, identity) is None
    manifest.write_text("broken JSON")
    assert valid_cache(tmp_path, identity) is None
