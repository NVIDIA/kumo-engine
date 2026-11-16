# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from kumo_core.config import NativeRuntimeConfig
from kumo_core.manifest import RunManifest, RunPaths


def test_run_paths_are_explicit_and_manifest_round_trips(
        tmp_path: Path) -> None:
    config = NativeRuntimeConfig(scratch_root=tmp_path / "scratch",
                                 artifact_root=tmp_path / "artifacts")
    paths = RunPaths.from_config(config, "run-123")
    manifest = RunManifest.create(
        "run-123",
        paths,
        ("ingest", "train"),
        input_fingerprint="input-sha256",
    )

    restored = RunManifest.from_dict(manifest.to_dict())

    assert restored == manifest
    assert restored.input_fingerprint == "input-sha256"
    assert paths.scratch_dir == tmp_path / "scratch" / "run-123"
    assert paths.manifest_path == tmp_path / "artifacts" / "run-123" / "run-manifest.json"


@pytest.mark.parametrize(
    "run_id", ["", ".", "..", "nested/run", "/absolute", "spaces are unsafe"])
def test_run_id_must_be_a_safe_path_component(tmp_path: Path,
                                              run_id: str) -> None:
    config = NativeRuntimeConfig(tmp_path / "scratch", tmp_path / "artifacts")

    with pytest.raises(ValueError):
        RunPaths.from_config(config, run_id)


def test_manifest_filename_must_not_escape_run_directory(
        tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        NativeRuntimeConfig(tmp_path / "scratch", tmp_path / "artifacts",
                            manifest_filename="../manifest")
