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


def test_build_jobs_respect_override_and_cap_only_the_default(monkeypatch):
    from scripts import build_kumo_loader_wheels as builder

    monkeypatch.setattr(builder.os, "cpu_count", lambda: 64)
    monkeypatch.delenv("CARGO_BUILD_JOBS", raising=False)
    assert builder.cargo_jobs() == "8"
    monkeypatch.setattr(builder.os, "cpu_count", lambda: None)
    assert builder.cargo_jobs() == "1"
    monkeypatch.setenv("CARGO_BUILD_JOBS", "3")
    assert builder.cargo_jobs() == "3"


def test_cargo_target_identity_tracks_source_lock_toolchain_and_flags(tmp_path, monkeypatch):
    from scripts import build_kumo_loader_wheels as builder

    monkeypatch.delenv("RUSTFLAGS", raising=False)
    monkeypatch.delenv("CARGO_ENCODED_RUSTFLAGS", raising=False)
    package = tmp_path / "packages/diskgraph"
    package.mkdir(parents=True)
    (package / "Cargo.lock").write_text("first lock")
    original = builder.cargo_cache_key(tmp_path, "source-a", "nightly-a")
    assert builder.cargo_cache_key(tmp_path, "source-b", "nightly-a") != original
    assert builder.cargo_cache_key(tmp_path, "source-a", "nightly-b") != original
    (package / "Cargo.lock").write_text("different lock")
    assert builder.cargo_cache_key(tmp_path, "source-a", "nightly-a") != original
    changed = builder.cargo_cache_key(tmp_path, "source-a", "nightly-a")
    monkeypatch.setenv("RUSTFLAGS", "-C target-cpu=native")
    assert builder.cargo_cache_key(tmp_path, "source-a", "nightly-a") != changed


def test_cold_build_uses_constraints_parallelism_and_reuses_verified_wheels(tmp_path, monkeypatch):
    from scripts import build_kumo_loader_wheels as builder

    revision = "a" * 40
    monkeypatch.setattr(
        builder, "source_identity", lambda: ("https://example.invalid/repo", revision)
    )
    monkeypatch.setenv("CARGO_BUILD_JOBS", "3")
    monkeypatch.setattr(builder.subprocess, "check_output", lambda *args, **kwargs: revision)
    calls = []

    def run(command, *, cwd, env, check):
        assert check
        calls.append((command, dict(env)))
        if command[:2] == ("git", "init"):
            source = Path(command[2])
            package = source / "packages/diskgraph"
            package.mkdir(parents=True)
            (package / "Cargo.lock").write_text("fixed lock")
            (package / "rust-toolchain.toml").write_text('[toolchain]\nchannel="nightly-fixed"\n')
        if command[0] == "test-uv":
            assert command[command.index("--build-constraint") + 1] == str(
                builder.BUILD_CONSTRAINTS
            )
            stage = Path(command[command.index("--out-dir") + 1])
            stage.mkdir(exist_ok=True)
            name = next(name for name, path in builder.PACKAGES.items() if path == command[-1])
            (stage / f"{name}-1.0-py3-none-any.whl").write_bytes(name.encode())
            assert env["CARGO_BUILD_JOBS"] == "3"
            assert Path(env["CARGO_TARGET_DIR"]).is_relative_to(tmp_path / "cargo")

    monkeypatch.setattr(builder.subprocess, "run", run)
    monkeypatch.setattr(
        builder.sys,
        "argv",
        [
            "build",
            "--uv",
            "test-uv",
            "--output",
            str(tmp_path / "out"),
            "--cache",
            str(tmp_path / "cache"),
            "--cargo-cache",
            str(tmp_path / "cargo"),
        ],
    )
    builder.main()
    assert len([call for call, _env in calls if call[0] == "test-uv"]) == len(PACKAGES)
    calls.clear()
    builder.main()
    assert not calls
