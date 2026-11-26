# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI must never reuse a mixed-source or corrupted native wheel set."""

import json
import zipfile
from pathlib import Path

import pytest
from scripts.build_kumo_loader_wheels import PACKAGES, digest, source_identity, valid_cache

from tfm.extras.kumo_pytorch import SOURCE_REVISION


def write_wheel(path: Path, package: str, *, mode: int = 0o755) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        name = "diskgraph/bin/ingest" if package == "diskgraph" else package + "/__init__.py"
        info = zipfile.ZipInfo(name)
        info.external_attr = (0o100000 | mode) << 16
        archive.writestr(info, package.encode())


def test_dependency_pins_agree():
    url, revision = source_identity()
    assert url == "https://github.com/NVIDIA/kumo-engine.git"
    assert revision == SOURCE_REVISION


def test_cache_rejects_modified_bytes_or_revision(tmp_path: Path):
    identity = {"revision": SOURCE_REVISION}
    wheels = {}
    for package in PACKAGES:
        path = tmp_path / f"{package}-1.0-py3-none-any.whl"
        write_wheel(path, package)
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
            (package / "pyproject.toml").write_text(
                '[tool.maturin]\npython-source="diskgraph-pybind/python"\n'
                'include=[{path="diskgraph/bin/ingest",format="wheel"},'
                '{path="diskgraph/bin/ingest.exe",format="wheel"}]\n'
            )
            (package / "rust-toolchain.toml").write_text('[toolchain]\nchannel="nightly-fixed"\n')
        if command[0] == "test-uv":
            assert command[command.index("--build-constraint") + 1] == str(
                builder.BUILD_CONSTRAINTS
            )
            stage = Path(command[command.index("--out-dir") + 1])
            stage.mkdir(exist_ok=True)
            name = next(name for name, path in builder.PACKAGES.items() if path == command[-1])
            write_wheel(stage / f"{name}-1.0-py3-none-any.whl", name)
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


@pytest.mark.parametrize("already_fixed", [False, True])
def test_pinned_diskgraph_includes_resolve_from_python_source(tmp_path, already_fixed):
    import tomllib

    from scripts import build_kumo_loader_wheels as builder

    package = tmp_path / builder.PACKAGES["diskgraph"]
    package.mkdir(parents=True)
    prefix = "" if already_fixed else "diskgraph-pybind/python/"
    manifest = package / "pyproject.toml"
    manifest.write_text(
        '[tool.maturin]\npython-source="diskgraph-pybind/python"\ninclude=['
        + ",".join(
            f'{{path = "{prefix}diskgraph/bin/{binary}", format = "wheel"}}'
            for binary in ("ingest", "ingest.exe")
        )
        + "]\n"
    )
    staged = package / "diskgraph-pybind/python/diskgraph/bin"
    staged.mkdir(parents=True)
    for binary in ("ingest", "ingest.exe"):
        (staged / binary).write_bytes(b"executable")
    builder.prepare_diskgraph_packaging(tmp_path)
    config = tomllib.loads(manifest.read_text())["tool"]["maturin"]
    assert all(
        (package / config["python-source"] / item["path"]).is_file() for item in config["include"]
    )
    corrected = manifest.read_text()
    builder.prepare_diskgraph_packaging(tmp_path)
    assert manifest.read_text() == corrected


def test_cache_rejects_hashed_but_nonexecutable_or_misplaced_ingest(tmp_path):
    identity = {"revision": SOURCE_REVISION}
    wheels = {}
    for package in PACKAGES:
        wheel = tmp_path / f"{package}-1.0-py3-none-any.whl"
        write_wheel(wheel, package)
        wheels[wheel.name] = digest(wheel)
    diskgraph = tmp_path / next(name for name in wheels if name.startswith("diskgraph-"))
    for content, mode in [("diskgraph/bin/ingest", 0o644), ("wrong/ingest", 0o755)]:
        with zipfile.ZipFile(diskgraph, "w") as archive:
            info = zipfile.ZipInfo(content)
            info.external_attr = (0o100000 | mode) << 16
            archive.writestr(info, b"executable")
        wheels[diskgraph.name] = digest(diskgraph)
        (tmp_path / "manifest.json").write_text(
            json.dumps({"identity": identity, "wheels": wheels})
        )
        assert valid_cache(tmp_path, identity) is None
