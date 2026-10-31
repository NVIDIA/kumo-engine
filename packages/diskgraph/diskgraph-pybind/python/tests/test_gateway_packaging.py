# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require the ingestion and gateway executables in native package builds."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import tomllib
import zipfile
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def backend() -> ModuleType:
    """Load the package build backend without invoking its external toolchain."""
    path = Path(__file__).resolve().parents[2] / "diskgraph_build_backend.py"
    spec = importlib.util.spec_from_file_location("gateway_build_backend", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("missing", ["ingest", "diskgraph-gateway"])
def test_native_wheel_requires_both_executables(
    backend: ModuleType, tmp_path: Path, missing: str
) -> None:
    """Reject a wheel missing either worker executable."""
    wheel = tmp_path / "diskgraph.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in backend.NATIVE_BINARIES:
            if name != missing:
                archive.writestr(f"diskgraph/bin/{name}", b"native")
    with pytest.raises(RuntimeError, match=missing):
        backend._verify_wheel_contains_binaries(wheel)


@pytest.mark.parametrize("use_zig", [False, True])
def test_staging_builds_both_native_workers(
    backend: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_zig: bool
) -> None:
    """Stage the gateway through the same portable build path as ingestion."""
    monkeypatch.setattr(backend, "PROJECT_ROOT", tmp_path / "package")
    monkeypatch.setattr(
        backend, "_cargo_env", lambda: {"CARGO_TARGET_DIR": str(tmp_path / "target")}
    )
    monkeypatch.setattr(backend.shutil, "which", lambda _: "/tools/cargo")
    monkeypatch.setattr(backend, "_zig_env", dict)
    calls = []

    def build(command: list[str], *, env: dict[str, str]) -> None:
        name = command[command.index("--bin") + 1]
        output = Path(env["CARGO_TARGET_DIR"])
        if "--target" in command:
            output = output / command[command.index("--target") + 1]
        output = output / "release" / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(name.encode())
        calls.append(name)

    monkeypatch.setattr(backend, "_run", build)
    backend._stage_native_binaries(use_zig=use_zig)
    backend._verify_staged_binaries_present()
    assert calls == ["ingest", "diskgraph-gateway"]


@pytest.mark.parametrize("included", [False, True])
def test_source_archive_keeps_gateway_workspace_member(
    backend: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, included: bool
) -> None:
    """Keep the standalone crate buildable after maturin prunes workspace members."""
    archive = tmp_path / "diskgraph.tar.gz"
    members = '["diskgraph-pybind", "diskgraph-gateway"]' if included else "[]"
    manifest = f"[workspace]\nmembers = {members}\n".encode()
    with tarfile.open(archive, "w:gz") as output:
        for name, payload in (
            ("Cargo.toml", manifest),
            ("diskgraph-gateway/Cargo.toml", b"[package]\nname='diskgraph-gateway'"),
        ):
            entry = tarfile.TarInfo(f"diskgraph/{name}")
            entry.size = len(payload)
            output.addfile(entry, io.BytesIO(payload))
    monkeypatch.setattr(backend.maturin, "build_sdist", lambda *_: archive.name)

    result = backend.build_sdist(str(tmp_path))

    assert result == archive.name
    with tarfile.open(archive) as output:
        content = output.extractfile("diskgraph/Cargo.toml")
        assert content is not None
        workspace = tomllib.loads(content.read().decode())["workspace"]
        assert workspace["members"].count("diskgraph-gateway") == 1
        assert output.getmember("diskgraph/diskgraph-gateway/Cargo.toml").isfile()
