# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check native compatibility and shared wheel orchestration without compilers."""

import argparse
import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock
from zipfile import ZipFile

import pytest

BUILDER = (
    Path(__file__).resolve().parents[4]
    / "packages/kumo-engine/scripts/build/build_wheel.py"
)


@pytest.fixture
def builder() -> ModuleType:
    """Load the builder without executing its CLI."""
    spec = importlib.util.spec_from_file_location("engine_wheel_builder", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tool", ["objdump", "readelf"])
@pytest.mark.parametrize("ceiling", ["2.34", "2.39"])
def test_glibc_checks_ingest_and_extension(
    builder: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tool: str,
    ceiling: str,
) -> None:
    """Reject an ingest binary newer than the ceiling even if the extension fits."""
    wheel = tmp_path / "diskgraph.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("diskgraph/bin/ingest", b"ingest")
        archive.writestr("diskgraph/_c.abi3.so", b"extension")
        archive.writestr("diskgraph/__init__.py", "")
    monkeypatch.setattr(
        builder.shutil,
        "which",
        lambda name: f"/usr/bin/{tool}" if name == tool else None,
    )
    run = Mock(side_effect=["GLIBC_2.9 GLIBC_2.39", "GLIBC_2.17 GLIBC_2.34"])
    monkeypatch.setattr(builder, "_run", run)

    if ceiling == "2.34":
        with pytest.raises(RuntimeError, match="above the 2.34 ceiling"):
            builder._check_glibc(wheel, ceiling, {}, tmp_path)
    else:
        builder._check_glibc(wheel, ceiling, {}, tmp_path)

    assert run.call_count == 2
    option = "-T" if tool == "objdump" else "--dyn-syms"
    assert option in run.call_args.args[0]


def test_missing_symbol_tools_warns(
    builder: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Report a skipped compatibility check when no symbol tools are installed."""
    monkeypatch.setattr(builder.shutil, "which", lambda name: None)

    builder._check_glibc(tmp_path / "not-opened.whl", "2.34", {}, tmp_path)

    assert "skipping the glibc floor check" in caplog.text


def test_missing_component_wheel_fails(builder: ModuleType, tmp_path: Path) -> None:
    """Reject a successful component command that produced no wheel."""
    with pytest.raises(RuntimeError, match="Expected a wheel matching"):
        builder._record(
            tmp_path,
            "diskgraph-*.whl",
            tmp_path / "components.tsv",
            "diskgraph-native",
            "test",
        )


def test_notice_zig_and_docker_share_orchestration(
    builder: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Preserve release-only checks and forward native build settings."""
    repository = tmp_path / "repo"
    local = repository / "packages/kumo-engine"
    local.mkdir(parents=True)
    diskgraph = repository / "packages/diskgraph/diskgraph-pybind"
    diskgraph.mkdir(parents=True)
    (diskgraph / "Cargo.toml").touch()
    (repository / "LICENSE").touch()
    (repository / "THIRD-PARTY.txt").touch()
    monkeypatch.setattr(builder, "REPO_ROOT", repository)
    monkeypatch.setattr(builder.platform, "system", lambda: "Linux")
    monkeypatch.setattr(builder, "_git", lambda *args: "abc123")
    monkeypatch.setattr(
        builder.shutil,
        "which",
        lambda name: None if name == "cargo-zigbuild" else f"/usr/bin/{name}",
    )
    monkeypatch.setattr(builder, "_check_glibc", Mock())
    monkeypatch.delenv("USE_ZIG", raising=False)
    work = tmp_path / "work"
    work.mkdir()
    output = tmp_path / "output.whl"
    native = "diskgraph-1-cp311-abi3-manylinux_2_17_x86_64.whl"

    def run(
        command: list[str], environment: dict[str, str], *, capture: bool = False
    ) -> str:
        if command[1:4] == ["-m", "pip", "wheel"]:
            assert environment["DISKGRAPH_BUILD_ZIG"] == "1"
            (work / "wheels" / native).touch()
        if "--license-expression" in command and "--base-license" in command:
            return "Apache-2.0 AND MIT\n"
        if Path(command[1]).name == "merge_wheels.py":
            output.touch()
            return str(output)
        return ""

    invoked = Mock(side_effect=run)
    monkeypatch.setattr(builder, "_run", invoked)
    args = argparse.Namespace(
        kumo_repo=repository,
        no_zig=False,
        version="1.0",
        release=True,
        output_dir=tmp_path,
        platform_tag="",
        offline_report=True,
        verify_image="test:image",
    )

    assert builder._assemble(args, {"PYTHON": "python"}, work) == output

    commands = [call.args[0] for call in invoked.call_args_list]
    assert ["cargo", "install", "cargo-zigbuild", "--locked"] in commands
    assert any("--check" in command for command in commands)
    assert any("--components" in command for command in commands)
    assert any("--offline-report" in command for command in commands)
    docker = commands[-1]
    assert docker[:3] == ["docker", "run", "--rm"]
    assert "test:image" in docker
    assert (
        work / "verify.py"
    ).read_text() == builder.VERIFY_INSTALL + builder.VERIFY_SOURCE


@pytest.mark.parametrize("keep", [False, True])
def test_cli_preserves_failure_status_and_scratch_policy(
    builder: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    keep: bool,
) -> None:
    """Keep component failures actionable while honoring --keep-work."""
    parser = Mock()
    parser.parse_args.return_value = argparse.Namespace(keep_work=keep)
    monkeypatch.setattr(builder, "_parser", lambda: parser)
    monkeypatch.setattr(builder.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    error = subprocess.CalledProcessError(73, ["cargo"], output="compile failed")
    monkeypatch.setattr(builder, "_assemble", Mock(side_effect=error))
    cleanup = Mock()
    monkeypatch.setattr(builder.shutil, "rmtree", cleanup)

    assert builder.main() == 73

    assert cleanup.call_count == (0 if keep else 1)
