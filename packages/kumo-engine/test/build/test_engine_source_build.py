# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise pip's bundle backend without compiling native dependencies."""

import hashlib
import importlib.metadata
import importlib.util
import io
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, Mock
from zipfile import ZipFile

import pytest
from wheel.wheelfile import WheelFile  # type: ignore[import-untyped]

BACKEND = (
    Path(__file__).resolve().parents[4]
    / "packages/kumo-engine/scripts/build/kumo_engine_build_backend.py"
)


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Return an independently loaded backend module for each test."""
    spec = importlib.util.spec_from_file_location("engine_source_backend", BACKEND)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    temporary = MagicMock()
    temporary.return_value.__enter__.return_value = "/isolated-build"
    monkeypatch.setattr(module, "TemporaryDirectory", temporary)
    monkeypatch.setattr(module.venv, "EnvBuilder", Mock())
    monkeypatch.setattr(
        module.shutil, "which", Mock(return_value="/existing/bin/cargo")
    )
    return module


@pytest.mark.parametrize(
    ("host", "platform_tag", "expected_tag"),
    [
        ("linux", "linux-x86_64", "linux_x86_64"),
        ("linux", "linux-aarch64", "linux_aarch64"),
        ("darwin", "macosx-15.0-arm64", ""),
    ],
)
def test_source_build_reuses_assembler_with_native_tags(
    backend: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    host: str,
    platform_tag: str,
    expected_tag: str,
) -> None:
    """Use the shared assembler without claiming release portability."""
    monkeypatch.setattr(backend.sys, "platform", host)
    monkeypatch.setattr(backend.platform, "libc_ver", lambda: ("glibc", "2.39"))
    monkeypatch.setattr(backend.sysconfig, "get_platform", lambda: platform_tag)
    monkeypatch.setenv("KUMO_REPO", "/another-checkout")
    monkeypatch.setenv("USE_ZIG", "1")
    monkeypatch.setenv("PLATFORM_TAG", "manylinux_2_34_x86_64")
    monkeypatch.setenv("VERIFY_IMAGE", "unrequested-image")
    monkeypatch.setenv("_PYPROJECT_HOOKS_BACKEND_PATH", "/outer-backend")
    monkeypatch.setenv("_PYPROJECT_HOOKS_BUILD_BACKEND", "outer_backend")
    monkeypatch.setenv("PIP_BUILD_TRACKER", "/outer-tracker")
    monkeypatch.setenv("PYTHONPATH", "/managed-notebook-packages")
    monkeypatch.setenv("PYTHONHOME", "/managed-python")
    wheel = "kumo_engine-1.0-cp311-abi3-linux_x86_64.whl"
    run = Mock(return_value=subprocess.CompletedProcess([], 0, f"/wheels/{wheel}\n"))
    monkeypatch.setattr(backend.subprocess, "run", run)

    result = backend.build_wheel("/wheels")

    assert result == wheel
    command = run.call_args.args[0]
    assert command == [
        "bash",
        str(BACKEND.with_name("build_wheel.sh")),
        "--no-zig",
        "--output-dir",
        "/wheels",
    ]
    environment = run.call_args.kwargs["env"]
    assert environment["KUMO_REPO"] == str(BACKEND.parents[4])
    assert environment["PYTHON"] == "/isolated-build/bin/python"
    assert environment["VIRTUAL_ENV"] == "/isolated-build"
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    backend.venv.EnvBuilder.assert_called_once_with(with_pip=False)
    backend.venv.EnvBuilder.return_value.create.assert_called_once_with(
        "/isolated-build"
    )
    assert run.call_args_list[0].args[0] == [
        backend.sys.executable,
        "-m",
        "pip",
        "--python",
        "/isolated-build/bin/python",
        "install",
        "--ignore-installed",
        "--no-deps",
        "pip>=25",
    ]
    assert environment["USE_ZIG"] == "0"
    assert environment["VERIFY_IMAGE"] == ""
    assert environment["PLATFORM_TAG"] == expected_tag
    assert "_PYPROJECT_HOOKS_BACKEND_PATH" not in environment
    assert "_PYPROJECT_HOOKS_BUILD_BACKEND" not in environment
    assert "PIP_BUILD_TRACKER" not in environment
    if host == "linux":
        assert environment["MAX_GLIBC"] == "2.39"


def test_component_mode_delegates_without_recursion(
    backend: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build only the engine component when requested by the assembler."""
    build = Mock(return_value="engine-component.whl")
    requirements = Mock(return_value=["wheel"])
    monkeypatch.setattr(backend.build_meta, "build_wheel", build)
    monkeypatch.setattr(
        backend.build_meta, "get_requires_for_build_wheel", requirements
    )
    run = Mock()
    monkeypatch.setattr(backend.subprocess, "run", run)
    settings = {"kumo-component": "true"}

    assert backend.get_requires_for_build_wheel() == []
    assert backend.get_requires_for_build_wheel(settings) == ["wheel"]
    assert backend.build_wheel("/wheels", settings, "/metadata") == (
        "engine-component.whl"
    )
    build.assert_called_once_with("/wheels", settings, "/metadata")
    requirements.assert_called_once_with(settings)
    run.assert_not_called()


def test_workspace_editable_hooks_remain_setuptools_hooks(backend: ModuleType) -> None:
    """Keep workspace editables independent of the bundle build."""
    assert backend.build_editable is backend.build_meta.build_editable
    assert (
        backend.get_requires_for_build_editable
        is backend.build_meta.get_requires_for_build_editable
    )
    assert (
        backend.prepare_metadata_for_build_editable
        is backend.build_meta.prepare_metadata_for_build_editable
    )


@pytest.mark.parametrize(
    ("host", "libc"), [("win32", ("", "")), ("linux", ("musl", "1.2"))]
)
def test_unsupported_host_fails_before_assembly(
    backend: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    host: str,
    libc: tuple[str, str],
) -> None:
    """Reject hosts that cannot produce the supported native bundle."""
    monkeypatch.setattr(backend.sys, "platform", host)
    monkeypatch.setattr(backend.platform, "libc_ver", lambda: libc)
    run = Mock()
    monkeypatch.setattr(backend.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="require"):
        backend.build_wheel("/wheels")

    run.assert_not_called()


def test_assembly_failure_is_reported(
    backend: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Preserve assembler diagnostics and its failing exit status."""
    error = subprocess.CalledProcessError(73, ["bash"], output="component failed")
    monkeypatch.setattr(backend.subprocess, "run", Mock(side_effect=error))

    with pytest.raises(subprocess.CalledProcessError) as caught:
        backend.build_wheel("/wheels")

    assert caught.value is error
    assert "component failed" in caplog.text


@pytest.mark.parametrize("output", ["", "no wheel returned\n"])
def test_missing_wheel_path_is_reported(
    backend: ModuleType, monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    """Fail clearly when assembly does not produce its promised output."""
    run = Mock(return_value=subprocess.CompletedProcess([], 0, output))
    monkeypatch.setattr(backend.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="did not return a wheel"):
        backend.build_wheel("/wheels")


@pytest.mark.skipif(sys.platform != "linux", reason="Linux frontend fixture")
@pytest.mark.parametrize(
    "bootstrap",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not os.environ.get("KUMO_TEST_RUST_BOOTSTRAP"),
                reason="Explicit network test",
            ),
        ),
    ],
)
def test_pip_frontend_installs_the_assembled_wheel(
    tmp_path: Path,
    bootstrap: bool,
) -> None:
    """Exercise pip's metadata fallback with a mocked native assembler.

    Args:
        tmp_path: Isolated directory retaining the mocked repository and wheel.
        bootstrap: Whether to exercise real Rust provisioning with no Cargo on PATH.
    """
    repository = tmp_path / "repository"
    source = repository / "packages/kumo-engine"
    scripts = source / "scripts/build"
    scripts.mkdir(parents=True)
    build_tools = repository / "build-tools"
    build_tools.mkdir()
    pip = importlib.metadata.distribution("pip")
    with WheelFile(build_tools / f"pip-{pip.version}-py3-none-any.whl", "w") as wheel:
        for relative in pip.files or []:
            if relative.name == "RECORD" or relative.suffix == ".pyc":
                continue
            if relative.parts[0] == "pip" or relative.parts[0].endswith(".dist-info"):
                wheel.write(pip.locate_file(relative), str(relative))
    shutil.copyfile(BACKEND, scripts / BACKEND.name)
    project = tomllib.loads((BACKEND.parents[2] / "pyproject.toml").read_text())
    build = project["build-system"]
    (source / "pyproject.toml").write_text(
        "[build-system]\n"
        f"requires = {build['requires']!r}\n"
        f"build-backend = {build['build-backend']!r}\n"
        f"backend-path = {build['backend-path']!r}\n"
        '[project]\nname = "kumo-engine"\nversion = "1.0"\n'
        "[tool.setuptools]\npy-modules = []\n"
    )
    platform_tag = sysconfig.get_platform().replace("-", "_")
    wheel_name = f"kumo_engine-1.0-cp311-abi3-{platform_tag}.whl"
    assembled = repository / wheel_name
    with ZipFile(assembled, "w") as archive:
        archive.writestr(
            "kumo_engine-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: kumo-engine\nVersion: 1.0\n",
        )
        archive.writestr(
            "kumo_engine-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: false\n"
            f"Tag: cp311-abi3-{platform_tag}\n",
        )
        archive.writestr("kumo_engine-1.0.dist-info/RECORD", "")
        archive.writestr("kumo/engine/__init__.py", "ASSEMBLED = True\n")
    component = repository / "component"
    component.mkdir()
    (component / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools"]\n'
        'build-backend = "setuptools.build_meta"\n'
        '[project]\nname = "component"\nversion = "1.0"\n'
        "[tool.setuptools]\npy-modules = []\n"
    )
    (scripts / "build_wheel.sh").write_text(
        "set -eu\n"
        'test "$1" = --no-zig\n'
        'test "$2" = --output-dir\n'
        '"$PYTHON" -m pip --version\n'
        '"$PYTHON" -c '
        + shlex.quote(
            "from pathlib import Path; import sysconfig; "
            "Path(sysconfig.get_path('purelib'), 'test-build-tools.pth').write_text("
            f"{sysconfig.get_path('purelib')!r})"
        )
        + "\n"
        '"$PYTHON" -m pip wheel --no-deps --no-index --no-build-isolation '
        '"$KUMO_REPO/component" -w "$KUMO_REPO/component-wheels"\n'
        '"$PYTHON" -m pip wheel --no-deps --no-index --no-build-isolation '
        '--config-settings=kumo-component=true "$KUMO_REPO/packages/kumo-engine" '
        '-w "$KUMO_REPO/component-wheels"\n'
        f'cp "$KUMO_REPO/{wheel_name}" "$3/{wheel_name}"\n'
        f'echo "$3/{wheel_name}"\n'
    )
    destination = tmp_path / "wheels"

    environment = dict(
        os.environ,
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        PIP_NO_INDEX="1",
        PIP_FIND_LINKS=str(build_tools),
        PYTHONPATH=sysconfig.get_path("purelib"),
    )
    if bootstrap:
        environment.update(PATH="/usr/bin:/bin", XDG_CACHE_HOME=str(tmp_path / "cache"))
        for variable in ("RUSTUP_HOME", "CARGO_HOME", "RUSTUP_TOOLCHAIN"):
            environment.pop(variable, None)
    installed = tmp_path / "installed"
    for arguments in (
        [
            "wheel",
            "--no-build-isolation",
            "--wheel-dir",
            str(destination),
            str(source),
        ],
        ["install", "--target", str(installed), str(destination / wheel_name)],
    ):
        subprocess.run(
            [sys.executable, "-m", "pip", *arguments, "--no-deps", "--no-index"],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

    assert (destination / wheel_name).read_bytes() == assembled.read_bytes()
    assert (installed / "kumo/engine/__init__.py").read_text() == "ASSEMBLED = True\n"


def test_missing_cargo_fails_before_component_builds(tmp_path: Path) -> None:
    """Reject a missing native toolchain before building Python packages.

    Args:
        tmp_path: Directory containing the restricted executable search path.
    """
    for name in ("dirname", "uname"):
        executable = shutil.which(name)
        assert executable is not None
        (tmp_path / name).symlink_to(executable)

    result = subprocess.run(
        ["bash", str(BACKEND.with_name("build_wheel.sh")), "--no-zig"],
        env=dict(os.environ, PATH=str(tmp_path), PYTHON=sys.executable),
        executable=shutil.which("bash"),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "cargo is required to build DiskGraph" in result.stderr
    assert "building " not in result.stderr


@pytest.mark.parametrize("cached", [False, True])
def test_rust_bootstrap_uses_private_cache_and_native_binaries(
    backend: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cached: bool,
) -> None:
    """Provision missing Rust without depending on isolated Python shims."""
    rustup = tmp_path / "bootstrap/rustup"
    cargo = tmp_path / "toolchain/bin/cargo"
    cargo.parent.mkdir(parents=True)
    cargo.touch()
    monkeypatch.setattr(backend, "_download_rustup", Mock(return_value=rustup))
    monkeypatch.setattr(backend.shutil, "which", Mock(return_value=None))
    resolved = subprocess.CompletedProcess([], 0, f"{cargo}\n")
    calls = (
        [resolved]
        if cached
        else [subprocess.CompletedProcess([], 1, ""), resolved, resolved]
    )
    run = Mock(side_effect=calls)
    monkeypatch.setattr(backend.subprocess, "run", run)
    environment = {
        "PATH": "/usr/bin",
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    }

    backend._configure_rust(environment)

    assert environment["RUSTUP_HOME"] == str(tmp_path / "bootstrap/rustup-home")
    assert environment["CARGO_HOME"] == str(tmp_path / "bootstrap/cargo")
    assert environment["PATH"].split(os.pathsep)[0] == str(cargo.parent)
    assert run.call_count == (1 if cached else 3)
    if not cached:
        assert run.call_args_list[1].args[0] == [
            str(rustup),
            "toolchain",
            "install",
            "stable",
            "--profile",
            "minimal",
            "--no-self-update",
        ]
    assert "CARGO_BUILD_JOBS" not in environment


def test_existing_rust_is_reused(
    backend: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid downloads or environment changes when user Rust is available."""
    download = Mock()
    monkeypatch.setattr(backend, "_download_rustup", download)
    environment = {"PATH": "/existing/bin"}

    backend._configure_rust(environment)

    assert environment == {"PATH": "/existing/bin"}
    download.assert_not_called()


def test_rust_bootstrap_preserves_install_failure(
    backend: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fail source installation if Rust provisioning fails."""
    monkeypatch.setattr(
        backend, "_download_rustup", Mock(return_value=tmp_path / "rustup")
    )
    monkeypatch.setattr(backend.shutil, "which", Mock(return_value=None))
    error = subprocess.CalledProcessError(1, ["rustup", "toolchain", "install"])
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        Mock(side_effect=[subprocess.CompletedProcess([], 1, ""), error]),
    )

    with pytest.raises(subprocess.CalledProcessError) as caught:
        backend._configure_rust({"PATH": "/usr/bin", "XDG_CACHE_HOME": str(tmp_path)})

    assert caught.value is error


@pytest.mark.skipif(
    not os.environ.get("KUMO_TEST_RUST_BOOTSTRAP"),
    reason="Explicit network test downloading the Rust toolchain",
)
def test_downloaded_rust_compiles_a_program(tmp_path: Path) -> None:
    """Exercise actual Rust provisioning from a PATH with no Cargo installed."""
    spec = importlib.util.spec_from_file_location("real_rust_bootstrap", BACKEND)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    environment = dict(
        os.environ, PATH="/usr/bin:/bin", XDG_CACHE_HOME=str(tmp_path / "cache")
    )
    for variable in ("RUSTUP_HOME", "CARGO_HOME", "RUSTUP_TOOLCHAIN"):
        environment.pop(variable, None)
    assert shutil.which("cargo", path=environment["PATH"]) is None

    module._configure_rust(environment)
    source = tmp_path / "hello.rs"
    source.write_text('fn main() { println!("bootstrap-ok"); }\n')
    binary = tmp_path / "hello"
    subprocess.run(
        ["rustc", str(source), "-o", str(binary)], env=environment, check=True
    )
    result = subprocess.run(
        [str(binary)], env=environment, capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "bootstrap-ok"
    assert Path(environment["RUSTUP_HOME"]).is_relative_to(tmp_path)


@pytest.mark.parametrize(
    "host",
    [
        ("linux", "x86_64", "x86_64-unknown-linux-gnu"),
        ("linux", "aarch64", "aarch64-unknown-linux-gnu"),
        ("darwin", "x86_64", "x86_64-apple-darwin"),
        ("darwin", "arm64", "aarch64-apple-darwin"),
    ],
)
def test_rustup_download_is_verified_and_cached(
    backend: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    host: tuple[str, str, str],
) -> None:
    """Select the host installer, verify it, and avoid repeat downloads."""
    system, machine, target = host
    payload = b"verified-rustup-installer"
    monkeypatch.setattr(backend.sys, "platform", system)
    monkeypatch.setattr(backend.platform, "machine", lambda: machine)
    monkeypatch.setattr(backend, "TemporaryDirectory", tempfile.TemporaryDirectory)
    monkeypatch.setitem(
        backend._RUSTUP_SHA256, target, hashlib.sha256(payload).hexdigest()
    )
    download = Mock(return_value=io.BytesIO(payload))
    monkeypatch.setattr(backend, "urlopen", download)

    binary = backend._download_rustup(tmp_path)
    repeated = backend._download_rustup(tmp_path)

    assert binary == repeated == tmp_path / target / "rustup"
    assert binary.read_bytes() == payload
    assert os.access(binary, os.X_OK)
    download.assert_called_once_with(
        f"https://static.rust-lang.org/rustup/archive/1.29.0/{target}/rustup-init",
        timeout=60,
    )


def test_bad_rustup_download_is_never_installed(
    backend: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject a corrupt download while preserving the previous cache file."""
    target = "x86_64-unknown-linux-gnu"
    monkeypatch.setattr(backend.sys, "platform", "linux")
    monkeypatch.setattr(backend.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(backend, "TemporaryDirectory", tempfile.TemporaryDirectory)
    monkeypatch.setattr(backend, "urlopen", Mock(return_value=io.BytesIO(b"corrupt")))
    cached = tmp_path / target / "rustup"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"previous-corrupt-cache")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        backend._download_rustup(tmp_path)

    assert cached.read_bytes() == b"previous-corrupt-cache"


def test_rustup_rejects_unsupported_architecture(
    backend: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fail before downloading a binary for an unsupported host."""
    monkeypatch.setattr(backend.platform, "machine", lambda: "unsupported")
    download = Mock()
    monkeypatch.setattr(backend, "urlopen", download)

    with pytest.raises(RuntimeError, match="unsupported"):
        backend._download_rustup(tmp_path)

    download.assert_not_called()
