# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the engine bundle from a Git checkout through the release assembler.

Editable installs retain setuptools' component behavior for the uv workspace.
The assembler requests component mode when building its engine input wheel.
"""

import hashlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import venv
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from setuptools import build_meta
from setuptools.build_meta import (
    build_editable,
    get_requires_for_build_editable,
    prepare_metadata_for_build_editable,
)

__all__ = [
    "build_editable",
    "build_wheel",
    "get_requires_for_build_editable",
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_editable",
]

ConfigSettings = dict[str, str | list[str]]
logger = logging.getLogger(__name__)


# Download/bootstrap approach adapted from rustup-pypi; checksums are from
# the official Rustup 1.29.0 archive release files. Upstream source:
# commit a41b40da1edf5b01575c0f5bd4ff1bb5f336fbb2 (Apache-2.0 election).
# See 3rdparty/notices/rustup-pypi-LICENSE for attribution and modifications.
_RUSTUP_VERSION = "1.29.0"
_RUSTUP_SHA256 = {
    "x86_64-unknown-linux-gnu": (
        "4acc9acc76d5079515b46346a485974457b5a79893cfb01112423c89aeb5aa10"
    ),
    "aarch64-unknown-linux-gnu": (
        "9732d6c5e2a098d3521fca8145d826ae0aaa067ef2385ead08e6feac88fa5792"
    ),
    "x86_64-apple-darwin": (
        "33cf85df9142bc6d29cbc62fa5ca1d4c29622cddb55213a4c1a43c457fb9b2d7"
    ),
    "aarch64-apple-darwin": (
        "aeb4105778ca1bd3c6b0e75768f581c656633cd51368fa61289b6a71696ac7e1"
    ),
}


def _download_rustup(cache: Path) -> Path:
    """Download and verify a pinned Rustup installer, reusing verified cache files."""
    machine = platform.machine().lower()
    machine = {"arm64": "aarch64", "amd64": "x86_64"}.get(machine, machine)
    suffix = "apple-darwin" if sys.platform == "darwin" else "unknown-linux-gnu"
    target = f"{machine}-{suffix}"
    expected = _RUSTUP_SHA256.get(target)
    if expected is None:
        raise RuntimeError(f"Automatic Rust installation is unsupported on {target}")
    binary = cache / target / "rustup"
    if binary.is_file():
        with binary.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() == expected:
                binary.chmod(0o755)
                return binary
        logger.warning("Cached Rustup checksum failed; downloading a verified copy")
    binary.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://static.rust-lang.org/rustup/archive/{_RUSTUP_VERSION}/"
        f"{target}/rustup-init"
    )
    with TemporaryDirectory(prefix="rustup-download-", dir=binary.parent) as directory:
        downloaded = Path(directory) / "rustup"
        with urlopen(url, timeout=60) as response, downloaded.open("wb") as output:
            shutil.copyfileobj(response, output)
        with downloaded.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"Rustup checksum mismatch for {target}")
        downloaded.chmod(0o755)
        downloaded.replace(binary)
    return binary


def _configure_rust(environment: dict[str, str]) -> None:
    """Reuse existing Rust or bootstrap a minimal toolchain in a private cache."""
    path = environment.get("PATH", "")
    if shutil.which("cargo", path=path) and shutil.which("rustc", path=path):
        return
    cache = (
        Path(environment.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        / "kumo/rust"
    )
    rustup = _download_rustup(cache)
    environment.setdefault("RUSTUP_HOME", str(rustup.parent / "rustup-home"))
    environment.setdefault("CARGO_HOME", str(rustup.parent / "cargo"))
    toolchain = environment.get("RUSTUP_TOOLCHAIN") or "stable"
    command = [str(rustup), "which", "--toolchain", toolchain, "cargo"]
    result = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        logger.info("Installing Rust %s in %s", toolchain, environment["RUSTUP_HOME"])
        subprocess.run(
            [
                str(rustup),
                "toolchain",
                "install",
                toolchain,
                "--profile",
                "minimal",
                "--no-self-update",
            ],
            env=environment,
            stdout=sys.stderr,
            check=True,
        )
        result = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
    cargo = Path(result.stdout.strip())
    if not cargo.is_file():
        raise RuntimeError(f"Rust bootstrap did not produce Cargo: {cargo}")
    # Use the installed binaries without modifying the user's shell profile.
    environment["PATH"] = os.pathsep.join((str(cargo.parent), path))


def get_requires_for_build_wheel(
    config_settings: ConfigSettings | None = None,
) -> list[str]:
    """Return additional build requirements for the selected wheel mode.

    Args:
        config_settings: PEP 517 settings; the assembler selects component mode.

    Returns:
        Setuptools requirements for components, or none for the bundle.
    """
    if config_settings and config_settings.get("kumo-component") == "true":
        return build_meta.get_requires_for_build_wheel(config_settings)
    return []


def build_wheel(
    wheel_directory: str,
    config_settings: ConfigSettings | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Build a native engine bundle, or the assembler's engine component.

    Args:
        wheel_directory: Destination directory for the wheel.
        config_settings: PEP 517 settings; ``kumo-component=true`` is internal.
        metadata_directory: Prepared metadata for a component build, if supplied.

    Returns:
        The generated wheel's filename, relative to ``wheel_directory``.

    Raises:
        RuntimeError: If the host or installer is invalid, or assembly returns no path.
        subprocess.CalledProcessError: If Rust provisioning or wheel assembly fails.
        OSError: If downloading or caching the installer fails.
    """
    if config_settings and config_settings.get("kumo-component") == "true":
        return build_meta.build_wheel(
            wheel_directory, config_settings, metadata_directory
        )

    script = Path(__file__).with_name("build_wheel.sh").resolve()
    environment = dict(
        os.environ,
        KUMO_REPO=str(script.parents[4]),
        USE_ZIG="0",
        VERIFY_IMAGE="",
        PLATFORM_TAG="",
    )
    # Nested pip builds must select their own backend and backend path.
    environment.pop("_PYPROJECT_HOOKS_BACKEND_PATH", None)
    environment.pop("_PYPROJECT_HOOKS_BUILD_BACKEND", None)
    environment.pop("PIP_BUILD_TRACKER", None)
    if sys.platform == "linux":
        libc, version = platform.libc_ver()
        if libc != "glibc" or not version:
            raise RuntimeError("Linux source builds require glibc")
        environment["MAX_GLIBC"] = version
        environment["PLATFORM_TAG"] = sysconfig.get_platform().replace("-", "_")
    elif sys.platform != "darwin":
        raise RuntimeError("Source builds require Linux with glibc or macOS")

    try:
        _configure_rust(environment)
        with TemporaryDirectory(prefix="kumo-source-build-") as directory:
            venv.EnvBuilder(with_pip=False).create(directory)
            python = str(Path(directory) / "bin/python")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "--python",
                    python,
                    "install",
                    "--ignore-installed",
                    "--no-deps",
                    "pip>=25",
                ],
                env=environment,
                stdout=sys.stderr,
                check=True,
            )
            # Managed notebook paths can precede pip's isolated build packages.
            environment.pop("PYTHONPATH", None)
            environment.pop("PYTHONHOME", None)
            environment.update(PYTHON=python, VIRTUAL_ENV=directory)
            result = subprocess.run(
                ["bash", str(script), "--no-zig", "--output-dir", wheel_directory],
                env=environment,
                stdout=subprocess.PIPE,
                text=True,
                check=True,
            )
    except subprocess.CalledProcessError as error:
        if error.stdout:
            logger.error(f"Engine assembly failed:\n{error.stdout}")
        raise
    sys.stderr.write(result.stdout)
    output = result.stdout.splitlines()
    if not output or not output[-1].endswith(".whl"):
        raise RuntimeError("The engine assembler did not return a wheel path")
    return Path(output[-1]).name
