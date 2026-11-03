# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the engine bundle from a Git checkout through the release assembler.

Editable installs retain setuptools' component behavior for the uv workspace.
The assembler requests component mode when building its engine input wheel.
"""

import logging
import os
import platform
import subprocess
import sys
import sysconfig
import venv
from pathlib import Path
from tempfile import TemporaryDirectory

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
        RuntimeError: If the host is unsupported or the assembler returns no path.
        subprocess.CalledProcessError: If component or bundle assembly fails.
    """
    if config_settings and config_settings.get("kumo-component") == "true":
        return build_meta.build_wheel(
            wheel_directory, config_settings, metadata_directory
        )

    script = Path(__file__).resolve().parents[2] / "scripts/build/build_wheel.py"
    environment = dict(
        os.environ,
        KUMO_REPO=str(script.parents[2]),
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
                [python, str(script), "--no-zig", "--output-dir", wheel_directory],
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
