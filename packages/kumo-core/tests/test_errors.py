# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from kumo_core.errors import PublicError

PACKAGE_ROOT = Path(__file__).parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"


def _isolated_import(package_path: Path) -> subprocess.CompletedProcess[str]:
    script = """
import importlib.abc
import sys

sys.path.insert(0, sys.argv[1])
blocked = {'common_utils', 'temporalio', 'temporalio_commons'}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in blocked:
            raise RuntimeError('forbidden import: ' + fullname)
        return None

sys.meta_path.insert(0, Blocker())
from kumo_core.errors import PublicError

error = PublicError('safe message', error=RuntimeError('private'))
assert str(error) == 'safe message'
assert error.get_user_message() == 'safe message'
assert not blocked.intersection(sys.modules)
"""
    return subprocess.run(
        [sys.executable, "-I", "-c", script,
         str(package_path)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_errors_have_no_platform_transitive_imports() -> None:
    result = _isolated_import(SOURCE_ROOT)

    assert result.returncode == 0, result.stderr


def test_runtime_error_contract() -> None:
    wrapped = RuntimeError("private provider response")
    error = PublicError("Could not read source", error=wrapped)

    assert str(error) == "Could not read source"
    assert error.message == "Could not read source"
    assert error.error is wrapped
    assert error.to_dict()["message"] == "Could not read source"
    assert "private provider response" not in str(error.to_dict())


def test_fresh_wheel_imports_without_platform_packages(tmp_path: Path) -> None:
    if importlib.util.find_spec("pip") is None:
        pytest.skip("building the wheel needs pip, absent from a uv venv")
    wheel_dir = tmp_path / "wheel"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            str(PACKAGE_ROOT),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    wheels = list(wheel_dir.glob("kumo_core-*.whl"))
    assert len(wheels) == 1
    installed = tmp_path / "installed"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(installed),
            str(wheels[0]),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    result = _isolated_import(installed)
    assert result.returncode == 0, result.stderr
