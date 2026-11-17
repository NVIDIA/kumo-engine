# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A faithful copy of the kumo-core package for tests that build it.

Building in place writes ``build/`` and ``src/kumo_core.egg-info`` into the
source tree, where the egg-info then shadows the installed distribution's
metadata for every process with ``src`` on its path. Build tests copy the
package first and build the copy.
"""
import shutil
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[2]
# Build outputs and caches (the package's .gitignore); everything else is
# copied, including the tests, which the wheel must keep excluding.
_ROOT_OUTPUTS = frozenset({'build', 'dist', '.pytest_cache'})


def _is_output(name: str) -> bool:
    return (name == '__pycache__' or name.endswith('.egg-info')
            or name.endswith(('.pyc', '.pyo', '.pyd')))


def _ignore(directory: str, names: list[str]) -> set[str]:
    ignored = {name for name in names if _is_output(name)}
    if Path(directory) == PACKAGE_ROOT:
        ignored |= _ROOT_OUTPUTS.intersection(names)
    return ignored


def copy_package_source(destination: Path) -> Path:
    """Copy the package directory to ``destination`` (which must not exist)
    without build outputs or caches, and return ``destination``."""
    shutil.copytree(PACKAGE_ROOT, destination, ignore=_ignore)
    return destination
