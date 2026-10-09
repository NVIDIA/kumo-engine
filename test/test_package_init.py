# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guardrail that every directory containing `.py` source files also ships an
``__init__.py`` so that ``setuptools.find_packages()`` picks it up and the
installed wheel/image can actually import it.

Without this, ``pip install .`` silently drops the directory from the package,
and a runtime ``ModuleNotFoundError`` appears only once the image is built and
run. See kumo-ai/kumo#28300.

Directories that are legitimately not Python packages (scripts, tooling,
examples, agent/IDE hooks, migration roots, etc.) are listed in
``ALLOWED_MISSING_INIT`` below. Add to that set only with intent; each entry
should be a directory that will never be imported as a package.
"""
from __future__ import annotations

import os
import runpy
from pathlib import Path

import setuptools

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directory names whose entire subtree is skipped from the walk.
_SKIP_DIR_NAMES = {
    '.git',
    '.mise',
    '.nodeenv',
    '.pytest_cache',
    '.venv',
    '__pycache__',
    'node_modules',
    'venv',
}

# Directories (relative to repo root, forward-slash separated) that are
# allowed to contain `.py` files without an `__init__.py`. These are
# intentionally NOT Python packages — they hold standalone scripts,
# tooling, hooks, or examples that are never imported as packages.
#
# Keep this list intentional and tight. If a new directory shows up in test
# failures, the default should be to add an `__init__.py`, not to add it here.
ALLOWED_MISSING_INIT: frozenset[str] = frozenset({
    '.agents/hooks',
    '.agents/scripts',
    '.claude/hooks',
    'benchmark',
    'benchmark/explain',
    'docker',
    'examples',
    'examples/sdk-local',
    'scripts/docs',
    'test/scripts/docs',
})


def _iter_python_dirs():
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_NAMES]
        has_py = any(f.endswith('.py') for f in filenames)
        if not has_py:
            continue
        rel = os.path.relpath(dirpath, REPO_ROOT).replace(os.sep, '/')
        if rel == '.':
            continue
        yield rel, filenames


def test_every_python_dir_has_init() -> None:
    missing = []
    for rel, filenames in _iter_python_dirs():
        if '__init__.py' in filenames:
            continue
        if rel in ALLOWED_MISSING_INIT:
            continue
        missing.append(rel)

    assert not missing, (
        'The following directories contain .py files but no __init__.py. '
        'This breaks `setuptools.find_packages()` and Docker image builds. '
        'Add an __init__.py (preferred) or, if the directory is intentionally '
        'not a Python package, add it to ALLOWED_MISSING_INIT in '
        'test/test_package_init.py.\nMissing:\n  - ' +
        '\n  - '.join(sorted(missing)))


def test_allowed_missing_init_is_still_valid() -> None:
    # Prevent the allowlist from going stale: every listed path should still
    # exist and still legitimately be missing an __init__.py.
    stale = []
    for rel in ALLOWED_MISSING_INIT:
        abs_path = REPO_ROOT / rel
        if not abs_path.is_dir():
            stale.append(f'{rel} (directory no longer exists)')
            continue
        if (abs_path / '__init__.py').exists():
            stale.append(f'{rel} (now has __init__.py)')

    assert not stale, (
        'ALLOWED_MISSING_INIT contains entries that are no longer valid. '
        'Remove them from test/test_package_init.py:\n  - ' +
        '\n  - '.join(sorted(stale)))


def test_shipped_memory_tools_declare_psutil(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(setuptools, 'setup',
                        lambda **kwargs: captured.update(kwargs))

    runpy.run_path(str(REPO_ROOT / 'setup.py'), run_name='kumo_setup_metadata')

    assert 'tools.memory' in captured['packages']
    assert 'psutil==7.2.2' in captured['install_requires']
