#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Set the release version.

The release version lives in the root ``VERSION`` file, shared by kumo-sdk,
kumo-engine, the platform, the Studio images and the Helm charts. For a version
of the form ``X.Y.Z``, ``X.Y.ZbN``, ``X.Y.ZrcN`` or ``X.Y.Z.devN`` this writes
``VERSION`` and runs ``scripts/stamp_version.py``, which updates every site
(including ``kumoai_studio/_version.py``). The release workflows then stage the
result with ``git add -u``.

Any other version (an ad hoc temporary build version) only replaces the first
line of ``kumoai_studio/_version.py`` and leaves ``VERSION`` alone. Replacing
only the first line keeps the SPDX header OSRB requires and keeps the
``exec(input())`` readers in the release workflows working -- they read line 1
and nothing else.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
STAMP = REPO_ROOT / 'scripts' / 'stamp_version.py'
VERSION_FILE = REPO_ROOT / 'VERSION'
SDK_VERSION_FILE = Path(
    __file__).resolve().parent.parent / 'kumoai_studio' / '_version.py'


def _version_pattern():
    spec = importlib.util.spec_from_file_location('stamp_version', STAMP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # the script's dataclasses need this
    spec.loader.exec_module(module)
    return module.VERSION_PATTERN


def main() -> int:
    if len(sys.argv) != 2:
        print(f'usage: {Path(sys.argv[0]).name} <version>', file=sys.stderr)
        return 2
    version = sys.argv[1]

    if _version_pattern().fullmatch(version):
        VERSION_FILE.write_text(version + '\n', encoding='utf-8')
        subprocess.check_call([sys.executable, str(STAMP)])
        return 0

    print(
        f'{version!r} is not a release version; only updating '
        f'{SDK_VERSION_FILE.name}', file=sys.stderr)
    lines = SDK_VERSION_FILE.read_text(encoding='utf-8').split('\n')
    lines[0] = f"__version__ = '{version}'"
    SDK_VERSION_FILE.write_text('\n'.join(lines), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
