# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

kumo_api_release_ver = '3.0.0'
api_pyproject = Path(__file__).parent.parent / 'kumo-api' / 'pyproject.toml'
api_ver = kumo_api_release_ver
if api_pyproject.exists():
    match = re.search(
        r'^\[project\]\s*$.*?^version\s*=\s*[\'\"]([^\'\"]+)[\'\"]',
        api_pyproject.read_text(encoding='utf-8'),
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f'No [project].version found in {api_pyproject}')
    api_ver = match.group(1)

kumo_api = f'kumo-api=={api_ver}'


def _build_commit():
    commit = os.environ.get('KUMO_ENGINE_BUILD_COMMIT')
    if commit:
        return commit
    command = ['git', '-C', str(Path(__file__).parent), 'rev-parse',
               '--short=12', 'HEAD']
    try:
        return subprocess.run(command, capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class _StampBuildCommit(build_py):
    def run(self):
        super().run()
        target = Path(self.build_lib) / 'kumo' / 'engine' / '_version.py'
        if target.is_file():
            text = target.read_text(encoding='utf-8')
            target.unlink()
            target.write_text(
                f'{text}BUILD_COMMIT = {_build_commit()!r}\n',
                encoding='utf-8')


setup(
    cmdclass={'build_py': _StampBuildCommit},
    install_requires=[
        "pandas",
        "pyarrow>=8.0.0",
        "typing_extensions>=4.5.0",  # for override
        kumo_api,
        "pydantic>=1.10.21",
        # Guards Graph.snapshot()'s check-then-build DiskGraph ingest
        # against concurrent builders (see kumo/engine/graph.py). filelock.
        # FileLock is a real OS-level advisory lock (flock/LockFileEx), so
        # it's released by the kernel even if the process crashes mid-build
        # -- unlike its own SoftFileLock fallback, deliberately not used
        # here.
        "filelock>=3.0.0",
        # Unlike kumo-sdk (where this is an optional extra behind
        # CloudBackend), this package has no non-training use case, so
        # kumo-core -- and the Torch/PyG/DiskGraph stack it pulls in -- is a
        # plain, required dependency here.
        "kumo-core>=0.1.0.dev1",
        # Select one Faiss distribution: both provide the `faiss` module.
        # Keep platform markers on these external dependencies so wheel
        # merging preserves them when the bundled core reference is removed.
        'faiss-gpu==1.14.3; sys_platform == "linux"',
        'faiss-cpu==1.14.3; sys_platform == "darwin"',
    ], )
