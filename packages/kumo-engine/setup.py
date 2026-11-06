# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
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

# The agent skills ship inside this wheel, one package per host, so
# `kumo.engine.install_skills()` installs the skill written for this exact
# SDK. A build from a checkout regenerates them from agent-skills/; a build
# from an sdist has no agent-skills/ and keeps the copy the sdist carries.
package_skill_py = (Path(__file__).resolve().parents[2] / 'agent-skills' /
                    'scripts' / 'package_skill.py')
if package_skill_py.exists():
    spec = importlib.util.spec_from_file_location('package_skill',
                                                  package_skill_py)
    assert spec and spec.loader
    package_skill = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package_skill)
    package_skill.bundle(Path(__file__).parent / 'kumo' / 'engine' / '_skills')


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
        # here. 3.20.4 through 3.29.4 unlink the lock file on release, so
        # two processes can each hold a lock on a different inode at one
        # path; 3.29.5 keeps the file.
        "filelock>=3.29.5",
        # Unlike kumo-sdk (where this is an optional extra behind
        # CloudBackend), this package has no non-training use case, so
        # kumo-core -- and the Torch/PyG/DiskGraph stack it pulls in -- is a
        # plain, required dependency here.
        "kumo-core>=0.1.0.dev1",
        # Select one Faiss distribution: both provide the `faiss` module.
        # Keep platform markers on these external dependencies so wheel
        # merging preserves them when the bundled core reference is removed.
        'faiss-gpu==1.14.3; sys_platform == "linux" '
        'and platform_machine == "x86_64"',
        'faiss-cpu==1.14.3; sys_platform == "darwin"',
    ], )
