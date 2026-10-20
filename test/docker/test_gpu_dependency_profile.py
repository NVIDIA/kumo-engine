# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the GPU security profile without changing the CPU dependency set."""

import subprocess
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('profile', ['cpu', 'cu132'])
def test_backend_profile_exports_compatible_cuda_dependencies(profile):
    result = subprocess.run([
        'uv', 'export', '--frozen', '--offline', '--extra', 'full', '--extra',
        profile, '--no-dev', '--no-emit-workspace', '--no-hashes',
        '--no-header', '--no-annotate'
    ], cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    requirements = {
        requirement.name: requirement
        for line in result.stdout.splitlines()
        if line and not line.startswith('#')
        for requirement in [Requirement(line)]
    }
    assert Version(f'2.12.0+{profile}') in requirements['torch'].specifier
    if profile == 'cpu':
        assert 'cuda-toolkit' not in requirements
    else:
        toolkit = requirements['cuda-toolkit']
        assert all(
            Version(pin.version) >= Version('13.1') for pin in toolkit.specifier
            if pin.operator == '==')
        assert any(pin.operator == '==' for pin in toolkit.specifier)
