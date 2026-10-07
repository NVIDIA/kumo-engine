# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""kumo-engine's provider for `kumo_core.version.get_kumo_version`."""

import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import kumo_engine_version
import pytest

PYPROJECT = Path(__file__).parents[1] / 'pyproject.toml'


def test_reports_the_installed_distribution_version(monkeypatch):
    calls = []

    def fake(name):
        calls.append(name)
        return '1.0.0'

    monkeypatch.setattr(kumo_engine_version, 'installed_distribution_version',
                        fake)
    assert kumo_engine_version.installed_version() == '1.0.0'
    assert calls == ['kumo-engine']


def test_defers_when_the_helper_does(monkeypatch):
    # Editable installs and a missing distribution both come back as None.
    monkeypatch.setattr(kumo_engine_version, 'installed_distribution_version',
                        lambda name: None)
    assert kumo_engine_version.installed_version() is None


def test_pyproject_registers_the_provider():
    pyproject = PYPROJECT.read_text()
    assert re.search(
        r'\[project\.entry-points\."kumo_core\.version_providers"\]\n'
        r'kumo-engine = "kumo_engine_version:installed_version"', pyproject)
    assert 'py-modules=["kumo_engine_version"]' in pyproject


def test_loading_the_provider_does_not_import_the_engine():
    # Entry points import their module; that must not drag in `kumo.engine`
    # (and with it torch) just to read a version.
    code = ('import sys, kumo_engine_version; '
            'sys.exit(any(m == "kumo.engine" or m == "torch" '
            'for m in sys.modules))')
    result = subprocess.run([sys.executable, '-c', code])
    assert result.returncode == 0


def test_the_provider_is_a_registered_entry_point_when_installed():
    from importlib.metadata import distribution
    try:
        dist = distribution('kumo-engine')
    except PackageNotFoundError:
        pytest.skip('kumo-engine is not installed')
    points = {
        e.name: e.value
        for e in dist.entry_points if e.group == 'kumo_core.version_providers'
    }
    assert points == {'kumo-engine': 'kumo_engine_version:installed_version'}
