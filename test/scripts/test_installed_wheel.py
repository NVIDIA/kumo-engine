# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import runpy
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[2] /
          'packages/kumo-engine/scripts/verify_installed_wheel.py')
HELPER = runpy.run_path(str(SCRIPT))


def test_checkout_origin_rejected(tmp_path):
    with pytest.raises(ValueError, match='outside'):
        HELPER['check_origin'](str(tmp_path / 'checkout/module.py'),
                               tmp_path / 'venv')


def test_installed_origin_accepted(tmp_path):
    origin = tmp_path / 'venv/site-packages/module.py'
    assert HELPER['check_origin'](str(origin), tmp_path / 'venv') == str(origin)


@pytest.mark.parametrize('expected,system,machine,release', [
    ('windows-native', 'Windows', 'AMD64', '2022'),
    ('windows-wsl', 'Linux', 'x86_64', '6.6-microsoft-standard-WSL2'),
    ('linux-arm64', 'Linux', 'aarch64', '6.8'),
])
def test_platform_accepted(expected, system, machine, release):
    HELPER['check_platform'](expected, system, machine, release)


@pytest.mark.parametrize('system,machine,release', [
    ('Windows', 'AMD64', '2022'),
    ('Linux', 'x86_64', '6.8'),
    ('Linux', 'aarch64', '6.6-microsoft-standard-WSL2'),
])
def test_wsl_requires_linux_guest(system, machine, release):
    with pytest.raises(ValueError, match='Unexpected runtime'):
        HELPER['check_platform']('windows-wsl', system, machine, release)


def test_install_matches_artifact(tmp_path):
    wheel = tmp_path / 'engine+dev.whl'
    source = json.dumps({'url': wheel.as_uri(), 'archive_info': {}})
    HELPER['check_install'](source, wheel)
    HELPER['check_install'](source.replace('%2B', '+'), wheel)
    with pytest.raises(ValueError, match='differs'):
        HELPER['check_install'](source, tmp_path / 'other.whl')


def test_install_requires_provenance(tmp_path):
    with pytest.raises(ValueError, match='provenance'):
        HELPER['check_install'](None, tmp_path / 'engine.whl')


def test_install_rejects_remote_artifact(tmp_path):
    with pytest.raises(ValueError, match='not from a local'):
        HELPER['check_install']('{"url":"https://example.com/engine.whl"}',
                                tmp_path / 'engine.whl')
