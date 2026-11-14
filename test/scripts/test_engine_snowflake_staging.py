# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh dispatch wheels and immutable dependency-cache admission."""
import importlib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'packages/kumo-engine/test/snowflake_notebook'


@pytest.fixture
def staging(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    return importlib.import_module('staging')


def wheel(tmp_path, source='a' * 40, version=None):
    version = version or '1.0.0.dev0+' + source[:7]
    path = tmp_path / (
        f'kumo_engine-{version}-cp311-abi3-manylinux_2_34_x86_64.whl')
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('kumo_engine.dist-info/METADATA',
                         f'Name: kumo-engine\nVersion: {version}\n')
    return path


def test_wheel_metadata_and_source_must_agree(staging, tmp_path):
    path = wheel(tmp_path)
    value = staging.identity(path, 'a' * 40)
    assert value['wheel_sha256'] == staging.launch.sha(path)
    assert value['expected_version'] == '1.0.0.dev0+aaaaaaa'
    with pytest.raises(ValueError, match='source revision'):
        staging.identity(path, 'b' * 40)
    bad = wheel(tmp_path, version='1.0.0.dev0+author')
    with pytest.raises(ValueError, match='source revision'):
        staging.identity(bad, 'a' * 40)


def test_candidate_admission_requires_explicit_manifest_before_cloud(
        staging, monkeypatch):
    import resources
    monkeypatch.delenv('SNOWFLAKE_TEST_CANDIDATE', raising=False)
    with pytest.raises(ValueError, match='must name the staged candidate'):
        resources.admit_candidate()
