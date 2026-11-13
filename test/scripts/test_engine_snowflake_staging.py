# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh dispatch wheels and immutable dependency-cache admission."""
import importlib
import json
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


def test_staging_resolves_current_wheel_and_records_inventory(
        staging, monkeypatch, tmp_path):
    from contextlib import nullcontext

    import resolve_dependencies
    import resources

    source = 'a' * 40
    path = wheel(tmp_path)
    output = tmp_path / 'candidate.json'
    manifest = dict(archive_sha256='b' * 64, wheels={'new-dependency': {}})
    monkeypatch.setattr(resources, 'require', lambda: {})
    monkeypatch.setattr(staging.subprocess, 'check_output',
                        lambda *a, **kw: source)

    def resolve(candidate, destination, plan):
        assert candidate == path
        destination.mkdir()
        archive = destination / 'wheelhouse.tar'
        archive.write_bytes(b'new candidate dependencies')
        (destination / 'dependencies.json').write_text(json.dumps(manifest))
        return archive

    monkeypatch.setattr(resolve_dependencies, 'resolve', resolve)
    uploads = []

    def upload(cursor, wheel, digest, work, **kwargs):
        uploads.append((wheel.name, digest))
        return dict(name=wheel.name, sha256=digest, action='reused')

    monkeypatch.setattr(staging.launch, 'ensure_cached_file', upload)
    connection = type('Connection', (),
                      {'cursor': lambda self: nullcontext(object())})()
    monkeypatch.setattr(staging.launch, 'connect',
                        lambda key: nullcontext(connection))
    result = staging.stage(path, source, 123, output)
    assert result['build'] == dict(source_commit=source, github_run=123)
    assert result['dependencies'] == manifest
    assert uploads == [('wheelhouse.tar', manifest['archive_sha256']),
                       (path.name, result['candidate']['wheel_sha256'])]
    assert json.loads(output.read_text()) == result
    assert 'action' not in result['assets'][path.name]
    assert output.stat().st_mode & 0o777 == 0o600
    for invalid in (False, True, 0, -1, '123'):
        with pytest.raises(ValueError, match='checkout and run'):
            staging.stage(path, source, invalid, output)
    with pytest.raises(ValueError, match='checkout and run'):
        staging.stage(path, 'b' * 40, 123, output)


def test_candidate_admission_requires_explicit_manifest_before_cloud(
        staging, monkeypatch):
    import resources
    monkeypatch.delenv('SNOWFLAKE_TEST_CANDIDATE', raising=False)
    with pytest.raises(ValueError, match='must name the staged candidate'):
        resources.admit_candidate()
