# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh dispatch wheels and immutable dependency-cache admission."""
import copy
import importlib
import json
import zipfile
from pathlib import Path

import pytest
import yaml

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


def test_cached_dependencies_require_no_download_or_upload(
        staging, monkeypatch, tmp_path):
    descriptor = json.loads((DIRECTORY / 'dependency-asset.json').read_text())
    descriptor['location'] = (staging.launch.STAGE +
                              '/mlops-validation/payload-cache/' +
                              descriptor['sha256'] +
                              '/wheelhouse.tar/objects/' + 'a' * 12 +
                              '/wheelhouse.tar')
    calls = []
    monkeypatch.setattr(
        staging.launch, 'existing_remote_asset',
        lambda cursor, name, asset: calls.append(name) or dict(
            asset, action='reused'))
    monkeypatch.setattr(
        staging.launch, 'ensure_cached_file',
        lambda *args, **kwargs: pytest.fail('unexpected upload'))
    assert staging.dependencies(object(), descriptor, tmp_path) == descriptor
    assert calls == ['wheelhouse.tar']
    legacy = copy.deepcopy(descriptor)
    legacy['location'] = (staging.launch.STAGE + '/notebook-output-probe/' +
                          'b' * 12 + '/payload/wheelhouse.tar')
    with pytest.raises(ValueError, match='dedicated payload cache'):
        staging.dependencies(object(), legacy, tmp_path)


def test_dispatch_builds_and_downloads_same_run_wheel():
    workflow = yaml.safe_load(
        (ROOT /
         '.github/workflows/kumo-engine-snowflake-notebook.yml').read_text())
    jobs = workflow['jobs']
    build = jobs['wheel']
    assert build['needs'] == 'configuration'
    assert build['uses'] == './.github/workflows/kumo-engine-pr.yml'
    assert build['with'] == {'build_only': True}
    shared = yaml.safe_load(
        (ROOT / '.github/workflows/kumo-engine-pr.yml').read_text())
    for job in shared['jobs'].values():
        for permission, level in job.get('permissions', {}).items():
            assert build['permissions'].get(permission) == level
    wheel = shared['jobs']['wheel']
    assert 'inputs.build_only' in wheel['if']
    assert "!inputs.build_only || github.ref" not in wheel['if']
    assert wheel['steps'][0]['with']['ref'] == '${{ github.sha }}'
    artifact = next(
        step for step in wheel['steps']
        if step.get('name') == 'Retain the candidate wheel')['with']['name']
    for name in ('engine-lifecycle', 'notebook-ui'):
        job = jobs[name]
        assert job['needs'] == ['configuration', 'wheel']
        download = next(step for step in job['steps']
                        if step.get('name') == "Download this dispatch's wheel")
        assert download['with']['name'] == artifact
        assert 'run-id' not in download['with']
        stage = next(step for step in job['steps']
                     if step.get('name') == "Stage this dispatch's candidate")
        assert 'runner.temp' not in json.dumps(job.get('env', {}))
        assert '>> "$GITHUB_ENV"' in stage['run']
        assert 'export SNOWFLAKE_TEST_CANDIDATE=' in stage['run']
        assert '--source-ref "$GITHUB_SHA"' in stage['run']
        assert '--output "$SNOWFLAKE_TEST_CANDIDATE"' in stage['run']
    assert 'platform_candidate.py' not in json.dumps(jobs['configuration'])


@pytest.mark.parametrize('local_key', [False, True])
def test_staging_records_exact_checkout_after_cache_validation(
        staging, monkeypatch, tmp_path, local_key):
    from contextlib import nullcontext

    import resources

    source = 'a' * 40
    path = wheel(tmp_path)
    output = tmp_path / 'candidate.json'
    descriptor = tmp_path / 'dependencies.json'
    archive = dict(name='wheelhouse.tar', sha256='b' * 64)
    descriptor.write_text(json.dumps(archive))
    monkeypatch.setattr(resources, 'require', lambda: {})
    monkeypatch.setattr(staging.subprocess, 'check_output',
                        lambda *args, **kwargs: source)
    monkeypatch.setattr(staging, 'dependencies',
                        lambda *args, **kwargs: archive)
    monkeypatch.setattr(
        staging.launch, 'ensure_cached_file',
        lambda cursor, wheel, digest, work, **kwargs: dict(
            name=wheel.name, sha256=digest, action='reused'))
    connection = type('Connection', (),
                      {'cursor': lambda self: nullcontext(object())})()
    observed_keys = []
    monkeypatch.setattr(
        staging.launch, 'connect',
        lambda key: observed_keys.append(key) or nullcontext(connection))
    inventory = json.loads((DIRECTORY / 'dependencies.json').read_text())
    archive['sha256'] = inventory['archive_sha256']
    descriptor.write_text(json.dumps(archive))
    key = tmp_path / 'cached-key.p8' if local_key else None
    result = staging.stage(path, source, 123, output, descriptor, key_file=key)
    assert observed_keys == [key]
    assert result['build'] == dict(source_commit=source, github_run=123)
    assert json.loads(output.read_text()) == result
    assert 'action' not in result['assets'][path.name]
    assert output.stat().st_mode & 0o777 == 0o600
    local = staging.stage(path, source, None, output, descriptor)
    assert local['build'] == dict(source_commit=source)
    for invalid in (False, True, 0, -1, '123'):
        with pytest.raises(ValueError, match='checkout and run'):
            staging.stage(path, source, invalid, output, descriptor)
    with pytest.raises(ValueError, match='checkout and run'):
        staging.stage(path, 'b' * 40, 123, output, descriptor)


def test_dependency_seed_cannot_reintroduce_legacy_cache_location(
        staging, monkeypatch, tmp_path):
    descriptor = json.loads((DIRECTORY / 'dependency-asset.json').read_text())
    descriptor['location'] = (staging.launch.STAGE +
                              '/mlops-validation/payload-cache/' +
                              descriptor['sha256'] +
                              '/wheelhouse.tar/objects/' + 'a' * 12 +
                              '/wheelhouse.tar')
    adopted = dict(
        descriptor, location=staging.launch.STAGE + '/notebook-output-probe/' +
        'b' * 12 + '/payload/wheelhouse.tar')
    monkeypatch.setattr(staging.launch, 'ensure_cached_file',
                        lambda *a, **kw: adopted)
    with pytest.raises(ValueError, match='unexpected archive'):
        staging.dependencies(object(), descriptor, tmp_path,
                             tmp_path / 'wheelhouse.tar')


def test_cleanup_has_time_and_separate_environment_credentials():
    jobs = yaml.safe_load(
        (ROOT / '.github/workflows/kumo-engine-snowflake-notebook.yml'
         ).read_text())['jobs']
    job = jobs['notebook-ui']
    cleanup = next(
        step for step in job['steps']
        if step.get('name') == "Stop and delete this attempt's notebook")
    assert cleanup['if'] == 'always()'
    assert cleanup['timeout-minutes'] == 15
    assert 'SNOWFLAKE_UI_CREDENTIALS' in cleanup['env']
    assert job['timeout-minutes'] >= 140
    assert '--output-dir "$RUNNER_TEMP/snowflake-graph" --execute' in cleanup[
        'run']


def test_candidate_admission_requires_explicit_manifest_before_cloud(
        staging, monkeypatch):
    import resources
    monkeypatch.delenv('SNOWFLAKE_TEST_CANDIDATE', raising=False)
    with pytest.raises(ValueError, match='must name the staged candidate'):
        resources.admit_candidate()


def test_exact_protected_legacy_dependency_is_reused_without_transfer(
        staging, monkeypatch, tmp_path):
    descriptor = json.loads((DIRECTORY / 'dependency-asset.json').read_text())
    monkeypatch.setattr(
        staging.launch, 'existing_remote_asset',
        lambda cursor, name, asset: dict(asset, action='reused'))
    monkeypatch.setattr(staging.launch, 'ensure_cached_file',
                        lambda *a, **kw: pytest.fail('unexpected upload'))
    assert staging.dependencies(object(), descriptor, tmp_path) == descriptor
