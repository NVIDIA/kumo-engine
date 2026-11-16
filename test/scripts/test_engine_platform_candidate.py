# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve(
).parents[2] / 'packages/kumo-engine/test/platform_candidate.py'
spec = importlib.util.spec_from_file_location('platform_candidate', SOURCE)
admission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(admission)


def test_candidate_rejects_product_changes_but_allows_harness_edits(tmp_path):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=tmp_path,
                                       text=True).strip()

    git('init', '-q')
    git('config', 'user.email', 'test@example.invalid')
    git('config', 'user.name', 'Test')
    product = tmp_path / 'packages/kumo-engine/kumo/engine'
    tests = tmp_path / 'packages/kumo-engine/test'
    product.mkdir(parents=True)
    tests.mkdir()
    (product / 'model.py').write_text('original')
    (tests / 'harness.py').write_text('original')
    git('add', '.')
    git('-c', 'commit.gpgsign=false', 'commit', '-qm', 'base')
    source = git('rev-parse', 'HEAD')
    candidate = dict(build=dict(source_commit=source))
    (tests / 'harness.py').write_text('updated assertions')
    git('add', '.')
    git('-c', 'commit.gpgsign=false', 'commit', '-qm', 'harness')
    assert admission.verify(tmp_path, candidate) == source
    (product / 'model.py').write_text('changed product')
    git('add', '.')
    git('-c', 'commit.gpgsign=false', 'commit', '-qm', 'product')
    with pytest.raises(ValueError, match='does not match'):
        admission.verify(tmp_path, candidate)
    with pytest.raises(ValueError, match='lacks committed build provenance'):
        admission.verify(tmp_path, {})


def test_partial_component_provenance_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='lacks committed build provenance'):
        admission.verify(
            tmp_path,
            dict(build=dict(engine_source_commit='a' *
                            40, provider_source_commit='a' *
                            40, native_payload='historical base')))


def test_artifact_outputs_bind_build_and_safe_filename():
    filename = 'kumo_engine-1.0.0+abc-cp311-abi3-manylinux_2_34_x86_64.whl'
    candidate = dict(build=dict(source_commit='a' * 40, github_run=123),
                     candidate=dict(wheel_file=filename))
    outputs = admission.artifact_outputs(candidate)
    assert outputs == dict(run_id='123',
                           artifact_name='engine-pr-wheel-' + 'a' * 40,
                           wheel_file=candidate['candidate']['wheel_file'])
    for invalid in (True, 0, -1, '123\nrun_id=456'):
        candidate['build']['github_run'] = invalid
        with pytest.raises(ValueError):
            admission.artifact_outputs(candidate)
    candidate['build']['github_run'] = 123
    for invalid in ('../candidate.whl', 'kumo_engine-a.whl\nrun_id=456',
                    '$(command).whl', None):
        candidate['candidate']['wheel_file'] = invalid
        with pytest.raises(ValueError):
            admission.artifact_outputs(candidate)


@pytest.mark.parametrize('path', [
    'LICENSE', 'THIRD-PARTY.txt', '3rdparty/notices/inventory.json',
    'agent-skills/skills/engine/SKILL.md',
    'agent-skills/scripts/package_skill.py'
])
def test_candidate_rejects_changed_wheel_bundle_inputs(tmp_path, path):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=tmp_path,
                                       text=True).strip()

    git('init', '-q')
    git('config', 'user.email', 'test@example.invalid')
    git('config', 'user.name', 'Test')
    artifact = tmp_path / path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('original')
    git('add', '.')
    git('-c', 'commit.gpgsign=false', 'commit', '-qm', 'base')
    source = git('rev-parse', 'HEAD')
    artifact.write_text('changed')
    git('add', '.')
    git('-c', 'commit.gpgsign=false', 'commit', '-qm', 'wheel input')
    with pytest.raises(ValueError, match='does not match'):
        admission.verify(tmp_path, dict(build=dict(source_commit=source)))


@pytest.fixture
def wheel(tmp_path, monkeypatch):
    monkeypatch.setattr(admission.subprocess, 'check_output',
                        lambda *a, **kw: 'a' * 40 + '\n')
    path = tmp_path / 'kumo_engine-1.0.0.dev0+aaaaaaa-cp311-abi3-manylinux_2_34_x86_64.whl'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('kumo_engine-1.0.0.dev0+aaaaaaa.dist-info/METADATA',
                         'Name: kumo-engine\nVersion: 1.0.0.dev0+aaaaaaa\n')
    return path


def test_current_wheel_produces_runtime_manifest(wheel, tmp_path):
    output = tmp_path / 'runtime/candidate.json'
    result = admission.prepare(wheel, 'a' * 40, output, 123)
    assert result == json.loads(output.read_text())
    assert result['build'] == dict(source_commit='a' * 40, github_run=123)
    assert result['candidate']['wheel_file'] == wheel.name
    assert len(result['candidate']['wheel_sha256']) == 64
    assert output.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('change', ['source', 'metadata', 'symlink', 'run'])
def test_reject_substituted_wheel_before_writing_manifest(
        wheel, tmp_path, change):
    source, run = 'a' * 40, 123
    if change == 'source':
        source = 'b' * 40
    elif change == 'metadata':
        with zipfile.ZipFile(wheel, 'w') as archive:
            archive.writestr(
                'x.dist-info/METADATA',
                'Name: kumo-engine\nVersion: 1.0.0.dev0+bbbbbbb\n')
    elif change == 'symlink':
        target = wheel.with_suffix('.target')
        wheel.rename(target)
        wheel.symlink_to(target)
    else:
        run = True
    output = tmp_path / 'candidate.json'
    with pytest.raises(ValueError):
        admission.prepare(wheel, source, output, run)
    assert not output.exists()
