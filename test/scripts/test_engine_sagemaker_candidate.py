# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind same-dispatch wheels before AWS credentials or notebook execution."""
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parents[2]
SOURCE = DIRECTORY / 'packages/kumo-engine/test/sagemaker_notebook/candidate.py'
SPEC = importlib.util.spec_from_file_location('sagemaker_candidate', SOURCE)
candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(candidate)


@pytest.fixture
def wheel(tmp_path, monkeypatch):
    import platform_candidate

    monkeypatch.setattr(platform_candidate.subprocess, 'check_output',
                        lambda *a, **kw: 'a' * 40 + '\n')
    path = tmp_path / 'kumo_engine-1.0.0.dev0+aaaaaaa-cp311-abi3-manylinux_2_34_x86_64.whl'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('kumo_engine-1.0.0.dev0+aaaaaaa.dist-info/METADATA',
                         'Name: kumo-engine\nVersion: 1.0.0.dev0+aaaaaaa\n')
    return path


def test_current_wheel_produces_runtime_manifest(wheel, tmp_path):
    output = tmp_path / 'runtime/candidate.json'
    result = candidate.prepare(wheel, 'a' * 40, output, 123)
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
        candidate.prepare(wheel, source, output, run)
    assert not output.exists()


def test_workflow_builds_and_downloads_this_dispatch_without_historical_pin():
    import yaml

    workflow = yaml.safe_load(
        (DIRECTORY /
         '.github/workflows/kumo-engine-sagemaker-notebook.yml').read_text())
    jobs = workflow['jobs']
    assert jobs['notebook']['needs'] == ['configuration', 'wheel']
    assert jobs['wheel']['uses'] == './.github/workflows/kumo-engine-pr.yml'
    assert jobs['wheel']['with'] == {'build_only': True}
    shared = yaml.safe_load(
        (DIRECTORY / '.github/workflows/kumo-engine-pr.yml').read_text())
    upload = next(s for s in shared['jobs']['wheel']['steps']
                  if s.get('name') == 'Retain the candidate wheel')
    download = next(s for s in jobs['notebook']['steps']
                    if s.get('name') == "Download this dispatch's wheel")
    assert upload['with']['name'] == download['with'][
        'name'] == 'engine-pr-wheel-${{ github.sha }}'
    assert 'run-id' not in download['with']
    assert 'outputs' not in jobs['configuration']
    identify = next(s for s in jobs['notebook']['steps']
                    if s.get('name') == "Identify this dispatch's wheel")
    assert '--source-ref "$GITHUB_SHA"' in identify['run']


def test_runner_paths_are_initialized_in_steps_not_job_env():
    import yaml

    workflow = yaml.safe_load(
        (DIRECTORY /
         '.github/workflows/kumo-engine-sagemaker-notebook.yml').read_text())
    for job in workflow['jobs'].values():
        assert 'runner.' not in json.dumps(job.get('env', {}))
    identify = next(s for s in workflow['jobs']['notebook']['steps']
                    if s.get('name') == "Identify this dispatch's wheel")
    assert 'export SAGEMAKER_TEST_CANDIDATE="$RUNNER_TEMP/' in identify['run']
    assert '>> "$GITHUB_ENV"' in identify['run']
