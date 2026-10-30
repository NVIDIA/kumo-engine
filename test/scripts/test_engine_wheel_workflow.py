# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / (
    '.github/workflows/kumo-engine-wheel.yml')
JOBS = yaml.safe_load(WORKFLOW.read_text())['jobs']
BUILDS = JOBS['build']['strategy']['matrix']['include']


def _step(job, name):
    return next(step for step in JOBS[job]['steps'] if step['name'] == name)


def _capture_command(path, capture):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n'
        f'Path({str(capture)!r}).write_text(json.dumps({{'
        "'args': sys.argv[1:], 'target': os.getenv('MACOSX_DEPLOYMENT_TARGET')"
        '}))\n')
    path.chmod(0o755)


@pytest.mark.parametrize('build', BUILDS, ids=lambda build: build['platform'])
@pytest.mark.parametrize('release', [False, True])
def test_build_step_selects_native_platform_and_release_mode(
        tmp_path, build, release):
    capture = tmp_path / 'command.json'
    _capture_command(tmp_path / 'packages/kumo-engine/scripts/build_wheel.sh',
                     capture)
    uname = tmp_path / 'uname'
    uname.write_text(f'#!/bin/sh\necho {build["arch"]}\n')
    uname.chmod(0o755)
    step = _step('build', 'Build wheel')
    subprocess.run(
        ['bash', '-e', '-c', step['run']], cwd=tmp_path,
        env=dict(os.environ, PATH=f'{tmp_path}:{os.environ["PATH"]}',
                 WHEEL_ARCH=build['arch'], WHEEL_PLATFORM=build['platform'],
                 MACOSX_DEPLOYMENT_TARGET=step['env'][
                     'MACOSX_DEPLOYMENT_TARGET'],
                 IS_RELEASE=str(release).lower()), check=True)

    command = json.loads(capture.read_text())
    expected = (['--platform-tag', build['platform']]
                if build['platform'].startswith('manylinux_') else [])
    assert command['args'] == expected + (['--release'] if release else [])
    if build['platform'].startswith('macosx_'):
        assert command['target'] == '14.0'


@pytest.mark.parametrize('job,step_name', [
    ('check-requirements', 'Resolve the wheel and each extra from PyPI'),
    ('kitmaker', 'Publish through Kitmaker'),
])
@pytest.mark.parametrize('build', BUILDS, ids=lambda build: build['platform'])
def test_release_steps_use_the_matching_platform_and_wheel(
        tmp_path, job, step_name, build):
    row = next(row for row in JOBS[job]['strategy']['matrix']['include']
               if row['arch'] == build['arch'])
    step = _step(job, step_name)
    download = _step(job, 'Download wheel')['with']['name']
    upload = _step('build', 'Upload wheel')['with']['name']
    assert download == upload
    if job == 'kitmaker':
        assert row['platform'] == build['platform']

    wheel = tmp_path / 'dist' / f'kumo_engine-1.0-cp311-abi3-{build["platform"]}.whl'
    wheel.parent.mkdir()
    wheel.touch()
    capture = tmp_path / 'command.json'
    _capture_command(tmp_path / 'python3', capture)
    subprocess.run(
        ['bash', '-e', '-c', step['run']], cwd=tmp_path,
        env=dict(os.environ, PATH=f'{tmp_path}:{os.environ["PATH"]}',
                 WHEEL_PLATFORM=row['platform'], RELEASE_TAG='kumo-engine-v1.0',
                 REPO='NVIDIA/kumo-engine',
                 MACOSX_DEPLOYMENT_TARGET=step.get('env', {}).get(
                     'MACOSX_DEPLOYMENT_TARGET', '')), check=True)

    command = json.loads(capture.read_text())
    args = command['args']
    assert args[args.index('--platform') + 1] == row['platform']
    assert str(wheel.relative_to(tmp_path)) in args
    if job == 'check-requirements' and build['platform'].startswith('macosx_'):
        assert command['target'] == '14.0'


def test_all_platforms_require_validation_and_release_approval():
    arches = {build['arch'] for build in BUILDS}
    assert len(arches) == len(BUILDS)
    for job in ['check-requirements', 'kitmaker']:
        assert {row['arch'] for row in JOBS[job]['strategy']['matrix'][
            'include']} == arches
    assert JOBS['check-requirements']['needs'] == ['build']
    assert set(JOBS['attach-to-release']['needs']) == {
        'build', 'check-requirements', 'wheel-smoke'
    }
    assert JOBS['kitmaker']['needs'] == ['attach-to-release']
    assert JOBS['kitmaker']['environment'] == 'kumo-engine-release'
    for job in ['attach-to-release', 'kitmaker']:
        assert JOBS[job]['if'] == "${{ github.event_name == 'release' }}"
    smoke = yaml.safe_load((WORKFLOW.parent /
                            'kumo-engine-wheel-smoke.yml').read_text())
    assert set(smoke[True]) == {'workflow_call'}


def test_publication_forwards_the_shared_prediction_tolerance():
    root = WORKFLOW.parents[2]
    workflow = yaml.safe_load(
        (root / ('.github/workflows/kumo-engine-wheel-smoke.yml')).read_text())
    action = yaml.safe_load(
        (root / ('.github/actions/engine-gpu-smoke/action.yml')).read_text())
    call = next(step['with'] for step in workflow['jobs']['smoke']['steps']
                if step.get('uses') == './.github/actions/engine-gpu-smoke')
    run = next(step for step in action['runs']['steps']
               if step.get('name') == 'Run the eight GPU tasks outside the checkout')
    assert call['reload-atol'] == action['inputs']['reload-atol']['default'] == '0.00001'
    assert '--reload-atol "$RELOAD_ATOL"' in run['run']
