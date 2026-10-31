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

from io_utils.fsutil.api import read_binary, read_text, rm, write_binary

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


def test_checkout_preserves_notice_bytes(tmp_path):
    """Override a Windows-style checkout policy without changing Git config."""
    source = tmp_path / 'notice.txt'
    canonical = b'Copyright example\nLicense text\n'
    subprocess.run(['git', 'init', str(tmp_path)], check=True,
                   capture_output=True)
    write_binary(str(tmp_path / '.gitattributes'),
                 read_binary(str(WORKFLOW.parents[2] / '.gitattributes')))
    write_binary(str(source), canonical)
    subprocess.run(['git', '-c', 'core.autocrlf=false', 'add', 'notice.txt'],
                   cwd=tmp_path, check=True)
    subprocess.run(['git', 'config', '--local', 'core.autocrlf', 'true'],
                   cwd=tmp_path, check=True)
    rm(str(source))
    subprocess.run(['git', 'checkout-index', '--force', 'notice.txt'],
                   cwd=tmp_path, check=True)
    assert read_binary(str(source)) == canonical.replace(b'\n', b'\r\n')

    checkout_env = _step('build', 'Checkout repository')['env']
    rm(str(source))
    subprocess.run(['git', 'checkout-index', '--force', 'notice.txt'],
                   cwd=tmp_path, env=dict(os.environ, **checkout_env),
                   check=True)
    assert read_binary(str(source)) == canonical
    configured = subprocess.check_output(
        ['git', 'config', '--local', 'core.autocrlf'], cwd=tmp_path, text=True)
    assert configured.strip() == 'true'


@pytest.mark.parametrize('build', BUILDS, ids=lambda build: build['platform'])
@pytest.mark.parametrize('release', [False, True])
def test_build_step_selects_native_platform_and_release_mode(
        tmp_path, build, release):
    capture = tmp_path / 'command.json'
    _capture_command(tmp_path / 'packages/kumo-engine/scripts/build_wheel.sh',
                     capture)
    uname = tmp_path / 'uname'
    uname.write_text(f'#!/bin/sh\necho {build["machine"]}\n')
    uname.chmod(0o755)
    step = _step('build', 'Build wheel')
    subprocess.run(
        ['bash', '-e', '-c', step['run']], cwd=tmp_path,
        env=dict(os.environ, PATH=f'{tmp_path}:{os.environ["PATH"]}',
                 WHEEL_ARCH=build['machine'], WHEEL_PLATFORM=build['platform'],
                 MACOSX_DEPLOYMENT_TARGET=step['env'][
                     'MACOSX_DEPLOYMENT_TARGET'],
                 IS_RELEASE=str(release).lower()), check=True)

    command = json.loads(capture.read_text())
    expected = (['--platform-tag', build['platform']]
                if build['platform'].startswith('manylinux_') else [])
    assert command['args'] == expected + (['--release'] if release else [])
    if build['platform'].startswith('macosx_'):
        assert command['target'] == '14.0'


@pytest.mark.parametrize('job,step_name,build', [
    (job, step_name, build)
    for job, step_name in [
        ('check-requirements', 'Resolve the wheel and each extra from PyPI'),
        ('kitmaker', 'Publish through Kitmaker'),
    ]
    for build in BUILDS
    if build['arch'] in {row['arch'] for row in JOBS[job][
        'strategy']['matrix']['include']}
])
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
    assert {row['arch'] for row in JOBS['check-requirements'][
        'strategy']['matrix']['include']} == arches
    assert 'windows-x86_64' in arches
    release_arches = {row['arch'] for row in JOBS['attach-to-release'][
        'strategy']['matrix']['include']}
    assert {'x86_64', 'arm64'} <= release_arches
    assert release_arches <= {'x86_64', 'arm64', 'aarch64'}
    if 'aarch64' in release_arches:
        assert 'platform-qualification' in JOBS['attach-to-release']['needs']
    for job in ['attach-to-release', 'kitmaker']:
        published = {row['arch'] for row in JOBS[job]['strategy']['matrix'][
            'include']}
        assert published == release_arches
        download = _step(job, 'Download wheel')['with']
        assert 'pattern' not in download
        assert download['name'] == _step('build', 'Upload wheel')['with']['name']
    assert JOBS['check-requirements']['needs'] == ['build']
    assert set(JOBS['attach-to-release']['needs']) == {
        'build', 'check-requirements', 'platform-qualification'
    }
    assert JOBS['kitmaker']['needs'] == ['attach-to-release']
    assert JOBS['kitmaker']['environment'] == 'kumo-engine-release'
    assert _step('build', 'Checkout repository')['with'][
        'persist-credentials'] is False
    for job in ['attach-to-release', 'kitmaker']:
        assert JOBS[job]['if'] == "${{ github.event_name == 'release' }}"


def test_cpu_qualification_uses_candidate_artifacts_on_all_pythons():
    caller = JOBS['platform-qualification']
    assert caller['needs'] == ['build']
    assert caller['uses'] == (
        './.github/workflows/kumo-engine-platform-smoke.yml')
    path = WORKFLOW.parent / 'kumo-engine-platform-smoke.yml'
    workflow = yaml.safe_load(read_text(str(path)))
    # PyYAML's YAML 1.1 parser interprets the GitHub Actions `on` key as True.
    assert workflow[True] == {'workflow_call': None}
    job = workflow['jobs']['smoke']
    matrix = job['strategy']['matrix']
    assert set(matrix['python']) == {'3.11', '3.12', '3.13', '3.14'}
    assert len(matrix['target']) == 3
    assert {row['platform']: row['artifact'] for row in matrix['target']} == {
        'windows-native': 'windows-x86_64', 'windows-wsl': 'x86_64',
        'linux-arm64': 'aarch64',
    }
    assert {row['artifact'] for row in matrix['target']} <= {
        build['arch'] for build in BUILDS}
    assert job['env']['CUDA_VISIBLE_DEVICES'] == ''
    download = next(step for step in job['steps']
                    if step.get('uses', '').startswith(
                        'actions/download-artifact@'))
    assert download['with'] == {
        'name': ('kumo-engine-wheel-${{ matrix.target.artifact }}-'
                 '${{ github.sha }}'),
        'path': '${{ runner.temp }}/platform-wheel/artifacts',
    }
    # Supplying another run or token here could accidentally qualify a stale
    # artifact while publishing the current build.
    assert 'run-id' not in download['with']
    assert 'github-token' not in download['with']
