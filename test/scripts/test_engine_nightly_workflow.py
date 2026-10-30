# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the actual workflow guards without a GPU or model."""

import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(
    __file__).parents[2] / '.github/workflows/kumo-engine-nightly.yml'


def jobs():
    return yaml.safe_load(WORKFLOW.read_text())['jobs']


def step(name):
    return next(item['run'] for job in jobs().values() for item in job['steps']
                if item.get('name') == name)


def test_one_rtx_job_owns_build_install_all_tests_comparison_and_upload():
    [job] = jobs().values()
    assert job['runs-on'] == '${{ vars.KUMO_ENGINE_NIGHTLY_RTX_RUNNER }}'
    assert 'needs' not in job and 'matrix' not in job.get('strategy', {})
    assert str(job['env']['CUDA_VISIBLE_DEVICES']) == '0'
    assert 'SHARD' not in job['env']
    steps = job['steps']
    assert not any('download-artifact@' in item.get('uses', '')
                   for item in steps)
    build = next(i for i, item in enumerate(steps) if item.get('name') ==
                 'Build the exact product tree with the existing builder')
    install = next(i for i, item in enumerate(steps)
                   if item.get('uses') == './.github/actions/engine-gpu-smoke')
    assert steps[install]['with']['install-only'] == 'true'
    run = next(i for i, item in enumerate(steps)
               if '--behavior' in item.get('run', ''))
    assert '--shard' not in steps[run]['run']
    assert '--behavior' in steps[run]['run']
    compare = next(
        i for i, item in enumerate(steps)
        if item.get('name') == 'Validate the complete candidate comparison')
    pack = next(i for i, item in enumerate(steps) if item.get('name') ==
                'Retain results and failures without full input indexes')
    upload = next(i for i, item in enumerate(steps)
                  if 'upload-artifact@' in item.get('uses', '') and i > compare)
    assert build < install < run < pack < compare < upload
    assert steps[pack].get('if') == 'always()'
    assert steps[upload].get('if') == 'always()'


@pytest.mark.parametrize('exit_code', [0, 23])
def test_actual_final_comparison_exit_controls_the_gpu_job(tmp_path, exit_code):
    job = next(
        job for job in jobs().values()
        if job['runs-on'] == '${{ vars.KUMO_ENGINE_NIGHTLY_RTX_RUNNER }}')
    compare = next(
        (item for item in job['steps']
         if item.get('name') == 'Validate the complete candidate comparison'),
        None)
    assert compare is not None, 'The final comparison must run in the same GPU job'
    assert not compare.get('continue-on-error', False)
    source = tmp_path / 'nightly-result'
    source.mkdir()
    (source / 'proof.txt').write_text('retained original bytes')
    env = dict(os.environ, RUNNER_TEMP=str(tmp_path),
               NIGHTLY_ROOT=str(tmp_path))
    env.pop('SHARD', None)
    packed = subprocess.run([
        'bash', '-euo', 'pipefail', '-c',
        step('Retain results and failures without full input indexes')
    ], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert packed.returncode == 0, packed.stderr
    binary = tmp_path / 'engine-smoke-venv/bin'
    binary.mkdir(parents=True)
    python = binary / 'python'
    python.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_FILE"\nexit "$COMPARE_EXIT"\n'
    )
    python.chmod(0o755)
    (binary / 'python3').symlink_to(sys.executable)
    calls, marker = tmp_path / 'calls.txt', tmp_path / 'comparison-finished'
    env = dict(os.environ, RUNNER_TEMP=str(tmp_path),
               NIGHTLY_ROOT=str(tmp_path), INPUT_SHA256='a' * 64,
               REFERENCE_SHA256='b' * 64,
               PATH=f'{binary}{os.pathsep}{os.environ["PATH"]}',
               CALL_FILE=str(calls), COMPARE_EXIT=str(exit_code),
               MARKER=str(marker))
    result = subprocess.run([
        'bash', '-euo', 'pipefail', '-c', compare['run'] + '\ntouch "$MARKER"'
    ], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert calls.exists(), result.stderr
    assert any(path.read_text() == 'retained original bytes'
               for path in tmp_path.rglob('proof.txt')
               if path != source / 'proof.txt')
    assert 'compare' in shlex.split(calls.read_text().splitlines()[-1])
    assert result.returncode == exit_code
    assert marker.exists() is (exit_code == 0)


def test_packed_behavior_roundtrips_without_shortcuts_or_native_work(tmp_path):
    output = tmp_path / 'nightly-result'
    child = output / 'behavior/pytest-temp/reuse0'
    child.mkdir(parents=True)
    (child / 'child-result.json').write_text('original child proof')
    (child.parent / 'reusecurrent').symlink_to(child, target_is_directory=True)
    (child / 'input.parquet').write_bytes(b'original table bytes')
    (child / 'staged.parquet').symlink_to(child / 'input.parquet')
    work = output / 'work/native'
    work.mkdir(parents=True)
    (work / 'index.bin').write_bytes(b'keep on host')
    host = tmp_path / 'host'
    host.mkdir()
    env = dict(os.environ, RUNNER_TEMP=str(tmp_path), NIGHTLY_ROOT=str(host))
    env.pop('SHARD', None)
    command = step('Retain results and failures without full input indexes')
    result = subprocess.run(['bash', '-euo', 'pipefail', '-c', command],
                            env=env, cwd=tmp_path, capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr
    extracted = tmp_path / 'extracted'
    [packed] = tmp_path.glob('nightly-*.tar.gz')
    with tarfile.open(packed) as archive:
        archive.extractall(extracted, filter='data')
    retained = extracted / child.relative_to(output)
    assert (retained /
            'child-result.json').read_text() == 'original child proof'
    assert (retained / 'staged.parquet').read_bytes() == b'original table bytes'
    assert not (extracted / 'work').exists()
    assert not any(path.is_symlink() for path in extracted.rglob('*'))


def test_failed_gpu_query_stops_before_model_work(tmp_path):
    binary = tmp_path / 'bin'
    binary.mkdir()
    probe = binary / 'nvidia-smi'
    probe.write_text('#!/bin/sh\nexit 1\n')
    probe.chmod(0o755)
    command = next(item['run'] for job in jobs().values()
                   for item in job['steps']
                   if '--behavior' in item.get('run', ''))
    query = command.split('flock 9\n',
                          1)[1].split('test "$(git rev-parse HEAD)"', 1)[0]
    marker = tmp_path / 'model-work-started'
    env = dict(os.environ, PATH=f'{binary}{os.pathsep}{os.environ["PATH"]}',
               MARKER=str(marker))
    result = subprocess.run(
        ['bash', '-c', 'set -euo pipefail\n' + query + '\ntouch "$MARKER"'],
        env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert not marker.exists()
