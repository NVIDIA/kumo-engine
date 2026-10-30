# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the actual workflow guards without a GPU or model."""

import os
import subprocess
import tarfile
from pathlib import Path

import yaml

WORKFLOW = Path(
    __file__).parents[2] / '.github/workflows/kumo-engine-nightly.yml'


def step(name):
    steps = yaml.safe_load(WORKFLOW.read_text())['jobs']['gpu']['steps']
    return next(item['run'] for item in steps if item.get('name') == name)


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
    env = dict(os.environ, RUNNER_TEMP=str(tmp_path), NIGHTLY_ROOT=str(host),
               SHARD='node')
    command = step('Retain results and failures without full input indexes')
    result = subprocess.run(['bash', '-euo', 'pipefail', '-c', command],
                            env=env, cwd=tmp_path, capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr
    extracted = tmp_path / 'extracted'
    with tarfile.open(tmp_path / 'nightly-node.tar.gz') as archive:
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
    command = step('Run eight primary fits under the existing GPU lock')
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
