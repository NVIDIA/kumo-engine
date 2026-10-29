# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise weekly runner admission without allocating infrastructure."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(
    __file__).resolve().parents[2] / '.github/workflows/kumo-engine-weekly.yml'


def configuration():
    profile = dict(runner=['self-hosted', 'linux', 'x64', 'weekly-profile'],
                   root='/scratch/weekly-ci', manifest='/frozen/manifest.json',
                   dependency_profile='/frozen/dependencies.json',
                   reference_wheel='/frozen/reference.whl',
                   reference_root='/durable/reference',
                   host_lock='/scratch/gpu-study.lock')
    return dict(manifest_sha256='1' * 64, reference_sha256='2' * 64,
                profiles=dict(rtx=dict(profile), scale=dict(profile)))


def configure(tmp_path, config, mode):
    workflow = yaml.safe_load(WORKFLOW.read_text())
    step = next(row for row in workflow['jobs']['configure']['steps']
                if row.get('id') == 'config')
    output = tmp_path / 'output'
    env = dict(os.environ, WEEKLY_CONFIG=json.dumps(config),
               REQUESTED_MODE=mode, GITHUB_EVENT_NAME='workflow_dispatch',
               GITHUB_OUTPUT=str(output))
    completed = subprocess.run(['bash', '-euo', 'pipefail', '-c', step['run']],
                               env=env, text=True, capture_output=True,
                               check=False)
    fields = dict(
        line.split('=', 1)
        for line in output.read_text().splitlines()) if output.exists() else {}
    return completed, fields


def test_workflow_selects_two_existing_profiles_and_does_not_launch_candidate_in_reference_mode(
        tmp_path):
    result, output = configure(tmp_path, configuration(), 'reference')
    assert result.returncode == 0, result.stderr
    assert output['mode'] == 'reference'
    matrix = json.loads(output['matrix'])
    assert [row['profile'] for row in matrix['include']] == ['rtx', 'scale']
    workflow = yaml.safe_load(WORKFLOW.read_text())
    assert workflow['jobs']['build'][
        'if'] == "needs.configure.outputs.mode == 'candidate'"
    assert workflow['concurrency']['cancel-in-progress'] is False


@pytest.mark.parametrize(
    'change', ['missing_runner', 'bad_hash', 'relative_root', 'missing_scale'])
def test_workflow_refuses_unbound_or_malformed_existing_capacity(
        tmp_path, change):
    value = configuration()
    if change == 'missing_runner':
        value['profiles']['rtx']['runner'] = []
    elif change == 'bad_hash':
        value['manifest_sha256'] = 'latest'
    elif change == 'relative_root':
        value['profiles']['rtx']['root'] = '../elsewhere'
    else:
        del value['profiles']['scale']
    result, _ = configure(tmp_path, value, 'candidate')
    assert result.returncode != 0


def test_candidate_workflow_requires_a_pinned_complete_reference(tmp_path):
    value = configuration()
    del value['reference_sha256']
    result, _ = configure(tmp_path, value, 'candidate')
    assert result.returncode != 0
