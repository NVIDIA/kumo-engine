# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise weekly runner admission without allocating infrastructure."""

import json
import os
import re
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
    return dict(
        manifest_sha256='1' * 64, reference_sha256='2' * 64, profiles=dict(
            rtx=dict(profile), scale=dict(
                profile, auxiliary={
                    check:
                    dict(
                        runner=dict(path=f'/frozen/{check}/weekly_aux.py',
                                    sha256='3' * 64),
                        template=dict(path=f'/frozen/{check}/template.json',
                                      sha256='4' * 64))
                    for check in ('P1', 'B9')
                })))


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


@pytest.mark.parametrize('change', [
    'missing_runner', 'bad_hash', 'relative_root', 'missing_scale',
    'missing_auxiliary', 'missing_b9', 'bad_auxiliary_hash',
    'relative_auxiliary'
])
def test_workflow_refuses_unbound_or_malformed_existing_capacity(
        tmp_path, change):
    value = configuration()
    if change == 'missing_runner':
        value['profiles']['rtx']['runner'] = []
    elif change == 'bad_hash':
        value['manifest_sha256'] = 'latest'
    elif change == 'relative_root':
        value['profiles']['rtx']['root'] = '../elsewhere'
    elif change == 'missing_scale':
        del value['profiles']['scale']
    elif change == 'missing_auxiliary':
        del value['profiles']['scale']['auxiliary']
    elif change == 'missing_b9':
        del value['profiles']['scale']['auxiliary']['B9']
    elif change == 'bad_auxiliary_hash':
        value['profiles']['scale']['auxiliary']['B9']['template'][
            'sha256'] = 'latest'
    else:
        value['profiles']['scale']['auxiliary']['P1']['runner'][
            'path'] = './weekly_aux.py'
    result, _ = configure(tmp_path, value, 'candidate')
    assert result.returncode != 0


def test_candidate_workflow_requires_a_pinned_complete_reference(tmp_path):
    value = configuration()
    del value['reference_sha256']
    result, _ = configure(tmp_path, value, 'candidate')
    assert result.returncode != 0


def test_both_native_shards_admit_all_bindings_before_any_model_fit():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    assert 'admit' in workflow['jobs']['gpu']['needs']
    admission = workflow['jobs']['admit']
    assert admission['strategy'][
        'matrix'] == '${{ fromJSON(needs.configure.outputs.matrix) }}'
    scripts = '\n'.join(step.get('run', '') for step in admission['steps'])
    assert 'admit_bindings' in scripts
    assert 'verify_case_inputs' in scripts
    assert "'run_case'" not in scripts


def test_scale_runs_bound_p1_then_p2_then_retained_b9_and_collects_both():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow['jobs']['gpu']['steps']
    names = [step.get('name') for step in steps]
    p1 = names.index('Run five bound CPU sampling operations')
    fit = names.index('Run each selected case exactly once')
    b9 = names.index('Run retained P2 prediction on one then two matching GPUs')
    assert p1 < fit < b9
    for index in (p1, b9):
        assert steps[index]['if'] == "matrix.profile == 'scale'"
        assert "binder['bind']" in steps[index]['run']
    assert 'retained_case=' in steps[b9]['run']
    compare = '\n'.join(
        step.get('run', '') for step in workflow['jobs']['compare']['steps'])
    assert 'ancillary_records=records' in compare
    assert "bundle['weekly_milestone_complete']" in compare
    assert 'workers are not bound or verified' not in compare


def test_embedded_python_programs_compile_without_running_hosts():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    programs = 0
    for job in workflow['jobs'].values():
        for step in job['steps']:
            for source in re.findall(r"python(?:3)? - <<'PY'\n(.*?)\nPY",
                                     step.get('run', ''), re.S):
                compile(source, step.get('name', 'workflow'), 'exec')
                programs += 1
    assert programs >= 8
