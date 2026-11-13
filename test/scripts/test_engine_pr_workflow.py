# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / (
    '.github/workflows/kumo-engine-pr.yml')
JOBS = yaml.safe_load(WORKFLOW.read_text())['jobs']
HEAD = 'a' * 40


def _run(job, **env):
    assert job in JOBS, f'The {job} policy job is missing'
    script = next(step['run'] for step in JOBS[job]['steps'] if 'run' in step)
    return subprocess.run(['bash', '-e', '-c', script], capture_output=True,
                          text=True, env=dict(os.environ, **env))


@pytest.mark.parametrize(
    'files,head,count,expected',
    [
        ([{
            'filename': 'docs/check.md'
        }], HEAD, 1, 'false'),
        ([{
            'filename': 'ui/src/App.tsx'
        }], HEAD, 1, 'false'),
        # A docs-only follow-up push still tests code changed earlier in the PR.
        ([{
            'filename': 'docs/check.md'
        }, {
            'filename': 'packages/kumo-engine/a.py'
        }], HEAD, 2, 'true'),
        ([{
            'filename': 'docs/check.md',
            'previous_filename': 'packages/kumo-engine/a.py'
        }], HEAD, 1, 'true'),
        ([{
            'filename': 'docs/check.md'
        }], 'b' * 40, 1, 'true'),
        ([{
            'filename': 'docs/check.md'
        }], HEAD, 3001, 'true'),
    ])
def test_gpu_selection_uses_the_complete_pr_and_rename_source(
        tmp_path, files, head, count, expected):
    gh = tmp_path / 'gh'
    gh.write_text('''#!/bin/bash
set -euo pipefail
case "$*" in
  *'/files'*) data="$PR_FILES" ;;
  *) data="$PR_METADATA" ;;
esac
printf '%s\\n' "$data" | jq -r "${!#}"
''')
    gh.chmod(0o755)
    output = tmp_path / 'output'
    result = _run(
        'changes', PATH=f'{tmp_path}:{os.environ["PATH"]}',
        GITHUB_REPOSITORY='NVIDIA/kumo-engine',
        GITHUB_REF_NAME='pull-request/135', GITHUB_SHA=HEAD,
        GITHUB_OUTPUT=str(output), PR_FILES=json.dumps(files),
        PR_METADATA=json.dumps(dict(head=dict(sha=head), changed_files=count)))
    assert result.returncode == 0, result.stderr
    assert output.read_text() == f'gpu={expected}\n'


def test_main_seeds_caches_without_querying_a_pr(tmp_path):
    output = tmp_path / 'output'
    result = _run('changes', GITHUB_REF_NAME='main',
                  GITHUB_OUTPUT=str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_text() == 'gpu=false\n'


def test_wheel_step_propagates_one_target_directory_and_exact_build_tools(tmp_path):
    import sys

    calls = tmp_path / 'calls.jsonl'
    executable = '''#!{python}
import json, os, subprocess, sys
from pathlib import Path
with Path(os.environ['CAPTURE']).open('a') as stream:
    stream.write(json.dumps(dict(command=Path(sys.argv[0]).name,
        args=sys.argv[1:], target=os.environ.get('CARGO_TARGET_DIR'),
        constraints=os.environ.get('PIP_BUILD_CONSTRAINT'))) + '\\n')
if Path(sys.argv[0]).name == 'build_wheel.sh':
    for variable, arguments in [
        ('CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER', ['-shared', 'input.o']),
        ('CC_x86_64_unknown_linux_gnu', ['-c', 'input.c']),
        ('CXX_x86_64_unknown_linux_gnu', ['-c', 'input.cc']),
        ('RANLIB_x86_64_unknown_linux_gnu', ['libinput.a']),
    ]:
        subprocess.run([os.environ[variable], *arguments], check=True)
'''.format(python=sys.executable)
    for name in ['python', 'cargo-zigbuild', 'packages/kumo-engine/scripts/build_wheel.sh']:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(executable)
        path.chmod(0o755)
    (tmp_path / 'temp').mkdir()
    job = JOBS['wheel']
    step = next(s for s in job['steps'] if s.get('name') == 'Build the candidate wheel')
    env = {k: v.replace('${{ github.workspace }}', str(tmp_path)).replace(
        '${{ runner.temp }}', str(tmp_path / 'temp'))
           for k, v in job['env'].items()}
    result = subprocess.run(['bash', '-e', '-c', step['run']], cwd=tmp_path,
                            capture_output=True, text=True,
                            env=dict(os.environ, **env, GITHUB_WORKSPACE=str(tmp_path),
                                     RUNNER_TEMP=str(tmp_path / 'temp'),
                                     PATH=f'{tmp_path}:{os.environ["PATH"]}', CAPTURE=str(calls)))
    assert result.returncode == 0, result.stderr
    recorded = [json.loads(v) for v in calls.read_text().splitlines()]
    expected = str(tmp_path / 'packages/diskgraph/target')
    assert {v['target'] for v in recorded} == {expected}
    tools = recorded[0]['args']
    assert 'ziglang==0.17.0' in tools and 'cargo-zigbuild==0.24.0' in tools
    assert Path(recorded[-1]['constraints']).read_text().strip() == 'ziglang==0.17.0'
    compiler_calls = [row['args'] for row in recorded if row['command'] == 'cargo-zigbuild']
    assert compiler_calls == [
        ['zig', 'cc', '--', '-g', '-fno-sanitize=all', '-target', 'x86_64-linux-gnu.2.17',
         '-mcpu=x86_64+aes+sse2', '-shared', 'input.o'],
        ['zig', 'cc', '--', '-g', '-fno-sanitize=all', '-target', 'x86_64-linux-gnu.2.17',
         '-mcpu=x86_64+aes+sse2', '-c', 'input.c'],
        ['zig', 'c++', '--', '-g', '-fno-sanitize=all', '-target', 'x86_64-linux-gnu.2.17',
         '-mcpu=x86_64+aes+sse2', '-c', 'input.cc'],
        ['zig', 'ranlib', '--', 'libinput.a'],
    ]


@pytest.mark.parametrize('event,changes,required,wheel,lifecycle,gpu,expected', [
    ('push', 'success', 'false', 'skipped', 'skipped', 'skipped', 0),
    ('push', 'success', 'true', 'success', 'success', 'success', 0),
    ('push', 'failure', 'false', 'skipped', 'skipped', 'skipped', 1),
    ('push', 'cancelled', '', 'skipped', 'skipped', 'skipped', 1),
    ('push', 'success', '', 'skipped', 'skipped', 'skipped', 1),
    ('push', 'success', 'true', 'failure', 'skipped', 'skipped', 1),
    ('push', 'success', 'true', 'success', 'success', 'failure', 1),
    ('push', 'success', 'true', 'success', 'failure', 'success', 1),
    ('push', 'success', 'true', 'success', 'skipped', 'success', 1),
    ('workflow_dispatch', 'skipped', '', 'success', 'success', 'success', 0),
    ('workflow_dispatch', 'skipped', '', 'success', 'failure', 'success', 1),
    ('workflow_dispatch', 'skipped', '', 'skipped', 'skipped', 'success', 1),
    ('pull_request', 'success', 'false', 'skipped', 'skipped', 'skipped', 1),
])
def test_required_check_only_accepts_a_pr_exemption_or_all_successful_jobs(
        tmp_path, event, changes, required, wheel, lifecycle, gpu, expected):
    result = _run('required', EVENT_NAME=event, CHANGES_RESULT=changes,
                  GPU_REQUIRED=required, WHEEL_RESULT=wheel,
                  LIFECYCLE_RESULT=lifecycle, GPU_RESULT=gpu)
    assert result.returncode == expected, result.stdout + result.stderr
