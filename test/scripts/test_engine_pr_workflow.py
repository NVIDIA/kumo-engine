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


@pytest.mark.parametrize('event,changes,required,wheel,gpu,expected', [
    ('push', 'success', 'false', 'skipped', 'skipped', 0),
    ('push', 'success', 'true', 'success', 'success', 0),
    ('push', 'failure', 'false', 'skipped', 'skipped', 1),
    ('push', 'cancelled', '', 'skipped', 'skipped', 1),
    ('push', 'success', '', 'skipped', 'skipped', 1),
    ('push', 'success', 'true', 'failure', 'skipped', 1),
    ('push', 'success', 'true', 'success', 'failure', 1),
    ('workflow_dispatch', 'skipped', '', 'success', 'success', 0),
    ('workflow_dispatch', 'skipped', '', 'skipped', 'success', 1),
    ('pull_request', 'success', 'false', 'skipped', 'skipped', 1),
])
def test_required_check_only_accepts_a_pr_exemption_or_both_successful_jobs(
        tmp_path, event, changes, required, wheel, gpu, expected):
    result = _run('required', EVENT_NAME=event, CHANGES_RESULT=changes,
                  GPU_REQUIRED=required, WHEEL_RESULT=wheel, GPU_RESULT=gpu)
    assert result.returncode == expected, result.stdout + result.stderr
