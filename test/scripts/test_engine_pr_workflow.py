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

ROOT = Path(__file__).resolve().parents[2]


def test_selection_shell_materializes_paginated_files_with_supported_gh_flags(
        tmp_path):
    workflow = yaml.safe_load(
        (ROOT / '.github/workflows/kumo-engine-pr.yml').read_text())
    step = next(s for s in workflow['jobs']['changes']['steps']
                if s.get('id') == 'select')
    revision = 'a' * 40
    files = [[dict(filename='docs/example.md', status='modified')],
             [dict(filename='runtime.py', status='modified')]]
    metadata = dict(head=dict(sha=revision), changed_files=2)
    commands = tmp_path / 'bin'
    commands.mkdir()
    (commands / 'python3').symlink_to(sys.executable)
    gh = commands / 'gh'
    gh.write_text(f'''#!/usr/bin/env python3
import json,sys
if '--slurp' in sys.argv and '--jq' in sys.argv:
    raise SystemExit('The --slurp option is not supported with --jq')
print(json.dumps({files!r} if any('/files?' in a for a in sys.argv) else {metadata!r}))
''')
    gh.chmod(0o755)
    output = tmp_path / 'github-output'
    completed = subprocess.run(
        ['bash', '-e', '-c', step['run']], cwd=ROOT,
        env=dict(os.environ, PATH=f'{commands}:{os.environ["PATH"]}',
                 INPUT_PR='135', GITHUB_REF_NAME='pull-request/135',
                 GITHUB_REPOSITORY='NVIDIA/kumo-engine', GITHUB_SHA=revision,
                 GITHUB_OUTPUT=str(output), RUNNER_TEMP=str(tmp_path)),
        capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = json.loads((tmp_path / 'pr-selection/selection.json').read_text())
    assert receipt['paths'] == ['docs/example.md', 'runtime.py']
    assert receipt['changed_files'] == 2
    assert dict(line.split('=', 1)
                for line in output.read_text().splitlines()) == {
                    'learning': 'true',
                    'install': 'true',
                    'example': 'false',
                    'behavior': 'true',
                    'source': 'true',
                    'gpu': 'true'
                }


@pytest.mark.parametrize('token', ['', 'test-only-opaque-credential'])
def test_generator_token_preflight_preserves_failure_without_exposing_secret(
        tmp_path, token):
    workflow = yaml.safe_load(
        (ROOT / '.github/workflows/kumo-engine-pr.yml').read_text())
    step = next(
        s for s in workflow['jobs']['gpu']['steps']
        if s.get('name') == 'Require access to the pinned synthetic generators')
    completed = subprocess.run(['bash', '-e', '-c', step['run']], cwd=ROOT,
                               env=dict(os.environ, SOURCE_TOKEN=token,
                                        RUNNER_TEMP=str(tmp_path),
                                        GITHUB_SHA='a' * 40),
                               capture_output=True, text=True)
    receipt_path = tmp_path / 'pr-preflight/source-token.json'
    assert receipt_path.exists(), completed.stdout + completed.stderr
    receipt = json.loads(receipt_path.read_text())
    assert receipt['revision'] == 'a' * 40
    assert receipt['token_configured'] is bool(token)
    assert receipt['physical_fits'] == 0
    assert receipt['source_checkout_verified'] is False
    assert (completed.returncode == 0) is bool(token)
    if token:
        assert token not in receipt_path.read_text()
        assert token not in completed.stdout + completed.stderr
    else:
        assert receipt['status'] == 'FAIL'
        assert 'KUMO_GITHUB_BOT_TOKEN' in completed.stdout + completed.stderr
