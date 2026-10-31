# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the actual CPU controller command and published-wheel receipt guards."""
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(
    __file__).parents[3] / '.github/workflows/kumo-engine-nightly.yml'


def jobs():
    return yaml.safe_load(WORKFLOW.read_text())['jobs']


def step(name):
    return next(item['run'] for job in jobs().values() for item in job['steps']
                if item.get('name') == name)


@pytest.mark.parametrize('reference_only', ['false', 'true'])
@pytest.mark.parametrize('exit_code', [0, 23])
def test_actual_controller_command_propagates_validation_failure_without_local_fits(
        tmp_path, reference_only, exit_code):
    job = jobs()['nightly']
    assert job['runs-on'] == 'ubuntu-22.04' and job['needs'] == 'build'
    source = step('Run and validate the fixed-VM nightly')
    binary = tmp_path / 'bin'
    binary.mkdir()
    executable = binary / 'python'
    executable.write_text(
        f'#!{sys.executable}\nimport json,os,sys\n'
        'open(os.environ["CALL_FILE"],"w").write(json.dumps(sys.argv[1:]))\n'
        'raise SystemExit(int(os.environ["COMMAND_EXIT"]))\n')
    executable.chmod(0o755)
    wheel = tmp_path / 'nightly-wheel'
    wheel.mkdir()
    (wheel / 'candidate.whl').touch()
    repo = WORKFLOW.parents[2]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=repo,
                                       text=True).strip()
    output = tmp_path / 'calls.json'
    env = dict(os.environ,
               RUNNER_TEMP=str(tmp_path),
               GITHUB_SHA=revision,
               GITHUB_RUN_ID='123',
               GITHUB_RUN_ATTEMPT='2',
               REFERENCE_ONLY=reference_only,
               CALL_FILE=str(output),
               COMMAND_EXIT=str(exit_code),
               PATH=f'{binary}{os.pathsep}{os.environ["PATH"]}')
    result = subprocess.run(['bash', '-euo', 'pipefail', '-c', source],
                            cwd=repo,
                            env=env,
                            capture_output=True,
                            text=True)
    assert result.returncode == exit_code, result.stderr
    args = json.loads(output.read_text())
    assert args[:2] == ['-I', 'scripts/engine_nightly_run.py']
    assert args[args.index('--run-id') + 1] == '123-2'
    assert args[args.index('--wheel') + 1] == str(wheel / 'candidate.whl')
    assert ('--reference-only' in args) is (reference_only == 'true')
    assert not jobs()['build'].get('permissions', {}).get('id-token')


@pytest.mark.parametrize('damage', [None, 'bytes', 'version'])
def test_published_wheel_receipt_checks_exact_bytes_and_version(
        tmp_path, monkeypatch, damage):
    root = tmp_path / 'nightly-wheel'
    root.mkdir()
    wheel = root / 'kumo_engine-1.0.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr(
            'kumo_engine-1.0.0.dist-info/METADATA',
            'Name: kumo-engine\nVersion: ' +
            ('1.1.0' if damage == 'version' else '1.0.0') + '\n')
    expected = hashlib.sha256(wheel.read_bytes()).hexdigest()
    monkeypatch.setenv('REFERENCE_WHEEL_SHA256', expected)
    monkeypatch.setenv('RUNNER_TEMP', str(tmp_path))
    monkeypatch.setenv('GITHUB_SHA', 'a' * 40)
    if damage == 'bytes':
        wheel.write_bytes(wheel.read_bytes() + b'changed')
    code = step('Download the approved reference wheel').split(
        "python - <<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
    if damage:
        with pytest.raises(SystemExit):
            exec(compile(code, 'published', 'exec'), {})
        assert not (root / 'build.json').exists()
    else:
        exec(compile(code, 'published', 'exec'), {})
        assert json.loads((root / 'build.json').read_text()) == dict(
            format='kumo-engine-published-reference-v1',
            version='1.0.0',
            wheel_sha256=expected,
            harness='a' * 40)


def test_cancelled_controller_has_a_separate_cleanup_job(tmp_path):
    assert 'cleanup' in jobs(
    ), 'A cancelled controller needs independent cleanup'
    job = jobs()['cleanup']
    assert job['needs'] == 'nightly' and 'always()' in job['if']
    source = step('Cancel and confirm the fixed-VM workload')
    binary = tmp_path / 'bin'
    binary.mkdir()
    command = binary / 'python'
    command.write_text(
        f'#!{sys.executable}\nimport json,os,sys\n'
        'open(os.environ["CALL_FILE"],"w").write(json.dumps(sys.argv[1:]))\n')
    command.chmod(0o755)
    output = tmp_path / 'calls.json'
    result = subprocess.run(
        ['bash', '-euo', 'pipefail', '-c', source],
        capture_output=True,
        text=True,
        env=dict(os.environ,
                 RUNNER_TEMP=str(tmp_path),
                 GITHUB_RUN_ID='123',
                 GITHUB_RUN_ATTEMPT='2',
                 GITHUB_SHA='a' * 40,
                 CALL_FILE=str(output),
                 PATH=f'{binary}{os.pathsep}{os.environ["PATH"]}'))
    assert result.returncode == 0, result.stderr
    args = json.loads(output.read_text())
    assert '--cancel-only' in args and '--wheel' not in args
    assert args[args.index('--run-id') + 1] == '123-2'
