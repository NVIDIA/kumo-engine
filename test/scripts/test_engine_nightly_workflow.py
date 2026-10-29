# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise nightly configuration and selected-product build receipts."""

import fcntl
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / '.github/workflows/kumo-engine-nightly.yml'
CONFIGURATION = {
    'NIGHTLY_RTX_RUNNER': 'engine-nightly-rtx',
    'NIGHTLY_ROOT': '/srv/engine-nightly',
    'NIGHTLY_DATA': '/srv/data',
    'NIGHTLY_SCALE_SOURCE': '/srv/scale-source',
    'NIGHTLY_FIXTURES': '/srv/fixtures',
    'NIGHTLY_DEPENDENCY_PROFILE': '/srv/dependencies.json',
    'NIGHTLY_REFERENCE_CACHE': '/srv/reference',
    'NIGHTLY_REFERENCE_SHA256': 'a' * 64,
}


def workflow():
    assert WORKFLOW.is_file(), 'The nightly workflow is missing'
    return yaml.safe_load(WORKFLOW.read_text())


def step(job, identifier):
    result = next((s for s in workflow()['jobs'][job]['steps']
                   if s.get('id') == identifier), None)
    assert result is not None, f'Missing {job}/{identifier} workflow step'
    return result


@pytest.mark.parametrize('missing', list(CONFIGURATION))
def test_missing_trusted_configuration_fails_before_runner_selection(
        tmp_path, missing):
    values = dict(CONFIGURATION, **{missing: ''})
    output = tmp_path / 'output'
    completed = subprocess.run(
        ['bash', '-e', '-c',
         step('configure', 'configure')['run']],
        env=dict(os.environ, **values,
                 GITHUB_OUTPUT=str(output)), capture_output=True, text=True)
    assert completed.returncode != 0
    assert missing in completed.stdout + completed.stderr
    assert not output.exists() or not output.read_text().strip()


def test_explicit_configuration_emits_only_the_requested_runner(tmp_path):
    output = tmp_path / 'output'
    completed = subprocess.run(
        ['bash', '-e', '-c',
         step('configure', 'configure')['run']],
        env=dict(os.environ, **CONFIGURATION,
                 GITHUB_OUTPUT=str(output)), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert output.read_text().splitlines() == ['runner=engine-nightly-rtx']
    gpu = workflow()['jobs']['gpu']
    assert gpu['needs'] == ['configure', 'build']
    assert gpu['runs-on'] == '${{ needs.configure.outputs.runner }}'
    tasks = [
        task for shard in gpu['strategy']['matrix']['include']
        for task in shard['tasks'].split()
    ]
    assert sorted(tasks) == [
        'binary', 'forecasting', 'multiclass', 'multilabel',
        'multilabel_ranking', 'regression', 'static_link', 'temporal_link'
    ]
    assert [
        len(s['tasks'].split()) for s in gpu['strategy']['matrix']['include']
    ] == [4, 4]


def test_build_receipt_identifies_the_selected_product_tree(tmp_path):
    """A tiny builder tests receipt wiring, not an actual Engine wheel build."""
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'harness').symlink_to(ROOT, target_is_directory=True)
    product = workspace / 'product'
    product.mkdir()
    (product / 'VERSION').write_text('0.1.0\n')
    builder = product / 'packages/kumo-engine/scripts/build_wheel.sh'
    builder.parent.mkdir(parents=True)
    builder.write_text('''#!/usr/bin/env bash
set -eu
python3 - "$2" <<'PY'
import pathlib,subprocess,sys,zipfile
out=pathlib.Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
sha=subprocess.check_output(['git','rev-parse','--short','HEAD'],text=True).strip()
version='0.1.0+'+sha
with zipfile.ZipFile(out/('kumo_engine-'+version+'-py3-none-any.whl'),'w') as z:
 z.writestr('kumo_engine-'+version+'.dist-info/METADATA',
            'Name: kumo-engine\\nVersion: '+version+'\\n')
PY
''')
    builder.chmod(0o755)
    for arguments in [['init', '-q'], ['config', 'user.name', 'Workflow test'],
                      ['config', 'user.email', 'workflow@example.com'],
                      ['add', '.'],
                      ['commit', '-qm', 'Selected product fixture']]:
        subprocess.run(['git', *arguments], cwd=product, check=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=product, text=True).strip()
    commands = tmp_path / 'bin'
    commands.mkdir()
    (commands / 'python').symlink_to(sys.executable)
    (commands / 'python3').symlink_to(sys.executable)
    temporary = tmp_path / 'runner-temp'
    temporary.mkdir()
    output = tmp_path / 'github-output'
    completed = subprocess.run(
        ['bash', '-e', '-c', step('build', 'wheel')['run']], cwd=workspace,
        env=dict(os.environ, PATH=f'{commands}:{os.environ["PATH"]}',
                 GITHUB_WORKSPACE=str(workspace), RUNNER_TEMP=str(temporary),
                 GITHUB_OUTPUT=str(output)), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = json.loads((temporary / 'nightly-wheel/build.json').read_text())
    assert receipt['revision'] == revision
    assert receipt['version'] == '0.1.0+' + revision[:7]
    assert len(receipt['wheel_sha256']) == 64
    assert json.loads((temporary / 'nightly-wheel/source.json'
                       ).read_text())['tree'] == subprocess.check_output(
                           ['git', 'rev-parse', 'HEAD^{tree}'], cwd=product,
                           text=True).strip()


def test_frozen_runtime_profile_does_not_require_pytest(tmp_path):
    """The input guard must accept the measured runtime without test tools."""
    values = dict(CONFIGURATION)
    for name in ('NIGHTLY_DATA', 'NIGHTLY_SCALE_SOURCE', 'NIGHTLY_FIXTURES',
                 'NIGHTLY_REFERENCE_CACHE'):
        directory = tmp_path / name
        directory.mkdir()
        values[name] = str(directory)
    manifest = Path(values['NIGHTLY_REFERENCE_CACHE']) / 'manifest.json'
    manifest.write_text('{"files": {}}\n')
    values['NIGHTLY_REFERENCE_SHA256'] = hashlib.sha256(
        manifest.read_bytes()).hexdigest()
    profile = tmp_path / 'dependencies.json'
    packages = {f'package{i}': '1.0' for i in range(97)}
    packages['duckdb'] = '1.5.5'
    profile.write_text(json.dumps(packages))
    values['NIGHTLY_DEPENDENCY_PROFILE'] = str(profile)
    run_root = tmp_path / 'run'
    completed = subprocess.run(
        ['bash', '-e', '-c', step('gpu', 'inputs')['run']],
        env=dict(os.environ, **values,
                 RUN_ROOT=str(run_root)), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(
        (run_root / 'setup/dependencies.json').read_text()) == packages
    assert 'pytest' not in (run_root / 'setup/dependencies.txt').read_text()


def test_trusted_push_includes_nightly_runtime_and_comparison_dependencies():
    document = workflow()
    triggers = document.get('on', document.get(True))
    push = triggers['push']
    assert push['branches'] == ['pull-request/[0-9]+']
    assert set(push['paths']) == {
        '.github/workflows/kumo-engine-nightly.yml',
        'docs/development/engine-nightly-ci.md',
        'test/scripts/test_engine_nightly_workflow.py',
        'packages/kumo-engine/scripts/nightly_check.py',
        'packages/kumo-engine/scripts/nightly_compare.py',
        'packages/kumo-engine/scripts/pr_check.py',
        'test/scripts/test_engine_nightly_runtime.py',
        'test/scripts/test_engine_nightly_compare.py',
        'packages/kumo-engine/test/wheel_gpu_learning.py',
    }
    assert document['env']['PIP_CONFIG_FILE'] == '/dev/null'


@pytest.mark.parametrize('matches', [True, False])
def test_harness_receipt_requires_the_workflow_revision(tmp_path, matches):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'harness').symlink_to(ROOT, target_is_directory=True)
    run_root = tmp_path / 'run'
    (run_root / 'setup').mkdir(parents=True)
    revision = subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    completed = subprocess.run(
        ['bash', '-e', '-c', step('gpu', 'harness')['run']],
        env=dict(os.environ, GITHUB_WORKSPACE=str(workspace),
                 RUN_ROOT=str(run_root),
                 GITHUB_SHA=revision if matches else 'a' * 40,
                 GITHUB_RUN_ID='123', GITHUB_RUN_ATTEMPT='2'),
        capture_output=True, text=True)
    receipt = run_root / 'setup/harness.json'
    if matches:
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert json.loads(receipt.read_text()) == {
            'revision': revision,
            'run_id': '123',
            'run_attempt': '2'
        }
    else:
        assert completed.returncode != 0
        assert not receipt.exists()


@pytest.mark.parametrize('job, identifier', [('compare', 'environment'),
                                             ('gpu', 'install'),
                                             ('gpu', 'pack')])
def test_runtime_work_stops_while_host_lock_is_busy(tmp_path, job, identifier):
    shared = tmp_path / 'shared'
    shared.mkdir()
    run_root = tmp_path / 'run'
    commands = tmp_path / 'bin'
    commands.mkdir()
    (commands / 'python3').symlink_to(sys.executable)
    # macOS has no flock executable. This shim uses the same kernel lock.
    if not shutil.which('flock'):
        command = commands / 'flock'
        command.write_text('''#!/usr/bin/env python3
import fcntl,sys
try:
 fcntl.flock(int(sys.argv[-1]),fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
 raise SystemExit(1)
''')
        command.chmod(0o755)
    with (shared / 'gpu-study.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = subprocess.run(
            ['bash', '-e', '-c',
             step(job, identifier)['run']],
            env=dict(os.environ, NIGHTLY_ROOT=str(shared),
                     RUN_ROOT=str(run_root),
                     PATH=f'{commands}:{os.environ["PATH"]}'),
            capture_output=True, text=True)
    assert completed.returncode != 0
    assert not run_root.exists()


def test_required_job_accepts_success_and_rejects_actual_unmet_jobs(tmp_path):
    command = workflow()['jobs']['required']['steps'][0]['run']
    results = {
        name: dict(result='success')
        for name in ('configure', 'build', 'gpu', 'compare')
    }
    for failed in (None, 'gpu', 'compare'):
        changed = {name: dict(value) for name, value in results.items()}
        if failed is not None:
            changed[failed]['result'] = 'failure'
        completed = subprocess.run(['bash', '-e', '-c', command], env=dict(
            os.environ, RESULTS=json.dumps(changed),
            GITHUB_STEP_SUMMARY=str(tmp_path / 'summary')), capture_output=True,
                                   text=True)
        assert (completed.returncode == 0) is (failed is None), completed.stderr


@pytest.mark.parametrize('missing_marker', [False, True])
def test_coverage_requires_three_phases_and_checked_b7(tmp_path,
                                                       missing_marker):
    fixtures = runpy.run_path(
        str(Path(__file__).with_name('test_engine_nightly_runtime.py')))
    pr = runpy.run_path(str(ROOT / 'packages/kumo-engine/scripts/pr_check.py'))
    assert callable(pr.get('verify_b7')), 'Missing required B7 check'
    root = tmp_path / 'run'
    correctness = root / 'correctness'
    fixtures['b7_evidence'](correctness / 'behavior')
    for phase in ('install', 'example', 'behavior'):
        directory = correctness / phase
        directory.mkdir(exist_ok=True)
        report = dict(status='PASS', physical_fits=1)
        if phase == 'behavior' and not missing_marker:
            report['b7'] = pr['verify_b7'](directory, 'candidate-test')
        (directory / 'result.json').write_text(json.dumps(report))
        (directory / 'fits.json').write_text(
            json.dumps(
                [dict(status='PASS', device='cuda:0', steps=1,
                      cuda_batches=1)]))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'harness').symlink_to(ROOT, target_is_directory=True)
    completed = subprocess.run(
        ['bash', '-e', '-c', step('gpu', 'coverage')['run']],
        env=dict(os.environ, RUN_ROOT=str(root),
                 GITHUB_WORKSPACE=str(workspace),
                 CANDIDATE_VERSION='candidate-test'), capture_output=True,
        text=True)
    assert (completed.returncode == 0) is (not missing_marker), completed.stderr
    assert json.loads(
        (correctness /
         'coverage.json').read_text())['status'] == ('FAIL' if missing_marker
                                                     else 'PASS')
