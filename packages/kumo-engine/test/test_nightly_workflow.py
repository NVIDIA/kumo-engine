# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the admission, launch, and cleanup programs used by Actions."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).parents[3]
WORKFLOW = ROOT / '.github/workflows/kumo-engine-nightly.yml'


def step(name, job_name=None):
    jobs = yaml.safe_load(WORKFLOW.read_text())['jobs']
    selected = [jobs[job_name]] if job_name else jobs.values()
    return next(s['run'] for j in selected for s in j.get('steps', [])
                if s.get('name') == name)


def configuration():
    return json.dumps(
        dict(
            format='kumo-engine-nightly-job-v1', min_free_bytes=1024,
            input_manifest_sha256='a' * 64, **{
                name:
                dict(bucket='shared-fixtures', region='us-west-2',
                     key='engine/nightly/' + name, sha256='b' * 64)
                for name in ('inputs', 'reference', 'policy')
            }))


@pytest.mark.parametrize('ref,head,accepted', [
    ('refs/heads/main', 'a' * 40, True),
    ('refs/heads/pull-request/137', 'a' * 40, True),
    ('refs/heads/pull-request/137', 'b' * 40, False),
    ('refs/heads/unreviewed', 'a' * 40, False),
])
def test_admission_rejects_arbitrary_and_stale_branches(tmp_path, ref, head,
                                                        accepted):
    binary = tmp_path / 'bin'
    binary.mkdir()
    (binary / 'python').symlink_to(sys.executable)
    gh = binary / 'gh'
    gh.write_text(
        f'#!{sys.executable}\nimport sys\n'
        'assert sys.argv[1:]==["api","repos/NVIDIA/kumo-engine/pulls/137"]\n'
        f'print({json.dumps(dict(state="open", head=dict(sha=head), base=dict(repo=dict(full_name="NVIDIA/kumo-engine"))))!r})\n'
    )
    gh.chmod(0o755)
    output = tmp_path / 'output'
    env = dict(os.environ, RUNNER_TEMP=str(tmp_path), GITHUB_REF=ref,
               GITHUB_EVENT_NAME='push', GITHUB_SHA='a' * 40,
               GITHUB_REPOSITORY='NVIDIA/kumo-engine',
               GITHUB_OUTPUT=str(output), NIGHTLY_CONFIGURATION=configuration(),
               PATH=str(binary) + os.pathsep + os.environ['PATH'])
    result = subprocess.run([
        'bash', '-euo', 'pipefail', '-c',
        step('Admit main or a current approved PR copy')
    ], cwd=ROOT, env=env, capture_output=True, text=True)
    assert (result.returncode == 0) is accepted, result.stderr
    if accepted:
        assert json.loads(output.read_text().split(
            '=', 1)[1])['min_free_bytes'] == 1024
    else:
        assert not output.exists()


@pytest.mark.parametrize('exit_code', [0, 7])
def test_actual_launch_binds_the_wheel_and_retains_worker_failure(
        tmp_path, exit_code):
    environment_file = tmp_path / 'environment'
    env = dict(
        os.environ, RUNNER_TEMP=str(tmp_path), GITHUB_ENV=str(environment_file),
        NIGHTLY_CONFIGURATION=configuration(),
        PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'])
    reserved = subprocess.run([
        'bash', '-euo', 'pipefail', '-c',
        step('Reserve time for artifact upload', 'gpu')
    ], cwd=ROOT, env=env, capture_output=True, text=True)
    assert reserved.returncode == 0, reserved.stderr
    created = subprocess.run([
        'bash', '-euo', 'pipefail', '-c',
        step('Create the job directory and require free space')
    ], cwd=ROOT, env=env, capture_output=True, text=True)
    assert created.returncode == 0, created.stderr
    env.update(
        line.split('=', 1)
        for line in environment_file.read_text().splitlines())
    work = Path(env['NIGHTLY_JOB_ROOT'])
    spec = importlib.util.spec_from_file_location(
        'fresh_nightly', ROOT / 'packages/kumo-engine/scripts/nightly_check.py')
    nightly = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nightly)
    with pytest.raises(FileNotFoundError) as missing:
        nightly.run(
            SimpleNamespace(input_root=work / 'inputs',
                            output_dir=Path(env['NIGHTLY_EVIDENCE_ROOT']),
                            scratch=work / 'scratch', input_sha256='a' * 64))
    assert Path(missing.value.filename) == work / 'inputs/manifest.json'
    (work / 'env/bin').mkdir(parents=True)
    worker = work / 'env/bin/python'
    calls = tmp_path / 'calls.json'
    worker.write_text(
        f'#!{sys.executable}\nimport json,sys\n'
        f'open({str(calls)!r},"w").write(json.dumps(sys.argv[1:]))\n'
        f'sys.exit({exit_code})\n')
    worker.chmod(0o755)
    wheel = tmp_path / 'candidate.whl'
    wheel.write_bytes(b'candidate')
    env.update(NIGHTLY_JOB_ROOT=str(work), NIGHTLY_MANIFEST_SHA256='a' * 64,
               NIGHTLY_REFERENCE_SHA256='b' * 64,
               NIGHTLY_POLICY_SHA256='c' * 64, ENGINE_NIGHTLY_WHEEL=str(wheel),
               ENGINE_NIGHTLY_VERSION='1.2.0+abc123', REFERENCE_ONLY='false')
    result = subprocess.run(
        ['bash', '-euo', 'pipefail', '-c',
         step('Run sixteen cases')], cwd=ROOT, env=env, capture_output=True,
        text=True)
    assert result.returncode == exit_code, result.stderr
    args = json.loads(calls.read_text())
    for option, expected in [('--wheel', str(wheel)),
                             ('--expected-version', '1.2.0+abc123'),
                             ('--input-sha256', 'a' * 64),
                             ('--reference-sha256', 'b' * 64),
                             ('--policy-sha256', 'c' * 64),
                             ('--scratch', str(work / 'scratch'))]:
        assert args[args.index(option) + 1] == expected
    assert not Path(args[args.index('--output-dir') + 1]).is_relative_to(work)
    evidence = Path(env['NIGHTLY_EVIDENCE_ROOT'])
    assert not (evidence / 'run.log').exists()
    staged = subprocess.run(
        ['bash', '-euo', 'pipefail', '-c',
         step('Stage the native log')], cwd=ROOT, env=env, capture_output=True,
        text=True)
    assert staged.returncode == 0, staged.stderr
    assert (evidence / 'run.log').is_file()
    env['NIGHTLY_EVIDENCE_UPLOADED'] = 'true'
    cleaned = subprocess.run([
        'bash', '-euo', 'pipefail', '-c',
        step("Remove this job's temporary work")
    ], cwd=ROOT, env=env, capture_output=True, text=True)
    assert cleaned.returncode == 0, cleaned.stderr
    assert not work.exists() and not evidence.exists()


def test_credentials_are_confined_to_the_private_runner_job():
    jobs = yaml.safe_load(WORKFLOW.read_text())['jobs']
    for job in jobs.values():
        receives_credentials = any(
            'configure-aws-credentials' in s.get('uses', '')
            for s in job.get('steps', []))
        if receives_credentials:
            assert job['runs-on']['group'].startswith('nv-')
            assert job['environment']['name'] == 'gha-kumo-workflow-tests'
        elif isinstance(job['runs-on'],
                        str) and job['runs-on'].startswith('ubuntu-'):
            assert job.get('permissions', {}).get('id-token') != 'write'


@pytest.mark.skipif(
    shutil.which('uv') is None, reason='The installer test needs uv')
def test_install_step_resolves_new_candidate_requirements_without_the_frozen_profile(
        tmp_path):
    import hashlib
    import zipfile

    work = tmp_path / 'job'
    evidence = tmp_path / 'evidence'
    (work / 'wheel').mkdir(parents=True)
    evidence.mkdir()
    links = tmp_path / 'wheels'
    links.mkdir()

    def wheel(directory, name, version, dependency=''):
        target = directory / f'{name}-{version}-py3-none-any.whl'
        info = f'{name}-{version}.dist-info'
        with zipfile.ZipFile(target, 'w') as archive:
            archive.writestr(
                info + '/METADATA',
                f'Metadata-Version: 2.1\nName: {name.replace("_", "-")}\nVersion: {version}\n'
                + dependency)
            archive.writestr(
                info + '/WHEEL',
                'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n'
            )
            archive.writestr(info + '/RECORD', '')
            archive.writestr(name + '/__init__.py', '')
        return target

    wheel(links, 'nightly_example_dep', '1.0')
    wheel(links, 'nightly_example_dep', '2.0')
    wheel(links, 'duckdb', '1.4.0')
    candidate = wheel(work / 'wheel', 'kumo_engine', '1.2.0',
                      'Requires-Dist: nightly-example-dep==2.0\n')
    (work / 'wheel/build.json').write_text(
        json.dumps(
            dict(
                version='1.2.0', wheel_sha256=hashlib.sha256(
                    candidate.read_bytes()).hexdigest())))
    (work / 'inputs').mkdir()
    (work / 'inputs/candidate-dependencies.json'
     ).write_text('{"nightly-example-dep":"1.0"}')
    env = dict(
        os.environ, NIGHTLY_JOB_ROOT=str(work),
        NIGHTLY_EVIDENCE_ROOT=str(evidence), RUNNER_TEMP=str(tmp_path),
        GITHUB_ENV=str(tmp_path / 'environment'), UV_NO_INDEX='true',
        UV_OFFLINE='true', UV_FIND_LINKS=str(links),
        PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'])
    result = subprocess.run([
        'bash', '-euo', 'pipefail', '-c',
        step("Resolve the candidate wheel's own requirements")
    ], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    installed = subprocess.check_output([
        str(work / 'env/bin/python'), '-I', '-c',
        'import importlib.metadata as m;print(m.version("nightly-example-dep"))'
    ], text=True)
    assert installed.strip() == '2.0'
