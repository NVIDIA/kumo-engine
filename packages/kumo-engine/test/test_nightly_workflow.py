# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the actual workflow guards without a GPU or model."""

import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

WORKFLOW = Path(
    __file__).parents[3] / '.github/workflows/kumo-engine-nightly.yml'


def jobs():
    return yaml.safe_load(WORKFLOW.read_text())['jobs']


def step(name):
    selected = [
        item['run'] for job in jobs().values() for item in job['steps']
        if item.get('name') == name
    ]
    assert len(selected) == 1, name
    return selected[0]


@pytest.mark.parametrize('reference_only', ['false', 'true'])
@pytest.mark.parametrize('exit_code', [0, 23])
def test_actual_final_comparison_exit_controls_the_gpu_job(
        tmp_path, exit_code, reference_only):
    [job] = jobs().values()
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
               MARKER=str(marker), REFERENCE_ONLY=reference_only)
    result = subprocess.run([
        'bash', '-euo', 'pipefail', '-c', compare['run'] + '\ntouch "$MARKER"'
    ], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert calls.exists(), result.stderr
    assert (source / 'proof.txt').read_text() == 'retained original bytes'
    arguments = shlex.split(calls.read_text().splitlines()[-1])
    assert 'compare' in arguments
    assert arguments[arguments.index('--reference-sha256') + 1] == 'b' * 64
    assert ('--reference-only' in arguments) is (reference_only == 'true')
    assert result.returncode == exit_code
    assert marker.exists() is (exit_code == 0)


def test_failed_gpu_query_stops_before_model_work(tmp_path):
    binary = tmp_path / 'bin'
    binary.mkdir()
    probe = binary / 'nvidia-smi'
    probe.write_text('#!/bin/sh\ntouch "$PROBE_FILE"\nexit 1\n')
    probe.chmod(0o755)
    command = next(item['run'] for job in jobs().values()
                   for item in job['steps']
                   if item.get('name') == 'Run sixteen primary fits')
    query = command.split('test "$(git rev-parse HEAD)"', 1)[0]
    marker = tmp_path / 'model-work-started'
    env = dict(os.environ, PATH=f'{binary}{os.pathsep}{os.environ["PATH"]}',
               MARKER=str(marker), PROBE_FILE=str(tmp_path / 'queried'))
    result = subprocess.run(
        ['bash', '-c', 'set -euo pipefail\n' + query + '\ntouch "$MARKER"'],
        env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert (tmp_path / 'queried').exists()
    assert not marker.exists()


def inline_python(name):
    return step(name).split("python - <<'PY'\n", 1)[1].split('\nPY', 1)[0]


@pytest.mark.parametrize('reference_only', ['false', 'true'])
@pytest.mark.parametrize(
    'damage', [None, 'archive', 'reference', 'manifest', 'predecessor'])
def test_actual_private_download_pins_bytes_and_selects_the_runtime(
        tmp_path, monkeypatch, reference_only, damage):
    candidate = b'{"pytorch-lightning":"2.6.6"}'
    reference = b'{"pytorch-lightning":"2.6.5"}'
    files = {'candidate.json': candidate, 'reference.json': reference}
    manifest = json.dumps(
        dict(candidate_dependencies='candidate.json',
             reference_dependencies='reference.json', gpu_model='T4', files={
                 k: hashlib.sha256(v).hexdigest()
                 for k, v in files.items()
             })).encode()
    input_hash = hashlib.sha256(manifest).hexdigest()
    baseline = json.dumps(
        dict(
            version='1.0.0', cases=[{}] * 16,
            input_manifest_sha256=('9' * 64 if damage == 'predecessor' else
                                   input_hash))).encode()
    archive = tmp_path / 'inputs.tar.gz'
    with tarfile.open(archive, 'w:gz') as target:
        for name, value in {**files, 'manifest.json': manifest}.items():
            info = tarfile.TarInfo(name)
            info.size = len(value)
            target.addfile(info, io.BytesIO(value))
    baseline_file = tmp_path / 'baseline.json'
    baseline_file.write_bytes(baseline)
    descriptor = dict(version='nightly-v1', bucket='kumo-test-dataplane-bucket',
                      key='inputs', region='us-west-2',
                      sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                      input_manifest_sha256=input_hash,
                      reference_key='baseline',
                      reference_sha256=hashlib.sha256(baseline).hexdigest())
    if damage == 'archive':
        descriptor['sha256'] = '0' * 64
    if damage == 'reference':
        descriptor['reference_sha256'] = '0' * 64
    if damage == 'manifest':
        descriptor['input_manifest_sha256'] = '0' * 64
    spec = tmp_path / 'packages/kumo-engine/test/nightly/inputs.json'
    spec.parent.mkdir(parents=True)
    spec.write_text(json.dumps(descriptor))
    objects = {'inputs': archive, 'baseline': baseline_file}

    def client(service, region_name):
        assert (service, region_name) == ('s3', 'us-west-2')

        def download(bucket, key, destination):
            assert bucket == 'kumo-test-dataplane-bucket'
            shutil.copyfile(objects[key], destination)

        return SimpleNamespace(download_file=download)

    monkeypatch.setitem(sys.modules, 'boto3', SimpleNamespace(client=client))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('NIGHTLY_ROOT', str(tmp_path / 'nightly'))
    monkeypatch.setenv('REFERENCE_ONLY', reference_only)
    for key in ['GITHUB_ENV', 'GITHUB_OUTPUT']:
        monkeypatch.setenv(key, str(tmp_path / key))
    source = inline_python('Download the frozen inputs and reference')
    if damage:
        with pytest.raises(SystemExit):
            exec(compile(source, 'download', 'exec'), {})
        assert not (tmp_path / 'GITHUB_OUTPUT').exists()
    else:
        exec(compile(source, 'download', 'exec'), {})
        selected = str(
            tmp_path /
            'nightly/inputs/reference.json') if reference_only == 'true' else ''
        assert (tmp_path /
                'nightly/inputs/manifest.json').read_bytes() == manifest
        assert f'profile={selected}\n' in (tmp_path /
                                           'GITHUB_OUTPUT').read_text()
        assert f'INPUT_SHA256={input_hash}\n' in (tmp_path /
                                                  'GITHUB_ENV').read_text()


@pytest.mark.parametrize('damage', [None, 'bytes', 'version'])
def test_published_wheel_receipt_checks_bytes_without_inventing_a_build(
        tmp_path, monkeypatch, damage):
    import zipfile
    wheel_root = tmp_path / 'nightly-wheel'
    wheel_root.mkdir()
    wheel = wheel_root / 'kumo_engine-1.0.0-py3-none-any.whl'
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
    source = inline_python('Download the approved reference wheel')
    if damage:
        with pytest.raises(SystemExit):
            exec(compile(source, 'published', 'exec'), {})
        assert not (wheel_root / 'build.json').exists()
    else:
        exec(compile(source, 'published', 'exec'), {})
        assert json.loads((wheel_root / 'build.json').read_text()) == dict(
            format='kumo-engine-published-reference-v1', version='1.0.0',
            wheel_sha256=expected, harness='a' * 40)


def test_job_storage_is_under_runner_temp_without_host_configuration(tmp_path):
    environment = tmp_path / 'github-env'
    result = subprocess.run(
        ['bash', '-euo', 'pipefail', '-c',
         step('Use temporary job storage')],
        env=dict(os.environ, GITHUB_ENV=str(environment),
                 RUNNER_TEMP=str(tmp_path / 'runner temp')),
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert environment.read_text(
    ) == f'NIGHTLY_ROOT={tmp_path}/runner temp/engine-nightly\n'


@pytest.mark.parametrize('reference_only', ['false', 'true'])
def test_actual_primary_command_passes_reference_without_extra_fits(
        tmp_path, reference_only):
    for name in [
            'nightly-wheel', 'engine-gpu-smoke', 'engine-smoke-venv/bin', 'bin'
    ]:
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / 'nightly-wheel/build.json').write_text('{}')
    (tmp_path / 'engine-gpu-smoke/wheel.json').write_text('{}')
    python = tmp_path / 'engine-smoke-venv/bin/python'
    python.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
                      'with open(os.environ["CALL_FILE"], "a") as stream:\n'
                      '    stream.write(json.dumps(sys.argv[1:]) + "\\n")\n')
    python.chmod(0o755)
    gpu = tmp_path / 'bin/nvidia-smi'
    gpu.write_text('#!/bin/sh\nexit 0\n')
    gpu.chmod(0o755)
    repo = WORKFLOW.parents[2]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo,
                                       text=True).strip()
    calls = tmp_path / 'calls.jsonl'
    env = dict(os.environ, RUNNER_TEMP=str(tmp_path),
               NIGHTLY_ROOT=str(tmp_path), GITHUB_SHA=revision,
               ENGINE_SMOKE_VERSION='1.0.0', INPUT_SHA256='a' * 64,
               REFERENCE_SHA256='b' * 64, REFERENCE_ONLY=reference_only,
               CALL_FILE=str(calls),
               PATH=f'{tmp_path}/bin{os.pathsep}{os.environ["PATH"]}')
    result = subprocess.run(
        ['bash', '-euo', 'pipefail', '-c',
         step('Run sixteen primary fits')], env=env, cwd=repo,
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    [primary] = [args for args in commands if 'run' in args]
    assert primary[primary.index('--reference') + 1] == str(tmp_path /
                                                            'reference.json')
    assert primary[primary.index('--reference-sha256') + 1] == 'b' * 64
    assert ('--reference-only' in primary) is (reference_only == 'true')
    assert len(commands) == 1
    assert not any('--shard' in args for args in commands)


@pytest.mark.parametrize('query_exit', [0, 7])
def test_capacity_snapshot_precedes_heavy_work_and_survives_gpu_query_failure(
        tmp_path, query_exit):
    binary = tmp_path / 'bin'
    binary.mkdir()
    gpu = binary / 'nvidia-smi'
    gpu.write_text(
        f'#!/bin/sh\necho "Tesla T4, 15360, 595.91.07"\nexit {query_exit}\n')
    gpu.chmod(0o755)
    result = subprocess.run([
        'bash', '-euo', 'pipefail', '-c',
        step('Record managed runner capacity')
    ], env=dict(os.environ, RUNNER_TEMP=str(tmp_path),
                PATH=f'{binary}{os.pathsep}{os.environ["PATH"]}'),
                            capture_output=True, text=True)
    assert (result.returncode == 0) is (query_exit == 0), result.stderr
    receipt = json.loads((tmp_path / 'nightly-wheel/capacity.json').read_text())
    assert receipt['cpu_count'] > 0
    assert receipt['system_memory_bytes'] > 0
    assert receipt['free_scratch_bytes'] > 0
    if query_exit == 0:
        assert receipt['gpus'] == [
            dict(model='Tesla T4', memory_mib=15360, driver='595.91.07')
        ]
    packed = subprocess.run([
        'bash', '-euo', 'pipefail', '-c',
        step('Retain results and failures without full input indexes')
    ], env=dict(os.environ, RUNNER_TEMP=str(tmp_path),
                NIGHTLY_ROOT=str(tmp_path)), capture_output=True, text=True)
    assert packed.returncode == 0, packed.stderr
    assert json.loads(
        (tmp_path /
         'nightly-result/setup/capacity.json').read_text()) == receipt
    [job] = jobs().values()
    names = [item.get('name') for item in job['steps']]
    assert names.index('Record managed runner capacity') < names.index(
        'Download the frozen inputs and reference')
