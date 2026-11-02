# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise nightly artifact, capacity, and cleanup boundaries."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(
    __file__).parents[2] / 'packages/kumo-engine/scripts/nightly_job.py'
SPEC = importlib.util.spec_from_file_location('nightly_job', SOURCE)
job = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(job)


def config():
    return dict(
        format='kumo-engine-nightly-job-v1', min_free_bytes=1024,
        input_manifest_sha256='a' * 64, **{
            key:
            dict(bucket='shared-fixtures', region='us-west-2',
                 key='engine/nightly/' + key, sha256='b' * 64)
            for key in ('inputs', 'reference', 'policy')
        })


def test_configuration_rejects_missing_and_unsafe_object_bindings():
    value = config()
    assert job.configuration(json.dumps(value)) == value
    for key, damage in [('min_free_bytes', 0), ('input_manifest_sha256', 'bad'),
                        ('inputs', {}), ('policy', None)]:
        with pytest.raises(ValueError):
            job.configuration(json.dumps(dict(value, **{key: damage})))
    value['inputs']['key'] = '../other'
    with pytest.raises(ValueError):
        job.configuration(json.dumps(value))


def test_local_reference_and_policy_are_checked_before_staging(tmp_path):
    source = tmp_path / 'repository'
    source.mkdir()
    reference = source / 'reference.json'
    reference.write_text('{"format":"historical-proposal"}')
    item = dict(path='reference.json',
                sha256=hashlib.sha256(reference.read_bytes()).hexdigest())
    value = config()
    value['reference'] = item
    assert job.configuration(json.dumps(value)) == value
    output = tmp_path / 'staged-reference.json'
    job.stage_object(item, output, source)
    assert output.read_bytes() == reference.read_bytes()
    reference.write_text('changed after review')
    with pytest.raises(ValueError, match='checksum'):
        job.stage_object(item, tmp_path / 'rejected.json', source)
    assert not (tmp_path / 'rejected.json').exists()
    for path in ('../reference.json', '/reference.json'):
        value['reference'] = dict(item, path=path)
        with pytest.raises(ValueError):
            job.configuration(json.dumps(value))


def test_capacity_uses_free_bytes_even_when_total_disk_is_large(
        tmp_path, monkeypatch):
    usage = shutil.disk_usage(tmp_path)._replace(total=1 << 50, free=1)
    monkeypatch.setattr(job.shutil, 'disk_usage', lambda path: usage)
    with pytest.raises(ValueError, match='free'):
        job.create_work(tmp_path, 2)
    assert list(tmp_path.iterdir()) == []


def test_cleanup_removes_only_the_marked_job_and_preserves_evidence(tmp_path):
    work = job.create_work(tmp_path, 1)
    (work / 'env').mkdir()
    (work / 'env/temporary').write_text('temporary')
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    (evidence / 'result.json').write_text('{"status":"FAIL"}')
    job.cleanup_work(work, tmp_path)
    assert not work.exists()
    assert (evidence / 'result.json').read_text() == '{"status":"FAIL"}'
    for unsafe in [tmp_path, evidence, tmp_path.parent]:
        with pytest.raises(ValueError):
            job.cleanup_work(unsafe, tmp_path)


def test_cleanup_rejects_a_symlink_to_another_job(tmp_path):
    first = job.create_work(tmp_path, 1)
    second = tmp_path / 'engine-nightly-link'
    second.symlink_to(first, target_is_directory=True)
    with pytest.raises(ValueError):
        job.cleanup_work(second, tmp_path)
    assert first.is_dir()


def test_cleanup_checks_job_ownership_before_removing_any_evidence(tmp_path):
    other = tmp_path / 'other'
    other.mkdir()
    evidence = tmp_path / 'other-evidence'
    evidence.mkdir()
    (evidence / '.nightly-evidence').write_text(str(other))
    (evidence / 'valuable.json').write_text('keep')
    result = subprocess.run(
        [sys.executable, str(SOURCE), 'cleanup'],
        env=dict(os.environ, RUNNER_TEMP=str(tmp_path),
                 NIGHTLY_JOB_ROOT=str(other), NIGHTLY_EVIDENCE_UPLOADED='true'),
        capture_output=True, text=True)
    assert result.returncode != 0
    assert (evidence / 'valuable.json').read_text() == 'keep'


@pytest.mark.parametrize('uploaded', [False, True])
def test_cli_removes_job_artifacts_and_keeps_evidence_until_upload(
        tmp_path, uploaded):
    environment_file = tmp_path / 'environment'
    environment = dict(os.environ, RUNNER_TEMP=str(tmp_path),
                       GITHUB_ENV=str(environment_file),
                       NIGHTLY_CONFIGURATION=json.dumps(config()))
    created = subprocess.run(
        [sys.executable, str(SOURCE), 'create'], env=environment,
        capture_output=True, text=True)
    assert created.returncode == 0, created.stderr
    environment.update(
        line.split('=', 1)
        for line in environment_file.read_text().splitlines())
    work = Path(environment['NIGHTLY_JOB_ROOT'])
    evidence = Path(environment['NIGHTLY_EVIDENCE_ROOT'])
    for relative in ('env/package', 'cache/wheel', 'scratch/graph1/index',
                     'scratch/graph2/index', 'scratch/graph3/index'):
        artifact = work / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text('temporary work')
    for relative in ('models/checkpoint', 'predictions/output.parquet'):
        artifact = evidence / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text('retained result')
    previous = tmp_path / 'engine-nightly-prior-evidence'
    previous.mkdir()
    (previous / 'keep').write_text('earlier evidence')
    cleaned = subprocess.run(
        [sys.executable, str(SOURCE), 'cleanup'],
        env=dict(environment, NIGHTLY_EVIDENCE_UPLOADED=str(uploaded).lower()),
        capture_output=True, text=True)
    assert cleaned.returncode == 0, cleaned.stderr
    assert not work.exists()
    assert evidence.exists() is not uploaded
    if not uploaded:
        assert (evidence / 'models/checkpoint').read_text() == 'retained result'
        assert (evidence /
                'predictions/output.parquet').read_text() == 'retained result'
    assert (previous / 'keep').read_text() == 'earlier evidence'


def test_build_command_produces_the_receipt_consumed_by_the_case_runner(
        tmp_path):
    import shlex

    product = tmp_path / 'product'
    product.mkdir()

    def git(*args):
        return subprocess.check_output(
            ['git', '-C', str(product), *args], text=True).strip()

    git('init', '-b', 'main')
    git('config', 'user.name', 'Test')
    git('config', 'user.email', 'test@example.com')
    (product / 'VERSION').write_text('1.2.3\n')
    script = product / 'packages/kumo-engine/scripts/build_wheel.sh'
    script.parent.mkdir(parents=True)
    script.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) +
                      ''' - "$2" <<'BUILD'
import pathlib, subprocess, sys, zipfile
output = pathlib.Path(sys.argv[1])
version = '1.2.3+' + subprocess.check_output(
    ['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()
with zipfile.ZipFile(output / 'candidate.whl', 'w') as archive:
    archive.writestr('kumo_engine.dist-info/METADATA',
                     'Name: kumo-engine\\nVersion: ' + version + '\\n')
BUILD
''')
    script.chmod(0o755)
    git('add', '.')
    git('commit', '-m', 'fixture')
    output = tmp_path / 'build'
    result = subprocess.run([
        sys.executable,
        str(SOURCE), 'build', '--source',
        str(product), '--output',
        str(output)
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    receipt = json.loads((output / 'build.json').read_text())
    assert receipt['revision'] == git('rev-parse', 'HEAD')
    assert receipt['wheel_sha256'] == hashlib.sha256(
        (output / 'candidate.whl').read_bytes()).hexdigest()
