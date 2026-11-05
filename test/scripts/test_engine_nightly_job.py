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
import tarfile
from pathlib import Path

import pytest

SOURCE = Path(
    __file__).parents[2] / 'packages/kumo-engine/scripts/nightly_job.py'
SPEC = importlib.util.spec_from_file_location('nightly_job', SOURCE)
job = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(job)


def config():
    return dict(
        format='kumo-engine-nightly-job-v2', min_free_bytes=1024,
        reference_identity=dict(version='1.0.0', wheel_sha256='c' * 64),
        scales={
            '100M':
            dict(
                input_manifest_sha256='a' * 64,
                inputs=dict(bucket='shared-fixtures', region='us-west-2',
                            key='engine/nightly/inputs', sha256='b' * 64)),
            '1B':
            dict(input_manifest_sha256=None, inputs=None,
                 pending='Await a frozen 1B archive')
        }, **{
            key: dict(path=key + '.json', sha256='b' * 64)
            for key in ('reference', 'policy', 'requirements')
        })


def test_configuration_rejects_missing_and_unsafe_object_bindings():
    value = config()
    assert job.configuration(json.dumps(value)) == value
    for key, damage in [('min_free_bytes', 0), ('scales', {}),
                        ('reference_identity', {}), ('policy', None)]:
        with pytest.raises(ValueError):
            job.configuration(json.dumps(dict(value, **{key: damage})))
    value['scales']['100M']['inputs']['key'] = '../other'
    with pytest.raises(ValueError):
        job.configuration(json.dumps(value))


def test_compact_records_require_local_bindings_but_archive_can_use_s3():
    value = config()
    remote = value['scales']['100M']['inputs']
    value['archive'] = remote
    assert job.configuration(json.dumps(value)) == value
    for name in ('reference', 'policy', 'requirements'):
        with pytest.raises(ValueError, match='relative path'):
            job.configuration(json.dumps(dict(value, **{name: remote})))


def variants():
    return dict(
        streaming=dict(task='temporal_link', graph='streaming',
                       metric='recall@50', query='PREDICT streaming links',
                       top_k=50, raw_limit=51),
        community=dict(task='regression', graph='community', metric='mae',
                       query='PREDICT community outcomes'),
        payments=dict(task='binary', graph='payments', metric='auroc',
                      query='PREDICT payment outcomes', min_hops=4, max_hops=6),
        fraud=dict(task='binary', graph='fraud', metric='auroc',
                   query='PREDICT fraud outcomes', min_hops=4, max_hops=6))


def test_configuration_accepts_all_four_workload_variants():
    value = dict(config(), variants=variants())
    assert job.configuration(json.dumps(value)) == value


@pytest.mark.parametrize(
    'damage',
    [{
        'streaming': None
    }, {
        'other': dict(task='binary', graph='fraud', metric='auroc', query='x')
    }, {
        'streaming': dict(task='binary')
    }, {
        'community': dict(query='')
    }, {
        'payments': dict(min_hops=0)
    }, {
        'payments': dict(max_hops=3)
    }, {
        'fraud': dict(min_hops=3)
    }, {
        'streaming': dict(top_k=True)
    }, {
        'streaming': dict(top_k=50.0)
    }, {
        'streaming': dict(raw_limit=50)
    }])
def test_configuration_rejects_incomplete_or_inconsistent_variants(damage):
    records = variants()
    for name, fields in damage.items():
        if fields is None:
            records.pop(name)
        elif name not in records:
            records[name] = fields
        else:
            records[name].update(fields)
    with pytest.raises(ValueError, match='variant'):
        job.configuration(json.dumps(dict(config(), variants=records)))


def test_prepare_rejects_pending_scale_before_downloading(tmp_path):
    with pytest.raises(ValueError, match='1B.*pending'):
        job.prepare(config(), tmp_path, scale='1B')
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('linked', [False, True])
def test_prepare_preserves_an_existing_input_destination(tmp_path, linked):
    work = tmp_path / 'work'
    work.mkdir()
    destination = tmp_path / 'existing-inputs'
    destination.mkdir()
    (destination / 'keep.json').write_text('preserve')
    if linked:
        (work / 'inputs').symlink_to(destination, target_is_directory=True)
    else:
        destination.rename(work / 'inputs')
        destination = work / 'inputs'
    value = config()
    value['scales']['100M']['inputs'] = dict(path='absent.tar', sha256='b' * 64)
    with pytest.raises(FileExistsError):
        job.prepare(value, work, local_root=tmp_path)
    assert (destination / 'keep.json').read_text() == 'preserve'


def workload_manifest():
    names = [
        'graph-schema.json',
        'reference-dependencies.json',
        'raw/adtech/events/part.parquet',
        'raw/commerce/events/part.parquet',
        'source-manifests/adtech.json',
        'source-manifests/commerce.json',
    ]
    cases = []
    for label, graph in [('binary', 'adtech'),
                         ('static_link', 'commerce-static'),
                         ('temporal_link', 'commerce')]:
        for leaf in ('query.json', 'population.parquet', 'normal.parquet',
                     'control.parquet'):
            names.append(f'cases/{label}/{leaf}')
        for arm in ('normal', 'control'):
            cases.append(
                dict(
                    task=label, arm=arm, seed=0, graph=graph,
                    descriptor=f'cases/{label}/query.json',
                    prediction_input=f'cases/{label}/population.parquet',
                    normal_rows=[f'cases/{label}/normal.parquet'],
                    **({
                        'control_rows': [f'cases/{label}/control.parquet']
                    } if arm == 'control' else {})))
    graphs = {
        name:
        dict(profile=profile, schema='graph-schema.json',
             sources={'events': f'raw/{profile}/events'})
        for name, profile in [('adtech', 'adtech'), (
            'commerce', 'commerce'), ('commerce-static', 'commerce')]
    }
    return dict(
        format='kumo-engine-nightly-inputs-v2', scale='100M', files={
            name: hashlib.sha256(name.encode()).hexdigest()
            for name in names
        }, cases=cases, graphs=graphs,
        reference_dependencies='reference-dependencies.json')


def test_workload_selection_keeps_both_commerce_views_without_other_data():
    manifest = workload_manifest()
    assert job.workload_for_case(manifest, 'static_link-control') == 'commerce'
    assert job.workload_for_case(manifest, 'temporal_link-normal') == 'commerce'
    assert job.workload_files(manifest, 'commerce') == {
        'graph-schema.json',
        'reference-dependencies.json',
        'raw/commerce/events/part.parquet',
        'source-manifests/commerce.json',
        'cases/static_link/query.json',
        'cases/static_link/population.parquet',
        'cases/static_link/normal.parquet',
        'cases/static_link/control.parquet',
        'cases/temporal_link/query.json',
        'cases/temporal_link/population.parquet',
        'cases/temporal_link/normal.parquet',
        'cases/temporal_link/control.parquet',
    }
    with pytest.raises(ValueError):
        job.workload_for_case(manifest, 'absent-normal')


@pytest.mark.parametrize('bad', [
    '../escape.json', '/absolute.json', 'raw/../escape.json', 'raw\\escape.json'
])
def test_workload_selection_rejects_unsafe_inactive_file_names(bad):
    manifest = workload_manifest()
    manifest['files'][bad] = 'a' * 64
    with pytest.raises(ValueError):
        job.workload_files(manifest, 'adtech')


def test_workload_configuration_requires_complete_safe_archive_bindings():
    value = config()
    value['scales']['100M']['workloads'] = {
        family:
        dict(bucket='shared-fixtures', region='us-west-2',
             key=f'workloads/{family}/inputs.tar', sha256='d' * 64)
        for family in ('adtech', 'delivery', 'commerce', 'streaming',
                       'community', 'payments', 'fraud')
    }
    assert job.configuration(json.dumps(value)) == value
    value['scales']['100M']['workloads']['fraud']['key'] = '../escape'
    with pytest.raises(ValueError):
        job.configuration(json.dumps(value))
    value['scales']['100M']['workloads'].pop('fraud')
    with pytest.raises(ValueError):
        job.configuration(json.dumps(value))


@pytest.mark.parametrize(
    'damage',
    [None, 'extra', 'missing', 'duplicate', 'wrong_manifest', 'link', 'escape'])
def test_prepare_downloads_only_the_selected_workload_archive(tmp_path, damage):
    repository = tmp_path / 'repository'
    repository.mkdir()
    manifest = workload_manifest()
    manifest_bytes = (json.dumps(manifest) + '\n').encode()
    family_paths = {
        'graph-schema.json',
        'reference-dependencies.json',
        'raw/adtech/events/part.parquet',
        'source-manifests/adtech.json',
        'cases/binary/query.json',
        'cases/binary/population.parquet',
        'cases/binary/normal.parquet',
        'cases/binary/control.parquet',
    }
    import io
    archive = repository / 'adtech.tar'
    names = ['manifest.json', *sorted(family_paths)]
    if damage == 'extra':
        names.append('raw/commerce/events/part.parquet')
    elif damage == 'missing':
        names.remove('raw/adtech/events/part.parquet')
    elif damage == 'duplicate':
        names.append('cases/binary/query.json')
    elif damage == 'escape':
        names.append('../outside.json')
    with tarfile.open(archive, 'w') as output:
        for name in names:
            body = manifest_bytes if name == 'manifest.json' else name.encode()
            if name == 'manifest.json' and damage == 'wrong_manifest':
                body = b'{}'
            member = tarfile.TarInfo(name)
            if name == 'cases/binary/query.json' and damage == 'link':
                member.type = tarfile.SYMTYPE
                member.linkname = '../outside.json'
                output.addfile(member)
                continue
            member.size = len(body)
            output.addfile(member, io.BytesIO(body))
    value = config()
    value['scales']['100M'].update(
        input_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        inputs=dict(path='whole-bundle-that-must-not-be-read.tar',
                    sha256='b' * 64),
        workloads={
            'adtech':
            dict(path='adtech.tar',
                 sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        })
    for name in ('reference', 'policy', 'requirements'):
        path = repository / (name + '.json')
        path.write_text('{}')
        value[name] = dict(path=path.name,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    work = tmp_path / 'work'
    work.mkdir()
    if damage:
        with pytest.raises(ValueError):
            job.prepare(value, work, scale='100M', local_root=repository,
                        case='binary-normal')
        assert not (tmp_path / 'outside.json').exists()
        return
    with pytest.raises(ValueError, match='Select a case'):
        job.prepare(value, work, scale='100M', local_root=repository)
    job.prepare(value, work, scale='100M', local_root=repository,
                case='binary-normal')
    assert (work / 'inputs/manifest.json').read_bytes() == manifest_bytes
    assert {
        p.relative_to(work / 'inputs').as_posix()
        for p in (work / 'inputs').rglob('*') if p.is_file()
    } == (family_paths | {'manifest.json'})
    assert not (work / 'inputs/raw/commerce').exists()


def test_create_exports_the_selected_scale_identity(tmp_path):
    value = config()
    value['scales']['1B'] = dict(value['scales']['100M'],
                                 input_manifest_sha256='d' * 64)
    environment_file = tmp_path / 'environment'
    result = subprocess.run(
        [sys.executable, str(SOURCE), 'create', '--scale', '1B'],
        env=dict(os.environ, RUNNER_TEMP=str(tmp_path),
                 GITHUB_ENV=str(environment_file), NIGHTLY_SCALE='100M',
                 NIGHTLY_CONFIGURATION=json.dumps(value)), capture_output=True,
        text=True)
    assert result.returncode == 0, result.stderr
    exported = dict(
        line.split('=', 1)
        for line in environment_file.read_text().splitlines())
    assert exported['NIGHTLY_SCALE'] == '1B'
    assert exported['NIGHTLY_MANIFEST_SHA256'] == 'd' * 64
    assert exported['NIGHTLY_REQUIREMENTS_SHA256'] == 'b' * 64


def test_prepare_stages_proposed_records_without_requiring_approved_numbers(
        tmp_path):
    repository = tmp_path / 'repository'
    repository.mkdir()
    manifest = repository / 'manifest.json'
    manifest.write_text('{"scale":"100M"}')
    archive_path = repository / 'input.tar'
    with tarfile.open(archive_path, 'w') as archive:
        archive.add(manifest, arcname='manifest.json')
    value = config()
    value['scales']['100M']['inputs'] = dict(
        path='input.tar',
        sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest())
    for name in ('reference', 'policy', 'requirements'):
        record = repository / (name + '.json')
        record.write_text('{"approved":false,"value":null}')
        value[name] = dict(
            path=record.name,
            sha256=hashlib.sha256(record.read_bytes()).hexdigest())
    work = tmp_path / 'work'
    work.mkdir()
    job.prepare(value, work, scale='100M', local_root=repository)
    assert (work / 'inputs/manifest.json').read_bytes() == manifest.read_bytes()
    for name in ('reference', 'policy', 'requirements'):
        assert (work / (name + '.json')).read_text() == (
            '{"approved":false,"value":null}')
    assert not (work / 'inputs.tar').exists()


@pytest.mark.parametrize('command', ['run', 'combine'])
def test_shared_commands_preserve_reference_mode_without_candidate_receipt(
        tmp_path, command):
    caller = tmp_path / 'nightly_job.py'
    shutil.copyfile(SOURCE, caller)
    (tmp_path / 'nightly_check.py').write_text(
        'import json\ndef main(argv):\n print(json.dumps(argv)); return 7\n')
    args = [command, '--mode', 'reference', '--scale', '100M']
    result = subprocess.run([sys.executable, str(caller), *args],
                            capture_output=True, text=True)
    assert result.returncode == 7, result.stderr
    forwarded = json.loads(result.stdout)
    assert forwarded[:len(args)] == args
    assert forwarded[len(args):] == [
        '--configuration',
        str(caller.parents[1] / 'test/nightly/inputs.json')
    ]


def test_shared_command_preserves_an_explicit_configuration(tmp_path):
    caller = tmp_path / 'nightly_job.py'
    shutil.copyfile(SOURCE, caller)
    (tmp_path / 'nightly_check.py').write_text(
        'import json\ndef main(argv):\n print(json.dumps(argv)); return 0\n')
    args = [
        'run', '--configuration',
        str(tmp_path / 'chosen.json'), '--mode', 'candidate', '--build-receipt',
        'build.json'
    ]
    result = subprocess.run([sys.executable, str(caller), *args],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == args


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
