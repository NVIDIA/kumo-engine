# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage source-only weekly inputs through the public preparation command."""

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_nightly_reference_inputs as fixtures

SCRIPTS = Path(__file__).parents[1] / 'scripts'
SPEC = importlib.util.spec_from_file_location('source_staging_job',
                                              SCRIPTS / 'nightly_job.py')
job = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(job)


@pytest.fixture
def source_only(tmp_path, monkeypatch):
    root, _, manifest = fixtures.source_inputs.__wrapped__(tmp_path)
    manifest['scale'] = '10B'
    for case in manifest['cases']:
        case['scale'] = '10B'
        for key in ('prediction_input', 'normal_rows', 'control_rows'):
            case.pop(key, None)
        if case['arm'] == 'control':
            case['descriptor'] = case['descriptor'].replace(
                '/control/', '/normal/')
    manifest['files'] = {
        name: sha
        for name, sha in manifest['files'].items() if (root / name).is_file()
    }
    payload = b'Pinned source table'
    inventory = root / 'source-objects/adtech.json'
    inventory.parent.mkdir()
    inventory.write_text(
        json.dumps(
            dict(objects=[
                dict(path='raw/adtech/part.parquet', bucket='raw-fixtures',
                     region='us-west-2', key='source/adtech.parquet',
                     sha256=hashlib.sha256(payload).hexdigest(), bytes=len(
                         payload))
            ])))
    manifest['source_objects'] = {'adtech': 'source-objects/adtech.json'}
    manifest['files']['source-objects/adtech.json'] = fixtures.nightly.digest(
        inventory)
    fixtures.nightly.save(root / 'manifest.json', manifest)
    archive = tmp_path / 'metadata.tar'
    names = {
        'schema.json', 'dependencies.json', 'source-objects/adtech.json',
        'cases/binary/normal/query.json'
    }
    with tarfile.open(archive, 'w') as output:
        for name in sorted(names):
            output.add(root / name, arcname=name)
    objects = {
        ('fixtures', 'metadata.tar'): archive.read_bytes(),
        ('fixtures', 'manifest.json'): (root / 'manifest.json').read_bytes(),
        ('raw-fixtures', 'source/adtech.parquet'): payload
    }

    def binding(key):
        return dict(bucket='fixtures', region='us-west-2', key=key,
                    sha256=hashlib.sha256(objects['fixtures', key]).hexdigest())

    config = dict(
        format='kumo-engine-weekly-job-v2', min_free_bytes=1024,
        reference_identity=dict(version='1.0.0', wheel_sha256='c' * 64),
        scales={
            '10B':
            dict(
                inputs=dict(
                    manifest=binding('manifest.json'), by_graph={
                        family: binding('metadata.tar')
                        for family in job.WORKLOADS
                    }),
                input_manifest_sha256=binding('manifest.json')['sha256']),
            '25B':
            dict(inputs=None, input_manifest_sha256=None,
                 pending='Not used by this fixture')
        })
    for name in ('reference', 'policy', 'requirements'):
        path = root / (name + '.json')
        path.write_text('{}')
        config[name] = dict(path=path.name,
                            sha256=fixtures.nightly.digest(path))

    class S3:
        def download_file(self, bucket, key, destination):
            Path(destination).write_bytes(objects[bucket, key])

        def get_object(self, Bucket, Key):
            return dict(Body=io.BytesIO(objects[Bucket, Key]))

    monkeypatch.setitem(sys.modules, 'boto3',
                        SimpleNamespace(client=lambda *args, **kwargs: S3()))
    work = tmp_path / 'work'
    work.mkdir()
    return root, config, manifest, objects, work


def test_source_only_prepare_fetches_raw_for_only_the_selected_family(
        source_only):
    root, config, _, _, work = source_only
    assert job.configuration(json.dumps(config)) == config
    job.prepare(config, work, scale='10B', local_root=root,
                case='binary-normal')
    actual = work / 'inputs/raw/adtech/part.parquet'
    assert actual.read_bytes() == b'Pinned source table'
    assert not actual.stat().st_mode & 0o222
    assert not (work / 'inputs/raw/delivery').exists()
    loaded = fixtures.nightly.load_inputs(
        work / 'inputs', config['scales']['10B']['input_manifest_sha256'],
        '10B', config, selected_case='binary-normal', reference_only=True)
    assert loaded['scale'] == '10B'


def test_source_only_prepare_rejects_changed_remote_bytes(source_only):
    root, config, _, objects, work = source_only
    objects['raw-fixtures', 'source/adtech.parquet'] = b'Changed table'
    with pytest.raises(ValueError, match='checksum'):
        job.prepare(config, work, scale='10B', local_root=root,
                    case='binary-normal')
    assert not (work / 'inputs/raw/adtech/part.parquet').exists()


def test_source_only_archive_cannot_smuggle_undeclared_files(source_only):
    root, config, _, objects, work = source_only
    original = objects['fixtures', 'metadata.tar']
    contents = io.BytesIO(original)
    with tarfile.open(fileobj=contents, mode='a') as output:
        member = tarfile.TarInfo('surprise.json')
        member.size = 2
        output.addfile(member, io.BytesIO(b'{}'))
    objects['fixtures', 'metadata.tar'] = contents.getvalue()
    checksum = hashlib.sha256(contents.getvalue()).hexdigest()
    config['scales']['10B']['inputs']['by_graph']['adtech']['sha256'] = checksum
    with pytest.raises(ValueError, match='missing or extra'):
        job.prepare(config, work, scale='10B', local_root=root,
                    case='binary-normal')
    assert not (work / 'inputs').exists()


def test_raw_stream_cannot_exceed_the_pinned_length(source_only):
    root, config, _, objects, work = source_only
    objects['raw-fixtures', 'source/adtech.parquet'] *= 2
    with pytest.raises(ValueError, match='length exceeds'):
        job.prepare(config, work, scale='10B', local_root=root,
                    case='binary-normal')
    assert list((work / 'inputs/raw/adtech').iterdir()) == []


def test_inventory_cannot_replace_manifest_raw_checksum(source_only):
    root, _, manifest, _, _ = source_only
    path = root / 'source-objects/adtech.json'
    record = json.loads(path.read_text())
    record['objects'][0]['sha256'] = '0' * 64
    path.write_text(json.dumps(record))
    manifest['files']['source-objects/adtech.json'] = fixtures.nightly.digest(
        path)
    sources = job.load_script(SCRIPTS / 'nightly_sources.py')
    with pytest.raises(ValueError, match='source identity'):
        sources.stage(root, manifest, 'adtech', input_path=job.input_path)


def test_prepare_records_source_transfer_costs(source_only):
    root, config, _, _, work = source_only
    job.prepare(config, work, scale='10B', local_root=root,
                case='binary-normal')
    report = json.loads((work / 'source-staging.json').read_text())
    assert report['files'] == 1 and report['bytes'] == 19
    assert report['family'] == 'adtech' and report['case'] == 'binary-normal'
    assert report['seconds'] >= 0
