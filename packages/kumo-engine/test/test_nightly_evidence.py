# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference publication must finish before fitting and preserve old proposals."""

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

SOURCE = Path(__file__).parents[1] / 'scripts/nightly_evidence.py'


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('evidence_tested', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / 'case'
    (root / 'reference-data').mkdir(parents=True)
    (root / 'reference-data/answers.parquet').write_bytes(b'proposed answers')
    (root / 'current-population.parquet').write_bytes(b'current IDs')
    (root /
     'reference-data/reference-data.json').write_text('{"status":"PREPARED"}')

    class S3:
        def __init__(self):
            self.objects, self.actions, self.parts = {}, [], {}
            self.failure = self.corrupt = None

        def put_object(self, *, Bucket, Key, Body, IfNoneMatch, **options):
            assert Bucket == 'test-evidence' and IfNoneMatch == '*'
            if Key in self.objects:
                raise ClientError({'Error': {
                    'Code': 'PreconditionFailed'
                }}, 'PutObject')
            self.actions.append(('put', Key))
            if self.failure and self.failure in Key:
                raise OSError('S3 upload failed')
            self.objects[Key] = Body.read() if hasattr(Body, 'read') else Body
            return {'VersionId': 'version-1'}

        def get_object(self, *, Bucket, Key):
            assert Bucket == 'test-evidence'
            data = self.objects[Key]
            if self.corrupt and self.corrupt in Key:
                data += b'corruption'
            self.actions.append(('read', Key))
            return dict(Body=io.BytesIO(data), VersionId='version-1')

        def create_multipart_upload(self, *, Bucket, Key, **options):
            assert Bucket == 'test-evidence'
            self.parts = {}
            self.actions.append(('start', Key))
            return dict(UploadId='upload-1')

        def upload_part(self, *, Bucket, Key, Body, PartNumber, UploadId):
            assert Bucket == 'test-evidence' and UploadId == 'upload-1'
            if self.failure:
                raise OSError('S3 multipart failed')
            self.parts[PartNumber] = Body
            return dict(ETag=str(PartNumber))

        def complete_multipart_upload(self, *, Bucket, Key, MultipartUpload,
                                      UploadId, IfNoneMatch):
            return self.put_object(Bucket=Bucket, Key=Key,
                                   IfNoneMatch=IfNoneMatch,
                                   Body=b''.join(self.parts.values()))

        def abort_multipart_upload(self, **options):
            self.actions.append(('abort', options['Key']))

    client = S3()
    monkeypatch.setitem(sys.modules, 'boto3',
                        SimpleNamespace(client=lambda name: client))
    return module, root, client


def test_publish_reads_back_every_payload_before_committing_manifest(evidence):
    module, root, s3 = evidence
    receipt = module.publish(root, 's3://test-evidence/reference-proposals')
    manifest_key = receipt['uri'].removeprefix('s3://test-evidence/')
    manifest = json.loads(s3.objects[manifest_key])
    assert receipt['status'] == 'PREPARED_IN_S3'
    assert receipt['sha256'] == hashlib.sha256(
        s3.objects[manifest_key]).hexdigest()
    assert receipt['version_id'] == 'version-1'
    assert set(manifest['files']) == {
        'reference-data/answers.parquet', 'current-population.parquet',
        'reference-data/reference-data.json'
    }
    final_put = s3.actions.index(('put', manifest_key))
    for name, item in manifest['files'].items():
        assert s3.objects[item['key']] == (root / name).read_bytes()
        assert ('read', item['key']) in s3.actions[:final_put]
    assert [key for action, key in s3.actions
            if action == 'put'][-1] == manifest_key
    assert s3.actions[-1] == ('read', manifest_key)


def test_retry_preserves_old_proposal_and_reuses_identical_objects(evidence):
    module, root, s3 = evidence
    first = module.publish(root, 's3://test-evidence/reference-proposals')
    old = dict(s3.objects)
    second = module.publish(root, 's3://test-evidence/reference-proposals')
    assert first == second and s3.objects == old
    (root / 'reference-data/answers.parquet').write_bytes(b'new answers')
    third = module.publish(root, 's3://test-evidence/reference-proposals')
    assert third['uri'] != first['uri']
    assert all(s3.objects[key] == value for key, value in old.items())


@pytest.mark.parametrize('failure', ['upload', 'readback'])
def test_failed_payload_never_publishes_a_complete_manifest(evidence, failure):
    module, root, s3 = evidence
    if failure == 'upload':
        s3.failure = 'answers.parquet'
    else:
        s3.corrupt = 'answers.parquet'
    with pytest.raises((OSError, ValueError), match='upload|checksum'):
        module.publish(root, 's3://test-evidence/reference-proposals')
    assert not any(key.endswith('/manifest.json') for key in s3.objects)


def test_changed_existing_object_is_not_overwritten(evidence):
    module, root, s3 = evidence
    first = module.publish(root, 's3://test-evidence/reference-proposals')
    manifest = json.loads(
        s3.objects[first['uri'].removeprefix('s3://test-evidence/')])
    key = manifest['files']['reference-data/answers.parquet']['key']
    s3.objects[key] = b'changed remote bytes'
    with pytest.raises(ValueError, match='checksum'):
        module.publish(root, 's3://test-evidence/reference-proposals')
    assert s3.objects[key] == b'changed remote bytes'


@pytest.mark.parametrize('destination', [
    'https://test-evidence/a',
    's3://test-evidence',
    's3://test-evidence/../approved',
    's3://test-evidence/a//b',
    's3://test-evidence/a?query=1',
    's3://test-evidence/a#fragment',
])
def test_bad_destination_is_rejected_before_cloud_writes(evidence, destination):
    module, root, s3 = evidence
    with pytest.raises(ValueError, match='S3'):
        module.publish(root, destination)
    assert not s3.objects


@pytest.mark.parametrize('bad_file', ['symbolic-link', 'program.py'])
def test_publication_rejects_non_data_files(evidence, bad_file):
    module, root, s3 = evidence
    if bad_file == 'symbolic-link':
        (root / 'link.parquet').symlink_to(root / 'current-population.parquet')
    else:
        (root / bad_file).write_text('print("not evidence")')
    with pytest.raises(ValueError, match='regular|data'):
        module.publish(root, 's3://test-evidence/reference-proposals')
    assert not s3.objects


def test_large_file_uses_multipart_and_still_verifies_readback(
        evidence, monkeypatch):
    module, root, s3 = evidence
    monkeypatch.setattr(module, 'SINGLE_PUT_BYTES', 4)
    receipt = module.publish(root, 's3://test-evidence/reference-proposals')
    assert receipt['status'] == 'PREPARED_IN_S3'
    assert any(action == 'start' for action, _ in s3.actions)
    assert not any(action == 'abort' for action, _ in s3.actions)


def test_failed_multipart_is_aborted(evidence, monkeypatch):
    module, root, s3 = evidence
    monkeypatch.setattr(module, 'SINGLE_PUT_BYTES', 4)
    s3.failure = 'multipart'
    with pytest.raises(OSError, match='multipart'):
        module.publish(root, 's3://test-evidence/reference-proposals')
    assert any(action == 'abort' for action, _ in s3.actions)
    assert not any(key.endswith('/manifest.json') for key in s3.objects)
