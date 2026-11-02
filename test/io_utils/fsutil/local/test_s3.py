# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

import boto3
import pytest

from io_utils import fsutil
from io_utils.fsutil import impl_s3, s3_filesystem
from io_utils.fsutil.s3_connection import S3Connection
from io_utils.fsutil.uri import S3URI
from kumo.testing.s3 import withS3Credentials
from service_framework._raw_deploy_config import S3AddressingStyle


def test_is_s3():
    assert fsutil.is_s3('s3://bucket')
    assert fsutil.is_s3('s3://bucket/prefix')
    assert not fsutil.is_s3('/home/tester')
    assert not fsutil.is_s3('../tester/path')


def test_parse_s3():
    assert s3_filesystem.parse_s3('s3://bucket/prefix') == ('bucket', 'prefix')
    assert s3_filesystem.parse_s3('s3://bucket2/path1/path2/file') == (
        'bucket2', 'path1/path2/file')


def test_listdir_recursive_streams_discovered_prefixes(monkeypatch):
    slow_release = Event()
    deep_started = Event()

    def list_objects(_bucket, prefix):
        if prefix == 'root/':
            return s3_filesystem._ListResponse(prefixes=['fast/', 'slow/'])
        if prefix == 'fast/':
            return s3_filesystem._ListResponse(prefixes=['deep/'])
        if prefix == 'slow/':
            assert slow_release.wait(5.0)
            return s3_filesystem._ListResponse(keys=['slow/key'])
        if prefix == 'deep/':
            deep_started.set()
            return s3_filesystem._ListResponse(keys=['deep/key'])
        raise AssertionError(prefix)

    monkeypatch.setattr(s3_filesystem, '_list_objects_v2', list_objects)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(s3_filesystem.listdir_recursive,
                                 's3://bucket/root/')
        try:
            assert deep_started.wait(2.0)
        finally:
            slow_release.set()
        assert set(result.result()) == {'deep/key', 'slow/key'}


def _assert_sigv4(url: str, *, expires_in: int | None = None) -> None:
    query = parse_qs(urlsplit(url).query)
    assert query['X-Amz-Algorithm'] == ['AWS4-HMAC-SHA256']
    assert query['X-Amz-SignedHeaders'] == ['host']
    if expires_in is not None:
        assert query['X-Amz-Expires'] == [str(expires_in)]
    assert 'AWSAccessKeyId' not in query
    assert 'Signature' not in query


def test_presigned_urls_use_sigv4(monkeypatch):
    boto = s3_filesystem.ThreadLocalBotoSession()
    boto._session = boto3.session.Session(
        aws_access_key_id='test-access-key',
        aws_secret_access_key='test-secret-key',
        aws_session_token='test-session-token',
        region_name='us-west-2',
    )
    monkeypatch.setattr(s3_filesystem, 'active_s3_connection', lambda: None)
    monkeypatch.setattr(s3_filesystem, 'threadsafe_boto', boto)
    monkeypatch.setattr(impl_s3, 'threadsafe_boto', boto)
    boto._bucket_regions['test-bucket'] = 'us-west-2'
    boto._region = 'us-west-2'
    client = boto.s3_client()
    client.create_multipart_upload = Mock(
        return_value={
            'ResponseMetadata': {
                'HTTPStatusCode': 200,
            },
            'UploadId': 'test-upload-id',
        })
    uri = S3URI(bucket='test-bucket', path='uploads/data.csv')

    _, part_urls = impl_s3.S3PresignedHelper(uri)._create_multipart_upload(
        uri, num_parts=1)
    download_url = s3_filesystem.generate_presigned_url_for_download(
        uri.as_s3())

    _assert_sigv4(part_urls[1], expires_in=3600)
    _assert_sigv4(download_url)


def test_presigned_upload_selects_ambient_bucket_region(monkeypatch):
    initial_client = Mock()
    initial_client.head_bucket.return_value = {'BucketRegion': 'us-west-2'}
    initial_client.create_multipart_upload.side_effect = AssertionError(
        'presigning must use a client in the bucket region')
    regional_client = Mock()
    regional_client.create_multipart_upload.return_value = {
        'ResponseMetadata': {
            'HTTPStatusCode': 200,
        },
        'UploadId': 'test-upload-id',
    }
    regional_client.generate_presigned_url.return_value = 'presigned-url'
    boto = Mock(
        _bucket_regions={},
        _endpoint_url=None,
        _region=None,
        s3_client=Mock(side_effect=[initial_client, regional_client]),
    )
    monkeypatch.setattr(s3_filesystem, 'active_s3_connection', lambda: None)
    monkeypatch.setattr(s3_filesystem, 'threadsafe_boto', boto)
    monkeypatch.setattr(impl_s3, 'threadsafe_boto', boto)
    uri = S3URI(bucket='test-bucket', path='uploads/data.csv')

    upload_id, part_urls = (
        impl_s3.S3PresignedHelper(uri)._create_multipart_upload(
            uri, num_parts=1))

    assert upload_id == 'test-upload-id'
    assert part_urls == {1: 'presigned-url'}
    assert boto._region == 'us-west-2'
    boto._reset.assert_called_once_with()


def test_complete_multipart_upload_lists_all_part_pages(monkeypatch):
    client = Mock()
    part_pages = [{'Parts': range(1000)}, {'Parts': range(24)}]
    client.get_paginator.return_value.paginate.return_value = part_pages
    client.complete_multipart_upload.return_value = {
        'ResponseMetadata': {
            'HTTPStatusCode': 200
        },
    }
    monkeypatch.setattr(
        impl_s3,
        'threadsafe_boto',
        Mock(s3_client=Mock(return_value=client)),
    )
    uri = S3URI(bucket='test-bucket', path='uploads/data.csv')
    parts = [Mock(part_number=i, etag=f'etag-{i}') for i in range(1, 1025)]

    impl_s3.S3PresignedHelper(uri)._complete_multipart_upload(
        uri, 'test-upload-id', parts)

    assert len(
        client.complete_multipart_upload.call_args.kwargs['MultipartUpload']
        ['Parts']) == 1024


def test_presigned_url_keeps_minio_config(monkeypatch):
    connection = S3Connection(
        endpoint_url='https://minio.example.test:9000',
        region='us-east-1',
        addressing_style=S3AddressingStyle.PATH,
        credential_secret_name='MINIO_CREDENTIALS',
        access_key='minio-access-key',
        secret_key='minio-secret-key',
        session_token=None,
        ca_bundle_path='/etc/kumo/minio-ca/ca.crt',
        send_bucket_owner_full_control=False,
    )
    monkeypatch.setattr(s3_filesystem, 'active_s3_connection',
                        lambda: connection)
    client = s3_filesystem.ThreadLocalBotoSession().s3_client()

    url = client.generate_presigned_url(
        'upload_part',
        Params={
            'Bucket': 'test-bucket',
            'Key': 'uploads/data.csv',
            'UploadId': 'test-upload-id',
            'PartNumber': 1,
        },
        HttpMethod='PUT',
        ExpiresIn=600,
    )

    parsed = urlsplit(url)
    assert parsed.netloc == 'minio.example.test:9000'
    assert parsed.path == '/test-bucket/uploads/data.csv'
    _assert_sigv4(url)


def test_zip_files_passes_exact_keys_without_prefix_listing(monkeypatch):
    zipper = Mock(return_value='s3://bucket/archive.zip')
    monkeypatch.setattr(impl_s3, 'zip_files_with_same_prefix', zipper)

    output = fsutil.zip_files(
        ['s3://bucket/root/part-1', 's3://bucket/root/part-3'],
        's3://bucket/archive.zip',
    )

    assert output == 's3://bucket/archive.zip'
    zipper.assert_called_once_with(
        's3://bucket/root/part-1',
        output_path='s3://bucket/archive.zip',
        object_keys=['root/part-1', 'root/part-3'],
    )


def test_s3_credentials_decorator_does_not_probe_provider(monkeypatch, tmp_path):
    credential_env = (
        'AWS_ACCESS_KEY_ID',
        'AWS_ACCESS_KEY',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_SECRET_KEY',
        'AWS_ROLE_ARN',
        'AWS_WEB_IDENTITY_TOKEN_FILE',
        'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI',
        'AWS_CONTAINER_CREDENTIALS_FULL_URI',
        'AWS_SHARED_CREDENTIALS_FILE',
    )
    for name in credential_env:
        monkeypatch.delenv(name, raising=False)

    def test_target():
        pass

    # An existing developer credential file must not affect this no-credentials case.
    monkeypatch.setenv('AWS_SHARED_CREDENTIALS_FILE', str(tmp_path / 'missing'))
    with patch.object(s3_filesystem.threadsafe_boto, 's3_client') as client:
        list_objects = client.return_value.list_objects_v2
        decorated = withS3Credentials(test_target)
        module_mark = withS3Credentials()

    client.assert_not_called()
    list_objects.assert_not_called()
    skip_marks = [
        mark for mark in decorated.pytestmark if mark.name == 'skipif'
    ]
    assert len(skip_marks) == 1
    assert skip_marks[0].kwargs['reason'] == 'No AWS credentials found for S3'
    assert module_mark.mark.name == 'skipif'
    assert module_mark.mark.args == (True, )
    assert module_mark.mark.kwargs[
        'reason'] == 'No AWS credentials found for S3'


def test_object_identity_uses_etag_without_version_permission(monkeypatch):
    from io_utils.fsutil import impl_s3

    monkeypatch.setattr(
        impl_s3,
        '_read_with_unsigned_fallback',
        lambda _operation: {
            'ContentLength': 123,
            'ETag': '"etag-value"',
            'VersionId': 'must-not-be-required',
        },
    )
    identity = fsutil.object_identity('s3://bucket/path/file.parquet')

    assert identity == fsutil.ObjectIdentity(
        size=123,
        etag='"etag-value"',
        version=None,
    )


def test_object_identity_maps_head_404_to_file_not_found(monkeypatch):
    from botocore.exceptions import ClientError

    from io_utils.fsutil import impl_s3

    error = ClientError(
        {'Error': {
            'Code': '404',
            'Message': 'not found'
        }},
        'HeadObject',
    )
    monkeypatch.setattr(
        impl_s3,
        '_read_with_unsigned_fallback',
        lambda _operation: (_ for _ in ()).throw(error),
    )

    with pytest.raises(FileNotFoundError):
        fsutil.object_identity('s3://bucket/missing.parquet')


def test_s3_listdir_detail_paginates_and_preserves_provider_types(monkeypatch):
    client = Mock()
    paginator = client.get_paginator.return_value
    paginator.paginate.return_value = [{
        'Contents': [{
            'Key': 'events/part.parquet'
        }],
        'CommonPrefixes': [{
            'Prefix': 'events/year=2026/'
        }],
    }]
    monkeypatch.setattr(
        impl_s3,
        '_read_with_unsigned_fallback',
        lambda operation: operation(client),
    )

    details = impl_s3._listdir_details(
        S3URI(bucket='customer-data', path='events'))

    assert [(entry.path, entry.is_directory) for entry in details] == [
        ('s3://customer-data/events/part.parquet', False),
        ('s3://customer-data/events/year=2026', True),
    ]
    paginator.paginate.assert_called_once_with(
        Bucket='customer-data',
        Prefix='events/',
        Delimiter='/',
    )
