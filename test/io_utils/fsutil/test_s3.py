# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor, as_completed
from os import path as osp
from threading import Event
from typing import Generator, List
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

import boto3
import pytest

from io_utils import fsutil
from io_utils.fsutil import impl_s3, s3_filesystem
from io_utils.fsutil.common import fsutil_ctx
from io_utils.fsutil.s3_connection import S3Connection
from io_utils.fsutil.uri import S3URI
from kumo.testing.s3 import withS3Credentials
from service_framework._raw_deploy_config import S3AddressingStyle

from .common import no_retry, random_name


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


def test_s3_credentials_decorator_does_not_probe_provider(monkeypatch):
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

    with patch.object(s3_filesystem.s3_ctx._s3_client,
                      'list_objects_v2') as list_objects:
        decorated = withS3Credentials(test_target)
        module_mark = withS3Credentials()

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


@pytest.fixture
def temp_prefix(k: int = 16) -> Generator[S3URI, None, None]:
    uri = S3URI(
        bucket='kumo-test-dataplane-bucket',
        path=random_name(k),
    )

    fsutil.makedirs(uri)
    yield uri
    fsutil.rm(uri, ignore_errors=True)


@withS3Credentials
def test_s3_rmdir(temp_prefix):
    bucket = "kumo-test-dataplane-bucket"
    dir_path = temp_prefix.as_s3()

    import io_utils.fsutil.s3_filesystem as util

    def touch_all(paths):
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(fsutil.touch, p) for p in paths]
            for f in as_completed(futures):
                f.result()

    touch_all([f"{dir_path}/{i}/{j}" for i in range(5) for j in range(5)])

    assert len(util.listdir_recursive(dir_path)) == 25
    util.rmdir(dir_path)
    assert len(util.listdir_recursive(dir_path)) == 0

    # Invalid bucket
    with pytest.raises(Exception):
        util.rmdir("s3://invalid-bucket/path")

    with pytest.raises(Exception):
        util._delete_batch("invalid-bucket", ["path"])

    # Should succeed if path is already deleted
    util._delete_batch(bucket, ["non-existant-path"])
    util.rmdir(dir_path)


@withS3Credentials
def test_s3_copydir(temp_prefix):
    bucket = temp_prefix.bucket
    prefix = temp_prefix.as_s3()

    src_relative = "tests/src_dir"
    src_key = osp.join(temp_prefix.path, src_relative)
    src_path = f"{prefix}/{src_relative}"
    dst_path = f"{prefix}/tests/dst_dir"
    dst_path2 = f"{prefix}/tests/dst_dir2"

    import io_utils.fsutil.s3_filesystem as util

    def touch_all(paths):
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(fsutil.touch, p) for p in paths]
            for f in as_completed(futures):
                f.result()

    touch_all([f"{src_path}/{i}/{j}" for i in range(3) for j in range(3)])

    assert len(util.listdir_recursive(src_path)) == 9
    util.copydir(src_path, dst_path2)
    assert len(util.listdir_recursive(dst_path2)) == 9

    # Set multiprocessing_threshold = 5 to force many processes to be launched
    util.s3_ctx._copy_multiprocessing_threshold = 5

    util.copydir(src_path, dst_path)
    assert len(util.listdir_recursive(dst_path)) == 9

    # Should overwrite existing file
    util.copydir(src_path, dst_path)
    assert len(util.listdir_recursive(dst_path)) == 9

    # Invalid src bucket
    with no_retry():
        with pytest.raises(Exception, match="Access Denied"):
            util.copydir("s3://invalid-bucket/path", dst_path)

    with no_retry():
        with pytest.raises(Exception, match="Forbidden"):
            util._copy_batch("invalid-bucket", "path", ["key"], bucket, "path",
                             util.s3_ctx._retry_delay_sec,
                             fsutil_ctx.thread_pool, util.s3_ctx._s3_client)

    # Invalid dst bucket
    with no_retry():
        with pytest.raises(Exception, match="Access Denied"):
            util.copydir(src_path, "s3://invalid-bucket")

    with pytest.raises(Exception, match="Access Denied"):
        util._copy_batch(
            bucket,
            f"{src_key}/0/",
            [f"{src_key}/0/0"],
            "invalid-bucket",
            "path",
            util.s3_ctx._retry_delay_sec,
            fsutil_ctx.thread_pool,
            util.s3_ctx._s3_client,
        )

    # Empty src path
    with pytest.raises(ValueError, match="source directory is empty"):
        util.copydir(f"s3://{bucket}/empty_path/", dst_path)

    with pytest.raises(Exception, match="Not Found"):
        util._copy_batch(
            bucket,
            "empty_dir/",
            ["empty_dir/0"],
            bucket,
            "path",
            util.s3_ctx._retry_delay_sec,
            fsutil_ctx.thread_pool,
            util.s3_ctx._s3_client,
        )


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


@withS3Credentials
def test_filesize():
    assert fsutil.filesize("s3://kumo-test-dataplane-bucket") is None
    assert fsutil.filesize("s3://kumo-test-dataplane-bucket/") is None
    assert fsutil.filesize(
        "s3://kumo-test-dataplane-bucket/test_fixtures") is None
    assert fsutil.filesize(
        "s3://kumo-test-dataplane-bucket/test_fixtures/") is None
    assert fsutil.filesize(
        "s3://kumo-test-dataplane-bucket/test_fixtures/hnm_10k_churn.json"
    ) == 7037
    assert fsutil.filesize(
        "s3://kumo-test-dataplane-bucket/test_fixtures/hnm_10k_churn.json/"
    ) is None


@withS3Credentials
def test_s3_listdir(temp_prefix):
    prefix = temp_prefix.as_s3()

    import io_utils.fsutil.s3_filesystem as util

    for i in range(5):
        for j in range(5):
            fsutil.touch(f"{prefix}/{i}/{j}")

    def listdir(path) -> List[str]:
        return util.listdir_recursive(path)

    # Base case
    assert len(listdir(f"{prefix}/")) == 25
    assert len(listdir(f"{prefix}")) == 25

    # Empty Dir
    assert len(listdir(f"{prefix}/empty_path")) == 0

    # Invalid bucket
    with no_retry():
        with pytest.raises(Exception, match="Access Denied"):
            listdir("s3://invalid-bucket/path")

    # Path with //
    fsutil.touch(f"{prefix}///slashpath//file")
    fsutil.touch(f"{prefix}///slashpath/file")
    assert len(listdir(f"{prefix}///slashpath")) == 2
    assert len(listdir(f"{prefix}///slashpath/")) == 2
    assert len(listdir(f"{prefix}///slashpath//")) == 1
    assert len(listdir(f"{prefix}///slashpath//file")) == 0
    assert len(listdir(f"{prefix}///slashpath//file/")) == 0


@withS3Credentials
def test_listdir(temp_prefix):
    prefix = temp_prefix.as_s3()

    def touch_all(paths):
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(fsutil.touch, p) for p in paths]
            for f in as_completed(futures):
                f.result()

    # Ensure all previous state is deleted.
    fsutil.rm(prefix)

    touch_all([f"{prefix}/{i}.txt" for i in range(10)] +
              [f"{prefix}/subdir/{i}.txt" for i in range(10)])

    assert len(fsutil.listdir(prefix)) == 11
    assert len(fsutil.listdir(prefix, max_items=5)) == 5
    assert len(fsutil.listdir(prefix, max_items=11)) == 11

    # Test max_items being out of range.
    with pytest.raises(ValueError):
        fsutil.listdir(prefix, max_items=-1)
    with pytest.raises(ValueError):
        fsutil.listdir(prefix, max_items=0)
    with pytest.raises(ValueError):
        fsutil.listdir(prefix, max_items=1005)

    fsutil.rm(prefix)

    touch_all([f"{prefix}/{i}.txt" for i in range(1005)])

    # Test that paginator returns all files.
    assert len(fsutil.listdir(prefix)) == 1005
    assert len(fsutil.listdir(prefix, max_items=1000)) == 1000
    with pytest.raises(ValueError):
        fsutil.listdir(prefix, max_items=1001)

    fsutil.rm(prefix)


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


@withS3Credentials
def test_isdir_bucket_root():
    bucket_path = "s3://kumo-test-dataplane-bucket"
    assert fsutil.isdir(bucket_path)
    assert fsutil.isdir(f"{bucket_path}/")


@withS3Credentials
def test_s3_generate_download_url():
    """S3PresignedHelper.generate_download_url produces a presigned HTTPS URL."""
    helper = fsutil.get_presigned_helper(
        's3://kumo-public-datasets/test_fsutil/existing_file')
    assert helper is not None
    url = helper.generate_download_url(
        's3://kumo-public-datasets/test_fsutil/existing_file')
    assert url.startswith('https://')
    assert 'kumo-public-datasets' in url


@withS3Credentials
def test_get_bucket_aws_region():
    assert s3_filesystem.get_bucket_aws_region(
        'kumo-public-datasets') == 'us-west-2'

    assert s3_filesystem.get_bucket_aws_region(
        'kumo-public-datasets-ue1') == 'us-east-1'

    # no permissions
    with pytest.raises(ValueError):
        s3_filesystem.get_bucket_aws_region(
            'kumo-inaccessible-bucket-for-automated-tests')

    # nonexistent
    with pytest.raises(ValueError):
        s3_filesystem.get_bucket_aws_region('kumo-fake-bucket')
