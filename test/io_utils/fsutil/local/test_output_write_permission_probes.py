# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

from io_utils import fsutil
from io_utils.fsutil.impl_adls import adls_ctx
from io_utils.fsutil.s3_filesystem import threadsafe_boto


def test_s3_write_probe_is_unique_and_aborted():
    client = Mock()
    client.create_multipart_upload.return_value = {'UploadId': 'upload-id'}

    with patch.object(threadsafe_boto, 's3_client', return_value=client):
        assert fsutil.has_write_permission(
            's3://bucket/authorized/output/') is True

    create_args = client.create_multipart_upload.call_args.kwargs
    assert create_args['Bucket'] == 'bucket'
    assert create_args['Key'].startswith('authorized/output/.kumo_write_probe_')
    client.abort_multipart_upload.assert_called_once_with(
        Bucket='bucket',
        Key=create_args['Key'],
        UploadId='upload-id',
    )


def test_s3_write_probe_omits_acl_for_custom_endpoint():
    client = Mock()
    client.create_multipart_upload.return_value = {'UploadId': 'upload-id'}

    with patch.object(threadsafe_boto, 's3_client', return_value=client), \
            patch('io_utils.fsutil.impl_s3.bucket_owner_write_args',
                  return_value={}):
        assert fsutil.has_write_permission('s3://bucket/output/') is True

    assert 'ACL' not in client.create_multipart_upload.call_args.kwargs


def test_s3_write_probe_denial_and_cleanup_errors_are_sanitized(caplog):
    secret = 'provider-payload-with-access-key'
    denied = ClientError(
        {'Error': {
            'Code': 'AccessDenied',
            'Message': secret,
        }}, 'CreateMultipartUpload')
    client = Mock()
    client.create_multipart_upload.side_effect = denied
    with patch.object(threadsafe_boto, 's3_client', return_value=client):
        assert fsutil.has_write_permission(
            's3://bucket/authorized/output/') is False
    assert secret not in caplog.text

    caplog.clear()
    client = Mock()
    client.create_multipart_upload.return_value = {'UploadId': 'upload-id'}
    client.abort_multipart_upload.side_effect = RuntimeError(secret)
    with patch.object(threadsafe_boto, 's3_client', return_value=client):
        assert fsutil.has_write_permission(
            's3://bucket/authorized/output/') is False
    assert secret not in caplog.text


def test_gcs_write_probe_targets_prefix_and_is_deleted():
    bucket = Mock()
    blob = Mock()
    bucket.blob.return_value = blob

    with patch(
            'io_utils.fsutil.impl_gcs._get_bucket',
            return_value=bucket,
    ):
        assert fsutil.has_write_permission(
            'gs://bucket/authorized/output/') is True

    probe_path = bucket.blob.call_args.args[0]
    assert probe_path.startswith('authorized/output/.kumo_write_probe_')
    blob.upload_from_string.assert_called_once_with(b'')
    blob.delete.assert_called_once_with()


def test_gcs_write_probe_cleanup_error_is_sanitized(caplog):
    bucket = Mock()
    blob = Mock()
    secret = 'provider-payload-with-access-token'
    blob.delete.side_effect = RuntimeError(secret)
    bucket.blob.return_value = blob

    with patch(
            'io_utils.fsutil.impl_gcs._get_bucket',
            return_value=bucket,
    ):
        assert fsutil.has_write_permission(
            'gs://bucket/authorized/output/') is False
    assert secret not in caplog.text


def test_adls_write_probe_preserves_cloud_endpoint_and_is_deleted():
    file_client = Mock()
    captured = []

    def _get_file_client(uri):
        captured.append(uri)
        return file_client

    with patch.object(adls_ctx, 'get_file_client',
                      side_effect=_get_file_client):
        assert fsutil.has_write_permission(
            'abfss://filesystem@account.dfs.core.usgovcloudapi.net/'
            'authorized/output/') is True

    assert len(captured) == 1
    probe_uri = captured[0]
    assert probe_uri.endpoint_suffix == 'dfs.core.usgovcloudapi.net'
    assert probe_uri.path.startswith('authorized/output/.kumo_write_probe_')
    file_client.create_file.assert_called_once_with()
    file_client.delete_file.assert_called_once_with()


def test_adls_write_probe_cleanup_error_is_sanitized(caplog):
    file_client = Mock()
    secret = 'provider-payload-with-client-secret'
    file_client.delete_file.side_effect = RuntimeError(secret)

    with patch.object(
            adls_ctx,
            'get_file_client',
            return_value=file_client,
    ):
        assert fsutil.has_write_permission(
            'abfss://filesystem@account.dfs.core.windows.net/'
            'authorized/output/') is False
    assert secret not in caplog.text
