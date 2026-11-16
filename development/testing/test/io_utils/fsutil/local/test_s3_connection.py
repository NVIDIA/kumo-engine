# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pyarrow.fs import AwsDefaultS3RetryStrategy

from io_utils.fsutil import s3_connection
from io_utils.fsutil.credentials import (
    ExplicitFilesystemCredentialsRequired,
    credential_scope,
    pyarrow_filesystem,
)
from io_utils.fsutil.s3_connection import S3Connection
from service_framework._raw_deploy_config import (
    S3AddressingStyle,
    S3StorageConfig,
)


@pytest.fixture(autouse=True)
def _clear_connection_cache():
    s3_connection.deployment_s3_connection.cache_clear()
    yield
    s3_connection.deployment_s3_connection.cache_clear()


def _config() -> S3StorageConfig:
    return S3StorageConfig(
        endpoint_url='https://minio.example.test:9000',
        region='us-east-1',
        addressing_style=S3AddressingStyle.PATH,
        credential_secret_name='MINIO_KUMO_CREDENTIALS',
        ca_bundle_path='/etc/kumo/minio-ca/ca.crt',
    )


def test_deployment_connection_resolves_named_secret_and_redacts_repr(
        monkeypatch):
    get_secret = Mock(return_value={
        'aws_access_key_id': 'long-lived-access',
        'aws_secret_access_key': 'long-lived-secret',
    })
    monkeypatch.setattr(s3_connection, 'deploy_env', lambda: SimpleNamespace(
        storage=SimpleNamespace(s3=_config())))
    monkeypatch.setattr(s3_connection, 'get_secrets', get_secret)

    connection = s3_connection.deployment_s3_connection()

    assert connection is not None
    assert connection.access_key == 'long-lived-access'
    assert connection.secret_key == 'long-lived-secret'
    assert connection.session_token is None
    assert 'long-lived-access' not in repr(connection)
    assert 'long-lived-secret' not in repr(connection)
    assert connection.identity == (
        'deployment',
        'startup',
        'https://minio.example.test:9000',
        'us-east-1',
        'path',
        'MINIO_KUMO_CREDENTIALS',
        False,
    )
    get_secret.assert_called_once_with('MINIO_KUMO_CREDENTIALS')


@pytest.mark.parametrize(
    'secret,missing',
    [
        ({}, 'aws_access_key_id'),
        ({'aws_access_key_id': 'SENTINEL_ACCESS_VALUE'},
         'aws_secret_access_key'),
        ({
            'aws_access_key_id': 'SENTINEL_ACCESS_VALUE',
            'aws_secret_access_key': 'SENTINEL_SECRET_VALUE',
            'aws_session_token': '',
        }, 'aws_session_token'),
    ],
)
def test_deployment_connection_fails_closed_on_partial_secret(
        monkeypatch, secret, missing):
    monkeypatch.setattr(s3_connection, 'deploy_env', lambda: SimpleNamespace(
        storage=SimpleNamespace(s3=_config())))
    monkeypatch.setattr(s3_connection, 'get_secrets', lambda _name: secret)

    with pytest.raises(RuntimeError, match=missing) as error:
        s3_connection.deployment_s3_connection()

    assert 'SENTINEL_ACCESS_VALUE' not in str(error.value)
    assert 'SENTINEL_SECRET_VALUE' not in str(error.value)


def test_connection_builds_matching_boto_and_pyarrow_options():
    connection = S3Connection(
        endpoint_url='https://minio.example.test:9000',
        region='us-east-1',
        addressing_style=S3AddressingStyle.PATH,
        credential_secret_name='MINIO_KUMO_CREDENTIALS',
        access_key='access',
        secret_key='secret',
        session_token=None,
        ca_bundle_path='/etc/kumo/minio-ca/ca.crt',
        send_bucket_owner_full_control=False,
    )

    assert connection.boto_config.signature_version == 's3v4'
    assert connection.boto_config.s3['addressing_style'] == 'path'
    assert connection.boto_verify == '/etc/kumo/minio-ca/ca.crt'
    options = connection.pyarrow_options()
    retry = options.pop('retry_strategy')
    assert isinstance(retry, AwsDefaultS3RetryStrategy)
    assert options == {
        'access_key': 'access',
        'secret_key': 'secret',
        'session_token': None,
        'anonymous': False,
        'region': 'us-east-1',
        'endpoint_override': 'https://minio.example.test:9000',
        'scheme': 'https',
        'force_virtual_addressing': False,
        'tls_ca_file_path': '/etc/kumo/minio-ca/ca.crt',
    }


def test_connection_builds_native_subprocess_environment(monkeypatch):
    connection = S3Connection(
        endpoint_url='https://minio.example.test:9000',
        region='us-east-1',
        addressing_style=S3AddressingStyle.PATH,
        credential_secret_name='MINIO_KUMO_CREDENTIALS',
        access_key='access',
        secret_key='secret',
        session_token=None,
        ca_bundle_path='/etc/kumo/minio-ca/ca.crt',
        send_bucket_owner_full_control=False,
    )
    monkeypatch.setattr(s3_connection, 'deployment_s3_connection',
                        lambda: connection)

    assert s3_connection.s3_subprocess_environment() == {
        'AWS_ACCESS_KEY_ID': 'access',
        'AWS_SECRET_ACCESS_KEY': 'secret',
        'AWS_REGION': 'us-east-1',
        'AWS_DEFAULT_REGION': 'us-east-1',
        'AWS_ENDPOINT_URL': 'https://minio.example.test:9000',
        'AWS_ENDPOINT_URL_S3': 'https://minio.example.test:9000',
        'AWS_EC2_METADATA_DISABLED': 'true',
        'AWS_S3_ADDRESSING_STYLE': 'path',
        'AWS_CA_BUNDLE': '/etc/kumo/minio-ca/ca.crt',
        'SSL_CERT_FILE': '/etc/kumo/minio-ca/ca.crt',
        'S3_ACCESS_KEY': 'access',
        'S3_SECRET_KEY': 'secret',
        'S3_REGION': 'us-east-1',
        'S3_ENDPOINT': 'https://minio.example.test:9000',
        'S3_VIRTUAL_HOSTED': 'false',
    }


def test_custom_endpoint_omits_aws_owner_acl(monkeypatch):
    connection = Mock(send_bucket_owner_full_control=False)
    monkeypatch.setattr(s3_connection, 'active_s3_connection',
                        lambda: connection)
    assert s3_connection.bucket_owner_write_args() == {}

    connection.send_bucket_owner_full_control = True
    assert s3_connection.bucket_owner_write_args() == {
        'ACL': 'bucket-owner-full-control'
    }


def test_request_credential_scope_isolates_deployment_connection(monkeypatch):
    deployment = Mock()
    monkeypatch.setattr(s3_connection, 'deployment_s3_connection',
                        lambda: deployment)

    assert s3_connection.active_s3_connection() is deployment
    with credential_scope({
            'provider': 's3',
            'access_key': 'connector-access',
            'secret_key': 'connector-secret',
    }, require_explicit=True):
        assert s3_connection.active_s3_connection() is None
    assert s3_connection.active_s3_connection() is deployment

    with credential_scope(None, require_explicit=False):
        assert s3_connection.active_s3_connection() is deployment

    with credential_scope({'provider': 'gcs'}, require_explicit=True):
        with pytest.raises(ExplicitFilesystemCredentialsRequired):
            s3_connection.active_s3_connection()

    monkeypatch.setattr(s3_connection, 'active_s3_connection', lambda: None)
    assert s3_connection.bucket_owner_write_args() == {
        'ACL': 'bucket-owner-full-control'
    }


def test_connection_filesystem_retries_503s():
    seen = []

    class Unavailable(BaseHTTPRequestHandler):
        def do_HEAD(self):
            seen.append(self.path)
            failing = len(seen) <= 3
            self.send_response(503 if failing else 200)
            self.send_header('Content-Length', '0' if failing else '8')
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Unavailable)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = S3Connection(
        endpoint_url=f'http://127.0.0.1:{server.server_address[1]}',
        region='us-east-1',
        addressing_style=S3AddressingStyle.PATH,
        credential_secret_name='MINIO_KUMO_CREDENTIALS',
        access_key='access',
        secret_key='secret',
        session_token=None,
        ca_bundle_path=None,
        send_bucket_owner_full_control=False,
    )
    try:
        with credential_scope(
            {
                'provider': 's3',
                'region': 'us-east-1',
                's3_connection': connection
            }, require_explicit=True):
            filesystem = pyarrow_filesystem('s3://bucket/part-0.parquet')
        assert filesystem.get_file_info('bucket/part-0.parquet').size == 8
    finally:
        server.shutdown()
        thread.join()
    assert seen == ['/bucket/part-0.parquet'] * 4
