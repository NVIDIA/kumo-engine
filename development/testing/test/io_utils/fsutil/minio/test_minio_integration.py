# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit, urlunsplit

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, SSLError
from mypy_boto3_s3 import S3Client

from io_utils import fsutil
from io_utils import pyarrow as pyarrow_utils
from io_utils.fsutil import s3_connection, s3_filesystem
from io_utils.fsutil.credentials import credential_scope
from io_utils.fsutil.s3_connection import S3Connection
from service_framework._raw_deploy_config import S3AddressingStyle

pytestmark = pytest.mark.compose_e2e

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
_SECRET_ROOT = os.path.join(
    _REPO_ROOT,
    'docker',
    'compose',
    'config',
    'minio-test',
    'secrets',
)


@dataclass(frozen=True)
class MinioHarness:
    connection: S3Connection
    client: S3Client
    data_root: str
    artifact_root: str


@dataclass(frozen=True)
class OrthogonalMinioHarness:
    deployment: S3Connection
    connector: S3Connection
    connector_credentials: dict[str, object]
    connector_root: str


@dataclass(frozen=True)
class UploadedPart:
    part_number: int
    etag: str


def _fixture_secret(name: str) -> str:
    return fsutil.read_text(os.path.join(_SECRET_ROOT, name)).strip()


@pytest.fixture
def minio(monkeypatch) -> MinioHarness:
    endpoint = os.environ.get('KUMO_MINIO_TEST_ENDPOINT',
                              'http://127.0.0.1:19010')
    ca_bundle_path = os.environ.get('KUMO_MINIO_TEST_CA')
    connection = S3Connection(
        endpoint_url=endpoint,
        region='us-east-1',
        addressing_style=S3AddressingStyle.PATH,
        credential_secret_name='MINIO_KUMO_CREDENTIALS',
        access_key=_fixture_secret('kumo-access-key'),
        secret_key=_fixture_secret('kumo-secret-key'),
        session_token=None,
        ca_bundle_path=ca_bundle_path,
        send_bucket_owner_full_control=False,
    )
    monkeypatch.setattr(s3_connection, 'active_s3_connection',
                        lambda: connection)
    monkeypatch.setattr(s3_filesystem, 'active_s3_connection',
                        lambda: connection)
    s3_filesystem.threadsafe_boto._reset()
    pyarrow_utils._arrow_cloud_filesystem.cache_clear()

    test_id = uuid.uuid4().hex
    harness = MinioHarness(
        connection=connection,
        client=s3_filesystem.threadsafe_boto.s3_client(),
        data_root=f's3://kumo-dataplane/kumo/minio-contract/{test_id}',
        artifact_root=f's3://kumo-artifacts/kumo/minio-contract/{test_id}',
    )
    harness.client.head_bucket(Bucket='kumo-dataplane')
    harness.client.head_bucket(Bucket='kumo-artifacts')
    yield harness

    fsutil.rm(harness.data_root, ignore_errors=True)
    fsutil.rm(harness.artifact_root, ignore_errors=True)
    pyarrow_utils._arrow_cloud_filesystem.cache_clear()
    s3_filesystem.threadsafe_boto._reset()


@pytest.fixture
def orthogonal_minio(monkeypatch) -> OrthogonalMinioHarness:
    endpoint = os.environ.get('KUMO_MINIO_TEST_ENDPOINT',
                              'http://127.0.0.1:19010')
    ca_bundle_path = os.environ.get('KUMO_MINIO_TEST_CA')
    ca_certificate = (fsutil.read_text(ca_bundle_path)
                      if ca_bundle_path else None)
    deployment = S3Connection(
        endpoint_url=endpoint,
        region='us-east-1',
        addressing_style=S3AddressingStyle.PATH,
        credential_secret_name='MINIO_KUMO_CREDENTIALS',
        access_key=_fixture_secret('kumo-access-key'),
        secret_key=_fixture_secret('kumo-secret-key'),
        session_token=None,
        ca_bundle_path=ca_bundle_path,
        send_bucket_owner_full_control=False,
    )
    connector = S3Connection(
        endpoint_url=endpoint,
        region='us-east-1',
        addressing_style=S3AddressingStyle.PATH,
        credential_secret_name='',
        access_key=_fixture_secret('connector-access-key'),
        secret_key=_fixture_secret('connector-secret-key'),
        session_token=None,
        ca_bundle_path=None,
        ca_certificate=ca_certificate,
        send_bucket_owner_full_control=False,
        owner='connector:minio-contract',
        rotation_id='fixture-1',
    )
    monkeypatch.setattr(s3_connection, 'deployment_s3_connection',
                        lambda: deployment)
    s3_filesystem.threadsafe_boto._reset()
    pyarrow_utils._arrow_cloud_filesystem.cache_clear()
    root = f's3://kumo-connector/kumo/minio-contract/{uuid.uuid4().hex}'
    credentials = {
        'provider': 's3',
        'aws_access_key_id': connector.access_key,
        'aws_secret_access_key': connector.secret_key,
        'aws_session_token': None,
        'region': connector.region,
        's3_connection': connector,
    }
    harness = OrthogonalMinioHarness(
        deployment=deployment,
        connector=connector,
        connector_credentials=credentials,
        connector_root=root,
    )
    with credential_scope(credentials, require_explicit=True):
        s3_filesystem.threadsafe_boto.s3_client().head_bucket(
            Bucket='kumo-connector')
    yield harness
    with credential_scope(credentials, require_explicit=True):
        fsutil.rm(root, ignore_errors=True)
    pyarrow_utils._arrow_cloud_filesystem.cache_clear()
    s3_filesystem.threadsafe_boto._reset()


def test_fsutil_contract(minio: MinioHarness) -> None:
    requests: list[tuple[str, dict[str, str]]] = []

    def record_request(model, params, **_kwargs) -> None:
        requests.append((model.name, {
            str(key).casefold(): str(value)
            for key, value in params['headers'].items()
        }))

    minio.client.meta.events.register('before-call.s3.*', record_request)
    source = f'{minio.data_root}/source/input.txt'
    same_bucket_copy = f'{minio.data_root}/copy/input.txt'
    cross_bucket_copy = f'{minio.artifact_root}/copy/input.txt'

    fsutil.write_text(source, 'minio-contract')
    assert fsutil.read_text(source) == 'minio-contract'
    assert fsutil.filesize(source) == len('minio-contract')
    assert fsutil.object_identity(source).size == len('minio-contract')

    fsutil.copy_file(source, same_bucket_copy)
    fsutil.copy_file(source, cross_bucket_copy)
    assert fsutil.read_text(same_bucket_copy) == 'minio-contract'
    assert fsutil.read_text(cross_bucket_copy) == 'minio-contract'

    nested_source = f'{minio.data_root}/tree/source'
    nested_copy = f'{minio.data_root}/tree/copy'
    fsutil.write_text(f'{nested_source}/top.txt', 'top')
    fsutil.write_text(f'{nested_source}/nested/child.txt', 'child')
    s3_filesystem.copydir(nested_source, nested_copy)
    assert fsutil.read_text(f'{nested_copy}/top.txt') == 'top'
    assert fsutil.read_text(f'{nested_copy}/nested/child.txt') == 'child'
    fsutil.rm(nested_source)
    assert not fsutil.exists(f'{nested_source}/top.txt')
    relevant = [headers for operation, headers in requests
                if operation in {'CopyObject', 'PutObject'}]
    assert relevant
    assert all('x-amz-acl' not in headers for headers in relevant)


def test_listing_paginates(minio: MinioHarness) -> None:
    root = f'{minio.data_root}/pagination'

    def create_object(index: int) -> None:
        fsutil.touch(f'{root}/{index:04d}.txt')

    with ThreadPoolExecutor(max_workers=32) as executor:
        list(executor.map(create_object, range(1005)))

    keys = s3_filesystem.listdir_recursive(root)
    assert len(keys) == 1005
    assert keys[0].startswith(
        minio.data_root.removeprefix('s3://kumo-dataplane/') + '/')


def test_multipart_cleanup(minio: MinioHarness) -> None:
    requests: list[dict[str, str]] = []

    def record_request(model, params, **_kwargs) -> None:
        del model
        requests.append({
            str(key).casefold(): str(value)
            for key, value in params['headers'].items()
        })

    minio.client.meta.events.register(
        'before-call.s3.CreateMultipartUpload',
        record_request,
    )
    bucket = 'kumo-dataplane'
    root_key = minio.data_root.removeprefix(f's3://{bucket}/')
    key = f'{root_key}/explicit-multipart-probe'
    response = minio.client.create_multipart_upload(Bucket=bucket, Key=key)
    minio.client.abort_multipart_upload(
        Bucket=bucket,
        Key=key,
        UploadId=response['UploadId'],
    )

    assert fsutil.has_write_permission(f'{minio.data_root}/output/')
    uploads = minio.client.list_multipart_uploads(Bucket=bucket,
                                                  Prefix=root_key)
    assert uploads.get('Uploads', []) == []
    assert requests
    assert all('x-amz-acl' not in headers for headers in requests)


def test_presigned_multipart_upload(minio: MinioHarness) -> None:
    path = f'{minio.data_root}/presigned/upload.csv'
    payload = b'customer_id,value\n1,42\n'
    helper = fsutil.get_presigned_helper(path)
    assert helper is not None

    upload_id, urls = helper.create_multipart_upload(1)
    upload_url = urls[1]
    query = parse_qs(urlsplit(upload_url).query)
    assert query['X-Amz-Algorithm'] == ['AWS4-HMAC-SHA256']
    assert query['X-Amz-SignedHeaders'] == ['host']

    upload = requests.put(
        upload_url,
        data=payload,
        headers={'Content-Type': 'text/csv'},
        timeout=10,
        verify=minio.connection.boto_verify,
    )
    upload.raise_for_status()
    helper.complete_multipart_upload(
        upload_id,
        [UploadedPart(part_number=1, etag=upload.headers['ETag'])],
    )

    download = requests.get(
        helper.generate_download_url(),
        timeout=10,
        verify=minio.connection.boto_verify,
    )
    download.raise_for_status()
    assert download.content == payload


def test_pyarrow_parquet(minio: MinioHarness) -> None:
    uri = f'{minio.artifact_root}/parquet/result.parquet'
    path = uri.removeprefix('s3://')
    filesystem = pyarrow_utils.arrow_cloud_filesystem(uri)
    expected = pa.table({'entity_id': [1, 2, 3], 'score': [0.1, 0.2, 0.3]})

    pq.write_table(expected, path, filesystem=filesystem)
    actual = pq.read_table(path, filesystem=filesystem)

    assert actual == expected


def test_connector_contract_is_orthogonal_from_dataplane(
        orthogonal_minio: OrthogonalMinioHarness) -> None:
    root = orthogonal_minio.connector_root
    source = f'{root}/source/input.txt'
    copied = f'{root}/copy/input.txt'
    parquet_uri = f'{root}/parquet/result.parquet'
    expected = pa.table({'customer_id': [1, 2], 'ltv': [10.0, 20.0]})

    assert orthogonal_minio.connector.identity != (
        orthogonal_minio.deployment.identity)
    with credential_scope(orthogonal_minio.connector_credentials,
                          require_explicit=True):
        client = s3_filesystem.threadsafe_boto.s3_client()
        assert client.meta.endpoint_url == orthogonal_minio.connector.endpoint_url
        fsutil.write_text(source, 'connector-minio-contract')
        assert fsutil.read_text(source) == 'connector-minio-contract'
        assert fsutil.object_identity(source).size == 24
        fsutil.copy_file(source, copied)
        assert s3_filesystem.listdir_recursive(root)
        filesystem = pyarrow_utils.arrow_cloud_filesystem(parquet_uri)
        pq.write_table(expected, parquet_uri.removeprefix('s3://'),
                       filesystem=filesystem)
        assert pq.read_table(parquet_uri.removeprefix('s3://'),
                             filesystem=filesystem) == expected
        assert fsutil.has_write_permission(f'{root}/permission-probe/')
        url = fsutil.get_presigned_helper(source).generate_download_url()
        response = requests.get(
            url,
            timeout=10,
            verify=(orthogonal_minio.connector.ca_path or True),
        )
        response.raise_for_status()
        assert response.text == 'connector-minio-contract'
        with pytest.raises(ClientError) as connector_denied:
            client.head_bucket(Bucket='kumo-dataplane')
        assert connector_denied.value.response['Error']['Code'] in {
            'AccessDenied', '403'
        }

    deployment_client = s3_filesystem.threadsafe_boto.s3_client()
    assert deployment_client.meta.endpoint_url == (
        orthogonal_minio.deployment.endpoint_url)
    with pytest.raises(ClientError) as deployment_denied:
        deployment_client.head_bucket(Bucket='kumo-connector')
    assert deployment_denied.value.response['Error']['Code'] in {
        'AccessDenied', '403'
    }


def test_invalid_key_fails(minio: MinioHarness) -> None:
    session = boto3.session.Session(
        aws_access_key_id='invalid-minio-access-key',
        aws_secret_access_key='invalid-minio-secret-key',
        region_name=minio.connection.region,
    )
    client = session.client(
        's3',
        endpoint_url=minio.connection.endpoint_url,
        region_name=minio.connection.region,
        config=minio.connection.boto_config,
        verify=minio.connection.boto_verify,
    )

    with pytest.raises(ClientError) as error:
        client.head_bucket(Bucket='kumo-dataplane')

    assert error.value.response['Error']['Code'] in {
        '403',
        'AccessDenied',
        'InvalidAccessKeyId',
    }


def test_insufficient_policy_fails(minio: MinioHarness) -> None:
    client = boto3.session.Session(
        aws_access_key_id=_fixture_secret('readonly-access-key'),
        aws_secret_access_key=_fixture_secret('readonly-secret-key'),
        region_name=minio.connection.region,
    ).client(
        's3',
        endpoint_url=minio.connection.endpoint_url,
        region_name=minio.connection.region,
        config=minio.connection.boto_config,
        verify=minio.connection.boto_verify,
    )

    with pytest.raises(ClientError) as error:
        client.put_object(
            Bucket='kumo-dataplane',
            Key='kumo/authorization-negative/denied.txt',
            Body=b'denied',
        )

    assert error.value.response['Error']['Code'] in {'AccessDenied', '403'}


def test_wrong_endpoint_fails(minio: MinioHarness) -> None:
    client = boto3.session.Session(
        aws_access_key_id=minio.connection.access_key,
        aws_secret_access_key=minio.connection.secret_key,
        region_name=minio.connection.region,
    ).client(
        's3',
        endpoint_url='http://127.0.0.1:1',
        region_name=minio.connection.region,
        config=minio.connection.boto_config.merge(
            Config(
                connect_timeout=1,
                read_timeout=1,
                retries={'max_attempts': 0},
            )),
    )

    with pytest.raises(EndpointConnectionError):
        client.head_bucket(Bucket='kumo-dataplane')


def test_wrong_ca_fails(minio: MinioHarness) -> None:
    if not minio.connection.endpoint_url.startswith('https://'):
        pytest.skip('private-CA failure applies only to the TLS fixture')
    default_ca = '/etc/ssl/certs/ca-certificates.crt'
    assert fsutil.exists(default_ca)
    client = boto3.session.Session(
        aws_access_key_id=minio.connection.access_key,
        aws_secret_access_key=minio.connection.secret_key,
        region_name=minio.connection.region,
    ).client(
        's3',
        endpoint_url=minio.connection.endpoint_url,
        region_name=minio.connection.region,
        verify=default_ca,
        config=minio.connection.boto_config.merge(
            Config(retries={'max_attempts': 0})),
    )

    with pytest.raises(SSLError):
        client.head_bucket(Bucket='kumo-dataplane')


def test_wrong_style_fails(minio: MinioHarness) -> None:
    if not minio.connection.endpoint_url.startswith('https://'):
        pytest.skip('addressing failure applies to the TLS fixture')
    endpoint = urlsplit(minio.connection.endpoint_url)
    virtual_endpoint = urlunsplit((
        endpoint.scheme,
        f'localhost:{endpoint.port}',
        endpoint.path,
        endpoint.query,
        endpoint.fragment,
    ))
    client = boto3.session.Session(
        aws_access_key_id=minio.connection.access_key,
        aws_secret_access_key=minio.connection.secret_key,
        region_name=minio.connection.region,
    ).client(
        's3',
        endpoint_url=virtual_endpoint,
        region_name=minio.connection.region,
        verify=minio.connection.boto_verify,
        config=Config(
            signature_version='s3v4',
            s3={'addressing_style': 'virtual'},
            retries={'max_attempts': 0},
        ),
    )

    with pytest.raises(SSLError):
        client.head_bucket(Bucket='kumo-dataplane')
