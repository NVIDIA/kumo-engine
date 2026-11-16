# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import stat
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.types import TaskType
from kumo_core.types.schema import IngestedTableFingerprint
from kumoapi.pquery import QueryType
from kumoapi.typing import Dtype
from pydantic import SecretStr

from io_utils.fsutil import ObjectIdentity
from io_utils.fsutil.credentials import credentials_for_provider
from io_utils.fsutil.s3_connection import S3Connection
from kumo.connector import file_connector
from kumo.connector.any_uri_adls_connector import AnyURIADLSConnector
from kumo.connector.any_uri_gcs_connector import AnyURIGCSConnector
from kumo.connector.any_uri_s3_connector import AnyURIS3Connector
from kumo.connector.composite_file_connector import CompositeFileConnector
from kumo.connector.connector import ConnectorError
from rest.backends.data_backend.activities import custom_train_table
from rest.config.credentials import (
    AWSFederatedCredentials,
    AzureFederatedCredentials,
    GoogleCloudCredentials,
)
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.file_type import FileType
from rest.config.table_v2 import DataSourceType, S3SourceTable
from statedb.config.common import ConnectorID, ProjectID
from statedb.utils.errors import UnsupportedInDiskgraphError


def _credentials() -> dict:
    return AWSFederatedCredentials(
        access_key_id=SecretStr('customer-access-key'),
        secret_access_key=SecretStr('customer-secret-key'),
        session_token=SecretStr('customer-session-token'),
    ).as_dict()


def _connector(*, suffix: str) -> CompositeFileConnector:
    connector = CompositeFileConnector(
        'customer_files',
        root_dir='s3://customer-bucket/root',
    )
    connector.set_credentials(_credentials())
    connector.require_explicit_credentials()
    assert suffix in {'csv', 'parquet'}
    return connector


def _fingerprint(source_name: str) -> IngestedTableFingerprint:
    return IngestedTableFingerprint(
        connector_name='customer_files',
        source_name=source_name,
        cols={'id': Dtype.int64},
        pkey='id',
        time_col=None,
        end_time_col=None,
    )


def _assert_customer_scope() -> None:
    scope = credentials_for_provider('s3')
    assert scope is not None
    assert scope.credentials == _credentials()


def _assert_deployment_scope() -> None:
    assert credentials_for_provider('s3') is None


def test_direct_manifest_uses_exact_ambient_identity_and_never_stages(
        monkeypatch):
    connector = CompositeFileConnector('ambient_files',
                                       root_dir='s3://customer-bucket/root')
    ambient_session = object()
    connector.require_explicit_credentials()
    connector.set_credentials({
        'provider': 's3',
        'ambient_session': ambient_session,
    })
    paths = [
        's3://customer-bucket/root/events/a.parquet',
        's3://customer-bucket/root/events/b.parquet',
    ]
    resolve_paths = MagicMock(side_effect=lambda source_name: paths
                              if source_name == 'events' else None)
    monkeypatch.setattr(connector, '_direct_parquet_paths', resolve_paths)
    identities = {
        paths[0]: ObjectIdentity(11, 'etag-a', None),
        paths[1]: ObjectIdentity(13, 'etag-b', None),
    }

    def identity(path):
        scope = credentials_for_provider('s3')
        assert scope is not None
        assert scope.credentials['ambient_session'] is ambient_session
        return identities[path]

    monkeypatch.setattr('kumo.connector.file_connector.fsutil.object_identity',
                        identity)
    copy_file = MagicMock(
        side_effect=AssertionError('direct manifest must not copy bytes'))
    monkeypatch.setattr('kumo.connector.file_connector.fsutil.copy_file',
                        copy_file)

    assert connector.uses_direct_diskgraph_source_for('events')
    source = connector.diskgraph_source(source_name='events')
    assert source == {
        'type':
        'object_store_parquet',
        'objects': [
            {
                'path': paths[0],
                'size': 11,
                'e_tag': 'etag-a',
                'version': None,
            },
            {
                'path': paths[1],
                'size': 13,
                'e_tag': 'etag-b',
                'version': None,
            },
        ],
        'auth_profile':
        'ambient',
    }
    resolve_paths.assert_called_once_with('events')
    copy_file.assert_not_called()


def test_ambient_s3_source_stages_for_separate_minio_dataplane(monkeypatch):
    connector = CompositeFileConnector('ambient_files',
                                       root_dir='s3://customer-bucket/root')
    connector.require_explicit_credentials()
    connector.set_credentials({
        'provider': 's3',
        'ambient_session': object(),
    })
    monkeypatch.setattr(
        connector,
        '_direct_parquet_paths',
        lambda _source_name: ['s3://customer-bucket/root/events/part.parquet'],
    )
    monkeypatch.setattr(
        'io_utils.fsutil.s3_connection.deployment_s3_connection',
        lambda: object(),
    )
    environment = SimpleNamespace(
        data_plane_storage_root='s3://deployment-bucket/root')
    monkeypatch.setattr(file_connector, 'deploy_env', lambda: environment)

    assert not connector.uses_direct_diskgraph_source_for('events')


def test_direct_s3_subprocess_environment_preserves_authorized_session(
        monkeypatch):
    connector = CompositeFileConnector(
        'ambient_files',
        root_dir='s3://customer-bucket/root',
    )
    session = MagicMock(region_name=None)
    source = MagicMock(method='container-role')
    session.get_credentials.return_value = source
    session.client.return_value.head_bucket.return_value = {
        'BucketRegion': 'us-west-2'
    }
    monkeypatch.setenv('AWS_CONTAINER_CREDENTIALS_RELATIVE_URI',
                       '/v2/credentials/pod')
    connector.require_explicit_credentials()
    connector.set_credentials({
        'provider': 's3',
        'ambient_session': session,
    })
    path = 's3://customer-bucket/root/events/part.parquet'
    monkeypatch.setattr(connector, '_direct_parquet_paths',
                        lambda _source_name: [path])

    environment = connector.diskgraph_subprocess_environment('events')

    assert environment == {
        'AWS_REGION': 'us-west-2',
        'AWS_DEFAULT_REGION': 'us-west-2',
    }
    assert session.get_credentials.call_count >= 1
    source.get_frozen_credentials.assert_not_called()
    session.client.return_value.head_bucket.assert_called_once_with(
        Bucket='customer-bucket')


@pytest.mark.parametrize('method,selectors', [
    ('assume-role-with-web-identity', {
        'AWS_ROLE_ARN': 'arn:aws:iam::123456789012:role/kumo-runtime',
        'AWS_WEB_IDENTITY_TOKEN_FILE':
        '/var/run/secrets/eks.amazonaws.com/token',
    }),
    ('container-role', {
        'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/pod',
    }),
    ('iam-role', {}),
])
def test_explicit_ambient_s3_hands_off_renewable_workload_provider(
        monkeypatch, method, selectors):
    connector = CompositeFileConnector(
        'ambient_files',
        root_dir='s3://customer-bucket/root',
    )
    source = MagicMock(method=method)
    session = MagicMock(region_name='us-west-2')
    session.get_credentials.return_value = source
    connector.require_explicit_credentials()
    connector.set_credentials({
        'provider': 's3',
        'ambient_session': session,
    })
    for name, value in selectors.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        connector,
        '_direct_parquet_paths',
        lambda _source_name: ['s3://customer-bucket/root/events/part.parquet'],
    )

    assert connector.uses_direct_diskgraph_source_for('events')
    assert connector.diskgraph_subprocess_environment('events') == {
        'AWS_REGION': 'us-west-2',
        'AWS_DEFAULT_REGION': 'us-west-2',
    }
    source.get_frozen_credentials.assert_not_called()


def test_profile_only_web_identity_retains_direct_static_handoff(monkeypatch):
    connector = CompositeFileConnector(
        'ambient_files',
        root_dir='s3://customer-bucket/root',
    )
    session = MagicMock(region_name='us-west-2')
    source = MagicMock(method='assume-role-with-web-identity')
    source.get_frozen_credentials.return_value = SimpleNamespace(
        access_key='profile-access-key',
        secret_key='profile-secret-key',
        token='profile-session-token',
    )
    session.get_credentials.return_value = source
    connector.require_explicit_credentials()
    connector.set_credentials({
        'provider': 's3',
        'ambient_session': session,
    })
    monkeypatch.delenv('AWS_ROLE_ARN', raising=False)
    monkeypatch.delenv('AWS_WEB_IDENTITY_TOKEN_FILE', raising=False)
    monkeypatch.setattr(
        connector,
        '_direct_parquet_paths',
        lambda _source_name: ['s3://customer-bucket/root/events/part.parquet'],
    )

    assert connector.uses_direct_diskgraph_source_for('events')
    assert connector.diskgraph_subprocess_environment('events') == {
        'AWS_REGION': 'us-west-2',
        'AWS_DEFAULT_REGION': 'us-west-2',
        'AWS_ACCESS_KEY_ID': 'profile-access-key',
        'AWS_SECRET_ACCESS_KEY': 'profile-secret-key',
        'AWS_SESSION_TOKEN': 'profile-session-token',
        'AWS_EC2_METADATA_DISABLED': 'true',
    }


def test_implicit_s3_retains_frozen_workload_credential_handoff(monkeypatch):
    connector = CompositeFileConnector(
        'ambient_files',
        root_dir='s3://customer-bucket/root',
    )
    session = MagicMock(region_name='us-west-2')
    source = MagicMock(method='container-role')
    source.get_frozen_credentials.return_value = SimpleNamespace(
        access_key='temporary-access-key',
        secret_key='temporary-secret-key',
        token='temporary-session-token',
    )
    session.get_credentials.return_value = source
    monkeypatch.setattr('boto3.Session', lambda: session)
    monkeypatch.setenv('AWS_CONTAINER_CREDENTIALS_RELATIVE_URI',
                       '/v2/credentials/pod')
    monkeypatch.setattr(
        connector,
        '_direct_parquet_paths',
        lambda _source_name: ['s3://customer-bucket/root/events/part.parquet'],
    )

    assert connector.uses_direct_diskgraph_source_for('events')
    assert connector.diskgraph_subprocess_environment('events') == {
        'AWS_REGION': 'us-west-2',
        'AWS_DEFAULT_REGION': 'us-west-2',
        'AWS_ACCESS_KEY_ID': 'temporary-access-key',
        'AWS_SECRET_ACCESS_KEY': 'temporary-secret-key',
        'AWS_SESSION_TOKEN': 'temporary-session-token',
        'AWS_EC2_METADATA_DISABLED': 'true',
    }
    source.get_frozen_credentials.assert_called_once_with()


def test_implicit_static_s3_credentials_still_require_a_session_token(
        monkeypatch):
    connector = CompositeFileConnector(
        'ambient_files',
        root_dir='s3://customer-bucket/root',
    )
    session = MagicMock(region_name='us-west-2')
    source = MagicMock(method='env')
    source.get_frozen_credentials.return_value = SimpleNamespace(
        access_key='static-access-key',
        secret_key='static-secret-key',
        token=None,
    )
    session.get_credentials.return_value = source
    monkeypatch.setattr('boto3.Session', lambda: session)
    monkeypatch.setattr(
        connector,
        '_direct_parquet_paths',
        lambda _source_name: ['s3://customer-bucket/root/events/part.parquet'],
    )

    assert connector.uses_direct_diskgraph_source_for('events')
    with pytest.raises(ConnectorError, match='no temporary credentials'):
        connector.diskgraph_subprocess_environment('events')


def test_connector_access_key_uses_authorized_staging(monkeypatch):
    connector = CompositeFileConnector(
        'minio_files',
        root_dir='s3://customer-bucket/root',
    )
    connection = S3Connection(
        endpoint_url='https://connector-minio.example.test:9443',
        region='us-east-1',
        addressing_style='path',
        credential_secret_name='',
        access_key='connector-access-key',
        secret_key='connector-secret-key',
        session_token=None,
        ca_bundle_path='/run/kumo/connector-ca.pem',
        send_bucket_owner_full_control=False,
        owner='connector:identity-1',
        rotation_id='rotation-1',
    )
    connector.require_explicit_credentials()
    connector.set_credentials({
        'provider': 's3',
        'aws_access_key_id': 'connector-access-key',
        'aws_secret_access_key': 'connector-secret-key',
        'region': 'us-east-1',
        's3_connection': connection,
    })
    path = 's3://customer-bucket/root/events/part.parquet'
    monkeypatch.setattr(connector, '_direct_parquet_paths',
                        lambda _source_name: [path])

    assert not connector.uses_direct_diskgraph_source_for('events')
    with pytest.raises(ConnectorError, match='authorized direct object'):
        connector.diskgraph_subprocess_environment('events')
    assert 'connector-secret-key' not in repr(connection)


def test_direct_s3_subprocess_environment_uses_deployment_long_lived_key(
        monkeypatch):
    connector = CompositeFileConnector(
        'ambient_files',
        root_dir='s3://deployment-bucket/root',
    )
    frozen = SimpleNamespace(
        access_key='worker-role-access-key',
        secret_key='worker-role-secret-key',
        token='worker-role-session-token',
    )
    session = MagicMock(region_name='us-east-1')
    session.get_credentials.return_value.get_frozen_credentials.return_value = (
        frozen)
    connector.require_explicit_credentials()
    connector.set_credentials({
        'provider': 's3',
        'ambient_session': session,
    })
    path = 's3://deployment-bucket/root/events/part.parquet'
    monkeypatch.setattr(connector, '_direct_parquet_paths',
                        lambda _source_name: [path])
    connection = SimpleNamespace(
        access_key='deployment-access-key',
        secret_key='deployment-secret-key',
    )
    expected = {
        'AWS_ACCESS_KEY_ID': 'deployment-access-key',
        'AWS_SECRET_ACCESS_KEY': 'deployment-secret-key',
        'AWS_ENDPOINT_URL_S3': 'https://minio:9000',
    }
    monkeypatch.setattr(
        'io_utils.fsutil.s3_connection.deployment_s3_connection',
        lambda: connection,
    )
    monkeypatch.setattr(
        'io_utils.fsutil.s3_connection.s3_subprocess_environment',
        lambda: expected,
    )
    monkeypatch.setattr(
        file_connector, 'deploy_env', lambda: SimpleNamespace(
            data_plane_storage_root='s3://deployment-bucket/root'))

    environment = connector.diskgraph_subprocess_environment('events')

    assert environment == expected
    session.get_credentials.assert_not_called()
    session.client.assert_not_called()


def test_other_long_lived_s3_source_stages_before_diskgraph(monkeypatch):
    connector = CompositeFileConnector(
        'ambient_files',
        root_dir='s3://customer-bucket/root',
    )
    frozen = SimpleNamespace(
        access_key='customer-access-key',
        secret_key='customer-secret-key',
        token=None,
    )
    session = MagicMock(region_name='us-east-1')
    session.get_credentials.return_value.get_frozen_credentials.return_value = (
        frozen)
    connector.require_explicit_credentials()
    connector.set_credentials({
        'provider': 's3',
        'ambient_session': session,
    })
    monkeypatch.setattr(
        connector,
        '_direct_parquet_paths',
        lambda _source_name: ['s3://customer-bucket/root/events/part.parquet'],
    )
    monkeypatch.setattr(
        'io_utils.fsutil.s3_connection.deployment_s3_connection',
        lambda: SimpleNamespace(
            access_key='deployment-access-key',
            secret_key='deployment-secret-key',
        ),
    )
    monkeypatch.setattr(
        file_connector, 'deploy_env', lambda: SimpleNamespace(
            data_plane_storage_root='s3://deployment-bucket/root'))

    assert not connector.uses_direct_diskgraph_source_for('events')


def test_direct_s3_subprocess_environment_uses_deployment_provider(monkeypatch):
    connector = CompositeFileConnector(
        'ambient_files',
        root_dir='s3://deployment-bucket/root/uploaded-datasets/default',
    )
    session = MagicMock(region_name=None)
    session.get_credentials.return_value = None
    connector.require_explicit_credentials()
    connector.set_credentials({
        'provider': 's3',
        'ambient_session': session,
    })
    monkeypatch.setattr(
        connector,
        '_direct_parquet_paths',
        lambda _source_name:
        ['s3://deployment-bucket/root/uploaded-datasets/default/part.parquet'],
    )
    monkeypatch.setattr(
        'io_utils.fsutil.s3_connection.deployment_s3_connection',
        lambda: SimpleNamespace(
            access_key='deployment-access-key',
            secret_key='deployment-secret-key',
        ),
    )
    expected = {'AWS_ENDPOINT_URL_S3': 'https://minio:9000'}
    monkeypatch.setattr(
        'io_utils.fsutil.s3_connection.s3_subprocess_environment',
        lambda: expected,
    )
    monkeypatch.setattr(
        file_connector, 'deploy_env', lambda: SimpleNamespace(
            data_plane_storage_root='s3://deployment-bucket/root'))

    assert connector.diskgraph_subprocess_environment('events') == expected
    session.get_credentials.assert_not_called()


def test_other_ambient_s3_source_stages_before_diskgraph(monkeypatch):
    connector = CompositeFileConnector(
        'ambient_files',
        root_dir='s3://customer-bucket/root',
    )
    session = MagicMock(region_name=None)
    session.get_credentials.return_value = None
    connector.require_explicit_credentials()
    connector.set_credentials({
        'provider': 's3',
        'ambient_session': session,
    })
    monkeypatch.setattr(
        connector,
        '_direct_parquet_paths',
        lambda _source_name: ['s3://customer-bucket/root/part.parquet'],
    )
    monkeypatch.setattr(
        'io_utils.fsutil.s3_connection.deployment_s3_connection',
        lambda: SimpleNamespace(
            access_key='deployment-access-key',
            secret_key='deployment-secret-key',
        ),
    )
    monkeypatch.setattr(
        file_connector, 'deploy_env', lambda: SimpleNamespace(
            data_plane_storage_root='s3://deployment-bucket/root'))

    assert not connector.uses_direct_diskgraph_source_for('events')


def test_explicit_file_identity_never_falls_back_to_ambient_direct(monkeypatch):
    connector = _connector(suffix='parquet')
    monkeypatch.setattr(
        connector,
        '_direct_parquet_paths',
        lambda _source_name: ['s3://customer-bucket/root/events/part.parquet'],
    )
    identity = MagicMock()
    monkeypatch.setattr('kumo.connector.file_connector.fsutil.object_identity',
                        identity)

    assert not connector.uses_direct_diskgraph_source_for('events')
    with pytest.raises(ConnectorError, match='ambient connector identity'):
        connector.diskgraph_source(source_name='events')
    identity.assert_not_called()


@pytest.mark.parametrize(
    ('path', 'credentials', 'require_explicit', 'expected'),
    [
        (
            's3://bucket/events/part.parquet',
            {},
            False,
            True,
        ),
        (
            'gs://bucket/events/part.parquet',
            GoogleCloudCredentials(
                provider='gcs',
                credential=object(),
                ambient=True,
            ).as_dict(),
            True,
            True,
        ),
        (
            'gs://bucket/events/part.parquet',
            GoogleCloudCredentials(
                provider='gcs',
                credential=object(),
            ).as_dict(),
            True,
            False,
        ),
        (
            'abfss://container@account.dfs.core.windows.net/events/'
            'part.parquet',
            {},
            False,
            True,
        ),
        (
            'abfss://container@account.dfs.core.windows.net/events/'
            'part.parquet',
            AzureFederatedCredentials(credential=object()).as_dict(),
            True,
            False,
        ),
    ],
)
def test_direct_provider_boundary_accepts_only_current_ambient_products(
        monkeypatch, path, credentials, require_explicit, expected):
    connector = CompositeFileConnector('files', root_dir=path)
    if require_explicit:
        connector.require_explicit_credentials()
    if credentials:
        connector.set_credentials(credentials)
    monkeypatch.setattr(connector, '_direct_parquet_paths',
                        lambda _source_name: [path])

    assert connector.uses_direct_diskgraph_source_for('events') is expected


@pytest.mark.parametrize(
    ('connector_type', 'uri'),
    [
        (
            AnyURIS3Connector,
            's3://customer-bucket/events/part.parquet',
        ),
        (
            AnyURIGCSConnector,
            'gs://customer-bucket/events/part.parquet',
        ),
        (
            AnyURIADLSConnector,
            'abfss://events@customer.dfs.core.windows.net/part.parquet',
        ),
    ],
)
def test_any_uri_ambient_flat_parquet_builds_direct_manifest(
        monkeypatch, connector_type, uri):
    connector = connector_type('ambient_files')
    version = '123' if connector_type is AnyURIGCSConnector else None

    def select_region(path):
        assert connector_type is AnyURIS3Connector
        assert path == uri

    monkeypatch.setattr('io_utils.fsutil.s3_filesystem.select_s3_region',
                        select_region)
    monkeypatch.setattr('kumo.connector.any_uri_file_connector.fsutil.isdir',
                        lambda path: False)
    monkeypatch.setattr('kumo.connector.any_uri_file_connector.fsutil.exists',
                        lambda path: path == uri)
    monkeypatch.setattr(
        'kumo.connector.file_connector.fsutil.object_identity',
        lambda path: ObjectIdentity(17, f'etag:{path}', version),
    )

    assert connector.uses_direct_diskgraph_source_for(uri)
    assert connector.diskgraph_source(source_name=uri) == {
        'type':
        'object_store_parquet',
        'objects': [{
            'path': uri,
            'size': 17,
            'e_tag':
            None if connector_type is AnyURIGCSConnector else f'etag:{uri}',
            'version': version,
        }],
        'auth_profile':
        'ambient',
    }


def test_flat_parquet_classifier_rejects_hive_and_accepts_flat(tmp_path):
    from kumo.connector.file_connector import _flat_parquet_files

    flat = tmp_path / 'flat'
    flat.mkdir()
    (flat / 'a.parquet').write_bytes(b'a')
    (flat / 'b.parquet').write_bytes(b'b')
    assert _flat_parquet_files(str(flat)) == [
        str(flat / 'a.parquet'),
        str(flat / 'b.parquet'),
    ]

    hive = tmp_path / 'hive'
    partition = hive / 'day=2026-07-28'
    partition.mkdir(parents=True)
    (partition / 'a.parquet').write_bytes(b'a')
    assert _flat_parquet_files(str(hive)) is None


def test_composite_direct_parquet_resolution_reuses_one_scan_per_identity(
        monkeypatch, tmp_path):
    from kumo.connector import composite_file_connector

    events = tmp_path / 'events'
    events.mkdir()
    (events / 'a.parquet').write_bytes(b'a')
    (events / 'b.parquet').write_bytes(b'b')
    (events / '_temporary.parquet').write_bytes(b'ignored')
    connector = CompositeFileConnector('files', root_dir=str(tmp_path))
    scan = MagicMock(wraps=composite_file_connector._scan_dataset_files)
    monkeypatch.setattr(composite_file_connector, '_scan_dataset_files', scan)

    expected = [
        str(events / 'a.parquet'),
        str(events / 'b.parquet'),
    ]
    assert connector._cached_direct_parquet_paths('events') == expected
    assert connector._cached_direct_parquet_paths('events') == expected
    scan.assert_called_once_with(str(events), ('parquet', 'csv'))
    connector.require_explicit_credentials()
    assert connector._cached_direct_parquet_paths('events') == expected
    assert scan.call_count == 2


def _mock_remote_file(monkeypatch, tmp_path, external_path):
    from pyarrow.fs import FileInfo, FileType

    payload = tmp_path / 'source'
    if external_path.endswith('.csv'):
        payload.write_text('id,value\n1,a\n2,b\n')
    else:
        pq.write_table(pa.table({'id': [1, 2]}), payload)
    events = []
    scratch = []
    published_sizes = []
    caller_thread = threading.get_ident()

    class Filesystem:
        def __init__(self):
            _assert_customer_scope()
            self.authorized = credentials_for_provider('s3')

        def get_file_info(self, path):
            _assert_customer_scope()
            assert path == external_path.removeprefix('s3://')
            return FileInfo(path, FileType.File, size=payload.stat().st_size)

    # The provider double also serves schema-only reads before materialization.
    read_schema = pq.read_schema

    def provider_schema(path, *, filesystem=None, **kwargs):
        if isinstance(filesystem, Filesystem):
            # Like the real Arrow filesystem, retain credentials captured at
            # construction; worker threads do not inherit the caller context.
            assert filesystem.authorized.credentials == _credentials()
            assert path == external_path.removeprefix('s3://')
            return read_schema(payload, **kwargs)
        return read_schema(path, filesystem=filesystem, **kwargs)

    monkeypatch.setattr(pq, 'read_schema', provider_schema)

    def access(path):
        from kumo_core.sources.object_store_access import ObjectStoreAccess

        _assert_customer_scope()
        assert path == external_path
        authorized = credentials_for_provider('s3')

        def fetch(source, destination):
            assert authorized.credentials == _credentials()
            assert source == external_path.removeprefix('s3://')
            assert threading.get_ident() != caller_thread
            ancestor = destination
            while not os.path.basename(ancestor).startswith(
                ('kumo-file-source-', 'kumo-raw-source-')):
                ancestor = os.path.dirname(ancestor)
                assert ancestor != '/'
            assert stat.S_IMODE(os.stat(ancestor).st_mode) == 0o700
            scratch.append(ancestor)
            with open(destination, 'wb') as output:
                output.write(payload.read_bytes())
            events.append('download')

        return ObjectStoreAccess(Filesystem(), fetch)

    def isdir(path):
        if path == external_path:
            _assert_customer_scope()
        else:
            _assert_deployment_scope()
        return False

    def exists(path):
        _assert_deployment_scope()
        assert path.startswith('s3://example-internal/')
        return False

    def copy_file(source, destination, overwrite=False, *args, **kwargs):
        assert overwrite
        assert source != external_path
        _assert_deployment_scope()
        assert os.path.isfile(source)
        assert destination.endswith('.parquet')
        assert pq.read_table(source).column('id').to_pylist() == [1, 2]
        published_sizes.append(os.path.getsize(source))
        events.append('publish')

    def rm(path, **kwargs):
        _assert_deployment_scope()
        assert kwargs['ignore_errors'] is False
        events.append('clear')

    def makedirs(path, **kwargs):
        _assert_deployment_scope()
        assert kwargs['exist_ok']

    monkeypatch.setattr('kumo.connector.object_store_access.source_access',
                        access)
    monkeypatch.setattr(file_connector.fsutil, 'isdir', isdir)
    monkeypatch.setattr(file_connector.fsutil, 'copy_file', copy_file)
    monkeypatch.setattr('io_utils.fsutil.generic.copy_file', copy_file)
    monkeypatch.setattr(file_connector.fsutil, 'rm', rm)
    monkeypatch.setattr(file_connector.fsutil, 'makedirs', makedirs)
    # Directory copies check the destination through the API module directly.
    monkeypatch.setattr('io_utils.fsutil.api.exists', exists)
    return payload, events, scratch, published_sizes


@pytest.mark.parametrize('suffix', ['parquet', 'csv'])
def test_composite_export_separates_customer_and_internal_s3(
        monkeypatch, tmp_path, suffix):
    connector = _connector(suffix=suffix)
    source_name = f'deep/nested/events.{suffix}'
    external_path = f's3://customer-bucket/root/{source_name}'
    payload, events, scratch, sizes = _mock_remote_file(monkeypatch, tmp_path,
                                                        external_path)

    def resolve(name):
        _assert_customer_scope()
        assert name == source_name
        return external_path

    monkeypatch.setattr(connector, '_get_path', resolve)
    monkeypatch.setattr('kumo.data.utils.get_materialize_path',
                        lambda *_a, **_k: 's3://example-internal/ingest/events')
    metrics = connector.export_table(source_name, _fingerprint(source_name),
                                     'v1')
    assert metrics.num_rows == (2 if suffix == 'csv' else None)
    assert metrics.source_size == (sum(sizes) if suffix == 'csv' else
                                   payload.stat().st_size)
    assert events == ['download', 'clear', 'publish']
    assert scratch and all(not os.path.exists(path) for path in scratch)


def test_composite_download_never_uses_customer_scope_or_external_source(
        monkeypatch, tmp_path):
    connector = _connector(suffix='parquet')
    internal_path = 's3://example-internal/ingest/events'
    local_destination = str(tmp_path / 'events')
    external_resolve = MagicMock(
        side_effect=AssertionError('download must not resolve customer data'))

    def isdir(path):
        _assert_deployment_scope()
        assert path == internal_path
        return False

    def copy_file(source, destination, *, overwrite=False, **_kwargs):
        _assert_deployment_scope()
        assert (source, destination, overwrite) == (
            internal_path,
            local_destination,
            True,
        )

    monkeypatch.setattr(connector, '_get_path', external_resolve)
    monkeypatch.setattr('kumo.connector.file_connector.fsutil.isdir', isdir)
    monkeypatch.setattr('kumo.connector.file_connector.fsutil.copy_file',
                        copy_file)
    monkeypatch.setattr(
        'kumo.data.utils.get_materialize_path',
        lambda *_args, **_kwargs: internal_path,
    )

    connector.download(
        'deep/events.parquet',
        _fingerprint('deep/events.parquet'),
        'v1',
        local_destination,
    )

    external_resolve.assert_not_called()


def test_parquet_dataset_directory_still_round_trips_through_local_stage(
        tmp_path):
    source_root = tmp_path / 'source'
    source_dir = source_root / 'deep' / 'events'
    source_dir.mkdir(parents=True)
    pq.write_table(pa.table({'id': [1, 2]}), source_dir / 'part-0.parquet')
    pq.write_table(pa.table({'id': [3]}), source_dir / 'part-1.parquet')

    source_name = 'deep/events'
    connector = CompositeFileConnector('customer_files', str(source_root))
    fingerprint = _fingerprint(source_name)

    metrics = connector.export_table(source_name, fingerprint, 'v1')
    local_download = tmp_path / 'download'
    connector.download(source_name, fingerprint, 'v1', str(local_download))

    assert metrics.num_rows is None  # file/parquet connector: no row scan
    assert sorted(path.name for path in local_download.glob('*.parquet')) == [
        'part-0.parquet',
        'part-1.parquet',
    ]


def _custom_spec(connector_ref=None, *, table_name='deep/events.parquet'):
    connector_id = getattr(connector_ref, 'connector_id', None)
    if not isinstance(connector_id, str):
        connector_id = 'connector_' + 'a' * 32
    return SimpleNamespace(
        source_table=S3SourceTable(
            connector_id=connector_id,
            source={
                'mode': 'relative',
                'path': table_name
            },
            file_type=FileType.PARQUET,
            data_source_type=DataSourceType.S3,
        ),
        connector_spec=connector_ref or MagicMock(),
    )


def _ingest(spec, destination):
    return custom_train_table.ingest_custom_train_table(
        custom_train_table_spec=spec,
        custom_train_table_path=destination,
        deploy_config=MagicMock(),
        id='training-job',
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
        list_distinct_target_is_array=False,
    )


@pytest.mark.parametrize('as_dataset', [False, True])
def test_custom_training_csv_is_converted_to_real_parquet(
        monkeypatch, tmp_path, as_dataset):
    source_root = tmp_path / 'source'
    source_root.mkdir()
    if as_dataset:
        source_name = 'deep/events'
        source = source_root / source_name
        source.mkdir(parents=True)
        (source / 'part-0.csv').write_text(
            'id,label,weight\n1,one,1.5\n2,two,2.5\n',
            encoding='utf-8',
        )
        (source / 'part-1.csv').write_text(
            'id,label,weight\n3,three,3.5\n',
            encoding='utf-8',
        )
    else:
        source_name = 'deep/events.csv'
        source = source_root / source_name
        source.parent.mkdir(parents=True)
        source.write_text(
            'id,label,weight\n1,one,1.5\n2,two,2.5\n3,three,3.5\n',
            encoding='utf-8',
        )

    connector = CompositeFileConnector('customer_files', str(source_root))
    monkeypatch.setattr(custom_train_table, 'get_connector_from_spec',
                        lambda _ref: connector)
    destination = str(tmp_path / 'custom-training')

    result_path, response = _ingest(
        _custom_spec(table_name=source_name),
        destination,
    )

    assert result_path == destination
    assert response.ok
    output = pq.read_table(destination)
    assert output.schema == pa.schema([
        pa.field('id', pa.int64()),
        pa.field('label', pa.string()),
        pa.field('weight', pa.float64()),
    ])
    assert output.to_pydict() == {
        'id': [1, 2, 3],
        'label': ['one', 'two', 'three'],
        'weight': [1.5, 2.5, 3.5],
    }


def test_custom_training_activity_stages_before_same_provider_write(
        monkeypatch, tmp_path):
    connector = _connector(suffix='parquet')
    connector.close = MagicMock()
    external_path = 's3://customer-bucket/root/deep/events.parquet'
    _, events, scratch, _ = _mock_remote_file(monkeypatch, tmp_path,
                                              external_path)
    monkeypatch.setattr(connector, '_get_path', lambda _name: external_path)
    monkeypatch.setattr(custom_train_table, 'get_connector_from_spec',
                        lambda _ref: connector)
    internal_path = 's3://example-internal/custom-training'
    result_path, response = _ingest(_custom_spec(), internal_path)
    assert result_path == internal_path
    assert response.ok
    assert events == ['download', 'clear', 'publish']
    connector.close.assert_called_once_with()
    assert scratch and all(not os.path.exists(path) for path in scratch)


def test_custom_training_rechecks_binding_before_external_read(
        monkeypatch, caplog):
    poison = 'provider-secret-from-unbound-identity'
    reference = ConnectorExecutionRef(
        connector_id=ConnectorID.generate(),
        project_id=ProjectID.generate(),
    )
    external_read = MagicMock()
    monkeypatch.setattr(custom_train_table.fsutil, 'copy_file', external_read)

    with patch(
            'rest.backends.data_backend.connector_broker.'
            'connector_credential_broker.resolve',
            side_effect=PermissionError(poison),
    ):
        _, response = _ingest(_custom_spec(reference), '/internal/custom')

    assert not response.ok
    external_read.assert_not_called()
    rendered = repr(response)
    assert poison not in rendered
    assert poison not in caplog.text


def test_custom_training_sanitizes_provider_failure_and_closes_connector(
        monkeypatch, caplog):
    connector = _connector(suffix='parquet')
    connector.close = MagicMock()
    poison = 'provider-secret-from-staging-failure'
    external_path = 's3://customer-bucket/root/deep/events.parquet'
    monkeypatch.setattr(connector, '_get_path', lambda _name: external_path)
    monkeypatch.setattr(custom_train_table, 'get_connector_from_spec',
                        lambda _ref: connector)
    monkeypatch.setattr(custom_train_table.fsutil, 'isdir', lambda _path: False)
    access = MagicMock(side_effect=RuntimeError(poison))
    monkeypatch.setattr('kumo.connector.object_store_access.source_access',
                        access)

    _, response = _ingest(_custom_spec(), '/internal/custom')

    access.assert_called_once_with(external_path)
    assert not response.ok
    assert poison not in repr(response)
    assert poison not in caplog.text
    connector.close.assert_called_once_with()


def test_custom_training_cancellation_cleans_stage_and_closes_connector(
        monkeypatch, tmp_path):
    connector = _connector(suffix='parquet')
    connector.close = MagicMock()
    external_path = 's3://customer-bucket/root/deep/events.parquet'
    _, events, scratch, _ = _mock_remote_file(monkeypatch, tmp_path,
                                              external_path)
    copy_file = file_connector.fsutil.copy_file

    def cancel_publication(source, destination, *args, **kwargs):
        if source == external_path:
            return copy_file(source, destination, *args, **kwargs)
        _assert_deployment_scope()
        events.append('cancel publication')
        raise asyncio.CancelledError()

    monkeypatch.setattr(connector, '_get_path', lambda _name: external_path)
    monkeypatch.setattr(custom_train_table, 'get_connector_from_spec',
                        lambda _ref: connector)
    monkeypatch.setattr(file_connector.fsutil, 'copy_file', cancel_publication)
    monkeypatch.setattr('io_utils.fsutil.generic.copy_file', cancel_publication)
    with pytest.raises(asyncio.CancelledError):
        _ingest(_custom_spec(), 's3://example-internal/custom-training')
    assert events == ['download', 'clear', 'cancel publication']
    connector.close.assert_called_once_with()
    assert scratch and all(not os.path.exists(path) for path in scratch)


def test_custom_training_unsupported_connector_still_closes(monkeypatch):
    connector = MagicMock()
    del connector.export_table_read_only
    del connector.export_table_to_blob_storage
    monkeypatch.setattr(custom_train_table, 'get_connector_from_spec',
                        lambda _ref: connector)

    with pytest.raises(UnsupportedInDiskgraphError):
        _ingest(_custom_spec(), '/internal/custom')

    connector.close.assert_called_once_with()


@pytest.mark.parametrize('export_fails', [False, True])
def test_custom_training_warehouse_connector_closes_on_every_path(
        monkeypatch, tmp_path, export_fails):
    connector = MagicMock()
    # A bare MagicMock synthesizes every attribute on access. Remove the
    # higher-priority read-only export explicitly so this exercises the blob
    # export path configured below, matching a real connector class.
    del connector.export_table_read_only

    def export(table_name, destination):
        assert table_name == 'deep/events.parquet'
        if export_fails:
            raise RuntimeError('sanitized-provider-failure')
        os.makedirs(destination, exist_ok=True)
        pq.write_table(pa.table({'ENTITY1': [1], 'TARGET1': [0]}),
                       os.path.join(destination, 'part-0.parquet'))

    connector.export_table_to_blob_storage.side_effect = export
    monkeypatch.setattr(custom_train_table, 'get_connector_from_spec',
                        lambda _ref: connector)
    monkeypatch.setattr(custom_train_table.fsutil, 'rm', lambda *_a, **_k: None)

    destination = tmp_path / 'custom'
    _, response = _ingest(_custom_spec(), str(destination))

    assert response.ok is not export_fails
    connector.close.assert_called_once_with()


@pytest.mark.parametrize(('scheme', 'provider'), [('s3', 's3'), ('gs', 'gcs')])
def test_source_versions_share_recursive_reader_under_customer_scope(
        tmp_path, monkeypatch, scheme, provider):
    from kumo_core.sources.object_store import ObjectStoreSourceReader
    from kumo_core.sources.object_store_access import ObjectStoreAccess
    from pyarrow.fs import FileSelector, LocalFileSystem, SubTreeFileSystem

    from kumo.connector import object_store_access
    from kumo.connector.connector import SourceTableNotFoundError

    root = tmp_path / 'bucket' / 'root'
    nested = root / 'events' / 'country=US'
    nested.mkdir(parents=True)
    direct = root / 'events' / 'part.parquet'
    direct.touch()
    (nested / 'part.parquet').touch()
    os.utime(direct, (2_000_000_000, 2_000_000_000))
    os.utime(nested / 'part.parquet', (3_000_000_000, 3_000_000_000))
    filesystem = SubTreeFileSystem(str(tmp_path), LocalFileSystem())

    class MetadataOnlyFilesystem:
        def get_file_info(self, target):
            if isinstance(target, FileSelector):
                assert target.recursive
            return filesystem.get_file_info(target)

    metadata = MetadataOnlyFilesystem()
    credentials = (_credentials()
                   if provider == 's3' else GoogleCloudCredentials(
                       provider='gcs', credential=object()).as_dict())
    connector = CompositeFileConnector('version_source',
                                       f'{scheme}://bucket/root')
    connector.set_credentials(credentials)
    connector.require_explicit_credentials()
    monkeypatch.setattr(connector, '_open', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(connector, 'validate_permissions', lambda: None)

    def access(_path):
        scope = credentials_for_provider(provider)
        assert scope is not None and scope.credentials == credentials
        return ObjectStoreAccess(metadata)

    def forbidden(*_args, **_kwargs):
        pytest.fail(
            'Version lookup must not use fsutil timestamps or format discovery')

    monkeypatch.setattr(object_store_access, 'source_access', access)
    monkeypatch.setattr(file_connector.fsutil, 'get_last_modified_time',
                        forbidden)
    monkeypatch.setattr(connector, '_resolve_table', forbidden)
    local_reader = ObjectStoreSourceReader(metadata, root='bucket/root')

    def version():
        return connector.get_data_version('events', skip_cache=True,
                                          save_to_cache=False)

    original = version()
    assert original == local_reader.data_version('events')
    # Changing an older partition must invalidate the version even when
    # another part keeps the maximum modification time unchanged.
    os.utime(direct, (2_000_000_001, 2_000_000_001))
    changed = version()
    assert changed == local_reader.data_version('events')
    assert changed != original
    direct.unlink()
    removed = version()
    assert removed == local_reader.data_version('events')
    assert removed != changed
    os.utime(nested / 'part.parquet', (3_000_000_001, 3_000_000_001))
    assert version() == local_reader.data_version('events') != removed
    with pytest.raises(SourceTableNotFoundError):
        connector.get_data_version('missing', skip_cache=True,
                                   save_to_cache=False)
