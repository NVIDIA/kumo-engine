# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import ANY, Mock, call

import pytest
from botocore.exceptions import ClientError

from io_utils.fsutil.credentials import (
    ContextPropagatingThreadPoolExecutor,
    ExplicitFilesystemCredentialsRequired,
    credential_scope,
    credentials_for_provider,
    pyarrow_filesystem,
)
from io_utils.fsutil.impl_adls import ThreadsafeADLS
from io_utils.fsutil.impl_gcs import ThreadLocalGCSClient, _client_bucket
from io_utils.fsutil.s3_connection import S3Connection, bucket_owner_write_args
from io_utils.fsutil.s3_filesystem import (
    ThreadLocalBotoSession,
    select_s3_region,
)
from kumo.connector import CompositeFileConnector
from service_framework._raw_deploy_config import S3AddressingStyle


def test_explicit_scope_never_falls_back_to_another_or_ambient_provider(
) -> None:
    with credential_scope({'provider': 's3'}, require_explicit=True):
        with pytest.raises(ExplicitFilesystemCredentialsRequired):
            credentials_for_provider('gcs')


def test_s3_client_uses_only_scoped_federated_session(monkeypatch) -> None:
    session = Mock()
    session.client.return_value = object()
    session.resource.return_value = object()
    constructor = Mock(return_value=session)
    monkeypatch.setattr('io_utils.fsutil.s3_filesystem.boto3.session.Session',
                        constructor)
    boto = ThreadLocalBotoSession()
    credentials = {
        'provider': 's3',
        'aws_access_key_id': 'temporary-access-key',
        'aws_secret_access_key': 'temporary-secret',
        'aws_session_token': 'temporary-token',
        'region': 'us-west-2',
    }

    with credential_scope(credentials, require_explicit=True):
        boto.s3_client()
        boto.s3_resource()

    constructor.assert_called_once_with(
        aws_access_key_id='temporary-access-key',
        aws_secret_access_key='temporary-secret',
        aws_session_token='temporary-token',
        region_name='us-west-2',
    )
    session.client.assert_called_once()
    session.resource.assert_called_once()


def test_deployment_s3_client_uses_minio_connection(monkeypatch) -> None:
    session = Mock()
    session.client.return_value = object()
    constructor = Mock(return_value=session)
    connection = S3Connection(
        endpoint_url='https://minio.example.test:9000',
        region='us-east-1',
        addressing_style=S3AddressingStyle.PATH,
        credential_secret_name='MINIO_KUMO_CREDENTIALS',
        access_key='long-lived-access',
        secret_key='long-lived-secret',
        session_token=None,
        ca_bundle_path='/etc/kumo/minio-ca/ca.crt',
        send_bucket_owner_full_control=False,
    )
    monkeypatch.setattr('io_utils.fsutil.s3_filesystem.boto3.session.Session',
                        constructor)
    monkeypatch.setattr(
        'io_utils.fsutil.s3_filesystem.active_s3_connection',
        lambda: connection,
    )

    boto = ThreadLocalBotoSession()
    assert boto.s3_client() is session.client.return_value

    constructor.assert_called_once_with(
        aws_access_key_id='long-lived-access',
        aws_secret_access_key='long-lived-secret',
        aws_session_token=None,
        region_name='us-east-1',
    )
    kwargs = session.client.call_args.kwargs
    assert kwargs['endpoint_url'] == 'https://minio.example.test:9000'
    assert kwargs['region_name'] == 'us-east-1'
    assert kwargs['verify'] == '/etc/kumo/minio-ca/ca.crt'
    assert kwargs['config'].signature_version == 's3v4'
    assert kwargs['config'].s3['addressing_style'] == 'path'


def test_connector_s3_connection_is_independent_from_deployment(
        monkeypatch) -> None:
    deployment = S3Connection(
        endpoint_url='https://dataplane-minio.example.test:9000',
        region='us-west-2',
        addressing_style=S3AddressingStyle.VIRTUAL,
        credential_secret_name='DEPLOYMENT_SECRET',
        access_key='deployment-access',
        secret_key='deployment-secret',
        session_token=None,
        ca_bundle_path=None,
        send_bucket_owner_full_control=False,
    )
    connector = S3Connection(
        endpoint_url='https://connector-minio.example.test:9443',
        region='us-east-1',
        addressing_style='path',
        credential_secret_name='',
        access_key='connector-access',
        secret_key='connector-secret',
        session_token=None,
        ca_bundle_path=None,
        send_bucket_owner_full_control=False,
        owner='connector:identity-1',
        rotation_id='rotation-2',
    )
    session = Mock()
    session.client.return_value = object()
    constructor = Mock(return_value=session)
    monkeypatch.setattr('io_utils.fsutil.s3_filesystem.boto3.session.Session',
                        constructor)
    monkeypatch.setattr(
        'io_utils.fsutil.s3_connection.deployment_s3_connection',
        lambda: deployment,
    )
    boto = ThreadLocalBotoSession()

    with credential_scope(
        {
            'provider': 's3',
            'aws_access_key_id': 'connector-access',
            'aws_secret_access_key': 'connector-secret',
            'aws_session_token': None,
            'region': 'us-east-1',
            's3_connection': connector,
        }, require_explicit=True):
        boto.s3_client()
        assert bucket_owner_write_args() == {}

    constructor.assert_called_once_with(
        aws_access_key_id='connector-access',
        aws_secret_access_key='connector-secret',
        aws_session_token=None,
        region_name='us-east-1',
    )
    kwargs = session.client.call_args.kwargs
    assert kwargs['endpoint_url'] == connector.endpoint_url
    assert kwargs['region_name'] == connector.region
    assert kwargs['config'].s3['addressing_style'] == 'path'
    assert deployment.endpoint_url not in repr(session.client.call_args)


def test_custom_connector_endpoint_skips_aws_bucket_region_discovery(
        monkeypatch) -> None:
    connection = S3Connection(
        endpoint_url='https://connector-minio.example.test:9443',
        region='us-east-1',
        addressing_style='path',
        credential_secret_name='',
        access_key='connector-access',
        secret_key='connector-secret',
        session_token=None,
        ca_bundle_path=None,
        send_bucket_owner_full_control=False,
        owner='connector:identity-1',
    )
    client = Mock()
    monkeypatch.setattr(
        'io_utils.fsutil.s3_filesystem.threadsafe_boto.s3_client',
        Mock(return_value=client),
    )

    with credential_scope(
        {
            'provider': 's3',
            'aws_access_key_id': 'connector-access',
            'aws_secret_access_key': 'connector-secret',
            'region': 'us-east-1',
            's3_connection': connection,
        }, require_explicit=True):
        select_s3_region('s3://customer-bucket/data')

    client.head_bucket.assert_not_called()


def test_rootless_s3_selects_each_uri_bucket_region(monkeypatch) -> None:
    client = Mock()
    client.head_bucket.side_effect = [
        ClientError(
            {
                'Error': {
                    'Code': '403',
                    'Message': 'Forbidden'
                },
                'ResponseMetadata': {
                    'HTTPHeaders': {
                        'x-amz-bucket-region': 'us-west-2'
                    }
                },
            }, 'HeadBucket'),
        {
            'BucketRegion': 'eu-central-1'
        },
    ]
    monkeypatch.setattr(
        'io_utils.fsutil.s3_filesystem.threadsafe_boto.s3_client',
        Mock(return_value=client),
    )
    credentials = {'provider': 's3', 'ambient_session': object()}
    with credential_scope(credentials, require_explicit=True):
        select_s3_region('s3://west-bucket/output')
        assert credentials_for_provider(
            's3').credentials['region'] == 'us-west-2'
        select_s3_region('s3://eu-bucket/output')
        assert credentials_for_provider(
            's3').credentials['region'] == 'eu-central-1'

    assert client.head_bucket.call_args_list == [
        call(Bucket='west-bucket'),
        call(Bucket='eu-bucket'),
    ]


def test_pyarrow_s3_resolves_region_for_ambient_session(monkeypatch) -> None:
    frozen = Mock(
        access_key='access-key',
        secret_key='secret-key',
        token='session-token',
    )
    session = Mock()
    session.get_credentials.return_value.get_frozen_credentials.return_value = (
        frozen)
    credentials = {'provider': 's3', 'ambient_session': session}

    def select_region(path: str) -> None:
        assert path == 's3://bucket/table'
        scope = credentials_for_provider('s3')
        assert scope is not None
        scope.credentials = {**(scope.credentials or {}), 'region': 'us-west-2'}

    constructor = Mock(return_value=object())
    monkeypatch.setattr(
        'io_utils.fsutil.s3_filesystem.select_s3_region',
        select_region,
    )
    monkeypatch.setattr('pyarrow.fs.S3FileSystem', constructor)

    with credential_scope(credentials, require_explicit=True):
        pyarrow_filesystem('s3://bucket/table')

    constructor.assert_called_once_with(
        access_key='access-key',
        secret_key='secret-key',
        session_token='session-token',
        anonymous=False,
        region='us-west-2',
        retry_strategy=ANY,
    )


def test_gcs_client_and_requester_pays_bucket_use_scoped_credential(
        monkeypatch) -> None:
    credential = object()
    client = Mock()
    constructor = Mock(return_value=client)
    monkeypatch.setattr('io_utils.fsutil.impl_gcs.storage.Client', constructor)
    clients = ThreadLocalGCSClient()
    credentials = {
        'provider': 'gcs',
        'credential': credential,
        'project': 'source-project',
        'requester_pays_project': 'billing-project',
    }

    with credential_scope(credentials, require_explicit=True):
        assert clients.client() is client
        _client_bucket(client, 'customer-bucket')
        _client_bucket(client, 'second-customer-bucket')

    constructor.assert_called_once_with(project='source-project',
                                        credentials=credential)
    assert client.bucket.call_args_list == [
        call('customer-bucket', user_project='billing-project'),
        call('second-customer-bucket', user_project='billing-project'),
    ]


def test_adls_client_uses_scoped_credential_and_connector_endpoint(
        monkeypatch) -> None:
    credential = object()
    client = object()
    constructor = Mock(return_value=client)
    monkeypatch.setattr('io_utils.fsutil.impl_adls._create_datalake_client',
                        constructor)
    clients = ThreadsafeADLS()

    with credential_scope({
            'provider': 'adls',
            'credential': credential,
    }, require_explicit=True):
        assert clients.data_lake_client(
            'customerlake',
            'dfs.core.usgovcloudapi.net',
        ) is client

    constructor.assert_called_once_with(
        account_url=('https://customerlake.dfs.core.usgovcloudapi.net'),
        credential=credential,
        audience='https://storage.azure.us',
    )


def test_identity_backed_file_connector_enters_credential_scope(
        monkeypatch) -> None:
    connector = CompositeFileConnector(
        name='connector_' + 'a' * 32,
        root_dir='s3://customer-bucket/prefix',
    )
    connector.require_explicit_credentials()
    connector.set_credentials({
        'provider': 's3',
        'aws_access_key_id': 'temporary-access-key',
        'aws_secret_access_key': 'temporary-secret',
        'aws_session_token': 'temporary-token',
    })

    from kumo_core.sources.object_store_access import ObjectStoreAccess
    from pyarrow.fs import FileInfo, FileSelector, FileType

    def access(_path):
        scope = credentials_for_provider('s3')
        assert scope is not None
        assert scope.credentials['aws_session_token'] == 'temporary-token'
        filesystem = Mock()
        filesystem.get_file_info.side_effect = lambda request: (
            [FileInfo('customer-bucket/prefix/events.parquet', FileType.File)]
            if isinstance(request, FileSelector) else FileInfo(
                request, FileType.Directory))
        return ObjectStoreAccess(filesystem)

    monkeypatch.setattr('kumo.connector.object_store_access.source_access',
                        access)

    assert connector.list_tables() == ['events']


@pytest.mark.parametrize('provider', ['s3', 'gcs', 'adls'])
def test_executor_submit_and_map_propagate_exact_provider_scope(
        provider) -> None:
    credentials = {
        'provider': provider,
        'marker': f'{provider}-customer-credential',
    }

    def read_marker(_value=None):
        scope = credentials_for_provider(provider)
        assert scope is not None
        return scope.credentials['marker']

    with ContextPropagatingThreadPoolExecutor(max_workers=2) as executor:
        with credential_scope(credentials, require_explicit=True):
            assert executor.submit(
                read_marker).result() == credentials['marker']
            assert list(executor.map(read_marker, range(3))) == [
                credentials['marker'],
                credentials['marker'],
                credentials['marker'],
            ]


def test_executor_worker_fails_closed_on_provider_mismatch_and_does_not_leak(
) -> None:
    with ContextPropagatingThreadPoolExecutor(max_workers=1) as executor:
        with credential_scope({'provider': 's3'}, require_explicit=True):
            with pytest.raises(ExplicitFilesystemCredentialsRequired):
                executor.submit(credentials_for_provider, 'gcs').result()

        # The same long-lived worker must not retain the preceding S3 scope.
        assert executor.submit(credentials_for_provider, 's3').result() is None


def test_large_identity_backed_s3_copy_never_uses_process_pool(
        monkeypatch) -> None:
    from io_utils.fsutil import s3_filesystem

    filesystem = s3_filesystem.S3FileSystem(
        process_pool_cls=Mock(
            side_effect=AssertionError('process pool must not be used')),
        retry_delay_sec=0,
    )
    filesystem._copy_multiprocessing_threshold = 0
    monkeypatch.setattr(s3_filesystem, 's3_ctx', filesystem)
    monkeypatch.setattr(s3_filesystem, 'listdir_recursive',
                        lambda _path: ['source/piece.parquet'])

    copied = []

    def copy_batch(*args):
        scope = credentials_for_provider('s3')
        assert scope is not None
        copied.append(args)

    monkeypatch.setattr(s3_filesystem, '_copy_batch', copy_batch)

    with credential_scope(
        {
            'provider': 's3',
            'aws_access_key_id': 'temporary-access-key',
            'aws_secret_access_key': 'temporary-secret',
            'aws_session_token': 'temporary-token',
        }, require_explicit=True):
        s3_filesystem.copydir('s3://source/source',
                              's3://destination/destination')

    assert len(copied) == 1
