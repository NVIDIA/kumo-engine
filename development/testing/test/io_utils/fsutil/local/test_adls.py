# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import datetime
import hashlib
import io
import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import pytest
from azure.core.exceptions import ServiceRequestError

from io_utils import fsutil
from io_utils.fsutil import impl_adls
from io_utils.fsutil.credentials import credential_scope
from io_utils.fsutil.impl_adls import retry_policy, threadsafe_adls
from io_utils.fsutil.uri import ADLSURI, ParseURI
from kumo.testing.adls import adls_client  # noqa: F401 - pytest fixture
from kumo.testing.adls import adls_test_base_uri, withADLSCredentials


def test_is_adls():
    assert fsutil.is_adls('abfss://container@account.dfs.core.windows.net/path')
    assert fsutil.is_adls('abfs://container@account.dfs.core.windows.net/path')
    assert fsutil.is_adls('abfss://mycontainer@storage.dfs.core.windows.net')
    assert not fsutil.is_adls('/home/tester')
    assert not fsutil.is_adls('../tester/path')
    assert not fsutil.is_adls('s3://bucket/path')


def test_adls_credentials_decorator_does_not_probe_provider(monkeypatch):
    credential_env = (
        'AZURE_CREDENTIALS_PATH',
        'AZURE_STORAGE_CONNECTION_STRING',
        'AZURE_TENANT_ID',
        'AZURE_CLIENT_ID',
        'AZURE_CLIENT_SECRET',
        'AZURE_FEDERATED_TOKEN_FILE',
        'IDENTITY_ENDPOINT',
        'MSI_ENDPOINT',
        'IMDS_ENDPOINT',
        'KUMO_RUN_ADLS_TESTS',
    )
    for name in credential_env:
        monkeypatch.delenv(name, raising=False)

    def test_target():
        pass

    with patch('kumo.testing.adls.threadsafe_adls.data_lake_client'
               ) as data_lake_client:
        decorated = withADLSCredentials(test_target)

    data_lake_client.assert_not_called()
    skip_marks = [
        mark for mark in decorated.pytestmark if mark.name == 'skipif'
    ]
    assert len(skip_marks) == 1
    assert skip_marks[0].args == (True, )
    assert skip_marks[0].kwargs[
        'reason'] == 'No Azure credentials found for ADLS'


def test_adls_credentials_decorator_accepts_explicit_ci_opt_in(monkeypatch):
    monkeypatch.setenv('KUMO_RUN_ADLS_TESTS', '1')

    def test_target():
        pass

    decorated = withADLSCredentials(test_target)
    skip_marks = [
        mark for mark in decorated.pytestmark if mark.name == 'skipif'
    ]
    assert len(skip_marks) == 1
    assert skip_marks[0].args == (False, )


def test_adls_test_base_uri(monkeypatch):
    monkeypatch.setenv('AZURE_STORAGE_ACCOUNT', 'account')
    monkeypatch.setenv('AZURE_CONTAINER_NAME', 'container')
    assert adls_test_base_uri() == (
        'abfss://container@account.dfs.core.windows.net')


def test_adls_retry_does_not_log_provider_exception_text(caplog):
    provider_secret = 'azure-provider-bearer-secret'
    attempts = 0

    @retry_policy()
    def transient_operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ServiceRequestError(provider_secret)

    with patch('io_utils.retry.time.sleep'):
        transient_operation()

    assert attempts == 3
    assert provider_secret not in caplog.text


def test_parse_adls():
    uri = ParseURI('abfss://container@account.dfs.core.windows.net/path')
    assert isinstance(uri, ADLSURI)
    assert (uri.storage_account, uri.container, uri.path) == (
        'account',
        'container',
        'path',
    )

    uri = ParseURI(
        'abfss://mycontainer@storage.dfs.core.windows.net/dir1/dir2/file.txt')
    assert isinstance(uri, ADLSURI)
    assert (uri.storage_account, uri.container, uri.path) == (
        'storage',
        'mycontainer',
        'dir1/dir2/file.txt',
    )

    # Handle URIs without path
    uri = ParseURI('abfss://container@account.dfs.core.windows.net')
    assert isinstance(uri, ADLSURI)
    assert (uri.storage_account, uri.container, uri.path) == (
        'account',
        'container',
        '',
    )


@pytest.mark.parametrize('suffix', [
    'dfs.core.windows.net',
    'dfs.core.usgovcloudapi.net',
    'dfs.core.chinacloudapi.cn',
])
def test_adls_signed_download_uses_uri_endpoint_suffix(suffix):
    uri = ADLSURI(storage_account='account', container='filesystem',
                  endpoint_suffix=suffix, path='dir/file name.txt')
    service = MagicMock()
    service.get_user_delegation_key.return_value = object()
    with (
            patch.object(threadsafe_adls, 'data_lake_client',
                         return_value=service),
            patch.object(impl_adls, 'generate_file_sas',
                         return_value='sig=read-only') as generate,
    ):
        url = impl_adls.ADLSPresignedHelper(uri).generate_download_url()

    assert url.startswith(f'https://account.{suffix}/filesystem/')
    assert '/dir/file%20name.txt?' in url
    assert url.endswith('?sig=read-only')
    assert generate.call_args.kwargs['permission'].read
    assert not generate.call_args.kwargs['permission'].write


def test_adls_directory_sas_backdates_start_for_clock_skew():
    uri = ADLSURI(storage_account='account', container='filesystem',
                  path='dir/output')
    service = MagicMock()
    service.get_user_delegation_key.return_value = object()
    before = datetime.datetime.now(datetime.timezone.utc)
    with (
            patch.object(threadsafe_adls, 'data_lake_client',
                         return_value=service),
            patch.object(impl_adls, 'generate_directory_sas',
                         return_value='sig=read-only') as generate,
    ):
        assert impl_adls.generate_directory_read_sas(uri) == 'sig=read-only'
        default_delegation = service.get_user_delegation_key.call_args.kwargs
        default_sas = generate.call_args.kwargs

        service.reset_mock()
        generate.reset_mock()
        requested = datetime.timedelta(days=3)
        explicit_before = datetime.datetime.now(datetime.timezone.utc)
        assert impl_adls.generate_directory_read_sas(
            uri, validity=requested) == 'sig=read-only'
        explicit_delegation = (service.get_user_delegation_key.call_args.kwargs)
        explicit_sas = generate.call_args.kwargs

    assert default_delegation['key_start_time'] <= (
        before - datetime.timedelta(minutes=4, seconds=59))
    assert default_delegation['key_expiry_time'] >= (
        before + datetime.timedelta(hours=2))
    assert default_sas['start'] == default_delegation['key_start_time']
    assert default_sas['expiry'] == default_delegation['key_expiry_time']
    assert explicit_delegation['key_expiry_time'] >= (explicit_before +
                                                      requested)
    assert explicit_sas['start'] == explicit_delegation['key_start_time']
    assert explicit_sas['expiry'] == explicit_delegation['key_expiry_time']


def test_directory_sas_rejects_non_adls_backends_clearly():
    with pytest.raises(NotImplementedError, match='S3URI'):
        fsutil.generate_directory_read_sas('s3://bucket/directory')


@pytest.mark.parametrize('operation', ['download', 'multipart'])
def test_adls_delegation_failure_does_not_leak_provider_error(
        operation, caplog):
    provider_secret = 'azure-delegation-provider-secret'
    uri = ADLSURI(storage_account='account', container='filesystem',
                  path='dir/file.csv')
    service = MagicMock()
    service.get_user_delegation_key.side_effect = ServiceRequestError(
        provider_secret)

    with patch.object(threadsafe_adls, 'data_lake_client',
                      return_value=service):
        with pytest.raises(ValueError) as exc_info:
            helper = impl_adls.ADLSPresignedHelper(uri)
            if operation == 'download':
                helper.generate_download_url()
            else:
                helper.create_multipart_upload(1)

    assert provider_secret not in str(exc_info.value)
    assert provider_secret not in caplog.text
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(('suffix', 'audience', 'authority'), [
    (
        'dfs.core.windows.net',
        'https://storage.azure.com',
        impl_adls.AzureAuthorityHosts.AZURE_PUBLIC_CLOUD,
    ),
    (
        'dfs.core.usgovcloudapi.net',
        'https://storage.azure.us',
        impl_adls.AzureAuthorityHosts.AZURE_GOVERNMENT,
    ),
    (
        'dfs.core.chinacloudapi.cn',
        'https://storage.azure.cn',
        impl_adls.AzureAuthorityHosts.AZURE_CHINA,
    ),
])
def test_adls_ambient_clients_use_sovereign_authority_and_audience(
        monkeypatch, suffix, audience, authority):
    for key in ('AZURE_STORAGE_CONNECTION_STRING', 'AZURE_TENANT_ID',
                'AZURE_CLIENT_ID', 'AZURE_CLIENT_SECRET'):
        monkeypatch.delenv(key, raising=False)
    credential = object()
    default_credential = MagicMock(return_value=credential)
    data_lake = MagicMock(return_value=object())
    blob = MagicMock(return_value=object())
    clients = impl_adls.ThreadsafeADLS()

    with (
            patch.object(impl_adls, 'DefaultAzureCredential',
                         default_credential),
            patch.object(impl_adls, 'DataLakeServiceClient', data_lake),
            patch.object(impl_adls, 'BlobServiceClient', blob),
    ):
        clients._build_client('account', suffix)
        clients._build_blob_client('account', suffix)

    assert [call.kwargs for call in default_credential.call_args_list] == [
        {
            'authority': authority
        },
        {
            'authority': authority
        },
    ]
    data_lake.assert_called_once_with(
        account_url=f'https://account.{suffix}',
        credential=credential,
        retry_total=3,
        retry_backoff_factor=0.8,
        audience=audience,
    )
    blob.assert_called_once_with(
        account_url=(f'https://account.'
                     f'{impl_adls._blob_endpoint_suffix(suffix)}'),
        credential=credential,
        audience=audience,
    )


def test_adls_ambient_clients_share_client_secret_credential_selection(
        monkeypatch):
    monkeypatch.delenv('AZURE_STORAGE_CONNECTION_STRING', raising=False)
    monkeypatch.setenv('AZURE_TENANT_ID', 'tenant')
    monkeypatch.setenv('AZURE_CLIENT_ID', 'client')
    monkeypatch.setenv('AZURE_CLIENT_SECRET', 'secret')
    credential = object()
    client_secret = MagicMock(return_value=credential)
    clients = impl_adls.ThreadsafeADLS()

    with (
            patch.object(impl_adls, 'ClientSecretCredential', client_secret),
            patch.object(impl_adls, 'DefaultAzureCredential') as default,
            patch.object(impl_adls, 'DataLakeServiceClient'),
            patch.object(impl_adls, 'BlobServiceClient'),
    ):
        clients._build_client('account')
        clients._build_blob_client('account', 'dfs.core.windows.net')

    expected = {
        'tenant_id': 'tenant',
        'client_id': 'client',
        'client_secret': 'secret',
        'authority': impl_adls.AzureAuthorityHosts.AZURE_PUBLIC_CLOUD,
    }
    assert [call.kwargs
            for call in client_secret.call_args_list] == [expected, expected]
    default.assert_not_called()


def test_adls_scoped_credential_precedes_connection_string(monkeypatch):
    monkeypatch.setenv('AZURE_STORAGE_CONNECTION_STRING', 'ambient-secret')
    credential = object()
    data_lake = MagicMock(return_value=object())
    blob = MagicMock(return_value=object())
    clients = impl_adls.ThreadsafeADLS()

    with (
            credential_scope({
                'provider': 'adls',
                'credential': credential,
            }, require_explicit=True),
            patch.object(impl_adls, 'DataLakeServiceClient', data_lake),
            patch.object(impl_adls, 'BlobServiceClient', blob),
    ):
        clients._build_client('account')
        clients._build_blob_client('account', 'dfs.core.windows.net')

    data_lake.assert_called_once_with(
        account_url='https://account.dfs.core.windows.net',
        credential=credential,
        retry_total=3,
        retry_backoff_factor=0.8,
        audience='https://storage.azure.com',
    )
    data_lake.from_connection_string.assert_not_called()
    blob.assert_called_once_with(
        account_url='https://account.blob.core.windows.net',
        credential=credential,
        audience='https://storage.azure.com',
    )
    blob.from_connection_string.assert_not_called()


def test_adls_multipart_urls_are_blob_staging_urls_bound_to_parts():
    uri = ADLSURI(storage_account='account', container='filesystem',
                  endpoint_suffix='dfs.core.usgovcloudapi.net',
                  path='uploads/table.csv')
    service = MagicMock()
    service.get_user_delegation_key.return_value = object()
    blob = MagicMock()
    blob_service = MagicMock()
    blob_service.get_blob_client.return_value = blob
    with (
            patch.object(threadsafe_adls, 'data_lake_client',
                         return_value=service),
            patch.object(threadsafe_adls, 'blob_service_client',
                         return_value=blob_service),
            patch.object(impl_adls, 'generate_blob_sas',
                         return_value='sig=write-only') as generate,
    ):
        upload_id, urls = (
            impl_adls.ADLSPresignedHelper(uri).create_multipart_upload(3))

    state = impl_adls._decode_upload_state(upload_id, uri)
    assert set(urls) == {1, 2, 3}
    assert all(
        url.startswith(
            'https://account.blob.core.usgovcloudapi.net/filesystem/')
        for url in urls.values())
    assert all('&comp=block&blockid=' in url for url in urls.values())
    assert len(set(urls.values())) == 3
    expected_ids = {
        impl_adls._encoded_block_id(state['nonce'], part)
        for part in range(1, 4)
    }
    assert all(
        any(
            block_id.rstrip('=') in url or block_id in url
            for url in urls.values()) for block_id in expected_ids)
    permission = generate.call_args.kwargs['permission']
    assert permission.write and permission.create and not permission.read
    blob_service.get_blob_client.assert_called_once_with(
        'filesystem', state['temp_path'])
    blob.upload_blob.assert_called_once_with(
        b'',
        overwrite=False,
        metadata={impl_adls._UPLOAD_METADATA_KEY: upload_id},
    )


def test_adls_upload_id_rejects_destination_substitution():
    first = ADLSURI(storage_account='account', container='filesystem',
                    path='uploads/first.csv')
    second = first.model_copy(update={'path': 'uploads/second.csv'})
    upload_id, _ = impl_adls._upload_state(first, 2)
    with pytest.raises(ValueError, match='does not match destination'):
        impl_adls._decode_upload_state(upload_id, second)


class _Part:
    def __init__(self, part_number: int, etag: str = 'etag'):
        self.part_number = part_number
        self.etag = etag


def _multipart_fixture(num_parts=3):
    uri = ADLSURI(storage_account='account', container='filesystem',
                  path='uploads/table.csv')
    upload_id, _ = impl_adls._upload_state(uri, num_parts)
    state = impl_adls._decode_upload_state(upload_id, uri)
    blocks = [
        SimpleNamespace(id=impl_adls._raw_block_id(state['nonce'], number))
        for number in range(1, num_parts + 1)
    ]
    blob = MagicMock()
    blob.get_blob_properties.return_value = SimpleNamespace(
        metadata={
            impl_adls._UPLOAD_METADATA_KEY: upload_id,
        })
    blob.get_block_list.return_value = ([], blocks)
    service = MagicMock()
    service.get_blob_client.return_value = blob
    return uri, upload_id, state, blob, service


@contextmanager
def _multipart_clients(service, final=None, promotion=None):
    if final is None:
        final = MagicMock()
        final.get_file_properties.side_effect = impl_adls.ResourceNotFoundError(
            'missing')
    promotion = promotion or MagicMock()
    with (
            patch.object(threadsafe_adls, 'blob_service_client',
                         return_value=service),
            patch.object(impl_adls.adls_ctx, 'get_file_client',
                         return_value=final),
            patch.object(impl_adls, '_atomic_promote', promotion),
    ):
        yield final, promotion


@pytest.fixture
def promotion_clients():
    source = ADLSURI(storage_account='account', container='filesystem',
                     path='input.tmp')
    destination = source.model_copy(update={'path': 'input.json'})
    source_client = MagicMock()
    destination_client = MagicMock()

    def get_client(uri):
        return source_client if uri.path == source.path else destination_client

    with patch.object(impl_adls.adls_ctx, 'get_file_client',
                      side_effect=get_client):
        yield SimpleNamespace(
            source=source,
            destination=destination,
            source_client=source_client,
            destination_client=destination_client,
        )


def test_adls_multipart_completion_commits_numeric_order_then_promotes():
    uri, upload_id, state, blob, service = _multipart_fixture()
    payloads = {
        impl_adls._raw_block_id(state['nonce'], 1): b'first row\n',
        impl_adls._raw_block_id(state['nonce'], 2): b'\x00\xff',
        impl_adls._raw_block_id(state['nonce'], 3): b'last row\n',
    }
    published = {}

    def commit(blocks, **_kwargs):
        published['staging'] = b''.join(payloads[block.id] for block in blocks)

    def promote(*_args):
        published['final'] = published['staging']

    blob.commit_block_list.side_effect = commit
    with _multipart_clients(service, promotion=MagicMock(side_effect=promote)):
        impl_adls.ADLSPresignedHelper(uri).complete_multipart_upload(
            upload_id, [_Part(3), _Part(1), _Part(2)])

    committed = [block.id for block in blob.commit_block_list.call_args.args[0]]
    assert committed == [
        impl_adls._raw_block_id(state['nonce'], n) for n in (1, 2, 3)
    ]
    assert blob.commit_block_list.call_args.kwargs['metadata'] == {
        impl_adls._UPLOAD_METADATA_KEY: upload_id,
    }
    expected = b'first row\n\x00\xfflast row\n'
    assert published['final'] == expected
    assert hashlib.sha256(published['final']).hexdigest() == (
        '88a08c1cc768ca1fbda4b6b876653830fd7d25afcdfc92a796a3348f904f5deb')
    blob.delete_blob.assert_not_called()


def test_adls_multipart_metadata_failure_cleans_staging_blob():
    uri, upload_id, _state, blob, service = _multipart_fixture()
    blob.get_blob_properties.return_value = SimpleNamespace(
        metadata={impl_adls._UPLOAD_METADATA_KEY: 'another-upload'})
    with (
            _multipart_clients(service) as (_, promote),
            pytest.raises(ValueError, match='state failed validation'),
    ):
        impl_adls.ADLSPresignedHelper(uri).complete_multipart_upload(
            upload_id, [_Part(1), _Part(2), _Part(3)])

    blob.delete_blob.assert_called_once_with(delete_snapshots='include')
    blob.commit_block_list.assert_not_called()
    promote.assert_not_called()


def test_adls_multipart_rejects_extra_raw_block_and_cleans_staging():
    uri, upload_id, state, blob, service = _multipart_fixture()
    _committed, uncommitted = blob.get_block_list.return_value
    blob.get_block_list.return_value = (
        [],
        uncommitted + [SimpleNamespace(id=f"kumo:{state['nonce']}:99999999")],
    )
    with (
            _multipart_clients(service),
            pytest.raises(ValueError, match='do not match multipart metadata'),
    ):
        impl_adls.ADLSPresignedHelper(uri).complete_multipart_upload(
            upload_id, [_Part(1), _Part(2), _Part(3)])

    blob.commit_block_list.assert_not_called()
    blob.delete_blob.assert_called_once_with(delete_snapshots='include')


@pytest.mark.parametrize('tamper', ['part-count', 'nonce-path'])
def test_adls_multipart_completion_rejects_tampered_upload_state(tamper):
    uri, upload_id, state, blob, service = _multipart_fixture()
    if tamper == 'part-count':
        state['num_parts'] = 1
    else:
        state['nonce'] = '../forged/path'
        state['temp_path'] = (f"{uri.path}.kumo-upload-{state['nonce']}-"
                              f"{state['num_parts']}")
    tampered = base64.urlsafe_b64encode(
        json.dumps(state, separators=(',', ':'),
                   sort_keys=True).encode()).decode().rstrip('=')
    # Even forged staging metadata cannot make a changed count valid: the
    # count is bound into the SAS-signed blob path.
    blob.get_blob_properties.return_value = SimpleNamespace(
        metadata={
            impl_adls._UPLOAD_METADATA_KEY: tampered,
        })
    with (
            _multipart_clients(service),
            pytest.raises(ValueError, match='Invalid ADLS upload id'),
    ):
        impl_adls.ADLSPresignedHelper(uri).complete_multipart_upload(
            tampered, [_Part(1)])

    service.get_blob_client.assert_not_called()
    blob.get_blob_properties.assert_not_called()
    blob.get_block_list.assert_not_called()
    blob.commit_block_list.assert_not_called()


def test_adls_multipart_promotion_failure_retries_committed_staging(
        monkeypatch):
    uri, upload_id, state, blob, service = _multipart_fixture()
    block_ids = [
        impl_adls._raw_block_id(state['nonce'], number) for number in (1, 2, 3)
    ]
    blob.get_block_list.side_effect = [
        ([], [SimpleNamespace(id=value) for value in block_ids]),
        SimpleNamespace(
            uncommitted_blocks=[],
            committed_blocks=[
                SimpleNamespace(name=base64.b64encode(value.encode()).decode())
                for value in block_ids
            ],
        ),
    ]
    promote = MagicMock(side_effect=[
        impl_adls.ServiceRequestError('transient promotion failure'), None
    ])
    monkeypatch.setattr('io_utils.retry.time.sleep', MagicMock())

    with _multipart_clients(service, promotion=promote):
        impl_adls.ADLSPresignedHelper(uri).complete_multipart_upload(
            upload_id, [_Part(1), _Part(2), _Part(3)])

    blob.commit_block_list.assert_called_once()
    assert promote.call_count == 2
    blob.delete_blob.assert_not_called()


@pytest.mark.parametrize('numbers', [
    [1, 2],
    [1, 2, 2],
    [1, 2, 4],
])
def test_adls_multipart_completion_rejects_bad_part_metadata(numbers):
    uri, upload_id, _state, blob, service = _multipart_fixture()
    with _multipart_clients(service):
        with pytest.raises(ValueError, match='every part exactly once'):
            impl_adls.ADLSPresignedHelper(uri).complete_multipart_upload(
                upload_id, [_Part(number) for number in numbers])
    blob.commit_block_list.assert_not_called()
    blob.delete_blob.assert_called_once_with(delete_snapshots='include')


def test_adls_multipart_completion_is_idempotent_after_promotion():
    uri, upload_id, _state, blob, service = _multipart_fixture()
    final = MagicMock()
    final.get_file_properties.return_value = SimpleNamespace(
        metadata={
            impl_adls._UPLOAD_METADATA_KEY: upload_id,
        })
    with _multipart_clients(service, final) as (_, promote):
        impl_adls.ADLSPresignedHelper(uri).complete_multipart_upload(
            upload_id, [_Part(1), _Part(2), _Part(3)])
    blob.get_block_list.assert_not_called()
    blob.commit_block_list.assert_not_called()
    promote.assert_not_called()


def test_adls_atomic_promote_uses_source_and_destination_etags(
        promotion_clients):
    source_client = promotion_clients.source_client
    destination_client = promotion_clients.destination_client
    source_client.get_file_properties.return_value = SimpleNamespace(
        etag='source-etag')
    destination_client.get_file_properties.return_value = SimpleNamespace(
        etag='destination-etag', metadata={})

    impl_adls._atomic_promote(promotion_clients.source,
                              promotion_clients.destination, 'operation')

    source_client.rename_file.assert_called_once_with(
        'filesystem/input.json',
        source_etag='source-etag',
        source_match_condition=impl_adls.MatchConditions.IfNotModified,
        etag='destination-etag',
        match_condition=impl_adls.MatchConditions.IfNotModified,
    )


def test_adls_atomic_promote_uses_if_missing_for_new_destination(
        promotion_clients):
    source_client = promotion_clients.source_client
    destination_client = promotion_clients.destination_client
    source_client.get_file_properties.return_value = SimpleNamespace(
        etag='source-etag')
    destination_client.get_file_properties.side_effect = (
        impl_adls.ResourceNotFoundError('missing'))

    impl_adls._atomic_promote(promotion_clients.source,
                              promotion_clients.destination, 'operation')

    source_client.rename_file.assert_called_once_with(
        'filesystem/input.json',
        source_etag='source-etag',
        source_match_condition=impl_adls.MatchConditions.IfNotModified,
        match_condition=impl_adls.MatchConditions.IfMissing,
    )


def test_adls_atomic_promote_rejects_destination_conflict(promotion_clients):
    source_client = promotion_clients.source_client
    destination_client = promotion_clients.destination_client
    source_client.get_file_properties.return_value = SimpleNamespace(
        etag='source-etag')
    source_client.rename_file.side_effect = impl_adls.ResourceModifiedError(
        'changed')
    destination_client.get_file_properties.side_effect = [
        SimpleNamespace(etag='old-etag', metadata={}),
        SimpleNamespace(etag='new-etag', metadata={}),
    ]

    with pytest.raises(RuntimeError, match='destination changed'):
        impl_adls._atomic_promote(promotion_clients.source,
                                  promotion_clients.destination, 'operation')


def test_adls_atomic_promote_retry_accepts_own_completed_operation(
        promotion_clients):
    source_client = promotion_clients.source_client
    destination_client = promotion_clients.destination_client
    source_client.get_file_properties.side_effect = (
        impl_adls.ResourceNotFoundError('already moved'))
    destination_client.get_file_properties.return_value = SimpleNamespace(
        metadata={impl_adls._UPLOAD_METADATA_KEY: 'operation'})

    impl_adls._atomic_promote(promotion_clients.source,
                              promotion_clients.destination, 'operation')
    source_client.rename_file.assert_not_called()


def test_adls_atomic_write_failure_cleans_staging_metadata():
    uri = ADLSURI(storage_account='account', container='filesystem',
                  path='input.json')
    staging = MagicMock()
    operation = SimpleNamespace(hex='operation-id')

    with (
            patch.object(impl_adls.uuid, 'uuid4', return_value=operation),
            patch.object(impl_adls.adls_ctx, 'get_file_client',
                         return_value=staging),
            patch.object(impl_adls, '_atomic_promote',
                         side_effect=RuntimeError('promotion failed')),
            pytest.raises(RuntimeError, match='promotion failed'),
    ):
        impl_adls.atomic_write_text(uri, 'complete payload')

    staging.upload_data.assert_called_once_with(
        b'complete payload',
        overwrite=True,
        metadata={impl_adls._UPLOAD_METADATA_KEY: operation.hex},
        match_condition=impl_adls.MatchConditions.IfMissing,
    )
    staging.delete_file.assert_called_once_with()


def test_adls_atomic_empty_write_conditionally_creates_staging_file():
    uri = ADLSURI(storage_account='account', container='filesystem',
                  path='empty.json')
    staging = MagicMock()
    operation = SimpleNamespace(hex='empty-operation-id')

    with (
            patch.object(impl_adls.uuid, 'uuid4', return_value=operation),
            patch.object(impl_adls.adls_ctx, 'get_file_client',
                         return_value=staging),
            patch.object(impl_adls, '_atomic_promote') as promote,
    ):
        impl_adls.atomic_write_text(uri, '')

    staging.upload_data.assert_not_called()
    staging.create_file.assert_called_once_with(
        metadata={impl_adls._UPLOAD_METADATA_KEY: operation.hex},
        match_condition=impl_adls.MatchConditions.IfMissing,
    )
    promote.assert_called_once()
    staging.delete_file.assert_called_once_with()


def test_zip_directory_adls_members_and_idempotency(adls_client):
    base = 'abfss://test-container@testaccount.dfs.core.windows.net'
    adls_client.create_container('test-container')
    adls_client.create_file(f'{base}/outputs/part-0.csv', b'zero')
    adls_client.create_file(f'{base}/outputs/part-1.csv', b'one')

    result = fsutil.zip_directory(f'{base}/outputs')
    assert result == f'{base}/outputs.zip'
    with ZipFile(io.BytesIO(fsutil.read_binary(result))) as archive:
        assert archive.namelist() == [
            'outputs/part-0.csv',
            'outputs/part-1.csv',
        ]
        assert archive.read('outputs/part-0.csv') == b'zero'
        assert archive.read('outputs/part-1.csv') == b'one'
    assert adls_client._metadata['test-container/outputs.zip'].get(
        impl_adls._UPLOAD_METADATA_KEY)
    assert fsutil.zip_directory(f'{base}/outputs') == result


def test_zip_directory_adls_supports_lifecycle_archive_path(adls_client):
    base = 'abfss://test-container@testaccount.dfs.core.windows.net'
    adls_client.create_container('test-container')
    adls_client.create_file(f'{base}/outputs/part-0.csv', b'zero')
    adls_client.create_file(f'{base}/outputs/part-1.csv', b'one')
    archive = f'{base}/tmp_archives/job.zip'
    assert fsutil.zip_directory(f'{base}/outputs',
                                output_path=archive) == archive
    assert fsutil.exists(archive)
    assert not fsutil.exists(f'{base}/outputs.zip')


def test_zip_directory_adls_empty_fails_without_output(adls_client):
    base = 'abfss://test-container@testaccount.dfs.core.windows.net'
    adls_client.create_container('test-container')
    with pytest.raises(ValueError, match='No files found'):
        fsutil.zip_directory(f'{base}/empty')
    assert not fsutil.exists(f'{base}/empty.zip')


def test_zip_directory_adls_partial_read_cleans_temporary_output(
        adls_client, monkeypatch):
    base = 'abfss://test-container@testaccount.dfs.core.windows.net'
    adls_client.create_container('test-container')
    adls_client.create_file(f'{base}/outputs/part-0.csv', b'zero')
    adls_client.create_file(f'{base}/outputs/part-1.csv', b'one')
    original_get_file_client = impl_adls.adls_ctx.get_file_client

    def get_file_client(uri):
        client = original_get_file_client(uri)
        if uri.path == 'outputs/part-1.csv':
            client.download_file = MagicMock(
                side_effect=RuntimeError('read failed'))
        return client

    monkeypatch.setattr(impl_adls.adls_ctx, 'get_file_client', get_file_client)
    with pytest.raises(RuntimeError, match='read failed'):
        fsutil.zip_directory(f'{base}/outputs')
    assert not fsutil.exists(f'{base}/outputs.zip')
    assert all('.kumo-atomic-' not in key for key in adls_client._files)


def test_write_permission_provider_error_returns_false():
    with patch.object(threadsafe_adls, 'data_lake_client',
                      side_effect=ServiceRequestError("connection failed")):
        assert fsutil.has_write_permission(
            "abfss://random@fakeaccount.dfs.core.windows.net/test/path"
        ) is False


def test_adls_copydir(adls_client):
    """Test the copydir fast path triggered by overwrite=True + raise_on_exists=False."""
    container_name = 'test-container'
    storage_account = 'testaccount'
    base_uri = f'abfss://{container_name}@{storage_account}.dfs.core.windows.net'

    adls_client.create_container(container_name)
    adls_client.create_file(f'{base_uri}/src_dir/a.txt', 'hello')
    adls_client.create_file(f'{base_uri}/src_dir/sub/b.txt', 'world')

    # This is the exact call signature that backend_cache.py uses — it must
    # trigger the copydir fast path, not the generic file-by-file path.
    fsutil.copy(
        f'{base_uri}/src_dir',
        f'{base_uri}/dst_dir',
        overwrite=True,
        raise_on_exists=False,
    )

    assert fsutil.exists(f'{base_uri}/dst_dir/a.txt')
    assert fsutil.exists(f'{base_uri}/dst_dir/sub/b.txt')
    assert fsutil.read_binary(f'{base_uri}/dst_dir/a.txt') == b'hello'
    assert fsutil.read_binary(f'{base_uri}/dst_dir/sub/b.txt') == b'world'

    # Source should still exist (copy, not move)
    assert fsutil.exists(f'{base_uri}/src_dir/a.txt')
    assert fsutil.exists(f'{base_uri}/src_dir/sub/b.txt')


def test_object_identity_uses_etag(monkeypatch):
    from io_utils.fsutil import impl_adls

    properties = SimpleNamespace(
        size=321,
        etag='etag-value',
        version_id=None,
        metadata={},
    )
    client = MagicMock()
    client.get_file_properties.return_value = properties
    monkeypatch.setattr(impl_adls.adls_ctx, 'get_file_client',
                        lambda _uri: client)

    identity = fsutil.object_identity(
        'abfss://container@account.dfs.core.windows.net/path/file.parquet')

    assert identity == fsutil.ObjectIdentity(
        size=321,
        etag='etag-value',
        version=None,
    )


def test_listdir_container_root(adls_client):
    """listdir on a bare container URI must not raise ValueError."""
    container = 'test-container'
    storage_account = 'testaccount'
    root_uri = f'abfss://{container}@{storage_account}.dfs.core.windows.net'

    adls_client.create_container(container)
    adls_client.create_file(f'{root_uri}/dir_a/f1.txt', 'x')
    adls_client.create_file(f'{root_uri}/dir_b/f2.txt', 'y')

    entries = fsutil.listdir(root_uri)
    assert any('dir_a' in e for e in entries)
    assert any('dir_b' in e for e in entries)
