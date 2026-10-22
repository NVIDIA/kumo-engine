# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import ANY, Mock

import pytest
from azure.identity import AzureAuthorityHosts
from kumo_adls import session
from kumo_core.sources.principals import Principal
from kumoapi.data_source import ADLSIdentityConfig, AzureCloud


@pytest.fixture(autouse=True)
def clean_sessions():
    session.close()
    yield
    session.close()


CLOUD_CASES = [
    (AzureCloud.PUBLIC, 'dfs.core.windows.net',
     AzureAuthorityHosts.AZURE_PUBLIC_CLOUD),
    (AzureCloud.US_GOVERNMENT, 'dfs.core.usgovcloudapi.net',
     AzureAuthorityHosts.AZURE_GOVERNMENT),
    (AzureCloud.CHINA, 'dfs.core.chinacloudapi.cn',
     AzureAuthorityHosts.AZURE_CHINA),
]


@pytest.mark.parametrize('cloud, suffix, authority', CLOUD_CASES)
def test_explicit_workload_identity_and_cloud_match(monkeypatch, cloud, suffix,
                                                    authority):
    principal = Principal(
        'test',
        ADLSIdentityConfig(tenant_id='tenant', client_id='client', cloud=cloud),
        None)
    factory = Mock()
    monkeypatch.setattr(session, 'WorkloadIdentityCredential', factory)
    credential, scope = session.credential_for(principal, suffix)
    assert credential is factory.return_value
    assert scope == 'https://storage.azure.com'
    factory.assert_called_once_with(tenant_id='tenant', client_id='client',
                                    authority=authority)
    other_suffix = next(value[0] for key, value in session.CLOUDS.items()
                        if key != cloud)
    with pytest.raises(ValueError, match='cloud'):
        session.credential_for(principal, other_suffix)


def test_ambient_identity_session_reuse_and_cleanup(monkeypatch):
    credential = Mock()
    credential_factory = Mock(return_value=credential)
    factory = Mock()
    monkeypatch.setattr(session, 'DefaultAzureCredential', credential_factory)
    monkeypatch.setattr(session, 'DataLakeServiceClient', factory)
    first = session.session_for(None, 'companylake', 'dfs.core.windows.net')
    assert session.session_for(None, 'companylake',
                               'dfs.core.windows.net') is first
    credential_factory.assert_called_once()
    factory.assert_called_once_with(
        account_url='https://companylake.dfs.core.windows.net',
        credential=credential, audience='https://storage.azure.com',
        max_single_get_size=16 << 20, max_chunk_get_size=16 << 20,
        connection_data_block_size=16 << 20, transport=ANY)
    session.close()
    factory.return_value.close.assert_called_once()
    credential.close.assert_called_once()


def test_close_one_identity_preserves_other_sessions(monkeypatch):
    factory = Mock(side_effect=lambda **kwargs: Mock())
    credentials = Mock(side_effect=lambda **kwargs: Mock())
    monkeypatch.setattr(session, 'WorkloadIdentityCredential', credentials)
    monkeypatch.setattr(session, 'DataLakeServiceClient', factory)
    config = ADLSIdentityConfig(tenant_id='tenant', client_id='client',
                                cloud=AzureCloud.PUBLIC)
    first = Principal('first', config, None)
    second = Principal('second', config, None)
    client = session.session_for(first, 'companylake', 'dfs.core.windows.net')
    other = session.session_for(second, 'companylake', 'dfs.core.windows.net')
    session.close(identity_name=first.name)
    client.close.assert_called_once()
    other.close.assert_not_called()


@pytest.mark.parametrize('cloud, suffix, authority', CLOUD_CASES)
def test_probe_acquires_token_without_touching_storage(monkeypatch, cloud,
                                                       suffix, authority):
    credential = Mock()
    credential.__enter__ = Mock(return_value=credential)
    credential.__exit__ = Mock()
    monkeypatch.setattr(session, 'WorkloadIdentityCredential',
                        Mock(return_value=credential))
    principal = Principal(
        'test',
        ADLSIdentityConfig(tenant_id='tenant', client_id='client', cloud=cloud),
        None)
    session.probe(principal)
    credential.get_token.assert_called_once_with(
        'https://storage.azure.com/.default')


@pytest.mark.parametrize('cloud, suffix, authority', CLOUD_CASES)
def test_ambient_session_uses_common_storage_audience(monkeypatch, cloud,
                                                      suffix, authority):
    credential = Mock()
    credential_factory = Mock(return_value=credential)
    client_factory = Mock()
    monkeypatch.setattr(session, 'DefaultAzureCredential', credential_factory)
    monkeypatch.setattr(session, 'DataLakeServiceClient', client_factory)

    session.session_for(None, 'companylake', suffix)

    credential_factory.assert_called_once_with(authority=authority)
    client_factory.assert_called_once_with(
        account_url=f'https://companylake.{suffix}', credential=credential,
        audience='https://storage.azure.com', max_single_get_size=16 << 20,
        max_chunk_get_size=16 << 20, connection_data_block_size=16 << 20,
        transport=ANY)


@pytest.mark.parametrize('failing_resource', ['client', 'credential'])
def test_cleanup_failure_does_not_leave_other_sessions_open(
        monkeypatch, failing_resource):
    clients = [Mock(), Mock()]
    credentials = [Mock(), Mock()]
    factory = Mock(side_effect=clients)
    credential_factory = Mock(side_effect=credentials)
    monkeypatch.setattr(session, 'DefaultAzureCredential', credential_factory)
    monkeypatch.setattr(session, 'DataLakeServiceClient', factory)
    session.session_for(None, 'firstlake', 'dfs.core.windows.net')
    session.session_for(None, 'otherlake', 'dfs.core.windows.net')
    resource = clients[0] if failing_resource == 'client' else credentials[0]
    resource.close.side_effect = RuntimeError('already closed')

    session.close()

    for item in clients + credentials:
        item.close.assert_called_once()
    assert not session._SESSIONS
