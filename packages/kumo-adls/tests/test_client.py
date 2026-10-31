# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from kumo_adls import client as clients
from kumo_adls.clouds import CLOUDS
from kumo_adls.source_access import source_access
from kumo_adls.transfer import NETWORK_BLOCK_BYTES


class Credential:
    def __init__(self):
        self.closed = False

    def get_token(self, *scopes, **kwargs):
        raise AssertionError('Construction must not acquire a token')

    def close(self):
        self.closed = True


@pytest.mark.parametrize('cloud', list(CLOUDS))
def test_factory_borrows_credential_and_owns_transport(monkeypatch, cloud):
    suffix, _, _ = CLOUDS[cloud]
    credential = Credential()
    owned_session = clients.Session()
    closed = Mock(wraps=owned_session.close)
    monkeypatch.setattr(owned_session, 'close', closed)
    monkeypatch.setattr(clients, 'Session', lambda: owned_session)
    client = clients.create_client(credential, 'companylake', suffix)
    assert clients.validate_client(client, 'companylake',
                                   suffix) == ('companylake', suffix)
    assert client._config.max_single_get_size == 16 << 20
    assert client._config.max_chunk_get_size == 16 << 20
    assert client._pipeline._transport.connection_config.data_block_size == NETWORK_BLOCK_BYTES
    client.close()
    closed.assert_called_once()
    assert not credential.closed
    credential.close()


@pytest.mark.parametrize('stage', ['adapter', 'mount', 'transport', 'client'])
def test_factory_failure_closes_owned_resources_not_credential(
        monkeypatch, stage):
    credential = Credential()
    owned_session = Mock()
    monkeypatch.setattr(clients, 'Session', lambda: owned_session)
    failure = Mock(side_effect=RuntimeError('construction failed'))
    if stage == 'adapter':
        monkeypatch.setattr(clients, 'HTTPAdapter', failure)
    elif stage == 'mount':
        owned_session.mount.side_effect = RuntimeError('construction failed')
    elif stage == 'transport':
        monkeypatch.setattr(clients, 'RequestsTransport', failure)
    else:
        monkeypatch.setattr(clients, 'DataLakeServiceClient', failure)
    with pytest.raises(RuntimeError, match='construction failed'):
        clients.create_client(credential, 'companylake')
    owned_session.close.assert_called_once()
    assert not credential.closed


def test_factory_requires_credential_before_creating_transport(monkeypatch):
    factory = Mock(side_effect=AssertionError('must not create transport'))
    monkeypatch.setattr(clients, 'Session', factory)
    with pytest.raises(TypeError, match='resolved credential'):
        clients.create_client(None, 'companylake')
    factory.assert_not_called()


@pytest.mark.parametrize('url', [
    'https://otherlake.dfs.core.windows.net/',
    'https://companylake.dfs.core.usgovcloudapi.net/',
    'https://companylake.dfs.core.windows.net.evil.example/',
    'https://companylake.blob.core.windows.net/',
    'http://companylake.dfs.core.windows.net/',
    'https://private-user:private-password@companylake.dfs.core.windows.net/',
    'https://companylake.dfs.core.windows.net:443/',
    'https://companylake.dfs.core.windows.net/filesystem/',
    'https://companylake.dfs.core.windows.net/#private-fragment',
    'https://[invalid/?sig=private-sas',
    'https://companylake.dfs.core.windows.net/\n',
    None,
])
def test_borrowed_endpoint_rejected_without_io_or_secret_echo(url):
    borrowed = Mock(account_name='companylake', url=url)
    with pytest.raises(ValueError, match='requested account') as error:
        source_access(borrowed, 'companylake')
    borrowed.get_file_system_client.assert_not_called()
    borrowed.close.assert_not_called()
    for value in ('private-user', 'private-password', 'private-fragment',
                  'private-sas'):
        assert value not in str(error.value)


def test_borrowed_client_account_name_must_match_endpoint():
    borrowed = Mock(account_name='otherlake',
                    url='https://companylake.dfs.core.windows.net/')
    with pytest.raises(ValueError, match='requested account'):
        source_access(borrowed, 'companylake')
    borrowed.get_file_system_client.assert_not_called()
    borrowed.close.assert_not_called()


def test_sas_client_url_is_supported_without_identity_resolution():
    from azure.storage.filedatalake import DataLakeServiceClient

    with DataLakeServiceClient(
            'https://companylake.dfs.core.windows.net/?sv=test&sig=synthetic',
            credential=None) as borrowed:
        access = source_access(borrowed, 'companylake')
        assert access.filesystem.handler.fs.client is borrowed
