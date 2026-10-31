# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest
from kumo_adls import native_output
from kumo_adls.session import CLOUDS
from kumoapi.data_source import ADLSIdentityConfig, AzureCloud
from pyarrow.fs import AzureFileSystem, FileSystem

TENANT = 'private-tenant-value'
CLIENT = 'private-client-value'
ACCOUNT = 'companylake'
CONTAINER = 'curated'


@pytest.fixture
def auth_environment(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith('AZURE_'):
            monkeypatch.delenv(name)
    token = tmp_path / 'private-token-file'
    token.write_text('private-token-contents')
    monkeypatch.setenv('AZURE_TENANT_ID', TENANT)
    monkeypatch.setenv('AZURE_CLIENT_ID', CLIENT)
    monkeypatch.setenv('AZURE_FEDERATED_TOKEN_FILE', str(token))
    return token


def principal(cloud=AzureCloud.PUBLIC):
    return SimpleNamespace(config=ADLSIdentityConfig(
        tenant_id=TENANT, client_id=CLIENT, cloud=cloud))


def assert_native_mapping(filesystem, suffix):
    assert isinstance(filesystem, AzureFileSystem)
    _, (options, ) = filesystem.__reduce__()
    assert options['account_name'] == ACCOUNT
    assert options['dfs_storage_authority'] == '.' + suffix
    assert options['blob_storage_authority'] == '.' + suffix.replace(
        'dfs.', 'blob.', 1)
    assert options['dfs_storage_scheme'] == 'https'
    assert options['blob_storage_scheme'] == 'https'


@pytest.mark.parametrize('cloud', list(CLOUDS))
def test_workload_native_construction_pins_identity_and_cloud(
        auth_environment, monkeypatch, cloud):
    suffix, _, authority = CLOUDS[cloud]
    monkeypatch.setenv('AZURE_AUTHORITY_HOST', 'https://' + authority)
    calls = []

    def from_uri(uri):
        calls.append(uri)
        parsed = urlsplit(uri)
        assert parsed.scheme == 'abfss'
        assert parsed.netloc == f'{ACCOUNT}@{CONTAINER}'
        assert parsed.path == '/'
        assert parse_qs(parsed.query) == {
            'credential_kind': ['workload_identity'],
            'dfs_storage_authority': ['.' + suffix],
            'blob_storage_authority':
            ['.' + suffix.replace('dfs.', 'blob.', 1)]
        }
        assert TENANT not in uri
        assert CLIENT not in uri
        assert str(auth_environment) not in uri
        return FileSystem.from_uri(uri)

    monkeypatch.setattr(native_output, 'FileSystem',
                        SimpleNamespace(from_uri=from_uri))
    before = dict(os.environ)
    filesystem = native_output.filesystem_for(principal(cloud), ACCOUNT, suffix,
                                              CONTAINER)
    assert_native_mapping(filesystem, suffix)
    assert len(calls) == 1
    assert dict(os.environ) == before
    # Arrow does not serialize credential_kind in __reduce__; the URI spy
    # above checks the selected credential rather than claiming round-trip proof.


@pytest.mark.parametrize('cloud', list(CLOUDS))
def test_ambient_native_construction_uses_selected_cloud_without_env_mutation(
        auth_environment, monkeypatch, cloud):
    suffix, _, authority = CLOUDS[cloud]
    monkeypatch.setenv('AZURE_AUTHORITY_HOST', 'https://' + authority + '/')
    before = dict(os.environ)
    filesystem = native_output.filesystem_for(None, ACCOUNT, suffix, CONTAINER)
    assert_native_mapping(filesystem, suffix)
    assert dict(os.environ) == before


def forbid_construction(monkeypatch):
    factory = Mock(
        side_effect=AssertionError('must reject before construction'))
    monkeypatch.setattr(native_output, 'AzureFileSystem', factory)
    monkeypatch.setattr(native_output, 'FileSystem',
                        SimpleNamespace(from_uri=factory))
    return factory


@pytest.mark.parametrize('name,value', [
    ('AZURE_TENANT_ID', None),
    ('AZURE_TENANT_ID', 'another-private-tenant'),
    ('AZURE_CLIENT_ID', None),
    ('AZURE_CLIENT_ID', 'another-private-client'),
    ('AZURE_FEDERATED_TOKEN_FILE', None),
    ('AZURE_FEDERATED_TOKEN_FILE', ''),
    ('AZURE_FEDERATED_TOKEN_FILE', '  '),
])
def test_workload_mismatch_never_falls_back_or_discloses_identity(
        auth_environment, monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    factory = forbid_construction(monkeypatch)
    before = dict(os.environ)
    with pytest.raises(NotImplementedError) as error:
        native_output.filesystem_for(principal(), ACCOUNT,
                                     CLOUDS[AzureCloud.PUBLIC][0], CONTAINER)
    factory.assert_not_called()
    assert dict(os.environ) == before
    for secret in [
            TENANT, CLIENT,
            str(auth_environment), 'private-token-contents',
            'another-private-tenant', 'another-private-client'
    ]:
        assert secret not in str(error.value)


@pytest.mark.parametrize('selected', list(CLOUDS))
def test_explicit_identity_cloud_mismatch_fails_before_construction(
        auth_environment, monkeypatch, selected):
    other = next(cloud for cloud in CLOUDS if cloud != selected)
    factory = forbid_construction(monkeypatch)
    before = dict(os.environ)
    with pytest.raises(ValueError, match='identity cloud'):
        native_output.filesystem_for(principal(selected), ACCOUNT,
                                     CLOUDS[other][0], CONTAINER)
    factory.assert_not_called()
    assert dict(os.environ) == before


@pytest.mark.parametrize('selected', [None, AzureCloud.PUBLIC])
@pytest.mark.parametrize('authority', [
    'http://login.microsoftonline.com',
    'login.microsoftonline.com',
    'https://private-user:private-password@login.microsoftonline.com',
    'https://login.microsoftonline.com:443',
    'https://login.microsoftonline.com/private-path',
    'https://login.microsoftonline.com?token=private-query',
    'https://login.microsoftonline.com#private-fragment',
    'https://login.microsoftonline.us',
    'https://login.microsoftonline.com.evil.example',
    '',
])
def test_authority_rejected_before_any_factory_without_secret_echo(
        auth_environment, monkeypatch, selected, authority):
    monkeypatch.setenv('AZURE_AUTHORITY_HOST', authority)
    factory = forbid_construction(monkeypatch)
    before = dict(os.environ)
    with pytest.raises(ValueError) as error:
        native_output.filesystem_for(None if selected is None else principal(),
                                     ACCOUNT, CLOUDS[AzureCloud.PUBLIC][0],
                                     CONTAINER)
    factory.assert_not_called()
    assert dict(os.environ) == before
    for secret in [
            'private-user', 'private-password', 'private-path', 'private-query',
            'private-fragment', TENANT, CLIENT,
            str(auth_environment)
    ]:
        assert secret not in str(error.value)


@pytest.mark.parametrize('cloud', [AzureCloud.US_GOVERNMENT, AzureCloud.CHINA])
def test_sovereign_cloud_rejects_unset_public_default_authority(
        auth_environment, monkeypatch, cloud):
    factory = forbid_construction(monkeypatch)
    with pytest.raises(ValueError, match='AZURE_AUTHORITY_HOST'):
        native_output.filesystem_for(None, ACCOUNT, CLOUDS[cloud][0], CONTAINER)
    factory.assert_not_called()


def test_default_public_authority_is_accepted(auth_environment):
    filesystem = native_output.filesystem_for(principal(), ACCOUNT,
                                              CLOUDS[AzureCloud.PUBLIC][0],
                                              CONTAINER)
    assert_native_mapping(filesystem, CLOUDS[AzureCloud.PUBLIC][0])


def test_changed_environment_is_revalidated_without_cached_filesystem(
        auth_environment, monkeypatch, tmp_path):
    factory = Mock(wraps=FileSystem.from_uri)
    monkeypatch.setattr(native_output, 'FileSystem',
                        SimpleNamespace(from_uri=factory))
    selected = principal()
    suffix = CLOUDS[AzureCloud.PUBLIC][0]
    first = native_output.filesystem_for(selected, ACCOUNT, suffix, CONTAINER)
    monkeypatch.setenv('AZURE_CLIENT_ID', 'changed-private-client')
    before = dict(os.environ)
    with pytest.raises(NotImplementedError):
        native_output.filesystem_for(selected, ACCOUNT, suffix, CONTAINER)
    assert dict(os.environ) == before
    assert factory.call_count == 1
    monkeypatch.setenv('AZURE_CLIENT_ID', CLIENT)
    next_token = tmp_path / 'another-private-token'
    next_token.write_text('another-private-token-contents')
    monkeypatch.setenv('AZURE_FEDERATED_TOKEN_FILE', str(next_token))
    second = native_output.filesystem_for(selected, ACCOUNT, suffix, CONTAINER)
    assert second is not first
    assert factory.call_count == 2


def test_other_identity_type_fails_before_native_construction(
        auth_environment, monkeypatch):
    factory = forbid_construction(monkeypatch)
    before = dict(os.environ)
    with pytest.raises(ValueError, match='not an ADLS identity'):
        native_output.filesystem_for(SimpleNamespace(config=object()), ACCOUNT,
                                     CLOUDS[AzureCloud.PUBLIC][0], CONTAINER)
    factory.assert_not_called()
    assert dict(os.environ) == before
