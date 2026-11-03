# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kumo_adls import native_output
from kumo_adls.session import CLOUDS
from kumo_adls.sink import ADLSSinkBackend
from kumoapi.data_source import ADLSIdentityConfig
from kumoapi.jobs import WriteMode


@pytest.mark.parametrize('cloud', list(CLOUDS))
@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('full_uri', [False, True])
def test_supported_preflight_constructs_no_clients(monkeypatch, connector,
                                                   cloud, explicit, full_uri):
    suffix, _, authority = CLOUDS[cloud]
    monkeypatch.setenv('AZURE_AUTHORITY_HOST', 'https://' + authority)
    monkeypatch.setenv('AZURE_TENANT_ID', 'tenant')
    monkeypatch.setenv('AZURE_CLIENT_ID', 'client')
    monkeypatch.setenv('AZURE_FEDERATED_TOKEN_FILE', '/projected/token')
    principal = SimpleNamespace(
        config=ADLSIdentityConfig(tenant_id='tenant', client_id='client',
                                  cloud=cloud)) if explicit else None
    backend = ADLSSinkBackend()
    monkeypatch.setattr(backend, '_principal', lambda _: principal)
    forbidden = Mock(
        side_effect=AssertionError('preflight constructed a client'))
    monkeypatch.setattr('kumo_adls.backend.source_access', forbidden)
    monkeypatch.setattr(native_output, 'AzureFileSystem', forbidden)
    monkeypatch.setattr(native_output, 'FileSystem',
                        SimpleNamespace(from_uri=forbidden))
    target = 'predictions/'
    if full_uri:
        target = f'abfss://curated@companylake.{suffix}/orders/{target}'
    before = dict(os.environ)

    backend.preflight_write_files(connector(suffix=suffix), target,
                                  write_mode=WriteMode.OVERWRITE)

    forbidden.assert_not_called()
    assert dict(os.environ) == before


def test_default_write_policy_checks_occupancy_but_overwrite_preserves_behavior(
        service, connector):
    from kumo_core.artifact_export import ArtifactDestinationConflict

    backend = ADLSSinkBackend()
    target = connector()
    backend.preflight_write_files(target, 'out/')
    service.files['curated/orders/out/part.parquet'] = b'data'
    service.directories.add('curated/orders/out')
    with pytest.raises(ArtifactDestinationConflict, match='already contains'):
        backend.preflight_write_files(target, 'out/')
    backend.preflight_write_files(target, 'out/',
                                  write_mode=WriteMode.OVERWRITE)
    with pytest.raises(NotImplementedError, match='OVERWRITE'):
        backend.preflight_write_files(target, 'out/',
                                      write_mode=WriteMode.APPEND)
