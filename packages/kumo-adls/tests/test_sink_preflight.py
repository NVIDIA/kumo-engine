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

    backend.preflight_write_files(connector(suffix=suffix), target)

    forbidden.assert_not_called()
    assert dict(os.environ) == before
