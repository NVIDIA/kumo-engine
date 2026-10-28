# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumo_core.sources.types import ReadMode
from kumoapi.data_source import DatabricksAuthenticationKind

from kumo.engine.connector.databricks import DatabricksConnector
from kumo.engine.connector.identity import DatabricksIdentity


@pytest.mark.parametrize(
    'staging_volume_path, require_staged, expected_mode',
    [
        (None, False, None),
        ('/Volumes/main/kumo/staging', False, None),
        ('/Volumes/main/kumo/staging', True, ReadMode.STAGED),
    ],
)
def test_read_configuration_reaches_the_cached_backend(staging_volume_path,
                                                       require_staged,
                                                       expected_mode):
    identity = DatabricksIdentity(host='https://workspace.example.com',
                                  client_id='test-client',
                                  client_secret='test-secret')
    connector = DatabricksConnector(
        catalog='main',
        schema_name='analytics',
        warehouse_id='warehouse-id',
        identity=identity,
        staging_volume_path=staging_volume_path,
        require_staged=require_staged,
    )

    backend = connector.source_backend

    assert connector.source_backend is backend
    assert backend._identity is identity
    assert backend.staging_volume_path == staging_volume_path
    assert backend.mode is expected_mode


@pytest.mark.parametrize('staging_volume_path', [None, ''])
def test_strict_staging_requires_a_volume(staging_volume_path):
    identity = DatabricksIdentity(host='https://workspace.example.com',
                                  client_id='test-client',
                                  client_secret='test-secret')
    with pytest.raises(ValueError,
                       match='require_staged needs a staging_volume_path'):
        DatabricksConnector(
            catalog='main',
            schema_name='analytics',
            warehouse_id='warehouse-id',
            identity=identity,
            staging_volume_path=staging_volume_path,
            require_staged=True,
        )


@pytest.mark.parametrize('build', [
    DatabricksConnector.from_notebook,
    DatabricksConnector,
])
def test_without_an_identity_the_connector_is_ambient(build):
    connector = build(
        catalog='main',
        schema_name='analytics',
        warehouse_id='warehouse-id',
    )

    assert connector.identity is not None
    identity_config = connector.identity._config
    assert identity_config.host is None
    assert identity_config.client_id is None
    assert (identity_config.authentication is
           DatabricksAuthenticationKind.AMBIENT)

    backend = connector.source_backend

    assert backend._identity is connector.identity
