# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import kumoapi.data_source as api
import pytest

from rest.public.routers.identity import IdentityRouter
from rest.routers.identities import IdentitiesResource
from rest.routers.identity_rotation import preserves_s3_ca_certificate
from statedb.config.common import IdentityID


@pytest.mark.parametrize(
    ('credentials', 'expected'),
    [
        ({
            'kind': 's3_access_key',
            'access_key_id': 'access',
            'secret_access_key': 'secret',
        }, True),
        ({
            'kind': 's3_access_key',
            'access_key_id': 'access',
            'secret_access_key': 'secret',
            'ca_certificate': None,
        }, False),
        ({
            'kind': 's3_access_key',
            'access_key_id': 'access',
            'secret_access_key': 'secret',
            'ca_certificate': 'replacement PEM',
        }, False),
    ],
)
@pytest.mark.asyncio
async def test_preserves_s3_ca_certificate_only_when_field_is_omitted(
        credentials: dict[str, object], expected: bool) -> None:
    request = MagicMock()
    request.json = AsyncMock(return_value={'credentials': credentials})

    assert await preserves_s3_ca_certificate(request) is expected


@pytest.mark.parametrize(
    ('resource_type', 'authorization_path'),
    [
        (IdentitiesResource,
         'rest.routers.identities.get_user_authorization_from_request'),
        (IdentityRouter,
         'rest.public.routers.identity.get_user_authorization_from_request'),
    ],
)
@pytest.mark.asyncio
async def test_identity_router_preserves_an_omitted_s3_ca_certificate(
        resource_type, authorization_path: str) -> None:
    service = MagicMock()
    response = MagicMock(spec=api.IdentityResponse)
    service.rotate_credentials = AsyncMock(return_value=response)
    services = MagicMock()
    services.identities = service
    resource = object.__new__(resource_type)
    resource._driver = MagicMock()
    resource._driver._require_state_services.return_value = services
    identity_id = IdentityID.generate()
    args = MagicMock(spec=api.RotateIdentityCredentialsArgs)
    principal = MagicMock()
    request = MagicMock()
    request.json = AsyncMock(return_value={
        'credentials': {
            'kind': 's3_access_key',
            'access_key_id': 'access',
            'secret_access_key': 'secret',
        }
    })

    with patch(authorization_path, return_value=principal):
        assert await resource.rotate_credentials(identity_id, args,
                                                 request) is response

    service.rotate_credentials.assert_awaited_once_with(
        identity_id,
        args,
        principal,
        preserve_s3_ca_certificate=True,
    )
