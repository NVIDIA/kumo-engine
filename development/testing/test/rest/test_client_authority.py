# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
from http import HTTPStatus

from fastapi.testclient import TestClient


def test_disabled_auth_request_has_customer_admin_authority(
        client: TestClient) -> None:
    admin_api_key = client.headers['X-API-Key']

    # Fixture authority is not application data and must not pollute API-key
    # repository tests or start StateDB services on a foreign event loop.
    api_keys = client.get('/v0/api-keys/admin', params={'type': 'admin'})
    assert api_keys.status_code == HTTPStatus.OK
    assert api_keys.json() == []

    # Auth is disabled in tests. Dropping the fixture credential leaves a
    # headerless request, which resolves to the anonymous customer-admin
    # principal -- there is no identity boundary to enforce -- so it must still
    # be able to perform admin-gated mutations such as creating a group.
    del client.headers['X-API-Key']
    try:
        headerless_response = client.post('/v0/groups', json={
            'display_name': f'disabled-admin-{uuid.uuid4().hex}',
        })
    finally:
        client.headers['X-API-Key'] = admin_api_key

    assert headerless_response.status_code == HTTPStatus.CREATED

    admin_response = client.post('/v0/groups', json={
        'display_name': f'admin-{uuid.uuid4().hex}',
    })
    assert admin_response.status_code == HTTPStatus.CREATED
