# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rest.public.app import AppBuilder
from rest.public.routers.authentication import AuthenticationRouter


def test_authentication_probe_is_group_neutral_and_requires_auth() -> None:
    calls = []

    async def authenticate_request() -> None:
        calls.append(True)

    app = FastAPI()
    app.include_router(
        AuthenticationRouter(authenticate_request).router,
        prefix='/v1',
    )

    response = TestClient(app).get('/v1/authentication')

    assert response.status_code == 204
    assert response.content == b''
    assert calls == [True]


@pytest.mark.asyncio
async def test_validate_api_key_logs_no_credential_substring(caplog) -> None:
    api_key = 'customer:userhash:key-name:known-secret-suffix'

    with caplog.at_level(logging.INFO):
        await AppBuilder.validate_api_key(SimpleNamespace(), api_key)

    assert 'Validated API key' in caplog.text
    for component in api_key.split(':'):
        assert component not in caplog.text
