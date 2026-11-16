# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from rest.deps import authentication as authentication_deps
from rest.driver import api_key_driver
from rest.driver.api_key_driver import APIKeyDriver, APIKeyVerificationError
from rest.routers import authentication as auth_router


@pytest.fixture(autouse=True)
def clear_api_key_pepper_cache():
    api_key_driver._clear_api_key_pepper_cache_for_test()
    yield
    api_key_driver._clear_api_key_pepper_cache_for_test()


def _request(headers: dict[str, str] | None = None) -> Request:
    scope = {
        "type":
        "http",
        "method":
        "GET",
        "path":
        "/",
        "headers": [(key.lower().encode(), value.encode())
                    for key, value in (headers or {}).items()],
        "query_string":
        b"",
    }
    return Request(scope)


def _driver(client: TestClient) -> APIKeyDriver:
    return APIKeyDriver(state_services=client.app_builder._state_services)


@pytest.mark.asyncio
async def test_create_and_verify_owner_bound_key(client: TestClient) -> None:
    driver = _driver(client)
    created = await driver.create_admin_api_key("admin-key", "admin",
                                                "direct_connect|owner",
                                                user_name="Owner")

    identity = await driver.verify_api_key(created.value)
    assert identity is not None
    assert identity.user_id == "direct_connect|owner"
    assert identity.is_admin
    assert len(created.value.split(":")) == 3


@pytest.mark.asyncio
async def test_wrong_secret_is_rejected(client: TestClient) -> None:
    driver = _driver(client)
    created = await driver.create_user_api_key("user-key", "external|owner")
    user_hash, name, _secret = created.value.split(":")
    with pytest.raises(APIKeyVerificationError):
        await driver.verify_api_key(f"{user_hash}:{name}:wrong")


@pytest.mark.asyncio
async def test_resolved_key_never_confers_employee_authority(
    client: TestClient, ) -> None:
    driver = _driver(client)
    created = await driver.create_admin_api_key("admin-key-2", "admin",
                                                "external|owner")
    principal = await authentication_deps._resolve_user_authorization_from_valid_api_key(
        created.value, api_key_driver=driver)
    assert principal.is_admin


@pytest.mark.asyncio
async def test_api_key_dependency_requires_key_when_enabled() -> None:
    dependency = auth_router.api_key_auth(auth_disabled=False)
    with pytest.raises(HTTPException) as error:
        await dependency(request=_request(), api_key=None)
    assert error.value.status_code == HTTPStatus.UNAUTHORIZED


@pytest.mark.asyncio
async def test_api_key_dependency_reuses_middleware_authorization() -> None:
    request = _request({"X-API-Key": "deadbeef:named:secret"})
    request.state.user_authorization = SimpleNamespace(error=None)
    dependency = auth_router.api_key_auth(auth_disabled=False)
    await dependency(request=request, api_key="deadbeef:named:secret")


def test_non_prod_pepper_uses_local_constant(monkeypatch) -> None:
    monkeypatch.setenv("DEPLOY_STAGE", "test")
    monkeypatch.delenv("API_KEY_PEPPER", raising=False)
    assert api_key_driver._get_api_key_pepper() == (
        api_key_driver._DEFAULT_DEVTEST_API_KEY_PEPPER)


def test_non_prod_pepper_preserves_explicit_override(monkeypatch) -> None:
    monkeypatch.setenv("DEPLOY_STAGE", "preprod")
    monkeypatch.setenv("API_KEY_PEPPER", "existing-pepper")
    assert api_key_driver._get_api_key_pepper() == "existing-pepper"


def test_prod_pepper_uses_fixed_secret_name(monkeypatch) -> None:
    calls = []
    monkeypatch.setenv("DEPLOY_STAGE", "prod")
    monkeypatch.delenv("API_KEY_PEPPER_SECRET_NAME", raising=False)
    monkeypatch.setattr(
        api_key_driver,
        "get_secret_prefer_firstboot",
        lambda name: calls.append(name) or {"api_key_pepper": "pepper"},
    )
    assert api_key_driver._get_api_key_pepper() == "pepper"
    assert calls == ["kumo-secret-key"]
