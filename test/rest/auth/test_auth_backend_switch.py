# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct Connect auth contract checks."""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rest.config.authentication import AuthenticationSettings
from rest.config.direct_connect import resolve_auth_settings_config
from rest.deps.authentication import Authentication
from rest.routers.authentication import AuthenticationResource
from service_framework.deploy_env_config import _reset_deploy_env


@pytest.fixture(autouse=True, scope="session")
def setup_rest_test_session():
    yield


@pytest.fixture(autouse=True, scope="function")
def auto_cleanup_per_func():
    _reset_deploy_env()
    yield
    _reset_deploy_env()


def test_default_auth_settings_are_direct_connect_only() -> None:
    settings = AuthenticationSettings(disable=True)

    assert not hasattr(settings, "auth_backend")
    assert settings.direct_connect_login_method == "oidc"


@pytest.mark.parametrize("backend", ["auth0", "keycloak", "okta"])
def test_env_backend_selector_rejects_non_direct_connect(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    monkeypatch.setenv("KUMO_AUTH_BACKEND", backend)

    with pytest.raises(ValueError, match="Direct Connect"):
        resolve_auth_settings_config()


@pytest.mark.parametrize("backend", ["auth0", "keycloak"])
def test_deploy_config_backend_rejects_non_direct_connect(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    monkeypatch.setenv(
        "DEPLOY_ENV_CONFIG_JSON",
        json.dumps({
            "auth": {
                "enabled": True,
                "backend": backend,
            },
            "logging": {
                "backend": "stdout",
            },
        }),
    )

    with pytest.raises(ValueError, match="Direct Connect"):
        resolve_auth_settings_config()


def test_client_config_exposes_direct_connect_settings_only() -> None:
    auth = Authentication(
        AuthenticationSettings(
            disable=False,
            direct_connect_login_method="password",
        ),
        is_unit_test=True,
    )
    app = FastAPI()
    app.include_router(AuthenticationResource(auth).router)

    with TestClient(app) as client:
        response = client.get("/v0/authentication/client-config")

    assert response.status_code == 200
    assert response.json() == {
        "auth_disabled": False,
        "direct_connect_login_method": "password",
    }
