# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from rest.config.authentication import AuthenticationSettings
from rest.settings import AppSettings, KumoRestSettings
from service_framework.deploy_stage import DeployStage


def test_default_rest_settings_disable_auth_in_test_mode(monkeypatch):
    monkeypatch.setenv('KUMO_TESTING', 'True')
    monkeypatch.setenv('DEPLOY_STAGE', DeployStage.TEST.value)

    rest_settings = KumoRestSettings()

    assert rest_settings.authentication.disable is True
    assert rest_settings.authentication.is_unit_test is True
    assert (rest_settings.authentication.
            direct_connect_password_login_ip_rate_limit_enabled is True)
    assert not hasattr(rest_settings.authentication, 'credentials')


def test_direct_connect_auth_allows_any_product_port_with_explicit_cors():
    settings = KumoRestSettings(
        app=AppSettings(
            port=42001,
            middleware_origin='https://kumo.example.com',
        ), )

    settings.validate()


def test_direct_connect_auth_rejects_wildcard_cors_with_credentials():
    settings = KumoRestSettings(
        app=AppSettings(port=42002),
        authentication=AuthenticationSettings(disable=False),
    )

    with pytest.raises(ValueError, match='REST_CORS_ALLOWED_ORIGINS'):
        settings.validate()
