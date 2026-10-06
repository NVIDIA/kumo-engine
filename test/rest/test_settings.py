# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from types import SimpleNamespace

import pytest
from pytest import MonkeyPatch

from io_utils.json import from_json, to_json
from rest import settings
from rest.settings import DataScaleLimits
from service_framework.deploy_env_config import _reset_deploy_env
from service_framework.deploy_stage import DeployStage


@pytest.fixture(scope='session')
def setup_rest_test_session() -> None:
    return None


def test_dev_local_db_storage_uses_configured_control_plane_root(
        monkeypatch: MonkeyPatch, tmp_path) -> None:
    control_root = tmp_path / 'configured-kumo-home'
    monkeypatch.setattr(
        settings,
        'deploy_env',
        lambda: SimpleNamespace(
            deploy_mode=DeployStage.DEV_LOCAL,
            control_plane_storage_root=str(control_root),
        ),
    )

    assert settings.get_db_storage_root() == str(control_root / 'db')
    assert settings.get_db_storage_root() != './db'


@pytest.fixture(autouse=True)
def _reset_singleton(
        monkeypatch: MonkeyPatch):  # pyright: ignore[reportUnusedFunction]
    """Reset the v2 deploy_env singleton before and after each test so it
    picks up the env vars set by monkeypatch."""
    # REST activity workers pin their shared test metadata root process-wide.
    # Settings parser tests model standalone deployments and must not inherit
    # that test-harness-only override; monkeypatch restores it after each test.
    monkeypatch.delenv('DB_STORAGE_ROOT', raising=False)
    monkeypatch.delenv('DISABLE_AUTH', raising=False)
    _reset_deploy_env()
    yield
    _reset_deploy_env()


# This test uses a "record-or-replay" approach. After making changes to
# rest/settings.py, this test would fail if you don't re-generate the test
# cases. To do that, run this test in "TEST_RECORD_MODE":
# `TEST_RECORD_MODE=1 pytest test/rest/test_settings.py`
# This will generate updated JSON setting files under test/rest/test_data/
# folder. Please review the changed JSON files, make sure the changes are
# expected, then add them to the PR.
#
# Note that the JSON files under test/rest/test_data/ are just test cases.
# Changes in the schema (KumoRestSettings) does not have to be backward
# compatible.
def _record_or_verify(rest_settings: settings.KumoRestSettings, file_name: str):
    record_mode = os.environ.get('TEST_RECORD_MODE')
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data')
    json_file_path = os.path.join(dir, file_name)
    if record_mode:
        with open(json_file_path, 'w') as out:
            out.write(to_json(rest_settings, insecure=True))
    else:
        with open(json_file_path, 'r') as f:
            data = f.read()
            # Strip deploy_config from snapshot — it was a v1 field,
            # now a @property on KumoRestSettings (not deserializable).
            import json as _json
            snapshot = _json.loads(data)
            snapshot.pop('deploy_config', None)
            data = _json.dumps(snapshot)
            json_rest_settings = from_json(data, settings.KumoRestSettings)

            assert json_rest_settings == rest_settings


def test_prod_settings_prpr(monkeypatch: MonkeyPatch):
    monkeypatch.setenv('DEPLOY_STAGE', DeployStage.PROD.value)
    monkeypatch.setenv('REST_API_PORT', '10001')
    monkeypatch.delenv('TEMPORAL_SERVER', raising=False)
    monkeypatch.setenv('REST_CORS_ALLOWED_ORIGINS',
                       'https://prod-customer-1.kumo.ai')

    rest_settings = settings.parse_from_env_vars()
    _record_or_verify(rest_settings, 'test_prod_settings_prpr.json')


def test_dev_all_features(monkeypatch: MonkeyPatch):
    monkeypatch.setenv('DEPLOY_STAGE', DeployStage.DEV.value)
    monkeypatch.setenv('INSTANCE_ID', 'devtest_instance_id')
    monkeypatch.setenv('REST_API_PORT', '12345')
    monkeypatch.setenv('LOGNAME', 'testuser')  # used in s3 db path
    # All "experimental" features turned on:
    monkeypatch.setenv('ENABLE_TEMPORAL_BACKEND', '1')
    monkeypatch.setenv('ENABLE_ASYNC_INGESTION', '1')
    monkeypatch.setenv('TEMPORAL_SERVER', 'localhost')

    rest_settings = settings.parse_from_env_vars(disable_auth=True)
    _record_or_verify(rest_settings, 'test_dev_settings.json')


def _set_direct_connect_parse_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv('DEPLOY_STAGE', DeployStage.DEV.value)
    monkeypatch.delenv('REST_CORS_ALLOWED_ORIGINS', raising=False)


@pytest.mark.parametrize('auth_backend', ['auth0', 'keycloak'])
def test_parse_rejects_auth0_keycloak_backends(
    monkeypatch: MonkeyPatch,
    auth_backend: str,
):
    monkeypatch.setenv('DEPLOY_STAGE', DeployStage.DEV.value)
    monkeypatch.setenv('KUMO_AUTH_BACKEND', auth_backend)

    with pytest.raises(ValueError, match='Direct Connect'):
        settings.parse_from_env_vars(disable_auth=True)


def test_cors_validation_allows_wildcard_without_credentials():
    settings.validate_cors_allowed_origins(['*'], allow_credentials=False)


def test_cors_validation_rejects_wildcard_with_credentials():
    with pytest.raises(ValueError, match='REST_CORS_ALLOWED_ORIGINS'):
        settings.validate_cors_allowed_origins(['*'], allow_credentials=True)


def test_direct_connect_rejects_default_wildcard_cors_origin(
        monkeypatch: MonkeyPatch):
    _set_direct_connect_parse_env(monkeypatch)

    with pytest.raises(ValueError, match='REST_CORS_ALLOWED_ORIGINS'):
        settings.parse_from_env_vars()


def test_parse_cors_allowed_origins_normalizes_url_origins():
    assert settings.parse_cors_allowed_origins('https://kumo.example.com/, '
                                               'https://kumo.example.com/app, '
                                               'http://localhost:5173/') == [
                                                   'https://kumo.example.com',
                                                   'https://kumo.example.com',
                                                   'http://localhost:5173',
                                               ]


def test_direct_connect_accepts_explicit_cors_origins(monkeypatch: MonkeyPatch):
    _set_direct_connect_parse_env(monkeypatch)
    middleware_origin = 'https://kumo.example.com, http://localhost:5173'
    monkeypatch.setenv('REST_CORS_ALLOWED_ORIGINS', middleware_origin)

    rest_settings = settings.parse_from_env_vars()

    assert rest_settings.app.middleware_origin == middleware_origin
    assert settings.parse_cors_allowed_origins(middleware_origin) == [
        'https://kumo.example.com',
        'http://localhost:5173',
    ]


def test_direct_connect_uses_deploy_config_return_origins_when_cors_env_unset(
        monkeypatch: MonkeyPatch):
    monkeypatch.setenv('DEPLOY_STAGE', DeployStage.DEV.value)
    monkeypatch.delenv('KUMO_AUTH_BACKEND', raising=False)
    monkeypatch.delenv('REST_CORS_ALLOWED_ORIGINS', raising=False)
    monkeypatch.setenv(
        'DEPLOY_ENV_CONFIG_JSON',
        json.dumps({
            'auth': {
                'enabled': True,
                'backend': 'direct_connect',
                'direct_connect': {
                    'allowed_return_origins': [
                        'https://kumo.example.com',
                    ],
                },
            },
            'logging': {
                'backend': 'stdout',
            },
        }))

    rest_settings = settings.parse_from_env_vars()

    assert not hasattr(rest_settings.authentication, 'auth_backend')
    assert rest_settings.authentication.direct_connect_login_method == 'oidc'
    assert rest_settings.app.middleware_origin == 'https://kumo.example.com'


def test_direct_connect_reads_auth_from_deploy_env_config(
        monkeypatch: MonkeyPatch):
    monkeypatch.setenv('DEPLOY_STAGE', DeployStage.DEV.value)
    monkeypatch.delenv('KUMO_AUTH_BACKEND', raising=False)
    monkeypatch.delenv('REST_CORS_ALLOWED_ORIGINS', raising=False)
    monkeypatch.setenv('DIRECT_CONNECT_PASSWORD_LOGIN_IP_RATE_LIMIT_ENABLED',
                       'false')
    monkeypatch.setenv(
        'DEPLOY_ENV_CONFIG_JSON',
        json.dumps({
            'auth': {
                'enabled': True,
                'backend': 'direct_connect',
                'direct_connect': {
                    'login_method': 'password',
                    'allowed_return_origins': [
                        'https://kumo.example.com',
                    ],
                },
            },
            'logging': {
                'backend': 'stdout',
            },
        }))

    rest_settings = settings.parse_from_env_vars()

    assert rest_settings.authentication.disable is False
    assert not hasattr(rest_settings.authentication, 'auth_backend')
    assert (
        rest_settings.authentication.direct_connect_login_method == 'password')
    assert (rest_settings.authentication.
            direct_connect_password_login_ip_rate_limit_enabled is False)
    assert rest_settings.app.middleware_origin == 'https://kumo.example.com'


def test_data_scale_edge_limits():
    limits = DataScaleLimits()
    limits.max_graph_total_edges = 10 * 10**9  # 10 billion

    # If a graph has 6 edge types with 10 billion edges each.
    # This should raise an error.
    resp = limits.validate_graph(edge_counts=6 * [10 * 10**9])
    assert len(resp.errors) == 1
