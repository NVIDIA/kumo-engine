# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

import rest.config.constants as constants


@pytest.fixture(autouse=True)
def _stable_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        constants,
        'deploy_env',
        lambda: SimpleNamespace(data_plane_storage_root='file:///tmp/kumo'),
    )
    monkeypatch.setattr(constants.fsutil, 'is_gcs', lambda _path: False)
    monkeypatch.setattr(
        constants,
        'supports_backend_file_upload',
        lambda _path: True,
    )
    monkeypatch.setattr(
        constants.developer_credentials,
        'dev_gcp_user_credentials_enabled',
        lambda: False,
    )


def test_gcp_workload_exchange_flag_defaults_off(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', raising=False)

    flags = constants.FeatureFlags.parse_from_env_vars()

    assert flags.GCP_WORKLOAD_EXCHANGE_ENABLED is False


def test_gcp_workload_exchange_flag_reaches_safe_feature_flags(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED', 'true')

    flags = constants.FeatureFlags.parse_from_env_vars()

    assert flags.GCP_WORKLOAD_EXCHANGE_ENABLED is True


def test_devserver_google_user_adc_enables_ambient_product_flow(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        constants.developer_credentials,
        'dev_gcp_user_credentials_enabled',
        lambda: True,
    )

    flags = constants.FeatureFlags.parse_from_env_vars()

    assert flags.GCP_AMBIENT_ENABLED is True
    assert flags.GCP_AMBIENT_DEVELOPER_CREDENTIALS is True
    assert flags.GCP_AMBIENT_SERVICE_ACCOUNT_EMAIL is None


def test_project_authorization_is_not_feature_gated() -> None:
    """Project/group authorization is the sole supported product model."""
    assert 'RBAC_ENABLED' not in constants.FeatureFlags.__dataclass_fields__
