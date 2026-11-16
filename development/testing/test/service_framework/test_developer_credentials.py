# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from service_framework.developer_credentials import (
    ENV_DEV_GCP_USER_CREDENTIALS_ENABLED,
    dev_gcp_user_credentials_enabled,
)


@pytest.mark.parametrize('deploy_stage', ['prod', 'staging', 'preprod', 'test'])
def test_google_user_adc_never_enabled_outside_dev_stage(
        monkeypatch: pytest.MonkeyPatch, deploy_stage: str) -> None:
    monkeypatch.setenv(ENV_DEV_GCP_USER_CREDENTIALS_ENABLED, 'true')
    monkeypatch.setenv('DEPLOY_STAGE', deploy_stage)

    assert dev_gcp_user_credentials_enabled() is False


@pytest.mark.parametrize('deploy_stage', ['dev', 'dev_local'])
def test_google_user_adc_enabled_for_opted_in_devserver(
        monkeypatch: pytest.MonkeyPatch, deploy_stage: str) -> None:
    monkeypatch.setenv(ENV_DEV_GCP_USER_CREDENTIALS_ENABLED, 'true')
    monkeypatch.setenv('DEPLOY_STAGE', deploy_stage)

    assert dev_gcp_user_credentials_enabled() is True


def test_google_user_adc_requires_explicit_boolean(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_DEV_GCP_USER_CREDENTIALS_ENABLED, 'enabled')
    monkeypatch.setenv('DEPLOY_STAGE', 'dev_local')

    with pytest.raises(ValueError, match='explicit true/false'):
        dev_gcp_user_credentials_enabled()
