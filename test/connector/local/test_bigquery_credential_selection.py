# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Credential selection for BigQuery permission tests without cloud access."""

import logging
import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, mock_open, patch

import orjson
import pytest

import io_utils


@pytest.fixture
def credentials_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Isolate the credential helper from connector SDKs and secret providers."""
    monkeypatch.setattr(io_utils, "logging", logging, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "io_utils.secrets",
        SimpleNamespace(
            get_secrets=Mock(
                return_value={
                    "type": "service_account",
                }
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "kumo.connector.bigquery_connector",
        SimpleNamespace(
            ENV_GCS_KUMO_BUCKET=SimpleNamespace(name="TEST_GCS_BUCKET"),
            ENV_GOOGLE_CREDENTIALS_DEFAULT_PATH="/mock-default.json",
            ENV_GOOGLE_CREDENTIALS_JSON="GOOGLE_CREDENTIALS_JSON",
        ),
    )
    path = Path(__file__).parents[3] / "kumo/testing/bigquery.py"
    spec = spec_from_file_location("isolated_bigquery_credentials", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/mock-adc.json")
    monkeypatch.delenv(module.ENV_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT, raising=False)
    monkeypatch.delenv(module.ENV_BIGQUERY_NO_JOB_SERVICE_ACCOUNT, raising=False)
    monkeypatch.delenv(module.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT, raising=False)
    monkeypatch.delenv(module.ENV_GOOGLE_CREDENTIALS_JSON, raising=False)
    return module


@pytest.mark.parametrize(
    "key_name",
    [
        "FAIL_TABLE_PERMISSION_BIGQUERY_KEY",
        "FAIL_PROJECT_PERMISSION_BIGQUERY_KEY",
    ],
)
def test_restricted_key_tests_load_their_own_credentials(
    credentials_module: ModuleType,
    key_name: str,
) -> None:
    """An existing service-account key must not bypass restricted-key selection."""
    key = getattr(credentials_module, key_name)
    with (
        patch.object(credentials_module.os.path, "exists", return_value=True),
        patch(
            "builtins.open",
            mock_open(
                read_data=orjson.dumps(
                    {
                        "type": "service_account",
                    }
                )
            ),
        ),
    ):
        credentials_module.setup_bigquery_credentials(key)
    credentials_module.get_secrets.assert_called_once_with(key)
    assert credentials_module.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT not in os.environ


@pytest.mark.parametrize(
    "key_name",
    [
        "FAIL_TABLE_PERMISSION_BIGQUERY_KEY",
        "FAIL_PROJECT_PERMISSION_BIGQUERY_KEY",
    ],
)
def test_keyless_tests_still_require_restricted_identity(
    credentials_module: ModuleType,
    key_name: str,
) -> None:
    """Missing keyless CI impersonation configuration remains a hard error."""
    with (
        patch.object(credentials_module.os.path, "exists", return_value=True),
        patch(
            "builtins.open",
            mock_open(
                read_data=orjson.dumps(
                    {
                        "type": "external_account",
                    }
                )
            ),
        ),
        pytest.raises(RuntimeError, match="must be configured"),
    ):
        credentials_module.setup_bigquery_credentials(
            getattr(credentials_module, key_name)
        )
    credentials_module.get_secrets.assert_not_called()


def test_keyless_tests_impersonate_configured_identity(
    credentials_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured keyless tests retain impersonation without loading stored keys."""
    monkeypatch.setenv(
        credentials_module.ENV_BIGQUERY_NO_TABLE_SERVICE_ACCOUNT,
        "restricted@example.invalid",
    )
    with patch.object(credentials_module.os.path, "exists", return_value=True):
        credentials_module.setup_bigquery_credentials(
            credentials_module.FAIL_TABLE_PERMISSION_BIGQUERY_KEY
        )
    credentials_module.get_secrets.assert_not_called()
    assert os.environ[credentials_module.ENV_GOOGLE_IMPERSONATE_SERVICE_ACCOUNT] == (
        "restricted@example.invalid"
    )
    assert (
        os.environ[credentials_module.ENV_GOOGLE_CREDENTIALS_JSON] == "/mock-adc.json"
    )
