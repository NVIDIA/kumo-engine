# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from kumo.testing.databricks import (
    withDatabricksCredentials,
    withDatabricksOauthCredentials,
)


def _result() -> str:
    return "ran"


def test_personal_access_token_decorator_uses_environment(monkeypatch) -> None:
    monkeypatch.setenv("DATABRICKS_HOST", "workspace.example.test")
    monkeypatch.setenv("DATABRICKS_TOKEN", "token")

    assert withDatabricksCredentials(_result)() == "ran"


def test_personal_access_token_decorator_skips_without_environment(
        monkeypatch) -> None:
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

    with pytest.raises(pytest.skip.Exception):
        withDatabricksCredentials(_result)()


def test_oauth_decorator_uses_environment(monkeypatch) -> None:
    monkeypatch.setenv("DATABRICKS_HOST", "workspace.example.test")
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "client")
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "secret")

    assert withDatabricksOauthCredentials(_result)() == "ran"
