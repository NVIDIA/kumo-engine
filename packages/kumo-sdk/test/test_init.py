# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for kumoai_studio.init."""
import os
import subprocess
import sys
import textwrap
from unittest.mock import patch

import pytest

import kumoai_studio
from kumoai_studio import global_state


@pytest.fixture(autouse=True)
def _reset_global_state():
    """Reset the singleton before/after each test."""
    global_state.clear()
    yield
    global_state.clear()


@pytest.fixture(autouse=True)
def _no_project_lookup():
    """These tests do not exercise project discovery."""
    with patch('kumoai_studio._resolve_project'):
        yield


def test_environment_auto_init_has_version_available() -> None:
    """Environment-based import initialization has its full module state."""
    script = textwrap.dedent("""
        import os
        import runpy
        from unittest.mock import patch

        import kumoai_studio
        from kumoai_studio.client.client import KumoClient
        from kumoai_studio.client.projects import ProjectsAPI
        from kumoai_studio.project import Project
        from kumoapi.data_source import DEFAULT_GROUP_ID, DEFAULT_PROJECT_ID

        os.environ["KUMO_API_ENDPOINT"] = "http://auto-init.example/api"
        os.environ["KUMO_API_KEY"] = "test:auto-init"
        project = Project(
            id=DEFAULT_PROJECT_ID,
            slug="default",
            name="Default",
            group_id=DEFAULT_GROUP_ID,
        )
        with patch.object(KumoClient, "authenticate"), patch.object(
                ProjectsAPI, "get_project", return_value=project):
            namespace = runpy.run_path(
                kumoai_studio.__file__, run_name="kumoai_auto_init_test")

        assert namespace["global_state"].initialized
        assert namespace["__version__"] == kumoai_studio.__version__
        """)
    env = dict(os.environ)
    env.pop("KUMO_API_ENDPOINT", None)
    env.pop("KUMO_API_KEY", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_init_without_an_api_key_is_refused(monkeypatch):
    """The API key is the only credential init() accepts.

    It used to accept Snowflake credentials as well, for a Kumo deployed to
    Snowpark Container Services; that path is gone, so a caller who reaches
    here with neither is told the one thing that is missing.
    """
    monkeypatch.delenv("KUMO_API_KEY", raising=False)

    with pytest.raises(ValueError, match="no API key was provided"):
        kumoai_studio.init(url="http://test.example.com/api")

    assert not global_state.initialized


def test_init_infers_an_https_endpoint_from_the_api_key(monkeypatch):
    monkeypatch.delenv("KUMO_API_ENDPOINT", raising=False)
    with patch("kumoai_studio.KumoClient") as client:
        client.return_value._url = "https://tenant.kumoai.cloud/api"
        client.return_value._api_key = "tenant:secret"
        kumoai_studio.init(api_key="tenant:secret")

    client.assert_called_once_with(url="https://tenant.kumoai.cloud/api",
                                   api_key="tenant:secret")
