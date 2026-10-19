# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify Ferry authentication and publication boundaries without network access."""

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import orjson
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/kitmaker_release.py"


@pytest.fixture
def release(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the release script with synthetic credentials and OIDC settings."""
    spec = importlib.util.spec_from_file_location("kitmaker_release", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in {
        "KITMAKER_API_TOKEN": "portal-token",
        "KITMAKER_PIC": "owner@example.com",
        "GITHUB_TOKEN": "github-token",
        "ACTIONS_ID_TOKEN_REQUEST_URL": "https://oidc.example/token?request=1",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "oidc-request-token",
    }.items():
        monkeypatch.setenv(name, value)
    return module


def test_ferry_refreshes_identity_and_preserves_portal_auth(
    release: ModuleType,
) -> None:
    """Each Portal request gets a new identity token and its bearer credential."""
    responses = []
    for data in [{"value": "identity-one"}, {}, {"value": "identity-two"}, {}]:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = orjson.dumps(data)
        responses.append(response)
    with patch.object(release.urllib.request, "urlopen", side_effect=responses) as send:
        portal = release.Api(
            "http://127.0.0.1:8888/kitmaker-portal/api/v0", "secret", True
        )
        portal.call("GET", "/projects/6903")
        portal.call("GET", "/status/example")

    requests = [call.args[0] for call in send.call_args_list]
    assert "audience=charon.nvidia.com" in requests[0].full_url
    assert requests[0].get_header("Authorization") == "Bearer oidc-request-token"
    assert requests[1].get_header("Authorization") == "Bearer secret"
    assert requests[1].get_header("X-charon-gha-token") == "identity-one"
    assert requests[3].get_header("X-charon-gha-token") == "identity-two"


@pytest.mark.parametrize("payload", [{}, {"value": ""}, {"value": 42}])
def test_ferry_rejects_missing_identity(
    release: ModuleType, payload: dict[str, object]
) -> None:
    """An invalid OIDC response fails before a Portal request can be sent."""
    with patch.object(release.urllib.request, "urlopen") as send:
        send.return_value.__enter__.return_value.read.return_value = orjson.dumps(
            payload
        )
        with pytest.raises(SystemExit, match="no Charon identity token"):
            release.Api("http://localhost", "secret", True).call(
                "GET", "/projects/6903"
            )
    assert send.call_count == 1


def test_ferry_requires_oidc_permission(
    release: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing OIDC configuration fails without network requests."""
    monkeypatch.delenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    with (
        patch.object(release.urllib.request, "urlopen") as send,
        pytest.raises(SystemExit, match="id-token: write"),
    ):
        release._ferry_headers()
    send.assert_not_called()


def test_github_does_not_receive_ferry_identity(release: ModuleType) -> None:
    """GitHub requests retain their own credentials without a Charon token."""
    with (
        patch.object(release, "_ferry_headers") as identity,
        patch.object(release.urllib.request, "urlopen") as send,
    ):
        send.return_value.__enter__.return_value.read.return_value = b"{}"
        release.Api("https://api.github.com", "github-token").call("GET", "/repos")
    identity.assert_not_called()
    assert send.call_args.args[0].get_header("X-charon-gha-token") is None


@pytest.mark.parametrize("wrong_owner", [False, True])
def test_connectivity_only_reads_project(
    release: ModuleType, wrong_owner: bool
) -> None:
    """Connectivity mode only reads the project and verifies the configured owner."""
    with patch.object(release, "Api") as api:
        api.return_value.call.return_value = {
            "name": "kumo-engine",
            "owner_email": "other@example.com" if wrong_owner else "owner@example.com",
        }
        arguments = [
            "--project-id",
            "6903",
            "--project",
            "kumo-engine",
            "--platform",
            "manylinux_2_34_x86_64",
            "--check-connectivity",
        ]
        if wrong_owner:
            with pytest.raises(SystemExit, match="owner does not match"):
                release.main(arguments)
        else:
            release.main(arguments)
    api.return_value.call.assert_called_once_with("GET", "/projects/6903")


@pytest.mark.parametrize("dry_run", [False, True])
def test_dry_run_never_publishes(release: ModuleType, dry_run: bool) -> None:
    """Dry runs submit validation only; release mode validates before publication."""
    with (
        patch.object(release, "Api") as api,
        patch.object(release, "attached", return_value={"wheel.whl": "asset"}),
        patch.object(release, "asset_urls", return_value=["https://example.com/wheel"]),
        patch.object(release, "submit", return_value=["uuid"]) as submit,
        patch.object(release, "wait"),
        patch.object(release, "verify_published") as verify,
    ):
        api.return_value.call.return_value = {
            "name": "kumo-engine",
            "owner_email": "owner@example.com",
        }
        arguments = [
            "--project-id",
            "6903",
            "--project",
            "kumo-engine",
            "--platform",
            "any",
            "--repo",
            "NVIDIA/kumo-engine",
            "--tag",
            "kumo-engine-v1.0.0",
            "wheel.whl",
        ]
        if dry_run:
            arguments.insert(0, "--dry-run")
        release.main(arguments)

    assert [call.args[-1] for call in submit.call_args_list] == (
        [False] if dry_run else [False, True]
    )
    assert verify.call_count == (0 if dry_run else 1)


def test_release_requires_assets(release: ModuleType) -> None:
    """An incomplete release command fails before API access."""
    with patch.object(release, "Api") as api, pytest.raises(SystemExit):
        release.main(
            [
                "--project-id",
                "6903",
                "--project",
                "kumo-engine",
                "--platform",
                "any",
            ]
        )
    api.assert_not_called()


def test_wheel_digest_streams_file(release: ModuleType) -> None:
    """Hash a wheel through its file handle without loading it all into memory."""
    from io import BytesIO

    wheel = MagicMock()
    wheel.open.return_value.__enter__.return_value = BytesIO(b"abc")
    assert release.sha256(wheel) == (
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    wheel.open.assert_called_once_with("rb")
    wheel.read_bytes.assert_not_called()
