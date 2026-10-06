# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

_SDK_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKFLOW = _REPO_ROOT / ".github/workflows/kumo-sdk-test-integration.yml"
_INTEGRATION_AUTH_FILES = (
    _WORKFLOW,
    _SDK_ROOT / "test/conftest.py",
    _SDK_ROOT / "test/test_client.py",
    _SDK_ROOT / "test/codegen/test_integration_e2e.py",
)


def test_integration_uses_a_revocable_owner_bound_key() -> None:
    for path in _INTEGRATION_AUTH_FILES:
        assert "test:DISABLED" not in path.read_text()

    workflow = _WORKFLOW.read_text()
    assert "KUMO_ENABLE_UNAUTHENTICATED_API_KEY_STAGING=1" in workflow
    assert "POST http://localhost:10001/v0/api-keys/stage" in workflow
    assert workflow.index(
        "http://localhost:10001/v0/authentication/organization") < (
            workflow.index("POST http://localhost:10001/v0/api-keys/stage"))
    assert "DELETE" in workflow
    assert "if: always() && env.KUMO_STAGED_API_KEY_ID != ''" in workflow


def test_integration_key_is_masked_before_export_and_never_printed() -> None:
    workflow = _WORKFLOW.read_text()
    mask = 'echo "::add-mask::$api_key"'
    export = 'echo "KUMO_API_KEY=$api_key" >> "$GITHUB_ENV"'

    assert workflow.index(mask) < workflow.index(export)
    assert 'echo "$api_key"' not in workflow
    assert 'echo "$response"' not in workflow
    assert "set -x" not in workflow
