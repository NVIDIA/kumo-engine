# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ensure the pinned API artifact exposes only the end-state contract."""

import kumoapi.data_source as api


def test_connector_identity_contract_is_available_from_pinned_artifact(
) -> None:
    assert api.IdentityResponse is not None
    assert api.CreateIdentityArgs is not None
    assert api.CreateConnectorArgs is not None
    assert api.ProjectUploadConnectorResponse is not None
    assert api.CreateConnectorArgs.__dataclass_fields__[
        "identity_id"].default is None


def test_identity_admin_lifecycle_contract_is_available() -> None:
    """The SDK identity administration surface (create/update/rotate/groups)
    depends on these models being present in the pinned API artifact."""
    assert api.UpdateIdentityArgs is not None
    assert api.RotateIdentityCredentialsArgs is not None
    assert api.ReplaceIdentityGroupsArgs is not None
    assert api.IdentityGroupsResponse is not None
