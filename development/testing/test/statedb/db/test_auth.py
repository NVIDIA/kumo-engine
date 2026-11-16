# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the StateDB-native principal abstraction."""
import pytest

from rest.deps.authentication import UserAuthorization
from statedb.db.auth import AuthPrincipal
from statedb.db.services import GroupService, ProjectService
from statedb.utils.errors import ProjectForbiddenError


def test_user_authorization_is_usable_as_auth_principal() -> None:
    # Structural typing smoke test: a UserAuthorization satisfies the protocol,
    # so the rest layer can pass it down without statedb importing rest.
    def read_identity(principal: AuthPrincipal) -> tuple[str, bool]:
        return str(principal.user_id), principal.is_admin

    user_id, is_admin = read_identity(UserAuthorization.disabled())
    assert user_id == "_AUTHORIZATION_DISABLED_"
    assert is_admin is True


def test_state_services_reject_missing_principal_instead_of_admin_bypass(
) -> None:
    principal = UserAuthorization.disabled()
    for service in (GroupService, ProjectService):
        with pytest.raises(ProjectForbiddenError,
                           match='Authenticated principal is required'):
            service._require_principal(None)
        assert service._require_principal(principal) is principal
