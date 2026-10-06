# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Dict, Optional

from statedb.db.services import StateServices

EXTERNAL_GROUP_CACHE_TTL_SECONDS = 5 * 60


@dataclass(frozen=True)
class ResolvedGroupMembership:
    """Request-local Kumo group membership and provider availability state.

    ``group_ids`` is always derived from current native rules plus a cached or
    freshly fetched provider result.  Provider identifiers themselves remain
    inside the resolver cache and are never persisted on the principal.
    """

    group_ids: tuple[str, ...]
    external_membership_unavailable: bool = False


class NativeGroupResolver:
    """Resolve live Kumo-native membership without an external adapter."""
    def __init__(self, state_services: Optional[StateServices] = None):
        self._state_services = state_services

    def configure_state_services(self, state_services: StateServices) -> None:
        self._state_services = state_services

    async def resolve_groups(
        self,
        user_id: str,
        auth_context: Optional[Dict[str, Any]],
    ) -> ResolvedGroupMembership:
        del auth_context
        if self._state_services is None:
            raise RuntimeError(
                "NativeGroupResolver StateServices is not configured")
        from statedb.rest.group import PrincipalType

        group_ids = await self._state_services.groups.resolve_native_memberships(
            user_id,
            principal_type=PrincipalType.USER,
            authenticated=True,
        )
        return ResolvedGroupMembership(tuple(sorted(set(group_ids))))
