# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, MutableMapping, Optional

import cachetools

from rest.deps.group_membership import (
    EXTERNAL_GROUP_CACHE_TTL_SECONDS,
    ResolvedGroupMembership,
)
from rest.driver.direct_connect_auth_storage import DirectConnectAuthStore
from statedb.db.services import StateServices

_GROUP_CACHE_MAXSIZE = 1000
_GROUP_CACHE_TTL_SECONDS = EXTERNAL_GROUP_CACHE_TTL_SECONDS
_IDENTITY_PROVIDER_ID = "direct_connect"


class DirectConnectGroupResolver:
    """Resolve native membership plus transient current OIDC membership."""
    def __init__(
        self,
        state_services: Optional[StateServices] = None,
        auth_store: Optional[DirectConnectAuthStore] = None,
        cache: Optional[MutableMapping[str, List[str]]] = None,
    ) -> None:
        self._state_services = state_services
        self._auth_store = auth_store
        self._group_cache: MutableMapping[str, List[str]] = (
            cache if cache is not None else cachetools.TTLCache(
                maxsize=_GROUP_CACHE_MAXSIZE,
                ttl=_GROUP_CACHE_TTL_SECONDS,
            ))

    def remember_external_groups(
        self,
        user_id: str,
        external_group_ids: List[str],
    ) -> None:
        """Seed the transient cache from a freshly verified OIDC login."""
        self._group_cache[self._cache_key(user_id)] = self._normalize_groups(
            external_group_ids)

    @staticmethod
    def _normalize_groups(external_group_ids) -> List[str]:
        return list(
            dict.fromkeys(cleaned for group_id in (external_group_ids or ())
                          if (cleaned := str(group_id).strip())))

    @staticmethod
    def _cache_key(user_id: str) -> str:
        return f"{_IDENTITY_PROVIDER_ID}|{user_id}"

    def configure_state_services(self, state_services: StateServices) -> None:
        self._state_services = state_services

    async def resolve_groups(
        self,
        user_id: str,
        auth_context: Optional[Dict[str, Any]],
    ) -> ResolvedGroupMembership:
        del auth_context
        native: List[str] = []
        if self._state_services is not None:
            from statedb.rest.group import PrincipalType
            native = await self._state_services.groups.resolve_native_memberships(
                user_id,
                principal_type=PrincipalType.USER,
                authenticated=True,
            )

        unavailable = False
        cache_key = self._cache_key(user_id)
        if cache_key in self._group_cache:
            external_group_ids = list(self._group_cache[cache_key])
        elif self._auth_store is None:
            external_group_ids = []
        else:
            try:
                fetched = await self._auth_store.fetch_external_group_ids(
                    user_id)
            except Exception:
                external_group_ids = []
                unavailable = True
            else:
                external_group_ids = self._normalize_groups(fetched)
                if fetched is not None:
                    # Cache both positive and negative provider results.
                    self._group_cache[cache_key] = external_group_ids

        external: List[str] = []
        if self._state_services is not None:
            external = await self._state_services.groups.map_idp_group_ids(
                external_group_ids,
                identity_provider_id=_IDENTITY_PROVIDER_ID,
            )
        return ResolvedGroupMembership(
            tuple(sorted(set(native) | set(external))),
            external_membership_unavailable=unavailable,
        )
