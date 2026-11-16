# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from rest.config.direct_connect import resolve_auth_settings_config
from rest.deps.authentication import (
    AUTHORIZATION_DISABLED,
    AUTHORIZATION_PROXIED,
)
from rest.deps.direct_connect_group_resolver import DirectConnectGroupResolver
from rest.driver.direct_connect_auth_storage import DirectConnectAuthStore
from service_framework.deploy_env_config import deploy_env
from statedb.config.common import UserID
from statedb.db.auth import AuthPrincipal
from statedb.db.services import StateServices
from statedb.rest.connector_authorization import ConnectorAuthorizationDBEntry


class PrincipalAuthorityDeniedError(PermissionError):
    """A capability owner no longer has a valid authoritative principal."""


class PrincipalAuthorityUnavailableError(RuntimeError):
    """Authoritative membership could not be resolved transiently."""


@dataclass(frozen=True)
class _CurrentPrincipal:
    user_id: UserID
    user_name: str
    kumo_group_ids: tuple[str, ...]
    external_membership_unavailable: bool
    is_admin: bool


class WorkerPrincipalAuthorityResolver:
    """Resolve a capability owner from current authoritative membership."""
    def __init__(self, services: StateServices) -> None:
        self._services = services
        self._direct_resolver: DirectConnectGroupResolver | None = None

    async def resolve(
        self,
        authorization: ConnectorAuthorizationDBEntry,
    ) -> AuthPrincipal:
        configured = resolve_auth_settings_config(deploy_env())
        synthetic_backend = {
            AUTHORIZATION_DISABLED: 'native_disabled',
            AUTHORIZATION_PROXIED: 'native_proxied',
        }.get(authorization.principal_id)
        if (synthetic_backend is not None
                and authorization.auth_backend != synthetic_backend):
            # Synthetic request principals are deployment-mode sentinels, not
            # identities that an external provider can resolve.  A capability
            # carrying one under any other backend is inconsistent (for
            # example after a mode change) and must fail before a Direct
            # Connect lookup is attempted.
            raise PrincipalAuthorityDeniedError(
                'Synthetic request authority is invalid')

        if authorization.auth_backend == 'native_disabled':
            if configured.enabled:
                raise PrincipalAuthorityDeniedError(
                    'Authentication mode changed')
            membership = await self._resolve_native_request(authorization)
            is_admin = authorization.principal_id == AUTHORIZATION_DISABLED
        elif authorization.auth_backend == 'native_proxied':
            if not configured.enabled or not configured.proxy:
                raise PrincipalAuthorityDeniedError(
                    'Authentication mode changed')
            membership = await self._resolve_native_request(authorization)
            is_admin = False
        elif authorization.auth_backend != 'direct_connect':
            raise PrincipalAuthorityDeniedError(
                'Authentication backend changed')
        elif configured.enabled and not configured.proxy:
            membership, is_admin = await self._resolve_direct_connect(
                authorization.principal_id)
        else:
            raise PrincipalAuthorityDeniedError('Authentication mode changed')

        if membership.external_membership_unavailable:
            raise PrincipalAuthorityUnavailableError(
                'External membership is unavailable')
        return _CurrentPrincipal(
            user_id=UserID(authorization.principal_id),
            user_name=authorization.principal_name,
            kumo_group_ids=membership.group_ids,
            external_membership_unavailable=False,
            is_admin=is_admin,
        )

    async def _resolve_native_request(
        self,
        authorization: ConnectorAuthorizationDBEntry,
    ):
        from rest.deps.group_membership import ResolvedGroupMembership
        from statedb.rest.group import PrincipalType
        groups = await self._services.groups.resolve_native_memberships(
            authorization.principal_id,
            principal_type=PrincipalType.USER,
            authenticated=True,
        )
        return ResolvedGroupMembership(tuple(sorted(set(groups))))

    async def _resolve_direct_connect(self, user_id: str):
        if self._direct_resolver is None:
            store = DirectConnectAuthStore(self._services.direct_connect_auth)
            self._direct_resolver = DirectConnectGroupResolver(
                state_services=self._services,
                auth_store=store,
            )
        membership = await self._direct_resolver.resolve_groups(user_id, None)
        record = await self._services.direct_connect_auth.get_user(user_id)
        if record is None or record.disabled:
            raise PrincipalAuthorityDeniedError(
                'Direct Connect user is inactive')
        return membership, record.role == 'admin'
