# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rest.backends.data_backend.principal_authority import (
    PrincipalAuthorityDeniedError,
    PrincipalAuthorityUnavailableError,
    WorkerPrincipalAuthorityResolver,
)
from rest.deps.authentication import (
    AUTHORIZATION_DISABLED,
    AUTHORIZATION_PROXIED,
    UserAuthorization,
)
from statedb.config.common import UserID
from statedb.db.identities import _authorization_issue_context


def _authorization(*, owner: str, backend: str, name='user'):
    return SimpleNamespace(
        principal_id=owner,
        principal_name=name,
        auth_backend=backend,
    )


def _auth_settings(enabled: bool, proxy: bool):
    return SimpleNamespace(
        enabled=enabled,
        proxy=proxy,
        backend='direct_connect',
    )


@pytest.mark.parametrize(
    ('enabled', 'proxy', 'expected_backend'),
    [
        (False, False, 'native_disabled'),
        (True, True, 'native_proxied'),
    ],
)
def test_request_native_modes_are_users_not_internal_services(
    monkeypatch,
    enabled,
    proxy,
    expected_backend,
) -> None:
    principal = UserAuthorization(
        user_id=UserID('verified-api-key-owner'),
        access_token='',
        permissions=(),
    )
    monkeypatch.setattr(
        'rest.config.direct_connect.resolve_auth_settings_config',
        lambda _config: _auth_settings(enabled, proxy),
    )
    monkeypatch.setattr(
        'service_framework.deploy_env_config.deploy_env',
        lambda: SimpleNamespace(),
    )
    backend, _ = _authorization_issue_context(principal)
    assert backend == expected_backend


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('backend', 'owner', 'enabled', 'proxy', 'expected_admin'),
    [
        ('native_disabled', AUTHORIZATION_DISABLED, False, False, True),
        ('native_disabled', 'phase1b-api-key-staging', False, False, False),
        ('native_proxied', AUTHORIZATION_PROXIED, True, True, False),
    ],
)
async def test_native_request_authority_uses_current_membership(
    monkeypatch,
    backend,
    owner,
    enabled,
    proxy,
    expected_admin,
) -> None:
    groups = SimpleNamespace(resolve_native_memberships=AsyncMock(
        return_value=['current-group']))
    resolver = WorkerPrincipalAuthorityResolver(SimpleNamespace(groups=groups))
    monkeypatch.setattr(
        'rest.backends.data_backend.principal_authority.'
        'resolve_auth_settings_config',
        lambda _config: _auth_settings(enabled, proxy),
    )
    monkeypatch.setattr(
        'rest.backends.data_backend.principal_authority.deploy_env',
        lambda: SimpleNamespace(deploy_mode=SimpleNamespace(is_preprod=False)),
    )

    principal = await resolver.resolve(
        _authorization(owner=owner, backend=backend))

    assert principal.kumo_group_ids == ('current-group', )
    assert principal.is_admin is expected_admin
    groups.resolve_native_memberships.assert_awaited_once()
    assert groups.resolve_native_memberships.await_args.args[0] == owner


@pytest.mark.asyncio
async def test_synthetic_request_owner_never_reaches_external_resolver(
        monkeypatch) -> None:
    resolver = WorkerPrincipalAuthorityResolver(
        SimpleNamespace(groups=SimpleNamespace()))
    resolver._resolve_direct_connect = AsyncMock()
    monkeypatch.setattr(
        'rest.backends.data_backend.principal_authority.'
        'resolve_auth_settings_config',
        lambda _config: _auth_settings(True, False),
    )
    monkeypatch.setattr(
        'rest.backends.data_backend.principal_authority.deploy_env',
        lambda: SimpleNamespace(deploy_mode=SimpleNamespace(is_preprod=False)),
    )

    with pytest.raises(PermissionError,
                       match='Synthetic request authority is invalid'):
        await resolver.resolve(
            _authorization(
                owner=AUTHORIZATION_DISABLED,
                backend='direct_connect',
            ))

    resolver._resolve_direct_connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_external_membership_outage_remains_retryable_unavailable(
        monkeypatch) -> None:
    resolver = WorkerPrincipalAuthorityResolver(
        SimpleNamespace(groups=SimpleNamespace()))
    resolver._resolve_native_request = AsyncMock(return_value=SimpleNamespace(
        group_ids=(),
        external_membership_unavailable=True,
    ))
    monkeypatch.setattr(
        'rest.backends.data_backend.principal_authority.'
        'resolve_auth_settings_config',
        lambda _config: _auth_settings(False, False),
    )
    monkeypatch.setattr(
        'rest.backends.data_backend.principal_authority.deploy_env',
        lambda: SimpleNamespace(deploy_mode=SimpleNamespace(is_preprod=False)),
    )

    with pytest.raises(PrincipalAuthorityUnavailableError):
        await resolver.resolve(
            _authorization(owner=AUTHORIZATION_DISABLED,
                           backend='native_disabled'))


def test_principal_authority_error_categories_preserve_retry_semantics(
) -> None:
    assert issubclass(PrincipalAuthorityDeniedError, PermissionError)
    assert not issubclass(PrincipalAuthorityUnavailableError, PermissionError)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('backend', 'owner', 'enabled', 'proxy'),
    [
        ('native_disabled', AUTHORIZATION_DISABLED, True, False),
        ('native_proxied', AUTHORIZATION_PROXIED, True, False),
    ],
)
async def test_native_request_authority_rejects_owner_or_mode_mismatch(
    monkeypatch,
    backend,
    owner,
    enabled,
    proxy,
) -> None:
    groups = SimpleNamespace(resolve_native_memberships=AsyncMock(
        return_value=['current-group']))
    resolver = WorkerPrincipalAuthorityResolver(SimpleNamespace(groups=groups))
    monkeypatch.setattr(
        'rest.backends.data_backend.principal_authority.'
        'resolve_auth_settings_config',
        lambda _config: _auth_settings(enabled, proxy),
    )
    monkeypatch.setattr(
        'rest.backends.data_backend.principal_authority.deploy_env',
        lambda: SimpleNamespace(deploy_mode=SimpleNamespace(is_preprod=False)),
    )

    with pytest.raises(PermissionError):
        await resolver.resolve(_authorization(owner=owner, backend=backend))


@pytest.mark.asyncio
async def test_external_user_cannot_impersonate_internal_service(
        monkeypatch) -> None:
    resolver = WorkerPrincipalAuthorityResolver(
        SimpleNamespace(groups=SimpleNamespace()))
    monkeypatch.setattr(
        'rest.backends.data_backend.principal_authority.'
        'resolve_auth_settings_config',
        lambda _config: _auth_settings(True, False),
    )
    monkeypatch.setattr(
        'rest.backends.data_backend.principal_authority.deploy_env',
        lambda: SimpleNamespace(deploy_mode=SimpleNamespace(is_preprod=False)),
    )
    authorization = _authorization(
        owner='internal_service:scheduled_workflow',
        backend='internal_service',
        name='scheduled_workflow',
    )

    with pytest.raises(PermissionError):
        await resolver.resolve(authorization)
