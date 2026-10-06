# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from io_utils.secrets import get_secrets
from service_framework._raw_deploy_config import (
    AuthConfig,
    DirectConnectAuthConfig,
    SecretKeyRef,
)
from service_framework.deploy_env_config import DeployEnvConfig, deploy_env

DirectConnectLoginMethod = Literal['oidc', 'password']


@dataclass(frozen=True)
class DirectConnectOIDCSeedConfig:
    issuer: str = ''
    client_id: str = ''
    client_secret: str = ''
    groups_claim: str = 'groups'
    # Google Workspace Directory API group lookup (see
    # DirectConnectGoogleGroupsConfig). When set, groups are fetched per
    # login instead of read from the ID-token groups claim.
    google_sa_key: str = ''
    google_admin_email: str = ''


@dataclass(frozen=True)
class DirectConnectResolvedConfig:
    login_method: DirectConnectLoginMethod = 'oidc'
    allowed_return_origins: tuple[str, ...] = ()
    admin_emails: tuple[str, ...] = ()
    oidc: DirectConnectOIDCSeedConfig = field(
        default_factory=DirectConnectOIDCSeedConfig)
    local_admin_temp_password: str = ''


@dataclass(frozen=True)
class AuthResolvedConfig:
    enabled: bool
    proxy: bool
    direct_connect: DirectConnectResolvedConfig


def _truthy_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 't', 'yes', 'y'}


def _auth_cfg(config: DeployEnvConfig) -> Optional[AuthConfig]:
    return config.auth


def _direct_connect_cfg(
        config: DeployEnvConfig) -> Optional[DirectConnectAuthConfig]:
    auth = _auth_cfg(config)
    return auth.direct_connect if auth is not None else None


def _login_method(value: Optional[str]) -> DirectConnectLoginMethod:
    method = (value or 'oidc').lower()
    if method not in {'oidc', 'password'}:
        raise ValueError(
            "Direct Connect login_method must be 'oidc' or 'password', "
            f"got {value!r}")
    return method  # type: ignore[return-value]


def _secret_ref_value(ref: Optional[SecretKeyRef]) -> str:
    if ref is None:
        return ''
    secret = get_secrets(ref.secret_name)
    value = secret.get(ref.secret_key)
    if value is None:
        raise ValueError(
            f"Secret {ref.secret_name!r} is missing key {ref.secret_key!r}")
    return str(value)


def resolve_auth_settings_config(
        config: Optional[DeployEnvConfig] = None) -> AuthResolvedConfig:
    """Resolve non-sensitive auth settings from deploy config with env fallback."""
    resolved_config = config or deploy_env()
    auth = _auth_cfg(resolved_config)
    direct_connect = _direct_connect_cfg(resolved_config)

    if auth is not None and auth.enabled is not None:
        enabled = auth.enabled
    else:
        enabled = not _truthy_env('DISABLE_AUTH')

    configured_backend = (auth.backend if auth is not None and auth.backend else
                          os.getenv('KUMO_AUTH_BACKEND', 'direct_connect'))
    backend = configured_backend.lower()
    if backend != 'direct_connect':
        raise ValueError("Direct Connect is the only supported auth backend. "
                         f"Got {configured_backend!r}.")

    login_method = _login_method(
        direct_connect.login_method if direct_connect is not None
        and direct_connect.login_method else 'oidc')
    allowed_return_origins = (tuple(direct_connect.allowed_return_origins)
                              if direct_connect is not None
                              and direct_connect.allowed_return_origins else ())
    admin_emails = (tuple(direct_connect.admin_emails)
                    if direct_connect is not None
                    and direct_connect.admin_emails else ())

    return AuthResolvedConfig(
        enabled=enabled,
        proxy=_truthy_env('PROXY_AUTH'),
        direct_connect=DirectConnectResolvedConfig(
            login_method=login_method,
            allowed_return_origins=allowed_return_origins,
            admin_emails=admin_emails,
        ),
    )


def resolve_direct_connect_runtime_config(
        config: Optional[DeployEnvConfig] = None
) -> DirectConnectResolvedConfig:
    """Resolve Direct Connect startup settings, including secret values."""
    resolved_config = config or deploy_env()
    settings = resolve_auth_settings_config(resolved_config).direct_connect
    direct_connect = _direct_connect_cfg(resolved_config)
    oidc = direct_connect.oidc if direct_connect is not None else None
    password = direct_connect.password if direct_connect is not None else None

    admin_emails = settings.admin_emails

    client_secret = (_secret_ref_value(oidc.client_secret_ref)
                     if oidc is not None and oidc.client_secret_ref is not None
                     else '')
    google_groups = oidc.google_groups if oidc is not None else None
    google_sa_key = (_secret_ref_value(google_groups.service_account_key_ref)
                     if google_groups is not None
                     and google_groups.service_account_key_ref is not None else
                     '')
    google_admin_email = (google_groups.admin_email if google_groups is not None
                          and google_groups.admin_email else '')
    temporary_password = (_secret_ref_value(password.temporary_password_ref)
                          if password is not None
                          and password.temporary_password_ref is not None else
                          '')

    return DirectConnectResolvedConfig(
        login_method=settings.login_method,
        allowed_return_origins=settings.allowed_return_origins,
        admin_emails=admin_emails,
        oidc=DirectConnectOIDCSeedConfig(
            issuer=oidc.issuer if oidc is not None and oidc.issuer else '',
            client_id=(oidc.client_id
                       if oidc is not None and oidc.client_id else ''),
            client_secret=client_secret,
            groups_claim=(oidc.groups_claim if oidc is not None
                          and oidc.groups_claim else 'groups'),
            google_sa_key=google_sa_key,
            google_admin_email=google_admin_email,
        ),
        local_admin_temp_password=temporary_password,
    )
