# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from dataclasses import fields, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from io_utils import kumo_dataclass
from statedb.config.common import DBEntry

DirectConnectUserRole = Literal["admin", "user"]
DirectConnectPasswordLoginStatus = Literal["ok", "password_reset_required"]

# These fields intentionally remain plain strings: io_utils.json rejects
# pydantic.SecretStr during normal state persistence unless insecure=True, while
# these values must round-trip through JsonRepository as their stored strings.
_REDACTED = "<redacted>"


def _redacted_dataclass_dict(
    value: object,
    redacted_fields: set[str],
) -> dict[str, Any]:
    if not is_dataclass(value):
        raise TypeError(f"Expected dataclass instance, got {type(value)!r}")
    return {
        field.name:
        _REDACTED if field.name in redacted_fields else getattr(
            value, field.name)
        for field in fields(value)
    }


def _redacted_dataclass_repr(
    value: object,
    redacted_fields: set[str],
) -> str:
    fields_repr = ", ".join(f"{name}={field_value!r}"
                            for name, field_value in _redacted_dataclass_dict(
                                value, redacted_fields).items())
    return f"{type(value).__name__}({fields_repr})"


_DIRECT_CONNECT_AUTH_CONFIG_REDACTED_FIELDS = {
    "encrypted_client_secret",
    "encrypted_google_sa_key",
}
_DIRECT_CONNECT_IDENTITY_LINK_REDACTED_FIELDS = {
    "encrypted_refresh_token",
}
_DIRECT_CONNECT_PASSWORD_CREDENTIAL_REDACTED_FIELDS = {
    "password_hash_b64",
    "salt_b64",
}


@kumo_dataclass(repr=False)
class DirectConnectAuthConfigRecord(DBEntry):
    """Stored customer Direct Connect OIDC configuration."""

    issuer: str
    client_id: str
    encrypted_client_secret: str
    groups_claim: str = "groups"
    admin_emails: List[str] = Field(default_factory=list)
    enabled: bool = True
    # Google Workspace Directory API lookup; empty means "read groups from
    # the ID-token claim" (the default for every other IdP).
    encrypted_google_sa_key: str = ""
    google_admin_email: str = ""

    def __repr__(self) -> str:
        return _redacted_dataclass_repr(
            self,
            _DIRECT_CONNECT_AUTH_CONFIG_REDACTED_FIELDS,
        )

    __str__ = __repr__

    def to_log_dict(self) -> dict[str, Any]:
        """Return a log-safe view; persistence must keep using JSON helpers."""
        return _redacted_dataclass_dict(
            self,
            _DIRECT_CONNECT_AUTH_CONFIG_REDACTED_FIELDS,
        )


@kumo_dataclass
class DirectConnectUserRecord(DBEntry):
    """Stored Kumo user created from a Direct Connect IdP login."""

    email: str
    display_name: str = ""
    role: DirectConnectUserRole = "user"
    disabled: bool = False


@kumo_dataclass(repr=False)
class DirectConnectIdentityLinkRecord(DBEntry):
    """Stored Direct Connect IdP issuer/subject lookup."""

    user_id: str
    encrypted_refresh_token: str

    def __repr__(self) -> str:
        return _redacted_dataclass_repr(
            self,
            _DIRECT_CONNECT_IDENTITY_LINK_REDACTED_FIELDS,
        )

    __str__ = __repr__

    def to_log_dict(self) -> dict[str, Any]:
        return _redacted_dataclass_dict(
            self,
            _DIRECT_CONNECT_IDENTITY_LINK_REDACTED_FIELDS,
        )


@kumo_dataclass
class DirectConnectSessionRecord(DBEntry):
    """Stored UI session; its StateDB key is a one-way token digest."""

    user_id: str
    expires_time: datetime


@kumo_dataclass
class DirectConnectUserCreateRequest:
    email: str
    display_name: str = ""
    role: DirectConnectUserRole = "user"
    disabled: bool = False
    password: Optional[str] = None


@kumo_dataclass
class DirectConnectUserUpdateRequest:
    role: Optional[DirectConnectUserRole] = None
    disabled: Optional[bool] = None


@kumo_dataclass
class DirectConnectUserResponse:
    user_id: str
    email: str
    display_name: str
    role: DirectConnectUserRole
    disabled: bool


@kumo_dataclass(repr=False)
class DirectConnectPasswordCredentialRecord(DBEntry):
    """Stored local password verifier for a Direct Connect user."""

    password_hash_b64: str
    salt_b64: str
    # scrypt cost parameters (n/r/p/dklen) used to derive this hash. Persisted
    # so the cost factor can be raised in future without invalidating
    # already-stored hashes: verification uses the params recorded here.
    hash_params: Dict[str, int]
    hash_algorithm: str = "scrypt"
    must_reset_password: bool = True

    def __repr__(self) -> str:
        return _redacted_dataclass_repr(
            self,
            _DIRECT_CONNECT_PASSWORD_CREDENTIAL_REDACTED_FIELDS,
        )

    __str__ = __repr__

    def to_log_dict(self) -> dict[str, Any]:
        """Return a log-safe view; persistence must keep using JSON helpers."""
        return _redacted_dataclass_dict(
            self,
            _DIRECT_CONNECT_PASSWORD_CREDENTIAL_REDACTED_FIELDS,
        )


@kumo_dataclass
class DirectConnectPasswordResetRecord(DBEntry):
    """Stored first-login password reset token state."""

    user_id: str
    expires_time: datetime
    used: bool = False
    credential_tag: Optional[str] = None


def direct_connect_password_credential_tag(
    credential: DirectConnectPasswordCredentialRecord | None,
) -> Optional[str]:
    if credential is None:
        return None
    material = "|".join((
        credential.hash_algorithm,
        credential.salt_b64,
        credential.password_hash_b64,
        str(bool(credential.must_reset_password)),
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@kumo_dataclass
class DirectConnectPasswordLoginAttemptRecord(DBEntry):
    """Stored failed password-login attempts for rate limiting."""

    first_failed_time: datetime
    failure_count: int = 0
    locked_until: Optional[datetime] = None


@kumo_dataclass
class DirectConnectSetPasswordRequest:
    password: str


@kumo_dataclass
class DirectConnectPasswordLoginRequest:
    email: str
    password: str


@kumo_dataclass
class DirectConnectPasswordResetRequest:
    reset_token: str
    new_password: str


@kumo_dataclass
class DirectConnectPasswordLoginResponse:
    status: DirectConnectPasswordLoginStatus
    reset_token: Optional[str] = None


@kumo_dataclass
class DirectConnectMeResponse:
    user_id: str
    email: str
    display_name: str
    role: DirectConnectUserRole
    permissions: List[str] = Field(default_factory=list)
    kumo_group_ids: List[str] = Field(default_factory=list)
    project_ids: List[str] = Field(default_factory=list)
