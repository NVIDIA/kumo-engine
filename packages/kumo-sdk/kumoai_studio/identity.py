# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from kumoapi.data_source import (
    IdentityConfig,
    IdentityResponse,
    IdentityType,
    S3AccessKeyCredentials,
)
from pydantic import SecretStr


def s3_access_key_credentials(
    *,
    access_key_id: str,
    secret_access_key: str,
    session_token: Optional[str] = None,
    ca_certificate: Optional[str] = None,
) -> S3AccessKeyCredentials:
    r"""Build protected credentials for an S3-compatible identity.

    Secret values use the same plain-string SDK boundary as other SDK
    credentials. They are wrapped immediately so representations remain
    redacted and are revealed only by the existing identity request
    serializer.
    """
    return S3AccessKeyCredentials(
        access_key_id=SecretStr(access_key_id),
        secret_access_key=SecretStr(secret_access_key),
        session_token=(SecretStr(session_token)
                       if session_token is not None else None),
        ca_certificate=(SecretStr(ca_certificate)
                        if ca_certificate is not None else None),
    )


@dataclass(frozen=True)
class Identity:
    r"""Immutable, credential-free summary of an external principal.

    Administrators create identities and bind them to groups -- through the UI
    or, with an administrator API key, through the SDK (see
    :func:`kumoai_studio.create_identity`). Project members may use a bound identity
    to create a connector. In every case the SDK receives only safe provider
    configuration and never the stored credentials: ``credential_configured``
    and ``credentials_last_rotated_at`` expose credential status only.

    BigQuery and GCS identities expose one safe
    ``config.authentication.mode``: ``"workload_identity"`` or
    ``"service_account_key"``. Workload identity is preferred because Kumo
    impersonates the configured service-account email without storing a
    customer key. Key mode exposes only non-secret service-account metadata;
    its private key and key ID remain encrypted, administrator-only material
    on the server. Neither mode returns credentials or provider tokens.

    Pass an :class:`Identity` returned by :func:`kumoai_studio.get_identity` to a
    connector for explicit selection. Omitting ``identity`` asks the server to
    find the one available identity that can access the exact target.
    Connector constructors accept only this safe identity reference and target
    configuration; they never accept Google credential fields.

    Args:
        id: Immutable identity ID.
        slug: Immutable short ID accepted by :func:`kumoai_studio.get_identity`.
        name: Administrator-defined display name.
        description: Safe human-readable description.
        type: Provider type; it must match the connector type.
        config: Safe provider-native principal configuration. For BigQuery and
            GCS, inspect ``config.authentication.mode`` and the corresponding
            safe service-account email and metadata.
        credential_configured: Whether the identity has the authentication
            configuration it requires. Workload mode reports ``True`` because
            it requires no stored customer secret; this does not prove that a
            live exchange or impersonation will succeed. Use connector or
            identity connection testing to verify current access.
        credentials_last_rotated_at: Time of the latest credential rotation,
            if credentials are stored for this identity.
    """

    id: str
    slug: str
    name: str
    description: str
    type: IdentityType
    config: IdentityConfig | None = None
    credential_configured: bool = False
    credentials_last_rotated_at: datetime | None = None

    @classmethod
    def _from_response(cls, response: IdentityResponse) -> 'Identity':
        return cls(
            id=str(response.id),
            slug=response.slug,
            name=response.display_name,
            description=response.description,
            type=response.config.type,
            config=response.config,
            credential_configured=response.credential_status.configured,
            credentials_last_rotated_at=(
                response.credential_status.last_rotated_at),
        )
