# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from kumoapi.data_source import (
    GoogleIdentityAuthentication,
    GoogleServiceAccountKeyAuthentication,
    GoogleServiceAccountKeyCredentials,
    GoogleWorkloadIdentityAuthentication,
    IdentityCredentials,
)

if TYPE_CHECKING:
    from google.auth.credentials import Credentials

# Keep this implementation in sync with kumo_gcs.auth. The provider packages
# deliberately own their small Google-auth adapters instead of making
# provider-neutral kumo-core depend on the Google SDK.


def service_account_info(
    authentication: GoogleServiceAccountKeyAuthentication,
    *,
    private_key_id: str,
    private_key: str,
) -> dict[str, str]:
    return {
        'type': 'service_account',
        'project_id': authentication.project_id,
        'client_email': authentication.client_email,
        'client_id': authentication.client_id,
        'auth_uri': authentication.auth_uri,
        'token_uri': authentication.token_uri,
        'universe_domain': authentication.universe_domain,
        'private_key_id': private_key_id,
        'private_key': private_key,
    }


def google_credentials(
    name: str,
    authentication: GoogleIdentityAuthentication,
    credentials: IdentityCredentials | None,
    *,
    scope: str,
) -> Credentials:
    import google.auth
    from google.auth import impersonated_credentials
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    if isinstance(authentication, GoogleWorkloadIdentityAuthentication):
        source, _ = google.auth.default(scopes=[scope])
        exchanged = impersonated_credentials.Credentials(
            source_credentials=source,
            target_principal=authentication.service_account_email,
            target_scopes=[scope],
        )
        # Report impersonation failures against the identity before any read.
        try:
            exchanged.refresh(Request())
        except Exception as error:
            raise ValueError(
                f"Identity {name!r} could not impersonate "
                f"{authentication.service_account_email!r}: {error}. The "
                f"ambient principal needs roles/iam.serviceAccountTokenCreator "
                f"on it.") from None
        return exchanged

    if not isinstance(authentication, GoogleServiceAccountKeyAuthentication):
        return google.auth.default()[0]

    if not isinstance(credentials, GoogleServiceAccountKeyCredentials):
        raise ValueError(
            f"Identity {name!r} authenticates with a Google service-account "
            f"key but holds {type(credentials).__name__} credentials.")
    return service_account.Credentials.from_service_account_info(
        service_account_info(
            authentication,
            private_key_id=credentials.private_key_id.get_secret_value(),
            private_key=credentials.private_key.get_secret_value()))


__all__ = ['google_credentials']
