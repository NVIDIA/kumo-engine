# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GCS identity binding, imported lazily by provider operations."""

from typing import TYPE_CHECKING, Any, Tuple, Union

from kumo_core.sources.principals import Principal
from kumoapi.data_source import GCSIdentityConfig, GoogleAmbientAuthentication

if TYPE_CHECKING:
    from pyarrow.fs import GcsFileSystem

#: What a GCS identity resolves to: a ready filesystem, a ``google.auth``
#: credential whose live access token is read off it, or ``None`` for the
#: ambient chain pyarrow resolves by itself.
GCSSession = Union['GcsFileSystem', Any]

#: What an impersonated GCS token is minted for. The platform scopes a GCS
#: identity the same way -- ``_GOOGLE_GCS_SCOPE`` in ``rest/config/identity``
#: -- so a local write and a platform write reach the bucket as the same
#: principal with the same grants.
_GCS_SCOPE = 'https://www.googleapis.com/auth/devstorage.read_write'

#: The storage namespace of Google Cloud Storage. A coordinate records this
#: rather than the principal that read it, so rotating a credential does not
#: re-identify the data. Kept equal to
#: ``kumoai_studio.backend.local.portable.GCS_STORAGE_NAMESPACE`` by a test; spelled
#: out rather than imported from there, which a graph conversion must not
#: need this package installed for.
GCS_NAMESPACE = 'google-gcs'


def _gcs_session_for(principal: Principal) -> GCSSession:
    """Return named credentials, or None for caller-owned ambient resolution."""
    config = principal.config
    if not isinstance(config, GCSIdentityConfig):
        raise ValueError(
            f"Identity {principal.name!r} is a {config.type.value} "
            f"identity, so it cannot sign for GCS.")
    if isinstance(config.authentication, GoogleAmbientAuthentication):
        return None
    from .auth import google_credentials
    return google_credentials(principal.name, config.authentication,
                              principal.credentials, scope=_GCS_SCOPE)


def _live_token(session: GCSSession) -> Tuple[str, Any]:
    """Refresh before binding an Arrow metadata filesystem."""
    # `valid` is `token is not None and not expired`, and google-auth reports
    # expiry early by its own refresh threshold. Only a credential that says
    # so is refreshed; anything that does not carry `valid` is left alone and
    # read as it is, which is what a caller's own duck type gets.
    if getattr(session, 'valid', None) is False:
        refresh = getattr(session, 'refresh', None)
        if not callable(refresh):
            raise TypeError(
                f"The GCS credential has expired and cannot refresh itself: "
                f"{type(session).__name__!r} has no 'refresh'. Create the "
                f"identity with a GCSIdentityConfig whose authentication "
                f"google-auth can refresh, or bind a "
                f"'pyarrow.fs.GcsFileSystem' built to refresh on its own.")
        try:
            from google.auth.transport.requests import Request
        except ImportError as error:  # pragma: no cover - environment
            raise ImportError(
                "Refreshing an expired GCS access token needs "
                "'google-auth[requests]', which is not installed. Install it, "
                "or use a GCS identity whose credential arrives already "
                "refreshed.") from error
        refresh(Request())
    token = getattr(session, 'token', None)
    expiry = getattr(session, 'expiry', None)
    if not token or expiry is None:
        raise TypeError(
            f"A GCS session must be a 'pyarrow.fs.GcsFileSystem' or a "
            f"refreshed google.auth credential carrying 'token' and "
            f"'expiry'; got {type(session).__name__!r}. Leaving the identity "
            f"unset is also supported: the reader then signs with "
            f"Application Default Credentials.")
    return token, expiry
