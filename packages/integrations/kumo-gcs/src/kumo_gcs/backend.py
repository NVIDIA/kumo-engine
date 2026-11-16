# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional

from kumo_core.sources.object_store_access import ObjectStoreAccess
from kumo_core.sources.object_store_backend import ObjectStoreSourceBackend
from kumo_core.sources.object_store_sessions import (
    ObjectStorePaths,
    ObjectStoreSessions,
)
from kumo_core.sources.principals import IdentityRef, Principal
from kumo_core.sources.types import ReadMode
from kumo_gcs.session import GCS_NAMESPACE, _gcs_session_for, _live_token
from kumoapi.data_source import (
    DataSourceType,
    GCSConnectorResourceConfig,
    GCSRootedConnectorTarget,
)
from kumoapi.gcs import (
    normalize_gcs_source_relative_path,
    normalize_gcs_source_uri,
)
from kumoapi.source_table import (
    GCSRelativeSource,
    GCSSourceTable,
    GCSURISource,
)
from pyarrow import fs


class GCSSessions(ObjectStoreSessions):
    r"""Resolving a connector and a table name to a reader over a live bucket.

    The source and sink backends share identity resolution. Sources use the
    SDK for requester-pays metadata and reads; sinks require native Arrow.

    Read off at *every* bind, and refreshed first where the credential says it
    has gone stale -- see :func:`_live_token`.

    Args:
        identity: A fallback identity, used only for connectors that name none
            themselves. Accepts an identity ID or the name it was created
            under. Unset is the common case, and the one the ambient
            credential chain serves.
        storage_namespace: The storage system this process reaches, recorded
            in every graph coordinate a table contributes. Defaults to
            Google Cloud Storage; name a compatible service explicitly, or its
            buckets share graph IDs with real GCS buckets of the same name.
    """

    namespace = GCS_NAMESPACE
    rooted = GCSRootedConnectorTarget
    resource_config = GCSConnectorResourceConfig
    paths = ObjectStorePaths(
        scheme='gs',
        provider='GCS',
        source_uri=normalize_gcs_source_uri,
        destination_uri=normalize_gcs_source_uri,
        destination_path=normalize_gcs_source_relative_path,
    )

    def _source_access(self, connector: Any, bucket: str,
                       region: Optional[str]) -> ObjectStoreAccess:
        from .source_access import source_access

        config = self._target_config(connector)
        principal = self._principal(connector)
        credentials = (_gcs_session_for(principal)
                       if principal is not None else None)
        if isinstance(credentials, fs.GcsFileSystem):
            if getattr(config, 'requester_pays_project', None) is not None:
                raise ValueError('A prebuilt GcsFileSystem cannot send '
                                 'requester-pays userProject.')
            return ObjectStoreAccess(credentials)
        if credentials is None:
            import google.auth

            from .session import _GCS_SCOPE

            credentials, project = google.auth.default(scopes=[_GCS_SCOPE])
        else:
            project = getattr(credentials, 'project_id', None)
        return source_access(
            credentials=credentials, project=project,
            requester_pays_project=getattr(config, 'requester_pays_project',
                                           None))

    def _filesystem(self, connector: Any, bucket: str,
                    region: Optional[str]) -> fs.GcsFileSystem:
        # A GCS bucket is global and carries no region, and the bucket is
        # named in every path rather than in the filesystem.
        del bucket, region
        # Arrow-backed sinks cannot send requester-pays userProject.
        config = self._target_config(connector)
        if getattr(config, 'requester_pays_project', None) is not None:
            raise ValueError(
                f"Connector {connector.id!r} names a requester_pays_project, "
                f"which the Arrow-backed sink does not support: "
                f"'pyarrow.fs.GcsFileSystem' has no way to send the "
                f"'userProject' a requester-pays bucket needs on every "
                f"request. Unset requester_pays_project, or read from a "
                f"bucket its owner pays for.")
        principal = self._principal(connector)
        session = (_gcs_session_for(principal)
                   if principal is not None else None)
        if isinstance(session, fs.GcsFileSystem):
            return session
        if session is None:
            return fs.GcsFileSystem()
        token, expiry = _live_token(session)
        return fs.GcsFileSystem(access_token=token,
                                credential_token_expiration=expiry)


class GCSSourceBackend(GCSSessions, ObjectStoreSourceBackend):
    r"""Resolves source tables by reading Google Cloud Storage in this
    process.

    Install with ``pip install 'kumoai-studio[gcs]'``. Without a named identity,
    reads use Application Default Credentials. Arrow serves metadata and
    the shared downloader handles native and SDK transfers.

    Args:
        identity: A fallback identity, used only for connectors that name none
            themselves. Accepts an :class:`~kumoai_studio.identity.Identity`, an
            identity ID, or the name it was created under.
        storage_namespace: The storage system this process reaches, recorded
            in every graph coordinate a GCS table contributes. Defaults to
            Google Cloud Storage.
        mode: A :class:`~kumo_core.sources.types.ReadMode` to force. An object
            store offers only ``streamed``, so this exists to assert that.

    Example:
        >>> import kumoai_studio as kumo  # doctest: +SKIP
        >>> from kumo_gcs import GCSSourceBackend  # doctest: +SKIP
        >>> kumo.set_backend(kumo.LocalBackend(  # doctest: +SKIP
        ...     sources={kumo.DataSourceType.GCS: GCSSourceBackend()}))
        >>> events = kumo.GCSConnector(  # doctest: +SKIP
        ...     name='events', root_dir='gs://company-data/events/')
    """

    source_type = DataSourceType.GCS

    relative_source = GCSRelativeSource
    uri_source = GCSURISource
    source_table = GCSSourceTable

    def __init__(
        self,
        identity: Optional[IdentityRef] = None,
        storage_namespace: Optional[str] = None,
        mode: Optional[ReadMode] = None,
    ) -> None:
        super().__init__(identity, storage_namespace)
        self.mode = mode

    def __repr__(self) -> str:
        return (f'{type(self).__name__}(identity={self._identity!r}, '
                f'storage_namespace={self.storage_namespace!r})')


def probe(principal: Principal) -> None:
    r"""Opens and drops one GCS connection with :obj:`principal`'s
    credentials.

    A named identity is resolved and its access token refreshed, which is the
    call that a revoked key, an expired assertion or an impersonation the
    caller may not perform fails on -- the same thing every bind does before
    signing, so a success here means a read would authenticate too.

    Ambient authentication resolves no credential of its own
    (:func:`_gcs_session_for` returns ``None`` for it), so what is exercised
    instead is what the ambient path actually signs with: constructing a
    :class:`pyarrow.fs.GcsFileSystem`, which resolves Application Default
    Credentials and fails when none are configured. Neither branch touches a
    bucket, so neither reports on the IAM grants over one.

    GCS opens a connection per call and caches nothing here, so there is no
    ``close`` beside this: a plugin that defines none says exactly that, and
    the identity area reads it that way.
    """

    session = _gcs_session_for(principal)
    if isinstance(session, fs.GcsFileSystem):
        return
    if session is None:
        fs.GcsFileSystem()
        return
    _live_token(session)
