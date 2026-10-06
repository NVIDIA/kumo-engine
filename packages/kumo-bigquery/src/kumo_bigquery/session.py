# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Turning a connector into BigQuery clients, and talking to them.

An identity resolves to ``google.auth`` credentials rather than to a client,
because BigQuery is reached through three clients and credentials are the one
thing all three are built from. This module is not lazy about the Google
client libraries the way the SDK's copy of it was: ``kumo_bigquery`` already
carries them as unconditional base dependencies (the platform reader needs
them too), so importing them here costs nothing that importing this package
at all did not already cost.
"""

import threading
from typing import Any, Dict, List, Optional, Tuple

import google.auth.transport.requests
from google.auth.credentials import Credentials
from google.cloud import bigquery, bigquery_storage, storage
from kumo_bigquery.auth import google_credentials
from kumo_core.sources.principals import IdentityRef, Principal, resolve
from kumoapi.data_source import (
    BigQueryConnectorResourceConfig,
    BigQueryIdentityConfig,
)
from kumoapi.plugin import SourceConnector

#: What an impersonated token is minted for. The platform uses the same scope
#: for BigQuery (``rest/config/identity.py``'s ``_google_scope``), so a local
#: read and a platform read reach the same APIs as the same principal.
_CLOUD_PLATFORM_SCOPE = 'https://www.googleapis.com/auth/cloud-platform'


def _target_config(
        connector: SourceConnector) -> BigQueryConnectorResourceConfig:
    r"""Narrows a connector's target to this provider's own.

    A host routes on ``connector.source_type``, so in practice this always
    holds. Checking it is what turns a host that got that wrong into a
    sentence naming what arrived, rather than an attribute error from
    somewhere further in -- the same reason the identity above narrows.
    """
    config = connector.config
    if not isinstance(config, BigQueryConnectorResourceConfig):
        raise ValueError(
            f"Connector {connector.id!r} points at a "
            f"{config.type.value} target, so it cannot be read as BigQuery.")
    return config


def _name(connector: SourceConnector, table_name: str) -> str:
    config = _target_config(connector)
    return f'{config.project_id}.{config.dataset_id}.{table_name}'


def _qualified(connector: SourceConnector, table_name: str) -> str:
    return f'`{_name(connector, table_name)}`'


def _billing_project(connector: SourceConnector) -> str:
    r"""A dataset is frequently shared read-only, so the project holding the
    data is one its readers cannot run a job in.
    """
    config = _target_config(connector)
    return config.billing_project_id or config.project_id


def _require_credentials(credentials: object) -> None:
    r"""Checked structurally rather than against
    ``google.auth.credentials.Credentials``, so refusing one is not itself
    what makes this module import Google.
    """
    if not (hasattr(credentials, 'refresh') and hasattr(credentials, 'token')):
        raise TypeError(
            f"A BigQuery identity must resolve to google.auth credentials; "
            f"got {type(credentials).__name__!r}.")


def _credentials_for(principal: Principal) -> Credentials:
    r"""Builds the Google credentials a BigQuery identity describes.

    The build itself is :func:`kumo_bigquery.auth.google_credentials`; what
    stays here is the fence, which has to name the area a caller reached for.
    """
    config = principal.config
    if not isinstance(config, BigQueryIdentityConfig):
        raise ValueError(
            f"Identity {principal.name!r} is a {config.type.value} "
            f"identity, so it cannot reach BigQuery.")
    return google_credentials(principal.name, config.authentication,
                              principal.credentials,
                              scope=_CLOUD_PLATFORM_SCOPE)


def probe(principal: Principal) -> None:
    r"""Proves an identity can authenticate.

    Half of the ``kumoai_studio.identities`` lifecycle. Deliberately not cached, and
    not routed through :data:`_clients`: this is asked when a caller wants to
    know whether the credentials work *now*.
    """
    credentials = _credentials_for(principal)
    _require_credentials(credentials)
    credentials.refresh(google.auth.transport.requests.Request())


def close(identity_name: Optional[str] = None) -> None:
    r"""Drops the cached clients built for :obj:`identity_name`, or all of
    them.

    The other half of the ``kumoai_studio.identities`` lifecycle. A client built from
    :data:`_clients` carries the credentials it was resolved from, so
    rotating or deleting the identity underneath it must drop the client too
    -- otherwise a rotated credential's caller would keep talking to BigQuery
    as the identity that no longer exists, through a client already built
    from the old material. There is nothing to close on a BigQuery client
    itself; forgetting it is enough, and the next call rebuilds one from
    whatever the identity now resolves to.
    """
    _clients.close(identity_name)


def _query_client(credentials: Credentials, *, project: str,
                  location: str) -> Any:
    r"""``location`` is the connector's, because a job submitted without one
    is resolved against the caller's default region and fails on a dataset
    that lives elsewhere.
    """
    return bigquery.Client(credentials=credentials, project=project,
                           location=location)


def _read_client(credentials: Credentials) -> Any:
    return bigquery_storage.BigQueryReadClient(credentials=credentials)


def _storage_client(credentials: Credentials, *, project: str) -> Any:
    return storage.Client(credentials=credentials, project=project)


class _ClientCache:
    r"""The three BigQuery clients :func:`_query_client`, :func:`_read_client`
    and :func:`_storage_client` build, by identity and target.

    Module-level rather than per backend, for the reason
    :class:`kumo_snowflake.session._SessionCache` is: a reader and a writer
    over the same connector must not resolve to two different clients, and a
    default backend instance is thrown away and rebuilt more often than the
    identity it authenticates as changes.

    Keyed on ``(identity, project, location)`` rather than holding the
    resolved credentials object the way an earlier version of this module
    did: caching the credentials alone and rebuilding a client from them on
    every call is not caching the client at all, and a client is what this
    module is actually asked for on every read.

    Calls the three module-level builders by name rather than inlining the
    client construction, so a caller can still replace one of them (a test
    stands in a fake client this way) and have every path that resolves a
    client -- cached or not -- see the replacement.
    """
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._query: Dict[Tuple[str, str, str], Any] = {}
        self._read: Dict[str, Any] = {}
        self._storage: Dict[Tuple[str, str], Any] = {}

    def query_client(self, principal: Principal, *, project: str,
                     location: str) -> Any:
        key = (principal.name, project, location)
        with self._lock:
            client = self._query.get(key)
            if client is None:
                credentials = _credentials_for(principal)
                _require_credentials(credentials)
                client = _query_client(credentials, project=project,
                                       location=location)
                self._query[key] = client
            return client

    def read_client(self, principal: Principal) -> Any:
        with self._lock:
            client = self._read.get(principal.name)
            if client is None:
                credentials = _credentials_for(principal)
                _require_credentials(credentials)
                client = _read_client(credentials)
                self._read[principal.name] = client
            return client

    def storage_client(self, principal: Principal, *, project: str) -> Any:
        key = (principal.name, project)
        with self._lock:
            client = self._storage.get(key)
            if client is None:
                credentials = _credentials_for(principal)
                _require_credentials(credentials)
                client = _storage_client(credentials, project=project)
                self._storage[key] = client
            return client

    def close(self, name: Optional[str] = None) -> None:
        with self._lock:
            for cache in (self._query, self._storage):
                for key in [
                        key for key in cache if name is None or key[0] == name
                ]:
                    del cache[key]
            for key in [
                    key for key in self._read if name is None or key == name
            ]:
                del self._read[key]


_clients = _ClientCache()


def _rows(client: Any, sql: str, job_config: Any = None) -> List[Any]:
    r"""Rows are normalized to tuples so that no client type escapes this
    function. A job configuration is passed only when there is one, so a read
    of a physical table asks for exactly what it always did.
    """
    from kumo_bigquery.sql import query_result

    return [tuple(row) for row in query_result(client.query, sql, job_config)]


class _BigQuerySessions:
    r"""Resolving a connector to the BigQuery clients built from its
    identity.

    Args:
        identity: A fallback identity, used only for connectors that name none
            themselves. Accepts the identity object the SDK handed back, an
            identity ID, or the name it was created under.
    """

    #: How a subclass spells "supply a default identity" in its own terms. The
    #: message resolution raises is the only place a caller learns the
    #: constructor to reach for.
    _default_hint = ''

    def __init__(self, identity: Optional[IdentityRef] = None) -> None:
        self._identity = identity

    def _credentials(self, connector: SourceConnector) -> Credentials:
        r"""Raw credentials for a connector, for the one caller that needs
        them directly rather than a client built from them:
        ``backend.py``'s reader construction hands a credentials object
        straight to ``BigQuerySourceReader``.

        Defined here rather than inlined at the call site so that a test
        replacing ``_credentials_for`` at the module level -- see
        ``test/_identities.py``'s ``use_session`` -- is seen by every path
        that resolves credentials, including this one. A module that
        imported ``_credentials_for`` directly would keep its own binding of
        the original function and miss the replacement.
        """
        credentials = _credentials_for(self._principal(connector))
        _require_credentials(credentials)
        return credentials

    def _client(self, connector: SourceConnector) -> Any:
        return _clients.query_client(
            self._principal(connector), project=_billing_project(connector),
            location=_target_config(connector).location)

    def _principal(self, connector: SourceConnector) -> Principal:
        return resolve(
            (getattr(connector, 'identity', None),
             getattr(connector, 'identity_id', None), self._identity),
            connector=connector,
            config_example='BigQueryIdentityConfig(...), credentials=...',
            connector_example='BigQueryConnector(identity=identity, ...)',
            fallback=(f'A default for sources that name none is also '
                      f'accepted, as {self._default_hint}.'),
        )
