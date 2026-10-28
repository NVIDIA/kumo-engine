# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Turning a connector into a live Databricks SQL connection.

An identity resolves to a :class:`~databricks.sdk.WorkspaceClient`, not to a
SQL connection, deliberately: ``databricks.sql`` fixes its CloudFetch download
pool when the connection is built, so handing back a ready-made one would pin
that pool at a constant no caller could correct on a cgroup-limited host.

A connection is opened per call and closed after it: one cached for the life of
a process would outlive the identity it was minted from.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, List, Optional

from databricks import sql
from databricks.sdk import WorkspaceClient
from databricks.sql.client import Connection
from kumo_core.sources.principals import IdentityRef, Principal, resolve
from kumoapi.data_source import (
    DatabricksAuthenticationKind,
    DatabricksConnectorResourceConfig,
    DatabricksIdentityConfig,
    DatabricksOAuthCredentials,
)
from kumoapi.plugin import SourceConnector
from pydantic import SecretStr

#: What this module builds and hands upward.
DatabricksSession = WorkspaceClient

#: A warehouse is addressed by its own ID alone; a cluster is addressed
#: through the workspace it belongs to, which only the workspace can report.
_WAREHOUSE_PATH = '/sql/1.0/warehouses/{compute}'
_CLUSTER_PATH = '/sql/protocolv1/o/{workspace}/{compute}'


def _target_config(
        connector: SourceConnector) -> DatabricksConnectorResourceConfig:
    r"""Narrows a connector's target to this provider's own.

    A host routes on ``connector.source_type``, so in practice this always
    holds. Checking it is what turns a host that got that wrong into a
    sentence naming what arrived, rather than an attribute error from
    somewhere further in -- the same reason the identity above narrows.
    """
    config = connector.config
    if not isinstance(config, DatabricksConnectorResourceConfig):
        raise ValueError(
            f"Connector {connector.id!r} points at a "
            f"{config.type.value} target, so it cannot be read as Databricks.")
    return config


def _require_workspace_client(client: object) -> None:
    r"""This module builds its own client, so a failure here means a caller
    reached in with something else, which would otherwise fail as a missing
    attribute somewhere inside ``databricks.sql``.
    """
    if not isinstance(client, WorkspaceClient):
        raise TypeError(f"A Databricks session must be a 'databricks.sdk."
                        f"WorkspaceClient'; got {type(client).__name__!r}.")


def _client_for(principal: Principal) -> WorkspaceClient:
    r"""Builds the workspace client an identity's credentials describe.

    An OAuth service principal is the common case; the SDK performs the
    token exchange and refresh itself. `AMBIENT` carries no
    credentials to pass at all -- a bare `WorkspaceClient()` is exactly what
    lets the SDK's own default auth chain resolve the environment (a
    notebook's execution context, `DATABRICKS_HOST`/`DATABRICKS_TOKEN`, ...),
    the same way `kumo_snowflake.session._authentication` reads a Snowpark
    Container Services-mounted token for
    `SnowflakeAuthenticationKind.OAUTH_TOKEN_FILE`.
    """
    config = principal.config
    credentials = principal.credentials
    if not isinstance(config, DatabricksIdentityConfig):
        raise ValueError(
            f"Identity {principal.name!r} is a {config.type.value} "
            f"identity, so it cannot open a Databricks connection.")
    if config.authentication is DatabricksAuthenticationKind.AMBIENT:
        return WorkspaceClient()
    if not isinstance(credentials, DatabricksOAuthCredentials):
        raise ValueError(
            f"Identity {principal.name!r} holds "
            f"{type(credentials).__name__} credentials; Databricks "
            f"authenticates a service principal with an OAuth secret.")
    # Explicit: a DATABRICKS_TOKEN another thread lends MLflow must not
    # make this client's configuration ambiguous.
    return WorkspaceClient(
        host=config.host,
        client_id=config.client_id,
        client_secret=_secret(credentials.client_secret),
        auth_type='oauth-m2m',
    )


def _secret(value: Optional[SecretStr]) -> Optional[str]:
    r"""Unwraps a ``SecretStr`` at the one point the material is used."""
    return None if value is None else value.get_secret_value()


def probe(principal: Principal) -> None:
    r"""Proves an identity can authenticate.

    The ``kumoai_studio.identities`` lifecycle, in full: this module opens a
    connection per call and closes it after, so there is nothing cached for a
    ``close`` to drop and it defines none.
    """
    _client_for(principal).config.authenticate()


def _rows(connection: Connection, statement: str) -> List[tuple]:
    r"""Rows are normalized to tuples so that no driver type escapes this
    function.
    """
    cursor = connection.cursor()
    try:
        cursor.execute(statement)
        return [tuple(row) for row in cursor.fetchall()]
    finally:
        cursor.close()


class _DatabricksSessions:
    r"""Resolving a connector to a live Databricks SQL connection.

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

    def _client(self, connector: SourceConnector) -> DatabricksSession:
        r"""Built per call because it holds the credentials it was resolved
        from and would outlive a rotation that replaced them.
        """
        client = _client_for(self._principal(connector))
        _require_workspace_client(client)
        return client

    @contextmanager
    def _connected(
        self,
        connector: SourceConnector,
        staging_from: Optional[str] = None,
    ) -> Iterator[Connection]:
        r"""``staging_from`` is the one directory ``PUT`` may read local files
        out of. The driver refuses a volume-ingestion command on a connection
        that names none, and it is fixed when the connection is built, so the
        sink opens its own around a scratch directory it owns.
        """
        from kumo_core.sources.cpu import download_workers

        client = self._client(connector)
        config = _target_config(connector)
        connection = sql.connect(
            server_hostname=client.config.hostname,
            http_path=_http_path(client, connector),
            # The SDK's resolved authentication, whatever form it took, as the
            # header factory the SQL client asks a provider for.
            credentials_provider=lambda: client.config.authenticate,
            catalog=config.catalog,
            schema=config.schema_name,
            staging_allowed_local_path=staging_from,
            # Never re-read after the connection is built, so a connection
            # opened anywhere else pins the pool to the client's constant 10.
            max_download_threads=download_workers(),
        )
        try:
            yield connection
        finally:
            connection.close()

    def _principal(self, connector: SourceConnector) -> Principal:
        return resolve(
            (getattr(connector, 'identity', None),
             getattr(connector, 'identity_id', None), self._identity),
            connector=connector,
            config_example='DatabricksIdentityConfig(...), credentials=...',
            connector_example='DatabricksConnector(identity=identity, ...)',
            fallback=(f'A default for sources that name none is also '
                      f'accepted, as {self._default_hint}.'),
        )

    def table_location(self, connector: SourceConnector,
                       table_name: str) -> str:
        """Fully qualified name of ``table_name`` in the connector's target."""
        return self._qualified(connector, self._bare(connector, table_name))

    @staticmethod
    def _qualified(connector: SourceConnector,
                   table_name: Optional[str] = None) -> str:
        r"""Borrowed from the connector package rather than restated, because
        backtick escaping is what stands between a catalog-derived name and the
        statement it is interpolated into.
        """
        from kumo_databricks import sql as dbx
        config = _target_config(connector)
        parts = (config.catalog, config.schema_name)
        if table_name is not None:
            parts += (table_name, )
        return dbx.qualified_name(*parts)

    @staticmethod
    def _bare(connector: SourceConnector, table_name: str) -> str:
        r"""Databricks is the one vendor whose ``SourceTable`` has nowhere to
        put the catalog and schema but inside the table field, so a portable
        graph carries ``catalog.schema.table`` there. The reader takes catalog
        and schema on itself, so the qualification is stripped once, here.

        One that disagrees with the connector's is refused rather than ignored:
        reading the connector's table anyway would answer a question nobody
        asked, with no sign it had happened.
        """
        from kumo_databricks.sql import normalize_table_coordinate

        config = _target_config(connector)
        _, table = normalize_table_coordinate(
            table_name,
            catalog=config.catalog,
            schema=config.schema_name,
        )
        return table


def _http_path(client: DatabricksSession, connector: SourceConnector) -> str:
    config = _target_config(connector)
    if config.warehouse_id:
        return _WAREHOUSE_PATH.format(compute=config.warehouse_id)
    if not config.cluster_id:
        # `DatabricksConnectorResourceConfig` requires exactly one of the two,
        # so a connector naming neither cannot have been constructed. Kept for
        # the day that rule is relaxed: saying so beats formatting `None` into
        # an HTTP path and letting the workspace reject it.
        raise ValueError(
            f"Connector {connector.id!r} names neither a warehouse nor a "
            f"cluster to run on. Set exactly one of 'warehouse_id' and "
            f"'cluster_id'.")
    return _CLUSTER_PATH.format(workspace=client.get_workspace_id(),
                                compute=config.cluster_id)
