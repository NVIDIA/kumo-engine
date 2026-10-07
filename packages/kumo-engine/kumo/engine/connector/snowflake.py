# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""`SnowflakeConnector`, mirroring the argument list of the cloud SDK's
`kumoai.connector.snowflake_connector.SnowflakeConnector`, minus everything
that exists only to round-trip a server-side connector registry (`name`,
`description`, `project_id`, `_response`). See `kumo.engine.connector.base` for why
that machinery is dropped rather than ported.
"""

import os
from functools import cached_property
from pathlib import Path
from typing import Dict, Optional

from kumoapi.data_source import (
    DataSourceType,
    SnowflakeConnectorResourceConfig,
)
from kumoapi.plugin import ArtifactSink, SourceBackend
from kumoapi.source_table import SnowflakeSourceTable
from typing_extensions import Self

from kumo.engine.connector._extras import import_provider
from kumo.engine.connector.base import Connector
from kumo.engine.connector.identity import SnowflakeIdentity

#: Warehouse/database/schema environment variables Snowpark Container
#: Services exports alongside the account/host `kumo_snowflake.spcs` already
#: reads (https://docs.snowflake.com/en/developer-guide/snowpark-container-
#: services/additional-considerations-services-jobs#label-spcs-env-variables).
#: Not defined in `kumo_snowflake.spcs`: that module builds an *identity*
#: (account/host/token only), and has no notion of a query target.
_SPCS_WAREHOUSE_ENV = "SNOWFLAKE_WAREHOUSE"
_SPCS_DATABASE_ENV = "SNOWFLAKE_DATABASE"
_SPCS_SCHEMA_ENV = "SNOWFLAKE_SCHEMA"


class SnowflakeConnector(Connector):
    r"""Connect directly to a Snowflake database and schema.

    Reads through `kumo_snowflake.SnowflakeSourceBackend` -- install the
    `snowflake` extra of this package to use it.

    Args:
        warehouse: Snowflake virtual warehouse used for queries.
        database: Snowflake database containing the target schema.
        schema_name: Schema within `database`.
        identity: A `SnowflakeIdentity`, passed directly. `None` defers to
            `kumo_snowflake`'s own ambient-session resolution.
        role: Optional Snowflake role to activate for this target.
    """
    def __init__(
        self,
        *,
        warehouse: str,
        database: str,
        schema_name: str,
        identity: Optional[SnowflakeIdentity] = None,
        role: Optional[str] = None,
    ) -> None:
        super().__init__(identity=identity)
        self._config = SnowflakeConnectorResourceConfig(
            role=role,
            warehouse=warehouse,
            database=database,
            schema_name=schema_name,
        )
        self.role = self._config.role
        self.warehouse = self._config.warehouse
        self.database = self._config.database
        self.schema_name = self._config.schema_name

    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.SNOWFLAKE

    @property
    def config(self) -> SnowflakeConnectorResourceConfig:
        return self._config

    @cached_property
    def source_backend(self) -> SourceBackend:
        # Lazily imported: kumo_snowflake is only installed with the
        # `snowflake` extra, and this package has plain `kumo.engine.Connector`
        # subclasses for every provider whether or not that extra is
        # installed.
        SnowflakeSourceBackend = import_provider(
            "kumo_snowflake.backend", "snowflake").SnowflakeSourceBackend
        return SnowflakeSourceBackend(identity=self.identity)

    @cached_property
    def sink_backend(self) -> ArtifactSink:
        r"""The `kumoapi.plugin.ArtifactSink`/`ModelPublishingSink`/
        `ModelFetchingSink`/`ModelListingSink` implementation this connector
        writes/publishes through -- backs `Trainer.fit(..., export_config=...)`
        and `Trainer.load_from_connector(...)`.
        """
        SnowflakeSinkBackend = import_provider("kumo_snowflake.sink",
                                               "snowflake").SnowflakeSinkBackend
        return SnowflakeSinkBackend(identity=self.identity)

    def source_table_for(self, table_name: str) -> SnowflakeSourceTable:
        return SnowflakeSourceTable(
            snowflake_connector_id=self.id,
            database=self.database,
            schema_name=self.schema_name,
            table=table_name,
        )

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(database={self.database!r}, "
                f"schema_name={self.schema_name!r})")

    @classmethod
    def from_connection_toml(
        cls,
        *,
        connection_name: Optional[str] = None,
        warehouse: Optional[str] = None,
        database: Optional[str] = None,
        schema_name: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Self:
        r"""Builds a connector (and its `SnowflakeIdentity`) from a
        `connections.toml` entry -- Snowflake's own developer configuration,
        and the file most people already have working before they reach
        Kumo. Mirrors the cloud SDK's `SnowflakeConnector.
        from_connection_toml`, minus the parts that only exist to round-trip
        a server-side identity registry (`name`, `identity_name`, `replace`,
        `description`): there is no registry here, so the entry's
        credentials go straight into a `SnowflakeIdentity` held by the
        returned connector, the same way any other `identity=` would.

        Args:
            connection_name: entry to read. `None` takes the file's
                `default_connection_name`.
            warehouse: overrides the entry's `warehouse`. Required when the
                entry names none, since a connector must name one.
            database: overrides the entry's `database`.
            schema_name: overrides the entry's `schema`.
            role: overrides the entry's `role`.

        Raises:
            ValueError: if the entry is absent, names neither a password
                nor a private key file, or leaves a required target field
                unset with no override.
        """
        # Snowflake's provider package owns its driver's configuration API;
        # this facade only turns the selected entry into Kumo objects.
        connection_toml_entry = import_provider(
            "kumo_snowflake.config", "snowflake").connection_toml_entry

        entry = connection_toml_entry(connection_name)
        warehouse = warehouse or entry.get('warehouse')
        database = database or entry.get('database')
        schema_name = schema_name or entry.get('schema')
        missing = sorted(field for field, value in (
            ('warehouse', warehouse),
            ('database', database),
            ('schema_name', schema_name),
        ) if not value)
        if missing:
            raise ValueError(
                f"connection {connection_name or 'default'!r} names no "
                f"{', '.join(missing)}, and none was passed. A connector "
                f"addresses one exact target, so it cannot be left to the "
                f"session's own default.")

        identity = _connections_toml_identity(entry)
        assert warehouse and database and schema_name
        return cls(warehouse=warehouse, database=database,
                   schema_name=schema_name, identity=identity, role=role
                   or entry.get('role'))

    @classmethod
    def from_spcs_env(
        cls,
        *,
        warehouse: Optional[str] = None,
        database: Optional[str] = None,
        schema_name: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Self:
        r"""Builds a connector (and its `SnowflakeIdentity`) from the
        Snowpark Container Services environment this process is already
        running in: the account/host Snowflake exports, and the OAuth token
        it mounts on the container's filesystem. Similar to `kumo_snowflake.
        spcs.spcs_identity_create_args`, minus the server-side identity
        registry that function builds towards -- the resolved account/host
        go straight into a `SnowflakeIdentity` held by the returned
        connector, the same way any other `identity=` would. Unlike that
        function, `token_path` is passed explicitly (the exact path this
        call already verified is mounted) rather than left for the reader
        to re-resolve.

        `warehouse`/`database`/`schema_name` default to the
        `SNOWFLAKE_WAREHOUSE`/`SNOWFLAKE_DATABASE`/`SNOWFLAKE_SCHEMA`
        environment variables SPCS exports; pass any of them explicitly to
        override.

        Args:
            warehouse: overrides `SNOWFLAKE_WAREHOUSE`. Required when unset,
                since a connector must name one.
            database: overrides `SNOWFLAKE_DATABASE`.
            schema_name: overrides `SNOWFLAKE_SCHEMA`.
            role: Snowflake role to activate for this target.

        Raises:
            ValueError: if this process is not running in a Snowpark
                Container Services container (no mounted OAuth token, or no
                account in the environment), or a required target field is
                left unset with no override.
        """
        # Imported lazily, like `source_backend`/`sink_backend` above: the
        # `snowflake` extra is optional, and reusing kumo_snowflake's own
        # env-var names keeps this in lockstep with the identity-creation
        # path it mirrors.
        spcs = import_provider("kumo_snowflake.spcs", "snowflake")
        ACCOUNT_ENV, HOST_ENV = spcs.ACCOUNT_ENV, spcs.HOST_ENV
        SPCS_TOKEN_PATH = spcs.SPCS_TOKEN_PATH

        if not os.path.exists(SPCS_TOKEN_PATH):
            raise ValueError(
                f"no snowflake token at {SPCS_TOKEN_PATH}, so this is not a "
                f"snowpark container services environment. This connector "
                f"reads a token the container runtime mounts, and there is "
                f"none to read; use `from_connection_toml` or pass an "
                f"`identity=` directly instead")
        account = os.environ.get(ACCOUNT_ENV)
        if not account:
            raise ValueError(
                f"the snowflake token at {SPCS_TOKEN_PATH} is mounted but "
                f"{ACCOUNT_ENV} is unset, so there is no account to "
                f"authenticate against. A container runtime sets both; one "
                f"without the other is an environment this cannot describe")
        host = os.environ.get(HOST_ENV) or None

        warehouse = warehouse or os.environ.get(_SPCS_WAREHOUSE_ENV)
        database = database or os.environ.get(_SPCS_DATABASE_ENV)
        schema_name = schema_name or os.environ.get(_SPCS_SCHEMA_ENV)
        missing = sorted(field for field, value in (
            ('warehouse', warehouse),
            ('database', database),
            ('schema_name', schema_name),
        ) if not value)
        if missing:
            raise ValueError(
                f"the SPCS environment names no {', '.join(missing)}, and "
                f"none was passed. A connector addresses one exact target, "
                f"so it cannot be left to the session's own default.")

        identity = SnowflakeIdentity(account=account, host=host,
                                     token_path=SPCS_TOKEN_PATH)
        assert warehouse and database and schema_name
        return cls(warehouse=warehouse, database=database,
                   schema_name=schema_name, identity=identity, role=role)


def _connections_toml_identity(entry: Dict[str, str]) -> SnowflakeIdentity:
    r"""Turns one entry's material into a `SnowflakeIdentity`.

    A key pair is preferred over a password where the entry has both,
    matching what `snowflake.connector` itself does with the same file.
    """
    key_file = entry.get('private_key_file') or entry.get('private_key_path')
    if key_file:
        return SnowflakeIdentity(
            account=entry['account'],
            username=entry.get('user'),
            private_key=Path(key_file).read_text(),
            private_key_passphrase=entry.get('private_key_file_pwd'),
        )
    if entry.get('password'):
        return SnowflakeIdentity(
            account=entry['account'],
            username=entry.get('user'),
            password=entry['password'],
        )
    raise ValueError(
        "that connection names neither a password nor a private key file. "
        "A SnowflakeIdentity holds its credentials directly, so one of the "
        "two has to be readable from the entry; an externalbrowser or SSO "
        "connection cannot be turned into one.")
