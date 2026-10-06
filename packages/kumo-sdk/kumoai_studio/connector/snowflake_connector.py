# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Mapping, Optional

from kumoapi.data_source import (
    ConnectorResponse,
    CreateIdentityArgs,
    DataSourceType,
    IdentityCredentials,
    SnowflakeAuthenticationKind,
    SnowflakeConnectorResourceConfig,
    SnowflakeIdentityConfig,
    SnowflakeKeyPairCredentials,
    SnowflakePasswordCredentials,
)
from pydantic import SecretStr
from typing_extensions import Self, override

from kumoai_studio.connector.identity_backed import IdentityBackedConnector
from kumoai_studio.identity import Identity


class SnowflakeConnector(IdentityBackedConnector):
    r"""Connect one Kumo project to a Snowflake database and schema.

    The Snowflake account, username, authentication kind, and credentials
    belong to an administrator-managed identity. This project-owned connector
    stores only the role, warehouse, database, and schema target. Passwords
    and private keys never enter the SDK.

    If ``identity`` is omitted, the server validates the target with compatible
    identities available to the active project's group and succeeds only when
    exactly one can access it. Select an identity explicitly when multiple
    Snowflake principals have access.

    .. code-block:: python

        import kumoai_studio

        analytics = kumoai_studio.SnowflakeConnector(
            name="analytics",
            warehouse="COMPUTE_WH",
            database="PROD",
            schema_name="ANALYTICS",
            role="KUMO_READER",
        )

        analytics = kumoai_studio.SnowflakeConnector(
            name="analytics",
            warehouse="COMPUTE_WH",
            database="PROD",
            schema_name="ANALYTICS",
            role="KUMO_READER",
            identity=kumoai_studio.get_identity("Production Snowflake"),
        )

        source_table = analytics["CUSTOMERS"]

    Args:
        name: Project-local connector display name.
        warehouse: Snowflake virtual warehouse used for queries.
        database: Snowflake database containing the target schema.
        schema_name: Schema within ``database``.
        identity: A credential-free :class:`~kumoai_studio.identity.Identity` or an
            immutable identity ID. ``None`` delegates unambiguous selection to
            the server. A string is never interpreted as a display name.
        role: Optional Snowflake role to activate for this target.
        description: Safe human-readable connector description.
        project_id: Exact Kumo project to use. ``None`` snapshots the active
            project without mutating global SDK state.

    Raises:
        :class:`ValueError`: If the target configuration is invalid or an
            explicit identity is empty or has the wrong provider type.
        :class:`~kumoai_studio.exceptions.HTTPException`: If the target cannot be
            accessed, identity resolution fails, or the name conflicts in this
            project.

    Note:
        All constructor arguments are keyword-only. Positional construction is
        rejected so warehouse target fields cannot be confused.

        Repeating construction with the same name and target is idempotent.
        An existing connector created with automatic resolution keeps its
        persisted identity.
    """
    def __init__(
        self,
        *,
        name: str,
        warehouse: str,
        database: str,
        schema_name: str,
        identity: str | Identity | None = None,
        role: Optional[str] = None,
        description: str = '',
        project_id: Optional[str] = None,
        _response: Optional[ConnectorResponse] = None,
    ) -> None:
        config = SnowflakeConnectorResourceConfig(
            role=role,
            warehouse=warehouse,
            database=database,
            schema_name=schema_name,
        )
        self.role = config.role
        self.warehouse = config.warehouse
        self.database = config.database
        self.schema_name = config.schema_name
        self._bind_connector(name, identity, config, description, _response,
                             project_id=project_id)

    @classmethod
    def get_by_name(cls, name: str, project_id: Optional[str] = None) -> Self:
        r"""Return an existing Snowflake connector from one exact project.

        Args:
            name: Project-local connector display name.
            project_id: Exact Kumo project, or ``None`` for the active project.

        Returns:
            The matching :class:`SnowflakeConnector` with safe target metadata
            and immutable identity ID.

        Raises:
            :class:`ValueError`: If the name is missing or belongs to another
                connector type. Connectors in other projects are not searched.
        """
        response = cls._response_by_name(name, project_id=project_id)
        config = response.config
        if not isinstance(config, SnowflakeConnectorResourceConfig):
            raise ValueError(f"Connector {name!r} is not Snowflake.")
        return cls(
            name=response.name,
            identity=response.identity_id,
            role=config.role,
            warehouse=config.warehouse,
            database=config.database,
            schema_name=config.schema_name,
            description=response.description,
            project_id=response.project_id,
            _response=response,
        )

    @override
    @property
    def source_type(self) -> DataSourceType:
        """Return the Snowflake data-source type."""
        return DataSourceType.SNOWFLAKE

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(name={self.name!r}, '
                f'database={self.database!r}, '
                f'schema_name={self.schema_name!r})')

    @classmethod
    def from_connection_toml(
        cls,
        name: str,
        *,
        connection_name: Optional[str] = None,
        warehouse: Optional[str] = None,
        database: Optional[str] = None,
        schema_name: Optional[str] = None,
        role: Optional[str] = None,
        identity_name: Optional[str] = None,
        replace: bool = False,
        description: str = '',
    ) -> Self:
        r"""Builds a connector from a ``connections.toml`` entry.

        ``connections.toml`` is Snowflake's own developer configuration, and
        the file most people already have working before they reach Kumo. An
        identity holds credentials, so this reads the named connection's
        material into one and binds a connector to the target beside it.

        The identity is created through the platform: the entry's material
        becomes a tenant identity like any other, readable by the project and
        outliving the process that uploaded it. A personal password or key
        file is a poor thing to share, so name an entry provisioned for Kumo
        rather than the one a developer signs in with, and prefer
        ``private_key_file`` to ``password`` where the account offers both.

        Args:
            name: Project-local connector display name.
            connection_name: Entry to read. ``None`` takes the file's
                ``default_connection_name``.
            warehouse: Overrides the entry's ``warehouse``. Required when the
                entry names none, since a connector must name one.
            database: Overrides the entry's ``database``.
            schema_name: Overrides the entry's ``schema``.
            role: Overrides the entry's ``role``.
            identity_name: What to create the identity under. Defaults to
                ``f'{name}-identity'``; connectors and saved graphs record it,
                so a run that must resolve elsewhere should name it.
            replace: Whether to delete an identity already registered under
                that name. Re-running a script is the case this exists for.
            description: Safe human-readable connector description.

        Raises:
            :class:`ValueError`: If the entry is absent, names neither a
                password nor a private key file, or leaves a required target
                field unset with no override.

        Example:
            >>> import kumoai_studio  # doctest: +SKIP
            >>> snow = kumoai_studio.SnowflakeConnector.from_connection_toml(
            ...     'analytics', schema_name='PUBLIC')  # doctest: +SKIP
        """
        from kumoai_studio import (
            create_identity,
            delete_identity,
            get_identity,
        )

        entry = _connections_toml_entry(connection_name)
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

        identity_name = identity_name or f'{name}-identity'
        if replace:
            # Looked up by name rather than held: `get_identity` raises for a
            # name the active project does not have, and holding nothing is
            # the ordinary case -- the first run of a script has nothing to
            # replace.
            try:
                existing = get_identity(identity_name)
            except ValueError:
                existing = None
            if existing is not None:
                delete_identity(existing.id)
        identity = create_identity(
            _connections_toml_identity_args(identity_name, entry))
        # Named rather than splatted: the check above is what makes these
        # three non-optional, and a type checker can only see that if the
        # narrowed locals are what get passed.
        assert warehouse and database and schema_name
        return cls(name=name, identity=identity, warehouse=warehouse,
                   database=database, schema_name=schema_name, role=role
                   or entry.get('role'), description=description)


def _connections_toml_entry(connection_name: Optional[str]) -> dict:
    r"""Returns one ``connections.toml`` entry, or says what the file holds.

    The library's own config manager is what resolves the file's location, so
    ``SNOWFLAKE_HOME`` and the default directory behave here exactly as they
    do for ``snowflake.connector`` itself.
    """
    from snowflake.connector.config_manager import CONFIG_MANAGER

    CONFIG_MANAGER.read_config()
    cache = CONFIG_MANAGER.conf_file_cache or {}
    connections = cache.get('connections', {})
    wanted = connection_name or cache.get('default_connection_name')
    entry = connections.get(wanted) if wanted else None
    if entry is None:
        known = ', '.join(repr(key) for key in sorted(connections)) or 'none'
        raise ValueError(
            f"connections.toml has no connection {wanted!r}. It defines: "
            f"{known}.")
    return dict(entry)


def _connections_toml_identity_args(
        name: str, entry: Mapping[str, Any]) -> CreateIdentityArgs:
    r"""Turns one entry's material into identity arguments.

    A key pair is preferred over a password where the entry has both, matching
    what ``snowflake.connector`` itself does with the same file.
    """
    key_file = entry.get('private_key_file') or entry.get('private_key_path')
    if key_file:
        passphrase = entry.get('private_key_file_pwd')
        authentication = SnowflakeAuthenticationKind.KEY_PAIR
        credentials: IdentityCredentials = SnowflakeKeyPairCredentials(
            private_key=SecretStr(Path(key_file).read_text()),
            private_key_passphrase=(SecretStr(passphrase)
                                    if passphrase else None))
    elif entry.get('password'):
        authentication = SnowflakeAuthenticationKind.PASSWORD
        credentials = SnowflakePasswordCredentials(
            password=SecretStr(entry['password']))
    else:
        raise ValueError(
            "that connection names neither a password nor a private key "
            "file. A Kumo identity holds its credentials, so one of the two "
            "has to be readable from the entry; an externalbrowser or SSO "
            "connection cannot be turned into one.")
    return CreateIdentityArgs(
        display_name=name,
        config=SnowflakeIdentityConfig(account=entry['account'],
                                       username=entry['user'],
                                       authentication=authentication),
        credentials=credentials,
    )
