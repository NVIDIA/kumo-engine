# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Resolving source tables by querying Snowflake.

The ``source`` area for one warehouse: it answers the four questions a
connector asks about its data -- which tables exist, what a table's shape is,
what its columns and sample rows are, and where its Parquet can be read from --
by querying Snowflake itself rather than asking the Kumo service to do it.

The bulk unload is not restated here: ``kumo_snowflake.SnowflakeSourceReader``
is the one implementation of it, and the hosted platform runs the same one.
"""

from typing import List, Mapping, Optional, Union

# One rule for what a dtype implies, shared with the table area.
from kumo_core.custom_sql import sample_timeout, subquery_safe_sql
from kumo_core.sources.backend import (
    ingest_result,
    sampled_table,
    select_columns,
    source_columns,
)
from kumo_core.sources.errors import PermanentSourceError, SourceNotFoundError
from kumo_core.sources.layout import staging_dir
from kumo_core.sources.principals import IdentityRef
from kumo_core.sources.types import ReadMode
from kumo_snowflake.session import (
    SnowflakeSession,
    _require_snowpark,
    _SnowflakeSessions,
    _target_config,
    resolved,
)
from kumo_snowflake.source import SnowflakeSourceReader
from kumo_snowflake.sql import describe_columns
from kumo_snowflake.types import dtype_for
from kumoapi.data_source import DataSourceType
from kumoapi.plugin import PrepareIngestResult, SourceConnector
from kumoapi.source_table import (
    SnowflakeSourceTable,
    SourceColumn,
    SourceTableConfigResponse,
    SourceTableDataResponse,
    SourceTableListResponse,
)
from kumoapi.typing import Dtype


class SnowflakeSourceBackend(_SnowflakeSessions):
    r"""Resolves source tables by querying Snowflake.

    Args:
        identity: A fallback identity, used only for connectors that name none
            themselves. Accepts an :class:`~kumoai_studio.identity.Identity`, an
            identity ID, or the name it was created under. Leaving it unset is the
            common case: a bound
            :class:`~kumoai_studio.connector.SnowflakeConnector` records the identity
            it was created with, which is what makes this reader stateless
            enough for :class:`~kumoai_studio.backend.LocalBackend` to register by
            default.

    Example:
        >>> import kumoai_studio as kumo  # doctest: +SKIP
        >>> from kumoapi.data_source import (  # doctest: +SKIP
        ...     CreateIdentityArgs, SnowflakeAuthenticationKind,
        ...     SnowflakeIdentityConfig)
        >>> identity = kumo.create_identity(  # doctest: +SKIP
        ...     CreateIdentityArgs(
        ...         display_name='prod-sf',
        ...         config=SnowflakeIdentityConfig(
        ...             account='xy12345',
        ...             authentication=(
        ...                 SnowflakeAuthenticationKind.OAUTH_TOKEN_FILE))))
        >>> kumo.set_backend(kumo.LocalBackend())  # doctest: +SKIP
        >>> snow = kumo.SnowflakeConnector(  # doctest: +SKIP
        ...     name='analytics', warehouse='WH', database='ANALYTICS',
        ...     schema_name='PUBLIC', identity=identity)
    """

    source_type = DataSourceType.SNOWFLAKE

    _default_hint = ('LocalBackend(sources={DataSourceType.SNOWFLAKE: '
                     'SnowflakeSourceBackend(identity=identity)})')

    def __init__(
        self,
        identity: Optional[IdentityRef] = None,
        mode: Optional[ReadMode] = None,
    ) -> None:
        super().__init__(identity)
        #: Snowflake unloads and offers nothing else; the argument exists so
        #: one caller can name a mode whichever warehouse it reaches.
        self.mode = mode

    # Source area ##########################################################

    def list_tables(
        self,
        connector: SourceConnector,
        project_id: Optional[str] = None,
    ) -> SourceTableListResponse:
        del project_id
        from kumo_snowflake.catalog import tables
        config = _target_config(connector)
        with self._session(connector) as session:
            _require_snowpark(session)
            cursor = session.connection.cursor()
            try:
                names = tables(cursor, database=resolved(config.database),
                               schema=resolved(config.schema_name))
            finally:
                cursor.close()
        return SourceTableListResponse(table_names=names)

    def get_table_config(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> SourceTableConfigResponse:
        del project_id
        if not self._exists(connector, table_name):
            # `Connector.table()` and `has_table()` distinguish a missing
            # table by a 404, which the source area translates this into.
            raise SourceNotFoundError(
                f'Invalid table {table_name}: it does not exist in '
                f'{resolved(_target_config(connector).database)}.'
                f'{resolved(_target_config(connector).schema_name)}.')
        return SourceTableConfigResponse(source_table=SnowflakeSourceTable(
            snowflake_connector_id=connector.id,
            database=_target_config(connector).database,
            schema_name=_target_config(connector).schema_name,
            table=table_name,
        ))

    def get_columns(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
        sql: Optional[str] = None,
    ) -> SourceTableDataResponse:
        del project_id
        return SourceTableDataResponse(
            table_name=table_name,
            cols=self._columns(connector, table_name, self._embeddable(sql)),
        )

    def get_sample(
        self,
        connector: SourceConnector,
        table_name: str,
        sample_rows: int,
        project_id: Optional[str] = None,
        sql: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> SourceTableDataResponse:
        del project_id

        query = self._embeddable(sql)
        source = (f'({query})' if query is not None else self._qualified(
            connector, table_name))
        # One span for both statements: the columns and the rows they name
        # have to come from the same warehouse.
        with self._session(connector) as session:
            columns = self._columns(connector, table_name, query)
            from kumo_snowflake.results import sample
            frame = sample(session.connection, source, sample_rows,
                           timeout=sample_timeout(timeout, query))
            rows = frame.itertuples(index=False, name=None)
        return sampled_table(table_name, columns, rows)

    def get_download_urls(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> List[str]:
        raise NotImplementedError(
            "Snowflake tables have no download URL; they are read through a "
            "session. Custom prediction tables must come from an object-store "
            "or Databricks connector.")

    def get_data_version(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> str:
        r"""Returns the table's data version from the catalog alone.

        One ``INFORMATION_SCHEMA`` read, and the same value
        :meth:`prepare_ingest` reports, so a caller can decide whether an
        existing snapshot still describes this table before paying for the
        unload that would otherwise establish it.
        """
        del project_id  # a local runtime has no projects
        with self._session(connector) as session:
            return self._reader(connector, session).data_version(table_name)

    def prepare_ingest(
        self,
        connector: SourceConnector,
        table_name: str,
        dest_dir: str,
        dtypes: Optional[Mapping[str, Dtype]] = None,
        project_id: Optional[str] = None,
        data_version: Optional[str] = None,
        sql: Optional[str] = None,
    ) -> PrepareIngestResult:
        r"""Unloads one table to a temporary stage and downloads what it made.

        The unload itself belongs to ``kumo_snowflake.SnowflakeSourceReader``;
        what this adds is the two things only a *local* caller knows: which
        columns the graph declared, and where on this disk the Parquet may
        land.

        The pre-signed URLs are spent inside that call rather than handed
        onward, so what ingest receives is a ``parquet`` source over local
        paths -- the source type every DiskGraph build reads, rather than one
        only a build that speaks this transport would accept.
        """
        del project_id  # a local runtime has no projects
        from kumo_core.sources.prepare import read_table

        query = self._embeddable(sql)
        # One directory per table: a read empties what it is handed, so two
        # tables sharing a `dest_dir` would leave the second holding only its
        # own files.
        destination = staging_dir(dest_dir, table_name)
        # Held for the whole unload, not just to select the warehouse: the
        # reader runs on the connection behind this session, so a connector
        # that took it mid-read would move the unload to its own warehouse.
        with self._session(connector) as session:
            staged = read_table(
                self._reader(connector, session), table_name, str(destination),
                columns=self._declared(connector, table_name, dtypes, query),
                mode=self.mode, sql=query, version=data_version)
        if not staged.paths:
            raise PermanentSourceError(
                f'Unloading {table_name} produced no files. An empty source '
                f'table cannot be ingested; DiskGraph needs at least one row '
                f'to infer an encoding.')

        return ingest_result(staged)

    def _declared(
        self,
        connector: SourceConnector,
        table_name: str,
        dtypes: Optional[Mapping[str, Dtype]],
        sql: Optional[str] = None,
    ) -> Mapping[str, Dtype]:
        catalog = self._columns(connector, table_name, sql)
        return select_columns(table_name,
                              {column.name: column.dtype
                               for column in catalog}, dtypes)

    def _reader(self, connector: SourceConnector,
                session: SnowflakeSession) -> SnowflakeSourceReader:
        r"""Built per call because it holds the connection behind the
        session, and a session resolved once and kept would outlive the
        identity it came from. ``Session.connection`` borrows the connection
        Snowpark is already running on rather than opening a second one.
        """
        return SnowflakeSourceReader(
            session.connection,
            database=resolved(_target_config(connector).database),
            schema=resolved(_target_config(connector).schema_name),
        )

    # Internals ############################################################

    def _exists(self, connector: SourceConnector, table_name: str) -> bool:
        from kumo_snowflake.catalog import table_exists
        config = _target_config(connector)
        with self._session(connector) as session:
            _require_snowpark(session)
            cursor = session.connection.cursor()
            try:
                return table_exists(cursor, database=resolved(config.database),
                                    schema=resolved(config.schema_name),
                                    table=table_name)
            finally:
                cursor.close()

    def _embeddable(self, sql: Optional[str]) -> Optional[str]:
        r"""The only form of a query this reader ever sends: the caller's text
        with its trailing terminator and comment removed, so it stands in
        subquery position.

        The warehouse is named here rather than read off ``source_type``,
        because this reader serves exactly one and asking itself which would
        only be a way to get it wrong.
        """
        if sql is None:
            return None
        from kumo_core.types import ConnectorType
        return subquery_safe_sql(sql, ConnectorType.SNOWFLAKE)

    def _described(self, connector: SourceConnector,
                   sql: str) -> List[SourceColumn]:
        with self._session(connector) as session:
            _require_snowpark(session)
            cursor = session.connection.cursor()
            try:
                return source_columns(
                    describe_columns(cursor, f'SELECT * FROM ({sql})'))
            finally:
                cursor.close()

    def _columns(self, connector: SourceConnector, table_name: str,
                 sql: Optional[str] = None) -> List[SourceColumn]:
        if sql is not None:
            return self._described(connector, sql)
        from kumo_snowflake.catalog import columns

        config = _target_config(connector)
        with self._session(connector) as session:
            _require_snowpark(session)
            cursor = session.connection.cursor()
            try:
                metadata = columns(cursor, database=resolved(config.database),
                                   schema=resolved(config.schema_name),
                                   table=table_name)
            finally:
                cursor.close()
        if not metadata:
            raise SourceNotFoundError(
                f'Invalid table {table_name}: it has no columns in '
                f'{resolved(config.database)}.{resolved(config.schema_name)}.')
        return source_columns(
            (name, dtype) for (_, name), dtype in metadata.items())

    @staticmethod
    def dtype_for(data_type: str,
                  numeric_scale: Optional[Union[int, str]] = None) -> Dtype:
        r"""Maps a Snowflake data type to a Kumo data type.

        Kept on the class because that is where callers reach it. The mapping
        itself lives in :mod:`~kumo_snowflake.types`, beside
        the reverse one it must not drift from.
        """
        return dtype_for(data_type, numeric_scale)

    def __repr__(self) -> str:
        return f'SnowflakeSourceBackend(identity={self._identity!r})'
