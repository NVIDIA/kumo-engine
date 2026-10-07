# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Resolving source tables by querying Databricks.

The ``source`` area for one warehouse: it answers the four questions a
connector asks about its data -- which tables exist, what a table's shape is,
what its columns and sample rows are, and where its Parquet can be read from
-- by querying Databricks itself rather than asking the Kumo service to do it.

The bulk read is not restated here: :class:`~kumo_databricks.
DatabricksSourceReader` (this package's ``source.py``) is the one
implementation of it, and the hosted platform runs the same one. Named
``backend.py``, not ``source.py``, because that name is already the platform
reader's.
"""

from __future__ import annotations

from typing import List, Mapping, Optional

from databricks.sql.client import Connection
from kumo_core.custom_sql import (
    gate_custom_sql,
    sample_timeout,
    subquery_safe_sql,
)
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
from kumo_databricks import sql as dbx_sql
from kumo_databricks.catalog import table_names
from kumo_databricks.session import _DatabricksSessions, _target_config
from kumo_databricks.source import DatabricksSourceReader
from kumo_databricks.types import dtype_for
from kumoapi.data_source import DataSourceType
from kumoapi.plugin import PrepareIngestResult, SourceConnector
from kumoapi.source_table import (
    DatabricksSourceTable,
    SourceColumn,
    SourceTableConfigResponse,
    SourceTableDataResponse,
    SourceTableListResponse,
)
from kumoapi.typing import Dtype


class DatabricksSourceBackend(_DatabricksSessions):
    r"""Resolves source tables by querying Databricks.

    Args:
        identity: A fallback identity, used only for connectors that name none
            themselves. Accepts an :class:`~kumoai_studio.identity.Identity`, an
            identity ID, or the name it was created under. Leaving it unset is the
            common case: a bound
            :class:`~kumoai_studio.connector.DatabricksConnector` records the
            identity it was created with.
        mode: A :class:`~kumo_core.sources.types.ReadMode` to force, when a
            run needs to know which path it took rather than only that the
            table was read. ``None`` takes the cheapest the reader offers.
        staging_volume_path: A Unity Catalog volume to unload through, such as
            ``/Volumes/main/kumo/staging``. Staging leads where it is
            available. ``None`` is not an error: the reader then offers a
            streamed CloudFetch read only, which needs no grant beyond the
            ``SELECT`` the table already required.

    Example:
        >>> import kumoai_studio as kumo  # doctest: +SKIP
        >>> from kumoapi.data_source import (  # doctest: +SKIP
        ...     CreateIdentityArgs, DatabricksIdentityConfig,
        ...     DatabricksOAuthCredentials)
        >>> from pydantic import SecretStr  # doctest: +SKIP
        >>> identity = kumo.create_identity(  # doctest: +SKIP
        ...     CreateIdentityArgs(
        ...         display_name='prod-dbx',
        ...         config=DatabricksIdentityConfig(
        ...             host='https://workspace.cloud.databricks.com',
        ...             client_id='service-principal-id'),
        ...         credentials=DatabricksOAuthCredentials(
        ...             client_secret=SecretStr('...'))))
        >>> kumo.set_backend(kumo.LocalBackend(  # doctest: +SKIP
        ...     sources={kumo.DataSourceType.DATABRICKS:
        ...              DatabricksSourceBackend(
        ...                  staging_volume_path='/Volumes/main/kumo/staging')}
        ... ))
        >>> bricks = kumo.DatabricksConnector(  # doctest: +SKIP
        ...     name='analytics', catalog='main', schema_name='features',
        ...     warehouse_id='4b9b953939869799', identity=identity)
    """

    source_type = DataSourceType.DATABRICKS

    _default_hint = ('LocalBackend(sources={DataSourceType.DATABRICKS: '
                     'DatabricksSourceBackend(identity=identity)})')

    def __init__(
        self,
        identity: Optional[IdentityRef] = None,
        mode: Optional[ReadMode] = None,
        staging_volume_path: Optional[str] = None,
    ) -> None:
        super().__init__(identity)
        self.mode = mode
        self.staging_volume_path = staging_volume_path

    # Source area ##########################################################

    def list_tables(
        self,
        connector: SourceConnector,
        project_id: Optional[str] = None,
    ) -> SourceTableListResponse:
        del project_id
        config = _target_config(connector)
        with self._connected(connector) as connection:
            cursor = connection.cursor()
            try:
                tables = table_names(cursor, catalog=config.catalog,
                                     schema=config.schema_name,
                                     target_schema=config.schema_name)
            finally:
                cursor.close()
        return SourceTableListResponse(table_names=tables)

    def get_table_config(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> SourceTableConfigResponse:
        del project_id
        bare = self._bare(connector, table_name)
        # Describing the table is the existence probe: Databricks reports a
        # missing one only when something asks for it.
        with self._connected(connector) as connection:
            self._columns(connector, connection, bare)
        config = _target_config(connector)
        return SourceTableConfigResponse(source_table=DatabricksSourceTable(
            databricks_connector_id=connector.id,
            # Qualified, because this field is the only place a Databricks
            # source table can carry its catalog and schema, and a portable
            # graph is read back by that name alone.
            table=f'{config.catalog}.{config.schema_name}.{bare}',
        ))

    def get_columns(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
        sql: Optional[str] = None,
    ) -> SourceTableDataResponse:
        del project_id
        bare = self._bare(connector, table_name)
        query = self._embeddable(connector, sql)
        with self._connected(connector) as connection:
            columns = self._columns(connector, connection, bare, query)
        return SourceTableDataResponse(table_name=table_name, cols=columns)

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

        bare = self._bare(connector, table_name)
        query = self._embeddable(connector, sql)
        source = (f'({query})' if query is not None else self._qualified(
            connector, bare))
        with self._connected(connector) as connection:
            columns = self._columns(connector, connection, bare, query)
            cursor = connection.cursor()
            try:
                bound = self._sample_timeout(timeout, query)
                if bound is not None:
                    cursor.execute(f'SET STATEMENT_TIMEOUT = {int(bound)}')
                dbx_sql.sample(cursor, source, sample_rows)
                rows = [tuple(row) for row in cursor.fetchall()]
            finally:
                cursor.close()
        return sampled_table(table_name, columns, rows)

    def get_download_urls(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> List[str]:
        raise NotImplementedError(
            "Databricks tables have no download URL; they are read through a "
            "SQL connection. Custom prediction tables must come from an "
            "object-store connector.")

    def get_data_version(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> str:
        r"""The same value :meth:`prepare_ingest` reports, so a caller can
        decide whether an existing snapshot still describes this table before
        paying for the read that would otherwise establish it.
        """
        del project_id  # a local runtime has no projects
        bare = self._bare(connector, table_name)
        with self._connected(connector) as connection:
            return self._reader(connector, connection).data_version(bare)

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
        r"""Reads one table into local Parquet and says where it landed.

        The read itself belongs to
        :class:`~kumo_databricks.DatabricksSourceReader`. What this
        adds is the two things only a *local* caller knows: which columns the
        graph declared, and where on this disk the Parquet may land.
        """
        del project_id  # a local runtime has no projects
        from kumo_core.sources.prepare import read_table

        bare = self._bare(connector, table_name)
        # One directory per table: a read empties what it is handed, so two
        # tables sharing a `dest_dir` would leave the second holding only its
        # own files.
        destination = staging_dir(dest_dir, bare)
        query = self._embeddable(connector, sql)
        with self._connected(connector) as connection:
            columns = self._declared(connector, connection, bare, dtypes, query)
            staged = read_table(self._reader(connector, connection), bare,
                                str(destination), columns=columns,
                                mode=self.mode, sql=query, version=data_version)
        if not staged.num_rows:
            raise PermanentSourceError(
                f'Reading {bare} returned no rows. An empty source '
                f'table cannot be ingested; DiskGraph needs at least one row '
                f'to infer an encoding.')

        return ingest_result(staged)

    def _declared(
        self,
        connector: SourceConnector,
        connection: Connection,
        table_name: str,
        dtypes: Optional[Mapping[str, Dtype]],
        sql: Optional[str] = None,
    ) -> Mapping[str, Dtype]:
        catalog = self._columns(connector, connection, table_name, sql)
        return select_columns(table_name,
                              {column.name: column.dtype
                               for column in catalog}, dtypes)

    def _reader(self, connector: SourceConnector,
                connection: Connection) -> DatabricksSourceReader:
        r"""Borrows the caller's connection rather than opening a second one,
        so the catalog reads framing a bulk read run on the same session.
        """
        from kumo_databricks.source import DatabricksSourceReader
        return DatabricksSourceReader(
            connection,
            catalog=_target_config(connector).catalog,
            schema=_target_config(connector).schema_name,
            workspace_client=self._client(connector),
            staging_volume_path=self.staging_volume_path,
        )

    # Internals ############################################################

    def _embeddable(self, connector: SourceConnector,
                    sql: Optional[str]) -> Optional[str]:
        r"""The only form of a query this reader ever sends: the caller's text
        with its trailing terminator and comment removed, so it stands in
        subquery position.
        """
        if sql is None:
            return None
        from kumo_core.types import ConnectorType
        gate_custom_sql(sql, ConnectorType.DATABRICKS,
                        _target_config(connector))
        return subquery_safe_sql(sql, ConnectorType.DATABRICKS)

    @staticmethod
    def _sample_timeout(timeout: Optional[int],
                        query: Optional[str]) -> Optional[int]:
        return sample_timeout(timeout, query)

    def _columns(self, connector: SourceConnector, connection: Connection,
                 table_name: str,
                 sql: Optional[str] = None) -> List[SourceColumn]:
        from kumo_databricks import errors, identity

        cursor = connection.cursor()
        try:
            if sql is None:
                declared = list(
                    identity.column_types(
                        cursor, self._qualified(connector, table_name)).items())
            else:
                declared = identity.query_columns(cursor, f'({sql})')
        except Exception as error:
            # `Connector.table()` and `has_table()` distinguish a missing
            # table by a 404; anything else is the warehouse's to report.
            if not errors.is_not_found(error):
                raise
            declared = []
        finally:
            cursor.close()
        if not declared:
            config = _target_config(connector)
            raise SourceNotFoundError(
                f'Invalid table {table_name}: it does not exist in '
                f'{config.catalog}.{config.schema_name}.')

        return source_columns(
            (name, self.dtype_for(data_type)) for name, data_type in declared)

    @staticmethod
    def dtype_for(data_type: str) -> Dtype:
        return dtype_for(data_type)

    def __repr__(self) -> str:
        return (f'DatabricksSourceBackend(identity={self._identity!r}, '
                f'staging_volume_path={self.staging_volume_path!r})')
