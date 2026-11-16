# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
import warnings
from copy import deepcopy
from io import StringIO
from threading import Lock
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd
from kumoapi.data_source import ConnectorType, S3RootedConnectorTarget
from kumoapi.source_table import (
    CUSTOM_SQL_SOURCE_TABLE_TYPE,
    ADLSRelativeSource,
    ADLSSourceTable,
    BigQuerySourceTable,
    CustomSQLSourceTable,
    DatabricksSourceTable,
    DataSourceType,
    GCSRelativeSource,
    GCSSourceTable,
    S3RelativeSource,
    S3SourceTable,
    SnowflakeSourceTable,
    SourceColumn,
    SourceTableDataResponse,
    SourceTableType,
)
from kumoapi.table import SqlTableResolveRequest, TableDefinition
from kumoapi.typing import Dtype
from typing_extensions import override

from kumoai_studio.backend import cloud_backend

if TYPE_CHECKING:
    from kumoai_studio.connector import Connector

logger = logging.getLogger(__name__)


def _synthetic_sql_table_name(sql: str) -> str:
    r"""Derives a stable synthetic table name for a custom-SQL source table.

    A SQL source table has no underlying physical table, so downstream cache,
    staging, and data-versioning pipelines key on this name. It is
    deterministic in the query text and satisfies the Kumo ``SourceTableName``
    charset (``[a-zA-Z0-9_-]``, <= 256 chars).
    """
    digest = hashlib.md5(sql.encode('utf-8')).hexdigest()[:16]
    return f'custom_sql_{digest}'


class SourceTable:
    r"""A source table is a reference to a table stored behind a backing
    :class:`~kumoai_studio.connector.Connector`. It can be used to examine basic
    information about raw data connected to Kumo, including a sample of the
    table's rows, basic statistics, and column data type information.

    Once you are ready to use a table as part of a
    :class:`~kumoai_studio.graph.Graph`, you may create a :class:`~kumoai_studio.graph.Table`
    object from this source table, which includes additional specifying
    information (including column semantic types and column constraint
    information).

    Args:
        name: The name of this table in the backing connector
        connector: The connector containing this table.

    Example:
        >>> import kumoai_studio
        >>> connector = kumoai_studio.S3Connector(  # doctest: +SKIP
        ...     name='events',
        ...     root_dir='s3://bucket/events/',
        ...     identity=kumoai_studio.get_identity('Production AWS'),
        ... )
        >>> articles_src = connector[  # doctest: +SKIP
        ...     'archive/2026/07/articles.parquet']
        >>> articles_src = connector.table_from_uri(  # doctest: +SKIP
        ...     's3://bucket/events/archive/2026/07/articles.parquet')
        >>> articles_src = kumoai_studio.SourceTable(  # doctest: +SKIP
        ...     'archive/2026/07/articles.parquet', connector)

    Note:
        A source table references its connector by immutable ID and keeps the
        connector's owning project. Changing the process-wide active project
        does not retarget an existing source table.

        Kumo resolves the canonical provider coordinate once when the source
        table is selected. Reusing the object in tables, graphs, validation,
        or save calls does not repeat provider metadata requests. Use
        :meth:`refresh_source_coordinate` to explicitly refresh that cached
        coordinate.
    """
    def __init__(
        self,
        name: str,
        connector: 'Connector',
        *,
        _api_source_table: Optional[SourceTableType] = None,
    ) -> None:
        # TODO(manan): existence check, if not too expensive?
        self.name = name
        self.connector = connector
        self._api_source_table = deepcopy(_api_source_table)
        self._api_source_table_lock = Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_api_source_table_lock', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._api_source_table_lock = Lock()

    # Metadata ################################################################

    @property
    def column_dict(self) -> Dict[str, SourceColumn]:
        r"""Returns the names of the columns in this table along with their
        :class:`SourceColumn` information.
        """
        return {col.name: col for col in self.columns}

    @property
    def columns(self) -> List[SourceColumn]:
        r"""Returns a list of the :class:`SourceColumn` metadata of the columns
        in this table.
        """
        resp: SourceTableDataResponse = self.connector._get_table_data(
            table_names=[self.name], sample_rows=0)[0]
        return resp.cols

    # Data Access #############################################################

    def head(self, num_rows: int = 5) -> pd.DataFrame:
        r"""Returns the first :obj:`num_rows` rows of this source table by
        reading data from the backing connector.

        Args:
            num_rows: The number of rows to select. If :obj:`num_rows` is
                larger than the number of available rows, all rows will be
                returned.

        Returns:
            The first :obj:`num_rows` rows of the source table as a
            :class:`~pandas.DataFrame`.
        """
        num_rows = int(num_rows)
        if num_rows <= 0:
            raise ValueError(
                f"'num_rows' must be an integer greater than 0; got {num_rows}"
            )
        try:
            resp = self.connector._get_table_data([self.name], num_rows)[0]

            # TODO(manan, siyang): consider returning `bytes` instead of `json`
            assert resp.sample_rows is not None
            return pd.read_json(StringIO(resp.sample_rows), orient='table')
        except TypeError as e:
            raise RuntimeError(f"Cannot read head of table {self.name}. "
                               "Please restart the kernel and try.") from e

    # Persistence #############################################################

    def _to_api_source_table(self) -> SourceTableType:
        r"""Cast this source table as an object of type :obj:`SourceTableType`
        for use with the public REST API.

        The exact source coordinate is preserved. Authorization, availability,
        and validation errors from the connector are returned unchanged; this
        method never retries a different path or mutates ``self.name``.
        """
        if self._api_source_table is None:
            with self._api_source_table_lock:
                if self._api_source_table is None:
                    resolved = self.connector._get_table_config(
                        self.name).source_table
                    self._api_source_table = deepcopy(resolved)
        return deepcopy(self._api_source_table)

    def refresh_source_coordinate(self) -> 'SourceTable':
        r"""Refresh the canonical provider coordinate used for serialization.

        A source coordinate is resolved once and then serialized without
        network I/O. Call this method after an administrator intentionally
        changes or recreates the backing connector and the current
        :class:`SourceTable` should be rebound to the provider's latest
        canonical coordinate.

        Returns:
            This source table, for convenient method chaining.
        """
        with self._api_source_table_lock:
            resolved = self.connector._get_table_config(self.name).source_table
            self._api_source_table = deepcopy(resolved)
        return self

    @staticmethod
    def _from_api_table_definition(
        table_definition: TableDefinition,
        project_id: Optional[str] = None,
    ) -> 'SourceTable':
        r"""Constructs a :class:`SourceTable` from a
        :class:`kumoapi.table.TableDefinition`.
        """
        from kumoai_studio.connector import (
            ADLSConnector,
            BigQueryConnector,
            DatabricksConnector,
            FileUploadConnector,
            GCSConnector,
            S3Connector,
            SnowflakeConnector,
        )
        source_table = table_definition.source_table
        source_type = source_table.data_source_type
        connector: Connector
        # The backend's connector area rather than the REST client, so a
        # definition can be rebuilt in a process with no Kumo service: on the
        # cloud backend this is the very same object, and locally it is the
        # in-process bind store.
        from kumoai_studio.backend import cloud_backend
        connector_api = cloud_backend().connector
        if source_type == DataSourceType.S3:
            assert isinstance(source_table, S3SourceTable)
            connector_id = source_table.connector_id
            # A managed upload is a hosted concept, so a backend with no
            # such notion answers `None` and leaves this an ordinary
            # object-store source.
            upload = connector_api.get_upload_connector(project_id=project_id)
            upload_connector_id = (upload.connector_id
                                   if upload is not None else None)
            if upload_connector_id is not None \
                    and connector_id == upload_connector_id:
                connector = FileUploadConnector(
                    file_type=(source_table.file_type.value.lower()
                               if source_table.file_type is not None else
                               'parquet'), project_id=project_id)
                if not isinstance(source_table.source, S3RelativeSource):
                    raise ValueError(
                        'Managed upload source must use a relative path')
                table_name = source_table.source.path
            else:
                connector_response = connector_api.get(connector_id,
                                                       project_id=project_id)
                if connector_response is None:
                    raise ValueError(
                        f'Connector {connector_id!r} was not found')
                from kumoapi.data_source import S3ConnectorResourceConfig
                config = connector_response.config
                assert isinstance(config, S3ConnectorResourceConfig)
                connector = S3Connector(
                    name=connector_response.name,
                    identity=connector_response.identity_id,
                    root_dir=S3Connector._root_dir(config),
                    region=(config.target.region if isinstance(
                        config.target, S3RootedConnectorTarget) else None),
                    description=connector_response.description,
                    project_id=connector_response.project_id,
                    _response=connector_response,
                )
                table_name = (source_table.source.path if isinstance(
                    source_table.source, S3RelativeSource) else
                              source_table.source.uri)
        elif source_type == DataSourceType.GCS:
            assert isinstance(source_table, GCSSourceTable)
            connector_id = source_table.connector_id
            connector_response = connector_api.get(connector_id,
                                                   project_id=project_id)
            if connector_response is None:
                raise ValueError(f'Connector {connector_id!r} was not found')
            from kumoapi.data_source import GCSConnectorResourceConfig
            config = connector_response.config
            assert isinstance(config, GCSConnectorResourceConfig)
            connector = GCSConnector(
                name=connector_response.name,
                identity=connector_response.identity_id,
                root_dir=GCSConnector._root_dir(config),
                requester_pays_project=config.requester_pays_project,
                description=connector_response.description,
                project_id=connector_response.project_id,
                _response=connector_response,
            )
            table_name = (source_table.source.path if isinstance(
                source_table.source, GCSRelativeSource) else
                          source_table.source.uri)
        elif source_type == DataSourceType.ADLS:
            assert isinstance(source_table, ADLSSourceTable)
            connector_id = source_table.connector_id
            connector_response = connector_api.get(connector_id,
                                                   project_id=project_id)
            if connector_response is None:
                raise ValueError(f'Connector {connector_id!r} was not found')
            from kumoapi.data_source import ADLSConnectorResourceConfig
            config = connector_response.config
            assert isinstance(config, ADLSConnectorResourceConfig)
            connector = ADLSConnector(
                name=connector_response.name,
                identity=connector_response.identity_id,
                root_dir=ADLSConnector._root_dir(config),
                description=connector_response.description,
                project_id=connector_response.project_id,
                _response=connector_response,
            )
            table_name = (source_table.source.path if isinstance(
                source_table.source, ADLSRelativeSource) else
                          source_table.source.uri)
        elif source_type == DataSourceType.SNOWFLAKE:
            assert isinstance(source_table, SnowflakeSourceTable)
            connector_response = connector_api.get(
                source_table.snowflake_connector_id, project_id=project_id)
            if connector_response is None:
                raise ValueError('Snowflake connector was not found')
            from kumoapi.data_source import SnowflakeConnectorResourceConfig
            config = connector_response.config
            assert isinstance(config, SnowflakeConnectorResourceConfig)
            connector = SnowflakeConnector(
                name=connector_response.name,
                identity=connector_response.identity_id,
                role=config.role,
                warehouse=config.warehouse,
                database=config.database,
                schema_name=config.schema_name,
                description=connector_response.description,
                project_id=connector_response.project_id,
                _response=connector_response,
            )
            table_name = source_table.table
        elif source_type == DataSourceType.DATABRICKS:
            assert isinstance(source_table, DatabricksSourceTable)
            connector_response = connector_api.get(
                source_table.databricks_connector_id, project_id=project_id)
            if connector_response is None:
                raise ValueError('Databricks connector was not found')
            from kumoapi.data_source import DatabricksConnectorResourceConfig
            config = connector_response.config
            assert isinstance(config, DatabricksConnectorResourceConfig)
            connector = DatabricksConnector(
                name=connector_response.name,
                identity=connector_response.identity_id,
                catalog=config.catalog,
                schema_name=config.schema_name,
                warehouse_id=config.warehouse_id,
                cluster_id=config.cluster_id,
                description=connector_response.description,
                project_id=connector_response.project_id,
                _response=connector_response,
            )
            table_name = source_table.table
        elif source_type == DataSourceType.BIGQUERY:
            assert isinstance(source_table, BigQuerySourceTable)
            connector_response = connector_api.get(
                source_table.bigquery_connector_id, project_id=project_id)
            if connector_response is None:
                raise ValueError('BigQuery connector was not found')
            from kumoapi.data_source import BigQueryConnectorResourceConfig
            config = connector_response.config
            assert isinstance(config, BigQueryConnectorResourceConfig)
            connector = BigQueryConnector(
                name=connector_response.name,
                identity=connector_response.identity_id,
                project_id=config.project_id,
                dataset_id=config.dataset_id,
                location=config.location,
                billing_project_id=config.billing_project_id,
                description=connector_response.description,
                kumo_project_id=connector_response.project_id,
                _response=connector_response,
            )
            table_name = source_table.table_name
        elif source_type == CUSTOM_SQL_SOURCE_TABLE_TYPE:
            assert isinstance(source_table, CustomSQLSourceTable)
            connector_response = connector_api.get(source_table.connector_id,
                                                   project_id=project_id)
            if connector_response is None:
                raise ValueError(
                    f'Connector {source_table.connector_id!r} was not found')
            from kumoapi.data_source import (
                BigQueryConnectorResourceConfig,
                DatabricksConnectorResourceConfig,
                SnowflakeConnectorResourceConfig,
            )
            config = connector_response.config
            # ``connector_type`` is server-owned and may be unset on the
            # def, so the backing warehouse is recovered from the fetched
            # connector's config type -- the authoritative source -- as the
            # sibling branches above do.
            if isinstance(config, SnowflakeConnectorResourceConfig):
                connector = SnowflakeConnector(
                    name=connector_response.name,
                    identity=connector_response.identity_id,
                    role=config.role,
                    warehouse=config.warehouse,
                    database=config.database,
                    schema_name=config.schema_name,
                    description=connector_response.description,
                    project_id=connector_response.project_id,
                    _response=connector_response,
                )
            elif isinstance(config, DatabricksConnectorResourceConfig):
                connector = DatabricksConnector(
                    name=connector_response.name,
                    identity=connector_response.identity_id,
                    catalog=config.catalog,
                    schema_name=config.schema_name,
                    warehouse_id=config.warehouse_id,
                    cluster_id=config.cluster_id,
                    description=connector_response.description,
                    project_id=connector_response.project_id,
                    _response=connector_response,
                )
            elif isinstance(config, BigQueryConnectorResourceConfig):
                connector = BigQueryConnector(
                    name=connector_response.name,
                    identity=connector_response.identity_id,
                    project_id=config.project_id,
                    dataset_id=config.dataset_id,
                    location=config.location,
                    billing_project_id=config.billing_project_id,
                    description=connector_response.description,
                    kumo_project_id=connector_response.project_id,
                    _response=connector_response,
                )
            else:
                raise NotImplementedError(
                    f"Custom SQL is not supported for connector config "
                    f"{type(config).__name__!r}.")
            return SQLSourceTable(
                sql=source_table.sql,
                connector=connector,
                name=source_table.table,  # Preserve the backend-assigned name.
                connector_type=source_table.connector_type,
                referenced_tables=source_table.referenced_tables,
            )
        else:
            raise NotImplementedError()

        assert table_name is not None
        return SourceTable(
            table_name,
            connector,
            _api_source_table=source_table,
        )

    # Class properties ########################################################

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'


class SQLSourceTable(SourceTable):
    r"""A :class:`SourceTable` produced by a read-only warehouse query.

    Unlike a plain source table, which references a physical table by name, a
    SQL source table carries the query itself. Create one with
    :meth:`~kumoai_studio.connector.Connector.sql` or
    :meth:`~kumoai_studio.graph.Table.from_sql`. SQL sources are immutable; construct
    a new source to change the query.

    Args:
        sql: The read-only SQL query to execute in the warehouse.
        connector: The warehouse connector the query runs against.
        name: A stable synthetic name, or the canonical name returned by Kumo.
        connector_type: The server-resolved warehouse type, when available.
        referenced_tables: Server-resolved physical table lineage.
    """
    def __init__(
        self,
        sql: str,
        connector: 'Connector',
        name: Optional[str] = None,
        connector_type: Optional[ConnectorType] = None,
        referenced_tables: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name=name or _synthetic_sql_table_name(sql),
                         connector=connector)
        self._sql = sql
        self.connector_type = connector_type
        self.referenced_tables = referenced_tables or []

    @property
    def sql(self) -> str:
        return self._sql

    @override
    def _to_api_source_table(self) -> CustomSQLSourceTable:
        # ``connector_type`` and ``referenced_tables`` are server-owned. They
        # are absent before resolution and carried forward afterward; the
        # server remains authoritative for both values.
        return CustomSQLSourceTable(
            connector_id=self.connector.id,
            sql=self.sql,
            table=self.name,
            connector_type=self.connector_type,
            referenced_tables=self.referenced_tables,
        )

    @property
    @override
    def columns(self) -> List[SourceColumn]:
        r"""Resolve the query and return its authoritative output columns."""
        response = cloud_backend().table.resolve_sql(
            SqlTableResolveRequest(source_table=self._to_api_source_table()),
            project_id=self.connector.kumo_project_id)
        if response.errors:
            raise ValueError(' '.join(error.message
                                      for error in response.errors))
        if response.table_definition is None:
            raise ValueError("The server returned no resolved definition.")
        return [
            SourceColumn(
                name=col.name, stype=col.stype, dtype=col.dtype,
                multicategorical_delimiter=col.multicategorical_delimiter,
                is_primary=False) for col in response.table_definition.cols
        ]

    @override
    def head(self, num_rows: int = 5) -> pd.DataFrame:
        num_rows = int(num_rows)
        if num_rows <= 0:
            raise ValueError(
                f"'num_rows' must be an integer greater than 0; got {num_rows}"
            )
        response = cloud_backend().table.resolve_sql(
            SqlTableResolveRequest(
                source_table=self._to_api_source_table(),
                preview_rows=num_rows,
            ),
            project_id=self.connector.kumo_project_id,
        )
        if response.errors:
            raise ValueError(' '.join(error.message
                                      for error in response.errors))
        if response.table_definition is None:
            raise ValueError("The server returned no resolved definition.")
        if response.preview_note is not None:
            warnings.warn(response.preview_note, UserWarning, stacklevel=2)
        columns = response.table_definition.cols
        result = pd.DataFrame(
            response.preview_rows,
            columns=pd.Index(column.name for column in columns),
        )
        numerical = {
            Dtype.int,
            Dtype.byte,
            Dtype.int16,
            Dtype.int32,
            Dtype.int64,
            Dtype.float,
            Dtype.float32,
            Dtype.float64,
        }
        for column in columns:
            if column.dtype in numerical:
                result[column.name] = pd.to_numeric(result[column.name])
            elif column.dtype == Dtype.bool:

                def parse_bool(value):
                    # A preview arrives as JSON, so a boolean may already be
                    # one. Anything unrecognised is left alone rather than
                    # guessed at, which would turn it into `False`.
                    if isinstance(value, str):
                        return value.casefold() == 'true'
                    return value

                result[column.name] = result[column.name].map(parse_bool)
            elif column.dtype in {Dtype.date, Dtype.time}:

                def parse_temporal(value):
                    if value is None:
                        return None
                    try:
                        parsed = pd.to_datetime(value)
                    except (OverflowError, TypeError, ValueError):
                        # Databricks supports years outside pandas' nanosecond
                        # range. Preserve those valid warehouse values instead
                        # of making a bounded preview fail.
                        return value
                    # The bounds are naive and comparing across awareness
                    # raises, so a zoned value is measured against them in UTC.
                    bounded = (parsed.tz_convert(None)
                               if parsed.tzinfo is not None else parsed)
                    if (bounded < pd.Timestamp.min
                            or bounded > pd.Timestamp.max):
                        return value
                    return parsed

                result[column.name] = result[column.name].map(parse_temporal)
        return result

    @override
    def __repr__(self) -> str:
        sql = self.sql if len(self.sql) <= 60 else f'{self.sql[:57]}...'
        return f'{self.__class__.__name__}(sql={sql!r})'
