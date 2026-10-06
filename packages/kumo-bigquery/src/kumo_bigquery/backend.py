# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Resolving source tables by querying BigQuery.

The ``source`` area for one warehouse: it answers the four questions a
connector asks about its data -- which tables exist, what a table's shape is,
what its columns and sample rows are, and where its Parquet can be read from --
by asking BigQuery itself rather than asking the Kumo service to do it.

The bulk read is not restated here: :class:`~kumo_bigquery.BigQuerySourceReader`
(this package's ``source.py``) is the one implementation of it, and the
hosted platform runs the same one. Named ``backend.py``, not ``source.py``,
because that name is already the platform reader's.
"""

import logging
from typing import Any, List, Mapping, Optional

from kumo_bigquery.session import (
    _BigQuerySessions,
    _billing_project,
    _clients,
    _name,
    _qualified,
    _rows,
    _target_config,
)
from kumo_bigquery.source import BigQuerySourceReader
from kumo_bigquery.sql import QueryPlan, sample_request, table_names
from kumo_bigquery.types import dtype_for, schema_columns
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
from kumoapi.data_source import DataSourceType
from kumoapi.plugin import PrepareIngestResult, SourceConnector
from kumoapi.source_table import (
    BigQuerySourceTable,
    SourceColumn,
    SourceTableConfigResponse,
    SourceTableDataResponse,
    SourceTableListResponse,
)
from kumoapi.typing import Dtype

logger = logging.getLogger(__name__)


class BigQuerySourceBackend(_BigQuerySessions):
    r"""Resolves source tables by querying BigQuery.

    Args:
        identity: A fallback identity, used only for connectors that name none
            themselves. Accepts an :class:`~kumoai_studio.identity.Identity`, an
            identity ID, or the name it was created under. Leaving it unset is
            the common case: a bound
            :class:`~kumoai_studio.connector.BigQueryConnector` records the identity
            it was created with.
        mode: The read mode every table is read under, or ``None`` to take
            whichever the reader offers first.
        staging_uri: A ``gs://`` prefix this run may export Parquet to. Without
            one -- the default -- every table is streamed through the Storage
            Read API instead, which is a slower path for a large table but not
            an error: no bucket is a complete configuration.

    Example:
        >>> import kumoai_studio as kumo  # doctest: +SKIP
        >>> from kumoapi.data_source import (  # doctest: +SKIP
        ...     BigQueryIdentityConfig, CreateIdentityArgs,
        ...     GoogleWorkloadIdentityAuthentication)
        >>> identity = kumo.create_identity(  # doctest: +SKIP
        ...     CreateIdentityArgs(
        ...         display_name='prod-bq',
        ...         config=BigQueryIdentityConfig(
        ...             authentication=GoogleWorkloadIdentityAuthentication(
        ...                 service_account_email='kumo@data-project'
        ...                 '.iam.gserviceaccount.com'))))
        >>> kumo.set_backend(kumo.LocalBackend(  # doctest: +SKIP
        ...     sources={kumo.DataSourceType.BIGQUERY: BigQuerySourceBackend(
        ...         staging_uri='gs://bucket/kumo')}))
        >>> events = kumo.BigQueryConnector(  # doctest: +SKIP
        ...     name='events', project_id='data-project',
        ...     dataset_id='analytics', identity=identity)
    """

    source_type = DataSourceType.BIGQUERY

    _default_hint = ('LocalBackend(sources={DataSourceType.BIGQUERY: '
                     'BigQuerySourceBackend(identity=identity)})')

    def __init__(
        self,
        identity: Optional[IdentityRef] = None,
        mode: Optional[ReadMode] = None,
        staging_uri: Optional[str] = None,
    ) -> None:
        super().__init__(identity)
        self.mode = mode
        self.staging_uri = staging_uri

    # Source area ##########################################################

    def list_tables(
        self,
        connector: SourceConnector,
        project_id: Optional[str] = None,
    ) -> SourceTableListResponse:
        del project_id
        config = _target_config(connector)
        dataset = f'{config.project_id}.{config.dataset_id}'
        return SourceTableListResponse(
            table_names=table_names(self._client(connector), dataset))

    def get_table_config(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> SourceTableConfigResponse:
        del project_id
        self._table(connector, table_name)  # 404s if it does not exist
        return SourceTableConfigResponse(source_table=BigQuerySourceTable(
            bigquery_connector_id=connector.id,
            table_name=table_name,
            project_id=_target_config(connector).project_id,
            dataset_id=_target_config(connector).dataset_id,
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
        r"""A query's sample is gated on what running it would scan.

        ``LIMIT`` does not reduce a BigQuery scan, so over the budget the
        columns come back with no rows and the caller infers from the data
        type alone. A physical table is not gated.
        """
        del project_id
        from kumo_core.custom_sql import (
            SQL_SAMPLE_MAX_SCAN_BYTES,
            sample_over_budget,
            sample_timeout,
        )
        query = self._embeddable(sql)
        if query is None:
            return self._table_sample(connector, table_name, sample_rows)
        plan = self._plan(connector, query)
        columns = self._described(plan)
        scanned = plan.scanned_bytes
        if sample_over_budget(scanned, gated=True):
            logger.warning(
                'Not sampling the Custom SQL query for %s: running it would '
                'scan %s, and a LIMIT does not reduce a BigQuery scan. '
                'Semantic types fall back to the data type alone; add a '
                'filter, or declare the columns.', table_name,
                'an amount BigQuery would not estimate' if scanned is None else
                f'~{scanned} bytes, over the {SQL_SAMPLE_MAX_SCAN_BYTES} '
                f'budget')
            return SourceTableDataResponse(table_name=table_name, cols=columns)
        # The estimate above is advisory, so the sample also carries a
        # ceiling BigQuery enforces itself.
        config = _target_config(connector)
        statement, job_config = sample_request(
            query, sample_rows, project=config.project_id,
            dataset=config.dataset_id,
            max_bytes_billed=SQL_SAMPLE_MAX_SCAN_BYTES,
            job_timeout_ms=_milliseconds(sample_timeout(timeout, query)))
        rows = _rows(self._client(connector), statement, job_config)
        return sampled_table(table_name, columns, rows)

    def _table_sample(self, connector: SourceConnector, table_name: str,
                      sample_rows: int) -> SourceTableDataResponse:
        from kumo_core.custom_sql import sample_select
        columns = self._columns(connector, table_name)
        rows = _rows(
            self._client(connector),
            sample_select(_qualified(connector, table_name), sample_rows))
        return sampled_table(table_name, columns, rows)

    def get_download_urls(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> List[str]:
        raise NotImplementedError(
            "BigQuery tables have no download URL; they are read through a "
            "client. Custom prediction tables must come from an object-store "
            "or Databricks connector.")

    def get_data_version(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> str:
        r"""One ``tables.get``, and the same value :meth:`prepare_ingest`
        reports, so a caller can decide whether an existing snapshot still
        describes this table before paying for the read that would otherwise
        establish it.
        """
        del project_id  # a local runtime has no projects
        return self._reader(connector).data_version(table_name)

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
        r"""Reads one table into local Parquet.

        The read itself belongs to ``kumo_bigquery.BigQuerySourceReader``, and
        which of its two modes runs depends on whether this backend was given
        a staging bucket. What this adds is the two things only a *local*
        caller knows: which columns the graph declared, and where on this disk
        the Parquet may land.
        """
        del project_id  # a local runtime has no projects
        from kumo_core.sources.prepare import read_table

        # One directory per table: a read empties what it is handed, so two
        # tables sharing a `dest_dir` would leave the second holding only its
        # own files.
        destination = staging_dir(dest_dir, table_name)
        query = self._embeddable(sql)
        staged = read_table(
            self._reader(connector), table_name, str(destination),
            columns=self._declared(connector, table_name, dtypes, query),
            mode=self.mode, sql=query, version=data_version)
        if not staged.paths:
            raise PermanentSourceError(
                f'Reading {table_name} produced no files. An empty source '
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

    def _reader(self, connector: SourceConnector) -> BigQuerySourceReader:
        r"""Built per call: the reader is a thin wrapper and cheap to build,
        even though the clients it wraps are now cached.
        """
        principal = self._principal(connector)
        credentials = self._credentials(connector)
        billing = _billing_project(connector)
        storage = (_clients.storage_client(principal, project=billing)
                   if self.staging_uri is not None else None)
        return BigQuerySourceReader(
            self._client(connector),
            project=_target_config(connector).project_id,
            dataset=_target_config(connector).dataset_id,
            read_client=_clients.read_client(principal),
            storage_client=storage,
            staging_uri=self.staging_uri,
            billing_project=_target_config(connector).billing_project_id,
            credentials=credentials,
        )

    # Internals ############################################################

    def _table(self, connector: SourceConnector, table_name: str) -> Any:
        r"""`Connector.table()` and `has_table()` distinguish a missing table
        by a ``404``, so this has to raise the same shape.
        """
        from google.api_core.exceptions import NotFound
        try:
            return self._client(connector).get_table(
                _name(connector, table_name))
        except NotFound as error:
            config = _target_config(connector)
            raise SourceNotFoundError(
                f'Invalid table {table_name}: it does not exist in '
                f'{config.project_id}.{config.dataset_id}.') from error

    def _embeddable(self, sql: Optional[str]) -> Optional[str]:
        r"""The only form of a query this reader ever sends: the caller's text
        with its trailing terminator and comment removed, so it stands in
        subquery position.
        """
        if sql is None:
            return None
        from kumo_core.custom_sql import subquery_safe_sql
        from kumo_core.types import ConnectorType
        return subquery_safe_sql(sql, ConnectorType.BIGQUERY)

    def _plan(self, connector: SourceConnector, sql: str) -> QueryPlan:
        from kumo_bigquery.sql import query_plan
        return query_plan(self._client(connector), sql,
                          project=_target_config(connector).project_id,
                          dataset=_target_config(connector).dataset_id)

    def _described(self, plan: QueryPlan) -> List[SourceColumn]:
        return source_columns(schema_columns(plan.schema))

    def _columns(self, connector: SourceConnector, table_name: str,
                 sql: Optional[str] = None) -> List[SourceColumn]:
        if sql is not None:
            return self._described(self._plan(connector, sql))
        return source_columns(
            schema_columns(self._table(connector, table_name).schema))

    @staticmethod
    def dtype_for(field_type: str, mode: Optional[str] = None,
                  precision: Optional[int] = None,
                  scale: Optional[int] = None) -> Dtype:
        return dtype_for(field_type, mode, precision, scale)

    def __repr__(self) -> str:
        return f'BigQuerySourceBackend(identity={self._identity!r})'


def _milliseconds(seconds: Optional[int]) -> Optional[int]:
    r"""BigQuery states a job's own deadline in milliseconds."""
    return None if seconds is None else int(seconds) * 1000
