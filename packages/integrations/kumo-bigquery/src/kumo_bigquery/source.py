# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any

from google.auth.credentials import Credentials
from kumo_bigquery.errors import resolving, translated
from kumo_bigquery.identity import data_version as read_data_version
from kumo_bigquery.identity import is_logical_view
from kumo_bigquery.sql import (
    build_projection,
    home_dataset,
    order_by_schema,
    output_columns,
    qualified_name,
    query_plan,
    table_path,
)
from kumo_bigquery.staging import StagingLocation, probe, staged_read
from kumo_bigquery.streaming import ReadTarget, query_target, target_streams
from kumo_core.custom_sql import subquery_safe_sql
from kumo_core.sources.arrow_parquet import (
    count_rows,
    materialize_arrow_streams,
)
from kumo_core.sources.identity import sql_data_version
from kumo_core.sources.prepare import ReadOutput, run_prepare
from kumo_core.sources.protocol import require_mode, staged_first
from kumo_core.sources.types import Capability, ReadMode, StagedTable
from kumo_core.types.enums import ConnectorType
from kumoapi.typing import Dtype


@dataclass
class BigQuerySourceReader:
    """A probe decides whether ``staging_uri`` may in fact be written to, so no
    bucket and no grant both mean a streamed read rather than a failure.
    ``credentials`` builds the parallel readers; missing it, a read fails
    rather than quietly collapsing onto a single connection.
    """

    client: Any
    _: KW_ONLY
    project: str
    dataset: str
    read_client: Any
    storage_client: Any = None
    staging_uri: str | None = None
    billing_project: str | None = None
    credentials: Credentials | None = None
    staging: StagingLocation | None = field(init=False, default=None)
    _staging_capability: Capability | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.staging_uri is None:
            return
        if self.storage_client is None:
            raise ValueError(
                "a staging uri needs a storage client to fetch the export")
        self.staging = StagingLocation.parse(self.staging_uri)

    @property
    def source_type(self) -> ConnectorType:
        return ConnectorType.BIGQUERY

    def probe_staging(self) -> Capability:
        """Answered once per reader and never persisted: a stale ``DENIED``
        would pin a deployment slow long after the grant was fixed."""
        if self.staging is None:
            return Capability.DENIED
        if self._staging_capability is None:
            answer = probe(self.client, self.storage_client,
                           location=self.staging)
            if answer is Capability.UNKNOWN:
                return answer
            self._staging_capability = answer
        return self._staging_capability

    def read_modes(self, table: str) -> Sequence[ReadMode]:
        return staged_first(self.probe_staging() is Capability.ALLOWED)

    def data_version(self, table: str, *, sql: str | None = None,
                     columns: Mapping[str, Dtype] | None = None) -> str:
        with translated(table, ReadMode.STREAMED):
            return self._stamp(table, sql, columns)

    def _stamp(self, table: str, sql: str | None = None,
               columns: Mapping[str, Dtype] | None = None) -> str:
        if sql is not None:
            return sql_data_version(
                sql, ConnectorType.BIGQUERY,
                lambda leaf: read_data_version(self.client, self._name(leaf)),
                columns, table=table)
        with resolving(table):
            return read_data_version(self.client, self._name(table))

    def prepare(
        self,
        table: str,
        dest: str,
        *,
        columns: Mapping[str, Dtype],
        mode: ReadMode,
        aliases: Mapping[str, str] | None = None,
        expected_rows: int | None = None,
        worker_ceiling: int | None = None,
        sql: str | None = None,
        version: str | None = None,
    ) -> StagedTable:
        """A query has no table to ask for a schema, so the dry run that
        classified it supplies the column order too."""
        with translated(table, mode):
            if mode is ReadMode.STAGED:
                # Only staging turns on the probe, so a streamed read never
                # pays for an answer that cannot change its outcome.
                require_mode(table, mode, self.read_modes(table))
            if sql is not None:
                schema = self._plan(sql).schema
                streamed_by_query = True
            else:
                with resolving(table):
                    info = self.client.get_table(self._name(table))
                # A read session opens on a table but not a view.
                schema, streamed_by_query = info.schema, is_logical_view(info)
            ordered = order_by_schema(columns,
                                      [column.name for column in schema])
            projection = build_projection(ordered, aliases)
            return run_prepare(
                table,
                dest,
                mode,
                stamp=lambda: self._stamp(table, sql, columns),
                read=lambda: self._read(table, dest, ordered, aliases,
                                        projection, mode, worker_ceiling,
                                        self._embedded(table, sql
                                                       ), streamed_by_query),
                expected_rows=expected_rows,
                version=version,
            )

    def _read(
        self,
        table: str,
        dest: str,
        columns: Mapping[str, Dtype],
        aliases: Mapping[str, str] | None,
        projection: str,
        mode: ReadMode,
        worker_ceiling: int | None,
        source: str,
        streamed_by_query: bool,
    ) -> ReadOutput:
        if mode is ReadMode.STAGED:
            paths, scanned = staged_read(self.client, self.storage_client,
                                         location=self.staging.child(table),
                                         projection=projection, source=source,
                                         dest=dest,
                                         worker_ceiling=worker_ceiling,
                                         job_config=self._job_config())
        else:
            paths, scanned = self._streamed(table, dest, columns, aliases,
                                            projection, worker_ceiling, source,
                                            streamed_by_query)
        return ReadOutput(paths=paths, num_rows=count_rows(paths),
                          scanned_bytes=scanned)

    def _streamed(
        self,
        table: str,
        dest: str,
        columns: Mapping[str, Dtype],
        aliases: Mapping[str, str] | None,
        projection: str,
        worker_ceiling: int | None,
        source: str,
        streamed_by_query: bool,
    ) -> tuple[tuple[str, ...], int | None]:
        """A read session opens on a table but not a view or a query, and
        selects fields but cannot rename one; all three are answered by the
        table BigQuery writes every query result to."""
        fields = output_columns(columns, aliases)
        if streamed_by_query or aliases:
            target = query_target(self.client,
                                  sql=f"SELECT {projection} FROM {source}",
                                  fields=fields, job_config=self._job_config())
        else:
            target = ReadTarget(
                path=table_path(self.project, self.dataset, table),
                fields=fields)
        with target_streams(self.read_client, target,
                            billing_project=self._billing(),
                            credentials=self.credentials,
                            worker_ceiling=worker_ceiling) as (streams,
                                                               workers):
            written = materialize_arrow_streams(streams, dest,
                                                worker_ceiling=workers)
        return written.paths, target.bytes_scanned

    def _name(self, table: str) -> str:
        return f"{self.project}.{self.dataset}.{table}"

    def _embedded(self, table: str, sql: str | None) -> str:
        if sql is None:
            return qualified_name(self.project, self.dataset, table)
        return f"({subquery_safe_sql(sql, self.source_type)})"

    def _plan(self, sql: str) -> Any:
        return query_plan(self.client, sql, project=self.project,
                          dataset=self.dataset)

    def _job_config(self) -> Any:
        return home_dataset(self.project, self.dataset)

    def _billing(self) -> str:
        """The project a read session is billed to, not the one it reads."""
        return self.billing_project or self.client.project
