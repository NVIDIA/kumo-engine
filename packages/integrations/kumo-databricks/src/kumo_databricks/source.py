# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from functools import partial
from typing import Mapping, Sequence
from uuid import uuid4

from databricks.sdk import WorkspaceClient
from databricks.sql.client import Connection, Cursor
from kumo_core.sources.arrow_parquet import count_rows
from kumo_core.sources.identity import sql_data_version
from kumo_core.sources.prepare import ReadOutput, run_prepare
from kumo_core.sources.protocol import require_mode, staged_first
from kumo_core.sources.transfer import staged_fetch
from kumo_core.sources.types import ReadMode, StagedTable
from kumo_core.types.enums import ConnectorType
from kumo_databricks.errors import translated
from kumo_databricks.identity import column_types
from kumo_databricks.identity import data_version as read_data_version
from kumo_databricks.identity import query_column_types
from kumo_databricks.sql import build_projection, qualified_name, read_query
from kumo_databricks.staging import clear_volume, download, list_parquet, stage
from kumo_databricks.streaming import stream_to_parquet
from kumoapi.typing import Dtype


@dataclass
class DatabricksSourceReader:
    """The staging volume belongs to the reader, and its absence is what
    makes :attr:`ReadMode.STAGED` unsupported rather than a failure."""

    connection: Connection
    _: KW_ONLY
    catalog: str
    schema: str
    workspace_client: WorkspaceClient | None = None
    staging_volume_path: str | None = None

    @property
    def source_type(self) -> ConnectorType:
        return ConnectorType.DATABRICKS

    def read_modes(self, table: str) -> Sequence[ReadMode]:
        """Databricks presigns its own CloudFetch objects, so a streamed read
        needs no grant beyond the ``SELECT``, and no probe to establish it."""
        return staged_first(self._stages())

    def data_version(self, table: str, *, sql: str | None = None,
                     columns: Mapping[str, Dtype] | None = None) -> str:
        cursor = self.connection.cursor()
        try:
            with translated(table, ReadMode.STREAMED):
                return self._stamp(cursor, table, sql, columns)
        finally:
            cursor.close()

    def _stamp(self, cursor: Cursor, table: str, sql: str | None = None,
               columns: Mapping[str, Dtype] | None = None) -> str:
        # `DESCRIBE HISTORY` over a query is reported as a missing table, so
        # a query is never routed through it.
        if sql is None:
            return read_data_version(cursor, self._source(table))
        return sql_data_version(
            sql, ConnectorType.DATABRICKS,
            lambda leaf: read_data_version(cursor, self._source(leaf)), columns,
            table=table)

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
        if not columns:
            raise ValueError("a projection of at least one column is required")
        cursor = self.connection.cursor()
        try:
            with translated(table, mode):
                require_mode(table, mode, self.read_modes(table))
                return run_prepare(
                    table,
                    dest,
                    mode,
                    stamp=lambda: self._stamp(cursor, table, sql, columns),
                    read=lambda: self._read(cursor, table, dest, columns,
                                            aliases, mode, worker_ceiling, sql),
                    expected_rows=expected_rows,
                    version=version,
                )
        finally:
            cursor.close()

    def _read(
        self,
        cursor: Cursor,
        table: str,
        dest: str,
        columns: Mapping[str, Dtype],
        aliases: Mapping[str, str] | None,
        mode: ReadMode,
        worker_ceiling: int | None,
        sql: str | None,
    ) -> ReadOutput:
        """The declared width of a column is known only to the catalog, and a
        ``DECIMAL`` beyond 64 bits has to be narrowed before it is read."""
        source = self._embedded(table, sql)
        declared = (column_types(cursor, source)
                    if sql is None else query_column_types(cursor, source))
        projection = build_projection(columns, aliases, declared)
        if mode is ReadMode.STREAMED:
            written = stream_to_parquet(cursor, read_query(source, projection),
                                        dest, worker_ceiling=worker_ceiling)
            return ReadOutput(paths=written.paths, num_rows=written.rows)
        return self._stage(cursor, table, dest, projection, worker_ceiling,
                           source)

    def _stage(self, cursor: Cursor, table: str, dest: str, projection: str,
               worker_ceiling: int | None, source: str) -> ReadOutput:
        """Every attempt unloads into a directory of its own because the
        volume is cleared before the unload and again after it, so one read of
        a table-named directory would delete another's parquet or download a
        mixture of the two.
        """
        volume = self.staging_volume_path.rstrip('/')
        location = f"{volume}/{table}/{uuid4().hex}"
        client = self.workspace_client

        def unload() -> None:
            stage(cursor, location=location, projection=projection,
                  source=source)

        def fetch(_: None) -> tuple[str, ...]:
            return download(client, list_parquet(client, location), dest,
                            worker_ceiling=worker_ceiling)

        _, paths = staged_fetch(
            clear=partial(clear_volume, client, location,
                          worker_ceiling=worker_ceiling), unload=unload,
            fetch=fetch, location=location)
        return ReadOutput(paths=paths, num_rows=count_rows(paths))

    def _stages(self) -> bool:
        return bool(self.staging_volume_path and self.workspace_client)

    def _source(self, table: str) -> str:
        return qualified_name(self.catalog, self.schema, table)

    def _embedded(self, table: str, sql: str | None) -> str:
        return self._source(table) if sql is None else f"({sql})"
