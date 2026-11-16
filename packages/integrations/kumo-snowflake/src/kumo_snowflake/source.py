# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import logging
from contextlib import contextmanager
from dataclasses import KW_ONLY, dataclass
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

from kumo_core.sources.errors import RowCountMismatchError
from kumo_core.sources.identity import sql_data_version
from kumo_core.sources.prepare import ReadOutput, run_prepare
from kumo_core.sources.protocol import require_mode
from kumo_core.sources.transfer import staged_fetch
from kumo_core.sources.types import Attestation, ReadMode, StagedTable
from kumo_core.types.enums import ConnectorType
from kumo_snowflake.errors import translated
from kumo_snowflake.identity import column_types as read_column_types
from kumo_snowflake.identity import data_version as read_data_version
from kumo_snowflake.identity import resolved_table_name
from kumo_snowflake.sql import build_projection, qualified_name
from kumo_snowflake.staging import (
    UnloadResult,
    clear_prefix,
    create_stage,
    download,
    list_prefix,
    presigned_urls,
    unload,
)
from kumoapi.typing import Dtype

STAGED_ONLY = (ReadMode.STAGED, )

logger = logging.getLogger(__name__)


@dataclass
class SnowflakeSourceReader:
    """Read Snowflake catalog names through an owned temporary stage.

    Connector adapters resolve Snowflake's quoted/unquoted input syntax before
    constructing a reader. SDK table discovery instead returns exact catalog
    names. The reader therefore preserves an exact table match first and falls
    back to Snowflake's unquoted folding only when the exact name is absent.
    Database, schema, resolved table, and stage names are always quoted in SQL.
    """

    connection: Any
    _: KW_ONLY
    database: str
    schema: str
    stage_name: str = "KUMO_INGEST"

    @property
    def source_type(self) -> ConnectorType:
        return ConnectorType.SNOWFLAKE

    def read_modes(self, table: str) -> Sequence[ReadMode]:
        """A temporary stage needs no grant, and a streamed read would move
        the same bytes more slowly, so it is not offered."""
        return STAGED_ONLY

    def data_version(self, table: str, *, sql: str | None = None,
                     columns: Mapping[str, Dtype] | None = None) -> str:
        cursor = self.connection.cursor()
        try:
            with translated(table, ReadMode.STAGED):
                catalog_table = (table if sql is not None else
                                 self._resolved_table(cursor, table))
                return self._stamp(cursor, catalog_table, sql, columns)
        finally:
            cursor.close()

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
        """``sql`` is embedded in parenthesized subquery position and never
        rewritten; ``table`` remains the synthetic name the stage prefix and
        every error are keyed on."""

        cursor = self.connection.cursor()
        try:
            with translated(table, mode):
                catalog_table = (table if sql is not None else
                                 self._resolved_table(cursor, table))
                # Interpolated, never bound: a literal `?` or `%` in a user's
                # query would corrupt a parameterized statement.
                source = (f"({sql})" if sql is not None else qualified_name(
                    self.database, self.schema, catalog_table))
                require_mode(table, mode, STAGED_ONLY)
                projection = build_projection(
                    columns, aliases,
                    read_column_types(cursor, source, [
                        name for name, dtype in columns.items()
                        if Dtype(dtype) in {Dtype.time, Dtype.date}
                    ]))
                return run_prepare(
                    table,
                    dest,
                    mode,
                    stamp=lambda: self._stamp(cursor, catalog_table, sql,
                                              columns),
                    read=lambda: self.
                    _unload(cursor, table, dest, projection, expected_rows,
                            worker_ceiling, source),
                    expected_rows=expected_rows,
                    version=version,
                )
        finally:
            cursor.close()

    @property
    def _stage(self) -> str:
        """Qualified because a session whose current schema is elsewhere
        would otherwise create the stage there, or fail for want of one."""
        return qualified_name(self.database, self.schema, self.stage_name)

    def _stamp(self, cursor: Any, table: str, sql: str | None = None,
               columns: Mapping[str, Dtype] | None = None) -> str:
        if sql is None:
            return read_data_version(cursor, database=self.database,
                                     schema=self.schema, table=table)
        return sql_data_version(
            sql, ConnectorType.SNOWFLAKE, lambda leaf: read_data_version(
                cursor, database=self.database, schema=self.schema, table=leaf),
            columns, table=table)

    def _resolved_table(self, cursor: Any, table: str) -> str:
        return resolved_table_name(cursor, database=self.database,
                                   schema=self.schema, table=table)

    def _unload(
        self,
        cursor: Any,
        table: str,
        dest: str,
        projection: str,
        expected_rows: int | None,
        worker_ceiling: int | None,
        source: str,
    ) -> ReadOutput:
        """``COPY INTO`` attests the count itself, so reconciling is free and
        precedes the download. A table of no rows unloads no file, so the count
        is also what tells an empty source from a stage that lost its objects.

        Each attempt unloads under an id of its own because the stage is
        cleared before the unload and again after it, and the SDK hands one
        cached session to every connector an identity opens, so two concurrent
        reads of a table-named prefix would delete each other's objects.
        """
        prefix = f"{table}/{uuid4().hex}/"
        stage = self._stage
        create_stage(cursor, stage)

        def unload_projection() -> UnloadResult:
            result = unload(cursor, stage=stage, prefix=prefix,
                            projection=projection, source=source)
            if (expected_rows is not None
                    and result.rows_unloaded != expected_rows):
                raise RowCountMismatchError(expected=expected_rows,
                                            actual=result.rows_unloaded,
                                            table=table)
            return result

        def fetch(result: UnloadResult) -> tuple[str, ...]:
            if not result.rows_unloaded:
                return ()
            files = list_prefix(cursor, stage, prefix)
            urls = presigned_urls(cursor, stage,
                                  [staged.path for staged in files])
            return download(files, urls, dest, worker_ceiling=worker_ceiling)

        result, paths = staged_fetch(
            clear=lambda: clear_prefix(cursor, stage, prefix),
            unload=unload_projection, fetch=fetch, location=f"{stage}/{prefix}")
        return ReadOutput(paths=paths, num_rows=result.rows_unloaded,
                          rows_attested_by=Attestation.WAREHOUSE,
                          scanned_bytes=result.input_bytes)


def microsecond_precision_kwargs(method: Any) -> dict[str, bool]:
    """Support cursor wrappers that omit the precision keyword without
    masking TypeErrors raised during the actual fetch.
    """
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return {}
    if any(parameter.name == 'force_microsecond_precision'
           or parameter.kind == inspect.Parameter.VAR_KEYWORD
           for parameter in parameters):
        return {'force_microsecond_precision': True}
    return {}


class ReadOnlyCursorCleanupError(RuntimeError):
    pass


@contextmanager
def read_cursor(connection: Any, query: str, *,
                timeout: int | None = None) -> Iterator[Any]:
    cursor = None
    primary_error: BaseException | None = None
    try:
        cursor = connection.cursor()
        if timeout is None:
            cursor.execute(query)
        else:
            cursor.execute(
                query, _statement_params={
                    'STATEMENT_TIMEOUT_IN_SECONDS': str(timeout)
                })
        yield cursor
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except BaseException as cleanup_error:
                if primary_error is None:
                    logger.error(
                        'Snowflake export cursor cleanup failed '
                        '(error_type=%s)',
                        type(cleanup_error).__name__)
                    raise ReadOnlyCursorCleanupError(
                        'Failed to finalize the Snowflake table read.'
                    ) from None
                logger.warning(
                    'Snowflake export cursor cleanup failed while preserving '
                    'primary failure (primary_error_type=%s cleanup_error_type=%s)',
                    type(primary_error).__name__,
                    type(cleanup_error).__name__)
