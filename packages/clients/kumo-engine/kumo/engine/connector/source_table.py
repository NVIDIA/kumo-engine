# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""`SourceTable`: a reference to a table living behind a `Connector`.

Mirrors the cloud SDK's `kumoai.connector.SourceTable` (`connector[name]` ->
`SourceTable`), minus what only exists to round-trip a server-side project:
construction here only stores the connector and table name.
`SQLSourceTable` (`connector.sql(query)`) is the Custom SQL variant.

`column_dict`/`columns` use the source-schema pass described in `kumo.engine._infer`.
A column whose stype the source dtype alone doesn't settle (most
numeric/string columns) comes back with
`stype=None`; resolving that via a sampled read is `Table.
from_source_table`'s job, not this class's -- see `kumo.engine._infer`'s module
docstring for why the split lands there rather than in a later no-arg
`infer_metadata()` call the way it does in the cloud SDK.

Cached (`functools.cached_property`) per `SourceTable` instance, computed
once on first access -- unlike the cloud SDK, which round-trips to the
backend on every access.
"""

from functools import cached_property
from typing import Dict, List, Optional

from kumoapi.data_source import ConnectorType as ApiConnectorType
from kumoapi.source_table import CustomSQLSourceTable, SourceColumn

from kumo.engine._local_schema import LocalSourceTableType
from kumo.engine.connector.base import Connector


class SourceTable:
    r"""A reference to `table_name` within `connector`, with its (possibly
    partially-resolved) schema read lazily -- and cached -- from the source
    catalog.

    Constructed via `connector[table_name]` (`Connector.__getitem__`) or
    `connector.table(table_name)`, not directly -- consumed by `Table.
    from_source_table(source_table, ...)`.
    """
    def __init__(self, table_name: str, connector: Connector) -> None:
        self.table_name = table_name
        self.connector = connector

    @cached_property
    def column_dict(self) -> Dict[str, SourceColumn]:
        r"""Every column of this table, name to `kumoapi.source_table.
        SourceColumn` -- dtype from the source schema, stype from the
        dtype where that settles it on its own, `None` otherwise. No
        primary-key/time-column awareness (that's a role
        within a particular `Graph`'s schema, not a property of the source
        table itself) -- both are `Table.from_source_table`'s job.
        """
        from kumo.engine._infer import infer_source_columns

        return infer_source_columns(self.connector, self.table_name,
                                    sql=self.sql)

    @property
    def sql(self) -> Optional[str]:
        return None

    @property
    def columns(self) -> List[SourceColumn]:
        return list(self.column_dict.values())

    def to_source_table(self) -> LocalSourceTableType:
        r"""Delegate source-coordinate resolution to
        `Connector.source_table_for` for `TableDefinition.source_table`.
        Resolution may perform source I/O to determine the path and format.
        """
        return self.connector.source_table_for(self.table_name)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(table_name={self.table_name!r}, "
                f"connector={self.connector!r})")


def checked_sql(connector: Connector, sql: str) -> CustomSQLSourceTable:
    r"""Runs the shared Custom SQL gates (read-only, single statement, named
    outputs, tables within the connector's namespace) and returns the
    resolved source, named the way hosted names it.
    """
    from kumo_core.custom_sql import custom_sql_synthetic_name, gate_custom_sql
    from kumo_core.types import ConnectorType

    kind = getattr(connector.source_type, 'name', None)
    if kind not in ('SNOWFLAKE', 'DATABRICKS', 'BIGQUERY'):
        raise ValueError(
            f"Custom SQL is only supported for warehouse connectors "
            f"(Snowflake, Databricks, BigQuery); {type(connector).__name__} "
            f"cannot execute SQL.")
    connector_type = ConnectorType[kind]
    refs = gate_custom_sql(sql, connector_type, connector.config)
    return CustomSQLSourceTable(
        connector_id=connector.id,
        sql=sql,
        table=custom_sql_synthetic_name(sql, connector_type),
        connector_type=ApiConnectorType(connector_type.value),
        referenced_tables=refs,
    )


class SQLSourceTable(SourceTable):
    r"""A source table whose rows come from a read-only `SELECT` run in the
    connector's warehouse. Mirrors the cloud SDK's `kumoai_studio.connector.
    SQLSourceTable`; constructed via `connector.sql(query)`.

    The query is checked up front with the same `kumo_core.custom_sql`
    gates hosted applies, and `table_name` is hosted's synthetic name for
    it. Reads pass the query to the connector's reader as `sql=`.
    """
    def __init__(self, sql: str, connector: Connector) -> None:
        self._source = checked_sql(connector, sql)
        super().__init__(self._source.table, connector)

    @property
    def sql(self) -> str:
        return self._source.sql

    @property
    def referenced_tables(self) -> List[str]:
        return list(self._source.referenced_tables)

    def to_source_table(self) -> CustomSQLSourceTable:
        return self._source

    def __repr__(self) -> str:
        sql = self.sql if len(self.sql) <= 60 else f'{self.sql[:57]}...'
        return f'{self.__class__.__name__}(sql={sql!r})'
