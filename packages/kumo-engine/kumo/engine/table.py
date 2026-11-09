# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""`Table`: metadata only, no I/O at construction.

A `Table` describes a table definition (schema/columns) plus a
`(Connector, table_name_or_path)` source pointer. Materializing the data
happens later, when a `Graph` containing it is ingested via
`Graph.snapshot()`. Schema inference can read the source first.

Two ways to get a `Table` with `definition` populated:

* Explicit construction for hosted source types: build a
  `kumoapi.table.TableDefinition` directly
  (`cols=[...]`, fully-specified `Column`s with `stype`/`dtype` set -- unlike
  the cloud SDK's own mutable `Column` wrapper, this `Column` requires both,
  so there is no partial/pre-inference state to represent) and pass it to
  `Table(connector, table_name, definition=...)`. No connector read at all.
* `from_source_table(source_table, ...)` -- auto-discovers the schema from
  the actual source data. `source_table` is a `kumo.engine.connector.
  source_table.SourceTable`, obtained via `connector[table_name]` or
  `connector.table(table_name)`. Its `column_dict` (`kumo.engine._infer.
  infer_source_columns`, cached per `SourceTable` instance) supplies the
  source-schema pass; `from_source_table` finishes resolving anything still
  ambiguous (`kumo.engine._infer.resolve_table_columns`, sampling only if needed).
  See `kumo.engine._infer`'s module docstring for why that resolution can't be
  deferred to a later call the way the cloud SDK's own `Table.
  from_source_table`/`infer_metadata()` split does it.
* `from_sql(connector, sql, ...)` -- the same, over a Custom SQL query
  (`connector.sql(sql)`).

Local filesystem sources use the local schema subclass defined in
`kumo.engine._local_schema`; shared hosted request schemas remain unchanged.

The cloud SDK also lets a caller edit an already-resolved column directly
(`table[name].stype = ...`, since its `Column` wrapper is mutable).
`with_column` is the immutable equivalent: it returns a new `Table` with one
column replaced by a copy carrying the given overrides, leaving everything
else untouched.
"""

from dataclasses import dataclass, replace
from typing import List, Mapping, Optional, Union

from kumo_core.stype_inference import multicategorical_separator
from kumoapi.source_table import CustomSQLSourceTable
from kumoapi.table import Column
from kumoapi.table import TableDefinition as HostedTableDefinition
from kumoapi.table import TimestampUnit
from kumoapi.typing import Dtype, MulticategoricalDelimiter, Stype
from typing_extensions import Self

from kumo.engine._local_schema import TableDefinition
from kumo.engine.connector.base import Connector
from kumo.engine.connector.source_table import SourceTable


@dataclass(frozen=True)
class Table:
    r"""An immutable, metadata-only description of one table.

    Args:
        connector: The connector this table's data will be read from at
            snapshot time. Referenced directly, never by id.
        table_name: The name of the table within `connector`.
        definition: The table's schema. `None` until resolved, via
            `from_source_table(...)` or `infer_metadata()`. A bare
            `Table(connector, table_name)` (no `definition`) is the entry
            point `infer_metadata()` needs to discover a schema from
            scratch -- without a default here, there would be no way to
            construct that starting point at all.
    """

    connector: Connector
    table_name: str
    definition: Optional[Union[TableDefinition, HostedTableDefinition]] = None

    @classmethod
    def from_source_table(
        cls,
        source_table: SourceTable,
        column_names: Optional[List[str]] = None,
        *,
        primary_key: Optional[str] = None,
        time_column: Optional[str] = None,
        end_time_column: Optional[str] = None,
        dtypes: Optional[Mapping[str, Dtype]] = None,
    ) -> Self:
        r"""Builds a `Table` from a `SourceTable` -- `connector[table_name]`
        or `connector.table(table_name)`. Mirrors the cloud SDK's `Table.
        from_source_table(source_table, column_names, ...)`, with one
        necessary difference: the cloud SDK's version is cheap (its own
        `Column` wrapper allows `stype=None`, so it can return a
        partially-resolved `Table` and leave sampling to a later, no-arg
        `Table.infer_metadata()` call), but `kumoapi.table.Column` here
        requires `stype`/`dtype` set -- `kumoapi.table.TableDefinition`,
        which this method must produce, cannot hold anything less. So this
        does whatever `source_table.column_dict`'s cheap pass left
        ambiguous itself, via `kumo.engine._infer.resolve_table_columns` (which
        samples the source only if something actually needs it) -- see that
        module's docstring for the full reasoning.

        A key or time column is named, not auto-detected: this reasserts
        its stype (`Stype.ID`/`Stype.timestamp`) regardless of what
        `column_dict` implied from dtype/statistics alone -- `column_dict`
        itself carries no such awareness, since it describes the source
        table on its own, not its role in a particular `Graph`'s schema.

        Args:
            source_table: the table to build from -- names both the
                connector and the provider-native table name.
            column_names: restrict to these columns. `None` keeps every
                column the source reports.
            primary_key: names the primary-key column.
            time_column: names the time column.
            end_time_column: names the end-time column.
            dtypes: declared dtypes, by column name, that replace the
                source's. DiskGraph converts the source values to them.
                This also resolves a column whose source type has no Kumo
                dtype; `Graph.snapshot()` refuses one left unresolved.
        """
        from kumo.engine._infer import resolve_table_columns

        columns = resolve_table_columns(
            source_table.connector,
            source_table.table_name,
            source_table.column_dict,
            column_names=column_names,
            primary_key=primary_key,
            time_column=time_column,
            end_time_column=end_time_column,
            sql=source_table.sql,
            dtypes=dtypes,
        )
        definition = TableDefinition(
            cols=columns,
            source_table=source_table.to_source_table(),
            pkey=primary_key,
            time_col=time_column,
            end_time_col=end_time_column,
        )
        return cls(connector=source_table.connector,
                   table_name=source_table.table_name, definition=definition)

    @classmethod
    def from_sql(
        cls,
        connector: Connector,
        sql: str,
        column_names: Optional[List[str]] = None,
        *,
        primary_key: Optional[str] = None,
        time_column: Optional[str] = None,
        end_time_column: Optional[str] = None,
    ) -> Self:
        r"""Builds a `Table` from a read-only `SELECT` run in `connector`'s
        warehouse -- `from_source_table(connector.sql(sql), ...)`. Mirrors
        the cloud SDK's `Table.from_sql`. Inference samples the query's
        result, so the query runs once here and again at `Graph.snapshot()`.
        """
        return cls.from_source_table(
            connector.sql(sql),
            column_names,
            primary_key=primary_key,
            time_column=time_column,
            end_time_column=end_time_column,
        )

    def infer_metadata(self) -> Self:
        r"""Re-resolves this table's schema from the current source data,
        returning a new `Table` with `definition` replaced. Takes no
        arguments -- unlike `from_source_table`, matching the cloud SDK's
        own argument-less `Table.infer_metadata()` -- reusing whatever
        column selection/primary key/time columns this table already
        carries: `self.definition`'s existing `cols`/`pkey`/`time_col`/
        `end_time_col` if already resolved (e.g. to pick up new data, or
        after `with_column` overrides were cleared some other way), or
        every column with no keys if this table was constructed bare
        (`Table(connector, table_name)`, `definition=None`). A dtype declared
        for a column whose source type has no Kumo dtype is kept.
        """
        if self.definition is not None:
            column_names = [col.name for col in self.definition.cols]
            primary_key = self.definition.pkey
            time_column = self.definition.time_col
            end_time_column = self.definition.end_time_col
        else:
            column_names = None
            primary_key = None
            time_column = None
            end_time_column = None
        source_table = self._source_table()
        unreadable = {
            name
            for name, col in source_table.column_dict.items()
            if col.dtype == Dtype.unsupported
        }
        table = self.from_source_table(
            source_table,
            column_names,
            primary_key=primary_key,
            time_column=time_column,
            end_time_column=end_time_column,
            dtypes={
                col.name: col.dtype
                for col in self.definition.cols if col.name in unreadable
                and col.dtype != Dtype.unsupported
            } if self.definition is not None else None,
        )
        return replace(table, table_name=self.table_name)

    def _source_table(self) -> SourceTable:
        source = (self.definition.source_table
                  if self.definition is not None else None)
        return (self.connector.sql(source.sql) if isinstance(
            source, CustomSQLSourceTable) else self.connector[self.table_name])

    def _inferred_stype(self, name: str, dtype: Dtype) -> Stype:
        from kumo.engine._infer import resolve_table_columns

        assert self.definition is not None
        source = self._source_table()
        stype = resolve_table_columns(
            source.connector,
            source.table_name,
            source.column_dict,
            column_names=[name],
            primary_key=None,
            time_column=None,
            end_time_column=None,
            sql=source.sql,
            dtypes={name: dtype},
        )[0].stype
        assert stype is not None
        return stype

    def with_column(
        self,
        name: str,
        *,
        stype: Optional[Stype] = None,
        dtype: Optional[Dtype] = None,
        timestamp_format: Optional[Union[TimestampUnit, str]] = None,
        multicategorical_delimiter: Optional[MulticategoricalDelimiter] = None,
    ) -> Self:
        r"""Returns a new `Table` with column `name` replaced by a copy that
        has the given fields overridden, leaving every other column (and
        the rest of `definition`) untouched. The immutable equivalent of
        the cloud SDK's `table[name].stype = ...`/`table[name].dtype = ...`
        in-place column editing -- a way to correct one column after
        `infer_metadata()` got it wrong (e.g. an `int64` id-like column
        read as `numerical` when it should be `categorical`), without
        re-running inference.

        Only fields actually passed override the existing column; `None`
        (the default for every argument) means "leave as is". Declaring a
        scalar-string multicategorical column without a delimiter samples
        the source to infer one using the hosted policy.

        Raises:
            ValueError: if `self.definition` isn't resolved yet, or names no
                column `name`.
        """
        if self.definition is None:
            raise ValueError(
                f"Table {self.table_name!r} has no resolved `definition` -- "
                "call from_source_table() or infer_metadata() first.")

        columns: List[Column] = list(self.definition.cols)
        for i, column in enumerate(columns):
            if column.name == name:
                resolved_stype = stype if stype is not None else column.stype
                if (stype is None and dtype is not None
                        and resolved_stype == Stype.unsupported):
                    resolved_stype = self._inferred_stype(name, dtype)
                resolved_dtype = dtype if dtype is not None else column.dtype
                delimiter = (multicategorical_delimiter
                             if multicategorical_delimiter is not None else
                             column.multicategorical_delimiter)
                if (multicategorical_delimiter is None
                        and (resolved_stype != Stype.multicategorical
                             or resolved_dtype != Dtype.string)):
                    delimiter = None
                if (resolved_stype == Stype.multicategorical
                        and resolved_dtype == Dtype.string
                        and delimiter is None):
                    from kumo.engine._infer import _sample

                    _, _, frame, _ = _sample(self.connector.source_backend,
                                             self.connector, self.table_name)
                    if frame is not None and name in frame:
                        delimiter = multicategorical_separator(
                            frame[name].dropna().astype('string'))
                columns[i] = replace(
                    column,
                    stype=resolved_stype,
                    dtype=resolved_dtype,
                    timestamp_format=(timestamp_format if timestamp_format
                                      is not None else column.timestamp_format),
                    multicategorical_delimiter=delimiter,
                )
                break
        else:
            raise ValueError(
                f"Table {self.table_name!r} has no column {name!r}.")

        definition = replace(self.definition, cols=columns)
        return replace(self, definition=definition)
