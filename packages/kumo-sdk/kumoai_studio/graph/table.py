# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import time
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union

import kumoapi.data_snapshot as snapshot_api
import kumoapi.table as api
import pandas as pd
from kumoapi.common import JobStatus
from kumoapi.data_snapshot import TableSnapshotID
from kumoapi.source_table import CustomSQLSourceTable
from kumoapi.typing import Dtype, Stype
from typing_extensions import Self

from kumoai_studio import global_state
from kumoai_studio.backend import cloud_backend
from kumoai_studio.client.table import TableID
from kumoai_studio.connector import (
    Connector,
    SourceColumn,
    SourceTable,
    SQLSourceTable,
)
from kumoai_studio.graph.column import Column

logger = logging.getLogger(__name__)

# Type alias for column-like inputs that can be cast to a `Column`:
ColumnLike: TypeAlias = Union[
    str,
    Column,
    Tuple[str, Optional[Stype], Optional[Dtype]],
    Dict[str, Union[str, Dtype, Stype, None]],
]

_DEFAULT_INTERVAL_S = 20


class Table:
    r"""A Table represents metadata information for a table in a Kumo
    :class:`~kumoai_studio.graph.Graph`.

    Whereas a :class:`~kumoai_studio.connector.SourceTable` is simply a reference to a
    table behind a backing :class:`~kumoai_studio.connector.Connector`, a table fully
    specifies the relevant metadata (including selected source columns, column
    data type and semantic type, and relational constraint information)
    necessary to train a :class:`~kumoai_studio.pquery.PredictiveQuery` on graph of
    tables. A table can either be constructed explicitly, or with the
    convenience method :meth:`~kumoai_studio.graph.Table.from_source_table`.

    The table belongs to the active project and its source connector must
    belong to that same project. A table cannot retain a source reference from
    another project.

    .. code-block:: python

        import kumoai_studio

        # Define connector to source data:
        connector = kumoai_studio.S3Connector(
            name='events', root_dir='s3://bucket/events/')

        # Create table using `from_source_table`:
        customer = kumoai_studio.Table.from_source_table(
            source_table=connector['customer'],
            primary_key='CustomerID',
        )

        # Create a table by constructing it directly:
        customer = kumoai_studio.Table(
            source_table=connector['customer'],
            columns=[kumoai_studio.Column(name='CustomerID', dtype='string', stype='ID')],
            primary_key='CustomerID',
        )

        # Infer any missing metadata in the table, from source table
        # properties:
        print("Current metadata: ", customer.metadata)
        customer.infer_metadata()

        # Validate the table configuration, for use in Kumo downstream models:
        customer.validate(verbose=True)

        # Fetch statistics from a snapshot of this table (this method will
        # take a table snapshot, and as a result may have high latency):
        customer.get_stats(wait_for="minimal")

    Args:
        source_table: The source table this Kumo table is created from.
        columns: The selected columns of the source table that are part of this
            Kumo table. Note that each column must specify its data type and
            semantic type; see the :class:`~kumoai_studio.graph.Column` documentation
            for more information, including the delimiter contract for scalar
            string multi-categorical columns. If `None`  all columns from the
            source table are included by default.
        primary_key: The primary key of the table, if present. The primary key
            must exist in the :obj:`columns` argument.
        time_column: The time column of the table, if present. The time column
            must exist in the :obj:`columns` argument.
        end_time_column: The end time column of the table, if present. The end
            time column must exist in the :obj:`columns` argument.
    """  # noqa: E501

    def __init__(
        self,
        source_table: SourceTable,
        columns: Optional[Sequence[Union[SourceColumn, ColumnLike]]] = None,
        primary_key: Optional[ColumnLike] = None,
        time_column: Optional[ColumnLike] = None,
        end_time_column: Optional[ColumnLike] = None,
    ) -> None:
        # Reference to the source (raw) table:
        self.source_table = source_table
        self.source_name = source_table.name
        self._kumo_project_id = source_table.connector.kumo_project_id

        # Columns. Note that there is no distinction between columns treated as
        # features and those treated as constraints at this stage. The
        # treatment of columns as "feature" or "schema-only" columns will be
        # decided at the model plan stage (e.g. by encoding as `Null()`):
        self._columns: Dict[str, Column] = {}

        # Basic schema. This information is defined at the table level:
        self._primary_key: Optional[str] = None
        self._time_column: Optional[str] = None
        self._end_time_column: Optional[str] = None
        self._discover_sql_columns = False

        # Update values:
        if columns is None:
            columns = list(source_table.column_dict.values())
        for col in (columns or []):
            if isinstance(col, SourceColumn):
                col = Column(
                    name=col.name,
                    stype=col.stype,
                    dtype=col.dtype,
                    multicategorical_delimiter=(
                        col.multicategorical_delimiter),
                )
            self.add_column(Column._cast(col))
        self.primary_key = Column._cast(primary_key)
        self.time_column = Column._cast(time_column)
        self.end_time_column = Column._cast(end_time_column)

        # Cached from backend. Note there is no such thing as a table resource
        # as tables are only persisted in the context of a graph. However,
        # table snapshot resources exist, as tables can be ingested and have
        # data fetched:
        self._table_snapshot_id: Optional[TableSnapshotID] = None

    @property
    def kumo_project_id(self) -> str:
        """Return the immutable Kumo project owning this table and source."""
        return self._kumo_project_id

    @staticmethod
    def from_source_table(
        source_table: SourceTable,
        column_names: Optional[List[str]] = None,
        primary_key: Optional[str] = None,
        time_column: Optional[str] = None,
        end_time_column: Optional[str] = None,
    ) -> 'Table':
        r"""Creates a Kumo Table from a source table. If no column names are
        specified, all source columns are included by default.

        Args:
            source_table: The :class:`~kumoai_studio.connector.SourceTable` object
                that this table is constructed on.
            column_names: A list of columns to include from the source table;
                if not specified, all columns are included by default.
            primary_key: The name of the primary key of this table, if it
                exists.
            time_column: The name of the time column of this table, if it
                exists.
            end_time_column: The name of the end time column of this table, if
                it exists.
        """
        cols = [
            Column(
                name,
                col.stype,
                col.dtype,
                multicategorical_delimiter=(col.multicategorical_delimiter),
            ) for name, col in source_table.column_dict.items()
            if (name in column_names if column_names is not None else True)
        ]
        out = Table(source_table, cols)
        out.primary_key = Column._cast(primary_key)
        out.time_column = Column._cast(time_column)
        out.end_time_column = Column._cast(end_time_column)
        return out

    @staticmethod
    def from_sql(
        connector: Connector,
        sql: str,
        columns: Optional[List[ColumnLike]] = None,
        primary_key: Optional[str] = None,
        time_column: Optional[str] = None,
        end_time_column: Optional[str] = None,
    ) -> 'Table':
        r"""Creates a Kumo Table from a read-only ``SELECT`` executed in a
        warehouse connector (pushdown). The query runs in the customer's
        warehouse and only its result set is ingested.

        No warehouse query runs at definition time; the output schema is
        resolved lazily on validation. Provide :obj:`columns` to declare the
        schema (and dtypes/stypes) up front; otherwise call
        :meth:`infer_metadata` (or validate the table) before use to resolve
        the query's output columns.

        Args:
            connector: The warehouse :class:`~kumoai_studio.connector.Connector` the
                query runs against (Snowflake, Databricks, or BigQuery).
            sql: The read-only ``SELECT`` statement.
            columns: The output columns to include; if unset, columns are
                resolved on validation.
            primary_key: The name of the primary key of this table, if it
                exists. A composite primary key can be expressed as a single
                computed column in the SQL.
            time_column: The name of the time column of this table, if it
                exists.
            end_time_column: The name of the end time column of this table, if
                it exists.

        .. note::
            The query runs in your own warehouse through the connector's
            credentials, and semantic types are inferred from a bounded
            sample of its result rather than from every row. Passing
            :obj:`columns` fixes the schema, which is what makes the
            resulting graph's ID independent of what the sample happened to
            contain -- but the sample still runs, so the query is still
            executed once against your warehouse. On BigQuery a ``LIMIT``
            does not reduce a scan, so a query the warehouse estimates above
            ten gibibytes is not sampled at all and semantic types fall back
            to the data type.
        """
        if columns == []:
            raise ValueError(
                "columns must be omitted for discovery or contain at least "
                "one Custom SQL output column.")
        out = Table(
            source_table=connector.sql(sql),
            columns=columns if columns is not None else [],
            primary_key=primary_key,
            time_column=time_column,
            end_time_column=end_time_column,
        )
        out._discover_sql_columns = columns is None
        return out

    def print_definition(self) -> None:
        r"""Prints the full definition for this table; this definition can be
        copied-and-pasted verbatim to re-create this table.
        """
        pkey_name = (repr(self.primary_key.name)
                     if self.primary_key is not None else "None")
        t_name = (repr(self.time_column.name)
                  if self.time_column is not None else "None")
        et_name = (repr(self.end_time_column.name)
                   if self.end_time_column is not None else "None")
        col_dict = "\n".join([f'    {c},' for c in self.columns])
        if isinstance(self.source_table, SQLSourceTable):
            columns = ('' if self._discover_sql_columns else
                       f'  columns=[\n{col_dict}\n  ],\n')
            print(f'{self.__class__.__name__}.from_sql(\n'
                  f'  connector={self.source_table.connector},\n'
                  f'  sql={self.source_table.sql!r},\n'
                  f'  primary_key={pkey_name},\n'
                  f'  time_column={t_name},\n'
                  f'  end_time_column={et_name},\n'
                  f'{columns}'
                  f')')
            return
        source_repr = f"{self.source_table.connector}[\"{self.source_name}\"]"
        print(f'{self.__class__.__name__}(\n'
              f'  source_table={source_repr},\n'
              f'  primary_key={pkey_name},\n'
              f'  time_column={t_name},\n'
              f'  end_time_column={et_name},\n'
              f'  columns=[\n{col_dict}\n'
              f'  ],\n'
              f')')

    # Data column #############################################################

    def has_column(self, name: str) -> bool:
        r"""Returns True if this table has column with name :obj:`name`; False
        otherwise.
        """
        return name in self._columns

    def column(self, name: str) -> Column:
        r"""Returns the data column named with name :obj:`name` in this table,
        or raises a :obj:`KeyError` if no such column is present.

        Raises:
            :class:`KeyError`
                if :obj:`name` is not present in this table.
        """
        if not self.has_column(name):
            raise KeyError(
                f"Column '{name}' not found in table '{self.source_name}'")
        return self._columns[name]

    @property
    def columns(self) -> List[Column]:
        r"""Returns a list of :class:`~kumoai_studio.Column` objects that represent
        the columns in this table.
        """
        return list(self._columns.values())

    def add_column(self, *args: Any, **kwargs: Any) -> None:
        r"""Adds a :obj:`~kumoai_studio.graph.Column` to this table. A column can
        either be added by directly specifying its configuration in this call,
        or by creating a Column object and passing it as an argument.

        Example:
            >>> import kumoai_studio
            >>> table = kumoai_studio.Table(source_table=...)  # doctest: +SKIP
            >>> table.add_column(name='col1', dtype='string')  # doctest: +SKIP
            >>> table.add_column(kumoai_studio.Column('col2', 'int'))  # doctest: +SKIP

        .. # noqa: E501
        """
        col = Column._cast(*args, **kwargs)
        if col is None:
            raise ValueError("Cannot add a 'None' column to a table.")
        if self.has_column(col.name):
            self._columns[col.name].update(col)
        else:
            self._columns[col.name] = col

    def remove_column(self, name: str) -> Self:
        r"""Removes a :obj:`~kumoai_studio.graph.Column` from this table.

        Raises:
            :class:`KeyError`
                if :obj:`name` is not present in this table.
        """
        if not self.has_column(name):
            raise KeyError(
                f"Column '{name}' not found in table '{self.source_name}'")

        if self.has_primary_key() and self._primary_key == name:
            self.primary_key = None
        if self.has_time_column() and self._time_column == name:
            self.time_column = None
        if self.has_end_time_column() and self._end_time_column == name:
            self.end_time_column = None
        del self._columns[name]
        return self

    # Primary key #############################################################

    def has_primary_key(self) -> bool:
        r"""Returns :obj:`True` if this table has a primary key; :obj:`False`
        otherwise.
        """
        return self._primary_key is not None

    @property
    def primary_key(self) -> Optional[Column]:
        r"""The primary key column of this table.

        The getter returns the primary key column of this table, or None if no
        such primary key is present.

        The setter sets a column as a primary key on this table, and raises a
        :class:`ValueError` if the primary key has a non-ID semantic type.
        """
        if not self.has_primary_key():
            return None
        assert self._primary_key is not None
        return self._columns[self._primary_key]

    @primary_key.setter
    def primary_key(self, *args: Any, **kwargs: Any) -> Self:
        col = Column._cast(*args, **kwargs)
        if col is None:
            self._primary_key = None
            return self

        if col.stype is not None and col.stype != Stype.ID:
            raise ValueError(
                f"The semantic type of a primary key must be 'ID' (got "
                f"{col.stype}).")

        col.stype = Stype.ID
        self.add_column(col)
        self._primary_key = col.name
        return self

    # Time column #############################################################

    def has_time_column(self) -> bool:
        r"""Returns :obj:`True` if this table has a time column; :obj:`False`
        otherwise.
        """
        return self._time_column is not None

    @property
    def time_column(self) -> Optional[Column]:
        r"""The time column of this table.

        The getter returns the time column of this table, or :obj:`None` if no
        such time column is present.

        The setter sets a column as a time column on this table, and raises a
        :class:`ValueError` if the time column is the same as the end time
        column, or has a non-timestamp semantic type.
        """
        if not self.has_time_column():
            return None
        assert self._time_column is not None
        return self._columns[self._time_column]

    @time_column.setter
    def time_column(self, *args: Any, **kwargs: Any) -> Self:
        col = Column._cast(*args, **kwargs)
        if col is None:
            self._time_column = None
            return self

        if self.has_end_time_column() and self._end_time_column == col.name:
            raise ValueError(f"Cannot set the time column ('{col.name}') "
                             f"to be the same as the end time column "
                             f"('{self._end_time_column}')")

        if col.stype is not None and col.stype != Stype.timestamp:
            raise ValueError(
                f"The semantic type of a time column must be 'timestamp' (got "
                f"{col.stype}).")

        col.stype = Stype.timestamp
        self.add_column(col)
        self._time_column = col.name
        return self

    # End time column #########################################################

    def has_end_time_column(self) -> bool:
        r"""Returns :obj:`True` if this table has an end time column;
        :obj:`False` otherwise.
        """
        return self._end_time_column is not None

    @property
    def end_time_column(self) -> Optional[Column]:
        r"""The end time column of this table.

        The getter returns the end time column of this table, or :obj:`None` if
        no such column is present.

        The setter sets a column as a time column on this table, and raises a
        :class:`ValueError` if the time column is the same as the end time
        column, or has a non-timestamp semantic type.
        """
        if not self.has_end_time_column():
            return None
        assert self._end_time_column is not None
        return self._columns[self._end_time_column]

    @end_time_column.setter
    def end_time_column(self, *args: Any, **kwargs: Any) -> Self:
        col = Column._cast(*args, **kwargs)
        if col is None:
            self._end_time_column = None
            return self

        if self.has_time_column() and self._time_column == col.name:
            raise ValueError(f"Cannot set the end time column ('{col.name}') "
                             f"to be the same as the time column "
                             f"('{self._time_column}')")

        if col.stype is not None and col.stype != Stype.timestamp:
            raise ValueError(
                f"The semantic type of an end time column must be 'timestamp' "
                f"(got {col.stype}).")

        col.stype = Stype.timestamp
        self.add_column(col)
        self._end_time_column = col.name
        return self

    # Metadata ################################################################

    @property
    def metadata(self) -> pd.DataFrame:
        r"""Returns a :class:`~pandas.DataFrame` object containing Kumo metadata
        information about the columns in this table.

        The returned dataframe has columns ``name``, ``dtype``, ``stype``,
        ``multicategorical_delimiter``, ``is_primary_key``,
        ``is_time_column``, and ``is_end_time_column``, which provide an
        aggregate view of the properties of the columns of this table.

        Example:
            >>> import kumoai_studio
            >>> table = kumoai_studio.Table(source_table=...)  # doctest: +SKIP
            >>> table.add_column(name='CustomerID', dtype='float64', stype='ID')  # doctest: +SKIP
            >>> table.metadata  # doctest: +SKIP
                name        dtype       stype    is_time_column  is_end_time_column
            0   CustomerID  float64     ID       False           False
        """  # noqa: E501
        items = self._columns.items()
        col_names: List[str] = [i[0] for i in items]
        cols: List[Column] = [i[1] for i in items]

        return pd.DataFrame({
            'name':
            pd.Series(dtype=str, data=col_names),
            'dtype':
            pd.Series(
                dtype=str, data=[
                    c.dtype.value if c.dtype is not None else None
                    for c in cols
                ]),
            'stype':
            pd.Series(
                dtype=str, data=[
                    c.stype.value if c.stype is not None else None
                    for c in cols
                ]),
            'multicategorical_delimiter':
            pd.Series(
                dtype=str,
                data=[c.multicategorical_delimiter for c in cols],
            ),
            'is_primary_key':
            pd.Series(dtype=bool, data=[self.primary_key == c for c in cols]),
            'is_time_column':
            pd.Series(dtype=bool, data=[self.time_column == c for c in cols]),
            'is_end_time_column':
            pd.Series(dtype=bool,
                      data=[self.end_time_column == c for c in cols]),
        })

    def infer_metadata(self, inplace: bool = True) -> Self:
        r"""Infers all metadata for this table's specified columns, including
        the column data types, semantic types, timestamp formats, primary keys,
        and time/end-time columns

        Args:
            inplace: Whether the method should modify the table columns in
                place or return a new :class:`~kumoai_studio.graph.Table` object.

        .. note::
            This method in-place modifies the Table object if `inplace = True`,
            and returns a copy if ``inplace = False``.
        """
        if isinstance(self.source_table, SQLSourceTable):
            out = self if inplace else copy.deepcopy(self)
            return out.validate()
        col_requests: List[api.ColumnMetadataRequest] = []
        for col in self.columns:
            col_requests.append(
                api.ColumnMetadataRequest(
                    name=col.name,
                    stype=col.stype,
                    dtype=col.dtype,
                    timestamp_format=col.timestamp_format,
                    multicategorical_delimiter=(
                        col.multicategorical_delimiter),
                ))

        pk_name: Optional[str] = None
        if self.primary_key is not None:
            pk_name = self.primary_key.name

        tc_name: Optional[str] = None
        if self.time_column is not None:
            tc_name = self.time_column.name

        request = api.TableMetadataRequest(
            cols=col_requests,
            source_table=self.source_table._to_api_source_table(),
            pkey=pk_name,
            time_col=tc_name,
        )

        response = cloud_backend().table.infer_metadata(
            request, project_id=self.kumo_project_id)
        inferred_cols: Dict[str, api.Column] = {
            col.name: col
            for col in response.cols
        }

        # Handle inplace:
        out = self
        if not inplace:
            out = copy.deepcopy(self)

        # TODO(manan): what happens when the ts format is set based on an
        # override?
        for col in out.columns:
            inferred_col = inferred_cols[col.name]

            col.dtype = col.dtype or inferred_col.dtype
            col.stype = col.stype or inferred_col.stype
            col.timestamp_format = (col.timestamp_format
                                    or inferred_col.timestamp_format)
            col.multicategorical_delimiter = (
                col.multicategorical_delimiter
                or inferred_col.multicategorical_delimiter)

        if not out.has_primary_key() and response.pkey is not None:
            out.primary_key = response.pkey
        if not out.has_time_column() and response.time_col is not None:
            out.time_column = response.time_col

        # Override for Kumo backend, always:
        if out.has_primary_key():
            assert out.primary_key is not None
            out.primary_key.stype = Stype.ID

        if out.has_time_column():
            assert out.time_column is not None
            out.time_column.stype = Stype.timestamp

        if out.has_end_time_column():
            assert out.end_time_column is not None
            out.end_time_column.stype = Stype.timestamp

        return out

    def _validate_definition(self) -> None:
        for col in self.columns:
            if col.dtype is None or col.stype is None:
                raise ValueError(
                    f"Column {col.name} is not fully specified. Please "
                    f"specify this column's data type and semantic type "
                    f"before proceeding. {col.name} currently has a "
                    f"data type of {col.dtype} and semantic type of "
                    f"{col.stype}.")
            col._validate_multicategorical_delimiter()

    def _apply_resolved_sql(self, definition: api.TableDefinition) -> None:
        source = definition.source_table
        if not isinstance(source, CustomSQLSourceTable):
            raise ValueError("The server returned a non-SQL source.")
        current = self.source_table
        assert isinstance(current, SQLSourceTable)
        if (source.connector_id != current.connector.id
                or source.sql != current.sql or source.connector_type is None):
            raise ValueError(
                "The server returned SQL metadata for a different source.")
        columns = definition.cols
        names = [col.name for col in columns]
        if len(names) != len(set(names)):
            raise ValueError(
                "Custom SQL output columns must have unique aliases.")
        if not self._discover_sql_columns and names != list(self._columns):
            raise ValueError(
                "The server returned a different explicit column selection.")
        requested_roles = (self._primary_key, self._time_column,
                           self._end_time_column)
        resolved_roles = (definition.pkey, definition.time_col,
                          definition.end_time_col)
        if any(requested is not None and resolved != requested for requested,
               resolved in zip(requested_roles, resolved_roles)):
            raise ValueError("The server changed a requested column role.")
        if any(role is not None and role not in names
               for role in (definition.pkey, definition.time_col,
                            definition.end_time_col)):
            raise ValueError(
                "The server returned a schema missing a role column.")
        resolved = {
            col.name:
            Column(col.name, col.stype, col.dtype, col.timestamp_format,
                   col.multicategorical_delimiter)
            for col in columns
        }
        resolved_source = SQLSourceTable(
            sql=source.sql, connector=current.connector, name=source.table,
            connector_type=source.connector_type,
            referenced_tables=source.referenced_tables)
        # Build and validate the complete candidate before replacing any
        # client state. Passing the resolved Column objects through the normal
        # setters preserves their type checks and role-exclusion rules.
        candidate = Table(
            source_table=resolved_source, columns=list(resolved.values()),
            primary_key=(resolved.get(definition.pkey)
                         if definition.pkey is not None else None),
            time_column=(resolved.get(definition.time_col)
                         if definition.time_col is not None else None),
            end_time_column=(resolved.get(definition.end_time_col)
                             if definition.end_time_col is not None else None))
        candidate._validate_definition()
        self.source_table = candidate.source_table
        self.source_name = candidate.source_name
        self._columns = candidate._columns
        self._primary_key = candidate._primary_key
        self._time_column = candidate._time_column
        self._end_time_column = candidate._end_time_column
        self._discover_sql_columns = False

    def _resolve_sql(
        self,
        timeout: Optional[Tuple[float, float]] = None,
    ) -> None:
        if not isinstance(self.source_table, SQLSourceTable):
            raise TypeError("SQL resolution requires a Custom SQL table.")
        hints = None if self._discover_sql_columns else [
            api.ColumnMetadataRequest(
                col.name, col.stype, col.dtype, col.timestamp_format,
                col.multicategorical_delimiter) for col in self.columns
        ]
        result = cloud_backend().table.resolve_sql(
            api.SqlTableResolveRequest(
                source_table=self.source_table._to_api_source_table(),
                column_hints=hints, pkey=self._primary_key,
                time_col=self._time_column,
                end_time_col=self._end_time_column),
            project_id=self.kumo_project_id, timeout=timeout)
        if result.errors:
            raise ValueError(' '.join(error.message
                                      for error in result.errors))
        if result.table_definition is None:
            raise ValueError("The server returned no resolved definition.")
        self._apply_resolved_sql(result.table_definition)

    def validate(self, verbose: bool = True,
                 timeout: Optional[Tuple[float, float]] = None) -> Self:
        r"""Validates a Table to ensure that all relevant metadata is specified
        for a table to be used in a downstream :class:`~kumoai_studio.graph.Graph` and
        :class:`~kumoai_studio.pquery.PredictiveQuery`.

        Concretely, validation ensures that all columns have valid data and
        semantic types with respect to the table's source data. For example,
        if a text column is assigned a ``dtype`` of ``"int"``, this method
        raises an exception detailing the mismatch. Similarly, if a column
        cannot be cast from its source data type to the specified data type
        (*e.g.* ``"int"`` to ``"binary"``), this method raises an exception.

        Custom SQL tables are resolved against their warehouse as part of the
        same validation call. Their complete server-owned schema, roles, and
        lineage are applied atomically only after validation succeeds.

        .. warning::
            Data type validation is performed on a sample of table data. A
            valid response may not indicate your entire data source is
            configured correctly.

        Args:
            verbose: Whether to log non-error output of this validation.
            timeout: Optional connect and read timeout for SQL resolution.

        Example:
            >>> import kumoai_studio
            >>> table = kumoai_studio.Table(...)  # doctest: +SKIP
            >>> table.validate()  # doctest: +SKIP

        Raises:
            ValueError: If validation fails.
        """
        if isinstance(self.source_table, SQLSourceTable):
            self._resolve_sql(timeout=timeout)
            return self

        self._validate_definition()

        # Actual heavy lifting:
        resp = cloud_backend().table.validate_table(
            api.TableValidationRequest(self._to_api_table_definition()),
            project_id=self.kumo_project_id)
        if not resp.ok:
            raise ValueError(resp.error_message())
        if verbose:
            if resp.empty():
                logger.info("Table %s is configured correctly.",
                            self.source_name)
            else:
                logger.warning(resp.message())
        return self

    # Snapshot ################################################################

    @property
    def snapshot_id(self) -> Optional[snapshot_api.TableSnapshotID]:
        r"""Returns the snapshot ID of this table's snapshot, if a snapshot
        has been taken. Returns `None` otherwise.

        .. warning::
            This property currently only returns a snapshot ID if a snapshot
            has been taken *in this session.*
        """
        return self._table_snapshot_id

    def snapshot(
        self,
        *,
        force_refresh: bool = False,
        non_blocking: bool = False,
    ) -> snapshot_api.TableSnapshotID:
        r"""Takes a *snapshot* of this table's underlying data, and returns a
        unique identifier for this snapshot.

        The *snapshot* functionality allows one to freeze a table in time, so
        that underlying data changes do not require Kumo to re-process the
        data. This allows for fast iterative machine learning model
        development, on a consistent set of input data.

        .. warning::
            Please note that snapshots are intended to freeze tables in
            time, and not to allow for "time-traveling" to an earlier version
            of data with a prior snapshot. In particular, this means that a
            table can only have one version of a snapshot, which represents
            the latest snapshot taken for that table.

        .. note::
            If you are using Kumo as a Snowpark Container Services native
            application, please note that *snapshot* is a no-op for all
            non-view tables.

        Args:
            force_refresh: Indicates whether a snapshot should be taken, if one
                already exists in Kumo. If :obj:`False`, a previously existing
                snapshot may be re-used. If :obj:`True`, a new snapshot is
                always taken.
            non_blocking: Whether this operation should return immediately
                after creating the snapshot, or await completion of the
                snapshot. If :obj:`True`, the snapshot will proceed in the
                background, and will be used for any downstream job.
        """
        if self._table_snapshot_id is None or force_refresh:
            self._table_snapshot_id = (cloud_backend().table.create_snapshot(
                table_definition=self._to_api_table_definition(),
                refresh_source=True,
                project_id=self.kumo_project_id,
            ))

            stage = snapshot_api.TableSnapshotStage.INGEST
            resource: snapshot_api.TableSnapshotResource = (
                cloud_backend().table.get_snapshot(
                    snapshot_id=self._table_snapshot_id,
                    project_id=self.kumo_project_id))

            if not non_blocking:
                status = resource.stages[stage].status
                while not status.is_terminal:
                    # TODO(manan, siyang): fix start and end time
                    resource = (cloud_backend().table.get_snapshot(
                        snapshot_id=self._table_snapshot_id,
                        project_id=self.kumo_project_id))
                    logger.info(
                        "Awaiting snapshot completion: current status is %s ",
                        status)
                    time.sleep(_DEFAULT_INTERVAL_S)
                    status = resource.stages[stage].status

                state = resource.stages[stage]
                status = state.status
                warnings = "\n".join([
                    f"{i}. {message}"
                    for i, message in enumerate(state.warnings)
                ])
                error = state.error
                if status == JobStatus.FAILED:
                    raise RuntimeError(
                        f"Table snapshot with identifier "
                        f"{self._table_snapshot_id} failed, with error "
                        f"{error} and warnings {warnings}")
                if len(state.warnings) > 0:
                    logger.warning(
                        "Table snapshot completed with the following "
                        "warnings: %s", warnings)

        # <prefix>@<data_version>
        assert self._table_snapshot_id is not None
        return self._table_snapshot_id

    def get_stats(
        self,
        wait_for: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        r"""Returns all currently computed statistics on the latest snapshot of
        this table. If a snapshot on this table has not been taken, this method
        will take a snapshot.

        .. note::
            Table statstics are computed in multiple stages after ingestion is
            complete. These stages are called *minimal* and *full*; minimal
            statistics are always computed before full statistics.

        Args:
            wait_for: Whether this operation should block on the existence of
                statistics availability. This argument can take one of three
                values: :obj:`None`, which indicates that the method should
                return immediately with whatever statistics are present,
                :obj:`"minimal"`, which indicates that the method should return
                the when the minimum, maximum, and fraction of NA values
                statistics are present, or :obj:`"full"`, which indicates that
                the method should return when all computed statistics are
                present.
        """
        assert wait_for is None or wait_for in {"minimal", "full"}

        # Attempt to snapshot, use cached snapshot if possible:
        if not self._table_snapshot_id:
            self.snapshot(force_refresh=False, non_blocking=False)
        assert self._table_snapshot_id is not None

        # Fetch resource:
        resource: snapshot_api.TableSnapshotResource = (
            cloud_backend().table.get_snapshot(
                snapshot_id=self._table_snapshot_id,
                project_id=self.kumo_project_id))

        # Wait for a stage, if we need to:
        if wait_for:
            if wait_for == "minimal":
                stage = snapshot_api.TableSnapshotStage.MIN_COL_STATS
            else:
                stage = snapshot_api.TableSnapshotStage.FULL_COL_STATS

            status = resource.stages[stage].status
            while not status.is_terminal:
                resource = (cloud_backend().table.get_snapshot(
                    snapshot_id=self._table_snapshot_id,
                    project_id=self.kumo_project_id))
                logger.info(
                    "Awaiting %s column statistics: current status is %s ",
                    wait_for, status)
                time.sleep(_DEFAULT_INTERVAL_S)
                status = resource.stages[stage].status

        # Write out statistics:
        out = {}
        col_stats = resource.column_stats
        for stat in (col_stats or []):
            out[stat.column_name] = stat.stats
        return out

    # Persistence #############################################################

    def _to_api_table_definition(self) -> api.TableDefinition:
        pk_name: Optional[str] = None
        if self.primary_key is not None:
            pk_name = self.primary_key.name

        tc_name: Optional[str] = None
        if self.time_column is not None:
            tc_name = self.time_column.name

        etc_name: Optional[str] = None
        if self.end_time_column is not None:
            etc_name = self.end_time_column.name

        cols = []
        for col in self.columns:
            if col.stype is None or col.dtype is None:
                missing = 'semantic' if col.stype is None else 'data'
                field = 'stype' if col.stype is None else 'dtype'
                raise ValueError(
                    f"Column {col.name!r} of table {self.source_name!r} has "
                    f"no {missing} type, so this table cannot be saved. Call "
                    f"table.infer_metadata() to fill it in from a sample of "
                    f"the source, or set the column explicitly with "
                    f"table[{col.name!r}].update({field}=...).")
            cols.append(
                api.Column(
                    name=col.name,
                    stype=col.stype,
                    dtype=col.dtype,
                    timestamp_format=col.timestamp_format,
                    multicategorical_delimiter=(
                        col.multicategorical_delimiter),
                ))

        return api.TableDefinition(
            cols=cols,
            source_table=self.source_table._to_api_source_table(),
            pkey=pk_name,
            time_col=tc_name,
            end_time_col=etc_name,
        )

    @staticmethod
    def _from_api_table_definition(
        table_definition: api.TableDefinition,
        project_id: Optional[str] = None,
    ) -> 'Table':
        return Table(
            source_table=SourceTable._from_api_table_definition(
                table_definition, project_id=project_id),
            columns=[
                Column(
                    name=col.name,
                    stype=col.stype,
                    dtype=col.dtype,
                    timestamp_format=col.timestamp_format,
                    multicategorical_delimiter=(
                        col.multicategorical_delimiter),
                ) for col in table_definition.cols
            ],
            primary_key=table_definition.pkey,
            time_column=table_definition.time_col,
            end_time_column=table_definition.end_time_col,
        )

    def save(self, name: str) -> Union[TableID, str]:
        r"""Associates this table with a unique name, that can later be
        used to fetch the table either in the Kumo UI or in the Kumo SDK
        with method :meth:`~kumoai_studio.Table.load`.

        Args:
            name: The name to associate with this table definition. If the
                name is already associated with another table, that table will
                be overridden.

        Raises:
            ValueError: If :obj:`name` is empty or whitespace-only.

        Example:
            >>> import kumoai_studio
            >>> table = kumoai_studio.Table(...)  # doctest: +SKIP
            >>> table_id = table.save("name")  # doctest: +SKIP
            >>> loaded = kumoai_studio.Table.load("name")  # doctest: +SKIP
        """
        if not name or not name.strip():
            raise ValueError(
                'A table name is required: call table.save("<name>"). A '
                'table saved without a name is hidden from the table list '
                'and cannot be deleted.')
        self.validate(verbose=False)
        template_resource = cloud_backend().table.get_table_if_exists(
            table_id_or_name=name, project_id=self.kumo_project_id)

        if template_resource is not None:
            config = self._from_api_table_definition(
                template_resource.table, project_id=self.kumo_project_id)
            logger.warning(
                ("Table template %s already exists, with configuration %s. "
                 "This template will be overridden with configuration %s."),
                name, str(config), str(self))

        # TODO(manan): fix
        _id = cloud_backend().table.create_table(
            table_def=self._to_api_table_definition(),
            name_alias=name,
            force_rename=True,
            project_id=self.kumo_project_id,
        )
        return f"table-{_id.split('-', maxsplit=1)[1]}"

    @classmethod
    def load(
        cls,
        table_id_or_template: str,
        project_id: Optional[str] = None,
    ) -> 'Table':
        r"""Load a table by ID or template name from one exact project.

        Args:
            table_id_or_template: Table ID or project-local template name.
            project_id: Exact project to search. ``None`` uses the active
                project. The returned table snapshots this project.

        Returns:
            A :class:`Table` containing the loaded columns and metadata.
        """
        api = cloud_backend().table
        if project_id is None:
            from kumoai_studio import _require_project
            _require_project()
            assert global_state._project_id is not None
            project_id = global_state._project_id
        res = api.get_table_if_exists(table_id_or_template,
                                      project_id=project_id)
        if not res:
            raise ValueError(f"Table {table_id_or_template} was not found.")
        out = cls._from_api_table_definition(res.table, project_id=project_id)
        return out

    # Class properties ########################################################

    def __hash__(self) -> int:
        return hash(
            tuple(self.columns +
                  [self.primary_key, self.time_column, self.end_time_column]))

    def __contains__(self, name: str) -> bool:
        return self.has_column(name)

    def __getitem__(self, name: str) -> Column:
        return self.column(name)

    def __delitem__(self, name: str) -> None:
        self.remove_column(name)

    def __repr__(self) -> str:
        col_names = str(list(self._columns.keys())).replace("'", "")
        pkey_name = (self.primary_key.name
                     if self.primary_key is not None else "None")
        t_name = (self.time_column.name
                  if self.time_column is not None else "None")
        et_name = (self.end_time_column.name
                   if self.end_time_column is not None else "None")
        return (f'{self.__class__.__name__}(\n'
                f'  source_name={self.source_name},\n'
                f'  data_source={self.source_table.connector.name},\n'
                f'  columns={col_names},\n'
                f'  primary_key={pkey_name},\n'
                f'  time_column={t_name},\n'
                f'  end_time_column={et_name},\n'
                f')')
