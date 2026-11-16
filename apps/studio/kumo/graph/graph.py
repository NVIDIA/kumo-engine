# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import os.path as osp
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from io import BytesIO
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import inflect
import pandas as pd
from kumo_core.utils.names import (
    fully_qualified_table_name,
    get_data_source,
    is_fully_qualified_name,
)
from kumo_core.utils.registry import register
from kumoapi.encoder import Encoder
from torch_geometric.typing import EdgeType

from async_utils.sync import blocking_method
from io_utils import fsutil, kumo_dataclass, logging
from io_utils.json import load_json, save_json
from kumo.config import ConfigMixin, GraphConfig
from kumo.connector import Connector
from kumo.connector.pandas_connector import PandasConnector
from kumo.data.backend_cache import ExactStats
from kumo.graph import ForeignKey, Table, TimeColumn
from kumo.graph.column import BaseColumn, Column, PrimaryKey
from kumo.graph.mixin import _HasData, raise_post_materialization
from kumo.graph.utils import check_for_exception, persist_pandas_connector
from kumo.utils.profiling import Timer
from kumo.utils.testing import pytest_only
from kumo.utils.visualization import visualize_graph
from kumo.validation.encoder_validation import (
    ColumnNotInTableError,
    TableNotInGraphError,
)
from kumo.validation.validation import ValidationException
from monitoring import tracing

logger = logging.getLogger(__name__)

GRAPH_THREAD_POOL = ThreadPoolExecutor(thread_name_prefix='Graph-Threadpool')


@kumo_dataclass
class ColumnInfo:
    r"""Dataclass to hold FQ table name and column name of an inferred group
    in the graph. Used to create inferred links."""
    table: str
    col: str


@kumo_dataclass
class GroupInfo:
    r"""Dataclass that defines an inferred group. pkey holds the column
    details about primary key of the group and fkeys are the list of table
    information about probable fkeys. Used to create inferred links."""
    pkey: ColumnInfo
    fkeys: List[ColumnInfo]


@register(data_cls=GraphConfig)
class Graph(ConfigMixin, _HasData):
    r"""A graph is a configuration object that represents a graph in Kumo. A
    graph consists of tables and their associated data sources; it defines the
    relationships (links) between these tables.

    .. code-block:: python

        import kumo

        graph = kumo.Graph(name='my-graph')

        graph.connect(kumo.CSVConnector(name='s3'))
        graph.connect(kumo.SnowflakeConnector(name='sf'))

        graph.add('A', connector_name='s3', pkey, fkey, cols, ...)
        graph.add('B', connector_name='snowflake', pkey, fkey, cols, ...)

    Args:
        name (str, optional): The name of the graph. (default: :obj:`None`)
        data_source_dict (Dict[str, Connector], optional): The data sources and
            their corresponding connectors of the graph. (default: :obj:`None`)
        tables (List[Table], optional): The tables in the graph.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        name: Optional[str] = None,
        data_source_dict: Optional[Dict[str, Connector]] = None,
        tables: Optional[List[Table]] = None,
    ):
        self._name = name

        # TODO(kiriti): Remove data_source_dict and take a list of connectors.
        # Convert list of connectors to data_source_dict.
        # Note: data_source is an internal terminology for connector_name in
        # the public API.
        self._data_source_dict = data_source_dict or {}

        # Stores all metapaths that need to be added during materialization.
        # Note that metapaths are currently unused:
        self._metapaths = []
        self._metapath_kwargs = {}
        self._intra_metapaths = []
        self._intra_metapath_kwargs = {}

        # Map of fully-qualified table names to `Table` objects.
        self._tables: Dict[str, Table] = {}
        for table in tables or []:
            self._add(table)

    # Python built-ins ########################################################

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Graph):
            return False

        cfg = asdict(self.config())
        other_cfg = asdict(other.config())

        del cfg['tables']
        del other_cfg['tables']
        return cfg == other_cfg and self.tables == other.tables

    def __len__(self) -> int:
        return self.num_tables

    def __contains__(self, name: str) -> bool:
        return self.has_table(name)

    def _get_nested_index(
        self,
        name: Any,
    ) -> Tuple[Tuple[str, str], Optional[List[Any]]]:
        r"""Parses a given :obj:`name` and tries to convert it into
        :obj:`((data_source, table_name), nested_index)`.
        Can parse the following formats:

        * :obj:`{table_name}`
        * :obj:`{data_source}.{table_name}`
        * :obj:`{table_name}.{col_name}`
        * :obj:`{data_source}.{table_name}.{col_name}`
        * :obj:`({table_name}, {col_name})`
        * :obj:`({data_soure}, {table_name}, {col_name})`

        :obj:`nested_index` refers to additional levels of access, e.g., the
        column name, or indices/slices/ellipsis to acces the data.

        .. note:

            Prefers :obj:`{table_name}.{col_name}` over
            :obj:`{data_source}.{table_name}`.
        """
        original_name = name
        if isinstance(name, str):
            if '.' not in name:
                return (None, name), None
            name = tuple(name.split('.'))

        if isinstance(name, (tuple, list)):
            # If only a single name, assume table name:
            if len(name) == 1:
                return (None, name[0]), None

            # If more than two strings, assume
            # {data_source}.{table_name}.{col_name}:
            if len(name) > 2 and all([isinstance(v, str) for v in name]):
                return (name[0], name[1]), name[2:]

            # If two strings, resolve ambiguity by checking column first, then
            # table
            maybe_cols = [name[1]] if isinstance(name[1], str) else name[1]
            maybe_cols = [] if not isinstance(maybe_cols,
                                              (list, tuple)) else maybe_cols
            is_valid_columns = (self.has_table(name[0]) and all(
                [self.table(name[0]).has_col(col) for col in maybe_cols]))
            is_valid_table = self.has_data_source(name[0]) and self.has_table(
                name[1], data_source=name[0])

            if is_valid_columns:
                if is_valid_table:
                    logger.warning(
                        f"Ambiguous access: {original_name} is a valid "
                        "column as well as a valid table. Assuming it is a "
                        "column.")
                return (None, name[0]), name[1:]
            if is_valid_table:
                return (name[0], name[1]), None

            # Fallback: Let `__getitem__` handle any errors gracefully:
            return (None, name[0]), name[1:]

        raise IndexError(f"Only strings and tuples can be used for "
                         f"indexing (got '{type(name).__name__}')")

    def __getitem__(
        self,
        name: Union[str, Tuple[str, str]],
    ) -> Union[Table, BaseColumn]:
        r"""Provides access to tables (and their data) as well as columns (and
        their data):

        .. code-block:: python

            graph[table_name]  # A table
            graph[table_name, :100]  # First 100 rows of the table
            graph[table_name, col_name]  # A column
            graph[table_name, col_name, :100]  # First 100 values of the column

        In addition, you can also use dot-syntax to access columns:

        .. code-block:: python

            graph[f"{table_name}.{col_name}"]  # A column
        """
        (data_source, table_name), nested_name = self._get_nested_index(name)

        table = self.table(table_name, data_source=data_source)
        return table[nested_name] if nested_name is not None else table

    @raise_post_materialization
    def __delitem__(self, name: Union[str, Tuple[str, str]]):
        (data_source, table_name), nested_name = self._get_nested_index(name)
        if nested_name is None:
            self.remove(table_name, data_source=data_source)
        else:
            del self.table(table_name, data_source=data_source)[nested_name]

    def __iter__(self) -> Iterator:
        for table in self._tables.values():
            yield table

    def __repr__(self) -> str:
        data_sources = str(self.data_sources).replace("'", "")
        table_names = str(self.table_names).replace("'", "")
        return (f'{self.__class__.__name__}(\n'
                f'  name={self.name},\n'
                f'  data_sources={data_sources},\n'
                f'  tables={table_names},\n'
                f')')

    # Copy ####################################################################

    def metacopy(self, name: Optional[str] = None) -> 'Graph':
        r"""Creates a copy of the graph, dropping any materialized data.

        Args:
            name (str, optional): The name of the graph. (default: :obj:`None`)
        """
        return self.__class__(
            name=name or self.name,
            data_source_dict=self.data_source_dict,
            tables=[table.metacopy() for table in self],
        )

    # Graph-Level Properties ##################################################

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def num_tables(self) -> int:
        r"""The number of tables in the graph."""
        return len(self._tables)

    @property
    def tables(self) -> List[Table]:
        r"""The tables in the graph."""
        return list(self._tables.values())

    @property
    def table_names(self) -> List[str]:
        r"""The fully qualified names of tables in the graph."""
        return list(self._tables.keys())

    def has_table(self, name: str, data_source: Optional[str] = None) -> bool:
        r"""Whether a table with name :obj:`name` exists in the graph."""
        try:
            name = self.fully_qualified_table_name(name, data_source)
            return name in self._tables
        except KeyError:
            return False

    # Table-level properties ##################################################

    @property
    def num_data_sources(self) -> int:
        r"""The number of data sources of the graph."""
        return len(self._data_source_dict)

    @property
    def data_sources(self) -> List[str]:
        r"""The data sources of the graph."""
        return list(self._data_source_dict.keys())

    def has_data_source(self, data_source: str) -> bool:
        r"""Whether the :obj:`data_source` exists in the graph."""
        return data_source in self._data_source_dict

    @property
    def data_source_dict(self) -> Dict[str, Connector]:
        return copy.copy(self._data_source_dict)

    @property
    def connectors(self) -> List[Connector]:
        return list(self._data_source_dict.values())

    def connector(self, data_source: str) -> Connector:
        r"""The connector of the data source with name :obj:`name`."""
        return self._data_source_dict[data_source]

    # Accessing tables ########################################################

    def copy_encoders(self, graph: 'Graph'):
        r"""Copy all encoders from `graph` to `self`."""
        for t1 in self.tables:
            for t2 in graph.tables:
                if t1.table_name == t2.table_name:
                    t1.copy_encoders(t2)
                    break

    def fully_qualified_table_name(
        self,
        name: Union[str, Tuple[str, str]],
        data_source: Optional[str] = None,
    ) -> str:
        r"""
        Args:
            name (str, tuple(str, str)): The name of the table.
                Supports these formats for table names and data sources:
                * name={name}, data_source={data_source}
                * name=({data_source}, {name}), data_source=None
                * name={data_source}.{name}, data_source=None
                * name={name}, data_source=None (requires unique table name)
        Returns:
            Fully qualified table name. E.g. "csv.table"
        """
        # If data_source is not provided, infer it based on unique table names.
        if (data_source is None and isinstance(name, str)
                and not is_fully_qualified_name(name)):
            matches = [
                table for table in self.tables if table.table_name == name
            ]
            if len(matches) > 1:
                raise ValidationException(message=(
                    "There exist multiple tables in the graph with name "
                    f"'{name}'. Please also specify the table data "
                    "source."))
            elif len(matches) == 0:
                raise KeyError(
                    f'Table "{name}" is not found in the graph. ',
                    'Please ensure that the correct table name is '
                    'used.')
            data_source = matches[0].data_source
        fq_table_name = fully_qualified_table_name(name, data_source)
        data_source = get_data_source(fq_table_name)

        # Ensure that the data source exists in the graph.
        if data_source not in self._data_source_dict.keys():
            raise ValidationException(message=(
                f"Data source: {data_source} does not exist in the graph."))

        return fq_table_name

    def table(self, name: str, data_source: Optional[str] = None) -> Table:
        r"""The table with name :obj:`name`.

        There are four valid ways to select a table by name:

        .. code-block:: python

            graph.table("table_name")  # Unique table name
            graph.table("data_source.table_name")
            graph.table(("data_source", "name"))
            graph.table("name", data_source="data_source")
        """
        return self._tables[self.fully_qualified_table_name(name, data_source)]

    def column(self, fully_qualified_table_name: str,
               column_name: str) -> BaseColumn:
        r"""The column with name :obj:`column_name`.
        """
        if fully_qualified_table_name not in self._tables:
            raise TableNotInGraphError(
                title=f'Table "{fully_qualified_table_name}" is not found in '
                'the Graph',
                message='Please ensure that the correct table name is used.')
        if column_name not in self._tables[fully_qualified_table_name]:
            raise ColumnNotInTableError(
                title=f'Column "{column_name}" does not exist in the table: '
                f'"{fully_qualified_table_name}"',
                message='Please ensure that the correct column name is used.')
        return self._tables[fully_qualified_table_name][column_name]

    def column_from_fqn(self, column_path: str) -> BaseColumn:
        r"""The column with a fully-qualified name :obj:`column_path`
        specified with its full "table.column" notation.
        """
        col = self[column_path]
        if isinstance(col, BaseColumn):
            return col
        raise KeyError(f'{column_path} is not a valid column name.')

    def set_table_data_versions(self, job_id: str):
        r"""Set the table data versions for tables in this graph according
        to the cached versions in data backend cache based on job_id.
        This is used to guarantee that the tables are consistently using
        the same data version throughout a workflow."""
        for table in self.tables:
            table.set_table_data_version(job_id)

    # Methods #################################################################

    def connect(self, connector: Connector) -> 'Graph':
        r"""Connects a data source to the graph.

        Args:
            connector (Connector): The connector.
        """
        data_source = connector.name
        if self.has_data_source(data_source):
            raise KeyError(f"Data source '{data_source}' already exists")
        self._data_source_dict[data_source] = connector
        return self

    @raise_post_materialization
    def add(
        self,
        name: str,
        data_source: Union[str, pd.DataFrame, None] = None,
        pkey: Optional[Union[PrimaryKey, str]] = None,
        fkeys: Optional[Union[List[ForeignKey], List[str]]] = None,
        cols: Optional[Union[List[Column], List[str]]] = None,
        time_col: Optional[Union[TimeColumn, str]] = None,
        end_time_col: Optional[Union[TimeColumn, str]] = None,
        source_name: Optional[str] = None,
        source_tables: Optional[List[Table]] = None,
        sql_query: Optional[str] = None,
        custom_sql: Optional[str] = None,
        ignore_errors: bool = False,
    ) -> 'Graph':
        r"""Adds a `Table` or `View` to the graph.

        Args:
            name (str): The name of the table.
            data_source (str or pandas.DataFrame, optional): The data source of
                the table. If set to :obj:`None`, requires that only a single
                data source is present. (default: :obj:`None`)
            pkey (PrimaryKey, optional): The primary key of the table.
                (default: :obj:`None`)
            fkeys (List[ForeignKey], optional): The foreign keys of the table.
                (default: :obj:`None`)
            cols (List[Column], optional): The columns of the table.
                (default: :obj:`None`)
            time_col (TimeColumn, optional): The time column of the table.
                (default: :obj:`None`)
            end_time_col (TimeColumn, optional): The end time column of the
                table. (default: :obj:`None`)
            source_name (str, optional): The name of the table in the
                :obj:`data_source`, in case it is not identical to :obj:`name`.
                Must be :obj:`None` for :obj:`View` tables.
                (default: :obj:`None`)
            source_tables (List[Table], optional): The source tables
                used to create a :obj:`View` table. (default: :obj:`None`)
            sql_query (str, optional): The SQL query used to create a
                :obj:`View` table using the :obj:`source_tables`.
                (default: :obj:`None`)
            custom_sql (str, optional): The read-only ``SELECT`` backing a
                Custom SQL Table, pushed down to the warehouse. Marks the table
                as :obj:`is_sql`. (default: :obj:`None`)
            ignore_errors (bool): Whether to ignore errors when adding the
                table. (default: :obj:`False`)
        """
        if isinstance(data_source, pd.DataFrame):
            df, data_source = data_source, 'pandas'
            if not self.has_data_source(data_source):
                self.connect(connector=PandasConnector())
            self.connector(data_source).add_df(name, df)
            if pkey is None and df.index.name is not None:
                pkey = PrimaryKey(name=df.index.name)

        if data_source is None:
            if self.num_data_sources != 1:
                raise ValueError(f"Can not add table '{name}' without any "
                                 f"data source")
            data_source = self.data_sources[0]

        table = Table(
            table_name=name,
            data_source=data_source,
            pkey=pkey,
            fkeys=fkeys,
            cols=cols,
            time_col=time_col,
            end_time_col=end_time_col,
            source_name=source_name,
            custom_sql=custom_sql,
        )

        return self._add(table, ignore_errors=ignore_errors)

    def _add(self, table: Table, ignore_errors: bool = False) -> 'Graph':
        if not self.has_data_source(table.data_source) and not ignore_errors:
            raise ValueError(
                f"Data source '{table.data_source}' does not yet exist")

        key = table.name
        if key in self._tables:
            raise ValueError(f"Table '{key}' already exists")

        table.connector = self._data_source_dict[table.data_source]
        self._tables[key] = table

        return self

    @raise_post_materialization
    def add_from_config(
        self,
        path: str,
        data_source: Optional[str] = None,
    ) -> 'Graph':
        r"""Adds a set of tables as defined by a :obj:`*.yml` configuration.

        Args:
            path (str): The path to the :obj:`*.yml` file.
            data_source (str, optional): The default data source of tables for
                which data sources are not further specified in the
                :obj:`*.yml` configuration. (default: :obj:`None`)
        """
        cfg = load_json(path, cls=GraphConfig)

        for table in cfg.tables:
            if table.data_source is None:
                if data_source is not None:
                    table.data_source = data_source
                elif self.num_data_sources != 1:
                    raise ValueError(f"Can not add table '{table.name}' "
                                     f"without any data source")
                else:
                    table.data_source = self.data_sources[0]
            self._add(Table.from_config(table))

        # The config file might not contain fully-qualified names for
        # foreign key destination table.
        self.fully_qualify_all_fkeys()
        return self

    @raise_post_materialization
    def remove(self, name: str, data_source: Optional[str] = None) -> 'Graph':
        r"""Removes the table with name :obj:`name` from the graph."""
        key = self.fully_qualified_table_name(name, data_source)

        del self._tables[key]

        for table in self._tables.values():
            for fkey in table.fkeys:
                if fkey.dst_table == name:
                    table.remove_fkey(fkey.name)

        return self

    @raise_post_materialization
    def link(
        self,
        src_table: str,
        src_key: str,
        dst_table: str,
        dst_key: Optional[str] = None,
        src_data_source: Optional[str] = None,
        dst_data_source: Optional[str] = None,
        *args,
        **kwargs,
    ) -> 'Graph':
        r"""Links two tables (:obj:`src_table` and :obj:`dst_table`) via the
        the keys :obj:`src_key` and :obj:`dst_key` in the two tables,
        respectively.

        Args:
            src_table (str): The name of the source table.
            src_key (str): The name of the key in :obj:`src_table`.
            dst_table (str): The name of the destination table.
            dst_key (str, optional): The name of the key in :obj:`dst_table`.
                If set to :obj:`None`, will link to its primary key.
                (default: :obj:`None`)
            src_data_source (str, optional): Name of the data source for the
                source table. This field is mandatory if multiple data sources
                are connected to the graph and the source table has the
                same name as another table in a different data source.
                (default: :obj:`None`)
            dst_data_source (str, optional): Name of the data source for the
                destination table. This field is mandatory if multiple data
                sources are connected to the graph and the destination table
                has the same name as another table in a different data source.
                (default: :obj:`None`)
        """
        fq_src_table_name = self.fully_qualified_table_name(
            src_table, src_data_source)
        fq_dst_table_name = self.fully_qualified_table_name(
            dst_table, dst_data_source)

        src_data_source = get_data_source(fq_src_table_name)
        dst_data_source = get_data_source(fq_dst_table_name)

        self._tables[fq_src_table_name].add_fkey(
            name=src_key,
            dst_table=fq_dst_table_name,
            dst_key=dst_key,
            dst_data_source=dst_data_source,
            *args,
            **kwargs,
        )

        # Add reverse connection if `dst_key` is not a primary key:
        dst_table_obj = self._tables[fq_dst_table_name]
        if dst_key is not None and not dst_table_obj.is_pkey(dst_key):
            dst_table_obj.add_fkey(
                name=dst_key,
                dst_table=fq_src_table_name,
                dst_key=src_key,
                dst_data_source=src_data_source,
                *args,
                **kwargs,
            )
        return self

    def remove_unused_cols(self) -> 'Graph':
        r"""Removes any columns from the :class:`Graph` for which
        materialization will be skipped. This avoids unnecessary loading of
        these columns in the first place."""
        for table in self.tables:
            table.remove_unused_cols()
        return self

    def get_total_num_edges(self) -> int:
        r"""Get the number of edges in the graph (estimated).
        """
        num_edges = 0
        for table in self.tables:
            # Despite the FK-FK relationship, each FK is considered as 2 edges
            # for undirected graphs.
            num_rows = table.num_rows()

            num_edges += num_rows * len(table.fkeys) * 2
        return num_edges

    # Persistence #############################################################

    def _persist_pandas_connector(self, root_dir: str):
        r"""Persists all :class:`PandasConnector` data sources and replaces
        them with :class:`ParquetConnector` instances.

        Args:
            root_dir (string): Directory to persist the data under.
        """
        # NOTE: Currently using PandasConnector in a graph doesn't work once we
        # save the object since we are converting them to ParquetConnector.
        # This conversion however doesn't handle the renaming of all other
        # metadata. For e.g. self._tables is keyed by FQN which needs to be
        # updated.
        # TODO(kiriti): Revisit later about figuring out how to make
        # PandasConnector work just within the graph.

        # Convert PandasConnector to ParquetConnector in a GraphConfig.
        for connector in self.connectors:
            if isinstance(connector, PandasConnector):
                new_connector = persist_pandas_connector(
                    connector, root_dir=root_dir)
                logger.warning(
                    f"Converting {connector} to {new_connector} when saving "
                    "the graph. If the PandasConnector is connected to data "
                    "**other than** training table, errors may occur "
                    "during training.")
                # Switch connector for tables
                for table in self.tables:
                    if table._data_source == 'pandas':
                        table.connector = new_connector

                # Add the new connector and delete the old one
                self.connect(new_connector)
                del self._data_source_dict[connector.name]

    def save(self, destination: str, name: Optional[str] = None):
        r"""Save the :class:`kumo.Graph` to storage.

        Args:
            destination (string): Base directory to save the graph under; the
                graph is written to ``<destination>/graphs/<name>``. Required —
                callers pass an explicit per-job dir so this never depends on
                (or mutates) the process-global home directory.
            name (string, optional): The name of the folder to store the data.
                If set to :obj:`None`, will use the name of the graph.
                (default: :obj:`None`)
        """
        if self.name is None:
            raise ValueError(
                "Please specify a name of the graph before saving it to disk")
        name = name or self.name
        root_dir = osp.join(destination, 'graphs', name)
        fsutil.makedirs(root_dir, exist_ok=True)

        # Convert `PandasConnector` to `ParquetConnector` in a graph before
        # save to disk.
        self._persist_pandas_connector(root_dir)

        # TODO(data-platform, ml-infra): remove all statistics from the graph
        # config before saving; these are stored in the data backend cache and
        # encoder statistics are stored in the *Data objects, so there should
        # be no reason to maintain a heavier configuration here. This can be
        # handled in post_config:
        graph_config = self.config()
        path = osp.join(root_dir, 'config.json')
        save_json(graph_config, path)

        for table in self.tables:
            if table.has_data:
                table.data.save_data(osp.join(root_dir, table.name))

        logger.info("Successfully saved Graph %s to %s", self.name, root_dir)

    # Statistics ##############################################################

    @pytest_only
    def compute_table_stats(self, ignore_errors: bool = False):
        r"""Computes all table stats for the graph. Only for testing."""
        for table in self.tables:
            try:
                table.compute_table_stats()
            except Exception as e:
                if not ignore_errors:
                    raise e

    # Metadata and Type Inferral ##############################################
    #
    # This set of methods helps infer graph metadata (missing links, and
    # missing metadata for all tables in a graph) for a given Graph object. The
    # main entrypoints are infer_metadata, which infers any missing metadata
    # tables in the graph, and infer_links, which infers any missing links in
    # the graph.

    def infer_col_groups(self) -> List[GroupInfo]:
        r"""Infers unspecified link (constraint) information of the graph:
        Creates linkage groups by parsing tables of this graph. For a pkey
        in each table, it is compared with cols of other tables to create a
        group. For each table with pkey there will be an inferred group if
        the pkey matches (with certain criteria ) with one of the column
        of other table.
        """
        groups: List[GroupInfo] = []
        inf = inflect.engine()
        for pkey_table in self:
            if not pkey_table.has_pkey():
                continue

            group = GroupInfo(
                pkey=ColumnInfo(table=pkey_table.name,
                                col=pkey_table.pkey_name),
                fkeys=[],
            )

            # Iterate over all tables. This is the second table that
            # we compare the pkey against.
            for fkey_table in self:
                for col in fkey_table.cols + fkey_table.fkeys:
                    # Criteria to identify fkey and add it to the group:
                    # 1. dtype of pkey and col match.
                    # 2. if pkey-fkey name matches as per one of these
                    #   a. col is pkey_table.pkey
                    #   b. col is `<pkey_table>_id`
                    #   c. col is `<singular(pkey_table)>_id` where
                    #      singular() is inflect.singular_noun

                    # First check whether the candidate fkey can link to
                    # the pkey dtype.
                    fkey_obj = ForeignKey(col.name, None, None, col.dtype)
                    if not fkey_obj.can_link_to_dtype(pkey_table.pkey_name,
                                                      pkey_table.pkey.dtype):
                        continue
                    # Prepare case folded column name for fkey column.
                    colname = col.name.casefold()
                    # Prepare case folded column name formats for pkey.
                    pkey_formata = pkey_table.pkey_name.casefold()
                    pkey_formatb = (f'{pkey_table.table_name}_id').casefold()
                    pkey_formatc = (
                        (f'{inf.singular_noun(pkey_table.table_name)}_id'
                         ).casefold())
                    if (colname == pkey_formata or colname == pkey_formatb
                            or colname == pkey_formatc):

                        # Add inferred fkey to the group
                        group.fkeys.append(
                            ColumnInfo(table=fkey_table.name, col=col.name))

            if len(group.fkeys) > 0:  # Group ready
                groups.append(group)

        return groups

    @blocking_method(GRAPH_THREAD_POOL)
    def infer_links(self, name_inferral: bool = True) -> 'Graph':
        r"""Infers unspecified link (constraint) information of the graph:

        * Infers link information of each table from the backing connectors,
            if possible.
            * Infers missing primary keys
            * Infers missing foreign keys
            * Infers missing destination tables of foreign keys
        * If name_inferral is True, infers the foreign keys by matching the
            names of columns and primary keys between tables in the graph.

        Fully qualifies the destination table of every foreign key.
        """
        meta_table_dict = defaultdict(list)
        for table in self:
            meta_table_dict[table.data_source].append(table.metadata_config())

        # At the data source level:
        for data_source, meta_tables in meta_table_dict.items():
            connector = self.connector(data_source)
            # Fetch constraints directly from the source connector.
            if connector.has_links_support():
                source_names = [table.source_name for table in meta_tables]
                table_names = [table.table_name for table in meta_tables]
                meta_tables = connector.get_table_metadata(source_names)
                # Correct table names that might vary from source names
                for meta_table, table_name in zip(meta_tables, table_names):
                    meta_table.table_name = table_name
            for meta_table in meta_tables:
                table = self.table(meta_table.table_name, data_source)
                if meta_table.pkey is not None:  # Update primary key info:
                    table.set_pkey(PrimaryKey.from_config(meta_table.pkey))
                for fkey in meta_table.fkeys:  # Update foreign key info:
                    if table.has_fkey(fkey.name):
                        table[fkey.name].dst_table = fkey.dst_table
                        table[fkey.name].dst_key = fkey.dst_key
                    else:
                        table.add_fkey(ForeignKey.from_config(fkey))

        # At the graph level:
        if name_inferral:
            # Get a list of all inferred groups. Each group is of type
            # GroupInfo
            groups = self.infer_col_groups()
            for group in groups:
                # Update fkeys in the tables of this graph.
                for fkey in group.fkeys:
                    table = self.table(fkey.table)
                    if (table.has_fkey(fkey.col)
                            and table[fkey.col].dst_table is None):
                        table[fkey.col].dst_table = group.pkey.table
                        table[fkey.col].dst_key = group.pkey.col
                    else:
                        table.add_fkey(name=fkey.col,
                                       dst_table=group.pkey.table,
                                       dst_key=group.pkey.col)
            uninferred_links = []
            for table in self:
                for fkey in table.fkeys:
                    if fkey.dst_table is None:
                        uninferred_links.append((table.name, fkey.name))

            if len(uninferred_links) > 0:
                logger.warning(
                    f"Cannot infer destination table for '{uninferred_links}'."
                    "Please manually assign the dst_table for these foreign "
                    "keys.")

        self.fully_qualify_all_fkeys()
        return self

    @blocking_method(GRAPH_THREAD_POOL)
    def infer_stypes(self) -> 'Graph':
        r"""Infers stypes for all tables in the graph by calling the
        corresponding function on the Table class
        """
        for table in self:
            table.infer_stypes()

    @Timer(logging_prefix="graph.infer_metadata")
    def infer_metadata(
        self,
        include_cols: Union[Dict[str, Set[str]], Set[str], None] = None,
        exclude_cols: Union[Dict[str, Set[str]], Set[str], None] = None,
        num_threads: int = 0,
        drop_id_cols: bool = False,
    ) -> 'Graph':
        r"""Infers unspecified metadata of the graph:

        * Infers remaining columns that exist in each table.
        * Fills missing :obj:`dtype` attribute of columns.

        Args:
            include_cols (Dict[str, Set[str]] or Set[str], optional):
                Set of columns that should be included in inferral if not
                already added.
                Can either be a common set for all the tables, or can provide
                a map of :obj:`table_name` to set of columns.
                Cannot be used with :obj:`exclude_cols` (default: :obj:`None`).
            exclude_cols (Dict[str, Set[str]] or Set[str], optional):
                Set of columns that should be excluded from inferral.
                Can either be a common set for all the tables, or can provide
                a map of table_name to set of columns.
                Cannot be used with :obj:`include_cols` (default: :obj:`None`).
            num_threads (int, optional): The number of threads to use for infer
                metadata. If set to :obj:`0`, will perform the operation on the
                main thread. (default: :obj:`0`).
            drop_id_cols (bool, optional): Ignore a column with `_ID_` (case
                insensitive) in the name. (default :obj:`True`).
        """
        # Convert include_cols, exclude_cols to dict if required.
        if not isinstance(include_cols, Dict):
            include_cols = {name: include_cols for name in self.table_names}
        include_cols = {
            self.fully_qualified_table_name(key): value
            for key, value in include_cols.items()
        }

        if not isinstance(exclude_cols, Dict):
            exclude_cols = {name: exclude_cols for name in self.table_names}
        exclude_cols = {
            self.fully_qualified_table_name(key): value
            for key, value in exclude_cols.items()
        }

        if num_threads == 0:
            for table in self.tables:
                table.infer_metadata(include_cols.get(table.name, None),
                                     exclude_cols.get(table.name, None),
                                     drop_id_cols=drop_id_cols)
        else:
            # TODO: verify that `table.infer_metadata` and calls into the
            # `connector` are thread-safe.
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for table in self.tables:
                    futures.append(
                        executor.submit(table.infer_metadata,
                                        include_cols.get(table.name, None),
                                        exclude_cols.get(table.name, None),
                                        drop_id_cols=drop_id_cols))
                check_for_exception(futures)

        return self

    # Encoder Inferral ########################################################

    @blocking_method(GRAPH_THREAD_POOL)
    def infer_encoders(
        self,
        encoder_stats: Optional[Dict[str, Optional[ExactStats]]] = None,
        **kwargs,
    ) -> 'Graph':
        r"""Infers unspecified encoders of the graph.

        Args:
            num_threads (int, optional): The number of threads to use for infer
                encoders. If set to :obj:`0`, will perform the operation on the
                main thread. (default: :obj:`0`).
            **kwargs: Keyword arguments that are accepted by
                :obj:`Connector.read_table_sample`.
        """
        if encoder_stats is None:
            encoder_stats = {}

        for table in self.tables:
            table.infer_encoders(encoder_stats=encoder_stats.get(table.name),
                                 **kwargs)

        return self

    def apply_encoder_overrides(
        self,
        encoder_overrides: Dict[str, Encoder],
    ) -> 'Graph':
        """Apply encoder overrides to this graph.

        Args:
            encoder_overrides (Dict[str, Encoder]): Dictionary of encoder
            overrides keyed by <table_name>.<col>, where table_name can be
            either fully qualified or not.
        """
        for fqcn, e in encoder_overrides.items():
            if fqcn.endswith('train_table.TARGET'):
                continue
            if len(fqcn.rsplit('.', 1)) != 2:
                raise ValueError(
                    'Column specified in encoder overrides needs to be of the '
                    'form <connector_name>.<table_name>.<column_name> '
                    f'(got {fqcn})')
            table_name, column_name = fqcn.rsplit('.', 1)
            if not is_fully_qualified_name(table_name):
                table_name = self.fully_qualified_table_name(table_name, None)
            self.column(table_name, column_name).encoder = e
        return self

    # Materialization #########################################################

    @blocking_method(
        GRAPH_THREAD_POOL,
        tracing_options=tracing.TraceOptions(trace_params=['ignore_encoders']))
    def validate(
        self,
        raise_on_error: bool = True,
        ignore_encoders: bool = False,
        enforce_connected: bool = False,
    ) -> bool:
        """Validate the graph's tables, keys, links, and encoders.

        A graph with no tables is invalid. Validation failures raise when
        ``raise_on_error`` is true; otherwise they emit warnings and this
        method returns ``False``. ``ignore_encoders`` skips encoder checks,
        and ``enforce_connected`` makes disconnected tables an error.
        """
        status = True

        if len(self) == 0:
            status = False
            logging.warn_or_raise(
                'A graph must contain at least one table. Please add at '
                'least one table to the graph.', raise_on_error)

        reachable_tables: Set[Tuple[str, str]] = set()
        self.fully_qualify_all_fkeys()
        for table in self.tables:

            # 1. Validate the correctness of each table.
            status &= table.validate(raise_on_error,
                                     ignore_encoders=ignore_encoders)
            for fkey in table.fkeys:
                # 2. For each foreign key, there exists a valid destination:
                if (fkey.dst_table is None
                        or not self.has_table(fkey.dst_table)):
                    status = False
                    logging.warn_or_raise(
                        f"The foreign key '{fkey.name}' of the table "
                        f"'{table.name}' is expected to point to a valid "
                        f"destination table (got {fkey.dst_table})",
                        raise_on_error)

                if not is_fully_qualified_name(fkey.dst_table):
                    status = False
                    logging.warn_or_raise(
                        f"The destination table of foreign key {fkey.name} "
                        f"is not fully qualified. Got {fkey.dst_table}",
                        raise_on_error)

                fkey_dst_table = self[fkey.dst_table]
                if fkey.dst_key is None:
                    # 3. Each table referenced by a foreign key without a
                    # destination column has a primary key:
                    if fkey_dst_table.pkey is None:
                        status = False
                        logging.warn_or_raise(
                            f"The foreign key '{fkey.name}' of the table "
                            f"'{table.name}' is expected to point to a "
                            f"destination table {fkey_dst_table.name} with "
                            "a primary key", raise_on_error)
                    fkey_dst_col = fkey_dst_table.pkey
                else:
                    # 4. Each table referenced by a foreign key with a
                    # destination column has that destination column
                    # as a key (primary or foreign):
                    if (not fkey_dst_table.has_fkey(fkey.dst_key)
                            and not fkey_dst_table.is_pkey(fkey.dst_key)):
                        status = False
                        logging.warn_or_raise(
                            f"The foreign key '{fkey.name}' of the table "
                            f"'{table.name}' points to the non-existent key "
                            f"'{fkey.dst_key}' in table '{fkey.dst_table}'",
                            raise_on_error)
                    fkey_dst_col = fkey_dst_table[fkey.dst_key]

                # 5. Each foreign key must have a compatible data type
                if not fkey.can_link_to_dtype(
                        fkey_dst_col.name, fkey_dst_col.dtype, raise_on_error):
                    logging.warn_or_raise(
                        f"The foreign key '{fkey.name}' of the table "
                        f"'{table.name}' has data type '{fkey.dtype}' "
                        f"which is incompatible with the destination "
                        f"column '{fkey.dst_key}' in table "
                        f"'{fkey.dst_table}' which has data type "
                        f"'{fkey_dst_col.dtype}'. Please make sure that "
                        f"the data types match exactly or that integer "
                        f"links widen from the foreign key to the "
                        f"destination key without narrowing. This can be "
                        f"done by reconfiguring or reuploading the table "
                        f"with compatible data types.", raise_on_error)

                # Float/double FK columns are dropped at ingest (empty graph); reject so the user declares an int key.
                if fkey.dtype is not None and fkey.dtype.is_float():
                    status = False
                    logging.warn_or_raise(
                        f"The foreign key '{fkey.name}' of table "
                        f"'{table.name}' has dtype '{fkey.dtype}'. ID/key "
                        f"columns must be integer or string; float/double "
                        f"key columns are not supported and would be dropped "
                        f"at ingest. Please set the dtype of '{fkey.name}' to "
                        f"'int'.", raise_on_error)

                reachable_tables.add(table.name)
                reachable_tables.add(fkey.dst_table)

        # 4. Every table in the graph is reachable.
        if len(self) > 1 and len(reachable_tables) < len(self):
            # This is not a hard requirement, so we just warn here:
            logging.warn_or_raise(
                f"Tables {set(self.table_names) - reachable_tables} are not "
                f"reachable from any table in the graph. Please make sure "
                f"that all tables are fully linked into a single graph via "
                f"column groups, or remove the tables that cannot be linked.",
                raise_on_error=enforce_connected)

        return status

    def clear_data(self) -> 'Graph':
        super().clear_data()
        for table in self.tables:
            table.clear_data()
        return self

    # Visualization ###########################################################

    @blocking_method(GRAPH_THREAD_POOL,
                     tracing_options=tracing.TraceOptions(trace_params=True))
    def visualize(
        self,
        path: Union[str, BytesIO, None] = None,
        show_cols: bool = True,
    ):
        r"""Visualizes the Kumo graph.

        Args:
            path (str or BytesIO, optional): The path to where the plot is
                saved. Pass in a :obj:`BytesIO` object if you would like the
                image written to an in-memory buffer instead. If set to
                :obj:`None`, will visualize the graph on-the-fly. (default:
                :obj:`None`)
        """
        meta_table_configs = [table.metadata_config() for table in self]
        visualize_graph(meta_table_configs, self.name, path, show_cols)

    def print_summary(self):
        r"""Prints a summary of the tables in the graph, *i.e.* the data types,
        encoder decisions, and number of :obj:`N/A` values per column."""
        for i, table in enumerate(self):
            table.print_summary()
            if i + 1 < self.num_tables:
                print()  # Force a line break between tables.

    # Metapaths ###############################################################

    @raise_post_materialization
    def add_metapath(self, metapath: List[EdgeType]):
        r""" Specify the metapath to be added during the
        materialization step. It will append the metapath to
        be added, and set / override the arguments for calling the
        :meth:`GraphData.add_metapath` method in materialization.

        Args:
            metapath: List[EdgeType]: A metapath expressed as a list of
                EdgeType tuples.
                The :obj:`EdgeType` is defined as a tuple specifying
                (source table, foreign key, destination table), which
                exists as an edge in the kumo graph.
                TODO: we could consider using a starting table and a
                sequence of foreign keys to specify the metapath instead
                of a list of tuples.
        """
        metapath = self._fully_qualify_metapaths(metapath)
        metapath = [self._verify_edge_type(metaedge) for metaedge in metapath]
        self._metapaths.append(metapath)

    @raise_post_materialization
    def remove_metapath(self, metapath: List[EdgeType]):
        r""" Remove the metapath from the list of metapaths to be added
        during the materialization step.

        Args:
            metapath: List[EdgeType]: A metapath expressed as a list of
                EdgeType tuples.
        """
        metapath = self._fully_qualify_metapaths(metapath)
        metapath = [self._verify_edge_type(metaedge) for metaedge in metapath]
        try:
            self._metapaths.remove(metapath)
        except ValueError:
            logger.warning(f"Attempting to remove a metapath that does not "
                           f"exist: {metapath}")

    @raise_post_materialization
    def set_metapath_kwargs(self, kwargs: Dict[str, Any]):
        r""" Set metapath arguments for :meth:`add_metapath`.

        Args:
            kwargs: the dictionary of arguments. Possible keys:

            - no_self_loops (bool, optional): Whether to exclude self loops
                from intra-table metapath edges (default: :obj:`None`).
            - max_sample (int, optional): Samples metapath edges such that
                at every step of metapath, the expected out-degree of nodes
                is less than :obj:`max_sample`. (default: :obj:`None`)
        """
        for k, v in kwargs.items():
            self._metapath_kwargs[k] = v

    @raise_post_materialization
    def add_intra_table_metapath(self, metapath: List[EdgeType]):
        r""" Specify the intra-table metapath to be added during the
        materialization step. These metapaths connect rows from the same
        table.
        It will append the list of intra-table metapaths to
        be added, and set / override the arguments for calling the
        :meth:`GraphData.add_intra_metapath` method in materialization.

        E.g. when a specified metapath is :obj:`[(A, B), (B, C)]`,
        the intra-table metapath formed is A->B->C->B->A, which are
        all (potentially sampled or self-loop-excluded) paths of
        length 4 with node types A, B, C, B, A.

        Args:
            metapath: List[EdgeType]: A metapath expressed as a list of
                EdgeType tuples.
            no_self_loops (bool, optional): Whether to exclude self loops
                from intra-table metapath edges (default: :obj:`None`).
            max_sample (int, optional): Samples metapath edges such that
                at every step of metapath, the expected out-degree of nodes
                is less than :obj:`max_sample`. (default: :obj:`None`)
        """
        metapath = self._fully_qualify_metapaths(metapath)
        metapath = [self._verify_edge_type(metaedge) for metaedge in metapath]
        self._intra_metapaths.append(metapath)

    @raise_post_materialization
    def remove_intra_metapath(self, metapath: List[EdgeType]):
        r""" Remove the metapath from the list of metapaths to be added
        during the materialization step.

        Args:
            metapath: List[EdgeType]: A metapath expressed as a list of
                EdgeType tuples.
                The :obj:`EdgeType` is defined as a tuple specifying
                (source table, foreign key, destination table), which
                exists as an edge in the kumo graph.
                TODO: we could consider using a starting table and a
                sequence of foreign keys to specify the metapath instead
                of a list of tuples.
        """
        metapath = self._fully_qualify_metapaths(metapath)
        metapath = [self._verify_edge_type(metaedge) for metaedge in metapath]
        try:
            self._intra_metapaths.remove(metapath)
        except ValueError:
            logger.warning(f"Attempting to remove an intra-table metapath "
                           f"that does not exist: {metapath}")

    @raise_post_materialization
    def set_intra_metapath_kwargs(self, kwargs: Dict[str, Any]):
        r""" Set metapath arguments for :meth:`add_intra_metapath`.

        Args:
            kwargs: the dictionary of arguments. Possible keys:

            - no_self_loops (bool, optional): Whether to exclude self loops
                from intra-table metapath edges (default: :obj:`None`).
            - max_sample (int, optional): Samples metapath edges such that
                at every step of metapath, the expected out-degree of nodes
                is less than :obj:`max_sample`. (default: :obj:`None`)
        """
        for k, v in kwargs.items():
            self._metapath_kwargs[k] = v

    def _fully_qualify_metapaths(self,
                                 metapaths: List[EdgeType]) -> List[EdgeType]:
        fully_qualified_metapaths = []
        for metapath in metapaths:
            fq_src_name = self.fully_qualified_table_name(metapath[0])
            edge_type = None
            # If edge type is not specified, we do not set it and infer the
            # second tuple argument to be destination
            if len(metapath) == 2:
                fq_dst_name = self.fully_qualified_table_name(metapath[1])
            else:
                edge_type = metapath[1]
                fq_dst_name = self.fully_qualified_table_name(metapath[2])
            if edge_type is not None:
                fully_qualified_metapaths.append(
                    (fq_src_name, edge_type, fq_dst_name))
            else:
                fully_qualified_metapaths.append((fq_src_name, fq_dst_name))
        return fully_qualified_metapaths

    def _verify_edge_type(self, edge_type: EdgeType):
        r"""
        Verifies an edge type, by either returning the unique edge
        type for a given edge_type, or raising a value error. Inputs
        can be
        1. `relation_type` will get mapped to the unique
         (`src_table`, `relation_type`, `dst_table`), or raise a ValueError if
         `relation_type` does not exist or corresponds to multiple edge types.
        2. (`src_table`, `dst_table`) will get mapped to the unique
         (`src_table`, `relation_type`, `dst_table`) edge type or raise a
         ValueError if multiple potential edges exist or none exist.
        3. (`src_table`, `relation_type`, `dst_table`) will raise a value error
         if the edge doesn't exist and will return
         (`src_table`, `relation_type`, `dst_table`) otherwise.

         Table names must be fully qualified.
        """
        def unknown():
            raise ValueError(f"Unknown edge type {edge_type}")

        def ambiguous():
            raise ValueError(f"Ambiguous edge type {edge_type}")

        fkey_map = defaultdict(list)
        table_map = defaultdict(list)
        for table in self.tables:
            for fkey in table.fkeys:
                fkey_map[fkey.name].append(
                    (table.name, fkey.name, fkey.dst_table))
                fkey_map[f'rev_{fkey.name}'].append(
                    (fkey.dst_table, f'rev_{fkey.name}', table.name))
                table_map[(table.name, fkey.dst_table)].append(
                    (table.name, fkey.name, fkey.dst_table))
                table_map[(fkey.dst_table, table.name)].append(
                    (fkey.dst_table, f'rev_{fkey.name}', table.name))

        if isinstance(edge_type, str):
            edge_type = (edge_type, )

        if len(edge_type) == 1:
            if len(fkey_map[edge_type[0]]) == 0:
                unknown()
            elif len(fkey_map[edge_type[0]]) > 1:
                ambiguous()
            return fkey_map[edge_type[0]][0]
        elif len(edge_type) == 2:
            start, end = edge_type
            matches = table_map[start, end]
            if len(matches) == 0:
                unknown()
            elif len(matches) > 1:
                ambiguous()
            else:
                return matches[0]
        elif len(edge_type) == 3:
            start, _, end = edge_type
            matches = table_map[start, end]
            if edge_type not in matches:
                unknown()
            return edge_type
        else:
            unknown()

    # Miscellaneous ###########################################################

    def fully_qualify_fkey(self, fkey: ForeignKey) -> ForeignKey:
        r"""Fully qualify the destination table of the foreign key. It also
        adds the destination table and data source if not already present."""
        fkey = ForeignKey.cast(fkey)
        fq_dst_table = None
        if fkey.dst_table is None:
            dst_col = (fkey.name if fkey.dst_key is None else fkey.dst_key)
            filtered_potential_fkeys: List[Table] = list(
                filter(
                    lambda table:
                    (table.pkey is not None and dst_col == table.pkey.name),
                    self.tables))
            if len(filtered_potential_fkeys) > 1:
                raise ValueError(
                    f"Multiple tables matching the dst_col: {dst_col} for "
                    f"fkey: {fkey}")
            elif len(filtered_potential_fkeys) == 1:
                fq_dst_table = filtered_potential_fkeys[0].name
        else:
            fq_dst_table = self.fully_qualified_table_name(fkey.dst_table)
        dst_data_source = (get_data_source(fq_dst_table)
                           if fq_dst_table else None)

        return ForeignKey(
            fkey.name,
            dst_table=fq_dst_table,
            dst_key=fkey.dst_key,
            dtype=fkey.dtype,
            dst_data_source=dst_data_source,
            stats=fkey.stats,
        )

    def fully_qualify_all_fkeys(self) -> 'Graph':
        r"""Fully qualify FKeys in all the tables of the graph"""
        for table in self.tables:
            for fkey in table.fkeys:
                if fkey.dst_table and not is_fully_qualified_name(
                        fkey.dst_table):
                    table.fkey(fkey.name).update(self.fully_qualify_fkey(fkey))
                if fkey.dst_key is None and fkey.dst_table is not None:
                    fkey.dst_key = self.table(fkey.dst_table).pkey.name
        return self
