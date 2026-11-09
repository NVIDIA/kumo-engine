# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import warnings
from dataclasses import asdict
from itertools import chain
from typing import Any, Dict, Iterator, List, Optional, Set

import pandas as pd
from diskgraph import MULTICAT_DELIMITERS
from kumo_core.stype_inference import (
    infer_stype_from_dtype,
    infer_stype_from_name,
    infer_stype_from_stats,
)
from kumo_core.types import ColStatType, ConnectorType, TableStatType
from kumo_core.types.schema import (
    TS_FORMAT_METADATA,
    BaseColumnConfig,
    IngestedTableFingerprint,
    PartialTableFingerprint,
)
from kumo_core.utils.names import fully_qualified_table_name
from kumo_core.utils.registry import register
from kumoapi.encoder import Null
from kumoapi.typing import Dtype, Stype
from pandas.api.types import is_timedelta64_dtype

import common_utils.string
from async_utils.sync import blocking_method
from common_utils import string
from common_utils.string import contains_identifier
from io_utils import kumo_dataclass, logging
from kumo.config import (
    ConfigMixin,
    TableConfig,
    TableFingerprint,
    TableMetadataConfig,
    TableOrViewMetadataConfig,
)
from kumo.config.stats import StatsConfig
from kumo.connector import DEFAULT_DATA_VERSION, Connector, cast_col_to_numeric
from kumo.connector.connector import (
    DuplicateColumnsError,
    _compute_col_stats_from_sample,
)
from kumo.connector.timestamp_infer import hint_to_storage, infer_temporal_hint
from kumo.data import get_active_cache
from kumo.data.backend_cache import BackendCacheManager, ColumnName, ExactStats
from kumo.data.common import DATA_BACKEND_THREAD_POOL
from kumo.data.custom_sql import (
    SQL_SAMPLE_MAX_SCAN_BYTES,
    UnsupportedColumnTypeError,
    normalized_sql_identifier_leaf,
    sample_over_budget,
    source_table_for_col,
)
from kumo.graph import Column, ForeignKey, PandasView, PrimaryKey, TimeColumn
from kumo.graph.column import BaseColumn, EncoderGenerator
from kumo.graph.mixin import (
    ConnectorMixin,
    _HasData,
    raise_post_materialization,
)
from kumo.graph.utils import get_full_stats, get_minimal_stats
from kumo.stats import (
    ALL_COL_STAT_TYPES,
    ALL_TABLE_STAT_TYPES,
    NUM_ROWS_FOR_APPROX_STATS,
    Stats,
)
from statedb.utils.errors import UnsupportedInDiskgraphError

logger = logging.getLogger(__name__)

NUM_UNIQUE = ColStatType.NUM_UNIQUE
UNIQUE_FRACTION = ColStatType.UNIQUE_FRACTION
NUM_UNIQUE_MULTI = ColStatType.NUM_UNIQUE_MULTI
MULTI_CATEGORIES_SEPARATOR = ColStatType.MULTI_CATEGORIES_SEPARATOR

# Sentinel distinguishing "sample not yet loaded" from a loaded-but-``None``
# sample (the Custom SQL cost gate deferred it) in ``infer_dtypes_from_stypes``.
_SAMPLE_UNSET = object()


@kumo_dataclass
class StypeInferralStats:
    # Required statistics to infer stypes with no
    # 1-1 mapping between stype and dtype
    REQUIRED_STATS = [
        ColStatType.NUM_UNIQUE,
        ColStatType.UNIQUE_FRACTION,
        ColStatType.NUM_UNIQUE_MULTI,
        ColStatType.MULTI_CATEGORIES_SEPARATOR,
    ]

    num_unique: Stats
    unique_fraction: Stats
    num_unique_multi: Stats
    multicategorical_proven: bool


@register(data_cls=TableConfig)
class Table(ConfigMixin, ConnectorMixin, _HasData['TableData']):
    r"""A table is a configuration object that represents a table in a Kumo
    graph. A table consists of a data source, a primary key, foreign keys and
    column information.

    Args:
        table_name (str): The name of the table.
        data_source (str): The data source of the table.
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
            (default: :obj:`None`)
        stats (dict, optional): Table statistics. (default: :obj:`None`)
    """
    def __init__(
        self,
        table_name: str,
        data_source: str,
        pkey: Optional[PrimaryKey] = None,
        fkeys: Optional[List[ForeignKey]] = None,
        cols: Optional[List[Column]] = None,
        time_col: Optional[TimeColumn] = None,
        end_time_col: Optional[TimeColumn] = None,
        source_name: Optional[str] = None,
        stats: Optional[Dict[TableStatType, Stats]] = None,
        data_version: Optional[str] = None,
        custom_sql: Optional[str] = None,
        is_sql: bool = False,
    ):
        assert table_name is not None
        assert data_source is not None

        common_utils.string.validate_id(table_name, "Table")
        if source_name:
            string.validate_connector_source_name(source_name)
        else:
            # Otherwise, table_name is also source_name so validate it:
            string.validate_source_name(table_name)

        self._name = table_name
        self._data_source = data_source
        self._source_name = source_name
        self._data_version = data_version
        # When set, this table's rows come from a Custom SQL query pushed down
        # to the warehouse rather than a physical source table; ``source_name``
        # is then a synthetic alias and schema/stats come from the query, not a
        # physical-table lookup.
        self._custom_sql = custom_sql
        # Durable Custom-SQL marker. Set from ``custom_sql`` on the authoring
        # path, or directly via ``is_sql`` when rebuilt from a serialized config
        # (which drops the SQL text).
        self._is_sql = is_sql or (custom_sql is not None)
        self._sql_inference_sample: Any = _SAMPLE_UNSET
        # Populated by the REST layer (which alone can read sibling tables) to
        # drive the Custom SQL stype-inheritance pre-pass: the connector's
        # dialect plus the column stypes of referenced external tables that are
        # independently Kumo Tables, keyed by dialect-normalized table/column.
        self._sql_connector_type: Optional[ConnectorType] = None
        self._sql_reference_stypes: Optional[Dict[str, Dict[str, Stype]]] = None
        self._sql_column_order: Optional[List[str]] = None
        self._sql_source_metadata: Optional[TableMetadataConfig] = None
        self._pkey: Optional[PrimaryKey] = None
        self._fkeys: Dict[str, ForeignKey] = {}
        self._cols: Dict[str, Column] = {}
        self._time_col: Optional[TimeColumn] = None
        self._end_time_col: Optional[TimeColumn] = None
        self._time_dtype_explicit = False
        self._end_dtype_explicit = False
        self.stats = stats or {}

        # Add columns first so pkey and fkeys can reference their stypes
        for col in cols or []:
            self.add_col(Column.cast(col))

        self.set_pkey(pkey)

        for fkey in fkeys or []:
            self.add_fkey(ForeignKey.cast(fkey))

        self.set_time_col(time_col)
        self.set_end_time_col(end_time_col)

    # Python built-ins ########################################################

    def __eq__(self, other: Any) -> bool:
        # Override equality to not include stats in equality requirement:
        if not isinstance(other, Table):
            return False

        cfg = asdict(self.config())
        other_cfg = asdict(other.config())

        del cfg['stats']
        del other_cfg['stats']
        return cfg == other_cfg

    def __len__(self) -> int:
        return self.num_fkeys + self.num_cols + (1 if self.has_pkey() else 0)

    def __contains__(self, name: str) -> bool:
        return self.is_pkey(name) or self.has_fkey(name) or self.has_col(
            name) or self.is_end_time_col(name)

    def __getitem__(self, name: Any) -> Any:
        r"""Returns a given column of the table. In case a certain column is
        set multiple times, *e.g.*, in case a column is set as a time column
        and a feature column, feature columns will always have higher priority.
        """
        idx = None
        if isinstance(name, tuple):
            if len(name) == 1:
                name = name[0]
            elif len(name) == 2:
                name, idx = name
            else:
                raise IndexError(f"Only tuples with at most two elements can "
                                 f"be used for indexing (got {name}).")

        if name == Ellipsis:
            out = self
        elif isinstance(name, list):
            out = self.subset(name)
        elif isinstance(name, str):
            if self.is_pkey(name):
                out = self.pkey
            elif self.has_fkey(name):
                out = self.fkey(name)
            elif self.has_col(name):
                out = self.col(name)
            # end_time_col is not used as a feature column
            elif self.is_end_time_col(name):
                out = self.end_time_col
            else:
                raise KeyError(f"'{name}' not in table '{self.table_name}'")
        else:
            raise IndexError(f"Only strings, list and tuples can be used for "
                             f"indexing (got '{type(name).__name__}')")

        return out if idx is None else out.pandas[idx]

    def subset(self, cols: List[str]) -> 'Table':
        table = copy.copy(self)
        table.clear_data()  # TODO We clear any materialized data for now.

        table._fkeys: Dict[str, ForeignKey] = {}
        for fkey_name in cols:
            fkey = self._fkeys.get(fkey_name, None)
            if fkey is not None:
                table.add_fkey(fkey)

        table._cols: Dict[str, Column] = {}
        for col_name in cols:
            col = self._cols.get(col_name, None)
            if col is not None:
                table.add_col(col)

        return table

    @raise_post_materialization
    def __delitem__(self, name: str):
        r"""Removes a given column of the table. Calling this on a time
        column will only delete the feature column, but not its role in
        temporal sampling. To remove that, set `table.time_col = None`.
        """
        if isinstance(name, (tuple, list)) and len(name) == 1:
            name = name[0]
        assert isinstance(name, str)

        if self.is_pkey(name):
            self.remove_pkey()
        elif self.has_fkey(name):
            self.remove_fkey(name)
        elif self.has_col(name):
            self.remove_col(name)
        elif self.is_end_time_col(name):
            raise KeyError(
                f"Cannot delete end_time_col {self.end_time_col_name} "
                "because it is not a feature column.")
        else:
            raise KeyError(f"Cannot delete non-existing column '{name}'")

    def __iter__(self) -> Iterator[BaseColumn]:
        r"""All columns of the table, including pkeys and fkeys."""
        if self.has_pkey():
            yield self.pkey
        for fkey in self._fkeys.values():
            yield fkey
        for col in self._cols.values():
            yield col

    def __copy__(self) -> 'Table':
        out = self.__class__.__new__(self.__class__)
        out.__dict__.update(self.__dict__)

        # Here, we also copy the private dictionaries of the object to allow
        # for modifications (e.g., adding/removing foreign keys) without
        # sharing these changes across multiple instances. Importantly, we
        # still want to share all objects that these dictionaries hold.
        out._fkeys = copy.copy(out._fkeys)
        out._cols = copy.copy(out._cols)
        out.stats = copy.copy(out.stats)

        return out

    def __repr__(self) -> str:
        fkey_names = str(self.fkey_names).replace("'", "")
        col_names = str(self.col_names).replace("'", "")
        return (f'{self.__class__.__name__}(\n'
                f'  name={self.name},\n'
                f'  data_source={self.data_source},\n'
                f'  pkey={self.pkey_name or "???"},\n'
                f'  fkeys={fkey_names},\n'
                f'  cols={col_names},\n'
                f'  time_col={self.time_col_name or "???"},\n'
                f'  end_time_col={self.end_time_col_name or "???"},\n'
                f')')

    # Configuration ###########################################################

    def metadata_config(self) -> TableMetadataConfig:
        return TableMetadataConfig(
            table_name=self.table_name,
            source_name=self.source_name,
            data_source=self.data_source,
            data_version=self.data_version,
            pkey=self.pkey.config() if self.pkey else None,
            fkeys=[fkey.config() for fkey in self.fkeys],
            cols=[col.base_config() for col in self.cols],
            time_col=self.time_col.config() if self.time_col else None,
            end_time_col=self.end_time_col.config()
            if self.end_time_col else None,
        )

    @classmethod
    def from_source_config(cls, meta_config: TableMetadataConfig, name: str,
                           data_source: str,
                           connector: Connector | None = None) -> 'Table':
        table_name = name or meta_config.table_name
        table = cls.from_config(
            meta_config.as_table_config(table_name, data_source))
        table.connector = connector
        return table

    def metacopy(self) -> 'Table':
        r"""Creates a copy of the table, dropping any materialized data."""
        return self.__class__(
            table_name=self.table_name,
            data_source=self.data_source,
            # NOTE Shallow-copies only
            # We do not want to re-instantiate encoders, statistics, ...
            pkey=copy.copy(self.pkey),
            fkeys=[copy.copy(fkey) for fkey in self.fkeys],
            cols=[copy.copy(col) for col in self.cols],
            source_name=self.source_name,
            is_sql=self.is_sql,
        )

    def fingerprint(self) -> IngestedTableFingerprint:
        cols = dict([(c.name, c.dtype) for c in self.data_cols])
        multicategorical_delimiters = {
            c.name: c.multicategorical_delimiter
            for c in self.data_cols if c.multicategorical_delimiter is not None
        }
        assert all([dtype is not None for dtype in list(cols.values())])
        return IngestedTableFingerprint(
            connector_name=self.data_source,
            source_name=self.source_name,
            cols=cols,  # type: ignore
            pkey=(self.pkey.name if self.pkey is not None else None),
            time_col=(self.time_col.config()
                      if self.time_col is not None else None),
            end_time_col=(self.end_time_col.config()
                          if self.end_time_col is not None else None),
            multicategorical_delimiters=multicategorical_delimiters,
        )

    def partial_fingerprint(self) -> TableFingerprint:
        cols = [c.name for c in self.data_cols]
        return PartialTableFingerprint(
            connector_name=self.data_source,
            source_name=self.source_name,
            cols=cols,
            pkey=(self.pkey.name if self.pkey is not None else None),
            time_col=(self.time_col.config()
                      if self.time_col is not None else None),
            end_time_col=(self.end_time_col.config()
                          if self.end_time_col is not None else None),
        )

    # Connector/Table Binding #################################################

    @ConnectorMixin.connector.setter
    def connector(self, connector: Connector):
        self._connector = connector
        if connector is not None:
            self._data_source = connector.name
        if self._data_version is not None:
            get_active_cache().table_version_cache.save_in_memory(
                self._data_source, self.source_name, self._data_version)
        if self._pkey is not None:
            self._pkey.connector = connector
            self._pkey.table_name = self.source_name
        for col in chain(self._fkeys.values(), self._cols.values()):
            col.connector = connector
            col.table_name = self.source_name
        if self._time_col is not None:
            self._time_col.connector = connector
            self._time_col.table_name = self.source_name

        # For training table, set the target_col connector
        if self.is_training_table:
            if self.target_col is not None:
                self.target_col.connector = connector
                self.target_col.table_name = self.source_name

    # Configuration Properties ################################################

    @property
    def name(self) -> str:
        r"""The fully qualified name of the table."""
        return fully_qualified_table_name(self._name, self._data_source)

    @property
    def table_name(self) -> str:
        r"""The name of the table."""
        return self._name

    @property
    def data_source(self) -> str:
        r"""The data source of the table."""
        return self._data_source

    @property
    def source_name(self) -> str:
        r"""The name of the table in the :obj:`data_source`."""
        return self._source_name or self.table_name

    @property
    def custom_sql(self) -> Optional[str]:
        r"""The read-only ``SELECT`` backing a Custom SQL Table, or ``None``
        for a physical source table."""
        return self._custom_sql

    @property
    def is_sql(self) -> bool:
        r"""True when this table's rows come from a Custom SQL query pushed
        down to the warehouse rather than from a physical source table.

        Carried as a bare boolean through ``TableConfig``, so it survives graph
        (de)serialization even when ``custom_sql`` is ``None``."""
        return self._is_sql

    @is_sql.setter
    def is_sql(self, value: bool) -> None:
        r"""Settable so config-driven materialization can restore the marker.

        ``ConfigMixin.from_config`` ``setattr``s every config field that isn't a
        constructor parameter; ``TrainingTable.__init__`` omits ``is_sql``, so
        without a setter that path raised ``property ... has no setter``."""
        self._is_sql = bool(value)

    @property
    def pkey(self) -> Optional[PrimaryKey]:
        r"""The primary key of the table."""
        return self._pkey

    @property
    def pkey_name(self) -> Optional[str]:
        r"""The name of the primary key of the table."""
        return None if self.pkey is None else self.pkey.name

    def fkey(self, name: str) -> Optional[ForeignKey]:
        r"""The foreign key of the table with name :obj:`name`."""
        return self._fkeys.get(name, None)

    @property
    def fkeys(self) -> List[ForeignKey]:
        r"""The foreign keys of the table."""
        return list(self._fkeys.values())

    @property
    def fkey_names(self) -> List[str]:
        r"""The names of foreign keys of the table."""
        return list(self._fkeys.keys())

    def col(self, name: str) -> Optional[Column]:
        r"""The column of the table with name :obj:`name`."""
        return self._cols.get(name, None)

    @property
    def cols(self) -> List[Column]:
        r"""The columns of the table."""
        return list(self._cols.values())

    @property
    def col_names(self) -> List[str]:
        r"""The names of columns of the table."""
        return list(self._cols.keys())

    @property
    def time_col(self) -> TimeColumn:
        r"""The time column of the table."""
        return self._time_col

    @property
    def end_time_col(self) -> TimeColumn:
        r"""The time column of the table."""
        return self._end_time_col

    @property
    def time_col_name(self) -> Optional[str]:
        r"""The column of the table with name :obj:`name`."""
        return None if self.time_col is None else self.time_col.name

    @property
    def end_time_col_name(self) -> Optional[str]:
        r"""The column of the table with name :obj:`name`."""
        return None if self.end_time_col is None else self.end_time_col.name

    def has_pkey(self) -> bool:
        r"""Returns :obj:`True` in case the table has a primary key."""
        return self.pkey is not None

    def is_pkey(self, name: str) -> bool:
        r"""Returns :obj:`True` in case the table has a primary key with name
        :obj:`name`."""
        return self.pkey is not None and self.pkey.name == name

    def has_fkey(self, name: str) -> bool:
        r"""Returns :obj:`True` in case the table has a foreign key with name
        :obj:`name`."""
        return name in self._fkeys

    def has_col(self, name: str) -> bool:
        r"""Returns :obj:`True` in case the table has a column with name
        :obj:`name`."""
        return name in self._cols

    def has_time_col(self) -> bool:
        r"""Returns :obj:`True` in case the table has a time column"""
        return self.time_col is not None

    def has_end_time_col(self) -> bool:
        r"""Returns :obj:`True` in case the table has an end time column"""
        return self.end_time_col is not None

    def is_time_col(self, name: str) -> bool:
        r"""Returns :obj:`True` in case the table has a time column with name
        :obj:`name`."""
        return self.has_time_col() and self.time_col.name == name

    def is_end_time_col(self, name: str) -> bool:
        r"""Returns :obj:`True` in case the table has an end time column with
        name :obj:`name`."""
        return self.has_end_time_col() and self.end_time_col.name == name

    @property
    def is_training_table(self) -> bool:
        from kumo.task.training_table import TrainingTable
        return isinstance(self, TrainingTable)

    @property
    def num_fkeys(self) -> int:
        r"""The number of foreign keys."""
        return len(self._fkeys)

    @property
    def num_cols(self) -> int:
        r"""The number of columns."""
        return len(self._cols)

    @property
    def degree(self) -> int:
        r"""Alias of :attr:`~kumo.graph.Table.num_fkeys`."""
        return self.num_fkeys

    @property
    def _data_cols(self) -> List[BaseColumn]:
        data_cols = self.fkeys + self.cols
        if self.has_pkey():
            data_cols.append(self.pkey)
        if self.has_time_col():
            data_cols.append(self.time_col)
        if self.has_end_time_col():
            data_cols.append(self.end_time_col)
        return data_cols

    @property
    def data_col_dict(self) -> Dict[str, BaseColumn]:
        data_col_dict: Dict[str, BaseColumn] = {}
        for col in self._data_cols:
            col = col.to_base()
            # Override dtypes and check if they match for duplicated columns:
            if col.name not in data_col_dict:
                data_col_dict[col.name] = col
            else:
                data_col_dict[col.name].update(col)
        return data_col_dict

    @property
    def data_cols(self) -> List[BaseColumn]:
        r"""Returns the data columns of a :class:`Table`, *i.e.* its primary
        key (if present), its foreign keys, its feature columns, and its time
        column (if present). Duplicates between :obj:`cols` and :obj:`time_col`
        will be removed."""
        cols = self.data_col_dict
        if self._sql_column_order is not None:
            return [cols[name] for name in self._sql_column_order]
        return list(cols.values())

    # Configuration Methods ###################################################

    @raise_post_materialization
    def set_pkey(self, *args, **kwargs) -> 'Table':
        r"""Sets the primary key of the table:

        .. code-block:: python

            table.set_pkey('TABLE_ID')
            table.set_pkey('TABLE_ID', dtype=Dtype.int)

        Any arguments of :obj:`kumo.graph.PrimaryKey` can be passed.
        """
        pkey = PrimaryKey.cast(*args, **kwargs)

        if pkey is None:
            self.remove_pkey()
            return self

        if self.is_pkey(pkey.name):
            self.pkey.update(pkey)
            return self

        if pkey.name in self.fkey_names:
            logger.info(f"'{pkey.name}' already exists as a foreign key in "
                        f"table '{self.name}'. Removing it in order to add it "
                        f"as a primary key")
            pkey.dtype = pkey.dtype or self.fkey(pkey.name).dtype
            pkey.stype = self.fkey(pkey.name).stype or pkey.stype
            self.remove_fkey(pkey.name)

        if pkey.name in self.col_names:
            logger.info(f"'{pkey.name}' already exists as a feature column in "
                        f"table '{self.name}'. Removing it in order to add it "
                        f"as a primary key")
            pkey.dtype = pkey.dtype or self.col(pkey.name).dtype
            pkey.stype = self.col(pkey.name).stype or pkey.stype
            self.remove_col(pkey.name)

        self._pkey = pkey
        if self.has_connector:
            self._pkey.connector = self.connector
        self._pkey.table_name = self.source_name

        return self

    @raise_post_materialization
    def remove_pkey(self) -> 'Table':
        r"""Removes the primary key from the table."""
        self._pkey = None
        return self

    @raise_post_materialization
    def add_fkey(self, *args, **kwargs) -> 'Table':
        r"""Sets a foreign key of the table:

        .. code-block:: python

            table.add_fkey('FOREIGN_TABLE_ID')
            table.add_fkey('FOREIGN_TABLE_ID', dst_table='DST_TABLE')
            table.add_fkey('FOREIGN_TABLE_ID', dtype=Dtype.int)
            table.add_fkey('FOREIGN_TABLE_ID', 'DST_TABLE', dtype=Dtype.int)

        Any arguments of :obj:`kumo.graph.ForeignKey` can be passed.
        """
        fkey = ForeignKey.cast(*args, **kwargs)
        assert fkey is not None

        if self.has_fkey(fkey.name):
            self.fkey(fkey.name).update(fkey)
            return self

        if self.is_pkey(fkey.name):
            logger.info(f"'{fkey.name}' already exists as a primary key in "
                        f"table '{self.name}'. Removing it in order to add it "
                        f"as a foreign key")
            fkey.dtype = fkey.dtype or self.pkey.dtype
            self.remove_pkey()

        if fkey.name in self.col_names:
            logger.info(f"'{fkey.name}' already exists as a feature column in "
                        f"table '{self.name}'. Removing it in order to add it "
                        f"as a foreign key")
            fkey.dtype = fkey.dtype or self.col(fkey.name).dtype
            fkey.stype = self.col(fkey.name).stype or fkey.stype
            self.remove_col(fkey.name)

        self._fkeys[fkey.name] = fkey
        if self.has_connector:
            fkey.connector = self.connector
        fkey.table_name = self.source_name

        return self

    @raise_post_materialization
    def link(self, *args, **kwargs) -> 'Table':
        r"""Alias of :attr:`~kumo.graph.Table.add_fkey`."""
        return self.add_fkey(*args, **kwargs)

    @raise_post_materialization
    def remove_fkey(self, name: str) -> 'Table':
        r"""Removes the foreign key with name :obj:`name` from the table."""
        if name not in self._fkeys:
            raise KeyError(f"Cannot delete non-existing foreign key '{name}'")
        del self._fkeys[name]
        return self

    @raise_post_materialization
    def unlink(self, name: str) -> 'Table':
        r"""Alias of :attr:`~kumo.graph.Table.remove_fkey`."""
        return self.remove_fkey(name)

    @raise_post_materialization
    def add_col(self, *args, **kwargs) -> 'Table':
        r"""Sets a column to the table:

        .. code-block:: python

            table.add_col('FEATURE')
            table.add_col('FEATURE', dtype=Dtype.float)

        Any arguments of :obj:`kumo.graph.Column` can be passed.
        """
        col = Column.cast(*args, **kwargs)

        # Do not add end_time_col as feature
        if (self.end_time_col is not None
                and col.name == self.end_time_col.name):
            return

        assert col is not None

        if self.has_col(col.name):
            self.col(col.name).update(col)
            return self

        if self.is_pkey(col.name):
            col.dtype = col.dtype or self.pkey.dtype
            logger.info(f"'{col.name}' already exists as a primary key in "
                        f"table '{self.name}'. Removing it in order to add "
                        f"it as a feature column")
            self.remove_pkey()

        if col.name in self.fkey_names:
            col.dtype = col.dtype or self.fkey(col.name).dtype
            logger.info(f"'{col.name}' already exists as a foreign key in "
                        f"table '{self.name}'. Removing it in order to add "
                        f"it as a feature column")
            self.remove_fkey(col.name)

        self._cols[col.name] = col
        if self.has_connector:
            col.connector = self.connector

        col.table_name = self.source_name

        return self

    @raise_post_materialization
    def remove_col(self, name: str) -> 'Table':
        r"""Removes the column with name :obj:`name` from the table."""
        if name not in self._cols:
            raise KeyError(f"Cannot delete non-existing column '{name}'")
        del self._cols[name]
        return self

    def _check_time_cols(self, time_col: Optional[TimeColumn] = None,
                         end_time_col: Optional[TimeColumn] = None):
        r"""Raises a value error if `time_col` is same as `self.end_time_col`
        OR if `end_time_col` is same as `self.time_col`."""
        if time_col is not None:
            if (self.has_end_time_col() and self.end_time_col.name.lower()
                    == time_col.name.lower()):
                raise ValueError(
                    f"Time column '{time_col.name}' cannot be same as "
                    "self.end_time_col.")
        if end_time_col is not None:
            if (self.has_time_col() and self.time_col.name.lower()
                    == end_time_col.name.lower()):
                raise ValueError(
                    f"End time column '{end_time_col.name}' cannot be same as "
                    "self.time_col.")

    @raise_post_materialization
    def set_time_col(self, *args, **kwargs) -> 'Table':
        r"""Sets the time column of the table:

        .. code-block:: python

            table.set_time_col('DATE')

        Any arguments of :obj:`kumo.graph.TimeColumn` can be passed.
        """
        self._time_dtype_explicit = (
            kwargs.get('dtype') is not None
            or (args and not isinstance(args[0], str)
                and getattr(args[0], 'dtype', None) is not None))
        time_col = TimeColumn.cast(*args, **kwargs)

        if time_col is None:
            self.remove_time_col()
            return self

        self._check_time_cols(time_col=time_col)
        if self.is_time_col(time_col.name):
            self.time_col.update(time_col)
            return self

        self._time_col = time_col
        if self.has_connector:
            self._time_col.connector = self.connector
        self._time_col.table_name = self.source_name

        return self

    @raise_post_materialization
    def set_end_time_col(self, *args, **kwargs) -> 'Table':
        r"""Sets the time column of the table:

        .. code-block:: python

            table.set_end_time_col('DATE')

        Any arguments of :obj:`kumo.graph.TimeColumn` can be passed.
        """
        self._end_dtype_explicit = (
            kwargs.get('dtype') is not None
            or (args and not isinstance(args[0], str)
                and getattr(args[0], 'dtype', None) is not None))
        end_time_col = TimeColumn.cast(*args, **kwargs)

        if end_time_col is None:
            self.remove_end_time_col()
            return self

        self._check_time_cols(end_time_col=end_time_col)
        if self.is_end_time_col(end_time_col.name):
            self.end_time_col.update(end_time_col)
            return self

        self._end_time_col = end_time_col
        if self.has_connector:
            self._end_time_col.connector = self.connector
        self._end_time_col.table_name = self.source_name

        if self.has_col(end_time_col.name):
            del self[end_time_col.name]

        return self

    @raise_post_materialization
    def remove_time_col(self) -> 'Table':
        r"""Removes the time column from the table."""
        self._time_col = None
        return self

    @raise_post_materialization
    def remove_end_time_col(self) -> 'Table':
        r"""Removes the end time column from the table."""
        self._end_time_col = None
        return self

    @raise_post_materialization
    def remove_unused_cols(self) -> 'Table':
        r"""Removes any columns from the :class:`Table` for which
        materialization will be skipped. This avoids unnecessary loading of
        these columns in the first place."""
        removed_cols: Set[str] = set()
        for col in self.cols:
            if isinstance(col.encoder, Null):
                removed_cols.add(col.name)
                del self[col.name]

        if len(removed_cols) > 0:
            logger.info(f"Removing columns {removed_cols} without "
                        f"an encoding from table '{self.name}'")

        return self

    # Encoders & Related Properties ###########################################

    @property
    def num_feature_cols(self) -> int:
        r"""Computes the number of feature columns that do not have `Null()`
        encoders. If the encoder is `None`, it counts as not `Null()`"""
        return len([
            col for col in self.cols
            if not hasattr(col, 'encoder') or col.encoder != Null()
        ])

    def copy_encoders(self, table: 'Table'):
        r"""Copies the encoders of `table` to the current table."""
        for col in self.cols:
            col.encoder = table.col(col.name).encoder

    # Data Version Handling ###################################################

    @property
    def data_version(self) -> str:
        r"""The version of the data in the table.

        Raises:
            SourceTableNotFoundError: If a physical source table is absent.
            SourceTableUnavailableError: If a physical source table is missing
                or inaccessible and the provider cannot distinguish the cases.
        """
        if self.has_data:
            return self.data.data_version
        else:
            if self._data_version is not None:
                return self._data_version
            elif self.is_sql:
                # Custom SQL Table: ``source_name`` is a synthetic
                # ``custom_sql_<md5>`` alias for a pushed-down SELECT, not a
                # physical table, so probing ``get_data_version(source_name)``
                # has nothing to hit (it 404s on BigQuery). The synthetic name
                # IS the SQL definition identity that the ingest
                # ``SqlTableFingerprint`` path-id is built from, so returning it
                # keeps serialization and ingest on the same identity without a
                # connector call. The content ``data_version`` (referenced-table
                # versions), when resolved by the materialize path, is carried in
                # ``_data_version`` above and still takes precedence.
                return self.source_name
            elif self.has_connector:
                self._data_version = self.connector.get_data_version(
                    self.source_name)
                return self._data_version
            else:
                return DEFAULT_DATA_VERSION

    def set_table_data_version(self, job_id: str):
        r"""Set the data version of the table according to the cached
        versions in data backend cache based on job_id. This is used to
        guarantee that the tables are consistently using the same data
        version throughout a workflow."""
        backend_cache = get_active_cache()
        data_version = (backend_cache.table_version_cache.get_for_job(
            job_id, self.data_source, self.source_name))
        logger.info(f"Setting data version for table {self.name} to "
                    f"{data_version}, which was cached for job id {job_id}")
        if data_version is not None and self.has_connector:
            self._data_version = data_version
            backend_cache.table_version_cache.save_in_memory(
                self.connector.name, self.source_name, data_version)

    # Metadata and Type Inferral ##############################################
    #
    # This set of methods helps infer table metadata (missing columns, dtypes,
    # and stypes) for a given Table object. The main entrypoint is
    # infer_metadata, which infers any missing columns, infers dtypes for all
    # columns without dtypes, and subsequently infers stypes for columns
    # without stypes (based on dtype or statistics). If at any point stypes are
    # updated, the dtypes are updated to match, e.g. via edit_stypes.

    def _infer_stype_from_dtype(
        self,
        dtype: Dtype,
    ) -> Optional[Stype]:
        r"""Infers a semantic type from a data type where possible. If not
        possible, returns None.

        The rule itself lives in :mod:`kumo_core.stype_inference`, so the
        engine, this package and the SDK all classify a column alike.
        """
        return infer_stype_from_dtype(dtype)

    def _infer_stype_from_stats(
        self,
        dtype: Dtype,
        stats: StypeInferralStats,
    ) -> Stype:
        r"""Infers a semantic type from a data type given a set of statistics
        for the column with that data type. An stype should always be returned
        by this function.

        The thresholds and the order they are applied in live in
        :mod:`kumo_core.stype_inference`; this method supplies the statistics
        and turns "no rule matched" into the error a table can report.
        """
        stype = infer_stype_from_stats(
            dtype,
            num_unique=stats.num_unique.value,
            unique_fraction=stats.unique_fraction.value,
            num_unique_multi=stats.num_unique_multi.value,
            multicategorical_proven=stats.multicategorical_proven,
        )
        if stype is None:
            raise ValueError(
                f"Table {self.name} failed inferring the semantic type for a "
                f"column with data type {dtype} and statistics {stats}.")
        return stype

    def infer_stypes(self, skip_cache: bool = False,
                     require_stats: bool = True) -> 'Table':
        r"""Infer semantic type for each column in the table.

        Args:
            skip_cache (bool): Skip cache when require stats for inference.
                (default: :obj:`False`)
            require_stats (bool): Require stats when infer the semantic type,
                if it is :obj:`False`, we will infer semantic type only based
                on column name and dtype. (default: :obj:`True`)
        """
        # Custom SQL cost gate (Option A): the stats sample executes the user's
        # query, so gate it here — the single boundary every caller
        # (``infer_metadata``, direct ``infer_stypes``, ``Graph.infer_stypes``)
        # passes through — rather than at one entry point. When the warehouse
        # scan estimate is over budget, drop to free name/dtype heuristics.
        if require_stats and self.is_sql:
            sql_sample = self._read_inference_sample()
            # A cost-gated query has no sample, while a valid empty query has
            # a zero-row sample. Neither can provide uniqueness statistics;
            # retain every warehouse-described column and use the same
            # dtype/name heuristic fallback in both cases.
            if sql_sample is None or sql_sample.empty:
                require_stats = False

        stype_inferred_from_dtype = lambda dtype: dtype in [
            Dtype.date,
            Dtype.time,
            Dtype.timedelta,
            Dtype.bool,
            Dtype.floatlist,
            Dtype.intlist,
            Dtype.stringlist,
            Dtype.unsupported,
        ]

        # First, attempt to infer semantic type based on data types and column
        # names:
        for col in self.cols:
            if col.name == self.pkey_name:
                col.stype = Stype.ID
            elif col.stype is not None:
                continue
            elif stype_inferred_from_dtype(col.dtype):
                col.stype = self._infer_stype_from_dtype(col.dtype)
                continue
            elif (named_stype := infer_stype_from_name(col.name, col.dtype)) \
                    is not None:
                col.stype = named_stype

        if not require_stats:
            # Determine the stype of the columne without stats, the preference
            # is numerical, timestamp, categorical, or sequence type.
            # TODO(dong) unify the `_infer_stype_from_dtype` function since
            # the function currently optimize for stype precision and here
            # we optimize for stype coverage.
            for col in self.cols:
                if col.stype is not None:
                    continue
                elif Stype.numerical.supports_dtype(col.dtype):
                    col.stype = Stype.numerical
                elif Stype.timestamp.supports_dtype(col.dtype):
                    col.stype = Stype.timestamp
                elif Stype.categorical.supports_dtype(col.dtype):
                    col.stype = Stype.categorical
                elif Stype.sequence.supports_dtype(col.dtype):
                    col.stype = Stype.sequence
                else:
                    col.stype = Stype.unsupported
        else:
            # Next, for columns that we still need to infer semantic types for,
            # compute the requisite statistics to do so:
            cols_to_infer = [col for col in self.cols if col.stype is None]
            declared_multicats_to_resolve = [
                col for col in self.cols
                if (col.stype == Stype.multicategorical and col.dtype ==
                    Dtype.string and col.multicategorical_delimiter is None)
            ]
            cols_needing_stats = cols_to_infer + declared_multicats_to_resolve
            if cols_needing_stats:
                computed_stats = self.compute_col_stats(
                    cols=cols_needing_stats,
                    col_stat_types=StypeInferralStats.REQUIRED_STATS,
                    approx=True,
                    num_rows=NUM_ROWS_FOR_APPROX_STATS,
                    skip_cache=skip_cache,
                    table_is_finalized=False,
                )

                # Semantic type inferral based on statistics:
                for col in cols_to_infer:
                    stats = computed_stats[col.name]
                    # Pre-ingest stats coverage may be partial when a sample
                    # realizes as a native list column. The dtype-supports
                    # gate rejects scalar-only stats but accepts
                    # ``NUM_UNIQUE_MULTI``. Detect that list-realized shape
                    # from the stats keys and infer multicategorical.
                    if NUM_UNIQUE not in stats:
                        if NUM_UNIQUE_MULTI in stats:
                            if col.dtype == Dtype.string:
                                col.dtype = Dtype.stringlist
                            col.stype = Stype.multicategorical
                        else:
                            col.stype = Stype.unsupported
                        continue
                    num_unique: Stats = stats[NUM_UNIQUE]
                    unique_fraction = stats[UNIQUE_FRACTION]
                    num_unique_multi = num_unique
                    if NUM_UNIQUE_MULTI in stats:
                        num_unique_multi = stats[NUM_UNIQUE_MULTI]

                    # Scalar-string multicategorical inference is valid only
                    # when sampling proves an encoding that DiskGraph can
                    # normalize into a native list. Punctuation such as ':'
                    # in time values and bracketed JSON-like strings must not
                    # create an stype/storage-shape mismatch.
                    separator_stat = stats.get(MULTI_CATEGORIES_SEPARATOR)
                    separator = (col.multicategorical_delimiter
                                 or (separator_stat.value
                                     if separator_stat is not None else None))
                    multicategorical_proven = (col.dtype.is_list() or separator
                                               in MULTICAT_DELIMITERS)

                    stats = StypeInferralStats(
                        num_unique=num_unique,
                        unique_fraction=unique_fraction,
                        num_unique_multi=num_unique_multi,
                        multicategorical_proven=multicategorical_proven,
                    )

                    col.stype = self._infer_stype_from_stats(col.dtype, stats)
                    if (col.stype == Stype.multicategorical
                            and col.dtype == Dtype.string):
                        col.multicategorical_delimiter = separator

                # Caller-declared scalar multicategorical columns use the
                # delimiter already computed by the same approximate-stats
                # sample. Explicit delimiters remain authoritative.
                for col in declared_multicats_to_resolve:
                    separator_stat = computed_stats.get(
                        col.name, {}).get(MULTI_CATEGORIES_SEPARATOR)
                    if separator_stat is not None:
                        col.multicategorical_delimiter = separator_stat.value

        for col in self.cols:
            if (col.stype != Stype.multicategorical
                    or col.dtype != Dtype.string):
                col.multicategorical_delimiter = None

        # Fill out any remaining semantic types for pkey and fkeys with the ID
        # type
        for fkey in self.fkeys:
            if fkey.stype is None:
                fkey.stype = Stype.ID
        if self.has_pkey() and self.pkey.stype is None:
            self.pkey.stype = Stype.ID

        return self

    @blocking_method()
    def _infer_dtypes_from_stypes_blocking(
        self,
        native_temporal_cols: Optional[Set[str]] = None,
    ) -> Dict[str, Dtype]:
        return self.infer_dtypes_from_stypes(native_temporal_cols)

    def infer_dtypes_from_stypes(
        self,
        native_temporal_cols: Optional[Set[str]] = None,
    ) -> Dict[str, Dtype]:
        r"""Updates stype of the input columns, with cast (if needed)

        ``native_temporal_cols`` is transient source-schema evidence. Columns
        that the connector reports as native date/time values need no value
        sample to infer an encoded representation.

        Returns: Dict[str, Dtype]: Returns mapping of column names to
            their new dtypes, if the dtype has changed.
        """
        native_temporal_cols = native_temporal_cols or set()
        samples_df: Any = _SAMPLE_UNSET
        cols_to_new_dtypes: Dict[str, Dtype] = {}

        for col in self._data_cols:
            if col.stype is None:
                raise ValueError(
                    f"Stype ({col.stype}) for column ({col.name}) with "
                    f"dtype ({col.dtype}) is None. Must infer stype first.")

            if col.stype == Stype.timestamp:
                try:
                    if (col.timestamp_unit is not None
                            or TS_FORMAT_METADATA in col.metadata):
                        # Gate: a column that already carries a unit or format
                        # hint is left alone; the detector never re-infers a
                        # hinted column. Only fix the dtype if needed.
                        if col.dtype not in {Dtype.date, Dtype.time}:
                            # Default to time:
                            col.dtype = Dtype.time
                            cols_to_new_dtypes[col.name] = col.dtype
                        continue

                    if col.name in native_temporal_cols:
                        # The provider schema is sufficient evidence for a
                        # native date/time column. This evidence is deliberately
                        # not persisted as a unit, format, or new column type.
                        continue

                    # No hint yet: read a (cost-gated) sample and let the
                    # detector decide unit XOR format (or native / unresolved).
                    # For Custom SQL tables the cost gate may defer the sample
                    # (``_read_inference_sample`` returns None) to avoid scanning
                    # a large query; in that case keep the declared dtype.
                    if samples_df is _SAMPLE_UNSET:
                        samples_df = self._read_inference_sample()
                    if samples_df is None:
                        # SQL cost gate deferred the sample; keep the declared
                        # dtype rather than scanning a large query.
                        col.dtype = col.dtype or Dtype.time
                        cols_to_new_dtypes[col.name] = col.dtype
                        continue
                    sample = samples_df[col.name]

                    if is_timedelta64_dtype(sample.dtype):
                        # A timedelta duration coerced to a timestamp is a path
                        # the detector does not model; persist the native
                        # duration resolution as the unit.
                        import numpy as np
                        col.timestamp_unit = np.datetime_data(sample.dtype)[0]
                    else:
                        # The detector is the single producer of the hint. It
                        # emits a UNIT hint for an epoch column, a FORMAT hint
                        # for an encoded-date / string column, or NATIVE /
                        # UNRESOLVED (which persist neither).
                        hint = infer_temporal_hint(
                            sample,
                            declared_unit=col.timestamp_unit,
                            declared_format=col.metadata.get(
                                TS_FORMAT_METADATA),
                        )
                        storage = hint_to_storage(hint)
                        # A UNIT hint writes ``timestamp_unit``; a FORMAT hint
                        # writes ``metadata['TS_FORMAT']``; NATIVE / UNRESOLVED
                        # write neither.
                        if storage.timestamp_unit is not None:
                            col.timestamp_unit = storage.timestamp_unit
                        if storage.ts_format is not None:
                            col.metadata[TS_FORMAT_METADATA] = storage.ts_format

                    # Set the dtype: a date column stays date, everything else
                    # becomes time.
                    if col.dtype == Dtype.date:
                        continue
                    # TODO (Dulloor): When should the `dtype` be `Dtype.date`
                    col.dtype = Dtype.time
                    cols_to_new_dtypes[col.name] = col.dtype
                    continue
                except Exception as e:
                    col.dtype = col.dtype or Dtype.time
                    # For Snowflake OutOfBoundsDatatime mentioned in github
                    # issue: https://github.com/kumo-ai/kumo/issues/13572
                    # The error is inside the snowflake connector convert data
                    # to Pandas DataFrame. The sampling logic will fail but
                    # the ingestion logic works fine. Therefore, we set the
                    # default dtype based on customer settings or time.
                    logger.warning(
                        f"Infer dtype failed for {col} with error"
                        f" {e}, set the default dtype to {col.dtype}.")

            if col.stype == Stype.unsupported:
                continue

            if col.stype.supports_dtype(col.dtype):
                continue

            if not col.stype.supports_cast_from_dtype(col.dtype):
                raise ValueError(
                    f"Stype ({col.stype}) for column ({col.name}) is not "
                    f"compatible for the column's dtype ({col.dtype})")

            if col.stype == Stype.numerical and col.dtype == Dtype.time:
                if samples_df is _SAMPLE_UNSET:
                    samples_df = self._read_inference_sample()
                if samples_df is None:
                    # SQL cost gate deferred the sample; keep the user's
                    # numerical override without a scan to verify the cast.
                    continue

                # if the inferred dtype was time but the user selects numeric
                # try casting to numeric - the col might, e.g., be in seconds
                # rather than a true time col
                cast_col_to_numeric(samples_df[col.name])
                col.dtype = Dtype.int64
                col.stype = Stype.numerical
                cols_to_new_dtypes[col.name] = col.dtype

            else:
                raise ValueError(
                    f"Infer dtype from stype for column ({col.name}) of "
                    f"Stype ({col.stype}) is not supported. Column's dtype "
                    f"is ({col.dtype})")

        return cols_to_new_dtypes

    def edit_stypes(
        self,
        cols_to_stypes: Dict[str, Stype],
    ) -> 'Table':
        r"""Updates stype of the input columns, with cast (if needed)
        Args:
            cols_to_stypes (Dict[str, Stype]): A dict containing
            the mapping of column names to desired kumo Stype.
        """
        # TODO(Dulloor): Enable this functionality at the column level
        # as part of the `stype` setter.

        # Update the stypes in place and call `infer_dtypes_from_stypes`
        for col in self._data_cols:
            if col.name in cols_to_stypes:
                col.stype = cols_to_stypes[col.name]

        self.infer_dtypes_from_stypes()
        return self

    def _set_dtype_from_metadata(self, table: TableOrViewMetadataConfig,
                                 include_cols: Optional[Set[str]] = None,
                                 exclude_cols: Optional[Set[str]] = None,
                                 drop_id_cols: bool = False) -> None:
        r"""Set dtype of `self.cols`, `self.pkey`, `self.fkeys`, and
        `self.time_col` based on metadata in `table`."""

        if include_cols and exclude_cols:
            raise ValueError(
                "Both 'include_cols' and 'exclude_cols' cannot be set.")

        # Filter result based on `include_cols` and `exclude_cols`.
        if include_cols is not None:
            if (table.pkey is not None and table.pkey.name not in self
                    and table.pkey.name not in include_cols):
                table.pkey = None

            table.fkeys = [
                fkey for fkey in table.fkeys
                if fkey.name in self or fkey.name in include_cols
            ]

            if (table.time_col is not None and table.time_col.name not in self
                    and table.time_col.name not in include_cols):
                table.time_col = None

            if table.end_time_col in include_cols:
                logger.warning('end_time_col cannot be added as a feature '
                               'column. Setting it in include_cols will '
                               'be ignored.')
                include_cols.remove(table.end_time_col)

            table.cols = [
                col for col in table.cols
                if col.name in self or col.name in include_cols
            ]

        elif exclude_cols is not None:
            if (table.pkey is not None and table.pkey.name not in self
                    and table.pkey.name in exclude_cols):
                table.pkey = None

            table.fkeys = [
                fkey for fkey in table.fkeys
                if fkey.name in self or fkey.name not in exclude_cols
            ]

            if (table.time_col is not None and table.time_col.name not in self
                    and table.time_col.name in exclude_cols):
                table.time_col = None

            if (table.end_time_col is not None
                    and table.end_time_col.name in exclude_cols):
                table.end_time_col = None

            table.cols = [
                col for col in table.cols
                if col.name in self or col.name not in exclude_cols
            ]

        # Modify existing table attributes and set new ones:
        if table.pkey is not None:
            if self.has_pkey():
                if self.pkey_name != table.pkey.name:
                    warnings.warn(
                        f"Inferred primary key '{table.pkey.name}' differs "
                        f" from set primary key '{self.pkey.name}'. "
                        f"Ignoring inferred primary key.")
                else:
                    self.pkey.update(table.pkey, override=False)
            else:
                self.set_pkey(table.pkey.name, dtype=table.pkey.dtype)

        for fkey in table.fkeys:
            if self.has_fkey(fkey.name):
                self[fkey.name].update(fkey, override=False)
            else:
                self.add_fkey(fkey.name, dtype=fkey.dtype)

        if table.time_col is not None:
            if self.time_col_name != table.time_col.name:
                warnings.warn(
                    f"Inferred time column '{table.time_col.name}' differs "
                    f"from set time column '{self.time_col_name}'. Ignoring "
                    f"inferred time column.")
            elif self.has_time_col():
                self.time_col.update(table.time_col, override=False)
            else:
                self.set_time_col(table.time_col.name,
                                  dtype=table.time_col.dtype)

        if table.end_time_col is not None:
            if self.end_time_col_name != table.end_time_col.name:
                warnings.warn(
                    f"Inferred end time column '{table.end_time_col.name}' "
                    f"differs from set end time column "
                    f"'{self.end_time_col.name}'. "
                    f"Ignoring inferred time column.")
            elif self.has_end_time_col():
                self.end_time_col.update(table.end_time_col, override=False)
            else:
                self.set_end_time_col(table.end_time_col.name,
                                      dtype=table.end_time_col.dtype)

        ignored_id_cols: Set[str] = set()
        for col in table.cols:
            if self.is_time_col(col.name):
                if col.dtype.is_timestamp() and not self._time_dtype_explicit:
                    self.time_col.dtype = col.dtype
                self.time_col.update(col, override=False)

            if self.is_end_time_col(col.name):
                if col.dtype.is_timestamp() and not self._end_dtype_explicit:
                    self.end_time_col.dtype = col.dtype
                self.end_time_col.update(col, override=False)

            if col.name in self:
                self[col.name].update(col, override=False)
                if ((self.is_pkey(col.name) or self.has_fkey(col.name))
                        and self.has_col(col.name)):
                    # In case a primary key column or a fkey column is also a
                    # feature column, force update the feature column.
                    self.col(col.name).update(col, override=False)
            elif drop_id_cols and contains_identifier(col.name):
                # Not adding potential foreign or primary keys as feature
                # columns by default. Collect the name to print out a warning:
                ignored_id_cols.add(col.name)
            else:
                self.add_col(**asdict(col))

        if len(ignored_id_cols) > 0:
            warnings.warn(
                f"Not including {ignored_id_cols} as feature columns in table "
                f"'{self.name}' since they potentially may refer to foreign "
                f"keys or primary keys. Consider explicitly including these "
                f"columns via `graph[table_name].add_col(...)` before calling "
                f"'infer_metadata', or explicitly excluding these columns via "
                f"`graph.infer_metadata(exclude_cols=[...])` to suppress this "
                f"warning")

        for col in table.all_cols():
            if col.dtype is Dtype.unsupported:
                logger.warning(f"{col.name} has unsupported raw dtype")

    @blocking_method()
    def _infer_metadata_blocking(
        self,
        include_cols: Optional[Set[str]] = None,
        exclude_cols: Optional[Set[str]] = None,
        drop_id_cols: bool = False,
    ) -> 'Table':
        return self.infer_metadata(include_cols, exclude_cols, drop_id_cols)

    def infer_metadata(
        self,
        include_cols: Optional[Set[str]] = None,
        exclude_cols: Optional[Set[str]] = None,
        drop_id_cols: bool = False,
    ) -> 'Table':
        r"""Infers unspecified metadata of the table:

        * Infers remaining columns that exist in the table.
        * Fills missing :obj:`stype` attribute of columns.
        * Fills missing :obj:`dtype` attribute of columns.

        Args:
            include_cols (Set[str], optional): A set containing the names of
                columns to include in inferral, if not already set. Cannot be
                used with :obj:`exclude_cols` (default: :obj:`None`).
            exclude_cols (Set[str], optional): A set containing the names of
                columns to exclude in inferral, if not already set. Cannot be
                used with :obj:`include_cols` (default: :obj:`None`).
            drop_id_cols (bool, optional): Ignore a column with `_ID_` (case
                insensitive) in the name. (default :obj:`True`).
        """
        if self.is_sql:
            return self._infer_sql_metadata(include_cols, exclude_cols,
                                            drop_id_cols)

        # Obtain metadata from backing connector:
        table_meta_list = self.connector.get_table_metadata([self.source_name])
        if not table_meta_list:
            raise ValueError(f"Failed to read schema for table '{self.name}' "
                             f"in connector '{self.connector}'. "
                             f"Please check whether table was "
                             f"accidentally dropped, renamed or emptied "
                             f"since it was first imported into Kumo.")
        table = table_meta_list[0]
        self._set_dtype_from_metadata(table, include_cols, exclude_cols,
                                      drop_id_cols)
        return self.infer_stypes()

    def _infer_sql_metadata(
        self,
        include_cols: Optional[Set[str]] = None,
        exclude_cols: Optional[Set[str]] = None,
        drop_id_cols: bool = False,
    ) -> 'Table':
        r"""Infer ordered outputs from one warehouse-authoritative description.

        Stypes use one bounded, cost-gated sample and are frozen into the table
        definition and fingerprint rather than corrected after ingestion.
        """
        metadata = self.connector.get_query_metadata(self.custom_sql)
        names = [col.name for col in metadata.cols]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise DuplicateColumnsError(
                f"Custom SQL output columns {duplicates} are duplicated. "
                "Give every output column a unique alias.")
        self._raise_on_missing_sql_columns(metadata)
        self._set_dtype_from_metadata(metadata, include_cols, exclude_cols,
                                      drop_id_cols)
        self._sql_source_metadata = metadata
        self._sql_column_order = [
            name for name in names if name in self.data_col_dict
        ]
        self._raise_on_unsupported_sql_dtypes(metadata)
        # Best-effort inheritance pre-pass: seed stypes for columns that trace
        # to a referenced Kumo table before the heuristic/stats inference below
        # fills whatever remains unset.
        self.infer_stypes_before_ingestion()
        # ``infer_stypes`` applies the Option A cost gate itself (dropping to
        # heuristics when the sample is over budget). The unmaterialized preview
        # stats are captured separately by the validate path via
        # ``persist_sql_preview_col_stats`` (keyed by definition identity), so
        # this inference does not need to persist them.
        return self.infer_stypes(require_stats=True)

    def set_sql_reference_stypes(
        self,
        connector_type: ConnectorType,
        reference_stypes: Dict[str, Dict[str, Stype]],
    ) -> None:
        r"""Provide normalized stypes from registered source tables.

        The REST layer resolves sibling-table state because ``Table`` cannot
        access the state database itself.
        """
        self._sql_connector_type = connector_type
        self._sql_reference_stypes = reference_stypes

    def infer_stypes_before_ingestion(self) -> None:
        r"""Inherit compatible stypes for direct pass-through outputs.

        Explicit stypes win; computed, ambiguous, and untraceable outputs fall
        through to normal heuristic inference.
        """
        reference_stypes = self._sql_reference_stypes
        if not reference_stypes or self._sql_connector_type is None:
            return
        assert self.custom_sql is not None
        # Mutate the feature columns in place, mirroring ``infer_stypes`` (which
        # runs next); ``data_cols`` would hand out throwaway ``to_base`` copies.
        for col in self.cols:
            if col.stype is not None or col.name == self.pkey_name:
                continue
            traced = source_table_for_col(self.custom_sql,
                                          self._sql_connector_type, col.name)
            if traced is None:
                continue
            table_name, source_col = traced
            # Match on the bare table name: ``reference_stypes`` is keyed by the
            # raw Kumo table's (unqualified) source name, so a qualified SQL
            # reference still reconciles.
            leaf = normalized_sql_identifier_leaf(table_name,
                                                  self._sql_connector_type)
            inherited = reference_stypes.get(leaf, {}).get(source_col)
            if (inherited == Stype.multicategorical
                    and col.dtype == Dtype.string
                    and col.multicategorical_delimiter is None):
                continue
            if inherited is not None and inherited.supports_dtype(col.dtype):
                col.stype = inherited

    def _raise_on_missing_sql_columns(
            self, metadata: TableOrViewMetadataConfig) -> None:
        r"""Reject requested columns absent from the query result."""
        output_names = {col.name for col in metadata.all_cols()}
        missing = sorted(col.name for col in self.data_cols
                         if col.name not in output_names)
        if missing:
            raise ValueError(
                f"Columns {missing} are not produced by the SQL query for "
                f"table '{self.name}'. Select or alias them in your SQL, or "
                f"remove them from the table definition.")

    def _raise_on_unsupported_sql_dtypes(
            self, metadata: TableOrViewMetadataConfig) -> None:
        r"""Reject unsupported warehouse-described output types.

        The authoritative query schema is checked so a request dtype override
        cannot mask an unsupported source representation.
        """
        query_dtypes = {col.name: col.dtype for col in metadata.all_cols()}
        table_col_names = {col.name for col in self.data_cols}
        unsupported = sorted(name for name in table_col_names
                             if query_dtypes.get(name) is Dtype.unsupported)
        if unsupported:
            raise UnsupportedColumnTypeError(
                f"Custom SQL columns {unsupported} have an unsupported output "
                f"type. Cast each to a supported type in your SQL (e.g. "
                f"`CAST(<expr> AS STRING) AS {unsupported[0]}`); an unaliased "
                f"computed column also needs an explicit alias.")

    # Encoder Inferral ########################################################

    def get_encoder_stats(self, **kwargs) -> ExactStats:
        # Fill in approximate statistics for columns
        approx_stats: Dict[str, Dict[ColStatType, Stats]] = \
            self.compute_col_stats(
                col_stat_types=EncoderGenerator.REQUIRED_APPROX_STATS,
                approx=True, **kwargs)

        # Fill in exact statistics for columns
        exact_stats: Dict[str, Dict[ColStatType, Stats]] = \
            self.compute_col_stats(
                col_stat_types=EncoderGenerator.REQUIRED_EXACT_STATS,
                approx=False, **kwargs)

        # Merge the two
        col_stats: Dict[ColumnName, Dict[ColStatType, StatsConfig]] = {}
        all_col_names: Set[str] = approx_stats.keys() | exact_stats.keys()
        for col_name in all_col_names:
            # Prefer exact:
            exact: Dict[ColStatType, StatsConfig] = {
                k: v.config()
                for k, v in exact_stats[col_name].items()
            }
            approx: Dict[ColStatType, StatsConfig] = {
                k: v.config()
                for k, v in approx_stats[col_name].items() if k not in exact
            }
            col_stats[col_name] = dict(**approx, **exact)

        return ExactStats(table_stats={}, col_stats=col_stats)

    def infer_encoders(self, encoder_stats: Optional[ExactStats] = None,
                       **kwargs) -> 'Table':
        r"""Infers unspecified encoders of the table, according to a sample
        of the table rows.

        Args:
            **kwargs: Keyword arguments that are accepted by
                :obj:`Connector.read_table_sample`.
        """
        # semantic types must be inferred by this point

        # Set reasonable default for encoder inferral if not specified
        # The read_table_sample default here is 1 row, which is clearly
        # insufficient.
        if 'num_rows' not in kwargs and 'frac' not in kwargs:
            kwargs['num_rows'] = NUM_ROWS_FOR_APPROX_STATS

        if kwargs.get('num_rows', 1000) < 1000:
            raise ValueError(
                "Inferring encoders requires at least reading 1,000 rows")

        raise_on_error = kwargs.get('raise_on_error', True)

        # Only compute stats for columns that need encoder inferral:
        cols_needing_encoders = [
            col for col in self.cols if col.encoder is None
        ]
        if encoder_stats is None and cols_needing_encoders:
            logger.info(f'Computing encoder stats for table {self.name}')
            encoder_stats: ExactStats = self.get_encoder_stats(
                cols=cols_needing_encoders, **kwargs)

        # Set the encoders for each column:
        for col in cols_needing_encoders:
            try:
                stats_configs: Dict[ColStatType,
                                    StatsConfig] = encoder_stats.col_stats.get(
                                        col.name, {})
                stats: Dict[ColStatType, Stats] = {
                    k: Stats.from_config(v)
                    for k, v in stats_configs.items()
                }
                col.encoder = EncoderGenerator.get_encoder(
                    col, stats, table_name=self.table_name)
            except Exception as e:
                if raise_on_error:
                    raise RuntimeError(
                        f"Failed to infer encoder for column '{self.name}.'"
                        f"'{col.name}' version: {self.data_version}. "
                        f"Cause: {e!r}") from e
                else:
                    logger.error(
                        f"Failed to infer encoder for column '{self.name}.'"
                        f"'{col.name}' version: {self.data_version} {e}")
                    col.encoder = Null()
        return self

    # Materialization #########################################################
    #
    # Materialization is the process by which a Table configuration object
    # generates a TableData data object. The main entry point is `materialize`,
    # which in turn performs ingestion and materialization. Soon, it will also
    # perform exact statistics computation, but this does not happen yet.

    def validate(
        self,
        raise_on_error: bool = True,
        require_key: bool = True,
        ignore_encoders: bool = False,
    ) -> bool:
        """Ensure that the Table has valid fields.

        Args:
            raise_on_error (bool): if true, raise a ValueError for any invalid
                fields
            require_key (bool): Require that the table has at least one primary
                or foreign key. Set to False if you are validating a table that
                has not been added to a graph.
            ignore_encoders (bool): if true, ignore the validations related to
                column encoders.

        Returns:
            bool: returns true iff the table is valid
        """
        status = True

        # 1. The table has at least one primary key or one foreign key:
        if require_key and self.pkey is None and len(self.fkeys) == 0:
            status = False
            logging.warn_or_raise(
                f"Table '{self.name}' is expected to have at least one "
                f"primary key or one foreign key", raise_on_error)

        # Float/double PK columns yield zero nodes at ingest; reject so the user declares an int key.
        if self.pkey is not None and self.pkey.dtype is not None \
                and self.pkey.dtype.is_float():
            status = False
            logging.warn_or_raise(
                f"The primary key '{self.pkey.name}' of table '{self.name}' "
                f"has dtype '{self.pkey.dtype}'. ID/key columns must be "
                f"integer or string; float/double key columns are not "
                f"supported and would be dropped at ingest (yielding zero "
                f"nodes). Please set the dtype of '{self.pkey.name}' to "
                f"'int'.", raise_on_error)

        # Get fingerprint for validation
        try:
            table_fingerprint = self.fingerprint()
        except AssertionError as e:
            logger.warning(
                "Failed to generate exact fingerprint, deferring to "
                "approximate fingerprint: %s", e)
            table_fingerprint = self.partial_fingerprint()

        # Load col stats if we need to validate encoders.
        if not ignore_encoders:
            get_active_cache().load_col_stats_cache(table_fingerprint,
                                                    self.data_version)

        for col in self.cols:
            # 2. Validate the correctness of each column.
            status &= col.validate(raise_on_error)

            if not ignore_encoders:
                # passing data version is a slow operation.
                required_stats = [ColStatType.NUM_UNIQUE] \
                    if col.stype == Stype.ID else []
                # DG-native path: read directly from BackendCacheManager
                # (populated by diskgraph ingest via
                # copy_ingest_stats_to_backend_cache). The severed legacy
                # factory no longer offers get_col_stats.
                data_backend_cache = get_active_cache()
                stats = data_backend_cache.get_col_stats(
                    table_fingerprint, {col.base_config(): required_stats},
                    data_version=self.data_version, approx=True)
                status &= col.validate_encoders(self.data_version,
                                                raise_on_error, stats)

        return status

    @property
    def pandas(self) -> PandasView:
        r"""Returns a :class:`pandas` view to access the data of the underlying
        :obj:`data_source`:

        .. code-block:: python

            graph[table_name].pandas()  # all data
            graph[table_name].pandas[:1000]  # first 1000 rows
        """
        cols = self.data_cols
        if self.has_pkey():
            cols = [col for col in cols if col.name != self.pkey.name]
        return PandasView(self.connector, self.source_name, cols, self.pkey)

    @property
    def pandas_with_pkey(self) -> PandasView:
        cols = self.data_cols
        if self.has_pkey():
            # Ensure pkey shows first
            cols = [self.pkey
                    ] + [col for col in cols if col.name != self.pkey.name]
        return PandasView(self.connector, self.source_name, cols, None)

    def compute_exact_stats(
        self,
        is_minimal: bool,
        cols: Optional[List[Column]] = None,
    ):
        r"""Computes all the exact stats on this table, if they are not
        already present. These stats are not stored on the columns, but are
        instead registered in the data backend cache."""
        cols = cols or self.cols

        # Load the stats from the cache if they exist, so we do not re-compute
        # here:
        data_backend_cache: BackendCacheManager = get_active_cache()
        data_backend_cache.load_col_stats_cache(
            self.fingerprint(),
            self.data_version,
        )

        if is_minimal:
            stat_dict = get_minimal_stats(cols)
            logger.info(
                f"Computing minimal statistics for table {self.name}, on "
                f"columns {[c.name for c in cols]}")
        else:
            stat_dict = get_full_stats(cols)
            logger.info(f"Computing full statistics for table {self.name}, on "
                        f"columns {[c.name for c in cols]}")

        stat_dict = {k.base_config(): v for k, v in stat_dict.items()}

        # DG-native path: col stats are populated into the backend cache by
        # diskgraph ingest (copy_ingest_stats_to_backend_cache). The legacy
        # Spark compute path is gone — verify the requested stats are
        # present in the cache and raise loudly on miss instead of silently
        # falling into deleted compute.
        missing = data_backend_cache.get_missing_col_stats(
            self.fingerprint(), stat_dict, self.data_version, approx=False)
        if missing:
            missing_desc = ', '.join(f"{cfg.name}:{[s.name for s in stats]}"
                                     for cfg, stats in missing.items())
            raise UnsupportedInDiskgraphError(
                "compute_exact_stats: requested col stats missing from the "
                f"DiskGraph-populated cache for table '{self.name}'@"
                f"{self.data_version}. DiskGraph ingest populates this "
                "cache; a miss means either ingest hasn't run for this "
                "table / version, or the requested stat type is not "
                f"emitted by DG. Missing: {missing_desc}.")

        # Persist to the cache:
        data_backend_cache.save_col_stats_cache(
            self.fingerprint(),
            self.data_version,
        )

    # Statistics ##############################################################

    def calculate_stats_for_column(
        self,
        col: Column,
    ) -> Dict[ColStatType, Stats]:
        r"""Automatically infer the stats to calculate and compute
        stats for the column.
        """
        all_possible_stat_types = EncoderGenerator.REQUIRED_STATS
        valid_stat_types = [
            stat_type for stat_type in all_possible_stat_types
            if stat_type.supports(col.dtype)
            and stat_type.supports_stype(col.stype)
        ]

        exact_stat_types = [
            stat_type for stat_type in valid_stat_types
            if stat_type in EncoderGenerator.REQUIRED_EXACT_STATS
        ]
        approx_stat_types = [
            stat_type for stat_type in valid_stat_types
            if stat_type in EncoderGenerator.REQUIRED_APPROX_STATS
        ]

        exact_stats = self.compute_col_stats([col], exact_stat_types,
                                             approx=False)
        approx_stats = self.compute_col_stats([col], approx_stat_types,
                                              approx=True)

        ret = exact_stats.get(col.name, {})
        ret.update(approx_stats.get(col.name, {}))
        return ret

    def compute_col_stats(
        self,
        cols: Optional[List[Column]] = None,
        col_stat_types: Optional[List[ColStatType]] = ALL_COL_STAT_TYPES,
        approx: bool = False,
        skip_cache: bool = False,
        table_is_finalized: bool = True,
        **kwargs,
    ) -> Dict[str, Dict[ColStatType, Stats]]:
        r"""Computes requested stats for columns in the table, with options
        to compute approximate stats on a sample of the table rows or exact
        stats on the entire data.

        Args:
            cols (List[Column]): List of columns to compute stats.
                Compute stats for all columns by default
                (default: :obj:`None`)
            col_stat_types (List[ColStatType]): List of column stat types to
                compute. Compute all column stats by default
                (default: :obj:`ALL_COL_STAT_TYPES`)
            approx (bool): Compute approximate column stats. Table stats
                are always exact (default: :obj:`True`)
            **kwargs: Keyword arguments that are passed down to the backend
                when they call `compute_col_stats`.
        Returns:
            Dict[str, Dict[ColStatType, Stats]]: For each table, dictionary
            mapping ColStatType to computed stats.
        """
        if not col_stat_types:
            return {}

        cols = cols or self.cols
        data_version = self._resolve_data_version(skip_cache)
        request = self._build_supported_stats_request(cols, col_stat_types)

        if not table_is_finalized:
            # ``infer_stypes`` is the only caller passing
            # ``table_is_finalized=False`` -- it runs before DG ingest, so
            # the DG cache is guaranteed empty and we go straight to the
            # connector sample path.
            col_stats_by_config = self._compute_stats_via_connector_sample(
                request, approx, data_version, **kwargs)
        else:
            col_stats_by_config = self._read_dg_cache_with_fallback(
                request, approx, data_version, **kwargs)

        return self._format_by_col_name(cols, col_stats_by_config)

    def _read_inference_sample(self) -> Optional[pd.DataFrame]:
        r"""Read a raw (uncast) sample for pre-ingest dtype inference — from the
        query for a Custom SQL Table, else from the physical source table.

        For a Custom SQL Table the sample executes the query, so it is subject
        to the same cost gate as stype inference; returns ``None`` when the gate
        defers it, so callers fall back to the declared dtype rather than
        scanning a large query."""
        if self.is_sql:
            if self._sql_inference_sample is _SAMPLE_UNSET:
                estimate = self.connector.get_query_scan_estimate_bytes(
                    self.custom_sql)
                if sample_over_budget(estimate,
                                      gated=self.connector.gates_query_scan):
                    cost = ('an amount BigQuery would not estimate'
                            if estimate is None else f'~{estimate} bytes')
                    logger.warning(
                        f"Custom SQL query for table '{self.name}' would scan "
                        f"{cost} (> {SQL_SAMPLE_MAX_SCAN_BYTES}); "
                        f"skipping the stype-inference sample and falling back "
                        f"to name/dtype heuristics. Add a filter or cast "
                        f"columns to refine stypes.")
                    self._sql_inference_sample = None
                else:
                    self._sql_inference_sample = self.connector.sample_query(
                        self.custom_sql, NUM_ROWS_FOR_APPROX_STATS)
            return self._sql_inference_sample
        return self.connector.read_table_sample(self.source_name,
                                                cols=self.cols,
                                                cast_response=False)

    def _resolve_data_version(self, skip_cache: bool) -> Optional[str]:
        if self.is_sql:
            # Check this before the ``data_version`` property, which would
            # otherwise call ``connector.get_data_version(source_name)`` on the
            # synthetic alias as if it were a physical table. A Custom SQL
            # Table's version is derived from its referenced tables by the
            # materialization path; pre-ingest inference samples the query
            # directly and does not need it.
            return self._data_version
        if self.data_version is not None:
            return self.data_version
        return self.connector.get_data_version(self.source_name,
                                               skip_cache=skip_cache)

    def _build_supported_stats_request(
        self,
        cols: List[Column],
        col_stat_types: List[ColStatType],
    ) -> Dict[BaseColumnConfig, List[ColStatType]]:
        # Drop AUTOREGRESSION_COL entirely (synthetic, no underlying data to
        # sample) and keep only stat types whose dtype/stype gates pass --
        # anything failing the gates would be silently dropped by the
        # downstream cache/connector path anyway.
        from kumo_core.train.autoregression import AUTOREGRESSION_COL
        request: Dict[BaseColumnConfig, List[ColStatType]] = {}
        for col in cols:
            if col.name == AUTOREGRESSION_COL:
                continue
            supported = [
                stat_type for stat_type in col_stat_types
                if (stat_type.supports(col.dtype)
                    and stat_type.supports_stype(col.stype))
            ]
            if supported:
                request[col.base_config()] = supported
        return request

    def _compute_stats_via_connector_sample(
        self,
        request: Dict[BaseColumnConfig, List[ColStatType]],
        approx: bool,
        data_version: Optional[str],
        **kwargs,
    ) -> Dict[BaseColumnConfig, Dict[ColStatType, Stats]]:
        # Default sample-and-python path on the base ``Connector`` class.
        # Engine-specific pushdowns (e.g. Snowpark) were removed in #28324;
        # all connectors now share this implementation. For a Custom SQL Table
        # the sample comes from executing the query (``sample_query``), so the
        # SQL is threaded through instead of the synthetic source name.
        if not request:
            return {}
        if self.is_sql:
            sample = self._read_inference_sample()
            if sample is None:
                return {}
            return _compute_col_stats_from_sample(request, sample,
                                                  approx=approx,
                                                  data_version=data_version)
        return self.connector.compute_col_stats(
            self.source_name,
            request,
            approx=approx,
            num_rows=kwargs.get('num_rows', NUM_ROWS_FOR_APPROX_STATS),
            data_version=data_version,
        )

    @blocking_method()
    def persist_sql_preview_col_stats(
        self,
        data_version: str,
        request: Dict[BaseColumnConfig, List[ColStatType]],
    ) -> None:
        r"""Compute this Custom SQL Table's Validate-time approx col stats from
        the bounded query sample and persist them under ``data_version`` — a
        *definition-identity* token (see ``custom_sql_stats_version``), not a
        content version — so an unmaterialized preview reuses them across
        upstream data churn while a semantic SQL/cols edit resets them.

        Cost-gated: an over-budget warehouse scan (Option A gate) or an empty
        request leaves the preview to fall back to schema-only. Out-of-range
        timestamps may be displayed as strings but cannot always be reduced by
        pandas, so they also leave the preview schema-only."""
        assert self.is_sql
        if not request:
            return
        try:
            by_config = self._compute_stats_via_connector_sample(
                request, approx=True, data_version=data_version,
                num_rows=NUM_ROWS_FOR_APPROX_STATS)
        except pd.errors.OutOfBoundsDatetime:
            logger.warning(
                "Skipping Custom SQL preview stats for out-of-range timestamps")
            return
        get_active_cache().persist_sampled_col_stats(self.fingerprint(),
                                                     data_version, by_config)

    def _read_dg_cache_with_fallback(
        self,
        request: Dict[BaseColumnConfig, List[ColStatType]],
        approx: bool,
        data_version: str,
        **kwargs,
    ) -> Dict[BaseColumnConfig, Dict[ColStatType, Stats]]:
        data_backend_cache: BackendCacheManager = get_active_cache()
        fingerprint = self.fingerprint()
        data_backend_cache.load_col_stats_cache(fingerprint, data_version)

        col_stats = data_backend_cache.get_col_stats(fingerprint, request,
                                                     data_version,
                                                     approx=approx)
        missing = data_backend_cache.get_missing_col_stats(
            fingerprint, request, data_version, approx=approx)
        if not missing:
            return col_stats

        # Approx-eligible stats fall back to the connector's sample +
        # python compute so flows that run before the DG snapshot (e.g.
        # pre-snapshot pquery validation, get_encoder_stats approx
        # branch) still produce stats. Exact requests still require the
        # DG snapshot and must strict-raise. Mirrors the connector-sample
        # fallback in ``DataBackend.get_min_col_stats``.
        from kumo.stats.stats import APPROX_ONLY_COL_STATS
        fallback_dict: Dict[BaseColumnConfig, List[ColStatType]] = {}
        for cfg, stats in missing.items():
            fallback_stats = [
                s for s in stats if approx or s in APPROX_ONLY_COL_STATS
            ]
            if fallback_stats:
                fallback_dict[cfg] = fallback_stats

        if fallback_dict:
            try:
                fallback = self._compute_stats_via_connector_sample(
                    fallback_dict, True, data_version, **kwargs)
            except Exception as exc:
                logger.warning(
                    "Table.compute_col_stats: connector sample fallback "
                    "failed for %s@%s: %r", fingerprint.source_name,
                    data_version, exc)
                fallback = {}
            for cfg, by_stat in fallback.items():
                if not by_stat:
                    continue
                col_stats.setdefault(cfg, {}).update(by_stat)
            # Persist the sampled fallback so later reads hit the cache instead
            # of re-sampling -- mirrors the REST ``get_min_col_stats`` path so
            # the two connector-sample fallbacks stay consistent. Best-effort.
            data_backend_cache.persist_sampled_col_stats(
                fingerprint, data_version, fallback)

        # Recompute what's still missing after the connector-sample merge:
        # the error fires only when no path produced the stat. Strict
        # (exact-only) stats are never in ``fallback_dict`` so they
        # remain in ``still_missing`` here.
        still_missing: Dict[BaseColumnConfig, List[ColStatType]] = {}
        for cfg, stats in missing.items():
            got = col_stats.get(cfg, {})
            remaining = [s for s in stats if s not in got]
            if remaining:
                still_missing[cfg] = remaining

        if still_missing:
            missing_desc = ', '.join(f"{cfg.name}:{[s.name for s in stats]}"
                                     for cfg, stats in still_missing.items())
            raise UnsupportedInDiskgraphError(
                "Requested col stats missing from the DiskGraph-"
                "populated cache for "
                f"{fingerprint.source_name}@{data_version}. "
                "DiskGraph ingest populates this cache; a miss means "
                "either ingest hasn't run for this table / version, "
                "or the requested stat type is not emitted by DG. "
                f"Missing: {missing_desc}.")

        return col_stats

    def _format_by_col_name(
        self,
        cols: List[Column],
        col_stats_by_config: Dict[BaseColumnConfig, Dict[ColStatType, Stats]],
    ) -> Dict[str, Dict[ColStatType, Stats]]:
        # Rekey config -> col.name; cols whose stats were dropped (autoreg or
        # unsupported dtype/stype) round-trip as empty dicts so callers can
        # iterate the requested ``cols`` uniformly.
        out: Dict[str, Dict[ColStatType, Stats]] = {}
        for col in cols:
            out[col.name] = col_stats_by_config.get(col.base_config(), {})
        return out

    def compute_table_stats(
        self,
        table_stat_types: Optional[List[TableStatType]] = ALL_TABLE_STAT_TYPES,
    ) -> Dict[TableStatType, Stats]:
        r"""Computes requested stats for the table.

        Args:
            table_stat_types (List[TableStatType]): List of table stat types
                to compute. Compute all table stats by default
                (default: :obj:`None`)
            approx (bool, optional): Whether to compute the statistic
                approximately or exactly (default: :obj:`False`).
        Returns:
            Dict[TableStatType, Stats]: Dictionary mapping TableStatType to
            computed stats.
        """
        if table_stat_types is None:
            return {}

        # DG-native: prefer the backend cache populated by DiskGraph
        # ingest (source tables). For training/prediction tables that
        # weren't ingested through DG, fall back to computing stats
        # directly off the parquet via compute_table_stats_native.
        from kumo.diskgraph.stats import compute_table_stats_native

        data_backend_cache: BackendCacheManager = get_active_cache()
        table_stats: Dict[TableStatType,
                          Stats] = (data_backend_cache.get_table_stats(
                              self.fingerprint(),
                              stats_list=table_stat_types,
                              data_version=self.data_version,
                          ))

        missing = [s for s in table_stat_types if s not in table_stats]
        if missing:
            # Compute the missing stats from the table's parquet source
            # (training-table / pred-table case, and source-table case
            # when the cache wasn't seeded).
            parquet_path = self.connector._get_path(self.source_name)
            native_stats = compute_table_stats_native(
                parquet_path,
                data_version=self.data_version or '',
                stats_list=missing,
            )
            table_stats = {**table_stats, **native_stats}
            if native_stats:
                data_backend_cache.add_table_stats(self.fingerprint(),
                                                   native_stats)

        # TODO(manan) remove, here for backwards compatibility:
        self.stats.update(table_stats)

        return table_stats

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def get_stats(self, stat_types: List[Any]) -> List[Optional[Stats]]:
        r"""Obtains the statistics :obj:`stat_types`."""
        self.stats.update(get_active_cache().get_table_stats(
            self.fingerprint(),
            stats_list=[TableStatType(st) for st in stat_types],
            data_version=self.data_version,
        ))

        return [self.stats.get(stat_type, None) for stat_type in stat_types]

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def num_rows(self, compute_table_stats_if_missing: bool = False):
        r"""Return number of rows for the table.

        Args:
            compute_table_stats_if_missing (bool): Compute number of rows if
                the stats is missing (default: :obj:`False`)
        """
        try:
            return self.get_stats([TableStatType.NUM_ROWS])[0].value
        except Exception as e:
            if compute_table_stats_if_missing:
                return self.compute_table_stats(
                    [TableStatType.NUM_ROWS])[TableStatType.NUM_ROWS].value
            raise e

    def num_nodes(
        self,
        compute_table_stats_if_missing: bool = False,
    ):
        num_nodes = self.num_rows(compute_table_stats_if_missing)
        # Dropping na pkey/time col is no longer done as part of table
        # ingestion after https://github.com/kumo-ai/kumo/pull/20807.
        # As a result of this, num_nodes == num_rows - dropped_rows.
        pkey_col_name = self.pkey_name
        time_col_name = self.time_col_name
        for col_name, col in self._cols.items():
            if col_name == pkey_col_name or col_name == time_col_name:
                try:
                    num_nodes -= col.stats[ColStatType.DROPPED_ROWS]
                    break
                except Exception:
                    logger.warning(
                        f"Failed to get number of dropped rows from column {col.name}."
                    )
        return num_nodes

    # Visualization ###########################################################

    def print_summary(self):
        r"""Prints a summary of the table, *i.e.* the data types, encoder
        decisions, and number of :obj:`N/A` values per column."""
        if not self.has_data:
            warnings.warn(f"Table {self.name} has not been "
                          "materialized yet. As a result, the encoder "
                          "statistics shown in the summary could be "
                          "inaccurate.")
        from tabulate import tabulate

        header = ['Name', 'SType', 'DType', 'Encoder', 'N/A']

        meta = []
        if self.has_pkey():
            name = f'{self.pkey.name} (PKEY)'
            dtype = getattr(self.pkey.dtype, 'value', '???')
            meta.append([name, self.pkey.stype.value, dtype, '', '???'])

        for fkey in self.fkeys:
            name = f'{fkey.name} (FKEY)'
            dtype = getattr(fkey.dtype, 'value', '???')
            meta.append([name, fkey.stype.value, dtype, '', '???'])

        if self.has_time_col():
            name = f'{self.time_col.name} (TIME)'
            dtype = getattr(self.time_col.dtype, 'value', '???')
            meta.append([name, self.time_col.stype.value, dtype, '', '???'])

        if self.has_end_time_col():
            name = f'{self.end_time_col.name} (END_TIME)'
            dtype = getattr(self.end_time_col.dtype, 'value', '???')
            meta.append([name, self.end_time_col.stype.value, dtype, '', '???'])

        if getattr(self, 'target_col', None) is not None:
            name = f'{self.target_col.name} (TARGET)'
            stype = getattr(self.target_col.stype, 'value', '???')
            dtype = getattr(self.target_col.dtype, 'value', '???')
            encoder = str(self.target_col.encoder)
            na = self.target_col.stats.get(ColStatType.NA_FRACTION, '???')
            if not isinstance(na, str):
                na = f'{"~" if na.approx else ""}{100 * na.value:.1f}%'
            meta.append([name, stype, dtype, encoder, na])

        data = []
        for col in self.cols:
            encoder = '???' if col.encoder is None else str(col.encoder)
            stype = getattr(col.stype, 'value', '???')
            dtype = getattr(col.dtype, 'value', '???')
            na = col.stats.get(ColStatType.NA_FRACTION, '???')
            if not isinstance(na, str):
                na = f'{"~" if na.approx else ""}{100 * na.value:.1f}%'
            data.append([col.name, stype, dtype, encoder, na])

        if self.has_data:
            print(f'{self.name} (#rows={self.data.num_nodes}):')
        else:
            print(f'{self.name}:')

        # Sort the feature summary based on semantic type first, then name:
        data = sorted(data, key=lambda x: (x[1], x[0]))

        print(tabulate(meta + data, headers=header, tablefmt='psql'))
