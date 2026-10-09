# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import Any, Dict, List, Optional, TypeAlias

from kumo_core.types import ColStatType, TableStatType
from kumo_core.types.schema import (
    BaseColumnConfig,
    IngestedTableFingerprint,
    PartialTableFingerprint,
    TableFingerprint,
)
from kumo_core.utils.names import fully_qualified_table_name
from kumoapi.encoder import EncoderType
from kumoapi.typing import Dtype, Stype
from pydantic import field_validator

from io_utils import kumo_dataclass, logging
from kumo.config import StatsConfig

logger = logging.getLogger(__name__)

# Column configurations #######################################################


def base_config_as_column_config(config: BaseColumnConfig) -> 'ColumnConfig':
    return ColumnConfig(
        name=config.name,
        stype=config.stype,
        dtype=config.dtype,
        na_values=config.na_values,
        metadata=config.metadata,
        timestamp_unit=config.timestamp_unit,
        multicategorical_delimiter=config.multicategorical_delimiter,
    )


@kumo_dataclass(unsafe_hash=True)
class PrimaryKeyConfig(BaseColumnConfig):
    r"""Primary key column configuration."""
    drop_duplicates: bool = False

    @field_validator('stype')
    @classmethod
    def default_stype(cls, val: Any) -> Stype:
        if val is None:
            return Stype.ID
        return val


@kumo_dataclass
class ForeignKeyConfig(BaseColumnConfig):
    r"""Foreign key column configuration. Includes a :obj:`dst_table` name."""
    dst_table: Optional[str] = None
    dst_key: Optional[str] = None
    dst_data_source: Optional[str] = None

    # NOTE: 'Any' here is Union[EdgeStatType, ColStatType]. Since Union types
    # are not well supported as dictionary keys, we use Any:
    stats: Dict[Any, StatsConfig] = field(default_factory=dict)

    @field_validator('stype')
    @classmethod
    def default_stype(cls, val: Any) -> Stype:
        if val is None:
            return Stype.ID
        return val


@kumo_dataclass
class ColumnConfig(BaseColumnConfig):
    r"""Full column configuration: includes all additional information not
    included in :class:`BaseColumnConfig`."""
    encoder: EncoderType | None = None
    stats: Dict[ColStatType, StatsConfig] = field(default_factory=dict)
    # Deprecated argument. Do not use:
    post_transform: Optional[Any] = field(default=None, hash=False)

    def as_base_column_config(self) -> BaseColumnConfig:
        return BaseColumnConfig(
            name=self.name,
            stype=self.stype,
            dtype=self.dtype,
            na_values=self.na_values,
            pre_transform=self.pre_transform,
            metadata=self.metadata,
            timestamp_unit=self.timestamp_unit,
            multicategorical_delimiter=self.multicategorical_delimiter,
        )


@kumo_dataclass
class TimeColumnConfig(BaseColumnConfig):
    is_pseudo: bool = False

    @field_validator('stype')
    @classmethod
    def default_stype(cls, val: Any) -> Stype:
        if val is None:
            return Stype.timestamp
        return val

    @field_validator('dtype')
    @classmethod
    def default_dtype(cls, val: Any) -> Dtype:
        if val is None:
            return Dtype.date
        return val


@kumo_dataclass
class BoolColumnConfig(BaseColumnConfig):
    @field_validator('dtype')
    @classmethod
    def default_dtype(cls, val: Any) -> Dtype:
        return val or Dtype.bool


# Table configurations ########################################################


@kumo_dataclass
class TableMetadataConfig:
    r"""Metadata for a TableConfig, consisting of the table name and the
    :class:`BaseColumnConfig` of each column. ONLY USED IN TRANSIT."""
    # Table name not fully-qualified.
    table_name: str
    # Original name of the table in the customer's data source
    source_name: str
    data_source: str
    data_version: Optional[str] = None
    pkey: Optional[PrimaryKeyConfig] = None
    fkeys: List[ForeignKeyConfig] = field(default_factory=list)
    cols: List[BaseColumnConfig] = field(default_factory=list)
    time_col: Optional[TimeColumnConfig] = None
    end_time_col: Optional[TimeColumnConfig] = None

    @property
    def name(self) -> str:
        r"""Fully qualified table name, read only."""
        return fully_qualified_table_name(self.table_name, self.data_source)

    # TODO: Make data_cols a property as in kumo.Table
    def data_cols(self) -> List[BaseColumnConfig]:
        cols = self.fkeys + self.cols
        if self.time_col is not None:
            cols.append(self.time_col)
        if self.end_time_col is not None:
            cols.append(self.end_time_col)
        # Dedupe potential duplicate columns
        col_dict: Dict[str, BaseColumnConfig] = {}
        for col in cols:
            if col.name not in col_dict:
                col_dict[col.name] = BaseColumnConfig.cast(col)
            else:
                col_dict[col.name].update(col)
        return list(col_dict.values())

    def all_cols(self) -> List[BaseColumnConfig]:
        # Retrieve all columns including primary key column
        cols = self.data_cols()
        if self.pkey is not None:
            cols.append(BaseColumnConfig.cast(self.pkey))
        return cols

    def as_table_config(self, table_name: str,
                        data_source: str) -> 'TableConfig':
        return TableConfig(
            table_name=table_name,
            # `TableMetadataConfig.table_name` generally refers to the name
            # of the table within the connector, so we assign it to
            # `Table.source_name`.
            source_name=self.source_name,
            data_source=data_source,
            pkey=self.pkey,
            fkeys=self.fkeys,
            cols=[base_config_as_column_config(col) for col in self.cols],
            time_col=self.time_col,
            end_time_col=self.end_time_col,
        )

    def fingerprint(self) -> 'TableFingerprint':
        cols = dict([(c.name, c.dtype) for c in self.all_cols()])
        multicategorical_delimiters = {
            c.name: c.multicategorical_delimiter
            for c in self.all_cols() if c.multicategorical_delimiter is not None
        }
        assert all([dtype is not None for dtype in list(cols.values())])
        return IngestedTableFingerprint(
            connector_name=self.data_source,
            source_name=self.source_name,
            cols=cols,  # type: ignore
            pkey=(self.pkey.name if self.pkey is not None else None),
            time_col=self.time_col,
            end_time_col=self.end_time_col,
            multicategorical_delimiters=multicategorical_delimiters,
        )

    def partial_fingerprint(self) -> 'TableFingerprint':
        cols = [c.name for c in self.all_cols()]
        return PartialTableFingerprint(
            connector_name=self.data_source,
            source_name=self.source_name,
            cols=cols,
            pkey=(self.pkey.name if self.pkey is not None else None),
            time_col=self.time_col,
            end_time_col=self.end_time_col,
        )


@kumo_dataclass
class TableConfig:
    table_name: str
    data_source: Optional[str] = None  # Allow default data source.
    data_version: Optional[str] = None
    pkey: Optional[PrimaryKeyConfig] = None
    fkeys: List[ForeignKeyConfig] = field(default_factory=list)
    cols: List[ColumnConfig] = field(default_factory=list)
    time_col: Optional[TimeColumnConfig] = None
    end_time_col: Optional[TimeColumnConfig] = None
    source_name: Optional[str] = None
    stats: Dict[TableStatType, StatsConfig] = field(default_factory=dict)
    # Marks a Custom SQL Table. Only this boolean is carried (never the raw
    # SQL text), letting ``Table.is_sql`` survive graph (de)serialization so
    # ``Table.data_version`` skips a physical connector probe after a round-trip.
    is_sql: bool = False

    @property
    def name(self) -> str:
        r"""Fully qualified table name, read only."""
        if self.data_source is None:
            return self.table_name
        return fully_qualified_table_name(self.table_name, self.data_source)

    @property
    def just_data_source(self) -> str:
        if self.data_source is None:
            raise RuntimeError(
                f"Data source is required on table {self.table_name}")
        return self.data_source

    @property
    def just_source_name(self) -> str:
        if self.source_name is None:
            raise RuntimeError(
                f"Data source is required on table {self.table_name}")
        return self.source_name

    def all_cols(self) -> List[ColumnConfig]:
        cols = self.cols.copy()
        if self.pkey is not None:
            cols.append(ColumnConfig.cast(self.pkey))
        for fkey in self.fkeys:
            cols.append(ColumnConfig.cast(fkey))
        return cols

    def fingerprint(self) -> 'TableFingerprint':
        cols = dict([(c.name, c.dtype) for c in self.all_cols()])
        multicategorical_delimiters = {
            c.name: c.multicategorical_delimiter
            for c in self.all_cols() if c.multicategorical_delimiter is not None
        }
        assert all([dtype is not None for dtype in list(cols.values())])
        return IngestedTableFingerprint(
            connector_name=self.just_data_source,
            source_name=self.just_source_name,
            cols=cols,  # type: ignore
            pkey=(self.pkey.name if self.pkey is not None else None),
            time_col=self.time_col,
            end_time_col=self.end_time_col,
            multicategorical_delimiters=multicategorical_delimiters,
        )

    def partial_fingerprint(self) -> 'TableFingerprint':
        cols = [c.name for c in self.all_cols()]
        return PartialTableFingerprint(
            connector_name=self.just_data_source,
            source_name=self.just_source_name,
            cols=cols,
            pkey=(self.pkey.name if self.pkey is not None else None),
            time_col=self.time_col,
            end_time_col=self.end_time_col,
        )

    def as_metadata_config(self) -> TableMetadataConfig:
        return TableMetadataConfig(
            table_name=self.table_name,
            source_name=self.source_name or self.table_name,
            data_version=self.data_version,
            data_source=str(self.data_source),
            pkey=self.pkey,
            fkeys=self.fkeys,
            cols=[col.as_base_column_config() for col in self.cols],
            time_col=self.time_col,
            end_time_col=self.end_time_col,
        )

    def get_col(self, col_name: str) -> ColumnConfig | None:
        for col in self.cols:
            if col.name == col_name:
                return col
        return None


# Graph configurations ########################################################


@kumo_dataclass
class GraphConfig:
    name: Optional[str] = None
    # Any: ConnectorConfig
    data_source_dict: Dict[str, Any] = field(default_factory=dict)
    tables: List[TableConfig] = field(default_factory=list)

    def clear_stats(self) -> 'GraphConfig':
        r"""
        Clears table, column, edge stats from the :obj:`GraphConfig`.
        This method should be used proactively when stats are not needed
        in the downstream process which consumes the GraphConfig, since column
        stats can be very large for high cardinality categorical columns.
        """
        for table in self.tables:
            table.stats = {}
            for col in table.cols:
                col.stats = {}
            for fkey in table.fkeys:
                fkey.stats = {}
        return self

    def clear_encoders(self) -> 'GraphConfig':
        r"""Clears encoder configs from the :obj:`GraphConfig`."""
        for table in self.tables:
            for col in table.cols:
                col.encoder = None
        return self

    def __getitem__(self, table_name: str) -> TableConfig:
        r"""Provides access to table configs:

        .. code-block:: python

            graph_config[table_name]  # table config for table_name
        """
        for table in self.tables:
            if table.name == table_name or table.table_name == table_name:
                return table
        raise KeyError(f'Table {table_name} not in GraphConfig {self.name}')

    def __contains__(self, table_name) -> bool:
        for table in self.tables:
            if table.name == table_name or table.table_name == table_name:
                return True
        return False


TableOrViewMetadataConfig: TypeAlias = TableMetadataConfig
