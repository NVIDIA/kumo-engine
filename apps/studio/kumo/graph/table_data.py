# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from dataclasses import field
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from kumo_core.types.schema import BaseColumnConfig
from kumo_core.utils.names import get_table_name, is_fully_qualified_name
from kumoapi.train import TrainingStage

import kumo.graph
import kumo.utils.mapping
from io_utils import (
    AllowArbitraryTypesConfig,
    fsutil,
    kumo_dataclass,
    logging,
    pydantic_dataclass,
)
from io_utils.json import load_json, save_json
from kumo.config import (
    ConfigMixin,
    TableOrViewMetadataConfig,
    TrainingTableConfig,
)
from kumo.config.task import TimePeriod
from kumo.connector import DEFAULT_DATA_VERSION
from kumo.connector.pandas_connector import PandasConnector
from kumo.connector.parquet_connector import ParquetConnector
from kumo.data.generic.generate_current_pred_table import TARGET_FILE_NAME
from kumo.graph import ForeignKey, PrimaryKey, Table
from kumo.graph.column import BaseColumn, BoolColumn, Column, TimeColumn
from kumo.task.training_table import TrainingTable
from kumo.utils import to_dataclass

logger = logging.getLogger(__name__)


class TableData:
    r"""Defines a table data object containing information associated with a
    materialized table.

    ``TableData`` is an in-memory shell around the training or prediction
    table; the raw data is persisted in the machine's RAM and lazily loaded
    on first access (see the ``connector``-driven property loaders below).
    Graph tables are served out of the on-VM diskgraph index.

    Note that feature data in the table is *lazily encoded*.

    Args:
        table (Table): the Table object that produces this materialized
            TableData object.
        data_version (str, optional): The version number of the data that is
            used for materialization of this table. (default: :obj:`"-1"`)
    """
    def __init__(
        self,
        table: Table,
        data_version: str = DEFAULT_DATA_VERSION,
    ):
        assert not isinstance(table.connector, PandasConnector)

        # TODO(manan): assert that the provided encoders are compatible with
        # the columns of the table

        # Set basic properties:
        self.name = table.name
        self.time_col = table.time_col
        self.end_time_col = table.end_time_col
        self.lp_is_array = getattr(table, 'lp_is_array', False)
        self.data_version = data_version
        self.table_metadata_config = table.metadata_config()

        # If this table is a training table with a temporal split, this
        # contains the time ranges that were used to generate the split:
        self.time_periods: Dict[TrainingStage, TimePeriod] = {}

        # In-memory attributes for lazy loading of training/prediction data:
        self.in_memory_attrs = InMemoryDataAttributes()

        # All fkeys:
        for fkey in table.fkeys:
            assert fkey.dst_table is not None
            if not is_fully_qualified_name(fkey.dst_table):
                raise ValueError(
                    f"The destination table of the foreign key {fkey} is not "
                    f"fully qualified (got {fkey.dst_table}).")

        logger.info(f"Materializing table {self.name} (in-memory).")

        # TODO(manan): this is currently required, since we `read_parquet`
        # to get foreign key data from the pointed-to table. Ensure that
        # this is standardized throughout.
        assert isinstance(table.connector, ParquetConnector)
        self.persist_table(table)

        logger.info(f"Completed materializing table {self.name}.")

    def persist_table(self, table: Table):
        r"""Initializes the TableData object with attributes derived from the
        raw data of `table`. Currently only invoked for the training or
        prediction tables.

        Args:
            table (kumo.Table): the table object to persist in-memory.
        """
        # Compute missing table stats when persist (mainly for training table).
        self.num_nodes = table.num_nodes(compute_table_stats_if_missing=True)

        fkeys = table.fkeys
        self._fkey_data: Dict[ForeignKey, Any] = {fkey: [] for fkey in fkeys}
        self._index_map_path = None

        # Store in-memory data:
        self.in_memory_attrs.connector = table.connector

        self.in_memory_attrs.target_col = getattr(table, 'target_col', None)
        self.in_memory_attrs.time_col = getattr(table, 'time_col', None)
        self.in_memory_attrs.weight_col = None
        # Only set weight_col for training tables, not for prediction tables
        is_prediction = isinstance(table, TrainingTable) and table.is_pred
        if not is_prediction:
            self.in_memory_attrs.weight_col = getattr(table, 'weight_col', None)

        self.in_memory_attrs.entity_include_bit_col = getattr(
            table, 'entity_include_bit_col', None)
        self.in_memory_attrs.is_pred = (isinstance(table, TrainingTable)
                                        and table.is_pred)

        path = table.connector._get_path(table.source_name)  # type: ignore
        for fkey in self._fkey_data:
            self._fkey_data[fkey] = (path, fkey.name)

    # TableData Properties ####################################################

    @property
    def table_name(self) -> str:
        r"""Returns the non-fully qualified name of the table."""
        return get_table_name(self.name)

    @property
    def persist_dir(self) -> Optional[str]:
        r"""Always ``None``: graph-table features live in the on-VM
        diskgraph index, not in a ``TableData``-owned directory. Retained
        for duck-typing with callers such as :class:`GraphData`."""
        return None

    @property
    def is_pred(self) -> Optional[bool]:
        r"""If the table is a prediction table."""
        return self.in_memory_attrs.is_pred

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  name={self.name},\n'
                f'  data_version={self.data_version},\n'
                f'  persist_dir=MEMORY,\n'
                f'  num_nodes={self.num_nodes}\n'
                f')')

    # Lazily Loaded In-Memory Properties ######################################
    #
    # These properties are attributes of tables that are stored in-memory, as
    # opposed to behind RocksDB as SST files. Currently, we only store the
    # training and prediction tables in memory, since these are treated as
    # external to the graph (and can be swapped in/out at will). In order to
    # avoid reading the data of these tables into memory in the REST driver
    # node, we lazily load all data-related properties to be read on first
    # access, which SHOULD ALWAYS OCCUR in the GPU node (e.g. in the trainer
    # or afterwards). Please be cautious when working with these properties,
    # to avoid memory issues downstream.

    @property
    def data(self) -> pd.DataFrame:
        r"""The raw table data if the table is persisted in-memory. Loaded
        lazily (at first access)."""
        if hasattr(self, '_data'):
            return self._data  # pylint: disable-all

        logger.info(
            f"Reading raw table data for table {self.name} into memory. "
            f"Please ensure this is intended.")

        assert self.in_memory_attrs.connector is not None
        assert self.table_metadata_config is not None

        # NOTE weight col is not currently stored in the table metadata
        # config, so we need to add it separately. Note that target col is
        # already part of data cols:
        cols: List[BaseColumnConfig] = self.table_metadata_config.data_cols()
        if self.in_memory_attrs.weight_col is not None:
            cols.append(self.in_memory_attrs.weight_col.base_config())
        if self.in_memory_attrs.target_col is not None:
            assert self.in_memory_attrs.target_col.name in [
                c.name for c in cols
            ]
        # NOTE: `entity_include_bit_col` is not stored in the
        # table metadata config, so we need to add it as well.
        if (self.in_memory_attrs.entity_include_bit_col is not None
                and not self.is_pred):
            cols.append(self.in_memory_attrs.entity_include_bit_col.config())

        targets_df: pd.DataFrame | None = None
        if getattr(self, 'lp_is_array', False):
            target_col_name = self.table_metadata_config.fkeys[-1].name
            logger.info(
                f"Dropping target column name '{target_col_name}' "
                f"from {[c.name for c in cols]} since the column is not in "
                f"the prediction table with `ALLOW_ARRAY_TARGETS=1`.")
            cols = [c for c in cols if c.name != target_col_name]
            targets_df = self.in_memory_attrs.connector.read_table(
                TARGET_FILE_NAME,
                cols=[target_col_name],
            )

        self._data: pd.DataFrame = self.in_memory_attrs.connector.read_table(
            self.table_name,
            pkey=self.table_metadata_config.pkey,
            cols=cols,
        )

        if getattr(self, 'lp_is_array', False):
            assert targets_df is not None
            max_rows = max(len(self._data), len(targets_df))
            if max_rows > len(self._data):
                first_row = self._data.iloc[0]
                self._data = self._data.reindex(
                    range(max_rows),
                    method=None,
                ).fillna(first_row)
            elif max_rows > len(targets_df):
                first_row = targets_df.iloc[0]
                targets_df = targets_df.reindex(
                    range(max_rows),
                    method=None,
                ).fillna(first_row)

            assert len(self._data) == len(targets_df)
            self._data = pd.concat([self._data, targets_df], axis=1)

        return self._data

    @property
    def weight(self) -> Optional[torch.Tensor]:
        r"""The weight column of the training or prediction table, if it
        exists. Loaded lazily (at first access)."""
        if hasattr(self, '_weight'):
            return self._weight  # pylint: disable-all

        self._weight: Optional[torch.Tensor] = None
        weight_col = self.in_memory_attrs.weight_col  # type: ignore
        if weight_col is not None:
            if hasattr(self, '_data'):
                weight = self.data[weight_col.name]
            else:
                # Only read the weight column:
                assert self.in_memory_attrs.connector is not None
                weight = self.in_memory_attrs.connector.read_table(
                    self.table_name,
                    cols=[weight_col.base_config()])[weight_col.name]
            if getattr(self, 'count', None) is not None:
                self._weight = torch.from_numpy(
                    weight.explode().astype(float).values)
            else:
                self._weight = torch.from_numpy(weight.values)
        return self._weight

    @property
    def target(self) -> Optional[torch.Tensor]:
        r"""The target column of the training, if it exists.
        Loaded lazily (at first access)."""
        if hasattr(self, '_target'):
            return self._target  # pylint: disable-all

        self._target: Optional[torch.Tensor] = None
        target_col = self.in_memory_attrs.target_col  # type: ignore
        if target_col is not None:
            if hasattr(self, '_data'):
                target = self.data[target_col.name]
            else:
                # Only read the target column:
                assert self.in_memory_attrs.connector is not None
                target = self.in_memory_attrs.connector.read_table(
                    self.table_name,
                    cols=[target_col.base_config()])[target_col.name]
            target = target.values
            if isinstance(target, (
                    pd.arrays.BooleanArray,
                    pd.arrays.IntegerArray,
            )):
                target = target._data
            target = torch.from_numpy(target)
            if target.is_floating_point():  # Cast to `torch.float32`:
                target = target.to(torch.get_default_dtype())
            self._target = target
        return self._target

    @property
    def time(self) -> Optional[torch.Tensor]:
        r"""The time column of a training/prediction table, if it exists.
        Loaded lazily (at first access).
        """
        if hasattr(self, '_time'):
            return self._time  # pylint: disable-all
        self._time: Optional[torch.Tensor] = None
        time_col = self.in_memory_attrs.time_col
        if time_col is not None:
            if hasattr(self, '_data'):
                time = self.data[time_col.name]
            else:
                # Only read the time column:
                assert self.in_memory_attrs.connector is not None
                time = self.in_memory_attrs.connector.read_table(
                    self.table_name,
                    cols=[time_col.base_config()])[time_col.name]
            if time_col.is_pseudo:
                # pseudo time is guaranteed to have no NA timestamp.
                # Force ns resolution before int64: under pandas >= 2
                # ``astype(int)`` preserves the source datetime resolution
                # (a ``datetime64[s]`` column would yield SECONDS, ~1e9x off),
                # whereas Kumo treats every timestamp tensor as nanoseconds.
                time = torch.from_numpy(
                    time.astype('datetime64[ns]').astype('int64').values)
            else:
                # Normalize missing timestamps to the sentinel expected by
                # downstream tensor consumers.
                mask = torch.from_numpy(time.isna().values)
                # Resolution-stable ns conversion (see pseudo branch above).
                time = torch.from_numpy(
                    time.astype('datetime64[ns]').astype('int64').values)
                time[mask] = torch.iinfo(time.dtype).max
            self._time = time
        return self._time

    @property
    def entity_include_bit(self) -> Optional[torch.Tensor]:
        r"""The entity include bit of the train table. Loaded lazily
        (at first access)."""
        if hasattr(self, '_entity_include_bit'):
            return self._entity_include_bit  # pylint: disable-all
        if self.in_memory_attrs.entity_include_bit_col is None:
            return None
        entity_include_bit_col = self.in_memory_attrs.entity_include_bit_col
        if entity_include_bit_col is not None:
            if hasattr(self, '_data'):
                entity_include_bit = self.data[entity_include_bit_col.name]
            else:
                # Only read the time column:
                assert self.in_memory_attrs.connector is not None
                entity_include_bit = self.in_memory_attrs.connector.read_table(
                    self.table_name,
                    cols=[entity_include_bit_col.base_config()
                          ])[entity_include_bit_col.name]
            self._entity_include_bit = entity_include_bit
        return self._entity_include_bit

    @property
    def index_map(self) -> pd.DataFrame:
        r"""The index map of the table. Loaded lazily (at first access). Note
        that users should be wary of loading and storing an index map of a
        large table, as this may result in excessive memory pressure."""
        if hasattr(self, '_index_map'):
            return self._index_map  # pylint: disable-all

        # TODO(manan): we don't really need to read the whole dataframe
        # here; we can instead just read the pkey column from the parquet
        # file. We can make this optimization later.
        self._index_map = kumo.utils.mapping.get_index_map(self.data, 0)
        return self._index_map

    # Persistence #############################################################

    def save_attributes(self, path: str):
        r"""Saves the attributes of the TableData object to disk."""
        attributes = TableDataAttributes(
            name=self.name,
            num_nodes=self.num_nodes,
            data_version=self.data_version,
            in_memory_attrs=self.in_memory_attrs.config(
                data_cls=InMemoryDataAttributesConfig),
            table_metadata_config=self.table_metadata_config,
            time_periods=self.time_periods,
        )
        save_json(attributes, path)

    def load_attributes(self, path: str):
        r"""Loads the attributes of the TableData object from disk."""
        attributes = load_json(path, TableDataAttributes)
        self.name = attributes.name
        self.num_nodes = attributes.num_nodes
        self.data_version = attributes.data_version
        self.time_periods = attributes.time_periods
        self.table_metadata_config = attributes.table_metadata_config
        self.time_col = self.table_metadata_config.time_col
        self.end_time_col = self.table_metadata_config.end_time_col

        # NOTE this is necessary in case we are loading from data platform
        # materialization, in which case we already have (empty) in-memory
        # attributes:
        if not hasattr(self, 'in_memory_attrs'):
            self.in_memory_attrs = InMemoryDataAttributes.from_config(
                attributes.in_memory_attrs)

    def save_data(self, root_dir: str):
        r"""Persists the data of a `TableData` object to disk.

        Args:
            root_dir (str): The root directory to save the data to.
        """
        fsutil.makedirs(root_dir, exist_ok=True)
        logger.info(f"TableData '{self.name}': saving data to {root_dir}")

        # Basic class attributes:
        self.save_attributes(osp.join(root_dir, 'attributes.json'))

        # Training/prediction table only ######################################

        if hasattr(self, 'fkey'):
            path = osp.join(root_dir, 'fkey.pt')
            fsutil.torch_save(self.fkey, path, pickle_protocol=2)

        if hasattr(self, 'train_mask'):
            path = osp.join(root_dir, 'train_mask.pt')
            fsutil.torch_save(self.train_mask, path, pickle_protocol=2)

        if hasattr(self, 'val_mask'):
            path = osp.join(root_dir, 'val_mask.pt')
            fsutil.torch_save(self.val_mask, path, pickle_protocol=2)

        if hasattr(self, 'test_mask'):
            path = osp.join(root_dir, 'test_mask.pt')
            fsutil.torch_save(self.test_mask, path, pickle_protocol=2)

    def load_data(
        self,
        root_dir: str,
        # TODO(manan): store pkey, fkeys and drop this field
        pkey: Optional[PrimaryKey] = None,
        fkeys: Optional[List[ForeignKey]] = None,
    ) -> 'TableData':
        """Load the data from disk.

        Args:
            root_dir (str): The root directory to load the data from.
            pkey (PrimaryKey, optional): Unused; retained for API
                compatibility with existing callers.
            fkeys (List[ForeignKey], optional): The table's foreign keys,
                used to populate ``_fkey_data`` for lazy loading.
        """
        del pkey  # Unused; retained for API compat.
        logger.info(f"Loading TableData object from directory {root_dir}")

        # Basic class attributes:
        self.load_attributes(osp.join(root_dir, 'attributes.json'))

        self._fkey_data = {}
        assert self.in_memory_attrs.connector is not None

        # Store foreign key path:
        path = self.in_memory_attrs.connector._get_path(self.table_name)
        for fkey in (fkeys or {}):
            self._fkey_data[fkey] = (path, fkey.name)

        # Training/prediction table only ######################################

        path = osp.join(root_dir, 'fkey.pt')
        if fsutil.exists(path):
            self.fkey = fsutil.torch_load(path)

        path = osp.join(root_dir, 'train_mask.pt')
        if fsutil.exists(path):
            self.train_mask = fsutil.torch_load(path)

        path = osp.join(root_dir, 'val_mask.pt')
        if fsutil.exists(path):
            self.val_mask = fsutil.torch_load(path)

        path = osp.join(root_dir, 'test_mask.pt')
        if fsutil.exists(path):
            self.test_mask = fsutil.torch_load(path)

        return self

    # TEMPORARY patches to align wtih `TableData` #############################

    def __setitem__(self, k: str, v: Any):
        setattr(self, k, v)

    def __getitem__(self, k: str):
        return getattr(self, k, None)

    def __contains__(self, k):
        return hasattr(self, k)


# Configuration Management ####################################################


@pydantic_dataclass(config=AllowArbitraryTypesConfig)  # type: ignore
class InMemoryDataAttributes(ConfigMixin):
    r"""Dataclass for storing attributes needed to load a table with
    data stored in-memory. This is currently necessary for the storage of the
    training and prediction tables, which are not stored behind the feature
    and graph store."""
    connector: Optional[Any] = None
    time_col: Optional[TimeColumn] = None  # type: ignore
    weight_col: Optional[BaseColumn] = None  # type: ignore
    target_col: Optional[Column] = None  # type: ignore
    entity_include_bit_col: Optional[BoolColumn] = None  # type: ignore
    is_pred: Optional[bool] = None


# InMemoryDataAttributes ser/deser configuration:
InMemoryDataAttributesConfig = to_dataclass(
    InMemoryDataAttributes,
    with_target=True,
)


def build_in_memory_attrs(
    table: Any,
    connector: Any,
) -> Optional[InMemoryDataAttributesConfig]:
    """Build an InMemoryDataAttributesConfig for tables stored in memory.

    Args:
        table: Table-like object that may define in-memory attributes such as
            time_col, weight_col, target_col, entity_include_bit_col, or
            is_pred.
        connector: Connector used to access the in-memory table data.

    Returns:
        An InMemoryDataAttributesConfig when the table exposes any supported
        in-memory attributes; otherwise, None.
    """
    if not isinstance(table, TrainingTableConfig):
        return None
    attrs = InMemoryDataAttributes(
        connector=connector,
        time_col=table.time_col,
        weight_col=None,
        target_col=table.target_col,
        entity_include_bit_col=table.entity_include_bit_col,
        is_pred=table.is_pred,
    )
    if not table.is_pred:
        attrs.weight_col = table.weight_col
    return attrs.config(data_cls=InMemoryDataAttributesConfig)


@kumo_dataclass
class TableDataAttributes:
    r"""Type-safe dataclass for serializing TableData attributes to JSON."""
    name: str = ""
    num_nodes: int = 0
    data_version: str = DEFAULT_DATA_VERSION
    in_memory_attrs: Optional[
        InMemoryDataAttributesConfig] = None  # pyright: ignore
    table_metadata_config: Optional[TableOrViewMetadataConfig] = None
    time_periods: Dict[TrainingStage, TimePeriod] = field(default_factory=dict)
