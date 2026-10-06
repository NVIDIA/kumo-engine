# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import copy
import os.path as osp
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from kumo_core.encoders.config import TargetEncoderConfig
from kumo_core.types import ColStatType
from kumo_core.types.conversions import to_pandas_type
from kumoapi.encoder import Datetime, Encoder
from kumoapi.graph import ColumnKey
from kumoapi.graph import GraphDefinition as ApiGraphDefinition
from kumoapi.graph import group_column_relationships, resolve_foreign_keys
from kumoapi.source_table import UnavailableSourceTable
from kumoapi.table import Column as ApiColumn
from kumoapi.table import TableDefinition as ApiTableDefinition
from kumoapi.typing import Dtype
from kumoapi.typing import Dtype as ApiDtype
from kumoapi.typing import Stype
from kumoapi.typing import Stype as ApiStype
from pandas import DataFrame, Series

from io_utils import fsutil, logging
from io_utils.parquet_util import parquet_num_rows
from kumo.config.mixin import _KUMO_INTERNAL_TIMESTAMP_PREFIX
from kumo.connector import ParquetConnector
from kumo.connector.pandas_connector import PandasConnector
from kumo.data import get_active_cache
from kumo.data.backend_cache import BackendCacheManager
from kumo.graph.column import (
    BaseColumn,
    Column,
    ForeignKey,
    PrimaryKey,
    TimeColumn,
)
from kumo.utils.profiling import Timer
from monitoring.tracing import trace

logger = logging.getLogger(__name__)

# Encoder Utilities ###########################################################

_CATEGORY_COUNT_STATS = {
    ColStatType.CATEGORY_COUNTS,
    ColStatType.MULTI_CATEGORY_COUNTS,
}


def _category_value_dtype(dtype: Optional[Dtype]) -> Optional[Dtype]:
    if dtype is None:
        return None
    if not isinstance(dtype, Dtype):
        try:
            dtype = Dtype(dtype)
        except ValueError:
            return None
    return dtype.list_element_type() if dtype.is_list() else dtype


def _coerce_category_value(value: Any, dtype: Optional[Dtype]) -> Any:
    dtype = _category_value_dtype(dtype)
    if dtype is None or dtype.is_string():
        return value

    if value is None:
        return value
    try:
        if pd.isna(value):
            return value
    except (TypeError, ValueError):
        # Some values are not supported by pd.isna; continue coercion.
        pass

    try:
        if dtype.is_bool():
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "1"}:
                    return True
                if normalized in {"false", "0"}:
                    return False
                return value
            if isinstance(value, bool):
                return value
            if value in {0, 1}:
                return bool(value)
            return value
        if dtype.is_int():
            return int(value.strip() if isinstance(value, str) else value)
        if dtype.is_float():
            return float(value.strip() if isinstance(value, str) else value)
    except (TypeError, ValueError):
        return value

    return value


def _normalize_category_counts_for_dtype(value: Any,
                                         dtype: Optional[Dtype]) -> Any:
    if not isinstance(value, list):
        return value

    normalized = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            normalized.append(item)
            continue

        normalized_item = [
            _coerce_category_value(item[0], dtype),
            *item[1:],
        ]
        normalized.append(
            tuple(normalized_item) if isinstance(item, tuple
                                                 ) else normalized_item)
    if not all(
            isinstance(item, (list, tuple)) and len(item) >= 2
            for item in normalized):
        return normalized

    def sort_key(item: Any) -> tuple[Any, str, str]:
        category, count = item[0], item[1]
        try:
            frequency = -float(count)
        except (TypeError, ValueError):
            frequency = 0
        return frequency, type(category).__name__, repr(category)

    return sorted(normalized, key=sort_key)


def _normalize_encoder_category_stats_for_dtype(
    encoder: Optional[Encoder],
    dtype: Optional[Dtype],
) -> None:
    if encoder is None:
        return

    for stat in _CATEGORY_COUNT_STATS:
        if stat in encoder._stats:
            encoder._stats[stat] = _normalize_category_counts_for_dtype(
                encoder._stats[stat], dtype)
        elif stat.value in encoder._stats:
            encoder._stats[stat.value] = _normalize_category_counts_for_dtype(
                encoder._stats[stat.value], dtype)


def get_full_stats(
        cols: List[BaseColumn]) -> Dict[BaseColumn, List[ColStatType]]:
    return {col: col.full_stats for col in cols}


def get_full_encoder_stats(
        cols: List[BaseColumn]) -> Dict[BaseColumn, List[ColStatType]]:
    return {col: col.full_encoder_stats for col in cols}


def get_minimal_stats(
        cols: List[BaseColumn]) -> Dict[BaseColumn, List[ColStatType]]:
    return {col: col.minimal_stats for col in cols}


def fetch_and_set_encoder_stats(
    cols: List[Column],
    table: Any,
    data_version: Optional[Any] = None,
):
    r"""Fetches all required encoder statistics from the data backend cache,
    and sets these statistics on the encoders of the corresponding columns.

    Arguments:
        cols (List[Column]): the list of columns that encoders should be
            initialized on. Note that the initialization here happens
            in-place.
        table (Table): the table corresponding to the columns.
        data_version (Any, optional): the data version of the statistics that
            must be fetched.
    """
    from kumo.graph import Table
    assert isinstance(table, Table)

    # Normalize preloaded stats before deciding which stats are missing.
    for col in cols:
        if col.encoder is not None:
            _normalize_encoder_category_stats_for_dtype(col.encoder, col.dtype)

    # Step 1: Compute all required statistics and store them on the columns:
    stats_to_fetch = {
        col.base_config():
        list(col.encoder.required_stats - set(col.encoder._stats.keys()))
        for col in cols if col.encoder is not None
    }
    stats_to_fetch = {
        col_cfg: stats
        for col_cfg, stats in stats_to_fetch.items() if len(stats) > 0
    }

    if len(stats_to_fetch) == 0:
        return

    # Step 2: Fetch statistics:
    logger.warning(
        f"Fetching statistics {stats_to_fetch} for encoder initialization on "
        f"table {table.name}")

    # Col stats are pre-populated in the cache by DiskGraph ingest;
    # read directly (no compute-on-miss fallback exists).
    cache: BackendCacheManager = get_active_cache()
    cache.load_col_stats_cache(table.fingerprint(), data_version)
    all_col_stats = cache.get_col_stats(
        table.fingerprint(),
        stats_to_fetch,
        data_version=data_version,
        approx=False,
    )

    # Step 3: Set statistics for each encoder:
    col_dict = {col.name: col for col in cols}
    for col_config, col_stats in all_col_stats.items():
        col = col_dict[col_config.name]
        encoder = col.encoder
        for stat in encoder.required_stats:
            value = col_stats[stat].value
            if stat in _CATEGORY_COUNT_STATS:
                value = _normalize_category_counts_for_dtype(value, col.dtype)
            if isinstance(encoder, Datetime):
                # TODO Move to `datetime` type and drop `TIMESTAMP_PREFIX`
                # to avoid hacky string serialization mechanism.
                value = _KUMO_INTERNAL_TIMESTAMP_PREFIX + str(value)
            encoder._stats[stat] = value


def materialize_encoders(
    table: Any, ) -> Tuple[Dict[str, Encoder], Optional[TargetEncoderConfig]]:
    r"""Materializes the encoders of the columns in the provided input table.
    Note that materialization of encoders depends on statistics that are
    assumed to have been computed by this point. If statistics are not present,
    encoder materialization will fail.

    Returns a tuple of materialized encoders:
        * First element is a list of materialized column encoders.
        * Second element is a target encoder, if present. None otherwise.
    """
    from kumo.task.training_table import TrainingTable
    logger.info(f"Computing all encoder statistics for table '{table.name}'")

    has_target = False
    table_cols = table.cols
    if isinstance(table, TrainingTable) and table.target_col is not None:
        table_cols.append(table.target_col)
        has_target = True

    # In-place modify:
    copied_cols = copy.deepcopy(table_cols)  # TODO Why?
    fetch_and_set_encoder_stats(
        cols=copied_cols,
        table=table,
        data_version=table.data_version,
    )

    # Write:
    encoders: Dict[str, Encoder] = {
        col.name: col.encoder
        for col in (copied_cols[:-1] if has_target else copied_cols)
    }

    target: Optional[TargetEncoderConfig] = None
    if has_target:
        target_col = copied_cols[-1]
        target = TargetEncoderConfig(target_col.name, target_col.encoder)

    return encoders, target


# Index Map / Foreign Key Data Utilities ######################################


def read_parquet(
    file_path: str,
    cols: Optional[List[str]] = None,
    max_row_limit: Optional[int] = None,
) -> DataFrame:
    r"""This function returns a :obj:`DataFrame`. If :obj:`max_row_limit`
    is supplied and
    Args:
        file_path(str): Path to the parquet
        cols (list[str]): List of columns to read
        max_row_limit (int, optional): Max rows to read. Since we load
            the data into memory, we cannot just read an arbitrarily
            large file. An error will be raised if we are attempting to
            read a file exceed max_row_limit.
    """
    logger.info(
        f"read_parquet: {file_path} cols: {cols} row limit: {max_row_limit}")
    from pyarrow import parquet

    from io_utils.pyarrow import arrow_cloud_filesystem
    if max_row_limit is not None:
        if parquet_num_rows(file_path) > max_row_limit:
            raise RuntimeError(f"{file_path} exceeds the row limit of "
                               f"{max_row_limit}.")
    if fsutil.is_s3(file_path) or fsutil.is_gcs(file_path):
        # Route cloud reads through the credentialed, fork-safe choke point;
        # local paths keep the native read. Normalize s3a first so the strip
        # and filesystem both see s3:// (no-op for plain s3://).
        file_path = fsutil.s3a_as_s3(file_path)
        bare = file_path.removeprefix("s3://").removeprefix("gs://")
        table = parquet.read_table(bare, columns=cols,
                                   filesystem=arrow_cloud_filesystem(file_path))
    else:
        table = parquet.read_table(file_path, columns=cols)
    return table.to_pandas(types_mapper=to_pandas_type)


# TODO clean up function signature and replace 'arange' cols with
# NODE_INDEX_COLUMN


def get_index_map(
    index_map: Union[DataFrame, str, Tuple[str, Dict[str,
                                                     str]]], ) -> DataFrame:
    r"""Returns the index map for a table. The index map could be
    * a string pointer to a DataFrame path for lazy loading,
    * a tuple of string path to a Parquet dataset and a mapping of columns
    type (index or pkey) to column name to read from the Parquet files
    for lazy loading,
    * a DataFrame itself."""
    if isinstance(index_map, str):
        with Timer(logging_prefix=f"get_index_map from {index_map}"):
            return read_parquet(index_map)
    elif isinstance(index_map, Tuple):
        path, cols = index_map
        with Timer(logging_prefix=f"get_index_map from {path}"):
            tmp_df = read_parquet(path, list(cols.values()))
            if 'pkey' in cols:
                tmp_df.set_index(cols['pkey'], inplace=True)
            return tmp_df
    return index_map


def get_fkey_data(fkey_data: Union[Series, str, Tuple[str, str]]) -> DataFrame:
    r"""Returns the data for a foreign key. The data could be
    * a string pointer to a DataFrame path for lazy loading,
    * a tuple of string path to a Parquet dataset and the fkey column
    name to read from the Parquet files for lazy loading,
    * a DataFrame itself."""
    if isinstance(fkey_data, str):
        with Timer(logging_prefix=f"get_fkey_data from {fkey_data}"):
            tmp_df = read_parquet(fkey_data)
        return tmp_df['key']
    elif isinstance(fkey_data, Tuple):
        path, col = fkey_data
        with Timer(logging_prefix=f"get_fkey_data from {path}"):
            tmp_df = read_parquet(path, [col])
        return tmp_df[col]
    return fkey_data


# Miscellaneous ###############################################################


def check_for_exception(futures: List[Future]):
    """Tasks in :obj:`ThreadPoolExecutor` fail silently. This function will
    raise the same exception if a call in :obj:`futures` raises an exception"""
    for future in concurrent.futures.as_completed(futures):
        future.result()


def persist_pandas_connector(
    connector: PandasConnector,
    root_dir: str,
) -> ParquetConnector:
    r"""Persists all the Pandas Dataframe in the PandasConnector as parquet
    files and returns a ParquetConnector pointing to those files. The parquet
    files will be created at
    'root_dir/{connector.name}_converted/{table_name}.parquet'"""
    new_data_source = f"{connector.name}_converted"
    parquet_dir = osp.join(root_dir, new_data_source)
    new_connector = ParquetConnector(name=new_data_source, root_dir=parquet_dir)
    logger.info(f"Converting PandasConnector({connector}) to "
                f"ParquetConnector({new_connector})")

    fsutil.makedirs(parquet_dir, exist_ok=True)
    for table_name in connector.df_dict.keys():
        # Get the Pandas dataframe and write it out as a parquet file
        df = connector.df_dict[table_name]
        output_file = osp.join(parquet_dir, f'{table_name}.parquet')
        df.to_parquet(output_file)
    # Return the ParquetConnector pointing to the root_dir
    return new_connector


@trace()
def make_graph_mock_object(graph_def: ApiGraphDefinition) -> Any:
    r'''Creates a kumo.Graph object but does not store it to a database nor
    does it perform any operations that are data-dependant, e.g. encoder
    inferral.'''
    from kumo.graph.graph import Graph

    fkey_dict = {table_name: [] for table_name in graph_def.tables}
    for edge in resolve_foreign_keys(graph_def.tables, graph_def.col_groups):
        source = graph_def.tables[edge.src_table]
        destination = graph_def.tables[edge.dst_table]
        source_column = next(column for column in source.cols
                             if column.name == edge.fkey)
        assert destination.pkey is not None
        fkey_dict[edge.src_table].append(
            ForeignKey(
                name=edge.fkey,
                dst_table=edge.dst_table,
                dst_key=destination.pkey,
                dtype=Dtype(source_column.dtype.value),
            ))
    graph = Graph(data_source_dict={'dummy_connector': None})
    for table_name, table_def in graph_def.tables.items():
        pkey = None
        if table_def.pkey is not None:
            dtype = None
            for col in table_def.cols:
                if col.name == table_def.pkey:
                    dtype = Dtype(col.dtype.value)
            pkey = PrimaryKey(
                table_def.pkey,
                dtype=dtype,
            )
        time_col = None
        if table_def.time_col is not None:
            time_col = TimeColumn(table_def.time_col, dtype=Dtype.time)
        end_time_col = None
        if table_def.end_time_col is not None:
            end_time_col = TimeColumn(table_def.end_time_col, dtype=Dtype.time)
        cols = []
        for col in table_def.cols:
            if col.name in fkey_dict[table_name]:
                continue
            cols.append(
                Column(
                    name=col.name, dtype=Dtype(col.dtype.value),
                    stype=Stype(col.stype.value), multicategorical_delimiter=(
                        col.multicategorical_delimiter)))
        graph.add(
            name=table_name,
            data_source='dummy_connector',
            pkey=pkey,
            fkeys=fkey_dict[table_name],
            cols=cols,
            time_col=time_col,
            end_time_col=end_time_col,
            ignore_errors=True,
        )
    return graph


def map_to_api_table_definition(table: Any) -> ApiTableDefinition:
    # Since this method is currently used for pquery validation only, we skip
    # source porting and timestamp_format
    source_table = UnavailableSourceTable(table=table.table_name)
    columns = []
    all_cols = [col for col in table]
    if table.end_time_col is not None:
        all_cols.append(table.end_time_col)
    for col in all_cols:
        assert col.stype is not None
        assert col.dtype is not None
        columns.append(
            ApiColumn(
                name=col.name,
                dtype=ApiDtype(col.dtype.value),
                stype=ApiStype(col.stype.value),
                timestamp_format=None,
                multicategorical_delimiter=(col.multicategorical_delimiter),
            ))
    return ApiTableDefinition(
        cols=columns,
        source_table=source_table,
        pkey=table.pkey_name,
        time_col=table.time_col_name,
        end_time_col=table.end_time_col_name,
    )


def map_to_api_graph_definition(graph: Any) -> ApiGraphDefinition:
    tables = {
        table.table_name: map_to_api_table_definition(table)
        for table in graph
    }
    relationships = []
    for table in graph:
        for fkey in table.fkeys:
            # fkey.dst_table_name is not set in some tests
            dst_table = fkey.dst_table_name
            dst_pkey = fkey.dst_key
            if dst_pkey is None:
                dst_pkey = graph[dst_table].pkey_name
            assert dst_pkey is not None
            relationships.append((
                ColumnKey(table.table_name, fkey.name),
                ColumnKey(dst_table, dst_pkey),
            ))
    return ApiGraphDefinition(
        tables=tables,
        col_groups=group_column_relationships(relationships),
    )
