# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import re
import time
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from kumo_core.sources.arrow_parquet import read_bounded_dataset
from kumo_core.sources.object_store import iter_column_dtypes
from kumo_core.types.conversions import dtype_to_pandas, to_pandas_type
from kumo_core.utils.registry import register
from pyarrow.lib import ArrowInvalid

from common_utils.error import PublicException
from io_utils import fsutil, logging, parquet_util
from io_utils.retry import retry
from kumo.config import BaseColumnConfig, ConnectorConfig, TableMetadataConfig
from kumo.connector.blocking import TraceOptions, blocking_method, trace
from kumo.connector.connector import ConnectorError
from kumo.connector.file_connector import FileConnector, _list_dataset_files
from kumo.connector.parquet_utils import (
    _MAX_READ_SAMPLE_TIME,
    S3_READ_TIMEOUT,
    clean_parquet_table,
    to_pandas_ns_safe,
    warn_on_large_parquet_partition,
)
from kumo.connector.utils import cast_df
from kumo.data.common import DATA_BACKEND_THREAD_POOL
from kumo.utils import to_dataclass

# Protect against large number of very small files.
BATCH_READ_SIZE = 4 * 1024 * 1024 * 1024  # 2GiB

logger = logging.getLogger(__name__)


class ParquetConnector(FileConnector):
    r"""Establishes a connection to a set of :obj:`*.parquet` files, either
    on-disk or remote.

    Args:
        name (str): The name of the connector.
        root_dir (str): Root directory where :obj:`*.parquet` files are stored.
        suffix (str, optional): Suffix of the files.
            (default: :obj:`"parquet"`)
    """
    def __init__(
        self,
        name: str,
        root_dir: str,
        suffix: str = 'parquet',
        **kwargs,
    ):
        super().__init__(name, root_dir, suffix=suffix, **kwargs)

    def _get_connector_type(self) -> str:
        return "parquet"

    def _supports_nested_dataset(self, path: str) -> bool:
        return bool(_list_dataset_files(path, (self.suffix, )))

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def _load(
        self,
        path: Union[str, StringIO],
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
        num_rows: Optional[int] = None,
        offset: int = 0,
        frac: Optional[float] = None,
        sample: Optional[bool] = False,
        **kwargs,
    ) -> pd.DataFrame:
        if offset > 0:
            # TODO (matthias) `pyarrow.Parquet` does not seem to support
            # `offset` option at the moment.
            raise NotImplementedError(
                f"'{self.__class__.__name__}' does not support the 'offset' "
                f"argument yet (got '{offset}')")

        columns = None if pkey is None and cols is None else []
        col_types: Dict[str, Any] = {}
        if pkey is not None:
            pkey: BaseColumnConfig = BaseColumnConfig.cast(pkey)
            columns.append(pkey.name)
            if pkey.dtype is not None:
                col_types[pkey.name] = dtype_to_pandas(pkey.dtype)

        if cols is not None:
            for col in cols:
                col: BaseColumnConfig = BaseColumnConfig.cast(col)
                columns.append(col.name)
                if col.dtype is not None:
                    col_types[col.name] = dtype_to_pandas(col.dtype)

        if fsutil.isdir(path):
            df = self.load_dataset(path, self.suffix, cols=columns,
                                   num_rows=num_rows, frac=frac, sample=sample,
                                   col_types=col_types)
        else:
            df = self.load_file(path, cols=columns, num_rows=num_rows,
                                frac=frac, sample=sample, col_types=col_types)

        # Cast the dataframe columns to the correct type.
        if len(df) > 0:
            df = cast_df(df, pkey, cols)

        if pkey is not None and pkey.name != df.index.name and len(df) > 0:
            df.set_index(pkey.name, inplace=True)

        # drop any parquet indexes that got read as columns
        drop_cols = [
            col for col in df.columns if re.match(r'__index_level_\d+__', col)
        ]
        df = df.drop(labels=drop_cols, axis=1)
        return df

    @trace(TraceOptions(trace_params=True))
    def _get_table_cols_metadata(
        self,
        table_name: str,
        skip_cache: bool = False,
        skip_inferral: bool = False,
    ) -> TableMetadataConfig:

        metadata = TableMetadataConfig(table_name=table_name,
                                       source_name=table_name,
                                       data_source=self.name)

        reader, source_name = self._source_reader(table_name)
        metadata.cols.extend(
            BaseColumnConfig(name, dtype=dtype)
            for name, dtype in iter_column_dtypes(
                reader.dataset(source_name, validate=False).schema))
        return metadata

    @trace(TraceOptions(trace_params=True))
    def _get_table_num_rows(self, table_name: str,
                            data_version: str | None = None) -> int:

        table_path = self._get_path(table_name, data_version)
        if not fsutil.exists(table_path):
            raise FileNotFoundError(f"File does not exist: {table_path}")

        return parquet_util.parquet_num_rows(table_path, skip_check=True)

    @trace(TraceOptions(trace_params=True))
    def _get_single_file_source_size(self, table_name: str,
                                     table_path: str) -> int:
        if fsutil.exists(table_path):
            return fsutil.filesize(table_path) or 0

        raise FileNotFoundError(
            f"Obtaining the source storage size for table '{table_name}' "
            f"is not implemented for the remote directory "
            f"{osp.join(self.root_dir, table_name)} or file does not exist")

    @trace(TraceOptions(trace_params=True))
    def _get_directory_source_size(self, table_name: str,
                                   table_path: str) -> int:
        total_size = 0
        if fsutil.exists(table_path):
            file_name_list = _list_dataset_files(table_path, (self.suffix, ))
            for file_name in file_name_list:
                total_size += fsutil.filesize(file_name) or 0
            return total_size

        raise FileNotFoundError(
            f"Obtaining the source storage size for table '{table_name}' "
            f"is not implemented for the remote directory "
            f"{osp.join(self.root_dir, table_name)} or file does not exist")

    @trace(TraceOptions(trace_params=True))
    def _get_table_source_size(self, table_name: str,
                               data_version: str | None = None) -> int:
        table_path = self._get_path(table_name, data_version)
        if fsutil.isdir(table_path):
            return self._get_directory_source_size(table_name, table_path)
        return self._get_single_file_source_size(table_name, table_path)

    def _create_iterator(
        self,
        path: str,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> Iterator[pd.DataFrame]:
        if fsutil.isdir(path):
            file_paths = _list_dataset_files(path, (self.suffix, ))
            df = pd.DataFrame()
            for file_path in file_paths:
                df = pd.concat(
                    [df, self._load(file_path, pkey=pkey, cols=cols)])
                if df.memory_usage(deep=True).sum() > BATCH_READ_SIZE:
                    yield df
                    df = pd.DataFrame()
            if len(df) > 0:
                yield df
        else:
            yield self._load(path, pkey=pkey, cols=cols)

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def _write(self, path: str, df: pd.DataFrame):
        df.to_parquet(path)

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    @retry(RuntimeError, tries=9, delay=1, backoff=2, logger=logger)
    def _parquet_dataset(self, path: str,
                         use_prebuffer: bool = False) -> ds.Dataset:
        from kumo.connector.object_store_access import source_reader

        try:
            reader = source_reader(path, file_format='parquet',
                                   file_suffix=self.suffix)
            return reader.dataset('', validate=False, pre_buffer=use_prebuffer)
        except (FileNotFoundError, ArrowInvalid) as e:
            raise ConnectorError(
                message=f"The uploaded file with path '{path}'"
                " is either missing or is not a valid parquet file.",
                error=e,
            )
        except PublicException:
            raise
        except Exception as e:
            raise ConnectorError(
                message="There was a problem uploading the file "
                f"with path '{path}'.",
                error=e,
            )

    # Utility functions ################################################
    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def _load_table(
        self,
        path: str,
        cols: Optional[List[str]] = None,
        num_rows: Optional[int] = None,
        sample: Optional[bool] = False,
    ) -> pa.Table:
        # Special handling for backends PyArrow can't read natively:
        # UCV paths are downloaded to a temp directory first.
        if fsutil.is_ucv(path):
            with fsutil.TemporaryDirectory() as temp_dir:
                fsutil.copy_file(path, temp_dir)
                parquet_file: pq.ParquetDataset = pq.ParquetDataset(temp_dir)
                if num_rows is not None:
                    table = read_bounded_dataset(parquet_file, cols, num_rows)
                else:
                    table = parquet_file.read(columns=cols)
                return table

        parquet_file = self._parquet_dataset(
            path, use_prebuffer=bool(sample and num_rows is None))

        logger.info(f"[load_file] path: {path}, num_rows: {num_rows}")

        if num_rows is not None:
            table = read_bounded_dataset(parquet_file, cols, num_rows)
        else:
            table = parquet_file.to_table(columns=cols)

        return table

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def load_file(
        self,
        path: str,
        cols: Optional[List[str]] = None,
        frac: Optional[float] = None,
        num_rows: Optional[int] = None,
        sample: Optional[bool] = False,
        col_types: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        # Skip directory path for loading single parquet path.
        if fsutil.isdir(path) or fsutil.filesize(path) == 0:
            return pd.DataFrame()

        table = self._load_table(path, cols, num_rows, sample)
        clean_table = clean_parquet_table(table, col_types=col_types,
                                          sample=sample)
        # :obj:`to_pandas` sometimes silently drops columns if pandas
        # metadata is not ignored.
        # Setting ``ignore_metadata=False`` can reproduce the issue.
        # :obj:`to_pandas_ns_safe` keeps datetime columns at nanosecond
        # resolution (Kumo's ``datetime64[ns]`` contract) while clamping any
        # out-of-range temporal values instead of silently wrapping them.
        df = to_pandas_ns_safe(clean_table, types_mapper=to_pandas_type,
                               ignore_metadata=True)

        return df

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    @retry(tries=9, delay=1, backoff=2, logger=logger)
    def load_dataset(
        self,
        path: str,
        suffix: str,
        cols: Optional[List[str]] = None,
        frac: Optional[float] = None,
        num_rows: Optional[int] = None,
        sample: Optional[bool] = False,
        col_types: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Load parquet dataset which contains multiple parquet files. Note
        that this function may duplicate memory usage for single table.
        """
        if sample:
            logger.warning("No random sample support for parquet connectors as"
                           " it slows down the UI. Return limit query.")

        from kumo.connector.file_connector import MAX_NUM_FILES_TO_SAMPLE
        start_time = time.time()
        if num_rows is not None:
            files = _list_dataset_files(path, (self.suffix, ))
            if files and any(
                    osp.dirname(file_path).rstrip('/') != path.rstrip('/')
                    for file_path in files):
                dataset = self._parquet_dataset(path, use_prebuffer=False)
                table = read_bounded_dataset(dataset, cols, num_rows)
                table = clean_parquet_table(table, col_types=col_types,
                                            sample=sample)
                return to_pandas_ns_safe(
                    table,
                    types_mapper=to_pandas_type,
                    ignore_metadata=True,
                )
            # Sort files by size to read the smallest files first when there
            # are few files. This will lead to further calls to the underlying
            # filesystem but reduces memory foorprint on loading the dataset.
            # Assumption here is that if the number of files are large,
            # they are likely of similar size.
            if files and len(files) < 10:
                files = sorted(files, key=lambda f: fsutil.filesize(f))

            # Check first 10 files to prevent OOM.
            # checking all is too expensive.
            warn_on_large_parquet_partition(
                files[:10], [fsutil.filesize(f) for f in files[:10]])

            partial_dfs: List[pd.DataFrame] = []
            num_files_scanned = 0
            for file_path in files:
                # Files that are not directories and are either empty or do
                # not have a Parquet suffix are not data files, so do not try
                # to read them.
                if (fsutil.filesize(file_path) == 0
                        or not file_path.endswith(suffix)):
                    continue
                partial_dfs.append(
                    self.load_file(file_path, cols=cols, num_rows=num_rows,
                                   col_types=col_types))
                num_files_scanned += 1
                num_rows -= partial_dfs[-1].shape[0]
                if (num_rows <= 0
                        or num_files_scanned >= MAX_NUM_FILES_TO_SAMPLE
                        or time.time() - start_time >= _MAX_READ_SAMPLE_TIME):
                    break

            ignore_index = isinstance(partial_dfs[0].index, pd.RangeIndex)
            df = pd.concat(partial_dfs, ignore_index=ignore_index)

        else:
            if fsutil.is_s3(path) or fsutil.is_gcs(path) or fsutil.is_ucv(
                    path) or fsutil.is_adls(path):
                df = self._load_pandas_df_from_directory(path, cols)
            else:
                pf = pq.ParquetDataset(path, pre_buffer=sample)
                table = pf.read(columns=cols)
                table = clean_parquet_table(table, col_types=col_types,
                                            sample=sample)
                df = to_pandas_ns_safe(table, types_mapper=to_pandas_type)

        assert len(df) > 0
        return df

    def _load_pandas_df_from_directory(
            self, path: str, cols: Optional[List[str]] = None) -> pd.DataFrame:
        r"""
        Create a pandas dataframe from a directory containing parquet files.

        Args:
            path (str): The path to the directory (this can be a local path,
                        s3 path or UC volume path).
            cols (List[str], optional): The columns to be loaded.
        """
        file_paths = _list_dataset_files(path, (self.suffix, ))

        # N.B. in the timeout case, rely on workflow-level retry
        # Retry here would quiet the issue into performance impact
        with ThreadPoolExecutor() as executor:
            loaded = executor.map(
                lambda path: self.load_file(path=path, cols=cols),
                file_paths,
                timeout=S3_READ_TIMEOUT,
            )
        parts = list(loaded)
        ignore_index = isinstance(parts[0].index, pd.RangeIndex)
        df = pd.concat(filter(lambda x: len(x) > 0, parts),
                       ignore_index=ignore_index)
        return df


# Explicit dataclass definition ###############################################

ParquetConnectorConfig = to_dataclass(ParquetConnector,
                                      base_cls=ConnectorConfig)
register(ParquetConnector, ParquetConnectorConfig, group='connector')
