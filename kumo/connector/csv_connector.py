# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from io import StringIO
from typing import Dict, List, Optional, Union

import pandas as pd
from kumo_core.sources import csv_format, layout
from kumo_core.types.conversions import to_dtype
from kumo_core.utils.registry import register
from kumoapi.typing import Dtype

from io_utils import fsutil, logging
from kumo.config import BaseColumnConfig, ConnectorConfig, TableMetadataConfig
from kumo.connector.blocking import blocking_func, blocking_method
from kumo.connector.connector import ConnectorError, DuplicateColumnsError
from kumo.connector.file_connector import FileConnector
from kumo.connector.utils import SAMPLING_SEED, cast_df_to_dtypes
from kumo.data.common import DATA_BACKEND_THREAD_POOL
from kumo.utils import detect_csv_sep, to_dataclass

logger = logging.getLogger(__name__)

# If sampling rows, this is how many times more rows to read than the requested
# number, so that they can be sampled
CSV_SAMPLE_MULTIPLIER = 10

# The ceilings a CSV table is held to, shared with the local backend so a
# graph this refuses cannot be built there either.
MAX_CSV_PARTITIONS = layout.MAX_CSV_PARTITIONS
MAX_CSV_DATASIZE = layout.MAX_CSV_DATASIZE


class CSVConnector(FileConnector):
    r"""Establishes a connection to a set of :obj:`*.csv` files, either on-disk
    or remote.

    Args:
        name (str): The name of the connector.
        root_dir (str): Root directory where :obj:`*.csv` files are stored.
        suffix (str, optional): Suffix of the files. (default: :obj:`"csv"`)
        sep (str, optional): Delimiter to use. (default: :obj:`None`)
        na_values (str or List[str], optional): Additional strings to recognize
            as :obj:`NA`/:obj:`NaN`. (default: :obj:`None`)
        compression (str, optional): On-the-fly decompression, such as
            :obj:`gzip`. (default: :obj:`"infer"`)
        on_bad_lines (str, optional): Specifies what to do upon encountering
            a bad line. Allowed values are :obj:`"error"`, :obj:`"warn"`, and
            :obj:`"skip"`. (default: :obj:`skip`)
        escapechar (str, optional): One-character string used to escape other
            characters. (default: :obj:`None`)
    """
    def __init__(
        self,
        name: str,
        root_dir: str,
        suffix: str = 'csv',
        sep: Optional[str] = None,
        na_values: Union[str, List[str], None] = None,
        compression: str = 'infer',
        on_bad_lines: str = 'skip',
        escapechar: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, root_dir, suffix=suffix, **kwargs)

        self.sep = sep
        self.na_values = na_values
        self.compression = compression
        self.on_bad_lines = on_bad_lines
        self.escapechar = escapechar

    def _get_connector_type(self) -> str:
        return "csv"

    def _get_sep(self, path: str) -> str:
        """
        Returns a separator detected using :func:`detect_csv_sep` using the
        first file found using .csv extension. If a separator is not detected,
        returns ','.
        """
        if self.sep is None:
            if fsutil.isdir(path):
                files = fsutil.listdir(path=path, suffix=self.suffix)
                for file_path in files:
                    if (fsutil.isdir(file_path)
                            or fsutil.filesize(file_path) == 0):
                        continue
                    return self._infer_separator(file_path)
            else:
                return self._infer_separator(path)
        return self.sep

    @staticmethod
    def _infer_separator(file_path):
        sep = detect_csv_sep(file_path)
        return sep

    def _get_table_cols_metadata(
        self,
        table_name: str,
        skip_cache: bool = False,
        skip_inferral: bool = False,
    ) -> Optional[TableMetadataConfig]:
        metadata = TableMetadataConfig(table_name=table_name,
                                       source_name=table_name,
                                       data_source=self._name)

        try:
            df = self.read_table_sample(table_name, skip_cache=skip_cache)
        except FileNotFoundError as e:
            raise (e)
        except DuplicateColumnsError as e:
            logger.warning(f'Duplicate columns found in Table {table_name}')
            raise (e)
        except Exception as e:
            logger.warning('Table metadata read failed (error_type=%s)',
                           type(e).__name__)
            return None

        from kumo_core.sources.csv_preview import column_dtypes

        dtypes = column_dtypes(df)
        if df.index.name is not None:
            dtypes[df.index.name] = to_dtype(df.index.dtype)

        for name, dtype in dtypes.items():
            metadata.cols.append(BaseColumnConfig(name, dtype=dtype))
        return metadata

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

        pkey = BaseColumnConfig.cast(pkey)
        if cols is not None:
            cols = [BaseColumnConfig.cast(col) for col in cols]

        if 'sep' not in kwargs and 'delimiter' not in kwargs:
            kwargs['sep'] = self._get_sep(path)

        if self.na_values is not None:
            kwargs['na_values'] = self.na_values

        kwargs['compression'] = self.compression
        kwargs['on_bad_lines'] = self.on_bad_lines
        kwargs['escapechar'] = self.escapechar

        dtypes: Dict[str, Dtype] = {}
        index_col: Optional[str] = None

        if pkey is not None:
            index_col = pkey.name
            if pkey.dtype is not None:
                dtypes[pkey.name] = pkey.dtype

        if cols is not None:
            kwargs['usecols'] = [] if pkey is None else [pkey.name]

            for col in cols:
                col = BaseColumnConfig.cast(col)
                kwargs['usecols'].append(col.name)
                if col.dtype is not None:
                    dtypes[col.name] = col.dtype
        if sample:
            if num_rows is not None:
                # Sampling from a large csv file on s3 is slow so this is a
                # pseude sampling {nrows} rows from top
                # {nrows * CSV_SAMPLE_MULTIPLIER} rows.
                kwargs['nrows'] = num_rows * CSV_SAMPLE_MULTIPLIER
            else:
                # nrows is not supported by pyarrow engine
                kwargs['engine'] = 'pyarrow'
            df = read_csv_safe(path, dtypes=dtypes, suffix=self.suffix,
                               index_col=index_col, sample=sample, **kwargs)
            if num_rows is not None:
                return df.sample(n=min(num_rows, df.shape[0]),
                                 random_state=SAMPLING_SEED)
            else:
                return df.sample(frac=frac, random_state=SAMPLING_SEED)
        else:
            if num_rows is not None:
                kwargs['nrows'] = num_rows
            if offset > 0:
                kwargs['skiprows'] = range(1, offset + 1)
            return read_csv_safe(path, dtypes=dtypes, suffix=self.suffix,
                                 index_col=index_col, sample=sample, **kwargs)

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def _get_table_num_rows(self, table_name: str,
                            data_version: str | None = None) -> int:
        table_path = self._get_path(table_name)
        if fsutil.exists(table_path):
            file_paths = self.table_path_to_files(path=table_path)
            refusal = layout.csv_partitions_refusal(len(file_paths))
            if refusal is not None:
                raise ConnectorError(message=refusal)

            num_rows = 0
            for file_path in file_paths:
                num_rows += fsutil.file_lines(file_path)
            return num_rows

        raise FileNotFoundError(
            f"Obtaining the number of rows for table '{table_name}' "
            f"is not implemented for the remote directory "
            f"{osp.join(self.root_dir, table_name)} or file does not exist")

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def _get_table_source_size(self, table_name: str,
                               data_version: str | None = None) -> int:
        path = self._get_path(table_name)
        file_paths = self.table_path_to_files(path=path)
        total_size = 0
        for path in file_paths:
            total_size += fsutil.filesize(path) or 0
            refusal = layout.csv_size_refusal(total_size)
            if refusal is not None:
                raise ConnectorError(message=refusal)
        return total_size

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def _write(self, path: str, df: pd.DataFrame):
        df.to_csv(path)

    def _source_format(self, path: str):
        return layout.CSV

    def _csv_read_options(self, path: str) -> csv_format.CsvOptions:
        nulls = (tuple(self.na_values) if isinstance(self.na_values, list) else
                 (self.na_values, ) if self.na_values is not None else None)
        return csv_format.CsvOptions(delimiter=self.sep,
                                     escape_char=self.escapechar or False,
                                     null_values=nulls)


@blocking_func()
def read_csv_safe(
    path: Union[str, StringIO],
    dtypes: Dict[str, Dtype],
    index_col: Optional[str] = None,
    sample: bool = False,
    suffix: str = 'csv',
    dtype: Optional[Dtype] = None,
    **kwargs,
) -> pd.DataFrame:
    r"""Reads a csv file with appropriate exception handling in the case
    of type mismatches.

    Note: the dtype could be either `None` or `str` indicates that we are
    loading all columns as `str` to better infer the data type. It should
    not be called outside of this function.
    """
    from kumo_core.sources import csv_preview

    from kumo.connector.file_connector import MAX_NUM_FILES_TO_SAMPLE
    from kumo.connector.object_store_access import source_reader

    if isinstance(path, str):
        reader = source_reader(path)
        files = [
            entry.path for entry in layout.objects(reader.filesystem,
                                                   reader.root, suffix=suffix)
            if entry.size != 0
        ]

        def read(options):
            return csv_preview.read_frames(
                reader.filesystem, files, max_files=MAX_NUM_FILES_TO_SAMPLE
                if sample and 'nrows' in options else None, **options)
    else:

        def read(options):
            path.seek(0)
            return pd.read_csv(path, **options)

    try:
        df = csv_preview.read_pandas(read, dtypes, index_col=index_col,
                                     dtype=dtype, **kwargs)
    except csv_preview.DuplicateColumnsError as error:
        raise DuplicateColumnsError(str(error)) from None

    # Pandas may not necessarily cast the result into the proper data type
    # (particularly in the case of datetime column that could not be parsed),
    # so we explicitly cast a second time.

    df = cast_df_to_dtypes(df, dtypes)

    if index_col is not None:
        df = df.set_index(index_col)

    return df


# Explicit dataclass definition ###############################################

CSVConnectorConfig = to_dataclass(CSVConnector, base_cls=ConnectorConfig)
register(CSVConnector, CSVConnectorConfig, group='connector')
