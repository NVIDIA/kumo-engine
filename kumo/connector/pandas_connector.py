# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

import pandas as pd
from kumo_core.types.conversions import to_dtype

from kumo.config import BaseColumnConfig, TableMetadataConfig
from kumo.connector import Connector
from kumo.connector.utils import SAMPLING_SEED, cast_df


class PandasConnector(Connector):
    r"""A wrapper to connect to in-memory :class:`pandas.DataFrame` objects.

    Args:
        df_dict (Dict[str, pandas.DataFrame], optional): Dictionary of
            :class:`pandas.DataFrame` objects with their respective names.
            (default: :obj:`None`)
    """
    def __init__(self, df_dict: Optional[Dict[str, pd.DataFrame]] = None):
        super().__init__(name='pandas')
        self.df_dict = df_dict or {}

    def _get_connector_type(self) -> str:
        return "pandas"

    def add_df(self, table_name: str, df: pd.DataFrame) -> 'PandasConnector':
        self.df_dict[table_name] = df
        return self

    # Connecting and closing connections ######################################

    def _open(self, credentials: Optional[Dict] = None) -> Any:
        pass

    def _close(self, context: Any):
        pass

    # Reading raw table data ##################################################

    def _read_table(
        self,
        context: Any,
        table_name: str,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> pd.DataFrame:
        return get_df(self.df_dict[table_name], pkey, cols)

    def _read_table_chunk(
        self,
        context: Any,
        table_name: str,
        num_rows: int,
        offset: int = 0,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> pd.DataFrame:
        df = get_df(self.df_dict[table_name], pkey, cols)
        return df[offset:offset + num_rows]

    def _read_table_sample(
        self,
        context: Any,
        table_name: str,
        num_rows: Optional[int],
        frac: Optional[float],
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> pd.DataFrame:
        df = get_df(self.df_dict[table_name], pkey, cols)
        if num_rows is not None:
            num_rows = min(num_rows, len(df))
        return df.sample(num_rows, frac, random_state=SAMPLING_SEED)

    # Read metadata ##################################################

    def list_tables(self, schema_name: Optional[str] = None) -> List[str]:
        raise NotImplementedError

    def _get_table_cols_metadata(
        self,
        table_name: str,
        skip_cache: bool = False,
        skip_inferral: bool = False,
    ) -> Optional[TableMetadataConfig]:
        metadata = TableMetadataConfig(table_name=table_name,
                                       source_name=table_name,
                                       data_source=self._name)

        df = self.df_dict[table_name]

        # Pandas Categorical type is not supported as input in Kumo right now.
        # So converting everything to a string which will be converted to a
        # categorical feature based on cardinality.
        category_cols: List[tuple] = list(
            filter(lambda val: val[1] == 'category', zip(df.columns,
                                                         df.dtypes)))
        if len(category_cols) != 0:
            cols_to_convert: Dict[str, str] = {
                category_col[0]: 'string'
                for category_col in category_cols
            }
            df = df.astype(cols_to_convert)
        dtypes = {name: to_dtype(dtype) for name, dtype in df.dtypes.items()}

        if df.index.name is not None:
            dtypes[df.index.name] = to_dtype(df.index.dtype)

        for name, dtype in dtypes.items():
            metadata.cols.append(BaseColumnConfig(name, dtype=dtype))
        return metadata

    # Data statistics #########################################################

    def _get_table_num_rows(self, table_name: str,
                            data_version: str | None = None) -> int:
        return len(self.df_dict[table_name].index)

    def _get_table_source_size(self, table_name: str,
                               data_version: str | None = None) -> int:
        return self.df_dict[table_name].memory_usage().sum()

    # Python built-ins #######################################################

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


# Utility functions ###########################################################


def get_df(
    df: pd.DataFrame,
    pkey: Optional[BaseColumnConfig] = None,
    cols: Optional[List[BaseColumnConfig]] = None,
):
    if pkey is not None and pkey.name != df.index.name:
        df = df.set_index(pkey.name)

    if pkey is None and df.index.name is not None:
        df = df.reset_index()

    df = cast_df(df, pkey, cols)

    if cols is not None:
        df = df[[col.name for col in cols]]

    return df


# Explicit dataclass definition ###############################################

# NOTE We do not register `PandasConnector` to the configuration store.
