# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any, Iterator, List, Optional, Union

import pandas as pd

from kumo.connector import Connector
from kumo.graph.column import BaseColumn

DUPLICATED_THRESHOLD = 0.5


class PandasView:
    r"""A view to access :class:`pd.DataFrame` or :class:`pd.Series` objects,
    given a :obj:`connector`, a :obj:`table_name`, a list of :obj:`cols`, and
    (optionally) a :obj:`pkey`.
    """
    def __init__(
        self,
        connector: Connector,
        table_name: str,
        cols: List[BaseColumn],
        pkey: Optional[BaseColumn] = None,
        return_series: bool = False,
    ):
        if return_series and len(cols) != 1:
            raise ValueError(f"'{self.__class__.__name__}' can only return a "
                             f"'pandas.Series' for single columns")

        self.connector = connector
        self.table_name = table_name
        self.cols = cols
        self.pkey = pkey
        self.return_series = return_series

    def __call__(self) -> Union[pd.DataFrame, pd.Series]:
        return self[:]

    def __iter__(self) -> Iterator[pd.DataFrame]:
        r"""Returns the iterator of a full table. Each elements of the iterator
        is a chunk of the table in pd.DataFrame format. The size of the element
        is controlled by the underlying connector."""
        return self.connector.get_iterator(
            self.table_name,
            pkey=self.pkey.config() if self.pkey is not None else None,
            cols=[col.base_config() for col in self.cols])

    def __getitem__(self, idx: Any) -> Union[pd.DataFrame, pd.Series]:
        if isinstance(idx, slice) and idx.stop is not None:
            if idx.step is not None and idx.step != 1:
                raise NotImplementedError(f"Slicing with 'step' parameter not "
                                          f"yet supported (got '{idx.step}')")

            start = 0 if idx.start is None else idx.start
            df = self.connector.read_table(
                self.table_name,
                pkey=self.pkey.config() if self.pkey is not None else None,
                cols=[col.base_config() for col in self.cols],
                num_rows=idx.stop - start,
                offset=start,
            )

        else:
            # TODO (matthias) Do not require full data for all index types.
            df = self.connector.read_table(
                self.table_name,
                pkey=self.pkey.config() if self.pkey is not None else None,
                cols=[col.base_config() for col in self.cols],
            )[idx]

        if self.return_series:
            df = df[self.cols[0].name]

        if self.pkey is None and not df.index.is_unique:
            df.reset_index(inplace=True)
        elif not df.index.is_unique:
            mask = df.index.duplicated(keep='first')
            frac = int(mask.sum()) / mask.shape[0]

            if frac > DUPLICATED_THRESHOLD and not self.pkey.drop_duplicates:
                raise ValueError(
                    f"The primary key column for table '{self.table_name}' "
                    f"has more than {100 * DUPLICATED_THRESHOLD:.0f}% "
                    f"duplicated values (got {100 * frac:.1f}%). Did you "
                    f"accidentally set a wrong primary key? If this is indeed "
                    f"intended, set "
                    f"`graph['{self.table_name}'].pkey.drop_duplicates` to "
                    f"`True` to ignore this error.")

            df = df[~mask]

            if not self.pkey.drop_duplicates:
                warnings.warn(
                    f"The primary key column for table '{self.table_name}' "
                    f"does not have unique values, please check if the "
                    f"primary key column is set correctly. We have dropped "
                    f"the rows with duplicated keys besides their first "
                    f"occurrences. "
                    f"Set `graph['{self.table_name}'].pkey.drop_duplicates` "
                    f"to `True` to remove this warning.")

        return df

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
    ) -> Union[pd.DataFrame, pd.Series]:

        df = self.connector.read_table_sample(
            self.table_name,
            pkey=self.pkey.config() if self.pkey is not None else None,
            cols=[col.base_config() for col in self.cols],
            num_rows=n,
            frac=frac,
        )

        if self.return_series:
            df = df[self.cols[0].name]

        return df

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.table_name})'
