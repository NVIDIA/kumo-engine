# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple, Union

import kumo_core.train.mapping as core_mapping
import pandas as pd
import pyarrow as pa
import torch
from kumo_core.types.conversions import PandasDtype
from pandas.api import types
from torch import Tensor

from io_utils import logging

logger = logging.getLogger(__name__)


def get_index_map(df: pd.DataFrame, index_offset: int = 0) -> pd.DataFrame:
    r"""Returns a :class:`pandas.DataFrame` that can be later used to map
    indices to a contiguous range in :obj:`{ 0, ..., len(df) - 1 }`."""
    arange = pd.RangeIndex(index_offset, index_offset + len(df))
    return pd.DataFrame(index=df.index,
                        data={core_mapping.NODE_INDEX_COLUMN: arange})


def node_idx_tensor(index_map: pd.DataFrame) -> torch.Tensor:
    r"""Returns a node idx :obj:`torch.Tensor` from the index map."""
    return torch.from_numpy(
        index_map[core_mapping.NODE_INDEX_COLUMN].values).to(torch.long)


def merge_size(
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
    left_on: str,
    right_on: str,
    how='inner',
) -> Tuple[int, Dict[str, int]]:
    # Compute sizes for each repeated component:
    left_groups = left_frame.groupby(left_on).size()
    right_groups = right_frame.groupby(right_on).size()

    intersection = left_groups.index.intersection(right_groups.index)
    left_diff = left_groups.index.difference(intersection)
    right_diff = right_groups.index.difference(intersection)

    # Handle NA values:
    left_nan = left_frame[left_on].isna().sum()
    right_nan = right_frame[right_on].isna().sum()

    if left_nan == 0 and right_nan > 0:
        left_nan = 1
    elif left_nan > 0 and right_nan == 0:
        right_nan = 1

    # Compute join size:
    key_sizes = {'NA': left_nan * right_nan}
    for key in intersection:
        key_sizes[key] = left_groups[key] * right_groups[key]

    largest_contributors = sorted(key_sizes.items(), key=lambda x: -x[1])[:10]

    left_size = sum(left_groups[key] for key in left_diff)
    right_size = sum(right_groups[key] for key in right_diff)

    total_size = sum(key_sizes.values())
    if how == 'left' or how == 'outer':
        total_size += left_size
    if how == 'right' or how == 'outer':
        total_size += right_size
    return total_size, largest_contributors


def map_values(
    src_data: pd.Series,
    src_index_map: pd.DataFrame,
    dst_data: pd.Series,
    dst_index_map: pd.DataFrame,
) -> Tuple[Tensor, Tensor]:
    r"""Returns an :obj:`(row, col)` that represents a mapping of values from
    :obj:`src_data` to :obj:`dst_data`, based on the values in each data
    object and indices in each data object's corresponding index map."""

    # Drop NA values to prevent merging on them:
    src_data = src_data.dropna()
    dst_data = dst_data.dropna()

    # Change the data types to make them more compatible
    src_data, dst_data = _maybe_cast_to_compatible_types(src_data, dst_data)

    # Join index map for src_data and dst_data:
    src = src_index_map.merge(
        src_data.rename('src'), left_index=True, right_index=True).rename(
            columns={core_mapping.NODE_INDEX_COLUMN: 'row'})

    dst = dst_index_map.merge(
        dst_data.rename('dst'), left_index=True, right_index=True).rename(
            columns={core_mapping.NODE_INDEX_COLUMN: 'col'})

    # Perform the actual merge:
    try:
        out = dst.merge(src, left_on='dst', right_on='src')
    except MemoryError:
        size, contributors = merge_size(src, dst, 'src', 'dst')
        logger.exception(
            "Edge index creation failed: cannot create %s edges "
            "(largest contributors: %s)", size, contributors)
        raise

    row = torch.from_numpy(out['row'].values).to(torch.long)
    col = torch.from_numpy(out['col'].values).to(torch.long)

    return row, col


def map_index(
    ser: Union[pd.Series, List[Any]],
    index_map: pd.DataFrame,
    return_mask: bool = False,
    default_val: Optional[int] = None,
    filter_nan: bool = True,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Maps entries in :obj:`ser` to a contiguous range based on the indices
    specified in :obj:`index_map`."""

    if isinstance(index_map.index, pd.RangeIndex):
        raise ValueError("Cannot map indices to a contiguous range without "
                         "an explicitly given primary key")

    if not isinstance(ser, pd.Series):
        ser = pd.Series(data=ser)

    # Change the data types to make them more compatible
    ser, new_index = _maybe_cast_to_compatible_types(ser, index_map.index)
    if new_index.dtype != index_map.index.dtype:
        index_map = index_map.set_index(new_index)

    mask: Optional[Tensor] = None
    if filter_nan:
        # If filter_nan is True, we filter out NaN values from the pd.Series.
        if ser.isnull().any():
            mask = ser.notnull()
            ser = ser[mask]
            mask = torch.from_numpy(mask.values)

    ser = ser.rename('key').astype(index_map.index.dtype)
    out = index_map.merge(ser, how='right', left_index=True, right_on='key')
    out = out[core_mapping.NODE_INDEX_COLUMN]
    if default_val is not None:
        out.fillna(default_val, inplace=True)
    out = torch.from_numpy(out.values)

    nan_mask: Optional[Tensor] = None
    if out.is_floating_point():  # are there any NA values?
        nan_mask = torch.isnan(out).logical_not_()
        out = out[nan_mask].to(torch.long)

    if not return_mask:
        return out

    if mask is not None and nan_mask is not None:
        mask[mask.clone()] = nan_mask
    elif mask is None:
        mask = nan_mask

    return out, mask


class MapIndex:
    def __init__(self, index_map: pd.DataFrame):
        r"""Create a :obj:`MapIndex` that maps index values from
        the :obj:`index_map` to the integer id.

        Args:
            index_map (pd.DataFrame): A :obj:`pd.DataFrame` that contains the
                index values and the corresponding integer id.
        """
        self.index_map = get_index_map(index_map)
        self.length = len(self.index_map)

    def __len__(self):
        return self.length

    def map_index(
        self,
        ser: Union[pd.Series, List[Any]],
        return_mask: bool = False,
        default_val: Optional[int] = None,
        filter_nan: Optional[bool] = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        r"""Maps entries in :obj:`ser` to a contiguous range based on the
        indices specified in :obj:`index_map`, refer to :func:`map_index`
        for more details.
        """
        return map_index(ser, self.index_map, return_mask,
                         default_val=default_val, filter_nan=filter_nan)


_SeriesIndex = Union[pd.Series, pd.Index]


def _maybe_cast_to_compatible_types(
        ser1: _SeriesIndex,
        ser2: _SeriesIndex) -> Tuple[_SeriesIndex, _SeriesIndex]:
    """Cast the two series into data types that are safe to merge
    together, using `pd.merge`

    The logic should match the validation specified by
    `ForeignKey.can_link_to_dtype`.

    Note, `pd.merge` already does a bunch of internal casting, to support
    situations such as merging a float with an integer. So, here we only have
    to handle the case of merging numbers with strings.
    """

    ser1 = _maybe_cast_str_series_to_numeric(ser1, ser2.dtype)
    ser2 = _maybe_cast_str_series_to_numeric(ser2, ser1.dtype)

    return ser1, ser2


def _maybe_cast_str_series_to_numeric(
        ser: Union[pd.Series,
                   pd.Index], dtype: PandasDtype) -> Union[pd.Series, pd.Index]:
    # Note, we cast strings to numbers (rather than numbers to strings) to
    # avoid differences in string formatting between floats and ints, such as
    # `1.0` vs `1`.
    if not types.is_string_dtype(ser.dtype):
        return ser

    if not types.is_numeric_dtype(dtype):
        return ser

    new_ser = pd.to_numeric(ser)

    logger.info("While mapping, cast series from data type %s to %s", ser.dtype,
                new_ser.dtype)

    return new_ser


def _maybe_combine(arr: pa.Array) -> pa.Array:
    if isinstance(arr, pa.ChunkedArray):
        return arr.combine_chunks()
    return arr
