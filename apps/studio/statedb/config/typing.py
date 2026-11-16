# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Dict, Optional, TypeAlias, Union

import db_dtypes
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from kumoapi.typing import Dtype
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_list_like,
    is_object_dtype,
)
from pandas.core.dtypes.base import ExtensionDtype
from torch import Tensor

Device: TypeAlias = Union[int, str, torch.device]

PandasDataFrame: TypeAlias = pd.DataFrame
DataFrame: TypeAlias = PandasDataFrame
PandasDtype: TypeAlias = Union[np.dtype, ExtensionDtype, None]
TrainModuleOutput: TypeAlias = dict[str, Tensor | dict[str, Tensor] | None]

_MAX_NUM_ROWS_FOR_DTYPE_INFERENCE = 1_000

DTYPE_TO_NUMPY = {
    Dtype.bool: np.dtype('bool'),
    Dtype.int: np.dtype('int64'),
    Dtype.byte: np.dtype('byte'),
    Dtype.int16: np.dtype('int16'),
    Dtype.int32: np.dtype('int32'),
    Dtype.int64: np.dtype('int64'),
    Dtype.float: np.dtype('float32'),
    Dtype.float32: np.dtype('float32'),
    Dtype.float64: np.dtype('float64'),
    Dtype.string: np.dtype('object'),
    Dtype.date: np.dtype('datetime64[ns]'),
    Dtype.time: np.dtype('datetime64[ns]'),
    Dtype.timedelta: np.dtype('timedelta64[ns]'),
    Dtype.floatlist: np.dtype('object'),
    Dtype.intlist: np.dtype('object'),
    Dtype.stringlist: np.dtype('object'),
    Dtype.binary: np.dtype('object'),
    Dtype.unsupported: None,
}

# Note that we cannot simply convert binary types from pyarrow to pandas
# object dtype, but we can convert pandas object dtype to pandas string
# dtype supported by pyarrow.
# See kumo.connector.utils.cast_df_to_dtypes.
DTYPE_TO_PANDAS: Dict[Dtype, PandasDtype] = {
    Dtype.bool: pd.BooleanDtype(),
    Dtype.int: pd.Int64Dtype(),
    Dtype.byte: pd.UInt8Dtype(),
    Dtype.int16: pd.Int16Dtype(),
    Dtype.int32: pd.Int32Dtype(),
    Dtype.int64: pd.Int64Dtype(),
    Dtype.float: np.dtype('float32'),
    Dtype.float32: np.dtype('float32'),
    Dtype.float64: np.dtype('float64'),
    Dtype.date: np.dtype('datetime64[ns]'),
    Dtype.time: np.dtype('datetime64[ns]'),
    Dtype.timedelta: np.dtype('timedelta64[ns]'),
    Dtype.string: pd.StringDtype(storage='pyarrow'),
    Dtype.floatlist: np.dtype('object'),
    Dtype.intlist: np.dtype('object'),
    Dtype.stringlist: np.dtype('object'),
    Dtype.binary: pd.StringDtype(),
    Dtype.unsupported: None,
}

DTYPE_TO_TORCH = {
    Dtype.bool: torch.bool,
    Dtype.int: torch.int64,
    Dtype.byte: torch.uint8,
    Dtype.int16: torch.int16,
    Dtype.int32: torch.int32,
    Dtype.int64: torch.int64,
    Dtype.float: torch.float32,
    Dtype.float32: torch.float32,
    Dtype.float64: torch.float64,
}

TO_DTYPE = {
    np.dtype('bool'): Dtype.bool,
    pd.BooleanDtype(): Dtype.bool,
    torch.bool: Dtype.bool,
    pa.bool_(): Dtype.bool,
    np.dtype('byte'): Dtype.int32,
    pd.UInt8Dtype(): Dtype.int32,
    pd.Int8Dtype(): Dtype.int32,
    torch.uint8: Dtype.int32,
    np.dtype('int16'): Dtype.int32,
    pd.Int16Dtype(): Dtype.int32,
    torch.int16: Dtype.int32,
    np.dtype('int32'): Dtype.int32,
    pd.Int32Dtype(): Dtype.int32,
    torch.int32: Dtype.int32,
    np.dtype('int64'): Dtype.int64,
    pd.Int64Dtype(): Dtype.int64,
    torch.int64: Dtype.int64,
    np.dtype('float32'): Dtype.float32,
    torch.float32: Dtype.float32,
    pd.Float32Dtype(): Dtype.float32,
    np.dtype('float64'): Dtype.float64,
    torch.float64: Dtype.float64,
    pd.Float64Dtype(): Dtype.float64,
    np.dtype('object'): Dtype.string,
    pd.StringDtype(storage='python'): Dtype.string,
    pd.StringDtype(storage='pyarrow'): Dtype.string,
    pa.string(): Dtype.string,
    pa.binary(): Dtype.binary,
    np.dtype('datetime64[ns]'): Dtype.date,
    np.dtype('timedelta64[ns]'): Dtype.timedelta,
    pa.list_(pa.float32()): Dtype.floatlist,
    pa.list_(pa.int64()): Dtype.intlist,
    pa.list_(pa.string()): Dtype.stringlist,
    db_dtypes.DateDtype(): Dtype.date,
}

PANDAS_TO_SQL = {
    pd.BooleanDtype(): 'BOOLEAN',
    np.dtype('bool'): 'BOOLEAN',
    np.dtype('object'): 'VARCHAR',
    pd.Int16Dtype(): 'SMALLINT',
    pd.Int32Dtype(): 'INT',
    pd.Int64Dtype(): 'BIGINT',
    np.dtype('int64'): 'BIGINT',
    np.dtype('int32'): 'INT',
    np.dtype('float32'): 'FLOAT',
    np.dtype('float64'): 'DOUBLE',
    np.dtype('datetime64[ns]'): 'TIMESTAMP_NTZ',
    np.dtype('timedelta64[ns]'): 'INTERVAL',
    pd.StringDtype(): 'VARCHAR',
    pd.StringDtype(storage='pyarrow'): 'VARCHAR',
}

TO_PYARROW_DTYPE = {
    Dtype.bool: pa.bool_(),
    Dtype.int: pa.int64(),
    Dtype.byte: pa.int8(),
    Dtype.int16: pa.int16(),
    Dtype.int32: pa.int32(),
    Dtype.int64: pa.int64(),
    Dtype.float: pa.float32(),
    Dtype.float32: pa.float32(),
    Dtype.float64: pa.float64(),
    Dtype.string: pa.string(),
    Dtype.binary: pa.binary(),
    Dtype.floatlist: pa.list_(pa.float32()),
    Dtype.intlist: pa.list_(pa.int64()),
    Dtype.stringlist: pa.list_(pa.string()),
    Dtype.unsupported: None,
}

PYARROW_TO_DTYPE = {
    v:
    k if k not in (Dtype.byte,
                   Dtype.int16) else Dtype.int32
    for k, v in TO_PYARROW_DTYPE.items()
    if k is not Dtype.unsupported
}


def to_pyarrow_type(dtype: Any) -> Any:
    r"""Map any data type to pyarrow data type and return None to use
    default mapping."""
    if dtype in TO_DTYPE:
        kumo_dtype = TO_DTYPE[dtype]
        # NOTE: Since this function is currently only used for table
        # metadata inference, this conditional is used to stop customer
        # data from using the internal list `Dtype`
        return TO_PYARROW_DTYPE[kumo_dtype] if not kumo_dtype.is_list(
        ) else None
    return None


def unsupported_pyarrow_composite(dtype: Any) -> bool:
    r"""Returns True if the input pyarrow data type is some sort of composite
    that kumo does not support. Kumo supports lists of floats, integers and
    strings but no other lists and no maps or structs. Input dtype may be from
    pa.lib or another pyarrow type.
    """
    if pa.types.is_map(dtype) or pa.types.is_struct(dtype):
        return True

    if pa.types.is_list(dtype):
        # if not a support type of list value, pyarrow_to_dtype
        # will raise a KeyError trying to look in PYARROW_TO_DTYPE
        try:
            pyarrow_to_dtype(dtype)
        except KeyError:
            return True

    return False


def pyarrow_to_dtype(dtype: Any) -> Any:
    r"""Not to be confused with to_dtype, because this mapping should not
    end up as `pa.Table.to_pandas(types_mapper)` argument. Since pandas
    ints and floats do not implement `__from_arrow__` method, that leads
    to errors."""
    if isinstance(dtype, pa.lib.DataType) and pa.types.is_temporal(dtype):
        if isinstance(dtype, pa.DurationType):
            return Dtype.timedelta
        if pa.types.is_time(dtype):
            return Dtype.time
        return Dtype.date

    if isinstance(dtype, pa.DictionaryType):  # Categorical type
        return Dtype.string

    if pa.types.is_list(dtype):
        if pa.types.is_floating(dtype.value_type):
            return Dtype.floatlist
        if pa.types.is_integer(dtype.value_type):
            return Dtype.intlist
        if pa.types.is_string(dtype.value_type):
            return Dtype.stringlist

    # Handle pyarrow 14 case that value type is element string instead of
    # item string.
    if pa.types.is_list(dtype) and pa.types.is_string(dtype.value_type):
        kumo_dtype = Dtype.stringlist
    else:
        kumo_dtype = PYARROW_TO_DTYPE[dtype]
    return kumo_dtype if not kumo_dtype.is_list(
    ) else Dtype.unsupported


def to_dtype(dtype: Any,
             ser: Optional[pd.Series] = None) -> Dtype:
    if is_datetime64_any_dtype(dtype):
        return Dtype.date
    if isinstance(dtype, pd.CategoricalDtype):
        return Dtype.string
    if is_object_dtype(dtype) and ser is not None and len(ser) > 0:
        if is_list_like(ser.iloc[0]):
            # the 0-th element might be an empty list or a list of N/A so
            # iterate over the first _MAX_NUM_ROWS_FOR_DTYPE_INFERENCE elements
            # to infer the dtype:
            for i in range(min(len(ser), _MAX_NUM_ROWS_FOR_DTYPE_INFERENCE)):
                # pd.isna can't be called on a list
                if (not isinstance(ser.iloc[i], list)
                        and not isinstance(ser.iloc[i], np.ndarray)
                        and pd.isna(ser.iloc[i])) or len(ser.iloc[i]) == 0:
                    continue
                elif isinstance(ser.iloc[i][0], float):
                    return Dtype.floatlist
                elif np.issubdtype(type(ser.iloc[i][0]), int):
                    return Dtype.intlist
                elif isinstance(ser.iloc[i][0], str):
                    return Dtype.stringlist
        else:
            # Snowflake object-dtype timestamps must be typed temporal, not
            # string, or MIN/MAX get dropped. infer_dtype also catches
            # np.datetime64 scalars an isinstance(date) probe would miss.
            if pd.api.types.infer_dtype(ser) in {
                    'datetime', 'date', 'datetime64'
            }:
                return Dtype.date

    # Check to propagate more meaningful error message
    if dtype not in TO_DTYPE:
        raise KeyError(f"Unsupported data type: {dtype}.")

    return TO_DTYPE[dtype]


def to_pandas_type(dtype: Any) -> Any:
    """Map any data type to pandas data type and return None to use default
    mapping."""
    if dtype in TO_DTYPE:
        kumo_dtype = TO_DTYPE[dtype]
        # NOTE: Since this function is currently only used for table
        # metadata inference, this conditional is used to stop customer
        # data from using the internal list `Dtype`
        return DTYPE_TO_PANDAS[kumo_dtype] if not kumo_dtype.is_list() else None
    return None


def cast_to_pandas_dtype(value: Any, dtype: Any) -> Any:
    r"""Casts a :obj:`value` to a pandas :obj:`dtype`."""
    return pd.Series([value], dtype=dtype).values[0]


# Dictionary for representing a set of paths keyed by str
PathDict = Dict[str, str]


class EdgeDirection(Enum):
    r"""Defines the ways to materialize edges."""
    SRC_TO_DST = 'src_to_dst'
    DST_TO_SRC = 'dst_to_src'
    BOTH = 'both'


# Types for Cohort Analysis ########################################

# Types for KumoBench #########################################################


class ContentType(Enum):
    r"""Defines the content to list from KumoBench and StateDB."""
    GRAPHS = 'graphs'
    QUERIES = 'queries'
