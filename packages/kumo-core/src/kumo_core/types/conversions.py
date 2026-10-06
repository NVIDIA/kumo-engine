# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conversions between Kumo dtypes and numerical-library dtypes."""

from __future__ import annotations

from typing import Any, TypeAlias

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

DataFrame: TypeAlias = pd.DataFrame
PandasDtype: TypeAlias = np.dtype | pd.api.extensions.ExtensionDtype | None

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
_PYARROW_TO_DTYPE = {
    value: key if key not in (Dtype.byte, Dtype.int16) else Dtype.int32
    for key, value in TO_PYARROW_DTYPE.items() if key is not Dtype.unsupported
}

_TO_DTYPE = {
    np.dtype("bool"): Dtype.bool,
    pd.BooleanDtype(): Dtype.bool,
    torch.bool: Dtype.bool,
    pa.bool_(): Dtype.bool,
    np.dtype("byte"): Dtype.int32,
    pd.UInt8Dtype(): Dtype.int32,
    pd.Int8Dtype(): Dtype.int32,
    torch.uint8: Dtype.int32,
    np.dtype("int16"): Dtype.int32,
    pd.Int16Dtype(): Dtype.int32,
    torch.int16: Dtype.int32,
    np.dtype("int32"): Dtype.int32,
    pd.Int32Dtype(): Dtype.int32,
    torch.int32: Dtype.int32,
    np.dtype("int64"): Dtype.int64,
    pd.Int64Dtype(): Dtype.int64,
    torch.int64: Dtype.int64,
    np.dtype("float32"): Dtype.float32,
    pd.Float32Dtype(): Dtype.float32,
    torch.float32: Dtype.float32,
    np.dtype("float64"): Dtype.float64,
    pd.Float64Dtype(): Dtype.float64,
    torch.float64: Dtype.float64,
    np.dtype("object"): Dtype.string,
    pd.StringDtype(storage="python"): Dtype.string,
    pd.StringDtype(storage="pyarrow"): Dtype.string,
    pa.string(): Dtype.string,
    pa.binary(): Dtype.binary,
    np.dtype("datetime64[ns]"): Dtype.date,
    np.dtype("timedelta64[ns]"): Dtype.timedelta,
    pa.list_(pa.float32()): Dtype.floatlist,
    pa.list_(pa.int64()): Dtype.intlist,
    pa.list_(pa.string()): Dtype.stringlist,
}


def to_dtype(dtype: Any, ser: pd.Series | None = None) -> Dtype:
    if isinstance(dtype, Dtype):
        return dtype
    if is_datetime64_any_dtype(dtype):
        return Dtype.date
    if isinstance(dtype, pd.CategoricalDtype):
        return Dtype.string
    if is_object_dtype(dtype) and ser is not None and len(ser) > 0:
        if is_list_like(ser.iloc[0]):
            for value in ser.iloc[:1000]:
                if ((not isinstance(value,
                                    (list, np.ndarray)) and pd.isna(value))
                        or len(value) == 0):
                    continue
                if isinstance(value[0], float):
                    return Dtype.floatlist
                if np.issubdtype(type(value[0]), int):
                    return Dtype.intlist
                if isinstance(value[0], str):
                    return Dtype.stringlist
        elif pd.api.types.infer_dtype(ser) in {
                "datetime", "date", "datetime64"
        }:
            return Dtype.date
    if dtype not in _TO_DTYPE:
        raise KeyError(f"Unsupported data type: {dtype}.")
    return _TO_DTYPE[dtype]


def to_pandas_type(dtype: Any) -> Any:
    if dtype not in _TO_DTYPE:
        return None
    kumo_dtype = _TO_DTYPE[dtype]
    return None if kumo_dtype.is_list() else dtype_to_pandas(kumo_dtype)


def pyarrow_to_dtype(dtype: Any) -> Dtype:
    if isinstance(dtype, pa.lib.DataType) and pa.types.is_temporal(dtype):
        if isinstance(dtype, pa.DurationType):
            return Dtype.timedelta
        if pa.types.is_time(dtype):
            return Dtype.time
        return Dtype.date
    if isinstance(dtype, pa.DictionaryType):
        return Dtype.string
    if pa.types.is_list(dtype):
        if pa.types.is_floating(dtype.value_type):
            return Dtype.floatlist
        if pa.types.is_integer(dtype.value_type):
            return Dtype.intlist
        if pa.types.is_string(dtype.value_type):
            return Dtype.stringlist
    kumo_dtype = _PYARROW_TO_DTYPE[dtype]
    return kumo_dtype if not kumo_dtype.is_list() else Dtype.unsupported


def unsupported_pyarrow_composite(dtype: Any) -> bool:
    if pa.types.is_map(dtype) or pa.types.is_struct(dtype):
        return True
    if pa.types.is_list(dtype):
        try:
            pyarrow_to_dtype(dtype)
        except KeyError:
            return True
    return False


def to_pyarrow_type(dtype: Any) -> Any:
    if isinstance(dtype, Dtype):
        return TO_PYARROW_DTYPE[dtype]
    if dtype not in _TO_DTYPE:
        return None
    kumo_dtype = _TO_DTYPE[dtype]
    return None if kumo_dtype.is_list() else TO_PYARROW_DTYPE[kumo_dtype]


def dtype_to_numpy(dtype: Dtype) -> Any:
    mapping = {
        Dtype.bool: np.dtype("bool"),
        Dtype.int: np.dtype("int64"),
        Dtype.byte: np.dtype("byte"),
        Dtype.int16: np.dtype("int16"),
        Dtype.int32: np.dtype("int32"),
        Dtype.int64: np.dtype("int64"),
        Dtype.float: np.dtype("float32"),
        Dtype.float32: np.dtype("float32"),
        Dtype.float64: np.dtype("float64"),
        Dtype.string: np.dtype("object"),
        Dtype.date: np.dtype("datetime64[ns]"),
        Dtype.time: np.dtype("datetime64[ns]"),
        Dtype.timedelta: np.dtype("timedelta64[ns]"),
        Dtype.floatlist: np.dtype("object"),
        Dtype.intlist: np.dtype("object"),
        Dtype.stringlist: np.dtype("object"),
        Dtype.binary: np.dtype("object"),
        Dtype.unsupported: None,
    }
    return mapping[dtype]


def dtype_to_pandas(dtype: Dtype) -> Any:
    mapping = {
        Dtype.bool: pd.BooleanDtype(),
        Dtype.int: pd.Int64Dtype(),
        Dtype.byte: pd.UInt8Dtype(),
        Dtype.int16: pd.Int16Dtype(),
        Dtype.int32: pd.Int32Dtype(),
        Dtype.int64: pd.Int64Dtype(),
        Dtype.float: np.dtype("float32"),
        Dtype.float32: np.dtype("float32"),
        Dtype.float64: np.dtype("float64"),
        Dtype.date: np.dtype("datetime64[ns]"),
        Dtype.time: np.dtype("datetime64[ns]"),
        Dtype.timedelta: np.dtype("timedelta64[ns]"),
        Dtype.string: pd.StringDtype(storage="pyarrow"),
        Dtype.floatlist: np.dtype("object"),
        Dtype.intlist: np.dtype("object"),
        Dtype.stringlist: np.dtype("object"),
        Dtype.binary: pd.StringDtype(),
        Dtype.unsupported: None,
    }
    return mapping[dtype]


def dtype_to_torch(dtype: Dtype) -> Any:
    mapping = {
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
    return mapping[dtype]


__all__ = [
    "DataFrame",
    "PandasDtype",
    "TO_PYARROW_DTYPE",
    "dtype_to_numpy",
    "dtype_to_pandas",
    "dtype_to_torch",
    "pyarrow_to_dtype",
    "to_dtype",
    "to_pandas_type",
    "to_pyarrow_type",
    "unsupported_pyarrow_composite",
]


def can_cast_safely(dtype: Dtype, raw_other_type: Any) -> bool:
    """Whether values of ``dtype`` survive a cast to ``raw_other_type``.

    A free function rather than a :class:`~kumoapi.typing.Dtype` method: it
    accepts a *raw* torch or pyarrow dtype and normalises it through
    :func:`to_dtype`, so it needs the numerical-library imports this module
    carries and the wire contract must not.

    Args:
        dtype: The type the values are held as today.
        raw_other_type: The target, either a :class:`~kumoapi.typing.Dtype` or
            anything :func:`to_dtype` understands.
    """
    if isinstance(raw_other_type, Dtype):
        other_type = raw_other_type
    else:
        other_type = to_dtype(raw_other_type)
    if dtype is Dtype.binary:
        return other_type in {Dtype.binary, Dtype.string}
    if dtype.is_bool():
        return other_type.is_bool()
    if dtype.is_int() or dtype.is_float():
        return other_type.is_numerical()
    if dtype.is_string():
        return other_type is Dtype.string or other_type.is_numerical()
    if dtype.is_temporal() or dtype.is_timestamp():
        return (other_type.is_temporal()
                or other_type.supports_cast_to_temporal())
    if dtype.is_list():
        return dtype is other_type
    return dtype.is_unsupported()
