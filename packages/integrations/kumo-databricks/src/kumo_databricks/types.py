# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

from kumoapi.typing import Dtype

#: Databricks type name to Kumo data type, for the names that carry no
#: parameters. Only the three element types the portable list dtypes can name
#: are listed, since every other ``ARRAY`` has nothing to be read as.
_DTYPE_BY_DATABRICKS_TYPE = {
    'TINYINT': Dtype.int64,
    'SMALLINT': Dtype.int64,
    'INT': Dtype.int64,
    'BIGINT': Dtype.int64,
    'BOOLEAN': Dtype.bool,
    'FLOAT': Dtype.float,
    'DOUBLE': Dtype.float,
    'STRING': Dtype.string,
    'BINARY': Dtype.binary,
    'DATE': Dtype.date,
    'TIMESTAMP': Dtype.time,
    'TIMESTAMP_NTZ': Dtype.time,
    'ARRAY<FLOAT>': Dtype.floatlist,
    'ARRAY<DOUBLE>': Dtype.floatlist,
    'ARRAY<INT>': Dtype.intlist,
    'ARRAY<BIGINT>': Dtype.intlist,
    'ARRAY<STRING>': Dtype.stringlist,
}

#: A bare DECIMAL means DECIMAL(10, 0).
_DECIMAL = re.compile(r'DECIMAL(?:\s*\(\s*\d+\s*,\s*(\d+)\s*\))?')


def dtype_for(databricks_type: str) -> Dtype:
    name = databricks_type.strip().upper()
    if 'DECIMAL' in name:
        return _decimal_dtype(name)
    if name in _DTYPE_BY_DATABRICKS_TYPE:
        return _DTYPE_BY_DATABRICKS_TYPE[name]
    if 'VARCHAR' in name or 'CHAR' in name:
        return Dtype.string
    return Dtype.unsupported


def _decimal_dtype(name: str) -> Dtype:
    r"""A ``DECIMAL`` is an integer exactly when it has no fractional digits."""
    match = _DECIMAL.fullmatch(name)
    if match is None:
        return Dtype.int64
    scale = match.group(1)
    return Dtype.int64 if not scale or int(scale) == 0 else Dtype.float
