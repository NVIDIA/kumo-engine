# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Reading a stored Snowflake type, which is many-to-one. The other
direction -- what an unload must produce -- lives in ``kumo_snowflake.sql``.
"""

from typing import Optional, Union

from kumoapi.typing import Dtype

#: Snowflake ``INFORMATION_SCHEMA.COLUMNS.DATA_TYPE`` to Kumo data type;
#: ``NUMBER`` is resolved by scale, not by name, so it is absent here. Every
#: timestamp variant maps to ``time``: reading one as ``date`` would leave the
#: unload without the UTC conversion a ``TIMESTAMP_TZ`` needs to be exportable
#: at all.
_DTYPE_BY_SNOWFLAKE_TYPE = {
    'FLOAT': Dtype.float64,
    'DOUBLE': Dtype.float64,
    'DOUBLE PRECISION': Dtype.float64,
    'REAL': Dtype.float64,
    'TEXT': Dtype.string,
    'VARCHAR': Dtype.string,
    'CHAR': Dtype.string,
    'CHARACTER': Dtype.string,
    'STRING': Dtype.string,
    'BOOLEAN': Dtype.bool,
    'DATE': Dtype.date,
    'TIMESTAMP_NTZ': Dtype.time,
    'TIMESTAMP_LTZ': Dtype.time,
    'TIMESTAMP_TZ': Dtype.time,
    'DATETIME': Dtype.time,
    'BINARY': Dtype.binary,
    'VARBINARY': Dtype.binary,
    # Snowflake refuses `CAST(<time> AS TIMESTAMP_NTZ)` outright, so calling
    # a time of day `time` would build an unload that does not compile.
    'TIME': Dtype.unsupported,
    'VARIANT': Dtype.string,
    'OBJECT': Dtype.unsupported,
    'ARRAY': Dtype.string,
    'VECTOR': Dtype.unsupported,
    'GEOGRAPHY': Dtype.unsupported,
    'GEOMETRY': Dtype.unsupported,
}


def dtype_for(data_type: str,
              numeric_scale: Optional[Union[int, str]] = None) -> Dtype:
    r"""``NUMBER`` covers both integers and decimals, separated by scale: a
    scale of zero is an integer.
    """
    name = data_type.strip().upper()
    if name in ('NUMBER', 'DECIMAL', 'NUMERIC'):
        if numeric_scale in (0, '0', None):
            return Dtype.int64
        return Dtype.float64
    if name in ('INT', 'INTEGER', 'BIGINT', 'SMALLINT', 'TINYINT', 'BYTEINT'):
        return Dtype.int64
    return _DTYPE_BY_SNOWFLAKE_TYPE.get(name, Dtype.unsupported)


#: ``cursor.describe`` names a result column's type differently from
#: ``INFORMATION_SCHEMA``: a ``NUMBER`` is described as ``FIXED`` and a
#: ``FLOAT`` as ``REAL``, neither of which appears in the catalog.
_DESCRIBED_AS = {
    'FIXED': 'NUMBER',
    'REAL': 'FLOAT',
    'TIMESTAMP': 'TIMESTAMP_NTZ',
    'MAP': 'OBJECT',
}


def dtype_for_described(
    data_type: str,
    numeric_scale: Optional[Union[int, str]] = None,
) -> Dtype:
    name = data_type.strip().upper()
    return dtype_for(_DESCRIBED_AS.get(name, name), numeric_scale)
