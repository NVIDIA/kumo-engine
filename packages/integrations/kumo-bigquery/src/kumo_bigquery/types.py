# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Iterable, Optional

from kumoapi.typing import Dtype

_REPEATED = 'REPEATED'

#: BigQuery's exact-numeric family. ``DECIMAL`` and ``BIGDECIMAL`` are the
#: standard-SQL aliases of ``NUMERIC`` and ``BIGNUMERIC``.
_DECIMAL_TYPES = {'NUMERIC', 'BIGNUMERIC', 'DECIMAL', 'BIGDECIMAL'}

#: Digits a scale-zero decimal may carry and still be certain to fit int64.
#: ``10**18 - 1`` is below ``i64::MAX`` while some 19-digit values are above
#: it, so 18 is the largest provably safe width.
_INT64_SAFE_PRECISION = 18

#: BigQuery ``TableFieldSchema.type`` to Kumo data type, for a column holding
#: a single value. Reading an instant as ``DATE`` would drop the ordering a
#: graph is built on; BigQuery's ``TIME`` is a time of day rather than an
#: instant, so it is deliberately absent and lands on ``unsupported``.
_DTYPE_BY_BIGQUERY_TYPE = {
    'STRING': Dtype.string,
    'INTEGER': Dtype.int64,
    'INT64': Dtype.int64,
    'FLOAT': Dtype.float,
    'FLOAT64': Dtype.float,
    'BOOLEAN': Dtype.bool,
    'BOOL': Dtype.bool,
    'DATE': Dtype.date,
    'TIMESTAMP': Dtype.time,
    'DATETIME': Dtype.time,
    'BYTES': Dtype.binary,
}

#: The same for a ``REPEATED`` column. Kumo's list dtypes are element-typed,
#: so only the element types it has a list of appear here.
_DTYPE_BY_REPEATED_BIGQUERY_TYPE = {
    'FLOAT': Dtype.floatlist,
    'FLOAT64': Dtype.floatlist,
    'INTEGER': Dtype.intlist,
    'INT64': Dtype.intlist,
    # `NUMERIC` is listed for the mapping this path has always had. The other
    # exact numerics are deliberately absent: BigQuery exports a repeated
    # decimal uncast and DiskGraph reads only Float16/32/64 child arrays.
    'NUMERIC': Dtype.floatlist,
}


def _decimal_dtype(precision: Optional[int], scale: Optional[int]) -> Dtype:
    r"""Read an exact-numeric column as an integer only where BigQuery states
    a scale of zero.

    BigQuery omits ``scale`` from a column's metadata when it is zero, so a
    parameterized ``NUMERIC(29, 0)`` reports ``precision=29, scale=None``.
    ``scale == 0`` is therefore never true, and ``scale is None`` on its own
    cannot separate "scale zero" from "unparameterized" -- a stated precision
    is what tells them apart.

    Scale zero says the values are whole; it does not say they fit. A
    scale-zero ``NUMERIC`` may declare 29 digits and a ``BIGNUMERIC`` 38,
    against int64's 19, so only ``_INT64_SAFE_PRECISION`` digits are provably
    in range and wider columns are read as ``float64``.

    An unparameterized column carries its type's default scale (9 for
    ``NUMERIC``, 38 for ``BIGNUMERIC``) and so is fractional.

    A fractional decimal is read as ``float64``, not ``float``: ``float`` is
    32-bit, and a ``NUMERIC`` at its default scale of 9 does not survive it.
    """
    if (precision is not None and scale is None
            and precision <= _INT64_SAFE_PRECISION):
        return Dtype.int64
    return Dtype.float64


def dtype_for(field_type: str, mode: Optional[str] = None,
              precision: Optional[int] = None,
              scale: Optional[int] = None) -> Dtype:
    r"""``mode`` separates a list from a single value of the same element
    type -- the one thing the type name alone does not say. ``precision`` and
    ``scale`` are read only for the exact-numeric types, whose Kumo dtype
    depends on the declared scale.
    """
    name = str(field_type).strip().upper()
    mode = str(mode).strip().upper() if mode is not None else None
    if mode == _REPEATED:
        if name == 'STRING':
            # Whether a list of strings is representable is the engine's
            # decision, since it is how a list-distinct target is stored.
            from kumo_core.training.options import allow_array_targets
            return (Dtype.stringlist
                    if allow_array_targets() else Dtype.unsupported)
        return _DTYPE_BY_REPEATED_BIGQUERY_TYPE.get(name, Dtype.unsupported)
    if name in _DECIMAL_TYPES:
        return _decimal_dtype(precision, scale)
    return _DTYPE_BY_BIGQUERY_TYPE.get(name, Dtype.unsupported)


def schema_columns(schema: Iterable[Any]) -> list[tuple[str, Dtype]]:
    return [(field.name,
             dtype_for(field.field_type, field.mode, field.precision,
                       field.scale)) for field in schema]
