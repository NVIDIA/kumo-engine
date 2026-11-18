# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Values worked out on the host, in Python, where upstream's Python semantics need them."""

from __future__ import annotations

from test.kumo_pytorch.reference_engine.gpu.data.text import python_texts
from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import Values
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:

    pass
from test.kumo_pytorch.reference_engine.gpu.sampling.rows import _Column


def _values_python(values: Values) -> list[Any]:
    """Sampled values as the Python values upstream's reference holds (None for NULL)."""

    import cupy

    present = values.valid().get().tolist()
    if values.dtype.name.startswith("decimal"):
        from test.kumo_pytorch.reference_engine.gpu.data.decimals import (
            python_decimal,
            words,
        )

        python = [python_decimal(row, values.dtype.scale) for row in words(values).get().tolist()]
    else:
        python = _python(values.dtype, values.texts, cupy.where(values.valid(), values.values, 0))
    return [value if ok else None for value, ok in zip(python, present, strict=True)]


def _column_python(column: _Column) -> list[Any]:
    """A column's values as the Python values upstream's reference holds (None for NULL)."""

    import cupy

    present = column.valid.get().tolist()
    if column.words is not None:
        from test.kumo_pytorch.reference_engine.gpu.data.decimals import (
            python_decimal,
        )

        values = [python_decimal(row, column.dtype.scale) for row in column.words.get().tolist()]
    else:
        values = _python(column.dtype, column.texts, cupy.where(column.valid, column.values, 0))
    return [value if ok else None for value, ok in zip(values, present, strict=True)]


def _python_at(operand: Values, row: int) -> Any:
    """One row's value as the Python value upstream's reference holds (a decimal's exponent its scale)."""

    if operand.dtype.name.startswith("decimal"):
        from test.kumo_pytorch.reference_engine.gpu.data.decimals import (
            python_decimal,
            words,
        )

        return python_decimal(words(operand)[row].get().tolist(), operand.dtype.scale)
    return _python(operand.dtype, operand.texts, operand.values[row:row + 1])[0]


def _python(dtype: Any, texts: Any, values: Any) -> list[Any]:
    """GPU values as the Python values upstream's reference holds."""

    import datetime as dt

    if texts is not None:
        strings = python_texts(texts, dtype)
        return [strings[code] if 0 <= code < len(strings) else None for code in values.get().tolist()]
    numbers = values.get().tolist()
    if dtype.name == "timestamp":
        return [dt.datetime.fromtimestamp(value, dt.UTC) for value in numbers]
    if dtype.name == "date32":
        return [dt.date(1970, 1, 1) + dt.timedelta(days=value) for value in numbers]
    if dtype.name == "bool":
        return [bool(value) for value in numbers]
    return numbers
