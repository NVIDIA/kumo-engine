# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The partition of each root key: a hash of its value alone."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:

    pass

__all__ = ['partition_of_keys']


def partition_of_keys(keys: Any, num_partitions: int, dtype: Any = None) -> Any:
    """The partition of each root primary-key value (a ``pylibcudf`` column, typed as the split read the key; text
    as text): ``murmurhash3_x86_32`` of the value (a text's bytes), modulo `num_partitions`, as an INT32 column.
    A key of another type of `dtype` is hashed from a form its value alone decides (`hashed_keys`), so equal
    values meet in one partition whatever width libcudf read them in."""

    import pyarrow as pa
    import pylibcudf as plc

    hashed = plc.hashing.murmurhash3_x86_32(hashed_keys(keys, dtype))
    modulus = plc.Scalar.from_arrow(pa.scalar(num_partitions, pa.uint32()))
    return plc.binaryop.binary_operation(hashed, modulus, plc.binaryop.BinaryOperator.PYMOD,
                                         plc.DataType(plc.TypeId.INT32))


def hashed_keys(keys: Any, dtype: Any = None) -> Any:
    """What a split hashes of key column `keys` (a libcudf table): integers and text (and bytes) as read, which
    splits written before keep; booleans, dates and timestamps as int64; floats as float64 bits, -0.0 as 0.0 (they
    are equal; NaN equals nothing, any partition will do); decimals as their four 64-bit words."""

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        _column,
    )

    import cupy
    import pylibcudf as plc

    if dtype is None or dtype.integer or dtype.name in ("string", "bytes"):
        return plc.Table([keys])
    if dtype.name.startswith("decimal"):
        from test.kumo_pytorch.reference_engine.gpu.data.decimals import (
            column_words,
        )

        held = column_words(keys)
        return plc.Table([_column(cupy.ascontiguousarray(held[:, k])) for k in range(held.shape[1])])
    from test.kumo_pytorch.reference_engine.gpu.data.tables import GpuTable

    values = GpuTable("keys", keys.size(), {"k": keys}, {"k": dtype}).values("k")
    if dtype.name.startswith("float"):
        from test.kumo_pytorch.reference_engine.gpu.data.keys import widened

        numbers = widened(values)  # a float32 exactly (CuPy's own widening flushes subnormals)
        return plc.Table([_column(cupy.where(numbers == 0, cupy.int64(0), numbers.view(cupy.int64)))])
    return plc.Table([_column(values.astype(cupy.int64))])
