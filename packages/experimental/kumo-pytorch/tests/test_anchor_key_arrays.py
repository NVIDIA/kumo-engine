# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Columnar anchor keys retain strict scalar typing at the native boundary."""

import pyarrow as pa
import pytest

from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.engines.parquet_anchors import _anchor_key_array


@pytest.mark.parametrize(
    "name", [f"{sign}int{bits}" for sign in ("", "u") for bits in (8, 16, 32, 64)]
)
def test_typed_integer_keys_keep_bounds_nulls_and_buffers(name):
    dtype = DType(name)
    bits = dtype.to_arrow().bit_width
    minimum = 0 if name.startswith("u") else -(1 << (bits - 1))
    maximum = (1 << (bits if name.startswith("u") else bits - 1)) - 1
    values = pa.array([minimum, maximum, None, minimum], type=dtype.to_arrow())
    assert _anchor_key_array(values, dtype) is values


def test_typed_string_keys_keep_nulls_duplicates_and_unicode():
    values = pa.array(["", None, "é🌱", "é🌱"])
    assert _anchor_key_array(values, DType("string")) is values


@pytest.mark.parametrize("source", [pa.int16(), pa.uint64()])
def test_different_integer_widths_remain_accepted(source):
    actual = _anchor_key_array(pa.array([0, None, 127], type=source), DType("int8"))
    assert actual.equals(pa.array([0, None, 127], type=pa.int8()))


@pytest.mark.parametrize("values", [[True], [1.0], ["1"], [128], [-129]])
def test_integer_keys_do_not_gain_implicit_casts_or_overflow(values):
    with pytest.raises(ValueError, match="anchors.entity_id:.*not a valid"):
        _anchor_key_array(pa.array(values), DType("int8"))


def test_large_string_keys_still_normalize_to_declared_storage():
    actual = _anchor_key_array(pa.array(["é", None], type=pa.large_string()), DType("string"))
    assert actual.equals(pa.array(["é", None], type=pa.string()))
