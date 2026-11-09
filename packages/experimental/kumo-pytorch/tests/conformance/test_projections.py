# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoded assertions must notice changed IDs, types, shapes and boundaries."""

from datetime import date
from decimal import Decimal

import pyarrow as pa
import pytest
import torch

from conformance.assertions import assert_projection
from conformance.fixtures import scalar


def test_complete_integer_buffer_is_asserted():
    assert_projection(
        {"value_ids": torch.tensor([2, 3], dtype=torch.int64)},
        {"value_ids": {"dtype": "int64", "shape": [2], "values": [2, 3]}},
    )


@pytest.mark.parametrize(
    "actual",
    (
        torch.tensor([3, 2], dtype=torch.int64),
        torch.tensor([2, 3], dtype=torch.int32),
        torch.tensor([2.0, 3.0], dtype=torch.float64),
        torch.tensor([[2, 3]], dtype=torch.int64),
        torch.tensor([2], dtype=torch.int64),
        [2, 3],
    ),
)
def test_wrong_id_dtype_shape_or_untyped_array_fails(actual):
    with pytest.raises(AssertionError):
        assert_projection(actual, {"dtype": "int64", "shape": [2], "values": [2, 3]})


def test_missing_requested_buffer_fails():
    with pytest.raises(AssertionError, match="missing requested field"):
        assert_projection({}, {"offsets": {"dtype": "int64", "shape": [3], "values": [0, 2, 3]}})


@pytest.mark.parametrize("actual", ([0, 1, 3], [0, 2, 4], [0, 2]))
def test_shifted_or_truncated_offsets_fail(actual):
    with pytest.raises(AssertionError):
        assert_projection(
            torch.tensor(actual), {"dtype": "int64", "shape": [3], "values": [0, 2, 3]}
        )


def test_integer_ids_cannot_use_float_answers():
    with pytest.raises(AssertionError, match="expected type float"):
        assert_projection(torch.tensor([2]), {"dtype": "int64", "shape": [1], "values": [2.0]})


def test_narrow_tolerance_is_explicit_and_float_only():
    assert_projection(
        torch.tensor([1.0 + 1e-8], dtype=torch.float64),
        {"dtype": "float64", "shape": [1], "values": [1.0], "atol": 1e-7},
    )
    with pytest.raises(AssertionError):
        assert_projection(
            torch.tensor([1.001], dtype=torch.float64),
            {"dtype": "float64", "shape": [1], "values": [1.0], "atol": 1e-7},
        )


def test_typed_arrow_null_and_decimal_are_exact():
    answer = {"dtype": "decimal128(4, 2)", "shape": [2], "values": [{"$decimal": "1.10"}, None]}
    assert_projection(pa.array([Decimal("1.10"), None], type=pa.decimal128(4, 2)), answer)
    with pytest.raises(AssertionError):
        assert_projection(
            pa.array([Decimal("1.10"), Decimal("0.00")], type=pa.decimal128(4, 2)), answer
        )


@pytest.mark.parametrize(
    "value,answer",
    (
        ({"$bytes": "AAE="}, b"\x00\x01"),
        ({"$decimal": "1.10"}, Decimal("1.10")),
        ({"$date": "1970-01-02"}, date(1970, 1, 2)),
    ),
)
def test_fixture_tags(value, answer):
    assert scalar(value) == answer
    assert type(scalar(value)) is type(answer)


@pytest.mark.parametrize(
    "value",
    (
        {"$wat": "x"},
        {"$bytes": 2},
        {"$bytes": "!"},
        {"$date": "not-a-date"},
        {"$decimal": "1", "$bytes": "AA=="},
    ),
)
def test_invalid_fixture_tags_fail(value):
    with pytest.raises((ValueError, TypeError)):
        scalar(value)
