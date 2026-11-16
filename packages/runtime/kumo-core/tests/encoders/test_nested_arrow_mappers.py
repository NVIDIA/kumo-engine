# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for Arrow-native nested feature encoding."""

import pyarrow as pa
import pytest
import torch
from kumo_core.encoders.tensor_mapper import (
    MultiCategoricalTensorMapper,
    SequenceTensorMapper,
)
from tests.support.mapper_reference import sequence_pandas_reference
from torch_geometric.testing import withCUDA


@pytest.mark.parametrize("large_list", [False, True])
@pytest.mark.parametrize("start", [0, 1, 3])
@pytest.mark.parametrize(
    "value_type, categories",
    [
        (pa.string(), ("a", "b")),
        (pa.large_string(), ("a", "b")),
        (pa.int64(), (1, 2)),
        (pa.float64(), (1.5, 2.5)),
        (pa.bool_(), (True, False)),
        (pa.binary(), (b"a", b"b")),
    ],
)
@withCUDA
def test_multicategorical_nested_arrow_parity(
    large_list: bool,
    start: int,
    value_type: pa.DataType,
    categories: tuple[str | float | bytes, str | float | bytes],
    device: torch.device,
) -> None:
    """Sliced nested categories preserve missing, empty, and unknown tokens."""
    first, second = categories
    list_type = pa.large_list(value_type) if large_list else pa.list_(value_type)
    array = pa.array(
        [[first, second], [], None, [None, first], [second], []], type=list_type
    ).slice(start, 3)
    mapper = MultiCategoricalTensorMapper([[first, 2], [second, 1]], sep=None)
    expected_offsets, expected_values = mapper.mapper.map_multi(array.to_pandas(), None)

    result = mapper(array, device=device)

    assert result.device == device
    assert result.offset.cpu().equal(expected_offsets)
    assert result.values.cpu().equal(expected_values)


@pytest.mark.parametrize("separator", ["|", "::", r"\s+", r"(,)", r"(?=b)", ""])
@withCUDA
def test_multicategorical_delimited_strings(
    separator: str, device: torch.device
) -> None:
    """String parsing retains empty filtering and Python regex semantics."""
    array = pa.array(["a|b|", "a::b", "a b", "a,b", "ab", "", None])
    mapper = MultiCategoricalTensorMapper([["a", 3], ["b", 2], [",", 1]], sep=separator)
    expected_offsets, expected_values = mapper.mapper.map_multi(
        array.to_pandas(), separator
    )

    result = mapper(array, device=device)

    assert result.offset.cpu().equal(expected_offsets)
    assert result.values.cpu().equal(expected_values)


@pytest.mark.parametrize("fixed_size", [False, True])
def test_multicategorical_string_lists_filter_empty_elements(fixed_size: bool) -> None:
    """Filtering blank elements rebuilds offsets without adding missing tokens."""
    list_type = pa.list_(pa.string(), 2) if fixed_size else pa.list_(pa.string())
    array = pa.array(
        [["ignored", "ignored"], ["a", ""], ["", ""], None], type=list_type
    )
    mapper = MultiCategoricalTensorMapper([["a", 1]], sep=None)

    result = mapper(array.slice(1))

    assert result.offset.tolist() == [0, 1, 1, 2]
    assert result.values.tolist() == [0, -1]
    assert [row.tolist() for row in mapper.backward(result)] == [["a"], [], []]


def test_multicategorical_large_integers_with_null_elements() -> None:
    """Null elements must not round large integer categories through float64."""
    first, second = 2**60 + 1, 2**60 + 2
    array = pa.array([[first, None, second]], type=pa.list_(pa.int64()))
    mapper = MultiCategoricalTensorMapper([[first, 2], [second, 1]], sep=None)

    result = mapper(array)

    assert result.offset.tolist() == [0, 3]
    assert result.values.tolist() == [0, -1, 1]


@pytest.mark.parametrize("fixed_size", [False, True])
def test_null_category_rows_ignore_hidden_values(fixed_size: bool) -> None:
    """Physical values underneath null list slots never become category tokens."""
    values = pa.array(["a", "b", "a"])
    mask = pa.array([False, True, False])
    array = (
        pa.FixedSizeListArray.from_arrays(values, 1, mask=mask)
        if fixed_size
        else pa.ListArray.from_arrays([0, 1, 2, 3], values, mask=mask)
    )
    mapper = MultiCategoricalTensorMapper([["a", 2], ["b", 1]], sep=None)

    result = mapper(array)

    assert result.offset.tolist() == [0, 1, 2, 3]
    assert result.values.tolist() == [0, -1, 0]


@pytest.mark.parametrize("dictionary", [False, True])
@pytest.mark.parametrize("max_length", [0, 2, 4])
@withCUDA
def test_sequence_strings_match_pandas(
    dictionary: bool, max_length: int, device: torch.device
) -> None:
    """String sequences retain parsing, truncation, padding, and reverse format."""
    array = pa.array(["[1, 2, 3]", None, "[]", "[4]", "[nan, inf, -inf]"])
    if dictionary:
        array = array.dictionary_encode()
    mapper = SequenceTensorMapper(0, max_length)
    reference_mapper = SequenceTensorMapper(0, max_length)
    reference = sequence_pandas_reference(
        reference_mapper,
        array.dictionary_decode().to_pandas() if dictionary else array.to_pandas(),
        device=device,
    )

    result = mapper(array, device=device)

    assert result.values.allclose(reference.values, equal_nan=True)
    assert result.offset.equal(reference.offset)
    assert mapper.backward(result).equals(reference_mapper.backward(reference))


@pytest.mark.parametrize("fixed_size", [False, True])
def test_null_sequence_rows_ignore_hidden_values(fixed_size: bool) -> None:
    """Null list rows encode as NaNs even when their child buffers contain values."""
    values = pa.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    mask = pa.array([False, True, False])
    array = (
        pa.FixedSizeListArray.from_arrays(values, 2, mask=mask)
        if fixed_size
        else pa.ListArray.from_arrays([0, 2, 4, 6], values, mask=mask)
    )

    result = SequenceTensorMapper(0, 2)(array.slice(1))

    assert result.values[0].isnan().all()
    assert result.values[1].tolist() == [5.0, 6.0]


def test_sequence_null_only_child_type() -> None:
    """Untyped null list elements become floating-point NaN values."""
    result = SequenceTensorMapper(0, 2)(pa.array([[], [None], None]))

    assert result.values.shape == (3, 2)
    assert result.values.isnan().all()
