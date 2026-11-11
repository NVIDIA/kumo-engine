# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve categorical identity and strict calendar parsing with Arrow."""

from datetime import datetime

import pyarrow as pa
import pytest
import torch
from kumo_core.encoders.tensor_mapper import (
    CategoricalTensorMapper,
    MultiCategoricalTensorMapper,
    TimestampTensorMapper,
)
from torch_frame.data import MultiNestedTensor


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("dictionary", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "case",
    [
        ([["1", 2], ["2", 1]], pa.array([1, 2, None]), [-1, -1, -1]),
        ([["1", 2], ["2", 1]], pa.array([1.0, 2.0, None]), [-1, -1, -1]),
        ([[1, 2], [2, 1]], pa.array(["1", "2", None]), [-1, -1, -1]),
        ([["true", 2]], pa.array([True, False, None]), [-1, -1, -1]),
        ([[1, 2]], pa.array([True, False, None]), [-1, -1, -1]),
        ([[True, 2]], pa.array([1, 0, None]), [-1, -1, -1]),
        ([[b"1", 2]], pa.array(["1", "2", None]), [-1, -1, -1]),
        ([["1", 2]], pa.array([b"1", b"2", None]), [-1, -1, -1]),
        ([[1, 2], [2, 1]], pa.array([1.0, 2.0, None]), [0, 1, -1]),
        ([[1.0, 2], [2.0, 1]], pa.array([1, 2, None]), [0, 1, -1]),
        (
            [["1", 2], ["2", 1]],
            pa.array(["1", "2", None], type=pa.large_string()),
            [0, 1, -1],
        ),
        (
            [[datetime(2020, 1, 1), 2]],
            pa.array([datetime(2020, 1, 1), None], type=pa.timestamp("us")),
            [0, -1],
        ),
    ],
)
def test_category_type_compatibility(
    case: tuple[list[list[object]], pa.Array, list[int]],
    nested: bool,
    dictionary: bool,
    chunked: bool,
    device: str,
) -> None:
    """Unknown types remain unknown, while numeric and temporal units compare."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    counts, array, expected = case
    if dictionary:
        array = array.dictionary_encode()
    if nested:
        array = pa.ListArray.from_arrays(pa.array(range(len(array) + 1)), array)
    mapper = (
        MultiCategoricalTensorMapper(counts, sep=None)
        if nested
        else CategoricalTensorMapper(counts)
    )
    source = pa.chunked_array([array[:1], array[1:]]) if chunked else array

    result = mapper(source, device=torch.device(device))

    codes = result.values if isinstance(result, MultiNestedTensor) else result
    assert codes.flatten().tolist() == expected


@pytest.mark.parametrize("dictionary", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "format,values",
    [
        (
            "%Y-%m-%d",
            ["2020-02-29", "2021-02-29", "2020-02-30", "2021-04-31", None, "2020-2-29"],
        ),
        (
            "%d/%m/%Y",
            ["29/02/2020", "29/02/2021", "30/02/2020", "31/04/2021", None, "29/2/2020"],
        ),
        (
            "%Y-%m-%d %H:%M:%S%z",
            [
                "2020-02-29 01:00:00+0100",
                "2021-02-29 01:00:00+0100",
                "2020-02-30 01:00:00+0100",
                "2021-04-31 01:00:00+0100",
                None,
                "2020-2-29 01:00:00+0100",
            ],
        ),
    ],
)
def test_explicit_formats_reject_invalid_calendar_dates(
    format: str,
    values: list[str | None],
    dictionary: bool,
    chunked: bool,
    device: str,
) -> None:
    """Invalid dates stay missing; leap days and unpadded valid dates survive."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    array = pa.array(values)
    if dictionary:
        array = array.dictionary_encode()
    source = pa.chunked_array([array[:2], array[2:]]) if chunked else array

    result = TimestampTensorMapper(format)(source, device=torch.device(device))

    assert result[1:5].eq(-1).all().item()
    assert result[[0, 5], TimestampTensorMapper.index["unix"]].tolist() == [
        1_582_934_400,
        1_582_934_400,
    ]
