# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target conversion allocates owned buffers and fills them from Arrow chunks."""

from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pytest
import torch
from kumo_core.diskgraph.runtime_adapter import _shape_target
from kumo_core.encoders.target.categorical import CategoricalTargetEncoder
from kumo_core.encoders.target.multicategorical import MultiCategoricalTargetEncoder
from kumo_core.encoders.tensor_mapper import (
    CategoricalTensorMapper,
    MultiCategoricalTensorMapper,
)
from kumoapi.task import TaskType
from torch import Tensor
from torch_frame.data import MultiNestedTensor


def _chunks(array: pa.Array) -> pa.ChunkedArray:
    """Split an array into sliced and empty chunks without copying its buffers."""
    return pa.chunked_array(
        [array.slice(0, 0), array.slice(0, 1), array.slice(1, 0), array.slice(1)]
    )


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("task_type", [None, "regression"])
def test_numeric_target_owns_destination(chunked: bool, task_type: str | None) -> None:
    """Numerical targets copy each chunk into a single float destination."""
    array = pa.array([1.0, None, 3.0], type=pa.float32())
    source = _chunks(array) if chunked else array

    with patch("torch.empty", wraps=torch.empty) as allocate:
        result = _shape_target(source, task_type)

    assert isinstance(result, Tensor)
    expected = torch.tensor([1.0, float("nan"), 3.0])
    if task_type is not None:
        expected = expected.view(-1, 1)
    torch.testing.assert_close(result, expected, equal_nan=True)
    assert allocate.call_count == 1
    result.fill_(0)
    assert source.to_pylist() == [1.0, None, 3.0]


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize(
    "task_type",
    ["static_link_prediction", "temporal_link_prediction", "multiclass_classification"],
)
def test_integer_target_preserves_precision(chunked: bool, task_type: str) -> None:
    """Node IDs and unencoded class IDs do not pass through floating point."""
    values = [2**54 + 1, 2**54 + 3]
    array = pa.array(values, type=pa.int64())

    result = _shape_target(_chunks(array) if chunked else array, task_type)

    assert isinstance(result, Tensor)
    assert result.dtype == torch.long
    assert result.view(-1).tolist() == values
    result.fill_(0)
    assert array.to_pylist() == values


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize(
    "values,categories,expected",
    [
        (["silver", "gold", "silver"], ["silver", "gold"], [0, 1, 0]),
        (["1", "2", "1"], [2, 1], [1, 0, 1]),
        ([1, 2, 1], ["2", "1"], [1, 0, 1]),
        (["1.0", "2.0"], [2, 1], [1, 0]),
        ([True, False], ["False", "True"], [1, 0]),
        ([None, "a"], ["a", "None"], [1, 0]),
    ],
)
def test_categorical_target_chunk_mapping(
    chunked: bool,
    values: list[object],
    categories: list[object],
    expected: list[int],
) -> None:
    """Chunked label lookup retains category order and legacy coercion rules."""
    encoder = CategoricalTargetEncoder(
        TaskType.MULTICLASS_CLASSIFICATION,
        CategoricalTensorMapper([[category, 1] for category in categories]),
    )
    array = pa.array(values)

    with (
        patch.object(
            pd.Index, "get_indexer", side_effect=AssertionError("pandas lookup")
        ),
        patch("torch.empty", wraps=torch.empty) as allocate,
    ):
        result = _shape_target(
            _chunks(array) if chunked else array,
            "multiclass_classification",
            encoder,
        )

    assert isinstance(result, Tensor)
    assert result.view(-1).tolist() == expected
    assert allocate.call_count == 1


@pytest.mark.parametrize("value", ["unknown", "01", "+1", " 1", "1 ", "inf", "NaN"])
def test_unknown_categorical_target_raises(value: str) -> None:
    """Unknown or noncanonical numeric labels never become valid class indices."""
    encoder = CategoricalTargetEncoder(
        TaskType.MULTICLASS_CLASSIFICATION, CategoricalTensorMapper([[1, 1]])
    )

    with pytest.raises(ValueError, match="categorical target contains labels"):
        _shape_target(
            _chunks(pa.array(["1", value])), "multiclass_classification", encoder
        )


def test_categorical_target_dictionary_chunks() -> None:
    """Separate dictionaries resolve into the same target vocabulary."""
    target = pa.chunked_array(
        [
            pa.array(["a", "b"]).dictionary_encode(),
            pa.array(["b", "a"]).dictionary_encode(),
        ]
    )
    encoder = CategoricalTargetEncoder(
        TaskType.MULTICLASS_CLASSIFICATION,
        CategoricalTensorMapper([["a", 2], ["b", 1]]),
    )

    result = _shape_target(target, "multiclass_classification", encoder)

    assert isinstance(result, Tensor)
    assert result.view(-1).tolist() == [0, 1, 1, 0]


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize(
    "task_type", ["multilabel_classification", "multilabel_ranking"]
)
@pytest.mark.parametrize(
    "array,sep,expected",
    [
        (pa.array([None, "1|2", "", "2||1"]), "|", ([1, 0, 0, 1], [0, 0, 2, 2, 4])),
        (pa.array([None, "1||2", "", "2||||1"]), "||", ([1, 0, 0, 1], [0, 0, 2, 2, 4])),
        (
            pa.array([None, ["1", None], [], ["2", "1"]]),
            None,
            ([1, 0, 1], [0, 0, 1, 1, 3]),
        ),
        (pa.array([None, [1, None], [], [2, 1]]), None, ([1, 0, 1], [0, 0, 1, 1, 3])),
    ],
)
def test_multilabel_target_chunk_offsets(
    chunked: bool,
    task_type: str,
    array: pa.Array,
    sep: str | None,
    expected: tuple[list[int], list[int]],
) -> None:
    """Ragged targets fill exact values and offsets while discarding missing labels."""
    encoder = MultiCategoricalTargetEncoder(
        MultiCategoricalTensorMapper([[2, 2], [1, 1]], sep)
    )

    with (
        patch.object(
            pd.Index, "get_indexer", side_effect=AssertionError("pandas lookup")
        ),
        patch("torch.empty", wraps=torch.empty) as allocate,
    ):
        result = _shape_target(_chunks(array) if chunked else array, task_type, encoder)

    assert isinstance(result, MultiNestedTensor)
    assert result.values.tolist() == expected[0]
    assert result.offset.tolist() == expected[1]
    assert allocate.call_count == 2
    original = array.to_pylist()
    result.values.fill_(0)
    assert array.to_pylist() == original


@pytest.mark.parametrize(
    "list_type",
    [pa.list_(pa.int64()), pa.large_list(pa.int64()), pa.list_(pa.int64(), 2)],
)
def test_sliced_multilabel_target(list_type: pa.DataType) -> None:
    """Visible list rows, child nulls, and duplicates retain their original order."""
    array = pa.array([[9, 9], [1, None], [2, 2], [9, 9]], type=list_type).slice(1, 2)
    encoder = MultiCategoricalTargetEncoder(
        MultiCategoricalTensorMapper([[2, 2], [1, 1]], None)
    )

    result = _shape_target(_chunks(array), "multilabel_classification", encoder)

    assert isinstance(result, MultiNestedTensor)
    assert result.values.tolist() == [1, 0, 0]
    assert result.offset.tolist() == [0, 1, 3]


def test_unknown_multilabel_target_raises() -> None:
    """Missing target classes remain an error rather than an unknown token."""
    encoder = MultiCategoricalTargetEncoder(
        MultiCategoricalTensorMapper([["a", 1]], "|")
    )

    with pytest.raises(ValueError, match="multilabel target contains labels.*unknown"):
        _shape_target(
            _chunks(pa.array(["a", "unknown"])), "multilabel_classification", encoder
        )


@pytest.mark.parametrize("chunked", [False, True])
def test_multilabel_target_without_encoder(chunked: bool) -> None:
    """Already-indexed target lists copy into integer buffers with empty null rows."""
    array = pa.array([[2.0, 1.0], None, [], [3.0, None]])

    result = _shape_target(
        _chunks(array) if chunked else array, "multilabel_classification"
    )

    assert isinstance(result, MultiNestedTensor)
    assert result.values.tolist() == [2, 1, 3]
    assert result.offset.tolist() == [0, 2, 2, 2, 3]


@pytest.mark.parametrize(
    "array,expected",
    [
        (pa.array(["+1| 2|3 |1_0"]), [1, 2, 3, 10]),
        (pa.array([[Decimal("1.5"), Decimal("-2.5")]]), [1, -2]),
    ],
)
def test_preindexed_multilabel_integer_coercion(
    array: pa.Array, expected: list[int]
) -> None:
    """Already-indexed targets retain Python integer parsing and truncation rules."""
    result = _shape_target(_chunks(array), "multilabel_classification")

    assert isinstance(result, MultiNestedTensor)
    assert result.values.tolist() == expected
    assert result.offset.tolist() == [0, len(expected)]


@pytest.mark.parametrize(
    "list_type",
    [pa.list_(pa.float32()), pa.large_list(pa.float32()), pa.list_(pa.float32(), 2)],
)
def test_rectangular_target_chunks(list_type: pa.DataType) -> None:
    """Numerical list targets copy visible children directly into a matrix."""
    array = pa.array([[9.0, 9.0], [1.0, 2.0], [3.0, 4.0]], type=list_type).slice(1)

    with patch("torch.empty", wraps=torch.empty) as allocate:
        result = _shape_target(_chunks(array), "regression")

    assert isinstance(result, Tensor)
    torch.testing.assert_close(result, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert allocate.call_count == 1
    result.fill_(0)
    assert array.to_pylist() == [[1.0, 2.0], [3.0, 4.0]]


def test_ragged_numeric_target_raises() -> None:
    """A ragged numeric target cannot silently truncate or pad labels."""
    with pytest.raises(ValueError, match="equal lengths"):
        _shape_target(_chunks(pa.array([[1.0], [2.0, 3.0]])), "regression")


@pytest.mark.parametrize(
    "task_type",
    ["regression", "multiclass_classification", "multilabel_classification"],
)
def test_no_chunk_target(task_type: str) -> None:
    """Zero-chunk Arrow inputs retain valid empty destination shapes."""
    result = _shape_target(pa.chunked_array([], type=pa.int64()), task_type)

    if isinstance(result, MultiNestedTensor):
        assert result.num_rows == 0
        assert result.values.numel() == 0
        assert result.offset.tolist() == [0]
    else:
        assert result.shape == (0, 1)
