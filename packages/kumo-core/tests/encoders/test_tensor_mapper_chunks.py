# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chunked encoding writes final buffers without consolidating Arrow inputs."""

from unittest.mock import patch

import pyarrow as pa
import pytest
import torch
from kumo_core.encoders.tensor_frame import combine_feats, get_tf
from kumo_core.encoders.tensor_mapper import (
    CategoricalTensorMapper,
    ImageTensorMapper,
    MultiCategoricalTensorMapper,
    NumericalTensorMapper,
    SequenceTensorMapper,
    TensorMapper,
    TextTensorMapper,
    TimestampTensorMapper,
    UnsupportedTensorMapper,
)
from kumo_core.types.features import TensorData
from kumoapi.typing import Stype
from tests.support.embedding_models import TinyModel
from torch import Tensor
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor


def _assert_same(actual: TensorData, expected: TensorData) -> None:
    """Compare tensor values, offsets, and dimensions, including missing values."""
    assert type(actual) is type(expected)
    assert actual.shape == expected.shape
    if isinstance(actual, Tensor):
        assert isinstance(expected, Tensor)
        torch.testing.assert_close(actual, expected, equal_nan=True)
    else:
        assert isinstance(expected, (MultiEmbeddingTensor, MultiNestedTensor))
        torch.testing.assert_close(actual.offset, expected.offset)
        torch.testing.assert_close(actual.values, expected.values, equal_nan=True)


@pytest.mark.parametrize(
    "mapper,array",
    [
        (NumericalTensorMapper(), pa.array([None, 1.0, 2.0, 3.0])),
        (NumericalTensorMapper(), pa.array([None, 1, 2, 3], pa.duration("ms"))),
        (CategoricalTensorMapper([["a", 2]]), pa.array([None, "a", "b", "a"])),
        (
            MultiCategoricalTensorMapper([["a", 2], ["b", 1]], None),
            pa.array([None, ["a", ""], [], ["b", None]]),
        ),
        (
            MultiCategoricalTensorMapper([["a", 2]], "|"),
            pa.array([None, "a|b", "", "a"]),
        ),
        (SequenceTensorMapper(0, 3), pa.array([None, [1.0], [], [2.0, 3.0]])),
        (SequenceTensorMapper(0, 3), pa.array([None, "[1]", "[]", "[2, 3]"])),
        (
            TimestampTensorMapper(),
            pa.array([None, "2020-01-01", "bad", "2021-02-03"]),
        ),
        (TextTensorMapper(TinyModel(10)), pa.array([None, "the of", "", "of"])),
        (UnsupportedTensorMapper(), pa.nulls(4)),
        (ImageTensorMapper(), pa.nulls(4)),
    ],
)
def test_mapper_preserves_chunks(mapper: TensorMapper, array: pa.Array) -> None:
    """Each nonempty visible chunk reaches the mapper with its buffers intact."""
    chunked = pa.chunked_array(
        [array.slice(0, 0), array.slice(0, 1), array.slice(1, 2), array.slice(3)]
    )
    expected = mapper.forward(array)

    with patch.object(mapper, "forward", wraps=mapper.forward) as forward:
        actual = mapper(chunked, device=torch.device("cpu"))

    _assert_same(actual, expected)
    inputs = [call.args[0] for call in forward.call_args_list]
    assert [len(chunk) for chunk in inputs] == [1, 2, 1]
    for actual_chunk, original in zip(inputs, chunked.chunks[1:], strict=True):
        assert actual_chunk.offset == original.offset
        assert actual_chunk.buffers() == original.buffers()
        assert [
            buffer.address if buffer else None for buffer in actual_chunk.buffers()
        ] == [buffer.address if buffer else None for buffer in original.buffers()]
    if isinstance(actual, Tensor):
        actual.fill_(0)
    else:
        actual.values.fill_(0)
    assert chunked.to_pylist() == array.to_pylist()


def test_dictionary_chunks_use_their_own_vocabularies() -> None:
    """Dictionary indices from separate chunks never share a dictionary implicitly."""
    chunks = [
        pa.array(["a", None, "b"]).dictionary_encode(),
        pa.array(["b", "a", "c"]).dictionary_encode(),
    ]
    mapper = CategoricalTensorMapper([["a", 2], ["b", 1]])

    result = mapper(pa.chunked_array(chunks))

    assert result.view(-1).tolist() == [0, -1, 1, 1, 0, -1]


@pytest.mark.parametrize("has_empty_chunks", [False, True])
def test_empty_chunked_columns(has_empty_chunks: bool) -> None:
    """Empty columns retain dense, ragged, and embedding output schemas."""
    cases = [
        (NumericalTensorMapper(), pa.float64()),
        (CategoricalTensorMapper([]), pa.string()),
        (MultiCategoricalTensorMapper([], None), pa.list_(pa.string())),
        (SequenceTensorMapper(0, 2), pa.list_(pa.float32())),
        (TimestampTensorMapper(), pa.timestamp("ns")),
        (TextTensorMapper(TinyModel(10)), pa.string()),
    ]
    for mapper, arrow_type in cases:
        empty = pa.array([], type=arrow_type)
        column = pa.chunked_array(
            [empty, empty] if has_empty_chunks else [], arrow_type
        )

        _assert_same(mapper(column), mapper(empty))


@pytest.mark.parametrize("as_record_batch", [False, True])
def test_frame_writes_final_buffers(as_record_batch: bool) -> None:
    """Differently chunked columns write scalar, embedding, and row-major ragged data."""
    arrays = {
        "num_a": pa.array([None, 1.0, 2.0, 3.0]),
        "num_b": pa.array([4.0, 5.0, None, 6.0]),
        "multi_a": pa.array([["a"], None, [], ["b", "a"]]),
        "multi_b": pa.array([[], ["b", "b"], ["a"], None]),
        "seq_a": pa.array([[1.0], None, [], [2.0, 3.0]]),
        "seq_b": pa.array([[], [4.0, 5.0, 6.0], None, [7.0]]),
        "time_a": pa.array([None, "2020-01-01", "bad", "2021-02-03"]),
        "time_b": pa.array(["2020-01-01", None, "2021-02-03", "bad"]),
    }
    mappers: dict[str, TensorMapper] = {
        "num_a": NumericalTensorMapper(),
        "num_b": NumericalTensorMapper(),
        "multi_a": MultiCategoricalTensorMapper([["a", 2], ["b", 1]], None),
        "multi_b": MultiCategoricalTensorMapper([["a", 2], ["b", 1]], None),
        "seq_a": SequenceTensorMapper(0, 2),
        "seq_b": SequenceTensorMapper(0, 3),
        "time_a": TimestampTensorMapper(),
        "time_b": TimestampTensorMapper(),
    }
    expected = combine_feats(
        {
            stype: [
                mapper(arrays[name])
                for name, mapper in mappers.items()
                if mapper.stype == stype
            ]
            for stype in {mapper.stype for mapper in mappers.values()}
        }
    )
    table = pa.table(
        {
            name: pa.chunked_array([array.slice(0, split), array.slice(split)])
            for (name, array), split in zip(
                arrays.items(), [1, 3, 2, 1, 3, 2, 1, 3], strict=True
            )
        }
    )
    source = pa.record_batch(arrays) if as_record_batch else table

    with patch(
        "kumo_core.encoders.tensor_frame.combine_feats",
        side_effect=AssertionError("Encoding must write the final buffers directly"),
    ):
        result = get_tf(source, mappers)

    assert result.num_rows == 4
    for stype, features in expected.items():
        _assert_same(result.feat_dict[stype], features)
    assert result.col_names_dict[Stype.sequence] == ["seq_a", "seq_b"]
