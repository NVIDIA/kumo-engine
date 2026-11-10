# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chunk preservation and destination ownership for raw feature encoding."""

from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pytest
import torch
from kumo_core.encoders import tensor_mapper as tm
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_raw_feature_mappers_preserve_chunks(
    empty: bool, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Encode original chunks without concatenation or aliasing input storage."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    cases: list[tuple[tm.TensorMapper, pa.Array]] = [
        (tm.NumericalTensorMapper(), pa.array([1, None, 3, 4])),
        (tm.CategoricalTensorMapper([["a", 2]]), pa.array(["a", None, "z", "a"])),
        (
            tm.MultiCategoricalTensorMapper([["a", 2]], None),
            pa.array([["a", ""], [], None, ["z", "a"]]),
        ),
        (tm.SequenceTensorMapper(0, 2), pa.array([[1.0], [], None, [2.0, 3.0, 4.0]])),
        (
            tm.TimestampTensorMapper(),
            pa.array(["2000-01-01", None, "2000-01-03", "1999-12-31"]),
        ),
        (tm.ImageTensorMapper(), pa.array(["a", None, "b", "c"])),
        (tm.UnsupportedTensorMapper(), pa.array([1, None, 3, 4])),
    ]
    chunks_spy = Mock(wraps=tm._arrow_chunks)
    monkeypatch.setattr(tm, "_arrow_chunks", chunks_spy)
    for mapper, source in cases:
        if empty:
            source = source.slice(0, 0)
        original = source.to_pylist()
        expected = mapper(source, device=torch.device(device))
        chunks_spy.reset_mock()
        chunks = pa.chunked_array(
            [source.slice(0, 1), source.slice(0, 0), source.slice(1)]
        )

        actual = mapper(chunks, device=torch.device(device))

        if mapper.stype not in {tm.Stype.image, tm.Stype.unsupported}:
            chunks_spy.assert_called_once_with(chunks)
        if isinstance(actual, (MultiNestedTensor, MultiEmbeddingTensor)):
            assert actual.offset.equal(expected.offset)
            torch.testing.assert_close(actual.values, expected.values, equal_nan=True)
            actual.values.fill_(0)
            actual.offset.fill_(0)
        else:
            torch.testing.assert_close(actual, expected, equal_nan=True)
            actual.fill_(0)
        assert source.to_pylist() == original


def test_categorical_chunk_dictionaries_are_independent() -> None:
    """Dictionary indices refer to each chunk's own dictionary."""
    source = pa.chunked_array(
        [
            pa.array(["a", "b"]).dictionary_encode(),
            pa.array(["b", "a"]).dictionary_encode(),
        ]
    )

    result = tm.CategoricalTensorMapper([["a", 2], ["b", 1]])(source)

    assert result.flatten().tolist() == [0, 1, 1, 0]


def test_text_tokenizes_original_chunks() -> None:
    """Tokenize each chunk once and shift pointers into owned final storage."""
    model = Mock()
    pointers = [torch.tensor([0, 1, 3]), torch.tensor([0]), torch.tensor([0, 0, 1])]
    tokens = [
        torch.tensor([2, 3, 4]),
        torch.empty(0, dtype=torch.long),
        torch.tensor([5]),
    ]
    model.tokenize.side_effect = list(zip(pointers, tokens))
    source = pa.chunked_array([["one", "two words"], [], [None, "last"]])
    mapper = tm.TextTensorMapper(model)

    result = mapper(source)

    assert model.tokenize.call_count == 3
    for call, chunk in zip(model.tokenize.call_args_list, source.chunks):
        assert call.args[0].equals(chunk)
    assert result.offset.tolist() == [0, 1, 3, 3, 4]
    assert result.values.tolist() == [2, 3, 4, 5]
    result.offset.fill_(0)
    result.values.fill_(0)
    assert pointers[0].tolist() == [0, 1, 3]
    assert tokens[0].tolist() == [2, 3, 4]


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_duration_chunk_conversion_preserves_rounding(unit: str) -> None:
    """Divide duration ticks before the destination dtype conversion."""
    source = pa.array([123456789012345, None, -123456789012345], type=pa.duration(unit))
    scale = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}[unit]
    expected = (source.cast(pa.int64()).to_numpy(zero_copy_only=False) / scale).astype(
        np.float32
    )

    result = tm.NumericalTensorMapper()(pa.chunked_array([source[:1], source[1:]]))

    np.testing.assert_array_equal(result[:, 0].numpy(), expected)


@pytest.mark.parametrize(
    "mapper", [tm.NumericalTensorMapper(), tm.SequenceTensorMapper(0, 2)]
)
def test_zero_chunk_input(mapper: tm.TensorMapper) -> None:
    """A typed column with no chunks still has the correct output shape."""
    arrow_type = (
        pa.float32()
        if isinstance(mapper, tm.NumericalTensorMapper)
        else pa.list_(pa.float32())
    )

    result = mapper(pa.chunked_array([], type=arrow_type))

    values = result.values if isinstance(result, MultiEmbeddingTensor) else result
    assert values.shape == (0, 1 if isinstance(mapper, tm.NumericalTensorMapper) else 2)


@pytest.mark.parametrize(
    "mapper,source",
    [
        (tm.NumericalTensorMapper(), pa.array([1.0, None, 3.0])),
        (tm.CategoricalTensorMapper([["a", 2]]), pa.array(["a", None, "z"])),
        (tm.SequenceTensorMapper(2, 2), pa.array([[1.0, 2.0]] * 3)),
        (tm.TimestampTensorMapper(), pa.array(["2000-01-01", None, "2000-01-03"])),
    ],
)
def test_chunks_write_into_final_storage(
    mapper: tm.TensorMapper, source: pa.Array, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each chunk copies Arrow views into the same final tensor allocation."""
    copy_spy = Mock(wraps=tm._copy_numpy)
    monkeypatch.setattr(tm, "_copy_numpy", copy_spy)

    result = mapper(pa.chunked_array([source[:1], source[1:]]))

    values = result.values if isinstance(result, MultiEmbeddingTensor) else result
    assert copy_spy.call_count == 2
    for call in copy_spy.call_args_list:
        assert (
            call.args[0].untyped_storage().data_ptr()
            == values.untyped_storage().data_ptr()
        )
