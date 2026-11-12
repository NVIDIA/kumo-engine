# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for Arrow text tokenization."""

from unittest.mock import Mock

import pandas as pd
import pyarrow as pa
import pytest
import torch
from kumo_core.embedding_models import TokenEmbeddingModel, resolve_encoder
from kumo_core.encoders.tensor_mapper import TextTensorMapper
from kumo_core.glove import VocabularyIndex
from kumoapi.encoder import TextEmbedding


@pytest.mark.parametrize("arrow_type", [pa.string(), pa.large_string()])
@pytest.mark.parametrize(
    ("texts", "offsets", "values"),
    [
        ([], [0], []),
        ([None, None], [0, 0, 0], []),
        (["", "  ", "unknown"], [0, 0, 0, 0], []),
        (
            ["THE,of!", None, "to<br />a", "a\nand\tthe", "to:of;the"],
            [0, 2, 2, 4, 7, 10],
            [0, 3, 4, 7, 7, 5, 0, 4, 3, 0],
        ),
        (
            ["the\u00a0of", "to\u2003a", "in\x1cthe", "the<br\u3000/>of"],
            [0, 2, 4, 6, 8],
            [0, 3, 4, 7, 6, 0, 0, 3],
        ),
    ],
)
def test_arrow_tokenization(
    arrow_type: pa.DataType,
    texts: list[str | None],
    offsets: list[int],
    values: list[int],
) -> None:
    """Preserve normalization, unknown-word filtering, and empty row offsets."""
    model = resolve_encoder(TextEmbedding("glove.test-10d"))
    array = pa.array(texts, type=arrow_type)

    result = TextTensorMapper(model)(array)

    assert result.num_rows == len(texts)
    assert result.offset.tolist() == offsets
    assert result.values.tolist() == values


@pytest.mark.parametrize("layout", ["sliced", "chunked", "dictionary", "null"])
def test_arrow_text_layouts(layout: str) -> None:
    """Handle Arrow layouts at the mapper's public boundary."""
    array = pa.array(["unknown", "the of", None, "to", "unknown"])
    offsets = [0, 0, 2, 2, 3, 3]
    values = [0, 3, 4]
    if layout == "sliced":
        array = array.slice(1, 3)
        offsets = [0, 2, 2, 3]
    elif layout == "chunked":
        array = pa.chunked_array([array.slice(0, 2), array.slice(2)])
    elif layout == "dictionary":
        array = array.dictionary_encode()
    else:
        array = pa.nulls(3)
        offsets, values = [0, 0, 0, 0], []
    model = resolve_encoder(TextEmbedding("glove.test-10d"))

    result = TextTensorMapper(model)(array)

    assert result.offset.tolist() == offsets
    assert result.values.tolist() == values


def test_arrow_unicode_case_and_duplicate_vocabulary() -> None:
    """Keep Python case semantics and the last row for repeated vocabulary words."""
    vocabulary = VocabularyIndex(pd.Index(["i\u0307", "ος", None, "é", "é", ""]))
    texts = pa.array(["İ ΟΣ", "É", None, "", "unknown"])

    offsets, values = vocabulary.tokenize(texts)

    assert offsets.tolist() == [0, 2, 3, 3, 3, 3]
    assert values.tolist() == [0, 1, 4]


def test_custom_tokenizer_receives_arrow() -> None:
    """Pass the Arrow input directly to custom model tokenizers."""
    model = Mock(spec=TokenEmbeddingModel)
    model.tokenize.return_value = (torch.tensor([0, 1, 1]), torch.tensor([0]))
    array = pa.array(["the", None])

    result = TextTensorMapper(model)(array)

    model.tokenize.assert_called_once_with(array)
    assert result.offset.tolist() == [0, 1, 1]
    assert result.values.tolist() == [0]
