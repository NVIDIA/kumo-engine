# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pickle
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
import torch
from kumoapi.encoder import TextEmbedding
from kumoapi.typing import Stype
from torch_geometric.testing import onlyCUDA

from kumo_core.encoders.fused.text import FusedTextEncoder
from kumo_core.encoders.tensor_mapper import TextTensorMapper
from kumo_core.encoders.text import normalize_text
from kumo_core.features.stats import get_stats
from kumo_core import embedding_models
from kumo_core.embedding_models import EmbeddingModelStore, GloVeVectors, resolve_encoder
from kumo_core.glove import TokenEmbeddingTable


@pytest.fixture
def glove_with_null_words(tmp_path) -> Iterator[tuple[GloVeVectors, np.ndarray]]:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    pd.DataFrame({"index": ["the", None, "of", None, "to", "of", "a"]}).to_parquet(
        artifacts / "glove_6B_index.parquet"
    )
    embedding = np.arange(7 * 50, dtype=np.float32).reshape(7, 50)
    np.save(artifacts / "glove_6B_50d_emb.npy", embedding)
    model = GloVeVectors(artifacts, cache_dir=tmp_path / "cache")
    store = EmbeddingModelStore()
    store.add("glove.6B-50d", model)
    previous = embedding_models.set_default_store(store)
    yield model, embedding
    embedding_models.set_default_store(previous)


def test_normalize_text() -> None:
    values = pd.Series(["Hello,  WORLD!", "get<br/>old", None])

    normalized = normalize_text(values)

    assert normalized.tolist() == [["hello", "world"], ["get", "old"], []]


def test_glove_tokenizer_preserves_non_range_rows() -> None:
    rowptr, tokens = resolve_encoder(TextEmbedding("glove.test-10d")).tokenize(
        pd.Series(["The invalid", "of, to"], index=[10, 20])
    )

    assert rowptr.tolist() == [0, 1, 3]
    assert tokens.tolist() == [0, 3, 4]


def test_glove_lookup_skips_null_and_repeated_words(glove_with_null_words) -> None:
    model, embedding = glove_with_null_words
    table = TokenEmbeddingTable(model).initialize()

    rowptr, tokens = model.tokenize(
        pd.Series(["The OF to", "null nan", None, "a"])
    )
    assert rowptr.tolist() == [0, 3, 3, 3, 4]
    assert tokens.tolist() == [0, 5, 4, 6]  # A repeated word maps to its last row.
    assert model.vocabulary()[tokens.numpy()].tolist() == ["the", "of", "to", "a"]
    assert torch.equal(table.emb[tokens], torch.from_numpy(embedding[[0, 5, 4, 6]]))

    rows, mask = model.lookup(["a", None, float("nan"), "unknown", "of"])
    assert rows.tolist() == [6, 5]
    assert mask.tolist() == [True, False, False, False, True]


def test_glove_text_encoding_with_null_words(glove_with_null_words) -> None:
    _, embedding = glove_with_null_words
    ser = pd.Series(["the of", "null", "to a to"])

    mapper = TextTensorMapper()
    feat = mapper(ser)
    assert mapper.backward(feat).tolist() == ["the of", None, "to a to"]

    encoder = FusedTextEncoder(
        {"text": TextEmbedding()}, exclude_cols=[], out_channels=4, enc_mode="column"
    )
    expected = np.stack([
        embedding[[0, 5]].mean(axis=0),
        np.zeros(50, dtype=np.float32),
        embedding[[4, 6, 4]].mean(axis=0),
    ])
    assert torch.allclose(
        encoder.forward_fixed(feat).squeeze(1), torch.from_numpy(expected)
    )

    stats = get_stats(ser, Stype.text, approx=False)
    assert stats.glove_overlap == pytest.approx(5 / 6)


def test_one_table_per_model_identity(tmp_path) -> None:
    model = resolve_encoder(TextEmbedding("glove.test-10d"))
    table = TokenEmbeddingTable.instance(model).cleanup()
    with pytest.raises(RuntimeError, match="not been initialized"):
        _ = table.emb

    assert table.initialize().is_initialized
    assert table.emb.shape == (8, 10)
    assert table is TokenEmbeddingTable.instance(pickle.loads(pickle.dumps(model)))
    assert table.cleanup() is table
    assert not table.is_initialized


def test_column_major_files_load_row_major(tmp_path) -> None:
    pd.DataFrame({"index": ["red", "shoe"]}).to_parquet(tmp_path / "glove_6B_index.parquet")
    vectors = np.asfortranarray(np.arange(100, dtype=np.float32).reshape(2, 50))
    np.save(tmp_path / "glove_6B_50d_emb.npy", vectors)
    table = TokenEmbeddingTable(GloVeVectors(tmp_path)).initialize()
    assert table.emb.is_contiguous()  # Else every lookup copies the table.
    assert table.emb.tolist() == vectors.tolist()


@onlyCUDA
def test_embedding_device() -> None:
    table = TokenEmbeddingTable.instance(resolve_encoder(TextEmbedding("glove.test-10d")))
    table.cleanup().initialize(torch.device("cuda"))
    assert table.emb.device.type == "cuda"
    table.to("cpu")
    assert table.emb.device.type == "cpu"


class _NumpyModel:
    family, label, dim = "numpy-words/1", None, 2

    def __init__(self, rows: int = 2, extra_pointer: bool = False) -> None:
        self.rows, self.extra_pointer = rows, extra_pointer

    def fingerprint(self) -> str:
        return f"{self.rows}"

    def vocabulary(self) -> pd.Index:
        return pd.Index(["red", "shoe"])

    def vectors(self) -> np.ndarray:
        return np.ones((self.rows, self.dim))  # float64

    def tokenize(self, texts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        rowptr = [0, 2] + [2] * (len(texts) - 1 + self.extra_pointer)
        return np.array(rowptr), np.array([0, 1])


def test_custom_models_are_checked_where_they_are_used() -> None:
    feat = TextTensorMapper(_NumpyModel())(pd.Series(["red shoe", "sock"]))
    assert feat.offset.tolist() == [0, 2, 2]
    assert feat.values.dtype == torch.long
    assert TokenEmbeddingTable(_NumpyModel()).initialize().emb.dtype == torch.float32

    with pytest.raises(ValueError, match=r"returned 4 row pointers for 2 texts"):
        TextTensorMapper(_NumpyModel(extra_pointer=True))(pd.Series(["a", "b"]))
    with pytest.raises(ValueError, match=r"has shape \(3, 2\), but it must be \(2, 2\)"):
        TokenEmbeddingTable(_NumpyModel(rows=3)).initialize()
