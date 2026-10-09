# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Generator
from typing import Literal

import numpy as np
import pandas as pd
import pytest
import torch
from kumoapi.encoder import Encoder, TextEmbedding
from torch_frame.data import MultiNestedTensor
from torch_geometric.testing import withCUDA

import kumo_core.modeling.context as modeling_context
from kumo_core import embedding_models
from kumo_core.embedding_models import EmbeddingModelStore
from kumo_core.encoders.fused.text import FusedTextEncoder
from kumo_core.encoders.tensor_mapper import TextTensorMapper
from kumo_core.glove import VocabularyIndex


def create_text_tensor_data() -> MultiNestedTensor:
    tensor_list = ([
        [
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1, 2, 3, 7], dtype=torch.long),
            torch.tensor([5], dtype=torch.long),
        ],
        [
            torch.tensor([6], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            torch.tensor([7], dtype=torch.long),
        ],
        [
            torch.tensor([3, 4], dtype=torch.long),
            torch.tensor([2], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
        ],
        [
            torch.tensor([7], dtype=torch.long),
            torch.tensor([6], dtype=torch.long),
            torch.tensor([3], dtype=torch.long),
        ],
    ])
    return MultiNestedTensor.from_tensor_mat(tensor_list)


@withCUDA
def test_forward(
    device: torch.device,
    teardown_glove_helper: Generator,
) -> None:
    out_channels = 5
    x = create_text_tensor_data().to(device)
    col_encoder_dict: dict[str, Encoder] = {
        'col0': TextEmbedding('glove.test-10d'),
        'col1': TextEmbedding('glove.test-10d'),
        'col2': TextEmbedding('glove.test-10d'),
    }
    fused_encoder = FusedTextEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=out_channels,
        enc_mode='fuse',
        device=device,
    )
    out = fused_encoder(x)
    assert out.size() == (x.num_rows, out_channels)


@withCUDA
@pytest.mark.parametrize('enc_mode', ['fuse', 'column'])
def test_empty_input(
    device: torch.device,
    enc_mode: Literal['fuse', 'column'],
    teardown_glove_helper: Generator,
) -> None:
    feat = MultiNestedTensor(
        num_rows=0,
        num_cols=3,
        values=torch.tensor([], dtype=torch.long),
        offset=torch.tensor([0]),
    ).to(device)
    encoder = FusedTextEncoder(
        col_encoder_dict={
            'col0': TextEmbedding('glove.test-10d'),
            'col1': TextEmbedding('glove.test-10d'),
            'col2': TextEmbedding('glove.test-10d'),
        },
        exclude_cols=[],
        out_channels=5,
        enc_mode=enc_mode,
        device=device,
    )
    out = encoder(feat)
    assert out.size() == (0, 5) if enc_mode == 'fuse' else (0, 3, 5)


@withCUDA
def test_enc_mode(
    device: torch.device,
    teardown_glove_helper: Generator,
) -> None:
    out_channels = 5
    x = create_text_tensor_data().to(device)
    col_encoder_dict: dict[str, Encoder] = {
        'col0': TextEmbedding('glove.test-10d'),
        'col1': TextEmbedding('glove.test-10d'),
        'col2': TextEmbedding('glove.test-10d'),
    }

    # ===============
    # enc_mode='fuse'
    # ===============
    fused_encoder = FusedTextEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=out_channels,
        enc_mode='fuse',
        device=device,
    )
    out = fused_encoder(x)
    assert out.size() == (x.num_rows, out_channels)

    # =================
    # enc_mode='column'
    # =================
    fused_encoder = FusedTextEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=out_channels,
        enc_mode='column',
        device=device,
    )
    out = fused_encoder(x)
    assert out.size() == (x.num_rows, x.num_cols, out_channels)

    # ================
    # invalid enc_mode
    # ================
    with pytest.raises(ValueError, match="Unsupported encoder mode 'invalid'"):
        FusedTextEncoder(
            col_encoder_dict=col_encoder_dict,
            exclude_cols=[],
            out_channels=out_channels,
            enc_mode='invalid',
        )


def test_inconsistent_col_order() -> None:
    """Test FusedTextEncoder raises error if the column order is different
    between in TF and in the fused encoder.
    """
    x = create_text_tensor_data()

    col_encoder_dict: dict[str, Encoder] = {
        'col0': TextEmbedding('glove.test-10d'),
        'col1': TextEmbedding('glove.test-10d'),
        'col2': TextEmbedding('glove.test-10d'),
    }
    fused_encoder = FusedTextEncoder(
        col_encoder_dict=col_encoder_dict,
        exclude_cols=[],
        out_channels=50,
        enc_mode='fuse',
    )
    invalid_col_names = ['col0', 'col2', 'col1']
    with pytest.raises(ValueError, match='The order of columns'):
        fused_encoder(x, invalid_col_names)


@withCUDA
@pytest.mark.parametrize('enc_mode', ['fuse', 'column'])
def test_gradient(
    device: torch.device,
    enc_mode: Literal['fuse', 'column'],
) -> None:
    encoder = FusedTextEncoder(
        col_encoder_dict={
            'col0': TextEmbedding('glove.test-10d'),
            'col1': TextEmbedding('glove.test-10d'),
            'col2': TextEmbedding('glove.test-10d'),
        },
        exclude_cols=[],
        out_channels=16,
        enc_mode=enc_mode,
        device=device,
    )
    with modeling_context.explaining():
        feat = create_text_tensor_data().to(device)
        out = encoder(feat)

    with pytest.raises(ValueError, match='Gradients not yet computed'):
        encoder.get_grad_wrt_input()

    out.backward(torch.randn_like(out))

    grad = encoder.get_grad_wrt_input()
    assert isinstance(grad, torch.Tensor)
    assert grad.size() == (4, 3)
    assert not torch.allclose(grad[:, 0], torch.zeros(4, device=device))
    assert not torch.allclose(grad[:, 1], torch.zeros(4, device=device))
    assert not torch.allclose(grad[:, 2], torch.zeros(4, device=device))
    assert not hasattr(encoder, 'saved_tensors')

    if encoder.enc_mode == 'fuse':
        # Note that in fuse mode, the gradients across different columns
        # are the same due to the GPU memory restriction.
        assert torch.allclose(grad[:, 0], grad[:, 1])
        assert torch.allclose(grad[:, 0], grad[:, 2])

    with pytest.raises(ValueError, match='Did not store required data'):
        encoder.get_grad_wrt_input()


class _Model:
    family = "test-words/1"
    label = None

    def __init__(self, words: list[str], dim: int, seed: int) -> None:
        self.words = pd.Index(words, dtype=object)
        self.dim = dim
        self.array = np.random.default_rng(seed).random(
            (len(words), dim), dtype=np.float32)

    def fingerprint(self) -> str:
        return embedding_models._fingerprint(self.words, self.array)

    def tokenize(self, texts: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
        return VocabularyIndex(self.words).tokenize(texts)

    def vectors(self) -> np.ndarray:
        return self.array

    def vocabulary(self) -> pd.Index:
        return self.words


@pytest.mark.parametrize('enc_mode', ['fuse', 'column'])
def test_each_column_uses_its_own_model(
    enc_mode: Literal['fuse', 'column'],
    teardown_glove_helper: Generator,
) -> None:
    # The same word has a different row (and dim) in each model.
    small = _Model(["red", "shoe", "blue"], dim=3, seed=0)
    large = _Model(["blue", "sock", "shoe", "red"], dim=4, seed=1)
    store = EmbeddingModelStore()
    store.add("small", small)
    store.add("large", large)
    encoders: dict[str, Encoder] = {
        'a': TextEmbedding("small"),
        'b': TextEmbedding("large"),
        'c': TextEmbedding("small"),
    }
    texts = {
        'a': ["red shoe", "blue", ""],
        'b': ["sock red", "", "shoe shoe"],
        'c': ["shoe", "red blue", "red"],
    }
    models = {'a': small, 'b': large, 'c': small}

    previous = embedding_models.set_default_store(store)
    try:
        feat = MultiNestedTensor.cat([
            TextTensorMapper.from_encoder(encoders[col])(
                pd.Series(texts[col])) for col in encoders
        ], dim=1)
        encoder = FusedTextEncoder(encoders, exclude_cols=[], out_channels=5,
                                   enc_mode=enc_mode)
    finally:
        embedding_models.set_default_store(previous)

    def vectors(col: str, row: int) -> np.ndarray:
        model = models[col]
        words = texts[col][row].split()
        return model.array[[model.words.get_loc(w) for w in words]]

    def mean(parts: list[np.ndarray], dim: int) -> np.ndarray:
        rows = np.concatenate([part.reshape(-1, dim) for part in parts])
        return rows.mean(0) if len(rows) else np.zeros(dim, np.float32)

    out = encoder.forward_fixed(feat).numpy()
    for row in range(3):
        if enc_mode == 'fuse':
            expected = np.concatenate([
                mean([vectors('a', row), vectors('c', row)], 3),
                mean([vectors('b', row)], 4),
            ])
            np.testing.assert_allclose(out[row], expected, rtol=1e-6)
        else:
            for idx, col in enumerate(encoders):
                dim = models[col].dim
                np.testing.assert_allclose(out[row, idx, :dim],
                                           mean([vectors(col, row)], dim),
                                           rtol=1e-6)
                assert not out[row, idx, dim:].any()
    assert encoder(feat).shape == ((3, 5) if enc_mode == 'fuse' else
                                   (3, 3, 5))
