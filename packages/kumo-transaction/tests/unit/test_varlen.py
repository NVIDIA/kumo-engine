# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Padding-free event encoding for the PRAGMA stack.

The padded path is the correctness baseline: every test here asserts the varlen
path reproduces it, per the performance rules in AGENTS.md.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import torch

from tfm.compositions.profile_history import ProfileHistory
from tfm.data.adapters.pragmatiq import generate_pragmatiq_fixture, load_pragmatiq_dataset
from tfm.data.sequence_builders.entity_sequences import build_entity_sequences
from tfm.features.tokenizers.key_value_time import KeyValueTimeTokenizer
from tfm.layers.profile_history import ProfileHistoryModel
from tfm.layers.varlen import compress_rows, scatter_rows


def _kvt_batch(tmp_path, *, n_users: int = 4, events_per_user: int = 6):
    root = generate_pragmatiq_fixture(
        tmp_path / "book", n_users=n_users, events_per_user=events_per_user, seed=5
    )
    dataset = load_pragmatiq_dataset(root)
    sequences = build_entity_sequences(
        list(dataset.events), datetime(2024, 12, 31, tzinfo=UTC), max_events=8
    )
    tokenizer = KeyValueTimeTokenizer()
    tokenizer.set_profiles({p.entity_id: p for p in dataset.profiles})
    tokenizer.fit(sequences)
    return tokenizer.collate(sequences, max_length=24), tokenizer.vocab_size


def test_compress_drops_empty_rows_and_trims_width() -> None:
    hidden = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])

    packed, packed_mask, index = compress_rows(hidden, mask)

    assert index.rows.tolist() == [0, 2]
    assert index.lengths.tolist() == [2, 1]
    assert packed.shape == (2, 2, 2)
    assert packed_mask.tolist() == [[1, 1], [1, 0]]
    torch.testing.assert_close(packed[0], hidden[0, :2])
    torch.testing.assert_close(packed[1], hidden[2, :2])


def test_cu_seqlens_is_the_exclusive_prefix_sum_of_lengths() -> None:
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]])
    _packed, _mask, index = compress_rows(torch.zeros(3, 4, 2), mask)

    assert index.cu_seqlens().tolist() == [0, 3, 5]
    assert index.cu_seqlens().dtype == torch.int32
    # 5 of 12 grid slots carry a token.
    assert index.padding_fraction() == pytest.approx(1 - 5 / 12)


def test_scatter_restores_the_padded_layout() -> None:
    hidden = torch.randn(3, 4, 2)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
    packed, _mask, index = compress_rows(hidden, mask)

    restored = scatter_rows(packed, index, like=hidden)

    torch.testing.assert_close(restored[0, :2], hidden[0, :2])
    torch.testing.assert_close(restored[2, :2], hidden[2, :2])
    assert torch.all(restored[1] == 0)
    # Trimmed tail positions are zero, not stale values.
    assert torch.all(restored[:, 2:] == 0)


def test_an_all_padding_batch_compresses_to_nothing() -> None:
    hidden = torch.randn(2, 3, 4)
    packed, _mask, index = compress_rows(hidden, torch.zeros(2, 3, dtype=torch.long))

    assert index.n_kept == 0
    assert packed.shape[0] == 0
    assert torch.all(scatter_rows(packed, index, like=hidden) == 0)


def test_compress_rejects_a_non_row_tensor() -> None:
    with pytest.raises(ValueError, match=r"\[R, T, D\]"):
        compress_rows(torch.zeros(2, 3), torch.ones(2, 3))


def test_varlen_matches_the_padded_baseline(tmp_path) -> None:
    batch, vocab = _kvt_batch(tmp_path)
    spec = ProfileHistory(hidden_size=32, num_heads=4, vocab_size=vocab, dropout=0.0)
    padded = ProfileHistoryModel(spec).eval()
    varlen = ProfileHistoryModel(
        ProfileHistory(hidden_size=32, num_heads=4, vocab_size=vocab, dropout=0.0, varlen=True)
    ).eval()
    varlen.load_state_dict(padded.state_dict())

    with torch.no_grad():
        want = padded(batch, compute_loss=False)
        got = varlen(batch, compute_loss=False)

    assert want.embeddings is not None and got.embeddings is not None
    torch.testing.assert_close(got.embeddings, want.embeddings, atol=1e-5, rtol=1e-4)
    assert want.logits is not None and got.logits is not None
    torch.testing.assert_close(got.logits, want.logits, atol=1e-4, rtol=1e-3)
    assert varlen.last_padding_fraction > 0.0
    assert padded.last_padding_fraction == 0.0


def test_varlen_trains_and_reports_what_it_skipped(tmp_path) -> None:
    batch, vocab = _kvt_batch(tmp_path)
    model = ProfileHistoryModel(
        ProfileHistory(hidden_size=32, num_heads=4, vocab_size=vocab, dropout=0.0, varlen=True)
    )
    model.train()

    output = model(batch, compute_loss=True)
    assert output.loss is not None
    output.loss.backward()

    assert torch.isfinite(output.loss)
    assert 0.0 < model.last_padding_fraction < 1.0
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
