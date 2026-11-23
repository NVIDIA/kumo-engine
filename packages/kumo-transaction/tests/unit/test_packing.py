# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sequence packing: row layout, boundary labels, and cross-entity isolation."""

from __future__ import annotations

import pytest
import torch

from tfm import Decoder, Flat, NextEvent, TrxFM
from tfm.contracts.errors import IncompatibleCompositionError
from tfm.data.packing import pack_token_rows

IGNORE = -100


def test_packing_fills_rows_and_marks_segments() -> None:
    batch, stats = pack_token_rows([[1, 2, 3], [4, 5], [6, 7, 8, 9]], max_tokens=5, pad_id=0)

    assert batch.segment_ids is not None
    assert batch.input_ids.shape == (2, 5)
    # First fit: [1,2,3] then [4,5] fills row 0; [6,7,8,9] opens row 1.
    assert batch.input_ids[0].tolist() == [1, 2, 3, 4, 5]
    assert batch.segment_ids[0].tolist() == [1, 1, 1, 2, 2]
    assert batch.attention_mask[0].tolist() == [1, 1, 1, 1, 1]
    assert batch.input_ids[1].tolist() == [6, 7, 8, 9, 0]
    assert batch.attention_mask[1].tolist() == [1, 1, 1, 1, 0]
    # Segment 0 is padding-only, so a padded tail never joins a real block.
    assert batch.segment_ids[1].tolist() == [1, 1, 1, 1, 0]
    assert stats.n_rows == 2
    assert stats.n_sequences == 3
    assert stats.n_real_tokens == 9
    assert stats.token_utilization == pytest.approx(9 / 10)


def test_first_token_of_each_segment_is_not_a_next_token_target() -> None:
    batch, stats = pack_token_rows([[1, 2, 3], [4, 5]], max_tokens=5, pad_id=0)

    assert batch.labels is not None
    # Position 3 opens segment 2; supervising it would predict entity 2's first
    # event from entity 1's last one.
    assert batch.labels[0].tolist() == [IGNORE, 2, 3, IGNORE, 5]
    assert stats.n_supervised_targets == 3


def test_padding_is_masked_and_unsupervised() -> None:
    batch, _ = pack_token_rows([[1, 2]], max_tokens=4, pad_id=7)

    assert batch.labels is not None
    assert batch.input_ids[0].tolist() == [1, 2, 7, 7]
    assert batch.attention_mask[0].tolist() == [1, 1, 0, 0]
    assert batch.labels[0].tolist() == [IGNORE, 2, IGNORE, IGNORE]


def test_oversized_sequence_is_truncated_to_the_context() -> None:
    batch, stats = pack_token_rows([[1, 2, 3, 4, 5]], max_tokens=3, pad_id=0)

    assert batch.input_ids[0].tolist() == [1, 2, 3]
    assert stats.n_truncated == 1


def test_packing_needs_a_next_token_pair() -> None:
    with pytest.raises(IncompatibleCompositionError, match="max_tokens >= 2"):
        pack_token_rows([[1, 2]], max_tokens=1)


def test_empty_input_packs_to_an_empty_batch() -> None:
    batch, stats = pack_token_rows([], max_tokens=4)

    assert batch.batch_size == 0
    assert stats.n_rows == 0
    assert stats.token_utilization == 0.0


def test_packed_segments_do_not_leak_across_entities() -> None:
    """A token's hidden state must not change when a later segment changes."""
    torch.manual_seed(0)
    model = TrxFM(
        granularity=Flat(Decoder(depth=2, hidden_size=16, num_heads=4, vocab_size=32)),
        objective=NextEvent(),
    ).eval()
    left, _ = pack_token_rows([[1, 2, 3], [4, 5]], max_tokens=5)
    right, _ = pack_token_rows([[1, 2, 3], [9, 9]], max_tokens=5)

    with torch.no_grad():
        out_left = model(left, compute_loss=False)
        out_right = model(right, compute_loss=False)

    assert out_left.hidden_states is not None
    assert out_right.hidden_states is not None
    torch.testing.assert_close(
        out_left.hidden_states[:, :3], out_right.hidden_states[:, :3], atol=1e-6, rtol=1e-5
    )
    assert not torch.allclose(out_left.hidden_states[:, 3:], out_right.hidden_states[:, 3:])


def test_packed_loss_matches_the_padded_loss_over_the_same_targets() -> None:
    """Packing changes throughput, not the next-event objective's value."""
    torch.manual_seed(0)
    model = TrxFM(
        granularity=Flat(Decoder(depth=2, hidden_size=16, num_heads=4, vocab_size=32)),
        objective=NextEvent(),
    ).eval()
    packed, stats = pack_token_rows([[1, 2, 3], [4, 5]], max_tokens=5)
    padded, _ = pack_token_rows([[1, 2, 3]], max_tokens=3)
    padded_tail, _ = pack_token_rows([[4, 5]], max_tokens=2)

    with torch.no_grad():
        packed_loss = model(packed).loss
        first = model(padded).loss
        second = model(padded_tail).loss

    assert packed_loss is not None and first is not None and second is not None
    # Token-mean loss: weight each padded loss by its supervised target count.
    expected = (first * 2 + second * 1) / stats.n_supervised_targets
    torch.testing.assert_close(packed_loss, expected, atol=1e-5, rtol=1e-4)
