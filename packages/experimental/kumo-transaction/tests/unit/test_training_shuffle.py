# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seeded row and batch-order shuffling for pretraining."""

from __future__ import annotations

import itertools

import torch

from tfm.contracts.batch import TransactionBatch
from tfm.data.collate import minibatches, select_rows, shuffled_rows
from tfm.training.trainer import epoch_order, iterate_batches


def _batch(n: int) -> TransactionBatch:
    ids = torch.arange(n).unsqueeze(1).repeat(1, 3)
    return TransactionBatch(
        input_ids=ids,
        attention_mask=torch.ones(n, 3, dtype=torch.long),
        labels=ids.clone(),
        fraud_labels=torch.arange(n, dtype=torch.float32),
        numeric_features=torch.arange(n, dtype=torch.float32).unsqueeze(1),
        entity_ids=tuple(f"e{i}" for i in range(n)),
        sequence_ids=tuple(f"s{i}" for i in range(n)),
        event_ids=tuple(f"v{i}" for i in range(n)),
    )


def test_select_rows_keeps_every_field_aligned() -> None:
    picked = select_rows(_batch(5), [3, 0, 4])

    assert picked.input_ids[:, 0].tolist() == [3, 0, 4]
    assert picked.labels is not None and picked.labels[:, 0].tolist() == [3, 0, 4]
    assert picked.fraud_labels is not None and picked.fraud_labels.tolist() == [3.0, 0.0, 4.0]
    assert picked.numeric_features is not None
    assert picked.numeric_features[:, 0].tolist() == [3.0, 0.0, 4.0]
    assert picked.entity_ids == ("e3", "e0", "e4")
    assert picked.sequence_ids == ("s3", "s0", "s4")
    assert picked.event_ids == ("v3", "v0", "v4")


def test_shuffled_rows_is_a_seeded_permutation() -> None:
    first = shuffled_rows(_batch(32), seed=42)
    again = shuffled_rows(_batch(32), seed=42)
    other = shuffled_rows(_batch(32), seed=7)

    order = first.input_ids[:, 0].tolist()
    assert sorted(order) == list(range(32))
    assert order != list(range(32))
    assert order == again.input_ids[:, 0].tolist()
    assert order != other.input_ids[:, 0].tolist()
    assert first.entity_ids == tuple(f"e{i}" for i in order)


def test_shuffled_minibatches_mix_entities() -> None:
    """Corpus order puts neighbouring windows of one entity in the same batch."""
    in_order = minibatches(_batch(64), 8)
    mixed = minibatches(shuffled_rows(_batch(64), seed=0), 8)

    assert in_order[0].input_ids[:, 0].tolist() == list(range(8))
    assert mixed[0].input_ids[:, 0].tolist() != list(range(8))


def test_epoch_order_is_identity_without_shuffle() -> None:
    assert epoch_order(5, seed=1, epoch=3, shuffle=False) == [0, 1, 2, 3, 4]


def test_epoch_order_changes_per_epoch_and_is_rank_independent() -> None:
    epochs = [epoch_order(20, seed=42, epoch=e, shuffle=True) for e in range(3)]

    for order in epochs:
        assert sorted(order) == list(range(20))
    assert epochs[0] != epochs[1] != epochs[2]
    # Same inputs, same order: every rank computes it without communicating.
    assert epochs[1] == epoch_order(20, seed=42, epoch=1, shuffle=True)


def test_iterate_batches_visits_every_batch_once_per_epoch() -> None:
    batches = minibatches(_batch(12), 3)

    seen = [
        int(b.input_ids[0, 0]) for b in itertools.islice(iterate_batches(batches, shuffle=True), 8)
    ]

    assert sorted(seen[:4]) == [0, 3, 6, 9]
    assert sorted(seen[4:]) == [0, 3, 6, 9]
    assert seen[:4] != seen[4:]
