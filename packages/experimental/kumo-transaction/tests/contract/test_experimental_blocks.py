# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the experimental auxiliary objectives on Blueprint v2.

`PastReconstruction` must only predict events its slot has already seen, and
`SubsequenceContrastive` must only crop inside the history it was given.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import torch

from tfm.contracts.kvt import KeyValueTimeBatch
from tfm.contracts.schema import TransactionEvent, TransactionSequence
from tfm.experimental import (
    NextEventFields,
    PastReconstruction,
    ProfileEventDecoder,
    SubsequenceContrastive,
    compose_blueprint_v2,
)
from tfm.features.tokenizers import KeyValueTimeTokenizer
from tfm.objectives.contrastive import random_crop
from tfm.objectives.past_reconstruction import past_targets

START = datetime(2024, 1, 1, tzinfo=UTC)


def _batch(lengths: tuple[int, ...] = (6, 4)) -> tuple[KeyValueTimeTokenizer, KeyValueTimeBatch]:
    sequences = []
    for row, n in enumerate(lengths):
        events = tuple(
            TransactionEvent(
                entity_id=f"u{row}",
                event_id=f"u{row}-{i}",
                event_timestamp=START + timedelta(hours=3 * i + row),
                categorical={"source": "transaction", "state": f"s{(row + i) % 4}"},
                numeric={"amount": float(10 * i + row)},
            )
            for i in range(n)
        )
        sequences.append(
            TransactionSequence(
                entity_id=f"u{row}",
                events=events,
                cutoff_time=events[-1].event_timestamp,
                observation_window=None,
            )
        )
    tok = KeyValueTimeTokenizer().fit(sequences)
    return tok, tok.collate(sequences, max_length=8)


@pytest.mark.parametrize("objective", [PastReconstruction(), SubsequenceContrastive()])
def test_objectives_declare_their_semantics(objective: object) -> None:
    for method in ("horizon_semantics", "cutoff_semantics", "gradient", "leakage_checks"):
        assert getattr(objective, method)().strip(), method
    assert objective.target_level() in {"event", "sequence"}
    assert "loss" in objective.required_outputs()


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_past_targets_are_events_the_slot_has_seen(lag: int) -> None:
    _tok, batch = _batch()
    targets = past_targets(batch, lag)
    assert targets.batch.numel() > 0
    # Slot s holds event s - 1 and has seen events 0..s-1; the target is `lag` earlier.
    assert torch.equal(targets.slot - 1 - lag, targets.event)
    assert torch.all(targets.event >= 0)
    lengths = batch.event_mask.sum(dim=1)
    assert torch.all(targets.slot - 1 < lengths[targets.batch])
    assert torch.all(targets.gap >= 0)


def test_random_crop_is_a_contiguous_retimed_slice_of_the_history() -> None:
    _tok, batch = _batch()
    crop = random_crop(batch, 0.5, generator=torch.Generator().manual_seed(0))
    for row, n in enumerate((6, 4)):
        kept = int(crop.event_mask[row].sum())
        assert kept == -(-n // 2)
        # Find the crop's start in the original row by matching value ids.
        start = next(
            s
            for s in range(n - kept + 1)
            if torch.equal(batch.value_ids[row, s : s + kept], crop.value_ids[row, :kept])
        )
        assert start + kept <= n
        assert crop.elapsed_log is not None
        assert crop.elapsed_log[row, 0].item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("auxiliary", [PastReconstruction(horizon=2), SubsequenceContrastive()])
def test_auxiliary_leaves_encoding_causal_and_reaches_the_encoders(auxiliary: object) -> None:
    tok, batch = _batch()
    torch.manual_seed(0)
    spec = ProfileEventDecoder(
        hidden_size=32,
        num_heads=4,
        profile_depth=1,
        event_depth=1,
        decoder_depth=2,
        vocab_size=tok.vocab_size,
    )
    model = compose_blueprint_v2(granularity=spec, objective=(NextEventFields(), auxiliary))
    model.eval()
    with torch.no_grad():
        base, _mask, _tok = model.encode(batch)
        value_ids = batch.value_ids.clone()
        value_ids[:, 3, :] = torch.where(
            batch.token_mask[:, 3, :].bool(),
            torch.full_like(value_ids[:, 3, :], 5),
            value_ids[:, 3, :],
        )
        changed, _m, _t = model.encode(
            KeyValueTimeBatch(**{**batch.__dict__, "value_ids": value_ids})
        )
    assert torch.allclose(base[:, :4], changed[:, :4], atol=1e-6)

    model.train()
    out = model(batch)
    name = auxiliary.name
    assert out.losses is not None and torch.isfinite(out.losses[name])
    out.losses[name].backward()
    for part in (model.event_encoder, model.decoder):
        grads = [p.grad for p in part.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads), type(part).__name__
