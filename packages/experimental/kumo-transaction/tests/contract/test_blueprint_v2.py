# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Leakage and alignment contracts for the Blueprint v2 hybrid (Blueprint v2 guide)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import torch

from tfm.contracts.errors import IncompatibleCompositionError
from tfm.contracts.kvt import KeyValueTimeBatch
from tfm.contracts.schema import TransactionEvent, TransactionSequence
from tfm.experimental import (
    NextEventFields,
    ProfileEventDecoder,
    TimeRoPE,
    TokenEventKeyMLM,
    UsrTokenPool,
    compose_blueprint_v2,
)
from tfm.features.tokenizers import KeyValueTimeTokenizer
from tfm.pooling.last_token import LastTokenPool
from tfm.position.none import NoPE

START = datetime(2024, 1, 1, tzinfo=UTC)
N_STATES = 4


def _sequence(
    entity: str, n_events: int, *, offset: int, gap_hours: float = 5.0
) -> TransactionSequence:
    """Events whose `state` advances by one each step, so event `j+1` is predictable from `j`."""
    events = tuple(
        TransactionEvent(
            entity_id=entity,
            event_id=f"{entity}-{i}",
            event_timestamp=START + timedelta(hours=gap_hours * i + 0.25 * (i % 3)),
            categorical={"source": "transaction", "state": f"s{(offset + i) % N_STATES}"},
        )
        for i in range(n_events)
    )
    return TransactionSequence(
        entity_id=entity,
        events=events,
        cutoff_time=events[-1].event_timestamp,
        observation_window=None,
    )


def _fitted(sequences: list[TransactionSequence]) -> KeyValueTimeTokenizer:
    tok = KeyValueTimeTokenizer()
    tok.fit(sequences)
    return tok


def _model(vocab_size: int, *, mlm: bool = False, seed: int = 0):
    torch.manual_seed(seed)
    spec = ProfileEventDecoder(
        hidden_size=32,
        num_heads=4,
        profile_depth=1,
        event_depth=1,
        decoder_depth=2,
        vocab_size=vocab_size,
    )
    objective = (NextEventFields(), TokenEventKeyMLM()) if mlm else NextEventFields()
    return compose_blueprint_v2(granularity=spec, objective=objective).eval()


def _replace_value(batch: KeyValueTimeBatch, event: int, new_value: int) -> KeyValueTimeBatch:
    value_ids = batch.value_ids.clone()
    value_ids[:, event, :] = torch.where(
        batch.token_mask[:, event, :].bool(),
        torch.full_like(value_ids[:, event, :], new_value),
        value_ids[:, event, :],
    )
    return KeyValueTimeBatch(**{**batch.__dict__, "value_ids": value_ids})


def test_decoder_slots_never_see_their_target_event() -> None:
    """Slot `k` predicts event `k`, so changing event `k` must leave slots `0..k` unchanged."""
    seqs = [_sequence("u0", 6, offset=0), _sequence("u1", 6, offset=2)]
    tok = _fitted(seqs)
    batch = tok.collate(seqs, max_length=8)
    model = _model(tok.vocab_size)
    with torch.no_grad():
        base, _mask, _tok = model.encode(batch)
        for k in range(1, 6):
            changed, _m, _t = model.encode(_replace_value(batch, k, new_value=5))
            assert torch.allclose(base[:, : k + 1], changed[:, : k + 1], atol=1e-6), k
            assert not torch.allclose(base[:, k + 1 :], changed[:, k + 1 :], atol=1e-6), k


def test_elapsed_time_depends_only_on_past_events() -> None:
    full = _sequence("u0", 6, offset=0)
    prefix = TransactionSequence(
        entity_id="u0",
        events=full.events[:3],
        cutoff_time=full.events[2].event_timestamp,
        observation_window=None,
    )
    tok = _fitted([full])
    long = tok.collate([full], max_length=8)
    short = tok.collate([prefix], max_length=8)
    assert long.elapsed_log is not None and short.elapsed_log is not None
    assert torch.allclose(long.elapsed_log[0, :3], short.elapsed_log[0, :3])
    assert float(long.elapsed_log[0, 0]) == 0.0
    assert bool((long.elapsed_log[0, 1:] >= long.elapsed_log[0, :-1]).all())
    # `time_log` is measured back from the last event, so it does change.
    assert not torch.allclose(long.time_log[0, :3], short.time_log[0, :3])


def test_padding_does_not_change_valid_outputs() -> None:
    short = _sequence("u0", 3, offset=1)
    longer = _sequence("u1", 7, offset=0)
    tok = _fitted([short, longer])
    model = _model(tok.vocab_size)
    alone = tok.collate([short], max_length=8)
    padded = tok.collate([short, longer], max_length=8)
    with torch.no_grad():
        a, _m, _t = model.encode(alone)
        b, _m2, _t2 = model.encode(padded)
        emb_a = model(alone, compute_loss=False).embeddings
        emb_b = model(padded, compute_loss=False).embeddings
    assert emb_a is not None and emb_b is not None
    assert torch.allclose(a[0], b[0, : a.shape[1]], atol=1e-5)
    assert torch.allclose(emb_a[0], emb_b[0], atol=1e-5)


def test_embedding_is_the_slot_after_the_last_valid_event() -> None:
    seqs = [_sequence("u0", 3, offset=0), _sequence("u1", 5, offset=0)]
    tok = _fitted(seqs)
    batch = tok.collate(seqs, max_length=8)
    model = _model(tok.vocab_size)
    with torch.no_grad():
        states, _mask, _tok = model.encode(batch)
        emb = model(batch, compute_loss=False).embeddings
    assert emb is not None
    assert torch.allclose(emb[0], states[0, 3])
    assert torch.allclose(emb[1], states[1, 5])


def test_next_event_training_does_not_learn_to_copy() -> None:
    """On data where event `j+1` is a shift of event `j`, next beats same-event loss.

    A model that could see its target would learn the identity instead, and
    score better from the slot after the event (Blueprint v1 objectives, 0011).
    """
    seqs = [_sequence(f"u{i}", 8, offset=i % N_STATES) for i in range(8)]
    tok = _fitted(seqs)
    batch = tok.collate(seqs, max_length=8)
    model = _model(tok.vocab_size, seed=1).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        loss = model(batch).loss
        assert loss is not None
        loss.backward()
        opt.step()
    diag = model.eval().alignment_losses(batch)
    assert not diag.learned_to_copy
    assert diag.next_token < diag.same_token


def test_mlm_auxiliary_targets_stay_unmasked() -> None:
    seqs = [_sequence("u0", 5, offset=0), _sequence("u1", 5, offset=1)]
    tok = _fitted(seqs)
    batch = tok.collate(seqs, max_length=8)
    model = _model(tok.vocab_size, mlm=True).train()
    out = model(batch)
    assert out.losses is not None
    assert set(out.losses) == {"next_event_fields", "token_event_key_mlm"}
    assert all(torch.isfinite(v) for v in out.losses.values())
    assert out.loss is not None
    assert torch.allclose(out.loss, sum(out.losses.values()))
    assert batch.mlm_labels is None


def test_composition_rejects_incompatible_components() -> None:
    spec = ProfileEventDecoder(vocab_size=16)
    with pytest.raises(IncompatibleCompositionError):
        compose_blueprint_v2(granularity=spec, readout=UsrTokenPool())  # type: ignore[arg-type]
    with pytest.raises(IncompatibleCompositionError):
        compose_blueprint_v2(granularity=spec, objective=TokenEventKeyMLM())  # type: ignore[arg-type]
    model = compose_blueprint_v2(granularity=spec, position=TimeRoPE(), readout=LastTokenPool())
    assert model.mlm_spec is not None
    assert all(block.rope is not None for block in model.decoder.blocks)


def test_nope_leaves_the_history_decoder_untimed() -> None:
    model = compose_blueprint_v2(granularity=ProfileEventDecoder(vocab_size=16), position=NoPE())
    assert all(block.rope is None for block in model.decoder.blocks)
    # The profile encoder still times its milestones.
    assert all(block.rope is not None for block in model.profile_encoder.blocks)
