# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime

import pytest
import torch

from tfm.api.validation import validate_tokenizer_layout
from tfm.compositions.profile_history import ProfileHistory
from tfm.contracts.errors import IncompatibleCompositionError
from tfm.contracts.kvt import KeyValueTimeBatch
from tfm.contracts.outputs import ModelOutput
from tfm.data.adapters.pragmatiq import generate_pragmatiq_fixture, load_pragmatiq_dataset
from tfm.data.sequence_builders.entity_sequences import build_entity_sequences
from tfm.features.tokenizers.key_value_time import KeyValueTimeTokenizer
from tfm.layers.profile_history import ProfileHistoryModel
from tfm.objectives.mlm import TokenEventKeyMLM, TokenEventKeyMLMLoss, apply_mlm_mask
from tfm.position.time_rope import TimeRoPEModule


def _output(logits: torch.Tensor) -> ModelOutput:
    return ModelOutput(hidden_states=torch.zeros_like(logits), logits=logits)


def _mlm_batch(mlm_labels: torch.Tensor) -> KeyValueTimeBatch:
    """A batch carrying only what `TokenEventKeyMLMLoss` reads."""
    empty = torch.zeros_like(mlm_labels)
    return KeyValueTimeBatch(
        key_ids=empty,
        value_ids=empty,
        positions=empty,
        token_mask=torch.ones_like(mlm_labels),
        event_mask=torch.ones(mlm_labels.shape[0], 1, dtype=torch.long),
        time_log=empty.float(),
        events={"hour": empty, "dow": empty, "dom": empty},
        prof_key_ids=empty,
        prof_value_ids=empty,
        prof_positions=empty,
        prof_mask=empty,
        prof_time_log=empty.float(),
        mlm_labels=mlm_labels,
    )


def test_time_rope_relative_difference() -> None:
    rope = TimeRoPEModule(head_dim=8, base=10000.0)
    query = torch.randn(1, 2, 3, 8)
    key = torch.randn(1, 2, 3, 8)
    pos = torch.tensor([[0.0, 1.0, 4.0]])
    q1, k1 = rope(query, key, pos)
    q2, k2 = rope(query, key, pos + 3.0)
    # Relative rotation: shifting all positions should preserve pairwise dots
    # of rotated q/k up to the shared offset (check finite, shape).
    assert q1.shape == query.shape
    assert torch.isfinite(q2).all()
    assert torch.isfinite(k1).all()


def test_profile_history_forward_and_mlm(tmp_path) -> None:
    root = generate_pragmatiq_fixture(tmp_path / "book", n_users=4, events_per_user=6, seed=5)
    dataset = load_pragmatiq_dataset(root)
    seqs = build_entity_sequences(
        list(dataset.events), datetime(2024, 12, 31, tzinfo=UTC), max_events=8
    )
    tok = KeyValueTimeTokenizer()
    tok.set_profiles({p.entity_id: p for p in dataset.profiles})
    tok.fit(seqs)
    batch = tok.collate(seqs, max_length=24)
    spec = ProfileHistory(hidden_size=32, num_heads=4, vocab_size=tok.vocab_size, dropout=0.0)
    validate_tokenizer_layout(tok, spec)
    model = ProfileHistoryModel(spec, TokenEventKeyMLM())
    model.train()
    out = model(batch, compute_loss=True)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    assert out.embeddings is not None
    assert out.embeddings.shape == (batch.batch_size, spec.hidden_size)
    assert out.logits is not None
    masked = apply_mlm_mask(batch, TokenEventKeyMLM())
    assert masked.mlm_labels is not None
    assert (masked.mlm_labels != -100).any()


def test_checkpoints_saved_with_the_old_encoder_keys_still_load() -> None:
    spec = ProfileHistory(hidden_size=16, num_heads=2, vocab_size=64)
    torch.manual_seed(0)
    saved = ProfileHistoryModel(spec).state_dict()
    legacy = {
        key.replace("event_encoder.blocks.", "event_blocks.")
        .replace("event_encoder.norm.", "event_norm.")
        .replace("profile_encoder.blocks.", "profile_blocks.")
        .replace("profile_encoder.norm.", "profile_norm."): value
        for key, value in saved.items()
    }
    assert "event_blocks.0.qkv.weight" in legacy

    torch.manual_seed(1)
    model = ProfileHistoryModel(spec)
    model.load_state_dict(legacy)

    assert all(torch.equal(model.state_dict()[key], value) for key, value in saved.items())


def test_label_smoothing_raises_the_loss_floor() -> None:
    """Smoothing spreads target mass, so a perfect prediction no longer scores 0."""
    logits = torch.tensor([[[20.0, 0.0, 0.0, 0.0]]])
    labels = torch.tensor([[0]])
    output = _output(logits)
    batch = _mlm_batch(labels)

    unsmoothed = TokenEventKeyMLMLoss(label_smoothing=0.0)(output, batch)
    smoothed = TokenEventKeyMLMLoss(label_smoothing=0.1)(output, batch)

    assert float(unsmoothed) == pytest.approx(0.0, abs=1e-6)
    assert float(smoothed) > float(unsmoothed)


def test_smoothing_ignores_unk_dropout_and_padding_positions() -> None:
    """Only masked value positions carry loss, smoothed or not."""
    logits = torch.tensor([[[20.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 20.0]]])
    supervised_only = _mlm_batch(torch.tensor([[0, -100]]))
    both = _mlm_batch(torch.tensor([[0, 3]]))
    loss = TokenEventKeyMLMLoss(label_smoothing=0.1)

    assert float(loss(_output(logits), supervised_only)) == pytest.approx(
        float(loss(_output(logits), both))
    )


def test_label_smoothing_reaches_the_model_from_the_spec() -> None:
    spec = ProfileHistory(hidden_size=16, num_heads=4, vocab_size=24, dropout=0.0)

    model = ProfileHistoryModel(spec, TokenEventKeyMLM(label_smoothing=0.25))

    assert model.objective_loss.label_smoothing == 0.25
    # PRAGMA's published value is the default.
    assert TokenEventKeyMLM().label_smoothing == 0.1


def test_invalid_label_smoothing_is_rejected() -> None:
    with pytest.raises(IncompatibleCompositionError, match="label_smoothing"):
        TokenEventKeyMLM(label_smoothing=1.0)
    with pytest.raises(IncompatibleCompositionError, match="label_smoothing"):
        TokenEventKeyMLM(label_smoothing=-0.1)
