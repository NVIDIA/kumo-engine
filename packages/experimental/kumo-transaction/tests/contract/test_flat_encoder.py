# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlatEncoder + MaskedFields + LastEventMeanPool contracts (flat-encoder guide)."""

import dataclasses
from pathlib import Path

import pytest
import torch

from tfm import Decoder, Flat, LastTokenPool, NextEvent, NoPE, TrxFM
from tfm.config import RecipeConfig, load_recipe_config
from tfm.contracts.batch import TransactionBatch
from tfm.contracts.errors import IncompatibleCompositionError
from tfm.contracts.tokenizers import FieldVocabulary
from tfm.experimental import (
    FlatEncoder,
    LastEventMeanPool,
    MaskedFields,
    MeanPool,
    UsrTokenPool,
    compose_flat_encoder,
)
from tfm.features.tokenizers.financial import TransactionFieldTokenizer
from tfm.objectives.masked_fields import apply_field_mask, masked_fields_loss
from tfm.pooling.last_event import last_event_mask
from tfm.runtime.data import tokenizer_for
from tfm.runtime.pipeline import build_model

FIELDS = ("AMT", "CHIP", "DOW")
ABLATION = Path("benchmarks/ablations/flat_encoder_tabformer.yaml")


def _vocab() -> FieldVocabulary:
    return TransactionFieldTokenizer(fields=FIELDS).field_vocabulary()


def _model(readout: LastEventMeanPool | MeanPool | UsrTokenPool | None = None):
    torch.manual_seed(0)
    spec = FlatEncoder(vocabulary=_vocab(), depth=2, hidden_size=32, num_heads=4, max_seq_len=64)
    return compose_flat_encoder(spec, readout=readout).eval()


def _row(events: list[tuple[int, int, int]]) -> list[int]:
    """`<bos>`, events of (AMT, CHIP, DOW) value offsets joined by `<sep>`, `<eos>`."""
    vocab = _vocab()
    ids = [vocab.bos_id]
    for index, event in enumerate(events):
        if index:
            ids.append(vocab.sep_id)
        ids.extend(
            start + value for (start, _), value in zip(vocab.field_ranges, event, strict=True)
        )
    return [*ids, vocab.eos_id]


def _batch(rows: list[list[int]], width: int | None = None) -> TransactionBatch:
    vocab = _vocab()
    width = width or max(len(r) for r in rows)
    ids = [r + [vocab.pad_id] * (width - len(r)) for r in rows]
    mask = [[1] * len(r) + [0] * (width - len(r)) for r in rows]
    return TransactionBatch(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))


def test_field_vocabulary_matches_the_tokens() -> None:
    tokenizer = TransactionFieldTokenizer(fields=FIELDS)
    vocab = tokenizer.field_vocabulary()
    assert vocab.field_names == FIELDS
    for name, (start, stop) in zip(FIELDS, vocab.field_ranges, strict=True):
        tokens = [tokenizer.id_to_token[i] for i in range(start, stop)]
        assert all(t.startswith(f"{name}_") for t in tokens)
    assert vocab.field_ranges[-1][1] == tokenizer.vocab_size


def test_later_tokens_change_earlier_states() -> None:
    model = _model()
    first = _batch([_row([(1, 0, 2), (3, 1, 4)])])
    second = _batch([_row([(1, 0, 2), (5, 2, 6)])])
    with torch.no_grad():
        a = model(first, compute_loss=False).hidden_states
        b = model(second, compute_loss=False).hidden_states
    assert not torch.allclose(a[0, 1], b[0, 1]), "attention is not bidirectional"


def test_padding_does_not_change_embeddings() -> None:
    model = _model()
    rows = [_row([(1, 0, 2), (3, 1, 4)]), _row([(2, 3, 5)])]
    with torch.no_grad():
        tight = model(_batch(rows), compute_loss=False).embeddings
        wide = model(_batch(rows, width=20), compute_loss=False).embeddings
    assert tight is not None and wide is not None
    torch.testing.assert_close(tight, wide, atol=1e-5, rtol=1e-5)


def test_last_event_mask_selects_only_the_last_events_fields() -> None:
    vocab = _vocab()
    model = _model()
    batch = _batch([_row([(1, 0, 2), (3, 1, 4), (5, 2, 6)]), _row([(2, 3, 5)])], width=16)
    mask = last_event_mask(batch.input_ids, batch.attention_mask, model.field_of_id, vocab.sep_id)
    # <bos> e1(3) <sep> e2(3) <sep> e3(3) <eos>: the last event is positions 9..11.
    assert mask[0].nonzero().flatten().tolist() == [9, 10, 11]
    assert mask[1].nonzero().flatten().tolist() == [1, 2, 3]


def test_last_event_readout_is_the_mean_of_those_states() -> None:
    model = _model()
    batch = _batch([_row([(1, 0, 2), (3, 1, 4)])])
    with torch.no_grad():
        out = model(batch, compute_loss=False)
    assert out.embeddings is not None
    torch.testing.assert_close(out.embeddings[0], out.hidden_states[0, 5:8].mean(dim=0))


def test_masking_chooses_field_tokens_only_and_keeps_targets_off_the_input() -> None:
    vocab = _vocab()
    model = _model()
    batch = _batch([_row([(1, 0, 2)] * 6)] * 4, width=40)
    spec = MaskedFields(p_field=0.5, p_event=0.0, p_mask_token=1.0, p_random_token=0.0)
    masked, targets = apply_field_mask(
        batch.input_ids,
        batch.attention_mask,
        model.field_of_id,
        model.field_ranges,
        sep_id=vocab.sep_id,
        mask_id=model.spec.mask_id,
        spec=spec,
        generator=torch.Generator().manual_seed(0),
    )
    chosen = targets != spec.ignore_index
    assert bool(chosen.any())
    assert bool((model.field_of_id[batch.input_ids][chosen] >= 0).all())
    assert bool((masked[chosen] == model.spec.mask_id).all())
    assert torch.equal(masked[~chosen], batch.input_ids[~chosen])
    assert torch.equal(targets[chosen], batch.input_ids[chosen])


def test_event_masking_hides_whole_events_and_random_values_stay_in_field() -> None:
    vocab = _vocab()
    model = _model()
    batch = _batch([_row([(1, 0, 2), (3, 1, 4)])])
    spec = MaskedFields(p_field=0.0, p_event=1.0, p_mask_token=0.0, p_random_token=1.0)
    masked, targets = apply_field_mask(
        batch.input_ids,
        batch.attention_mask,
        model.field_of_id,
        model.field_ranges,
        sep_id=vocab.sep_id,
        mask_id=model.spec.mask_id,
        spec=spec,
    )
    is_field = model.field_of_id[batch.input_ids] >= 0
    assert torch.equal(targets != spec.ignore_index, is_field)
    assert torch.equal(model.field_of_id[masked], model.field_of_id[batch.input_ids])


def test_loss_ranks_only_the_target_fields_values() -> None:
    model = _model()
    vocab = _vocab()
    amt_start, _ = vocab.field_ranges[0]
    target = torch.tensor([amt_start])
    logits = torch.zeros(1, vocab.vocab_size)
    logits[0, amt_start] = 5.0
    other = logits.clone()
    chip_start, chip_stop = vocab.field_ranges[1]
    other[0, chip_start:chip_stop] = 100.0
    base = masked_fields_loss(logits, target, model.field_of_id, model.field_allowed)
    noisy = masked_fields_loss(other, target, model.field_of_id, model.field_allowed)
    torch.testing.assert_close(base, noisy)


def test_training_forward_returns_the_masked_fields_loss() -> None:
    model = _model().train()
    batch = _batch([_row([(1, 0, 2), (3, 1, 4), (5, 2, 6)])] * 2)
    out = model(batch)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.losses is not None and set(out.losses) == {"masked_fields"}
    out.loss.backward()


def test_packed_batches_are_refused() -> None:
    model = _model()
    batch = _batch([_row([(1, 0, 2)])])
    packed = TransactionBatch(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        segment_ids=torch.zeros_like(batch.input_ids),
    )
    with pytest.raises(IncompatibleCompositionError, match="packed"):
        model(packed, compute_loss=False)


def test_next_event_on_the_encoder_breaks_rule_1() -> None:
    spec = FlatEncoder(vocabulary=_vocab(), hidden_size=32, num_heads=4)
    with pytest.raises(IncompatibleCompositionError, match="rule 1"):
        compose_flat_encoder(spec, objective=NextEvent())  # type: ignore[arg-type]


def test_masked_fields_on_the_decoder_breaks_rule_2() -> None:
    with pytest.raises(IncompatibleCompositionError, match="rule 2"):
        TrxFM(
            granularity=Flat(Decoder(vocab_size=64)),
            objective=[NextEvent(), MaskedFields()],  # type: ignore[list-item]
        )


def test_the_encoder_needs_rotary_positions_and_its_readouts() -> None:
    spec = FlatEncoder(vocabulary=_vocab(), hidden_size=32, num_heads=4)
    with pytest.raises(IncompatibleCompositionError, match="RoPE"):
        compose_flat_encoder(spec, position=NoPE())  # type: ignore[arg-type]
    with pytest.raises(IncompatibleCompositionError, match="reads out"):
        compose_flat_encoder(spec, readout=LastTokenPool())  # type: ignore[arg-type]


def test_config_refuses_surprise_and_other_tokenizers() -> None:
    raw = load_recipe_config(ABLATION).model_dump()
    raw["evaluation"]["surprise"] = True
    with pytest.raises(ValueError, match="surprise"):
        RecipeConfig.model_validate(raw)
    raw["evaluation"]["surprise"] = False
    raw["features"]["tokenizer"] = "key_value_time"
    raw["features"]["fields"] = None
    with pytest.raises(ValueError, match="FieldVocabularyTokenizer"):
        RecipeConfig.model_validate(raw)


def test_the_model_masks_by_the_fitted_tokenizers_fields() -> None:
    """A tokenizer that sets its field layout in `fit` is not rebuilt from the config."""
    config = load_recipe_config(ABLATION)
    default = tokenizer_for(config).field_vocabulary()  # type: ignore[attr-defined]
    fitted = dataclasses.replace(
        default, field_names=tuple(f"fit_{n}" for n in default.field_names)
    )

    class Fitted:
        def field_vocabulary(self) -> FieldVocabulary:
            return fitted

    model = build_model(config, fitted.vocab_size, tokenizer=Fitted())  # type: ignore[arg-type]
    assert model.spec.vocabulary is fitted  # type: ignore[union-attr]
