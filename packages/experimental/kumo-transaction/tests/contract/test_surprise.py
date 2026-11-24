# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Next-event surprise scores each event from its prefix only, row by row."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from tfm.adaptation.batching import select_rows
from tfm.config import load_recipe_config
from tfm.contracts.errors import IncompatibleCompositionError
from tfm.data.collate import collate_batch
from tfm.evaluation.surprise import (
    SURPRISE_WINDOWS,
    extract_surprise,
    next_event_ce_by_key,
    next_event_token_ce,
)
from tfm.objectives.next_event_fields import next_event_targets
from tfm.runtime.common import recipe_dir
from tfm.runtime.data import PreparedData, prepare_data
from tfm.runtime.pipeline import build_model

CPU = torch.device("cpu")
CONFIG = load_recipe_config(recipe_dir("blueprint_v2") / "config.smoke.yaml")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> tuple[PreparedData, torch.nn.Module]:
    data = prepare_data(CONFIG, fixture_dir=Path(tmp_path_factory.mktemp("book")))
    torch.manual_seed(0)
    model = build_model(CONFIG, data.tokenizer.vocab_size).eval()
    return data, model


def _ce_by_event(model: torch.nn.Module, batch: object) -> dict[tuple[int, int, int], float]:
    """Cross-entropy keyed by (row, event rank among the row's valid events, token)."""
    with torch.no_grad():
        states, _mask, _tokens = model.encode(batch)  # type: ignore[attr-defined]
        target = next_event_targets(batch)  # type: ignore[arg-type]
        ce = next_event_token_ce(model, states, target)  # type: ignore[arg-type]
    valid = batch.event_mask.bool()  # type: ignore[attr-defined]
    rank = valid.long().cumsum(dim=1) - 1
    return {
        (int(b), int(rank[b, e]), int(t)): float(v)
        for b, e, t, v in zip(target.batch, target.event, target.token, ce, strict=True)
    }


def test_columns_are_named_per_window_and_key(
    prepared: tuple[PreparedData, torch.nn.Module],
) -> None:
    data, model = prepared
    batch = data.batches[2]
    table = extract_surprise(model, batch, data.tokenizer, device=CPU)

    key_vocab = data.tokenizer.pairs.key_vocab  # type: ignore[attr-defined]
    n_keys = len(key_vocab)
    assert table.vectors.shape == (batch.batch_size, len(SURPRISE_WINDOWS) * (n_keys + 1))
    assert len(table.names) == table.vectors.shape[1]
    window = SURPRISE_WINDOWS[0]
    assert table.names[: n_keys + 1] == (
        *(f"last{window}/{key}" for key in sorted(key_vocab, key=key_vocab.get)),
        f"last{window}/all",
    )
    assert np.isfinite(table.vectors).all() and (table.vectors >= 0).all()
    assert table.entity_ids == tuple(batch.entity_ids)


def test_a_row_scores_the_same_alone_and_in_a_batch(
    prepared: tuple[PreparedData, torch.nn.Module],
) -> None:
    data, model = prepared
    batch = data.batches[2]
    together = extract_surprise(model, batch, data.tokenizer, device=CPU, minibatch_size=64)
    alone = np.concatenate(
        [
            extract_surprise(model, select_rows(batch, [i]), data.tokenizer, device=CPU).vectors
            for i in range(batch.batch_size)
        ]
    )
    np.testing.assert_allclose(together.vectors, alone, rtol=1e-4, atol=1e-5)


def test_dropping_later_events_leaves_earlier_scores_unchanged(
    prepared: tuple[PreparedData, torch.nn.Module],
) -> None:
    data, model = prepared
    sequences = [seq for seq in data.test_sequences if len(seq.events) >= 4][:4]
    keep = 2
    truncated = [dataclasses.replace(seq, events=seq.events[:keep]) for seq in sequences]
    max_tokens = CONFIG.data.max_tokens
    full = _ce_by_event(model, collate_batch(sequences, data.tokenizer, max_tokens=max_tokens))
    prefix = _ce_by_event(model, collate_batch(truncated, data.tokenizer, max_tokens=max_tokens))

    assert prefix and all(rank < keep for _row, rank, _token in prefix)
    for position, value in prefix.items():
        assert full[position] == pytest.approx(value, rel=1e-4, abs=1e-5)


def test_per_key_ce_averages_every_target_token_of_its_key(
    prepared: tuple[PreparedData, torch.nn.Module],
) -> None:
    data, model = prepared
    batch = data.batches[2]
    by_key = next_event_ce_by_key(model, batch, data.tokenizer, device=CPU, minibatch_size=3)
    with torch.no_grad():
        states, _mask, _tokens = model.encode(batch)  # type: ignore[attr-defined]
        target = next_event_targets(batch)  # type: ignore[arg-type]
        ce = next_event_token_ce(model, states, target)  # type: ignore[arg-type]
    key_vocab = data.tokenizer.pairs.key_vocab  # type: ignore[attr-defined]
    amount = target.key_ids == key_vocab["amount"]
    assert by_key["amount"] == pytest.approx(float(ce[amount].mean()), rel=1e-4)
    present = {int(k) for k in target.key_ids.unique()}
    assert set(by_key) == {name for name, key_id in key_vocab.items() if key_id in present}


class _FieldLabels:
    """Any layout's tokenizer, seen by surprise only through its field names."""

    def __init__(self, names: dict[int, str]) -> None:
        self.names = names

    def field_names(self) -> dict[int, str]:
        return self.names


def test_any_tokenizer_that_names_its_fields_gets_surprise(
    prepared: tuple[PreparedData, torch.nn.Module],
) -> None:
    data, model = prepared
    batch = data.batches[2]
    labels = sorted(data.tokenizer.field_names())  # type: ignore[attr-defined]
    other = _FieldLabels({label: f"field{i}" for i, label in enumerate(labels)})
    table = extract_surprise(model, batch, other, device=CPU)
    want = extract_surprise(model, batch, data.tokenizer, device=CPU)
    np.testing.assert_array_equal(table.vectors, want.vectors)
    renamed = [f"field{i}" for i in range(len(labels))]
    assert table.names == tuple(
        f"last{w}/{name}" for w in SURPRISE_WINDOWS for name in (*renamed, "all")
    )
    with pytest.raises(IncompatibleCompositionError, match="FieldNamedTokenizer"):
        extract_surprise(model, batch, object(), device=CPU)
    twice = _FieldLabels(dict.fromkeys(labels, "amount"))
    for score in (extract_surprise, next_event_ce_by_key):
        with pytest.raises(IncompatibleCompositionError, match="its own name"):
            score(model, batch, twice, device=CPU)


def test_refuses_a_model_without_a_next_event_head(
    prepared: tuple[PreparedData, torch.nn.Module],
) -> None:
    data, _model = prepared
    with pytest.raises(IncompatibleCompositionError, match="profile_event_decoder"):
        extract_surprise(torch.nn.Linear(2, 2), data.batches[2], data.tokenizer, device=CPU)


def test_all_is_reserved_for_aggregate_surprise(prepared):
    data, model = prepared
    names = data.tokenizer.field_names()
    names[next(iter(names))] = "all"
    for score in (extract_surprise, next_event_ce_by_key):
        with pytest.raises(IncompatibleCompositionError, match="reserves 'all'"):
            score(model, data.batches[2], _FieldLabels(names), device=CPU)
