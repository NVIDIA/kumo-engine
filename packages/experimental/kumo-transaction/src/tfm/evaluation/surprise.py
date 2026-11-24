# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Next-event surprise: how badly a causal model predicted a row's own latest events.

A model trained on `next_event_fields` predicts every field value of event `j`
from decoder slot `j`, which has seen the profile and events `0..j-1` only.
The cross-entropy of the true value is the model's surprise at that field.
Averaged per key over a row's last few events, it is a readout of its own: a
stolen card day, copied from another user, is unlikely under the victim's
history, so its surprise is high even when the last-token embedding does not
separate it.

Every score uses only events up to the row's cutoff, because the row's
sequence holds nothing later and each event is scored from its prefix. This
is a readout for evaluation and probing; it is not an objective and changes
no weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from tfm.adaptation.batching import minibatches
from tfm.contracts.errors import IncompatibleCompositionError
from tfm.contracts.tokenizers import FieldNamedTokenizer
from tfm.objectives.next_event_fields import NextEventTargets, next_event_targets

SURPRISE_WINDOWS: tuple[int, ...] = (1, 8)
"""The latest events each surprise feature averages over: the last one, and the last eight."""

CE_SLICE = 8192
"""Target tokens scored at once. Full-vocabulary logits for every token of a
minibatch do not fit beside a prepared 100k-user book in unified memory."""


def _field_names(tokenizer: object) -> dict[int, str]:
    """Field names by `key_ids` label; empty for a tokenizer that names none."""
    if isinstance(tokenizer, FieldNamedTokenizer):
        names = tokenizer.field_names()
        if "all" in names.values():
            raise IncompatibleCompositionError(
                "field_names() reserves 'all' for aggregate surprise"
            )
        if len(set(names.values())) < len(names):
            raise IncompatibleCompositionError(
                f"{type(tokenizer).__name__}.field_names() gives several labels one name; "
                "give each label its own name."
            )
        return names
    return {}


@dataclass(frozen=True)
class SurpriseTable:
    """One surprise vector per batch row, with a name for each column."""

    vectors: NDArray[np.float32]
    names: tuple[str, ...]
    entity_ids: tuple[str, ...]


@runtime_checkable
class NextEventScorer(Protocol):
    """A model whose decoder slot `j` predicts the values of event `j`.

    `encode` returns `(states [B, N+1, D], slot mask, token states)` for a
    key-value-time batch; `next_event_logits` scores one target token per row
    from its slot state and its key. `profile_event_decoder` implements it.
    """

    @property
    def vocab_size(self) -> int: ...

    def encode(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def next_event_logits(self, states: torch.Tensor, key_ids: torch.Tensor) -> torch.Tensor: ...


def _decoder(model: nn.Module) -> NextEventScorer:
    if not isinstance(model, NextEventScorer):
        raise IncompatibleCompositionError(
            "Next-event surprise needs a model with a next-event head (encode and "
            "next_event_logits), such as profile_event_decoder trained on "
            f"next_event_fields; got {type(model).__name__}."
        )
    return model


def _prepare(model: NextEventScorer, device: torch.device) -> None:
    module = cast(nn.Module, model)
    module.eval()
    module.to(device)


def next_event_token_ce(
    model: NextEventScorer, states: torch.Tensor, target: NextEventTargets
) -> torch.Tensor:
    """Cross-entropy of every next-event target token, from the decoder `states`.

    Token `t` of event `j` is scored from slot `j` and its own key, as the
    objective scores it. Computed in slices of `CE_SLICE` tokens.
    """
    out: list[torch.Tensor] = []
    for start in range(0, int(target.values.numel()), CE_SLICE):
        sl = slice(start, start + CE_SLICE)
        logits = model.next_event_logits(
            states[target.batch[sl], target.event[sl]], target.key_ids[sl]
        )
        out.append(
            torch.nn.functional.cross_entropy(logits.float(), target.values[sl], reduction="none")
        )
    return torch.cat(out) if out else torch.zeros(0, device=states.device)


def surprise_features(
    model: NextEventScorer,
    chunk: Any,
    states: torch.Tensor,
    key_ids: Sequence[int],
    windows: Sequence[int] = SURPRISE_WINDOWS,
) -> torch.Tensor:
    """`[rows, len(windows) * (len(key_ids) + 1)]` surprise features of one encoded chunk.

    For each window: the mean cross-entropy of each key in `key_ids` over the
    row's latest `window` events (0 where the key is absent), then the mean
    over every token in the window.
    """
    target = next_event_targets(chunk)
    ce = next_event_token_ce(model, states, target)
    rows = chunk.event_mask.shape[0]
    index = torch.arange(chunk.event_mask.shape[1], device=ce.device)
    last = torch.where(chunk.event_mask.bool(), index, -1).amax(dim=1)
    age = last[target.batch] - target.event
    slot = torch.full((int(model.vocab_size),), len(key_ids), device=ce.device)
    slot[torch.as_tensor(list(key_ids), device=ce.device, dtype=torch.long)] = torch.arange(
        len(key_ids), device=ce.device
    )
    column = slot[target.key_ids]
    parts: list[torch.Tensor] = []
    for window in windows:
        inside = (age < window).float()
        sums = torch.zeros(rows, len(key_ids) + 1, device=ce.device)
        counts = torch.zeros_like(sums)
        sums.index_put_((target.batch, column), ce * inside, accumulate=True)
        counts.index_put_((target.batch, column), inside, accumulate=True)
        per_key = (sums / counts.clamp(min=1))[:, : len(key_ids)]
        overall = sums.sum(1, keepdim=True) / counts.sum(1, keepdim=True).clamp(min=1)
        parts += [per_key, overall]
    return torch.cat(parts, dim=1)


def extract_surprise(
    model: nn.Module,
    batch: Any,
    tokenizer: Any,
    *,
    device: torch.device,
    windows: Sequence[int] = SURPRISE_WINDOWS,
    minibatch_size: int = 64,
) -> SurpriseTable:
    """Surprise features for every row of `batch`, over every field the tokenizer names
    (`FieldNamedTokenizer.field_names`).

    Columns are named `last{window}/{field}` and `last{window}/all`, fields in label
    order. Raises `IncompatibleCompositionError` for a model without a next-event head,
    or a tokenizer that names no fields.
    """
    decoder = _decoder(model)
    fields = _field_names(tokenizer)
    if not fields:
        raise IncompatibleCompositionError(
            f"{type(tokenizer).__name__} names no fields to name surprise columns by; "
            "implement tfm.contracts.tokenizers.FieldNamedTokenizer."
        )
    key_ids = sorted(fields)
    keys = [fields[key_id] for key_id in key_ids]
    names = tuple(f"last{window}/{key}" for window in windows for key in (*keys, "all"))
    _prepare(decoder, device)
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for chunk in minibatches(batch, minibatch_size):
            chunk = chunk.to(device)
            states, _mask, _tokens = decoder.encode(chunk)
            features = surprise_features(decoder, chunk, states, key_ids, windows)
            parts.append(features.float().cpu().numpy())
    vectors = (
        np.concatenate(parts).astype(np.float32) if parts else np.zeros((0, len(names)), np.float32)
    )
    return SurpriseTable(vectors=vectors, names=names, entity_ids=tuple(batch.entity_ids))


def next_event_ce_by_key(
    model: nn.Module,
    batch: Any,
    tokenizer: Any,
    *,
    device: torch.device,
    minibatch_size: int = 16,
) -> dict[str, float]:
    """Mean next-event cross-entropy per key over every event of `batch`, in nats.

    Inputs are unmasked, as an embedding sees them. Keys absent from `batch`
    are omitted.
    """
    decoder = _decoder(model)
    names = _field_names(tokenizer)
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    _prepare(decoder, device)
    with torch.no_grad():
        for chunk in minibatches(batch, minibatch_size):
            chunk = chunk.to(device)
            states, _mask, _tokens = decoder.encode(chunk)
            target = next_event_targets(chunk)
            ce = next_event_token_ce(decoder, states, target)
            for key in target.key_ids.unique().tolist():
                chosen = target.key_ids == key
                sums[key] = sums.get(key, 0.0) + float(ce[chosen].sum())
                counts[key] = counts.get(key, 0) + int(chosen.sum())
    return {str(names.get(key, key)): sums[key] / counts[key] for key in sorted(sums)}


__all__ = [
    "CE_SLICE",
    "NextEventScorer",
    "SURPRISE_WINDOWS",
    "SurpriseTable",
    "extract_surprise",
    "next_event_ce_by_key",
    "next_event_token_ce",
    "surprise_features",
]
