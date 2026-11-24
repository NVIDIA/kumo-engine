# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collate sequences into batches, attaching labels and raw features.

The tokenizer decides the layout (`TransactionBatch` for flat streams,
`KeyValueTimeBatch` for per-event rows). Labels and raw features are data
concerns, so they are attached here, the same way for every layout. The
batch's `fraud_labels` field holds whichever event label `label` names, and
the raw aggregates read the numeric field `amount_key`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import torch

from tfm.adaptation.batching import minibatches, select_rows, shuffled
from tfm.contracts.batch import TransactionBatch
from tfm.contracts.errors import IncompatibleCompositionError
from tfm.contracts.schema import TransactionSequence
from tfm.contracts.tokenizers import FLAT_LAYOUTS, SequenceTokenizer
from tfm.data.labels.horizon import available_label_value
from tfm.data.labels.windows import LabelWindow
from tfm.features.tokenizers.financial import event_time_features

RAW_FEATURE_NAMES: tuple[str, ...] = (
    "mean_amount",
    "max_amount",
    "n_events",
    "last_hour",
    "last_dow",
)


def raw_feature_vector(sequence: TransactionSequence, *, amount_key: str = "amount") -> list[float]:
    """Leakage-safe aggregates from events already truncated to cutoff."""
    if not sequence.events:
        return [0.0] * len(RAW_FEATURE_NAMES)
    amounts = [float(event.numeric.get(amount_key, 0.0)) for event in sequence.events]
    last = sequence.events[-1]
    time_feat = event_time_features(last)
    return [
        float(sum(amounts) / len(amounts)),
        float(max(amounts)),
        float(len(sequence.events)),
        time_feat["hour"],
        time_feat["dow"],
    ]


def sequence_label(
    sequence: TransactionSequence, window: LabelWindow | None = None, *, label: str = "fraud"
) -> float:
    """1.0 if an event's `label` available at cutoff is positive, else 0.0.

    With `window`, only events inside it count (ADR 0013), so a fraud event
    from an earlier period cannot make this row positive. Without it, every
    event in the history counts, which is right for a single-period table.
    """
    for event in sequence.events:
        if window is not None and not window.contains(event.event_timestamp):
            continue
        value = available_label_value(event, label, sequence.cutoff_time)
        if value is not None and float(value) >= 0.5:
            return 1.0
    return 0.0


def collate_batch(
    sequences: Sequence[TransactionSequence],
    tokenizer: SequenceTokenizer,
    *,
    max_tokens: int,
    label_window: LabelWindow | None = None,
    label: str = "fraud",
    amount_key: str = "amount",
) -> Any:
    """Tokenize `sequences` and attach labels, raw features, and ids.

    Works for every layout. `max_tokens` is tokens per row for flat layouts
    and tokens per event for `key_value_time`.
    """
    batch = tokenizer.collate(sequences, max_length=max_tokens)
    if not dataclasses.is_dataclass(batch) or isinstance(batch, type):
        raise IncompatibleCompositionError(
            f"Tokenizer {tokenizer.name!r} collate must return a batch dataclass."
        )
    updates: dict[str, Any] = {
        "fraud_labels": torch.tensor(
            [sequence_label(seq, label_window, label=label) for seq in sequences],
            dtype=torch.float32,
        ),
        "numeric_features": torch.tensor(
            [raw_feature_vector(seq, amount_key=amount_key) for seq in sequences],
            dtype=torch.float32,
        ).reshape(len(sequences), len(RAW_FEATURE_NAMES)),
        "entity_ids": tuple(seq.entity_id for seq in sequences),
        "sequence_ids": tuple(seq.sequence_id or seq.entity_id for seq in sequences),
    }
    if any(f.name == "event_ids" for f in dataclasses.fields(batch)):
        updates["event_ids"] = tuple(
            seq.events[-1].event_id if seq.events else "" for seq in sequences
        )
    return dataclasses.replace(batch, **updates)


def collate_sequences(
    sequences: Sequence[TransactionSequence],
    tokenizer: SequenceTokenizer,
    *,
    max_tokens: int,
    label_window: LabelWindow | None = None,
) -> TransactionBatch:
    """`collate_batch` for the flat layouts, typed as `TransactionBatch`."""
    if tokenizer.layout not in FLAT_LAYOUTS:
        raise IncompatibleCompositionError(
            f"collate_sequences packs the flat layouts {sorted(FLAT_LAYOUTS)} into "
            f"TransactionBatch. Got tokenizer {tokenizer.name!r} with layout {tokenizer.layout!r}."
        )
    batch = collate_batch(sequences, tokenizer, max_tokens=max_tokens, label_window=label_window)
    if not isinstance(batch, TransactionBatch):
        raise IncompatibleCompositionError(
            f"Flat-layout collate must return TransactionBatch, got {type(batch).__name__}."
        )
    return batch


def shuffled_rows(batch: Any, seed: int) -> Any:
    """`batch` with its rows in a seeded random order.

    Corpora are built entity by entity, so without this every minibatch holds
    neighbouring windows of one or two entities.
    """
    return shuffled(batch, seed)


__all__ = [
    "RAW_FEATURE_NAMES",
    "collate_batch",
    "collate_sequences",
    "minibatches",
    "raw_feature_vector",
    "select_rows",
    "sequence_label",
    "shuffled_rows",
]
