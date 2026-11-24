# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One probe row per labelled event, with that event's own history.

The entity-level fraud probe asks "is this entity's recent history fraudulent?".
Fraud is decided per transaction, so the question a fraud team asks is "is
*this* transaction fraudulent, given the history before it?". Each row here is
one labelled event: its history is every event up to and including it (the
transaction being scored is known; its label is not), and its target is that
event's own label.

The control row holds the event alone. Comparing the two isolates what the
history adds; the raw features for both are computed from the full history, so
the arms differ only in what the encoder sees.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from random import Random

from tfm.contracts.schema import TransactionEvent, TransactionSequence, ensure_utc
from tfm.data.labels.horizon import available_label_value
from tfm.data.labels.windows import LabelWindow
from tfm.data.transforms.ordering import order_events


@dataclass(frozen=True)
class EventHistoryRow:
    """A labelled event, its history up to itself, and the event on its own."""

    history: TransactionSequence
    event_only: TransactionSequence
    label: float


def build_event_history_rows(
    events: Sequence[TransactionEvent],
    *,
    entities: Collection[str],
    window: LabelWindow,
    label_cutoff: datetime,
    max_events: int,
    observation_window: timedelta | None,
    max_rows: int | None,
    seed: int,
    label: str = "fraud",
    negatives_per_positive: float | None = None,
) -> list[EventHistoryRow]:
    """Rows for events of `entities` inside `window` whose label is known by `label_cutoff`.

    More than `max_rows` candidates are sampled uniformly with `seed`, which
    keeps the class prevalence of the window; `None` keeps every candidate. With `negatives_per_positive`,
    every positive is kept (up to `max_rows`) and negatives are sampled to that
    ratio within the cap instead; use it on a training side only, since it
    changes the prevalence.
    """
    by_entity: dict[str, list[TransactionEvent]] = defaultdict(list)
    wanted = set(entities)
    for event in events:
        if event.entity_id in wanted:
            by_entity[event.entity_id].append(event)
    candidates: list[tuple[str, int, float]] = []
    ordered: dict[str, list[TransactionEvent]] = {}
    for entity_id in sorted(by_entity):
        history = order_events(by_entity[entity_id])
        ordered[entity_id] = history
        for index, event in enumerate(history):
            if not window.contains(event.event_timestamp):
                continue
            value = available_label_value(event, label, label_cutoff)
            if value is None:
                continue
            candidates.append((entity_id, index, float(value)))
    rng = Random(seed)
    if negatives_per_positive is not None:
        positives = [c for c in candidates if c[2] >= 0.5]
        negatives = [c for c in candidates if c[2] < 0.5]
        if max_rows is not None and len(positives) > max_rows:
            positives = rng.sample(positives, max_rows)
        n_negatives = min(len(negatives), round(negatives_per_positive * len(positives)))
        if max_rows is not None:
            n_negatives = min(n_negatives, max_rows - len(positives))
        candidates = positives + rng.sample(negatives, n_negatives)
        candidates.sort(key=lambda item: (item[0], item[1]))
    elif max_rows is not None and len(candidates) > max_rows:
        candidates = rng.sample(candidates, max_rows)
        candidates.sort(key=lambda item: (item[0], item[1]))
    rows: list[EventHistoryRow] = []
    for entity_id, index, value in candidates:
        history = ordered[entity_id]
        event = history[index]
        cutoff = ensure_utc(event.event_timestamp)
        start = None if observation_window is None else cutoff - observation_window
        first = max(0, index + 1 - max_events) if max_events > 0 else 0
        past = [
            e
            for e in history[first : index + 1]
            if start is None or ensure_utc(e.event_timestamp) >= start
        ]
        sequence_id = f"{entity_id}/{event.event_id}"
        window_text = None if observation_window is None else str(observation_window)
        rows.append(
            EventHistoryRow(
                history=TransactionSequence(
                    entity_id=entity_id,
                    events=tuple(past),
                    cutoff_time=cutoff,
                    observation_window=window_text,
                    sequence_id=sequence_id,
                ),
                event_only=TransactionSequence(
                    entity_id=entity_id,
                    events=(event,),
                    cutoff_time=cutoff,
                    observation_window=window_text,
                    sequence_id=sequence_id,
                ),
                label=value,
            )
        )
    return rows
