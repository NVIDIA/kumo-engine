# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build per-entity sequences that exclude future events."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Protocol

from tfm.contracts.schema import TransactionEvent, TransactionSequence, ensure_utc
from tfm.data.transforms.ordering import order_events
from tfm.data.validation.schema import validate_events


def events_at_or_before(
    events: Sequence[TransactionEvent],
    cutoff: datetime,
) -> list[TransactionEvent]:
    cutoff_utc = ensure_utc(cutoff)
    return [
        event for event in order_events(events) if ensure_utc(event.event_timestamp) <= cutoff_utc
    ]


def build_entity_sequences(
    events: Sequence[TransactionEvent],
    cutoff: datetime,
    *,
    max_events: int,
    observation_window: timedelta | None = None,
    causal: bool = True,
) -> list[TransactionSequence]:
    """One sequence per entity using events with timestamp <= cutoff."""
    validated = validate_events(events)
    cutoff_utc = ensure_utc(cutoff)
    window_start = None if observation_window is None else cutoff_utc - observation_window
    grouped: dict[str, list[TransactionEvent]] = defaultdict(list)
    for event in validated:
        ts = ensure_utc(event.event_timestamp)
        if ts > cutoff_utc:
            continue
        if window_start is not None and ts < window_start:
            continue
        grouped[event.entity_id].append(event)

    sequences: list[TransactionSequence] = []
    for entity_id, entity_events in grouped.items():
        ordered = order_events(entity_events)
        if max_events > 0:
            ordered = ordered[-max_events:]
        if not ordered:
            continue
        sequences.append(
            TransactionSequence(
                entity_id=entity_id,
                events=tuple(ordered),
                cutoff_time=cutoff_utc,
                observation_window=None if observation_window is None else str(observation_window),
                causal=causal,
            )
        )
    sequences.sort(key=lambda seq: seq.entity_id)
    return sequences


def build_entity_windows(
    events: Sequence[TransactionEvent],
    cutoff: datetime,
    *,
    window_events: int,
    stride: int | None = None,
    observation_window: timedelta | None = None,
    causal: bool = True,
) -> list[TransactionSequence]:
    """Many sequences per entity, tiling each history with fixed-size windows.

    `build_entity_sequences` keeps only an entity's most recent `max_events`,
    which is right for a probe that wants one embedding per entity but discards
    most of the book when building a pretraining corpus. This walks each
    entity's ordered history instead, emitting a sequence per window.

    Every window still ends at or before `cutoff`, so the temporal guarantee is
    unchanged. A trailing partial window is kept; entities shorter than one
    window yield exactly one sequence, as before.
    """
    if window_events <= 0:
        raise ValueError(f"window_events must be positive, got {window_events}.")
    step = window_events if stride is None else stride
    if step <= 0:
        raise ValueError(f"stride must be positive, got {step}.")

    validated = validate_events(events)
    cutoff_utc = ensure_utc(cutoff)
    window_start = None if observation_window is None else cutoff_utc - observation_window
    grouped: dict[str, list[TransactionEvent]] = defaultdict(list)
    for event in validated:
        ts = ensure_utc(event.event_timestamp)
        if ts > cutoff_utc:
            continue
        if window_start is not None and ts < window_start:
            continue
        grouped[event.entity_id].append(event)

    sequences: list[TransactionSequence] = []
    for entity_id in sorted(grouped):
        ordered = order_events(grouped[entity_id])
        for start in range(0, len(ordered), step):
            chunk = ordered[start : start + window_events]
            if not chunk:
                continue
            sequences.append(
                TransactionSequence(
                    entity_id=entity_id,
                    events=tuple(chunk),
                    cutoff_time=cutoff_utc,
                    observation_window=(
                        None if observation_window is None else str(observation_window)
                    ),
                    causal=causal,
                )
            )
            if start + window_events >= len(ordered):
                break
    return sequences


class _TimedLabel(Protocol):
    @property
    def entity_id(self) -> str: ...

    @property
    def timestamp(self) -> datetime: ...


def build_sequences_at_label_cutoffs[TLabel: _TimedLabel](
    events: Sequence[TransactionEvent],
    rows: Sequence[TLabel],
    *,
    max_events: int,
    observation_window: timedelta | None = None,
    causal: bool = True,
) -> list[tuple[TransactionSequence, TLabel]]:
    """One history per label row using events with timestamp <= that row's cutoff."""
    grouped: dict[str, list[TransactionEvent]] = defaultdict(list)
    for event in events:
        grouped[event.entity_id].append(event)
    pairs: list[tuple[TransactionSequence, TLabel]] = []
    for row in rows:
        built = build_entity_sequences(
            grouped.get(row.entity_id, ()),
            row.timestamp,
            max_events=max_events,
            observation_window=observation_window,
            causal=causal,
        )
        if built:
            pairs.append((built[0], row))
    return pairs
