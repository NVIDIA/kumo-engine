# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporal train/validation/test splits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from tfm.contracts.errors import SchemaError, TemporalLeakageError
from tfm.contracts.schema import TransactionEvent, ensure_utc
from tfm.data.transforms.ordering import order_events
from tfm.data.validation.schema import validate_events


def parse_duration(text: str) -> timedelta:
    text = text.strip().lower()
    if text.endswith("d"):
        return timedelta(days=int(text[:-1]))
    if text.endswith("h"):
        return timedelta(hours=int(text[:-1]))
    raise ValueError(f"Unsupported duration {text!r}; use e.g. 7d or 24h.")


def parse_timestamp(text: str) -> datetime:
    raw = text.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    ts = datetime.fromisoformat(raw)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


@dataclass(frozen=True)
class TemporalSplitSpec:
    train_end: datetime
    validation_end: datetime
    label_delay: timedelta
    observation_window: timedelta | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_end", ensure_utc(self.train_end))
        object.__setattr__(self, "validation_end", ensure_utc(self.validation_end))
        if self.validation_end <= self.train_end:
            raise SchemaError("validation_end must be after train_end.")
        if self.label_delay.total_seconds() < 0:
            raise SchemaError("label_delay must be non-negative.")


@dataclass(frozen=True)
class SplitAssignment:
    train: tuple[TransactionEvent, ...]
    validation: tuple[TransactionEvent, ...]
    test: tuple[TransactionEvent, ...]


def assign_temporal_split(
    events: Sequence[TransactionEvent],
    spec: TemporalSplitSpec,
) -> SplitAssignment:
    """Split by event timestamp. Does not attach delayed labels."""
    ordered = order_events(validate_events(events))
    train: list[TransactionEvent] = []
    validation: list[TransactionEvent] = []
    test: list[TransactionEvent] = []
    for event in ordered:
        ts = ensure_utc(event.event_timestamp)
        if spec.observation_window is not None:
            window_start = spec.train_end - spec.observation_window
            if ts < window_start and ts < spec.train_end:
                # Still eligible for train if inside a longer history; keep all
                # events before train_end in train so sequences can use history.
                pass
        if ts < spec.train_end:
            train.append(event)
        elif ts < spec.validation_end:
            validation.append(event)
        else:
            test.append(event)
    if not train or not validation or not test:
        raise TemporalLeakageError(
            "Temporal split produced an empty partition. "
            "Adjust cutoffs or generate a longer synthetic history."
        )
    return SplitAssignment(tuple(train), tuple(validation), tuple(test))
