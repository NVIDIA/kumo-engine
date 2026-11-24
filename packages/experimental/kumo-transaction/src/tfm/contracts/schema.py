# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical transaction event and sequence contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from tfm.contracts.errors import SchemaError

ALLOWED_LABEL_LEVELS = frozenset({"event", "entity", "sequence"})


def ensure_utc(ts: datetime, tz_name: str = "UTC") -> datetime:
    """Return a timezone-aware UTC datetime."""
    if ts.tzinfo is None:
        if tz_name != "UTC":
            raise SchemaError(
                f"Naive timestamp provided with timezone={tz_name!r}; attach tzinfo or use UTC."
            )
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


@dataclass(frozen=True)
class EventLabel:
    """A label attached to an event, entity, or sequence."""

    name: str
    value: float
    available_at: datetime
    level: str = "event"

    def __post_init__(self) -> None:
        if self.level not in ALLOWED_LABEL_LEVELS:
            raise SchemaError(
                f"Label {self.name!r} has invalid level {self.level!r}; "
                f"expected one of {sorted(ALLOWED_LABEL_LEVELS)}."
            )
        object.__setattr__(self, "available_at", ensure_utc(self.available_at))


@dataclass(frozen=True)
class TransactionEvent:
    """Canonical transaction event.

    Identifiers must be synthetic or otherwise non-sensitive. Do not store
    real account, card, or customer numbers.
    """

    entity_id: str
    event_id: str
    event_timestamp: datetime
    timezone: str = "UTC"
    categorical: Mapping[str, str] = field(default_factory=dict)
    numeric: Mapping[str, float] = field(default_factory=dict)
    text: Mapping[str, str] = field(default_factory=dict)
    labels: Mapping[str, EventLabel] = field(default_factory=dict)
    group: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_timestamp",
            ensure_utc(self.event_timestamp, self.timezone),
        )
        object.__setattr__(self, "categorical", dict(self.categorical))
        object.__setattr__(self, "numeric", dict(self.numeric))
        object.__setattr__(self, "text", dict(self.text))
        object.__setattr__(self, "labels", dict(self.labels))


@dataclass(frozen=True)
class TransactionSequence:
    """Ordered events for one entity up to a cutoff time."""

    entity_id: str
    events: tuple[TransactionEvent, ...]
    cutoff_time: datetime
    observation_window: str | None
    causal: bool = True
    sequence_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cutoff_time", ensure_utc(self.cutoff_time))
        if self.sequence_id is None:
            object.__setattr__(
                self, "sequence_id", f"{self.entity_id}:{self.cutoff_time.isoformat()}"
            )


@dataclass(frozen=True)
class LifelongItem:
    """Dated profile milestone (account opened, KYC passed, ...)."""

    key: str
    timestamp: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))


@dataclass(frozen=True)
class EntityProfile:
    """Optional entity snapshot. Unused by the flat decoder recipe."""

    entity_id: str
    as_of: datetime
    attributes: Mapping[str, str] = field(default_factory=dict)
    lifelong: tuple[LifelongItem, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", ensure_utc(self.as_of))
        object.__setattr__(self, "attributes", dict(self.attributes))
        object.__setattr__(self, "lifelong", tuple(self.lifelong))


@dataclass(frozen=True)
class TransferEdge:
    """Directed money-flow edge. Optional; unused by the decoder recipe."""

    from_entity: str
    to_entity: str
    timestamp: datetime
    amount: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
