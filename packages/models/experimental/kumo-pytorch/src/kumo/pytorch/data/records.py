# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage-independent records passed from history construction to tokenization."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any, TypeAlias

from kumo.pytorch.contract.dtypes import DType, _cutoff_seconds, _timestamp_seconds

Scalar: TypeAlias = str | int | float | bool | bytes | Decimal | dt.date | dt.datetime


@dataclass(frozen=True)
class FieldValue:
    key: str
    value: Scalar | None
    dtype: DType


@dataclass(frozen=True)
class DatedValue:
    key: str
    value: Scalar | None
    dtype: DType
    timestamp: int
    fields: tuple[FieldValue, ...] = ()


@dataclass(frozen=True)
class Event:
    source: str
    timestamp: int
    fields: tuple[FieldValue, ...]


@dataclass(frozen=True)
class EntityState:
    """Scalar and dated values describing an entity at one sample time."""

    values: tuple[FieldValue, ...]
    dated_values: tuple[DatedValue, ...]


@dataclass(frozen=True)
class EntityHistory:
    """Point-in-time values and events for one entity.

    ``entity_type`` describes what owns the sequence (for example ``user`` or
    ``card_account``). The events may still refer to other related entities.
    """

    entity_id: int
    evaluation_time: int
    entity: EntityState
    events: tuple[Event, ...]
    entity_type: str = "user"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EntityHistory:
        entity = payload["entity"]
        return cls(
            entity_id=int(payload["entity_id"]),
            evaluation_time=int(payload["evaluation_time"]),
            entity=EntityState(
                values=tuple(
                    FieldValue(item["key"], item["value"], DType(**item["dtype"]))
                    for item in entity["values"]
                ),
                dated_values=tuple(
                    DatedValue(
                        item["key"],
                        item["value"],
                        DType(**item["dtype"]),
                        int(item["timestamp"]),
                        tuple(
                            FieldValue(field["key"], field["value"], DType(**field["dtype"]))
                            for field in item.get("fields", ())
                        ),
                    )
                    for item in entity["dated_values"]
                ),
            ),
            events=tuple(
                Event(
                    source=item["source"],
                    timestamp=int(item["timestamp"]),
                    fields=tuple(
                        FieldValue(field["key"], field["value"], DType(**field["dtype"]))
                        for field in item["fields"]
                    ),
                )
                for item in payload["events"]
            ),
            entity_type=str(payload.get("entity_type", "user")),
        )


def timestamp_to_epoch(value: Any) -> int:
    if isinstance(value, dt.date):
        return _timestamp_seconds(value)
    return int(value)


def cutoff_to_epoch(value: Any) -> int:
    if isinstance(value, dt.date):
        return _cutoff_seconds(value)
    return int(value)


def parse_utc_timestamp(value: str) -> int:
    normalized = value.replace("Z", "+00:00")
    parsed = dt.datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return timestamp_to_epoch(parsed)
