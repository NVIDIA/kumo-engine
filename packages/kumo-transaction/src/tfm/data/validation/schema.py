# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema validation for transaction events."""

from __future__ import annotations

from collections.abc import Sequence

from tfm.contracts.errors import SchemaError
from tfm.contracts.schema import TransactionEvent, ensure_utc

PROHIBITED_INPUT_KEYS = frozenset(
    {
        "is_fraud",
        "fraud",
        "chargeback",
        "chargeback_resolved",
        "label",
        "target",
    }
)


def validate_event(event: TransactionEvent) -> TransactionEvent:
    """Reject incomplete or unsafe event records."""
    if not event.entity_id or not str(event.entity_id).strip():
        raise SchemaError("entity_id is required.")
    if not event.event_id or not str(event.event_id).strip():
        raise SchemaError("event_id is required.")
    ensure_utc(event.event_timestamp, event.timezone)

    for key in event.categorical:
        if key.lower() in PROHIBITED_INPUT_KEYS:
            raise SchemaError(
                f"Categorical field {key!r} looks like a label and cannot be a model input."
            )
    for key in event.numeric:
        if key.lower() in PROHIBITED_INPUT_KEYS:
            raise SchemaError(
                f"Numeric field {key!r} looks like a label and cannot be a model input."
            )
    for key, label in event.labels.items():
        if label.name != key:
            raise SchemaError(f"Label map key {key!r} does not match label.name {label.name!r}.")
    return event


def validate_events(events: Sequence[TransactionEvent]) -> list[TransactionEvent]:
    if not events:
        raise SchemaError("At least one event is required.")
    return [validate_event(event) for event in events]
