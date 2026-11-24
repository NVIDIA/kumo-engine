# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema validation for transaction events."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from tfm.contracts.errors import SchemaError
from tfm.contracts.schema import TransactionEvent, ensure_utc
from tfm.contracts.tokenizers import FIELD_CATEGORICAL, FIELD_NUMERIC, FIELD_TEXT, EventBlock

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

    _reject_label_keys("Categorical", event.categorical)
    _reject_label_keys("Numeric", event.numeric)
    _reject_label_keys("Text", event.text)
    for key, label in event.labels.items():
        if label.name != key:
            raise SchemaError(f"Label map key {key!r} does not match label.name {label.name!r}.")
    return event


def validate_event_block(block: EventBlock) -> None:
    """Reject a block whose categorical, numeric or text fields include a label-like key,
    as `validate_event` does for each event."""
    for kind, name in (
        (FIELD_CATEGORICAL, "Categorical"),
        (FIELD_NUMERIC, "Numeric"),
        (FIELD_TEXT, "Text"),
    ):
        used = np.bincount(block.pair_key[block.pair_kind == kind], minlength=len(block.keys))
        _reject_label_keys(name, [block.keys[k] for k in np.flatnonzero(used).tolist()])


def _reject_label_keys(kind: str, keys: Iterable[str]) -> None:
    for key in keys:
        if key.strip().lower() in PROHIBITED_INPUT_KEYS:
            raise SchemaError(
                f"{kind} field {key!r} looks like a label and cannot be a model input."
            )


def validate_events(events: Sequence[TransactionEvent]) -> list[TransactionEvent]:
    if not events:
        raise SchemaError("At least one event is required.")
    return [validate_event(event) for event in events]
