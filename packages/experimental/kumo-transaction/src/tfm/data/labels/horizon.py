# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Label availability relative to a cutoff time."""

from __future__ import annotations

from datetime import datetime

from tfm.contracts.errors import MissingLabelError
from tfm.contracts.schema import EventLabel, TransactionEvent, ensure_utc


def is_label_available(label: EventLabel, cutoff: datetime) -> bool:
    return ensure_utc(label.available_at) <= ensure_utc(cutoff)


def event_label_at_cutoff(
    event: TransactionEvent,
    name: str,
    cutoff: datetime,
) -> EventLabel:
    """Return `name` if it is available at `cutoff`; otherwise raise."""
    label = event.labels.get(name)
    if label is None:
        raise MissingLabelError(f"Event {event.event_id!r} has no label {name!r}.")
    if not is_label_available(label, cutoff):
        raise MissingLabelError(
            f"Label {name!r} for event {event.event_id!r} is not available until "
            f"{label.available_at.isoformat()} (cutoff {ensure_utc(cutoff).isoformat()})."
        )
    return label


def available_label_value(
    event: TransactionEvent,
    name: str,
    cutoff: datetime,
) -> float | None:
    """Return the label value if available, else None (no exception)."""
    label = event.labels.get(name)
    if label is None or not is_label_available(label, cutoff):
        return None
    return label.value
