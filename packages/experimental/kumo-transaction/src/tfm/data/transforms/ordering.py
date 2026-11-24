# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic event ordering."""

from __future__ import annotations

from collections.abc import Sequence

from tfm.contracts.schema import TransactionEvent, ensure_utc


def order_events(events: Sequence[TransactionEvent]) -> list[TransactionEvent]:
    """Sort by timestamp, then event_id so ties are stable."""
    return sorted(
        events,
        key=lambda event: (ensure_utc(event.event_timestamp), event.event_id),
    )
