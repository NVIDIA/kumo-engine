# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic synthetic transaction generator (no network, no real PII)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from random import Random

from tfm.contracts.schema import EventLabel, TransactionEvent

STATES = ("CA", "NY", "TX", "FL", "WA", "IL", "ONLINE")
CHIPS = ("SWIPE TRANSACTION", "CHIP TRANSACTION", "ONLINE TRANSACTION")
MCCS = (5411, 5812, 5541, 5311, 5999, 4111, 7011)
MERCHANTS = ("ACME MART", "FUEL STOP", "CAFE 12", "TRANSIT AUTH", "DIGITAL STORE")


def generate_synthetic_events(
    *,
    seed: int = 42,
    n_entities: int = 32,
    events_per_entity: int = 16,
    start: datetime | None = None,
    step_hours: int = 12,
    label_delay: timedelta | None = None,
    fraud_rate: float = 0.15,
) -> list[TransactionEvent]:
    """Generate a tiny, seeded transaction history with delayed fraud labels."""
    rng = Random(seed)
    start_ts = start or datetime(2024, 1, 1, tzinfo=UTC)
    delay = label_delay if label_delay is not None else timedelta(days=7)
    events: list[TransactionEvent] = []
    for entity_idx in range(n_entities):
        entity_id = f"ent-{entity_idx:04d}"
        for event_idx in range(events_per_entity):
            ts = start_ts + timedelta(hours=step_hours * event_idx + (entity_idx % 5))
            amount = round(rng.uniform(3.0, 250.0), 2)
            is_fraud = 1.0 if rng.random() < fraud_rate else 0.0
            if is_fraud:
                amount = round(amount * rng.uniform(2.0, 6.0), 2)
            mcc = rng.choice(MCCS)
            event_id = f"{entity_id}-e{event_idx:04d}"
            events.append(
                TransactionEvent(
                    entity_id=entity_id,
                    event_id=event_id,
                    event_timestamp=ts,
                    timezone="UTC",
                    categorical={
                        "merchant_name": rng.choice(MERCHANTS),
                        "merchant_state": rng.choice(STATES),
                        "use_chip": rng.choice(CHIPS),
                        "mcc": str(mcc),
                    },
                    numeric={
                        "amount": amount,
                        "card": float(entity_idx % 10),
                        "zip": float(10000 + (entity_idx * 17) % 90000),
                        "user": float(entity_idx),
                    },
                    labels={
                        "fraud": EventLabel(
                            name="fraud",
                            value=is_fraud,
                            available_at=ts + delay,
                            level="event",
                        )
                    },
                )
            )
    return events
