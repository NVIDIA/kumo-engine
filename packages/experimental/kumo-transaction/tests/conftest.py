# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime, timedelta, timezone

import pytest

from tfm.contracts.schema import EventLabel, TransactionEvent
from tfm.data.synthetic.generator import generate_synthetic_events


@pytest.fixture(autouse=True)
def _no_local_dataset_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests use generated fixtures, never a book a developer has configured locally."""
    for name in ("TFM_PRAGMATIQ_ROOT", "TFM_TABFORMER_ROOT"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def utc() -> timezone:
    return UTC


@pytest.fixture
def sample_events() -> list[TransactionEvent]:
    return generate_synthetic_events(seed=42, n_entities=8, events_per_entity=10)


def make_event(
    entity: str,
    event_id: str,
    ts: datetime,
    *,
    amount: float = 10.0,
    fraud: float | None = None,
    delay: timedelta = timedelta(0),
) -> TransactionEvent:
    labels = {}
    if fraud is not None:
        labels["fraud"] = EventLabel(
            name="fraud",
            value=fraud,
            available_at=ts + delay,
            level="event",
        )
    return TransactionEvent(
        entity_id=entity,
        event_id=event_id,
        event_timestamp=ts,
        numeric={"amount": amount, "card": 0.0, "zip": 10000.0, "user": 0.0},
        categorical={
            "merchant_name": "ACME",
            "merchant_state": "CA",
            "use_chip": "CHIP TRANSACTION",
            "mcc": "5411",
        },
        labels=labels,
    )
