# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Windowed pretraining corpora must cover history without crossing the cutoff."""

from datetime import UTC, datetime, timedelta

import pytest

from tfm.contracts.schema import TransactionEvent
from tfm.data.sequence_builders.entity_sequences import (
    build_entity_sequences,
    build_entity_windows,
)

BASE = datetime(2020, 1, 1, tzinfo=UTC)


def _events(entity: str, n: int, start: int = 0) -> list[TransactionEvent]:
    return [
        TransactionEvent(
            entity_id=entity,
            event_id=f"{entity}-{i}",
            event_timestamp=BASE + timedelta(hours=i),
            timezone="UTC",
            categorical={"merchant_name": f"M{i}"},
            numeric={"amount": float(i)},
            labels={},
        )
        for i in range(start, start + n)
    ]


def test_windows_cover_every_event_exactly_once() -> None:
    events = _events("a", 10)
    cutoff = BASE + timedelta(days=30)

    windows = build_entity_windows(events, cutoff, window_events=4)

    assert [len(w.events) for w in windows] == [4, 4, 2]
    seen = [e.event_id for w in windows for e in w.events]
    assert seen == [e.event_id for e in events]


def test_single_sequence_builder_keeps_only_the_tail() -> None:
    """The contrast the windowing exists to fix."""
    events = _events("a", 10)
    cutoff = BASE + timedelta(days=30)

    tail = build_entity_sequences(events, cutoff, max_events=4)
    windows = build_entity_windows(events, cutoff, window_events=4)

    assert len(tail) == 1
    assert len(tail[0].events) == 4
    assert sum(len(w.events) for w in windows) == 10


def test_windows_never_cross_the_cutoff() -> None:
    events = _events("a", 10)
    cutoff = BASE + timedelta(hours=4)

    windows = build_entity_windows(events, cutoff, window_events=3)

    assert windows
    for window in windows:
        for event in window.events:
            assert event.event_timestamp <= cutoff


def test_stride_controls_overlap() -> None:
    events = _events("a", 8)
    cutoff = BASE + timedelta(days=30)

    overlapping = build_entity_windows(events, cutoff, window_events=4, stride=2)

    assert [w.events[0].event_id for w in overlapping] == ["a-0", "a-2", "a-4"]


def test_windows_are_built_per_entity() -> None:
    events = _events("a", 6) + _events("b", 3)
    cutoff = BASE + timedelta(days=30)

    windows = build_entity_windows(events, cutoff, window_events=4)

    by_entity: dict[str, int] = {}
    for window in windows:
        by_entity[window.entity_id] = by_entity.get(window.entity_id, 0) + 1
        assert len({e.entity_id for e in window.events}) == 1
    assert by_entity == {"a": 2, "b": 1}


def test_observation_window_still_applies() -> None:
    events = _events("a", 10)
    cutoff = BASE + timedelta(hours=9)

    windows = build_entity_windows(
        events, cutoff, window_events=4, observation_window=timedelta(hours=3)
    )

    kept = [e.event_id for w in windows for e in w.events]
    assert kept == ["a-6", "a-7", "a-8", "a-9"]


@pytest.mark.parametrize(("window_events", "stride"), [(0, None), (-1, None), (4, 0), (4, -2)])
def test_invalid_window_parameters_raise(window_events: int, stride: int | None) -> None:
    with pytest.raises(ValueError):
        build_entity_windows(
            _events("a", 4), BASE + timedelta(days=1), window_events=window_events, stride=stride
        )
