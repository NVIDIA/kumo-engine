# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Label windows: which events' labels a sequence-level label may count (probe split rules).

A sequence's history can reach far back past the period it is scored on. If
its label counted every labelled event in the history, the same fraud event
would make an entity positive on both sides of an out-of-time probe. A
`LabelWindow` restricts the count to one period. Events before the window stay
in the history as inputs; only their labels are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from tfm.contracts.schema import ensure_utc


@dataclass(frozen=True)
class LabelWindow:
    """The half-open interval `(start, end]` of event times whose labels count.

    `start=None` means unbounded below. The window is passed explicitly and
    written to `split_manifest.json`; it is never inferred from a sequence.
    """

    start: datetime | None
    end: datetime

    def __post_init__(self) -> None:
        if self.start is not None and ensure_utc(self.start) >= ensure_utc(self.end):
            raise ValueError(
                f"LabelWindow start {self.start.isoformat()} must precede end "
                f"{self.end.isoformat()}."
            )

    def contains(self, timestamp: datetime) -> bool:
        ts = ensure_utc(timestamp)
        if ts > ensure_utc(self.end):
            return False
        return self.start is None or ts > ensure_utc(self.start)

    def as_dict(self) -> dict[str, str | None]:
        return {
            "start_exclusive": None if self.start is None else ensure_utc(self.start).isoformat(),
            "end_inclusive": ensure_utc(self.end).isoformat(),
        }


def side_label_windows(
    train_cutoff: datetime,
    validation_cutoff: datetime,
    test_cutoff: datetime,
    observation_window: timedelta,
) -> dict[str, LabelWindow]:
    """Each side's label window: the train lookback, then back to back up to the test cutoff."""
    return {
        "train": LabelWindow(train_cutoff - observation_window, train_cutoff),
        "validation": LabelWindow(train_cutoff, validation_cutoff),
        "test": LabelWindow(validation_cutoff, test_cutoff),
    }
