# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The clock that times are counted in: whole ticks of one unit for the whole schema."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Literal

import pyarrow as pa

from kumo.pytorch.contract.dtypes import _since_epoch
from kumo.pytorch.contract.schema import DataSchema

__all__ = ["SECONDS", "Clock", "ClockUnit"]

ClockUnit = Literal["seconds", "milliseconds", "microseconds", "nanoseconds"]

_UNITS: dict[str, ClockUnit] = {
    "s": "seconds",
    "ms": "milliseconds",
    "us": "microseconds",
    "ns": "nanoseconds",
}
_TICK_NANOS: dict[ClockUnit, int] = {
    "seconds": 10**9,
    "milliseconds": 10**6,
    "microseconds": 10**3,
    "nanoseconds": 1,
}


@dataclass(frozen=True)
class Clock:
    """Times as integer ticks of ``unit``, rounded as the DiskGraph index rounds them.

    Stored times round up to a tick and request cutoffs round down, so no
    row inside a cutoff's tick is later than the cutoff. An integer is epoch seconds.
    """

    unit: ClockUnit = "seconds"

    @classmethod
    def of(cls, schema: DataSchema) -> Clock:
        """The finest timestamp unit the schema declares; seconds when it declares none."""

        units = [
            dtype.unit
            for table in schema.tables
            for dtype in table.columns.values()
            if dtype.name == "timestamp"
        ]
        return cls(_UNITS[max(units, key=list(_UNITS).index)] if units else "seconds")

    @property
    def tick_nanos(self) -> int:
        return _TICK_NANOS[self.unit]

    def stored(self, value: Any) -> int:
        """Ticks of a stored time, rounded toward positive infinity."""

        return -(-_epoch_nanos(value) // self.tick_nanos)

    def cutoff(self, value: Any) -> int:
        """Ticks of a request cutoff, rounded toward negative infinity."""

        return _epoch_nanos(value) // self.tick_nanos

    def ticks(self, seconds: int) -> int:
        """A whole-second offset, such as a window bound, in ticks."""

        return seconds * (10**9 // self.tick_nanos)

    def seconds(self, ticks: int) -> float:
        """Elapsed ticks as seconds."""

        if self.unit == "seconds":
            return float(ticks)
        return float(ticks) * self.tick_nanos / 1_000_000_000


SECONDS = Clock()


def _epoch_nanos(value: Any) -> int:
    if isinstance(value, pa.TimestampScalar):
        return value.cast(pa.timestamp("ns", tz=value.type.tz)).value
    if isinstance(value, dt.date):
        delta = _since_epoch(value)
        return (
            (delta.days * 86_400 + delta.seconds) * 10**9
            + delta.microseconds * 1_000
            + getattr(value, "nanosecond", 0)
        )
    return int(value) * 10**9
