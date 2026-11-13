# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serializable scalar types and checked values."""

from __future__ import annotations

import datetime as dt
import struct
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

_INTEGER_WIDTHS = {
    **{f"int{width}": (width, True) for width in (8, 16, 32, 64)},
    **{f"uint{width}": (width, False) for width in (8, 16, 32, 64)},
}
_SCALARS = frozenset(
    (
        *_INTEGER_WIDTHS,
        "bool",
        "float32",
        "float64",
        "string",
        "bytes",
        "date32",
        "timestamp",
        "decimal128",
        "decimal256",
    )
)


def _since_epoch(value: dt.date | dt.datetime) -> dt.timedelta:
    if not isinstance(value, dt.datetime):
        value = dt.datetime.combine(value, dt.time(), tzinfo=dt.UTC)
    elif value.tzinfo is None:
        value = value.replace(tzinfo=dt.UTC)
    else:
        value = value.astimezone(dt.UTC)
    return value - dt.datetime(1970, 1, 1, tzinfo=dt.UTC)


def _timestamp_seconds(value: dt.date | dt.datetime) -> int:
    """Native stored clocks round fractional instants toward positive infinity."""
    delta = _since_epoch(value)
    return delta.days * 86_400 + delta.seconds + int(delta.microseconds != 0)


def _cutoff_seconds(value: dt.date | dt.datetime) -> int:
    """Request cutoffs round toward negative infinity, so no later event can share their second."""
    delta = _since_epoch(value)
    return delta.days * 86_400 + delta.seconds


@dataclass(frozen=True)
class DType:
    """The stored scalar type, independent of its eventual model encoding.

    Supported names: ``bool``; ``int8``, ``int16``, ``int32``, ``int64``;
    ``uint8``, ``uint16``, ``uint32``, ``uint64``; ``float32``, ``float64``;
    ``string``, ``bytes``; ``date32``; ``timestamp`` (``unit="s"|"ms"|"us"|"ns"``,
    ``timezone="UTC"``); ``decimal128`` (precision up to 38) and ``decimal256``
    (precision up to 76), which require ``precision`` and ``scale``. No
    parameters are accepted for other families. Arrays, lists and structs are
    not scalar column types. NULL is allowed for every type; it is not a dtype.

    The type describes stored values, not an encoding: values are validated,
    never converted. Numeric strings are not parsed, numbers are not
    stringified, integers are not booleans, and decimals are not rounded to fit;
    integer bounds and decimal precision/scale are checked.

    Timestamp precision is declared explicitly. The default ``unit="s"``
    preserves whole-second UTC instants; finer declared units retain their ticks.
    In the seconds mode, finer physical units round toward
    positive infinity, including before 1970; timezone metadata does not change
    the instant, and a naive ``datetime`` means UTC. Anchor times
    (``SampleRequest.as_of_time``) instead round toward negative infinity, so an
    event later in the anchor's second is never history. ``date32`` stays a
    date. Plain integer anchor times are always epoch seconds, independently
    of the graph clock; a typed Arrow timestamp column carries its physical
    unit. Integers supplied as timestamp *field values* instead use that
    field's declared ticks. Finer physical timestamp columns are accepted
    during ingestion and converted to the declared graph clock; timestamps
    round upward to that clock's tick, while finer anchor cutoffs are refused
    for nonsecond graph clocks.

    :raises ValueError: for an unknown name, missing or out-of-range decimal
        parameters, or parameters on a family that takes none.
    """

    name: str
    unit: str | None = None
    timezone: str | None = None
    precision: int | None = None
    scale: int | None = None

    def __post_init__(self) -> None:
        if self.name not in _SCALARS:
            raise ValueError(f"unsupported scalar dtype {self.name!r}")
        if self.name == "timestamp":
            if self.unit not in {"s", "ms", "us", "ns"} or self.timezone != "UTC":
                raise ValueError("timestamp dtype requires unit in s/ms/us/ns and timezone='UTC'")
        elif self.unit is not None or self.timezone is not None:
            raise ValueError(f"{self.name} does not accept time parameters")
        if self.name.startswith("decimal"):
            limit = 38 if self.name == "decimal128" else 76
            if (
                type(self.precision) is not int
                or not 1 <= self.precision <= limit
                or type(self.scale) is not int
                or not -self.precision <= self.scale <= self.precision
            ):
                raise ValueError(
                    f"{self.name} requires precision in [1, {limit}] and a valid scale"
                )
        elif self.precision is not None or self.scale is not None:
            raise ValueError(f"{self.name} does not accept decimal parameters")

    @property
    def integer(self) -> bool:
        """Whether this is a signed or unsigned integer type."""
        return self.name in _INTEGER_WIDTHS

    @property
    def numeric(self) -> bool:
        """Whether ``ZScore`` and ``Buckets`` accept it: integers, floats and decimals."""
        return self.integer or self.name in {"float32", "float64", "decimal128", "decimal256"}

    @property
    def temporal(self) -> bool:
        """Whether it can be compared with an anchor time: ``timestamp`` or ``date32``."""
        return self.name in {"timestamp", "date32"}

    def normalize(self, value: Any, *, path: str = "value") -> Any:
        """Validate a scalar; only the declared storage normalization is applied."""

        if value is None:
            return None
        error = f"{path}: {value!r} is not a valid {self} value"
        if self.integer:
            width, signed = _INTEGER_WIDTHS[self.name]
            minimum = -(1 << (width - 1)) if signed else 0
            maximum = (1 << (width - int(signed))) - 1
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(error)
            return value
        if self.name == "bool":
            if type(value) is not bool:
                raise ValueError(error)
            return value
        if self.name.startswith("float"):
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(error)
            try:
                result = float(value)
                if self.name == "float32":
                    result = struct.unpack("!f", struct.pack("!f", result))[0]
            except (OverflowError, struct.error) as exc:
                raise ValueError(error) from exc
            return result
        if self.name == "string":
            if not isinstance(value, str):
                raise ValueError(error)
            return value
        if self.name == "bytes":
            if not isinstance(value, bytes):
                raise ValueError(error)
            return value
        if self.name == "date32":
            if not isinstance(value, dt.date) or isinstance(value, dt.datetime):
                raise ValueError(error)
            return value
        if self.name == "timestamp":
            if type(value) is not int and not isinstance(value, dt.datetime):
                raise ValueError(error)
            try:
                if self.unit != "s":
                    import pyarrow as pa

                    return pa.scalar(value, type=self.to_arrow()).as_py()
                seconds = value if type(value) is int else _timestamp_seconds(value)
                return dt.datetime(1970, 1, 1, tzinfo=dt.UTC) + dt.timedelta(seconds=seconds)
            except (OverflowError, ValueError) as exc:
                raise ValueError(error) from exc
        if not isinstance(value, Decimal) or not value.is_finite():
            raise ValueError(error)
        with localcontext() as context:
            context.prec = max(100, self.precision + abs(self.scale) + 1)
            scaled = value.scaleb(self.scale)
            if scaled != scaled.to_integral_value() or abs(scaled) >= 10**self.precision:
                raise ValueError(error)
        return value

    def to_arrow(self) -> Any:
        """Return the corresponding Arrow type without importing Arrow at startup."""

        import pyarrow as pa

        if self.name == "timestamp":
            return pa.timestamp(self.unit, tz=self.timezone)
        if self.name.startswith("decimal"):
            return getattr(pa, self.name)(self.precision, self.scale)
        return {"bytes": pa.binary, "string": pa.string, "bool": pa.bool_}.get(
            self.name, getattr(pa, self.name, None)
        )()

    @classmethod
    def from_arrow(cls, value: Any) -> DType:
        """Resolve supported physical scalar metadata to its stored type."""

        import pyarrow as pa

        if pa.types.is_timestamp(value):
            return cls("timestamp", unit="s", timezone="UTC")
        if pa.types.is_decimal(value):
            return cls(f"decimal{value.bit_width}", precision=value.precision, scale=value.scale)
        if pa.types.is_string(value) or pa.types.is_large_string(value):
            return cls("string")
        if pa.types.is_binary(value) or pa.types.is_large_binary(value):
            return cls("bytes")
        if pa.types.is_boolean(value):
            return cls("bool")
        if pa.types.is_date32(value):
            return cls("date32")
        name = str(value)
        if name in _SCALARS:
            return cls(name)
        if pa.types.is_float32(value):
            return cls("float32")
        if pa.types.is_float64(value):
            return cls("float64")
        raise ValueError(f"unsupported physical scalar dtype {value}")


def arrow_float64(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Cast to float64, rounding decimals exactly as ``float(Decimal)`` does.

    Arrow's direct decimal cast is not correctly rounded; its decimal text parse is.
    """

    import pyarrow as pa
    import pyarrow.compute as pc

    if pa.types.is_decimal(values.type):
        values = pc.cast(values, pa.string())
    return pc.cast(values, pa.float64(), safe=False)
