"""Calculations on entity values: arithmetic, `log1p`, absolute values, signs and calendars."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import Values
if TYPE_CHECKING:

    pass


_BOUNDS = {f"{sign}int{width}": ((0, (1 << width) - 1) if sign else (-(1 << (width - 1)), (1 << (width - 1)) - 1))
           for sign in ("", "u") for width in (8, 16, 32, 64)}

_INT64_MIN, _INT64_MAX = _BOUNDS["int64"]

def _arithmetic(operation: str, operands: list[Values], dtype: Any) -> tuple[Any, Any]:
    """add/subtract/multiply in the result type: (result, rows upstream refuses or None)."""

    import cupy

    a, b = operands[0].values, operands[1].values
    apply = {"add": lambda x, y: x + y, "subtract": lambda x, y: x - y, "multiply": lambda x, y: x * y}[operation]
    if dtype.name.startswith("float"):  # upstream: operands normalized to the result type, then float64 math
        if dtype.name == "float32":
            a, b = (x.astype(cupy.float64).astype(cupy.float32) for x in (a, b))
        result = apply(a.astype(cupy.float64), b.astype(cupy.float64))
        return result.astype(cupy.dtype(dtype.name)), None
    low, high = _BOUNDS[dtype.name]
    bad = _outside(operands[0], low, high) | _outside(operands[1], low, high)
    if dtype.name == "uint64":
        x, y = a.astype(cupy.uint64), b.astype(cupy.uint64)
        result = apply(x, y)
        if operation == "add":
            bad |= result < x
        elif operation == "subtract":
            bad |= x < y
        else:
            nonzero = x != 0
            safe = cupy.where(nonzero, x, cupy.uint64(1))
            bad |= nonzero & ((result % safe != 0) | (result // safe != y))
        return result, bad
    if dtype.name == "int64":
        x, y = a.astype(cupy.int64), b.astype(cupy.int64)
        result = apply(x.view(cupy.uint64), y.view(cupy.uint64)).view(cupy.int64)  # wraps, then checked
        if operation == "add":
            bad |= ((x ^ result) & (y ^ result)) < 0
        elif operation == "subtract":
            bad |= ((x ^ y) & (x ^ result)) < 0
        else:
            minus_one = x == -1
            plain = (x != 0) & ~minus_one
            safe = cupy.where(plain, x, 1)
            bad |= (plain & ((result % safe != 0) | (result // safe != y))) | (minus_one & (y == _INT64_MIN))
        return result, bad
    # narrower results: exact in 64 bits (an unsigned product in uint64), then checked against the range
    wide = cupy.uint64 if operation == "multiply" and dtype.name.startswith("u") else cupy.int64
    result = apply(a.astype(wide), b.astype(wide))
    bad |= (result > high) if wide == cupy.uint64 else (result < low) | (result > high)
    return result.astype(cupy.dtype(dtype.name)), bad

def _outside(operand: Values, low: int, high: int) -> Any:
    """Which integer operand values do not fit [low, high]."""

    import cupy

    values = operand.values
    source_low, source_high = _BOUNDS[operand.dtype.name]
    if low <= source_low and source_high <= high:
        return cupy.zeros(values.size, dtype=bool)
    if values.dtype == cupy.uint64:
        return values > cupy.uint64(high)
    wide = values.astype(cupy.int64)
    outside = wide < low
    return outside | (wide > high) if high <= _INT64_MAX else outside

def _divide(operands: list[Values]) -> tuple[Any, Any]:
    """float64 division: (result, rows that are NULL: a zero divisor, -0.0 included)."""

    import cupy

    zero = operands[1].values == 0
    divisor = cupy.where(zero, 1.0, operands[1].values.astype(cupy.float64))
    return operands[0].values.astype(cupy.float64) / divisor, zero

def _log1p(operand: Values) -> tuple[Any, Any]:
    """float64 log1p: (result, rows that are NULL: at or below -1; NaN stays NaN)."""

    import cupy

    outside = operand.values <= -1
    return cupy.log1p(cupy.where(outside, 0.0, operand.values.astype(cupy.float64))), outside

def _absolute(operand: Values, dtype: Any) -> tuple[Any, Any]:
    import cupy

    values = operand.values
    if dtype.name.startswith("float"):
        return cupy.abs(values.astype(cupy.float64)).astype(cupy.dtype(dtype.name)), None
    if operand.dtype.name.startswith("u"):
        return values.astype(cupy.dtype(dtype.name)), None
    result = cupy.abs(values.astype(cupy.int64))  # the smallest int64 stays negative: refused below
    _, high = _BOUNDS[dtype.name]
    return result.astype(cupy.dtype(dtype.name)), (result < 0) | (result > high)

def _sign(operand: Values) -> tuple[Any, Any]:
    import cupy

    values = operand.values
    result = (values > 0).astype(cupy.int64) - (values < 0).astype(cupy.int64)
    return result, (cupy.isnan(values) if values.dtype.kind == "f" else None)

def _calendar(part: str, operand: Values) -> Any:
    """One UTC calendar part of timestamps (seconds) or dates (days), as int64 (Monday is weekday 0)."""

    import cupy

    stored = operand.values.astype(cupy.int64)
    seconds = stored * 86_400 if operand.dtype.name == "date32" else stored
    days = cupy.floor_divide(seconds, 86_400)
    within = seconds - days * 86_400
    if part == "hour":
        return within // 3_600
    if part == "minute":
        return within % 3_600 // 60
    if part == "second":
        return within % 60
    if part == "weekday":
        return (days + 3) % 7  # 1970-01-01 was a Thursday
    # civil date from days since 1970-01-01 (proleptic Gregorian)
    z = days + 719_468
    era = cupy.floor_divide(z, 146_097)
    day_of_era = z - era * 146_097
    year_of_era = (day_of_era - day_of_era // 1_460 + day_of_era // 36_524 - day_of_era // 146_096) // 365
    day_of_year = day_of_era - (365 * year_of_era + year_of_era // 4 - year_of_era // 100)
    shifted_month = (5 * day_of_year + 2) // 153
    if part == "day":
        return day_of_year - (153 * shifted_month + 2) // 5 + 1
    month = cupy.where(shifted_month < 10, shifted_month + 3, shifted_month - 9)
    if part == "month":
        return month
    return year_of_era + era * 400 + (month <= 2)

def _seconds(dtype: Any, values: Any) -> Any:
    """Clock values as whole UTC seconds (a date's midnight)."""

    import cupy

    if dtype.name == "date32":
        return values.astype(cupy.int64) * 86_400
    return values.astype(cupy.int64)
