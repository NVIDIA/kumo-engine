# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Apply-time conversion of temporal column values to a canonical epoch domain.

Converts a timestamp/date column into epoch seconds (for ``timestamp``) or epoch
days (for ``date``). Every value-level decision -- which format parses, which
epoch unit a magnitude implies, what a format means -- is DiskGraph's
``resolve_temporal``, the same code ingest runs, so the stats computed here match
the on-disk index by construction rather than by two implementations agreeing. A
format string is an opaque token here: stored, hashed and displayed, never
interpreted.

What remains on this side is the standardized output model:
  * a timestamp collapses to an ``i64`` epoch-seconds value (UTC);
  * a date collapses to an ``i32`` days-since-epoch value.
Sub-second precision and timezone are never retained, and everything is computed
in the integer epoch-seconds / epoch-days domain; only
:func:`epoch_seconds_to_timestamp` lifts the final scalar back into a
``pd.Timestamp`` for the stats value.

The stats path is best-effort: an out-of-range or unconvertible value becomes
``None`` and is dropped from the MIN/MAX reduction rather than raising.
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from diskgraph import resolve_temporal
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
)

from kumo.connector.timestamp_infer import HintKind, TemporalHint, _as_arrow

# Valid range for a Date32 day value at apply time.
# Lower: pandas Timestamp.min (~1677-09-21 = day -106752).
# Upper: i32 epoch-seconds overflow (~2038-01-19 = day 24855).
DATE32_MIN_DAY = -106752
DATE32_MAX_DAY = 24855

_NATIVE_UNIT_DIVISOR = {'s': 1, 'ms': 1_000, 'us': 1_000_000, 'ns': 1_000_000_000}


# ---------------------------------------------------------------------------
# Core integer converters.
# ---------------------------------------------------------------------------
def div_ceil(value: int, divisor: int) -> int:
    r"""Ceiling division toward positive infinity.

    Sub-second timestamps round UP so a value is never assigned to an earlier
    second, which would leak future data into temporal aggregation.

    Sign behavior:
      * 1500 ms with divisor 1000 -> 2 (rounds up).
      * -1500 ms with divisor 1000 -> -1 (rounds toward +inf, not toward zero).

    ``value`` must be a Python int (not a fixed-width numpy int64) so the minimum
    int64 magnitude is handled without overflow.
    """
    d = value // divisor
    r = value % divisor
    return d + 1 if r > 0 else d


def timestamp_from_int(value: int, unit: Optional[str]) -> int:
    r"""Convert an epoch integer in ``unit`` to epoch seconds.

    A ``None`` unit defaults to seconds: the value is interpreted as epoch seconds
    and returned unchanged, so a temporal integer column with no resolved unit is
    read as epoch seconds rather than an error or a guess.
    """
    if unit is None or unit == 's':
        return value
    divisor = _NATIVE_UNIT_DIVISOR.get(unit)
    if divisor is None:
        raise ValueError(f"unsupported timestamp unit: {unit!r}")
    return div_ceil(value, divisor)


def days_from_secs(secs: int) -> int:
    r"""Convert epoch seconds to epoch days.

    Also ceiling: an epoch-seconds value that is not an exact multiple of 86400
    rounds UP to the next day. For example ``days_from_secs(-1)`` is ``0``, not
    ``-1``.
    """
    return div_ceil(secs, 86_400)


# ---------------------------------------------------------------------------
# The DiskGraph crossing: one call per column, no per-value Python work.
# ---------------------------------------------------------------------------
def _resolve(series: pd.Series, hint: TemporalHint, *,
             days: bool) -> np.ndarray:
    r"""Apply ``hint`` to a whole column, returning epoch seconds or epoch days.

    An ``UNRESOLVED`` column declares seconds rather than declaring nothing: a
    declared hint is what stops DiskGraph re-detecting the column at apply time.
    """
    unit = hint.unit or ('s' if hint.kind is HintKind.UNRESOLVED else None)
    epochs, _, _ = resolve_temporal(_as_arrow(series), hint.strftime, unit, days)
    out = np.empty(len(epochs), dtype=object)
    out[:] = epochs
    return out


# ---------------------------------------------------------------------------
# Date32 apply-time conversion.
# ---------------------------------------------------------------------------
def date32_to_epoch_seconds(days: Optional[int]) -> Optional[int]:
    r"""Convert a Date32 day value to epoch seconds.

    In-range arithmetic is ``days * 86400`` (a plain multiply, not ceiling, since
    days are already integral); the valid range is ``[-106752, 24855]`` days. An
    out-of-range day returns ``None`` so the caller drops the value from the
    MIN/MAX reduction rather than raising.
    """
    if days is None:
        return None
    days = int(days)
    if not (DATE32_MIN_DAY <= days <= DATE32_MAX_DAY):
        return None
    return days * 86_400


# ---------------------------------------------------------------------------
# Native Arrow/pandas temporal conversion.
# ---------------------------------------------------------------------------
def _as_naive_utc_datetime_series(series: pd.Series) -> pd.Series:
    r"""Coerce to a tz-naive datetime64 series in UTC.

    A tz-aware column is normalized to UTC so its int64 view is UTC epoch counts.
    """
    s = pd.to_datetime(series)
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_convert('UTC').dt.tz_localize(None)
    return s


def _native_divisor(dtype) -> int:
    unit = np.datetime_data(np.dtype(dtype))[0]
    return _NATIVE_UNIT_DIVISOR.get(unit, 1_000_000_000)


def timestamp_secs_from_array(series: pd.Series) -> np.ndarray:
    r"""Convert a native temporal series to epoch seconds.

    Reads the series' own native unit and ceiling-divides each value to seconds
    (second resolution passes through, ms/us/ns go through the ceiling), so even a
    native timestamp is ceiling-converted. Null (``NaT``) values map to ``None``.
    Returns a numpy object array aligned with the input positions.

    Fully vectorized: the ceiling is ``-((-a) // divisor)`` on the int64 view
    (floor-division floors toward negative infinity, so negating the floor of the
    negation reproduces the scalar ceiling). The only int64 value whose negation
    overflows is the minimum int64, which is exactly the ``NaT`` sentinel; ``NaT``
    is masked out and mapped to ``None`` before any negation, so a null never
    becomes a bogus epoch.
    """
    s = _as_naive_utc_datetime_series(series)
    divisor = _native_divisor(s.dtype)
    ints = s.to_numpy().view('int64')
    isna = pd.isna(s).to_numpy()
    # An object dtype array initializes to None, which is the NaT/null result.
    out = np.empty(len(s), dtype=object)
    computable = ~isna
    if not computable.any():
        return out
    sub = ints[computable]  # excludes NaT (the minimum-int64 sentinel)
    if divisor == 1:
        secs = sub  # Second resolution: already whole epoch seconds
    else:
        secs = -((-sub) // divisor)
    out[computable] = secs.tolist()  # store Python ints, leave nulls as None
    return out


# ---------------------------------------------------------------------------
# Vectorized apply + MIN/MAX.
# ---------------------------------------------------------------------------
def to_epoch_seconds(series: pd.Series, hint: TemporalHint) -> np.ndarray:
    r"""Vectorized apply-time conversion of a column to epoch seconds.

    An already-typed temporal column carries its unit on its dtype and needs no
    parsing; everything else crosses into DiskGraph once. Best-effort: an
    unconvertible or null value becomes ``None``, never raising. Returns a numpy
    object array of ints and ``None`` aligned with the input.
    """
    if hint.kind is HintKind.NATIVE or is_datetime64_any_dtype(series):
        return timestamp_secs_from_array(series)
    return _resolve(series, hint, days=False)


def to_epoch_days(series: pd.Series, hint: TemporalHint) -> np.ndarray:
    r"""Vectorized apply-time conversion of a column to epoch days.

    Best-effort: unconvertible or null values become ``None``.
    """
    if hint.kind is HintKind.NATIVE or is_datetime64_any_dtype(series):
        secs = timestamp_secs_from_array(series)
        return np.array(
            [None if x is None else days_from_secs(int(x)) for x in secs],
            dtype=object)
    return _resolve(series, hint, days=True)


def min_max_epoch(series: pd.Series,
                  hint: TemporalHint) -> Tuple[Optional[int], Optional[int]]:
    r"""Return ``(min, max)`` epoch seconds for a column, or ``(None, None)``.

    Uses the monotonic fast path for ``UNIT`` and ``NATIVE``: take the raw min/max
    first, then convert only those two scalars, because the conversion is
    monotonic in the raw value. Otherwise the raw order is not the temporal order,
    so it converts then reduces.
    """
    if hint.kind is HintKind.NATIVE:
        return _min_max_native(series)

    numeric = is_integer_dtype(series) or is_float_dtype(series)
    if hint.kind is HintKind.UNIT and numeric:
        s = series.dropna()
        if s.empty:
            return (None, None)
        return (timestamp_from_int(int(s.min()), hint.unit),
                timestamp_from_int(int(s.max()), hint.unit))

    return _reduce_min_max(to_epoch_seconds(series, hint))


def _min_max_native(
        series: pd.Series) -> Tuple[Optional[int], Optional[int]]:
    s = _as_naive_utc_datetime_series(series).dropna()
    if s.empty:
        return (None, None)
    arr = s.to_numpy()
    divisor = _native_divisor(arr.dtype)
    raw_min = int(arr.min().astype('int64'))
    raw_max = int(arr.max().astype('int64'))
    return (div_ceil(raw_min, divisor), div_ceil(raw_max, divisor))


def _reduce_min_max(
        converted: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    ints: List[int] = [int(x) for x in converted if x is not None]
    if not ints:
        return (None, None)
    return (min(ints), max(ints))


def epoch_seconds_to_timestamp(secs: Optional[int]) -> pd.Timestamp:
    r"""Wrap an epoch-seconds scalar into a ``pd.Timestamp`` for the stats value.

    This is the only place the integer epoch domain is lifted back into a pandas
    timestamp.
    An out-of-bounds value (outside the pandas datetime range) becomes ``NaT``
    rather than raising, consistent with the best-effort stats contract.
    """
    if secs is None:
        return pd.NaT
    try:
        return pd.Timestamp(int(secs), unit='s')
    except (OverflowError, ValueError, pd.errors.OutOfBoundsDatetime):
        return pd.NaT
