# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Tests for the timestamp/date converter.

Covers the epoch-domain arithmetic kumo still owns (``div_ceil`` /
``timestamp_from_int`` / ``days_from_secs`` / ``timestamp_secs_from_array`` /
Date32), and the DiskGraph-backed apply path: the exhaustive
format-x-value matrix, the vectorized entry points, and MIN/MAX.

All expected epoch values are fixed literals; nothing depends on ``now()``,
randomness, or the machine timezone.
"""
import datetime

import numpy as np
import pandas as pd
import pytest
from diskgraph import TIMESTAMP_FORMATS

from kumo.connector import timestamp_convert as tc
from kumo.connector.timestamp_convert import HintKind, TemporalHint

_EPOCH_DATE = datetime.date(1970, 1, 1)


def _hint(fmt):
    return (TemporalHint.of_format(fmt) if fmt is not None
            else TemporalHint.unresolved())


def _secs(values, fmt):
    return list(tc.to_epoch_seconds(pd.Series(values, dtype='string'),
                                    _hint(fmt)))


def _epoch_days(values, fmt):
    return list(tc.to_epoch_days(pd.Series(values, dtype='string'),
                                 _hint(fmt)))


def _days(year, month, day):
    return (datetime.date(year, month, day) - _EPOCH_DATE).days


# ===========================================================================
# Every TIMESTAMP_FORMATS entry x a value set that includes a literal "Z".
#
# kumo#30447 shipped because the offset formats were only ever tested against
# "+0000"/"+00:00"/"-0530", and a "Z" that Python's %z accepts is one chrono's
# rejects. A new format entry with no row here fails
# ``test_matrix_covers_every_format``.
# ===========================================================================
E = 1562198400          # 2019-07-04T00:00:00Z
E530 = 1562238000       # 2019-07-04T05:30:00-05:30
APR7 = 1554595200       # 2019-04-07, the %d/%m reading of "07/04/2019"

_VALUES = [
    '2019-07-04',
    '2019-07-04 00:00',
    '2019-07-04 00:00:00',
    '2019-07-04T00:00:00',
    '2019-07-04 00:00:00Z',
    '2019-07-04T00:00:00Z',
    '2019-07-04 00:00:00+0000',
    '2019-07-04T00:00:00+00:00',
    '2019-07-04 05:30:00 -05:30',
    '07/04/2019',
    '07/04/2019 00:00',
    '07/04/2019 00:00:00',
    '2019/07/04',
    '2019/07/04 00:00',
    '2019/07/04 00:00:00',
]

# ``None`` means the format must not parse that value.
_MATRIX = {
    '%Y-%m-%d':
        [E, None, None, None, None, None, None, None, None, None, None, None, E, None, None],
    '%Y-%m-%d %H:%M:%S':
        [None, None, E, E, None, None, None, None, None, None, None, None, None, None, E],
    '%Y-%m-%d %H:%M:%S%.f':
        [None, None, E, None, None, None, None, None, None, None, None, None, None, None, None],
    '%Y-%m-%d %H:%M':
        [None, E, None, None, None, None, None, None, None, None, None, None, None, E, None],
    '%Y-%m-%dT%H:%M:%S':
        [None, None, E, E, None, None, None, None, None, None, None, None, None, None, E],
    '%Y-%m-%dT%H:%M:%S%.f':
        [None, None, None, E, None, None, None, None, None, None, None, None, None, None, None],
    '%Y-%m-%d %H:%M:%S%z':
        [None, None, E, None, E, E, E, E, E530, None, None, None, None, None, None],
    '%Y-%m-%d %H:%M:%S %z':
        [None, None, E, None, E, E, E, E, E530, None, None, None, None, None, None],
    '%Y-%m-%d %H:%M:%S%:z':
        [None, None, E, None, E, E, E, E, E530, None, None, None, None, None, None],
    '%Y-%m-%d %H:%M:%S %:z':
        [None, None, E, None, E, E, E, E, E530, None, None, None, None, None, None],
    '%Y-%m-%dT%H:%M:%S%z':
        [None, None, None, E, E, E, E, E, None, None, None, None, None, None, None],
    '%Y-%m-%dT%H:%M:%S%:z':
        [None, None, None, E, E, E, None, E, None, None, None, None, None, None, None],
    '%Y-%m-%dT%H:%M:%S%#z':
        [None, None, None, E, E, E, None, E, None, None, None, None, None, None, None],
    '%Y-%m-%d %H:%M:%S%#z':
        [None, None, E, None, E, E, E, E, E530, None, None, None, None, None, None],
    '%m/%d/%Y':
        [None, None, None, None, None, None, None, None, None, E, None, None, None, None, None],
    '%m/%d/%Y %H:%M:%S':
        [None, None, None, None, None, None, None, None, None, None, None, E, None, None, None],
    '%m/%d/%Y %H:%M':
        [None, None, None, None, None, None, None, None, None, None, E, None, None, None, None],
    '%d/%m/%Y':
        [None, None, None, None, None, None, None, None, None, APR7, None, None, None, None, None],
    '%d/%m/%Y %H:%M:%S':
        [None, None, None, None, None, None, None, None, None, None, None, APR7, None, None, None],
    '%d/%m/%Y %H:%M':
        [None, None, None, None, None, None, None, None, None, None, APR7, None, None, None, None],
    '%Y/%m/%d':
        [None, None, None, None, None, None, None, None, None, None, None, None, E, None, None],
    '%Y/%m/%d %H:%M:%S':
        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, E],
    '%Y/%m/%d %H:%M':
        [None, None, None, None, None, None, None, None, None, None, None, None, None, E, None],
}


def test_matrix_covers_every_format():
    assert set(_MATRIX) == set(TIMESTAMP_FORMATS)


@pytest.mark.parametrize('fmt', list(_MATRIX))
def test_format_value_matrix(fmt):
    assert _secs(_VALUES, fmt) == _MATRIX[fmt]


# ===========================================================================
# Compact encoded-date formats (integer-sourced, never string-detected).
# ===========================================================================
@pytest.mark.parametrize('values, fmt, expected', [
    (['20230408', '19700101'], '%Y%m%d', [1680912000, 0]),
    (['2023040'], '%Y%m%d', [2023040]),              # too short -> parse as int
    (['20230408120530', '20240115000000'], '%Y%m%d%H%M%S',
     [1680955530, 1705276800]),
    # Civil-days arithmetic is non-rejecting: Feb 30 rolls into early March.
    (['20230230', '20230431'], '%Y%m%d',
     [_days(2023, 3, 2) * 86400, _days(2023, 5, 1) * 86400]),
])
def test_compact_formats(values, fmt, expected):
    assert _secs(values, fmt) == expected


# ===========================================================================
# Integer apply-time: format vs unit vs epoch default
# ===========================================================================
@pytest.mark.parametrize('values, hint, expected', [
    ([20240301, 19930715], TemporalHint.of_format('%Y%m%d'),
     [1709251200, 742694400]),
    ([1709251200], TemporalHint.unresolved(), [1709251200]),
    ([1709251200000], TemporalHint.of_unit('ms'), [1709251200]),
    # "%Y" / "%Y%m" never build a date, so the value stays an epoch second.
    ([2023], TemporalHint.of_format('%Y'), [2023]),
    ([202304], TemporalHint.of_format('%Y%m'), [202304]),
    # Feb 30 as an integer: the date parse rejects it, epoch seconds win.
    ([20230230, 20230431], TemporalHint.of_format('%Y%m%d'),
     [20230230, 20230431]),
])
def test_int_apply(values, hint, expected):
    ser = pd.Series(values, dtype='int64')
    assert list(tc.to_epoch_seconds(ser, hint)) == expected


def test_int_nat_sentinel_is_null_not_an_epoch():
    # The pandas NaT sentinel must never decode as 1677-09-21.
    ser = pd.Series([1709251200, -(2**63)], dtype='int64')
    assert list(tc.to_epoch_seconds(ser, TemporalHint.of_unit('s'))) == [
        1709251200, None]


def test_float_out_of_range_is_null_and_does_not_poison_the_column():
    ser = pd.Series([1709251200.0, 1e300, np.inf, np.nan], dtype='float64')
    assert list(tc.to_epoch_seconds(ser, TemporalHint.of_unit('s'))) == [
        1709251200, None, None, None]


# ===========================================================================
# Date32 apply-time range clamp (out-of-range -> None)
# ===========================================================================
@pytest.mark.parametrize('day, expected', [
    (19738, 19738 * 86400),        # 2024, in range
    (24855, 24855 * 86400),        # inclusive upper bound (~2038-01-19)
    (-106752, -106752 * 86400),    # inclusive lower bound (~1677-09-21)
    (24856, None),                 # just past upper bound
    (-106753, None),               # just past lower bound
    (None, None),
])
def test_date32_to_epoch_seconds(day, expected):
    assert tc.date32_to_epoch_seconds(day) == expected


# ===========================================================================
# Converter core: div_ceil, timestamp_from_int, days_from_secs
# ===========================================================================
@pytest.mark.parametrize('value, divisor, expected', [
    (1500, 1000, 2),      # rounds up
    (1000, 1000, 1),      # exact multiple
    (999, 1000, 1),       # rounds up
    (0, 1000, 0),         # zero
    (-1500, 1000, -1),    # ceil toward +inf, not toward zero
    (-1000, 1000, -1),    # exact negative multiple
    (-1, 1000, 0),        # ceil(-0.001) == 0
    (-2500, 1000, -2),
    (86400, 86400, 1),
    (86401, 86400, 2),
    (-86401, 86400, -1),
])
def test_div_ceil(value, divisor, expected):
    assert tc.div_ceil(value, divisor) == expected


@pytest.mark.parametrize('value, unit, expected', [
    (5, None, 5),                 # seconds default
    (5, 's', 5),
    (1500, 'ms', 2),              # sub-second boundary rounds up
    (1000, 'ms', 1),
    (1001, 'ms', 2),
    (-1500, 'ms', -1),            # negative ceil
    (1_500_000, 'us', 2),
    (1_500_000_000, 'ns', 2),
])
def test_timestamp_from_int(value, unit, expected):
    assert tc.timestamp_from_int(value, unit) == expected


def test_timestamp_from_int_bad_unit_raises():
    with pytest.raises(ValueError):
        tc.timestamp_from_int(1, 'weeks')


@pytest.mark.parametrize('secs, expected', [
    (0, 0),
    (1, 1),               # ceil(1/86400) == 1
    (86400, 1),           # exact day
    (86401, 2),           # rounds up to next day
    (-1, 0),              # ceil(-1/86400) == 0
    (-86400, -1),         # exact negative day
    (-86401, -1),         # ceil toward +inf
])
def test_days_from_secs(secs, expected):
    assert tc.days_from_secs(secs) == expected


def test_i64_min_magnitude_no_overflow():
    # The minimum int64 must not overflow the way a fixed-width numpy int64 abs
    # would. Python big ints handle it; the result is the true ceiling.
    i64_min = -(2**63)
    assert tc.div_ceil(i64_min, 1) == i64_min
    assert tc.timestamp_from_int(i64_min, 'ns') == -9223372036


# ===========================================================================
# Native temporal path (no parsing: the unit rides the dtype)
# ===========================================================================
def test_timestamp_secs_from_array_ceiling_ns():
    base = int(pd.Timestamp('2020-01-01 00:00:00').timestamp())
    ser = pd.Series([
        pd.Timestamp('2020-01-01 00:00:00.500'),   # sub-second -> ceil up
        pd.Timestamp('2020-01-01 00:00:01'),        # exact second
    ])
    assert list(tc.timestamp_secs_from_array(ser)) == [base + 1, base + 1]


def test_timestamp_secs_from_array_ceiling_ms_unit():
    base = int(pd.Timestamp('2020-01-01 00:00:00').timestamp())
    ser = pd.Series([pd.Timestamp('2020-01-01 00:00:00.500')]).astype(
        'datetime64[ms]')
    assert list(tc.timestamp_secs_from_array(ser)) == [base + 1]


def test_timestamp_secs_from_array_nat_is_none():
    ser = pd.Series([pd.Timestamp('2020-01-01'), pd.NaT])
    assert tc.timestamp_secs_from_array(ser)[1] is None


def test_timestamp_secs_from_array_tz_aware_normalized_to_utc():
    ser = pd.Series([pd.Timestamp('2020-01-01 00:00:00', tz='US/Eastern')])
    want = int(pd.Timestamp('2020-01-01 00:00:00', tz='US/Eastern').timestamp())
    assert list(tc.timestamp_secs_from_array(ser)) == [want]


def _scalar_native_seconds(ser):
    # Element-wise scalar reference for the vectorized native path.
    s = tc._as_naive_utc_datetime_series(ser)
    divisor = tc._native_divisor(s.dtype)
    ints = s.to_numpy().view('int64')
    isna = pd.isna(s).to_numpy()
    return [None if isna[i] else tc.div_ceil(int(ints[i]), divisor)
            for i in range(len(s))]


@pytest.mark.parametrize('resolution', ['ns', 'us', 'ms', 's'])
def test_parity_native_vectorized_matches_scalar(resolution):
    ts = pd.Series([
        pd.Timestamp('2020-01-01 00:00:00.500'),   # sub-second must ceil up
        pd.Timestamp('2020-01-01 00:00:01'),        # exact second
        pd.Timestamp('1969-12-31 23:59:59.250'),    # pre-epoch sub-second
        pd.Timestamp('2038-01-19 03:14:07'),
        pd.NaT,                                     # null -> None, never epoch
    ])
    ser = ts.astype(f'datetime64[{resolution}]')
    out = tc.timestamp_secs_from_array(ser)
    assert list(out) == _scalar_native_seconds(ser)


def test_parity_native_all_nat_and_empty():
    assert list(tc.timestamp_secs_from_array(
        pd.Series([pd.NaT, pd.NaT]))) == [None, None]
    assert list(tc.timestamp_secs_from_array(
        pd.Series([], dtype='datetime64[ns]'))) == []


def test_native_dispatch_epoch_seconds_and_days():
    ser = pd.Series(pd.to_datetime(['2020-01-01', '2020-03-01']))
    assert list(tc.to_epoch_seconds(ser, TemporalHint.native())) == [
        1577836800, 1583020800]
    assert list(tc.to_epoch_days(ser, TemporalHint.native())) == [
        _days(2020, 1, 1), _days(2020, 3, 1)]


def test_native_dispatch_days_subsecond_ceils_and_null_drops():
    ser = pd.Series([pd.Timestamp('2020-01-01 00:00:00.500'), pd.NaT])
    out = tc.to_epoch_days(ser, TemporalHint.native())
    assert out[0] == _days(2020, 1, 2)   # 00:00:00.5 -> ceil to next day
    assert out[1] is None


# ===========================================================================
# Epoch days
# ===========================================================================
@pytest.mark.parametrize('values, fmt, expected', [
    (['2024-01-15', '1969-12-31'], '%Y-%m-%d',
     [_days(2024, 1, 15), -1]),
    (['2024-01-15', 'not-a-date'], None, [_days(2024, 1, 15), None]),
    (['2024-01-15', 'garbage'], '%Y-%m-%d', [_days(2024, 1, 15), None]),
    # Non-rejecting civil-days arithmetic: month 13 rolls into the next year.
    (['2024-13-01'], '%Y-%m-%d', [_days(2025, 1, 1)]),
    # A value carrying a time of day belongs to the following day, never an
    # earlier one -- the same ceiling ingest applies.
    (['2024-01-15 10:30:00'], '%Y-%m-%d %H:%M:%S', [_days(2024, 1, 16)]),
])
def test_string_to_epoch_days(values, fmt, expected):
    assert _epoch_days(values, fmt) == expected


def test_to_epoch_days_integer_epoch():
    out = tc.to_epoch_days(pd.Series([1709251200], dtype='int64'),
                           TemporalHint.of_unit('s'))
    assert out[0] == _days(2024, 3, 1)


# ===========================================================================
# MIN/MAX
# ===========================================================================
def test_min_max_fast_path_unit_ms():
    ser = pd.Series([1709251200000, 1709251200500, 1000], dtype='int64')
    lo, hi = tc.min_max_epoch(ser, TemporalHint.of_unit('ms'))
    assert (lo, hi) == (1, 1709251201)


def test_min_max_unresolved_numeric_is_identity_seconds():
    lo, hi = tc.min_max_epoch(pd.Series([5, 1, 9]), TemporalHint.unresolved())
    assert (lo, hi) == (1, 9)


def test_min_max_native():
    base = int(pd.Timestamp('2020-01-01 00:00:00').timestamp())
    ser = pd.Series([
        pd.Timestamp('2020-01-01 00:00:00.500'),
        pd.Timestamp('2020-01-01 00:00:00.900'),
    ])
    assert tc.min_max_epoch(ser, TemporalHint.native()) == (base + 1, base + 1)


def test_min_max_format_string_convert_then_reduce():
    # Raw lexical order differs from temporal order, so FORMAT must convert
    # then reduce (not raw min/max).
    ser = pd.Series(['03/04/2024', '01/15/2024', '12/31/2024'])
    lo, hi = tc.min_max_epoch(ser, TemporalHint.of_format('%m/%d/%Y'))
    assert lo == int(pd.Timestamp('2024-01-15').timestamp())
    assert hi == int(pd.Timestamp('2024-12-31').timestamp())


def test_min_max_empty_returns_none():
    assert tc.min_max_epoch(pd.Series([], dtype='int64'),
                            TemporalHint.of_unit('s')) == (None, None)
    assert tc.min_max_epoch(pd.Series([np.nan, np.nan]),
                            TemporalHint.of_unit('s')) == (None, None)


def test_min_max_native_all_nat_and_empty_are_none():
    assert tc.min_max_epoch(pd.Series([pd.NaT, pd.NaT]),
                            TemporalHint.native()) == (None, None)
    assert tc.min_max_epoch(pd.Series([], dtype='datetime64[ns]'),
                            TemporalHint.native()) == (None, None)


def test_min_max_format_all_unparseable_is_none():
    assert tc.min_max_epoch(pd.Series(['x', 'y', 'z']),
                            TemporalHint.of_format('%Y-%m-%d')) == (None, None)


def test_null_values_never_raise_and_drop():
    ser = pd.Series(['2024-01-15', 'not-a-date', None, '2024-06-01'])
    lo, hi = tc.min_max_epoch(ser, TemporalHint.of_format('%Y-%m-%d'))
    assert lo == int(pd.Timestamp('2024-01-15').timestamp())
    assert hi == int(pd.Timestamp('2024-06-01').timestamp())


# ===========================================================================
# UNRESOLVED and mixed columns
# ===========================================================================
def test_unresolved_string_column_is_the_default_chain():
    ser = pd.Series(['2020-01-02T03:04:05Z', '2020-01-02 03:04:05',
                     '2020-01-02', '1709251200', 'garbage', None])
    assert list(tc.to_epoch_seconds(ser, TemporalHint.unresolved())) == [
        1577934245, 1577934245, 1577923200, 1709251200, None, None]


def test_format_column_falls_back_to_int_epoch_seconds():
    ser = pd.Series(['20230408', '1709251200'])
    out = tc.to_epoch_seconds(ser, TemporalHint.of_format('%Y%m%d'))
    assert list(out) == [_days(2023, 4, 8) * 86400, 1709251200]


def test_empty_column_is_empty():
    assert list(tc.to_epoch_seconds(pd.Series([], dtype=object),
                                    TemporalHint.of_format('%Y-%m-%d'))) == []


def test_epoch_seconds_to_timestamp_adapter():
    assert tc.epoch_seconds_to_timestamp(1709251200) == pd.Timestamp(
        '2024-03-01')
    assert tc.epoch_seconds_to_timestamp(None) is pd.NaT
    # Out-of-bounds epoch -> NaT (best-effort, no raise).
    assert tc.epoch_seconds_to_timestamp(10**19) is pd.NaT


def test_hint_constructors_shape():
    assert TemporalHint.of_unit('ms').kind is HintKind.UNIT
    assert TemporalHint.of_format('%Y%m%d').kind is HintKind.FORMAT
    assert TemporalHint.native().kind is HintKind.NATIVE
    assert TemporalHint.unresolved().kind is HintKind.UNRESOLVED


_TYPED = pd.Series(
    pd.to_datetime([
        '1969-12-31 23:59:59.5', '2024-01-01 00:00:00.000000001',
        '1700-01-01 00:00:00.25', None
    ], format='ISO8601'))


@pytest.mark.parametrize('series', [
    _TYPED.astype('datetime64[s]'),
    _TYPED.astype('datetime64[ms]'),
    _TYPED.astype('datetime64[us]'),
    _TYPED,
    _TYPED.dt.tz_localize('US/Pacific'),
], ids=['s', 'ms', 'us', 'ns', 'tz'])
def test_typed_columns_on_the_seconds_clock_match_diskgraph(series):
    hint = TemporalHint.native()
    assert list(tc.to_epoch_ticks(series, hint, 'seconds')) == list(
        tc._resolve(series, hint, days=False, timestamp_precision='seconds'))


def test_statistics_convert_in_the_snapshot_clock(monkeypatch):
    from kumoapi.typing import Stype

    from kumo.config import BaseColumnConfig
    from kumo.connector.utils import coerce_timestamp_stype_series_for_stats

    clocks = []

    def resolve_temporal(values, fmt, unit, as_days, *, timestamp_precision):
        clocks.append(timestamp_precision)
        return [1_700_000_000_123], fmt, unit

    monkeypatch.setattr(tc, 'resolve_temporal', resolve_temporal)
    config = BaseColumnConfig(name='ts', stype=Stype.timestamp)
    coerced = coerce_timestamp_stype_series_for_stats(
        pd.Series(['2023-11-14 22:13:20.123']), config, 'milliseconds')

    assert clocks == ['milliseconds']
    assert coerced.tolist() == [pd.Timestamp('2023-11-14 22:13:20.123')]
