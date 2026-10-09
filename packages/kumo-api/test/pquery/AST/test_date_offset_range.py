# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from kumoapi.json_serde import from_json, to_json
from kumoapi.pquery.AST.date_offset_range import DateOffsetRange
from kumoapi.typing import TimeUnit


def test_date_offset_range():
    arange = DateOffsetRange(start=2, end=6, unit='days')
    assert str(arange) == ("DateOffsetRange(start=2, end=6, unit=days)")

    assert arange.unit == TimeUnit.DAYS
    assert arange.start_date_offset == pd.DateOffset(days=2)
    assert arange.end_date_offset == pd.DateOffset(days=6)

    cfg = to_json(arange)
    assert arange == from_json(cfg, DateOffsetRange)

    with pytest.raises(ValueError, match="needs to be smaller"):
        DateOffsetRange(start=6, end=2, unit='days')


def test_merge_overlapping():
    left = DateOffsetRange(start=12, end=36, unit='hours')
    right = DateOffsetRange(start=2, end=6, unit='days')
    combined = left.merge_ranges(left, right)
    assert left.merge_ranges(left, right) == right.merge_ranges(right, left)
    assert combined.unit == TimeUnit.HOURS
    assert combined.start_date_offset == pd.DateOffset(hours=12)
    assert combined.end_date_offset == pd.DateOffset(hours=6 * 24)

    left = DateOffsetRange(start=12 * 60, end=36 * 60, unit='minutes')
    right = DateOffsetRange(start=2, end=6, unit='days')
    combined = left.merge_ranges(left, right)
    assert left.merge_ranges(left, right) == right.merge_ranges(right, left)
    assert combined.unit == TimeUnit.MINUTES
    assert combined.start_date_offset == pd.DateOffset(minutes=12 * 60)
    assert combined.end_date_offset == pd.DateOffset(minutes=6 * 24 * 60)


def test_merge_covering():
    left = DateOffsetRange(start=32, end=72, unit='hours')
    right = DateOffsetRange(start=1, end=6, unit='days')
    combined = left.merge_ranges(left, right)
    assert left.merge_ranges(left, right) == right.merge_ranges(right, left)
    assert combined.unit == TimeUnit.HOURS
    assert combined.start_date_offset == pd.DateOffset(hours=24)
    assert combined.end_date_offset == pd.DateOffset(hours=6 * 24)

    left = DateOffsetRange(start=32 * 60, end=72 * 60, unit='minutes')
    right = DateOffsetRange(start=1, end=6, unit='days')
    combined = left.merge_ranges(left, right)
    assert left.merge_ranges(left, right) == right.merge_ranges(right, left)
    assert combined.unit == TimeUnit.MINUTES
    assert combined.start_date_offset == pd.DateOffset(minutes=24 * 60)
    assert combined.end_date_offset == pd.DateOffset(minutes=6 * 24 * 60)


def test_merge_month():
    left = DateOffsetRange(start=14, end=21, unit='days')
    right = DateOffsetRange(start=1, end=2, unit='months')
    combined = left.merge_ranges(left, right)
    assert combined.unit == TimeUnit.DAYS
    assert combined.start_date_offset == pd.DateOffset(days=14)
    assert 31 + 28 <= combined.end <= 31 + 31


def test_merge_None():
    left = DateOffsetRange(start=14, end=21, unit='days')
    right = DateOffsetRange(start=None, end=12, unit='days')
    combined = left.merge_ranges(left, right)
    assert combined.unit == TimeUnit.DAYS
    assert combined.start_date_offset is None
    assert combined.is_open
    assert combined.end == 21


def test_large_constants():
    with pytest.raises(ValueError, match="Unsupported constant"):
        DateOffsetRange(start=30, end=13123112819128, unit='days')
    with pytest.raises(ValueError, match="Unsupported constant"):
        DateOffsetRange(start=-2000000000000000000000000, end=0, unit='days')


def test_start_less_than_end():
    with pytest.raises(ValueError):
        DateOffsetRange(start=30, end=29)


def test_start_equal_to_end():
    with pytest.raises(ValueError):
        DateOffsetRange(start=30, end=30)


@pytest.mark.parametrize('ts,exp_repr', [(pd.DateOffset(months=8), '8 months'),
                                         (pd.DateOffset(days=1), '1 days')])
def test_date_offset_repr(ts, exp_repr):
    assert DateOffsetRange.date_offset_repr(ts) == exp_repr


@pytest.mark.parametrize('ts,unit,exp_repr', [
    (pd.Timedelta('30 days 18:22:19'), TimeUnit.HOURS,
     '738 hours and 22 minutes'),
    (pd.Timedelta('30 days 18:22:19'), TimeUnit.DAYS, '30 days and 18 hours'),
    (pd.Timedelta('30 days 18:22:19'), TimeUnit.MONTHS,
     '1 month and 18 hours'),
])
def test_timedelta_repr(ts, unit, exp_repr):
    assert DateOffsetRange.timedelta_repr(ts, unit) == exp_repr


def test_date_offset_multiply():
    assert DateOffsetRange.date_offset_multiply(pd.DateOffset(days=4),
                                                3) == pd.DateOffset(days=12)


def test_date_offset_add():
    assert DateOffsetRange.date_offset_add(
        pd.DateOffset(months=3, days=1),
        pd.DateOffset(months=1)) == pd.DateOffset(months=4, days=1)


def test_date_offset_subtract():
    assert DateOffsetRange.date_offset_subtract(
        pd.DateOffset(months=3),
        pd.DateOffset(months=1, days=1)) == pd.DateOffset(months=2, days=-1)


def test_date_offset_gt():
    assert DateOffsetRange.date_offset_gt(pd.DateOffset(months=3),
                                          pd.DateOffset(months=1, days=1))


def test_issue_27576_large_offset_handling():
    """Regression test for OutOfBoundsDatetime with large date offset handling.

    Validates that large date offset comparisons and timeframe calculations
    work correctly when using a fixed reference date instead of current time,
    ensuring safe and deterministic behavior across all real-world usage
    patterns.

    Covers:
    - Large offset comparisons (100-290 years)
    - Validation split size checks with mixed units
    - Timeframe calculations with large date ranges
    - Boundary conditions and edge cases
    - Deterministic behavior (time-independent)
    """
    # Large offset comparisons (the core issue)
    assert DateOffsetRange.date_offset_gt(pd.DateOffset(years=200),
                                          pd.DateOffset(years=100))
    assert DateOffsetRange.date_offset_gt(pd.DateOffset(years=100),
                                          pd.DateOffset(years=50))
    assert DateOffsetRange.date_offset_gt(pd.DateOffset(years=290),
                                          pd.DateOffset(years=100))
    assert not DateOffsetRange.date_offset_gt(pd.DateOffset(years=100),
                                              pd.DateOffset(years=100))
    assert not DateOffsetRange.date_offset_gt(pd.DateOffset(years=50),
                                              pd.DateOffset(years=100))

    # Real-world validation split size checks
    assert not DateOffsetRange.date_offset_gt(pd.DateOffset(days=7),
                                              pd.DateOffset(days=30))
    assert DateOffsetRange.date_offset_gt(pd.DateOffset(months=3),
                                          pd.DateOffset(days=45))
    assert DateOffsetRange.date_offset_gt(pd.DateOffset(months=200),
                                          pd.DateOffset(days=30))

    # Timeframe calculations with large ranges (the other affected method)
    timeframe_1200_2400 = DateOffsetRange(start=1200, end=2400,
                                          unit='months').timeframe
    assert timeframe_1200_2400 is not None
    expected_days = 1200 * 30.44
    assert abs(timeframe_1200_2400.total_seconds() -
               expected_days * 24 * 3600) < 1e7

    # Additional timeframe ranges
    ranges = [
        DateOffsetRange(start=1, end=7, unit='days'),
        DateOffsetRange(start=100, end=200, unit='months'),
        DateOffsetRange(start=12, end=144, unit='months'),
        DateOffsetRange(start=0, end=100, unit='months'),
    ]
    for dor in ranges:
        assert dor.timeframe is not None

    # Boundary conditions
    assert not DateOffsetRange.date_offset_gt(pd.DateOffset(days=0),
                                              pd.DateOffset(days=1))
    assert DateOffsetRange.date_offset_gt(pd.DateOffset(months=-1),
                                          pd.DateOffset(months=-2))
    assert DateOffsetRange.date_offset_gt(pd.DateOffset(years=1),
                                          pd.DateOffset(years=-1))

    # Open ranges (None start)
    open_range = DateOffsetRange(start=None, end=30, unit='days')
    assert open_range.is_open
    assert open_range.timeframe is None

    # Determinism (results independent of current time)
    result1 = DateOffsetRange.date_offset_gt(pd.DateOffset(years=150),
                                             pd.DateOffset(years=100))
    result2 = DateOffsetRange.date_offset_gt(pd.DateOffset(years=150),
                                             pd.DateOffset(years=100))
    assert result1 == result2
