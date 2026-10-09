# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Tests for the column-level temporal detector and the ``TemporalHint`` type.

The detector is one DiskGraph call, so there are no per-value classifiers left to
test in isolation: every case below is a column. Covers epoch-magnitude bands,
encoded-date integers, string formats, plurality across a column, the robustness
contract (float/null, negative, messy, all-null guards), the round-trip
self-consistency between the hint and the converter, the declared-hint gate, and
native-dtype passthrough.

Everything is deterministic: fixed literals only, no ``now()``, no randomness, no
machine-timezone dependence.
"""
import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from kumo.connector import timestamp_convert as tc
from kumo.connector import timestamp_infer as ti
from kumo.connector.timestamp_infer import HintKind, TemporalHint


def infer(values, **kwargs):
    return ti.infer_temporal_hint(pd.Series(values), **kwargs)


# ===========================================================================
# TemporalHint type and storage mapping
# ===========================================================================
def test_hint_constructors_and_kind():
    assert TemporalHint.of_unit('ms').kind is HintKind.UNIT
    assert TemporalHint.of_unit('ms').unit == 'ms'
    assert TemporalHint.of_format('%Y%m%d').kind is HintKind.FORMAT
    assert TemporalHint.of_format('%Y%m%d').strftime == '%Y%m%d'
    assert TemporalHint.native().kind is HintKind.NATIVE
    assert TemporalHint.unresolved().kind is HintKind.UNRESOLVED


def test_hint_is_frozen_hashable_and_equal():
    assert TemporalHint.of_unit('s') == TemporalHint.of_unit('s')
    assert len({TemporalHint.of_unit('s'), TemporalHint.of_unit('s')}) == 1
    assert TemporalHint.of_unit('s') != TemporalHint.of_unit('ms')


@pytest.mark.parametrize('kind, unit, strftime', [
    (HintKind.UNIT, 'bogus', None),   # invalid unit token
    (HintKind.UNIT, None, None),      # UNIT without a unit
    (HintKind.UNIT, 's', '%Y'),       # UNIT must not carry a format
    (HintKind.FORMAT, None, None),    # FORMAT without a format
    (HintKind.FORMAT, 's', '%Y'),     # FORMAT must not carry a unit
    (HintKind.NATIVE, 's', None),     # NATIVE must carry neither
    (HintKind.UNRESOLVED, None, '%Y'),
])
def test_hint_mutual_exclusion_enforced(kind, unit, strftime):
    with pytest.raises(ValueError):
        TemporalHint(kind, unit=unit, strftime=strftime)


def test_hint_storage_mapping():
    # UNIT -> timestamp_unit; FORMAT -> metadata['TS_FORMAT']; else neither.
    assert ti.hint_to_storage(TemporalHint.of_unit('ms')) == ti.HintStorage(
        timestamp_unit='ms')
    assert ti.hint_to_storage(TemporalHint.of_format('%Y%m%d')) == ti.HintStorage(
        ts_format='%Y%m%d')
    assert ti.hint_to_storage(TemporalHint.native()) == ti.HintStorage()
    assert ti.hint_to_storage(TemporalHint.unresolved()) == ti.HintStorage()
    assert ti.TIMESTAMP_UNIT_CONFIG_FIELD == 'timestamp_unit'
    assert ti.TS_FORMAT_METADATA_KEY == 'TS_FORMAT'


# ===========================================================================
# Epoch-magnitude bands (strict '<' at 1e10 / 1e13 / 1e16)
# ===========================================================================
@pytest.mark.parametrize('value, unit', [
    (1709251200, 's'),
    (-86400, 's'),                    # negative uses unsigned magnitude
    (9999999999, 's'),                # 1e10 - 1
    (10000000000, 'ms'),              # 1e10 flips
    (9999999999999, 'ms'),            # 1e13 - 1
    (10000000000000, 'us'),           # 1e13 flips
    (9999999999999999, 'us'),         # 1e16 - 1
    (10000000000000000, 'ns'),        # 1e16 flips
    (1709251200000, 'ms'),
    (1709251200000000, 'us'),
    (1709251200000000000, 'ns'),
])
def test_epoch_magnitude_bands(value, unit):
    assert infer([value], ) == TemporalHint.of_unit(unit)


def test_zero_is_no_vote():
    assert infer([0]) == TemporalHint.unresolved()


def test_nat_sentinel_is_not_a_nanosecond_epoch():
    # The pandas NaT sentinel is a null, not the largest int64 magnitude.
    assert infer([-(2**63)]) == TemporalHint.unresolved()


# ===========================================================================
# Encoded-date integers: a FORMAT hint, never a UNIT hint (applying a unit to
# an encoded-date column would reinterpret it as epoch seconds and corrupt it).
# ===========================================================================
@pytest.mark.parametrize('value, expected', [
    (20230408, TemporalHint.of_format('%Y%m%d')),
    (20230408120530, TemporalHint.of_format('%Y%m%d%H%M%S')),
    (2023, TemporalHint.of_format('%Y')),
    (202304, TemporalHint.of_format('%Y%m')),
    (19000101, TemporalHint.of_format('%Y%m%d')),        # earliest valid
    (21001231, TemporalHint.of_format('%Y%m%d')),        # latest valid
    (19000101000000, TemporalHint.of_format('%Y%m%d%H%M%S')),
    (21001231235959, TemporalHint.of_format('%Y%m%d%H%M%S')),
    (-20230408, TemporalHint.of_format('%Y%m%d')),       # sign discarded
    # Component ranges are checked, calendar validity is not.
    (20230230, TemporalHint.of_format('%Y%m%d')),        # Feb 30
    (20230431, TemporalHint.of_format('%Y%m%d')),        # Apr 31
    # Out of range / no pattern: falls through to the epoch magnitude.
    (99999999, TemporalHint.of_unit('s')),               # year 9999
    (18991231, TemporalHint.of_unit('s')),               # year 1899
    (21010101, TemporalHint.of_unit('s')),               # year 2101
    (20231301, TemporalHint.of_unit('s')),               # month 13
    (20230001, TemporalHint.of_unit('s')),               # month 00
    (20230100, TemporalHint.of_unit('s')),               # day 00
    (20230132, TemporalHint.of_unit('s')),               # day 32
    (12345, TemporalHint.of_unit('s')),                  # 5 digits, no pattern
    (1709251200, TemporalHint.of_unit('s')),             # 10 digits, no pattern
    (20230408240000, TemporalHint.of_unit('us')),        # hour 24
    (20230408006000, TemporalHint.of_unit('us')),        # minute 60
    (20230408000060, TemporalHint.of_unit('us')),        # second 60
])
def test_encoded_int_format(value, expected):
    assert infer([value]) == expected


# ===========================================================================
# String columns
# ===========================================================================
@pytest.mark.parametrize('value, expected', [
    ('2024-01-15', '%Y-%m-%d'),
    ('2024-01-15 10:30:00', '%Y-%m-%d %H:%M:%S'),
    ('2024-01-15T10:30:00', '%Y-%m-%d %H:%M:%S'),
    ('2019-07-04T00:00:00Z', '%Y-%m-%dT%H:%M:%S%#z'),
    ('01/15/2024', '%m/%d/%Y'),
    ('01/15/2024 10:30:00', '%m/%d/%Y %H:%M:%S'),
    ('15/01/2024', '%d/%m/%Y'),                # day>12, only d/m/Y matches
    ('20230408', '%Y%m%d'),                    # numeric string, encoded date
])
def test_string_format_detection(value, expected):
    assert infer([value]) == TemporalHint.of_format(expected)


def test_ambiguous_slash_is_month_first():
    # %m/%d/%Y precedes %d/%m/%Y, so an ambiguous date is month-first.
    assert infer(['03/04/2024']) == TemporalHint.of_format('%m/%d/%Y')


def test_numeric_string_epoch_is_a_unit_not_a_date():
    # A 10-digit epoch-seconds string must not read as a %Y-%m-%d date.
    assert infer(['1709251200', '1710000000']) == TemporalHint.of_unit('s')


def test_non_timestamp_string_is_unresolved():
    assert infer(['not-a-date']) == TemporalHint.unresolved()


# ===========================================================================
# Plurality across a column
# ===========================================================================
def test_plurality_winner():
    assert infer([1709251200, 1710000000, 20230408]) == TemporalHint.of_unit('s')


def test_format_beats_unit_on_tie():
    assert infer([20230408, 1709251200]) == TemporalHint.of_format('%Y%m%d')
    assert infer([1709251200, 20230408]) == TemporalHint.of_format('%Y%m%d')


def test_format_plurality_over_tied_units():
    assert infer([20230408, 20240115, 1709251200,
                  1709251200000]) == TemporalHint.of_format('%Y%m%d')


# ===========================================================================
# Robustness contract
# ===========================================================================
def test_robust_float_null_epoch_resolves_to_unit():
    # An int epoch column with nulls becomes float64; it must still resolve to
    # the correct UNIT, not fall through to UNRESOLVED.
    assert infer([1709251200.0, None,
                  1710000000.0]) == TemporalHint.of_unit('s')
    assert infer([1709251200000.0, np.nan]) == TemporalHint.of_unit('ms')


def test_robust_float_encoded_date_stays_format():
    # A float-typed encoded-date column must still become FORMAT (corruption
    # trap: a unit on an encoded-date column mis-encodes it as epoch seconds).
    assert infer([20230408.0, 20240115.0,
                  None]) == TemporalHint.of_format('%Y%m%d')


def test_robust_negative_epoch_sign_agnostic():
    assert infer([-86400, -172800]) == TemporalHint.of_unit('s')


def test_robust_messy_mixed_magnitude_does_not_raise():
    # Mixed magnitudes never raise; the plurality unit wins (three seconds votes
    # beat one ms vote).
    assert infer([1000000000, 2000000000, 3000000000,
                  1709251200000]) == TemporalHint.of_unit('s')


def test_robust_mixed_object_column_does_not_raise():
    # No single Arrow type: the column is stringified rather than rejected.
    assert infer(['2024-01-15', 5, None]) == TemporalHint.of_format('%Y-%m-%d')


def test_robust_category_column_resolves_and_applies():
    # A low-cardinality column arrives dictionary-encoded; the values are
    # timestamps regardless of how they are physically laid out.
    ser = pd.Series(['2024-01-15', '2024-02-20', '2024-01-15'],
                    dtype='category')
    hint = ti.infer_temporal_hint(ser)
    assert hint == TemporalHint.of_format('%Y-%m-%d')
    assert list(tc.to_epoch_seconds(ser, hint)) == [
        1705276800, 1708387200, 1705276800]


def test_robust_bool_column_is_unresolved():
    assert infer([True, False]) == TemporalHint.unresolved()


def test_robust_leading_sentinels_do_not_hide_the_timestamps():
    # Sentinel rows classify to nothing; they must not consume the vote budget
    # and leave a real timestamp column UNRESOLVED (fatal for a create_time).
    values = ['0'] * 50 + ['2024-01-15'] * 50
    assert infer(values) == TemporalHint.of_format('%Y-%m-%d')
    assert infer(values, is_create_time=True) == TemporalHint.of_format(
        '%Y-%m-%d')


def test_robust_all_null_non_create_time_is_unresolved():
    assert infer([None, None, np.nan],
                 is_create_time=False) == TemporalHint.unresolved()


def test_robust_all_unparseable_non_create_time_is_unresolved():
    assert infer(['x', 'y', 'z'],
                 is_create_time=False) == TemporalHint.unresolved()


def test_robust_all_null_create_time_raises():
    with pytest.raises(ValueError, match='create_time'):
        infer([None, None], is_create_time=True)


def test_robust_unresolvable_create_time_raises():
    with pytest.raises(ValueError, match='create_time'):
        infer(['x', 'y'], is_create_time=True)


# ===========================================================================
# Round-trip self-consistency. The detector and the converter are the same
# DiskGraph call, so a detected hint is applicable by construction -- including
# when the token that wins is not the one a human would have picked.
# ===========================================================================
@pytest.mark.parametrize('values, epochs', [
    (['2024/01/15', '2024/02/20'], [1705276800, 1708387200]),
    (['2024/01/15 10:30:00'], [1705314600]),
    (['01/15/2024', '02/20/2024'], [1705276800, 1708387200]),
    (['2019-07-04T00:00:00Z'], [1562198400]),
    (['20230408'], [1680912000]),
    (['1709251200'], [1709251200]),
])
def test_detected_hint_is_applicable(values, epochs):
    ser = pd.Series(values)
    hint = ti.infer_temporal_hint(ser)
    assert list(tc.to_epoch_seconds(ser, hint)) == epochs
    assert tc.min_max_epoch(ser, hint) == (min(epochs), max(epochs))


def test_slash_content_resolves_to_the_separator_tolerant_dash_format():
    # The token is opaque to kumo: DiskGraph's %Y-%m-%d parser is separator
    # tolerant, so it wins the first-match over %Y/%m/%d and still applies.
    assert infer(['2024/01/15']) == TemporalHint.of_format('%Y-%m-%d')


# ===========================================================================
# Gate + native passthrough.
# ===========================================================================
def test_gate_declared_unit_returned_unchanged():
    # A declared hint short-circuits inference, even on messy data.
    assert infer([20230408, 20240115],
                 declared_unit='ms') == TemporalHint.of_unit('ms')


def test_gate_declared_format_returned_unchanged():
    assert infer([1709251200, 1710000000],
                 declared_format='%Y%m%d') == TemporalHint.of_format('%Y%m%d')


def test_gate_declared_format_precedes_declared_unit():
    assert infer([1, 2], declared_unit='s',
                 declared_format='%Y-%m-%d') == TemporalHint.of_format(
                     '%Y-%m-%d')


def test_infer_unrecognized_declared_unit_falls_through_to_inference():
    # 'weeks' is not a known alias; inference runs rather than raising.
    assert infer([1709251200, 1710000000],
                 declared_unit='weeks') == TemporalHint.of_unit('s')


def test_infer_non_series_sample_is_coerced():
    assert ti.infer_temporal_hint(
        [1709251200, 1710000000]) == TemporalHint.of_unit('s')


def test_native_dtype_passthrough_datetime64():
    ser = pd.to_datetime(pd.Series(['2024-01-01', '2024-02-01']))
    assert ti.infer_temporal_hint(ser) == TemporalHint.native()


def test_native_dtype_passthrough_object_datetimes():
    ser = pd.Series([datetime.datetime(2024, 1, 1),
                     datetime.datetime(2024, 2, 1)], dtype=object)
    assert ti.infer_temporal_hint(ser) == TemporalHint.native()


def test_native_dtype_passthrough_object_np_datetime64():
    # An ns-precision np.datetime64 must not be misread as an epoch integer.
    ser = pd.Series([np.datetime64('2024-01-01')], dtype=object)
    assert ti.infer_temporal_hint(ser) == TemporalHint.native()


def test_is_native_temporal_series_arrow_dtypes():
    ts = pd.Series(pd.to_datetime(['2024-01-01', '2024-02-01']),
                   dtype=pd.ArrowDtype(pa.timestamp('us')))
    assert ti._is_native_temporal_series(ts)
    assert ti.infer_temporal_hint(ts) == TemporalHint.native()

    dt = pd.Series([datetime.date(2024, 1, 1), datetime.date(2024, 2, 1)],
                   dtype=pd.ArrowDtype(pa.date32()))
    assert ti._is_native_temporal_series(dt)
    assert ti.infer_temporal_hint(dt) == TemporalHint.native()


# ===========================================================================
# map_timestamp_unit: every alias -> canonical token, plus the None / empty /
# unrecognized / `.value`-object cases.
# ===========================================================================
@pytest.mark.parametrize('alias, expected', [
    ('s', 's'), ('sec', 's'), ('secs', 's'), ('second', 's'), ('seconds', 's'),
    ('ms', 'ms'), ('milli', 'ms'), ('millis', 'ms'), ('millisecond', 'ms'),
    ('milliseconds', 'ms'),
    ('us', 'us'), ('µs', 'us'), ('micro', 'us'), ('micros', 'us'),
    ('microsecond', 'us'), ('microseconds', 'us'),
    ('ns', 'ns'), ('nano', 'ns'), ('nanos', 'ns'), ('nanosecond', 'ns'),
    ('nanoseconds', 'ns'),
    (None, None), ('', None), ('bogus', None),
])
def test_map_timestamp_unit_aliases(alias, expected):
    assert ti.map_timestamp_unit(alias) == expected


def test_map_timestamp_unit_value_attr_object():
    # An already-``TimestampUnit``-valued input is matched by its ``.value``.
    class _Unit:
        value = 'ms'

    assert ti.map_timestamp_unit(_Unit()) == 'ms'
