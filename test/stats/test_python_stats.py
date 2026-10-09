# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import os
import os.path as osp
import shutil
from typing import List

import numpy as np
import pandas as pd
import pytest
import torch
from kumo_core import embedding_models
from kumo_core.types import ColStatType, EdgeStatType, TableStatType
from kumo_core.types.schema import TS_FORMAT_METADATA, BaseColumnConfig
from kumoapi.typing import Dtype, Stype
from pandas import bdate_range

from kumo.config.mixin import (
    _KUMO_INTERNAL_TIMESTAMP_PREFIX,
    _recursive_from_config,
)
from kumo.connector.csv_connector import CSVConnector
from kumo.connector.parquet_connector import ParquetConnector
from kumo.connector.snowflake_connector import SnowflakeConnector
from kumo.connector.utils import coerce_timestamp_stype_series_for_stats
from kumo.graph.column import PrimaryKey
from kumo.graph.graph import Graph
from kumo.home import get_home_dir
from kumo.stats.compute_stats_python import (
    _histogram_half_month,
    _max,
    _min,
    compute_col_stats_python,
    compute_edge_stats_python,
)
from kumo.stats.stats import NUM_BINS_HISTOGRAM
from kumo.stats.utils import detect_multicat_sep, detect_multicat_sep_values
from kumo.testing import withSnowflakeCredentials
from kumo.testing.s3 import withS3Credentials
from kumo.utils.embedding_models import embedding_model_store
from statedb.config.typing import to_dtype

TEST_GRAPH_NAME = 'kumo_generator_test_graph'
STATS_DUMP_FILE = 'testfile.json'

# The full CI test environment does not bake the GloVe artifact used by
# STRING_GLOVE_OVERLAP. Keep this stats module self-contained instead of
# depending on workflow-level env.
os.environ.setdefault('KUMO_ENABLE_GLOVE', 'False')


def teardown_function():
    stats_file = osp.join(get_home_dir(), 'graph_stats', STATS_DUMP_FILE)
    if os.path.exists(stats_file):
        os.remove(stats_file)
    folder = osp.join(get_home_dir(), 'test_fsutil')
    shutil.rmtree(folder, ignore_errors=True)


def test_compute_stats():
    test_data = {
        "A": [0.0, 1.0, pd.NA, 3.0, 4.0],
        "B": [0.0, 1.0, 0.0, 1.0, 0.0],
        "C": ["foo1", pd.NA, "foo3", "foo4", "foo5"],
        "D": bdate_range("1/1/2009", periods=5),
    }

    test_df = pd.DataFrame(test_data)
    test_df['A'] = pd.to_numeric(test_df['A'])
    test_df['B'] = pd.to_numeric(test_df['B'])
    test_df['D'] = pd.to_datetime(test_df['D'])

    cols = [
        BaseColumnConfig(name, dtype=to_dtype(dtype))
        for name, dtype in dict(test_df.dtypes).items()
    ]

    for c in cols:
        possible_stats = [
            x for x in ColStatType
            if hasattr(x, 'value') and x.supports(c.dtype)
        ]
        stats = compute_col_stats_python(test_df[c.name], possible_stats)
        # TODO: Something more concrete here.
        assert stats


def test_string_stats():
    test_data = {
        "A": ["two words", "three words here", pd.NA, "", "punctuation!"],
        "B": [
            "notinGloVeprobably", "ein deutscher Satz", "hash123940uhash==",
            "john@doe.com", "鸡汤"
        ],
    }

    test_df = pd.DataFrame(test_data)
    cols = [
        BaseColumnConfig(name, dtype=to_dtype(dtype))
        for name, dtype in dict(test_df.dtypes).items()
    ]

    expected = {
        "A": {
            ColStatType.STRING_AVG_LEN: 9.25,
            ColStatType.STRING_MAX_LEN: 16,
            ColStatType.STRING_AVG_TOKENS: 1.5,
            ColStatType.STRING_MAX_TOKENS: 3,
            ColStatType.STRING_GLOVE_OVERLAP: 1.0,
            ColStatType.STRING_AVG_NON_CHAR: 0.0,
        },
        "B": {
            ColStatType.STRING_AVG_LEN: 13.4,
            ColStatType.STRING_MAX_LEN: 18,
            ColStatType.STRING_AVG_TOKENS: 1.6,
            ColStatType.STRING_MAX_TOKENS: 3,
            ColStatType.STRING_GLOVE_OVERLAP: 0.5,
            ColStatType.STRING_AVG_NON_CHAR: 0.375,
        },
    }

    # When CI sets ``KUMO_ENABLE_GLOVE=False`` (see
    # ``.github/workflows/testing.yml:289``) the column-stats dispatcher
    # skips ``STRING_GLOVE_OVERLAP`` entirely, so the assertion on its
    # exact value must be conditioned on the env-var. Mirrors the parsing
    # used by ``KUMO_ENABLE_GLOVE`` (``service_framework.env_var.EnvVar``).
    _disabled_values = {'false', '0', 'no', 'off'}
    glove_enabled = (os.environ.get('KUMO_ENABLE_GLOVE', 'True').lower()
                     not in _disabled_values)
    for c in cols:
        possible_stats = [
            x for x in ColStatType
            if hasattr(x, 'value') and x.supports(c.dtype)
        ]
        stats = compute_col_stats_python(test_df[c.name], possible_stats)
        for exp_stat, exp_value in expected[c.name].items():
            if exp_stat == ColStatType.STRING_GLOVE_OVERLAP \
                    and not glove_enabled:
                # GloVe disabled: dispatcher correctly omits this stat.
                assert exp_stat not in stats
                continue
            assert exp_stat in stats
            assert abs(stats[exp_stat].value - exp_value) < 0.01


def test_glove_overlap_with_and_without_default_model(monkeypatch, tmp_path):
    monkeypatch.setenv('KUMO_ENABLE_GLOVE', 'True')
    ser = pd.Series(['hello world', 'unknown words'])
    overlap = ColStatType.STRING_GLOVE_OVERLAP
    stats = compute_col_stats_python(ser, [overlap])
    assert stats[overlap].value == 0.5

    monkeypatch.setenv('KUMO_GLOVE_PATH', str(tmp_path))
    monkeypatch.setattr(embedding_models, 'entry_points', lambda group: [])
    monkeypatch.setattr(embedding_models, '_INSTALLED', None)
    embedding_model_store.cache_clear()
    try:
        stats = compute_col_stats_python(
            ser, [overlap, ColStatType.STRING_AVG_TOKENS])
    finally:
        embedding_model_store.cache_clear()
    assert overlap not in stats
    assert stats[ColStatType.STRING_AVG_TOKENS].value == 2


def test_category_counts():
    ser = pd.Series(["foo1", pd.NA, "foo3", "foo4", "foo1"])
    stats = compute_col_stats_python(ser, [ColStatType.CATEGORY_COUNTS])
    expected = [("foo1", 2), ("foo3", 1), ("foo4", 1)]
    category_counts: pd.Series = stats[ColStatType.CATEGORY_COUNTS].value
    assert category_counts == expected

    ser = pd.Series(['x'] * 200000 + ['y'] * 100 + ['z'])
    stats = compute_col_stats_python(ser, [ColStatType.CATEGORY_COUNTS])
    expected = [('x', 200000), ('y', 100), ('z', 1)]
    category_counts: pd.Series = stats[ColStatType.CATEGORY_COUNTS].value
    assert category_counts == expected


def test_multi_categories():
    ser = pd.Series(["a,b", "b,c", "c,d", "a,b,c", pd.NA])
    stats = compute_col_stats_python(ser, [
        ColStatType.NUM_UNIQUE_MULTI,
        ColStatType.MULTI_CATEGORY_COUNTS,
        ColStatType.MULTI_CATEGORIES_SEPARATOR,
    ])
    expected_counts = [("b", 3), ("c", 3), ("a", 2), ("d", 1)]
    assert stats[ColStatType.MULTI_CATEGORIES_SEPARATOR].value == ","
    assert stats[ColStatType.NUM_UNIQUE_MULTI].value == 4
    assert stats[ColStatType.MULTI_CATEGORY_COUNTS].value == expected_counts

    # bar separator - checks that we're not separating out each char
    ser = pd.Series(["abc|b", "b|c", "c|d", "abc|b|c", pd.NA])
    stats = compute_col_stats_python(ser, [
        ColStatType.NUM_UNIQUE_MULTI,
        ColStatType.MULTI_CATEGORY_COUNTS,
        ColStatType.MULTI_CATEGORIES_SEPARATOR,
    ])
    expected_counts = [("b", 3), ("c", 3), ("abc", 2), ("d", 1)]
    assert stats[ColStatType.MULTI_CATEGORIES_SEPARATOR].value == "|"
    assert stats[ColStatType.NUM_UNIQUE_MULTI].value == 4
    assert stats[ColStatType.MULTI_CATEGORY_COUNTS].value == expected_counts

    # No separator
    ser = pd.Series(["a", "b", "c", pd.NA, "b"])
    stats = compute_col_stats_python(ser, [
        ColStatType.NUM_UNIQUE_MULTI,
        ColStatType.MULTI_CATEGORY_COUNTS,
        ColStatType.MULTI_CATEGORIES_SEPARATOR,
    ])
    expected = [("b", 2), ("a", 1), ("c", 1)]
    assert stats[ColStatType.MULTI_CATEGORIES_SEPARATOR].value is None
    assert stats[ColStatType.NUM_UNIQUE_MULTI].value == 3
    assert stats[ColStatType.MULTI_CATEGORY_COUNTS].value == expected


def test_multi_categories_array():
    for column_dtype in [str, int, float]:
        a, b, c, d = map(column_dtype, [0, 1, 2, 3])

        # array targets
        ser = pd.Series([[a, b], [b, c], [c, d], [a, b, c], [], pd.NA])
        stats = compute_col_stats_python(ser, [
            ColStatType.NUM_UNIQUE_MULTI,
            ColStatType.MULTI_CATEGORY_COUNTS,
            ColStatType.MULTI_CATEGORIES_SEPARATOR,
        ])
        expected_counts = [(b, 3), (c, 3), (a, 2), (d, 1)]
        assert stats[ColStatType.MULTI_CATEGORIES_SEPARATOR].value is None
        assert stats[ColStatType.NUM_UNIQUE_MULTI].value == 4
        assert stats[ColStatType.MULTI_CATEGORY_COUNTS].value == expected_counts

        # array targets with the first element as N/A
        ser = pd.Series([pd.NA, [a, b], [b, c], [c, d], [a, b, c]])
        stats = compute_col_stats_python(ser, [
            ColStatType.NUM_UNIQUE_MULTI,
            ColStatType.MULTI_CATEGORY_COUNTS,
            ColStatType.MULTI_CATEGORIES_SEPARATOR,
        ])
        expected_counts = [(b, 3), (c, 3), (a, 2), (d, 1)]
        assert stats[ColStatType.MULTI_CATEGORIES_SEPARATOR].value is None
        assert stats[ColStatType.NUM_UNIQUE_MULTI].value == 4
        assert stats[ColStatType.MULTI_CATEGORY_COUNTS].value == expected_counts

        # array targets with the first element as an empty list
        ser = pd.Series([[], [a, b], [b, c], [c, d], [a, b, c]])
        stats = compute_col_stats_python(ser, [
            ColStatType.NUM_UNIQUE_MULTI,
            ColStatType.MULTI_CATEGORY_COUNTS,
            ColStatType.MULTI_CATEGORIES_SEPARATOR,
        ])
        expected_counts = [(b, 3), (c, 3), (a, 2), (d, 1)]
        assert stats[ColStatType.MULTI_CATEGORIES_SEPARATOR].value is None
        assert stats[ColStatType.NUM_UNIQUE_MULTI].value == 4
        assert stats[ColStatType.MULTI_CATEGORY_COUNTS].value == expected_counts

    ser = pd.Series([[True], [False], [True, False], [False], [], pd.NA])
    stats = compute_col_stats_python(ser, [
        ColStatType.NUM_UNIQUE_MULTI,
        ColStatType.MULTI_CATEGORY_COUNTS,
        ColStatType.MULTI_CATEGORIES_SEPARATOR,
    ])
    expected_counts = [(False, 3), (True, 2)]
    assert stats[ColStatType.MULTI_CATEGORIES_SEPARATOR].value is None
    assert stats[ColStatType.NUM_UNIQUE_MULTI].value == 2
    assert stats[ColStatType.MULTI_CATEGORY_COUNTS].value == expected_counts

    ser = pd.Series([], dtype='object')
    stats = compute_col_stats_python(ser, [
        ColStatType.NUM_UNIQUE_MULTI,
        ColStatType.MULTI_CATEGORY_COUNTS,
        ColStatType.MULTI_CATEGORIES_SEPARATOR,
    ])
    assert stats[ColStatType.MULTI_CATEGORIES_SEPARATOR].value is None
    assert stats[ColStatType.NUM_UNIQUE_MULTI].value == 0
    assert stats[ColStatType.MULTI_CATEGORY_COUNTS].value == []

    ser = pd.Series([['']])
    stats = compute_col_stats_python(ser, [
        ColStatType.NUM_UNIQUE_MULTI,
        ColStatType.MULTI_CATEGORY_COUNTS,
        ColStatType.MULTI_CATEGORIES_SEPARATOR,
    ])
    assert stats[ColStatType.MULTI_CATEGORIES_SEPARATOR].value is None
    assert stats[ColStatType.NUM_UNIQUE_MULTI].value == 0
    assert stats[ColStatType.MULTI_CATEGORY_COUNTS].value == []

    ser = pd.Series([pd.NA])
    stats = compute_col_stats_python(ser, [
        ColStatType.NUM_UNIQUE_MULTI,
        ColStatType.MULTI_CATEGORY_COUNTS,
        ColStatType.MULTI_CATEGORIES_SEPARATOR,
    ])
    assert stats[ColStatType.MULTI_CATEGORIES_SEPARATOR].value is None
    assert stats[ColStatType.NUM_UNIQUE_MULTI].value == 0
    assert stats[ColStatType.MULTI_CATEGORY_COUNTS].value == []


def test_multi_categories_quoted_strings():
    ser = pd.Series([
        '["10", "11", "12"]', '["11", "12", "13"]', '["12", "13", "14"]', pd.NA
    ])
    stats = compute_col_stats_python(ser, [
        ColStatType.NUM_UNIQUE_MULTI,
        ColStatType.MULTI_CATEGORY_COUNTS,
        ColStatType.MULTI_CATEGORIES_SEPARATOR,
    ])
    expected_counts = [("12", 3), ("11", 2), ("13", 2), ("10", 1), ("14", 1)]

    assert stats[ColStatType.MULTI_CATEGORIES_SEPARATOR].value is None
    assert stats[ColStatType.NUM_UNIQUE_MULTI].value == 5
    assert stats[ColStatType.MULTI_CATEGORY_COUNTS].value == expected_counts


def test_col_invalid_fraction(tempdir):
    tables_path = os.path.join(tempdir.name, 'tables')
    os.makedirs(tables_path)

    dummy_table_path = os.path.join(tables_path, 'DUMMY.csv')
    id_vals = [1] * 100000
    feature_vals = [10] * 90000 + [pd.NA] * 5000 + ['x'] * 5000
    # id_vals = [range(0, 10000, 1)]
    # feature_vals = [10] * 10000
    pd.DataFrame({
        'ID': id_vals,
        'FEATURE': feature_vals,
    }).to_csv(dummy_table_path, index=False)

    connector = CSVConnector('csv', root_dir=tables_path)

    id_col = BaseColumnConfig('ID', None, Dtype.int64)
    feature_col = BaseColumnConfig('FEATURE', None, Dtype.int64)

    stats_request_dict = {
        id_col: [ColStatType.INVALID_FRACTION],
        feature_col: [ColStatType.INVALID_FRACTION],
    }
    out = connector._get_approx_col_stats('DUMMY', stats_request_dict)
    assert len(out) == 2
    id_stats = out[id_col]
    feature_stats = out[feature_col]
    assert id_stats[ColStatType.INVALID_FRACTION].value == 0.0
    assert feature_stats[ColStatType.INVALID_FRACTION].value == pytest.approx(
        0.05, 0.1)

    stats_request_dict = {
        id_col: [ColStatType.INVALID_FRACTION],
        feature_col: [ColStatType.INVALID_FRACTION],
    }
    out = connector._get_exact_col_stats('DUMMY', stats_request_dict)
    assert len(out) == 2
    id_stats = out[id_col]
    feature_stats = out[feature_col]
    assert id_stats[ColStatType.INVALID_FRACTION].value == 0.0
    assert feature_stats[ColStatType.INVALID_FRACTION].value == 0.05


def test_time_col_hist():
    # day month pairs to test
    test_dates = [(12, 1), (15, 1), (1, 2), (2, 2), (1, 6), (12, 6)]
    timestamp_list = [
        pd.Timestamp(day=date[0], month=date[1], year=2000)
        for date in test_dates
    ]
    timestamp_ser = pd.Series(timestamp_list)
    hist = _histogram_half_month(timestamp_ser)
    # _histogram_half_month returns datetime.date bins; pandas 2.x no longer
    # treats datetime.date == Timestamp as equal, so compare against date.
    assert hist[0][0] == datetime.date(day=1, month=1, year=2000)
    assert len(hist) == 11
    assert hist[10][0] == datetime.date(day=1, month=6, year=2000)
    assert hist[10][1] == 2
    assert hist[0][1] == 2

    test_dates = [(17, 1, 2000), (18, 1, 2000), (1, 2, 2000), (2, 2, 2000),
                  (1, 6, 2000), (16, 8, 2000), (16, 8, 2000), (16, 8, 2001)]
    timestamp_list = [
        pd.Timestamp(day=date[0], month=date[1], year=date[2])
        for date in test_dates
    ]
    timestamp_ser = pd.Series(timestamp_list)
    hist = _histogram_half_month(timestamp_ser)
    assert hist[0][0] == datetime.date(day=16, month=1, year=2000)
    assert len(hist) == 39
    assert hist[38][0] == datetime.date(day=16, month=8, year=2001)
    assert hist[38][1] == 1
    assert hist[0][1] == 2

    timestamp_ser = pd.Series([pd.NaT, pd.NaT, pd.NaT])
    hist = _histogram_half_month(timestamp_ser)
    assert len(hist) == 0

    test_dates = [(17, 1, 2000), None, (1, 2, 2000), None, (1, 6, 2000), None,
                  (16, 8, 2000), (16, 8, 2001), (16, 1, 2000)]
    timestamp_list = [
        pd.Timestamp(day=date[0], month=date[1], year=date[2])
        if date else pd.NaT for date in test_dates
    ]
    timestamp_ser = pd.Series(timestamp_list)
    hist = _histogram_half_month(timestamp_ser)
    assert hist[0][0] == datetime.date(day=16, month=1, year=2000)
    assert len(hist) == 39
    assert hist[38][0] == datetime.date(day=16, month=8, year=2001)
    assert hist[38][1] == 1
    assert hist[0][1] == 2


def test_min_max_object_dtype_datetime_series():
    # MIN/MAX must handle Snowflake object-dtype timestamps; the numeric
    # np.isnan path raises TypeError on object arrays.
    timestamp_ser = pd.Series([
        pd.Timestamp("2021-06-15"),
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2022-03-10"),
    ], dtype=object)
    assert timestamp_ser.dtype == object
    assert _min(timestamp_ser) == pd.Timestamp("2020-01-01")
    assert _max(timestamp_ser) == pd.Timestamp("2022-03-10")

    datetime_ser = pd.Series([
        datetime.datetime(2021, 6, 15),
        datetime.datetime(2020, 1, 1),
    ], dtype=object)
    assert _min(datetime_ser) == datetime.datetime(2020, 1, 1)
    assert _max(datetime_ser) == datetime.datetime(2021, 6, 15)

    date_ser = pd.Series([
        datetime.date(2021, 6, 15),
        datetime.date(2020, 1, 1),
    ], dtype=object)
    assert _min(date_ser) == datetime.date(2020, 1, 1)
    assert _max(date_ser) == datetime.date(2021, 6, 15)

    # np.datetime64 scalars are not a datetime.date subclass but must still be
    # detected as temporal, not routed to the numeric np.isnan path.
    np_dt64_ser = pd.Series([
        np.datetime64("2021-06-15"),
        np.datetime64("2020-01-01"),
        np.datetime64("2022-03-10"),
    ], dtype=object)
    assert np_dt64_ser.dtype == object
    assert _min(np_dt64_ser) == np.datetime64("2020-01-01")
    assert _max(np_dt64_ser) == np.datetime64("2022-03-10")

    # Nulls mixed into the object series must be ignored, not crash.
    mixed_ser = pd.Series([
        None,
        pd.Timestamp("2020-01-01"),
        pd.NaT,
        pd.Timestamp("2021-01-01"),
    ], dtype=object)
    assert _min(mixed_ser) == pd.Timestamp("2020-01-01")
    assert _max(mixed_ser) == pd.Timestamp("2021-01-01")

    # Regression guard: numeric series still computed via the numeric path.
    numeric_ser = pd.Series([3.0, 1.0, np.nan, 2.0])
    assert _min(numeric_ser) == 1.0
    assert _max(numeric_ser) == 3.0

    # Regression guard: datetime64 series unchanged.
    datetime64_ser = pd.Series(pd.to_datetime(["2020-01-01", "2021-06-15"]))
    assert _min(datetime64_ser) == pd.Timestamp("2020-01-01")
    assert _max(datetime64_ser) == pd.Timestamp("2021-06-15")


def test_timestamp_stype_int_epoch_min_max_are_typed():
    # A FILE/parquet connector reads a `timestamp`-stype column raw, so an
    # int64 epoch-millisecond series reaches the col-stat kernel untyped. The
    # numeric branch of `_min`/`_max` would then emit a bare int, which is
    # serialized without the `kumo_timestamp_` prefix and later overflows when
    # reinterpreted as a datetime. The production-site coercion must turn it
    # into a `pd.Timestamp` before the kernel runs.
    #
    # 13-digit epoch-ms values around 2022-07-31.
    epoch_ms = [1659304801388, 1659304901388, 1659305001388]
    col_config = BaseColumnConfig(name="ts", stype=Stype.timestamp,
                                  timestamp_unit=None)

    raw_ser = pd.Series(epoch_ms, dtype="int64")

    # Without coercion the numeric branch returns a bare int.
    assert isinstance(_min(raw_ser), (int, np.integer))
    assert not isinstance(_min(raw_ser), pd.Timestamp)

    # Coerce at the production site, then the kernel yields typed `pd.Timestamp`
    # MIN/MAX values with the correct dates.
    coerced = coerce_timestamp_stype_series_for_stats(raw_ser, col_config)
    assert pd.api.types.is_datetime64_any_dtype(coerced)

    stats = compute_col_stats_python(coerced,
                                     [ColStatType.MIN, ColStatType.MAX])
    min_value = stats[ColStatType.MIN].value
    max_value = stats[ColStatType.MAX].value
    assert isinstance(min_value, pd.Timestamp)
    assert isinstance(max_value, pd.Timestamp)
    # The coercion collapses epoch-milliseconds to epoch-seconds with CEILING
    # division, so the sub-second `.388` is ceiled up to the next whole second
    # (`:01.388` -> `:02`, `:21.388` -> `:22`), matching what ingest writes to
    # the on-disk index.
    assert min_value == pd.Timestamp("2022-07-31 22:00:02")
    assert max_value == pd.Timestamp("2022-07-31 22:03:22")

    # An explicit `timestamp_unit` from the column config is honored over
    # magnitude inference and yields the same ceiled-second dates.
    unit_config = BaseColumnConfig(name="ts", stype=Stype.timestamp,
                                   timestamp_unit="ms")
    coerced_unit = coerce_timestamp_stype_series_for_stats(
        pd.Series(epoch_ms, dtype="int64"), unit_config)
    assert _min(coerced_unit) == pd.Timestamp("2022-07-31 22:00:02")

    # Scoping guard: a non-`timestamp`-stype numeric column passes through
    # untouched.
    numeric_config = BaseColumnConfig(name="n", stype=Stype.numerical)
    passthrough = coerce_timestamp_stype_series_for_stats(
        pd.Series(epoch_ms, dtype="int64"), numeric_config)
    assert pd.api.types.is_integer_dtype(passthrough)

    # An already-datetime series is still routed through the converter (NATIVE
    # hint), which ceils its sub-second precision to the epoch-seconds domain
    # rather than passing it through unchanged.
    already_dt = pd.Series(pd.to_datetime(epoch_ms, unit="ms"))
    coerced_native = coerce_timestamp_stype_series_for_stats(
        already_dt, col_config)
    assert pd.api.types.is_datetime64_any_dtype(coerced_native)
    assert coerced_native.min() == pd.Timestamp("2022-07-31 22:00:02")
    assert coerced_native.max() == pd.Timestamp("2022-07-31 22:03:22")


def test_ts_infer_float_null_epoch_min_max_correct():
    # An int64 epoch-millisecond column that contains a null arrives from a
    # FILE/parquet read as float64 (the null forces the float upcast). The
    # detector's float branch classifies the integral float epochs as
    # milliseconds, and the converter ceils them to epoch-seconds, so the values
    # resolve to the correct 2022 dates.
    #
    # 13-digit epoch-ms values around 2022-07-31, plus a trailing null.
    epoch_ms = [1659304801388, 1659304901388, 1659305001388, None]
    raw_ser = pd.Series(epoch_ms, dtype="float64")
    assert pd.api.types.is_float_dtype(raw_ser)

    col_config = BaseColumnConfig(name="ts", stype=Stype.timestamp,
                                  timestamp_unit=None)
    coerced = coerce_timestamp_stype_series_for_stats(raw_ser, col_config)

    stats = compute_col_stats_python(coerced,
                                     [ColStatType.MIN, ColStatType.MAX])
    min_value = stats[ColStatType.MIN].value
    max_value = stats[ColStatType.MAX].value

    # The null is dropped; the surviving three ms values span :01.388 to
    # :21.388, which the converter ceils to :02 and :22 (second precision).
    assert min_value == pd.Timestamp("2022-07-31 22:00:02")
    assert max_value == pd.Timestamp("2022-07-31 22:03:22")


def test_ts_infer_messy_epoch_does_not_raise():
    # A ``timestamp``-stype epoch column whose values do not cleanly fall into a
    # single digit-length bucket (here a 50/50 mix of 10-digit second and
    # 13-digit millisecond epochs) is genuinely ambiguous. The detector takes a
    # plurality vote and the converter is best-effort, so col-stat production
    # never raises on messy input.
    messy_epoch = [1659304801, 1659304901, 1659304801388, 1659304901388]
    raw_ser = pd.Series(messy_epoch, dtype="int64")

    col_config = BaseColumnConfig(name="ts", stype=Stype.timestamp,
                                  timestamp_unit=None)

    coerced = coerce_timestamp_stype_series_for_stats(raw_ser, col_config)
    stats = compute_col_stats_python(coerced,
                                     [ColStatType.MIN, ColStatType.MAX])

    # Best-effort AND deterministic: the plurality vote breaks the s/ms tie in
    # favor of the first-seen unit (seconds), so the two 13-digit ms values
    # overflow when read as seconds and drop to NaT, leaving the two 10-digit
    # second epochs. MIN/MAX are therefore typed pd.Timestamp values that land
    # in a sane 2022 range rather than bare ints or a far-future overflow.
    min_value = stats[ColStatType.MIN].value
    max_value = stats[ColStatType.MAX].value
    assert isinstance(min_value, pd.Timestamp)
    assert isinstance(max_value, pd.Timestamp)
    assert pd.Timestamp("2022-01-01") <= min_value <= max_value <= pd.Timestamp(
        "2023-01-01")


def test_ts_format_metadata_string_and_encoded_int_min_max_are_typed():
    # (a) A FILE/parquet connector reads a `timestamp`-stype STRING date column
    # raw. With an explicit `metadata['TS_FORMAT']` strftime hint, the col-stat
    # coercion resolves a FORMAT hint (via `_resolve_stats_temporal_hint`) and
    # applies the converter string path, yielding typed MIN/MAX.
    str_config = BaseColumnConfig(name="ts", stype=Stype.timestamp,
                                  metadata={TS_FORMAT_METADATA: "%Y-%m-%d"})
    str_ser = pd.Series(["2024-01-15", "2024-06-01", "garbage"])
    coerced_str = coerce_timestamp_stype_series_for_stats(str_ser, str_config)
    assert pd.api.types.is_datetime64_any_dtype(coerced_str)
    str_stats = compute_col_stats_python(coerced_str,
                                         [ColStatType.MIN, ColStatType.MAX])
    # The unparseable value drops; the two dates survive as typed timestamps.
    assert str_stats[ColStatType.MIN].value == pd.Timestamp("2024-01-15")
    assert str_stats[ColStatType.MAX].value == pd.Timestamp("2024-06-01")

    # (b) An encoded-date (YYYYMMDD) integer column carries no explicit hint, so
    # the coercion runs the detector, which resolves a FORMAT ("%Y%m%d") rather
    # than a UNIT (the corruption trap), and the converter applies it.
    int_config = BaseColumnConfig(name="d", stype=Stype.timestamp,
                                  timestamp_unit=None)
    int_ser = pd.Series([20230408, 20240115, 20250301], dtype="int64")
    coerced_int = coerce_timestamp_stype_series_for_stats(int_ser, int_config)
    int_stats = compute_col_stats_python(coerced_int,
                                         [ColStatType.MIN, ColStatType.MAX])
    assert int_stats[ColStatType.MIN].value == pd.Timestamp("2023-04-08")
    assert int_stats[ColStatType.MAX].value == pd.Timestamp("2025-03-01")


def test_ts_bare_int_min_roundtrip_does_not_overflow():
    # When a Datetime encoder receives a bare-integer MIN stat,
    # ``fetch_and_set_encoder_stats`` (kumo/graph/utils.py) serializes it by
    # prefixing ``str(value)`` with the internal timestamp marker. On
    # deserialization ``_recursive_from_config`` (kumo/config/mixin.py) strips
    # the prefix and routes the bare-integer string through
    # ``normalize_timestamp_like_value``, which magnitude-infers the unit for a
    # bare epoch, so the exact 2022 timestamp (millisecond precision) is
    # recovered rather than crashing on ``OutOfBoundsDatetime``. The value is a
    # 13-digit epoch-ms for 2022-07-31 22:00:01.388.
    bare_int_min = 1659304801388
    serialized = _KUMO_INTERNAL_TIMESTAMP_PREFIX + str(bare_int_min)

    result = _recursive_from_config(serialized)
    assert result == pd.Timestamp("2022-07-31 22:00:01.388")


def test_recursive_from_config_timestamp_deserialization_overflow_safe():
    # Consumer-side defense-in-depth for the ``kumo_timestamp_`` branch of
    # ``_recursive_from_config`` (kumo/config/mixin.py).

    # (a) A normal formatted timestamp string still deserializes to its
    # ``pd.Timestamp``.
    formatted = _KUMO_INTERNAL_TIMESTAMP_PREFIX + "2022-07-31 22:00:02"
    assert _recursive_from_config(formatted) == pd.Timestamp(
        "2022-07-31 22:00:02")

    # (b) A bare-integer epoch (13-digit ms) recovers a sane in-range timestamp
    # via magnitude inference.
    bare = _KUMO_INTERNAL_TIMESTAMP_PREFIX + "1659304801388"
    recovered = _recursive_from_config(bare)
    assert isinstance(recovered, pd.Timestamp)
    assert recovered == pd.Timestamp("2022-07-31 22:00:01.388")

    # (c) A wildly out-of-range value clamps to a valid ``pd.Timestamp``
    # (or degrades to ``NaT``) instead of raising.
    huge = _KUMO_INTERNAL_TIMESTAMP_PREFIX + ("9" * 30)
    clamped = _recursive_from_config(huge)
    assert clamped is pd.NaT or isinstance(clamped, pd.Timestamp)
    if isinstance(clamped, pd.Timestamp):
        assert pd.Timestamp.min <= clamped <= pd.Timestamp.max


def test_time_col_hist_object_dtype_datetime_series():
    # _histogram_half_month must also handle object-dtype temporal series.
    timestamp_ser = pd.Series([
        pd.Timestamp("2020-01-05"),
        pd.Timestamp("2020-02-20"),
        pd.Timestamp("2020-03-10"),
    ], dtype=object)
    hist = _histogram_half_month(timestamp_ser)
    assert len(hist) > 0
    assert all(isinstance(bin_date, datetime.date) for bin_date, _ in hist)

    date_ser = pd.Series([
        datetime.date(2020, 1, 5),
        datetime.date(2020, 3, 10),
    ], dtype=object)
    hist = _histogram_half_month(date_ser)
    assert len(hist) > 0


def test_time_col_hist_out_of_range_object_dtype():
    # Out-of-range timestamps must be binned best-effort over the in-range
    # portion, never raise OutOfBoundsDatetime.
    out_of_range_ser = pd.Series(
        [
            datetime.datetime(3122, 5, 1),  # > 2262 (out of range)
            datetime.datetime(2020, 1, 1),
            datetime.datetime(2021, 6, 15),
            datetime.datetime(11, 1, 1),  # < 1677 (out of range)
        ],
        dtype=object)
    hist = _histogram_half_month(out_of_range_ser)
    assert len(hist) > 0
    assert all(isinstance(bin_date, datetime.date) for bin_date, _ in hist)
    # Only the two in-range rows are counted.
    assert sum(int(count) for _, count in hist) == 2

    # Fully out-of-range -> empty result, not a crash.
    all_out_of_range_ser = pd.Series([
        datetime.datetime(3122, 5, 1),
        datetime.datetime(11, 1, 1),
    ], dtype=object)
    assert _histogram_half_month(all_out_of_range_ser) == []


def test_compute_col_stats_out_of_range_object_dtype_full_temporal_set():
    # An out-of-range object-dtype timestamp must not fail the whole column
    # compute when the full temporal stat set is requested.
    ser = pd.Series(
        [
            datetime.datetime(3122, 5, 1),  # out of range
            datetime.datetime(2020, 1, 1),
            datetime.datetime(2021, 6, 15),
        ],
        dtype=object)
    assert to_dtype(ser.dtype, ser=ser) == Dtype.date

    temporal_stats = [
        ColStatType.COUNT,
        ColStatType.NUM_NA,
        ColStatType.NA_FRACTION,
        ColStatType.NUM_UNIQUE,
        ColStatType.MIN,
        ColStatType.MAX,
        ColStatType.HISTOGRAM_HALF_MONTH,
    ]
    out = compute_col_stats_python(ser, stats=temporal_stats)

    assert ColStatType.MIN in out
    assert ColStatType.MAX in out
    assert out[ColStatType.MIN].value is not None
    assert out[ColStatType.MAX].value is not None

    # HISTOGRAM_HALF_MONTH is best-effort over the in-range rows.
    assert ColStatType.HISTOGRAM_HALF_MONTH in out
    hist = out[ColStatType.HISTOGRAM_HALF_MONTH].value
    assert len(hist) > 0
    assert sum(int(count) for _, count in hist) == 2

    assert out[ColStatType.COUNT].value == 3


def test_compute_col_stats_np_datetime64_object_dtype():
    # np.datetime64 object series must type as Dtype.date and survive MIN/MAX
    # without the numeric-path TypeError.
    ser = pd.Series([
        np.datetime64("2021-06-15"),
        np.datetime64("2020-01-01"),
        np.datetime64("2022-03-10"),
    ], dtype=object)
    assert ser.dtype == object
    assert to_dtype(ser.dtype, ser=ser) == Dtype.date

    out = compute_col_stats_python(ser,
                                   stats=[ColStatType.MIN, ColStatType.MAX])
    assert out[ColStatType.MIN].value == np.datetime64("2020-01-01")
    assert out[ColStatType.MAX].value == np.datetime64("2022-03-10")


def test_compute_edge_stats():
    num_src_nodes = 10
    num_dst_nodes = 10
    test_data = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 0]]
    edge_index = torch.Tensor(test_data).type(torch.int64)
    stat_result = compute_edge_stats_python(edge_index, num_src_nodes,
                                            num_dst_nodes)

    assert stat_result[EdgeStatType.EDGE_COUNT].value == 6
    assert stat_result[EdgeStatType.NUM_SRC_NONZERO_DEGREE].value == 6
    assert stat_result[EdgeStatType.NUM_DST_NONZERO_DEGREE].value == 5
    assert stat_result[EdgeStatType.MAX_SRC_DEGREE].value == 1
    assert stat_result[EdgeStatType.MAX_DST_DEGREE].value == 2

    test_data = [[0, 1, 2, 3, 4, 5, 0, 1], [0, 1, 2, 3, 4, 0, 5, 1]]
    edge_index = torch.Tensor(test_data).type(torch.int64)
    stat_result = compute_edge_stats_python(edge_index, num_src_nodes,
                                            num_dst_nodes)

    assert stat_result[EdgeStatType.EDGE_COUNT].value == 8
    assert stat_result[EdgeStatType.NUM_SRC_NONZERO_DEGREE].value == 6
    assert stat_result[EdgeStatType.NUM_DST_NONZERO_DEGREE].value == 6
    assert stat_result[EdgeStatType.MAX_SRC_DEGREE].value == 2
    assert stat_result[EdgeStatType.MAX_DST_DEGREE].value == 2


def test_table_stats_parquet():
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'parquet_data', 'FINANCIAL')

    graph = Graph(name='FINANCIAL')
    graph.connect(ParquetConnector('parquet', root_dir))
    graph.add('TRANS', pkey=PrimaryKey('TRANS_ID', dtype=Dtype.int64))
    graph['TRANS'].add_col('AMOUNT', dtype=Dtype.float)
    graph.compute_table_stats()

    n_rows = graph['TRANS'].num_rows()
    assert n_rows == 5000

    # This new dataset has multiple parquet files for the same table.
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'parquet_data')

    graph = Graph(name='FINANCIAL')
    graph.connect(ParquetConnector('parquet', root_dir))
    graph.add('trans_dataset', pkey=PrimaryKey('TRANS_ID', dtype=Dtype.int64))
    graph['trans_dataset'].add_col('AMOUNT', dtype=Dtype.float)
    graph.compute_table_stats()

    n_rows = graph['trans_dataset'].num_rows()
    assert n_rows == 3072


@pytest.mark.snowflake
@withSnowflakeCredentials
def test_table_stats_snowflake():
    connector = SnowflakeConnector('snowflake', account='xva19026',
                                   warehouse='WH_XS', database='KUMO',
                                   schema_name='SCHEMA_TEST')

    graph = Graph(name='test_graph')
    graph.connect(connector)
    graph.add('TYPES_TEST')
    graph.add('TYPES_TEST_2')
    graph.add('TEST_VIEW_2')

    for approx in [True, False]:
        stats = connector.get_table_stats(
            'TYPES_TEST', [TableStatType.NUM_ROWS, TableStatType.SOURCE_SIZE],
            approx=approx)
        assert stats[TableStatType.NUM_ROWS].value == 100
        assert stats[TableStatType.SOURCE_SIZE].value == 13312

        # views have number of rows but won't have source size
        stats = connector.get_table_stats(
            'TEST_VIEW_2', [TableStatType.NUM_ROWS, TableStatType.SOURCE_SIZE],
            approx=approx)
        assert stats[TableStatType.NUM_ROWS].value == 100


def test_table_stats_csv(create_graph):
    self_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(self_dir, '..', 'csv_data', 'FINANCIAL')

    graph = Graph(name='FINANCIAL')
    graph.connect(CSVConnector('csv', root_dir, na_values='?', sep=';'))
    graph.add('TRANS', pkey=PrimaryKey('TRANS_ID', dtype=Dtype.int64))

    root_dir = osp.join(self_dir, '..', 'csv_data', 'shard_test')
    graph = Graph(name='test')
    graph.connect(CSVConnector('csv', root_dir, na_values='?', sep=';'))
    graph.add('sharded_csv', pkey=PrimaryKey('DISTRICT_ID', dtype=Dtype.int64))


@withS3Credentials
def test_table_stats_s3_parquet(seed_col_stats):
    # This test bypasses the ``_create_graph`` conftest path that
    # triggers ``_seed_file_backed_graph_col_stats``. The autouse col-stats
    # fallback only handles local file-backed / Pandas / BigQuery connectors;
    # S3-backed ParquetConnector reads aren't seeded automatically. Seed each
    # table explicitly so ``infer_metadata`` can read col-stats from the cache.
    s3_connector = ParquetConnector('parquet',
                                    's3://kumo-datasets/financial/parquet')
    s3_df = s3_connector.read_table('TRANS')
    s3_data_version = s3_connector.get_data_version('TRANS')
    seed_col_stats(s3_df, table_name='TRANS', data_version=s3_data_version,
                   primary_key='TRANS_ID')

    graph = Graph(name='FINANCIAL')
    graph.connect(s3_connector)
    graph.add('TRANS', pkey=PrimaryKey('TRANS_ID', dtype=Dtype.int64))
    graph['TRANS'].add_col('AMOUNT', dtype=Dtype.float)
    graph.infer_metadata()

    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'parquet_data')

    local_connector = ParquetConnector('parquet', root_dir)
    local_df = local_connector.read_table('trans_dataset')
    local_data_version = local_connector.get_data_version('trans_dataset')
    seed_col_stats(local_df, table_name='trans_dataset',
                   data_version=local_data_version, primary_key='TRANS_ID')

    graph = Graph(name='FINANCIAL')
    graph.connect(local_connector)
    graph.add('trans_dataset', pkey=PrimaryKey('TRANS_ID', dtype=Dtype.int64))
    graph['trans_dataset'].add_col('AMOUNT', dtype=Dtype.float)
    graph.infer_metadata()


def test_detect_multicat_sep_for_single_col():
    text = 'SingleColumn'
    assert detect_multicat_sep(text) is None

    df = pd.DataFrame(data={'col': [1, 2, 3, 4]})
    text = df.to_csv(index=False)
    assert detect_multicat_sep(text) is None


def test_detect_multicat_sep_non_allowlist():
    text = 'col1#col2#col3#col4*col5'
    assert detect_multicat_sep(text) is None


@pytest.mark.parametrize(
    'values',
    [
        ['08:15:00', '09:30:00', '10:45:00'],
        ['["red", "blue"]', '["green"]'],
        ['', '', ''],
        [None, None],
        ['red^blue', 'green^yellow'],
    ],
)
def test_detect_multicat_sep_rejects_unusable_scalar_encodings(values):
    assert detect_multicat_sep_values(values) is None


@pytest.mark.parametrize('separator', [',', '|', ';', '\t'])
def test_detect_multicat_sep_matches_diskgraph_supported_delimiters(separator):
    values = [f'red{separator}blue'] * 10 + ['green'] * 990
    assert detect_multicat_sep_values(values) == separator


def test_detect_multicat_sep_rejects_candidate_below_one_percent():
    values = ['red,blue'] * 9 + ['green'] * 991
    assert detect_multicat_sep_values(values) is None


def test_detect_multicat_sep_threshold_counts_only_non_null_rows():
    values = ['red,blue'] + [None] * 999
    assert detect_multicat_sep_values(values) == ','


def test_detect_multicat_sep_tie_uses_deterministic_candidate_order():
    values = ['red,blue'] * 10 + ['green|yellow'] * 10
    assert detect_multicat_sep_values(values) == ','


_IRRATIONAL_NUMBERS = pd.Series(
    [1.0, 2.0, 3.0, np.nan, np.inf, -np.inf, None, pd.NA], dtype='Float64')


def _test_stat(stat_type: ColStatType, vals: List[float] | List[int]) -> List:
    is_data_int = len(vals) != 0 and isinstance(vals[0], int)
    ser = pd.Series(vals, dtype=np.dtype('int64' if is_data_int else 'float64'))
    stats = compute_col_stats_python(ser, [stat_type])
    return stats[stat_type].value


def test_histogram():
    assert _test_stat(ColStatType.HISTOGRAM, []) == []
    assert _test_stat(ColStatType.HISTOGRAM, [1.0]) == []
    assert _test_stat(ColStatType.HISTOGRAM, [1.0, 1.0]) == []
    assert _test_stat(ColStatType.HISTOGRAM, [1]) == []
    assert _test_stat(ColStatType.HISTOGRAM, [1, 1]) == []

    histogram = _test_stat(ColStatType.HISTOGRAM, _IRRATIONAL_NUMBERS)
    expected = [(1.0, 33.33333333), (1.1, 0.0), (1.2, 0.0), (1.3, 0.0),
                (1.4, 0.0), (1.5, 0.0), (1.6, 0.0), (1.7, 0.0), (1.8, 0.0),
                (1.9, 0.0), (2.0, 33.33333333), (2.1, 0.0), (2.2, 0.0),
                (2.3, 0.0), (2.4, 0.0), (2.5, 0.0), (2.6, 0.0), (2.7, 0.0),
                (2.8, 0.0), (2.9, 33.33333333)]
    assert histogram == expected

    histogram = _test_stat(ColStatType.HISTOGRAM, [1.0, 2.0, 3.0])
    # expected is same as above
    assert histogram == expected

    histogram = _test_stat(ColStatType.HISTOGRAM, [1, 2, 3])
    expected = [(1, 33.33333333), (2, 66.66666667)]
    assert isinstance(histogram[0][0], int)
    assert histogram == expected

    histogram = _test_stat(ColStatType.HISTOGRAM, [0, NUM_BINS_HISTOGRAM])
    expected = [(0, 50.0), (1, 0.0), (2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0),
                (6, 0.0), (7, 0.0), (8, 0.0), (9, 0.0), (10, 0.0), (11, 0.0),
                (12, 0.0), (13, 0.0), (14, 0.0), (15, 0.0), (16, 0.0),
                (17, 0.0), (18, 0.0), (19, 50.0)]
    assert isinstance(histogram[0][0], int)
    assert histogram == expected

    histogram = _test_stat(ColStatType.HISTOGRAM, [0, NUM_BINS_HISTOGRAM + 1])
    expected = [(0, 50.0), (2, 0.0), (4, 0.0), (6, 0.0), (8, 0.0), (10, 0.0),
                (12, 0.0), (14, 0.0), (16, 0.0), (18, 0.0), (20, 50.0)]
    assert isinstance(histogram[0][0], int)
    assert histogram == expected

    histogram = _test_stat(ColStatType.HISTOGRAM, [-2**63, 2**63 - 1])
    expected = [(-9223372036854775808, 50.0), (-8301034833169298227, 0.0),
                (-7378697629483820646, 0.0), (-6456360425798343065, 0.0),
                (-5534023222112865484, 0.0), (-4611686018427387903, 0.0),
                (-3689348814741910322, 0.0), (-2767011611056432741, 0.0),
                (-1844674407370955160, 0.0), (-922337203685477579, 0.0),
                (2, 0.0), (922337203685477583, 0.0), (1844674407370955164, 0.0),
                (2767011611056432745, 0.0), (3689348814741910326, 0.0),
                (4611686018427387907, 0.0), (5534023222112865488, 0.0),
                (6456360425798343069, 0.0), (7378697629483820650, 0.0),
                (8301034833169298231, 50.0)]
    assert isinstance(histogram[0][0], int)
    assert histogram == expected


def test_irrational_numbers():
    assert _test_stat(ColStatType.MIN, _IRRATIONAL_NUMBERS) == -np.inf
    assert _test_stat(ColStatType.MAX, _IRRATIONAL_NUMBERS) == np.inf
    assert _test_stat(ColStatType.COUNT, _IRRATIONAL_NUMBERS) == 8
    assert _test_stat(ColStatType.NUM_NA, _IRRATIONAL_NUMBERS) == 5
    assert _test_stat(ColStatType.NA_FRACTION, _IRRATIONAL_NUMBERS) == 5 / 8
    assert _test_stat(ColStatType.NUM_IRRATIONAL, _IRRATIONAL_NUMBERS) == 5
    assert _test_stat(ColStatType.MEAN, _IRRATIONAL_NUMBERS) == 2.0
    assert (_test_stat(ColStatType.QUANTILES,
                       _IRRATIONAL_NUMBERS) == [1.5, 2.0, 2.5])
    assert _test_stat(ColStatType.QUANTILE25, _IRRATIONAL_NUMBERS) == 1.5
    assert _test_stat(ColStatType.MEDIAN, _IRRATIONAL_NUMBERS) == 2.0
    assert _test_stat(ColStatType.QUANTILE75, _IRRATIONAL_NUMBERS) == 2.5
    assert _test_stat(ColStatType.STD, _IRRATIONAL_NUMBERS) == 1.0
    assert pd.isna(_test_stat(ColStatType.KURTOSIS, _IRRATIONAL_NUMBERS))


def test_seq_stat():
    ser = pd.Series([[1.0, 2.0, 3.0], [4.0, 5.0], [0.0, 0.1, 0.2],
                     [0.3, 0.4, 0.5]])
    stats = compute_col_stats_python(ser, [
        ColStatType.SEQUENCE_MAX_LENGTH,
        ColStatType.SEQUENCE_MIN_LENGTH,
    ])
    print(stats)
    assert stats[ColStatType.SEQUENCE_MAX_LENGTH].value == 3
    assert stats[ColStatType.SEQUENCE_MIN_LENGTH].value == 2


def test_decimal_column_yields_numerical_stats():
    r"""A warehouse ``DECIMAL``/``NUMERIC`` reaches pandas as an ``object``
    column of :class:`decimal.Decimal`, which ``to_dtype`` reports as
    ``string``. Uncoerced, the caller keeps only the stats ``string``
    supports and a ``numerical`` column arrives at
    ``ColumnStats.from_stats_dict`` without the numerical stats it requires
    (kumo-ai/kumo#30917).
    """
    from decimal import Decimal

    from kumo.connector.connector import _compute_col_stats_from_sample
    from kumo.connector.utils import coerce_decimal_series_for_stats

    col_config = BaseColumnConfig(name='loan_prncpl_usd_amt',
                                  stype=Stype.numerical, dtype=Dtype.float64)
    requested = [
        ColStatType.NUM_NA, ColStatType.COUNT, ColStatType.NUM_UNIQUE,
        ColStatType.MIN, ColStatType.MAX, ColStatType.MEAN,
        ColStatType.QUANTILES, ColStatType.STD
    ]
    values = [Decimal(f'{index}.250000000') for index in range(1000)]

    # Uncoerced, every numerical stat is silently dropped.
    raw = pd.Series(values)
    assert to_dtype(raw.dtype, ser=raw) is Dtype.string
    assert not ColStatType.QUANTILES.supports(Dtype.string)

    stats = _compute_col_stats_from_sample(
        {col_config: requested},
        pd.DataFrame({col_config.name: values}),
        approx=True,
        data_version=None,
    )[col_config]

    assert set(stats) == set(requested)
    assert stats[ColStatType.MIN].value == 0.25
    assert stats[ColStatType.MAX].value == 999.25
    assert len(stats[ColStatType.QUANTILES].value) == 3

    coerced = coerce_decimal_series_for_stats(raw, col_config)
    assert pd.api.types.is_float_dtype(coerced)


def test_decimal_coercion_targets_the_declared_dtype():
    r"""The stats describe the values as ingested, so an integer column is
    coerced to an integer rather than to a general float, and a column that
    is not decimal-backed is left alone."""
    from decimal import Decimal

    from kumo.connector.utils import coerce_decimal_series_for_stats

    keys = pd.Series([Decimal('7000000000001'), Decimal('7000007000000')])
    as_int = coerce_decimal_series_for_stats(
        keys, BaseColumnConfig(name='k', stype=Stype.ID, dtype=Dtype.int64))
    assert pd.api.types.is_integer_dtype(as_int)
    assert list(as_int) == [7000000000001, 7000007000000]

    text = pd.Series(['a', 'b'])
    assert coerce_decimal_series_for_stats(
        text,
        BaseColumnConfig(name='s', stype=Stype.categorical,
                         dtype=Dtype.string)).equals(text)

    floats = pd.Series([1.25, 2.5])
    assert coerce_decimal_series_for_stats(
        floats,
        BaseColumnConfig(name='f', stype=Stype.numerical,
                         dtype=Dtype.float64)).equals(floats)


@pytest.mark.parametrize('values, expected', [
    ([1, 2, 3, 4] + [None] * 96, (4, 0.04)),
    ([None] * 4, (0, 0.0)),
    ([], (0, None)),
])
def test_distinct_sample_statistics_include_null_rows(values, expected):
    from kumo_core.stype_inference import infer_stype_from_stats

    series = pd.Series(values, dtype='Int64')
    stats = compute_col_stats_python(
        series, [ColStatType.NUM_UNIQUE, ColStatType.UNIQUE_FRACTION])
    actual = (stats[ColStatType.NUM_UNIQUE].value,
              stats[ColStatType.UNIQUE_FRACTION].value)
    assert actual == expected
    if expected[0]:
        assert infer_stype_from_stats(Dtype.int64, *actual) == Stype.categorical
