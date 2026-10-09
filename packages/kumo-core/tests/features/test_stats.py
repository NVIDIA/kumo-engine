# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import pytest
from kumoapi.typing import Stype

from kumo_core.features.stats import get_stats
from kumo_core.features.stats import (
    CategoricalStats,
    MultiCategoricalStats,
    NumericalStats,
    SequenceStats,
    Stats,
    TextStats,
)


@pytest.mark.parametrize('approx', [True, False])
def test_numerical_stats(approx: bool) -> None:
    ser = pd.Series([0.0, 10.0, float('NaN'), 20.0, 50.0, 100.0, float('inf')])
    stats = get_stats(ser, Stype.numerical, approx=approx)
    assert isinstance(stats, NumericalStats)
    assert stats.approx == approx
    assert stats.count == 5
    assert stats.na_frac == pytest.approx(2 / 7)
    assert stats.mean == 180 / 5
    assert stats.std == pytest.approx(36.11094)
    assert stats.min == 0
    assert stats.max == 100
    assert stats.kurtosis == pytest.approx(0.947345)
    assert stats.quantiles == (10, 20, 50)
    assert stats.median == 20

    ser = pd.Series([float('NaN')])
    stats = get_stats(ser, Stype.numerical, approx=approx)
    assert isinstance(stats, NumericalStats)
    assert stats.approx == approx
    assert stats.count == 0
    assert stats.na_frac == 1
    assert stats.mean == 0
    assert stats.std == 1
    assert stats.min == 0
    assert stats.max == 0
    assert stats.kurtosis == 0
    assert stats.quantiles == (0, 0, 0)
    assert stats.median == 0


@pytest.mark.parametrize('stype', [Stype.categorical, Stype.ID])
@pytest.mark.parametrize('approx', [True, False])
def test_categorical_stats(stype: Stype, approx: bool) -> None:
    ser = pd.Series(['A', 'A', '', None, 'B', 'C', 'C'])
    stats = get_stats(ser, stype, approx=approx)
    assert isinstance(stats, CategoricalStats)
    assert stats.approx == approx
    assert stats.count == 5
    assert stats.na_frac == pytest.approx(2 / 7)
    assert stats.num_unique == 3
    assert stats.category_count == [['A', 2], ['C', 2], ['B', 1]]

    ser = pd.Series([None])
    stats = get_stats(ser, stype, approx=approx)
    assert isinstance(stats, CategoricalStats)
    assert stats.approx == approx
    assert stats.count == 0
    assert stats.na_frac == 1
    assert stats.num_unique == 0
    assert stats.category_count == []


@pytest.mark.parametrize('approx', [True, False])
def test_string_multicategorical_stats(approx: bool) -> None:
    ser = pd.Series(['A|B', 'A|C', '', None, 'B|A', 'C|', 'C'])
    stats = get_stats(ser, Stype.multicategorical, approx=approx)
    assert isinstance(stats, MultiCategoricalStats)
    assert stats.approx == approx
    assert stats.count == 5
    assert stats.na_frac == pytest.approx(2 / 7)
    assert stats.sep == '|'
    assert stats.num_unique == 3

    ser = pd.Series([None])
    stats = get_stats(ser, Stype.multicategorical, approx=approx)
    assert isinstance(stats, MultiCategoricalStats)
    assert stats.approx == approx
    assert stats.count == 0
    assert stats.na_frac == 1
    assert stats.sep is None
    assert stats.num_unique == 0


@pytest.mark.parametrize('approx', [True, False])
def test_array_multicategorical_stats(approx: bool) -> None:
    ser = pd.Series([['A', 'B'], ['A', 'C'], [], None, ['B', 'A'], ['C', ''],
                     ['C']])
    stats = get_stats(ser, Stype.multicategorical, approx=approx)
    assert isinstance(stats, MultiCategoricalStats)
    assert stats.approx == approx
    assert stats.count == 6
    assert stats.na_frac == pytest.approx(1 / 7)
    assert stats.sep is None
    assert stats.num_unique == 3


@pytest.mark.parametrize('approx', [True, False])
def test_string_sequence_stats(approx: bool) -> None:
    ser = pd.Series(['[0,1,2]', '[0,2]', None])
    stats = get_stats(ser, Stype.sequence, approx=approx)
    assert isinstance(stats, SequenceStats)
    assert stats.approx == approx
    assert stats.count == 2
    assert stats.na_frac == pytest.approx(1 / 3)
    assert stats.min_length == 2
    assert stats.max_length == 3
    assert stats.mean == 1
    assert stats.std == pytest.approx(0.894427)

    ser = pd.Series([None])
    stats = get_stats(ser, Stype.sequence, approx=approx)
    assert isinstance(stats, SequenceStats)
    assert stats.approx == approx
    assert stats.count == 0
    assert stats.na_frac == 1
    assert stats.min_length == 0
    assert stats.max_length == 0
    assert stats.mean == 0
    assert stats.std == 1


@pytest.mark.parametrize('approx', [True, False])
def test_array_sequence_stats(approx: bool) -> None:
    ser = pd.Series([[0, 1, 2], [0, 2], None])
    stats = get_stats(ser, Stype.sequence, approx=approx)
    assert isinstance(stats, SequenceStats)
    assert stats.approx == approx
    assert stats.count == 2
    assert stats.na_frac == pytest.approx(1 / 3)
    assert stats.min_length == 2
    assert stats.max_length == 3
    assert stats.mean == 1
    assert stats.std == pytest.approx(0.894427)


@pytest.mark.parametrize('approx', [True, False])
def test_text_stats(approx: bool) -> None:
    ser = pd.Series([
        'Hello, greetings from Europe',
        'Wow, t1hs is cool dawdajoadaw!',
        None,
        '',
    ])
    stats = get_stats(ser, Stype.text, approx=approx)
    assert isinstance(stats, TextStats)
    assert stats.approx == approx
    assert stats.count == 2
    assert stats.na_frac == pytest.approx(1 / 2)
    assert stats.avg_tokens == pytest.approx((4 + 5) / 2)
    assert stats.max_tokens == 5
    assert stats.avg_non_char == pytest.approx(1 / 9)
    assert stats.glove_overlap == pytest.approx(7 / 9)

    ser = pd.Series([None])
    stats = get_stats(ser, Stype.text, approx=approx)
    assert isinstance(stats, TextStats)
    assert stats.approx == approx
    assert stats.count == 0
    assert stats.na_frac == 1
    assert stats.avg_tokens == 0
    assert stats.max_tokens == 0
    assert stats.avg_non_char == 0
    assert stats.glove_overlap == 1


@pytest.mark.parametrize('approx', [True, False])
def test_fallback_stats(approx: bool) -> None:
    ser = pd.Series(['1990-01-01', None])
    stats = get_stats(ser, Stype.timestamp, approx=approx)
    assert isinstance(stats, Stats)
    assert stats.approx == approx
    assert stats.count == 1
    assert stats.na_frac == pytest.approx(1 / 2)

    ser = pd.Series([None])
    stats = get_stats(ser, Stype.timestamp, approx=approx)
    assert isinstance(stats, Stats)
    assert stats.approx == approx
    assert stats.count == 0
    assert stats.na_frac == 1
