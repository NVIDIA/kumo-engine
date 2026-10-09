# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from kumoapi.encoder import (
    Datetime,
    Hash,
    Index,
    MaxLogNumerical,
    MinLogNumerical,
    MultiCategorical,
    Numerical,
    NumericalList,
    TextEmbedding,
)
from kumoapi.typing import ColStatType, Stype

from kumo_core.encoders.selection import compute_missing_stats_
from tests.support.encoders import get_encoder


@pytest.mark.parametrize('i', range(10))
def test_get_numerical_encoder(i: int) -> None:
    encoder = get_encoder(Stype.numerical)
    assert isinstance(encoder, Numerical | MinLogNumerical | MaxLogNumerical)

    ser = pd.Series([0.1, 0.5, None])
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert encoder._stats.get(ColStatType.MEAN, 0.3) == 0.3
    assert encoder._stats.get(ColStatType.STD, 0.2) == 0.2
    assert encoder._stats.get(ColStatType.MIN, 0.1) == 0.1
    assert encoder._stats.get(ColStatType.MAX, 0.5) == 0.5
    assert encoder._stats.get(ColStatType.QUANTILES,
                              (0.2, 0.3, 0.4)) == (0.2, 0.3, 0.4)

    ser = pd.Series([None])
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert encoder._stats.get(ColStatType.MEAN, 0) == 0
    assert encoder._stats.get(ColStatType.STD, 1) == 1
    assert encoder._stats.get(ColStatType.MIN, 0) == 0
    assert encoder._stats.get(ColStatType.MAX, 0) == 0
    assert encoder._stats.get(ColStatType.QUANTILES, (0, 0, 0)) == (0, 0, 0)

    ser = pd.Series([], dtype=float)
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert encoder._stats.get(ColStatType.MEAN, 0) == 0
    assert encoder._stats.get(ColStatType.STD, 1) == 1
    assert encoder._stats.get(ColStatType.MIN, 0) == 0
    assert encoder._stats.get(ColStatType.MAX, 0) == 0
    assert encoder._stats.get(ColStatType.QUANTILES, (0, 0, 0)) == (0, 0, 0)


@pytest.mark.parametrize('i', range(10))
@pytest.mark.parametrize('stype', [Stype.categorical, Stype.ID])
def test_get_categorical_encoder(i: int, stype: Stype) -> None:
    encoder = get_encoder(stype)
    if isinstance(encoder, Index):
        assert encoder.min_occ >= 1 and encoder.min_occ <= 10
    else:
        assert isinstance(encoder, Hash)
        assert encoder.num_components >= 1 and encoder.num_components <= 10

    ser = pd.Series(['A', 'A', 'B', None])
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert encoder._stats[ColStatType.CATEGORY_COUNTS] == [['A', 2], ['B', 1]]

    ser = pd.Series([None], dtype=str)
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert encoder._stats[ColStatType.CATEGORY_COUNTS] == []

    ser = pd.Series([], dtype=str)
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert encoder._stats[ColStatType.CATEGORY_COUNTS] == []


@pytest.mark.parametrize('i', range(10))
def test_get_multi_categorical_encoder(i: int) -> None:
    encoder = get_encoder(Stype.multicategorical)
    assert isinstance(encoder, MultiCategorical)
    assert encoder.min_occ >= 1 and encoder.min_occ <= 10

    ser = pd.Series(['A|B', 'A', 'B|A', None])
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert encoder._stats[ColStatType.MULTI_CATEGORIES_SEPARATOR] == '|'
    assert encoder._stats[ColStatType.MULTI_CATEGORY_COUNTS] == [
        ['A', 3],
        ['B', 2],
    ]

    ser = pd.Series([
        np.array(['A', 'B']),
        np.array(['A']),
        np.array(['B', 'A']),
        ['A', 'C'],  # list, not an array
        None,
    ])
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert encoder._stats[ColStatType.MULTI_CATEGORY_COUNTS] == [
        ['A', 4],
        ['B', 2],
        ['C', 1],
    ]

    ser = pd.Series([None], dtype=str)
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert encoder._stats[ColStatType.MULTI_CATEGORIES_SEPARATOR] is None
    assert encoder._stats[ColStatType.MULTI_CATEGORY_COUNTS] == []

    ser = pd.Series([], dtype=str)
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert encoder._stats[ColStatType.MULTI_CATEGORIES_SEPARATOR] is None
    assert encoder._stats[ColStatType.MULTI_CATEGORY_COUNTS] == []


def test_get_timestamp_encoder() -> None:
    encoder = get_encoder(Stype.timestamp)
    assert isinstance(encoder, Datetime)

    ser = pd.Series(['2018-01-05', '2019-01-05', '', None])
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert (encoder._stats[ColStatType.MIN] ==
            'kumo_timestamp_2018-01-05 00:00:00')
    assert (encoder._stats[ColStatType.MAX] ==
            'kumo_timestamp_2019-01-05 00:00:00')

    ser = pd.Series(['', None], dtype=str)
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert (encoder._stats[ColStatType.MIN] ==
            'kumo_timestamp_1970-01-01 00:00:00')
    assert (encoder._stats[ColStatType.MAX] ==
            'kumo_timestamp_1970-01-01 00:00:00')

    ser = pd.Series([], dtype=str)
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert (encoder._stats[ColStatType.MIN] ==
            'kumo_timestamp_1970-01-01 00:00:00')
    assert (encoder._stats[ColStatType.MAX] ==
            'kumo_timestamp_1970-01-01 00:00:00')


def test_get_text_encoder() -> None:
    encoder = get_encoder(Stype.text)
    assert isinstance(encoder, TextEmbedding)

    ser = pd.Series(['get old', 'get', '', None])
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert len(encoder.required_stats) == 0

    ser = pd.Series([None], dtype=str)
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert len(encoder.required_stats) == 0

    ser = pd.Series([], dtype=str)
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert len(encoder.required_stats) == 0


def test_get_sequence_encoder() -> None:
    encoder = get_encoder(Stype.sequence)
    assert isinstance(encoder, NumericalList)

    ser = pd.Series([[0.1, 0.2], [0.3], None])
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert encoder._stats[ColStatType.SEQUENCE_MIN_LENGTH] == 1
    assert encoder._stats[ColStatType.SEQUENCE_MAX_LENGTH] == 2

    ser = pd.Series(['[0.1, 0.2]', '[0.3]', None])
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert encoder._stats[ColStatType.SEQUENCE_MIN_LENGTH] == 1
    assert encoder._stats[ColStatType.SEQUENCE_MAX_LENGTH] == 2

    ser = pd.Series([None], dtype=str)
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert encoder._stats[ColStatType.SEQUENCE_MIN_LENGTH] == 0
    assert encoder._stats[ColStatType.SEQUENCE_MAX_LENGTH] == 0

    ser = pd.Series([], dtype=str)
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert encoder._stats[ColStatType.SEQUENCE_MIN_LENGTH] == 0
    assert encoder._stats[ColStatType.SEQUENCE_MAX_LENGTH] == 0

    # test NumericalList with Standard
    encoder = NumericalList(scaler='standard')
    ser = pd.Series([[0.1, 0.2], [0.3], None])
    encoder = compute_missing_stats_(encoder, ser, force_refresh=True)
    assert set(encoder._stats.keys()) == encoder.required_stats
    assert encoder._stats[ColStatType.SEQUENCE_MIN_LENGTH] == 1
    assert encoder._stats[ColStatType.SEQUENCE_MAX_LENGTH] == 2
    test_values = np.array([0.1, 0.2, 0.3])
    np.testing.assert_almost_equal(encoder._stats[ColStatType.SEQUENCE_MEAN],
                                   np.mean(test_values))
    np.testing.assert_almost_equal(encoder._stats[ColStatType.SEQUENCE_STD],
                                   np.std(test_values))
