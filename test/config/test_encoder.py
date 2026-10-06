# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict

import pandas as pd
import pytest
from kumo_core.encoders.selection import compute_missing_stats_
from kumo_core.types import ColStatType
from kumoapi.encoder import (
    Datetime,
    EncoderType,
    Hash,
    Index,
    MaxLogNumerical,
    MinLogNumerical,
    MultiCategorical,
    NAStrategy,
    Null,
    Numerical,
    NumericalList,
    Scaler,
    TextEmbedding,
)
from pydantic import TypeAdapter, ValidationError

from io_utils import kumo_dataclass
from io_utils.locate import locate
from kumo.testing import withTestDataFrame


@kumo_dataclass
class Encoders:
    """Some config class that holds an Encoder as one of its attributes."""
    encoders: list[EncoderType]


@pytest.mark.parametrize(
    ('encoder_type', 'deprecated_args'),
    [
        (Index, {
            'na_values': None
        }),
        (Hash, {
            'num_components': 2,
            'min_occ': 1
        }),
        (MultiCategorical, {
            'na_values': None
        }),
        (MaxLogNumerical, {
            'base': 'natural'
        }),
        (MinLogNumerical, {
            'base': 'natural'
        }),
        (NumericalList, {
            'num_dimensions': 3
        }),
        (Datetime, {
            'format': None
        }),
        (Datetime, {
            'include_month': True
        }),
        (Datetime, {
            'include_week_of_year': False
        }),
    ],
)
def test_deprecated_encoder_args_are_rejected(encoder_type, deprecated_args):
    with pytest.raises(ValidationError, match='Unexpected keyword argument'):
        encoder_type(**deprecated_args)


def test_supported_encoder_values_round_trip():
    encoder = Numerical(na_strategy=NAStrategy.RAISE)
    encoders = Encoders(encoders=[encoder])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))


@pytest.mark.parametrize(
    'encoder',
    [
        Null(),
        Numerical(),
        MaxLogNumerical(),
        MinLogNumerical(),
        Index(),
        Hash(num_components=2),
        MultiCategorical(),
        TextEmbedding(),
        NumericalList(),
        Datetime(),
    ],
)
def test_encoder_target_is_canonical_and_resolvable(encoder):
    assert encoder._target_.startswith('kumoapi.encoder.')
    assert locate(encoder._target_) is encoder.__class__


def test_invalid_na_strategy():
    with pytest.raises(ValueError,
                       match=r"Input should be.*input_value='mean'"):
        Index(na_strategy='mean')

    with pytest.raises(ValueError,
                       match=r"Input should be.*input_value='mean'"):
        Hash(num_components=10, na_strategy='mean')

    with pytest.raises(ValueError,
                       match=r"Input should be.*input_value='mean'"):
        MultiCategorical(na_strategy='mean')

    with pytest.raises(ValueError,
                       match=r"Input should be.*input_value='mean'"):
        TextEmbedding(na_strategy='mean')

    with pytest.raises(ValueError,
                       match=r"Input should be.*input_value='most_frequent'"):
        Numerical(na_strategy='most_frequent')

    with pytest.raises(ValueError,
                       match=r"Input should be.*input_value='most_frequent'"):
        MaxLogNumerical(na_strategy='most_frequent')

    with pytest.raises(ValueError,
                       match=r"Input should be.*input_value='most_frequent'"):
        MinLogNumerical(na_strategy='most_frequent')

    with pytest.raises(ValueError,
                       match=r"Input should be.*input_value='mean'"):
        NumericalList(na_strategy='mean')

    with pytest.raises(ValueError,
                       match=r"Input should be.*input_value='mean'"):
        Datetime(na_strategy='mean')


def test_null():
    cfg = Null()
    encoders = Encoders(encoders=[cfg])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))
    assert asdict(encoders)['encoders'][0] == {
        'name': 'Null',
        '_target_': 'kumoapi.encoder.Null',
        '_stats': {},
    }


@withTestDataFrame(dataset='FINANCIAL', filename='LOAN.csv')
def test_index(df):
    # Test computing stats from series with categories
    cfg = Index(na_strategy='separate')
    assert cfg.na_strategy is NAStrategy.SEPARATE
    compute_missing_stats_(cfg, df['STATUS'])
    assert cfg._stats[ColStatType.CATEGORY_COUNTS] == [
        ['C', 403],
        ['A', 203],
        ['D', 45],
        ['B', 31],
    ]
    encoders = Encoders(encoders=[cfg])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))

    # Test computing stats from series with NA values
    cfg = Index(na_strategy='separate')
    assert cfg.na_strategy is NAStrategy.SEPARATE
    compute_missing_stats_(
        cfg,
        pd.Series([True, False, True, True, pd.NA], dtype=pd.BooleanDtype()),
    )
    assert cfg._stats[ColStatType.CATEGORY_COUNTS] == [
        [True, 3],
        [False, 1],
    ]
    encoders = Encoders(encoders=[cfg])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))

    # Test computing stats from series with no categories
    cfg = Index(na_strategy='separate')
    assert cfg.na_strategy is NAStrategy.SEPARATE
    compute_missing_stats_(cfg, pd.Series(['', '', '', '']))
    assert cfg._stats[ColStatType.CATEGORY_COUNTS] == []
    encoders = Encoders(encoders=[cfg])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))


@withTestDataFrame(dataset='FINANCIAL', filename='LOAN.csv')
def test_hash(df):
    # Test computing stats from series with categories
    cfg = Hash(num_components=2, na_strategy='separate')
    assert cfg.na_strategy is NAStrategy.SEPARATE
    compute_missing_stats_(cfg, df['STATUS'])
    assert cfg._stats[ColStatType.CATEGORY_COUNTS] == [
        ['C', 403],
        ['A', 203],
        ['D', 45],
        ['B', 31],
    ]
    encoders = Encoders(encoders=[cfg])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))
    assert asdict(encoders)['encoders'][0] == {
        'name': 'Hash',
        '_target_': 'kumoapi.encoder.Hash',
        'na_strategy': NAStrategy.SEPARATE,
        '_stats': {
            ColStatType.CATEGORY_COUNTS: [
                ['C', 403],
                ['A', 203],
                ['D', 45],
                ['B', 31],
            ],
        },
        'num_components': 2,
    }

    # Test computing stats from series with NA values
    cfg = Hash(num_components=2, na_strategy='separate')
    assert cfg.na_strategy is NAStrategy.SEPARATE
    compute_missing_stats_(
        cfg,
        pd.Series([True, False, True, True, pd.NA], dtype=pd.BooleanDtype()),
    )
    assert cfg._stats[ColStatType.CATEGORY_COUNTS] == [
        [True, 3],
        [False, 1],
    ]
    encoders = Encoders(encoders=[cfg])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))
    assert asdict(encoders)['encoders'][0] == {
        'name': 'Hash',
        '_target_': 'kumoapi.encoder.Hash',
        'na_strategy': NAStrategy.SEPARATE,
        '_stats': {
            ColStatType.CATEGORY_COUNTS: [
                [True, 3],
                [False, 1],
            ],
        },
        'num_components': 2,
    }

    # Test computing stats from series with no categories
    cfg = Hash(num_components=2, na_strategy='separate')
    assert cfg.na_strategy is NAStrategy.SEPARATE
    compute_missing_stats_(cfg, pd.Series(['', '', '', '']))
    assert cfg._stats[ColStatType.CATEGORY_COUNTS] == []
    encoders = Encoders(encoders=[cfg])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))
    assert asdict(encoders)['encoders'][0] == {
        'name': 'Hash',
        '_target_': 'kumoapi.encoder.Hash',
        'na_strategy': NAStrategy.SEPARATE,
        '_stats': {
            ColStatType.CATEGORY_COUNTS: [],
        },
        'num_components': 2,
    }


def test_multicategorical():
    # Test computing stats from series with categories
    cfg = MultiCategorical(na_strategy='zero')
    assert cfg.na_strategy is NAStrategy.ZERO
    compute_missing_stats_(
        cfg,
        pd.Series([
            pd.NA,
            '',
            'A|B|C|D|E',
            'A|C|E',
            'D|F',
            'D|G|F',
            'D',
        ]),
    )
    assert cfg._stats[ColStatType.MULTI_CATEGORY_COUNTS] == [
        ['D', 4],
        ['A', 2],
        ['C', 2],
        ['E', 2],
        ['F', 2],
        ['B', 1],
        ['G', 1],
    ]
    encoders = Encoders(encoders=[cfg])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))
    assert asdict(encoders)['encoders'][0] == {
        'name': 'MultiCategorical',
        '_target_': 'kumoapi.encoder.MultiCategorical',
        'na_strategy': NAStrategy.ZERO,
        '_stats': {
            ColStatType.MULTI_CATEGORIES_SEPARATOR:
            '|',
            ColStatType.MULTI_CATEGORY_COUNTS: [
                ['D', 4],
                ['A', 2],
                ['C', 2],
                ['E', 2],
                ['F', 2],
                ['B', 1],
                ['G', 1],
            ],
        },
        'min_occ': 1,
    }

    # Test computing stats from series with no categories
    for sep in [None, '|']:
        cfg = MultiCategorical(na_strategy='zero')
        cfg._stats[ColStatType.MULTI_CATEGORIES_SEPARATOR] = sep
        assert cfg.na_strategy is NAStrategy.ZERO
        compute_missing_stats_(cfg, pd.Series(['', '', '', '']))
        assert cfg._stats[ColStatType.MULTI_CATEGORY_COUNTS] == []
        encoders = Encoders(encoders=[cfg])
        assert encoders == TypeAdapter(Encoders).validate_python(
            asdict(encoders))
        assert asdict(encoders)['encoders'][0] == {
            'name': 'MultiCategorical',
            '_target_': 'kumoapi.encoder.MultiCategorical',
            'na_strategy': NAStrategy.ZERO,
            '_stats': {
                ColStatType.MULTI_CATEGORIES_SEPARATOR: sep,
                ColStatType.MULTI_CATEGORY_COUNTS: [],
            },
            'min_occ': 1,
        }

    # Test computing stats from series with single category in each row
    cfg = MultiCategorical(na_strategy='zero')
    assert cfg.na_strategy is NAStrategy.ZERO
    compute_missing_stats_(cfg, pd.Series([pd.NA, 'A', 'A', 'B', 'B', 'B']))
    encoders = Encoders(encoders=[cfg])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))
    assert asdict(encoders)['encoders'][0] == {
        'name': 'MultiCategorical',
        '_target_': 'kumoapi.encoder.MultiCategorical',
        'min_occ': 1,
        'na_strategy': NAStrategy.ZERO,
        '_stats': {
            ColStatType.MULTI_CATEGORIES_SEPARATOR: None,
            ColStatType.MULTI_CATEGORY_COUNTS: [
                ['B', 3],
                ['A', 2],
            ],
        },
    }

    # Test computing stats from series with single category or empty string
    cfg = MultiCategorical(na_strategy='zero')
    assert cfg.na_strategy is NAStrategy.ZERO
    compute_missing_stats_(cfg, pd.Series([
        '',
        'A',
        'A',
        '',
        '',
        'A',
        'A',
    ]))
    encoders = Encoders(encoders=[cfg])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))
    assert asdict(encoders)['encoders'][0] == {
        'name': 'MultiCategorical',
        '_target_': 'kumoapi.encoder.MultiCategorical',
        'min_occ': 1,
        'na_strategy': NAStrategy.ZERO,
        '_stats': {
            ColStatType.MULTI_CATEGORIES_SEPARATOR: None,
            ColStatType.MULTI_CATEGORY_COUNTS: [
                ['A', 4],
            ],
        },
    }

    # Test computing stats from series for separate n/a strategy
    cfg = MultiCategorical(na_strategy='separate')
    assert cfg.na_strategy is NAStrategy.SEPARATE
    compute_missing_stats_(
        cfg, pd.Series([
            pd.NA,
            '',
            'A|B|C|D|E',
            'A|C|E',
            'D|F',
            'D|G|F',
            'D',
        ]))
    encoders = Encoders(encoders=[cfg])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))
    assert asdict(encoders)['encoders'][0] == {
        'name': 'MultiCategorical',
        '_target_': 'kumoapi.encoder.MultiCategorical',
        'min_occ': 1,
        'na_strategy': NAStrategy.SEPARATE,
        '_stats': {
            ColStatType.MULTI_CATEGORIES_SEPARATOR:
            '|',
            ColStatType.MULTI_CATEGORY_COUNTS: [
                ['D', 4],
                ['A', 2],
                ['C', 2],
                ['E', 2],
                ['F', 2],
                ['B', 1],
                ['G', 1],
            ],
        },
    }

    # Test computing stats from series with repeated categories
    cfg = MultiCategorical(na_strategy='separate')
    assert cfg.na_strategy is NAStrategy.SEPARATE
    # Series where categories are repeated
    compute_missing_stats_(
        cfg,
        pd.Series([
            pd.NA,
            '',
            'A|B|C|D|E',
            'A|A|C|E',
            'D|F|D',
            'D|G|G|F',
            'D',
        ]),
    )
    encoders = Encoders(encoders=[cfg])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))
    assert asdict(encoders)['encoders'][0] == {
        'name': 'MultiCategorical',
        '_target_': 'kumoapi.encoder.MultiCategorical',
        'min_occ': 1,
        'na_strategy': NAStrategy.SEPARATE,
        '_stats': {
            ColStatType.MULTI_CATEGORIES_SEPARATOR:
            '|',
            ColStatType.MULTI_CATEGORY_COUNTS: [
                # Counts of D, A and G have increase since they are repeated.
                ['D', 5],
                ['A', 3],
                ['C', 2],
                ['E', 2],
                ['F', 2],
                ['G', 2],
                ['B', 1],
            ],
        },
    }

    # Test computing stats from series for most_frequent n/a strategy
    cfg = MultiCategorical(na_strategy='most_frequent')
    assert cfg.na_strategy is NAStrategy.MOST_FREQUENT
    compute_missing_stats_(
        cfg, pd.Series([
            pd.NA,
            '',
            'A|B|C|D|E',
            'A|C|E',
            'D|F',
            'D|G|F',
            'D',
        ]))
    encoders = Encoders(encoders=[cfg])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))
    assert asdict(encoders)['encoders'][0] == {
        'name': 'MultiCategorical',
        '_target_': 'kumoapi.encoder.MultiCategorical',
        'min_occ': 1,
        'na_strategy': NAStrategy.MOST_FREQUENT,
        '_stats': {
            ColStatType.MULTI_CATEGORIES_SEPARATOR:
            '|',
            ColStatType.MULTI_CATEGORY_COUNTS: [
                ['D', 4],
                ['A', 2],
                ['C', 2],
                ['E', 2],
                ['F', 2],
                ['B', 1],
                ['G', 1],
            ],
        },
    }


def test_text_embedding():
    cfg = TextEmbedding()
    encoders = Encoders(encoders=[cfg])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))
    assert asdict(encoders)['encoders'][0] == {
        'name': 'TextEmbedding',
        '_target_': 'kumoapi.encoder.TextEmbedding',
        'na_strategy': NAStrategy.ZERO,
        '_stats': {},
        'model': 'glove.6B-50d',
        '_identity': None,
    }


def test_numerical():
    cfg = Numerical(na_strategy='mean', scaler='minmax')
    assert cfg.na_strategy is NAStrategy.MEAN
    assert cfg.scaler is Scaler.MINMAX
    compute_missing_stats_(cfg, pd.Series([1, 2, 3, 4, 5]))
    encoders = Encoders(encoders=[cfg])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))
    assert asdict(encoders)['encoders'][0] == {
        'name': 'Numerical',
        '_target_': 'kumoapi.encoder.Numerical',
        'na_strategy': NAStrategy.MEAN,
        '_stats': {
            ColStatType.MEAN: 3.,  # required by NAStrategy.MEAN
            ColStatType.MIN: 1,  # required by Scaler.MINMAX
            ColStatType.MAX: 5,  # required by Scaler.MINMAX
        },
        'scaler': Scaler.MINMAX,
    }


def test_max_log_numerical():
    cfg = MaxLogNumerical()
    assert cfg.na_strategy is NAStrategy.MEAN
    compute_missing_stats_(cfg, pd.Series([1, 2, 3, 4, 5]))
    encoders = Encoders(encoders=[cfg])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))
    assert asdict(encoders)['encoders'][0] == {
        'name': 'MaxLogNumerical',
        '_target_': 'kumoapi.encoder.MaxLogNumerical',
        'na_strategy': NAStrategy.MEAN,
        '_stats': {
            ColStatType.MEAN: 3.0,  # required by NAStrategy.MEAN
            ColStatType.MIN: 1.0,
        },
    }


def test_min_log_numerical():
    cfg = MinLogNumerical()
    assert cfg.na_strategy is NAStrategy.MEAN
    compute_missing_stats_(cfg, pd.Series([1, 2, 3, 4, 5]))
    encoders = Encoders(encoders=[cfg])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))
    assert asdict(encoders)['encoders'][0] == {
        'name': 'MinLogNumerical',
        '_target_': 'kumoapi.encoder.MinLogNumerical',
        'na_strategy': NAStrategy.MEAN,
        '_stats': {
            ColStatType.MEAN: 3.0,  # required by NAStrategy.MEAN
            ColStatType.MAX: 5.0,
        },
    }


def test_numerical_list():
    cfg = NumericalList(na_strategy='zero')
    cfg._stats[ColStatType.SEQUENCE_MAX_LENGTH] = 3
    cfg._stats[ColStatType.SEQUENCE_MIN_LENGTH] = 3
    assert cfg.na_strategy is NAStrategy.ZERO
    encoders = Encoders(encoders=[cfg])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))
    assert asdict(encoders)['encoders'][0] == {
        'name': 'NumericalList',
        '_target_': 'kumoapi.encoder.NumericalList',
        'scaler': None,
        'na_strategy': NAStrategy.ZERO,
        '_stats': {
            ColStatType.SEQUENCE_MAX_LENGTH: 3,
            ColStatType.SEQUENCE_MIN_LENGTH: 3,
        },
    }


@withTestDataFrame(dataset='FINANCIAL', filename='LOAN.csv')
def test_datetime(df):
    cfg = Datetime(include_minute=False, include_hour=False)
    assert cfg.na_strategy is NAStrategy.ZERO
    encoders = Encoders(encoders=[cfg])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))
    assert asdict(encoders)['encoders'][0] == {
        'name': 'Datetime',
        '_target_': 'kumoapi.encoder.Datetime',
        'na_strategy': NAStrategy.ZERO,
        '_stats': {},
        'include_minute': False,
        'include_hour': False,
        'include_day_of_week': True,
        'include_day_of_month': True,
        'include_day_of_year': True,
        'include_year': True,
        'num_year_periods': None,
    }

    compute_missing_stats_(
        cfg,
        pd.to_datetime(df['DATE'], format='%Y%m%d'),  # type: ignore
    )
    encoders = Encoders(encoders=[cfg])
    assert encoders == TypeAdapter(Encoders).validate_python(asdict(encoders))
    assert asdict(encoders)['encoders'][0] == {
        'name': 'Datetime',
        '_target_': 'kumoapi.encoder.Datetime',
        'na_strategy': NAStrategy.ZERO,
        '_stats': {
            ColStatType.MAX: 'kumo_timestamp_1998-12-08 00:00:00',
            ColStatType.MIN: 'kumo_timestamp_1993-07-05 00:00:00',
        },
        'include_minute': False,
        'include_hour': False,
        'include_day_of_week': True,
        'include_day_of_month': True,
        'include_day_of_year': True,
        'include_year': True,
        'num_year_periods': None,
    }
