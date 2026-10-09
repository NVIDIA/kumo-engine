# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from kumoapi.encoder import (
    Hash,
    Index,
    MaxLogNumerical,
    MinLogNumerical,
    MultiCategorical,
    Null,
    Numerical,
    NumericalList,
    Scaler,
    TextEmbedding,
)
from kumoapi.typing import ColStatType, Stype

from kumo_core.encoders.selection import infer_encoder
from kumo_core.features.stats import (
    CategoricalStats,
    MultiCategoricalStats,
    NumericalStats,
    SequenceStats,
    TextStats,
)


def test_infer_numerical_encoder() -> None:
    def get_default_stats() -> NumericalStats:
        return NumericalStats(
            approx=False,
            count=100,
            na_frac=0.0,
            mean=0.0,
            std=1.0,
            min=0.0,
            max=0.0,
            kurtosis=0.0,
            quantiles=(0.0, 0.0, 0.0),
        )

    stats = get_default_stats()
    stats.na_frac = 1.0
    encoder = infer_encoder(stats, Stype.numerical)
    assert isinstance(encoder, Null)

    stats = get_default_stats()
    encoder = infer_encoder(stats, Stype.numerical)
    assert isinstance(encoder, Numerical)
    assert encoder.scaler is None

    stats = get_default_stats()
    stats.max = 1_000_000
    stats.mean = 800_000
    stats.quantiles = (0, 100_000, 200_000)
    encoder = infer_encoder(stats, Stype.numerical)
    assert isinstance(encoder, MaxLogNumerical)

    stats = get_default_stats()
    stats.max = 1_000_000
    stats.mean = 100_000
    stats.quantiles = (800_000, 900_000, 1_000_000)
    encoder = infer_encoder(stats, Stype.numerical)
    assert isinstance(encoder, MinLogNumerical)

    stats = get_default_stats()
    stats.kurtosis = 2_000
    stats.quantiles = (0, 1_000, 10_000)
    encoder = infer_encoder(stats, Stype.numerical)
    assert isinstance(encoder, Numerical)
    assert encoder.scaler == Scaler.ROBUST

    stats = get_default_stats()
    stats.max = 100
    encoder = infer_encoder(stats, Stype.numerical)
    assert isinstance(encoder, Numerical)
    assert encoder.scaler == Scaler.STANDARD


def test_infer_categorical_encoder() -> None:
    def get_default_stats() -> CategoricalStats:
        return CategoricalStats(
            approx=False,
            count=100,
            na_frac=0.0,
            category_count=[],
        )

    stats = get_default_stats()
    stats.na_frac = 1.0
    encoder = infer_encoder(stats, Stype.categorical)
    assert isinstance(encoder, Null)

    stats = get_default_stats()
    stats.category_count = [(i, 50) for i in range(2)]
    encoder = infer_encoder(stats, Stype.categorical)
    assert isinstance(encoder, Index)
    assert encoder.min_occ == 2

    stats = get_default_stats()
    stats.category_count = [(i, 10) for i in range(10)]
    # TODO This is a very awkward encoder proposal:
    encoder = infer_encoder(stats, Stype.categorical)
    assert isinstance(encoder, Hash)
    assert encoder.num_components == 3


def test_infer_multicategorical_encoder() -> None:
    def get_default_stats() -> MultiCategoricalStats:
        return MultiCategoricalStats(
            approx=False,
            count=100,
            na_frac=0.0,
            sep='|',
            num_unique=10,
        )

    stats = get_default_stats()
    stats.na_frac = 1.0
    encoder = infer_encoder(stats, Stype.multicategorical)
    assert isinstance(encoder, Null)

    stats = get_default_stats()
    encoder = infer_encoder(stats, Stype.multicategorical)
    assert isinstance(encoder, MultiCategorical)
    assert encoder._stats[ColStatType.MULTI_CATEGORIES_SEPARATOR] == '|'

    stats = get_default_stats()
    stats.num_unique = 501
    encoder = infer_encoder(stats, Stype.multicategorical)
    assert isinstance(encoder, Null)


def test_infer_id_encoder() -> None:
    def get_default_stats() -> CategoricalStats:
        return CategoricalStats(
            approx=False,
            count=100,
            na_frac=0.0,
            category_count=[],
        )

    stats = get_default_stats()
    stats.na_frac = 1.0
    encoder = infer_encoder(stats, Stype.ID)
    assert isinstance(encoder, Null)

    stats = get_default_stats()
    stats.category_count = [(i, 50) for i in range(8_000)]
    stats.count = 10_000
    encoder = infer_encoder(stats, Stype.ID)
    assert isinstance(encoder, Hash)
    assert encoder.num_components == 3989

    stats = get_default_stats()
    stats.category_count = [(i, 50) for i in range(8_001)]
    stats.count = 10_000
    encoder = infer_encoder(stats, Stype.ID)
    assert isinstance(encoder, Null)


def test_infer_text_encoder() -> None:
    def get_default_stats() -> TextStats:
        return TextStats(
            approx=False,
            count=100,
            na_frac=0.0,
            avg_tokens=0,
            max_tokens=0,
            avg_non_char=0,
            glove_overlap=1,
        )

    stats = get_default_stats()
    stats.na_frac = 1.0
    encoder = infer_encoder(stats, Stype.text)
    assert isinstance(encoder, Null)

    stats = get_default_stats()
    stats.avg_tokens = 4
    stats.max_tokens = 9
    stats.avg_non_char = 0.3
    stats.glove_overlap = 0.5
    encoder = infer_encoder(stats, Stype.text)
    assert encoder == TextEmbedding()

    stats = get_default_stats()
    stats.avg_tokens = 3
    stats.max_tokens = 9
    stats.avg_non_char = 0.3
    stats.glove_overlap = 0.5
    encoder = infer_encoder(stats, Stype.text)
    assert isinstance(encoder, Null)

    stats = get_default_stats()
    stats.avg_tokens = 4
    stats.max_tokens = 8
    stats.avg_non_char = 0.3
    stats.glove_overlap = 0.5
    encoder = infer_encoder(stats, Stype.text)
    assert isinstance(encoder, Null)

    stats = get_default_stats()
    stats.avg_tokens = 4
    stats.max_tokens = 9
    stats.avg_non_char = 0.31
    stats.glove_overlap = 0.5
    encoder = infer_encoder(stats, Stype.text)
    assert isinstance(encoder, Null)

    stats = get_default_stats()
    stats.avg_tokens = 4
    stats.max_tokens = 9
    stats.avg_non_char = 0.3
    stats.glove_overlap = 0.49
    encoder = infer_encoder(stats, Stype.text)
    assert isinstance(encoder, Null)


def test_infer_sequence_encoder() -> None:
    def get_default_stats() -> SequenceStats:
        return SequenceStats(
            approx=False,
            count=100,
            na_frac=0.0,
            min_length=0,
            max_length=0,
            mean=0,
            std=1,
        )

    stats = get_default_stats()
    stats.na_frac = 1.0
    encoder = infer_encoder(stats, Stype.sequence)
    assert isinstance(encoder, Null)

    stats = get_default_stats()
    stats.min_length = 10
    stats.max_length = 10
    encoder = infer_encoder(stats, Stype.sequence)
    assert isinstance(encoder, NumericalList)

    stats = get_default_stats()
    stats.min_length = 9
    stats.max_length = 10
    encoder = infer_encoder(stats, Stype.sequence)
    assert isinstance(encoder, Null)

    stats = get_default_stats()
    stats.min_length = 0
    stats.max_length = 0
    encoder = infer_encoder(stats, Stype.sequence)
    assert isinstance(encoder, Null)
