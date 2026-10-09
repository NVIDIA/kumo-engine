# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
import pandas as pd
from kumoapi.encoder import (
    Datetime,
    Encoder,
    Hash,
    Image,
    Index,
    MaxLogNumerical,
    MinLogNumerical,
    MultiCategorical,
    Null,
    Numerical,
    NumericalList,
    TextEmbedding,
)
from kumoapi.typing import ColStatType, Stype

import kumo_core.features.stats
from kumo_core.encoders.tensor_mapper import SequenceTensorMapper
from kumo_core.features.stats import (
    CategoricalStats,
    MultiCategoricalStats,
    NumericalStats,
    SequenceStats,
    Stats,
    TextStats,
    get_stats,
)
from kumo_core.types.features import TIMESTAMP_PREFIX
from kumo_core.features.time import to_datetime


def infer_encoder(
    stats_or_ser: Stats | pd.Series,
    stype: Stype,
    fill_stats: bool = False,
    use_absolute_time: bool = True,
) -> Encoder:
    r"""Infers an encoder based on column statistics."""
    if isinstance(stats_or_ser, pd.Series):
        approx = stype not in {Stype.categorical, Stype.ID}
        if pd.api.types.is_timedelta64_dtype(stats_or_ser):
            stats_or_ser = stats_or_ser.dt.total_seconds()
        stats = get_stats(stats_or_ser, stype, approx)
    else:
        assert isinstance(stats_or_ser, Stats)
        stats = stats_or_ser

    if stats.na_frac > 1.0 - 1e-6:
        return Null()

    if stype == Stype.numerical:
        assert isinstance(stats, NumericalStats)
        return infer_numerical_encoder(stats, fill_stats)

    if stype == Stype.categorical:
        assert isinstance(stats, CategoricalStats)
        return infer_categorical_encoder(stats)

    if stype == Stype.ID:
        assert isinstance(stats, CategoricalStats)
        return infer_id_encoder(stats)

    if stype == Stype.multicategorical:
        assert isinstance(stats, MultiCategoricalStats)
        return infer_multicategorical_encoder(stats)

    if stype == Stype.text:
        assert isinstance(stats, TextStats)
        return infer_text_encoder(stats)

    if stype == Stype.timestamp:
        return Datetime(include_year=use_absolute_time)

    if stype == Stype.sequence:
        assert isinstance(stats, SequenceStats)
        return infer_sequence_encoder(stats)

    if stype == Stype.image:
        return Image()

    return Null()


def infer_numerical_encoder(
    stats: NumericalStats,
    fill_stats: bool = False,
) -> Encoder:

    if (stats.max - stats.min) > 10:
        # Calculate Pearson’s median skewness:
        skewness = 3 * (stats.mean - stats.median) / stats.std
        if skewness <= -1:
            encoder = MinLogNumerical()
            if fill_stats:
                encoder._stats[ColStatType.MEAN] = stats.mean
                encoder._stats[ColStatType.MAX] = stats.max
            return encoder
        if skewness >= 1:
            encoder = MaxLogNumerical()
            if fill_stats:
                encoder._stats[ColStatType.MEAN] = stats.mean
                encoder._stats[ColStatType.MIN] = stats.min
            return encoder

    # Large kurtosis means that the target column has long tail or high
    # outliers. Using 'robust' scaler can be more robust to those outliers.
    if stats.kurtosis > 100 and stats.quantiles[2] - stats.quantiles[0] != 0:
        encoder = Numerical(scaler='robust')
    elif abs(stats.max) > 10 or abs(stats.min) > 10:
        encoder = Numerical(scaler='standard')
    else:
        encoder = Numerical()

    if fill_stats:
        encoder._stats[ColStatType.MEAN] = stats.mean
        encoder._stats[ColStatType.STD] = stats.std
        encoder._stats[ColStatType.MIN] = stats.min
        encoder._stats[ColStatType.MAX] = stats.max
        encoder._stats[ColStatType.QUANTILES] = stats.quantiles

    return encoder


def infer_categorical_encoder(stats: CategoricalStats) -> Encoder:
    if stats.approx:
        raise ValueError("Inferring an encoder for a categorical column "
                         "requires exact statistics")

    # We infer the maximum allowed unique number of categories using
    # the heuristic defined below. The larger the number of non-N/A rows, the
    # more unique categories are allowed in a sublinear rate.
    unique_frac_threshold = 0.3 * (1 / math.sqrt(stats.count))
    unique_threshold = int(stats.count * unique_frac_threshold)

    # The minimum occurrence for a category is inferred to be the square
    # root of the number of unique categories.
    min_occ = max(int(math.sqrt(stats.num_unique)), 2)
    num_unique_pass_min_occ = len(
        [name for name, occ in stats.category_count if occ > min_occ])

    if stats.count < 100:  # Always use `Index` encoder for small tables.
        encoder = Index(min_occ=min_occ)
        encoder._stats[ColStatType.CATEGORY_COUNTS] = stats.category_count
        return encoder

    if num_unique_pass_min_occ <= 4000:
        if num_unique_pass_min_occ <= unique_threshold:
            encoder = Index(min_occ=min_occ)
            encoder._stats[ColStatType.CATEGORY_COUNTS] = stats.category_count
            return encoder
        elif num_unique_pass_min_occ <= 20 * unique_threshold:
            encoder = Hash(num_components=unique_threshold)
            encoder._stats[ColStatType.CATEGORY_COUNTS] = stats.category_count
            return encoder

    return Null()


def infer_multicategorical_encoder(stats: MultiCategoricalStats) -> Encoder:
    if stats.num_unique <= 500:
        encoder = MultiCategorical()
        encoder._stats[ColStatType.MULTI_CATEGORIES_SEPARATOR] = stats.sep
        return encoder

    return Null()


def infer_id_encoder(stats: CategoricalStats) -> Encoder:
    if stats.approx:
        raise ValueError("Inferring an encoder for an ID column requires "
                         "exact statistics")

    if stats.num_unique <= 4000:
        return infer_categorical_encoder(stats)

    # If the number of unique values is large but still smaller than a
    # threshold (80 % entries), we use a `Hash` encoder:
    if (stats.num_unique <= 0.8 * stats.count):
        # Determine the number of components for the hash encoder based
        # on number of unique categorical values. The threshold is
        # determined based on primary number within range:
        # (4k~10k], (10k~50k], (50k, 200k], (200k~1m], (1m, ...)
        if stats.num_unique <= 10_000:
            num_components = 3_989
        elif stats.num_unique <= 50_000:
            num_components = 4_999
        elif stats.num_unique <= 200_000:
            num_components = 19_997
        elif stats.num_unique <= 1_000_000:
            num_components = 99_991
        else:
            num_components = 199_999

        encoder = Hash(num_components=num_components)
        encoder._stats[ColStatType.CATEGORY_COUNTS] = stats.category_count
        return encoder

    return Null()


def infer_text_encoder(stats: TextStats) -> Encoder:
    if (stats.glove_overlap >= 0.5  # Ensure GloVe overlap
            and stats.avg_non_char <= 0.3  # Safeguard against non-text
            and stats.avg_tokens > 3  # Ensure meaningful deviation
            and stats.max_tokens > 2 * stats.avg_tokens):
        return TextEmbedding()

    return Null()


def infer_sequence_encoder(stats: SequenceStats) -> Encoder:
    if stats.min_length > 0 and stats.min_length == stats.max_length:
        return NumericalList()

    return Null()


def compute_missing_stats_(
    encoder: Encoder,
    ser: pd.Series,
    force_refresh: bool = False,
) -> Encoder:

    if force_refresh:
        encoder._stats = {}

    missing_stats = encoder.required_stats - set(encoder._stats.keys())
    if len(missing_stats) == 0:
        return encoder

    if isinstance(encoder, Datetime):
        ser = to_datetime(ser)

    notna_mask = kumo_core.features.stats.notna(ser)
    ser = ser[notna_mask]

    if pd.api.types.is_timedelta64_dtype(ser):
        ser = ser.dt.total_seconds()

    # Ensure that we first infer the separator:
    if ColStatType.MULTI_CATEGORIES_SEPARATOR in missing_stats:
        sep = kumo_core.features.stats.sep_multi(ser)
        encoder._stats[ColStatType.MULTI_CATEGORIES_SEPARATOR] = sep

    for stat in encoder.required_stats - set(encoder._stats.keys()):
        if stat == ColStatType.CATEGORY_COUNTS:
            encoder._stats[stat] = kumo_core.features.stats.category_count(ser)
        elif stat == ColStatType.MULTI_CATEGORY_COUNTS:
            sep = encoder._stats[ColStatType.MULTI_CATEGORIES_SEPARATOR]
            _ser = ser.astype(str).str.split(sep) if sep is not None else ser
            _ser = _ser.explode()
            encoder._stats[stat] = kumo_core.features.stats.category_count(
                _ser)
        elif stat == ColStatType.MEAN:
            encoder._stats[stat] = kumo_core.features.stats.mean(ser)
        elif stat == ColStatType.STD:
            encoder._stats[stat] = kumo_core.features.stats.std(ser)
        elif stat == ColStatType.MIN:
            if isinstance(encoder, Datetime):
                value = np.min(ser) if len(ser) > 0 else pd.Timestamp('1970')
                encoder._stats[stat] = f'{TIMESTAMP_PREFIX}{value}'
            else:
                encoder._stats[stat] = kumo_core.features.stats.min(ser)
        elif stat == ColStatType.MAX:
            if isinstance(encoder, Datetime):
                value = np.max(ser) if len(ser) > 0 else pd.Timestamp('1970')
                encoder._stats[stat] = f'{TIMESTAMP_PREFIX}{value}'
            else:
                encoder._stats[stat] = kumo_core.features.stats.max(ser)
        elif stat == ColStatType.QUANTILES:
            encoder._stats[stat] = kumo_core.features.stats.quantiles(ser)
        elif stat == ColStatType.SEQUENCE_MIN_LENGTH:
            ser = SequenceTensorMapper.to_list_like(ser)
            encoder._stats[
                stat] = kumo_core.features.stats.sequence_min_length(ser)
        elif stat == ColStatType.SEQUENCE_MAX_LENGTH:
            ser = SequenceTensorMapper.to_list_like(ser)
            encoder._stats[
                stat] = kumo_core.features.stats.sequence_max_length(ser)
        elif stat == ColStatType.SEQUENCE_MEAN:
            ser = SequenceTensorMapper.to_list_like(ser)
            encoder._stats[stat] = kumo_core.features.stats.sequence_mean(ser)
        elif stat == ColStatType.SEQUENCE_STD:
            ser = SequenceTensorMapper.to_list_like(ser)
            encoder._stats[stat] = kumo_core.features.stats.sequence_std(ser)
        else:
            raise ValueError(f"Cannot compute stat '{stat.value}'")

    return encoder
