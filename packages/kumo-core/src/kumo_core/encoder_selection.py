# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure feature eligibility and encoder-selection decisions."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from kumoapi.encoder import (
    Datetime,
    Encoder,
    EncoderType,
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
from kumoapi.typing import ColStatType, Dtype, Stype

from . import stype_inference

logger = logging.getLogger(__name__)

NA_THRESHOLD = 0.6
SMALL_NUM_ROWS_THRESHOLD = 100
CATEGORY_UNIQUE_THRESHOLD_MULTIPLIER = 0.3
MAX_ONE_HOT_CATEGORIES = 10
MAX_VALUE_TO_STANDARDIZE = 10.0
UNIQUE_ID_THRESHOLD = 0.8
KURTOSIS_THRESHOLD = 100.0
GLOVE_COVERAGE_THRESHOLD = 0.5
NON_CHAR_THRESHOLD = 0.3
AVG_TOKENS_THRESHOLD = 3

MAX_AUTO_CATEGORICAL_CONVERSION_THRESHOLD = (
    stype_inference.MAX_AUTO_CATEGORICAL_CONVERSION_THRESHOLD)
MAX_CATEGORIES = stype_inference.MAX_CATEGORIES
MAX_MULTI_LABEL_CATEGORIES = stype_inference.MAX_MULTI_LABEL_CATEGORIES
UNIQUE_INT_THRESHOLD = stype_inference.UNIQUE_INT_THRESHOLD

REQUIRED_EXACT_STATS = (
    ColStatType.COUNT,
    ColStatType.NUM_NA,
    ColStatType.NUM_UNIQUE,
    ColStatType.CATEGORY_COUNTS,
)
REQUIRED_APPROX_STATS = (
    ColStatType.NA_FRACTION,
    ColStatType.UNIQUE_FRACTION,
    ColStatType.NUM_UNIQUE_MULTI,
    ColStatType.MULTI_CATEGORIES_SEPARATOR,
    ColStatType.QUANTILES,
    ColStatType.KURTOSIS,
    ColStatType.MEAN,
    ColStatType.MEDIAN,
    ColStatType.MIN,
    ColStatType.MAX,
    ColStatType.STD,
    ColStatType.STRING_AVG_TOKENS,
    ColStatType.STRING_MAX_TOKENS,
    ColStatType.STRING_GLOVE_OVERLAP,
    ColStatType.STRING_AVG_NON_CHAR,
    ColStatType.SEQUENCE_MAX_LENGTH,
    ColStatType.SEQUENCE_MIN_LENGTH,
)
REQUIRED_STATS = REQUIRED_EXACT_STATS + REQUIRED_APPROX_STATS


@dataclass(frozen=True)
class _EncoderStat:
    """The statistic facts that affect encoder selection."""

    value: Any
    approx: bool
    num_entries: int | None = None


_EncoderStats = Mapping[ColStatType, _EncoderStat]


def feature_column_names(
    column_names: Iterable[str],
    *,
    primary_key: str | None,
    foreign_keys: Iterable[str],
    end_time_column: str | None,
    excluded: Iterable[str] = (),
) -> list[str]:
    structural = {*foreign_keys, *excluded}
    structural.update(name for name in (primary_key, end_time_column) if name)
    return [name for name in column_names if name not in structural]


def supported_encoder_types(dtype: Dtype) -> list[type[Encoder]]:
    if dtype is None:
        return [Null]
    if dtype.is_bool():
        return [Index]
    if dtype.is_int():
        return [Index, Hash, Numerical, MinLogNumerical, MaxLogNumerical]
    if dtype.is_float():
        return [Numerical, MinLogNumerical, MaxLogNumerical, Index, Hash]
    if dtype.is_string():
        return [
            Index, Hash, MultiCategorical, TextEmbedding, NumericalList, Image
        ]
    if dtype.is_temporal():
        return [Datetime, Numerical]
    if dtype.is_list():
        if dtype == Dtype.floatlist:
            return [NumericalList]
        if dtype == Dtype.intlist:
            return [NumericalList, MultiCategorical]
        if dtype == Dtype.stringlist:
            return [MultiCategorical]
    return [Null]


def supports_encoder(dtype: Dtype, encoder: Encoder) -> bool:
    return type(encoder) in supported_encoder_types(dtype) + [Null, Index]


def _stat(
    stats: _EncoderStats,
    stat_type: ColStatType,
    *,
    exact: bool,
) -> _EncoderStat:
    stat = stats.get(stat_type)
    if stat is None:
        raise ValueError(
            f"Column Stat {stat_type} is required for encoder inferral but "
            "was not provided.")
    if exact and stat.approx:
        raise ValueError(
            f"The provided Column Stat {stat_type} must be exact for proper "
            "encoder inferral but was passed in as approx")
    return stat


def _is_finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def select_encoder(
    column_name: str,
    dtype: Dtype,
    stype: Stype,
    stats: _EncoderStats,
) -> EncoderType:
    na_fraction = _stat(stats, ColStatType.NA_FRACTION, exact=False).value
    if na_fraction is not None and na_fraction > 1.0 - 1e-6:
        return Null()
    if stype == Stype.numerical:
        return select_numerical_encoder(stats)
    if stype == Stype.categorical:
        return select_categorical_encoder(column_name, dtype, stats)
    if stype == Stype.multicategorical:
        return select_multicategorical_encoder(column_name, stats)
    if stype == Stype.ID:
        return select_id_encoder(column_name, dtype, stats)
    if stype == Stype.text:
        return select_text_encoder(column_name, stats)
    if stype == Stype.timestamp:
        return Datetime()
    if stype == Stype.sequence:
        return select_sequence_encoder(dtype, stats)
    if stype == Stype.image:
        return Image()
    raise NotImplementedError


def select_target_encoder(stype: Stype, stats: _EncoderStats) -> EncoderType:
    if stype == Stype.categorical:
        return Index()
    if stype == Stype.multicategorical:
        return MultiCategorical()
    if stype == Stype.numerical:
        return select_numerical_encoder(stats)
    raise ValueError(f"no TARGET encoder for stype {stype!r}")


def select_numerical_encoder(stats: _EncoderStats) -> EncoderType:
    values = [
        _stat(stats, stat_type, exact=False).value for stat_type in (
            ColStatType.MEAN,
            ColStatType.MEDIAN,
            ColStatType.STD,
            ColStatType.MIN,
            ColStatType.MAX,
            ColStatType.KURTOSIS,
            ColStatType.QUANTILES,
        )
    ]
    mean, median, std, minimum, maximum, kurtosis, quantiles = values
    if not all(_is_finite(value) for value in (mean, median, minimum, maximum)):
        return Numerical()
    if not _is_finite(std):
        std = 0
    if not _is_finite(kurtosis):
        kurtosis = 0
    if (not isinstance(quantiles, list) or len(quantiles) < 3
            or any(not _is_finite(value) for value in quantiles[:3])):
        quantiles = [0, 0, 0]

    skewness = 3 * (mean - median) / std if std else 0.0
    if maximum - minimum > 10:
        if skewness <= -1:
            return MinLogNumerical()
        if skewness >= 1:
            return MaxLogNumerical()
    if (kurtosis and kurtosis > KURTOSIS_THRESHOLD
            and quantiles[2] - quantiles[0] != 0):
        return Numerical(scaler="robust")
    if (abs(maximum) > MAX_VALUE_TO_STANDARDIZE
            or abs(minimum) > MAX_VALUE_TO_STANDARDIZE):
        return Numerical(scaler="standard")
    return Numerical()


def select_categorical_encoder(
    column_name: str,
    dtype: Dtype,
    stats: _EncoderStats,
) -> EncoderType:
    if dtype.is_bool():
        return Index()
    count = _stat(stats, ColStatType.COUNT, exact=True).value
    num_na = _stat(stats, ColStatType.NUM_NA, exact=True).value
    non_na = count - num_na
    if non_na == 0:
        logger.warning(
            "%r will not be encoded because it only contains N/A values",
            column_name,
        )
        return Null()

    unique_fraction_threshold = CATEGORY_UNIQUE_THRESHOLD_MULTIPLIER * (
        1 / math.sqrt(non_na))
    unique_threshold = int(non_na * unique_fraction_threshold)
    num_unique = _stat(stats, ColStatType.NUM_UNIQUE, exact=True).value
    min_occ = max(int(math.sqrt(num_unique)), 2)
    category_counts = _stat(stats, ColStatType.CATEGORY_COUNTS,
                            exact=True).value
    retained = sum(1 for _, occurrences in category_counts
                   if occurrences > min_occ)
    supported = supported_encoder_types(dtype)
    if count < SMALL_NUM_ROWS_THRESHOLD and Index in supported:
        return Index(min_occ=min_occ)
    if retained <= MAX_CATEGORIES:
        if retained <= unique_threshold and Index in supported:
            return Index(min_occ=min_occ)
        if retained <= 20 * unique_threshold and Hash in supported:
            return Hash(num_components=unique_threshold)
    logger.warning(
        "%r will not be encoded because it has more than %s unique "
        "categories (got %s) and %s NA rows in a table with %s rows.",
        column_name,
        unique_threshold,
        retained,
        num_na,
        count,
    )
    return Null()


def select_multicategorical_encoder(
    column_name: str,
    stats: _EncoderStats,
) -> EncoderType:
    num_unique = _stat(stats, ColStatType.NUM_UNIQUE_MULTI, exact=False).value
    if num_unique <= MAX_MULTI_LABEL_CATEGORIES:
        return MultiCategorical()
    logger.warning(
        "%r will not be encoded because it has more than %s unique "
        "multi-label categories (got %s)",
        column_name,
        MAX_MULTI_LABEL_CATEGORIES,
        num_unique,
    )
    return Null()


def select_id_encoder(
    column_name: str,
    dtype: Dtype,
    stats: _EncoderStats,
) -> EncoderType:
    num_unique = _stat(stats, ColStatType.NUM_UNIQUE, exact=False)
    if num_unique.value <= MAX_CATEGORIES:
        return select_categorical_encoder(column_name, dtype, stats)
    num_entries = num_unique.num_entries
    if num_entries is None:
        raise ValueError(
            f"Column Stat {ColStatType.NUM_UNIQUE} for {column_name!r} must "
            "carry num_entries for ID encoder inferral.")
    if (not is_mostly_unique_id(num_unique.value, num_entries)
            and Hash in supported_encoder_types(dtype)):
        if num_unique.value <= 4_000:
            components = num_unique.value
        elif num_unique.value <= 10_000:
            components = 3_989
        elif num_unique.value <= 50_000:
            components = 4_999
        elif num_unique.value <= 200_000:
            components = 19_997
        elif num_unique.value <= 1_000_000:
            components = 99_991
        else:
            components = 199_999
        return Hash(num_components=components)
    logger.warning(
        "%r will not be encoded because it has more than %s unique values",
        column_name,
        num_entries * UNIQUE_ID_THRESHOLD,
    )
    return Null()


def is_mostly_unique_id(num_unique: int, num_entries: int) -> bool:
    return (num_unique > MAX_CATEGORIES
            and num_unique > UNIQUE_ID_THRESHOLD * num_entries)


def select_text_encoder(
    column_name: str,
    stats: _EncoderStats,
) -> EncoderType:
    try:
        coverage, average, maximum, non_char = [
            _stat(stats, stat_type, exact=False).value for stat_type in (
                ColStatType.STRING_GLOVE_OVERLAP,
                ColStatType.STRING_AVG_TOKENS,
                ColStatType.STRING_MAX_TOKENS,
                ColStatType.STRING_AVG_NON_CHAR,
            )
        ]
    except ValueError as exc:
        logger.warning(
            "Unable to get text stats for column %s due to error %s. Using "
            "Null encoder.",
            column_name,
            exc,
        )
        return Null()
    if (coverage >= GLOVE_COVERAGE_THRESHOLD and non_char <= NON_CHAR_THRESHOLD
            and average > AVG_TOKENS_THRESHOLD and maximum > 2 * average):
        logger.debug("%r will be encoded with TextEmbedding", column_name)
        return TextEmbedding()
    logger.warning(
        "Text column %r will not be encoded by default: its values are too "
        "short or have too few known words. To encode it anyway, set its "
        "encoder to TextEmbedding().",
        column_name,
    )
    return Null()


def select_sequence_encoder(
    dtype: Dtype,
    stats: _EncoderStats,
) -> EncoderType:
    if dtype.is_list() or dtype.is_string():
        minimum = _stat(stats, ColStatType.SEQUENCE_MIN_LENGTH,
                        exact=False).value
        maximum = _stat(stats, ColStatType.SEQUENCE_MAX_LENGTH,
                        exact=False).value
        if minimum == maximum and minimum > 0:
            return NumericalList()
    return Null()
