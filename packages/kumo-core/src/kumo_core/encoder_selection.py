# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure feature eligibility and encoder-selection decisions."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from numbers import Real
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


def _stats_error(
    stats: _EncoderStats,
    stat_types: Iterable[ColStatType],
    *,
    exact: bool,
) -> str | None:
    for stat_type in stat_types:
        stat = stats.get(stat_type)
        if stat is None:
            return (f"Column Stat {stat_type} is required for encoder inferral "
                    "but was not provided.")
        if exact and stat.approx:
            return (f"The provided Column Stat {stat_type} must be exact for "
                    "proper encoder inferral but was passed in as approx")
    return None


def _required_stats(
    column_name: str,
    stats: _EncoderStats,
    stat_types: tuple[ColStatType, ...],
    *,
    exact: bool = False,
) -> list[_EncoderStat] | None:
    reason = _stats_error(stats, stat_types, exact=exact)
    if reason is not None:
        logger.warning("%r will not be encoded: %s Using Null encoder.",
                       column_name, reason)
        return None
    for stat_type in stat_types:
        value = stats[stat_type].value
        if stat_type in (
                ColStatType.COUNT,
                ColStatType.NUM_NA,
                ColStatType.NUM_UNIQUE,
                ColStatType.NUM_UNIQUE_MULTI,
        ) and (not isinstance(value, Real) or not _is_finite(value)
               or value < 0):
            logger.warning(
                "%r will not be encoded: Column Stat %s must be a finite "
                "nonnegative count (got %r). Using Null encoder.",
                column_name, stat_type, value)
            return None
    return [stats[stat_type] for stat_type in stat_types]


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
    """Select a feature encoder, warning and returning Null for unusable stats."""
    required = _required_stats(column_name, stats, (ColStatType.NA_FRACTION,))
    if required is None:
        return Null()
    na_fraction = required[0].value
    if na_fraction is not None and na_fraction > 1.0 - 1e-6:
        return Null()
    if stype == Stype.numerical:
        return select_numerical_encoder(stats, column_name=column_name)
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
        return select_sequence_encoder(dtype, stats, column_name=column_name)
    if stype == Stype.image:
        return Image()
    logger.warning("%r will not be encoded: no feature encoder for stype %r.",
                   column_name, stype)
    return Null()


def select_target_encoder(stype: Stype, stats: _EncoderStats) -> EncoderType:
    if stype == Stype.categorical:
        return Index()
    if stype == Stype.multicategorical:
        return MultiCategorical()
    if stype == Stype.numerical:
        reason = _stats_error(stats, _NUMERICAL_STATS, exact=False)
        if reason is not None:
            raise ValueError(reason)
        return _numerical_encoder([stats[k].value for k in _NUMERICAL_STATS])
    raise ValueError(f"no TARGET encoder for stype {stype!r}")


_NUMERICAL_STATS = (
    ColStatType.MEAN,
    ColStatType.MEDIAN,
    ColStatType.STD,
    ColStatType.MIN,
    ColStatType.MAX,
    ColStatType.KURTOSIS,
    ColStatType.QUANTILES,
)


def select_numerical_encoder(
    stats: _EncoderStats,
    *,
    column_name: str,
) -> EncoderType:
    required = _required_stats(column_name, stats, _NUMERICAL_STATS)
    if required is None:
        return Null()
    return _numerical_encoder([stat.value for stat in required])


def _numerical_encoder(values: list[Any]) -> EncoderType:
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
    required = _required_stats(column_name, stats, (
        ColStatType.COUNT,
        ColStatType.NUM_NA,
        ColStatType.NUM_UNIQUE,
        ColStatType.CATEGORY_COUNTS,
    ), exact=True)
    if required is None:
        return Null()
    count, num_na, num_unique, category_counts = [s.value for s in required]
    non_na = count - num_na
    if non_na < 0:
        logger.warning(
            "%r will not be encoded: NUM_NA (%s) exceeds COUNT (%s). "
            "Using Null encoder.", column_name, num_na, count)
        return Null()
    if non_na == 0:
        logger.warning(
            "%r will not be encoded because it only contains N/A values",
            column_name,
        )
        return Null()

    unique_fraction_threshold = CATEGORY_UNIQUE_THRESHOLD_MULTIPLIER * (
        1 / math.sqrt(non_na))
    unique_threshold = int(non_na * unique_fraction_threshold)
    min_occ = max(int(math.sqrt(num_unique)), 2)
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
    required = _required_stats(column_name, stats,
                               (ColStatType.NUM_UNIQUE_MULTI,))
    if required is None:
        return Null()
    num_unique = required[0].value
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
    required = _required_stats(column_name, stats, (ColStatType.NUM_UNIQUE,))
    if required is None:
        return Null()
    num_unique = required[0]
    if num_unique.value <= MAX_CATEGORIES:
        return select_categorical_encoder(column_name, dtype, stats)
    num_entries = num_unique.num_entries
    if num_entries is None:
        logger.warning(
            "%r will not be encoded: Column Stat %s must carry num_entries "
            "for ID encoder inferral. Using Null encoder.",
            column_name, ColStatType.NUM_UNIQUE)
        return Null()
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
    required = _required_stats(column_name, stats, (
        ColStatType.STRING_GLOVE_OVERLAP,
        ColStatType.STRING_AVG_TOKENS,
        ColStatType.STRING_MAX_TOKENS,
        ColStatType.STRING_AVG_NON_CHAR,
    ))
    if required is None:
        return Null()
    coverage, average, maximum, non_char = [s.value for s in required]
    if (coverage >= GLOVE_COVERAGE_THRESHOLD and non_char <= NON_CHAR_THRESHOLD
            and average > AVG_TOKENS_THRESHOLD and maximum > 2 * average):
        logger.debug("%r will be encoded with TextEmbedding", column_name)
        return TextEmbedding()
    num_unique = stats.get(ColStatType.NUM_UNIQUE)
    if (num_unique is not None and num_unique.value is not None
            and num_unique.value > MAX_CATEGORIES):
        logger.warning(
            "Text column %r will not be encoded by default: it has %s%s "
            "distinct values, exceeding MAX_CATEGORIES=%s, and its values "
            "are too short or have too few known words. To encode it as "
            "categorical data, use Hash() or Index() via "
            "model_plan.column_processing.encoder_overrides[%r].",
            column_name,
            "approximately " if num_unique.approx else "",
            num_unique.value,
            MAX_CATEGORIES,
            column_name,
        )
        return Null()
    logger.warning(
        "Text column %r will not be encoded by default: its values are too "
        "short or have too few known words. To encode it anyway, set its "
        "encoder to TextEmbedding().%s",
        column_name,
        _no_default_model(),
    )
    return Null()


def _no_default_model() -> str:
    from kumo_core.embedding_models import (
        DEFAULT_MODEL,
        INSTALL_DEFAULT_MODEL,
        _installed,
    )
    if DEFAULT_MODEL in _installed():
        return ""
    return (f" No word-vector model is installed for it either: "
            f"{INSTALL_DEFAULT_MODEL}, or pass embedding_models= to Trainer.")


def select_sequence_encoder(
    dtype: Dtype,
    stats: _EncoderStats,
    *,
    column_name: str,
) -> EncoderType:
    if dtype.is_list() or dtype.is_string():
        required = _required_stats(column_name, stats, (
            ColStatType.SEQUENCE_MIN_LENGTH,
            ColStatType.SEQUENCE_MAX_LENGTH,
        ))
        if required is None:
            return Null()
        minimum, maximum = [s.value for s in required]
        if minimum == maximum and minimum > 0:
            return NumericalList()
    return Null()
