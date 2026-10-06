# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Semantic type inference: the rule that reads a column's statistics.

A column carries a data type and a semantic type, and the second does not
follow from the first. ``int64`` describes a transaction amount and a 1-5 star
rating equally well; only the values separate them. This module owns that
separation, because it is engine policy rather than client convenience:
:data:`MAX_CATEGORIES` states what the categorical encoder can represent, so it
has to move whenever that encoder does.

Two rules live here. :func:`infer_stype_from_dtype` covers the families a data
type settles on its own, and :func:`infer_stype_from_stats` covers the rest from
statistics. Both are pure -- no I/O, no torch, no filesystem -- so any caller
can import this module cheaply.

The rules read :class:`~kumoapi.typing.Stype`'s own ``supports_dtype`` rather
than restating which data types each family accepts, so the two stay in step:
tightening the vocabulary in :mod:`kumoapi.typing` tightens inference with it.
"""

from __future__ import annotations

from typing import Any, Optional

from kumoapi.typing import Dtype, Stype

#: Few enough distinct values that a categorical encoding loses nothing. Sized
#: for binary features held as a truthy value, a falsey value, and NA.
MAX_AUTO_CATEGORICAL_CONVERSION_THRESHOLD = 3

#: The most categories the categorical encoder will represent.
MAX_CATEGORIES = 4000

#: The most categories the multicategorical encoder will represent.
MAX_MULTI_LABEL_CATEGORIES = 500

#: Above this fraction of distinct values, an integer column reads as a
#: measurement rather than a code.
UNIQUE_INT_THRESHOLD = 0.05


def _as_dtype(dtype: Any) -> Dtype:
    """Returns ``dtype`` as a :class:`~kumoapi.typing.Dtype`.

    Callers hold the enum itself or its wire value -- a bare string off a
    schema payload -- and both should reach the same rule.
    """
    return dtype if isinstance(dtype, Dtype) else Dtype(dtype)


def infer_stype_from_dtype(dtype: Any) -> Optional[Stype]:
    """Infers a semantic type from a data type where possible.

    Returns ``None`` for the families a data type cannot settle -- the numeric
    and string ones -- which is where :func:`infer_stype_from_stats` takes over.
    """
    dtype = _as_dtype(dtype)
    if dtype in (Dtype.intlist, Dtype.stringlist):
        return Stype.multicategorical
    if dtype == Dtype.floatlist:
        return Stype.sequence
    if dtype == Dtype.bool:
        return Stype.categorical
    if dtype == Dtype.time:
        return Stype.timestamp
    if dtype == Dtype.date:
        return Stype.timestamp
    if dtype == Dtype.timedelta:
        return Stype.numerical
    if dtype == Dtype.unsupported:
        return Stype.unsupported
    return None


def infer_stype_without_stats(dtype: Any) -> Stype:
    """Infers a semantic type from a data type alone, preferring coverage.

    Unlike :func:`infer_stype_from_dtype`, which answers only where the data
    type is decisive, this always answers.
    """
    dtype = _as_dtype(dtype)
    decided = infer_stype_from_dtype(dtype)
    if decided is not None:
        return decided
    for stype in (Stype.numerical, Stype.timestamp, Stype.categorical,
                  Stype.sequence):
        if stype.supports_dtype(dtype):
            return stype
    return Stype.unsupported


def distinct_statistics(nonnull_values: Any,
                        total_rows: int) -> tuple[int, Optional[float]]:
    """Count distinct nonnull values against the full sample, including nulls."""
    num_unique = len(nonnull_values.unique())
    return num_unique, num_unique / total_rows if total_rows else None


def column_statistics(frame: Any) -> dict[str, tuple[int, float]]:
    """Scalar inference statistics with all sampled rows in the denominator."""
    statistics = {}
    for name in frame.columns:
        values = frame[name].dropna()
        if values.empty:
            continue
        try:
            num_unique, fraction = distinct_statistics(values, len(frame))
        except TypeError as error:
            # Native list columns have a dtype-derived semantic type; scalar
            # cardinality is neither defined nor needed for these values.
            if ('unhashable type' not in str(error) or not all(
                    isinstance(value, (list, tuple)) for value in values)):
                raise
            continue
        if fraction is not None:
            statistics[name] = (num_unique, fraction)
    return statistics


def infer_stype_from_stats(
    dtype: Any,
    num_unique: int,
    unique_fraction: float,
    num_unique_multi: Optional[int] = None,
    multicategorical_proven: bool = False,
) -> Optional[Stype]:
    """Infers a semantic type from a data type and the column's statistics.

    The order of the tests is load-bearing; they run cheapest and most specific
    first.

    Args:
        dtype: The column's data type.
        num_unique: Distinct values in the column.
        unique_fraction: Distinct values as a fraction of the rows counted.
        num_unique_multi: Distinct values after splitting on the multi-label
            separator. ``None`` -- the default -- skips the multicategorical
            test, which is what a caller that cannot split the column wants.
        multicategorical_proven: Whether a multi-label separator was found.

    Returns:
        The semantic type, or ``None`` when no rule matches. Callers that
        require an answer raise on ``None``; the statistics themselves are the
        useful thing to report when that happens.
    """
    dtype = _as_dtype(dtype)

    # If there are multi-labels, they're not nearly all unique, and the
    # count falls within a threshold, classify as multicategorical
    if (multicategorical_proven and num_unique_multi is not None
            and Stype.multicategorical.supports_dtype(dtype)
            and num_unique > 1.5 * num_unique_multi
            and num_unique_multi <= MAX_MULTI_LABEL_CATEGORIES):
        return Stype.multicategorical

    # If binary 0/1 features are represented as
    # a truthy value (1, 1.0, True) and a falsey value
    # (0, 0.0, False, NA), representing them as categorical features or
    # targets has equivalent representation power
    # TODO: Check that the underlying data does fit this schema
    if (Stype.categorical.supports_dtype(dtype)
            and num_unique <= MAX_AUTO_CATEGORICAL_CONVERSION_THRESHOLD):
        return Stype.categorical

    # Excluding floats, which are currently the only dtype with a more
    # constrained auto-categorical conversion threshold, check the
    # more restrictive threshold to see if we should classify as
    # categorical
    # TODO(Dulloor): float can be categorical
    if (Stype.categorical.supports_dtype(dtype) and not dtype.is_float()
            and num_unique <= MAX_CATEGORIES):
        if dtype.is_int() and unique_fraction > UNIQUE_INT_THRESHOLD:
            return Stype.numerical
        return Stype.categorical

    # If the numerical dtype (currently ints and floats) has more
    # unique values than both thresholds, classify as numerical
    if Stype.numerical.supports_dtype(dtype):
        return Stype.numerical

    if Stype.text.supports_dtype(dtype):
        return Stype.text

    return None
