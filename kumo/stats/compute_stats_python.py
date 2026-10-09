# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import math
import os
import re
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from kumo_core.embedding_models import (
    EmbeddingModelNotFoundError,
    resolve_encoder,
)
from kumo_core.glove import GloveDisabledError
from kumo_core.stype_inference import (
    distinct_statistics,
    multicategorical_separator,
    multicategorical_token_counts,
)
from kumo_core.types import ColStatType, EdgeStatType
from kumo_core.types.conversions import to_dtype
from kumoapi.encoder import TextEmbedding
from kumoapi.typing import Dtype
from torch_geometric.utils import degree

from kumo.stats.stats import (
    ALL_EDGE_STAT_TYPES,
    BASE_FKEY_STATS,
    MIN_CATEGORY_FREQUENCY_FRAC,
    NUM_BINS_HISTOGRAM,
    HistogramValueType,
    Stats,
)
from kumo.stats.utils import (
    HISTOGRAM_BIN_PRECISION,
    _safe_divide,
    calc_histogram_integer_bin_width_and_num_bins,
    update_derived_fkey_stats,
)
from kumo.utils.embedding_models import embedding_model_store
from kumo.utils.text import Tokenizer

# Column stats implementations ################################################


def _count(ser: pd.Series, **kwargs) -> float:
    return kwargs['original_len']


# NUM_NA includes nulls, NUM_IRRATIONAL is nans and +-inf
def _num_irrational(ser: pd.Series, **kwargs) -> float:
    if not pd.api.types.is_numeric_dtype(ser):
        # in this case, return na count only
        return kwargs['original_len'] - len(ser)
    return len(
        ser[~np.isfinite(ser.values)]) + kwargs['original_len'] - len(ser)


def _num_na(ser: pd.Series, **kwargs) -> float:
    return _num_irrational(ser, **kwargs) + pd.isnull(ser).sum()


def _na_fraction(ser: pd.Series, **kwargs) -> float:
    return _num_na(ser, **kwargs) / kwargs['original_len']


def _mean(ser: pd.Series, **kwargs) -> float:
    return np.mean(ser[np.isfinite(ser.values)])


def _quantiles(ser: pd.Series, **kwargs) -> Union[int, float]:
    finite_ser = ser[np.isfinite(ser.values)]
    if len(finite_ser) == 0:
        return [0, 0, 0]
    return list(np.quantile(finite_ser.values, [0.25, 0.5, 0.75]))


def _quantile25(ser: pd.Series, **kwargs) -> Union[int, float]:
    finite_ser = ser[np.isfinite(ser.values)]
    if len(finite_ser) == 0:
        return 0
    return np.quantile(finite_ser.values, 0.25)


def _median(ser: pd.Series, **kwargs) -> Union[int, float]:
    return np.median(ser[np.isfinite(ser.values)])


def _quantile75(ser: pd.Series, **kwargs) -> Union[int, float]:
    finite_ser = ser[np.isfinite(ser.values)]
    if len(finite_ser) == 0:
        return 0
    return np.quantile(finite_ser.values, 0.75)


def _is_datetime_object_series(ser: pd.Series) -> bool:
    """Whether an object-dtype series holds temporal scalars.

    Snowflake object-dtype timestamps must skip the numeric ``np.isnan`` path,
    which raises ``TypeError`` on object arrays. infer_dtype also catches
    np.datetime64 scalars an isinstance(date) probe would miss.
    """
    if not pd.api.types.is_object_dtype(ser):
        return False
    return pd.api.types.infer_dtype(ser) in {'datetime', 'date', 'datetime64'}


def _min(ser: pd.Series, **kwargs) -> Union[int, float, pd.Timestamp]:
    if pd.api.types.is_datetime64_any_dtype(ser):
        return ser.min() if len(ser) > 0 else pd.NaT
    if _is_datetime_object_series(ser):
        non_null = ser.dropna()
        return non_null.min() if len(non_null) > 0 else pd.NaT
    if len(ser) == 0:
        return float('NaN')
    arr = ser.values
    arr = arr[~np.isnan(arr)]
    return np.min(arr)


def _max(ser: pd.Series, **kwargs) -> Union[int, float, pd.Timestamp]:
    if pd.api.types.is_datetime64_any_dtype(ser):
        return ser.max() if len(ser) > 0 else pd.NaT
    if _is_datetime_object_series(ser):
        non_null = ser.dropna()
        return non_null.max() if len(non_null) > 0 else pd.NaT
    if len(ser) == 0:
        return float('NaN')
    arr = ser.values
    arr = arr[~np.isnan(arr)]
    return np.max(arr)


def _std(ser: pd.Series, **kwargs) -> float:
    return np.std(ser[np.isfinite(ser.values)], ddof=1)


def _kurtosis(ser: pd.Series, **kwargs) -> float:
    filtered_ser = ser[np.isfinite(ser.values)]
    # Pandas>=2 `Series.kurtosis()` raises on TimedeltaArray (no `kurt`
    # reduction); the downstream consumer tolerates NaN kurtosis.
    if not pd.api.types.is_numeric_dtype(filtered_ser):
        return float('nan')
    return filtered_ser.kurtosis()


def _histogram_half_month(ser: pd.Series,
                          **kwargs) -> List[Tuple[datetime.date, float]]:
    # Coerce out-of-range Snowflake timestamps to NaT and drop them so the
    # histogram is best-effort over the in-range portion rather than raising
    # OutOfBoundsDatetime.
    coerced = pd.to_datetime(ser, errors='coerce').dropna()

    if len(coerced) == 0:
        return []

    mindate = coerced.min()
    maxdate = coerced.max()

    month_diff = (maxdate.to_period('M') - mindate.to_period('M')).n
    min_bin = pd.Timestamp(month=mindate.month, year=mindate.year,
                           day=1 + 15 * math.floor(mindate.day / 16))
    bins = [
        min_bin + pd.offsets.SemiMonthBegin(m, day_of_month=16)
        for m in range(2 * month_diff + 3)
    ]

    # if last date is in last half of a month, can drop the middle of the
    # next month to drop an empty range
    if maxdate.replace(tzinfo=None) < bins[-2]:
        bins = bins[:-1]
    bins_numeric = [b.timestamp() for b in bins]

    timestamp_ser = coerced.map(lambda x: x.timestamp())
    # hist requires max of last bin included, but we drop in return zip
    hist = np.histogram(timestamp_ser, bins=bins_numeric)
    # create a cleaned up set of bins w/out a freq field
    bins_final = [
        datetime.date(year=b.year, month=b.month, day=b.day) for b in bins
    ]
    # return hist values but using timestamp bins
    return list(zip(bins_final[:-1], hist[0]))


def _histogram(ser: pd.Series, **kwargs) -> HistogramValueType:
    # Filter out non-numerical values
    if is_data_int := pd.api.types.is_integer_dtype(ser):
        arr = ser.to_numpy(dtype="long")
    else:
        arr = ser.to_numpy(dtype="float")
        arr = arr[np.isfinite(arr)]

    # For short arrays, match the spark behavior
    if len(arr) <= 1:
        return []

    min_value = arr.min().item()
    max_value = arr.max().item()

    if (range_to_cover := max_value - min_value) == 0:
        return []

    if is_data_int:
        bin_width, num_bins = calc_histogram_integer_bin_width_and_num_bins(
            range_to_cover)

        # need to precompute integer bin edges, otherwise np.histogram will
        # return whole-number float bin edges
        bin_arg = [min_value + bin_width * i for i in range(num_bins + 1)]
        bin_arg[-1] = min(bin_arg[-1], 2**63 - 1)
    else:
        bin_arg = NUM_BINS_HISTOGRAM

    hist, bin_edges = np.histogram(
        arr,
        bins=bin_arg,
    )
    X = bin_edges[:-1]
    Y = (hist / len(arr) * 100).round(HISTOGRAM_BIN_PRECISION)

    if not is_data_int:
        X = X.round(HISTOGRAM_BIN_PRECISION)

    return list(zip(X.tolist(), Y.tolist()))


def _num_unique(ser: pd.Series, **kwargs) -> int:
    return distinct_statistics(ser, len(ser))[0]


def _unique_fraction(ser: pd.Series, **kwargs) -> Optional[float]:
    return distinct_statistics(ser, kwargs['original_len'])[1]


def _category_counts(ser: pd.Series, **kwargs) -> List[Tuple[Any, Any]]:
    # Returns list of (category, count) in descending order by count. Note that
    # this drops any NA values in the series:
    counts = ser.value_counts()

    # Sort by index then use an in-place method to preserve index order on tied
    # values - result in descending by value but alphabetical by index within
    # groups of same value
    sort_value = counts.sort_index().sort_values(ascending=False,
                                                 kind='mergesort')

    # Apply MIN_CATEGORY_FREQUENCY_FRAC
    min_count = MIN_CATEGORY_FREQUENCY_FRAC * len(ser)

    # For using this function to get num unique multicategorical
    if 'all_categories' in kwargs and kwargs['all_categories']:
        return list(sort_value.items())

    def _is_empty_str(x) -> bool:
        if isinstance(x, str) and len(x) == 0:
            return True
        return False

    from kumo.data import MAX_NUM_CLASSES
    return [
        item for (i, item) in enumerate(list(sort_value.items()))
        if ((i < MAX_NUM_CLASSES or item[1] >= min_count)
            and not _is_empty_str(item[0]))
    ]


def _multi_categories_separator(ser: pd.Series, **kwargs) -> Optional[str]:
    return multicategorical_separator(ser)


# Since Python stats computation is only used for approximate stats, we
# are detecting the delimiter on the entire series upto 10k rows.
# This implementation will differ for PySpark when computing exact stats.
def _multi_category_counts(ser: pd.Series, **kwargs) -> List[Tuple[str, int]]:
    r"""
    Returns list of tuples (category, count) sorted in decreasing order
    of the counts.
    """

    if len(ser) == 0:
        return []

    # ser is already with na's dropped
    kumo_dtype = to_dtype(ser.dtype, ser=ser)
    if (isinstance(ser.iloc[0], list) or isinstance(ser.iloc[0], np.ndarray)
            or kumo_dtype.is_list()):  # for array targets
        flattened_categories = []
        for s in ser:
            for category in s:
                # Ignore empty strings
                if isinstance(category, str) and len(category) == 0:
                    continue
                if pd.isna(category):
                    continue
                flattened_categories.append(category)
        item_count = len(flattened_categories)
        counts_list = pd.Series(
            flattened_categories).value_counts().sort_index().sort_values(
                ascending=False, kind='mergesort').items()
    else:
        # if column is object but above condition wasn't true,
        # object might still be processable as a string containing a list
        if ser.dtype != "string":
            ser = ser.astype("string")

        counts_list = ser.value_counts().items()
        sep = _multi_categories_separator(ser, **kwargs)
        if sep is None:
            # Quoted multicat case
            if ser.dropna().apply(lambda x: isinstance(x, str) and x.strip(
            ).startswith('[') and x.strip().endswith(']')).all():
                counts = defaultdict(int)
                item_count = 0
                for s in ser:
                    # Removes outer [ and ]
                    s = s.strip()[1:-1]
                    # Split and remove surrounding whitespace and ""
                    for category in s.split(","):
                        category = category.strip().strip('"')
                        if len(category) == 0 or pd.isna(category):
                            continue
                        counts[category] += 1
                        item_count += 1
                counts_list = counts.items()
            else:
                counts_list = _category_counts(ser, **kwargs)
                item_count = len(ser)
        else:
            counts = multicategorical_token_counts(ser, sep)
            item_count = sum(counts.values())
            counts_list = counts.items()

    # Apply MIN_CATEGORY_FREQUENCY_FRAC
    min_count = MIN_CATEGORY_FREQUENCY_FRAC * item_count

    # Lambda function sorts the count list by count descending and then by name
    # in ascending order. Then cap the return list at the fraction
    # MIN_CATEGORY_FREQUENCY_FRAC of non-null items
    full_count_list = sorted(counts_list, key=lambda k: (-k[1], k[0]))
    # For using this function to get num unique multicategorical
    if 'all_categories' in kwargs and kwargs['all_categories']:
        return full_count_list
    from kumo.data import MAX_NUM_CLASSES
    return [
        item for i, item in enumerate(full_count_list)
        if i < MAX_NUM_CLASSES or item[1] >= min_count
    ]


def _num_unique_multi(ser: pd.Series, **kwargs) -> int:
    return len(_multi_category_counts(ser, all_categories=True, **kwargs))


def snowflake_vector_str_udf(x: str) -> str:
    r"""
    Given a string from snowflake vector type, return a string
    in the same format as snowflake array type.

    Snowflake array and vector are formatted to string in different ways:
    array: [1, 2, 3] --> '[\n  1,\n  2,\n  3\n]'  # snowpark_func.to_array()
    vector: [1, 2, 3] --> '[1 2 3]'

    Example:
        snowflake_vector_str_udf('[\n  1,\n  2,\n  3\n]') -> '[1,2,3]'
        snowflake_vector_str_udf('[1.  2.2 3. ]') -> '[1.,2,3.]'
    """
    # No need to process if not a list-like string
    if not x.startswith('[') or not x.endswith(']'):
        return x
    x = x[1:-1]
    if x.startswith('\n') and x.endswith('\n'):
        elements = [e.strip(' \n') for e in x[1:-1].split(',')]
    else:
        # there could be consecutive spaces, ignore empty splits
        elements = [e for e in x.split(' ') if e]
    return f'[{",".join(elements)}]'


def _sequence_max_length(ser: pd.Series, **kwargs) -> int:
    if len(ser) == 0:
        return 0
    if isinstance(ser.iloc[0], list) or isinstance(ser.iloc[0], np.ndarray):
        try:
            lengths = ser.apply(len)
        except AttributeError:
            return 0
        return lengths.max()
    elif isinstance(ser.iloc[0], str):
        # For snowflake vector types
        ser = ser.apply(snowflake_vector_str_udf)
        return ser.apply(lambda x: x.count(',')).max() + 1
    return 0


def _sequence_min_length(ser: pd.Series, **kwargs) -> int:
    if len(ser) == 0:
        return 0
    if isinstance(ser.iloc[0], list) or isinstance(ser.iloc[0], np.ndarray):
        try:
            lengths = ser.apply(len)
        except AttributeError:
            return 0
        # Filter out empty lists in the case of bigquery
        lengths = lengths[lengths > 0]
        return lengths.min()
    elif isinstance(ser.iloc[0], str):
        # For snowflake vector types
        ser = ser.apply(snowflake_vector_str_udf)
        return ser.apply(lambda x: x.count(',')).min() + 1
    return 0


def _string_avg_len(ser: pd.Series, **kwargs) -> float:
    try:
        str_len_ser = ser.str.len()
    except AttributeError:
        return None
    return np.mean(str_len_ser.values)


def _string_max_len(ser: pd.Series, **kwargs) -> int:
    try:
        str_len_ser = ser.str.len()
    except AttributeError:
        return None
    return np.max(str_len_ser.values)


def _num_tokens_batch(ser: pd.Series) -> List[int]:
    tokenizer = Tokenizer()
    return [len(tokenizer(sentence)) for sentence in ser]


def _string_max_tokens(ser: pd.Series, **kwargs) -> int:
    return max(_num_tokens_batch(ser)) if len(ser) > 0 else 0


def _string_avg_tokens(ser: pd.Series, **kwargs) -> float:
    if len(ser) > 0:
        return sum(_num_tokens_batch(ser)) / len(ser)
    return float('NaN')


def _string_avg_non_char(ser: pd.Series, **kwargs) -> float:
    if len(ser) == 0:
        return float('NaN')
    tokenizer = Tokenizer()
    total_tokens = 0
    non_char_tokens = 0
    for i, sentence in enumerate(ser):
        for token in tokenizer(sentence):
            if not re.fullmatch('[a-zA-Z]+', token):
                non_char_tokens += 1
            total_tokens += 1
    return non_char_tokens / total_tokens


def _string_glove_overlap(ser: pd.Series, **kwargs) -> float:
    r"""Find the percentage of text that GloVe vocabulary covers."""
    if len(ser) == 0:
        return 0
    # The dispatcher catches ``GloveDisabledError``.
    if os.environ.get("KUMO_ENABLE_GLOVE", "true").lower() == "false":
        raise GloveDisabledError(
            "STRING_GLOVE_OVERLAP skipped: KUMO_ENABLE_GLOVE is False.")
    try:
        model = resolve_encoder(TextEmbedding(), embedding_model_store())
    except EmbeddingModelNotFoundError as error:
        raise GloveDisabledError(
            "STRING_GLOVE_OVERLAP skipped: the default text embedding model "
            "is not installed.") from error
    tokenizer = Tokenizer()
    flat_tokens = []
    for i, sentence in enumerate(ser):
        tokens = tokenizer(sentence)
        flat_tokens.extend(tokens)
    if len(flat_tokens) == 0:
        return 0
    _, mask = model.lookup(flat_tokens)
    return mask.sum().item() / len(mask)


def refine_list_dtype(ser: pd.Series, kumo_dtype: Dtype) -> Dtype:
    """Refine an object-dtyped series to a typed list ``Dtype`` if applicable.

    Parquet round-trips list-typed columns as ``object`` series whose
    cells are ``np.ndarray``/``list``; ``to_dtype`` then reports
    ``Dtype.string`` because it inspects only the column dtype. That
    mis-classification would request hashing-based stats (NUM_UNIQUE,
    CATEGORY_COUNTS, ...) which blow up with
    ``TypeError: unhashable type: 'numpy.ndarray'``. Sniff the first
    non-null cell to recover the typed list dtype DG ingest would use.

    Shared by the test seed helper and the production seed entry point
    ``Task.seed_train_table_col_stats_from_dataframe`` so dtype refinement
    stays in lock-step.
    """
    if kumo_dtype.is_list():
        return kumo_dtype
    valid = ser.dropna()
    if len(valid) == 0:
        return kumo_dtype
    first = valid.iloc[0]
    if not isinstance(first, (list, np.ndarray)):
        return kumo_dtype
    elements = list(first) if not isinstance(first, np.ndarray) else first
    if len(elements) == 0:
        # Empty list element — fall back to stringlist (most permissive
        # multi-cat stat set).
        return Dtype.stringlist
    sample = elements[0]
    if isinstance(sample, str):
        return Dtype.stringlist
    if isinstance(sample, (bool, np.bool_)):
        return Dtype.stringlist
    if isinstance(sample, (int, np.integer)):
        return Dtype.intlist
    if isinstance(sample, (float, np.floating)):
        return Dtype.floatlist
    return kumo_dtype


_HEAVY_OR_UPSTREAM_STATS: 'frozenset[ColStatType]' = frozenset({
    ColStatType.INVALID_FRACTION,
    ColStatType.SUMMARY_STATS_BY_CLASS,
    ColStatType.DROPPED_ROWS,
    ColStatType.STRING_AVG_TOKENS,
    ColStatType.STRING_MAX_TOKENS,
    ColStatType.STRING_GLOVE_OVERLAP,
    ColStatType.STRING_AVG_NON_CHAR,
})

# Subset of ``_HEAVY_OR_UPSTREAM_STATS`` that is genuinely Python-computable
# by ``compute_col_stats_python`` (per-series text/token NLP stats) and is
# excluded from the seed path only by policy (these are heavy to compute at
# production scale, so DG ingest's Rust pass owns them). When a caller
# explicitly opts in via ``include_heavy_nlp=True`` we re-admit exactly this
# subset so DG-native unit tests that assert on text encoder behavior
# can seed real values without running full DG ingest.
#
# Notably this does NOT include ``SUMMARY_STATS_BY_CLASS`` (no per-series
# Python computation exists: it needs per-class grouping over a target
# column, which the single-Series API cannot express; the DG Rust producer
# does not emit it either), nor the vestigial cast-phase stats
# (``INVALID_FRACTION`` / ``DROPPED_ROWS``).
_PYTHON_COMPUTABLE_NLP_STATS: 'frozenset[ColStatType]' = frozenset({
    ColStatType.STRING_AVG_TOKENS,
    ColStatType.STRING_MAX_TOKENS,
    ColStatType.STRING_GLOVE_OVERLAP,
    ColStatType.STRING_AVG_NON_CHAR,
})


def supported_supplemental_stats_for_dtype(
        dtype: Dtype, include_heavy_nlp: bool = False) -> List[ColStatType]:
    """Return the ColStatType set DG-side seed helpers emit for a dtype.

    ``compute_col_stats_python`` raises if asked for a stat it cannot
    produce for the dtype, so the per-dtype filter is required.

    Args:
        dtype: The column dtype to filter supported stats for.
        include_heavy_nlp: When ``True``, additionally admit the
            Python-computable text/token NLP stats
            (``_PYTHON_COMPUTABLE_NLP_STATS``) that are otherwise excluded by
            policy. ``SUMMARY_STATS_BY_CLASS`` is never admitted (no
            per-series Python computation exists). The ``STRING_GLOVE_OVERLAP``
            stat additionally honors the ``KUMO_ENABLE_GLOVE`` env gate inside
            ``compute_col_stats_python`` (skipped, not failed, when disabled).
    """
    excluded = _HEAVY_OR_UPSTREAM_STATS
    if include_heavy_nlp:
        excluded = excluded - _PYTHON_COMPUTABLE_NLP_STATS
    return [
        stat for stat in ColStatType
        if stat not in excluded and stat.supports(dtype)
    ]


def compute_supplemental_col_stats_from_series(
    ser: pd.Series,
    *,
    data_version: Any,
    approx: bool = False,
    include_heavy_nlp: bool = False,
) -> Tuple[Dtype, Dict[ColStatType, Stats]]:
    """Compute the canonical DG-supplemental :class:`Stats` set for a Series.

    Shared entry point so the test seed helper
    (``test/util/cache_seeding.py::seed_col_stats_cache``) and the
    production seed entry point
    (:meth:`kumo.task.task.Task.seed_train_table_col_stats_from_dataframe`)
    emit identical payloads to DG ingest's supplemental pass.

    Returns the refined :class:`Dtype` (post-list-refinement) alongside
    the stats dict so callers key their cache with the same dtype the
    stats were computed against.

    Args:
        ser: The column data to compute stats over.
        data_version: Data-version tag stamped on the resulting ``Stats``.
        approx: Whether to compute stats approximately.
        include_heavy_nlp: Opt-in flag (default ``False`` preserves the
            existing seed behavior for all production callers). When ``True``,
            additionally emit the Python-computable text/token NLP stats
            (avg/max tokens, GloVe overlap, non-char fraction). Used by
            DG-native unit tests that assert on text encoder behavior
            without running full DG ingest.
    """
    kumo_dtype = to_dtype(ser.dtype, ser=ser)
    kumo_dtype = refine_list_dtype(ser, kumo_dtype)
    wanted = supported_supplemental_stats_for_dtype(
        kumo_dtype, include_heavy_nlp=include_heavy_nlp)
    if not wanted:
        return kumo_dtype, {}
    stats_for_col = compute_col_stats_python(
        ser,
        stats=wanted,
        approx=approx,
        data_version=data_version,
    )
    return kumo_dtype, stats_for_col


col_methods: Dict[ColStatType, Callable] = {
    ColStatType.COUNT: _count,
    ColStatType.NUM_NA: _num_na,
    ColStatType.NA_FRACTION: _na_fraction,
    ColStatType.MEAN: _mean,
    ColStatType.QUANTILES: _quantiles,
    ColStatType.QUANTILE25: _quantile25,
    ColStatType.MEDIAN: _median,
    ColStatType.QUANTILE75: _quantile75,
    ColStatType.MIN: _min,
    ColStatType.MAX: _max,
    ColStatType.STD: _std,
    ColStatType.HISTOGRAM: _histogram,
    ColStatType.HISTOGRAM_HALF_MONTH: _histogram_half_month,
    ColStatType.NUM_UNIQUE: _num_unique,
    ColStatType.UNIQUE_FRACTION: _unique_fraction,
    ColStatType.NUM_UNIQUE_MULTI: _num_unique_multi,
    ColStatType.MULTI_CATEGORY_COUNTS: _multi_category_counts,
    ColStatType.MULTI_CATEGORIES_SEPARATOR: _multi_categories_separator,
    ColStatType.CATEGORY_COUNTS: _category_counts,
    ColStatType.STRING_AVG_LEN: _string_avg_len,
    ColStatType.STRING_MAX_LEN: _string_max_len,
    ColStatType.STRING_AVG_TOKENS: _string_avg_tokens,
    ColStatType.STRING_MAX_TOKENS: _string_max_tokens,
    ColStatType.STRING_GLOVE_OVERLAP: _string_glove_overlap,
    ColStatType.STRING_AVG_NON_CHAR: _string_avg_non_char,
    ColStatType.KURTOSIS: _kurtosis,
    ColStatType.NUM_IRRATIONAL: _num_irrational,
    ColStatType.SEQUENCE_MAX_LENGTH: _sequence_max_length,
    ColStatType.SEQUENCE_MIN_LENGTH: _sequence_min_length,
}


# TODO(Alex, Kiriti): Current implementation calls multi_categories_separator
# multiple times. Cache the result to re-use for other multi-categorical stats.
def compute_col_stats_python(
    ser: pd.Series,
    stats: Iterable[ColStatType],
    approx: bool = False,
    data_version: Any = None,
) -> Dict[ColStatType, Stats]:
    r"""
    Computes statistics for all requested statistic types, given a column
    configuration and a pd.Series containing values from that column. If a
    statistic is not supported on the column datatype, it is simply ignored.

    Args:
        col (BaseColumnConfig): the column configuration.
        ser (pd.Series): the column data.
        stats (List[ColStatType]): the list of column statistics to
            approximate for this column.
        approx (bool, optional): Whether to compute statistics
            approximately or exactly (default: :obj:`False`).
        data_version (Any, optional): the data version of the column (default:
            None).

    Returns:
        Dict[ColStatType, Stats]: A dictionary mapping the :obj:`ColStatType`
        to a :obj:`Stats` object that contains the computed statistics.
    """
    dtype = to_dtype(ser.dtype, ser=ser)
    valid_ser = ser.dropna()
    # `INVALID_FRACTION` is handled at higher layer and we can't group by
    # column in python.
    skip_stats = [
        ColStatType.INVALID_FRACTION, ColStatType.SUMMARY_STATS_BY_CLASS,
        ColStatType.DROPPED_ROWS
    ]
    stats = [stat for stat in stats if stat not in skip_stats]

    out = {}

    for stat_type in stats:
        if stat_type not in col_methods:
            raise ValueError(f"Inappropriate statistic type {stat_type}.")
        if stat_type.supports(dtype):
            try:
                value = col_methods[stat_type](valid_ser, original_len=len(ser))
            except ZeroDivisionError:  # some stats fail for empty arrays
                value = None
            except GloveDisabledError:
                # ``KUMO_ENABLE_GLOVE=False`` or a missing default model
                # skips GloVe-dependent stats (currently only
                # STRING_GLOVE_OVERLAP) without failing the stats compute.
                continue

            # TODO(manan): unpack and handle iterable values:
            if isinstance(value, np.generic):
                value = value.item()  # Unpack numpy scalar.
            if isinstance(value, pd._libs.missing.NAType):
                value = np.nan  # Pandas NA is buggy and cannot be serialized.

            out[stat_type] = Stats(
                value=value,
                approx=approx,
                data_version=data_version,
                num_entries=len(ser),
                timestamp=None,
            )
        else:
            raise ValueError(
                f"Cannot calculate stat {stat_type} for dtype {dtype}.")

    return out


# Edge stats implementations ##################################################


def _edge_count(edge_index: torch.Tensor, **kwargs) -> int:
    return edge_index.size(1)


def _edge_density(edge_index: torch.Tensor, **kwargs) -> Optional[float]:
    num_src_nodes = kwargs['num_src_nodes']
    num_dst_nodes = kwargs['num_dst_nodes']
    return _safe_divide(edge_index.size(1), (num_src_nodes * num_dst_nodes))


def _num_src_nonzero_degree(edge_index: torch.Tensor, **kwargs) -> int:
    num_src_nodes = kwargs['num_src_nodes']
    mask = edge_index.new_zeros(num_src_nodes, dtype=torch.bool)
    mask[edge_index[0]] = True
    return int(mask.sum())


def _num_dst_nonzero_degree(edge_index: torch.Tensor, **kwargs) -> int:
    num_dst_nodes = kwargs['num_dst_nodes']
    mask = edge_index.new_zeros(num_dst_nodes, dtype=torch.bool)
    mask[edge_index[1]] = True
    return int(mask.sum())


def _max_src_degree(edge_index: torch.Tensor, **kwargs) -> int:
    if _edge_count(edge_index) == 0:
        return 0
    return int(degree(edge_index[0]).max())


def _max_dst_degree(edge_index: torch.Tensor, **kwargs) -> int:
    if _edge_count(edge_index) == 0:
        return 0
    return int(degree(edge_index[1]).max())


def _num_src_nodes(edge_index: torch.Tensor, **kwargs) -> int:
    return kwargs['num_src_nodes']


def _num_dst_nodes(edge_index: torch.Tensor, **kwargs) -> int:
    return kwargs['num_dst_nodes']


# Mapping from ColStatType -> implementations
edge_methods: Dict[EdgeStatType, Callable] = {
    EdgeStatType.EDGE_COUNT: _edge_count,
    EdgeStatType.EDGE_DENSITY: _edge_density,
    EdgeStatType.NUM_SRC_NONZERO_DEGREE: _num_src_nonzero_degree,
    EdgeStatType.NUM_DST_NONZERO_DEGREE: _num_dst_nonzero_degree,
    EdgeStatType.MAX_SRC_DEGREE: _max_src_degree,
    EdgeStatType.MAX_DST_DEGREE: _max_dst_degree,
    EdgeStatType.NUM_SRC_NODES: _num_src_nodes,
    EdgeStatType.NUM_DST_NODES: _num_dst_nodes,
}


def compute_edge_stats_python(
    edge_index: torch.Tensor,
    num_src_nodes: int,
    num_dst_nodes: int,
    data_version: Any = None,
) -> Dict[EdgeStatType, Stats]:
    r"""
    Computes statistics in BASE_FKEY_STATS and DERIVED_FKEY_STATS

    Args:
        edge_index (Tensor): the edge_index tensor
        num_src_nodes (int): number of source nodes/rows
        num_dst_nodes (int): number of dst nodes/rows
        data_version (Any, optional): the data version of the column (default:
            None).

    Returns:
        Dict[EdgeStatType, Stats]: A dictionary mapping the :obj:`EdgeStatType`
        to a :obj:`Stats` object that contains the computed statistics.
    """

    out = {}

    for stat_type in BASE_FKEY_STATS:
        if stat_type not in edge_methods:
            raise ValueError(f"Inappropriate statistic type {stat_type}.")
        val = edge_methods[stat_type](edge_index, num_src_nodes=num_src_nodes,
                                      num_dst_nodes=num_dst_nodes)

        out[stat_type] = Stats(
            value=val,
            approx=False,
            data_version=data_version,
            num_entries=edge_index.size(1),
            timestamp=None,
        )

    # TODO: implement degree distribution analysis here
    out[EdgeStatType.IN_DEGREE_PERCENTILE_LIST] = Stats(value=[], approx=False,
                                                        data_version=None)

    # Since edge stats via python doesn't take an argument of which stats
    # to compute, pass ALL_EDGE_STAT_TYPES so the derived method adds all
    # derived stats
    update_derived_fkey_stats(out, ALL_EDGE_STAT_TYPES, data_version)
    return out
