# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import combinations
from types import SimpleNamespace

import pandas as pd
import pytest
from kumo_core.stype_inference import (
    column_statistics,
    infer_stype_from_stats,
    multicategorical_separator,
    multicategorical_token_counts,
)
from kumo_core.types import ColStatType
from kumoapi.typing import Dtype, Stype

from kumo.graph.table import StypeInferralStats, Table
from kumo.stats.compute_stats_python import compute_col_stats_python


@pytest.mark.parametrize('delimiter', [',', '|', ';', '\t'])
def test_local_multicat_inputs_match_hosted_statistics(delimiter):
    values = [delimiter.join(pair) for pair in combinations('abcde', 2)] * 12
    series = pd.Series(values + [None], name='tags')
    statistics = compute_col_stats_python(series, [
        ColStatType.NUM_UNIQUE,
        ColStatType.UNIQUE_FRACTION,
        ColStatType.NUM_UNIQUE_MULTI,
        ColStatType.MULTI_CATEGORIES_SEPARATOR,
    ])
    inferred_delimiter = multicategorical_separator(series.dropna())
    num_unique_multi = len(
        multicategorical_token_counts(series.dropna(), inferred_delimiter))
    assert inferred_delimiter == statistics[
        ColStatType.MULTI_CATEGORIES_SEPARATOR].value == delimiter
    assert num_unique_multi == statistics[ColStatType.NUM_UNIQUE_MULTI].value
    scalar_stats = column_statistics(series.to_frame())['tags']
    local = infer_stype_from_stats(Dtype.string, *scalar_stats,
                                   num_unique_multi, True)
    hosted = Table._infer_stype_from_stats(
        SimpleNamespace(name='users'), Dtype.string,
        StypeInferralStats(
            num_unique=statistics[ColStatType.NUM_UNIQUE],
            unique_fraction=statistics[ColStatType.UNIQUE_FRACTION],
            num_unique_multi=statistics[ColStatType.NUM_UNIQUE_MULTI],
            multicategorical_proven=True))
    assert local == hosted == Stype.multicategorical


def test_hosted_token_counts_preserve_whitespace_and_ignore_empty_tokens():
    series = pd.Series(['a|| b|', ' b|a', '|c|'])
    stats = compute_col_stats_python(series,
                                     [ColStatType.MULTI_CATEGORY_COUNTS])
    assert stats[ColStatType.MULTI_CATEGORY_COUNTS].value == [
        (' b', 2),
        ('a', 2),
        ('c', 1),
    ]
