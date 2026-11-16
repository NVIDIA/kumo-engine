# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from kumo_core.types import ColStatType, EdgeStatType, TableStatType
from kumo_core.utils.mixins import CastMixin
from kumo_core.utils.registry import register

from kumo.config import ConfigMixin, StatsConfig
from service_framework.env_var import EnvVar

ALL_COL_STAT_TYPES: List[ColStatType] = list(ColStatType)
ALL_TABLE_STAT_TYPES: List[TableStatType] = list(TableStatType)
INGESTED_TABLE_STAT_TYPES = [TableStatType.MATERIALIZED_SIZE]
ALL_EDGE_STAT_TYPES: List[EdgeStatType] = list(EdgeStatType)

APPROX_ONLY_COL_STATS = [
    ColStatType.STRING_AVG_TOKENS,
    ColStatType.STRING_MAX_TOKENS,
    ColStatType.STRING_GLOVE_OVERLAP,
    ColStatType.STRING_AVG_NON_CHAR,
    ColStatType.INVALID_FRACTION,
    ColStatType.SEQUENCE_MIN_LENGTH,
    ColStatType.SEQUENCE_MAX_LENGTH,
]
EXACT_ONLY_COL_STATS = [ColStatType.SUMMARY_STATS_BY_CLASS]
# Common constant between python and pyspark stats
NUM_BINS_HISTOGRAM = 20
# No.of rows to use for detecting the separator in multi-categorical columns.
NUM_ROWS_FOR_SEPARATOR_INFERENCE = 10000

# General number of rows to use to compute approximate stats
NUM_ROWS_FOR_APPROX_STATS = 10000

# BASE_FKEY_STATS are stats computed directly on an edge_index
BASE_FKEY_STATS = [
    EdgeStatType.EDGE_COUNT,
    EdgeStatType.NUM_SRC_NONZERO_DEGREE,
    EdgeStatType.NUM_DST_NONZERO_DEGREE,
    EdgeStatType.MAX_SRC_DEGREE,
    EdgeStatType.MAX_DST_DEGREE,
    EdgeStatType.NUM_SRC_NODES,
    EdgeStatType.NUM_DST_NODES,
]

# DERIVED_FKEY_STATS are simple calculations using values in BASE_FKEY_STATS,
# so they are computed in a secondary step to avoid repeating more expensive
# edge_index operations
DERIVED_FKEY_STATS = [
    EdgeStatType.AVG_SRC_DEGREE,
    EdgeStatType.AVG_DST_DEGREE,
    EdgeStatType.AVG_SRC_NONZERO_DEGREE,
    EdgeStatType.AVG_DST_NONZERO_DEGREE,
    EdgeStatType.FRAC_SRC_NONZERO_DEGREE,
    EdgeStatType.FRAC_DST_NONZERO_DEGREE,
    EdgeStatType.EDGE_DENSITY,
]

# maximum number of category counts to be returned - will return the top
# all categories that are at least 0.0001% of the non-N/A values for
# CATEGORY_COUNTS and MULTI_CATEGORY_COUNTS (at most 1000000 categories)
# Meanwhile, the databackend required at least MAX_NUM_CLASSES (1000) in order
# to support multi-label and multi-class tasks.
MIN_CATEGORY_FREQUENCY_FRAC = EnvVar('MIN_CATEGORY_FREQUENCY_FRAC', type=float,
                                     default_value=1e-6).parse_value(0.000001)

# Type for histogram represented by list of (bin_lower_bound, percent) pairs.
# Both bin and percent are rounded to 2 places, and percent is on the scale of
# [0,100] (not [0,1])
HistogramValueType = List[Tuple[float, float]]

MULTI_CAT_TYPES = [
    ColStatType.MULTI_CATEGORIES_SEPARATOR,
    ColStatType.NUM_UNIQUE_MULTI,
    ColStatType.MULTI_CATEGORY_COUNTS,
]


@register(data_cls=StatsConfig)
class Stats(ConfigMixin, CastMixin):
    r"""A representation of a single statistic. Stats are used to represent
    information about the data that is independent of Kumo structures.
    """
    def __init__(
        self,
        value: Any,
        approx: bool,
        data_version: Optional[Any] = None,
        num_entries: Optional[int] = None,
        timestamp: Optional[float] = None,
    ):
        self.value = value
        self.approx = approx
        self.num_entries = num_entries
        self.data_version = data_version
        self.timestamp = timestamp

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(value={self.value}, '
                f'approx={self.approx}, data_version={self.data_version}, '
                f'num_entries={self.num_entries or "???"})')

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any):
        # pydantic v1 used ``__get_validators__`` yielding ``None`` (i.e. accept
        # the value unchanged, no validation). The v2 equivalent is a plain
        # validator that returns the input as-is.
        from pydantic_core import core_schema
        return core_schema.no_info_plain_validator_function(lambda value: value)


def is_missing_stat_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def is_finite_stat_number(value: Any) -> bool:
    if is_missing_stat_value(value):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
