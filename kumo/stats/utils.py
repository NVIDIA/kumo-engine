# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Annotated, Any, Dict, List

import pandas as pd
from kumo_core import stype_inference
from kumo_core.types import ColStatType, EdgeStatType
from kumo_core.types.schema import BaseColumnConfig
from pydantic import PlainSerializer, PlainValidator, WithJsonSchema

from io_utils import logging

from .stats import NUM_BINS_HISTOGRAM, Stats

# Round up to 8 floating point precision in histogram generation
HISTOGRAM_BIN_PRECISION = 8

logger = logging.getLogger(__name__)


def _validate_pandas_timestamp(value: object) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value)  # type: ignore[arg-type]


# ``pd.Timestamp`` has no pydantic-core schema of its own, so a bare
# ``TypeAdapter[pd.Timestamp]`` raises ``PydanticSchemaGenerationError`` (and
# FastAPI ``response_model=`` registration raises
# ``PydanticInvalidForJsonSchema``) under pydantic v2. Wrap it with a plain
# validator/serializer + an explicit JSON schema so dataclasses carrying
# ``pd.Timestamp`` fields can be used as FastAPI response models and round-trip
# through ``to_json``/``from_json``.
PandasTimestamp = Annotated[
    pd.Timestamp,
    PlainValidator(_validate_pandas_timestamp),
    PlainSerializer(lambda ts: ts.isoformat(), return_type=str),
    WithJsonSchema({
        'type': 'string',
        'format': 'date-time'
    }),
]


def _safe_divide(n: Any, d: Any, default_value: Any = None) -> Any:
    try:
        return n / d
    except ZeroDivisionError:
        caller_name = inspect.stack()[1].function
        logger.error(
            'Function %s attempted to divide %d by %d, setting output to %s.',
            caller_name, n, d, str(default_value))
        return default_value


def col_stat_dict_complete(
        result_dict: Dict[BaseColumnConfig, Dict[ColStatType, Stats]],
        request_dict: Dict[BaseColumnConfig, List[ColStatType]]) -> bool:
    r"""Check if all the stats in the request dict were included in the result
    dict

    Args:
        result_dict (Dict[BaseColumnConfig, List[ColStatType]]): the dict to be
            checked for completeness
        request_dict (Dict[BaseColumnConfig, Dict[ColStatType, Stats]]): the
            dict of all col stats that should be included

    """
    for col_config, request_list in request_dict.items():
        if request_list and col_config not in result_dict.keys():
            logger.warning(f"col missing from stats entirely: {col_config}")
            return False
        for stat in request_list:
            if stat not in result_dict[col_config].keys():
                logger.warning(f"missing stat for {col_config.name}: {stat}")
                return False
    return True


# Given a stat dict with the BASE_FKEY_STATS already added, compute the easily
# derived fkey stats, DERIVED_FKEY_STATS. These simple operations are shared
# across backends.
def update_derived_fkey_stats(
    stat_dict: Dict[EdgeStatType, Stats],
    include_list: List[EdgeStatType],
    data_version: str,
):
    r"""Use the values in stat dict (BASE_FKEY_STATS) to compute the easily
    derived fkey stats, DERIVED_FKEY_STATS. These simple operations are shared
    across backends.

    Args:
        stat_dict (Dict[EdgeStatType, Stats]): stats compute so far and dict
            that newly computed stats will be added to
        include_list (List[EdgeStatType]): list specifying which derived stats
            to add to stat_dict
        data_version (str): data version stats are computed on
    """
    # Count stats will always be available
    num_src_nodes = stat_dict[EdgeStatType.NUM_SRC_NODES].value
    num_dst_nodes = stat_dict[EdgeStatType.NUM_DST_NODES].value
    edge_count = stat_dict[EdgeStatType.EDGE_COUNT].value

    # If derived stats based on nonzero degree counts
    num_src_nonzero_deg = 1
    if EdgeStatType.NUM_SRC_NONZERO_DEGREE in stat_dict:
        num_src_nonzero_deg = stat_dict[
            EdgeStatType.NUM_SRC_NONZERO_DEGREE].value
    num_dst_nonzero_deg = 1
    if EdgeStatType.NUM_DST_NONZERO_DEGREE in stat_dict:
        num_dst_nonzero_deg = stat_dict[
            EdgeStatType.NUM_DST_NONZERO_DEGREE].value

    # dict maps the stat type to the numerator and denominator to compute it
    stats_formulas = {
        EdgeStatType.AVG_SRC_DEGREE: (edge_count, num_src_nodes),
        EdgeStatType.AVG_DST_DEGREE: (edge_count, num_dst_nodes),
        EdgeStatType.AVG_SRC_NONZERO_DEGREE: (edge_count, num_src_nonzero_deg),
        EdgeStatType.AVG_DST_NONZERO_DEGREE: (edge_count, num_dst_nonzero_deg),
        EdgeStatType.FRAC_SRC_NONZERO_DEGREE:
        (num_src_nonzero_deg, num_src_nodes),
        EdgeStatType.FRAC_DST_NONZERO_DEGREE:
        (num_dst_nonzero_deg, num_dst_nodes),
        EdgeStatType.EDGE_DENSITY: (edge_count, (num_src_nodes * num_dst_nodes))
    }

    for stat_type, formula in stats_formulas.items():
        if stat_type in include_list:
            stat_dict[stat_type] = Stats(
                value=_safe_divide(formula[0], formula[1]),
                approx=False,
                data_version=data_version,
            )


# Preserve the hosted helper imports while sharing their implementation.
MULTICAT_MAX_SAMPLES = stype_inference.MULTICAT_MAX_SAMPLES
_multicat_sample_rows = stype_inference._multicat_sample_rows
detect_multicat_sep_values = stype_inference.detect_multicat_sep_values
detect_multicat_sep = stype_inference.detect_multicat_sep


def convert_str_to_timestamp(obj: Any, vars_to_convert: List[str]) -> Any:
    r"""Converts the string representation of a timestamp to a timestamp object
    for the specified variables. This is needed because sometimes when loading
    from json, the timestamp is stored as a string.

    Args:
        obj (Any): Object with variables to convert.
        vars_to_convert (List[str]): List of variable names to convert.
    """
    for var_name in vars_to_convert:
        var = getattr(obj, var_name, None)
        if var is not None and isinstance(var, str):
            setattr(obj, var_name, pd.Timestamp(var))
    return obj


def calc_histogram_integer_bin_width_and_num_bins(
        range_to_cover: int) -> tuple[int, int]:
    # calculate the minimum integer bin width which can cover the range with
    # NUM_BINS_HISTOGRAM bins or less
    bin_width = (range_to_cover +
                 (NUM_BINS_HISTOGRAM - 1)) // NUM_BINS_HISTOGRAM

    # trim off empty bins at the right
    num_bins = (range_to_cover + (bin_width - 1)) // bin_width

    return bin_width, num_bins
