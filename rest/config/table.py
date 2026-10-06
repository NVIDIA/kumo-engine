# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import calendar
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from kumo_core.types import ColStatType
from kumo_core.types.schema import ColumnArgs
from kumoapi.typing import Dtype, Stype

from io_utils import kumo_dataclass, logging
from kumo.graph.graph import ForeignKey
from kumo.stats.stats import APPROX_ONLY_COL_STATS, Stats
from kumo.utils.cast import to_python_list, to_python_scalar
from statedb.config.common import ConnectorID, TableName
from statedb.config.table import TableDataVersion

Numeric = Union[float, int]
DateTime = Union[datetime.datetime, datetime.date, datetime.time,
                 datetime.timedelta]

MAX_CATEGORIES = 20
EMPTY_TIME = '00:00:00'

MISSING_STR = 'MISSING (<NA>)'

logger = logging.getLogger(__name__)

# Validation error messages
PKEY_PRESENT_IN_COLS = (
    "Primary Key (pkey) must be present in cols, but was not "
    "found. Please make sure your primary key is valid and "
    "present in your table and try again!")

PKEY_IS_ID = ("Primary Key (pkey) must be an ID. Please change the "
              "semantic type (stype) of your column to ID "
              "and try again!")

TIME_COL_PRESENT_IN_COLS = (
    "Time Column (time_col) must be present in cols, but was "
    "not found. Please make sure your time column is valid "
    "and present in your table and try again!")

END_TIME_COL_PRESENT_IN_COLS = (
    "End Time Column (end_time_col) must be present in cols, but was "
    "not found. Please make sure your end time column is valid "
    "and present in your table and try again!")

END_TIME_COL_IN_NON_DIM_TABLE = (
    "End Time Column (end_time_col) can only be "
    "specified for dimension tables. Please remove "
    "the end time column and try again!")

END_TIME_SAME_AS_TIME_COL = (
    "End Time Column (end_time_col) cannot be the same "
    "as the time column. Please select a different "
    "column and try again!")

TIME_COL_NEQ_PKEY = ("The time column cannot be the same as "
                     "the primary key")


def column_args_to_foreign_key(
    args: ColumnArgs,
    dst_connector_id: ConnectorID,
    dst_table_name: TableName,
    dst_col_name: str,
) -> ForeignKey:
    out = ForeignKey(
        name=args.name,
        dtype=args.dtype,
        dst_table=dst_table_name,
        dst_data_source=dst_connector_id,
        dst_key=dst_col_name,
    )
    out.stype = args.stype
    out.metadata = {}
    out.timestamp_unit = args.timestamp_unit
    return out


@kumo_dataclass
class TableStats:
    size_bytes: int
    num_rows: int


@kumo_dataclass
class FactTableTimeRange:
    min: Optional[datetime.datetime]
    max: Optional[datetime.datetime]
    approx: bool

    def __str__(self):
        range = f"{self.min} - {self.max}"
        if self.approx:
            range += ' (Approximate)'
        return range


@kumo_dataclass
class ColumnStatsCounts:
    total: int
    missing_percentage: float
    num_unique: int
    invalid_percentage: float
    # TODO(Dulloor): unique_fraction


PointDistribution = List[Tuple[Numeric, Numeric]]
CategoryDistribution = List[Tuple[str, Numeric]]

# A sequence of [start, end] date intervals, and a number associated with each
# interval. Intervals are expected to be contiguous (no gaps).
DateHistDistribution = List[Tuple[datetime.date, datetime.date, Numeric]]

BatchPredictionStats = Dict[str, Dict[str, Optional[Union[
    float,
    List[float],
]]]]
log = logging.getLogger(__name__)


def _format_time(dt: Any) -> str:
    if pd.isnull(dt):
        return "NA"

    if not isinstance(dt, datetime.datetime):
        # Due to backend bug, dt might not be a datetime #6730. In this
        # case, return a string rather than crashing.
        return "NA"

    dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    date, time = dt_str.split()
    if time == EMPTY_TIME:
        return date
    return dt.strftime("%Y-%m-%d %H:%M")


def _format_time_range(min: Any, max: Any) -> str:
    # Pandas timestamps are true for isinstance(datetime.datetime) and are
    # generally interchangeable but need to be converted to avoid possible
    # overflow errors in the timedelta
    if isinstance(min, pd.Timestamp):
        min = min.to_pydatetime()
    if isinstance(max, pd.Timestamp):
        max = max.to_pydatetime()

    if pd.isnull(min) or min is None or not isinstance(min, datetime.datetime):
        return "NA"

    if pd.isnull(max) or max is None or not isinstance(max, datetime.datetime):
        return "NA"

    range = max - min

    return f"{range.days:,} days"


@kumo_dataclass
class ColumnStats:
    counts: ColumnStatsCounts
    summary_stats: List[Tuple[str, Union[Numeric, str, DateTime, List]]]
    distribution_points: Optional[PointDistribution]
    distribution_category: Optional[CategoryDistribution]
    distribution_date: Optional[DateHistDistribution]

    # If Batch Prediction statistics were requested, this will be not None
    # Batch Prediction stats maps a class name -> metric name (min | max |
    # mean | distribution) to the value (float) or distribution (list)
    bp_stats: Optional[BatchPredictionStats]
    approx: bool

    @classmethod
    def from_stats_dict(cls, stype: Stype, stats_dict: Dict[ColStatType, Stats],
                        approx_sample_size: Optional[int] = None,
                        dtype: Optional[Dtype] = None) -> 'ColumnStats':
        approx = False  # if any of the retrieved stats are approx, set to True
        for stat_type, stat in stats_dict.items():
            if stat.approx and stat_type not in APPROX_ONLY_COL_STATS:
                approx = True
                break

        # NOTE: these getters tolerate a MISSING key, not just a None value.
        # Vestigial stats (``_VESTIGIAL_FULL_STATS`` in kumo/graph/column.py:
        # INVALID_FRACTION, DROPPED_ROWS) are intentionally dropped from the
        # requested ``full_stats`` set, so they won't be present in
        # ``stats_dict`` even when ``from_stats_dict`` reads them
        # unconditionally (e.g. INVALID_FRACTION below). The vestigial design
        # promises "downstream consumers tolerate absence (default-to-zero)";
        # ``stats_dict.get(...)`` is what makes that true here. Without it the
        # DG col-stats path (ingest + batch-prediction) raises KeyError.
        def get_int_stat(stat: ColStatType) -> int:
            entry = stats_dict.get(stat)
            if entry is None or entry.value is None:
                return 0
            return int(entry.value)

        def get_int_approx_rows_used() -> int:
            for stats in stats_dict.values():
                if (stats.num_entries is not None and stats.num_entries > 0
                        and stats.approx):
                    return stats.num_entries
            return 0

        def get_float_stat(stat: ColStatType) -> float:
            entry = stats_dict.get(stat)
            if entry is None or entry.value is None:
                return 0
            return float(entry.value)

        def get_list_stat(stat: ColStatType) -> List:
            entry = stats_dict.get(stat)
            if entry is None or entry.value is None:
                return []
            return to_python_list(entry.value)

        def get_bp_stats(
        ) -> Dict[str, Dict[str, Optional[Union[float, List[float]]]]]:
            """
            Extract BP stats (type dict) from stats_dict. If none are available
            return an empty dict

            Returns:
                Dict[str, Dict[str, Optional[Union[float, List[float]]]]]:
                    Dict mapping a class to a stat name (min | max | mean |
                    distribution) to the value (float) or distribution (list).
            """
            entry = stats_dict.get(ColStatType.SUMMARY_STATS_BY_CLASS)
            if entry is None or entry.value is None:
                return {}
            return entry.value

        def get_datetime_stat(stat: ColStatType) -> DateTime:
            entry = stats_dict.get(stat)
            if entry is None or entry.value is None:
                return datetime.datetime.utcfromtimestamp(0)
            return entry.value

        def get_str_stat(stat: ColStatType) -> str:
            entry = stats_dict.get(stat)
            if entry is None or entry.value is None:
                return ""
            return str(entry.value)

        rows_used = get_int_stat(ColStatType.COUNT)
        approx_rows_used = get_int_approx_rows_used()

        def top_k(
                counts: List[Tuple[str, int]], na_count: int,
                rows_used_for_stat: Optional[int] = 0
        ) -> List[Tuple[str, float]]:
            if rows_used_for_stat is None or rows_used_for_stat == 0:
                return []
            count_to_percent = lambda x: round(
                (to_python_scalar(x) / rows_used_for_stat) * 100, 2)
            missing_count = 0
            if (na_count != 0):
                sum_count = sum(b for _, b in counts)
                missing_count = rows_used_for_stat - sum_count
            counts.append((MISSING_STR, missing_count))
            new_counts: List[Tuple[str, float]] = [(str(a), count_to_percent(b))
                                                   for a, b in counts]
            return sorted(new_counts, key=lambda x: x[1],
                          reverse=True)[:MAX_CATEGORIES]

        def missing_percent(num_na: int, rows_used_for_stat: int) -> float:
            if rows_used_for_stat == 0:
                return 0
            return (num_na / rows_used_for_stat) * 100

        def missing_percent_result() -> float:
            missing_percent_result: float = 0.0
            # Tolerate a MISSING NUM_NA stat: the DG col-stats path emits an
            # empty per-column stats dict for ID/ENTITY columns, so indexing
            # ``NUM_NA`` directly raises ``KeyError``. A column with no NUM_NA
            # stat has no recorded missing values -> 0% missing.
            num_na_stat = stats_dict.get(ColStatType.NUM_NA)
            if num_na_stat is None:
                return 0.0
            if stype == Stype.categorical:
                missing_percent_count = (approx_rows_used if num_na_stat.approx
                                         and approx_rows_used > 0 else
                                         rows_used)
                return missing_percent(num_na_stat.value, missing_percent_count)
            else:
                missing_percent_count = (approx_rows_used if num_na_stat.approx
                                         and approx_rows_used > 0 else
                                         rows_used)
                missing_percent_result = missing_percent(
                    get_int_stat(ColStatType.NUM_NA), missing_percent_count)
            return missing_percent_result

        if stype == Stype.multicategorical:
            num_unique = get_int_stat(ColStatType.NUM_UNIQUE_MULTI)
        elif stype in [Stype.categorical, Stype.ID, Stype.timestamp]:
            num_unique = get_int_stat(ColStatType.NUM_UNIQUE)
        else:
            # For other data type such as numiercal, sequence, ..., we don't
            # provide unique count since it does not make sense.
            num_unique = get_int_stat(ColStatType.COUNT)
        counts = ColumnStatsCounts(
            total=rows_used, missing_percentage=missing_percent_result(),
            num_unique=num_unique,
            invalid_percentage=get_float_stat(ColStatType.INVALID_FRACTION) *
            100)
        summary_stats: List[Tuple[str, Union[Numeric, str, DateTime,
                                             List]]] = []
        distribution_points = None
        distribution_category = None
        distribution_date = None
        bp_stats = None
        if stype == Stype.numerical:
            # Say which stat is absent. Indexing the quantile list unguarded
            # turned a missing stat into a bare `IndexError` several frames
            # from the cause (kumo-ai/kumo#30917).
            quantiles = get_list_stat(ColStatType.QUANTILES)
            if len(quantiles) < 3:
                raise ValueError(
                    f"A numerical column needs quartile statistics, but "
                    f"{ColStatType.QUANTILES.value} holds {len(quantiles)} "
                    f"of the 3 expected. Stats present: "
                    f"{sorted(stat.value for stat in stats_dict)}.")
            summary_stats = [
                ('Mean', get_float_stat(ColStatType.MEAN)),
                ('Min', get_float_stat(ColStatType.MIN)),
                ('25%', quantiles[0]),
                ('50%', quantiles[1]),
                ('75%', quantiles[2]),
                ('Max', get_float_stat(ColStatType.MAX)),
            ]
            if ColStatType.HISTOGRAM in stats_dict:
                distribution_points = get_list_stat(ColStatType.HISTOGRAM)
            if ColStatType.SUMMARY_STATS_BY_CLASS in stats_dict:
                bp_stats = get_bp_stats()
        elif stype == Stype.timestamp:
            min = get_datetime_stat(ColStatType.MIN)
            max = get_datetime_stat(ColStatType.MAX)

            summary_stats = [
                ('Range', _format_time_range(min, max)),
                ('Min', _format_time(min)),
                ('Max', _format_time(max)),
            ]

            distribution_date = _get_date_hist(stats_dict)
        elif stype == Stype.categorical:
            summary_stats = []
            category_counts = stats_dict[ColStatType.CATEGORY_COUNTS]
            top_k_count = (approx_rows_used if category_counts.approx
                           and approx_rows_used > 0 else rows_used)
            distribution_category = top_k(
                get_list_stat(ColStatType.CATEGORY_COUNTS),
                get_int_stat(ColStatType.NUM_NA), top_k_count)
        elif stype == Stype.ID:
            summary_stats = []

        elif stype == Stype.multicategorical:
            summary_stats = []
            if ColStatType.MULTI_CATEGORIES_SEPARATOR in stats_dict:
                # Only add separator stats if available.
                summary_stats.append(
                    ('Separator',
                     get_str_stat(ColStatType.MULTI_CATEGORIES_SEPARATOR)))
            # multicat can also be list types which do not have category counts
            if ColStatType.CATEGORY_COUNTS in stats_dict:
                distribution_category = top_k(
                    get_list_stat(ColStatType.CATEGORY_COUNTS),
                    get_int_stat(ColStatType.NUM_NA))

        elif stype == Stype.sequence:
            # Sequence lengths are integer counts. They must be produced as
            # ints so the ``Union[Numeric, str, ...]`` summary-stat field keeps
            # them numeric: pydantic v1 silently coerced the old ``str`` values
            # back to numbers, but pydantic v2's smart-union keeps a ``str``
            # input as a ``str``, which broke the numeric contract.
            summary_stats = [
                ('Min Length', get_int_stat(ColStatType.SEQUENCE_MIN_LENGTH)),
                ('Max Length', get_int_stat(ColStatType.SEQUENCE_MAX_LENGTH)),
            ]

        return ColumnStats(
            counts,
            summary_stats,
            distribution_points,
            distribution_category,
            distribution_date,
            bp_stats,
            approx,
        )


@kumo_dataclass
class ColumnSummary:
    name: str
    stype: str
    dtype: str
    sample_rows: List[Optional[str]]
    statistics: Optional[ColumnStats] = None


@kumo_dataclass
class ColStatsResponse:
    statistics_available: bool
    columns: List[ColumnSummary]
    approx: bool
    rows_used: int


@kumo_dataclass
class ColStatsAtVersion:
    """
    Dataclass containing the column stats in dictionary form along with the
    version that the source data was at when the stats were calculated
    """
    version: Optional[TableDataVersion]
    stats_dict: Dict[str, ColumnStats]
    approx: bool
    num_rows_used: int


def _half_month_end(dt: datetime.date) -> datetime.date:
    # assumes dt is either the first or 16th
    # returns the 15th, or the last day of month
    if dt.day < 16:
        return datetime.date(dt.year, dt.month, 15)
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    return datetime.date(dt.year, dt.month, last_day)


def _get_date_hist(
        stats_dict: Dict[ColStatType, Stats]) -> DateHistDistribution:
    result: DateHistDistribution = []
    # In case of error, it's better to degrade gracefully rather than throw an
    # exception and crash the UI.
    try:
        # Tolerate a missing key the same way the numerical-histogram path
        # guards ``ColStatType.HISTOGRAM`` with an ``in`` check: a timestamp
        # column may carry no half-month histogram (e.g. the value was
        # dropped), and indexing it directly raises a spurious ``KeyError``
        # that gets logged as an error on every such column.
        stat = stats_dict.get(ColStatType.HISTOGRAM_HALF_MONTH)
        if stat is None or not stat.value:
            logger.debug("Histogram is empty or filled with missing values")
            return result

        for bin, val in stat.value:
            if not isinstance(bin, datetime.date):
                logger.error("Date hist bin should be date but got: %s",
                             stat.value)
                return []
            result.append((bin, _half_month_end(bin), int(val)))
    except Exception:
        logger.exception("failed to parse date histogram")
    return result
