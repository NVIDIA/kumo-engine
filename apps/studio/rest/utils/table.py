# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import os.path as osp
from enum import Enum
from typing import Dict, List, Optional

from kumo_core.types import ColStatType, TableStatType
from kumo_core.types.schema import BaseColumnConfig
from kumoapi.typing import Stype

from async_utils.sync import blocking_func
from io_utils import fsutil, logging
from kumo.connector.parquet_connector import ParquetConnector
from kumo.connector.parquet_utils import validate_parquet_schema
from kumo.data.backend_cache import ExactStats
from kumo.data.common import TableRefreshStage
from kumo.graph.utils import (
    get_full_encoder_stats,
    get_full_stats,
    get_minimal_stats,
)
from kumo.stats import is_missing_stat_value
from kumo.stats.stats import Stats
from kumo.utils import detect_csv_sep
from kumo.validation import ValidationException
from rest.config.table import FactTableTimeRange, TableStats
from rest.config.table_v2 import TableDefinition

log = logging.getLogger(__name__)


def generate_ui_stats_dict(cols) -> Dict[BaseColumnConfig, List[ColStatType]]:
    r"""Returns all the statistics that are used to display in the UI.
    Accepts either v1 ``ColumnArgs`` or v2 ``Column`` — only ``name``,
    ``dtype``, ``stype``, and the optional multicategorical delimiter are
    read; both expose those.
    """
    stats_dict = {}
    for col in cols:
        col_config = BaseColumnConfig(
            name=col.name, dtype=col.dtype, stype=col.stype,
            multicategorical_delimiter=(col.multicategorical_delimiter))
        stats_dict[col_config] = [
            ColStatType.NUM_NA,
            ColStatType.COUNT,
        ]

        if (col_config.stype != Stype.sequence
                and not col_config.dtype.is_list()):
            stats_dict[col_config] += [
                ColStatType.NUM_UNIQUE,
            ]

        if col_config.stype == Stype.numerical:
            stats_dict[col_config] += [
                ColStatType.MIN, ColStatType.MAX, ColStatType.MEAN,
                ColStatType.QUANTILES, ColStatType.STD, ColStatType.HISTOGRAM
            ]
        elif col_config.stype == Stype.categorical:
            stats_dict[col_config] += [
                ColStatType.CATEGORY_COUNTS,
            ]
        elif col_config.stype == Stype.multicategorical:
            stats_dict[col_config] += [
                ColStatType.NUM_UNIQUE_MULTI,
            ]
            if not col_config.dtype.is_list():
                stats_dict[col_config] += [
                    ColStatType.CATEGORY_COUNTS,
                    ColStatType.MULTI_CATEGORIES_SEPARATOR,
                ]
        elif col_config.stype == Stype.timestamp:
            stats_dict[col_config] += [
                ColStatType.MIN,
                ColStatType.MAX,
            ]
        if col_config.stype == Stype.sequence:
            stats_dict[col_config] += [
                ColStatType.SEQUENCE_MIN_LENGTH,
                ColStatType.SEQUENCE_MAX_LENGTH,
            ]
    return stats_dict


class ColStatsSet(str, Enum):
    """Predefined set of col stats required for different purpose."""

    UI = "UI"
    MINIMAL = 'MINIMAL'
    REMAINING_FULL_ENCODER = 'REMAINING_FULL_ENCODER'
    FULL_ENCODER = 'FULL_ENCODER'
    FULL = 'FULL'
    BP = 'BP'

    def to_table_refresh_stage(self) -> TableRefreshStage:
        if self == ColStatsSet.UI:
            return TableRefreshStage.COMPUTE_MIN_COL_STATS
        elif self == ColStatsSet.MINIMAL:
            return TableRefreshStage.COMPUTE_MIN_COL_STATS
        elif self == ColStatsSet.FULL_ENCODER:
            return TableRefreshStage.COMPUTE_FULL_COL_STATS
        elif self in {
                ColStatsSet.FULL,
                ColStatsSet.BP,
        }:
            return TableRefreshStage.COMPUTE_FULL_COL_STATS
        elif self == ColStatsSet.REMAINING_FULL_ENCODER:
            return TableRefreshStage.COMPUTE_FULL_COL_STATS
        else:
            raise ValueError(f"Unknown ColStatsSet: {self}")

    def get_stats_types(
            self, table: TableDefinition
    ) -> Dict[BaseColumnConfig, List[ColStatType]]:
        if self == ColStatsSet.UI:
            return generate_ui_stats_dict(table.cols)

        min_stat = get_minimal_stats(table.to_base_columns())
        min_stat = {k.base_config(): v for k, v in min_stat.items()}
        if self == ColStatsSet.MINIMAL:
            return min_stat

        full_stat = {
            k.base_config(): v
            for k, v in get_full_stats(table.to_col_objects()).items()
        }
        for stats in full_stat.values():
            stats[:] = [
                stat for stat in stats
                if stat != ColStatType.SUMMARY_STATS_BY_CLASS
            ]
        if self == ColStatsSet.FULL:
            return full_stat

        if self == ColStatsSet.BP:
            # DiskGraph's batch-prediction stats engine does not emit
            # NUM_IRRATIONAL. Keep the producer and reader on one explicit
            # current-engine contract so an otherwise complete result is not
            # reported as unavailable.
            for stats in full_stat.values():
                stats[:] = [
                    stat for stat in stats if stat != ColStatType.NUM_IRRATIONAL
                ]

            base_columns = table.to_base_columns()
            columns = {col.name for col in base_columns}
            if {'SCORE', 'CLASS'} <= columns:
                score_column_config = next(col.base_config()
                                           for col in base_columns
                                           if col.name == 'SCORE')
                full_stat[score_column_config].append(
                    ColStatType.SUMMARY_STATS_BY_CLASS)
            return full_stat

        full_encoder_stat = {
            k.base_config(): v
            for k, v in get_full_encoder_stats(table.to_col_objects()).items()
        }
        if self == ColStatsSet.FULL_ENCODER:
            return full_encoder_stat

        if self == ColStatsSet.REMAINING_FULL_ENCODER:
            return {
                col: list(set(full) - set(min_stat.get(col, [])))
                for col, full in full_encoder_stat.items()
            }

        raise RuntimeError(f'Unknown ColStatsSet: {self}')


@blocking_func()
def validate_csv_file(file_path: str):
    r"""Validates whether a CSV file is supported by Kumo. This currently
    checks:
        - Whether a valid separator can be detected

    Raises a ValidationException if the file is invalid.
    """
    log.debug(f"Validating CSV file: {file_path}")
    try:
        # Try detecting a separator:
        separator = detect_csv_sep(file_path)
        with fsutil.get_handle(file_path, 'r') as csv_file:
            rows = csv.reader(csv_file, delimiter=separator)
            next(rows, None)
            if next((row for row in rows if row), None) is None:
                raise ValueError("Uploaded file is empty - has no rows.")
    except UnicodeDecodeError as err:
        log.error(err)
        raise ValidationException(
            title=("Non UTF-8 encoded character detected in this file. "
                   "Please make sure that your CSV file is encoded using "
                   "UTF-8 before ingesting to Kumo."), message=f"{str(err)}")
    except Exception as err:
        log.error(f"Error when valdidating CSV file: {file_path}: {err}")
        raise ValidationException(title="Failed to validate CSV file",
                                  message=f"{str(err)}")


@blocking_func()
def validate_parquet_file(file_path: str):
    r"""Validates whether a Parquet file is supported by Kumo. This currently
    checks:
        - The file can be parsed and opened as a Parquet file
        - All the dtypes are supported by Kumo

    Raises a ValidationException if the file is invalid.
    """
    log.debug(f"Validating parquet file: {file_path}")
    # Create a temporary parquet connector for the file_path.
    dir_path = osp.dirname(file_path)
    parquet_connector = ParquetConnector(name="temp_conn", root_dir=dir_path)
    try:
        pq_dataset = parquet_connector._parquet_dataset(file_path)
        validate_parquet_schema(pq_dataset.schema)

        if not any(fragment.metadata.num_rows > 0
                   for fragment in pq_dataset.get_fragments()):
            raise ValueError("Uploaded file is empty - has no rows.")

    except ValidationException as val_ex:
        raise val_ex
    except Exception as err:
        raise ValidationException(title="Failed to validate parquet file",
                                  message=f"{str(err)}")


def get_time_range_from_stats(
    time_col: str,
    stats: Optional[ExactStats],
    col_stats_computed: bool = True,
) -> Optional[FactTableTimeRange]:
    """Extract time range from stats for a given time column.

    Args:
        time_col: The name of the time column
        stats: Optional exact stats containing column statistics
        col_stats_computed: Whether column statistics have been computed

    Returns:
        FactTableTimeRange if time range can be determined, None otherwise
    """
    # If no stats are available, then return approx=True, which will render as
    # "calculating" in the UI. Note that we need 'col_stats_computed' in the
    # case that we have table statistics present but column statistics are
    # still being computed:
    if stats is None or not col_stats_computed:
        return FactTableTimeRange(min=None, max=None, approx=True)

    # If there are no stats for the time col, then there is a bug in the
    # backend code. Return None, to indicate to the user that it is no longer
    # calculating.
    if time_col not in stats.col_stats:
        log.error(f"Exact stats for time col {time_col} are not present")
        return None

    col_stats = stats.col_stats[time_col]

    # If stats are computed but the specific stats are not available, then
    # there is a backend bug
    try:
        min = Stats.from_config(col_stats[ColStatType.MIN])
        max = Stats.from_config(col_stats[ColStatType.MAX])
    except KeyError:
        log.exception(f"Exact stats for time col {time_col} are "
                      f"missing min or max stats")
        return None

    min_value = None if is_missing_stat_value(min.value) else min.value
    max_value = None if is_missing_stat_value(max.value) else max.value
    return FactTableTimeRange(min=min_value, max=max_value, approx=min.approx
                              or max.approx)


def get_full_time_range(
    table: TableDefinition,
    stats: Optional[ExactStats],
    col_stats_computed: bool = True,
) -> Optional[FactTableTimeRange]:
    # If no time_col exists, show nothing
    if table.time_col is None:
        return None

    return get_time_range_from_stats(table.time_col, stats, col_stats_computed)


def get_table_stats(stats: Optional[ExactStats]) -> TableStats:
    """Table stats are saved on the entry in StateDB, but these may be out of
    date. If exact table stats are available, we should use those instead.
    """
    size_bytes, num_rows = 0, 0
    if stats is not None:
        if TableStatType.SOURCE_SIZE in stats.table_stats:
            size_bytes = stats.table_stats[TableStatType.SOURCE_SIZE].value
        if TableStatType.NUM_ROWS in stats.table_stats:
            num_rows = stats.table_stats[TableStatType.NUM_ROWS].value
    return TableStats(size_bytes=size_bytes, num_rows=num_rows)
