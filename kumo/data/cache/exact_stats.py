# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from kumo_core.types import ColStatType, TableStatType
from kumoapi.typing import Stype

from io_utils.json import fast_from_json
from kumo.config import StatsConfig
from kumo.stats.stats import APPROX_ONLY_COL_STATS

logger = logging.getLogger(__name__)


def parse_table_stat_from_key(key: str) -> Tuple[Optional[TableStatType], bool]:
    if len(key.split("__")) < 2:
        raise ValueError(f"Table stat key needs one '__' but got {key}")
    parts = key.rsplit("__", 1)

    # Fix backward incompatibility where DROPPED_ROWS is a ColStat and is no
    # longer a TableStat. This change is in v2.14.
    # Any ingestion performed with a release version <= v2.13 will have
    # DROPPED_ROWS as a TableStat.
    if parts[0] == "DROPPED_ROWS":
        return None, True
    stat = TableStatType(parts[0])
    approx = parts[1] == 'True'
    return stat, approx


def parse_col_stat_from_key(key: str) -> Tuple[str, ColStatType, bool]:
    if len(key.split("__")) == 3:
        # No 'stype' in the column statistic:
        parts = key.rsplit("__", 2)
        col = parts[0]
        stat = ColStatType(parts[1])
        approx = parts[2] == 'True'
    elif len(key.split("__")) >= 4:
        # Yes 'stype' in the column statistic (parts[1]). Note that this is a
        # '>' and not '==' since some columns may have '__' in their name,
        # which needs to be handled:
        parts = key.rsplit("__", 3)
        try:
            # If parts[1] is an stype, then the key is well-formed. If it is
            # not an stype, then the key is not well-formed (parts[1] is just
            # a part of the column name), and so we need to split by 2 __,
            # not 3:
            _ = Stype(parts[1].split(".")[-1])  # convert kumo.X to X
            col = parts[0]
            stat = ColStatType(parts[2])
            approx = parts[3] == 'True'
        except ValueError:
            parts = key.rsplit("__", 2)
            col = parts[0]
            stat = ColStatType(parts[1])
            approx = parts[2] == 'True'
    else:
        raise ValueError(f"Table stat key needs two/three '__' but got {key}")
    return col, stat, approx


class NoExactStatsFound(RuntimeError):
    pass


@dataclass(frozen=True)
class ExactStats:
    """All exact stats that are related to a single table.
    """
    table_stats: Dict[TableStatType, StatsConfig]
    col_stats: Dict[str, Dict[ColStatType, StatsConfig]]

    @staticmethod
    def parse(table_id: str, table_json: bytes,
              col_json: Optional[bytes]) -> 'ExactStats':
        """Parse exact stats from the given JSON files. Raise exception if
        exact stats are missing or failed to be parsed, for either table_json
        or col_json (both must be valid to succeed).

        If col_json is None, skip parsing col stats.
        """
        table_dict = fast_from_json(table_json, Dict[str, StatsConfig])

        table_stats: Dict[TableStatType, StatsConfig] = {}
        for key, stat in table_dict.items():
            stat_type, approx = parse_table_stat_from_key(key)
            if approx or stat_type is None:
                continue
            table_stats[stat_type] = stat
        if len(table_stats) == 0:
            raise NoExactStatsFound(
                "No exact stats could be found in the table_json")
        if col_json is None:
            return ExactStats(table_stats, {})

        col_dict = fast_from_json(col_json, Dict[str, StatsConfig])
        col_stats: Dict[str, Dict[ColStatType, StatsConfig]] = defaultdict(dict)
        for key, stat in col_dict.items():
            try:
                col_name, stat_type, approx = parse_col_stat_from_key(key)
                if approx and stat_type not in APPROX_ONLY_COL_STATS:
                    continue
                col_stats[col_name][stat_type] = stat
            except ValueError as e:
                # If the stats in S3 have a stat type that doesn't exist
                # anymore don't return it but return the rest instead of
                # failing the whole parse function
                logger.exception(
                    'Unable to parse table %s key %s from exact stats, %s',
                    table_id, key, e)
        if len(col_stats) == 0:
            raise NoExactStatsFound(
                "No exact stats could be found in the col_json")

        return ExactStats(table_stats, col_stats)

    def get_table_stat(self, typ: TableStatType, default: float) -> float:
        """Get the given stat and cast to float. Return default if failing.
        """
        if typ not in self.table_stats:
            return default
        stat = self.table_stats[typ]
        try:
            return float(stat.value)
        except Exception:
            return default
