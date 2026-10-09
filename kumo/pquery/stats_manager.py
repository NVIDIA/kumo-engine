# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from kumo_core.types import TaskType, TrainTableType
from kumoapi.pquery import QueryType, ValidatedPredictiveQuery

import kumo.diskgraph.diskgraph_native_tt_artifacts as tt_artifacts
from async_utils.sync import blocking_method
from io_utils import logging
from kumo.data import get_active_cache
from kumo.graph import Graph
from kumo.stats import Stats, StatsConfig
from kumo.stats.train_table_stats import (
    ALL_STATIC_LINK_STATS,
    ALL_STATIC_NODE_STATS,
    ALL_TEMPORAL_LINK_STATS,
    ALL_TEMPORAL_NODE_STATS,
    ANCHOR_DATE,
    AVG_TARGETS_PER_TIMEFRAME,
    MAX_DATE,
    MIN_DATE,
    NUM_ENTITIES,
    NUM_LINKS,
    NUM_TIMEFRAMES,
    NUM_TIMEFRAMES_POST_FILTER,
    PRED_TIME,
    STATS_PER_TIMEFRAME,
    STATS_TYPE,
    TemporalTrainTableStats,
    load_train_table_stats_from_cache,
)
from kumo.utils.cast import normalize_timestamp_to_utc_local

logger = logging.getLogger(__name__)

_OPTIONAL_NATIVE_TRAIN_STATS = frozenset({
    AVG_TARGETS_PER_TIMEFRAME,
    NUM_LINKS,
    STATS_PER_TIMEFRAME,
})


class StatsManager:
    r"""A class that handles any actions related to training table stats
    computation, storage, or caching."""
    def __init__(
        self,
        task_type: TaskType,
        query: ValidatedPredictiveQuery,
        graph: Graph,
    ):
        self.task_type = task_type
        self.query = query
        self.graph = graph
        self._train_table_stats = None

    def write_to_cache(
        self,
        job_id: str,
        stat_dict: Dict[str, Stats],
    ):
        r"""Write the train table stats to backend cache.

        Args:
            job_id (str): Job id corresponding to table table stats.
            stat_dict (Dict[str, Stats]): The dictionary storing train table
                stats.
        """
        get_active_cache().pquery_table_stats_cache.update(job_id, stat_dict)

    def load_from_cache(
        self,
        job_id: str,
        stats_list: List[str] | None = None,
    ) -> Dict[str, Stats]:
        r"""Load the table stats from backend cache.

        Args:
            job_id (str): Job id corresponding to table table stats.
            stats_list (List[str] | None): The list of stats to load from
                cache.

        Returns:
            Dict[str, Stats]: A dictionary mapping stat names to their
                corresponding Stats objects loaded from the cache.
        """
        stats = get_active_cache().pquery_table_stats_cache.get_stats(
            job_id, stats_list)
        return stats

    def _expected_train_stats_keys_for_mode(self) -> list[str]:
        if self.query.query_type == QueryType.TEMPORAL:
            if self.task_type.is_node_pred:
                return list(ALL_TEMPORAL_NODE_STATS)
            if self.task_type == TaskType.TEMPORAL_LINK_PREDICTION:
                return list(ALL_TEMPORAL_LINK_STATS)
        else:
            if self.task_type.is_node_pred:
                return list(ALL_STATIC_NODE_STATS)
            if self.task_type == TaskType.STATIC_LINK_PREDICTION:
                return list(ALL_STATIC_LINK_STATS)
        raise NotImplementedError(
            f'Unsupported task/query combination for stats keys: '
            f'task_type={self.task_type}, query_type={self.query.query_type}')

    def seed_train_stats_cache_from_native_artifact(
        self,
        job_id: str,
        parsed_stats: dict[str, Any],
    ) -> list[str]:
        """Seed train-table stats cache directly from diskgraph artifact.

        Returns missing expected cache keys for the current task/query mode.
        Caller should fail fast in strict native mode when this list is non-empty.
        """
        if not isinstance(parsed_stats, dict):
            raise RuntimeError('parsed_stats must be a dict')

        if self.query.query_type == QueryType.TEMPORAL:
            if self.task_type.is_node_pred:
                table_type = TrainTableType.TEMPORAL_NODE_PRED
            elif self.task_type == TaskType.TEMPORAL_LINK_PREDICTION:
                table_type = TrainTableType.TEMPORAL_LINK_PRED
            else:
                raise NotImplementedError
        else:
            if self.task_type.is_node_pred:
                table_type = TrainTableType.STATIC_NODE_PRED
            elif self.task_type == TaskType.STATIC_LINK_PREDICTION:
                table_type = TrainTableType.STATIC_LINK_PRED
            else:
                raise NotImplementedError

        expected_keys = self._expected_train_stats_keys_for_mode()

        table_stats: Dict[str, Stats] = {
            STATS_TYPE: Stats(table_type, approx=False),
        }
        for key in expected_keys:
            if key in parsed_stats:
                value = parsed_stats[key]
                if key in {MIN_DATE, MAX_DATE, ANCHOR_DATE}:
                    value = tt_artifacts.normalize_timestamp_like_value(value)
                table_stats[key] = Stats(value, approx=False)
            elif key in _OPTIONAL_NATIVE_TRAIN_STATS:
                # FK-free LP resolution-mode native artifacts do not emit the
                # per-timeframe target / link counts. Seed them to None so the
                # downstream progress-stats consumer does not KeyError. Native
                # v2 artifacts omit per-timeframe diagnostics because pass-B
                # stage tables and split_info are authoritative.
                table_stats[key] = Stats(None, approx=False)

        self.write_to_cache(job_id, table_stats)

        present = self.load_from_cache(job_id, expected_keys)
        missing = sorted(set(expected_keys) - set(present.keys()))
        return missing

    def update_min_max_anchor_date_cache(
        self,
        job_id: str,
        min_date: pd.Timestamp,
        max_date: pd.Timestamp,
        anchor_date: pd.Timestamp,
    ):
        r"""Stores information about the minimum and maximum date to the
        stats cache."""
        date_stats = {
            MIN_DATE: Stats(min_date, approx=False),
            MAX_DATE: Stats(max_date, approx=False),
            ANCHOR_DATE: Stats(anchor_date, approx=False),
        }
        self.write_to_cache(job_id, date_stats)

    def update_train_table_stats(self, job_id: str,
                                 train_table_stats: TemporalTrainTableStats,
                                 boundaries: List[Tuple[pd.Timestamp,
                                                        pd.Timestamp]]):
        r"""Update timeframe stats in the stats cache.
        Args:
            job_id (str): Unique operation id associated with the
                training table generation process.
            train_table_stats (TemporalTrainTableStats): Temporal train table
                stats.
            boundaries (List[Tuple[pd.Timestamp, pd.Timestamp]]): The split
                boundaries used for timeframe stats update.
        """
        # NOTE: Right now the train table stats update logic is only supported
        # for `ApproxDateOffsetSplit` in PQLv2. Other temporal split methods
        # do not need train table stats update because their train table
        # timeframes are selected for train/val/test set before train table
        # generation in PQLv2.
        timeframe_stats = train_table_stats.stats_per_timeframe
        assert isinstance(timeframe_stats, list)
        anchor_time_list = [
            timeframe_stat.min_date for timeframe_stat in timeframe_stats
        ]
        assert anchor_time_list == sorted(anchor_time_list, reverse=True)

        timeframe_stats_updated = []
        boundary_index_curr = len(boundaries) - 1
        for timeframe_stat in timeframe_stats:
            # Normalize all timestamps to UTC for consistent comparison
            min_date = normalize_timestamp_to_utc_local(
                pd.Timestamp(timeframe_stat.min_date))
            max_date = normalize_timestamp_to_utc_local(
                pd.Timestamp(timeframe_stat.max_date))
            boundary_start = normalize_timestamp_to_utc_local(
                boundaries[boundary_index_curr][0])
            boundary_end = normalize_timestamp_to_utc_local(
                boundaries[boundary_index_curr][1])

            # Move split boundary once is enough because for the split method
            # this logic is applied to, `ApproxDateOffsetSplit`, train/val/test
            # set all contains at least one train table timeframe, o.w. a
            # validation error will be raised earlier in split table generation
            # logic.
            if min_date < boundary_start:
                boundary_index_curr -= 1
                # Update boundary references after index change and normalize to UTC
                boundary_start = normalize_timestamp_to_utc_local(
                    boundaries[boundary_index_curr][0])
                boundary_end = normalize_timestamp_to_utc_local(
                    boundaries[boundary_index_curr][1])
            # The assertion should be `True` at all times sicne each split set
            # contains at least one train table timeframe.
            assert min_date >= boundary_start
            if max_date <= boundary_end:
                timeframe_stats_updated.append(timeframe_stat)

        num_timeframes_post_filter_updated = len(timeframe_stats_updated)
        num_filtered_timeframes = (
            train_table_stats.num_timeframes_post_filter -
            len(timeframe_stats_updated))
        num_timeframes_updated = (train_table_stats.num_timeframes -
                                  num_filtered_timeframes)

        # TODO: Adjust the other train table stats, e.g. num_entities,
        # num_train_rows, etc. once we decide to show them in the UI.
        # Update train table stats stored in the cache
        train_table_stats_updated = {
            STATS_PER_TIMEFRAME:
            Stats(timeframe_stats_updated, approx=False),
            NUM_TIMEFRAMES:
            Stats(num_timeframes_updated, approx=False),
            NUM_TIMEFRAMES_POST_FILTER:
            Stats(num_timeframes_post_filter_updated, approx=False),
        }
        self.write_to_cache(job_id, train_table_stats_updated)

        # Update train table stats stored in training table object in the
        # current session
        train_table_stats.stats_per_timeframe = timeframe_stats_updated
        train_table_stats.num_timeframes_post_filter = (
            num_timeframes_post_filter_updated)
        train_table_stats.num_timeframes = num_timeframes_updated

    def update_pred_time(
        self,
        job_id: str,
        pred_time: pd.Timestamp,
    ):
        r"""Stores information about the prediction time to the
        stats cache."""
        date_stats = {
            PRED_TIME: Stats(pred_time, approx=False),
        }
        self.write_to_cache(job_id, date_stats)

    def update_pred_num_entities(
        self,
        job_id: str,
        num_entities: int,
    ):
        r"""Stores the number of entities in a prediction table."""
        entity_stats = {NUM_ENTITIES: Stats(num_entities, approx=False)}
        self.write_to_cache(job_id, entity_stats)

    def get_batch_pred_table_stats_config(
            self, job_id: str) -> Optional[Dict[str, StatsConfig]]:
        r"""Retrieve the batch prediction table statistics configuration from
        cache. This method loads prediction table statistics from the cache
        and converts them into a configuration format. It is typically used to
        get the prediction anchor time and entity-count configuration for a
        batch prediction job.

        Args:
            job_id (str): The ID of the batch prediction job to retrieve
                stats for.

        Returns:
            Optional[Dict[str, StatsConfig]]: A dictionary mapping stat names
                to their configurations. Returns None if no stats are found in
                the cache.
        """
        stats = self.load_from_cache(job_id, [PRED_TIME, NUM_ENTITIES])
        if len(stats) == 0:
            return None
        return {name: stat.config() for name, stat in stats.items()}

    @blocking_method()
    def get_train_table_stats(self, job_id: Optional[str] = None) -> Any:
        r"""Gets important statistics of the generated
        :class:`~kumo.task.TrainingTable` in order to rate the quality of a
        :class:`PredictiveQuery`.
        Args:
            job_id: job id to retrieve the training table stats.
        """
        if self._train_table_stats is not None:
            return self._train_table_stats

        assert job_id is not None
        self._train_table_stats = load_train_table_stats_from_cache(job_id)
        return self._train_table_stats
