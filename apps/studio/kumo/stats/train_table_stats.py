# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union

import pandas as pd
from kumo_core.types import TrainTableType
from kumoapi.train import TrainingStage
from typing_extensions import TypedDict

from async_utils.sync import blocking_func
from io_utils import kumo_dataclass, logging
from io_utils.json import load_json
from kumo.config import ConfigMixin
from kumo.data import get_active_cache
from kumo.home import HomeDirManager
from kumo.stats import Stats
from kumo.stats.utils import PandasTimestamp, convert_str_to_timestamp
from monitoring.tracing import TraceOptions

NUM_PRED_ROWS = 'num_pred_rows'
NUM_PRED_ENTITIES = 'num_pred_entities'
NUM_PRED_TARGETS = 'num_pred_targets'
NUM_TARGET_ROWS = 'num_target_rows'
NUM_TOTAL_ROWS = 'num_total_rows'
MAX_DATE = 'max_date'
MIN_DATE = 'min_date'
ANCHOR_DATE = 'anchor_date'
STATS_TYPE = 'stats_type'
AVG_ENTITIES_PER_TIMEFRAME = 'avg_entities_per_timeframe'
NUM_TRAIN_ROWS = 'num_train_rows'
NUM_TARGETS = 'num_targets'
NUM_ENTITIES = 'num_entities'
NUM_LINKS = 'num_links'
NUM_TIMEFRAMES = 'num_timeframes'
NUM_TIMEFRAMES_POST_FILTER = 'num_timeframes_post_filter'
TARGET_DISTRIBUTION = 'target_distribution'
STATS_PER_TIMEFRAME = 'stats_per_timeframe'
SCORING_TABLE_COLUMN_STATS = 'scoring_table_column_stats'
AVG_TARGETS_PER_TIMEFRAME = 'avg_targets_per_timeframe'
NODE_DEGREE_STATS = 'node_degree_stats'
NUM_EXPECTED_ITERATIONS = 'num_expected_iterations'
ITERATION_DATA = 'iteration_data'

PRED_TIME = 'pred_time'

TARGET_COUNT = 'target_count'
ENTITY_COUNT = 'entity_count'
ROW_COUNT = 'row_count'
LINK_COUNT = 'link_count'
ANCHOR_DATE = 'anchor_date'
COUNT = 'count'
TARGET_PERCENTAGE = 'target_percentage'
AGGREGATED = 'aggregated'
MIN_DATE = 'min_date'
MAX_DATE = 'max_date'
MIN = '__numerical_target_MIN_stat__'
MAX = '__numerical_target_MAX_stat__'
MEAN = '__numerical_target_MEAN_stat__'
MODE = '__numerical_target_MODE_stat__'
PERCENTILES = '__numerical_target_PERCENTILES_stat__'
PERCENTILE_25 = '__numerical_target_PERCENTILE_25_stat__'
MEDIAN = '__numerical_target_MEDIAN_stat__'
PERCENTILE_75 = '__numerical_target_PERCENTILE_75_stat__'
MODE_COUNT = '__numerical_target_MODE_COUNT_stat__'
MODE_PERCENTAGE = '__numerical_target_MODE_PERCENTAGE_stat__'
MODE = '__numerical_target_MODE_stat__'

GROUP_ROW_COUNT = 'group_row_count'
TARGET_FRACTION = 'TARGET_FRACTION'

TARGET = 'TARGET'
ENTITY = 'ENTITY'
TIMESTAMP = 'TIMESTAMP'

ALL_STATS = [
    NUM_ENTITIES,
    NUM_TARGETS,
    NUM_TRAIN_ROWS,
    STATS_TYPE,
]
ALL_TEMPORAL_STATS = ALL_STATS + [
    MIN_DATE,
    MAX_DATE,
    ANCHOR_DATE,
    NUM_TIMEFRAMES,
    NUM_TIMEFRAMES_POST_FILTER,
    AVG_ENTITIES_PER_TIMEFRAME,
    STATS_PER_TIMEFRAME,
    NODE_DEGREE_STATS,
]

ALL_TEMPORAL_NODE_STATS = ALL_TEMPORAL_STATS + [
    NUM_TARGET_ROWS,
    TARGET_DISTRIBUTION,
    SCORING_TABLE_COLUMN_STATS,
]

ALL_TEMPORAL_LINK_STATS = ALL_TEMPORAL_STATS + [
    AVG_TARGETS_PER_TIMEFRAME, NUM_LINKS
]

ALL_STATIC_LINK_STATS = ALL_STATS + [NUM_LINKS]

ALL_STATIC_NODE_STATS = ALL_STATS + [
    NUM_TOTAL_ROWS,
    NUM_TARGET_ROWS,
    TARGET_DISTRIBUTION,
]

logger = logging.getLogger(__name__)


@dataclass
class TrainTableStats(ConfigMixin, ABC):
    num_entities: int
    num_targets: int
    num_train_rows: int
    stats_type: TrainTableType

    @classmethod
    @abstractmethod
    def from_cache(cls, job_id: str):
        pass


@dataclass(repr=False)
class NodeDegreeStats:
    r"""Node degree statistics for train table."""
    # Average entity node degree
    avg_entity_degree: float
    # Minimum entity node degree
    min_entity_degree: int
    # Maximum entity node degree
    max_entity_degree: int
    # Average target node degree
    avg_target_degree: Optional[float]
    # Minimum target node degree
    min_target_degree: Optional[int]
    # Maximum target node degree
    max_target_degree: Optional[int]

    def __repr__(self) -> str:
        avg_entity_degree = round(self.avg_entity_degree, 2)
        avg_target_degree = None
        if self.avg_target_degree is not None:
            avg_target_degree = round(self.avg_target_degree, 2)
        return (f"{self.__class__.__name__}(\n"
                f"  avg_entity_degree={avg_entity_degree},\n"
                f"  min_entity_degree={self.min_entity_degree},\n"
                f"  max_entity_degree={self.max_entity_degree},\n"
                f"  avg_target_degree={avg_target_degree},\n"
                f"  min_target_degree={self.min_target_degree},\n"
                f"  max_target_degree={self.max_target_degree},\n"
                ")")


CategoricalDictTargetDistribution = TypedDict(
    'CategoricalDictTargetDistribution', {
        'count': Dict[str, int],
        'percentage': Dict[str, float]
    })


@kumo_dataclass
class CategoricalTargetDistribution:
    count: Dict[str, int]
    percentage: Dict[str, float]

    def as_dict(self) -> CategoricalDictTargetDistribution:
        return asdict(self)  # type: ignore


TargetHistogramDictType = TypedDict('TargetHistogramDictType', {
    'bins': List[float],
    'counts': List[int],
    'bin_width': float,
    'num_bins': int
})

# Dict version of TargetHistogramAndStats
NumericalDictTargetDistribution = TypedDict(
    'NumericalDictTargetDistribution', {
        'target_mean': float,
        'target_min': float,
        'target_max': float,
        'target_percentile_25': float,
        'target_percentile_75': float,
        'target_median': float,
        'target_mode': float,
        'target_mode_percentage': float,
        'target_mode_count': int,
        'histogram': TargetHistogramDictType,
    })


@dataclass(repr=False)
class TemporalNodeTimeFrameStats:
    # Minimum date in the timeframe. ``PandasTimestamp`` (not bare
    # ``pd.Timestamp``) so that when this dataclass is validated as an element
    # of ``TrainTableGenerationProgress.iteration_data`` (a pydantic v2
    # dataclass), string dates loaded from the job-dir JSON are coerced to
    # ``pd.Timestamp`` instead of failing strict ``is_instance_of`` validation.
    min_date: PandasTimestamp
    # Maximum date in the timeframe
    max_date: PandasTimestamp
    # Anchor date in the timeframe
    anchor_date: PandasTimestamp
    # Distribution of target column values
    target_distribution: Mapping[str, Any]
    # The order of this timeframe regarding all the timeframes
    # in train table stats
    index: Optional[int] = None

    num_train_rows: int = 0
    num_entities: int = 0
    num_targets: int = 0

    def __post_init__(self):
        convert_str_to_timestamp(self, ['min_date', 'max_date', 'anchor_date'])

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(\n"
                f"  min_date={self.min_date},\n"
                f"  max_date={self.max_date},\n"
                f"  anchor_date={self.anchor_date},\n"
                f"  num_train_rows={self.num_train_rows},\n"
                f"  num_entities={self.num_entities},\n"
                f"  num_targets={self.num_targets},\n"
                f"  index={self.index},\n"
                f"  target_distribution={self.target_distribution},\n"
                ")")


@dataclass(repr=False)
class TemporalLinkTimeFrameStats:
    r"""Train table stats for each timeframe."""
    # Minimum date in the timeframe. ``PandasTimestamp`` (not bare
    # ``pd.Timestamp``) so that when this dataclass is validated as an element
    # of ``TrainTableGenerationProgress.iteration_data`` (a pydantic v2
    # dataclass), string dates loaded from the job-dir JSON are coerced to
    # ``pd.Timestamp`` instead of failing strict ``is_instance_of`` validation.
    min_date: PandasTimestamp
    # Maximum date in the timeframe
    max_date: PandasTimestamp
    # Anchor date in the timeframe
    anchor_date: PandasTimestamp
    # Number of rows in this timeframe
    num_train_rows: int
    # Number of unique entities
    num_entities: int
    # Number of unique targets
    num_targets: int
    # Combined length of all target arrays in the timeframe
    num_links: Optional[int] = None
    # The order of this timeframe regarding all the timeframes
    # in train table stats
    index: Optional[int] = None
    # Node degree stats for the train table in this timeframe
    node_degree_stats: Optional[NodeDegreeStats] = None

    def __post_init__(self):
        convert_str_to_timestamp(self, ['min_date', 'max_date', 'anchor_date'])

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(\n"
                f"  min_date={self.min_date},\n"
                f"  max_date={self.max_date},\n"
                f"  anchor_date={self.anchor_date},\n"
                f"  num_train_rows={self.num_train_rows},\n"
                f"  num_entities={self.num_entities},\n"
                f"  num_targets={self.num_targets},\n"
                f"  num_links={self.num_links},\n"
                f"  index={self.index},\n"
                f"  node_degree_stats={self.node_degree_stats},\n"
                ")")


IterationStats = Union[TemporalNodeTimeFrameStats, TemporalLinkTimeFrameStats]


@dataclass(kw_only=True)
class TemporalTrainTableStats(TrainTableStats):
    min_date: pd.Timestamp
    max_date: pd.Timestamp
    anchor_date: pd.Timestamp
    num_timeframes: int
    num_timeframes_post_filter: int
    avg_entities_per_timeframe: int
    node_degree_stats: Optional[NodeDegreeStats] = None
    stats_per_timeframe: List[TemporalNodeTimeFrameStats] | List[
        TemporalLinkTimeFrameStats] | None = field(default_factory=list)

    def __post_init__(self):
        convert_str_to_timestamp(self, ['min_date', 'max_date', 'anchor_date'])


@dataclass(repr=False, kw_only=True)
class TemporalNodeTrainTableStats(TemporalTrainTableStats):
    num_target_rows: Optional[int] = 0
    target_distribution: Optional[Dict[str, Any]] = field(default_factory=dict)
    scoring_table_column_stats: Optional[Dict[str, Dict[str, float]]] = field(
        default_factory=dict)
    stats_type: TrainTableType = TrainTableType.TEMPORAL_NODE_PRED

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  min_date={self.min_date},\n"
            f"  max_date={self.max_date},\n"
            f"  anchor_date={self.anchor_date},\n"
            f"  num_entities={self.num_entities},\n"
            f"  num_target_rows={self.num_target_rows},\n"
            f"  num_timeframes={self.num_timeframes},\n"
            f"  num_timeframes_post_filter={self.num_timeframes_post_filter},\n"  # noqa  E501
            f"  avg_entities_per_timeframe={self.avg_entities_per_timeframe},\n"  # noqa  E501
            f"  num_train_rows={self.num_train_rows},\n"
            f"  num_targets={self.num_targets},\n"
            f"  target_distribution={self.target_distribution},\n"
            f"  stats_per_timeframe={self.stats_per_timeframe},\n"
            f"  node_degree_stats={self.node_degree_stats},\n"
            f"  scoring_table_column_stats="
            f"{self.scoring_table_column_stats},\n"
            f"  stats_type={self.stats_type},\n"
            ")")

    @classmethod
    def from_cache(cls, job_id: str) -> 'TemporalNodeTrainTableStats':
        from kumo.data.backend_cache import BackendCacheManager
        cache: BackendCacheManager = get_active_cache()
        training_table_stats = cache.pquery_table_stats_cache.get_stats(
            job_id, ALL_TEMPORAL_NODE_STATS)

        node_degree_stats = None
        stats_per_timeframe = None
        if training_table_stats[NODE_DEGREE_STATS].value is not None:
            node_degree_stats = NodeDegreeStats(
                **training_table_stats[NODE_DEGREE_STATS].value)
        if training_table_stats[STATS_PER_TIMEFRAME].value is not None:
            stats_per_timeframe = [
                TemporalNodeTimeFrameStats(**stat)
                for stat in training_table_stats[STATS_PER_TIMEFRAME].value
            ]

        return cls(
            min_date=training_table_stats[MIN_DATE].value,
            max_date=training_table_stats[MAX_DATE].value,
            anchor_date=training_table_stats[ANCHOR_DATE].value,
            num_entities=training_table_stats[NUM_ENTITIES].value,
            num_target_rows=training_table_stats[NUM_TARGET_ROWS].value,
            num_timeframes=training_table_stats[NUM_TIMEFRAMES].value,
            num_timeframes_post_filter=(
                training_table_stats[NUM_TIMEFRAMES_POST_FILTER].value),
            avg_entities_per_timeframe=(
                training_table_stats[AVG_ENTITIES_PER_TIMEFRAME].value),
            num_train_rows=training_table_stats[NUM_TRAIN_ROWS].value,
            num_targets=training_table_stats[NUM_TARGETS].value,
            target_distribution=training_table_stats[TARGET_DISTRIBUTION].value,
            node_degree_stats=node_degree_stats,
            stats_per_timeframe=stats_per_timeframe,
            scoring_table_column_stats=training_table_stats[
                SCORING_TABLE_COLUMN_STATS].value,
        )

    @classmethod
    def from_stats(
        cls,
        training_table_stats: Dict[str,
                                   Stats]) -> 'TemporalNodeTrainTableStats':
        node_degree_stats = None
        stats_per_timeframe = None
        if training_table_stats[NODE_DEGREE_STATS].value is not None:
            node_degree_stats = NodeDegreeStats(
                **training_table_stats[NODE_DEGREE_STATS].value)
        if training_table_stats[STATS_PER_TIMEFRAME].value is not None:
            stats_per_timeframe = [
                TemporalNodeTimeFrameStats(**stat)
                for stat in training_table_stats[STATS_PER_TIMEFRAME].value
            ]

        # TODO: Figure out the reason using keyword argument for initialization
        # does not work here
        return TemporalNodeTrainTableStats(
            min_date=training_table_stats[MIN_DATE].value,
            max_date=training_table_stats[MAX_DATE].value,
            anchor_date=training_table_stats[ANCHOR_DATE].value,
            num_entities=training_table_stats[NUM_ENTITIES].value,
            num_targets=training_table_stats[NUM_TARGETS].value,
            num_train_rows=training_table_stats[NUM_TRAIN_ROWS].value,
            num_timeframes=training_table_stats[NUM_TIMEFRAMES].value,
            num_timeframes_post_filter=training_table_stats[
                NUM_TIMEFRAMES_POST_FILTER].value,
            avg_entities_per_timeframe=training_table_stats[
                AVG_ENTITIES_PER_TIMEFRAME].value,
            num_target_rows=training_table_stats[NUM_TARGET_ROWS].value,
            stats_per_timeframe=stats_per_timeframe,
            target_distribution=training_table_stats[TARGET_DISTRIBUTION].value,
            scoring_table_column_stats=training_table_stats[
                SCORING_TABLE_COLUMN_STATS].value,
            node_degree_stats=node_degree_stats,
        )


@dataclass(repr=False)
class TemporalPredTableStats:
    pred_time: pd.Timestamp
    num_pred_rows: int = 0

    def __post_init__(self):
        convert_str_to_timestamp(self, ['pred_time'])


@dataclass(repr=False, kw_only=True)
class TemporalLinkTrainTableStats(TemporalTrainTableStats):
    avg_targets_per_timeframe: int = 0
    num_links: Optional[int] = None
    stats_type: TrainTableType = TrainTableType.TEMPORAL_LINK_PRED

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  min_date={self.min_date},\n"
            f"  max_date={self.max_date},\n"
            f"  anchor_date={self.anchor_date},\n"
            f"  num_train_rows={self.num_train_rows},\n"
            f"  num_entities={self.num_entities},\n"
            f"  num_targets={self.num_targets},\n"
            f"  num_timeframes={self.num_timeframes},\n"
            f"  num_timeframes_post_filter={self.num_timeframes_post_filter},\n"  # noqa  E501
            f"  avg_entities_per_timeframe={self.avg_entities_per_timeframe},\n"  # noqa  E501
            f"  avg_targets_per_timeframe={self.avg_targets_per_timeframe},\n"
            f"  num_links={self.num_links},\n"
            f"  node_degree_stats={self.node_degree_stats},\n"
            f"  stats_per_timeframe={self.stats_per_timeframe},\n"
            f"  stats_type={self.stats_type},\n"
            ")")

    @classmethod
    def from_cache(cls, job_id: str) -> 'TemporalLinkTrainTableStats':
        from kumo.data.backend_cache import BackendCacheManager
        cache: BackendCacheManager = get_active_cache()
        training_table_stats = cache.pquery_table_stats_cache.get_stats(
            job_id, ALL_TEMPORAL_LINK_STATS)

        node_degree_stats = None
        stats_per_timeframe = None
        if training_table_stats[NODE_DEGREE_STATS].value is not None:
            node_degree_stats = NodeDegreeStats(
                **training_table_stats[NODE_DEGREE_STATS].value)
        if training_table_stats[STATS_PER_TIMEFRAME].value is not None:
            stats_per_timeframe = [
                TemporalLinkTimeFrameStats(**stat)
                for stat in training_table_stats[STATS_PER_TIMEFRAME].value
            ]

        from kumo_core.training.options import allow_array_targets
        return cls(
            min_date=training_table_stats[MIN_DATE].value,
            max_date=training_table_stats[MAX_DATE].value,
            anchor_date=training_table_stats[ANCHOR_DATE].value,
            num_train_rows=training_table_stats[NUM_TRAIN_ROWS].value,
            num_entities=training_table_stats[NUM_ENTITIES].value,
            num_targets=training_table_stats[NUM_TARGETS].value,
            num_timeframes=training_table_stats[NUM_TIMEFRAMES].value,
            num_timeframes_post_filter=training_table_stats[
                NUM_TIMEFRAMES_POST_FILTER].value,
            avg_entities_per_timeframe=(
                training_table_stats[AVG_ENTITIES_PER_TIMEFRAME].value),
            avg_targets_per_timeframe=training_table_stats[
                AVG_TARGETS_PER_TIMEFRAME].value,
            num_links=(training_table_stats[NUM_LINKS].value
                       if allow_array_targets() else None),
            node_degree_stats=node_degree_stats,
            stats_per_timeframe=stats_per_timeframe,
        )

    @classmethod
    def from_stats(
        cls,
        training_table_stats: Dict[str,
                                   Stats]) -> 'TemporalLinkTrainTableStats':
        node_degree_stats = None
        stats_per_timeframe = None
        if training_table_stats[NODE_DEGREE_STATS].value is not None:
            node_degree_stats = NodeDegreeStats(
                **training_table_stats[NODE_DEGREE_STATS].value)
        if training_table_stats[STATS_PER_TIMEFRAME].value is not None:
            stats_per_timeframe = [
                TemporalLinkTimeFrameStats(**stat)
                for stat in training_table_stats[STATS_PER_TIMEFRAME].value
            ]

        from kumo_core.training.options import allow_array_targets
        return cls(
            min_date=training_table_stats[MIN_DATE].value,
            max_date=training_table_stats[MAX_DATE].value,
            anchor_date=training_table_stats[ANCHOR_DATE].value,
            num_timeframes=training_table_stats[NUM_TIMEFRAMES].value,
            num_timeframes_post_filter=training_table_stats[
                NUM_TIMEFRAMES_POST_FILTER].value,
            num_train_rows=training_table_stats[NUM_TRAIN_ROWS].value,
            num_entities=training_table_stats[NUM_ENTITIES].value,
            num_targets=training_table_stats[NUM_TARGETS].value,
            avg_entities_per_timeframe=training_table_stats[
                AVG_ENTITIES_PER_TIMEFRAME].value,
            avg_targets_per_timeframe=training_table_stats[
                AVG_TARGETS_PER_TIMEFRAME].value,
            num_links=training_table_stats[NUM_LINKS].value
            if allow_array_targets() else None,
            node_degree_stats=node_degree_stats,
            stats_per_timeframe=stats_per_timeframe,
        )


@dataclass(repr=False)
class TemporalLinkPredTableStats:
    r"""Temporal link pred task prediction table statistics."""
    # Time used for prediction
    pred_time: pd.Timestamp
    # Number of total rows
    num_pred_rows: int
    # Number of unique entities
    num_pred_entities: int
    # Number of unique targets
    num_pred_targets: int

    def __post_init__(self):
        convert_str_to_timestamp(self, ['pred_time'])

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(\n"
                f"  pred_time={self.pred_time},\n"
                f"  num_pred_rows={self.num_pred_rows},\n"
                f"  num_pred_entities={self.num_pred_entities},\n"
                f"  num_pred_targets={self.num_pred_targets},\n"
                ")")


@dataclass(repr=False, kw_only=True)
class StaticNodeTrainTableStats(TrainTableStats):
    num_total_rows: Optional[int] = 0
    num_target_rows: Optional[int] = 0
    target_distribution: Optional[Dict[str, Any]] = field(default_factory=dict)
    stats_type: TrainTableType = TrainTableType.STATIC_NODE_PRED

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(\n"
                f"  num_entities={self.num_entities},\n"
                f"  num_target_rows={self.num_target_rows},\n"
                f"  num_total_rows={self.num_total_rows},\n"
                f"  num_train_rows={self.num_train_rows},\n"
                f"  num_targets={self.num_targets},\n"
                f"  target_distribution={self.target_distribution},\n"
                f"  stats_type={self.stats_type},\n"
                ")")

    @classmethod
    def from_cache(cls, job_id: str) -> 'StaticNodeTrainTableStats':
        from kumo.data.backend_cache import BackendCacheManager
        cache: BackendCacheManager = get_active_cache()
        training_table_stats = cache.pquery_table_stats_cache.get_stats(
            job_id, ALL_STATIC_NODE_STATS)
        return cls(
            num_entities=training_table_stats[NUM_ENTITIES].value,
            num_target_rows=training_table_stats[NUM_TARGET_ROWS].value,
            num_total_rows=training_table_stats[NUM_TOTAL_ROWS].value,
            num_train_rows=training_table_stats[NUM_TRAIN_ROWS].value,
            num_targets=training_table_stats[NUM_TARGETS].value,
            target_distribution=training_table_stats[TARGET_DISTRIBUTION].value,
        )

    @classmethod
    def from_stats(
            cls,
            training_table_stats: Dict[str,
                                       Stats]) -> 'StaticNodeTrainTableStats':
        return cls(
            num_entities=training_table_stats[NUM_ENTITIES].value,
            num_target_rows=training_table_stats[NUM_TARGET_ROWS].value,
            num_total_rows=training_table_stats[NUM_TOTAL_ROWS].value,
            num_train_rows=training_table_stats[NUM_TRAIN_ROWS].value,
            num_targets=training_table_stats[NUM_TARGETS].value,
            target_distribution=training_table_stats[TARGET_DISTRIBUTION].value,
        )


@dataclass
class StaticPredTableStats:
    num_pred_rows: int = 0


# TODO: (xinwei, zecheng): Log more stats here e.g. node degree, number of
# total entities, number of total rhs entities etc.


@dataclass(repr=False, kw_only=True)
class StaticLinkTrainTableStats(TrainTableStats):
    num_links: Optional[int] = None
    stats_type: TrainTableType = TrainTableType.STATIC_LINK_PRED

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(\n"
                f"  num_entities={self.num_entities},\n"
                f"  num_targets={self.num_targets},\n"
                f"  num_train_rows={self.num_train_rows},\n"
                f"  num_links={self.num_links},\n"
                f"  stats_type={self.stats_type},\n"
                ")")

    @classmethod
    def from_cache(cls, job_id: str) -> 'StaticLinkTrainTableStats':
        from kumo.data.backend_cache import BackendCacheManager
        cache: BackendCacheManager = get_active_cache()
        training_table_stats = cache.pquery_table_stats_cache.get_stats(
            job_id, ALL_STATIC_LINK_STATS)
        from kumo_core.training.options import allow_array_targets
        return cls(
            num_entities=training_table_stats[NUM_ENTITIES].value,
            num_train_rows=training_table_stats[NUM_TRAIN_ROWS].value,
            num_targets=training_table_stats[NUM_TARGETS].value,
            num_links=training_table_stats[NUM_LINKS].value
            if allow_array_targets() else None,
        )

    @classmethod
    def from_stats(
            cls,
            training_table_stats: Dict[str,
                                       Stats]) -> 'StaticLinkTrainTableStats':
        from kumo_core.training.options import allow_array_targets
        return cls(
            num_entities=training_table_stats[NUM_ENTITIES].value,
            num_targets=training_table_stats[NUM_TARGETS].value,
            num_train_rows=training_table_stats[NUM_TRAIN_ROWS].value,
            num_links=training_table_stats[NUM_LINKS].value
            if allow_array_targets() else None,
        )


@blocking_func()
def load_train_table_stats_from_cache(job_id: str):
    from kumo.data.backend_cache import BackendCacheManager
    cache: BackendCacheManager = get_active_cache()
    train_table_stats_type = cache.pquery_table_stats_cache.get_stats(
        job_id, [STATS_TYPE]).get(STATS_TYPE)
    if train_table_stats_type is None:
        return None
    train_table_stats_type = TrainTableType(train_table_stats_type.value)
    if train_table_stats_type == TrainTableType.STATIC_NODE_PRED:
        training_table_stats = cache.pquery_table_stats_cache.get_stats(
            job_id, ALL_STATIC_NODE_STATS)
        return StaticNodeTrainTableStats.from_stats(training_table_stats)
    elif train_table_stats_type == TrainTableType.STATIC_LINK_PRED:
        training_table_stats = cache.pquery_table_stats_cache.get_stats(
            job_id, ALL_STATIC_LINK_STATS)
        return StaticLinkTrainTableStats.from_stats(training_table_stats)
    elif train_table_stats_type == TrainTableType.TEMPORAL_NODE_PRED:
        training_table_stats = cache.pquery_table_stats_cache.get_stats(
            job_id, ALL_TEMPORAL_NODE_STATS)
        return TemporalNodeTrainTableStats.from_stats(training_table_stats)
    elif train_table_stats_type == TrainTableType.TEMPORAL_LINK_PRED:
        training_table_stats = cache.pquery_table_stats_cache.get_stats(
            job_id, ALL_TEMPORAL_LINK_STATS)
        return TemporalLinkTrainTableStats.from_stats(training_table_stats)

    raise ValueError(
        f"Non-supported train table stats type: '{train_table_stats_type}'")


@kumo_dataclass(repr=False)
class TrainTableGenerationProgress:
    num_expected_iterations: int = 0
    num_finished_iterations: int = 0
    iteration_data: List[TemporalNodeTimeFrameStats] | List[
        TemporalLinkTimeFrameStats] | None = None

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(\n"
                f"  num_finished_iterations={self.num_finished_iterations},\n"
                f"  num_expected_iterations={self.num_expected_iterations},\n"
                f"  iteration_data={self.iteration_data},\n"
                ")")

    @classmethod
    def from_cache(cls,
                   job_id: str) -> Optional['TrainTableGenerationProgress']:
        from kumo.data.backend_cache import BackendCacheManager
        cache: BackendCacheManager = get_active_cache()
        train_table_progress = cache.pquery_table_progress_cache.get_stats(
            job_id)
        if (ITERATION_DATA not in train_table_progress
                or NUM_EXPECTED_ITERATIONS not in train_table_progress):
            return None
        else:
            iteration_data = train_table_progress[ITERATION_DATA].value
            stats_per_timeframe = []
            for timeframe_stats in iteration_data:
                # TODO (vid) make loading of stats not rely on below heuristics
                if TARGET_DISTRIBUTION in timeframe_stats:
                    # saved stats are TemporalNodeTimeFrameStats
                    # TODO (vid) update when implemented for regular stats and
                    # remove type ignore at the cls init

                    # Necessary for backwards compatitbility
                    is_regression = (
                        'histogram' in timeframe_stats[TARGET_DISTRIBUTION]
                        or 'target_max' in timeframe_stats[TARGET_DISTRIBUTION])

                    if 'count' not in timeframe_stats[
                            TARGET_DISTRIBUTION] and not is_regression:
                        total = sum(
                            timeframe_stats[TARGET_DISTRIBUTION].values())
                        timeframe_stats[TARGET_DISTRIBUTION] = {
                            "count": timeframe_stats[TARGET_DISTRIBUTION],
                            "percentage": {
                                k: v / total
                                for k, v in
                                timeframe_stats[TARGET_DISTRIBUTION].items()
                            }
                        }

                    stats_per_timeframe.append(
                        TemporalNodeTimeFrameStats(**timeframe_stats))
                else:
                    # saved stats are TemporalLinkTimeFrameStats
                    stats_per_timeframe.append(
                        TemporalLinkTimeFrameStats(**timeframe_stats))
            return cls(
                num_expected_iterations=train_table_progress[
                    NUM_EXPECTED_ITERATIONS].value,
                num_finished_iterations=len(iteration_data),
                iteration_data=stats_per_timeframe,  # type: ignore
            )


@blocking_func()
def train_table_generation_stats_from_cache_v2(
        job_id: str) -> Optional['TrainTableGenerationProgress']:
    r"""Load train table generation stats from backend cache in PQLv2 setup.

    Args:
        job_id (str): The job id corresponds to the train table stats.
    Returns
        TrainTableGenerationProgress if train table stats is available
            in backend cache and that the predictive query problem type
            is temporal, None otherwise.
    """
    train_table_stats = load_train_table_stats_from_cache(job_id)
    if (train_table_stats is not None and isinstance(
            train_table_stats,
            TemporalNodeTrainTableStats | TemporalLinkTrainTableStats)):
        iteration_data = train_table_stats.stats_per_timeframe
        progress = TrainTableGenerationProgress(
            num_expected_iterations=train_table_stats.num_timeframes,
            num_finished_iterations=train_table_stats.
            num_timeframes_post_filter,
            iteration_data=iteration_data,
        )
        return progress
    return None


# TODO (xinwei): Support link prediction split table stats with a different
# data class


@kumo_dataclass(repr=False)
class SplitTableStats:
    num_train_rows: int
    num_entities: int
    num_targets: Optional[int]
    target_distribution: Optional[Dict[str, Any]]
    min_date: Optional[PandasTimestamp] = None
    max_date: Optional[PandasTimestamp] = None
    anchor_date: Optional[PandasTimestamp] = None

    def __post_init__(self):
        convert_str_to_timestamp(self, ['min_date', 'max_date', 'anchor_date'])

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(\n"
                f"  num_train_rows={self.num_train_rows},\n"
                f"  num_entities={self.num_entities},\n"
                f"  num_targets={self.num_targets},\n"
                f"  target_distribution={self.target_distribution},\n"
                f"  min_date={self.min_date},\n"
                f"  max_date={self.max_date},\n"
                f"  anchor_date={self.anchor_date},\n"
                ")")


# TODO: Change `val_table_stats` to be required after separate the predict
# table statistics with train table statistics.


@kumo_dataclass(repr=False)
class SplitTrainTableStats:
    train_table_stats: SplitTableStats
    val_table_stats: Optional[SplitTableStats] = None
    test_table_stats: Optional[SplitTableStats] = None

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(\n"
                f"  train_table_stats={self.train_table_stats},\n"
                f"  val_table_stats={self.val_table_stats},\n"
                f"  test_table_stats={self.test_table_stats},\n"
                ")")

    @classmethod
    def from_cache(cls, job_id: str) -> Optional['SplitTrainTableStats']:
        from kumo.data.backend_cache import BackendCacheManager
        cache: BackendCacheManager = get_active_cache()
        training_table_stats = \
            cache.training_table_split_stats_cache.get_stats(
                job_id,
                ['train_table_stats', 'val_table_stats', 'test_table_stats'])

        if ('train_table_stats' not in training_table_stats
                and 'val_table_stats' not in training_table_stats
                and 'test_table_stats' not in training_table_stats):
            return None

        # TODO (xinwei): Fix the serialization problem
        train_table_stats = SplitTableStats(
            **training_table_stats['train_table_stats'].value)
        val_table_stats = None
        if training_table_stats['val_table_stats'].value is not None:
            val_table_stats = SplitTableStats(
                **training_table_stats['val_table_stats'].value)
        test_table_stats = None
        if 'test_table_stats' in training_table_stats:
            test_table_stats = SplitTableStats(
                **training_table_stats['test_table_stats'].value)
        return cls(
            train_table_stats=train_table_stats,
            val_table_stats=val_table_stats,
            test_table_stats=test_table_stats,
        )

    def get_stage_table_stats(
            self, stage: TrainingStage) -> Optional[SplitTableStats]:
        split_table_stats = None
        if stage == TrainingStage.TRAIN:
            split_table_stats = self.train_table_stats
        elif stage == TrainingStage.VAL:
            split_table_stats = self.val_table_stats
        elif stage == TrainingStage.TEST:
            split_table_stats = self.test_table_stats
        return split_table_stats


@blocking_func()
def train_table_split_stats_from_cache(
        job_id: str) -> Optional['SplitTrainTableStats']:
    """ Helper function to call from_cache()
    which had to be made blocking. """
    return SplitTrainTableStats.from_cache(job_id)


@blocking_func(tracing_options=TraceOptions(trace_params=True))
def train_table_generation_stats_from_job_dir(
        job_dir: str) -> Optional[TrainTableGenerationProgress]:
    r"""Load train table generation stats from a job directory.

    Args:
        job_dir (str): The full path to the training job directory

    Returns:
        (Optional[TrainTableGenerationProgress]): if train table stats is
            available and the predictive query problem type is temporal,
            None otherwise.
    """
    home_dir_mgr = HomeDirManager(job_dir)
    logger.info(
        f"Loading train table stats from {home_dir_mgr.train_table_stats}")

    # Load stats as dict first
    stats_dict = load_json(home_dir_mgr.train_table_stats, dict, default=None,
                           raise_on_missing=False)
    if stats_dict is None:
        logger.warning(f"No stats found at {home_dir_mgr.train_table_stats}")
        return None

    stats_type = stats_dict.get('stats_type')
    assert stats_type is not None
    stats_type = TrainTableType(stats_type)
    logger.info(f"Found stats_type: {stats_type}")

    # Convert dict to proper type based on stats_type
    if stats_type == TrainTableType.TEMPORAL_NODE_PRED:
        train_table_stats = TemporalNodeTrainTableStats(**stats_dict)
    elif stats_type == TrainTableType.TEMPORAL_LINK_PRED:
        train_table_stats = TemporalLinkTrainTableStats(**stats_dict)
    elif stats_type == TrainTableType.STATIC_NODE_PRED:
        train_table_stats = StaticNodeTrainTableStats(**stats_dict)
    elif stats_type == TrainTableType.STATIC_LINK_PRED:
        train_table_stats = StaticLinkTrainTableStats(**stats_dict)
    else:
        raise ValueError(
            f"Non-supported train table stats type: '{stats_type}'")

    if isinstance(train_table_stats,
                  TemporalNodeTrainTableStats | TemporalLinkTrainTableStats):
        iteration_data = train_table_stats.stats_per_timeframe
        progress = TrainTableGenerationProgress(
            num_expected_iterations=train_table_stats.num_timeframes,
            num_finished_iterations=train_table_stats.
            num_timeframes_post_filter,
            iteration_data=iteration_data,
        )
        logger.info("Successfully created TrainTableGenerationProgress")
        return progress

    return None


@blocking_func(tracing_options=TraceOptions(trace_params=True))
def train_table_split_stats_from_job_dir(
    job_dir: str,
    split_fpid: Optional[str] = None,
) -> Optional[SplitTrainTableStats]:
    """Load split train table stats from a job directory.

    Args:
        job_dir (str): The full path to the training job directory
        split_fpid (Optional[str]): The fpid of the split to load stats for.
            If not provided, the stats will be loaded from the old split
            location.

    Returns:
        (SplitTrainTableStats) if available in the job directory, None
            otherwise.
    """
    home_dir_mgr = HomeDirManager(job_dir)
    path = home_dir_mgr.split_train_table_stats(split_fpid)
    logger.info(f"Loading split train table stats from "
                f"{path}")

    split_stats = load_json(path, SplitTrainTableStats, default=None,
                            raise_on_missing=False)

    if split_stats is None:  # type: ignore
        logger.warning(f"No split train table stats found at "
                       f"{path}")
        return None

    return split_stats
