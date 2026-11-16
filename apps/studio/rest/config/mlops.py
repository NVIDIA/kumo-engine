# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from io_utils import kumo_dataclass


class MLOpsStatType(str, Enum):
    PREDICTION_COUNT = 'PREDICTION_COUNT'
    PREDICTION_DISTRIBUTION = 'PREDICTION_DISTRIBUTION'
    TRANSFORMED_PREDICTION_COUNT = 'TRANSFORMED_PREDICTION_COUNT'
    TRANSFORMED_PREDICTION_DISTRIBUTION = 'TRANSFORMED_PREDICTION_DISTRIBUTION'
    PREDICTION_VERSION = 'PREDICTION_VERSION'
    TRAINING_VERSION = 'TRAINING_VERSION'
    # Currently we don't store graph version so this returns a dict of
    # table versions
    GRAPH_VERSION = 'GRAPH_VERSION'
    # dict of table stats for input tables:
    # row count, data size, earliest+latest time
    TABLE_STATS = 'TABLE_STATS'
    START_TIME = 'START_TIME'
    END_TIME = 'END_TIME'

    # for prediction jobs only:
    TRAINED_JOB_ID = 'TRAINED_JOB_ID'
    PREDICTION_OUTPUTS = 'PREDICTION_OUTPUTS'
    # PSI comparing predict to training time data for all cols in job graph
    # - currently just used to compare BP to training time data but could be
    # more general
    COL_PSI_SCORES = 'COL_PSI_SCORES'


@kumo_dataclass
class JobRegistryQuery:
    # for now with just one db table this includes optionally filtering for
    # job_type; in the future may make it a separate and required argument
    job_filters: Dict[str, Union[List[str], str]] = field(default_factory=dict)


@kumo_dataclass
class MLOpsStatsQuery:
    stat_request: List[MLOpsStatType] = field(default_factory=list)


@kumo_dataclass
class MLOpsQueryConfig:
    job_query: JobRegistryQuery
    stat_query: MLOpsStatsQuery


@kumo_dataclass
class MLOpsStats:
    job_status_dict: Dict[str, Optional[str]]
    # mapping of job_ids to stat values returned by MLOPs backend, separated
    # by job type
    bp_stats: Dict[str, Dict[MLOpsStatType, Any]] = field(default_factory=dict)
    train_stats: Dict[str, Dict[MLOpsStatType,
                                Any]] = field(default_factory=dict)


@kumo_dataclass
class BatchPredictionTemplate:
    job_query: JobRegistryQuery
