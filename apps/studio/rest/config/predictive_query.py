# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

from kumo_core.model_plan import ModelPlan
from kumo_core.types import TaskType
from kumoapi.pquery import QueryType

from io_utils import kumo_dataclass
from io_utils.dataclasses import ArbitraryTypesBaseModel
from kumo.config.predictive_query import PredictiveQueryV2Config
from kumo.config.task import SplitConfig
from kumo.pquery.predictive_query import PredictiveQueryV2
from rest.config.graph_v2 import GraphDefinition
from rest.config.ml_backend import MaterializationTaskInfo
from statedb.config.common import GraphID, PredictiveQueryID
from statedb.config.predictive_query import \
    PQueryDefinition as PQueryDefinition
from statedb.config.predictive_query import \
    PQueryRuntimeOptions as PQueryRuntimeOptions
from statedb.config.predictive_query import QueryDialect as QueryDialect
from statedb.config.predictive_query import \
    compute_pquery_fpid as compute_pquery_fpid
from statedb.config.predictive_query import \
    pquery_runtime_options_or_none as pquery_runtime_options_or_none

__all__ = [
    'PQueryDefinition',
    'PQueryRuntimeOptions',
    'QueryDialect',
    'compute_pquery_fpid',
    'pquery_runtime_options_or_none',
]


@kumo_dataclass
class InferTaskTypeRequest:
    query_string: str
    graph_id: GraphID


@kumo_dataclass
class InsertFilterOverridesRequest:
    entity_filter: str = ''
    target_filter: str = ''


@kumo_dataclass
class MaterializationInfoResponse:
    # Human readable string - for debug purposes only
    status: str

    # Empty if the materialization info is not available. For example, this may
    # happen if the training job has not yet finished the materialization
    # phase.
    info: Optional[MaterializationTaskInfo]


@kumo_dataclass
class AdvancedOptionYaml:
    advanced_options_yaml: str


@kumo_dataclass
class OutputPresignedUrlResponse:
    presigned_url: str


@kumo_dataclass
class ModelPlansResponse:
    model_plans: List[str]
    advanced_options: List[str]


class RESTParsedPredictiveQuery(ArbitraryTypesBaseModel):
    id: PredictiveQueryID
    config: PredictiveQueryV2Config
    predictive_query: PredictiveQueryV2
    auto_trainer_options: ModelPlan
    task_type: TaskType
    query_type: QueryType
    has_train_table_time_col: bool
    split_config: Optional[SplitConfig] = None


@kumo_dataclass
class DefaultModelPlanRequest:
    query_string: str
    # graph_id is deprecated in V2 query and model planner validation
    graph_id: str
    run_mode: str
    has_train_table_weight_col: bool = False
    graph_def: Optional[GraphDefinition] = None


@kumo_dataclass
class DefaultModelPlanInfo:
    model_plan: ModelPlan
    task_type: TaskType
    query_type: QueryType
    has_train_table_weight_col: bool = False
