# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
from dataclasses import field
from typing import List, Optional

from kumo_core.types import TaskType
from kumo_core.types.schema import FPID
from kumoapi.model_plan import RunMode
from kumoapi.pquery import QueryType
from pydantic import field_validator

from io_utils import kumo_dataclass
from rest.config.graph_v2 import GraphDefinition
from rest.config.prediction_output import PredictionOutputConfig
from rest.config.table_v2 import TableDefinition
from statedb.config.common import (
    BatchPredictionWorkflowID,
    JobSource,
    PredictiveQueryID,
)
from statedb.config.predictive_query import FilterOverrideArgs


@kumo_dataclass
class FixturePQueryRecord:
    """
    Predictive-query record embedded in a fixture. Carries the per-query
    parameters needed to recreate a query when a fixture is loaded: the
    query string, the parent graph (referenced by FPID), display name,
    and task/query/run-mode hints plus optional model-planner overrides.
    """

    query_str: str
    graph_fpid: FPID
    pquery_name: str
    description: str = ''
    model_planner_yaml: str = ''
    run_mode: Optional[RunMode] = None
    task_type: Optional[TaskType] = None
    query_type: Optional[QueryType] = None
    list_distinct_target_is_array: Optional[bool] = None
    lp_bp_layout: Optional[str] = None

    @field_validator('graph_fpid')
    @classmethod
    def _validate_graph_fpid(cls, v: str) -> FPID:
        return FPID(v)


@kumo_dataclass
class FixtureBatchPredictionRecord:
    """
    Batch-prediction record embedded in a fixture. Carries the parameters
    needed to recreate a batch-prediction job when a fixture is loaded.
    """

    predictive_query_id: PredictiveQueryID
    batch_prediction_workflow_id: Optional[BatchPredictionWorkflowID] = None
    # If unset, the batch-prediction job reuses the training query under
    # `predictive_query_id`; if set, the filters override that query.
    filter_override_args: Optional[FilterOverrideArgs] = None
    transformation: Optional[str] = None
    outputs: List[PredictionOutputConfig] = field(default_factory=list)
    binary_classification_threshold: Optional[float] = 0.5
    # Top-K classes per entity; ignored for regression and forecasting tasks.
    num_classes_to_return: Optional[int] = None
    prediction_time: Optional[datetime.datetime] = None
    num_parallel_workers: int = 1
    explanations: bool = False
    job_source: Optional[JobSource] = None
    description: str = ''


@kumo_dataclass
class FixtureDefinition:
    tables: List[TableDefinition]
    graphs: List[GraphDefinition]
    queries: List[FixturePQueryRecord]
    predictions: Optional[List[FixtureBatchPredictionRecord]] = None
