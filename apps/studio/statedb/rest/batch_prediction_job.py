# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from datetime import datetime
from typing import List, Optional

from kumo_core.types import TaskType
from kumo_core.types.schema import FPID
from kumoapi.jobs import PredictionTableSource
from pydantic import model_validator

from io_utils import kumo_dataclass, logging
from statedb.config.common import (
    GeneratePredictionTableJobID,
    GraphID,
    PredictiveQueryID,
    TrainingJobID,
)
from statedb.config.jobs import (
    PredictionOutputConfig,
    validate_prediction_outputs,
)
from statedb.config.predictive_query import FilterOverrideArgs

log = logging.getLogger(__name__)


@kumo_dataclass
class BatchPredictionJobConfig:
    """Captures persisted user options for initiating a batch prediction job."""
    predictive_query_id: PredictiveQueryID
    training_job_id: TrainingJobID
    task_type: TaskType
    num_classes_to_return: Optional[int] = None

    # Present if the predictive query applies a sigmoid threshold to outputs.
    binary_classification_threshold: Optional[float] = None

    # PredictionTableGenerationPlan options
    prediction_time: Optional[datetime] = None
    lag_timesteps: Optional[int] = None
    year_over_year: bool = False
    forecast_length: Optional[int] = None

    filter_override_args: Optional[FilterOverrideArgs] = None
    transformation: Optional[str] = None

    outputs: List[PredictionOutputConfig] = field(default_factory=list)

    # No.of workers/parallelism to use when generating batch predictions. When
    # set to >1, the prediction table is partitioned into smaller parts and
    # processed in parallel.
    # The name has "_limit" suffix for backward compatibility.
    num_parallel_workers_limit: int = 1

    # Exactly one generated job or connector-backed custom object-store source
    # is used.
    generate_pred_table_job_id: Optional[GeneratePredictionTableJobID] = None
    custom_pred_table_source: Optional[PredictionTableSource] = None

    # Graph ID (v1 or v2), optional(None) for backward compatibility. New BP
    # jobs should always record the graph_id.
    graph_id: Optional[GraphID] = None
    # Record which graph data version this bp job was executing over.
    graph_data_version: Optional[str] = None

    # Key of an immutable `PQueryDefinition` frozen at the time of job launch.
    pquery_def_fpid: Optional[FPID] = None

    # Whether to enable XAI: Default is True
    explanations: bool = False

    @model_validator(mode='after')
    def _check_invariants(self) -> 'BatchPredictionJobConfig':
        task_type: Optional[TaskType] = self.task_type
        binary_classification_threshold = self.binary_classification_threshold
        validate_prediction_outputs(self.outputs)

        thresholded_task_types = {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        }

        if (task_type in thresholded_task_types
                and binary_classification_threshold is None):
            self.binary_classification_threshold = 0.5
            binary_classification_threshold = 0.5
            log.warning("`binary_classification_threshold` is not set "
                        "for a thresholded classification task. "
                        "Use 0.5 as the default value.")

        if (task_type in thresholded_task_types
                and binary_classification_threshold is not None
                and not 0 <= binary_classification_threshold <= 1):
            raise ValueError(
                '`binary_classification_threshold` must be between 0 and 1.')

        if (task_type not in thresholded_task_types
                and binary_classification_threshold is not None):
            raise ValueError(
                "`binary_classification_threshold` may not be present "
                "for non-thresholded task types.")

        if (self.generate_pred_table_job_id
                and self.custom_pred_table_source is not None):
            raise ValueError(
                '"custom_pred_table_source" and '
                '"generate_pred_table_job_id" cannot be set at the same time '
                'in BatchPredictionJobConfig.')

        return self
