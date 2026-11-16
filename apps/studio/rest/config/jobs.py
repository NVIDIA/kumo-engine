# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
========================= Training =======================
* Start a training job:
POST /v1/training_jobs {TrainingJobRequest}
response body(200) -> {IDResponse}

* Get training job status:
GET /v1/training_jobs/{training_job_id}
response body(200) -> {TrainingJobResource}

* Cancel a training job:
POST /v1/training_jobs/{training_job_id}/cancel {}
response body(200) -> {CancelTrainingJobResponse}

==================== Batch Prediction =======================
* Start a batch prediction job:
POST /predictive_queries/{pquery_id}/start_prediction {BatchPredictionRequest}
response body(200) -> {BatchPredictionJobResource}

* Get batch prediction job status:
GET /predictive_queries/{pquery_id}/prediction_status
response body(200) -> {{BatchPredictionJobResource}}

* Cancel a batch prediction job:
POST /predictive_queries/{pquery_id}/cancel_prediction {}
reponse body(200) -> {}

"""

from dataclasses import field
from datetime import datetime
from typing import List, Optional

import kumoapi.jobs as api
from kumo_core.types import PredictionArtifactType, TaskType
from pydantic import Field, model_validator

from io_utils import kumo_dataclass
from rest.config.batch_prediction_job import BatchPredictionJobConfig
from rest.config.prediction_output import (
    PredictionOutputConfig,
    validate_prediction_outputs,
)
from statedb.config import jobs as serialized_jobs
from statedb.config.common import (
    BatchPredictionWorkflowID,
    GraphID,
    JobEventLogEntry,
    JobStatus,
    PredictiveQueryID,
    TrainingJobID,
)
from statedb.config.predictive_query import FilterOverrideArgs

MetadataField = serialized_jobs.MetadataField
WriteMode = serialized_jobs.WriteMode


# Generic job status report for both training and batch prediction.
@kumo_dataclass
class JobStatusReport:
    status: JobStatus

    # URL to the training job tracking UI page:
    # http://xxx.kumoai.cloud/queries/trainquery/<encoded tracking id>
    # or
    # https://xxx.kumoai.cloud/predictions/status/<batch pred id>
    # This is useful for human to inspect the training job summary after the
    # job reaches the terminal state.
    tracking_url: str

    # Set when a queue-managed job is admitted to execution.
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None  # Present when status is not RUNNING

    # Generic events or messages for information purpose (logging, debugging).
    event_log: List[JobEventLogEntry] = field(default_factory=list)


Metric = api.Metric

ModelEvaluationMetrics = api.ModelEvaluationMetrics
TrainingJobSummary = api.TrainingJobSummary


@kumo_dataclass
class CreateTrainingJobOptions:  # Config/options to create a training job
    advanced_options_yaml: Optional[str] = Field(default=None, hidden=True)
    model_planner_yaml: Optional[str] = Field(default=None, hidden=True)


@kumo_dataclass
class TrainingJobResource:
    job_id: str

    predictive_query_id: str

    job_status_report: JobStatusReport

    # Present if job status is DONE.
    result: Optional[TrainingJobSummary] = None

    advanced_options_yaml: Optional[str] = Field(default=None, hidden=True)


@kumo_dataclass
class BatchPredictionOptions(api.BatchPredictionOptions):
    # Args to specify new query for prediction (otherwise query from training
    # under predictive_query_id used)
    filter_override_args: Optional[FilterOverrideArgs] = None

    # SQL Transformation to apply to the generated predictions
    # TODO(bp-v1): unused by the v1 path (to_config never reads it); delete
    # this serialized field once all prod releases are past the v1 cutover.
    transformation: Optional[str] = None

    # No.of workers/parallelism to use when generating batch predictions.
    # TODO(bp-v1): legacy duplicate of `num_workers` (the v1 path uses
    # `num_workers`); delete this serialized field once all prod releases are
    # past the v1 cutover.
    num_parallel_workers: int = 1

    # If set, prediction is done at this time by limiting the prediction table
    prediction_time: Optional[datetime] = None


@kumo_dataclass
class BatchPredictionRequest:
    outputs: List[PredictionOutputConfig]

    predict_options: BatchPredictionOptions

    batch_prediction_workflow_id: Optional[BatchPredictionWorkflowID] = None

    explanations: bool = False

    # v1 @root_validator (post-validation) -> @model_validator(mode='after').
    @model_validator(mode='after')
    def _check_invariants(self) -> 'BatchPredictionRequest':
        outputs: List[PredictionOutputConfig] = self.outputs
        validate_prediction_outputs(outputs)
        return self

    @property
    def only_embeddings(self) -> bool:
        no_predictions = PredictionArtifactType.PREDICTIONS not in [
            output.artifact_type for output in self.outputs
        ]
        has_embeddings = PredictionArtifactType.EMBEDDINGS in [
            output.artifact_type for output in self.outputs
        ]
        return no_predictions and has_embeddings

    def to_config(
        self,
        *,
        predictive_query_id: PredictiveQueryID,
        training_job_id: TrainingJobID,
        graph_id: GraphID,
        task_type: TaskType,
        filter_override_args: Optional[FilterOverrideArgs],
        effective_num_classes_to_return: Optional[int],
        effective_threshold: Optional[float],
    ) -> BatchPredictionJobConfig:
        return BatchPredictionJobConfig(
            predictive_query_id=predictive_query_id,
            training_job_id=training_job_id,
            filter_override_args=filter_override_args,
            task_type=task_type,
            num_classes_to_return=(
                effective_num_classes_to_return if task_type
                not in {TaskType.REGRESSION, TaskType.FORECASTING} else None),
            binary_classification_threshold=effective_threshold,
            outputs=self.outputs,
            prediction_time=self.predict_options.prediction_time,
            num_parallel_workers_limit=self.predict_options.num_workers,
            graph_id=graph_id,
            explanations=self.explanations,
        )


@kumo_dataclass
class BatchPredictionJobSummary:
    """Summary of a successful batch prediction job."""
    num_entities_predicted: int
    # TODO(zack): Add more stats #1847
    # TODO(zack): maybe add stats per artifact?


@kumo_dataclass
class BatchPredictionJobResource:
    job_id: str
    predictive_query_id: str
    outputs: List[PredictionOutputConfig]
    predict_options: BatchPredictionOptions

    job_status_report: JobStatusReport

    batch_prediction_workflow_id: Optional[BatchPredictionWorkflowID] = None

    # Present if job status is DONE.
    result: Optional[BatchPredictionJobSummary] = None
