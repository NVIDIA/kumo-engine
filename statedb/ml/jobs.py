# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional

from kumo_core.types.schema import FPID
from pydantic import model_validator

from io_utils import kumo_dataclass
from rest.config.batch_prediction_job import (
    BatchPredictionJobConfig,
    BatchPredictionJobOutput,
)
from statedb.config.common import (
    BatchPredictionJobID,
    BatchPredictionWorkflowID,
    JobSource,
    JobStatus,
    PredictiveQueryID,
    ProjectID,
    TrainingJobID,
)
from statedb.config.timestamp import ResourceTimestamps


@kumo_dataclass
class MLJobRegistryMetadata:
    instance_id: Optional[str] = None
    pquery_id: Optional[str] = None
    pquery_str: Optional[str] = None
    kumo_version: Optional[str] = None
    generator_advanced_options: Optional[str] = None
    trained_job_id: Optional[str] = None
    bp_filters: Optional[str] = None
    override_model_path: Optional[str] = None


@kumo_dataclass
class BatchPredictionJobDBEntry:
    """The single authoritative StateDB batch-prediction job row."""

    id: BatchPredictionJobID
    timestamps: ResourceTimestamps
    project_id: ProjectID
    config: BatchPredictionJobConfig
    execution: BatchPredictionJobOutput
    description: str = ''
    batch_prediction_workflow_id: Optional[BatchPredictionWorkflowID] = None
    source: Optional[JobSource] = None
    user: Optional[str] = None
    kumo_version: Optional[str] = None
    storage_path: Optional[str] = None
    input_data_stats: dict[str, Any] | None = None
    registry_metadata: Optional[MLJobRegistryMetadata] = None

    # Explicit index fields keep SQL predicates independent from nested JSON.
    status: Optional[JobStatus] = None
    pquery_id: Optional[PredictiveQueryID] = None
    pquery_def_fpid: Optional[FPID] = None
    training_job_id: Optional[TrainingJobID] = None

    @model_validator(mode='after')
    def _sync_index_fields(self) -> 'BatchPredictionJobDBEntry':
        self.status = self.execution.status
        self.pquery_id = self.config.predictive_query_id
        self.pquery_def_fpid = self.config.pquery_def_fpid
        self.training_job_id = self.config.training_job_id
        return self

    def to_generic(self, ctx):
        from rest.backends.ml_backend.job_models import (
            GenericJob,
            JobFilterMetadata,
        )
        from statedb.config.common import JobType

        self.execution.remove_employee_only_fields()
        return GenericJob(
            job_id=self.id,
            job_type=JobType.BATCH_PREDICTION_JOB,
            execution=self.execution,
            metadata=JobFilterMetadata(
                start_time=self.execution.start_time,
                end_time=(self.execution.last_updated_at
                          if self.execution.status.is_terminal else None),
                created_at=self.timestamps.created_at,
                job_type=JobType.BATCH_PREDICTION_JOB,
                status=self.execution.status,
            ),
            source=self.source,
            user=self.user,
            project_id=self.project_id,
        )
