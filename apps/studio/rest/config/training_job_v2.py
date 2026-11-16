# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import Dict, List, Optional

import kumoapi.jobs as jobs_api
import kumoapi.pquery as pquery_api
import kumoapi.source_table as source_table_api
from kumo_core.model_plan import ModelPlan, TrainingTableGenerationPlan
from kumo_core.types import TaskType
from kumoapi.data_snapshot import GraphSnapshotID
from pydantic import ConfigDict, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

from io_utils import kumo_dataclass
from rest.config.graph_v2 import GraphDefinition
from rest.config.training_job import TrainingJobResponse
from statedb.config.common import GraphID, JobSource, ProjectID


@kumo_dataclass
class TrainingJobInternalRequest:
    """Driver-facing training job launch request.

    Mirrors :class:`kumoapi.jobs.TrainingJobRequest`, except that it carries
    the internal :class:`~kumo_core.model_plan.ModelPlan`. Callers translate
    the public model plan at the REST boundary so that the driver layer never
    round-trips a plan through the wire contract.
    """
    # ID of the predictive query to train.
    pquery_id: str

    # Full internal model plan, including the training-table-generation
    # section that the public contract omits.
    model_plan: ModelPlan = field(default_factory=ModelPlan)

    # Custom key-value tags associated with the job.
    job_tags: Dict[str, str] = field(default_factory=dict)

    # Optional, a specific graph data snapshot to use in this training job.
    graph_snapshot_id: Optional[GraphSnapshotID] = None

    # Optional ID of a generate-train-table job created for the same pquery.
    # If unset, a generate-train-table job is launched as a child job.
    train_table_job_id: Optional[str] = None

    # Train table that was modified post generation via PQ.
    custom_train_table: Optional[jobs_api.CustomTrainingTable] = None

    # If set, training starts from the best model of this training job.
    warm_start_job_id: Optional[str] = None

    @field_validator('custom_train_table')
    @classmethod
    def _refuse_a_local_file_source(
        cls, v: Optional[jobs_api.CustomTrainingTable]
    ) -> Optional[jobs_api.CustomTrainingTable]:
        # Nothing above resolves a local path, so without this it would reach
        # the connector lookup in a Temporal activity
        # (`driver_common.py:401`) rather than the request that sent it.
        local_file = source_table_api.LocalFileSourceTable
        if v is not None and isinstance(v.source_table, local_file):
            raise ValueError(
                'A local file custom training table is readable only by a '
                'backend running in the calling process, not by this server.')
        return v


@pydantic_dataclass(config=ConfigDict(extra='forbid'))
class TrainingJobRequest:
    """
    Training job request with additional parameters needed for UI.
    """
    # Predictive query for this training job.
    predictive_query: str

    # Graph Definition to train the quert in.
    graph_definition: GraphDefinition

    # Plan used for training table generation. Note that the
    # api.TrainingJobRequest ModelPlan doesn't include this.
    training_table_gen_plan: TrainingTableGenerationPlan

    # Training job model plan excluding TrainTableGen spec from the plan.
    model_plan: ModelPlan

    project_id: ProjectID

    # List of custom tags associated with the job.
    tags: Optional[List[str]] = None

    # Custom key-value tags associated with the UI job.
    job_tags: Optional[Dict[str, str]] = None

    # Optional training job ID to warm start from. If provided, the training
    # will start from the best model checkpoint of the specified training job.
    warm_start_job_id: Optional[str] = None

    # Field to retain information about user that triggered the job.
    # Optional because old jobs don't have this field.
    user: Optional[str] = None

    # Name of the original graph template this snapshot was derived from,
    # populated when the user edits an existing graph before training.
    # Used to set graph_name as "<base_graph_name> (modified)" on the snapshot.
    base_graph_name: Optional[str] = None


@kumo_dataclass
class TrainingWorkflowResponse:
    pquery_definition: pquery_api.PQueryResource
    training_job: TrainingJobResponse
    task_type: TaskType
    query_type: Optional[pquery_api.QueryType]
    graph_snapshot_id: GraphSnapshotID | None
    generation_job: Optional[jobs_api.GenerateTrainTableJobResource] = None
    legacy_graph_id: Optional[GraphID] = None
    job_source: Optional[JobSource] = None
