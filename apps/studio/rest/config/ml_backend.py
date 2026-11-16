# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from kumoapi.typing import Dtype, Stype
from pydantic import Field, model_validator

from io_utils import before_validator_dict, kumo_dataclass
from rest.config.data_backend import (
    GraphDataVersionInfo,
    bind_predictive_query_to_execution_refs,
)
from rest.config.graph_v2 import GraphDataComputeInput
from rest.config.training_job import TrainingSpec
from statedb.config.common import (
    BatchPredictionJobID,
    ConnectorID,
    GenerateTrainTableJobID,
    GraphID,
    JobID,
    PredictiveQueryID,
    TableID,
    TableName,
    TrainingJobID,
)


@kumo_dataclass
class MaterializedColumnInfo:
    name: str
    dtype: Dtype
    stype: Stype
    num_empty: int
    # TODO: Add # invalid


@kumo_dataclass
class TimeColInfo:
    name: str
    min: Optional[datetime]
    max: Optional[datetime]


@kumo_dataclass
class MaterializedTableInfo:
    """Metadata about an individual table that was materialized.
    """
    # Size of the source table pre-ingestion
    size_bytes_source: Optional[int]

    # The UTC time at which the data was read from the data source and imported
    # into our system.
    ingest_start_time: datetime
    ingest_end_time: datetime

    # This is generally the time that the underlying table was last modified.
    # This may differ depending on the specific connector type
    data_version: str

    # Number of rows that were read from the source table.
    num_rows: Optional[int]

    # The list of columns that were ingested
    columns: List[MaterializedColumnInfo]

    # Names of pkey and time_col, if present
    pkey: Optional[str]
    time_col: Optional[TimeColInfo]

    # ID of the table in StateDB that was materialized. Note, the table may have
    # been edited since the table was materialized, so its current columns may
    # not match the columns that were actually materialized.
    table_id: TableID
    table_name: TableName
    connector_id: ConnectorID
    source_table_name: str

    # Version of the code as well as the ID of the temporal workflow that was
    # used to materialize the table.
    code_version: str
    temporal_workflow_id: str

    def clear_private_fields(self) -> None:
        """Obfuscate the fields that should not be returned to the UI.
        """
        self.temporal_workflow_id = ""


@kumo_dataclass
class MaterializationTaskInfo:
    """Information about the tables that were ingested & materialized, during a
    training or batch prediction job. Some of information is made available to
    the user, so that they can understand exactly which input data was used to
    train the model.

    This also contains a bunch of additional debug information that can be
    exposed in the UI for Kumo engineers.
    """
    table_stats: List[MaterializedTableInfo]

    # Version of the temporal worker that performed materialization
    code_version: str

    # time/duration of materialization task
    start_time: datetime
    end_time: datetime

    graph_id: GraphID
    predictive_query_id: PredictiveQueryID
    temporal_workflow_id: str

    # ID of Job that triggered the materialization task
    job_id: Optional[JobID] = None
    #
    # DO NOT USE THESE FIELDS, kept for backward compatibility.
    batch_prediction_job_id: Optional[BatchPredictionJobID] = None
    training_job_id: Optional[TrainingJobID] = None

    @property
    def id(self) -> JobID:
        if self.job_id is not None:
            return self.job_id
        if self.training_job_id is not None:
            return self.training_job_id
        if self.batch_prediction_job_id is not None:
            return self.batch_prediction_job_id
        raise RuntimeError("Job ID is missing in MaterializationTaskInfo")

    # v1 @root_validator(skip_on_failure=True) -> @model_validator(mode='after').
    @model_validator(mode='after')
    def validate_id_present(self) -> 'MaterializationTaskInfo':
        training_job_id = self.training_job_id
        batch_prediction_job_id = self.batch_prediction_job_id
        job_id = self.job_id
        training_present = training_job_id is not None
        batch_pred_present = batch_prediction_job_id is not None
        job_id_present = job_id is not None
        if sum([training_present, batch_pred_present, job_id_present]) != 1:
            raise ValueError(
                "Exactly one of training_job_id, batch_prediction_job_id and "
                f"job_id fields must be present, but got: {training_job_id}, "
                f"{batch_prediction_job_id}, {job_id}")
        return self

    def clear_private_fields(self) -> None:
        """Obfuscate the fields that should not be returned to the UI.
        """
        self.temporal_workflow_id = ""
        for table in self.table_stats:
            table.clear_private_fields()


class JobState(str, Enum):
    """The terminal state of a pooled ML activity."""
    SUCCEEDED = 'SUCCEEDED'
    FAILED = 'FAILED'


@kumo_dataclass
class TrainingJobInput:
    """A complete specification of a training job (or genereate-train-table
    job). Meant for internal use within ML backend.

    Should not be confused with TrainingSpec, which only contains the
    parameters related to business logic/problem formulation for the training
    job.
    """
    id: TrainingJobID
    spec: TrainingSpec

    # Start time of the job (eg, when it was submitted).
    # Stored for purpose of logging and debugging.
    start_time: datetime

    # Child job id of the GenerateTrainingTableWorkflow, which may or may not
    # be separately triggered.
    generate_train_table_job_id: GenerateTrainTableJobID = Field()

    graph_data_compute_input: GraphDataComputeInput = Field()

    # Optional, run the training job over a specific fixed graph data version.
    graph_version_info: Optional[GraphDataVersionInfo] = None

    # Optional training job ID to warm start from.
    warm_start_job_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.spec.predictive_query = bind_predictive_query_to_execution_refs(
            self.spec.predictive_query,
            self.graph_data_compute_input.connectors,
        )

    # For backward compatibility, during deserialization, if the
    # "generate_train_table_job_id" field is absent, we'll set a default value.
    # v1 @root_validator(pre=True) -> @model_validator(mode='before').
    @model_validator(mode='before')
    @classmethod
    def set_default_child_job_id(cls, values: Dict[str, Any]):
        values = before_validator_dict(values)
        if 'generate_train_table_job_id' not in values:
            id = TrainingJobID(values['id'])
            child_id = GenerateTrainTableJobID.as_child_of(id)
            values['generate_train_table_job_id'] = child_id
        return values

    @property
    def weight_col_name(self) -> Optional[str]:
        """If the training job is using a custom training table and
        it has a weight col return its name. Otherwise,
        return None.
        """
        if hasattr(self.spec, 'custom_train_table'):
            if self.spec.custom_train_table is not None:
                return self.spec.custom_train_table.weight_col_name
        if hasattr(self.spec, 'advanced_options'):
            train_table_opts = (
                self.spec.advanced_options.training_table_generation)
            if train_table_opts.weight_col is not None:
                from kumo.pquery.weight_spec import DEFAULT_WEIGHT_COL_NAME
                return DEFAULT_WEIGHT_COL_NAME
        return None


# TODO(frg100): Create a proper GenerateTrainTableJobInput class
GenerateTrainTableJobInput = TrainingJobInput
