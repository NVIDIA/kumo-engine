# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from dataclasses import field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Union
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import Field
from pydantic.dataclasses import dataclass
from typing_extensions import Annotated, Literal

from kumoapi._legacy_object_store import (
    migrate_batch_prediction,
    migrate_validator_values,
)
from kumoapi.adls import normalize_adls_source_uri
from kumoapi.common import JobSource, JobStatus, StrEnum, ValidationResponse
from kumoapi.data_snapshot import GraphSnapshotID
from kumoapi.gcs import normalize_gcs_source_uri
from kumoapi.model_plan import (
    ModelPlan,
    PredictionTableGenerationPlan,
    TrainingTableGenerationPlan,
)
from kumoapi.s3 import normalize_s3_destination_uri
from kumoapi.source_table import (
    ADLSSource,
    GCSSource,
    LocalFileSourceTable,
    S3Source,
    SourceTableType,
)
from kumoapi.train import TrainingTableSpec
from kumoapi.typing import (
    _FORBID_EXTRA_CONFIG,
    compatible_field_validator,
    compatible_model_validator,
)

logger = logging.getLogger(__name__)


def _normalize_rooted_object_store_output_path(
    value: str,
    *,
    legacy_uri_normalizer: Optional[Callable[[str], str]] = None,
) -> str:
    """Validate a relative coordinate or canonical provider URI."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError('path must be non-empty without surrounding space')
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        if legacy_uri_normalizer is None:
            raise ValueError('path must be a safe relative path')
        normalized_uri = legacy_uri_normalizer(value)
        if normalized_uri != value:
            raise ValueError('full URI must use its canonical spelling')
        return normalized_uri
    if (parsed.query or parsed.fragment or '?' in value or '#' in value
            or value.startswith('/') or '\\' in value
            or re.search(r'%[0-9a-fA-F]{2}', value)
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError('path must be a safe relative path')
    has_trailing_slash = value.endswith('/')
    normalized = value[:-1] if has_trailing_slash else value
    components = normalized.split('/')
    if (not normalized
            or any(component in {'', '.', '..'} for component in components)):
        raise ValueError(
            'path contains an empty, current, or parent component')
    return f'{normalized}/' if has_trailing_slash else normalized


def _normalize_object_store_destination_uri(value: str) -> str:
    """Canonicalize one exact S3, GCS, or ADLS destination URI."""
    scheme = urlsplit(value).scheme.casefold()
    if scheme == 's3':
        return normalize_s3_destination_uri(value)
    if scheme == 'gs':
        return normalize_gcs_source_uri(value)
    if scheme == 'abfss':
        return normalize_adls_source_uri(value)
    raise ValueError(
        'object-store destination URI must use s3://, gs://, or abfss://')


@dataclass
class EpochMetrics:
    time_taken: Optional[float] = None
    train_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[
        str,
        Optional[float],  # Support for `None` in case metrics are N/A.
    ] = field(default_factory=dict)


@dataclass
class TrainerProgress:
    start_time: Optional[datetime] = None
    # Actual device selected by Lightning for this trial (for example
    # ``cuda``, ``mps``, or ``cpu``).
    accelerator: Optional[str] = None
    num_epochs_completed: int = 0
    elapsed_time: float = 0.0
    estimated_train_time: Optional[float] = None
    # TODO Make this a list
    metrics: Dict[int, EpochMetrics] = field(default_factory=dict)
    # The final best validation model performance after training is finished:
    final_validation_metrics: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.num_epochs_completed != len(self.metrics):
            raise ValueError("Number of epochs and metrics size do not match")


@dataclass
class AutoTrainerProgress:
    start_time: Optional[datetime] = None
    total_trials: int = 0
    completed_trials: int = 0
    elapsed_training_time: float = 0.0
    # Provides an estimated amount of time for the training to complete.
    # This is generally an upper bound and the estimate gets more accurate
    # as time progresses.
    estimated_training_time: Optional[float] = None
    # Map of trial ID to training progress for all running/completed trials
    trial_progress: Dict[str, TrainerProgress] = field(default_factory=dict)


@dataclass
class PredictionProgress:
    start_time: Optional[datetime] = None
    total_iterations: int = 0
    completed_iterations: int = 0
    elapsed_prediction_time: timedelta = timedelta(seconds=0)
    # Provides an estimated amount of time for the prediction to complete.
    # This is generally an upper bound and the estimate gets more accurate
    # as time progresses.
    estimated_prediction_time: Optional[timedelta] = None


class JobType(StrEnum):
    GENERATE_TRAIN_TABLE_JOB = "GENERATE_TRAIN_TABLE_JOB"
    GENERATE_PREDICTION_TABLE_JOB = "GENERATE_PREDICTION_TABLE_JOB"
    TRAINING_JOB = "TRAINING_JOB"
    BATCH_PREDICTION_JOB = "BATCH_PREDICTION_JOB"


# Execution status of a Training or Batch Prediction job.
# Log entry to recorded detailed events throughout multi-step job execution.
@dataclass
class JobEventLogEntry:
    # Name of current stage (step).
    stage_name: str
    last_updated_at: datetime
    detail: Optional[str] = None


@dataclass
class JobStatusReport:
    status: JobStatus

    # URL to the Kumo web UI page that allows human to track and monitor job
    # progress, and also view the job summary after the job finishes.
    tracking_url: str

    start_time: datetime
    end_time: Optional[datetime] = None  # Present when status is not RUNNING

    # Informational job execution event log for logging/debugging purpose.
    event_log: List[JobEventLogEntry] = field(default_factory=list)

    # Errors associated with this job mostly caused by failures in
    # async workflows
    validation_response: Optional[ValidationResponse] = None


@dataclass
class Metric:
    name: str
    value: Optional[float]


@dataclass
class ModelEvaluationMetrics:
    # Eval metrics on the test(holdout) data split.
    test_metrics: List[Metric] = field(default_factory=list)

    # Eval metrics on the validation data split.
    validation_metrics: List[Metric] = field(default_factory=list)

    # Eval metrics on the training data split.
    training_metrics: List[Metric] = field(default_factory=list)


@dataclass
class TrainingJobSummary:
    """Summary report of a successful query training job."""
    # Model eval metrics are available when job status is DONE.
    eval_metrics: ModelEvaluationMetrics

    # TODO(siyang): other stats/info such as cost (GPU hours), etc.
    total_elapsed_time: timedelta
    automl_experiments_completed: int


@dataclass(config=_FORBID_EXTRA_CONFIG)
class CustomTrainingTable:
    """Specifies the custom training table to be used for training.

    Args:
        source_table: The source table to the custom training table.
        table_mod_spec: The modifications made to the original training table.
        validated: Whether the custom training table has been validated
            against the original training table.
    """
    source_table: Union[SourceTableType, LocalFileSourceTable]
    table_mod_spec: TrainingTableSpec
    validated: bool = False


@dataclass(config=_FORBID_EXTRA_CONFIG)
class JobRequestBase:
    """Common job launch request options applicable to all job types."""

    # Custom key-value pair tags to be associated with the job.
    # Tags are useful for grouping, searching and managing jobs (and models).
    #
    # Requirements:
    # 1. Key may be at most 64 characters long, and may only contain
    #    alphanumeric, dot, underscore and dash characters.
    # 2. Value may be at most 256 characters long.
    job_tags: Dict[str, str]


@dataclass
class JobResourceBase:
    """Common info/metadata for job resource of any kind"""
    job_id: str
    job_status_report: JobStatusReport

    # Time when job was created (launched)
    created_at: datetime

    # All tags attached to this job, including both system-defined and
    # custom-defined tags at job launch time, as well as additional tag updates
    # made (if any) after job was launched.
    tags: Dict[str, str]

    # Jobs are exclusively project-owned. This is required on every durable
    # job resource; an unowned/global job is not a valid API state.
    project_id: str = field(kw_only=True)


@dataclass(config=_FORBID_EXTRA_CONFIG)
class GenerateTrainTableRequest(JobRequestBase):
    """POST request body to create a generate-train-table job."""
    pquery_id: str
    plan: 'TrainingTableGenerationPlan'
    graph_snapshot_id: Optional[GraphSnapshotID]

    # Rows the caller supplies, rather than rows the platform generates.
    # Set, the job records the supplied table instead of generating one: it
    # resolves the query and the graph snapshot exactly as a generating job
    # does, reads the rows into storage the platform owns, and completes.
    # The plan above is still what the run reads its split out of, which is
    # why a supplied table is named against a job at all.
    custom_train_table: Optional[CustomTrainingTable] = None


@dataclass
class GenerateTrainTableJobResource(JobResourceBase):
    config: GenerateTrainTableRequest
    source: JobSource
    user: Optional[str] = None

    # Whether this job recorded supplied rows rather than generating them.
    # `config.custom_train_table` above answers *what* was registered, and
    # reads `None` both for a job that registered nothing and for a record
    # that cannot produce the spec. This separates the two, so a reader can
    # refuse the second rather than treating it as the first and training on
    # rows the caller did not supply.
    custom_train_table_used: bool = False


@dataclass(config=_FORBID_EXTRA_CONFIG)
class GeneratePredictionTableRequest(JobRequestBase):
    pquery_id: str
    plan: 'PredictionTableGenerationPlan'
    graph_snapshot_id: Optional[GraphSnapshotID]


@dataclass
class GeneratePredictionTableJobResource(JobResourceBase):
    config: GeneratePredictionTableRequest
    source: JobSource
    user: Optional[str] = None


@dataclass(config=_FORBID_EXTRA_CONFIG)
class TrainingJobRequest(JobRequestBase):
    """POST request body to create a training job."""
    pquery_id: str

    # Required field without default?  Or optional field default to None?
    model_plan: 'ModelPlan' = field(default_factory=ModelPlan)

    # Optional, a specific Graph data snapshot to use in this training job.
    graph_snapshot_id: Optional[GraphSnapshotID] = None

    # Optionally we can specify the ID of generate-train-table job that was
    # created with the same pquery_name.
    # If not specified (by default), a generate-train-table job will be
    # launched with default plan.
    train_table_job_id: Optional[str] = None
    # Depreciated in favor of `custom_train_table`.
    train_table_override: Optional[SourceTableType] = None

    # Used to specify a train table that is modified
    # post generation via PQ. Currently only supports adding
    # a wieght colum.
    custom_train_table: Optional[CustomTrainingTable] = None

    # Training will start from the best model from
    # the below training job.
    warm_start_job_id: Optional[str] = None


@dataclass
class TrainingJobResource(JobResourceBase):
    config: TrainingJobRequest

    # Present if job status is DONE.
    result: Optional[TrainingJobSummary] = None


@dataclass(config=_FORBID_EXTRA_CONFIG)
class BatchPredictionOptions:
    # Required if prediction task is to perform binary classification.
    binary_classification_threshold: Optional[float] = None

    # On classification tasks, for each entity, we will only return predictions
    # for the K classes with the highest predicted values for the entity.
    # If empty, predict all class. This field is ignored for regression tasks.
    num_classes_to_return: Optional[int] = None

    # No.of workers to use when generating batch predictions. When set to a
    # value greater than 1, the prediction table is partitioned into smaller
    # parts and processed in parallel.
    #
    # Default: 1 - Sequential inference over the prediction table.
    num_workers: int = 1


class TrainingTableArtifactType(StrEnum):
    FULL_TRAIN_TABLE = "FULL_TRAIN_TABLE"
    # TODO: Add more artifact types like train table split


class PredictionArtifactType(StrEnum):
    """Specifies what kind of batch predictions should be generated.
    The user may specify multiple types of predictions to be computed, and
    each one will be output to a separate file.
    """
    PREDICTIONS = "PREDICTIONS"
    EMBEDDINGS = "EMBEDDINGS"


class PredictionStorageType(StrEnum):
    S3 = "S3"
    GCS = "GCS"
    ADLS = "ADLS"
    SNOWFLAKE = "SNOWFLAKE"
    DATABRICKS = "DATABRICKS"
    BIGQUERY = "BIGQUERY"
    LOCAL = "LOCAL"


# Metadata fields that can be optionally selected and included as additional
# columns in the output table.
class MetadataField(StrEnum):
    ANCHOR_TIMESTAMP = 'ANCHOR_TIMESTAMP'
    JOB_TIMESTAMP = 'JOB_TIMESTAMP'


class WriteMode(StrEnum):
    OVERWRITE = "OVERWRITE"
    APPEND = "APPEND"


@dataclass(config=_FORBID_EXTRA_CONFIG)
class LocalPredictionOutput:
    artifact_type: PredictionArtifactType
    storage_type: Literal[
        PredictionStorageType.LOCAL] = PredictionStorageType.LOCAL
    extra_fields: List[MetadataField] = field(default_factory=list)


@dataclass(config=_FORBID_EXTRA_CONFIG)
class SnowflakePredictionOutput:
    artifact_type: PredictionArtifactType
    connector_id: str
    table_name: str
    storage_type: Literal[
        PredictionStorageType.SNOWFLAKE] = PredictionStorageType.SNOWFLAKE
    # Select additional metadata fields to be included as columns in data.
    extra_fields: List[MetadataField] = field(default_factory=list)
    # Whether to APPEND or OVERWRITE the data in the existing table. If using
    # APPEND, it is strongly recommended to use JOB_TIMESTAMP as extra_fields.
    write_mode: WriteMode = WriteMode.OVERWRITE


@dataclass(config=_FORBID_EXTRA_CONFIG)
class S3PredictionOutput:
    r"""Prediction output addressed through a project-owned S3 connector."""
    artifact_type: PredictionArtifactType
    connector_id: str
    file_path: str
    storage_type: Literal[PredictionStorageType.S3] = PredictionStorageType.S3
    # Select additional metadata fields to be included as columns in data.
    extra_fields: List[MetadataField] = field(default_factory=list)

    @compatible_field_validator('file_path')
    def normalize_file_path(cls, value: str) -> str:
        return _normalize_rooted_object_store_output_path(
            value, legacy_uri_normalizer=normalize_s3_destination_uri)


@dataclass(config=_FORBID_EXTRA_CONFIG)
class GCSPredictionOutput:
    r"""Prediction output addressed through a project-owned GCS connector."""
    artifact_type: PredictionArtifactType
    file_path: str
    connector_id: str
    storage_type: Literal[
        PredictionStorageType.GCS] = PredictionStorageType.GCS
    # Select additional metadata fields to be included as columns in data.
    extra_fields: List[MetadataField] = field(default_factory=list)

    @compatible_field_validator('file_path')
    def normalize_file_path(cls, value: str) -> str:
        return _normalize_rooted_object_store_output_path(
            value, legacy_uri_normalizer=normalize_gcs_source_uri)


@dataclass(config=_FORBID_EXTRA_CONFIG)
class ADLSPredictionOutput:
    r"""Prediction output addressed through a project-owned ADLS connector."""
    artifact_type: PredictionArtifactType
    file_path: str
    connector_id: str
    storage_type: Literal[
        PredictionStorageType.ADLS] = PredictionStorageType.ADLS
    # Select additional metadata fields to be included as columns in data.
    extra_fields: List[MetadataField] = field(default_factory=list)

    @compatible_field_validator('file_path')
    def normalize_file_path(cls, value: str) -> str:
        return _normalize_rooted_object_store_output_path(
            value, legacy_uri_normalizer=normalize_adls_source_uri)


@dataclass(config=_FORBID_EXTRA_CONFIG)
class DatabricksPredictionOutput:
    artifact_type: PredictionArtifactType
    connector_id: str
    table_name: str
    storage_type: Literal[
        PredictionStorageType.DATABRICKS] = PredictionStorageType.DATABRICKS
    # Select additional metadata fields to be included as columns in data.
    extra_fields: List[MetadataField] = field(default_factory=list)
    # Whether to replace or append to an existing destination table.
    # OVERWRITE is retry-safe because each attempt recreates the table;
    # APPEND disables workflow retries to avoid duplicate rows.
    write_mode: WriteMode = WriteMode.OVERWRITE


@dataclass(config=_FORBID_EXTRA_CONFIG)
class BigQueryPredictionOutput:
    storage_type: Literal[PredictionStorageType.BIGQUERY]
    artifact_type: PredictionArtifactType
    connector_id: str
    table_name: str
    # Select additional metadata fields to be included as columns in data.
    extra_fields: List[MetadataField] = field(default_factory=list)

    # There are two ways to write to Bigquery:
    # 1. Write directly to the table_name. This is the default behavior
    # with staging_table_name=None. The output is *APPENDED* to the table_name.
    # It's a safe way to not overwrite the existing data in the table_name.
    # 2. Write to a staging table and then copy the staging table to the
    # table_name. After the staging table is written, the table_name will be
    # *OVERWRITTEN* with the staging table. This is done by replacing the
    # table_name with the staging table in an atomic operation. This is
    # recommended when the table_name already exists and we want to overwrite
    # the existing data in the table_name while ensuring that the table_name
    # is never corrupted.
    staging_table_name: Optional[str] = None
    # The default write mode is overwrite and if `staging_table_name`` is None,
    # if default to `table_name_{staging}`.
    write_mode: WriteMode = WriteMode.OVERWRITE


PredictionOutputConfig = Annotated[Union[
    SnowflakePredictionOutput,
    S3PredictionOutput,
    GCSPredictionOutput,
    ADLSPredictionOutput,
    DatabricksPredictionOutput,
    BigQueryPredictionOutput,
    LocalPredictionOutput,
],
                                   Field(discriminator='storage_type')]


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class S3PredictionTableSource:
    r"""Custom prediction data behind one project-owned S3 connector.

    Servers enforce relative coordinates only with rooted connectors and full
    URIs only with any-URI connectors. The connector supplies all identity and
    authorization context; this value contains no credentials.
    """
    connector_id: str
    source: S3Source = Field(discriminator='mode')
    storage_type: Literal[PredictionStorageType.S3] = PredictionStorageType.S3

    def __post_init__(self):
        if not self.connector_id:
            raise ValueError('connector_id must be set')


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class GCSPredictionTableSource:
    r"""Custom prediction data behind one project-owned GCS connector."""
    connector_id: str
    source: GCSSource = Field(discriminator='mode')
    storage_type: Literal[
        PredictionStorageType.GCS] = PredictionStorageType.GCS

    def __post_init__(self):
        if not self.connector_id:
            raise ValueError('connector_id must be set')


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class ADLSPredictionTableSource:
    r"""Custom prediction data behind one project-owned ADLS connector."""
    connector_id: str
    source: ADLSSource = Field(discriminator='mode')
    storage_type: Literal[
        PredictionStorageType.ADLS] = PredictionStorageType.ADLS

    def __post_init__(self):
        if not self.connector_id:
            raise ValueError('connector_id must be set')


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class DatabricksPredictionTableSource:
    r"""Custom prediction data read through a project-owned Databricks
    connector.

    Exactly one of ``table`` or ``sql`` identifies the source. SQL must be a
    read-only query; the server validates its catalog/schema scope.
    """
    connector_id: str
    table: Optional[str] = None
    sql: Optional[str] = None
    storage_type: Literal[
        PredictionStorageType.DATABRICKS] = PredictionStorageType.DATABRICKS

    def __post_init__(self):
        if (not self.connector_id
                or self.connector_id != self.connector_id.strip()):
            raise ValueError(
                'connector_id must be non-empty without surrounding space')
        sources = {'table': self.table, 'sql': self.sql}
        if sum(value is not None for value in sources.values()) != 1:
            raise ValueError('Exactly one of "table" or "sql" must be set')
        name, value = next((name, value) for name, value in sources.items()
                           if value is not None)
        if not value or value != value.strip():
            raise ValueError(
                f'{name} must be non-empty without surrounding space')


PredictionTableSource = Annotated[Union[
    S3PredictionTableSource,
    GCSPredictionTableSource,
    ADLSPredictionTableSource,
    DatabricksPredictionTableSource,
],
                                  Field(discriminator='storage_type')]


@dataclass(config=_FORBID_EXTRA_CONFIG)
class BatchPredictionRequest(JobRequestBase):
    """POST request body to create a Batch Prediction job."""
    # ID of a (successful) modeling job.
    model_training_job_id: str

    predict_options: BatchPredictionOptions
    outputs: List[PredictionOutputConfig] = field(default_factory=list)

    # Optional, a specific Graph data snapshot to use in this training job.
    graph_snapshot_id: Optional[GraphSnapshotID] = None

    # Optionally use a generated prediction-table job or an exact custom
    # object-store source behind a project-owned connector. If neither is
    # specified, a generate-prediction-table child job is launched with the
    # default plan.
    #
    # ID of an in-progress or successfully completed Generate-Prediction-Table
    # Job.
    pred_table_job_id: Optional[str] = None
    pred_table_source: Optional[PredictionTableSource] = None

    # Anchor time for the inline prediction table generation. Only used when
    # pred_table_job_id and pred_table_source are both unset, i.e., the
    # prediction table is created as an inline child of this job. If None,
    # the anchor time is inferred from the data.
    prediction_time: Optional[datetime] = None

    # Whether to enable explanations for the Batch Prediction job
    explanations: bool = False

    @compatible_model_validator(mode='before')
    @classmethod
    def migrate_legacy_s3_prediction_source(cls, values):
        return migrate_validator_values(values, migrate_batch_prediction)

    def _validate_prediction_table_selection(self) -> None:
        selected_modes = sum(
            (self.pred_table_job_id is not None, self.pred_table_source
             is not None, self.prediction_time is not None))
        if selected_modes > 1:
            raise ValueError(
                'Only one of "pred_table_job_id", "pred_table_source", or '
                '"prediction_time" may be set.')

    def __post_init__(self) -> None:
        self._validate_prediction_table_selection()


@dataclass
class BatchPredictionJobSummary:
    """Summary of a successful batch prediction job."""
    num_entities_predicted: int
    # TODO: Add more stats


@dataclass
class BatchPredictionJobResource(JobResourceBase):
    config: BatchPredictionRequest
    # Present if job status is DONE.
    result: Optional[BatchPredictionJobSummary] = None


@dataclass(config=_FORBID_EXTRA_CONFIG)
class NamedTrainingTableOutputConfig:
    r"""Training-table output addressed by a warehouse table name."""
    connector_id: str
    table_name: str
    output_kind: Literal['named'] = 'named'
    artifact_type: TrainingTableArtifactType =\
        TrainingTableArtifactType.FULL_TRAIN_TABLE

    # Write mode is applicable only for warehouse connectors.
    # Whether to APPEND or OVERWRITE the data in the existing table. If using
    # APPEND, it is strongly recommended to use JOB_TIMESTAMP as extra_fields.
    write_mode: WriteMode = WriteMode.OVERWRITE

    # Select additional metadata fields to be included as columns in data.
    extra_fields: List[MetadataField] = field(default_factory=list)

    def __post_init__(self):
        if not self.connector_id:
            raise ValueError('connector_id must be set')
        if not self.table_name:
            raise ValueError('table_name must be set')


@dataclass(config=_FORBID_EXTRA_CONFIG)
class FileTrainingTableOutputConfig:
    r"""Training-table output through a rooted or rootless object store."""
    connector_id: str
    path: str
    output_kind: Literal['file'] = 'file'
    artifact_type: TrainingTableArtifactType =\
        TrainingTableArtifactType.FULL_TRAIN_TABLE
    extra_fields: List[MetadataField] = field(default_factory=list)

    @compatible_field_validator('path')
    def normalize_path(cls, value: str) -> str:
        return _normalize_rooted_object_store_output_path(
            value,
            legacy_uri_normalizer=_normalize_object_store_destination_uri)

    def __post_init__(self):
        if not self.connector_id:
            raise ValueError('connector_id must be set')


TrainingTableOutputConfig = Annotated[
    Union[
        NamedTrainingTableOutputConfig,
        FileTrainingTableOutputConfig,
    ],
    Field(discriminator='output_kind'),
]


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class S3ModelArtifactOutput:
    r"""Model artifacts written through a project-owned S3 connector."""
    connector_id: str
    path: str
    output_kind: Literal['s3'] = 's3'

    @compatible_field_validator('path')
    def normalize_path(cls, value: str) -> str:
        return _normalize_rooted_object_store_output_path(
            value, legacy_uri_normalizer=normalize_s3_destination_uri)

    def __post_init__(self):
        if not self.connector_id:
            raise ValueError('connector_id must be set')


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class GCSModelArtifactOutput:
    r"""Model artifacts written through a project-owned GCS connector."""
    connector_id: str
    path: str
    output_kind: Literal['gcs'] = 'gcs'

    @compatible_field_validator('path')
    def normalize_path(cls, value: str) -> str:
        return _normalize_rooted_object_store_output_path(
            value, legacy_uri_normalizer=normalize_gcs_source_uri)

    def __post_init__(self):
        if not self.connector_id:
            raise ValueError('connector_id must be set')


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class ADLSModelArtifactOutput:
    r"""Model artifacts written through a project-owned ADLS connector."""
    connector_id: str
    path: str
    output_kind: Literal['adls'] = 'adls'

    @compatible_field_validator('path')
    def normalize_path(cls, value: str) -> str:
        return _normalize_rooted_object_store_output_path(
            value, legacy_uri_normalizer=normalize_adls_source_uri)

    def __post_init__(self):
        if not self.connector_id:
            raise ValueError('connector_id must be set')


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class SnowflakeModelArtifactOutput:
    r"""Model artifacts published to a Snowflake Model Registry."""
    connector_id: str
    model_name: str
    version_name: Optional[str] = None
    comment: Optional[str] = None
    output_kind: Literal['snowflake'] = 'snowflake'

    def __post_init__(self):
        if not self.connector_id:
            raise ValueError('connector_id must be set')
        _validate_model_registry_location(self.model_name, self.version_name)


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class DatabricksModelArtifactOutput:
    r"""Model artifacts published to a Databricks Model Registry."""
    connector_id: str
    model_name: str
    version_name: Optional[str] = None
    comment: Optional[str] = None
    output_kind: Literal['databricks'] = 'databricks'

    def __post_init__(self):
        if not self.connector_id:
            raise ValueError('connector_id must be set')
        _validate_model_registry_location(self.model_name, self.version_name)


def _validate_model_registry_location(model_name: str,
                                      version_name: Optional[str]) -> None:
    if (not isinstance(model_name, str) or not model_name
            or model_name != model_name.strip()):
        raise ValueError(
            'model_name must be non-empty without surrounding space')
    if (version_name is not None
            and (not isinstance(version_name, str) or not version_name
                 or version_name != version_name.strip())):
        raise ValueError(
            'version_name must be non-empty without surrounding space')


ModelArtifactOutput = Union[
    S3ModelArtifactOutput,
    GCSModelArtifactOutput,
    ADLSModelArtifactOutput,
    SnowflakeModelArtifactOutput,
    DatabricksModelArtifactOutput,
]


@dataclass(config=_FORBID_EXTRA_CONFIG)
class ModelOutputConfig:
    """Config for publishing an offline model bundle."""
    training_job_id: str
    # Project-owned connector and exact connector-relative destination.
    output: ModelArtifactOutput = Field(discriminator='output_kind')


@dataclass(config=_FORBID_EXTRA_CONFIG)
class ArtifactExportRequest:
    """POST request body to create an artifact export job.

    Args:
        request_id: Canonical UUID idempotency key for this logical export.
            Retries must reuse the same value and payload.
        job_id: ID of a job that generates the artifact.
        prediction_output: Optional prediction output configuration. For
            uploading prediction/embedding artifacts.
        training_table_output: Optional training table output configuration.
            For uploading training table artifacts.
        model_output: Optional offline model bundle publication configuration.
    """
    request_id: str

    # ID of a job that generates the artifact.
    job_id: str

    prediction_output: Optional[PredictionOutputConfig] = None

    training_table_output: Optional[TrainingTableOutputConfig] = None

    model_output: Optional[ModelOutputConfig] = None

    @compatible_field_validator('request_id')
    def validate_request_id(cls, value: str) -> str:
        try:
            parsed = UUID(value)
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError('request_id must be a canonical UUID') from error
        if str(parsed) != value:
            raise ValueError('request_id must be a canonical UUID')
        return value

    def __post_init__(self):
        configs = [
            self.prediction_output, self.training_table_output,
            self.model_output
        ]
        n = sum(c is not None for c in configs)
        if n == 0:
            raise ValueError(
                'Exactly one of "prediction_output", "training_table_output", '
                'or "model_output" must be set.')
        elif n > 1:
            raise ValueError('Only one output config may be set.')


@dataclass
class ArtifactExportResponse:
    # ID of a artifact export job.
    job_id: str


@dataclass
class CancelTrainingJobResponse:
    is_cancelled: bool


@dataclass
class CancelBatchPredictionJobResponse:
    is_cancelled: bool


@dataclass
class GetPredictionsDfUrlResponse:
    """
    Response class for /prediction_jobs/{job_id}/get_prediction_df_urls.
    """
    # List of presigned URLs, each entry corresponding to one parquet partition
    prediction_partitions: List[str]


@dataclass
class GetEmbeddingsDfUrlResponse:
    """
    Response class for /prediction_jobs/{job_id}/get_embedding_df_urls.
    """
    # List of presigned URLs, each entry corresponding to one parquet partition
    embedding_partitions: List[str]


class ErrorType(Enum):
    """
    Enumeration of different error/info response types returned to the
    user.
    """
    # Mostly errors which prevent user from proceeding with an
    # operation.
    ERROR = 0
    # Mostly warnings/info which does not prevent user from
    # proceeding but could be good insights.
    INFO = 1


@dataclass
class ErrorCTA:
    """
    Class representing potential actionable items for the users.
    """
    # While name is mostly useful for the UI, it can be use to
    # format error responses in SDK APIs as well.
    name: str
    # The url associated with the the click to action.
    url: str


@dataclass
class ErrorDetail:
    """
    Each error in workflow execution is associated with one instance
    of :class:ErrorDetail.
    """
    type: ErrorType
    description: str
    title: Optional[str] = None
    # Captures potential actionable items based on the error.
    cta: Optional[ErrorCTA] = None


@dataclass
class ErrorDetails:
    """
    Response class for errors and warnings returned by long running
    jobs.
    """
    items: List[ErrorDetail]
