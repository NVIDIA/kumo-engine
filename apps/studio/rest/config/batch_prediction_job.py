# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from kumo_core.types import PredictionArtifactType, TaskType
from kumo_core.types.schema import ColumnArgs
from kumoapi.jobs import TrainingTableArtifactType
from kumoapi.model_plan import PredictionTableGenerationPlan
from kumoapi.typing import Dtype, Stype
from pandas import Timestamp
from pydantic import field_validator, model_validator
from temporalio.client import WorkflowExecutionStatus

from async_utils.sync import blocking_method
from io_utils import before_validator_dict, fsutil, kumo_dataclass, logging
from io_utils.json import load_json
from io_utils.logging import traceback
from kumo.config import PredictiveQueryV2Config
from kumo.data import get_active_cache
from kumo.home import HomeDirManager
from kumo.stats import is_missing_stat_value
from kumo.stats.stats import Stats
from kumo.stats.train_table_stats import PRED_TIME
from kumo.train.artifact_manifest import (
    PREDICTION_PRODUCER_ABI,
    PredictionManifestRef,
)
from kumo.validation import ValidationResponse
from rest.config.data_backend import (
    ConnectorExecutionRef,
    GraphDataVersionInfo,
    bind_predictive_query_to_execution_refs,
)
from rest.config.error_handling import extract_failure_info
from rest.config.graph_v2 import GraphDataComputeInput
from rest.config.ml_backend import JobState
from rest.config.prediction_output import (
    ADLSPredictionOutput,
    BigQueryPredictionOutput,
    DatabricksPredictionOutput,
    GCSPredictionOutput,
    LocalPredictionOutput,
    PredictionOutputConfig,
    PredictionStorageType,
    S3PredictionOutput,
    SnowflakePredictionOutput,
)
from rest.config.table import ColStatsResponse
from statedb.config.common import (
    BatchPredictionJobID,
    BatchPredictionWorkflowID,
    ConnectorID,
    GeneratePredictionTableJobID,
    GraphID,
    GraphLink,
    JobEventLogEntry,
    JobExecution,
    JobFailureInfo,
    JobProgress,
    JobSource,
    JobStatus,
    PredictiveQueryID,
    PredictiveQueryLink,
    ProjectID,
    TrainingJobID,
)
from statedb.config.job_tag import JobTagKey, JobTagValue
from statedb.config.timestamp import ResourceTimestamps
from statedb.rest.batch_prediction_job import \
    BatchPredictionJobConfig as SerializedBatchPredictionJobConfig

log = logging.getLogger(__name__)

_REST_OUTPUT_TYPES: Dict[PredictionStorageType, type] = {
    PredictionStorageType.S3: S3PredictionOutput,
    PredictionStorageType.GCS: GCSPredictionOutput,
    PredictionStorageType.ADLS: ADLSPredictionOutput,
    PredictionStorageType.SNOWFLAKE: SnowflakePredictionOutput,
    PredictionStorageType.DATABRICKS: DatabricksPredictionOutput,
    PredictionStorageType.BIGQUERY: BigQueryPredictionOutput,
    PredictionStorageType.LOCAL: LocalPredictionOutput,
}
_REST_OUTPUT_CLASSES = tuple(_REST_OUTPUT_TYPES.values())


def _output_values(output: Any) -> Dict[str, Any]:
    if isinstance(output, dict):
        return output
    if dataclasses.is_dataclass(output):
        return dataclasses.asdict(output)
    return dict(output)


def _coerce_rest_output(output: Any) -> PredictionOutputConfig:
    if isinstance(output, _REST_OUTPUT_CLASSES):
        return output
    values = _output_values(output)
    output_type = _REST_OUTPUT_TYPES[PredictionStorageType(
        values['storage_type'])]
    return output_type(**values)


@kumo_dataclass
class BatchPredictionJobConfig(SerializedBatchPredictionJobConfig):
    outputs: List[PredictionOutputConfig] = field(default_factory=list)

    @property
    def pred_table_generation_plan(self) -> PredictionTableGenerationPlan:
        plan = PredictionTableGenerationPlan()
        if self.prediction_time:
            plan.anchor_time = int(self.prediction_time.strftime('%s'))
        if self.lag_timesteps is not None:
            plan.lag_timesteps = self.lag_timesteps
        plan.year_over_year = self.year_over_year
        plan.forecast_length = self.forecast_length
        return plan

    @field_validator('outputs', mode='before')
    @classmethod
    def _coerce_rest_outputs_before_union(cls, value: Optional[Any]) -> Any:
        if value is None:
            return value
        # Pydantic discriminated unions validate dataclass inputs as dicts.
        # Rehydrate REST output behavior in the post-validation root validator.
        return [_output_values(output) for output in value]

    # v1 @root_validator(skip_on_failure=True) -> @model_validator(mode='after').
    @model_validator(mode='after')
    def _coerce_rest_outputs(self) -> 'BatchPredictionJobConfig':
        self.outputs = [_coerce_rest_output(output) for output in self.outputs]
        return self


def requested_prediction_artifact_types(
    config: BatchPredictionJobConfig, ) -> set[PredictionArtifactType]:
    """Return the producer artifacts explicitly requested by a BP job."""
    artifact_types = {
        PredictionArtifactType(output.artifact_type)
        for output in config.outputs
    }
    if not artifact_types:
        raise ValueError(
            'Batch prediction requires at least one requested output artifact.')
    return artifact_types


def has_only_databricks_outputs(config: BatchPredictionJobConfig) -> bool:
    """Whether every requested artifact is written directly to Databricks."""
    return bool(config.outputs) and all(
        isinstance(output, DatabricksPredictionOutput)
        for output in config.outputs)


def direct_databricks_prediction_output_index(
        config: BatchPredictionJobConfig) -> Optional[int]:
    """Return one raw-prediction stage usable for pre-COPY statistics."""
    if not has_only_databricks_outputs(config):
        return None
    return next(
        (index for index, output in enumerate(config.outputs)
         if output.artifact_type == PredictionArtifactType.PREDICTIONS),
        None,
    )


@kumo_dataclass
class BatchPredictionJobProgress(JobProgress):
    pass


@kumo_dataclass
class OutputArtifactStats:
    """Information about each output artifact from the batch prediction job.
    """
    artifact_type: Union[PredictionArtifactType, TrainingTableArtifactType]
    num_rows: int
    num_cols: int
    size_bytes: int
    output_path: str


@kumo_dataclass
class DatabricksVolumeShard:
    """One prediction worker's files in a managed Unity Catalog volume."""

    output_index: int
    artifact_type: PredictionArtifactType
    stage_path: str
    staged_files: List[str]
    num_rows: int
    num_cols: int
    size_bytes: int
    uploaded_bytes: int = 0
    upload_seconds: float = 0.0
    file_count: int = 0
    producer_abi: str = PREDICTION_PRODUCER_ABI


@kumo_dataclass
class PredictionShardResult:
    state: JobState
    databricks_uploads: List[DatabricksVolumeShard] = field(
        default_factory=list)
    artifact_manifest: PredictionManifestRef | None = None


@kumo_dataclass
class BatchPredictionJobResults:
    """Information to display when the job is done.
    """
    # This is currently shown as "Prediction Target" in the UI, which is the
    # predictive_query.target field.
    key_cols: List[str]

    # For each output configured in BatchPredictionJobConfig, this contains
    # the stats of the output artifact generated.
    # NOTE: Setting this to list | None to simplify in the front-end for
    # checking if the field is present.
    output_artifact_stats: list[OutputArtifactStats] | None = None

    # All fields below are depracated in favor of output_artifact_stats
    num_rows: int = -1
    num_cols: int = -1
    size_bytes: int = -1
    output_paths: List[str] = field(default_factory=list)
    # End deprecated fields


@kumo_dataclass
class BatchPredictionJobInput:
    id: BatchPredictionJobID
    project_id: ProjectID
    config: BatchPredictionJobConfig

    # TODO: add information about the trained model
    predictive_query: PredictiveQueryV2Config
    # Secret-free, project-bound references for every external output.
    output_connectors: Dict[ConnectorID, ConnectorExecutionRef]

    # Start time of the job (eg, when it was submitted).
    # Stored for purpose of logging and debugging.
    start_time: datetime

    # TODO(Dulloor): Is it ok to make graph_data_compute_input non-optional?
    graph_data_compute_input: GraphDataComputeInput

    # Exact project connector used to stage a custom prediction-table source.
    # The source coordinate itself lives in ``config``; credentials never do.
    custom_pred_table_connector: Optional[ConnectorExecutionRef] = None

    # Optional, run the pred job over a specific fixed graph data version.
    graph_version_info: Optional[GraphDataVersionInfo] = None

    # For unit tests - up to table generation can be done on fake training job
    stop_after_table_gen: bool = False

    def __post_init__(self) -> None:
        self.predictive_query = bind_predictive_query_to_execution_refs(
            self.predictive_query,
            self.graph_data_compute_input.connectors,
        )

    @property
    def generate_pred_table_job_id(
            self) -> Optional[GeneratePredictionTableJobID]:
        return self.config.generate_pred_table_job_id

    @property
    def training_job_id(self) -> TrainingJobID:
        return self.config.training_job_id

    @property
    def predictive_query_id(self) -> PredictiveQueryID:
        return self.config.predictive_query_id

    @property
    def graph_id(self) -> GraphID:
        assert self.config.graph_id
        return self.config.graph_id

    @property
    def prediction_time(self) -> Optional[datetime]:
        return self.config.prediction_time

    # SQL Transformation to apply to the generated predictions
    @property
    def transformation(self) -> Optional[str]:
        return self.config.transformation

    # v1 @root_validator(pre=True) -> @model_validator(mode='before').
    @model_validator(mode='before')
    @classmethod
    def _migrate_config_fields(cls, values: Dict[str, Any]):
        values = before_validator_dict(values)
        config = values['config']
        if not isinstance(config, dict):
            config = dataclasses.asdict(config)

        prediction_time = 'prediction_time'
        if prediction_time in values and prediction_time not in config:
            config[prediction_time] = values.pop(prediction_time)
        transformation = 'transformation'
        if transformation in values and transformation not in config:
            config[transformation] = values.pop(transformation)

        graph_id = 'graph_id'
        if graph_id in values and graph_id not in config:
            config[graph_id] = values.pop(graph_id)
        if graph_id not in config:
            raise ValueError(
                'graph_id is missing in BatchPredictionJobInput.config')

        values['config'] = config
        return values

    # v1 @root_validator(skip_on_failure=True) -> @model_validator(mode='after').
    @model_validator(mode='after')
    def _check_invariants(self) -> 'BatchPredictionJobInput':
        self.validate_connector_execution_refs()
        return self

    def validate_connector_execution_refs(self) -> None:
        """Reassert that every durable connector capability is exact.

        This is intentionally callable at activity boundaries after loading a
        persisted input, in addition to normal model construction validation.
        """
        for connector_id, reference in self.graph_data_compute_input.connectors.items(
        ):
            if not isinstance(reference, ConnectorExecutionRef):
                raise TypeError(
                    'Graph connectors must be ConnectorExecutionRef values')
            if ConnectorID(connector_id) != reference.connector_id:
                raise ValueError(
                    'Graph connector key does not match its execution reference'
                )
            if reference.project_id != self.project_id:
                raise ValueError(
                    'Graph connector execution reference belongs to a '
                    'different project')

        config: BatchPredictionJobConfig = self.config
        configured_output_ids: set[ConnectorID] = set()
        for output in config.outputs:
            output_types = (
                SnowflakePredictionOutput,
                DatabricksPredictionOutput,
                BigQueryPredictionOutput,
                S3PredictionOutput,
                GCSPredictionOutput,
                ADLSPredictionOutput,
            )
            if isinstance(output, output_types):
                connector_id = ConnectorID(output.connector_id)
                configured_output_ids.add(connector_id)
                if connector_id not in self.output_connectors:
                    raise ValueError(
                        f"Prediction output with ID '{connector_id}' is "
                        "missing its authorized connector reference")

        if set(self.output_connectors) != configured_output_ids:
            raise ValueError(
                'Authorized output connector references must exactly match '
                'the configured prediction outputs')
        for connector_id, reference in self.output_connectors.items():
            if not isinstance(reference, ConnectorExecutionRef):
                raise TypeError(
                    'Output connectors must be ConnectorExecutionRef values')
            if ConnectorID(connector_id) != reference.connector_id:
                raise ValueError(
                    'Output connector key does not match its execution reference'
                )
            if reference.project_id != self.project_id:
                raise ValueError(
                    'Output connector execution reference belongs to a '
                    'different project')

        source = config.custom_pred_table_source
        reference = self.custom_pred_table_connector
        if (source is None) != (reference is None):
            raise ValueError(
                'Custom prediction source and connector reference must be '
                'provided together')
        if reference is not None:
            if not isinstance(reference, ConnectorExecutionRef):
                raise TypeError('Custom prediction connector must be a '
                                'ConnectorExecutionRef value')
            if reference.project_id != self.project_id:
                raise ValueError(
                    'Custom prediction connector execution reference belongs '
                    'to a different project')
            if (source is not None
                    and source.connector_id != str(reference.connector_id)):
                raise ValueError(
                    'Custom prediction source connector does not match its '
                    'execution reference')


@kumo_dataclass
class BatchPredictionJobOutput(JobExecution):
    # Set by StateDbJobQueue at admission, never at submission.
    start_time: Optional[datetime] = None
    # present iff status=PROCESSING
    progress: Optional[BatchPredictionJobProgress] = None
    # present iff status=DONE
    results: Optional[BatchPredictionJobResults] = None

    @classmethod
    def failed(
        cls, err_msg: Optional[str] = None, stacktrace: Optional[str] = None,
        exception: Optional[BaseException] = None,
        validation: Optional['ValidationResponse'] = None,
        workflow_end_state: Optional['WorkflowExecutionStatus'] = None
    ) -> 'BatchPredictionJobOutput':
        """Create failed job output with proper failure_info.

        Encapsulates failure_info extraction - callers just pass raw error data.

        Args:
            err_msg: User-facing error message
            stacktrace: Debug stacktrace
            exception: Exception object (``BaseException`` accepted since
                workflow crash handlers catch ``BaseException``)
            validation: ValidationResponse (if validation failed)
            workflow_end_state: Optional workflow end state (e.g., FAILED, COMPLETED)

        Returns:
            BatchPredictionJobOutput with status=FAILED and failure_info populated
        """
        failure_info = extract_failure_info(err_msg=err_msg,
                                            stacktrace=stacktrace,
                                            exception=exception,
                                            validation=validation)

        if not failure_info.message:
            failure_info.message = "Batch prediction job failed"

        return cls(
            status=JobStatus.FAILED,
            log=JobEventLogEntry.failed(message=failure_info.message),
            failure_info=failure_info,
            workflow_end_state=workflow_end_state,
        )

    def terminal_invariant_violations(self) -> List[str]:
        """Return human-readable messages for any status->required-field
        invariant this output violates.

        Single source of truth shared by the (warn-only) ``mode='after'``
        validator and the strict write-time gate in ``modify_job_if_open_*``.
        An empty list means the output is internally consistent for its
        current status.
        """
        status: Optional[JobStatus] = self.status
        log_entry: Optional[JobEventLogEntry] = self.log
        progress: Optional[BatchPredictionJobProgress] = self.progress
        metrics: Optional[BatchPredictionJobResults] = self.results
        failure_info: Optional[JobFailureInfo] = self.failure_info

        violations: List[str] = []

        if status == JobStatus.RUNNING and (log_entry and log_entry.stage_name
                                            == 'Predicting'):
            if not progress:
                violations.append(
                    f"BatchPredictionJobOutput.progress must be present "
                    f"if status = {JobStatus.RUNNING} and "
                    f"log.stage_name == Predicting")

        if status == JobStatus.DONE:
            if not metrics:
                violations.append(
                    f"BatchPredictionJobOutput.results must be present "
                    f"if status = {JobStatus.DONE}")

        if status == JobStatus.FAILED:
            if not failure_info:
                violations.append(
                    f"BatchPredictionJobOutput.failure_info must be present "
                    f"if status = {JobStatus.FAILED}")

        return violations

    # v1 @root_validator (post-validation) -> @model_validator(mode='after').
    @model_validator(mode='after')
    def _check_invariants(self) -> 'BatchPredictionJobOutput':
        # These are soft progress-completeness diagnostics here, not structural
        # data-integrity gates. Under pydantic v1 the equivalent
        # ``@root_validator`` only ran once, at the object's own construction.
        # Under v2 a ``mode='after'`` validator also re-runs whenever an
        # already-constructed instance is re-validated -- e.g. when this output
        # is read back through ``JsonRepository`` (which re-validates the full
        # payload on every read) or re-wrapped into a parent model on a
        # read/response path. Hard-raising there turns a build-time sanity
        # check into a fatal error on legitimate read/mutation flows. Log a
        # warning instead of raising so these flows keep working while the
        # diagnostic signal is retained.
        #
        # The strict v1 WRITE-time enforcement is restored only at the persist
        # chokepoint (``MLJobService.modify_job_if_open_generic``), which calls
        # ``terminal_invariant_violations`` and raises so a broken terminal job
        # cannot be saved. The read/response path stays warn-only.
        for violation in self.terminal_invariant_violations():
            log.warning("%s", violation)
        return self

    @staticmethod
    def missing_from_ml_db(
            id: BatchPredictionJobID) -> 'BatchPredictionJobOutput':
        return BatchPredictionJobOutput(
            status=JobStatus.FAILED, log=JobEventLogEntry.failed(),
            failure_info=JobFailureInfo(
                message=(f"Job with ID '{id}' could not be "
                         "found in the ML backend"),
                stacktrace=''.join(traceback.format_stack()),
            ))

    def remove_employee_only_fields(self) -> None:
        """Delete fields that should not be shown to customers.
        """
        if self.failure_info is not None:
            self.failure_info.remove_employee_only_fields()


@kumo_dataclass
class BatchPredictionJobResponse:
    """Output of the GetBatchPrediction and ListBatchPrediction REST calls.
    """
    id: BatchPredictionJobID
    config: BatchPredictionJobConfig
    timestamps: ResourceTimestamps
    predictive_query_link: PredictiveQueryLink
    execution: BatchPredictionJobOutput
    graph_link: GraphLink
    project_id: str
    description: str = ''
    batch_prediction_workflow_id: Optional[BatchPredictionWorkflowID] = None
    tags: Dict[JobTagKey, JobTagValue] = field(default_factory=dict)
    job_source: Optional[JobSource] = None
    user: Optional[str] = None
    kumo_version: Optional[str] = None
    download_supported: Optional[bool] = None


@kumo_dataclass
class PaginatedBatchPredictionJobResponse:
    """
    Contains `BatchPredictionJobResponses` and
    associated metadata (page number, number of pages, etc.)
    required to implement pagination in the frontend.

    Note: Replace BatchPredictionJobResponse wherever
    pagination is implemented.
    """
    jobs: List[BatchPredictionJobResponse]
    current_page: int
    number_of_pages: int
    total_number_of_jobs: int


@kumo_dataclass
class BatchPredictionJobStats:
    raw_predictions: ColStatsResponse
    transformed_predictions: Optional[ColStatsResponse] = None


@kumo_dataclass
class BatchPredictionJobAnchorTime:
    anchor_time: Optional[datetime] = None

    @blocking_method()
    def get_stats_from_per_job_dir(
        self,
        job_id: str,
        home_dir: str,
    ) -> Optional['BatchPredictionJobAnchorTime']:
        """Load batch prediction anchor time from the job's output directory.

        Args:
            home_dir (str): Base directory containing job outputs.
            bp_job_id (str): Batch prediction job ID.

        Returns:
            Optional[BatchPredictionJobAnchorTime]: if the file exists and
            can be loaded, None otherwise.
        """
        batch_pred_table_stats_path = HomeDirManager(home_dir).pred_table_stats

        if not fsutil.exists(batch_pred_table_stats_path):
            log.debug(f"Stats file not found at {batch_pred_table_stats_path}")
            return None

        pred_table_stats = load_json(batch_pred_table_stats_path,
                                     Dict[str, Dict[str, Any]])
        pred_time_stat = Stats.from_config(pred_table_stats.get(PRED_TIME))
        if pred_time_stat is not None:
            assert isinstance(pred_time_stat, Stats)
            pred_time = pred_time_stat.value
            if is_missing_stat_value(pred_time):
                log.warning(
                    'Prediction anchor time is invalid or unavailable in '
                    'the stats cache for job %s', job_id)
            else:
                assert isinstance(pred_time, Timestamp)
                log.info('Using prediction time of %s', pred_time)
                self.anchor_time = pred_time.to_pydatetime()
        else:
            log.warning(
                'Prediction anchor time is not yet available in '
                'the stats cache for job %s', job_id)
        return self

    @blocking_method()
    def load_from_cache(
        self,
        job_id: BatchPredictionJobID,
    ) -> 'BatchPredictionJobAnchorTime':
        from kumo.data.backend_cache import BackendCacheManager
        cache: BackendCacheManager = get_active_cache()
        pred_table_stats = cache.pquery_table_stats_cache.get_stats(
            job_id, [PRED_TIME])
        pred_time_stat = pred_table_stats.get(PRED_TIME)

        if pred_time_stat is not None:
            assert isinstance(pred_time_stat, Stats)
            pred_time = pred_time_stat.value
            if is_missing_stat_value(pred_time):
                log.warning(
                    'Prediction anchor time is invalid or unavailable in '
                    'the stats cache for job %s', job_id)
            else:
                assert isinstance(pred_time, Timestamp)
                log.info('Using prediction time of %s', pred_time)
                self.anchor_time = pred_time.to_pydatetime()
        else:
            log.warning(
                'Prediction anchor time is not yet available in '
                'the stats cache for job %s', job_id)
        return self


@kumo_dataclass
class PaginationArgs:
    """
    Arguments to implement pagination in REST calls.
    Currently only used for list_batch_prediction_jobs REST call.
    """
    page: int = 1
    page_size: int = 50


def get_bp_columns_by_task_type(task_type: TaskType, is_temporal: bool,
                                entity_col_name: str) -> List[ColumnArgs]:
    base = [
        ColumnArgs(name=entity_col_name, stype=Stype.ID, dtype=Dtype.string),
    ]
    if task_type in [
            TaskType.REGRESSION,
            TaskType.FORECASTING,
    ]:
        return base + [
            ColumnArgs(name='TARGET_PRED', stype=Stype.numerical,
                       dtype=Dtype.float)
        ]
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return base + [
            ColumnArgs(name='TARGET_PRED', stype=Stype.categorical,
                       dtype=Dtype.bool),
            ColumnArgs(name='False_PROB', stype=Stype.numerical,
                       dtype=Dtype.float),
            ColumnArgs(name='True_PROB', stype=Stype.numerical,
                       dtype=Dtype.float)
        ]
    if task_type in [
            TaskType.MULTILABEL_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
    ]:
        return base + [
            ColumnArgs(name='PREDICTED', stype=Stype.categorical,
                       dtype=Dtype.int),
            ColumnArgs(name='CLASS', stype=Stype.categorical,
                       dtype=Dtype.string),
            ColumnArgs(name='SCORE', stype=Stype.numerical, dtype=Dtype.float)
        ]
    if task_type.is_ranking:
        return base + [
            ColumnArgs(name='CLASS', stype=Stype.categorical,
                       dtype=Dtype.string),
            ColumnArgs(name='SCORE', stype=Stype.numerical, dtype=Dtype.float)
        ]
    else:
        log.error('Invalid task type for Batch Prediction stats: %s', task_type)
        return []
