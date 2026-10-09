# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import field
from datetime import datetime
from typing import Dict, List, Optional

from kumo_core.model_plan import ModelPlan
from kumo_core.types import TaskType
from kumo_core.types.schema import FPID
from kumoapi.source_table import SourceTableType
from kumoapi.train import TimestampUnit, TrainingStage
from pydantic import field_validator, model_validator
from temporalio.client import WorkflowExecutionStatus

from async_utils.sync import blocking_method
from io_utils import kumo_dataclass, logging
from io_utils.json import load_json, save_json
from kumo.config import PredictiveQueryV2Config
from kumo.config.explain import ColumnContributionSummary
from kumo.config.task import SplitConfigSerializable, TimePeriod
from kumo.utils.evaluate import BinaryConfusionMatrix
from kumo.validation.validation import ValidationResponse
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.error_handling import extract_failure_info
from service_framework.env_var import EnvVar
from statedb.config.common import (
    GraphID,
    JobEventLogEntry,
    JobExecution,
    JobFailureInfo,
    JobProgress,
    JobSource,
    JobStatus,
    PredictiveQueryID,
    TrainingJobID,
)
from statedb.config.job_tag import JobTagKey, JobTagValue

log = logging.getLogger(__name__)

KUMO_CUSTOM_TT_ALLOW_ROW_COUNT_MISMATCH = (
    'KUMO_CUSTOM_TT_ALLOW_ROW_COUNT_MISMATCH')
CUSTOM_TT_ALLOW_ROW_COUNT_MISMATCH = EnvVar(
    KUMO_CUSTOM_TT_ALLOW_ROW_COUNT_MISMATCH,
    type=bool,
    required=False,
    default_value=False,
    description=('Allow custom training tables to differ in row count from the '
                 'original generated training table during validation.'),
)


@kumo_dataclass
class CustomTrainingTable:
    """Defines the custom training table to be used for training.

    Args:
        connector_spec: The connector spec for the custom training table.
        source_table: The physical table or read-only SQL source.
        weight_col_name: The name of the weight column in the custom
            training table, or ``None`` when the table carries none. A
            custom table without weights is a complete table supplied as it
            stands; DiskGraph reads one with no weight column and every row
            weighs the same.
        split_col_name: The name of the column carrying each row's split
            assignment, or ``None`` when the run takes its split from the
            plan. Named per table because DiskGraph names it per table --
            ``split_contract.py`` takes a ``split_column`` parameter rather
            than matching a literal, so unlike ``ENTITY1`` there is no fixed
            name to assume.
        timestamp_unit: Unit of integer TIMESTAMP values in the supplied table.
        validated: Whether the custom training table has been validated.
    """
    connector_spec: ConnectorExecutionRef
    source_table: SourceTableType
    weight_col_name: Optional[str] = None
    split_col_name: Optional[str] = None
    validated: bool = False
    timestamp_unit: TimestampUnit = 'nanoseconds'


@kumo_dataclass
class TrainingSpec:
    """Defines the business logic/problem formulation for a training job.
    """
    predictive_query: PredictiveQueryV2Config
    advanced_options: ModelPlan

    # These fields are used for debug purpose only. You should NOT use these
    # IDs to directly look up data in StateDB, because StateDB may have
    # changed since the training job was launched.
    predictive_query_id: PredictiveQueryID
    graph_id: GraphID
    split: Optional[SplitConfigSerializable] = None

    task_type: Optional[TaskType] = None
    custom_train_table: Optional[CustomTrainingTable] = None

    # Stop the training job after performing materialization. This is used for
    # writing unit tests against the materialization code.
    stop_after_materialization: bool = False


@kumo_dataclass
class Metric:
    name: str
    val: Optional[float]

    # Convert inf/nan to None as they are not supported by json
    @field_validator('val')
    @classmethod
    def ensure_valid_float(cls, val: Optional[float]) -> Optional[float]:
        if val is None or math.isnan(val) or math.isinf(val):
            return None
        return val


@kumo_dataclass
class TrialProgress:
    trial_id: str
    # The best metric value across all the epochs until now.
    # "Best" is defined by the type of metric. For example, it's the maximum
    # for accuracy, but minimum for sMAPE etc.

    # This will be :obj:`None` during the time when the trial has started but
    # zero epochs completed. This will never be :obj:`None` for trials that
    # are finished.
    best_train_metric: Optional[Metric] = None
    best_validation_metric: Optional[Metric] = None


@kumo_dataclass
class TrainingJobProgress(JobProgress):
    # Contains the metrics for each trial that is running/completed.
    trial_progress: List[TrialProgress] = field(default_factory=list)
    metric_mode: Optional[str] = None  # {max, min}


@kumo_dataclass
class TestEvalResults:
    confusion_matrix: BinaryConfusionMatrix


@kumo_dataclass
class TestExplanation:
    __test__ = False
    contribution_summary: ColumnContributionSummary


@kumo_dataclass
class TimePeriods:
    __test__ = False
    test: Optional[TimePeriod]
    val: Optional[TimePeriod]
    train: Optional[TimePeriod]


@kumo_dataclass
class TestPeriod:
    __test__ = False
    test_start: datetime
    test_end: datetime


@kumo_dataclass
class SplitSizes:
    """Number of training/eval examples in each of the splits
    """
    train: int
    val: int
    test: Optional[int] = None


@kumo_dataclass
class TrainingJobResults:
    total_training_time_sec: int = 0
    total_iterations: int = 0

    # Metrics on test split
    test_metrics: Optional[List[Metric]] = None
    # Metrics on validation split
    val_metrics: Optional[List[Metric]] = None
    # Metrics on train split
    train_metrics: Optional[List[Metric]] = None

    # TODO(siyang): Protect against direct access to "test_explain" field.
    test_explain: Optional[TestExplanation] = None
    # If test_explain data is too large, this field points to the file path
    # from which we can load test_explain data
    _test_explain_file_path: Optional[str] = None

    # present for binary classification
    test_eval: Optional[TestEvalResults] = None
    # present for temporal pquery. Ranges of anchor time.
    time_periods: Optional[TimePeriods] = None
    # present for temporal pquery. Ranges of the actual time ranges used
    # to generate prediction labels. Will be same as `time_periods`` if
    # target aggregation in PQL starts from `0` eg. `SUM(col, 0, 10)`.
    prediction_horizons: Optional[TimePeriods] = None

    split_sizes: Optional[SplitSizes] = None

    # This field is deprecated, and is replaced by time_periods
    # TODO: figure out how to actually deprecate this from the DB
    test_period: Optional[TestPeriod] = None

    def with_metrics(self, metrics: List[Metric]) -> 'TrainingJobResults':
        # Parse test_* and val_* metrics into their respective fields.
        self.test_metrics = []
        self.val_metrics = []
        self.train_metrics = []
        remaining_metrics = []
        for metric in metrics:
            stage, name = TrainingStage.split_staged_name(metric.name)
            if stage == TrainingStage.TEST:
                self.test_metrics.append(Metric(name, metric.val))
            elif stage == TrainingStage.VAL:
                self.val_metrics.append(Metric(name, metric.val))
            elif stage == TrainingStage.TRAIN:
                self.train_metrics.append(Metric(name, metric.val))
            else:
                remaining_metrics.append(metric)
        if remaining_metrics:
            raise RuntimeError(
                f"All metrics must be prefixed by 'train', 'test', or 'val'. "
                f"Got {remaining_metrics}. "
                "This usually indicates a trainer bug or a custom metric without a stage prefix."
            )
        return self

    # Read test_explain field by optionally loading from external file path.
    @blocking_method()
    def load_test_explain(self) -> TestExplanation:
        if not self.test_explain:
            assert self._test_explain_file_path
            self.test_explain = load_json(self._test_explain_file_path,
                                          TestExplanation, compressed=True)
        return self.test_explain

    @blocking_method()
    def save_test_explain_to_remote(self, test_explain_file_path: str) -> None:
        assert self.test_explain
        save_json(self.test_explain, test_explain_file_path, compressed=True)
        self._test_explain_file_path = test_explain_file_path
        self.test_explain = None


@kumo_dataclass
class TrainingJobOutput(JobExecution):
    # Set by StateDbJobQueue at admission, never at submission.
    start_time: Optional[datetime] = None
    # present if status=TRAINING
    progress: Optional[TrainingJobProgress] = None
    # present if status=DONE
    results: Optional[TrainingJobResults] = None

    @classmethod
    def failed(
        cls, err_msg: Optional[str] = None, stacktrace: Optional[str] = None,
        exception: Optional[BaseException] = None,
        validation: Optional[ValidationResponse] = None,
        workflow_end_state: Optional['WorkflowExecutionStatus'] = None
    ) -> 'TrainingJobOutput':
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
            TrainingJobOutput with status=FAILED and failure_info populated
        """
        failure_info = extract_failure_info(err_msg=err_msg,
                                            stacktrace=stacktrace,
                                            exception=exception,
                                            validation=validation)

        if not failure_info.message:
            failure_info.message = "Training job failed"

        return cls(status=JobStatus.FAILED,
                   log=JobEventLogEntry.failed(message=failure_info.message),
                   failure_info=failure_info,
                   workflow_end_state=workflow_end_state)

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
        progress: Optional[TrainingJobProgress] = self.progress
        results: Optional[TrainingJobResults] = self.results
        failure_info: Optional[JobFailureInfo] = self.failure_info

        violations: List[str] = []

        if status == JobStatus.RUNNING and (log_entry and log_entry.stage_name
                                            == 'Training'):
            if not progress:
                violations.append(
                    f"Missing progress field for running training job "
                    f"(status={JobStatus.RUNNING}). Ensure the training job "
                    f"reporter is writing progress.")

        if status == JobStatus.DONE:
            if not results:
                violations.append(
                    f"Missing results field for completed training job "
                    f"(status={JobStatus.DONE}). Ensure the postprocessing "
                    f"step completed successfully.")

        if status == JobStatus.FAILED:
            if not failure_info:
                violations.append(
                    f"Missing failure_info field for failed training job "
                    f"(status={JobStatus.FAILED}). Ensure errors are captured "
                    f"and propagated from the training workflow.")

        return violations

    # v1 @root_validator (post-validation) -> @model_validator(mode='after').
    @model_validator(mode='after')
    def _check_invariants(self) -> 'TrainingJobOutput':
        # These are soft progress-completeness diagnostics here, not structural
        # data-integrity gates. Under pydantic v1 the equivalent
        # ``@root_validator`` only ran once, at the object's own construction.
        # Under v2 a ``mode='after'`` validator also re-runs whenever an
        # already-constructed instance is re-validated -- e.g. when a mutated
        # ``TrainingJobOutput`` is nested into a parent model on a read/response
        # path (``_training_job_response_from_entry`` re-wraps ``job.execution``
        # after flipping ``status``). Hard-raising there turns a build-time
        # sanity check into a fatal error on legitimate read/mutation flows.
        # Log a warning instead of raising so these flows keep working while
        # the diagnostic signal is retained.
        #
        # The strict v1 WRITE-time enforcement is restored only at the persist
        # chokepoint (``MLJobService.modify_job_if_open_generic``), which calls
        # ``terminal_invariant_violations`` and raises so a broken terminal job
        # cannot be saved. The read/response path stays warn-only.
        for violation in self.terminal_invariant_violations():
            log.warning("%s", violation)
        return self

    def strip_out_expensive_fields(self):
        """Remove all large/expensive fields that may cause RPC message to be
        too large for TemporalIO, especially in batch requests.
        """
        if self.results is not None:
            self.results.test_explain = None
            self.results.test_eval = None

    def remove_employee_only_fields(self) -> None:
        """Delete fields that should not be shown to customers.
        """
        if self.failure_info is not None:
            self.failure_info.remove_employee_only_fields()


@kumo_dataclass
class TrainingJobResponse:  # object model for lookups from MLJobBackend
    id: TrainingJobID
    execution: TrainingJobOutput
    project_id: str
    # Immutable submission time from the StateDB entry.
    created_at: Optional[datetime] = None
    advanced_options: Optional[ModelPlan] = None

    # Fields below are associated identifiers related to this job.
    #
    # Looked up from secondary/optional ML SQL DB therefore might be absent.
    pquery_id: Optional[PredictiveQueryID] = None

    # ID to an immutable copy of PQueryDefinition frozen at job launch time.
    # This field should always be present for future jobs, but missing for
    # old/existing jobs launched in the past.
    pquery_def_fpid: Optional[FPID] = None

    # Which graph data version this training job was executing over. Absent if
    # 1) older jobs never recorded the data version; or 2) the training job
    # has not yet refreshed & saved the graph data version to DB.
    graph_data_version: Optional[str] = None

    # Job id that generates the train table.  Absent for older jobs that were
    # executed before we decouple train-table-generation from training.
    generate_train_table_job_id: Optional[str] = None

    job_source: Optional[JobSource] = None

    # Tags associated with the job.
    tags: Optional[Dict[JobTagKey, JobTagValue]] = None

    # Wether a custom train table is used for this training job
    custom_train_table: Optional[bool] = None

    # Optional training job ID to warm start from.
    warm_start_job_id: Optional[str] = None

    # The Kumo version training was performed on;
    # Optional to assure backward compatability.
    # Model Config also contains the kumo version.
    kumo_version: Optional[str] = None
    # Field to retain information about user that triggered the job.
    # Optional because old jobs don't have this field.
    user: Optional[str] = None
