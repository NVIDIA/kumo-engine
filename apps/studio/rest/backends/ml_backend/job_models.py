# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from datetime import datetime
from enum import Enum
from math import inf
from typing import Any, Dict, List, Optional, Union

from kumo_core.model_plan import ModelPlan, TrainingTableGenerationPlan
from kumo_core.types.schema import FPID
from kumoapi.data_snapshot import GraphSnapshotID, TableSnapshotID
from kumoapi.jobs import CustomTrainingTable
from kumoapi.model_plan import PredictionTableGenerationPlan
from pydantic import field_validator
from temporalio.client import WorkflowExecutionStatus

from io_utils import kumo_dataclass
from kumo.data.cache.graph_cache import GraphVersion
from kumo.validation.validation import ValidationResponse
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.error_handling import extract_failure_info
from rest.config.training_job import Metric, TrainingJobOutput
from rest.deps.authentication import RequestContext
from statedb.config.common import (
    BatchPredictionJobID,
    DBEntry,
    GeneratePredictionTableJobID,
    GenerateTrainTableJobID,
    JobEventLogEntry,
    JobExecution,
    JobSource,
    JobStatus,
    JobType,
    PredictiveQueryID,
    ProjectID,
    TrainingJobID,
)
from statedb.config.job_tag import TAG_KEY_ARCHIVED, TAG_VALUE_ARCHIVED
from statedb.ml.jobs import BatchPredictionJobDBEntry, MLJobRegistryMetadata
from statedb.rest import rest_api_db_entries as rest_db


class JobSortField(str, Enum):
    """Enum for job list sorting fields."""

    JOB_ID = "job_id"
    JOB_TYPE = "metadata.job_type"
    START_TIME = "metadata.start_time"
    STATUS = "metadata.status"
    SOURCE = "source"
    DURATION = "duration"
    TAGS = "metadata.tags"
    TUNE_METRIC = "test_metric"


@kumo_dataclass
class JobFilterMetadata:
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    created_at: datetime
    job_type: JobType
    status: JobStatus
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def ordering_and_filter_time(self) -> datetime:
        """Use submission time until a job has been admitted to run."""
        return self.start_time or self.created_at


@kumo_dataclass
class GenericJob:
    job_id: Union[
        TrainingJobID,
        BatchPredictionJobID,
        GenerateTrainTableJobID,
        GeneratePredictionTableJobID,
        GraphSnapshotID,
        TableSnapshotID,
    ]
    job_type: JobType
    execution: JobExecution
    metadata: JobFilterMetadata
    project_id: ProjectID
    source: Optional[JobSource] = None
    user: Optional[str] = None
    # Field only present for training jobs. Added here to avoid having to
    # fetch the exact same data in a separate endpoint, and unneccessarily
    # add latency.
    test_metric: Optional[Metric] = None

    def __lt__(self, other: 'GenericJob') -> bool:
        return (self.metadata.ordering_and_filter_time
                > other.metadata.ordering_and_filter_time)


@kumo_dataclass
class ListJobsQuery:
    project_id: ProjectID
    limit: int = 50
    page: int = 0
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    status: List[JobStatus] = field(default_factory=list)
    job_type: List[JobType] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    tags_filter: List[str] = field(default_factory=list)
    search_keyword: Optional[str] = ""
    sort_by: Optional[JobSortField] = None
    sort_desc: bool = False

    @field_validator('job_type')
    @classmethod
    def _default_all_job_types(cls, v: List):
        if len(v) == 0:
            return [jt for jt in JobType]
        return v


class ListJobsProcessor:
    r"""Utility class to handle filtering, sorting and pagination of job
    lists.
    """
    @staticmethod
    def apply_filters(query: ListJobsQuery,
                      jobs: List[GenericJob]) -> List[GenericJob]:
        """Apply filters to list of jobs based on query parameters.
        Filters jobs by:
            - Time range (start and end)
            - Job status
            - Job type
            - Tags
            - Search keyword

        Args:
            query: Query parameters for filtering
            jobs: List of GenericJob objects to filter

        Returns:
            List[GenericJob]: Filtered list of jobs matching all criteria
        """
        jobs = [job for job in jobs if job.project_id == query.project_id]

        # Apply filters
        if query.time_range_start is not None:
            jobs = [
                job for job in jobs if job.metadata.ordering_and_filter_time >=
                query.time_range_start
            ]
        if query.time_range_end is not None:
            jobs = [
                job for job in jobs
                if job.metadata.ordering_and_filter_time <= query.time_range_end
            ]
        if len(query.status) > 0:
            jobs = [job for job in jobs if job.metadata.status in query.status]
        if len(query.job_type) > 0:
            jobs = [
                job for job in jobs if job.metadata.job_type in query.job_type
            ]
        if len(query.tags_filter) > 0:
            jobs_out = []
            # job id set to keep track of jobs already appended to jobs_out
            job_ids_out = set()
            for tag in query.tags_filter:
                for job in jobs:
                    if job.job_id in job_ids_out:
                        continue
                    jobs_tags = job.metadata.tags.values()
                    if tag in jobs_tags:
                        job_ids_out.add(job.job_id)
                        jobs_out.append(job)
            jobs = jobs_out
        want_archived = TAG_VALUE_ARCHIVED in query.tags_filter
        jobs = [
            job for job in jobs
            if (job.metadata.tags.get(TAG_KEY_ARCHIVED) == TAG_VALUE_ARCHIVED
                ) == want_archived
        ]
        if query.search_keyword is not None and query.search_keyword != "":
            search_term = query.search_keyword.lower()
            jobs = [
                job for job in jobs
                if ListJobsProcessor._matches_search_criteria(job, search_term)
            ]
        return jobs

    @staticmethod
    def apply_sorting(query: ListJobsQuery,
                      jobs: List[GenericJob]) -> List[GenericJob]:
        """Apply sorting to list of jobs based on query parameters.

        Args:
            query: Query parameters for sorting
            jobs: List of GenericJob objects to sort

        Returns:
            List[GenericJob]: Sorted list of jobs based on sort_by and
                sort_desc parameters. If sort_by is None, returns jobs
                sorted by default ordering.

        Notes:
            - For duration sorting, jobs with no end_time are placed at
                start/end based on sort order.
            - For tag sorting, concatenates sorted tag values.
            - For other fields, supports dot notation to access nested
                attributes.
        """
        sort_by = query.sort_by
        if sort_by is None:
            return sorted(jobs)

        # The str value is accessed via sort_by.value
        def get_sort_key(job: GenericJob):
            # Handle nested fields with dot notation
            if sort_by == JobSortField.DURATION:
                # Duration is end_time - start_time if end_time exists
                if not job.metadata.end_time or not job.metadata.start_time:
                    # Put jobs with no end_time at the start/end based on sort
                    # order
                    return float('-inf')
                return (job.metadata.end_time -
                        job.metadata.start_time).total_seconds()
            elif sort_by == JobSortField.START_TIME:
                return job.metadata.ordering_and_filter_time
            elif sort_by == JobSortField.TAGS:
                # For tags, sort by concatenated dictionary values
                return ",".join(
                    sorted(str(v) for v in job.metadata.tags.values()))
            elif sort_by == JobSortField.TUNE_METRIC:
                test_metric = job.test_metric
                if test_metric is None or test_metric.val is None:
                    # use -inf and inf to place None values at the end of the sorted list
                    return (0, -inf) if query.sort_desc else (1, inf)
                val = test_metric.val
                return (1, val) if query.sort_desc else (0, val)
            else:
                # Handle other fields with dot notation
                assert sort_by is not None
                keys = sort_by.value.split('.')  # Use .value to get string
                value = job
                for key in keys:
                    value = getattr(value, key)
                return value

        jobs = sorted(jobs, key=get_sort_key, reverse=query.sort_desc)
        return jobs

    @staticmethod
    def apply_pagination(query: ListJobsQuery,
                         jobs: List[GenericJob]) -> List[GenericJob]:
        """Apply pagination to list of jobs based on query parameters.

        Args:
            query: Query parameters for pagination
            jobs: List of GenericJob objects to paginate

        Returns:
            List[GenericJob]: Paginated subset of jobs based on page and
                limit parameters.

        Notes:
            - Returns a slice of jobs from index (page * limit) to
              ((page + 1) * limit).
            - If page or limit results in indices beyond list bounds,
              returns empty list.
        """
        start_idx = query.page * query.limit
        end_idx = (query.page + 1) * query.limit
        return jobs[start_idx:end_idx]

    @staticmethod
    def _matches_search_criteria(job: GenericJob, search_term: str) -> bool:
        """Check if a job matches the search criteria.

        Args:
            job: The job to check
            search_term: The lowercase search term to match against

        Returns:
            bool: True if the job matches any of the search criteria,
                False otherwise

        Note:
            The search is case-insensitive and matches against:
            - Job ID
            - Job type (with underscores replaced by spaces)
            - Job source (if available)
            - User (if available)
            - Any tag values
        """
        # Match job ID
        matches_job_id = search_term in job.job_id.lower()

        # Match job type
        matches_job_type = search_term in job.metadata.job_type.lower().replace(
            "_", " ")

        # Match job source
        matches_source = job.source is not None and search_term in job.source.lower(
        )

        # Match user (if available)
        matches_user = (job.user is not None
                        and search_term in job.user.lower())

        # Match any tag value
        # NOTE: we add tags in keyword search to unblock for now, if we decided
        # to add tags search in the UI, we need to deprecate this logic.
        matches_tags = any(search_term in str(tag_value).lower()
                           for tag_value in job.metadata.tags.values())

        return (matches_job_id or matches_job_type or matches_source
                or matches_user or matches_tags)


@kumo_dataclass
class TrainingJobDBEntry(DBEntry):
    id: TrainingJobID
    execution: TrainingJobOutput
    project_id: ProjectID
    # NOTE: Order is important, ModelPlan must go first due to pydantic
    # resolution
    advanced_options: Optional[ModelPlan
                               | rest_db.DeprecatedAdvancedAutoTrainerOptions] = None

    # Record which graph data version this training job was executing over.
    graph_data_version: Optional[str] = None

    # Job id that generates the train table.
    # This field would be missing for legacy training jobs that were created
    # prior to decoupling generate-train-table job from training job.
    generate_train_table_job_id: Optional[str] = None

    # Key of an immutable `PQueryDefinition` frozen at the time of job launch.
    pquery_def_fpid: Optional[FPID] = None

    # PredictiveQueryID captured at launch for MLOps filtering.
    pquery_id: Optional[PredictiveQueryID] = None

    # Path to the job-specific artifact directory.
    storage_path: Optional[str] = None

    # JSON serialized InputDataStats.
    input_data_stats: Dict[str, Any] = field(default_factory=dict)

    # Job source -- REST, WEB, or SDK.
    source: Optional[JobSource] = None

    custom_train_table_used: bool = False

    # Durable, secret-free ownership reference. This lets accepted work resume
    # safely and lets connector deletion detect nonterminal consumers.
    custom_train_table_connector_ref: Optional[ConnectorExecutionRef] = None

    # Optional training job ID to warm start from.
    warm_start_job_id: Optional[str] = None

    # The Kumo version training was performed on;
    # Optional to assure backward compatability.
    kumo_version: Optional[str] = None
    # Field to retain information about user that triggered the job.
    # Optional because old jobs don't have this field.
    user: Optional[str] = None

    # Legacy-compatible registry metadata used by MLOps queries.
    registry_metadata: Optional[MLJobRegistryMetadata] = None

    def to_generic(self, ctx: RequestContext) -> GenericJob:
        self.execution.remove_employee_only_fields()

        tune_test_metric = None
        if isinstance(self.advanced_options, ModelPlan):
            tune_metric = self.advanced_options.training_job.tune_metric
            test_metrics = (self.execution.results.test_metrics
                            if self.execution.results is not None else None)
            if tune_metric and test_metrics:
                # When refit_trainval is enabled, prefer the refit metric over the regular test metric
                refit_trainval = self.advanced_options.training_job.refit_trainval
                target_metric_name = f"refit_{tune_metric}" if refit_trainval else tune_metric

                for metric in test_metrics:
                    if metric.name == target_metric_name:
                        tune_test_metric = metric
                        break

        return GenericJob(
            job_id=self.id,
            job_type=JobType.TRAINING_JOB,
            execution=self.execution,
            metadata=JobFilterMetadata(
                start_time=self.execution.start_time,
                end_time=self.execution.last_updated_at
                if self.execution.status.is_terminal else None,
                created_at=self.timestamps.created_at,
                job_type=JobType.TRAINING_JOB,
                status=self.execution.status,
            ),
            source=self.source,
            user=self.user,
            project_id=self.project_id,
            test_metric=tune_test_metric,
        )


@kumo_dataclass
class GenerateTrainTableJobConfig:
    pquery_id: PredictiveQueryID
    plan: TrainingTableGenerationPlan
    graph_id: str  # GraphIDv1 | GraphIDv2
    graph_data_version: Optional[GraphVersion] = None

    # Key of an immutable `PQueryDefinition` frozen at the time of job launch.
    pquery_def_fpid: Optional[FPID] = None

    # The table a *registering* job recorded rather than generated, exactly as
    # the caller named it. Durable because it is the only place the answer
    # survives: the SDK reconstructs a `TrainingTable` from a job ID alone,
    # and reads back off this record which rows a later `fit()` must train on.
    # Without it, a registered table read back as a generated one and the run
    # silently generated its own rows.
    #
    # Deliberately the *public* `kumoapi` shape rather than the internal
    # `rest.config.training_job.CustomTrainingTable`: the internal one carries
    # a `ConnectorExecutionRef`, whose `authorization_token` must not be
    # written into a job payload or echoed back over REST. The connector this
    # job owns is recorded separately, on the entry below.
    custom_train_table: Optional[CustomTrainingTable] = None


@kumo_dataclass
class GenerateTrainTableJobOutput(JobExecution):
    # Set by StateDbJobQueue at admission, never at submission.
    start_time: Optional[datetime] = None
    # Result fields produced after job succeeds.
    train_table_path: Optional[str] = None
    train_table_materialization_path: Optional[str] = None
    validation: Optional[ValidationResponse] = None
    train_table_home_dir: Optional[str] = None

    @classmethod
    def failed(
        cls,
        err_msg: Optional[str] = None,
        stacktrace: Optional[str] = None,
        exception: Optional[Exception] = None,
        validation: Optional['ValidationResponse'] = None,
        workflow_end_state: Optional["WorkflowExecutionStatus"] = None,
    ) -> 'GenerateTrainTableJobOutput':
        """Create failed job output with proper failure_info.

        Encapsulates failure_info extraction - callers just pass raw error data.

        Args:
            err_msg: User-facing error message
            stacktrace: Debug stacktrace
            exception: Exception object
            validation: ValidationResponse (if validation failed)
            workflow_end_state: Optional workflow end state (e.g., FAILED, COMPLETED)

        Returns:
            GenerateTrainTableJobOutput with status=FAILED and failure_info populated
        """
        failure_info = extract_failure_info(
            err_msg=err_msg,
            stacktrace=stacktrace,
            exception=exception,
            validation=validation,
        )

        if not failure_info.message:
            failure_info.message = "Generating train table failed"

        return cls(
            status=JobStatus.FAILED,
            log=JobEventLogEntry.failed(message=failure_info.message),
            failure_info=failure_info,
            workflow_end_state=workflow_end_state,
        )


@kumo_dataclass
class GenerateTrainTableJobDBEntry(DBEntry):
    id: GenerateTrainTableJobID

    # configuration of the gen-train-table job.
    config: GenerateTrainTableJobConfig

    # job execution state.
    execution: GenerateTrainTableJobOutput
    project_id: ProjectID

    # Job source -- REST, WEB, or SDK.
    source: Optional[JobSource] = None

    # Field to retain information about user that triggered the job.
    # Optional because old jobs don't have this field.
    user: Optional[str] = None

    # Path to the job-specific artifact directory.
    storage_path: Optional[str] = None
    # Legacy-compatible registry metadata used by MLOps queries.
    registry_metadata: Optional[MLJobRegistryMetadata] = None

    # Whether this job registered supplied rows rather than generating them.
    # Separate from `config.custom_train_table` on purpose: it is what lets a
    # reader tell "this job had no custom table" from "this record cannot say
    # what it had", which are the same `None` otherwise. Mirrors
    # `TrainingJobDBEntry.custom_train_table_used`.
    custom_train_table_used: bool = False

    # Durable, secret-free ownership reference, as on a training job: it lets
    # accepted work resume safely and lets connector deletion detect
    # nonterminal consumers.
    custom_train_table_connector_ref: Optional[ConnectorExecutionRef] = None

    def to_generic(self, ctx: RequestContext) -> GenericJob:
        return GenericJob(
            job_id=self.id,
            job_type=JobType.GENERATE_TRAIN_TABLE_JOB,
            execution=self.execution,
            metadata=JobFilterMetadata(
                start_time=self.execution.start_time,
                end_time=self.execution.last_updated_at
                if self.execution.status.is_terminal else None,
                created_at=self.timestamps.created_at,
                job_type=JobType.GENERATE_TRAIN_TABLE_JOB,
                status=self.execution.status,
            ),
            source=self.source,
            user=self.user,
            project_id=self.project_id,
        )


@kumo_dataclass
class GeneratePredTableJobConfig:
    pquery_id: PredictiveQueryID
    plan: PredictionTableGenerationPlan
    graph_id: str  # GraphIDv1 | GraphIDv2
    graph_data_version: Optional[GraphVersion] = None

    # Key of an immutable `PQueryDefinition` frozen at the time of job launch.
    pquery_def_fpid: Optional[FPID] = None


@kumo_dataclass
class GeneratePredTableJobOutput(JobExecution):
    # Set by StateDbJobQueue at admission, never at submission.
    start_time: Optional[datetime] = None
    # Result fields produced after job succeeds.
    pred_table_path: Optional[str] = None
    validation: Optional[ValidationResponse] = None

    @classmethod
    def failed(
        cls,
        err_msg: Optional[str] = None,
        stacktrace: Optional[str] = None,
        exception: Optional[Exception] = None,
        validation: Optional['ValidationResponse'] = None,
        workflow_end_state: Optional["WorkflowExecutionStatus"] = None,
    ) -> 'GeneratePredTableJobOutput':
        """Create failed job output with proper failure_info.

        Encapsulates failure_info extraction - callers just pass raw error data.

        Args:
            err_msg: User-facing error message
            stacktrace: Debug stacktrace
            exception: Exception object
            validation: ValidationResponse (if validation failed)
            workflow_end_state: Optional workflow end state (e.g., FAILED, COMPLETED)

        Returns:
            GeneratePredTableJobOutput with status=FAILED and failure_info populated
        """
        failure_info = extract_failure_info(
            err_msg=err_msg,
            stacktrace=stacktrace,
            exception=exception,
            validation=validation,
        )

        if not failure_info.message:
            failure_info.message = "Generating prediction table failed"

        return cls(
            status=JobStatus.FAILED,
            log=JobEventLogEntry.failed(message=failure_info.message),
            failure_info=failure_info,
            workflow_end_state=workflow_end_state,
        )


@kumo_dataclass
class GeneratePredTableJobDBEntry(DBEntry):
    id: GeneratePredictionTableJobID

    # configuration of the gen-prediction-table job.
    config: GeneratePredTableJobConfig

    # job execution state.
    execution: GeneratePredTableJobOutput
    project_id: ProjectID

    # Job source -- REST, WEB, or SDK.
    source: Optional[JobSource] = None

    # Field to retain information about user that triggered the job.
    # Optional because old jobs don't have this field.
    user: Optional[str] = None
    # Path to the job-specific artifact directory.
    storage_path: Optional[str] = None
    # Legacy-compatible registry metadata used by MLOps queries.
    registry_metadata: Optional[MLJobRegistryMetadata] = None

    def to_generic(self, ctx: RequestContext) -> GenericJob:
        return GenericJob(
            job_id=self.id,
            job_type=JobType.GENERATE_PREDICTION_TABLE_JOB,
            execution=self.execution,
            metadata=JobFilterMetadata(
                start_time=self.execution.start_time,
                end_time=self.execution.last_updated_at
                if self.execution.status.is_terminal else None,
                created_at=self.timestamps.created_at,
                job_type=JobType.GENERATE_PREDICTION_TABLE_JOB,
                status=self.execution.status,
            ),
            source=self.source,
            user=self.user,
            project_id=self.project_id,
        )


MLJobDBEntry = Union[
    TrainingJobDBEntry,
    BatchPredictionJobDBEntry,
    GeneratePredTableJobDBEntry,
    GenerateTrainTableJobDBEntry,
]
