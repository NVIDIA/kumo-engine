# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Collection, Iterable, Mapping

from kumo_core.model_plan import ModelPlan
from kumo_core.types.schema import FPID
from kumo_core.version import get_kumo_version
from temporalio.client import WorkflowExecutionStatus

from io_utils import logging
from rest.backends.ml_backend.job_models import (
    GeneratePredTableJobConfig,
    GeneratePredTableJobDBEntry,
    GeneratePredTableJobOutput,
    GenerateTrainTableJobConfig,
    GenerateTrainTableJobDBEntry,
    GenerateTrainTableJobOutput,
    GenericJob,
    JobSortField,
    ListJobsProcessor,
    ListJobsQuery,
    MLJobRegistryMetadata,
    TrainingJobDBEntry,
)
from rest.config.batch_prediction_job import (
    BatchPredictionJobConfig,
    BatchPredictionJobOutput,
)
from rest.config.training_job import CustomTrainingTable, TrainingJobOutput
from statedb.config.common import (
    BatchPredictionJobID,
    BatchPredictionWorkflowID,
    GeneratePredictionTableJobID,
    GenerateTrainTableJobID,
    GraphID,
    JobEventLogEntry,
    JobFailureInfo,
    JobID,
    JobSource,
    JobStatus,
    JobType,
    PredictiveQueryID,
    ProjectID,
    TrainingJobID,
    now_utc,
)
from statedb.config.job_tag import (
    _IMMUTABLE_TAG_KEYS,
    TAG_KEY_ARCHIVED,
    TAG_VALUE_ARCHIVED,
    JobTagKey,
    JobTags,
    JobTagValue,
)
from statedb.config.timestamp import ResourceTimestamps
from statedb.db.connector_admission import require_connectors_for_admission
from statedb.db.job_tags import JobTagsRepository, coerce_job_type
from statedb.db.repositories import JsonRepository
from statedb.db.row_parsing import row_field, row_str
from statedb.db.runtime import StateDB, StateTx
from statedb.ml.jobs import BatchPredictionJobDBEntry

log = logging.getLogger(__name__)


class JobInvariantViolationError(ValueError):
    """Raised when a job that violates a status->required-field invariant is
    about to be persisted (e.g. a ``DONE`` training job without ``results`` or
    a ``FAILED`` job without ``failure_info``).

    Restores the pydantic-v1 write-time guard that was downgraded to a warning
    during the v2 migration: the per-output ``mode='after'`` validators stay
    warn-only (v2 re-runs them on read/response paths, where a hard-raise would
    be a false positive), while persistence is gated here so a broken terminal
    job can never be written to StateDB.
    """


def _enforce_job_output_invariants(job: Any) -> None:
    """Raise if ``job.execution`` reports any status->required-field violation.

    Generic across the three job-output types that carry these invariants
    (training or batch prediction). Generate-table outputs do
    not expose ``terminal_invariant_violations`` and are skipped.
    """
    execution = getattr(job, "execution", None)
    if execution is None:
        return
    check = getattr(execution, "terminal_invariant_violations", None)
    if check is None:
        return
    violations = check()
    if violations:
        job_id = getattr(job, "id", "<unknown>")
        raise JobInvariantViolationError(f"Refusing to persist job {job_id}: " +
                                         "; ".join(violations))


_RANGE_JOB_ATTRS = {"start_time", "end_time"}
_TIME_RANGE_FORMAT = "%Y-%m-%d %H:%M:%S"
_AGGREGATE_JOB_TYPES = {
    JobType.TRAINING_JOB,
    JobType.BATCH_PREDICTION_JOB,
    JobType.GENERATE_TRAIN_TABLE_JOB,
    JobType.GENERATE_PREDICTION_TABLE_JOB,
}


@dataclass
class MLJobRegistryEntry:
    job_id: JobID
    job_type: JobType
    instance_id: str | None = None
    pquery_id: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    storage_path: str | None = None
    input_data_stats: dict[str, Any] = field(default_factory=dict)
    kumo_version: str | None = None
    pquery_str: str | None = None
    target_table_stats: dict[str, str] = field(default_factory=dict)
    pquery_plan_metrics: dict[str, str] = field(default_factory=dict)
    generator_advanced_options: str | None = None
    trained_job_id: str | None = None
    bp_filters: str | None = None
    override_model_path: str | None = None
    status: str | None = None
    stage: str | None = None


def _placeholder(tx: StateTx, position: int) -> str:
    return f"${position}" if tx.backend == "postgres" else "?"


def _json_value(tx: StateTx, field_name: str) -> str:
    if tx.backend == "postgres":
        return f"payload->>'{field_name}'"
    return f"json_extract(payload, '$.{field_name}')"


def _json_path_value(tx: StateTx, *field_names: str) -> str:
    if tx.backend == "postgres":
        return "payload #>> '{" + ",".join(field_names) + "}'"
    return "json_extract(payload, '$." + ".".join(field_names) + "')"


def _batch_prediction_has_execution_clause(tx: StateTx) -> str:
    return (f"({_json_value(tx, 'status')} IS NOT NULL OR "
            f"{_json_path_value(tx, 'execution', 'status')} IS NOT NULL)")


def _batch_prediction_countable_execution_clause(tx: StateTx) -> str:
    return f"{_json_path_value(tx, 'execution', 'status')} IS NOT NULL"


def _clone_clauses(clauses: list[str]) -> list[str]:
    return list(clauses)


def _lower_sql(expression: str) -> str:
    return f"lower({expression})"


def _like_pattern(value: str) -> str:
    escaped = (value.replace("\\", "\\\\").replace("%",
                                                   "\\%").replace("_", "\\_"))
    return f"%{escaped}%"


def _job_id_from_raw(job_type: JobType, id: str) -> JobID:
    if job_type == JobType.TRAINING_JOB:
        return TrainingJobID(id)
    if job_type == JobType.BATCH_PREDICTION_JOB:
        return BatchPredictionJobID(id)
    if job_type == JobType.GENERATE_TRAIN_TABLE_JOB:
        return GenerateTrainTableJobID(id)
    if job_type == JobType.GENERATE_PREDICTION_TABLE_JOB:
        return GeneratePredictionTableJobID(id)
    raise AssertionError(f"Unsupported job type {job_type}")


class MLJobService:
    """State DB service boundary for ML job metadata updates."""
    def __init__(self, state_db: StateDB) -> None:
        self._state_db = state_db
        self._training_jobs = JsonRepository("training_jobs",
                                             TrainingJobDBEntry)
        self._batch_prediction_jobs = JsonRepository(
            "batch_prediction_jobs",
            BatchPredictionJobDBEntry,
        )
        self._generate_train_table_jobs = JsonRepository(
            "generate_train_table_jobs",
            GenerateTrainTableJobDBEntry,
        )
        self._generate_pred_table_jobs = JsonRepository(
            "generate_pred_table_jobs",
            GeneratePredTableJobDBEntry,
        )
        self._job_tags_repository = JobTagsRepository()

    @staticmethod
    def _require_same_project(
        referenced: Any,
        project_id: ProjectID,
        resource_name: str,
    ) -> None:
        if referenced.project_id != project_id:
            raise ValueError(f'{resource_name} belongs to project '
                             f"'{referenced.project_id}', not '{project_id}'")

    async def _require_connector_ref_in_project(
        self,
        tx: StateTx,
        reference,
        project_id: ProjectID,
    ) -> None:
        if reference.project_id != project_id:
            raise ValueError(
                f"Connector '{reference.connector_id}' belongs to another "
                'project')
        await require_connectors_for_admission(tx, project_id,
                                               [reference.connector_id])

    async def _require_batch_output_connectors_in_project(
        self,
        tx: StateTx,
        config: BatchPredictionJobConfig,
        project_id: ProjectID,
    ) -> None:
        from statedb.config.common import ConnectorID

        connector_ids = sorted(
            {
                ConnectorID(str(raw_connector_id))
                for output in config.outputs if (raw_connector_id := getattr(
                    output, 'connector_id', None)) is not None
            } | ({ConnectorID(config.custom_pred_table_source.connector_id)}
                 if config.custom_pred_table_source is not None else set()),
            key=str,
        )
        await require_connectors_for_admission(tx, project_id, connector_ids)

    async def save_train_table_graph_version(
        self,
        id: GenerateTrainTableJobID,
        graph_data_version: str,
    ) -> None:
        async with self._state_db.write_tx() as tx:
            job = await self._get_required(tx, self._generate_train_table_jobs,
                                           id)
            self._set_graph_data_version(job, graph_data_version)
            await self._modify_if_open(tx, self._generate_train_table_jobs, job)

    async def save_training_job_graph_version(
        self,
        id: TrainingJobID,
        graph_data_version: str,
    ) -> None:
        async with self._state_db.write_tx() as tx:
            job = await self._get_if_exists(tx, self._training_jobs, id)
            if job is None:
                raise ValueError(f"Training Job with ID '{id}' does not exist.")
            assert (job.graph_data_version is None
                    or job.graph_data_version == graph_data_version)
            job.graph_data_version = graph_data_version
            await self._modify_if_open(tx, self._training_jobs, job)

    async def save_pred_job_graph_version(
        self,
        id: BatchPredictionJobID,
        graph_data_version: str,
    ) -> None:
        async with self._state_db.write_tx() as tx:
            job = await self._get_if_exists(tx, self._batch_prediction_jobs, id)
            if job is None:
                raise ValueError(
                    f"Batch Prediction Job with ID '{id}' does not exist.")
            assert job.config
            assert (job.config.graph_data_version is None
                    or job.config.graph_data_version == graph_data_version)
            job.config.graph_data_version = graph_data_version
            await self._modify_if_open(tx, self._batch_prediction_jobs, job)

    async def save_pred_table_graph_version(
        self,
        id: GeneratePredictionTableJobID,
        graph_data_version: str,
    ) -> None:
        async with self._state_db.write_tx() as tx:
            job = await self._get_required(tx, self._generate_pred_table_jobs,
                                           id)
            self._set_graph_data_version(job, graph_data_version)
            await self._modify_if_open(tx, self._generate_pred_table_jobs, job)

    async def save_job_storage_path(
        self,
        id: JobID,
        storage_path: str,
    ) -> None:
        async with self._state_db.write_tx() as tx:
            job = await self._get_job_if_exists_generic_in_tx(tx, id)
            if job is None:
                raise KeyError(f"Job with id {id} not found in database")
            job.storage_path = storage_path
            await self._modify_job_metadata_if_open_in_tx(tx, job)

    async def save_job_override_model_path(
        self,
        id: JobID,
        override_model_path: str,
    ) -> None:
        async with self._state_db.write_tx() as tx:
            job = await self._get_job_if_exists_generic_in_tx(tx, id)
            if job is None:
                raise KeyError(f"Job with id {id} not found in database")
            metadata = deepcopy(getattr(job, "registry_metadata", None))
            if metadata is None:
                metadata = MLJobRegistryMetadata()
            metadata.override_model_path = override_model_path
            job.registry_metadata = metadata
            await self._modify_job_metadata_if_open_in_tx(tx, job)

    async def save_job_input_data_stats(
        self,
        id: JobID,
        input_data_stats: Mapping[str, Any],
    ) -> None:
        async with self._state_db.write_tx() as tx:
            job = await self._get_job_if_exists_generic_in_tx(tx, id)
            if job is None:
                raise KeyError(f"Job with id {id} not found in database")
            if not hasattr(job, "input_data_stats"):
                raise ValueError(
                    f"Job with id {id} does not support input_data_stats")
            job.input_data_stats = dict(input_data_stats)
            await self._modify_job_metadata_if_open_in_tx(tx, job)

    async def reserve_training_job_id(self) -> TrainingJobID:
        while True:
            id = TrainingJobID.make_random_id()
            async with self._state_db.read_tx() as tx:
                existing = await self._get_if_exists(tx, self._training_jobs,
                                                     id)
            if existing is None:
                return id

    async def create_training_job(
        self,
        *,
        pquery_id: PredictiveQueryID,
        pquery_def_fpid: FPID | None,
        graph_id: GraphID,
        now,
        advanced_options: ModelPlan,
        graph_data_version: str | None = None,
        generate_train_table_job_id: GenerateTrainTableJobID | None = None,
        job_source: JobSource | None = None,
        custom_train_table: CustomTrainingTable | None = None,
        warm_start_job_id: str | None = None,
        user_info: str | None = None,
        reserved_job_id: TrainingJobID | None = None,
        project_id: ProjectID,
    ) -> tuple[TrainingJobID, GenerateTrainTableJobID]:
        async with self._state_db.write_tx() as tx:
            if reserved_job_id is not None:
                id = reserved_job_id
                existing = await self._get_if_exists(tx, self._training_jobs,
                                                     id)
                if existing is not None:
                    raise ValueError(
                        f"Reserved training job id={id} is already in use.")
            else:
                while True:
                    id = TrainingJobID.make_random_id()
                    existing = await self._get_if_exists(
                        tx, self._training_jobs, id)
                    if existing is None:
                        break

            if custom_train_table is not None:
                await self._require_connector_ref_in_project(
                    tx, custom_train_table.connector_spec, project_id)
            if warm_start_job_id is not None:
                referenced_job = await self._get_if_exists(
                    tx, self._training_jobs, TrainingJobID(warm_start_job_id))
                if referenced_job is not None:
                    self._require_same_project(referenced_job, project_id,
                                               'Warm-start training job')

            if generate_train_table_job_id:
                child_job = await self._get_if_exists(
                    tx,
                    self._generate_train_table_jobs,
                    generate_train_table_job_id,
                )
                if child_job is None:
                    raise ValueError(
                        "Generate train table job is not found for "
                        f"{generate_train_table_job_id}")
                self._require_same_project(child_job, project_id,
                                           'Generate train table job')
                if child_job.execution.status in (
                        JobStatus.FAILED,
                        JobStatus.CANCELLED,
                ):
                    raise ValueError(
                        "Unable to create a training job, using a previously "
                        f"{child_job.execution.status} generate-train-table "
                        f"job id={generate_train_table_job_id}")
                if child_job.config.pquery_id != pquery_id:
                    raise ValueError(
                        f"Unable to create training job for pquery={pquery_id} "
                        "while using training table that was generated for a "
                        f"different pquery={child_job.config.pquery_id}")
            else:
                train_table_plan = advanced_options.training_table_generation
                generate_train_table_job_id = (
                    GenerateTrainTableJobID.as_child_of(id))
                await self._create_generate_train_table_job_in_tx(
                    tx,
                    id=generate_train_table_job_id,
                    config=GenerateTrainTableJobConfig(
                        pquery_id,
                        train_table_plan,
                        graph_id,
                        graph_data_version,
                        pquery_def_fpid,
                    ),
                    now=now,
                    source=job_source,
                    user_info=user_info,
                    project_id=project_id,
                )

            await self._training_jobs.upsert(
                tx,
                str(id),
                TrainingJobDBEntry(
                    id=id,
                    pquery_def_fpid=pquery_def_fpid,
                    pquery_id=pquery_id,
                    timestamps=ResourceTimestamps(now, now),
                    advanced_options=advanced_options,
                    execution=TrainingJobOutput(
                        status=JobStatus.QUEUED,
                        log=JobEventLogEntry.not_started(now),
                        start_time=None,
                        last_updated_at=now,
                    ),
                    graph_data_version=graph_data_version,
                    generate_train_table_job_id=generate_train_table_job_id,
                    source=job_source,
                    custom_train_table_used=custom_train_table is not None,
                    custom_train_table_connector_ref=(
                        custom_train_table.connector_spec
                        if custom_train_table is not None else None),
                    warm_start_job_id=warm_start_job_id,
                    kumo_version=get_kumo_version(),
                    user=user_info,
                    project_id=project_id,
                ),
            )
        return id, generate_train_table_job_id

    async def create_generate_train_table_job(
        self,
        *,
        config: GenerateTrainTableJobConfig,
        now=None,
        id: GenerateTrainTableJobID | None = None,
        source=None,
        user_info: str | None = None,
        project_id: ProjectID,
        custom_train_table: CustomTrainingTable | None = None,
    ) -> GenerateTrainTableJobID:
        """Create a generate-train-table job row in ``QUEUED``.

        ``custom_train_table`` is the table a *registering* job records
        rather than generates. It is admitted exactly as a training job
        admits one (:meth:`create_training_job`): under the same transaction,
        so the connector is locked and its identity binding rechecked live
        rather than found dead halfway through the read.
        """
        if now is None:
            now = now_utc()

        async with self._state_db.write_tx() as tx:
            if custom_train_table is not None:
                await self._require_connector_ref_in_project(
                    tx, custom_train_table.connector_spec, project_id)
            if id is not None:
                existing = await self._get_if_exists(
                    tx, self._generate_train_table_jobs, id)
                if existing is not None:
                    raise ValueError(f"Job id={id} already exists.")
            else:
                while True:
                    id = GenerateTrainTableJobID.make_random_id()
                    existing = await self._get_if_exists(
                        tx, self._generate_train_table_jobs, id)
                    if existing is None:
                        break

            await self._generate_train_table_jobs.upsert(
                tx,
                str(id),
                GenerateTrainTableJobDBEntry(
                    id=id,
                    timestamps=ResourceTimestamps(now, now),
                    config=config,
                    execution=GenerateTrainTableJobOutput(
                        status=JobStatus.QUEUED,
                        log=JobEventLogEntry.not_started(now),
                        start_time=None,
                        last_updated_at=now,
                    ),
                    source=source,
                    user=user_info,
                    project_id=project_id,
                    # Durable, so a reader can tell a registering job from a
                    # generating one, and so connector deletion can see this
                    # job as a consumer. The spec itself rides on `config`,
                    # in its secret-free public shape.
                    custom_train_table_used=(
                        custom_train_table is not None
                        or config.custom_train_table is not None),
                    custom_train_table_connector_ref=(
                        custom_train_table.connector_spec
                        if custom_train_table is not None else None),
                ),
            )
        return id

    async def create_generate_pred_table_job(
        self,
        *,
        config: GeneratePredTableJobConfig,
        id: GeneratePredictionTableJobID | None = None,
        now=None,
        source=None,
        user: str | None = None,
        project_id: ProjectID,
    ) -> GeneratePredictionTableJobID:
        if now is None:
            now = now_utc()

        async with self._state_db.write_tx() as tx:
            if id is not None:
                existing = await self._get_if_exists(
                    tx, self._generate_pred_table_jobs, id)
                if existing is not None:
                    raise ValueError(f"Job id={id} already exists.")
            else:
                while True:
                    id = GeneratePredictionTableJobID.make_random_id()
                    existing = await self._get_if_exists(
                        tx, self._generate_pred_table_jobs, id)
                    if existing is None:
                        break

            await self._generate_pred_table_jobs.upsert(
                tx,
                str(id),
                GeneratePredTableJobDBEntry(
                    id=id,
                    timestamps=ResourceTimestamps(now, now),
                    config=config,
                    execution=GeneratePredTableJobOutput(
                        status=JobStatus.QUEUED,
                        log=JobEventLogEntry.not_started(now),
                        start_time=None,
                        last_updated_at=now,
                    ),
                    source=source,
                    user=user,
                    project_id=project_id,
                ),
            )
        return id

    async def create_batch_prediction_job(
        self,
        *,
        config: BatchPredictionJobConfig,
        now=None,
        id: BatchPredictionJobID | None = None,
        source=None,
        project_id: ProjectID,
        user: str | None = None,
        description: str = "",
        batch_prediction_workflow_id: BatchPredictionWorkflowID | None = None,
    ) -> tuple[BatchPredictionJobID, bool]:
        if now is None:
            now = now_utc()

        async with self._state_db.write_tx() as tx:
            if id is not None:
                existing = await self._get_if_exists(
                    tx, self._batch_prediction_jobs, id)
                if existing is not None:
                    raise ValueError(
                        f"Batch prediction job id={id} already exists.")
            else:
                while True:
                    id = BatchPredictionJobID.make_random_id()
                    existing = await self._get_if_exists(
                        tx, self._batch_prediction_jobs, id)
                    if existing is None:
                        break

            assert bool(config.custom_pred_table_source) ^ bool(
                config.generate_pred_table_job_id)
            assert config.graph_id

            training_job = await self._get_if_exists(tx, self._training_jobs,
                                                     config.training_job_id)
            if training_job is not None:
                self._require_same_project(training_job, project_id,
                                           'Training job')
            await self._require_batch_output_connectors_in_project(
                tx, config, project_id)

            pred_table_job_id = config.generate_pred_table_job_id
            created_pred_table_job = False
            if pred_table_job_id:
                pred_table_job = await self._get_if_exists(
                    tx, self._generate_pred_table_jobs, pred_table_job_id)
                if pred_table_job is None:
                    created_pred_table_job = True
                    assert pred_table_job_id == (
                        GeneratePredictionTableJobID.as_child_of(id))
                    await self._create_generate_pred_table_job_in_tx(
                        tx,
                        id=pred_table_job_id,
                        config=GeneratePredTableJobConfig(
                            pquery_id=config.predictive_query_id,
                            plan=config.pred_table_generation_plan,
                            graph_id=config.graph_id,
                            graph_data_version=config.graph_data_version,
                        ),
                        now=now,
                        source=source,
                        project_id=project_id,
                    )
                    pred_table_job = await self._get_required(
                        tx,
                        self._generate_pred_table_jobs,
                        pred_table_job_id,
                    )
                self._require_same_project(pred_table_job, project_id,
                                           'Generate prediction table job')
                if pred_table_job.execution.status in (
                        JobStatus.FAILED,
                        JobStatus.CANCELLED,
                ):
                    raise ValueError(
                        "Unable to create a batch prediction job, using a "
                        f"previously {pred_table_job.execution.status} "
                        "generate-prediction-table job "
                        f"id={pred_table_job_id}")

            entry = BatchPredictionJobDBEntry(
                id=id,
                timestamps=ResourceTimestamps(now, now),
                execution=BatchPredictionJobOutput(
                    status=JobStatus.QUEUED,
                    log=JobEventLogEntry.not_started(now),
                    start_time=None,
                    last_updated_at=now,
                ),
                config=config,
                source=source,
                project_id=project_id,
                kumo_version=get_kumo_version(),
                user=user,
                description=description,
                batch_prediction_workflow_id=batch_prediction_workflow_id,
            )
            await self._batch_prediction_jobs.upsert(
                tx,
                str(id),
                entry,
            )
        return id, created_pred_table_job

    async def delete_batch_prediction_metadata(
        self,
        id: BatchPredictionJobID,
    ) -> bool:
        deleted_ids = await self.delete_batch_prediction_metadata_and_tags([id])
        return id in deleted_ids

    async def delete_batch_prediction_metadata_and_tags(
        self,
        ids: Iterable[BatchPredictionJobID],
    ) -> set[BatchPredictionJobID]:
        id_values = [BatchPredictionJobID(str(id)) for id in dict.fromkeys(ids)]
        if not id_values:
            return set()
        raw_ids = [str(id) for id in id_values]
        async with self._state_db.write_tx() as tx:
            rows = await tx.fetch(
                f"""
                DELETE FROM {self._batch_prediction_jobs._table_name}
                WHERE {self._in_predicate(tx, "id", raw_ids, 1)}
                RETURNING id
                """,
                *raw_ids,
            )
            await self._job_tags_repository.delete_by_ids(
                tx,
                JobType.BATCH_PREDICTION_JOB,
                id_values,
            )
        return {BatchPredictionJobID(row_str(row, "id", 0)) for row in rows}

    async def save_training_job_output(
        self,
        id: TrainingJobID,
        output: TrainingJobOutput,
    ) -> None:
        async with self._state_db.write_tx() as tx:
            await self._save_job_output_if_open(
                tx,
                self._training_jobs,
                id,
                output,
                f"Training Job with ID '{id}' does not exist.",
            )

    async def save_batch_prediction_job_output(
        self,
        id: BatchPredictionJobID,
        output: BatchPredictionJobOutput,
    ) -> None:
        async with self._state_db.write_tx() as tx:
            job = await self._get_if_exists(tx, self._batch_prediction_jobs, id)
            if job is None:
                raise ValueError(
                    f"Batch Prediction Job with ID '{id}' does not exist.")
            if job.execution is not None and (job.execution.progress is not None
                                              and output.progress is None):
                output.progress = job.execution.progress
            job.execution = output
            await self._modify_job_if_open_in_tx(tx, deepcopy(job))

    async def save_generate_train_table_job_output(
        self,
        id: GenerateTrainTableJobID,
        output: GenerateTrainTableJobOutput,
    ) -> None:
        async with self._state_db.write_tx() as tx:
            await self._save_job_output_if_open(
                tx,
                self._generate_train_table_jobs,
                id,
                output,
                None,
            )

    async def save_generate_pred_table_job_output(
        self,
        id: GeneratePredictionTableJobID,
        output: GeneratePredTableJobOutput,
    ) -> None:
        async with self._state_db.write_tx() as tx:
            await self._save_job_output_if_open(
                tx,
                self._generate_pred_table_jobs,
                id,
                output,
                None,
            )

    async def get_generate_train_table_job(
        self,
        id: GenerateTrainTableJobID,
        *,
        project_ids: Collection[ProjectID] | None,
    ) -> GenerateTrainTableJobDBEntry | None:
        rows = await self.get_generate_train_table_jobs([id],
                                                        project_ids=project_ids)
        return rows[0] if rows else None

    async def get_generate_train_table_jobs(
        self,
        ids: Iterable[GenerateTrainTableJobID],
        *,
        project_ids: Collection[ProjectID] | None,
    ) -> list[GenerateTrainTableJobDBEntry]:
        return await self._list_project_scoped_payloads_by_ids(
            self._generate_train_table_jobs,
            ids,
            project_ids=project_ids,
        )

    async def get_generate_pred_table_job(
        self,
        id: GeneratePredictionTableJobID,
    ) -> GeneratePredTableJobDBEntry | None:
        async with self._state_db.read_tx() as tx:
            return await self._get_if_exists(tx, self._generate_pred_table_jobs,
                                             id)

    async def get_generate_pred_table_jobs(
        self,
        ids: Iterable[GeneratePredictionTableJobID],
        *,
        project_ids: Collection[ProjectID] | None,
    ) -> list[GeneratePredTableJobDBEntry]:
        return await self._list_project_scoped_payloads_by_ids(
            self._generate_pred_table_jobs,
            ids,
            project_ids=project_ids,
        )

    async def get_training_job(
        self,
        id: TrainingJobID,
    ) -> TrainingJobDBEntry | None:
        async with self._state_db.read_tx() as tx:
            return await self._get_if_exists(tx, self._training_jobs, id)

    async def get_training_jobs(
        self,
        ids: Iterable[TrainingJobID],
    ) -> list[TrainingJobDBEntry]:
        async with self._state_db.read_tx() as tx:
            return await self._list_payloads_by_ids(tx, self._training_jobs,
                                                    ids)

    async def list_training_job_rows(
        self,
        *,
        job_ids: Collection[TrainingJobID],
        project_ids: Collection[ProjectID] | None,
        sort_created_at: str = "asc",
        limit: int | None = None,
    ) -> list[TrainingJobDBEntry]:
        """List tagged training candidates within an explicit project scope."""
        return await self._list_project_scoped_payloads_by_ids(
            self._training_jobs,
            job_ids,
            project_ids=project_ids,
            sort_created_at=sort_created_at,
            limit=limit,
            copy_payloads=True,
        )

    async def get_batch_prediction_jobs(
        self,
        ids: Iterable[BatchPredictionJobID],
        *,
        project_ids: Collection[ProjectID] | None,
    ) -> list[BatchPredictionJobDBEntry]:
        return await self.get_batch_prediction_rows(ids,
                                                    project_ids=project_ids)

    async def get_batch_prediction_rows(
        self,
        ids: Iterable[BatchPredictionJobID],
        *,
        project_ids: Collection[ProjectID] | None,
    ) -> list[BatchPredictionJobDBEntry]:
        async with self._state_db.read_tx() as tx:
            clauses, args = self._batch_prediction_where(
                tx,
                project_ids=project_ids,
                job_ids=ids,
            )
            rows = await self._list_payloads_where(tx,
                                                   self._batch_prediction_jobs,
                                                   clauses, args)
        return [deepcopy(row) for row in rows]

    async def list_batch_prediction_rows_by_workflow_id(
        self,
        workflow_id: BatchPredictionWorkflowID | str | None,
        *,
        project_ids: Collection[ProjectID] | None,
    ) -> list[BatchPredictionJobDBEntry]:
        expected_workflow_id = (None if workflow_id is None else
                                BatchPredictionWorkflowID(str(workflow_id)))
        async with self._state_db.read_tx() as tx:
            workflow_expr = _json_value(tx, "batch_prediction_workflow_id")
            clauses, args = self._project_ids_where(tx, project_ids)
            if expected_workflow_id is None:
                clauses.append(f"{workflow_expr} IS NULL")
            else:
                args.append(str(expected_workflow_id))
                clauses.append(
                    f"{workflow_expr} = {_placeholder(tx, len(args))}")
            rows = await self._list_payloads_where(tx,
                                                   self._batch_prediction_jobs,
                                                   clauses, args)
        return [deepcopy(row) for row in rows]

    async def list_batch_prediction_rows(
        self,
        *,
        project_ids: Collection[ProjectID] | None,
        model_id: TrainingJobID | None = None,
        pquery_id: PredictiveQueryID | None = None,
        job_ids: Iterable[BatchPredictionJobID] | None = None,
        status: JobStatus | None = None,
        sort_created_at: str = "asc",
        limit: int | None = None,
        offset: int = 0,
        match_pquery_def_fpid: bool = True,
    ) -> list[BatchPredictionJobDBEntry]:
        async with self._state_db.read_tx() as tx:
            clauses, args = self._batch_prediction_where(
                tx,
                project_ids=project_ids,
                model_id=model_id,
                pquery_id=pquery_id,
                job_ids=job_ids,
                status=status,
                match_pquery_def_fpid=match_pquery_def_fpid,
            )
            rows = await self._list_payloads_where(
                tx,
                self._batch_prediction_jobs,
                clauses,
                args,
                sort_created_at=sort_created_at,
                limit=limit,
                offset=offset,
            )
        return [deepcopy(row) for row in rows]

    async def count_batch_prediction_rows(
        self,
        *,
        project_ids: Collection[ProjectID] | None,
        model_id: TrainingJobID | None = None,
        pquery_id: PredictiveQueryID | None = None,
        job_ids: Iterable[BatchPredictionJobID] | None = None,
        status: JobStatus | None = None,
        match_pquery_def_fpid: bool = True,
    ) -> int:
        async with self._state_db.read_tx() as tx:
            clauses, args = self._batch_prediction_where(
                tx,
                project_ids=project_ids,
                model_id=model_id,
                pquery_id=pquery_id,
                job_ids=job_ids,
                status=status,
                match_pquery_def_fpid=match_pquery_def_fpid,
            )
            where_sql = (" WHERE " + " AND ".join(clauses) if clauses else "")
            row = await tx.fetchrow(
                f"""
                SELECT COUNT(*) AS count
                FROM {self._batch_prediction_jobs._table_name}
                {where_sql}
                """,
                *args,
            )
        if row is None:
            return 0
        return int(row_field(row, "count", 0))

    async def has_batch_prediction_rows(
        self,
        *,
        project_ids: Collection[ProjectID] | None,
        model_id: TrainingJobID | None = None,
        pquery_id: PredictiveQueryID | None = None,
        job_ids: Iterable[BatchPredictionJobID] | None = None,
        status: JobStatus | None = None,
        match_pquery_def_fpid: bool = True,
    ) -> bool:
        async with self._state_db.read_tx() as tx:
            clauses, args = self._batch_prediction_where(
                tx,
                project_ids=project_ids,
                model_id=model_id,
                pquery_id=pquery_id,
                job_ids=job_ids,
                status=status,
                match_pquery_def_fpid=match_pquery_def_fpid,
            )
            where_sql = (" WHERE " + " AND ".join(clauses) if clauses else "")
            row = await tx.fetchrow(
                f"""
                SELECT 1
                FROM {self._batch_prediction_jobs._table_name}
                {where_sql}
                LIMIT 1
                """,
                *args,
            )
        return row is not None

    async def get_current_batch_prediction_job_ids(
        self,
        pquery_ids: Iterable[PredictiveQueryID],
        *,
        project_ids: Collection[ProjectID] | None,
    ) -> dict[PredictiveQueryID, BatchPredictionJobID]:
        pquery_id_values = sorted({str(pquery_id) for pquery_id in pquery_ids})
        if not pquery_id_values:
            return {}

        result: dict[PredictiveQueryID, BatchPredictionJobID] = {}
        async with self._state_db.read_tx() as tx:
            pquery_expr = _json_value(tx, "pquery_id")
            clauses, args = self._project_ids_where(tx, project_ids)
            clauses.append(
                self._in_predicate(
                    tx,
                    pquery_expr,
                    pquery_id_values,
                    len(args) + 1,
                ))
            args.extend(pquery_id_values)
            rows = await self._list_payloads_where(
                tx,
                self._batch_prediction_jobs,
                clauses,
                args,
                sort_created_at="desc",
            )
            for row in rows:
                if row.pquery_id is None or row.pquery_id in result:
                    continue
                result[row.pquery_id] = row.id
        return result

    async def list_batch_prediction_prune_candidate_ids(
        self,
        workflow_id: BatchPredictionWorkflowID | str | None,
        *,
        project_ids: Collection[ProjectID] | None,
        keep_most_recent: int,
    ) -> list[BatchPredictionJobID]:
        expected_workflow_id = (None if workflow_id is None else
                                BatchPredictionWorkflowID(str(workflow_id)))
        async with self._state_db.read_tx() as tx:
            workflow_expr = _json_value(tx, "batch_prediction_workflow_id")
            clauses, args = self._project_ids_where(tx, project_ids)
            if expected_workflow_id is None:
                clauses.append(f"{workflow_expr} IS NULL")
            else:
                args.append(str(expected_workflow_id))
                clauses.append(
                    f"{workflow_expr} = {_placeholder(tx, len(args))}")
            rows = await self._list_payloads_where(
                tx,
                self._batch_prediction_jobs,
                clauses,
                args,
                sort_created_at="desc",
                offset=keep_most_recent,
            )
        return [row.id for row in rows]

    async def get_training_job_training_table_plan(
        self,
        id: TrainingJobID,
    ) -> Any | None:
        from kumo_core.model_plan import ModelPlan

        job = await self.get_training_job(id)
        if job is None or not isinstance(job.advanced_options, ModelPlan):
            return None
        return job.advanced_options.training_table_generation

    async def get_running_batch_prediction_job_ids(
        self,
        *,
        project_ids: Collection[ProjectID] | None,
    ) -> list[BatchPredictionJobID]:
        submitted_statuses = [
            status.value for status in JobStatus if status.is_submitted
        ]
        async with self._state_db.read_tx() as tx:
            clauses, args = self._project_ids_where(tx, project_ids)
            status_start = len(args) + 1
            clauses.append("(" + " OR ".join([
                self._in_predicate(
                    tx,
                    _json_value(tx, "status"),
                    submitted_statuses,
                    status_start,
                ),
                self._in_predicate(
                    tx,
                    _json_path_value(tx, "execution", "status"),
                    submitted_statuses,
                    status_start + len(submitted_statuses),
                ),
            ]) + ")")
            args.extend(submitted_statuses)
            args.extend(submitted_statuses)
            rows = await self._list_ids_where(
                tx,
                self._batch_prediction_jobs,
                clauses,
                args,
            )
        return [BatchPredictionJobID(id) for id in rows]

    async def get_job_if_exists_generic(self, id: JobID) -> Any | None:
        job_type = JobType.from_id(id)
        if job_type == JobType.TRAINING_JOB:
            return await self.get_training_job(TrainingJobID(id))
        if job_type == JobType.BATCH_PREDICTION_JOB:
            jobs = await self.get_batch_prediction_jobs(
                [BatchPredictionJobID(id)], project_ids=None)
            return jobs[0] if jobs else None
        if job_type == JobType.GENERATE_TRAIN_TABLE_JOB:
            return await self.get_generate_train_table_job(
                GenerateTrainTableJobID(id), project_ids=None)
        if job_type == JobType.GENERATE_PREDICTION_TABLE_JOB:
            return await self.get_generate_pred_table_job(
                GeneratePredictionTableJobID(id))
        raise AssertionError(f"Unsupported job type {job_type}")

    async def upsert_job_registry_metadata(
        self,
        id: JobID,
        metadata: MLJobRegistryMetadata,
    ) -> None:
        job_type = JobType.from_id(id)
        async with self._state_db.write_tx() as tx:
            if job_type == JobType.TRAINING_JOB:
                repository = self._training_jobs
                key = TrainingJobID(id)
            elif job_type == JobType.BATCH_PREDICTION_JOB:
                repository = self._batch_prediction_jobs
                key = BatchPredictionJobID(id)
            elif job_type == JobType.GENERATE_TRAIN_TABLE_JOB:
                repository = self._generate_train_table_jobs
                key = GenerateTrainTableJobID(id)
            elif job_type == JobType.GENERATE_PREDICTION_TABLE_JOB:
                repository = self._generate_pred_table_jobs
                key = GeneratePredictionTableJobID(id)
            else:
                raise AssertionError(f"Unsupported job type {job_type}")

            job = await self._get_if_exists(tx, repository, key)
            if job is None:
                raise ValueError(f"Job id={id} does not exist.")
            job = deepcopy(job)
            job.registry_metadata = metadata
            await repository.upsert(tx, str(id), job)

    async def select_jobs(
        self,
        filters: Mapping[str, list[str] | str],
    ) -> list[MLJobRegistryEntry]:
        entries = await self._list_registry_entries()
        return [
            entry for entry in entries
            if self._registry_entry_matches_filters(entry, filters)
        ]

    async def get_multi_train_jobs(
        self,
        train_job_ids: list[str],
    ) -> dict[str, MLJobRegistryEntry]:
        ids = set(train_job_ids)
        entries = await self.select_jobs({
            "job_type": [JobType.TRAINING_JOB],
            "job_id": list(ids),
        })
        return {str(entry.job_id): entry for entry in entries}

    async def list_job_ids_by_type(self, job_type: JobType) -> list[JobID]:
        repository = self._repository_for_job_type(job_type)
        async with self._state_db.read_tx() as tx:
            ids = await self._list_ids_where(tx, repository, [], [])
        return [_job_id_from_raw(job_type, id) for id in ids]

    async def get_job_statuses_by_ids(
        self,
        ids: Iterable[JobID],
    ) -> dict[JobID, JobStatus]:
        ids_by_type: dict[JobType, list[JobID]] = {}
        for id in ids:
            ids_by_type.setdefault(JobType.from_id(id), []).append(id)

        statuses: dict[JobID, JobStatus] = {}
        async with self._state_db.read_tx() as tx:
            for job_type, typed_ids in ids_by_type.items():
                rows = await self._list_rows_by_ids(
                    tx,
                    self._repository_for_job_type(job_type),
                    typed_ids,
                )
                for row in rows:
                    job = row.payload
                    execution = getattr(job, "execution", None)
                    if execution is not None:
                        statuses[_job_id_from_raw(job_type,
                                                  row.id)] = execution.status
        return statuses

    async def delete_batch_prediction_jobs(
        self,
        ids: Iterable[BatchPredictionJobID],
        *,
        force: bool = False,
    ) -> set[BatchPredictionJobID]:
        async with self._state_db.write_tx() as tx:
            deleted_ids = await self._delete_batch_prediction_ids_in_tx(
                tx,
                ids,
                force=force,
            )
            await self._job_tags_repository.delete_by_ids(
                tx,
                JobType.BATCH_PREDICTION_JOB,
                [BatchPredictionJobID(id) for id in deleted_ids],
            )
        return {BatchPredictionJobID(id) for id in deleted_ids}

    async def modify_job_if_open_generic(self, job: Any) -> None:
        async with self._state_db.write_tx() as tx:
            await self._modify_job_if_open_in_tx(tx, job)

    async def mark_workflow_not_found(self, id: JobID, now) -> Any:
        job_type = JobType.from_id(id)
        async with self._state_db.write_tx() as tx:
            job = await self._get_job_if_exists_generic_in_tx(tx, id)
            if job is None:
                raise KeyError(f"Job with id {id} not found in database")
            execution = job.execution
            execution.workflow_end_state = (execution.workflow_end_state or
                                            WorkflowExecutionStatus.TIMED_OUT)
            if execution.status.is_submitted:
                execution.status = JobStatus.FAILED
                execution.failure_info = JobFailureInfo(
                    message=(f"This {job_type.value} has expired and its"
                             " workflow is not found."),
                    stacktrace="",
                )
                execution.log = JobEventLogEntry.failed(
                    now,
                    f"{job_type.value} workflow has expired and not found.",
                )
            await self._modify_job_if_open_in_tx(tx, job)
            return job

    async def cancel_job_if_open_generic(self, id: JobID) -> bool:
        job_type = JobType.from_id(id)
        async with self._state_db.write_tx() as tx:
            job = await self._get_job_if_exists_generic_in_tx(tx, id)
            if job is None:
                raise KeyError(f"Job with id {id} not found in database")
            if job.execution.workflow_end_state:
                assert job.execution.status.is_terminal
                return job.execution.status == JobStatus.CANCELLED

            job.execution.workflow_end_state = WorkflowExecutionStatus.CANCELED
            job.execution.status = JobStatus.CANCELLED
            job.execution.log = JobEventLogEntry.cancelled(
                message=f"{job_type.value} was cancelled by user")
            await self._modify_job_if_open_in_tx(tx, job)

        return True

    async def delete_job_generic(
        self,
        id: JobID,
        *,
        force: bool = False,
        delete_created_child: bool = False,
    ) -> bool:
        async with self._state_db.write_tx() as tx:
            job = await self._get_job_if_exists_generic_in_tx(tx, id)
            if job is None:
                return False
            execution = getattr(job, "execution", None)
            if (execution is not None and execution.status.is_submitted
                    and not force):
                raise ValueError(f"Unable to delete a queued/running job {id}")
            child = (await self._find_child_job_in_tx(tx, job)
                     if delete_created_child else None)
            await self._delete_job_in_tx(tx, job)
            if child is not None:
                await self._delete_job_in_tx(tx, child)
        return True

    async def is_child_job(self, job: Any) -> bool:
        async with self._state_db.read_tx() as tx:
            if isinstance(job, GenerateTrainTableJobDBEntry):
                training_parent = await self._get_if_exists(
                    tx,
                    self._training_jobs,
                    TrainingJobID.as_parent_of(job.id),
                )
                return training_parent is not None

            if isinstance(job, GeneratePredTableJobDBEntry):
                batch_parent = await self._get_if_exists(
                    tx,
                    self._batch_prediction_jobs,
                    BatchPredictionJobID.as_parent_of(job.id),
                )
                return batch_parent is not None

        return False

    async def get_job_tags(
        self,
        job_type: JobType | str,
        job_id: JobID,
    ) -> JobTags | None:
        job_type = coerce_job_type(job_type)
        async with self._state_db.read_tx() as tx:
            return await self._job_tags_repository.get(tx, job_type, job_id)

    async def apply_job_tags(
        self,
        job_type: JobType | str,
        job_id: JobID,
        tags: dict[JobTagKey, JobTagValue],
    ) -> bool:
        job_type = coerce_job_type(job_type)
        async with self._state_db.write_tx() as tx:
            return await self._job_tags_repository.apply(
                tx,
                job_type,
                job_id,
                tags,
            )

    async def delete_job_tag(
        self,
        job_type: JobType | str,
        job_id: JobID,
        tag_key: JobTagKey,
    ) -> bool:
        job_type = coerce_job_type(job_type)
        if tag_key in _IMMUTABLE_TAG_KEYS:
            return False
        async with self._state_db.write_tx() as tx:
            return await self._job_tags_repository.delete_tag(
                tx,
                job_type,
                job_id,
                tag_key,
            )

    async def delete_job_tags(
        self,
        job_type: JobType | str,
        job_id: JobID,
    ) -> bool:
        job_type = coerce_job_type(job_type)
        async with self._state_db.write_tx() as tx:
            return await self._job_tags_repository.delete_all(
                tx,
                job_type,
                job_id,
            )

    async def archive_job(
        self,
        job_id: JobID,
        *,
        project_id: ProjectID,
    ) -> bool:
        job = await self.get_job_if_exists_generic(job_id)
        if job is None:
            return False
        if job.project_id != project_id:
            return False
        return await self.apply_job_tags(
            JobType.from_id(job_id),
            job_id,
            {TAG_KEY_ARCHIVED: TAG_VALUE_ARCHIVED},
        )

    async def unarchive_job(
        self,
        job_id: JobID,
        *,
        project_id: ProjectID,
    ) -> bool:
        job = await self.get_job_if_exists_generic(job_id)
        if job is None:
            return False
        if job.project_id != project_id:
            return False
        return await self.apply_job_tags(
            JobType.from_id(job_id),
            job_id,
            {TAG_KEY_ARCHIVED: JobTagValue("null")},
        )

    async def list_job_tags(
        self,
        job_type: JobType | str,
        filters: Mapping[str, str],
        *,
        job_ids: Iterable[JobID] | None = None,
    ) -> dict[JobID, JobTags]:
        job_type = coerce_job_type(job_type)
        async with self._state_db.read_tx() as tx:
            if job_ids is not None:
                return await self._job_tags_repository.list_by_ids(
                    tx, job_type, job_ids)
            return await self._job_tags_repository.list_by_filters(
                tx, job_type, filters)

    async def search_jobs(
        self,
        ctx,
        list_query: ListJobsQuery,
        tags: Mapping[JobID, JobTags],
        *,
        project_id: ProjectID,
    ) -> tuple[list[GenericJob], int]:
        pushdown_page = self._can_pushdown_aggregate_page(list_query)
        count = None
        if pushdown_page:
            count = await self._count_aggregate_jobs_for_query(
                list_query,
                tags,
                project_id=project_id,
            )

        entries = await self._list_aggregate_job_entries_for_query(
            list_query,
            tags=tags,
            project_id=project_id,
            pushdown_page=pushdown_page,
        )
        jobs = [entry.to_generic(ctx) for entry in entries]
        for job in jobs:
            if job.job_id in tags:
                job.metadata.tags.update(**tags[job.job_id].tags)

        if pushdown_page and count is not None:
            return jobs, count

        filtered_jobs = ListJobsProcessor.apply_filters(
            query=list_query,
            jobs=jobs,
        )
        num_items = len(filtered_jobs)
        sorted_jobs = ListJobsProcessor.apply_sorting(
            query=list_query,
            jobs=filtered_jobs,
        )
        return (
            ListJobsProcessor.apply_pagination(
                query=list_query,
                jobs=sorted_jobs,
            ),
            num_items,
        )

    async def count_jobs(
        self,
        ctx,
        list_query: ListJobsQuery,
        tags: Mapping[JobID, JobTags],
        *,
        project_id: ProjectID,
    ) -> int:
        count = await self._count_aggregate_jobs_for_query(
            list_query,
            tags,
            project_id=project_id,
        )
        if count is not None:
            return count

        entries = await self._list_aggregate_job_entries_for_query(
            list_query,
            project_id=project_id,
        )
        jobs = [entry.to_generic(ctx) for entry in entries]
        for job in jobs:
            if job.job_id in tags:
                job.metadata.tags.update(**tags[job.job_id].tags)
        return len(
            ListJobsProcessor.apply_filters(
                query=list_query,
                jobs=jobs,
            ))

    async def list_jobs_by_project_visibility(
        self,
        visible_project_ids: Iterable[ProjectID],
    ) -> list[Any]:
        visible = {
            ProjectID(str(project_id))
            for project_id in visible_project_ids
        }
        return await self._list_aggregate_job_entries_for_project_visibility(
            visible, )

    async def list_aggregate_job_entries(
        self,
        *,
        project_id: ProjectID,
    ) -> list[Any]:
        async with self._state_db.read_tx() as tx:
            project_clauses, project_args = self._project_where(
                tx,
                project_id=project_id,
            )
            training_jobs = await self._list_payloads_where(
                tx, self._training_jobs, project_clauses, project_args)
            batch_rows = await self._list_payloads_where(
                tx,
                self._batch_prediction_jobs,
                [
                    *_clone_clauses(project_clauses),
                    _batch_prediction_has_execution_clause(tx),
                ],
                list(project_args),
            )
            generate_train_table_jobs = await self._list_payloads_where(
                tx,
                self._generate_train_table_jobs,
                project_clauses,
                project_args,
            )
            generate_pred_table_jobs = await self._list_payloads_where(
                tx,
                self._generate_pred_table_jobs,
                project_clauses,
                project_args,
            )
        return [
            *training_jobs,
            *batch_rows,
            *generate_train_table_jobs,
            *generate_pred_table_jobs,
        ]

    async def _list_aggregate_job_entries_for_query(
        self,
        list_query: ListJobsQuery,
        *,
        tags: Mapping[JobID, JobTags] | None = None,
        project_id: ProjectID,
        pushdown_page: bool = False,
    ) -> list[Any]:
        requested_job_types = set(list_query.job_type)
        tag_filter_ids = None
        archived_ids = None
        search_tag_ids = None
        if tags is not None:
            tag_filter_ids, archived_ids = self._job_tag_count_predicates(
                list_query,
                tags,
            )
            search_tag_ids = self._job_tag_search_predicates(list_query, tags)
        limit = list_query.limit if pushdown_page else None
        offset = list_query.limit * list_query.page if pushdown_page else 0
        sort_created_at = (self._aggregate_created_at_sort_direction(list_query)
                           if pushdown_page else None)
        async with self._state_db.read_tx() as tx:
            project_clauses, project_args = self._project_where(
                tx,
                project_id=project_id,
            )
            training_jobs = await self._list_aggregate_payloads_for_type(
                tx,
                self._training_jobs,
                JobType.TRAINING_JOB,
                requested_job_types,
                project_clauses,
                project_args,
                list_query,
                tag_filter_ids=tag_filter_ids,
                archived_ids=archived_ids,
                search_tag_ids=search_tag_ids,
                sort_created_at=sort_created_at,
                limit=limit,
                offset=offset,
            )
            batch_rows = await self._list_aggregate_payloads_for_type(
                tx,
                self._batch_prediction_jobs,
                JobType.BATCH_PREDICTION_JOB,
                requested_job_types,
                project_clauses,
                project_args,
                list_query,
                tag_filter_ids=tag_filter_ids,
                archived_ids=archived_ids,
                extra_clauses=[_batch_prediction_has_execution_clause(tx)],
                search_tag_ids=search_tag_ids,
                sort_created_at=sort_created_at,
                limit=limit,
                offset=offset,
            )
            generate_train_table_jobs = (await
                                         self._list_aggregate_payloads_for_type(
                                             tx,
                                             self._generate_train_table_jobs,
                                             JobType.GENERATE_TRAIN_TABLE_JOB,
                                             requested_job_types,
                                             project_clauses,
                                             project_args,
                                             list_query,
                                             tag_filter_ids=tag_filter_ids,
                                             archived_ids=archived_ids,
                                             search_tag_ids=search_tag_ids,
                                             sort_created_at=sort_created_at,
                                             limit=limit,
                                             offset=offset,
                                         ))
            generate_pred_table_jobs = (
                await self._list_aggregate_payloads_for_type(
                    tx,
                    self._generate_pred_table_jobs,
                    JobType.GENERATE_PREDICTION_TABLE_JOB,
                    requested_job_types,
                    project_clauses,
                    project_args,
                    list_query,
                    tag_filter_ids=tag_filter_ids,
                    archived_ids=archived_ids,
                    search_tag_ids=search_tag_ids,
                    sort_created_at=sort_created_at,
                    limit=limit,
                    offset=offset,
                ))
        return [
            *training_jobs,
            *batch_rows,
            *generate_train_table_jobs,
            *generate_pred_table_jobs,
        ]

    async def _count_aggregate_jobs_for_query(
        self,
        list_query: ListJobsQuery,
        tags: Mapping[JobID, JobTags],
        *,
        project_id: ProjectID,
    ) -> int | None:
        tag_filter_ids, archived_ids = self._job_tag_count_predicates(
            list_query,
            tags,
        )
        search_tag_ids = self._job_tag_search_predicates(list_query, tags)
        requested_job_types = set(list_query.job_type)
        async with self._state_db.read_tx() as tx:
            project_clauses, project_args = self._project_where(
                tx,
                project_id=project_id,
            )
            return sum([
                await self._count_aggregate_jobs_for_type(
                    tx,
                    self._training_jobs,
                    JobType.TRAINING_JOB,
                    requested_job_types,
                    project_clauses,
                    project_args,
                    list_query,
                    tag_filter_ids,
                    archived_ids,
                    search_tag_ids=search_tag_ids,
                ),
                await self._count_aggregate_jobs_for_type(
                    tx,
                    self._batch_prediction_jobs,
                    JobType.BATCH_PREDICTION_JOB,
                    requested_job_types,
                    project_clauses,
                    project_args,
                    list_query,
                    tag_filter_ids,
                    archived_ids,
                    extra_clauses=[
                        _batch_prediction_countable_execution_clause(tx)
                    ],
                    search_tag_ids=search_tag_ids,
                ),
                await self._count_aggregate_jobs_for_type(
                    tx,
                    self._generate_train_table_jobs,
                    JobType.GENERATE_TRAIN_TABLE_JOB,
                    requested_job_types,
                    project_clauses,
                    project_args,
                    list_query,
                    tag_filter_ids,
                    archived_ids,
                    search_tag_ids=search_tag_ids,
                ),
                await self._count_aggregate_jobs_for_type(
                    tx,
                    self._generate_pred_table_jobs,
                    JobType.GENERATE_PREDICTION_TABLE_JOB,
                    requested_job_types,
                    project_clauses,
                    project_args,
                    list_query,
                    tag_filter_ids,
                    archived_ids,
                    search_tag_ids=search_tag_ids,
                ),
            ])

    async def _list_aggregate_payloads_for_type(
        self,
        tx: StateTx,
        repository: JsonRepository,
        job_type: JobType,
        requested_job_types: set[JobType],
        project_clauses: list[str],
        project_args: list[Any],
        list_query: ListJobsQuery,
        *,
        tag_filter_ids: dict[JobType, set[str]] | None = None,
        archived_ids: dict[JobType, set[str]] | None = None,
        extra_clauses: list[str] | None = None,
        search_tag_ids: dict[JobType, set[str]] | None = None,
        sort_created_at: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Any]:
        clauses_and_args = self._aggregate_query_clauses(
            tx,
            job_type,
            requested_job_types,
            project_clauses,
            project_args,
            list_query,
            tag_filter_ids=tag_filter_ids,
            archived_ids=archived_ids,
            extra_clauses=extra_clauses,
            search_tag_ids=search_tag_ids,
        )
        if clauses_and_args is None:
            return []
        clauses, args = clauses_and_args
        return await self._list_payloads_where(
            tx,
            repository,
            clauses,
            args,
            sort_created_at=sort_created_at,
            limit=limit,
            offset=offset,
        )

    async def _count_aggregate_jobs_for_type(
        self,
        tx: StateTx,
        repository: JsonRepository,
        job_type: JobType,
        requested_job_types: set[JobType],
        project_clauses: list[str],
        project_args: list[Any],
        list_query: ListJobsQuery,
        tag_filter_ids: dict[JobType, set[str]] | None,
        archived_ids: dict[JobType, set[str]],
        *,
        extra_clauses: list[str] | None = None,
        search_tag_ids: dict[JobType, set[str]] | None = None,
    ) -> int:
        clauses_and_args = self._aggregate_query_clauses(
            tx,
            job_type,
            requested_job_types,
            project_clauses,
            project_args,
            list_query,
            tag_filter_ids=tag_filter_ids,
            archived_ids=archived_ids,
            extra_clauses=extra_clauses,
            search_tag_ids=search_tag_ids,
        )
        if clauses_and_args is None:
            return 0
        clauses, args = clauses_and_args
        where_sql = (" WHERE " + " AND ".join(clauses) if clauses else "")
        row = await tx.fetchrow(
            f"""
            SELECT COUNT(*) AS count
            FROM {repository._table_name}
            {where_sql}
            """,
            *args,
        )
        if row is None:
            return 0
        return int(row_field(row, "count", 0))

    def _aggregate_query_clauses(
        self,
        tx: StateTx,
        job_type: JobType,
        requested_job_types: set[JobType],
        project_clauses: list[str],
        project_args: list[Any],
        list_query: ListJobsQuery,
        *,
        tag_filter_ids: dict[JobType, set[str]] | None = None,
        archived_ids: dict[JobType, set[str]] | None = None,
        extra_clauses: list[str] | None = None,
        search_tag_ids: dict[JobType, set[str]] | None = None,
    ) -> tuple[list[str], list[Any]] | None:
        if job_type not in requested_job_types:
            return None

        clauses = [
            *_clone_clauses(project_clauses),
            *([] if extra_clauses is None else extra_clauses),
        ]
        args = list(project_args)

        if list_query.status:
            status_values = [
                str(getattr(status, "value", status))
                for status in list_query.status
            ]
            clauses.append(
                self._in_predicate(
                    tx,
                    _json_path_value(tx, "execution", "status"),
                    status_values,
                    len(args) + 1,
                ))
            args.extend(status_values)

        created_expr = _json_path_value(tx, "timestamps", "created_at")
        if list_query.time_range_start is not None:
            args.append(list_query.time_range_start.isoformat())
            clauses.append(f"{created_expr} >= {_placeholder(tx, len(args))}")
        if list_query.time_range_end is not None:
            args.append(list_query.time_range_end.isoformat())
            clauses.append(f"{created_expr} <= {_placeholder(tx, len(args))}")

        if tag_filter_ids is not None:
            job_ids = sorted(tag_filter_ids.get(job_type, set()))
            clauses.append(self._in_predicate(tx, "id", job_ids, len(args) + 1))
            args.extend(job_ids)
        elif archived_ids is not None:
            archived_job_ids = sorted(archived_ids.get(job_type, set()))
            if archived_job_ids:
                clauses.append(
                    self._not_in_predicate(
                        tx,
                        "id",
                        archived_job_ids,
                        len(args) + 1,
                    ))
                args.extend(archived_job_ids)

        search_clause = self._aggregate_search_clause(
            tx,
            job_type,
            list_query.search_keyword,
            search_tag_ids,
            args,
        )
        if search_clause is not None:
            clauses.append(search_clause)

        return clauses, args

    def _aggregate_search_clause(
        self,
        tx: StateTx,
        job_type: JobType,
        search_keyword: str | None,
        search_tag_ids: dict[JobType, set[str]] | None,
        args: list[Any],
    ) -> str | None:
        if not search_keyword:
            return None
        search_term = search_keyword.lower()
        search_pattern = _like_pattern(search_term)
        search_clauses = []
        for expression in (
                "id",
                _json_value(tx, "source"),
                _json_value(tx, "user"),
        ):
            args.append(search_pattern)
            search_clauses.append(
                f"{_lower_sql(expression)} LIKE {_placeholder(tx, len(args))} "
                "ESCAPE '\\'")

        job_type_label = job_type.value.lower().replace("_", " ")
        if search_term in job_type_label:
            search_clauses.append("1 = 1")

        tag_job_ids = sorted((search_tag_ids or {}).get(job_type, set()))
        if tag_job_ids:
            search_clauses.append(
                self._in_predicate(tx, "id", tag_job_ids,
                                   len(args) + 1))
            args.extend(tag_job_ids)
        return "(" + " OR ".join(search_clauses) + ")"

    @staticmethod
    def _aggregate_created_at_sort_direction(
        list_query: ListJobsQuery, ) -> str:
        if list_query.sort_by is None:
            return "desc"
        if list_query.sort_by == JobSortField.START_TIME:
            return "desc" if list_query.sort_desc else "asc"
        raise ValueError(f"Unsupported aggregate job SQL sort: "
                         f"{list_query.sort_by}")

    def _can_pushdown_aggregate_page(self, list_query: ListJobsQuery) -> bool:
        if list_query.sort_by not in (None, JobSortField.START_TIME):
            return False
        requested_job_types = set(list_query.job_type)
        return (len(requested_job_types) == 1
                and next(iter(requested_job_types)) in _AGGREGATE_JOB_TYPES)

    def _job_tag_count_predicates(
        self,
        list_query: ListJobsQuery,
        tags: Mapping[JobID, JobTags],
    ) -> tuple[dict[JobType, set[str]] | None, dict[JobType, set[str]]]:
        tag_filter_values = {str(value) for value in list_query.tags_filter}
        want_archived = str(TAG_VALUE_ARCHIVED) in tag_filter_values
        if tag_filter_values:
            matching_ids: dict[JobType, set[str]] = {}
        else:
            matching_ids = None
        archived_ids: dict[JobType, set[str]] = {}

        for job_id, job_tags in tags.items():
            job_type = JobType.from_id(job_id)
            tag_values = {str(value) for value in job_tags.tags.values()}
            archived = (str(
                job_tags.tags.get(TAG_KEY_ARCHIVED)) == str(TAG_VALUE_ARCHIVED))
            if matching_ids is not None:
                if archived != want_archived:
                    continue
                if tag_values.isdisjoint(tag_filter_values):
                    continue
                matching_ids.setdefault(job_type, set()).add(str(job_id))
            elif archived:
                archived_ids.setdefault(job_type, set()).add(str(job_id))

        return matching_ids, archived_ids

    def _job_tag_search_predicates(
        self,
        list_query: ListJobsQuery,
        tags: Mapping[JobID, JobTags],
    ) -> dict[JobType, set[str]] | None:
        if not list_query.search_keyword:
            return None
        search_term = list_query.search_keyword.lower()
        matching_ids: dict[JobType, set[str]] = {}
        for job_id, job_tags in tags.items():
            if any(search_term in str(value).lower()
                   for value in job_tags.tags.values()):
                matching_ids.setdefault(JobType.from_id(job_id),
                                        set()).add(str(job_id))
        return matching_ids

    async def _list_aggregate_job_entries_for_project_visibility(
        self,
        visible_project_ids: set[ProjectID],
    ) -> list[Any]:
        async with self._state_db.read_tx() as tx:
            project_clauses, project_args = self._project_visibility_where(
                tx,
                visible_project_ids,
            )
            training_jobs = await self._list_payloads_where(
                tx, self._training_jobs, project_clauses, project_args)
            batch_rows = await self._list_payloads_where(
                tx,
                self._batch_prediction_jobs,
                [
                    *_clone_clauses(project_clauses),
                    _batch_prediction_has_execution_clause(tx),
                ],
                list(project_args),
            )
            generate_train_table_jobs = await self._list_payloads_where(
                tx,
                self._generate_train_table_jobs,
                project_clauses,
                project_args,
            )
            generate_pred_table_jobs = await self._list_payloads_where(
                tx,
                self._generate_pred_table_jobs,
                project_clauses,
                project_args,
            )
        return [
            *training_jobs,
            *batch_rows,
            *generate_train_table_jobs,
            *generate_pred_table_jobs,
        ]

    async def list_open_parent_jobs(self) -> list[Any]:
        async with self._state_db.read_tx() as tx:
            return await self.list_open_parent_jobs_in_tx(tx)

    async def list_open_parent_jobs_in_tx(self, tx: StateTx) -> list[Any]:
        """``list_open_parent_jobs`` against a caller-provided transaction.

        Lets callers (e.g. the StateDB-backed job queue's admission pass) read
        the open top-level jobs inside their own write transaction, reusing the
        SQL-level ``workflow_end_state IS NULL`` filter and orphan detection
        instead of scanning every historical job row.
        """
        training_jobs = await self._list_open_job_payloads_where(
            tx, self._training_jobs)
        batch_rows = await self._list_open_job_payloads_where(
            tx,
            self._batch_prediction_jobs,
            [_batch_prediction_has_execution_clause(tx)],
            [],
        )
        generate_train_table_jobs = (
            await self._list_open_orphan_generate_train_table_jobs(tx))
        generate_pred_table_jobs = (
            await self._list_open_orphan_generate_pred_table_jobs(tx))

        return [
            *training_jobs,
            *batch_rows,
            *generate_train_table_jobs,
            *generate_pred_table_jobs,
        ]

    async def _get_if_exists(
        self,
        tx: StateTx,
        repository: JsonRepository,
        id: JobID,
    ) -> Any | None:
        row = await repository.get(tx, str(id))
        return row.payload if row is not None else None

    async def _get_required(
        self,
        tx: StateTx,
        repository: JsonRepository,
        id: JobID,
    ) -> Any:
        job = await self._get_if_exists(tx, repository, id)
        if job is None:
            raise KeyError(f"Job with ID '{id}' does not exist.")
        return job

    async def _list_rows_by_ids(
        self,
        tx: StateTx,
        repository: JsonRepository,
        ids: Iterable[JobID],
    ) -> list[Any]:
        id_values = [str(id) for id in ids]
        if not id_values:
            return []
        rows = await repository.list_by_ids(tx, id_values)
        rows_by_id = {row.id: row for row in rows}
        return [rows_by_id[id] for id in id_values if id in rows_by_id]

    async def _list_payloads_by_ids(
        self,
        tx: StateTx,
        repository: JsonRepository,
        ids: Iterable[JobID],
    ) -> list[Any]:
        return [
            row.payload
            for row in await self._list_rows_by_ids(tx, repository, ids)
        ]

    async def _delete_batch_prediction_ids_in_tx(
        self,
        tx: StateTx,
        ids: Iterable[BatchPredictionJobID],
        *,
        force: bool,
    ) -> list[str]:
        id_values = [str(id) for id in ids]
        if not id_values:
            return []

        status_expr = _json_path_value(tx, "execution", "status")
        args: list[Any] = list(id_values)
        where_clauses = [
            self._in_predicate(tx, "id", id_values, 1),
            f"{status_expr} IS NOT NULL",
        ]
        if not force:
            submitted_statuses = [
                status.value for status in JobStatus if status.is_submitted
            ]
            where_clauses.append(
                self._not_in_predicate(
                    tx,
                    status_expr,
                    submitted_statuses,
                    len(args) + 1,
                ))
            args.extend(submitted_statuses)

        rows = await tx.fetch(
            f"""
            DELETE FROM {self._batch_prediction_jobs._table_name}
            WHERE {" AND ".join(where_clauses)}
            RETURNING id
            """,
            *args,
        )
        return [row_str(row, "id", 0) for row in rows]

    def _repository_for_job(self, job: Any) -> JsonRepository:
        if isinstance(job, TrainingJobDBEntry):
            return self._training_jobs
        if isinstance(job, GenerateTrainTableJobDBEntry):
            return self._generate_train_table_jobs
        if isinstance(job, GeneratePredTableJobDBEntry):
            return self._generate_pred_table_jobs
        raise AssertionError(f"Unsupported job db entry type: {type(job)}")

    def _repository_for_job_type(self, job_type: JobType) -> JsonRepository:
        if job_type == JobType.TRAINING_JOB:
            return self._training_jobs
        if job_type == JobType.BATCH_PREDICTION_JOB:
            return self._batch_prediction_jobs
        if job_type == JobType.GENERATE_TRAIN_TABLE_JOB:
            return self._generate_train_table_jobs
        if job_type == JobType.GENERATE_PREDICTION_TABLE_JOB:
            return self._generate_pred_table_jobs
        raise AssertionError(f"Unsupported job type {job_type}")

    async def _list_payloads_where(
        self,
        tx: StateTx,
        repository: JsonRepository,
        where_clauses: list[str],
        args: list[Any],
        *,
        sort_created_at: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Any]:
        where_sql = (" WHERE " +
                     " AND ".join(where_clauses) if where_clauses else "")
        order_sql = "ORDER BY id"
        if sort_created_at is not None:
            if sort_created_at not in ("asc", "desc"):
                raise ValueError(
                    f"Unsupported created_at sort: {sort_created_at}")
            created_expr = _json_path_value(tx, "timestamps", "created_at")
            order_sql = f"ORDER BY {created_expr} {sort_created_at.upper()}, id"
        query_args = list(args)
        limit_sql = ""
        if limit is not None:
            query_args.append(limit)
            limit_sql = f" LIMIT {_placeholder(tx, len(query_args))}"
        offset_sql = ""
        if offset:
            if limit is None and tx.backend == "sqlite":
                limit_sql = " LIMIT -1"
            query_args.append(offset)
            offset_sql = f" OFFSET {_placeholder(tx, len(query_args))}"
        rows = await tx.fetch(
            f"""
            SELECT id, payload, created_at, updated_at
            FROM {repository._table_name}
            {where_sql}
            {order_sql}
            {limit_sql}
            {offset_sql}
            """,
            *query_args,
        )
        return [repository._row_from_list_row(row).payload for row in rows]

    async def _list_open_job_payloads_where(
        self,
        tx: StateTx,
        repository: JsonRepository,
        where_clauses: list[str] | None = None,
        args: list[Any] | None = None,
    ) -> list[Any]:
        return await self._list_payloads_where(
            tx,
            repository,
            [
                *([] if where_clauses is None else where_clauses),
                f"{_json_path_value(tx, 'execution', 'workflow_end_state')} "
                "IS NULL",
            ],
            [] if args is None else args,
        )

    async def _list_open_orphan_generate_train_table_jobs(
        self,
        tx: StateTx,
    ) -> list[GenerateTrainTableJobDBEntry]:
        child_id = f"{self._generate_train_table_jobs._table_name}.id"
        training_parent_id = self._child_parent_id_expr(
            tx,
            child_id,
            parent_prefix="trainingjob-",
            child_prefix="gen-traintable-job-",
        )
        return await self._list_open_job_payloads_where(
            tx,
            self._generate_train_table_jobs,
            [
                "NOT EXISTS ("
                f"SELECT 1 FROM {self._training_jobs._table_name} parent "
                f"WHERE parent.id = {training_parent_id})",
            ],
            [],
        )

    async def _list_open_orphan_generate_pred_table_jobs(
        self,
        tx: StateTx,
    ) -> list[GeneratePredTableJobDBEntry]:
        child_id = f"{self._generate_pred_table_jobs._table_name}.id"
        batch_parent_id = self._child_parent_id_expr(
            tx,
            child_id,
            parent_prefix="bp-job-",
            child_prefix="gen-predtable-job-",
        )
        return await self._list_open_job_payloads_where(
            tx,
            self._generate_pred_table_jobs,
            [
                "NOT EXISTS ("
                f"SELECT 1 FROM {self._batch_prediction_jobs._table_name} "
                f"parent WHERE parent.id = {batch_parent_id})",
            ],
            [],
        )

    @staticmethod
    def _child_parent_id_expr(
        tx: StateTx,
        child_id_expr: str,
        *,
        parent_prefix: str,
        child_prefix: str,
    ) -> str:
        suffix_start = len(child_prefix) + 1
        if tx.backend == "postgres":
            return (f"'{parent_prefix}' || substring({child_id_expr} "
                    f"from {suffix_start})")
        return f"'{parent_prefix}' || substr({child_id_expr}, {suffix_start})"

    async def _list_ids_where(
        self,
        tx: StateTx,
        repository: JsonRepository,
        where_clauses: list[str],
        args: list[Any],
    ) -> list[str]:
        where_sql = (" WHERE " +
                     " AND ".join(where_clauses) if where_clauses else "")
        rows = await tx.fetch(
            f"""
            SELECT id
            FROM {repository._table_name}
            {where_sql}
            ORDER BY id
            """,
            *args,
        )
        return [row_str(row, "id", 0) for row in rows]

    def _batch_prediction_where(
        self,
        tx: StateTx,
        *,
        project_ids: Collection[ProjectID] | None,
        model_id: TrainingJobID | None = None,
        pquery_id: PredictiveQueryID | None = None,
        job_ids: Iterable[BatchPredictionJobID] | None = None,
        status: JobStatus | None = None,
        match_pquery_def_fpid: bool = True,
    ) -> tuple[list[str], list[Any]]:
        clauses, args = self._project_ids_where(tx, project_ids)
        if job_ids is not None:
            job_id_args = sorted(str(job_id) for job_id in job_ids)
            clauses.append(
                self._in_predicate(tx, "id", job_id_args,
                                   len(args) + 1))
            args.extend(job_id_args)
        if model_id is not None:
            args.append(str(model_id))
            clauses.append(f"{_json_value(tx, 'training_job_id')} = "
                           f"{_placeholder(tx, len(args))}")
        if status is not None:
            args.append(status.value)
            clauses.append(f"{_json_value(tx, 'status')} = "
                           f"{_placeholder(tx, len(args))}")
        if pquery_id is not None:
            pquery_clauses = []
            args.append(str(pquery_id))
            pquery_clauses.append(f"{_json_value(tx, 'pquery_id')} = "
                                  f"{_placeholder(tx, len(args))}")
            _, fpid = PredictiveQueryID.parse(pquery_id)
            if (match_pquery_def_fpid and fpid is not None
                    and PredictiveQueryID.compose(fpid=fpid) == pquery_id):
                args.append(str(fpid))
                pquery_clauses.append(f"{_json_value(tx, 'pquery_def_fpid')} = "
                                      f"{_placeholder(tx, len(args))}")
            clauses.append("(" + " OR ".join(pquery_clauses) + ")")
        return clauses, args

    def _project_ids_where(
        self,
        tx: StateTx,
        project_ids: Collection[ProjectID] | None,
    ) -> tuple[list[str], list[Any]]:
        """Build a required, explicit project scope for a job query.

        ``None`` deliberately means an internal/admin global scope. An empty
        collection is an explicit no-access scope and matches no rows.
        """
        if project_ids is None:
            return [], []
        return self._project_visibility_where(tx, set(project_ids))

    async def _list_project_scoped_payloads_by_ids(
        self,
        repository: JsonRepository,
        ids: Iterable[object],
        *,
        project_ids: Collection[ProjectID] | None,
        sort_created_at: str | None = None,
        limit: int | None = None,
        copy_payloads: bool = False,
    ) -> list[Any]:
        async with self._state_db.read_tx() as tx:
            clauses, args = self._project_ids_where(tx, project_ids)
            id_args = sorted(str(id) for id in ids)
            clauses.append(self._in_predicate(tx, "id", id_args, len(args) + 1))
            args.extend(id_args)
            payloads = await self._list_payloads_where(
                tx,
                repository,
                clauses,
                args,
                sort_created_at=sort_created_at,
                limit=limit,
            )
        if copy_payloads:
            return [deepcopy(payload) for payload in payloads]
        return payloads

    def _in_predicate(
        self,
        tx: StateTx,
        expression: str,
        values: Iterable[object],
        first_position: int,
    ) -> str:
        values = list(values)
        if not values:
            return "1 = 0"
        placeholders = [
            _placeholder(tx, position)
            for position in range(first_position, first_position + len(values))
        ]
        return f"{expression} IN ({', '.join(placeholders)})"

    def _not_in_predicate(
        self,
        tx: StateTx,
        expression: str,
        values: Iterable[object],
        first_position: int,
    ) -> str:
        values = list(values)
        if not values:
            return "1 = 1"
        placeholders = [
            _placeholder(tx, position)
            for position in range(first_position, first_position + len(values))
        ]
        return f"{expression} NOT IN ({', '.join(placeholders)})"

    async def _list_registry_entries(self) -> list[MLJobRegistryEntry]:
        async with self._state_db.read_tx() as tx:
            training_jobs = await self._training_jobs.list_all(tx)
            batch_prediction_rows = await self._batch_prediction_jobs.list_all(
                tx)
            generate_train_table_jobs = (
                await self._generate_train_table_jobs.list_all(tx))
            generate_pred_table_jobs = (
                await self._generate_pred_table_jobs.list_all(tx))

        return [
            *[
                self._registry_entry_from_job(self._repository_payload(job),
                                              JobType.TRAINING_JOB)
                for job in training_jobs
            ],
            *[
                self._registry_entry_from_job(
                    self._repository_payload(row),
                    JobType.BATCH_PREDICTION_JOB,
                ) for row in batch_prediction_rows
            ],
            *[
                self._registry_entry_from_job(
                    self._repository_payload(job),
                    JobType.GENERATE_TRAIN_TABLE_JOB,
                ) for job in generate_train_table_jobs
            ],
            *[
                self._registry_entry_from_job(
                    self._repository_payload(job),
                    JobType.GENERATE_PREDICTION_TABLE_JOB,
                ) for job in generate_pred_table_jobs
            ],
        ]

    @staticmethod
    def _repository_payload(row: Any) -> Any:
        return getattr(row, "payload", row)

    def _registry_entry_from_job(
        self,
        job: Any,
        job_type: JobType,
    ) -> MLJobRegistryEntry:
        metadata = getattr(job, "registry_metadata", None)
        execution = getattr(job, "execution", None)
        config = getattr(job, "config", None)
        timestamps = getattr(job, "timestamps", None)
        status = getattr(execution, "status", None)
        log_entry = getattr(execution, "log", None)
        pquery_id = None
        trained_job_id = None
        if config is not None:
            pquery_id = getattr(config, "pquery_id", None)
            if pquery_id is None:
                pquery_id = getattr(config, "predictive_query_id", None)
            trained_job_id = getattr(config, "training_job_id", None)
        if pquery_id is None:
            pquery_id = getattr(job, "pquery_id", None)
        metadata_pquery_id = self._metadata_value(metadata, "pquery_id")
        metadata_trained_job_id = self._metadata_value(metadata,
                                                       "trained_job_id")
        if metadata_pquery_id is not None:
            pquery_id = metadata_pquery_id
        if metadata_trained_job_id is not None:
            trained_job_id = metadata_trained_job_id

        start_time = None
        end_time = None
        if timestamps is not None:
            start_time = timestamps.created_at
            is_terminal = (
                status.is_terminal if hasattr(status, "is_terminal") else
                JobStatus(status).is_terminal if status is not None else False)
            if is_terminal:
                end_time = timestamps.updated_at
        status_value = None if status is None else getattr(
            status, "value", status)

        return MLJobRegistryEntry(
            job_id=job.id,
            job_type=job_type,
            instance_id=self._metadata_value(metadata, "instance_id"),
            pquery_id=None if pquery_id is None else str(pquery_id),
            start_time=start_time,
            end_time=end_time,
            storage_path=getattr(job, "storage_path", None),
            input_data_stats=getattr(job, "input_data_stats", {}),
            kumo_version=(self._metadata_value(metadata, "kumo_version")
                          or getattr(job, "kumo_version", None)),
            pquery_str=self._metadata_value(metadata, "pquery_str"),
            generator_advanced_options=(self._metadata_value(
                metadata, "generator_advanced_options")),
            trained_job_id=None
            if trained_job_id is None else str(trained_job_id),
            bp_filters=self._metadata_value(metadata, "bp_filters"),
            override_model_path=self._metadata_value(metadata,
                                                     "override_model_path"),
            status=status_value,
            stage=getattr(log_entry, "stage_name", None),
        )

    @staticmethod
    def _registry_entry_matches_filters(
        entry: MLJobRegistryEntry,
        filters: Mapping[str, list[str] | str],
    ) -> bool:
        valid_fields = MLJobRegistryEntry.__dataclass_fields__
        for key, value in filters.items():
            if key not in valid_fields:
                continue
            entry_value = getattr(entry, key)
            if isinstance(value, list):
                if key in _RANGE_JOB_ATTRS and len(value) == 2:
                    lower_bound = MLJobService._try_time_range_format(value[0])
                    upper_bound = MLJobService._try_time_range_format(value[1])
                    if lower_bound and (entry_value is None
                                        or entry_value < lower_bound):
                        return False
                    if upper_bound and (entry_value is None
                                        or entry_value > upper_bound):
                        return False
                    continue
                expected_values = {
                    MLJobService._registry_filter_value(item)
                    for item in value
                }
                if (MLJobService._registry_filter_value(entry_value)
                        not in expected_values):
                    return False
                continue
            if (MLJobService._registry_filter_value(entry_value)
                    != MLJobService._registry_filter_value(value)):
                return False
        return True

    @staticmethod
    def _metadata_value(metadata: Any, key: str) -> Any:
        if metadata is None:
            return None
        if isinstance(metadata, Mapping):
            return metadata.get(key)
        return getattr(metadata, key, None)

    @staticmethod
    def _registry_filter_value(value: object) -> str:
        return str(getattr(value, "value", value))

    @staticmethod
    def _try_time_range_format(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.strptime(value, _TIME_RANGE_FORMAT)
        except (TypeError, ValueError):
            return None

    def _project_where(
        self,
        tx: StateTx,
        *,
        project_id: ProjectID,
    ) -> tuple[list[str], list[Any]]:
        project_expr = _json_value(tx, "project_id")
        return [f"{project_expr} = {_placeholder(tx, 1)}"], [str(project_id)]

    def _project_visibility_where(
        self,
        tx: StateTx,
        visible_project_ids: set[ProjectID],
    ) -> tuple[list[str], list[Any]]:
        project_expr = _json_value(tx, "project_id")
        project_args = sorted(
            str(project_id) for project_id in visible_project_ids)
        project_clauses = []
        if project_args:
            project_clauses.append(
                self._in_predicate(tx, project_expr, project_args, 1))
        if not project_clauses:
            return ["1 = 0"], []
        return ["(" + " OR ".join(project_clauses) + ")"], project_args

    async def _get_job_if_exists_generic_in_tx(
        self,
        tx: StateTx,
        id: JobID,
    ) -> Any | None:
        job_type = JobType.from_id(id)
        if job_type == JobType.TRAINING_JOB:
            return await self._get_if_exists(tx, self._training_jobs,
                                             TrainingJobID(id))
        if job_type == JobType.BATCH_PREDICTION_JOB:
            return await self._get_if_exists(tx, self._batch_prediction_jobs,
                                             BatchPredictionJobID(id))
        if job_type == JobType.GENERATE_TRAIN_TABLE_JOB:
            return await self._get_if_exists(
                tx,
                self._generate_train_table_jobs,
                GenerateTrainTableJobID(id),
            )
        if job_type == JobType.GENERATE_PREDICTION_TABLE_JOB:
            return await self._get_if_exists(
                tx,
                self._generate_pred_table_jobs,
                GeneratePredictionTableJobID(id),
            )
        raise AssertionError(f"Unsupported job type {job_type}")

    async def _save_job_entry_if_open(
        self,
        tx: StateTx,
        job: Any,
    ) -> bool:
        if isinstance(job, BatchPredictionJobDBEntry):
            existing = await self._get_required(tx, self._batch_prediction_jobs,
                                                job.id)
            existing_execution = existing.execution
            if (existing_execution is not None
                    and existing_execution.workflow_end_state):
                log.warning(
                    "ML job %s is already in terminal state, reject "
                    "further modification: %s",
                    job.id,
                    job,
                )
                return False
            _enforce_job_output_invariants(job)
            now = now_utc()
            if (existing_execution.status == JobStatus.QUEUED
                    and job.execution.status != JobStatus.QUEUED):
                job.execution.start_time = now
            else:
                job.execution.start_time = existing_execution.start_time
            job.execution.last_updated_at = now
            job._sync_index_fields()
            await self._batch_prediction_jobs.upsert(tx, str(job.id), job)
            return True

        return await self._modify_if_open(tx, self._repository_for_job(job),
                                          deepcopy(job))

    async def _modify_job_if_open_in_tx(
        self,
        tx: StateTx,
        job: Any,
    ) -> None:
        """Save a job row, terminalizing its implicit child alongside it.

        Every write that can move a job to a terminal state must route here.
        Parent-before-child is the canonical cross-table lock order.
        """
        # Only a write the open-gate accepts may sweep the child, so a late or
        # replayed write to an already-terminal parent cannot relabel a child
        # that parent never owned.
        if not await self._save_job_entry_if_open(tx, job):
            return
        if not job.execution.status.is_terminal:
            return
        child_job = await self._find_child_job_in_tx(tx, job)
        if child_job and not child_job.execution.status.is_terminal:
            child_status = child_job.execution.status
            # A swept child never reported its own outcome, so its row holds no
            # generated-table path.  CANCELLED rather than DONE keeps it out of
            # the reuse gates in ``create_training_job`` and
            # ``create_batch_prediction_job``, which reject only FAILED and
            # CANCELLED; a pathless DONE row would pass them and then trip the
            # ``train_table_path`` assertion in ``MLJobBackend``.
            # ``workflow_end_state`` freezes the row on purpose: a child that
            # later reports a failure of its own has that write rejected with a
            # log warning rather than reopening a closed job.
            child_job.execution.status = JobStatus.CANCELLED
            child_job.execution.workflow_end_state = (
                WorkflowExecutionStatus.CANCELED)
            child_job.execution.log = JobEventLogEntry.cancelled(
                message=(f"This job is cancelled by its parent job "
                         f"{job.id} which has reached "
                         f"{job.execution.status.value} state"))
            log.info(
                "Updating child job %s status from %s to %s as its "
                "parent job becomes %s",
                child_job.id,
                child_status,
                child_job.execution.status,
                job.execution.status,
            )
            await self._save_job_entry_if_open(tx, child_job)

    async def _modify_job_metadata_if_open_in_tx(
        self,
        tx: StateTx,
        job: Any,
    ) -> None:
        await self._save_job_entry_if_open(tx, job)

    async def _find_child_job_in_tx(
        self,
        tx: StateTx,
        parent_job: Any,
    ) -> Any | None:
        if isinstance(parent_job, TrainingJobDBEntry):
            child_id = GenerateTrainTableJobID.as_child_of(parent_job.id)
        elif isinstance(parent_job, BatchPredictionJobDBEntry):
            child_id = GeneratePredictionTableJobID.as_child_of(parent_job.id)
        else:
            return None
        return await self._get_job_if_exists_generic_in_tx(tx, child_id)

    async def _delete_job_in_tx(
        self,
        tx: StateTx,
        job: Any,
    ) -> None:
        if isinstance(job, TrainingJobDBEntry):
            await self._training_jobs.delete(tx, str(job.id))
        elif isinstance(job, BatchPredictionJobDBEntry):
            await self._batch_prediction_jobs.delete(tx, str(job.id))
        elif isinstance(job, GenerateTrainTableJobDBEntry):
            await self._generate_train_table_jobs.delete(tx, str(job.id))
        elif isinstance(job, GeneratePredTableJobDBEntry):
            await self._generate_pred_table_jobs.delete(tx, str(job.id))
        else:
            raise AssertionError(f"Unsupported job db entry type: {type(job)}")
        await self._job_tags_repository.delete_by_ids(
            tx,
            JobType.from_id(job.id),
            [job.id],
        )

    async def _create_generate_pred_table_job_in_tx(
        self,
        tx: StateTx,
        *,
        id: GeneratePredictionTableJobID,
        config: GeneratePredTableJobConfig,
        now,
        source=None,
        user: str | None = None,
        project_id: ProjectID,
    ) -> None:
        await self._generate_pred_table_jobs.upsert(
            tx,
            str(id),
            GeneratePredTableJobDBEntry(
                id=id,
                timestamps=ResourceTimestamps(now, now),
                config=config,
                execution=GeneratePredTableJobOutput(
                    status=JobStatus.QUEUED,
                    log=JobEventLogEntry.not_started(now),
                    start_time=None,
                    last_updated_at=now,
                ),
                source=source,
                user=user,
                project_id=project_id,
            ),
        )

    async def _create_generate_train_table_job_in_tx(
        self,
        tx: StateTx,
        *,
        id: GenerateTrainTableJobID,
        config: GenerateTrainTableJobConfig,
        now,
        source=None,
        user_info: str | None = None,
        project_id: ProjectID,
    ) -> None:
        await self._generate_train_table_jobs.upsert(
            tx,
            str(id),
            GenerateTrainTableJobDBEntry(
                id=id,
                timestamps=ResourceTimestamps(now, now),
                config=config,
                execution=GenerateTrainTableJobOutput(
                    status=JobStatus.QUEUED,
                    log=JobEventLogEntry.not_started(now),
                    start_time=None,
                    last_updated_at=now,
                ),
                source=source,
                user=user_info,
                project_id=project_id,
            ),
        )

    async def _save_job_output_if_open(
        self,
        tx: StateTx,
        repository: JsonRepository,
        id: JobID,
        output: Any,
        missing_message: str | None,
    ) -> None:
        if missing_message is None:
            job = await self._get_required(tx, repository, id)
        else:
            job = await self._get_if_exists(tx, repository, id)
            if job is None:
                raise ValueError(missing_message)
        job.execution = output
        # A workflow that writes its own terminal output also sets
        # ``workflow_end_state``, so ``finalize_job``'s safety net never runs
        # and this write owns terminalizing the child.
        await self._modify_job_if_open_in_tx(tx, job)

    @staticmethod
    def _set_graph_data_version(job: Any, graph_data_version: str) -> None:
        assert (job.config.graph_data_version is None
                or job.config.graph_data_version == graph_data_version)
        job.config.graph_data_version = graph_data_version

    async def _modify_if_open(
        self,
        tx: StateTx,
        repository: JsonRepository,
        job: Any,
    ) -> bool:
        existing_job = await self._get_required(tx, repository, job.id)
        existing_execution = getattr(existing_job, "execution", None)
        if (existing_execution is not None
                and existing_execution.workflow_end_state):
            log.warning(
                "ML job %s is already in terminal state, reject further "
                "modification: %s",
                job.id,
                job,
            )
            return False
        _enforce_job_output_invariants(job)
        now = now_utc()
        if (existing_execution is not None
                and existing_execution.status == JobStatus.QUEUED
                and job.execution.status != JobStatus.QUEUED):
            job.execution.start_time = now
        elif existing_execution is not None:
            job.execution.start_time = existing_execution.start_time
        # An execution-less legacy row has no reliable prior runtime boundary.
        # Preserve the caller-provided value rather than inventing one.
        job.execution.last_updated_at = now
        await repository.upsert(tx, str(job.id), job)
        return True
