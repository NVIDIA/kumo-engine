# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import kumoapi.common as common_api
import kumoapi.jobs as api
from kumo_core.job_config import TrainDirOverrides
from kumo_core.model_plan import ModelPlan, TrainingTableGenerationPlan
from kumo_core.types.schema import FPID
from temporalio.client import (
    WorkflowExecutionDescription,
    WorkflowExecutionStatus,
    WorkflowFailureError,
)
from temporalio.exceptions import (
    ActivityError,
    ChildWorkflowError,
    WorkflowAlreadyStartedError,
)

from async_utils.sync import blocking_method
from io_utils import fsutil, json, logging
from kumo.config.mixin import construct_object_string_from_config
from kumo.config.model_plan_adapter import to_predictive_query_kwargs
from kumo.config.task import TaskConfig
from kumo.home import HomeDirManager
from kumo.utils.formatting import format_exception
from kumo.validation import ValidationResponse
from rest.backends.ml_backend.activities.ml_backend_activities import (
    create_job_registry_metadata_in_state_db,
    persist_job_inputs,
)
from rest.backends.ml_backend.data_util import get_job_base_dir, get_job_dir
from rest.backends.ml_backend.job_queue import (
    JobConcurrencyLimitError,
    StateDbJobQueue,
)
from rest.backends.ml_backend.ml_job_output_store import (
    MLJobOutputStore,
    get_many_materialization_info,
    get_materialization_info,
)
from rest.backends.ml_backend.settings import MLBackendSettings, MLJobSettings
from rest.backends.ml_backend.workflows import (
    BatchPredictionParentWorkflow,
    GeneratePredictionTableWorkflow,
    GeneratePredTableResult,
    GenerateTrainingTableWorkflow,
    GenerateTrainTableResult,
    TrainingWorkflow,
)
from rest.config.batch_prediction_job import (
    BatchPredictionJobInput,
    BatchPredictionJobOutput,
    BatchPredictionJobResults,
)
from rest.config.data_backend import (
    ConnectorExecutionRef,
    GraphDataVersionInfo,
)
from rest.config.error_handling import extract_failure_info
from rest.config.graph_v2 import GraphDataComputeInput
from rest.config.ml_backend import MaterializationTaskInfo, TrainingJobInput
from rest.config.mlops import MLOpsQueryConfig, MLOpsStats, MLOpsStatType
from rest.config.training_job import (
    ColumnContributionSummary,
    Metric,
    TestExplanation,
    TrainingJobOutput,
    TrainingJobResponse,
    TrainingJobResults,
    TrainingSpec,
)
from rest.constants import ML_WORKFLOW_TASK_QUEUE
from rest.deps.authentication import (
    RequestContext,
    visible_project_ids_or_none,
)
from rest.deps.context import _current_user_ctx
from rest.utils.public import get_base_kumo_url
from statedb.config.common import (
    BatchPredictionJobID,
    BatchPredictionWorkflowID,
    DataSnapshot,
    GeneratePredictionTableJobID,
    GenerateTrainTableJobID,
    GraphIDv2,
    JobEventLogEntry,
    JobExecution,
    JobID,
    JobSource,
    JobStatus,
    JobType,
    PredictiveQueryID,
    ProjectID,
    TrainingJobID,
    now_utc,
    to_graph_id,
)
from statedb.config.job_tag import JobTagKey, JobTags, JobTagValue
from statedb.db.ml_jobs import (
    GeneratePredTableJobConfig,
    GeneratePredTableJobDBEntry,
    GeneratePredTableJobOutput,
    GenerateTrainTableJobConfig,
    GenerateTrainTableJobDBEntry,
    GenerateTrainTableJobOutput,
    GenericJob,
    ListJobsQuery,
    MLJobService,
    TrainingJobDBEntry,
)
from statedb.db.runtime import StateDB
from statedb.ml.jobs import BatchPredictionJobDBEntry

# isort: off
from statedb.rest.rest_api_db_entries import (
    DeprecatedAdvancedAutoTrainerOptions, )
# isort: on
from temporalio_commons import TemporalClient, WFError, is_cancelled_exception
from temporalio_commons.error_utils import wf_failure_to_temporal_trace

log = logging.getLogger(__name__)

# Fake-backend completion fixtures (unit tests only): the duration/iteration
# count and explanation reported when a fake job is explicitly marked DONE via
# mock_training_job_complete / mock_bp_job_complete. The fake backend no longer
# simulates timed status progression, so there is no time-to-schedule constant.
MOCK_TRAINING_TIME_SEC = 120
MOCK_CONTRIBUTION = TestExplanation(ColumnContributionSummary([]))
# Cancellation is best effort, the target workflow might not be immediately
# cancelled after cancellation request is made.
CANCELLATION_MAX_WAIT_SECONDS = 5.0

MLJobDBEntry = Union[
    TrainingJobDBEntry,
    BatchPredictionJobDBEntry,
    GenerateTrainTableJobDBEntry,
    GeneratePredTableJobDBEntry,
]


class MLJobBackend:
    """
    Backend service to perform Training and Batch Prediction operations.
    """
    def __init__(
            self,
            settings: MLBackendSettings,
            temporal_client: TemporalClient,
            state_db: Optional[StateDB] = None,
            now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        log.info(f"Initializing MLBackend with {settings}")
        self._settings = settings
        self._ml_job_settings = settings.ml_job_settings
        self._now_fn = now_fn
        self._temporal_client = temporal_client
        self._job_output_store = MLJobOutputStore()
        self._cancellation_tasks: Set[asyncio.Task] = set()

        # StateDB-backed services are late-bound: the REST/public apps
        # construct MLJobBackend before their StateDB is open and call
        # `_configure_state_db()` in startup; workers pass `state_db` here.
        self._state_db: Optional[StateDB] = None
        self._ml_job_service: Optional[MLJobService] = None
        self._job_queue: Optional[StateDbJobQueue] = None
        if state_db is not None:
            self._configure_state_db(state_db)

    def _configure_state_db(self, state_db: StateDB) -> None:
        """Bind the StateDB-backed services (idempotent)."""
        self._state_db = state_db
        self._ml_job_service = MLJobService(state_db)
        # DB-backed admission queue (replaces the AutoRPC MLJobQueueManager).
        self._job_queue = StateDbJobQueue(state_db,
                                          self._settings.job_queue_settings)
        self._job_output_store._configure_ml_job_service(self._ml_job_service)

    def _require_ml_job_service(self) -> MLJobService:
        ml_job_service = getattr(self, "_ml_job_service", None)
        if ml_job_service is None:
            raise RuntimeError(
                "MLJobService is not configured for MLJobBackend. "
                "Call start() before accessing ML job metadata.")
        return ml_job_service

    async def get_default_ml_job_settings(self) -> MLJobSettings:
        return self._ml_job_settings

    # TODO(siyang): Deprecate start/shutdown of background thread once temporal
    # workflow migration completes.
    #
    # =============== Start & Shutdown =============== #

    async def start(self, read_only: bool = False):
        if self._state_db is None:
            raise RuntimeError(
                "MLJobBackend.start() requires a configured StateDB; pass "
                "state_db to the constructor or call _configure_state_db() "
                "first.")
        await self._temporal_client.init()
        await self._configure_state_db_services()

        # read_only consumers (in-process REST/public apps) only wire services
        # + init Temporal; the single lifecycle owner (read_only=False) performs
        # reconciliation and queue admission recovery so they run once across
        # processes. Both passes are idempotent and concurrency-safe.
        if read_only:
            return

        # Reconcile jobs whose Temporal workflow has vanished (server data
        # loss / retention expiry / a worker crash before the workflow's
        # finally-block finalize ran).  Such a job would otherwise stay
        # RUNNING/QUEUED forever, permanently occupying its queue slot and
        # starving everything behind it.  This restores the recovery that
        # the removed watcher's `_handle_workflow_not_found` used to do.
        await self._reconcile_lost_workflows()

        # Restart recovery: re-run admission against the DB-backed queue so any
        # jobs left QUEUED before a restart get a chance to start once capacity
        # is recomputed from the persisted job rows (and any slots just freed
        # by reconciliation become available).
        await self._job_queue.drain()

    async def _configure_state_db_services(self) -> None:
        self._job_output_store._configure_ml_job_service(self._ml_job_service)

    async def _reconcile_lost_workflows(self) -> None:
        """Fail-and-release jobs whose Temporal workflow no longer exists.

        Finalization is normally owned by each workflow's ``finally`` block
        (``finalize_job``).  But if a workflow disappears before that runs
        (Temporal data loss, retention expiry, or a hard worker crash mid
        finalize), the job is stranded ``RUNNING``/``QUEUED`` in StateDB,
        holding its queue slot indefinitely.

        This sweep lists open (non-terminal, un-frozen) top-level jobs and, for
        any whose workflow describe returns NOT_FOUND, marks it failed via
        ``mark_workflow_not_found`` and releases its queue slot.  It is
        idempotent and best-effort: per-job errors are logged and skipped so
        one bad job can't block startup.  The caller runs admission
        (``drain``) afterwards to fill the freed capacity.
        """
        service = self._require_ml_job_service()
        try:
            open_jobs = await service.list_open_parent_jobs()
        except Exception:
            log.exception(
                "Lost-workflow reconciliation: failed to list open jobs; "
                "skipping sweep")
            return

        for job in open_jobs:
            job_id = job.id
            try:
                wf_desc = await self._temporal_client.get_workflow_desc(
                    _job_workflow_id(job_id))
                if wf_desc is not None:
                    continue  # workflow still exists; leave it alone
                log.warning(
                    "Lost-workflow reconciliation: workflow for %s not found; "
                    "marking failed and releasing its queue slot", job_id)
                await service.mark_workflow_not_found(job_id, self._now_fn())
                await self._job_queue.release_job(job_id)
            except Exception:
                log.exception(
                    "Lost-workflow reconciliation: error reconciling %s; "
                    "skipping", job_id)

    # =============== Training =============== #
    async def _get_job_if_exists_generic(self,
                                         id: JobID) -> Optional[MLJobDBEntry]:
        return await self._require_ml_job_service().get_job_if_exists_generic(id
                                                                              )

    async def _submit_created_job(
        self,
        job_id: JobID,
        *,
        delete_created_child: bool,
    ) -> None:
        try:
            await self._job_queue.submit_job(job_id)
        except JobConcurrencyLimitError:
            # Admission follows StateDB creation but precedes Temporal.
            await self._require_ml_job_service().delete_job_generic(
                job_id,
                force=True,
                delete_created_child=delete_created_child,
            )
            raise

    async def start_generate_train_table(
        self,
        training_spec: TrainingSpec,
        graph_data_compute_input: GraphDataComputeInput,
        pquery_def_fpid: Optional[FPID],
        *,
        project_id: ProjectID,
        graph_version_info: Optional[GraphDataVersionInfo] = None,
        job_source: Optional[JobSource] = None,
        registered_custom_table: Optional[api.CustomTrainingTable] = None,
    ) -> GenerateTrainTableJobID:
        """
        Launch a stand-alone, stateful job to generate training table.

        The workflow always kicks off a graph refresh first: a specified
        graph_version_info pins which snapshot to read, but only the refresh
        guarantees it was built (kumo-ai/kumo#30363).
        """
        # Must have a model plan containing train-table-generation plan.
        self._require_connector_references_in_project(
            graph_data_compute_input.connectors.values(), project_id)
        model_plan = training_spec.advanced_options
        assert isinstance(model_plan, ModelPlan)
        # Make sure to apply ModelPlan as override to PQ config.
        to_predictive_query_kwargs(model_plan).apply_overrides(
            training_spec.predictive_query)
        plan = model_plan.training_table_generation
        ml_job_settings = self._ml_job_settings
        graph_data_version = (None if not graph_version_info else
                              graph_version_info.graph_data_version)
        # A registering job reads a customer connector, so its reference is
        # admitted with the job -- under the same transaction the training
        # path admits one in, rather than left to fail mid-read.
        custom_train_table = getattr(training_spec, 'custom_train_table', None)
        if custom_train_table is not None:
            self._require_connector_references_in_project(
                [custom_train_table.connector_spec], project_id)

        # On the config, so it is written into the job row. The SDK
        # reconstructs a `TrainingTable` from a job ID alone and asks the job
        # record what was registered; a record that cannot answer made
        # `from_source_table` a no-op on the cloud -- the rows were read, and
        # the training run that followed generated its own anyway. The public
        # shape, not `training_spec`'s internal one, so no connector token is
        # persisted or echoed back.
        config = GenerateTrainTableJobConfig(
            training_spec.predictive_query_id,
            plan,
            graph_data_compute_input.graph_key,
            graph_data_version,
            pquery_def_fpid,
            custom_train_table=registered_custom_table,
        )

        service = self._require_ml_job_service()
        gen_train_table_job_id = await service.create_generate_train_table_job(
            config=config,
            now=now_utc(),
            source=job_source,
            project_id=project_id,
            custom_train_table=custom_train_table,
        )
        await self._submit_created_job(
            gen_train_table_job_id,
            delete_created_child=False,
        )

        # GenereateTrainTableJob and TrainingJob both share the same input
        # dataclass `TrainingJobInput`. Here we simply set the job id to
        # "gen-traintrable-job-xxxxxx".
        job_input = TrainingJobInput(
            id=TrainingJobID(gen_train_table_job_id),
            graph_data_compute_input=graph_data_compute_input,
            spec=training_spec,
            start_time=now_utc(),
            graph_version_info=graph_version_info,
            generate_train_table_job_id=gen_train_table_job_id,
        )

        if ml_job_settings.use_fake_job_backend:
            await self._persist_job_fake_backend(gen_train_table_job_id,
                                                 job_input, ml_job_settings)
            return gen_train_table_job_id

        client = await self._temporal_client.get()
        home_dir = await self.get_job_home_dir(gen_train_table_job_id)
        await client.start_workflow(
            GenerateTrainingTableWorkflow.run,
            args=[
                job_input,
                ml_job_settings,
                True,  # run gen-train-table wf in standalone mode
                home_dir,
            ],
            id=gen_train_table_job_id.wf_id,
            task_queue=ML_WORKFLOW_TASK_QUEUE,
        )

        return gen_train_table_job_id

    @staticmethod
    def _raise_if_training_replay_value_changed(
        training_job_id: TrainingJobID,
        existing_value: object,
        requested_value: object,
        field_description: str,
    ) -> None:
        if existing_value != requested_value:
            raise ValueError(
                f'Training job id={training_job_id} already exists for a '
                f'different {field_description}.')

    async def _validate_idempotent_training_replay_with_service(
        self,
        training_job_id: TrainingJobID,
        existing_training_job: TrainingJobDBEntry,
        *,
        pquery_def_fpid: Optional[FPID],
        graph_data_compute_input: GraphDataComputeInput,
        training_spec: TrainingSpec,
        graph_data_version: Optional[str],
        gen_train_table_job_id: Optional[GenerateTrainTableJobID],
        warm_start_job_id: Optional[str],
        project_id: ProjectID,
    ) -> GenerateTrainTableJobID:
        assert self._ml_job_service is not None
        replay_checks = [
            (existing_training_job.pquery_def_fpid, pquery_def_fpid,
             'predictive query definition'),
            (existing_training_job.graph_data_version, graph_data_version,
             'graph data version'),
            (existing_training_job.warm_start_job_id, warm_start_job_id,
             'warm-start job'),
            (existing_training_job.project_id, project_id, 'project'),
        ]
        for existing_value, requested_value, field_description in replay_checks:
            self._raise_if_training_replay_value_changed(
                training_job_id, existing_value, requested_value,
                field_description)

        if existing_training_job.generate_train_table_job_id is None:
            raise ValueError(
                f'Training job id={training_job_id} already exists without '
                'a generate-train-table job id.')

        existing_gen_train_table_job_id = GenerateTrainTableJobID(
            existing_training_job.generate_train_table_job_id)
        existing_gen_train_table_job = (
            await self._ml_job_service.get_generate_train_table_job(
                existing_gen_train_table_job_id,
                project_ids={project_id},
            ))
        if existing_gen_train_table_job is None:
            raise KeyError(
                'Generate-Train-Table Job with id '
                f'{existing_gen_train_table_job_id} not found in database')

        generate_train_table_replay_checks = [
            (existing_gen_train_table_job.config.pquery_id,
             training_spec.predictive_query_id, 'predictive query'),
            (existing_gen_train_table_job.config.graph_id,
             to_graph_id(graph_data_compute_input.graph_key), 'graph'),
        ]
        if gen_train_table_job_id is not None:
            generate_train_table_replay_checks.append(
                (existing_gen_train_table_job_id, gen_train_table_job_id,
                 'generate-train-table job'))

        for existing_value, requested_value, field_description in (
                generate_train_table_replay_checks):
            self._raise_if_training_replay_value_changed(
                training_job_id, existing_value, requested_value,
                field_description)

        return existing_gen_train_table_job_id

    @staticmethod
    def _require_connector_references_in_project(
        references: Iterable[ConnectorExecutionRef],
        project_id: ProjectID,
    ) -> None:
        if not project_id:
            raise ValueError("project_id is required")
        for reference in references:
            if not isinstance(reference, ConnectorExecutionRef):
                raise TypeError(
                    "Job inputs must contain ConnectorExecutionRef values")
            if reference.project_id != project_id:
                raise ValueError(
                    "A job cannot reference a connector from another project")

    async def start_training(
        self,
        pquery_def_fpid: Optional[FPID],
        graph_data_compute_input: GraphDataComputeInput,
        training_spec: TrainingSpec,
        *,
        project_id: ProjectID,
        ml_job_settings: Optional[MLJobSettings] = None,
        graph_version_info: Optional[GraphDataVersionInfo] = None,
        gen_train_table_job_id: Optional[GenerateTrainTableJobID] = None,
        job_source: Optional[JobSource] = None,
        warm_start_job_id: Optional[str] = None,
        user_info: Optional[str] = None,
        reserved_job_id: Optional[TrainingJobID] = None,
    ) -> TrainingJobID:
        """Start a training workflow."""
        service = self._require_ml_job_service()
        self._require_connector_references_in_project(
            graph_data_compute_input.connectors.values(), project_id)
        advanced_options = training_spec.advanced_options
        graph_data_version = (None if not graph_version_info else
                              graph_version_info.graph_data_version)
        existing_training_job = (await service.get_training_job(reserved_job_id)
                                 if reserved_job_id is not None else None)

        if existing_training_job is None:
            created_child = gen_train_table_job_id is None
            training_job_id, gen_train_table_job_id = (
                await service.create_training_job(
                    pquery_id=training_spec.predictive_query_id,
                    pquery_def_fpid=pquery_def_fpid,
                    graph_id=to_graph_id(graph_data_compute_input.graph_key),
                    now=self._now_fn(),
                    advanced_options=advanced_options,
                    graph_data_version=graph_data_version,
                    generate_train_table_job_id=gen_train_table_job_id,
                    job_source=job_source,
                    custom_train_table=training_spec.custom_train_table,
                    warm_start_job_id=warm_start_job_id,
                    user_info=user_info,
                    reserved_job_id=reserved_job_id,
                    project_id=project_id,
                ))
            await self._submit_created_job(
                training_job_id,
                delete_created_child=created_child,
            )
        else:
            assert reserved_job_id is not None
            if existing_training_job.project_id != project_id:
                raise ValueError(
                    f"Training job id={reserved_job_id} belongs to a "
                    "different project.")
            training_job_id = reserved_job_id
            gen_train_table_job_id = await (
                self._validate_idempotent_training_replay_with_service(
                    training_job_id,
                    existing_training_job,
                    pquery_def_fpid=pquery_def_fpid,
                    graph_data_compute_input=graph_data_compute_input,
                    training_spec=training_spec,
                    graph_data_version=graph_data_version,
                    gen_train_table_job_id=gen_train_table_job_id,
                    warm_start_job_id=warm_start_job_id,
                    project_id=project_id,
                ))
            log.info(
                'Reusing existing training job %s for idempotent '
                'start_training replay.', training_job_id)

        home_dir = await self.get_job_home_dir(training_job_id)
        await HomeDirManager(home_dir).touch_init_file.coroutine()
        ml_job_settings = ml_job_settings or self._ml_job_settings

        if isinstance(advanced_options, ModelPlan):
            gen_train_table_job = (await service.get_generate_train_table_job(
                gen_train_table_job_id, project_ids={project_id}))
            assert gen_train_table_job is not None
            advanced_options.training_table_generation = (
                gen_train_table_job.config.plan)
            to_predictive_query_kwargs(advanced_options).apply_overrides(
                training_spec.predictive_query)
        training_job_input = TrainingJobInput(
            id=training_job_id,
            graph_data_compute_input=graph_data_compute_input,
            spec=training_spec,
            start_time=now_utc(),
            graph_version_info=graph_version_info,
            generate_train_table_job_id=gen_train_table_job_id,
            warm_start_job_id=warm_start_job_id,
        )

        if ml_job_settings.use_fake_job_backend:
            if existing_training_job is None:
                await self._persist_job_fake_backend(training_job_id,
                                                     training_job_input,
                                                     ml_job_settings)
            return training_job_id

        client = await self._temporal_client.get()

        wf_id = _training_workflow_id(training_job_id)
        try:
            await client.start_workflow(
                TrainingWorkflow.run,
                args=[
                    training_job_input,
                    ml_job_settings,
                ],
                id=wf_id,
                task_queue=ML_WORKFLOW_TASK_QUEUE,
            )
        except WorkflowAlreadyStartedError:
            log.info(
                'Training workflow %s for job %s already exists; '
                'treating start_training as an idempotent replay.', wf_id,
                training_job_id)

        return training_job_id

    async def reserve_training_job_id(self) -> TrainingJobID:
        """
        Reserve a TrainingJobID without creating the full job entry.

        The actual training job entry will be created later via start_training()
        which will use the reserved ID.

        Returns:
            TrainingJobID: A new, unique TrainingJobID
        """
        return await self._require_ml_job_service().reserve_training_job_id()

    async def get_training_job(
        self,
        ctx: RequestContext,
        training_job_id: TrainingJobID,
        strip_expensive_fields: bool = False,
    ) -> Optional[TrainingJobResponse]:
        results = await self._get_training_jobs(
            ctx,
            {training_job_id},
            strip_expensive_fields=strip_expensive_fields,
        )
        return results[training_job_id]

    async def get_training_jobs(
        self,
        ctx: RequestContext,
        training_job_ids: Set[TrainingJobID],
        job_status: Optional[JobStatus] = None,
        limit: Optional[int] = None,
        project_id: Optional[ProjectID] = None,
    ) -> Dict[TrainingJobID, TrainingJobResponse]:
        _current_user_ctx.set(ctx.user_authorization)
        jobs = await self._get_training_jobs(
            ctx,
            training_job_ids,
            strip_expensive_fields=True,
            project_id=project_id,
        )
        # Filter null
        jobs = {k: v for k, v in jobs.items() if v}
        if job_status:  # Optionally, filter by job status
            jobs = {
                id: job
                for id, job in jobs.items()
                if job.execution.status == job_status
            }
        # Use submission time for queued jobs that have not started execution.
        jobs = {
            job.id: job
            for job in sorted(
                [job for job in jobs.values() if job],
                key=lambda j: j.execution.start_time or j.created_at,
            )
        }
        if limit:  # keep most recent N=limit jobs
            jobs = dict(list(jobs.items())[-limit:])
        return jobs

    async def _get_training_jobs(
        self,
        ctx: RequestContext,
        training_job_ids: Set[TrainingJobID],
        *,
        strip_expensive_fields: bool,
        project_id: Optional[ProjectID] = None,
    ) -> Dict[TrainingJobID, Optional[TrainingJobResponse]]:
        """Fetch multiple training jobs from DB for specified job ids.

        Args:
            ctx (RequestContext): Request context propagated from REST API.
            training_job_ids (Set[TrainingJobID]): Set of training job IDs to
            lookup; if missing, value entry would be None in returned dict.

            strip_expensive_fields (bool): Whether to strip large expensive
            fields in training job result, such as test_eval (binary
            classification confusion matrix) and test_explain (legacy column
            XAI). See `TrainingJobOutput.strip_out_expensive_fields()`
        """
        _current_user_ctx.set(ctx.user_authorization)
        service = self._require_ml_job_service()
        project_ids = ({project_id} if project_id is not None else
                       visible_project_ids_or_none(ctx.user_authorization))
        jobs_by_id = {
            job.id: job
            for job in await service.list_training_job_rows(
                job_ids=training_job_ids,
                project_ids=project_ids,
            )
        }
        results = [
            self._training_job_response_from_entry(ctx, id, jobs_by_id.get(id))
            for id in training_job_ids
        ]
        if strip_expensive_fields:
            for result in results:
                if result is not None:
                    result.execution.strip_out_expensive_fields()
        return dict(zip(training_job_ids, results))

    async def get_train_table_result(
        self,
        job_id: GenerateTrainTableJobID,
    ) -> GenerateTrainTableResult:
        service = self._require_ml_job_service()
        job = await service.get_generate_train_table_job(
            job_id, project_ids=None)
        if job is None:
            raise KeyError(f'Job with id {job_id} not found in database')
        return self._generate_train_table_result_from_job(job_id, job)

    @staticmethod
    def _generate_train_table_result_from_job(
        job_id: GenerateTrainTableJobID,
        job: GenerateTrainTableJobDBEntry,
    ) -> GenerateTrainTableResult:
        if job.execution.status != JobStatus.DONE:
            raise ValueError(
                f"Unable to get GenerateTrainTableJob result because the "
                f"job status is: {job.execution.status}")
        validation = job.execution.validation or ValidationResponse()
        if not validation.ok:
            raise ValueError(
                f"GenerateTrainTableJob id={job_id} has failed validation, "
                f"details: {validation}")
        assert job.execution.train_table_path is not None
        assert job.execution.train_table_materialization_path is not None
        return GenerateTrainTableResult(
            train_table_path=job.execution.train_table_path,
            train_table_materialization_path=job.execution.
            train_table_materialization_path,
            validation=validation,
            train_table_home_dir=job.execution.train_table_home_dir,
        )

    async def get_pred_table_result(
        self,
        job_id: GeneratePredictionTableJobID,
    ) -> GeneratePredTableResult:
        service = self._require_ml_job_service()
        job = await service.get_generate_pred_table_job(job_id)
        if job is None:
            raise KeyError(f'Job with id {job_id} not found in database')
        return self._generate_pred_table_result_from_job(job_id, job)

    @staticmethod
    def _generate_pred_table_result_from_job(
        job_id: GeneratePredictionTableJobID,
        job: GeneratePredTableJobDBEntry,
    ) -> GeneratePredTableResult:
        if job.execution.status != JobStatus.DONE:
            raise ValueError(
                f"Unable to get GeneratePredictionTableJob result because the "
                f"job status is: {job.execution.status}")
        validation = job.execution.validation or ValidationResponse()
        if not validation.ok:
            raise ValueError(
                f"GeneratePredictionTableJob id={job_id} failed validation, "
                f"details: {validation}")
        assert job.execution.pred_table_path
        return GeneratePredTableResult(job.execution.pred_table_path,
                                       validation)

    async def get_generate_pred_table_jobs(
        self,
        job_ids: Set[GeneratePredictionTableJobID],
        *,
        project_ids: set[ProjectID] | None,
        status: Optional[JobStatus] = None,
        # Chronologically ordered by job start(create) timestamp, limit is
        # applied to return most recent jobs.
        limit: Optional[int] = None,
    ) -> List[api.GeneratePredictionTableJobResource]:
        service = self._require_ml_job_service()
        job_entries = await service.get_generate_pred_table_jobs(
            job_ids, project_ids=project_ids)
        # Filter by status, if specified.
        if status:
            job_entries = [
                job for job in job_entries if job.execution.status == status
            ]
        # Sort job entries
        job_entries = sorted(job_entries, key=lambda entry: entry.timestamps)
        # Apply limit to keep the most recent N entries
        if limit and len(job_entries) > limit:
            job_entries = job_entries[-limit:]

        # Convert job DB entry to public-facing resource object.
        def _entry_to_resource(
            entry: GeneratePredTableJobDBEntry,
        ) -> api.GeneratePredictionTableJobResource:
            graph_id = to_graph_id(entry.config.graph_id)
            if isinstance(graph_id,
                          GraphIDv2) and entry.config.graph_data_version:
                _, graph_fpid = GraphIDv2.parse(graph_id)
                assert graph_fpid
                graph_snapshot_id = DataSnapshot.GRAPH.format_id(
                    graph_fpid, entry.config.graph_data_version)
            else:
                graph_snapshot_id = None
            if entry.config.pquery_def_fpid:
                pquery_id = PredictiveQueryID.public(
                    entry.config.pquery_def_fpid)
            else:
                pquery_id = entry.config.pquery_id
            tracking_url = (f'{get_base_kumo_url()}'
                            f'/predictions/predictiontable/{entry.id}')
            return api.GeneratePredictionTableJobResource(
                job_id=entry.id,
                config=api.GeneratePredictionTableRequest(
                    {},
                    pquery_id=pquery_id,
                    plan=entry.config.plan,
                    graph_snapshot_id=graph_snapshot_id,
                ),
                job_status_report=api.JobStatusReport(
                    status=api.JobStatus(entry.execution.status),
                    tracking_url=tracking_url,
                    start_time=(entry.execution.start_time
                                or entry.timestamps.created_at),
                    end_time=(entry.execution.last_updated_at
                              if entry.execution.status.is_terminal else None),
                    event_log=[entry.execution.log],
                ),
                created_at=entry.timestamps.created_at,
                source=(json.from_json(
                    json.to_json(entry.source or JobSource.UNKNOWN),
                    common_api.JobSource,
                )),
                tags={},  # to be filled by REST
                user=entry.user,
                project_id=entry.project_id,
            )

        return [_entry_to_resource(job) for job in job_entries]

    async def get_generate_train_table_jobs(
        self,
        job_ids: Set[GenerateTrainTableJobID],
        *,
        project_ids: set[ProjectID] | None,
        status: Optional[JobStatus] = None,
        # Chronologically ordered by job start(create) timestamp, limit is
        # applied to return most recent jobs.
        limit: Optional[int] = None,
    ) -> List[api.GenerateTrainTableJobResource]:
        service = self._require_ml_job_service()
        job_entries = await service.get_generate_train_table_jobs(
            job_ids, project_ids=project_ids)
        # Filter by status, if specified.
        if status:
            job_entries = [
                job for job in job_entries if job.execution.status == status
            ]
        # Sort job entries
        job_entries = sorted(job_entries, key=lambda entry: entry.timestamps)
        # Apply limit to keep the last N entries
        if limit and len(job_entries) > limit:
            job_entries = job_entries[-limit:]

        # Convert job DB entry to public-facing resource object.
        def _entry_to_resource(
            entry: GenerateTrainTableJobDBEntry,
        ) -> api.GenerateTrainTableJobResource:
            graph_snapshot_id = None
            graph_id = to_graph_id(entry.config.graph_id)
            graph_ver = entry.config.graph_data_version
            if isinstance(graph_id, GraphIDv2) and graph_ver:
                _, fpid = GraphIDv2.parse(graph_id)
                assert fpid
                graph_snapshot_id = DataSnapshot.GRAPH.format_id(
                    fpid, graph_ver)
            if entry.config.pquery_def_fpid:
                pquery_id = PredictiveQueryID.public(
                    entry.config.pquery_def_fpid)
            else:
                pquery_id = entry.config.pquery_id
            tracking_url = (f'{get_base_kumo_url()}'
                            f'/models/trainingtable/{entry.id}')
            return api.GenerateTrainTableJobResource(
                job_id=entry.id,
                config=api.GenerateTrainTableRequest(
                    {},
                    pquery_id=pquery_id,
                    graph_snapshot_id=graph_snapshot_id,
                    plan=json.from_json(json.to_json(entry.config.plan),
                                        api.TrainingTableGenerationPlan),
                    custom_train_table=entry.config.custom_train_table,
                ),
                # Reported separately from the spec so the SDK can tell a job
                # that had no custom table from one whose record cannot say
                # what it had -- both of which read as `None` above.
                custom_train_table_used=entry.custom_train_table_used,
                job_status_report=api.JobStatusReport(
                    status=api.JobStatus(entry.execution.status),
                    tracking_url=tracking_url,
                    start_time=(entry.execution.start_time
                                or entry.timestamps.created_at),
                    end_time=(entry.execution.last_updated_at
                              if entry.execution.status.is_terminal else None),
                    event_log=[entry.execution.log],
                    validation_response=entry.execution.validation,
                ),
                created_at=entry.timestamps.created_at,
                source=(json.from_json(
                    json.to_json(entry.source or JobSource.UNKNOWN),
                    common_api.JobSource,
                )),
                tags={},  # to be filled by REST
                user=entry.user,
                project_id=entry.project_id,
            )

        return [_entry_to_resource(job) for job in job_entries]

    async def get_training_gen_tt_job(
        self,
        training_job_id: TrainingJobID,
    ) -> Optional[GenerateTrainTableJobDBEntry]:
        """Return the generate-train-table job entry for a training job.

        Performs a single DB lookup to fetch the training job, then a second
        lookup for the associated gen-TT job entry (if it exists). This
        replaces the previous two-step pattern of fetching the ID and then
        calling ``get_generate_train_table_jobs``.

        Raises:
            ValidationException: If the gen-TT job ID exists on the training
                job but the job entry is missing from the database (data
                corruption).
        """
        from kumo.validation import ValidationException

        service = self._require_ml_job_service()
        entry = await service.get_training_job(training_job_id)
        if entry is None or entry.generate_train_table_job_id is None:
            return None
        gen_tt_id = GenerateTrainTableJobID(entry.generate_train_table_job_id)
        gen_tt_entry = await service.get_generate_train_table_job(
            gen_tt_id, project_ids={ProjectID(entry.project_id)})
        if gen_tt_entry is None:
            raise ValidationException(
                title="Internal Data Error",
                message=(f"Training table generation job {gen_tt_id} "
                         f"is missing from database but was referenced by "
                         f"training job {training_job_id}. "
                         f"Please contact Kumo support."),
            )
        return gen_tt_entry

    async def get_training_job_training_table_plan(
        self,
        training_job_id: TrainingJobID,
    ) -> Optional[TrainingTableGenerationPlan]:
        """Return the training table generation plan for a training job.

        This provides access to forecasting params when a generate-train-table
        job ID is unavailable (e.g., SDK path using pre-generated train tables).
        """
        return await (self._require_ml_job_service().
                      get_training_job_training_table_plan(training_job_id))

    async def _get_generate_table_job_entry(
        self,
        job_id: Union[GenerateTrainTableJobID, GeneratePredictionTableJobID],
    ) -> Optional[Union[GenerateTrainTableJobDBEntry,
                        GeneratePredTableJobDBEntry]]:
        service = self._require_ml_job_service()
        if isinstance(job_id, GenerateTrainTableJobID):
            return await service.get_generate_train_table_job(
                job_id, project_ids=None)
        return await service.get_generate_pred_table_job(job_id)

    async def get_generate_train_table_job_path(
        self,
        job_id: GenerateTrainTableJobID,
    ):
        """
        Get the S3 path where a training table from a training
        table job is stored.

        It returns None, if the path doesn't exist.

        ** This method will not support custom training tables and
        is currently not supported for snowflake users.

        Args:
            job_id(GenerateTrainTableJobID): The job ID associated
            with the training table generation job.
        Response:
            If there is a valid S3 path for the training table, return
            the path. Otherwise raise a validation error.
        """
        job_entry = await self._get_generate_table_job_entry(job_id)
        if job_entry is None:
            raise ValueError(f'Generate train table entry for {job_id}' +
                             ' not found.')
        if not job_entry.execution.train_table_path:
            raise ValueError(f'Generate train table path for {job_id}' +
                             ' not found.')
        return job_entry.execution.train_table_path

    async def get_generate_table_job_execution(
        self,
        job_id: Union[GenerateTrainTableJobID, GeneratePredictionTableJobID],
    ) -> Optional[Union[GenerateTrainTableJobOutput,
                        GeneratePredTableJobOutput]]:
        """
        Get the execution details for a GenerateTrainTable or
        GeneratePrediction job.

        This provides access to validation, failure_info, and log fields
        for error reporting.

        Args:
            job_id: The job ID (GenerateTrainTableJobID or
                GeneratePredictionTableJobID)

        Returns:
            The job execution output containing validation, failure_info,
            and log, or None if the job doesn't exist.
        """
        job_entry = await self._get_generate_table_job_entry(job_id)
        return None if job_entry is None else job_entry.execution

    async def override_model(self, training_job_id: TrainingJobID,
                             overrides: TrainDirOverrides) -> None:
        home_dir = await self.get_job_home_dir(training_job_id)

        if not fsutil.exists(home_dir):
            raise RuntimeError(
                f"Job dir '{home_dir}' does not exist. Please check if the "
                f"training job ID is correct.")

        overrides.validate()

        output_path = HomeDirManager(home_dir).train_dir_overrides_file
        await json.save_json.coroutine(overrides, output_path)

        service = self._require_ml_job_service()
        job = await service.get_training_job(training_job_id)
        if job is None:
            raise RuntimeError(
                f"Job '{training_job_id}' does not exist in StateDB")
        job.execution.results = TrainingJobResults(test_explain=TestExplanation(
            contribution_summary=ColumnContributionSummary(
                global_column_contributions=[])))
        job.execution.status = JobStatus.DONE
        await service.modify_job_if_open_generic(job)

    async def get_job_home_dir(
        self,
        id: Union[
            TrainingJobID,
            BatchPredictionJobID,
            GenerateTrainTableJobID,
            GeneratePredictionTableJobID,
        ],
    ) -> str:
        return get_job_dir(get_job_base_dir(), id)

    async def get_job_base_dir(self) -> str:
        return get_job_base_dir()

    # =============== Batch Prediction =============== #

    async def start_generate_pred_table(
        self,
        job_input: BatchPredictionJobInput,
        *,
        project_id: ProjectID,
        graph_version_info: Optional[GraphDataVersionInfo] = None,
        job_source: Optional[JobSource] = None,
        user: Optional[str] = None,
        is_child_job: Optional[bool] = False,
    ) -> GeneratePredictionTableJobID:
        """
        Launch a stand-alone, stateful job to generate prediction table.

        The workflow always kicks off a graph refresh first: a specified
        graph_version_info pins which snapshot to read, but only the refresh
        guarantees it was built (kumo-ai/kumo#30363).
        """
        # For a standalone generete-pred-table job, the primary job id should
        # be the same as config.generate_pred_table_job_id
        assert job_input.id == job_input.generate_pred_table_job_id
        assert job_input.config.pquery_def_fpid
        self._require_connector_references_in_project(
            job_input.graph_data_compute_input.connectors.values(), project_id)
        self._require_connector_references_in_project(
            job_input.output_connectors.values(), project_id)
        if job_input.custom_pred_table_connector is not None:
            self._require_connector_references_in_project(
                [job_input.custom_pred_table_connector], project_id)
        ml_job_settings = self._ml_job_settings
        graph_data_version = (None if not graph_version_info else
                              graph_version_info.graph_data_version)
        expected_config = GeneratePredTableJobConfig(
            job_input.predictive_query_id,
            job_input.config.pred_table_generation_plan,
            job_input.graph_id,
            graph_data_version,
            job_input.config.pquery_def_fpid,
        )
        assert job_input.generate_pred_table_job_id
        requested_job_id = GeneratePredictionTableJobID(job_input.id)

        service = self._require_ml_job_service()
        existing_job = await service.get_generate_pred_table_job(
            requested_job_id)
        if existing_job is None:
            job_id = await service.create_generate_pred_table_job(
                config=expected_config,
                id=requested_job_id,
                now=now_utc(),
                source=job_source,
                user=user,
                project_id=project_id,
            )
            await self._submit_created_job(job_id, delete_created_child=False)
        else:
            if existing_job.project_id != project_id:
                raise ValueError(
                    f'Job id={requested_job_id} belongs to a different '
                    'project.')
            if existing_job.config != expected_config:
                raise ValueError(
                    f'Job id={requested_job_id} already exists with '
                    'different configuration.')
            job_id = requested_job_id
            log.info(
                'Reusing existing generate-pred-table job %s for '
                'idempotent start_generate_pred_table replay.', job_id)
        assert job_id == job_input.generate_pred_table_job_id
        assert isinstance(job_id, GeneratePredictionTableJobID)

        # Use the same gen-pred-table job ID as ID of job input.
        job_input.id = BatchPredictionJobID(job_id)

        if ml_job_settings.use_fake_job_backend:
            if existing_job is None:
                await self._persist_job_fake_backend(job_id, job_input,
                                                     ml_job_settings)
            return job_id

        home_dir = get_job_dir(
            get_job_base_dir(),
            job_id,
        )

        client = await self._temporal_client.get()
        try:
            await client.start_workflow(
                GeneratePredictionTableWorkflow.run,
                args=[
                    job_input,
                    ml_job_settings,
                    True,  # run gen-pred-table wf in standalone mode
                    home_dir,
                    is_child_job,
                ],
                id=job_id.wf_id,
                task_queue=ML_WORKFLOW_TASK_QUEUE,
            )
        except WorkflowAlreadyStartedError:
            log.info(
                'Generate-pred-table workflow %s for job %s already '
                'exists; treating start_generate_pred_table as an '
                'idempotent replay.', job_id.wf_id, job_id)

        return job_id

    async def start_batch_prediction(
        self,
        job_input: BatchPredictionJobInput,
        *,
        project_id: ProjectID,
        job_source: Optional[JobSource] = None,
        user: Optional[str] = None,
        description: str = "",
        batch_prediction_workflow_id: Optional[
            BatchPredictionWorkflowID] = None,
    ):
        # `job_input.graph_data_compute_input` is resolved at launch time and
        # threaded through to BatchPredictionParentWorkflow -> materialization,
        # so gen-pred-table and materialize agree on the diskgraph graph key.
        job_id = job_input.id
        self._require_connector_references_in_project(
            job_input.graph_data_compute_input.connectors.values(), project_id)
        self._require_connector_references_in_project(
            job_input.output_connectors.values(), project_id)
        ml_job_settings = deepcopy(self._ml_job_settings)

        # Check "OneOf" precondition.
        assert bool(job_input.config.custom_pred_table_source) ^ bool(
            job_input.config.generate_pred_table_job_id)
        if job_input.config.custom_pred_table_source is not None:
            reference = job_input.custom_pred_table_connector
            if (reference is None or str(reference.connector_id)
                    != job_input.config.custom_pred_table_source.connector_id):
                raise ValueError(
                    'Custom prediction source must match its connector '
                    'execution reference')
        else:
            assert job_input.generate_pred_table_job_id

        service = self._require_ml_job_service()
        assert not await service.get_batch_prediction_jobs([job_id],
                                                           project_ids=None)
        _, created_child = await service.create_batch_prediction_job(
            id=job_id,
            now=self._now_fn(),
            config=job_input.config,
            source=job_source,
            project_id=project_id,
            user=user,
            description=description,
            batch_prediction_workflow_id=batch_prediction_workflow_id,
        )
        await self._submit_created_job(
            job_id,
            delete_created_child=created_child,
        )
        if ml_job_settings.use_fake_job_backend:
            await self._persist_job_fake_backend(job_id, job_input,
                                                 ml_job_settings)
            return

        client = await self._temporal_client.get()
        # TODO(siyang): Consider specifying run timeout and
        # retry policy
        await client.start_workflow(
            BatchPredictionParentWorkflow.run,
            args=[
                job_input,
                ml_job_settings,
            ],
            id=_batch_prediction_workflow_id(job_id),
            task_queue=ML_WORKFLOW_TASK_QUEUE,
        )

    async def get_batch_prediction_job(
        self, ctx: RequestContext, batch_prediction_job_id: BatchPredictionJobID
    ) -> BatchPredictionJobOutput:
        _current_user_ctx.set(ctx.user_authorization)
        service = self._require_ml_job_service()
        job_entries = await service.get_batch_prediction_jobs(
            [batch_prediction_job_id],
            project_ids=visible_project_ids_or_none(ctx.user_authorization),
        )
        return self._batch_prediction_job_output_from_entry(
            ctx,
            batch_prediction_job_id,
            job_entries[0] if job_entries else None,
        )

    async def _has_batch_prediction_job(
        self,
        ctx: RequestContext,
        batch_prediction_job_id: BatchPredictionJobID,
    ) -> bool:
        _current_user_ctx.set(ctx.user_authorization)
        service = self._require_ml_job_service()
        job_entries = await service.get_batch_prediction_jobs(
            [batch_prediction_job_id],
            project_ids=visible_project_ids_or_none(ctx.user_authorization),
        )
        if not job_entries:
            return False
        self._state_service_job_visible_or_raise(job_entries[0].project_id)
        return True

    async def get_batch_prediction_jobs(
        self,
        ctx: RequestContext,
        ids: Set[BatchPredictionJobID],
        job_status: Optional[JobStatus] = None,
        limit: Optional[int] = None,
    ) -> Dict[BatchPredictionJobID, BatchPredictionJobOutput]:
        _current_user_ctx.set(ctx.user_authorization)
        service = self._require_ml_job_service()
        job_entries_by_id = {
            job.id: job
            for job in await service.get_batch_prediction_jobs(
                ids,
                project_ids=visible_project_ids_or_none(ctx.user_authorization),
            )
        }
        results = {
            id:
            self._batch_prediction_job_output_from_entry(
                ctx, id, job_entries_by_id.get(id))
            for id in ids
        }
        # Filter by job_status, if specified.
        if job_status:
            results = {
                k: v
                for k, v in results.items() if v.status == job_status
            }
        # Sort by job start_time (chronologically) and keep most recent N jobs
        sorted_jobs = sorted(results.items(), key=lambda e: e[1].start_time)
        if limit:
            sorted_jobs = sorted_jobs[-limit:]
        return dict(sorted_jobs)

    async def list_batch_prediction_rows_by_workflow_id(
        self,
        ctx: RequestContext,
        workflow_id: BatchPredictionWorkflowID | str | None,
    ) -> List[BatchPredictionJobDBEntry]:
        _current_user_ctx.set(ctx.user_authorization)
        return await (self._require_ml_job_service(
        ).list_batch_prediction_rows_by_workflow_id(
            workflow_id,
            project_ids=visible_project_ids_or_none(ctx.user_authorization),
        ))

    async def get_running_batch_prediction_job_ids(
        self,
        *,
        project_ids: set[ProjectID] | None,
    ) -> List[BatchPredictionJobID]:
        return await (
            self._require_ml_job_service().get_running_batch_prediction_job_ids(
                project_ids=project_ids))

    async def delete_batch_prediction(self, ctx: RequestContext,
                                      id: BatchPredictionJobID,
                                      force: bool = False) -> bool:
        _current_user_ctx.set(ctx.user_authorization)
        service = self._require_ml_job_service()
        job_entries = await service.get_batch_prediction_jobs(
            [id],
            project_ids=visible_project_ids_or_none(ctx.user_authorization),
        )
        job_entry = job_entries[0] if job_entries else None
        if job_entry is None:
            return False
        job = self._batch_prediction_job_output_from_entry(
            ctx,
            id,
            job_entry,
        )
        if job.status.is_submitted and not force:
            raise ValueError(f'Unable to delete a queued/running job {id}')
        home_dir = await self.get_job_home_dir(id)
        entry_deleted = await service.delete_job_generic(id, force=force)
        if entry_deleted:
            # Release job resource slot by removing it from job queue manager
            await self._job_queue.release_job(id)
            await self._delete_internal_job_home_best_effort(id, home_dir)
        return entry_deleted

    async def delete_many_batch_prediction_jobs(
        self,
        ctx: RequestContext,
        ids: Set[BatchPredictionJobID],
        force: bool = False,
    ) -> Set[BatchPredictionJobID]:
        r"""Delete multiple batch prediction jobs,
        which are in a terminal state, by ID.
        if `force` is set to True, then the deletion will proceed even if the
        job is in a non-terminal state

        Returns ::
           Set[BatchPredictionJobID]: the set of job IDs that were deleted.
        """
        _current_user_ctx.set(ctx.user_authorization)
        service = self._require_ml_job_service()
        job_entries_by_id = {
            job.id: job
            for job in await service.get_batch_prediction_jobs(
                ids,
                project_ids=visible_project_ids_or_none(ctx.user_authorization),
            )
        }
        deletable_ids = set()
        for id in ids:
            job = self._batch_prediction_job_output_from_entry(
                ctx, id, job_entries_by_id.get(id))
            if job.status.is_submitted and not force:
                continue
            deletable_ids.add(id)
        home_dirs = {
            id: await self.get_job_home_dir(id)
            for id in deletable_ids
        }
        successfully_deleted = await service.delete_batch_prediction_jobs(
            deletable_ids,
            force=force,
        )
        for id in successfully_deleted:
            # Release job resource slot by removing it from queue manager
            await self._job_queue.release_job(id)
        await asyncio.gather(
            *(self._delete_internal_job_home_best_effort(id, home_dirs[id])
              for id in successfully_deleted))
        return successfully_deleted

    async def delete_training_job(self, ctx: RequestContext, id: TrainingJobID,
                                  force: bool = False) -> bool:
        _current_user_ctx.set(ctx.user_authorization)
        service = self._require_ml_job_service()
        rows = await service.list_training_job_rows(
            job_ids={id},
            project_ids=visible_project_ids_or_none(ctx.user_authorization),
            limit=1,
        )
        job_entry = rows[0] if rows else None
        job = self._training_job_response_from_entry(ctx, id, job_entry)
        if job is None:
            return False
        if job.execution.status.is_submitted and not force:
            raise ValueError(f'Unable to delete a queued/running job {id}')
        home_dir = await self.get_job_home_dir(id)
        entry_deleted = await service.delete_job_generic(id, force=force)
        if entry_deleted:
            # Release job resource slot by removing it from job queue manager
            await self._job_queue.release_job(id)
            await self._delete_internal_job_home_best_effort(id, home_dir)
        return entry_deleted

    async def _cancel_job_workflow(self, job_id: JobID, wf_id: str):
        await self._temporal_client.cancel(wf_id)

        # Due to https://github.com/temporalio/sdk-python/issues/620
        # it's possible for the workflow to silently swallow cancelaltion
        # request and continue to run. As a best-effort workaround, we sleep a
        # few seconds and check if workflow is still running, if so, we send a
        # cancellation signal directly to the workflow as best effort t
        await asyncio.sleep(2.0)
        wf_handle = await self._temporal_client.get_workflow_handle(wf_id)
        wf_desc = await wf_handle.describe()
        if wf_desc.status == WorkflowExecutionStatus.RUNNING:
            await wf_handle.signal(
                signal='cancel_wf',
                arg=f'Job {job_id} was requested to be canceled.')
            log.warning(
                "Workflow %s still running after cancellation was requested, "
                "send a direct cancel_wf signal now.",
                wf_id,
            )

    async def _get_and_cancel_parent_job(
        self,
        child_id: GenerateTrainTableJobID | GeneratePredictionTableJobID,
    ) -> bool:
        """Cancel the parent job if the given job is a child job.

        When a child job (GenerateTrainTable or GeneratePredTable) is cancelled,
        its parent job (Training or BatchPrediction) should also be
        cancelled to avoid the parent continuing to run while waiting for a
        child that will never complete.

        Returns:
            True if no parent job exists or parent was successfully cancelled.
        """
        if isinstance(child_id, GenerateTrainTableJobID):
            parent_ids = [
                TrainingJobID.as_parent_of(child_id),
            ]
        else:
            parent_ids = [
                BatchPredictionJobID.as_parent_of(child_id),
            ]

        for parent_id in parent_ids:
            parent_job = await self._get_job_if_exists_generic(parent_id)
            if parent_job is not None:
                log.info(
                    "Cancelling parent job %s because child job %s was cancelled",
                    parent_id,
                    child_id,
                )
                return await self._cancel_job_generic(parent_id)

        return True

    async def _cancel_job_generic(self, id: JobID) -> bool:
        job_type = JobType.from_id(id)
        log.info('Cancel %s %s in ML job backend', job_type, id)
        job = await self._get_job_if_exists_generic(id)
        if job is None:
            raise KeyError(f'Job with id {id} not found in database')
        if job.execution.workflow_end_state:  # noop, already terminal state.
            assert job.execution.status.is_terminal
            return job.execution.status == JobStatus.CANCELLED

        # Asynchronously request cancellation of the workflow.
        wf_id = _job_workflow_id(id)
        cancel_task = asyncio.create_task(self._cancel_job_workflow(id, wf_id))
        self._cancellation_tasks.add(cancel_task)
        cancel_task.add_done_callback(self._cancellation_tasks.discard)

        # Eagerly mark job as cancelled in the DB and freeze it from being
        # later modified by setting the `workflow_end_state`.
        service = self._require_ml_job_service()
        is_cancelled = await service.cancel_job_if_open_generic(id)
        if is_cancelled:
            await self._job_queue.release_job(id)
        return is_cancelled

    async def cancel_training_job(self, id: TrainingJobID) -> bool:
        return await self._cancel_job_generic(id)

    async def cancel_batch_prediction_job(self,
                                          id: BatchPredictionJobID) -> bool:
        return await self._cancel_job_generic(id)

    async def cancel_generate_train_table_job(
            self, id: GenerateTrainTableJobID) -> bool:
        is_cancelled = await self._cancel_job_generic(id)
        await self._get_and_cancel_parent_job(id)
        return is_cancelled

    async def cancel_generate_pred_table_job(
            self, id: GeneratePredictionTableJobID) -> bool:
        is_cancelled = await self._cancel_job_generic(id)
        await self._get_and_cancel_parent_job(id)
        return is_cancelled

    async def get_materialization_info(
        self, id: Union[TrainingJobID, BatchPredictionJobID]
    ) -> Optional[MaterializationTaskInfo]:
        return await asyncio.to_thread(get_materialization_info, id)

    async def get_many_materialization_info(
        self, id: Union[List[TrainingJobID], List[BatchPredictionJobID]]
    ) -> List[Optional[MaterializationTaskInfo]]:
        return await asyncio.to_thread(get_many_materialization_info, id)

    async def get_mlops_jobs_and_db_stats(
            self, query: MLOpsQueryConfig) -> Optional[MLOpsStats]:
        r"""Runs the db querying step of an MLOps query to (1) get the ml
        backend job db entries that satisfy the query
        (MLOpsQueryConfig.job_query) and (2) collect any stats in the stats
        component of the query (MLOpsQueryConfig.stat_query) that are
        available directly in the retrieved ml backend db entries. Not all
        requested stats will necessarily be returned by this method, since
        other stats are only in StateDB or the data backend cache.

        Args:
            query (MLOpsQueryConfig): Query specifying the jobs to retrieve
                as well as what stats are requested for those jobs

        Returns:
            MLOpsStats: mlops query resonse containing a dict of job ids and
                their statuses, as well as any retrieved stats for each job.
                Note that the keys of MLOpsStast.train_stats and
                MLOpsStats.bp_stats are later used to determine what BP and
                train jobs were retrieved, so they need to be populated.
        """
        ml_job_service = self._require_ml_job_service()
        job_models = await ml_job_service.select_jobs(
            query.job_query.job_filters)

        # return the response with the stats that are in the db (MLOpsLogic
        # will request remaining stats from elsewhere)
        job_status_dict = {job.job_id: job.status for job in job_models}
        bp_stats = {}
        train_stats = {}

        # filter for stat types that can be answered by the db here -
        # map stat types to attributions of JobModel class
        stats_to_db_col = {
            MLOpsStatType.TABLE_STATS:
            'input_data_stats',
            MLOpsStatType.START_TIME:
            'start_time',
            MLOpsStatType.END_TIME:
            'end_time',
            MLOpsStatType.PREDICTION_VERSION:
            'kumo_version',
            # this will be overridden for BP jobs:
            MLOpsStatType.TRAINING_VERSION:
            'kumo_version',
            # only set for BP versions:
            MLOpsStatType.TRAINED_JOB_ID:
            "trained_job_id",
        }

        db_stats_requested = [
            stat for stat in query.stat_query.stat_request
            if stat in stats_to_db_col
        ]
        train_jobs_for_bps = {}
        for job in job_models:
            if job.job_type == JobType.BATCH_PREDICTION_JOB:
                # put all bp jobs in bp_stats, even if no db stats added
                bp_stats[job.job_id] = {}
                train_jobs_for_bps[job.job_id] = job.trained_job_id
                for stat in db_stats_requested:
                    if stat != MLOpsStatType.TRAINING_VERSION:
                        bp_stats[job.job_id][stat] = getattr(
                            job, stats_to_db_col[stat])
            if job.job_type == JobType.TRAINING_JOB:
                # put all train jobs in train_stats, even if no db stats added
                train_stats[job.job_id] = {}
                for stat in db_stats_requested:
                    if stat != MLOpsStatType.PREDICTION_VERSION:
                        train_stats[job.job_id][stat] = getattr(
                            job, stats_to_db_col[stat])

        # if we need training versions for bp jobs, need to query db again
        # to get the corresponding training job entries. This gathers all
        # training jobs we need and then looks them up as one query to
        # reduce overhead with db usage
        if (MLOpsStatType.TRAINING_VERSION in query.stat_query.stat_request
                and len(train_jobs_for_bps.keys())) > 0:
            train_job_dict = await ml_job_service.get_multi_train_jobs(
                list(train_jobs_for_bps.values()))
            for bp_id in bp_stats.keys():
                train_id = train_jobs_for_bps[bp_id]
                if train_id in train_job_dict.keys():
                    bp_stats[bp_id][
                        MLOpsStatType.TRAINING_VERSION] = train_job_dict[
                            train_id].kumo_version
                else:
                    bp_stats[bp_id][MLOpsStatType.TRAINING_VERSION] = None

        return MLOpsStats(job_status_dict=job_status_dict,
                          train_stats=train_stats, bp_stats=bp_stats)

    async def get_all_ml_jobs(
        self,
        ctx: RequestContext,
        list_query: ListJobsQuery,
        tags: Dict[JobID, JobTags],
    ) -> Tuple[List[GenericJob], int]:
        _current_user_ctx.set(ctx.user_authorization)
        service = self._require_ml_job_service()
        if not self._state_service_job_visible(list_query.project_id):
            return [], 0
        return await service.search_jobs(
            ctx,
            list_query,
            tags,
            project_id=list_query.project_id,
        )

    # =============== Internal Implementation =============== #

    async def _get_generic_job_or_status(
        self,
        ids: Set[JobID],
        ctx: RequestContext,
        only_status_job_ids: Optional[Set[JobID]] = None,
        project_id: Optional[ProjectID] = None,
        prefetched_jobs: Optional[Dict[JobID, MLJobDBEntry]] = None,
    ) -> Dict[Union[TrainingJobID, BatchPredictionJobID,
                    GenerateTrainTableJobID, GeneratePredictionTableJobID],
              Union[JobStatus, GenericJob]]:
        async def _get_generic_or_status_from_db(
            job_id: JobID, jobs
        ) -> Union[
                TrainingJobDBEntry,
                BatchPredictionJobDBEntry,
                GenerateTrainTableJobDBEntry,
                GeneratePredTableJobDBEntry,
                None,
        ]:
            """
            Function to get a generic job from job id. If the generic job
            is already fetched in the UI, then we only return the
            job status. Otherwise, return the generic job.
            """
            job = (prefetched_jobs or {}).get(job_id)
            if job is None:
                job = await self._get_generic_job(job_id)
            # Filter by project if specified.
            if job and project_id is not None:
                if getattr(job, 'project_id', None) != project_id:
                    return None
            if job:
                if only_status_job_ids is not None and job_id in only_status_job_ids:
                    # if the job's metadata is already fetched in the UI,
                    # only return the job status
                    if job_id in bulk_status_ids and job.id in bulk_statuses:
                        jobs[job.id] = bulk_statuses[job.id]
                    else:
                        jobs[job.id] = job.execution.status

                else:
                    # otherwise, return the generic job
                    jobs[job.id] = job.to_generic(ctx)
                return job
            else:
                log.warning(f'Generic job for {id} is None and cannot' +
                            ' return anything because the job does not exist.')

        bulk_status_ids: Set[JobID] = set()
        bulk_statuses: Dict[JobID, JobStatus] = {}
        service = self._require_ml_job_service()
        if only_status_job_ids is not None and prefetched_jobs is not None:
            # Only prefetched jobs have already supplied the parent metadata
            # needed below for child discovery and project filtering.
            for job_id in ids:
                job = prefetched_jobs.get(job_id)
                if job is None or job_id not in only_status_job_ids:
                    continue
                if project_id is not None and getattr(job, 'project_id',
                                                      None) != project_id:
                    continue
                bulk_status_ids.add(job_id)

            if bulk_status_ids:
                bulk_statuses = await service.get_job_statuses_by_ids(
                    bulk_status_ids)

        jobs = {}
        for id in ids:
            job = await _get_generic_or_status_from_db(id, jobs)

            # make sure to fetch child training table jobs
            if isinstance(
                    job,
                    TrainingJobDBEntry) and job.generate_train_table_job_id:
                training_table_job_id = JobID(job.generate_train_table_job_id)
                if training_table_job_id not in jobs:
                    await _get_generic_or_status_from_db(
                        training_table_job_id, jobs)

            # make sure to fetch child prediction table jobs
            if isinstance(job, BatchPredictionJobDBEntry):
                if job.config and job.config.generate_pred_table_job_id:
                    pred_table_job_id = job.config.generate_pred_table_job_id
                    if pred_table_job_id not in ids:
                        await _get_generic_or_status_from_db(
                            pred_table_job_id, jobs)

        return jobs

    async def _get_generic_job(
        self, id: JobID
    ) -> Union[
            TrainingJobDBEntry,
            BatchPredictionJobDBEntry,
            GenerateTrainTableJobDBEntry,
            GeneratePredTableJobDBEntry,
            None,
    ]:
        return await self._require_ml_job_service().get_job_if_exists_generic(id
                                                                              )

    @blocking_method()
    def _delete_internal_job_home(self, home_dir: str) -> None:
        """Delete only Kumo's internal home for one terminal job."""
        if fsutil.exists(home_dir):
            fsutil.rm(home_dir)

    async def _delete_internal_job_home_best_effort(
        self,
        job_id: JobID,
        home_dir: str,
    ) -> None:
        """Clean storage without reversing an already-durable job deletion."""
        try:
            await self._delete_internal_job_home.coroutine(home_dir)
        except Exception as error:
            log.error(
                'Internal home cleanup failed for durably deleted job %s at %s '
                '(error_type=%s)',
                job_id,
                home_dir,
                type(error).__name__,
            )

    def _training_job_response_from_entry(
        self,
        ctx: RequestContext,
        id: TrainingJobID,
        job: Optional[TrainingJobDBEntry],
    ) -> Optional[TrainingJobResponse]:
        if job is None:
            return None
        if not self._state_service_job_visible_or_raise(job.project_id):
            return None

        job.execution.remove_employee_only_fields()

        if self._settings.ml_job_settings.use_fake_job_backend:
            execution = self._make_fake_training_job(job).execution
            if isinstance(job.advanced_options,
                          DeprecatedAdvancedAutoTrainerOptions):
                advanced_options = None
            else:
                advanced_options = job.advanced_options
            return TrainingJobResponse(
                id=id,
                execution=execution,
                project_id=job.project_id,
                created_at=job.timestamps.created_at,
                advanced_options=advanced_options,
                pquery_def_fpid=job.pquery_def_fpid,
                generate_train_table_job_id=job.generate_train_table_job_id,
                warm_start_job_id=job.warm_start_job_id,
                user=job.user,
            )

        if isinstance(job.advanced_options,
                      DeprecatedAdvancedAutoTrainerOptions):
            advanced_options = None
        else:
            advanced_options = job.advanced_options
        return TrainingJobResponse(
            id=id,
            execution=job.execution,
            project_id=job.project_id,
            created_at=job.timestamps.created_at,
            advanced_options=advanced_options,
            pquery_def_fpid=job.pquery_def_fpid,
            graph_data_version=job.graph_data_version,
            generate_train_table_job_id=job.generate_train_table_job_id,
            job_source=job.source,
            custom_train_table=job.custom_train_table_used,
            warm_start_job_id=job.warm_start_job_id,
            kumo_version=job.kumo_version,
            user=job.user,
        )

    def _make_fake_training_job(
        self,
        job: TrainingJobDBEntry,
    ) -> TrainingJobDBEntry:
        # Fake backend (unit tests): no worker advances status, so a
        # still-submitted job is presented as RUNNING/"Not Started". Terminal
        # state (DONE/FAILED) is written explicitly via
        # mock_training_job_complete and returned as-is. No timed simulation.
        if not job.execution.status.is_submitted:
            return job
        job.execution.status = JobStatus.RUNNING
        job.execution.log = JobEventLogEntry.not_started(self._now_fn())
        job.execution.progress = None
        job.execution.results = None
        return job

    def _batch_prediction_job_output_from_entry(
        self,
        ctx: RequestContext,
        id: BatchPredictionJobID,
        job: Optional[BatchPredictionJobDBEntry],
    ) -> BatchPredictionJobOutput:
        if not self._state_service_job_visible_or_raise(
                None if job is None else job.project_id):
            return BatchPredictionJobOutput.missing_from_ml_db(id)
        if job is not None:
            job.execution.remove_employee_only_fields()
        if job is None:
            return BatchPredictionJobOutput.missing_from_ml_db(id)
        # simulate a batch prediction job that starts after 10 seconds, and
        # finishes after 2 minutes
        if self._settings.ml_job_settings.use_fake_job_backend:
            return self._make_fake_batch_prediction_job(job)

        return job.execution

    def _state_service_job_visible_or_raise(
        self,
        project_id: Optional[ProjectID],
    ) -> bool:
        if self._state_service_job_visible(project_id):
            return True
        raise ValueError("Project access denied")

    def _state_service_job_visible(
        self,
        project_id: Optional[ProjectID],
    ) -> bool:
        if project_id is None:
            return False
        user_auth = _current_user_ctx.get()
        if user_auth is None:
            return False
        if user_auth.is_admin or project_id in user_auth.project_ids:
            return True
        return False

    def _make_fake_batch_prediction_job(
            self, job: BatchPredictionJobDBEntry) -> BatchPredictionJobOutput:
        # Fake backend (unit tests): no worker advances status, so a
        # still-submitted job is presented as RUNNING/"Not Started". Terminal
        # state (DONE/FAILED) is written explicitly via mock_bp_job_complete and
        # returned as-is. No timed simulation.
        if not job.execution.status.is_submitted:
            return job.execution
        job.execution.status = JobStatus.RUNNING
        job.execution.log = JobEventLogEntry.not_started(self._now_fn())
        job.execution.progress = None
        job.execution.results = None
        return job.execution

    async def mock_bp_job_complete(self, id: BatchPredictionJobID) -> None:
        """For use in unit tests, to simulate the completion of a BP job."""
        assert self._settings.ml_job_settings.use_fake_job_backend
        output = BatchPredictionJobOutput(
            status=JobStatus.DONE,
            log=JobEventLogEntry.done(),
            progress=None,
            last_updated_at=self._now_fn(),
            results=BatchPredictionJobResults(
                num_rows=500,
                num_cols=45,
                size_bytes=3020394,
                key_cols=['user_id'],
                output_paths=[],
            ),
        )
        await self._require_ml_job_service().save_batch_prediction_job_output(
            id, output)

    async def mock_training_job_complete(self, id: TrainingJobID) -> None:
        """For use in unit tests, to mark a fake training job DONE."""
        assert self._settings.ml_job_settings.use_fake_job_backend
        output = TrainingJobOutput(
            status=JobStatus.DONE,
            log=JobEventLogEntry.done(self._now_fn()),
            progress=None,
            last_updated_at=self._now_fn(),
            results=TrainingJobResults(
                total_training_time_sec=MOCK_TRAINING_TIME_SEC,
                total_iterations=MOCK_TRAINING_TIME_SEC,
                test_explain=MOCK_CONTRIBUTION,
            ).with_metrics([
                Metric("test_rmse", 0.4665),
                Metric("test_mae", 0.643),
                Metric("test_rmsle", 0.73),
                Metric("test_mape", 0.2231),
            ]),
        )
        await self._require_ml_job_service().save_training_job_output(
            id, output)

    async def get_encoders_for_job(
            self, ctx: RequestContext,
            id: TrainingJobID) -> Optional[Dict[str, str]]:
        """
        Gets encoders used for a query by reading the task config and
        parsing the encoders from it

        Output: Dictionary mapping from Fully Qualified Table Name to Encoder
        object representation

        {
            'connector_id.table_id.column_name': 'Encoder(p1=2, p2=[41])'
        }

        The encoder is meant to be used in the Model Planner and can
        be parsed by ModelPlan.parse_encoder_overrides
        into an encoder object in code
        """
        _current_user_ctx.set(ctx.user_authorization)
        home_dir = await self.get_job_home_dir(id)
        home_dir_manager = HomeDirManager(home_dir)
        task_config_path = home_dir_manager.task_config_path()
        if task_config_path is None:
            return None
        task_config = await json.load_json_async(task_config_path, TaskConfig,
                                                 default=None,
                                                 raise_on_missing=False)
        if task_config is None:
            log.debug('Task config is not yet available')
            return None

        encoders: Dict[str, str] = {}
        for table in task_config.graph.tables:
            for col in table.cols:
                if col.encoder is None:
                    log.debug('Missing encoder')
                    raise ValueError(
                        'Error in getting encoders for query. '
                        f'Encoder is missing for column {col.name}. '
                        'Please contact Kumo for assistance.')
                encoders[f"{table.name}.{col.name}"] = (
                    construct_object_string_from_config(col.encoder))

        return encoders

    async def get_in_progress_job_or_job_status(
        self,
        ctx: RequestContext,
        only_status_job_ids: Optional[Set[JobID]] = None,
        project_id: Optional[ProjectID] = None,
    ) -> Dict[Union[TrainingJobID, BatchPredictionJobID,
                    GenerateTrainTableJobID, GeneratePredictionTableJobID],
              Union[JobStatus, GenericJob]]:
        """
        Function that 1. fetches all the job IDs for jobs that are currently
        either queued or running and 2. fetches the generic job or job status
        for each job ID.
        If the generic job is already retrieved in the UI, then we
        only return the job status. Otherwise, return the
        generic job.
        """
        _current_user_ctx.set(ctx.user_authorization)
        job_entries = {}

        # Get job ids for jobs that are currently running or queued.
        open_parent_jobs = await (
            self._require_ml_job_service().list_open_parent_jobs())
        prefetched_jobs = {job.id: job for job in open_parent_jobs}
        current_jobs = set(prefetched_jobs.keys())

        if len(current_jobs) != 0:
            # Get generic job or job status for each ID
            job_entries = await self._get_generic_job_or_status(
                current_jobs,
                ctx,
                only_status_job_ids,
                project_id=project_id,
                prefetched_jobs=prefetched_jobs,
            )
        return job_entries

    async def _persist_job_fake_backend(
        self,
        job_id: JobID,
        job_input: Union[
            TrainingJobInput,
            BatchPredictionJobInput,
        ],
        ml_job_settings: MLJobSettings,
    ):
        await create_job_registry_metadata_in_state_db(
            job_id,
            job_input,
            ml_job_settings,
            self._require_ml_job_service(),
        )
        await asyncio.get_running_loop().run_in_executor(
            None, persist_job_inputs, job_input, ml_job_settings)

    async def list_job_tags(
        self,
        job_type: JobType,
        tag_filters: Mapping[str, str],
    ) -> Dict[JobID, JobTags]:
        return await self._require_ml_job_service().list_job_tags(
            job_type, tag_filters)

    async def list_project_job_tags(
        self,
        job_type: JobType,
        project_id: ProjectID,
        tag_filters: Mapping[str, str],
    ) -> Dict[JobID, JobTags]:
        """Return tags only for jobs owned by the exact project."""
        service = self._require_ml_job_service()
        tags = await service.list_job_tags(job_type, tag_filters)
        jobs = await asyncio.gather(*(service.get_job_if_exists_generic(job_id)
                                      for job_id in tags))
        return {
            job_id: job_tags
            for (job_id, job_tags), job in zip(tags.items(), jobs)
            if job is not None and job.project_id == project_id
        }

    async def get_job_tags_if_exists(
        self,
        job_type: JobType,
        job_id: JobID,
    ) -> Optional[JobTags]:
        return await self._require_ml_job_service().get_job_tags(
            job_type, job_id)

    async def get_job_tags(
        self,
        job_type: JobType,
        job_id: JobID,
    ) -> JobTags:
        job_tags = await self.get_job_tags_if_exists(job_type, job_id)
        if job_tags is None:
            raise KeyError(f"{job_type} id={job_id} is not found in DB")
        return job_tags

    async def apply_job_tags(
        self,
        job_type: JobType,
        job_id: JobID,
        tags: Dict[JobTagKey, JobTagValue],
    ):
        return await self._require_ml_job_service().apply_job_tags(
            job_type, job_id, tags)

    async def update_job_tags(
        self,
        job_type: JobType,
        job_id: JobID,
        tags: Dict[JobTagKey, JobTagValue],
        *,
        project_id: ProjectID,
    ) -> bool:
        """Update tags for an existing job.

        Tag rows are created lazily, so validate against the generic job row
        instead of requiring a pre-existing tag row.
        """
        job = await self._get_job_if_exists_generic(job_id)
        if job is None or job.project_id != project_id:
            raise KeyError(f"{job_type} id={job_id} is not found in DB")
        return await self.apply_job_tags(job_type, job_id, tags)

    async def delete_job_tag(
        self,
        job_type: JobType,
        job_id: JobID,
        tag: JobTagKey,
        *,
        project_id: ProjectID,
    ) -> bool:
        job = await self._get_job_if_exists_generic(job_id)
        if job is None or job.project_id != project_id:
            raise KeyError(f"{job_type} id={job_id} is not found in DB")
        return await self._require_ml_job_service().delete_job_tag(
            job_type, job_id, tag)

    async def archive_job(
        self,
        job_id: JobID,
        archived: bool,
        project_id: ProjectID,
    ) -> bool:
        service = self._require_ml_job_service()
        if archived:
            return await service.archive_job(
                job_id,
                project_id=project_id,
            )
        return await service.unarchive_job(
            job_id,
            project_id=project_id,
        )

    async def get_job_validation_response(self,
                                          job_id: JobID) -> ValidationResponse:
        return await get_validation_response(self._ml_job_settings, job_id)


def _training_workflow_id(id: TrainingJobID) -> str:
    return f'training-wf-{id}'


def _batch_prediction_workflow_id(id: BatchPredictionJobID) -> str:
    return f'batch-prediction-wf-{id}'


def _job_workflow_id(job_id: JobID) -> str:
    job_type = JobType.from_id(job_id)
    if job_type == JobType.TRAINING_JOB:
        return _training_workflow_id(TrainingJobID(job_id))
    if job_type == JobType.BATCH_PREDICTION_JOB:
        return _batch_prediction_workflow_id(BatchPredictionJobID(job_id))
    if job_type == JobType.GENERATE_TRAIN_TABLE_JOB:
        return GenerateTrainTableJobID(job_id).wf_id
    if job_type == JobType.GENERATE_PREDICTION_TABLE_JOB:
        return GeneratePredictionTableJobID(job_id).wf_id
    raise AssertionError(f'Unsupported job type {job_type} for {job_id}')


def _job_cancelled_message(
        wf_error: WorkflowFailureError,
        wf_description: WorkflowExecutionDescription) -> Optional[str]:
    if wf_description.status == WorkflowExecutionStatus.TERMINATED:
        return f'Job has been terminated: {wf_error.cause}'

    if wf_description.status == WorkflowExecutionStatus.CANCELED or (
            wf_description.status == WorkflowExecutionStatus.FAILED
            and is_cancelled_exception(wf_error.cause)):
        return 'Job has been cancelled by user'

    return None


def _wf_failure_to_stacktrace(err: WFError) -> str:
    stacktrace = format_exception(err)
    if isinstance(err.cause, ChildWorkflowError):
        err = err.cause
        stacktrace = format_exception(err)
        if isinstance(err.cause, ActivityError):
            err = err.cause
            stacktrace = format_exception(err)
    return stacktrace


def _wf_failure_to_log_msg(id: JobID, err: WFError) -> str:
    job_type = JobType.from_id(id)
    log_msg = f'{job_type} {id} workflow failed, cause: {err.cause}\n'
    if isinstance(err.cause, ChildWorkflowError):
        err = err.cause
        log_msg += (f'Child workflow {err.workflow_id} {err.run_id} '
                    f'failed, cause: {err.cause}\n')
        if isinstance(err.cause, ActivityError):
            err = err.cause
            log_msg += (f"Activity {err.activity_type} in worker "
                        f'{err.identity} failed, cause: {err.cause}\n')
    return log_msg


def _job_failed_message(job_type: JobType) -> str:
    display = job_type.value.replace('_', ' ').capitalize()
    return f'{display} failed'


def _update_execution_upon_wf_error(
    job_type: JobType,
    execution: JobExecution,
    wf_error: WorkflowFailureError,
    wf_description: WorkflowExecutionDescription,
    ts_now: datetime,
):
    assert wf_description.close_time
    assert wf_description.status != WorkflowExecutionStatus.RUNNING

    # Check if workflow already set failure_info (using extract_failure_info())
    # If so, preserve it instead of overwriting with generic Temporal error
    workflow_set_failure_info = execution.failure_info is not None and (
        execution.failure_info.message or execution.failure_info.stacktrace)

    if execution.status.is_submitted:
        cancel_msg = _job_cancelled_message(wf_error, wf_description)
        if cancel_msg:
            execution.status = JobStatus.CANCELLED
            execution.log = JobEventLogEntry.cancelled(
                ts_now, f'{job_type} is cancelled: {cancel_msg}')
        else:
            execution.status = JobStatus.FAILED

            if not workflow_set_failure_info:
                # Try to extract PublicException from the Temporal error chain as a safety net
                failure_info = extract_failure_info(exception=wf_error)

                # For internal logging, include Temporal error details
                log_message = failure_info.message or (
                    f"{job_type} failed: {wf_failure_to_temporal_trace(wf_error)}"
                )

                # Never store an empty user-facing message on a FAILED
                # execution; mirror the TrainingJobOutput.failed() and
                # BatchPredictionJobOutput.failed() constructors' fallback.
                # Applied after log_message so the log keeps the detailed
                # Temporal trace.
                if not failure_info.message:
                    failure_info.message = _job_failed_message(job_type)

                execution.log = JobEventLogEntry.failed(ts_now, log_message)
                execution.failure_info = failure_info
            else:
                assert execution.failure_info is not None  # Type narrowing
                log_message = (execution.failure_info.message
                               or _job_failed_message(job_type))
                execution.log = JobEventLogEntry.failed(ts_now, log_message)

    execution.workflow_end_state = wf_description.status
    execution.last_updated_at = wf_description.close_time


async def get_validation_response(
    settings: MLJobSettings,
    id: JobID,
) -> ValidationResponse:
    r"""Returns a validation response for a training or batch prediction
    workflow, if present. Raises a `RuntimeError` if the validation response
    is not found."""
    home_dir = get_job_dir(get_job_base_dir(), id)
    path = HomeDirManager(home_dir).validation_path(id)
    return await json.load_json_async(path, ValidationResponse)
