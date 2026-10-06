# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import os.path as osp
from typing import Optional, Tuple, Union

from kumo_core.job_config import BestTrainerConfig, JobConfig
from kumo_core.training.config import TrainerConfig
from kumo_core.training.explain_aggregation import aggregate_subgraph_features
from kumo_core.version import get_kumo_version
from kumoapi.explain import LinkPredRFM, NodePredRFM
from kumoapi.train import TrainingStage
from temporalio import activity
from temporalio.exceptions import CancelledError

from io_utils import fsutil, logging
from io_utils.json import load_json, save_json, save_json_atomic
from io_utils.yaml import to_yaml
from kumo.config.explain import StrategyMappings
from kumo.config.task import TaskConfig
from kumo.data.cache.graph_cache import GraphVersion
from kumo.diskgraph import get_service
from kumo.home import HomeDirManager
from kumo.train.config_io import load_job_task
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.ml_backend.data_util import get_job_base_dir, get_job_dir
from rest.backends.ml_backend.job_models import (
    GeneratePredTableJobOutput,
    GenerateTrainTableJobOutput,
    MLJobRegistryMetadata,
)
from rest.backends.ml_backend.job_queue import StateDbJobQueue
from rest.backends.ml_backend.ml_job_output_store import MLJobOutputStore
from rest.backends.ml_backend.settings import MLJobQueueSettings, MLJobSettings
from rest.config.batch_prediction_job import (
    BatchPredictionJobInput,
    BatchPredictionJobOutput,
)
from rest.config.ml_backend import TrainingJobInput
from rest.config.training_job import TrainingJobOutput
from rest.utils.explain import (
    CATEGORY_RFM_DIR,
    MAX_NUM_POST_XAI_ENTITIES,
    link_pred_rfm_avg,
    node_pred_rfm_avg,
)
from statedb.config.common import (
    BatchPredictionJobID,
    GeneratePredictionTableJobID,
    GenerateTrainTableJobID,
    JobID,
    JobStatus,
    JobType,
    ProjectID,
    TrainingJobID,
)
from statedb.db.ml_jobs import MLJobService
from temporalio_commons.heartbeat_utils import auto_heartbeater

log = logging.getLogger(__name__)


@activity.defn
def init_home_dir_activity(home_dir: str) -> None:
    # Initialize the S3 home_dir by creating an '__init__' file
    HomeDirManager(home_dir).touch_init_file()


@activity.defn
def xai_aggregation_activity(
    job_input: TrainingJobInput
    | BatchPredictionJobInput,
    settings: MLJobSettings,
) -> None:
    r"""This activity is used to aggregate the RFMs and features for XAI
    in the post-processing step of either training or batch prediction
    workflows. Notice that failure of this activity will not affect the
    overall job completion. There are post-processing steps for aggregating
    global RFMs and global topk features.

    Args:
        job_input (TrainingJobInput | BatchPredictionJobInput): The job input.
            Either the training job input or the batch prediction job input.
        settings (MLJobSettings): The ML job settings.
    """
    home_dir = HomeDirManager(get_job_dir(get_job_base_dir(), job_input.id))
    if isinstance(job_input, TrainingJobInput):
        # Best trainer dir is only available for training
        best_trainer_dir = home_dir.get_best_trainer_dir(
            job_input.spec.predictive_query_id.name)
        stage = TrainingStage.TEST
        task_type = job_input.spec.task_type
    else:
        best_trainer_dir = None
        stage = TrainingStage.PRED
        task_type = job_input.config.task_type

    subgraph_collector_dir = home_dir.subgraph_collector_dir(
        stage, best_trainer_dir)

    if not fsutil.exists(subgraph_collector_dir):
        log.error(f"Subgraph collector dir "
                  f"{subgraph_collector_dir} does not exist")
        return None
    mapping_path = home_dir.subgraph_mappings_path(stage, best_trainer_dir)
    # Read via the in-repo writer schema, which accepts int primary keys in
    # orig_mappings. Long-term fix: widen EntityMappings.orig_mappings upstream.
    mappings = load_json(mapping_path, StrategyMappings).mappings
    rfm_dir = home_dir.rfm_dir(stage, best_trainer_dir)
    rfm_exists = fsutil.exists(rfm_dir)
    if not rfm_exists:
        log.warning(
            "[TT-NG] RFM dir %s does not exist; skipping RFM "
            "aggregation", rfm_dir)

    # Aggregate RFMs ##########################################################
    if rfm_exists:
        if task_type.is_node_pred:
            rfm_cls = NodePredRFM
            avg_func = node_pred_rfm_avg
        else:
            rfm_cls = LinkPredRFM
            avg_func = link_pred_rfm_avg

        # Folder to save the average RFMs for each category
        cat_rfm_dir = osp.join(subgraph_collector_dir, CATEGORY_RFM_DIR)
        fsutil.makedirs(cat_rfm_dir, exist_ok=True)
        # For each category, we average the RFMs of all entities in the category
        # to path at <cat_rfm_dir>/<category>.json
        for category, entities in mappings.items():
            # Limit the number of entities to postprocess XAI
            entities = entities[:MAX_NUM_POST_XAI_ENTITIES]
            rfms = []
            for entity in entities:
                rfm_path = home_dir.rfm_path(stage, entity, best_trainer_dir)
                if not fsutil.exists(rfm_path):
                    log.error(f"RFM path {rfm_path} does not exist")
                    continue
                rfm = load_json(rfm_path, rfm_cls)
                rfms.append(rfm)
            if len(rfms) == 0:
                log.error(f"No RFMs found for category {category}")
                continue
            avg_rfm = avg_func(rfms)
            save_json(avg_rfm, osp.join(cat_rfm_dir, f'{category}.json'))

    aggregate_subgraph_features(subgraph_collector_dir, stage,
                                exists=fsutil.exists, load_json=load_json,
                                save_json=save_json)


@activity.defn
def persist_training_job_dir_activity(job_input: TrainingJobInput,
                                      settings: MLJobSettings) -> None:
    # Copy data that needs to be persisted from the training job dir
    # into the persisted location.
    # This is done to make sure later predictions can use the persisted
    # training job data.
    # TODO(zeyuan): Test this with a finished training job dir.
    log.info("Persisting training job dir for job %s", job_input.id)
    home_dir = HomeDirManager(get_job_dir(get_job_base_dir(), job_input.id))
    persist_home_dir = HomeDirManager(
        get_job_dir(get_job_base_dir(), job_input.id))
    persist_home_dir.touch_init_file()
    fsutil.copy(home_dir.inputs_dir, persist_home_dir.inputs_dir,
                raise_on_exists=False)
    fsutil.copy(home_dir.outputs_dir, persist_home_dir.outputs_dir,
                raise_on_exists=False)
    fsutil.copy(home_dir.graphs_dir, persist_home_dir.graphs_dir,
                raise_on_exists=False)
    fsutil.copy(home_dir.jobs_dir, persist_home_dir.jobs_dir,
                raise_on_exists=False)
    fsutil.copy(home_dir.queries_dir, persist_home_dir.queries_dir,
                raise_on_exists=False)
    # TODO (min) Use multi process copying for the models and tasks dir which
    # contain larger files. The existing multi process copy is not reliable,
    # and it could miss files.
    fsutil.copy(home_dir.models_dir, persist_home_dir.models_dir,
                raise_on_exists=False)
    fsutil.copy(home_dir.tasks_dir, persist_home_dir.tasks_dir,
                raise_on_exists=False)
    task_name = job_input.spec.predictive_query.name
    # Update the job dir to the persisted job dir in json files
    # best trainer config JSON
    best_trainer_dir = persist_home_dir.get_best_trainer_dir(task_name)
    best_trainer_dir = best_trainer_dir.replace(home_dir._home_dir,
                                                persist_home_dir._home_dir)
    best_trainer_config_path = persist_home_dir.get_best_trainer_config_path(
        task_name)
    best_trainer_config = load_json(best_trainer_config_path, BestTrainerConfig)
    best_trainer_config.best_trainer = best_trainer_dir
    save_json(best_trainer_config, best_trainer_config_path)
    # task config JSON
    task_path = osp.join(persist_home_dir.tasks_dir, task_name, 'config.json')
    task_config = load_json(task_path, TaskConfig)
    if task_config.train_table.connector:
        root_dir = task_config.train_table.connector['root_dir']
        task_config.train_table.connector['root_dir'] = root_dir.replace(
            home_dir._home_dir, persist_home_dir._home_dir)
        save_json(task_config, task_path)
    # trainer config JSON
    trainer_config_path = osp.join(best_trainer_dir, 'config.json')
    trainer_config = load_json(trainer_config_path, TrainerConfig)
    if trainer_config.root_dir:
        trainer_config.root_dir = trainer_config.root_dir.replace(
            home_dir._home_dir, persist_home_dir._home_dir)
        save_json(trainer_config, trainer_config_path)


class JobOutputActivities:
    """Async activities that persist ML-job output / graph-version through an
    in-process, StateDB-bound ``MLJobOutputStore``.

    The store is bound (``_configure_ml_job_service``) to the
    ml-workflow-worker's own already-open StateDB during worker setup, so these
    activities write to the exact StateDB the job rows were created in. (A
    self-contained store would re-resolve its StateDB from the global
    ``deploy_env()``, which in the in-process multi-StateDB integ harness can
    diverge from the per-test ``DeployEnvConfig`` the worker uses -> "job does
    not exist".) They run on the worker's event loop, so the loop-bound pool is
    safe. Activity names are pinned so workflow references resolve unchanged.
    """
    def __init__(self, job_output_store: MLJobOutputStore):
        self._store = job_output_store

    @activity.defn(name='save_training_job_output')
    async def save_training_job_output(self, id: TrainingJobID,
                                       output: TrainingJobOutput):
        await self._store.save_training_job_output(id, output)

    @activity.defn(name='save_training_job_graph_version')
    async def save_training_job_graph_version(self, id: TrainingJobID,
                                              graph_data_version: GraphVersion):
        await self._store.save_training_job_graph_version(
            id, graph_data_version)

    @activity.defn(name='save_pred_job_graph_version')
    async def save_pred_job_graph_version(self, id: BatchPredictionJobID,
                                          graph_data_version: GraphVersion):
        await self._store.save_pred_job_graph_version(id, graph_data_version)

    @activity.defn(name='save_generate_train_table_graph_version')
    async def save_generate_train_table_graph_version(
            self, id: GenerateTrainTableJobID,
            graph_data_version: GraphVersion):
        await self._store.save_train_table_graph_version(id, graph_data_version)

    @activity.defn(name='save_generate_pred_table_graph_version')
    async def save_generate_pred_table_graph_version(
            self, id: GeneratePredictionTableJobID,
            graph_data_version: GraphVersion):
        await self._store.save_pred_table_graph_version(id, graph_data_version)

    @activity.defn(name='save_batch_prediction_job_output')
    async def save_batch_prediction_job_output(
            self, id: BatchPredictionJobID, output: BatchPredictionJobOutput):
        await self._store.save_batch_prediction_job_output(id, output)

    @activity.defn(name='save_generate_train_table_job_output')
    async def save_generate_train_table_job_output(
            self, id: GenerateTrainTableJobID,
            output: GenerateTrainTableJobOutput):
        await self._store.save_generate_train_table_job_output(id, output)

    @activity.defn(name='save_generate_pred_table_job_output')
    async def save_generate_pred_table_job_output(
            self, id: GeneratePredictionTableJobID,
            output: GeneratePredTableJobOutput):
        await self._store.save_generate_pred_table_job_output(id, output)


def _load_persisted_gen_pred_table_input(
    bp_job_id: BatchPredictionJobID,
    gen_pred_table_job_id: GeneratePredictionTableJobID,
) -> Tuple[BatchPredictionJobID, Optional[BatchPredictionJobInput]]:
    """Load a generator input from its relationship-owned job directory.

    A coupled generate-prediction-table workflow receives the parent batch
    prediction home directory and persists the shared ``BatchPredictionJobInput``
    there. A standalone generator owns its own directory (its input ID is the
    generator ID represented by ``BatchPredictionJobID``). Derive that location
    exclusively from the two durable IDs so callers cannot select an arbitrary
    input path.

    The expected input ID is returned with the exact deserialized object so the
    authorization activity can validate and consume that single snapshot.
    """
    if gen_pred_table_job_id.is_child_of(bp_job_id):
        expected_input_id = bp_job_id
    else:
        expected_input_id = BatchPredictionJobID(str(gen_pred_table_job_id))
    home_dir = get_job_dir(get_job_base_dir(), expected_input_id)
    input_path = HomeDirManager(home_dir).prediction_job_input_file
    try:
        gen_job_input = load_json(input_path, BatchPredictionJobInput)
    except FileNotFoundError:
        return expected_input_id, None
    if (gen_job_input is None or gen_job_input.graph_data_compute_input is None
            or gen_job_input.graph_version_info is None):
        return expected_input_id, None
    return expected_input_id, gen_job_input


@activity.defn
def load_task_config_activity(job_config_path: str) -> TaskConfig:
    job_config = JobConfig.load(job_config_path)
    return load_job_task(job_config).config()


def _bp_col_stat_workflow_id(job_id: str,
                             use_transformed_output: bool = False) -> str:
    wfid = f'bp-stats-{job_id}'
    if use_transformed_output:
        wfid += '-transformed'
    return wfid


def build_job_registry_metadata(
    job_id: JobID,
    job_input: Union[TrainingJobInput, BatchPredictionJobInput],
    ml_job_settings: MLJobSettings,
) -> MLJobRegistryMetadata:
    job_type = JobType.from_id(job_id)
    generator_advanced_options = None
    pquery_str = None
    trained_job_id = None
    bp_filters = None
    if job_type == JobType.TRAINING_JOB:
        assert isinstance(job_input, TrainingJobInput)
        pq_id = job_input.spec.predictive_query_id
        pquery_str = job_input.spec.predictive_query.query_string()
        generator_advanced_options = (
            job_input.spec.predictive_query.options_string())
    elif job_type == JobType.GENERATE_TRAIN_TABLE_JOB:
        assert isinstance(job_input, TrainingJobInput)
        pq_id = job_input.spec.predictive_query_id
        generator_advanced_options = to_yaml(
            job_input.spec.advanced_options.training_table_generation)
    elif job_type == JobType.BATCH_PREDICTION_JOB:
        assert isinstance(job_input, BatchPredictionJobInput)
        pq_id = job_input.predictive_query_id
        pquery_str = job_input.predictive_query.query_string()
        trained_job_id = job_input.training_job_id
        if job_input.config.filter_override_args:
            bp_filters = job_input.config.filter_override_args.to_yaml()
    elif job_type == JobType.GENERATE_PREDICTION_TABLE_JOB:
        assert isinstance(job_input, BatchPredictionJobInput)
        pq_id = job_input.predictive_query_id
    from service_framework.deploy_env_config import deploy_env
    config = deploy_env()
    return MLJobRegistryMetadata(
        instance_id=config.instance_id,
        pquery_id=str(pq_id),
        pquery_str=pquery_str,
        kumo_version=get_kumo_version(),
        generator_advanced_options=generator_advanced_options,
        trained_job_id=None if trained_job_id is None else str(trained_job_id),
        bp_filters=bp_filters,
    )


async def create_job_registry_metadata_in_state_db(
    job_id: JobID,
    job_input: Union[TrainingJobInput, BatchPredictionJobInput],
    ml_job_settings: MLJobSettings,
    ml_job_service: MLJobService,
) -> None:
    metadata = build_job_registry_metadata(job_id, job_input, ml_job_settings)
    await ml_job_service.upsert_job_registry_metadata(job_id, metadata)


class MLBackendActivities:
    """
    Temporal activity functions for ML training and batch prediction workflows.
    Each "activity" function executes a logical step/task, or more strictly
    speaking it represents a unit of state transition in the flow.

    We implement the activity functions as class instance methods so that
    activity executions can share and reuse long-lived, in-process
    dependencies (the StateDB-backed MLJobService, the job-admission queue,
    and the in-process MLJobBackend).

    The workflow classes below, TrainingWorkflow and
    BatchPredictionParentWorkflow are responsible for spawning these
    activities and orchestrate the execution. As the
    orchestration design evolves, we do expect to add more activities or
    break down a large/monolithic activity (training function) into smaller
    activities or a child workflow.
    """
    def __init__(
        self,
        job_queue: StateDbJobQueue,
        ml_backend,  # MLJobBackend
        ml_job_service: MLJobService,
        ctx: DataBackendContext,
        job_queue_settings: MLJobQueueSettings,
        slack_url: Optional[str] = None,
    ) -> None:
        from service_framework.deploy_env_config import deploy_env
        self._deploy_config = deploy_env()
        # DB-backed admission queue (replaces the AutoRPC MLJobQueueManager).
        self._job_queue = job_queue
        from rest.backends.ml_backend.ml_job_backend import MLJobBackend
        self._ml_backend: MLJobBackend = ml_backend
        self._ml_job_service = ml_job_service
        self._data_backend_ctx = ctx
        self._job_queue_poll_seconds = (
            job_queue_settings.job_queue_poll_seconds)
        self._slack_url = slack_url

    async def _load_project_bound_gen_pred_table_input(
        self,
        bp_job_input: BatchPredictionJobInput,
        gen_pred_table_job_id: GeneratePredictionTableJobID,
    ) -> Optional[BatchPredictionJobInput]:
        """Load a generator input only after durable ownership rechecks."""
        bp_job_input.validate_connector_execution_refs()
        project_ids = {bp_job_input.project_id}
        bp_rows = await self._ml_job_service.get_batch_prediction_rows(
            [bp_job_input.id], project_ids=project_ids)
        if len(bp_rows) != 1:
            raise ValueError(
                f"Batch prediction job '{bp_job_input.id}' does not belong "
                f"to project '{bp_job_input.project_id}'.")
        bp_row = bp_rows[0]
        if (BatchPredictionJobID(bp_row.id) != bp_job_input.id
                or ProjectID(bp_row.project_id) != bp_job_input.project_id
                or bp_row.config.generate_pred_table_job_id
                != gen_pred_table_job_id):
            raise ValueError(
                f"Batch prediction job '{bp_job_input.id}' does not "
                f"authorize prediction-table job "
                f"'{gen_pred_table_job_id}'.")

        gen_rows = await self._ml_job_service.get_generate_pred_table_jobs(
            [gen_pred_table_job_id], project_ids=project_ids)
        if len(gen_rows) != 1:
            raise ValueError(
                f"Prediction-table job '{gen_pred_table_job_id}' does not "
                f"belong to project '{bp_job_input.project_id}'.")
        gen_row = gen_rows[0]
        if (GeneratePredictionTableJobID(gen_row.id) != gen_pred_table_job_id
                or ProjectID(gen_row.project_id) != bp_job_input.project_id):
            raise ValueError(
                f"Prediction-table job '{gen_pred_table_job_id}' does not "
                f"belong to project '{bp_job_input.project_id}'.")

        expected_input_id, gen_job_input = (
            _load_persisted_gen_pred_table_input(bp_job_input.id,
                                                 gen_pred_table_job_id))
        if gen_job_input is None:
            return None
        if (gen_job_input.id != expected_input_id
                or gen_job_input.project_id != bp_job_input.project_id
                or gen_job_input.generate_pred_table_job_id
                != gen_pred_table_job_id):
            raise ValueError(
                f"Persisted input for prediction-table job "
                f"'{gen_pred_table_job_id}' does not match its durable "
                "job identity and project.")
        gen_job_input.validate_connector_execution_refs()
        return gen_job_input

    @activity.defn(name='load_decoupled_gen_pred_table_input')
    async def load_decoupled_gen_pred_table_input(
        self,
        bp_job_input: BatchPredictionJobInput,
        gen_pred_table_job_id: GeneratePredictionTableJobID,
    ) -> Optional[BatchPredictionJobInput]:
        return await self._load_project_bound_gen_pred_table_input(
            bp_job_input, gen_pred_table_job_id)

    @activity.defn
    async def upsert_job_registry_metadata(
        self,
        job_id: JobID,
        job_input: Union[TrainingJobInput, BatchPredictionJobInput],
        ml_job_settings: MLJobSettings,
    ) -> None:
        await create_job_registry_metadata_in_state_db(
            job_id,
            job_input,
            ml_job_settings,
            self._ml_job_service,
        )

    @activity.defn
    @auto_heartbeater
    async def wait_in_job_queue(self, job_id: JobID):
        """
        Submit a job to the DB-backed admission queue and wait for it to
        become ready to start running.

        ``submit_job`` is idempotent: it enqueues the job (no-op if already
        queued/running) and runs an admission pass, returning the resulting
        status.  We poll until the job is no longer QUEUED.
        """
        while True:
            log.info(f"Job {job_id} queueing... checking status.")
            job_status = await self._job_queue.submit_job(job_id)
            if job_status != JobStatus.QUEUED:
                log.info("Job %s status is %s", job_id, job_status)
                break
            try:
                log.info(f"Job {job_id} is waiting in queue.")
                await asyncio.sleep(self._job_queue_poll_seconds)
            except CancelledError:
                log.info(f"Job {job_id} cancelled. Not queueing.")
                raise

    @activity.defn
    async def finalize_job(self, job_id: JobID):
        """Workflow-owned finalization: terminalize (safety-net) + Slack +
        release the queue slot.

        Replaces the external ``_wait_for_job_wf_completion`` watcher.  Called
        from every parent workflow's ``finally`` block, so it runs on BOTH the
        success and failure (incl. cancellation) paths.

        Idempotent under Temporal activity retries:

        * If the workflow already wrote a terminal status + ``workflow_end_state``
          (the common case -- workflows write DONE/FAILED/CANCELLED inline), we
          leave it untouched.
        * Otherwise we apply the safety-net: mark the job FAILED with a
          "workflow finished but did not report a terminal status" message and
          freeze it via ``workflow_end_state``.  This mirrors the watcher's
          ``_apply_successful_workflow_completion`` fallback.

        We then send the Slack/incident failure notification (for training and
        batch-prediction jobs) and release the queue slot so the next queued
        job is admitted.
        """
        service = self._ml_job_service
        job = await service.get_job_if_exists_generic(job_id)
        if job is None:
            # Already deleted (e.g. cancel+delete); just release capacity.
            await self._job_queue.release_job(job_id)
            return

        job_type = JobType.from_id(job_id)
        execution = job.execution

        # Safety-net terminalization: if the workflow finished without writing
        # a terminal status, mark it FAILED and freeze it.  (When the workflow
        # already terminalized, workflow_end_state is set and we skip this.)
        if execution.workflow_end_state is None:
            from temporalio.client import WorkflowExecutionStatus

            from statedb.config.common import JobEventLogEntry, now_utc
            ts_now = now_utc()
            if not execution.status.is_terminal:
                from statedb.config.common import JobFailureInfo

                # Prefer the real validation error the workflow recorded
                # before returning non-terminal; fall back to the generic
                # message only when no failing validation response exists.
                error_msg: Optional[str] = None
                try:
                    response = await self._ml_backend.get_job_validation_response(
                        job_id)
                except Exception:
                    # The safety-net must always terminalize the job; never let
                    # a missing/corrupt validation response abort finalization.
                    # Fall back to the generic message in that case.
                    log.warning(
                        'finalize_job: could not read validation response for '
                        '%s; using generic terminal message', job_id,
                        exc_info=True)
                    response = None
                if response is not None and not response.ok:
                    error_msg = response.error_message()
                if not error_msg:
                    error_msg = (
                        'Workflow finished but did not report a terminal '
                        'status. Please retry this query.')
                execution.status = JobStatus.FAILED
                execution.failure_info = JobFailureInfo(message=error_msg,
                                                        stacktrace='')
                execution.log = JobEventLogEntry.failed(
                    ts_now, f'{job_type.value} failed: {error_msg}')
            execution.workflow_end_state = (WorkflowExecutionStatus.COMPLETED)
            execution.last_updated_at = max(execution.last_updated_at, ts_now)
            await service.modify_job_if_open_generic(job)

        # Slack notification on failure (training & BP only),
        # matching the legacy watcher behavior.
        if job_type in (JobType.TRAINING_JOB, JobType.BATCH_PREDICTION_JOB):
            refreshed = await service.get_job_if_exists_generic(job_id)
            if refreshed is not None:
                await self._send_failure_notification(
                    status=refreshed.execution.status,
                    job_type=job_type,
                    job_id=job_id,
                    failure_info=refreshed.execution.failure_info,
                )

        # Release the queue slot and admit the next queued job(s).
        await self._job_queue.release_job(job_id)
        log.info("Job %s finalized and unqueued.", job_id)

    async def _send_failure_notification(
        self,
        *,
        status: JobStatus,
        job_type: JobType,
        job_id: JobID,
        failure_info,
    ) -> None:
        """Send a Slack notification for a FAILED job during finalization."""
        if status != JobStatus.FAILED:
            return
        from rest.utils.public import get_base_kumo_url
        from service_framework.deploy_env import send_slack_message
        base_url = get_base_kumo_url()
        if job_type == JobType.TRAINING_JOB:
            url = f'{base_url}/models/training/{job_id}'
        elif job_type == JobType.BATCH_PREDICTION_JOB:
            url = f'{base_url}/predictions/prediction/{job_id}'
        else:
            url = None

        grafana_logs_dashboard = os.environ.get('GRAFANA_LOGS_DASHBOARD_URL',
                                                '').rstrip('?&')
        notification_recipient = os.environ.get(
            'INCIDENT_NOTIFICATION_RECIPIENT', '').strip()
        grafana_logs_line = ''
        if grafana_logs_dashboard:
            separator = '&' if '?' in grafana_logs_dashboard else '?'
            grafana_logs_line = (
                f'Grafana Logs: {grafana_logs_dashboard}{separator}'
                f'var-job_id={job_id}'
                '&from=now-14d&to=now\n')

        job_tags = await self._ml_job_service.get_job_tags(job_type, job_id)
        tag_vals = []
        if job_tags:
            tag_vals = [val for val in job_tags.tags.values()]
        is_prod = any("prod" in str(tag).lower() for tag in tag_vals)

        message = 'Unknown'
        if failure_info:
            message = failure_info.message
        err_msg = ""
        if is_prod:
            err_msg = ":warning::warning: Production :warning::warning: "
        err_msg += (f'{job_type} Failed\n\nURL: {url}\nError: {message}\n'
                    f'Tags: {tag_vals}\n'
                    f'{grafana_logs_line}')
        if notification_recipient:
            err_msg += f'cc {notification_recipient}'

        if self._slack_url:
            await send_slack_message.coroutine(err_msg, self._slack_url)
        import re
        log.error(re.sub(r'\n+', '; ', err_msg))


@activity.defn
def cleanup_diskgraph_staging(job_ids: list) -> None:
    """Remove diskgraph local staging dirs for specific job(s).

    Each staging dir is scoped by job_id, so this only cleans up
    the given job's artifacts without affecting concurrent jobs.
    Pass multiple IDs when a BP job needs to clean both its own
    pred_artifact_cache and the training job's edge_persist.

    Delegates to ``DiskgraphService.cleanup_staging``.
    """
    get_service().cleanup_staging(job_ids)


@activity.defn
def persist_job_inputs(job_input: Union[TrainingJobInput,
                                        BatchPredictionJobInput],
                       ml_job_settings: MLJobSettings,
                       home_dir: Optional[str] = None) -> None:

    home_dir = home_dir or get_job_dir(get_job_base_dir(), job_input.id)
    home_dir_mgr = HomeDirManager(home_dir)

    home_dir_mgr.touch_init_file()

    # A generate-table workflow re-publishes this input after resolving its
    # graph version. Deleting the whole directory first creates a window in
    # which concurrent consumers observe a missing input. Each exact-key
    # write is atomic on object stores, so preserve the published keys and
    # overwrite them in place.
    fsutil.makedirs(home_dir_mgr.inputs_dir, exist_ok=True)
    if fsutil.is_local(home_dir_mgr.inputs_dir):
        # Persisted job inputs are internal configuration. The directory is
        # private even before individual mode-0600 files are published.
        os.chmod(home_dir_mgr.inputs_dir, 0o700)

    storage_scheme = fsutil.get_scheme(home_dir_mgr.inputs_dir)
    if storage_scheme in {'local', 's3', 'gs', 'abfs', 'abfss'}:
        persist_json = save_json_atomic
    elif storage_scheme == 'dbfs':
        # The Databricks Files API does not document atomic overwrite
        # visibility. Preserve those existing deployments while making the
        # residual reader race explicit; do not mislabel their direct
        # overwrite as atomic.
        log.warning(
            'Persisting job inputs with legacy non-atomic overwrite because '
            'storage scheme %s has no proven atomic replacement primitive. '
            'Concurrent readers may observe transient read failures.',
            storage_scheme)
        persist_json = save_json
    else:
        # Let the atomic helper fail closed for an unknown storage provider.
        persist_json = save_json_atomic

    if isinstance(job_input, TrainingJobInput):
        job_file = home_dir_mgr.training_job_input_file
        persist_json(job_input.spec.advanced_options,
                     home_dir_mgr.advanced_options_file)
    else:
        job_file = home_dir_mgr.prediction_job_input_file

    # Settings/options are immutable for a job revision. Publish them first,
    # then publish the job input last as the marker that a complete revision is
    # available. The workflow's initial and post-refresh publications are
    # sequential, while atomic replacement keeps concurrent readers on either
    # the previous or complete new input.
    persist_json(ml_job_settings, home_dir_mgr.ml_job_settings_file)
    persist_json(job_input, job_file)
    log.info(f"Inputs persisted for job {job_input.id} to {job_file} "
             f"and {home_dir_mgr.ml_job_settings_file}")
