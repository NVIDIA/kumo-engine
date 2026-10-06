# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List, Tuple

import torch
from temporalio.worker import Worker

from io_utils import logging
from pool.activities import (
    get_pool_stats_activity,
    place_snapshot_activity,
    release_snapshot_activity,
)
from rest.backends import (
    ML_PRED_MAX_NUM_ENTITIES_PER_PARTITION,
    ML_PRED_MAX_NUM_ROWS_PER_PARTITION,
)
from rest.backends.common.activities import io_activities
from rest.backends.data_backend.activities.connector_authorization import (
    delegate_connector_authorizations,
    revoke_connector_authorizations,
)
from rest.backends.data_backend.connector_activity_registry import (
    validate_connector_activity_registration,
)
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.ml_backend.activities import (
    JobOutputActivities,
    MLBackendActivities,
    TrainingJobActivities,
    cleanup_diskgraph_staging,
    copy_train_table_stats,
    init_home_dir_activity,
    load_task_config_activity,
    persist_job_inputs,
    persist_training_job_dir_activity,
    xai_aggregation_activity,
)
from rest.backends.ml_backend.activities.parquet_io_activities import (
    cleanup_databricks_volume_outputs,
    collect_prediction_job_output,
    finalize_prediction_job_output,
    get_artifact_stats,
    write_artifacts_to_customer,
)
from rest.backends.ml_backend.job_queue import StateDbJobQueue
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.backends.ml_backend.ml_job_output_store import MLJobOutputStore
from rest.backends.ml_backend.settings import (
    MLBackendSettings,
    MLJobQueueSettings,
    MLJobSettings,
)
from rest.backends.ml_backend.workflows import (
    ArtifactExportWorkflow,
    BatchPredictionJobWorkflow,
    BatchPredictionParentWorkflow,
    BatchPredictionStatsWorkflow,
    CustomTrainTableWorkflow,
    GeneratePredictionTableWorkflow,
    GenerateTrainingTableWorkflow,
    MaterializePredictionTaskWorkflow,
    MaterializeTrainingTaskWorkflow,
    TrainingJobWorkflow,
    TrainingWorkflow,
)
from rest.constants import ML_WORKFLOW_WORKER
from rest.settings import DataBackendSettings

# isort: off
from rest.backends.ml_backend.workflows.split_workflow import (
    SplitTrainingTableWorkflow, )
# isort: on
from rest.settings import DataScaleLimits, get_db_storage_root
from service_framework.deploy_stage import DeployStage
from statedb.db import MLJobService, open_state_db_from_deploy_config
from temporalio_commons import WorkerEnv, register_bundled_workers

log = logging.getLogger(__name__)

# Env variable name to enable fast training in DEV or TEST mode.
ML_FAST_TRAINING_ENABLED = 'FAST_TRAIN'
PRED_COL_STATS = 'PRED_COL_STATS'
SLACK_URL = 'SLACK_URL'

# Reduce the number of iterations performed by the AutoTrainer in order to
# make the training jobs run 100x faster. Intended for local development
# and integration testing.
TRAINING_CFG_FAST_DEV = {
    'max_epochs': 5,
    'num_experiments': 2,
}


async def _create_backend_activities(
    worker_env: WorkerEnv
) -> Tuple[MLBackendActivities, TrainingJobActivities, JobOutputActivities]:
    deploy_config = worker_env.deploy_config
    log.info(f"Creating MLWorkflowWorker with deploy_config: {deploy_config}")
    ml_backend_settings = generate_ml_backend_settings(worker_env)
    state_db = await open_state_db_from_deploy_config(
        deploy_config,
        state_root=get_db_storage_root(),
    )
    ml_job_service = MLJobService(state_db)
    log.info('ML workflow metadata StateDB initialized')
    # JobOutputActivities (async save_* activities) write through a StateDB-BOUND
    # store over the worker's own StateDB, so they hit the exact StateDB the job
    # rows were created in (correct-by-construction; avoids the self-contained
    # path re-resolving via the global deploy_env()). They run on the worker's
    # loop, so the loop-bound pool is safe.
    bound_output_store = MLJobOutputStore()
    bound_output_store._configure_ml_job_service(ml_job_service)
    job_output_activities = JobOutputActivities(bound_output_store)
    job_queue_settings = MLJobQueueSettings.parse_from_env()
    # DB-backed admission queue over the already-open StateDB (replaces the
    # AutoRPC MLJobQueueManager).
    job_queue = StateDbJobQueue(state_db, job_queue_settings)
    # In-process MLJobBackend over the already-open StateDB (no AutoRPC client).
    # read_only=False: the ml-workflow-worker is now the single lifecycle owner
    # (reconcile + admission recovery). Both passes are idempotent and
    # concurrency-safe, so multiple workflow-worker replicas are fine.
    ml_job_backend = MLJobBackend(ml_backend_settings,
                                  worker_env.temporal_client, state_db)
    await ml_job_backend.start(read_only=False)
    data_backend_ctx = DataBackendContext(
        deploy_config, DataBackendSettings(disable_async_ingestion=False))
    ml_be_activities = MLBackendActivities(
        job_queue,
        ml_job_backend,
        ml_job_service,
        data_backend_ctx,
        job_queue_settings,
        slack_url=ml_backend_settings.slack_url,
    )

    training_activities = TrainingJobActivities()
    return (
        ml_be_activities,
        training_activities,
        job_output_activities,
    )


@register_bundled_workers(worker_name=ML_WORKFLOW_WORKER)
async def build_ml_workflow_worker(worker_env: WorkerEnv) -> List[Worker]:
    # The data persist dir is derived from deploy_env() on demand
    # (kumo.data.get_persist_dir); the engine layer (DiskGraph) holds no
    # process-global state to prime here.
    log.info("Starting to create backend activities")

    (
        ml_be_activities,
        training_activities,
        job_output_activities,
    ) = await _create_backend_activities(worker_env)
    log.info("Completed creating backend activities")

    # Training/prediction run in-process via SnapshotActivities (the pool
    # path); the legacy K8s launch/poll/delete activities have been removed.
    # Training failure reporting and postprocessing stay on this worker.
    ml_backend_activities = [
        delegate_connector_authorizations,
        revoke_connector_authorizations,
        ml_be_activities.wait_in_job_queue,
        ml_be_activities.finalize_job,
        ml_be_activities.upsert_job_registry_metadata,
        # base_training_workflow always copies train-table stats
        # into the per-job dir post-gen.
        copy_train_table_stats,
        init_home_dir_activity,
        ml_be_activities.load_decoupled_gen_pred_table_input,
        persist_training_job_dir_activity,
        xai_aggregation_activity,
        job_output_activities.save_training_job_output,
        job_output_activities.save_batch_prediction_job_output,
        job_output_activities.save_training_job_graph_version,
        job_output_activities.save_pred_job_graph_version,
        job_output_activities.save_generate_train_table_job_output,
        job_output_activities.save_generate_pred_table_job_output,
        job_output_activities.save_generate_pred_table_graph_version,
        job_output_activities.save_generate_train_table_graph_version,
        load_task_config_activity,
        io_activities.save_json_activity,
        io_activities.get_missing_files,
        cleanup_diskgraph_staging,
        persist_job_inputs,
        io_activities.save_validation_response_activity,
        io_activities.load_validation_response_activity,
        # Pool integration: workflow-side wrappers that bridge
        # Placement coordinators to PoolController
        # via AutoRPC.
        place_snapshot_activity,
        release_snapshot_activity,
        get_pool_stats_activity,
        # Shared post-completion activities.
        training_activities.postprocess_job_output,
        training_activities.log_failed_job,
        # I/O-bound parquet post-processing runs on the thread executor.
        # Artifact export keeps credentialed provider I/O in the activity
        # process, but isolates its credential-free Pandas/PyArrow CSV
        # conversion in a disposable child process.
        collect_prediction_job_output,
        finalize_prediction_job_output,
        cleanup_databricks_volume_outputs,
        write_artifacts_to_customer,
        get_artifact_stats,
    ]
    validate_connector_activity_registration(ml_backend_activities)
    ml_wf_worker = await worker_env.build_worker(
        activities=ml_backend_activities,
        workflows=[
            BatchPredictionJobWorkflow,
            BatchPredictionParentWorkflow,
            CustomTrainTableWorkflow,
            GeneratePredictionTableWorkflow,
            GenerateTrainingTableWorkflow,
            MaterializePredictionTaskWorkflow,
            MaterializeTrainingTaskWorkflow,
            TrainingJobWorkflow,
            TrainingWorkflow,
            ArtifactExportWorkflow,
            BatchPredictionStatsWorkflow,
            SplitTrainingTableWorkflow,
        ],
    )

    return [ml_wf_worker]


TRAINING_CFG_CPU_TEST = {
    'max_epochs': 5,
    'num_experiments': 2,
}


def generate_ml_backend_settings(worker_env: WorkerEnv) -> MLBackendSettings:
    data_limits = DataScaleLimits.parse_from_env()
    deploy_mode = worker_env.deploy_config.deploy_mode
    slack_url = worker_env.get_variable(SLACK_URL)
    log.info("Slack URL: %s", slack_url)
    if not slack_url or not slack_url.startswith('https'):
        slack_url = ''

    settings = MLBackendSettings(
        slack_url=slack_url,
        ml_job_settings=MLJobSettings(
            pred_max_num_entities_per_partition=worker_env.get_variable(
                ML_PRED_MAX_NUM_ENTITIES_PER_PARTITION),
            pred_max_num_rows_per_partition=worker_env.get_variable(
                ML_PRED_MAX_NUM_ROWS_PER_PARTITION),
            max_total_table_size=data_limits.max_graph_total_size_bytes,
            max_total_graph_edges=data_limits.max_graph_total_edges,
        ),
        job_queue_settings=MLJobQueueSettings.parse_from_env(),
        prediction_col_stats=worker_env.get_variable(PRED_COL_STATS),
    )
    if deploy_mode in (DeployStage.PROD, DeployStage.STAGING):
        return settings

    training_job_settings = settings.ml_job_settings
    if worker_env.get_variable(ML_FAST_TRAINING_ENABLED):
        training_job_settings.train_cfg_overrides = TRAINING_CFG_FAST_DEV
    elif deploy_mode == DeployStage.TEST:
        if torch.cuda.is_available():
            training_job_settings.train_cfg_overrides = TRAINING_CFG_FAST_DEV
        else:
            training_job_settings.train_cfg_overrides = TRAINING_CFG_CPU_TEST

    return settings
