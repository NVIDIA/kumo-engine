# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Core workflow: materialize training task
from datetime import timedelta
from typing import Optional, Tuple

from temporalio import workflow

from io_utils import logging
from pool.activities import SnapshotActivities
from pool.dataclasses import ResourceHint, SnapshotActivity
from pool.wf_coordinator import run_pooled_activity
from rest.backends import (
    DATA_BACKEND_IO_TIMEOUT_HOURS,
    DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS,
)
from rest.backends.ml_backend.activities import (
    JobOutputActivities,
    generate_prediction_config,
    generate_training_config,
)
from rest.backends.ml_backend.settings import MLJobSettings
from rest.backends.ml_backend.validation import ValidationResponse
from rest.config.batch_prediction_job import BatchPredictionJobInput
from rest.config.ml_backend import TrainingJobInput
from rest.config.training_job import TrainingJobOutput
from rest.constants import (
    COMPUTE_ACTIVITY_RETRY_POLICY,
    DATA_BACKEND_TASK_QUEUE,
    DEFAULT_HEARTBEAT_TIMEOUT,
    HEARTBEAT_TIMEOUT_COMPUTE_ACTIVITY,
    ML_WORKFLOW_TASK_QUEUE,
)
from statedb.config._graph_key_norm import _graph_key_to_v2
from statedb.config.common import (
    DataSnapshot,
    JobEventLogEntry,
    JobStatus,
    JobType,
    TrainingJobID,
)

log = logging.getLogger(__name__)


@workflow.defn
class MaterializeTrainingTaskWorkflow:
    @staticmethod
    def generate_wf_id(job_id: str) -> str:
        return f"materialize-training-task-workflow-{job_id}"

    async def _save_output(
        self,
        job_input: TrainingJobInput,
        ml_job_settings: MLJobSettings,
        output: TrainingJobOutput,
    ):
        await workflow.execute_activity(
            JobOutputActivities.save_training_job_output,
            args=[job_input.id, output],
            start_to_close_timeout=timedelta(minutes=10),
            schedule_to_close_timeout=timedelta(hours=1),
        )

    @workflow.run
    async def run(
        self,
        job_input: TrainingJobInput,
        ml_job_settings: MLJobSettings,
        train_table_path: str,
        train_table_materialization_path: str,
        weight_col_name: Optional[str] = None,
    ) -> Tuple[str, ValidationResponse]:
        log.info("Start materialize training task")

        # Update training job stats when this is part of a training workflow.
        if TrainingJobID.is_valid_id(job_input.id):
            job_output = TrainingJobOutput(status=JobStatus.RUNNING,
                                           log=JobEventLogEntry.materializing())
            await self._save_output(job_input, ml_job_settings, job_output)

        # generate_training_config was de-globalized off the compute process
        # pool onto the data-backend thread pool (workflow_worker), so target
        # the data-backend queue. The ML workflow runs on its own worker; derive
        # the sibling data-backend queue from this workflow's own (history-
        # recorded) queue name so the deploy-specific suffix is preserved
        # without a non-deterministic config read.
        data_backend_task_queue = workflow.info().task_queue.replace(
            ML_WORKFLOW_TASK_QUEUE, DATA_BACKEND_TASK_QUEUE)
        job_config_path = await workflow.execute_activity(
            generate_training_config,
            args=[job_input, ml_job_settings],
            task_queue=data_backend_task_queue,
            schedule_to_close_timeout=timedelta(
                hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
            heartbeat_timeout=DEFAULT_HEARTBEAT_TIMEOUT,
            retry_policy=COMPUTE_ACTIVITY_RETRY_POLICY,
        )
        log.info("Generated training job config at %s", job_config_path)

        # Training materialization runs via the pool path; snapshot
        # residency is ensured by the pool worker.
        assert job_input.graph_data_compute_input is not None
        assert job_input.graph_version_info is not None
        graph_key = job_input.graph_data_compute_input.graph_key
        graph_ver = job_input.graph_version_info.graph_data_version
        snapshot_id = DataSnapshot.GRAPH.canonicalize(
            f'{_graph_key_to_v2(graph_key)}@{graph_ver}')
        job_config_path, _, response = await run_pooled_activity(
            snapshot_id=snapshot_id,
            activity=SnapshotActivity.MATERIALIZE_TRAINING_TASK,
            activity_fn=SnapshotActivities.materialize_training_task,
            args=[
                snapshot_id,
                job_input,
                ml_job_settings,
                train_table_path,
                train_table_materialization_path,
                job_config_path,
                weight_col_name,
            ],
            schedule_to_close_timeout=timedelta(
                hours=DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS),
            heartbeat_timeout=HEARTBEAT_TIMEOUT_COMPUTE_ACTIVITY,
            retry_policy=COMPUTE_ACTIVITY_RETRY_POLICY,
        )
        log.info("Finish materialize training task")
        return job_config_path, response


# Core workflow: materialize training task
@workflow.defn
class MaterializePredictionTaskWorkflow:
    @staticmethod
    def generate_wf_id(job_id: str) -> str:
        return f"materialize-prediction-task-workflow-{job_id}"

    @workflow.run
    async def run(
        self,
        job_input: BatchPredictionJobInput,
        ml_job_settings: MLJobSettings,
        pred_table_path: str,
        serving: bool,
        home_dir: Optional[str] = None,
        job_type: JobType = JobType.BATCH_PREDICTION_JOB,
        resource_hint: Optional[ResourceHint] = None,
    ) -> Tuple[str, ValidationResponse]:
        log.info("Start materialize prediction task")
        # generate_prediction_config was de-globalized and moved to the
        # data-backend thread-pool worker; target its queue (suffix-preserving)
        # instead of the compute process-pool queue.
        data_backend_task_queue = workflow.info().task_queue.replace(
            ML_WORKFLOW_TASK_QUEUE, DATA_BACKEND_TASK_QUEUE)
        pred_config_path = await workflow.execute_activity(
            generate_prediction_config,
            args=[
                job_input,
                ml_job_settings,
                serving,
                pred_table_path,
                home_dir,
            ],
            task_queue=data_backend_task_queue,
            schedule_to_close_timeout=timedelta(
                hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
            heartbeat_timeout=DEFAULT_HEARTBEAT_TIMEOUT,
            retry_policy=COMPUTE_ACTIVITY_RETRY_POLICY,
        )
        log.info("Generated prediction config at %s", pred_config_path)

        # Prediction materialization runs via the pool path; snapshot
        # residency is ensured by the pool worker.
        assert job_input.graph_data_compute_input is not None
        assert job_input.graph_version_info is not None
        graph_key = job_input.graph_data_compute_input.graph_key
        graph_ver = job_input.graph_version_info.graph_data_version
        snapshot_id = DataSnapshot.GRAPH.canonicalize(
            f'{_graph_key_to_v2(graph_key)}@{graph_ver}')
        pred_config_path, _, response = await run_pooled_activity(
            snapshot_id=snapshot_id,
            activity=SnapshotActivity.MATERIALIZE_PREDICTION_TASK,
            activity_fn=SnapshotActivities.materialize_prediction_task,
            args=[
                snapshot_id,
                job_input,
                ml_job_settings,
                serving,
                pred_config_path,
                home_dir,
                job_type,
            ],
            schedule_to_close_timeout=timedelta(
                hours=DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS),
            heartbeat_timeout=HEARTBEAT_TIMEOUT_COMPUTE_ACTIVITY,
            retry_policy=COMPUTE_ACTIVITY_RETRY_POLICY,
            resource_hint=resource_hint,
        )
        log.info("Finish materialize prediction task")
        return pred_config_path, response
