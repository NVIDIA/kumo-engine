# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Parent workflow: the whole training pipeline.
from datetime import timedelta

from kumo_core.types import TaskType
from temporalio import workflow
from temporalio.client import WorkflowExecutionStatus
from temporalio.workflow import (
    ChildWorkflowCancellationType,
    ParentClosePolicy,
)

from io_utils import logging
from io_utils.exceptions import catch_and_log
from rest.backends.common.activities import (
    load_validation_response_activity,
    save_validation_response,
)
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.ml_backend.activities import (
    cleanup_diskgraph_staging,
    init_home_dir_activity,
    persist_training_job_dir_activity,
    xai_aggregation_activity,
)
from rest.backends.ml_backend.settings import MLJobSettings
from rest.backends.ml_backend.validation import ValidationResponse
from rest.config.ml_backend import TrainingJobInput
from rest.config.training_job import TrainingJobOutput, TrainingJobResults
from rest.constants import RETRY_ONCE_POLICY
from rest.settings import DataBackendSettings
from service_framework.deploy_env_config import deploy_env
from statedb.config.common import (
    GenerateTrainTableJobID,
    JobEventLogEntry,
    JobStatus,
)

from .base_training_workflow import BaseTrainingWorkflow
from .materialize_task_workflows import MaterializeTrainingTaskWorkflow
from .training_job_workflow import TrainingJobWorkflow

log = logging.getLogger(__name__)


@workflow.defn
class TrainingWorkflow(BaseTrainingWorkflow):
    @workflow.run
    async def run(  # temporal requires run to be defined in the subclass
        self,
        job_input: TrainingJobInput,
        ml_job_settings: MLJobSettings,
    ) -> ValidationResponse:
        return await super().run(job_input, ml_job_settings)

    async def _run(
        self,
        job_input: TrainingJobInput,
        ml_job_settings: MLJobSettings,
    ) -> ValidationResponse:
        # No outer session lease here: at _run entry the
        # graph_data_version isn't yet known (refresh happens further
        # down), so a canonical snapshot_id can't be formed.  Inner
        # leases — DiskgraphIngestWorkflow (held during ingest) and
        # TrainingJobWorkflow (held during the kumo-auto subprocess) —
        # carry the correct canonical id once it's available.
        self.job_input = job_input
        self.ml_job_settings = ml_job_settings
        validation_response = ValidationResponse()

        home_dir = await self._initialize_train_job()

        custom_train_table_spec = self.job_input.spec.custom_train_table
        gen_train_table_job_id = GenerateTrainTableJobID(
            self.job_input.generate_train_table_job_id)
        validation_response.merge(await self._validate_custom_train_table(
            gen_train_table_job_id, self.job_input.id, custom_train_table_spec))
        if not validation_response.ok:
            return await self._exit_cleanup(validation_response,
                                            self.job_input.id, home_dir)

        await workflow.execute_activity(
            init_home_dir_activity, args=[home_dir],
            schedule_to_close_timeout=timedelta(hours=1))
        self.check_cancelled()

        # Mark job status as RUNNING
        await self._save_output(
            TrainingJobOutput(status=JobStatus.RUNNING,
                              log=JobEventLogEntry.starting()))

        # graph_compute_input is passed in from the REST driver
        graph_compute_input = self.job_input.graph_data_compute_input
        assert graph_compute_input is not None

        # Step 1: Refresh Graph Data, including table ingestion, col stats, and
        # edge stats computation.
        data_backend_ctx = DataBackendContext(
            deploy_env(), DataBackendSettings(disable_async_ingestion=False))

        graph_input = await self._refresh_graph(graph_compute_input,
                                                validation_response, home_dir,
                                                data_backend_ctx)
        if graph_input is None:
            # _refresh_graph already wrote a terminal FAILED output (with the
            # real validation error) before returning None, so there is no
            # non-terminal exit to terminalize here.
            return validation_response

        train_table_result = await self._maybe_generate_train_table(
            home_dir, data_backend_ctx, validation_response)
        if train_table_result is None:
            return await self._exit_cleanup(validation_response,
                                            self.job_input.id, home_dir)

        # Step 3: Materialize Training Task
        job_config_path, response = await workflow.execute_child_workflow(
            MaterializeTrainingTaskWorkflow.run, args=[
                job_input,
                ml_job_settings,
                train_table_result.train_table_path,
                train_table_result.train_table_materialization_path,
                job_input.weight_col_name,
            ], id=MaterializeTrainingTaskWorkflow.generate_wf_id(job_input.id),
            cancellation_type=ChildWorkflowCancellationType.TRY_CANCEL,
            parent_close_policy=ParentClosePolicy.TERMINATE)

        validation_response.merge(response)
        if not validation_response.ok:
            return await self._exit_cleanup(validation_response, job_input.id,
                                            home_dir)

        if job_input.spec.stop_after_materialization:
            log.info("Workflow was requested to stop after materialization, so "
                     "it is exiting.")
            # Terminalize as DONE explicitly: without a terminal status,
            # finalize_job would mislabel this success-shaped early-return as
            # FAILED-generic. Empty results still satisfy the DONE invariant.
            await self._save_output(
                TrainingJobOutput(
                    status=JobStatus.DONE, log=JobEventLogEntry.done(),
                    results=TrainingJobResults(),
                    workflow_end_state=WorkflowExecutionStatus.COMPLETED))
            return validation_response

        # Step 4: Launch Training on K8S
        wf_id = TrainingJobWorkflow.generate_wf_id(job_input.id)
        await workflow.execute_child_workflow(
            TrainingJobWorkflow.run,
            args=[job_input, ml_job_settings, job_config_path], id=wf_id,
            cancellation_type=ChildWorkflowCancellationType.TRY_CANCEL,
            parent_close_policy=ParentClosePolicy.REQUEST_CANCEL)

        # Merge any warnings saved by TrainingJobWorkflow (e.g., batch size reduction)
        training_job_validation: ValidationResponse = await workflow.execute_activity(
            load_validation_response_activity, args=[home_dir, job_input.id],
            schedule_to_close_timeout=timedelta(minutes=5))
        validation_response.merge(training_job_validation)

        # Step 5: Post processing:
        if (deploy_env().data_plane_storage_root
                != deploy_env().control_plane_storage_root):
            await workflow.execute_activity(
                persist_training_job_dir_activity,
                args=[job_input, ml_job_settings],
                schedule_to_close_timeout=timedelta(minutes=30),
                retry_policy=RETRY_ONCE_POLICY)

        # Step 6: XAI aggregation — post-processing only; per its own
        # contract ("failure of this activity will not affect the overall
        # job completion") a failure here must not fail the finished
        # training job, same as the batch-prediction path.
        if job_input.spec.task_type not in {
                TaskType.FORECASTING, TaskType.MULTILABEL_RANKING
        }:
            with catch_and_log():
                await workflow.execute_activity(
                    xai_aggregation_activity, args=[job_input, ml_job_settings],
                    schedule_to_close_timeout=timedelta(minutes=30),
                    retry_policy=RETRY_ONCE_POLICY)

        # Clean up diskgraph local staging dirs (rehydration_cache,
        # edge_persist, pred_artifact_cache).  Safe now that training,
        # post-processing, and XAI are all done.
        with catch_and_log():
            await workflow.execute_activity(
                cleanup_diskgraph_staging,
                args=[[job_input.id]],
                schedule_to_close_timeout=timedelta(minutes=5),
            )

        # Write validation outputs to the job directory:
        await save_validation_response(validation_response, home_dir,
                                       job_input.id)
        return validation_response
