# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from datetime import timedelta
from typing import Optional

from temporalio import workflow
from temporalio.client import WorkflowExecutionStatus
from temporalio.exceptions import CancelledError
from temporalio.workflow import (
    ActivityCancellationType,
    ChildWorkflowCancellationType,
    ParentClosePolicy,
)

from io_utils import logging
from io_utils.exceptions import catch_and_log
from pool.activities import SnapshotActivities
from pool.dataclasses import ResourceHint, SnapshotActivity
from pool.wf_coordinator import run_pooled_activity
from rest.backends import (
    DATA_BACKEND_BP_STATS_TIMEOUT_HOURS,
    DATA_BACKEND_IO_TIMEOUT_HOURS,
    DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS,
    DATA_BACKEND_REFRESH_TIMEOUT_HOURS,
)
from rest.backends.common.activities import save_validation_response
from rest.backends.data_backend.activities import DataBackendActivities
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.workflows import GraphRefreshWorkflow
from rest.backends.ml_backend.activities import (
    JobOutputActivities,
    MLBackendActivities,
    cleanup_diskgraph_staging,
    init_home_dir_activity,
    xai_aggregation_activity,
)
from rest.constants import (
    CHILD_WF_WAITING_RETRY_POLICY,
    COMPUTE_ACTIVITY_RETRY_POLICY,
    DATA_BACKEND_TASK_QUEUE,
    HEARTBEAT_TIMEOUT_COMPUTE_ACTIVITY,
)
from rest.settings import DataBackendSettings

# isort: off
from rest.backends.ml_backend.activities.common import (
    update_graph_config_from_graph_input, )
# isort: on
from rest.backends.ml_backend.data_util import get_job_base_dir, get_job_dir
from rest.backends.ml_backend.settings import MLJobSettings
from rest.backends.ml_backend.validation import (
    ValidationError,
    ValidationResponse,
)

# isort: off
from rest.backends.ml_backend.workflows.mixins import (
    JobWorkflowCancellationMixin, )
# isort: on
from kumo_core.types import PredictionArtifactType, TaskType

from rest.backends.ml_backend.workflows.utils import (
    dequeue_job,
    enqueue_job,
    init_job_generic,
)
from rest.config.batch_prediction_job import (
    BatchPredictionJobInput,
    BatchPredictionJobOutput,
    BatchPredictionJobResults,
    requested_prediction_artifact_types,
)
from rest.config.error_handling import root_cause_message
from service_framework.deploy_env_config import deploy_env
from statedb.config._graph_key_norm import _graph_key_to_v2
from statedb.config.common import (
    DataSnapshot,
    GeneratePredictionTableJobID,
    JobEventLogEntry,
    JobStatus,
    JobType,
)
from statedb.utils.errors import maybe_get_public_message
from temporalio_commons import is_cancelled_exception

from .batch_prediction_job_workflow import (
    BatchPredictionJobWorkflow,
    _configured_max_durable_gpu_slots,
    _prediction_resource_hint,
)
from .batch_prediction_stats_workflow import BatchPredictionStatsWorkflow
from .generate_table_workflows import (
    GeneratePredictionTableWorkflow,
    GeneratePredTableResult,
)
from .materialize_task_workflows import MaterializePredictionTaskWorkflow

log = logging.getLogger(__name__)


# Parent workflow: the whole prediction pipeline.
@workflow.defn
class BatchPredictionParentWorkflow(JobWorkflowCancellationMixin):
    async def _save_output(self, output: BatchPredictionJobOutput):
        await workflow.execute_activity(
            JobOutputActivities.save_batch_prediction_job_output,
            args=[self._job_input.id,
                  output], schedule_to_close_timeout=timedelta(minutes=10))
        self.check_cancelled()

    async def _exit_cleanup(
        self,
        validation_response: ValidationResponse,
        job_id: str,
        home_dir: str,
        exception: Optional[BaseException] = None,
    ) -> ValidationResponse:
        """Terminalize a failed job via the workflow_end_state chokepoint.

        ``exception`` (crash path only) populates ``failure_info.stacktrace``,
        which would otherwise be empty for every workflow-crash failure;
        traceback formatting is deterministic, so this stays workflow-safe.
        """
        await save_validation_response(validation_response, home_dir, job_id)
        job_output = BatchPredictionJobOutput.failed(
            exception=exception, validation=validation_response,
            workflow_end_state=WorkflowExecutionStatus.COMPLETED)
        await self._save_output(job_output)
        return validation_response

    @workflow.run
    async def run(
        self,
        job_input: BatchPredictionJobInput,
        ml_job_settings: MLJobSettings,
    ) -> ValidationResponse:
        # Mirror _run's assignments so the except-handler can terminalize early.
        self._job_input = job_input
        self._ml_job_settings = ml_job_settings
        # Pool integration note: see TrainingWorkflow._run for the
        # rationale behind not opening a session-level lease at this
        # outer point. The meaningful leases live in
        # DiskgraphIngestWorkflow + BatchPredictionJobWorkflow, where
        # the canonical GraphSnapshotID is derivable.
        try:
            return await self.run_until_canceled(
                self._run(job_input, ml_job_settings))
        except BaseException as ex:
            # Record the real cause so finalize_job doesn't mask it with the
            # generic "did not report a terminal status" message; cancellation
            # and continue-as-new are control flow, not failures.
            if is_cancelled_exception(ex) or isinstance(
                    ex, workflow.ContinueAsNewError):
                raise
            # Never let a terminalize failure swallow the original exception.
            with catch_and_log():
                home_dir = get_job_dir(get_job_base_dir(), job_input.id)
                # User-facing message: prefer the public-error machinery
                # (a PublicException attached anywhere on the chain, or the
                # curated translation of a bare Temporal timeout) over the
                # deepest raw failure text, which leaks internals like
                # "activity Heartbeat timeout" (kumo-ai/kumo#30110). The raw
                # text stays in the worker log here and in the workflow
                # failure re-raised below, so no diagnostic info is lost.
                raw_message = root_cause_message(ex)
                log.error(
                    'Batch prediction job %s failed; internal root cause: %s',
                    job_input.id, raw_message)
                validation_response = ValidationResponse()
                validation_response.errors.append(
                    ValidationError(
                        title="Batch prediction failed",
                        message=maybe_get_public_message(ex)
                        or ("An internal error occurred while running the "
                            "batch prediction job. Please retry the job. If "
                            "the problem continues, contact Kumo support.")))
                await self._exit_cleanup(validation_response, job_input.id,
                                         home_dir, exception=ex)
            raise
        finally:
            # Finalize unconditionally so a pre-enqueue failure can't leave the
            # StateDB row non-terminal. dequeue_job runs the idempotent
            # finalize_job safety-net; the queue release is a no-op if the job
            # never enqueued.
            await dequeue_job(job_input.id)

    async def _run(
        self,
        job_input: BatchPredictionJobInput,
        ml_job_settings: MLJobSettings,
    ) -> ValidationResponse:
        self._job_input = job_input
        self._ml_job_settings = ml_job_settings
        self._resource_hint: ResourceHint | None = None
        if deploy_env().is_diskgraph:
            self._resource_hint = _prediction_resource_hint(
                job_input.config.num_parallel_workers_limit,
                _configured_max_durable_gpu_slots(),
            )
        self._data_backend_ctx = DataBackendContext(
            deploy_env(), DataBackendSettings(disable_async_ingestion=False))

        validation_response = ValidationResponse()

        # Init: create job entry in SQL DB, and persist job inputs
        await init_job_generic(job_input, ml_job_settings)

        # Queue the job.
        await enqueue_job(job_input.id)
        self.check_cancelled()

        # Initialize the S3 home_dir by creating an '__init__' file
        home_dir = get_job_dir(get_job_base_dir(), job_input.id)
        self._home_dir = home_dir
        await workflow.execute_activity(
            init_home_dir_activity, args=[home_dir],
            schedule_to_close_timeout=timedelta(hours=1))
        self.check_cancelled()

        # Mark job status as RUNNING
        await self._save_output(
            BatchPredictionJobOutput(status=JobStatus.RUNNING,
                                     log=JobEventLogEntry.starting()))

        # This batch-prediction job may reuse a separate prediction-table
        # generation ("gen") job. BP must score on the SAME graph version the
        # gen job used to build its table -- not the current graph (#29507).
        # The gen job records that version only after it finishes, so we wait
        # for it here and copy the version in, instead of reading it at
        # submission where the gen job may still be running. We only do this
        # when the version is still unset: TT-NG leaves it unset at submission
        # for this reason; the non-TT-NG path sets it at submission and skips.
        gen_pred_table_job_id = job_input.generate_pred_table_job_id
        if (gen_pred_table_job_id is not None
                and job_input.graph_version_info is None):
            gen_job_id = GeneratePredictionTableJobID(gen_pred_table_job_id)
            if not gen_job_id.is_child_of(job_input.id):
                # Wait for the gen job to finish, then read the version it used.
                await workflow.execute_activity(
                    DataBackendActivities.wait_on_workflow,
                    args=[self._data_backend_ctx, gen_job_id.wf_id],
                    task_queue=DATA_BACKEND_TASK_QUEUE,
                    retry_policy=CHILD_WF_WAITING_RETRY_POLICY,
                    schedule_to_close_timeout=timedelta(
                        hours=DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS),
                )
                self.check_cancelled()
                gen_input = await workflow.execute_activity(
                    MLBackendActivities.load_decoupled_gen_pred_table_input,
                    args=[job_input, gen_job_id],
                    schedule_to_close_timeout=timedelta(minutes=10))
                if gen_input is not None:
                    # Use the gen job's graph version for this BP job. The
                    # version step + refresh below run on it; the later wait
                    # for the gen job is then a no-op (it already finished).
                    job_input.graph_data_compute_input = (
                        gen_input.graph_data_compute_input)
                    job_input.graph_version_info = gen_input.graph_version_info
                    job_input.config.graph_data_version = (
                        gen_input.graph_version_info.graph_data_version)
                    self.check_cancelled()

        # Determine graph data version.
        if job_input.graph_version_info is None:
            # Determine & save graph data version if not specified.
            graph_version_info = await workflow.execute_activity(
                DataBackendActivities.refresh_graph_data_version,
                args=[job_input.graph_data_compute_input, job_input.id],
                task_queue=DATA_BACKEND_TASK_QUEUE,
                schedule_to_close_timeout=timedelta(
                    hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
                cancellation_type=(
                    ActivityCancellationType.WAIT_CANCELLATION_COMPLETED),
            )
            self.check_cancelled()
            # Save {job_id: graph data version} mapping explicitly into
            # job DB.  Note that today we write {job_id: table versions} to
            # data backend cache which we should get rid of.
            await workflow.execute_activity(
                JobOutputActivities.save_pred_job_graph_version, args=[
                    job_input.id,
                    graph_version_info.graph_data_version,
                ], schedule_to_close_timeout=timedelta(minutes=10))
            self.check_cancelled()
            # Set graph version info in job_input to propagate it to
            # child workflow(s).
            job_input.graph_version_info = graph_version_info

        # Step 1: Refresh Graph Data (table ingestion + col stats).
        result, graph_input = await workflow.execute_child_workflow(
            GraphRefreshWorkflow.run,
            args=[
                self._data_backend_ctx,
                job_input.graph_data_compute_input,
                job_input.graph_version_info,
                job_input.id,
                self._resource_hint,
            ],
            id=GraphRefreshWorkflow.compose_wf_id(
                job_input.graph_data_compute_input),
            execution_timeout=timedelta(
                hours=DATA_BACKEND_REFRESH_TIMEOUT_HOURS),
            task_queue=DATA_BACKEND_TASK_QUEUE,
            cancellation_type=ChildWorkflowCancellationType.TRY_CANCEL,
            parent_close_policy=ParentClosePolicy.REQUEST_CANCEL,
        )
        validation_response.merge(result.validation)
        if not validation_response.ok:
            await save_validation_response(validation_response, home_dir,
                                           job_input.id)

            job_output = BatchPredictionJobOutput.failed(
                validation=validation_response,
                workflow_end_state=WorkflowExecutionStatus.COMPLETED)
            await self._save_output(job_output)
            return validation_response

        # Update timestamp unit for graph config in job input
        updated_graph_config = update_graph_config_from_graph_input(
            job_input.predictive_query.graph, graph_input)
        job_input.predictive_query.graph = updated_graph_config

        # Step 2: Generate Prediction Table
        self.check_cancelled()
        pred_table_result = await self._generate_pred_table()

        validation_response.merge(pred_table_result.validation)
        if not validation_response.ok:
            await save_validation_response(validation_response, home_dir,
                                           job_input.id)
            job_output = BatchPredictionJobOutput.failed(
                validation=validation_response,
                workflow_end_state=WorkflowExecutionStatus.COMPLETED)
            await self._save_output(job_output)
            return validation_response

        if job_input.stop_after_table_gen:
            path = pred_table_result.pred_table_path
            log.info(f"Workflow was requested to stop after prediction table "
                     f"generation (output path {path}, so it is "
                     f"exiting.")
            # This is a SUCCESS-shaped early-return: the pred table was
            # generated and validation passed, but no prediction ran. Without
            # an explicit terminal status, finalize_job's safety-net would see
            # a non-terminal job with an OK validation response and mislabel it
            # FAILED-generic. Terminalize it as DONE/COMPLETED through the
            # canonical workflow_end_state chokepoint. results is honestly empty
            # (no predictions produced), which still satisfies the DONE
            # invariant (results present).
            job_output = BatchPredictionJobOutput(
                status=JobStatus.DONE, log=JobEventLogEntry.done(),
                results=BatchPredictionJobResults(key_cols=[]),
                workflow_end_state=WorkflowExecutionStatus.COMPLETED)
            await self._save_output(job_output)
            return validation_response

        # Step 3: Materialize Prediction Task
        self.check_cancelled()
        # parent's Databricks cluster.
        job_config_path, response = await workflow.execute_child_workflow(
            MaterializePredictionTaskWorkflow.run,
            # Serving flag is always False for batch prediction.
            args=[
                job_input,
                ml_job_settings,
                pred_table_result.pred_table_path,
                False,
                None,  # home_dir
                JobType.BATCH_PREDICTION_JOB,  # job_type
                self._resource_hint,
            ],
            id=MaterializePredictionTaskWorkflow.generate_wf_id(job_input.id),
            cancellation_type=ChildWorkflowCancellationType.TRY_CANCEL,
            parent_close_policy=ParentClosePolicy.TERMINATE)

        validation_response.merge(response)
        if not validation_response.ok:
            await save_validation_response(validation_response, home_dir,
                                           job_input.id)
            job_output = BatchPredictionJobOutput.failed(
                validation=validation_response,
                workflow_end_state=WorkflowExecutionStatus.COMPLETED)
            await self._save_output(job_output)
            return validation_response

        # Step 5: Launch Prediction on K8S
        self.check_cancelled()
        # parent's Databricks cluster.
        output = await workflow.execute_child_workflow(
            BatchPredictionJobWorkflow.run, args=[
                job_input,
                ml_job_settings,
                job_config_path,
                self._resource_hint,
            ], id=BatchPredictionJobWorkflow.generate_wf_id(job_input.id),
            cancellation_type=ChildWorkflowCancellationType.TRY_CANCEL,
            parent_close_policy=ParentClosePolicy.REQUEST_CANCEL)

        if output.status == JobStatus.DONE:
            self.check_cancelled()

            # Step 6 (may be optional): XAI aggregation
            if job_input.config.task_type not in {
                    TaskType.FORECASTING, TaskType.MULTILABEL_RANKING
            } and job_input.config.explanations:
                await workflow.execute_activity(
                    xai_aggregation_activity, args=[job_input, ml_job_settings],
                    schedule_to_close_timeout=timedelta(minutes=30))

        else:
            log.warning('Skipping BP stats activity because job failed')

        # Remove any stale job-scoped staging before detached BP stats creates
        # its own local cache under the same job namespace.
        with catch_and_log():
            await workflow.execute_activity(
                cleanup_diskgraph_staging,
                args=[[job_input.id, job_input.training_job_id]],
                schedule_to_close_timeout=timedelta(minutes=5),
            )

        # Write validation outputs to the job directory before terminalizing.
        await save_validation_response(validation_response, home_dir,
                                       job_input.id)

        if (output.status == JobStatus.DONE and PredictionArtifactType.PREDICTIONS
                in requested_prediction_artifact_types(job_input.config)):
            # Detailed BP column statistics are eventual MLOps data, not part
            # of BP success. Temporal owns this child after start is accepted;
            # parent completion and cancellation must not terminate it.
            with catch_and_log(ex_to_raise=(CancelledError, )):
                deadline_hours = DATA_BACKEND_BP_STATS_TIMEOUT_HOURS * 2
                await workflow.start_child_workflow(
                    BatchPredictionStatsWorkflow.run,
                    args=[
                        job_input,
                        ml_job_settings,
                    ],
                    id=f'bp_stats_wf-{job_input.id}',
                    execution_timeout=timedelta(hours=deadline_hours),
                    cancellation_type=ChildWorkflowCancellationType.ABANDON,
                    parent_close_policy=ParentClosePolicy.ABANDON,
                )

        if output.status == JobStatus.DONE:
            # Publish and freeze success after required parent cleanup and
            # validation, but without waiting for auxiliary MLOps statistics.
            output.log = JobEventLogEntry.done()
            output.workflow_end_state = WorkflowExecutionStatus.COMPLETED
            await self._save_output(output)
        return validation_response

    async def _generate_pred_table(self, ) -> GeneratePredTableResult:
        custom_source = self._job_input.config.custom_pred_table_source
        if custom_source is not None:
            reference = self._job_input.custom_pred_table_connector
            if reference is None:
                raise ValueError(
                    'Custom prediction source is missing its connector '
                    'execution reference')
            # On TT-NG the custom pred table needs a gen-only durable bundle
            # (build_prediction_table_artifacts_for_existing_table) beside the
            # pred table, mirroring the custom TRAIN path -- otherwise
            # materialize_prediction_task has no bundle to re-ingest and fails.
            # This single activity resolves the parquet (single-file or
            # multi-part dir), validates its schema (entity pkey present,
            # TIMESTAMP present/synthesizable, no target column), builds the
            # bundle, and persists it. Inlined (not a child workflow) because
            # -- unlike custom TRAIN, which orchestrates ingest + validate +
            # build + materialize -- custom PRED is a single gen-only step, so
            # a dedicated child workflow would add a round-trip for no
            # additional orchestration. The bundle activity no-ops on non-TT-NG,
            # whose engine consumes the Kumo-staged parquet directly. The gate
            # lives in the activity (not here) to keep the workflow deterministic
            # (no env/wheel probe in workflow code), exactly like
            # CustomTrainTableWorkflow.
            custom_pred_table_path = osp.join(self._home_dir, 'inputs',
                                              'custom_prediction_table')
            from rest.backends.ml_backend.activities import (
                stage_custom_prediction_table,
            )
            custom_pred_table_path = await workflow.execute_activity(
                stage_custom_prediction_table,
                args=[
                    reference,
                    custom_source,
                    custom_pred_table_path,
                ],
                task_queue=DATA_BACKEND_TASK_QUEUE,
                schedule_to_close_timeout=timedelta(
                    hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
            )
            assert self._job_input.graph_data_compute_input is not None
            assert self._job_input.graph_version_info is not None
            snapshot_id = DataSnapshot.GRAPH.canonicalize(
                f'{_graph_key_to_v2(self._job_input.graph_data_compute_input.graph_key)}'
                f'@{self._job_input.graph_version_info.graph_data_version}')
            await run_pooled_activity(
                snapshot_id=snapshot_id,
                activity=SnapshotActivity.GEN_CUSTOM_TT,
                activity_fn=(
                    SnapshotActivities.build_ttng_custom_pred_table_bundle),
                args=[
                    snapshot_id,
                    self._job_input,
                    self._ml_job_settings,
                    custom_pred_table_path,
                ],
                schedule_to_close_timeout=timedelta(
                    hours=DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS),
                heartbeat_timeout=HEARTBEAT_TIMEOUT_COMPUTE_ACTIVITY,
                retry_policy=COMPUTE_ACTIVITY_RETRY_POLICY,
            )
            return GeneratePredTableResult(custom_pred_table_path,
                                           ValidationResponse())

        # If generate-train-table is executed in a decoupled workflow,
        # wait for the workflow result; otherwise kick it off as child
        # workflow.
        assert self._job_input.generate_pred_table_job_id
        gen_pred_table_job_id = GeneratePredictionTableJobID(
            self._job_input.generate_pred_table_job_id)
        if not gen_pred_table_job_id.is_child_of(self._job_input.id):
            # Generate-Pred-Table job was executed in standalone
            # (decoupled) mode.
            await self._save_output(
                BatchPredictionJobOutput(
                    status=JobStatus.RUNNING,
                    log=JobEventLogEntry.waiting_pred_table(
                        gen_pred_table_job_id)))

            pred_table_result: GeneratePredTableResult = (
                await workflow.execute_activity(
                    DataBackendActivities.wait_on_workflow,
                    args=[self._data_backend_ctx, gen_pred_table_job_id.wf_id],
                    task_queue=DATA_BACKEND_TASK_QUEUE,
                    retry_policy=CHILD_WF_WAITING_RETRY_POLICY,
                    schedule_to_close_timeout=timedelta(
                        hours=DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS),
                ))

        else:
            home_dir = get_job_dir(
                get_job_base_dir(),
                self._job_input.id,
            )
            # Launch GeneratePredictionTableWorkflow as child.
            await self._save_output(
                BatchPredictionJobOutput(
                    status=JobStatus.RUNNING,
                    log=JobEventLogEntry.generating_pred_table()))
            # parent's Databricks cluster.
            pred_table_result = await workflow.execute_child_workflow(
                GeneratePredictionTableWorkflow.run,
                args=[
                    self._job_input,
                    self._ml_job_settings,
                    False,  # standalone_mode (child mode)
                    home_dir,
                    False,  # is_child_job (keep default behavior - warnings not errors)
                    self._resource_hint,
                ],
                id=gen_pred_table_job_id.wf_id,
                cancellation_type=ChildWorkflowCancellationType.TRY_CANCEL,
                parent_close_policy=ParentClosePolicy.TERMINATE)

        return pred_table_result
