# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from typing import Optional

from temporalio import workflow
from temporalio.client import WorkflowExecutionStatus
from temporalio.workflow import (
    ChildWorkflowCancellationType,
    ParentClosePolicy,
)

from io_utils import logging
from io_utils.exceptions import catch_and_log
from kumo.home import HomeDirManager
from rest.backends import (
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
    copy_train_table_stats,
)
from rest.constants import (
    CHILD_WF_WAITING_RETRY_POLICY,
    DATA_BACKEND_TASK_QUEUE,
)

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
from rest.backends.ml_backend.workflows.mixins import (
    JobWorkflowCancellationMixin,
)
from rest.backends.ml_backend.workflows.utils import (
    dequeue_job,
    enqueue_job,
    init_job_generic,
)
from rest.config.error_handling import root_cause_message
from rest.config.graph_v2 import GraphDataComputeInput
from rest.config.ml_backend import TrainingJobInput
from rest.config.training_job import CustomTrainingTable, TrainingJobOutput
from rest.settings import _USE_DECOUPLED_SPLITS
from statedb.config.common import (
    GenerateTrainTableJobID,
    JobEventLogEntry,
    JobStatus,
    TrainingJobID,
)
from statedb.utils.errors import maybe_get_public_message
from temporalio_commons import is_cancelled_exception

from .custom_train_table_workflow import CustomTrainTableWorkflow
from .generate_table_workflows import (
    GenerateTrainingTableWorkflow,
    GenerateTrainTableResult,
)

log = logging.getLogger(__name__)


class BaseTrainingWorkflow(JobWorkflowCancellationMixin):
    async def run(
        self,
        job_input: TrainingJobInput,
        ml_job_settings: MLJobSettings,
    ) -> ValidationResponse:
        """Runs the complete training workflow.

        This method orchestrates the full training pipeline, which includes:
        1. Initializing the training job (DB entry, job queuing, home directory setup)
        2. Refreshing graph data (table ingestion, column stats, edge stats)
        3. Generating the training table (decoupled or as child workflow)
        4. Materializing the training task
        5. Triggering model training on K8S

        Subclasses should implement the `_run` method to perform the actual workflow
        steps. Subclasses may add additional steps such as:
        - XAI (Explainable AI) aggregation

        Args:
            job_input: The training job input configuration
            ml_job_settings: ML backend job settings including deployment environment

        Returns:
            ValidationResponse: The validation response containing any errors or warnings
            encountered during the workflow execution
        """
        self.job_input = job_input
        self.ml_job_settings = ml_job_settings
        try:
            return await self.run_until_canceled(
                self._run(job_input, ml_job_settings))
        except BaseException as ex:
            # Record the real error before re-raising: if _run raises before its
            # child workflow terminalizes, finalize_job would otherwise overwrite
            # it with the generic "did not report a terminal status" message.
            # Cancellation and continue-as-new are control-flow signals with
            # their own handling, not failures -- re-raise untouched.
            if is_cancelled_exception(ex) or isinstance(
                    ex, workflow.ContinueAsNewError):
                raise
            # catch_and_log so a terminalize failure never swallows the original.
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
                log.error('Training job %s failed; internal root cause: %s',
                          job_input.id, raw_message)
                validation_response = ValidationResponse()
                validation_response.errors.append(
                    ValidationError(
                        title="Training failed",
                        message=maybe_get_public_message(ex) or (
                            "An internal error occurred while running the "
                            "training job. Please retry the job. If the "
                            "problem continues, contact Kumo support.")))
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
        job_input: TrainingJobInput,
        ml_job_settings: MLJobSettings,
    ) -> ValidationResponse:
        raise NotImplementedError("Subclasses must implement this method")

    async def _save_output(self, job_output: TrainingJobOutput):
        # Mark job status as QUEUED
        await workflow.execute_activity(
            JobOutputActivities.save_training_job_output,
            args=[self.job_input.id, job_output],
            schedule_to_close_timeout=timedelta(minutes=10),
        )
        self.check_cancelled()

    async def _initialize_train_job(self, ) -> str:
        """Initialize training job: create DB entry, queue job, and return home_dir."""
        # Init: create job entry in SQL DB, and persist job inputs
        await init_job_generic(self.job_input, self.ml_job_settings)

        # Queue the job.
        await enqueue_job(self.job_input.id)
        self.check_cancelled()

        # Initialize the home_dir by creating an '__init__' file
        home_dir: str = get_job_dir(get_job_base_dir(), self.job_input.id)
        return home_dir

    async def _refresh_graph(
        self,
        graph_compute_input: GraphDataComputeInput,
        validation_response: ValidationResponse,
        home_dir: str,
        data_backend_ctx: DataBackendContext,
    ) -> Optional[GraphDataComputeInput]:
        """Refresh graph data: determine version, launch refresh workflow,
        and update config."""

        if self.job_input.graph_version_info is None:
            # Determine & save graph data version if not specified.
            graph_version_info = await workflow.execute_activity(
                DataBackendActivities.refresh_graph_data_version,
                args=[graph_compute_input, self.job_input.id],
                task_queue=DATA_BACKEND_TASK_QUEUE,
                schedule_to_close_timeout=timedelta(
                    hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
            )
            self.check_cancelled()
            # Save {job_id: graph data version} mapping explicitly into
            # job DB.  Note that today we write {job_id: table versions} to
            # data backend cache which we should get rid of.
            await workflow.execute_activity(
                JobOutputActivities.save_training_job_graph_version,
                args=[
                    self.job_input.id,
                    graph_version_info.graph_data_version,
                ],
                schedule_to_close_timeout=timedelta(minutes=10),
            )
            self.check_cancelled()
            # Set graph version info in job_input to propagate it to
            # child workflow(s).
            self.job_input.graph_version_info = graph_version_info
        else:
            graph_version_info = self.job_input.graph_version_info

        # Launch GraphRefreshWorkflow on this graph data version.
        log.info(
            f"BaseTrainingWorkflow._refresh_graph: Starting "
            f"GraphRefreshWorkflow as CHILD with job_id={self.job_input.id}")
        result, graph_input = await workflow.execute_child_workflow(
            GraphRefreshWorkflow.run,
            args=[
                data_backend_ctx,
                graph_compute_input,
                graph_version_info,
                self.job_input.id,
            ],
            execution_timeout=timedelta(
                hours=DATA_BACKEND_REFRESH_TIMEOUT_HOURS),
            task_queue=DATA_BACKEND_TASK_QUEUE,
            cancellation_type=ChildWorkflowCancellationType.TRY_CANCEL,
            parent_close_policy=ParentClosePolicy.REQUEST_CANCEL,
        )
        self.check_cancelled()
        validation_response.merge(result.validation)
        if not validation_response.ok:
            await save_validation_response(validation_response, home_dir,
                                           self.job_input.id)

            job_output = TrainingJobOutput.failed(
                validation=validation_response,
                workflow_end_state=WorkflowExecutionStatus.COMPLETED)
            await self._save_output(job_output)
            return None

        # Update timestamp unit for graph config in job input
        updated_graph_config = update_graph_config_from_graph_input(
            self.job_input.spec.predictive_query.graph, graph_input)
        self.job_input.spec.predictive_query.graph = updated_graph_config
        return graph_input

    async def _maybe_generate_train_table(
        self,
        home_dir: str,
        data_backend_ctx: DataBackendContext,
        validation_response: ValidationResponse,
    ) -> Optional[GenerateTrainTableResult]:
        """Generate or wait for train table workflow result."""
        gen_train_table_job_id = GenerateTrainTableJobID(
            self.job_input.generate_train_table_job_id)
        custom_train_table_spec = self.job_input.spec.custom_train_table

        # If generate-train-table is executed in a decoupled workflow,
        # wait for the workflow result; otherwise kick it off as child
        # workflow.
        if not gen_train_table_job_id.is_child_of(self.job_input.id):
            await self._save_output(
                TrainingJobOutput(
                    status=JobStatus.RUNNING,
                    log=JobEventLogEntry.waiting_training_table(
                        gen_train_table_job_id)))
            # TODO(siyang): We'll not be able to fetch workflow result
            # beyound 30d retention period. This might be good enough
            # since we ususally don't want to train on a very old train
            # table anyway.  We should add error handling here if the
            # workflow is not found (has expired)
            train_table_result: GenerateTrainTableResult = (
                await workflow.execute_activity(
                    DataBackendActivities.wait_on_workflow,
                    args=[data_backend_ctx, gen_train_table_job_id.wf_id],
                    task_queue=DATA_BACKEND_TASK_QUEUE,
                    retry_policy=CHILD_WF_WAITING_RETRY_POLICY,
                    schedule_to_close_timeout=timedelta(
                        hours=DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS),
                ))
            if train_table_result.validation.ok:
                # Train table stats were saved under the id of
                # generate-train-table job, we need to copy it over to the
                # dir indexed by current training job id, as expected by
                # downstream materialize_train_task.
                await workflow.execute_activity(
                    copy_train_table_stats,
                    args=[
                        self.job_input.generate_train_table_job_id,
                        self.job_input.id
                    ],
                    schedule_to_close_timeout=timedelta(minutes=30),
                )
            # Update train table paths if custom train table is specified
            if custom_train_table_spec is not None:
                assert train_table_result.validation.ok
                home_dir_mgr = HomeDirManager(home_dir)
                custom_train_table_path =\
                    home_dir_mgr.custom_train_table_path
                custom_tt_wf_id = CustomTrainTableWorkflow.generate_wf_id(
                    self.job_input.id)
                log.info(
                    f"BaseTrainingWorkflow._maybe_generate_train_table: "
                    f"Starting CustomTrainTableWorkflow as CHILD with "
                    f"wf_id={custom_tt_wf_id}")
                custom_tt_result = await workflow.execute_child_workflow(
                    CustomTrainTableWorkflow.run,
                    args=[
                        custom_train_table_spec,
                        train_table_result.train_table_path,
                        custom_train_table_path,
                        self.job_input,
                        self.ml_job_settings,
                        train_table_result.registered,
                    ],
                    id=custom_tt_wf_id,
                    cancellation_type=ChildWorkflowCancellationType.TRY_CANCEL,
                    parent_close_policy=ParentClosePolicy.TERMINATE,
                )
                validation_response.merge(custom_tt_result.validation)
                if not validation_response.ok:
                    return None
                self.check_cancelled()
                train_table_result.train_table_path = (
                    custom_tt_result.train_table_path)
                train_table_result.train_table_materialization_path = (
                    custom_tt_result.train_table_materialization_path)

        else:
            # Update the job status to 'generating training table'
            await self._save_output(
                TrainingJobOutput(
                    status=JobStatus.RUNNING,
                    log=JobEventLogEntry.generating_training_table()))
            # parent's Databricks cluster.
            log.info(
                f"BaseTrainingWorkflow._maybe_generate_train_table: Starting "
                f"GenerateTrainingTableWorkflow as CHILD with "
                f"wf_id={gen_train_table_job_id.wf_id}")
            train_table_result = (
                await workflow.execute_child_workflow(
                    GenerateTrainingTableWorkflow.run,
                    args=[
                        self.job_input,
                        self.ml_job_settings,
                        False,  # run gen-train-table as child
                        home_dir,
                    ],
                    id=gen_train_table_job_id.wf_id,
                    cancellation_type=ChildWorkflowCancellationType.TRY_CANCEL,
                    parent_close_policy=ParentClosePolicy.TERMINATE,
                ))

        validation_response.merge(train_table_result.validation)
        if not validation_response.ok:
            return None

        return train_table_result

    async def _exit_cleanup(
        self,
        validation_response: ValidationResponse,
        job_id: TrainingJobID,
        home_dir: str,
        exception: Optional[BaseException] = None,
    ) -> ValidationResponse:
        r""" Terminalize the failed training job at the point of failure, then
        save the validation response and return it.

        This mirrors the batch-prediction parent workflow's validation-failure
        early-returns: rather than relying on ``finalize_job``'s safety-net to
        infer a terminal status, we write a terminal FAILED output through the
        canonical ``workflow_end_state`` chokepoint so the real validation error
        is preserved and training and batch-prediction share one terminalization
        contract.  Job dequeuing is handled by the caller's try/finally block.

        ``exception`` (present only on the crash path, not on validation-failure
        early-returns) populates ``failure_info.stacktrace``, which would
        otherwise be empty for every workflow-crash failure; traceback
        formatting is deterministic, so this stays workflow-safe.
        """
        await save_validation_response(validation_response, home_dir, job_id)
        job_output = TrainingJobOutput.failed(
            exception=exception, validation=validation_response,
            workflow_end_state=WorkflowExecutionStatus.COMPLETED)
        await self._save_output(job_output)
        return validation_response

    async def _validate_custom_train_table(
        self,
        train_table_job_id: GenerateTrainTableJobID,
        training_job_id: TrainingJobID,
        custom_train_table: Optional[CustomTrainingTable] = None,
    ) -> ValidationResponse:
        r""" If `custom_train_table` is provided validates that
        1. the train table workflow is a decoupled workflow.
        2. decoupled splits are not enabled.
        """
        validation_response = ValidationResponse()
        if custom_train_table is not None:
            if train_table_job_id.is_child_of(training_job_id):
                validation_response.errors.append(
                    ValidationError(
                        title="Custom training table not supported",
                        message="Custom training table is not supported when "
                        " the generate-train-table job is a coupled workflow.",
                    ), )
            if _USE_DECOUPLED_SPLITS:
                validation_response.errors.append(
                    ValidationError(
                        title="Custom train table not supported"
                        " with decoupled splits",
                        message="Custom training table is not supported when "
                        " `USE_DECOUPLED_SPLITS` env variable is enabled. "
                        "Contact Kumo for support."))
        return validation_response
