# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import traceback
from datetime import timedelta
from typing import Any, Optional, Sequence, Tuple

from temporalio import workflow
from temporalio.client import WorkflowExecutionStatus
from temporalio.workflow import (
    ChildWorkflowCancellationType,
    ParentClosePolicy,
)

from io_utils import kumo_dataclass, logging
from kumo.config.graph import TableConfig
from kumo.home import HomeDirManager
from pool.activities import SnapshotActivities
from pool.dataclasses import ResourceHint, SnapshotActivity
from pool.wf_coordinator import run_pooled_activity
from rest.backends import (
    DATA_BACKEND_INGEST_TIMEOUT_HOURS,
    DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS,
    DATA_BACKEND_REFRESH_TIMEOUT_HOURS,
)
from rest.backends.data_backend.activities import (
    ingest_custom_train_table,
    validate_weighted_train_table,
)
from rest.backends.data_backend.activities.custom_train_table import (
    anchor_is_sentinel,
)
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.workflows import (
    GraphRefreshResult,
    GraphRefreshWorkflow,
)
from rest.backends.data_backend.workflows.workflow_utils import (
    start_child_data_wf,
)
from rest.backends.ml_backend.activities import (
    materialize_training_table,
    persist_job_inputs,
)
from rest.backends.ml_backend.data_util import get_job_base_dir, get_job_dir
from rest.backends.ml_backend.job_models import (
    GeneratePredTableJobOutput,
    GenerateTrainTableJobOutput,
)
from rest.backends.ml_backend.settings import MLJobSettings
from rest.backends.ml_backend.validation import (
    ValidationError,
    ValidationResponse,
)
from rest.backends.ml_backend.workflows.utils import (
    dequeue_job,
    enqueue_job,
    init_job_generic,
)
from rest.config.batch_prediction_job import BatchPredictionJobInput
from rest.config.error_handling import extract_failure_info
from rest.config.ml_backend import TrainingJobInput
from rest.config.training_job import CustomTrainingTable
from rest.constants import (
    COMPUTE_ACTIVITY_RETRY_POLICY,
    DATA_BACKEND_TASK_QUEUE,
    HEARTBEAT_TIMEOUT_COMPUTE_ACTIVITY,
    HEARTBEAT_TIMEOUT_TRAIN_TABLE_SPLIT,
    ML_WORKFLOW_TASK_QUEUE,
)
from rest.settings import _USE_DECOUPLED_SPLITS, DataBackendSettings
from service_framework.deploy_env_config import deploy_env
from statedb.config._graph_key_norm import _graph_key_to_v2
from statedb.config.common import (
    DataSnapshot,
    GeneratePredictionTableJobID,
    GenerateTrainTableJobID,
    JobEventLogEntry,
    JobFailureInfo,
    JobStatus,
    TableName,
)
from statedb.config.timestamp import now_utc
from statedb.utils.errors import maybe_get_public_message
from temporalio_commons import is_cancelled_exception

from ...common.activities.io_activities import save_validation_response
from ..activities.common import update_graph_config_from_graph_input
from ..activities.ml_backend_activities import JobOutputActivities
from ..activities.split import construct_split_workflow_input
from .custom_train_table_workflow import _resolve_entity_time_col
from .mixins import JobWorkflowCancellationMixin
from .split_workflow import SplitTrainingTableWorkflow

log = logging.getLogger(__name__)


async def _refresh_graph_snapshot(
    ctx: DataBackendContext,
    wf_id: str,
    args: Sequence[Any],
) -> Tuple[GraphRefreshResult, Any]:
    """Build, or wait for, the graph snapshot this job is about to consume.

    ``GraphRefreshWorkflow`` owns the only DiskGraph ingest a pooled consumer
    can read from, and a pinned version names a snapshot rather than
    guaranteeing one. Its wf id is shared across every job on that version, so
    this goes through ``start_child_data_wf``, which joins an already-running
    refresh and restarts one that the job which really parented it cancelled
    out from under us.
    """
    child_task = await start_child_data_wf(
        GraphRefreshWorkflow.run,
        ctx,
        wf_id=wf_id,
        args=args,
        execution_timeout=timedelta(hours=DATA_BACKEND_REFRESH_TIMEOUT_HOURS),
        task_queue=DATA_BACKEND_TASK_QUEUE,
    )
    return await child_task


def _apply_graph_version_to_table_configs(graph_config, graph_version_info):
    for table in graph_config.tables:
        if isinstance(table, TableConfig):
            table_name = TableName(table.table_name)
            if table_name in graph_version_info.table_data_versions:
                table.data_version = (
                    graph_version_info.table_data_versions[table_name])


@kumo_dataclass
class GenerateTrainTableResult:
    # Path to table data.
    train_table_path: str
    # Path to materialized TrainingTable data (split masks)
    train_table_materialization_path: str

    validation: ValidationResponse
    train_table_home_dir: Optional[str] = None
    # True when `train_table_path` is a registration's own copy of rows the
    # caller supplied (`_register_supplied_table`), rather than a table this
    # job generated. A training run reads a registered copy as it stands
    # instead of re-ingesting the caller's original source a second time, so
    # this is what tells it which one `train_table_path` is.
    registered: bool = False


# Core workflow: generate training table.
#
# It can be executed in a "standalone_mode", i.e., decoupled from the training
# execution now. In the standalone mode, it will kick off a child workflow of
# graph refresh to ensure it's done.
#
# Otherwise, it's executed in a "child mode", i.e., as child workflow of a
# parent TrainingWorkflow.  In child mode, we assume the graph refresh is
# already completed by the parent TrainingWorkflow before launching this
# GenerateTrainingTableWorkflow!
@workflow.defn
class GenerateTrainingTableWorkflow(JobWorkflowCancellationMixin):
    async def _save_job(self, job_output: GenerateTrainTableJobOutput):
        await workflow.execute_activity(
            JobOutputActivities.save_generate_train_table_job_output, args=[
                self._job_input.generate_train_table_job_id,
                job_output,
            ], schedule_to_close_timeout=timedelta(minutes=10))
        self.check_cancelled()

    @workflow.run
    async def run(
        self,
        job_input: TrainingJobInput,
        ml_job_settings: MLJobSettings,
        # True if generate-train-table job is executed as a standalone
        # job, i.e., decoupled from any training job. False if this is
        # triggered as child workflow(job) of TrainingWorkflow(Job).
        standalone_mode: bool,
        home_dir: str,
    ) -> GenerateTrainTableResult:
        # Pool integration note: see TrainingWorkflow._run for the
        # rationale behind not opening a session-level lease at this
        # outer point. The meaningful leases live in
        # DiskgraphIngestWorkflow + the leaf job workflows where the
        # canonical GraphSnapshotID is derivable.
        return await self.run_until_canceled(
            self._run(
                job_input,
                ml_job_settings,
                standalone_mode,
                home_dir,
            ))

    async def _run(
        self,
        job_input: TrainingJobInput,
        ml_job_settings: MLJobSettings,
        standalone_mode: bool,
        home_dir: str,
    ) -> GenerateTrainTableResult:
        self._job_input = job_input
        self._ml_job_settings = ml_job_settings
        self._standalone_mode = standalone_mode
        self.job_id = GenerateTrainTableJobID(
            job_input.generate_train_table_job_id)
        log.info(f"GenerateTrainingTableWorkflow._run: received "
                 f"standalone_mode={standalone_mode}, job_id={self.job_id}")

        # Init: create job entry in SQL DB, and persist job inputs
        await init_job_generic(job_input, ml_job_settings, self.job_id)

        if standalone_mode:
            assert job_input.id == self.job_id
            await enqueue_job(job_input.id)
        else:  # child mode
            assert self.job_id.is_child_of(job_input.id)

        log.info("Start training table generation")
        await self._save_job(
            GenerateTrainTableJobOutput(
                JobStatus.RUNNING,
                JobEventLogEntry.generating_training_table()))

        try:
            result = await self._generate_table()
        except Exception as ex:
            log.exception(f'GenerateTrainingTableWorkflow failed '
                          f'for job id={self.job_id}')
            if is_cancelled_exception(ex):
                job_output = GenerateTrainTableJobOutput(
                    JobStatus.CANCELLED,
                    JobEventLogEntry('Generating Training Data', now_utc(),
                                     'Generating train table is canceled.'),
                    workflow_end_state=WorkflowExecutionStatus.CANCELED,
                    failure_info=JobFailureInfo(
                        message='Generating train table is canceled.',
                        stacktrace=traceback.format_exc()))
                await self._save_job(job_output)
            else:
                validation_response = ValidationResponse(errors=[
                    ValidationError(
                        title="Unexpected error in generating training table",
                        message=maybe_get_public_message(ex) or (
                            "An internal error occurred while generating the "
                            "training table. Please retry the job. If the "
                            "problem continues, contact Kumo support."))
                ])

                # This is the case when _generate_table() failed with an unhandled
                # error which raised an exception as opposed to returning a
                # ValidationResponse
                log.debug(
                    f'Saving Validation response to: {home_dir}/{self.job_id}')

                await save_validation_response(
                    validation_response,
                    home_dir,
                    self.job_id,
                )

                job_output = GenerateTrainTableJobOutput.failed(
                    validation=validation_response,
                    exception=ex,
                    workflow_end_state=WorkflowExecutionStatus.FAILED,
                )
                await self._save_job(job_output)

            raise
        else:
            failure_info = None
            if result.validation.ok:
                status = JobStatus.DONE
                msg = 'Successfully completed generating train table.'
            else:
                status = JobStatus.FAILED

                # Persist validation failures returned without an exception.
                await save_validation_response(
                    result.validation,
                    home_dir,
                    self.job_id,
                )

                failure_info = extract_failure_info(
                    validation=result.validation)
                msg = failure_info.message

            train_table_home_dir: str = get_job_dir(
                get_job_base_dir(),
                self.job_id,
            )
            await self._save_job(
                GenerateTrainTableJobOutput(
                    status,
                    JobEventLogEntry('Generating Training Data', now_utc(),
                                     msg),
                    workflow_end_state=WorkflowExecutionStatus.COMPLETED,
                    train_table_path=result.train_table_path,
                    train_table_materialization_path=result.
                    train_table_materialization_path,
                    validation=result.validation,
                    train_table_home_dir=train_table_home_dir,
                    failure_info=failure_info,
                ))
            return result
        finally:
            if self._standalone_mode:
                # enqueue only happens in this mode
                await dequeue_job(self._job_input.id)

    async def _generate_table(
        self,
        is_child_job: Optional[bool] = False,
    ) -> GenerateTrainTableResult:
        data_backend_ctx = DataBackendContext(
            deploy_env(), DataBackendSettings(disable_async_ingestion=False))

        validation_response = ValidationResponse()

        # Refresh even for an already-pinned ``graph_version_info``: a pinned
        # version names the snapshot to read, not one that exists. Full
        # rationale on the prediction path.
        if self._standalone_mode:
            assert self._job_input.graph_data_compute_input
            graph_refresh_wf_id = GraphRefreshWorkflow.compose_wf_id(
                self._job_input.graph_data_compute_input,
                self._job_input.graph_version_info)
            log.info(
                f"GenerateTrainingTableWorkflow._generate_table: Starting "
                f"GraphRefreshWorkflow as CHILD (standalone_mode=True) with "
                f"job_id={self.job_id}, "
                f"wf_id={graph_refresh_wf_id}")
            result, g_input = await _refresh_graph_snapshot(
                data_backend_ctx,
                graph_refresh_wf_id,
                args=[
                    data_backend_ctx,
                    self._job_input.graph_data_compute_input,
                    self._job_input.graph_version_info,
                    self.job_id,
                ],
            )
            validation_response.merge(result.validation)
            # A failed refresh returns before running the ingest, so the
            # snapshot gen_tt would consume was never built. Stop here rather
            # than dispatch a consumer that can only fail.
            if not validation_response.ok:
                return GenerateTrainTableResult('', '', validation_response,
                                                '')
            graph_ver = result.graph_version_info.graph_data_version
            # Propagate graph_version_info so downstream activities
            # (e.g. diskgraph) can locate the index.
            self._job_input.graph_version_info = \
                result.graph_version_info
            # Update timestamp unit for graph config in job input,
            # when graph refresh workflow is executed.
            updated_graph_config = update_graph_config_from_graph_input(
                self._job_input.spec.predictive_query.graph, g_input)
            self._job_input.spec.predictive_query.graph = \
                updated_graph_config
            # update_graph_config_from_graph_input carries timestamp units and
            # delimiters only, never data_version.
            _apply_graph_version_to_table_configs(
                self._job_input.spec.predictive_query.graph,
                result.graph_version_info,
            )
        else:
            assert self._job_input.graph_version_info
            graph_ver = (self._job_input.graph_version_info.graph_data_version)
            _apply_graph_version_to_table_configs(
                self._job_input.spec.predictive_query.graph,
                self._job_input.graph_version_info,
            )

        self.check_cancelled()
        # Persist graph data vesion for this job in ML-jobs-DB
        await workflow.execute_activity(
            JobOutputActivities.save_generate_train_table_graph_version,
            args=[self._job_input.generate_train_table_job_id,
                  graph_ver], schedule_to_close_timeout=timedelta(minutes=10))

        # Rows the caller supplied, rather than rows this job generates.
        # Everything above still ran -- the query, the plan and the graph
        # snapshot are resolved and pinned identically, because those are
        # what a training run reads back off this job.
        custom_train_table = self._job_input.spec.custom_train_table
        if custom_train_table is not None:
            return await self._register_supplied_table(custom_train_table,
                                                       validation_response)

        # Diskgraph dispatches gen_tt via the pool to ensure snapshot
        # residency on the worker before delegating to the underlying
        # generate_training_table body.
        log.info("GenerateTrainingTableWorkflow._generate_table: Calling "
                 "generate_training_table activity")
        graph_key = (self._job_input.graph_data_compute_input.graph_key)
        snapshot_id = DataSnapshot.GRAPH.canonicalize(
            f'{_graph_key_to_v2(graph_key)}@{graph_ver}')
        # gen_tt handles both training and prediction target tables;
        # home_dir/is_child_job are unused on the training path.
        path, response, home_dir = await run_pooled_activity(
            snapshot_id=snapshot_id,
            activity=SnapshotActivity.GEN_TT,
            activity_fn=SnapshotActivities.gen_tt,
            args=[
                snapshot_id,
                self._job_input,
                self._ml_job_settings,
                None,  # home_dir — prediction-only
                False,  # is_child_job — prediction-only
            ],
            schedule_to_close_timeout=timedelta(
                hours=DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS),
            heartbeat_timeout=HEARTBEAT_TIMEOUT_COMPUTE_ACTIVITY,
            retry_policy=COMPUTE_ACTIVITY_RETRY_POLICY,
        )

        validation_response.merge(response)
        if not validation_response.ok:
            return GenerateTrainTableResult(path, '', response, '')

        # Apply decoupled splits if allowed by the environment variable
        if _USE_DECOUPLED_SPLITS:
            materialized_path, response = (await
                                           self._materialize_and_split_table(
                                               path,
                                               validation_response,
                                               home_dir,
                                           ))
            if not response.ok or materialized_path is None:
                return GenerateTrainTableResult(path, '', response, '')

        else:
            # Materialize training table (TableData attributions,
            # splits etc)
            execute_splits = True
            materialized_path, response = (await
                                           self._materialize_training_table(
                                               path, execute_splits))

            validation_response.merge(response)
            if not validation_response.ok:
                return GenerateTrainTableResult(path, '', validation_response,
                                                '')

        log.info("Finish training table generation")
        return GenerateTrainTableResult(path, materialized_path, response,
                                        home_dir)

    async def _register_supplied_table(
        self,
        custom_train_table: CustomTrainingTable,
        validation_response: ValidationResponse,
    ) -> GenerateTrainTableResult:
        """Read supplied rows into storage the platform owns.

        Nothing is compared, because there is no generated table to compare
        against, which is the point of naming a supplied one. Everything that
        does not need one is still checked, on Kumo's own copy rather than on
        the caller's source: without that check, a registered table DiskGraph
        reads no rows out of would still become DONE and train, silently, at
        every run that names it.

        What the copy buys is that the platform now holds the rows this job
        stands for, and an export can return them.

        There is no separate materialization either. A supplied table is
        split and bundled by the run that trains on it
        (``CustomTrainTableWorkflow``), exactly as it is for a table supplied
        against a generated job -- except that a *registered* table's rows
        are this copy: the run that trains against this job does not re-read
        the caller's original source, so deleting or rewriting it after
        registration does not affect training.
        """
        home_dir = get_job_dir(get_job_base_dir(), self.job_id)
        task_type = self._job_input.spec.task_type
        assert task_type is not None
        predictive_query = self._job_input.spec.predictive_query
        entity_time_col = _resolve_entity_time_col(predictive_query)

        log.info("GenerateTrainingTableWorkflow._register_supplied_table: "
                 "Calling ingest_custom_train_table activity")
        path, response = await workflow.execute_activity(
            ingest_custom_train_table,
            args=[
                custom_train_table,
                HomeDirManager(home_dir).custom_train_table_path,
                deploy_env(),
                self._job_input.id,
                task_type,
                predictive_query.query_type,
                bool(predictive_query.list_distinct_target_is_array),
                entity_time_col,
            ],
            task_queue=DATA_BACKEND_TASK_QUEUE,
            schedule_to_close_timeout=timedelta(
                hours=DATA_BACKEND_INGEST_TIMEOUT_HOURS),
            retry_policy=COMPUTE_ACTIVITY_RETRY_POLICY,
        )
        validation_response.merge(response)
        if not validation_response.ok:
            return GenerateTrainTableResult('', '', validation_response, '')

        validation_response.merge(await workflow.execute_activity(
            validate_weighted_train_table,
            args=[
                None,
                path,
                custom_train_table.weight_col_name,
                deploy_env(),
                self._job_input.id,
                task_type,
                custom_train_table.split_col_name,
                not anchor_is_sentinel(
                    task_type,
                    predictive_query.query_type,
                    bool(predictive_query.list_distinct_target_is_array),
                    entity_time_col,
                ),
            ],
            task_queue=DATA_BACKEND_TASK_QUEUE,
            schedule_to_close_timeout=timedelta(
                hours=DATA_BACKEND_INGEST_TIMEOUT_HOURS),
            retry_policy=COMPUTE_ACTIVITY_RETRY_POLICY,
        ))
        if not validation_response.ok:
            return GenerateTrainTableResult('', '', validation_response, '')

        log.info("Finish registering supplied training table")
        return GenerateTrainTableResult(path, path, validation_response,
                                        home_dir, registered=True)

    async def _materialize_training_table(
        self,
        path: str,
        execute_split: bool,
    ) -> Tuple[str, ValidationResponse]:
        """Validate and materialize the durable TT-NG bundle."""
        return await workflow.execute_activity(
            materialize_training_table,
            args=[
                self._job_input,
                self._ml_job_settings,
                path,
                execute_split,
                self._job_input.weight_col_name,
                False,
            ],
            schedule_to_close_timeout=timedelta(
                hours=DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS),
            heartbeat_timeout=HEARTBEAT_TIMEOUT_TRAIN_TABLE_SPLIT,
            retry_policy=COMPUTE_ACTIVITY_RETRY_POLICY,
            task_queue=DATA_BACKEND_TASK_QUEUE,
        )

    async def _materialize_and_split_table(
        self,
        path: str,
        validation_response: ValidationResponse,
        home_dir: str,
    ) -> Tuple[Optional[str], ValidationResponse]:
        """
        Materialize the training table and apply the split in its own workflow.
        This will later be refactored to allow independent execution of the
        split workflow.

        Args:
            path (str): Path to the training table.
            validation_response (ValidationResponse): Validation response.
            home_dir (str): Home directory (where all job info is stored).

        Returns:
            Tuple[Optional[str], ValidationResponse]: Tuple of materialized
                path and validation response.
        """
        # First validate and materialize the base TrainingTable
        execute_split = False
        materialized_train_table_data_path, response = (
            await self._materialize_training_table(path, execute_split))

        validation_response.merge(response)
        if not validation_response.ok:
            return None, validation_response

        # Then apply the splits
        training_table_generation_plan = (
            self._job_input.spec.advanced_options.training_table_generation)
        graph_key = None
        graph_data_version = None
        if self._job_input.graph_data_compute_input is not None:
            graph_key = self._job_input.graph_data_compute_input.graph_key
        if self._job_input.graph_version_info is not None:
            graph_data_version = (
                self._job_input.graph_version_info.graph_data_version)
        split_wf_input, split_wf_id = construct_split_workflow_input(
            self._job_input.generate_train_table_job_id,
            training_table_generation_plan,
            materialized_train_table_data_path,
            self._job_input.spec.predictive_query.name,
            self._job_input.spec,
            self._job_input.graph_data_compute_input.connectors,
            graph_key=graph_key,
            graph_data_version=graph_data_version,
            diskgraph_manifest_path=f'{path}.manifest.json',
        )

        result = await workflow.execute_child_workflow(
            SplitTrainingTableWorkflow.run,
            args=[
                split_wf_input,
                self._ml_job_settings,
                home_dir,
                False,  # standalone_mode (called indep. of a training job)
            ],
            id=split_wf_id,
            execution_timeout=timedelta(
                hours=DATA_BACKEND_REFRESH_TIMEOUT_HOURS),
            task_queue=ML_WORKFLOW_TASK_QUEUE,
            cancellation_type=ChildWorkflowCancellationType.TRY_CANCEL,
            parent_close_policy=ParentClosePolicy.REQUEST_CANCEL,
        )
        validation_response.merge(result.validation)
        if not validation_response.ok or result.materialized_path is None:
            return None, validation_response

        # Set the new materialized path
        materialized_path = result.materialized_path
        return materialized_path, validation_response


@kumo_dataclass
class GeneratePredTableResult:
    pred_table_path: str
    validation: ValidationResponse


# Core workflow: generate prediction table
@workflow.defn
class GeneratePredictionTableWorkflow(JobWorkflowCancellationMixin):
    async def _save_job(self, job_output: GeneratePredTableJobOutput):
        await workflow.execute_activity(
            JobOutputActivities.save_generate_pred_table_job_output, args=[
                self._job_input.generate_pred_table_job_id,
                job_output,
            ], schedule_to_close_timeout=timedelta(minutes=10))
        self.check_cancelled()

    async def _generate_table(
        self,
        is_child_job: Optional[bool] = False,
    ) -> GeneratePredTableResult:
        data_backend_ctx = DataBackendContext(
            deploy_env(), DataBackendSettings(disable_async_ingestion=False))
        validation_response = ValidationResponse()
        # Refresh even when ``graph_version_info`` is already pinned: a pinned
        # version names the snapshot to read, not one that exists. A standalone
        # job has no parent that built it — the decoupled
        # BatchPredictionParentWorkflow only waits for this job and refreshes
        # afterwards — and the pooled gen_tt consumer can only report the miss,
        # never build.
        if self._standalone_mode:
            graph_refresh_wf_id = GraphRefreshWorkflow.compose_wf_id(
                self._job_input.graph_data_compute_input,
                self._job_input.graph_version_info)
            log.info(
                f"GeneratePredictionTableWorkflow._generate_table: Starting "
                f"GraphRefreshWorkflow as CHILD (standalone_mode=True) with "
                f"job_id={self._job_input.id}, "
                f"wf_id={graph_refresh_wf_id}")
            result, g_input = await _refresh_graph_snapshot(
                data_backend_ctx,
                graph_refresh_wf_id,
                args=[
                    data_backend_ctx,
                    self._job_input.graph_data_compute_input,
                    self._job_input.graph_version_info,
                    self._job_input.id,
                    getattr(self, '_resource_hint', None),
                ],
            )
            validation_response.merge(result.validation)
            # A failed refresh returns before running the ingest, so the
            # snapshot gen_tt would consume was never built. Stop here rather
            # than dispatch a consumer that can only fail.
            if not validation_response.ok:
                return GeneratePredTableResult('', validation_response)
            graph_ver = result.graph_version_info.graph_data_version
            # Propagate graph_version_info to job_input so downstream
            # activities (e.g., get_diskgraph_index_path_for_job) can access it.
            self._job_input.graph_version_info = result.graph_version_info
            # Update timestamp unit for graph config in job input,
            # when graph refresh workflow is executed.
            updated_graph_config = update_graph_config_from_graph_input(
                self._job_input.predictive_query.graph, g_input)
            self._job_input.predictive_query.graph = updated_graph_config
            # An unpinned job's version is only resolved by the refresh, so the
            # job input file written at submission time is stale. Re-apply the
            # resolved version to the graph config so the rest of the job uses
            # the versions it actually ran on.
            _apply_graph_version_to_table_configs(
                self._job_input.predictive_query.graph,
                result.graph_version_info,
            )
            self._job_input.config.graph_data_version = graph_ver
            await workflow.execute_activity(
                persist_job_inputs,
                args=[self._job_input, self._ml_job_settings],
                schedule_to_close_timeout=timedelta(minutes=10),
            )
        else:
            # Child mode: BatchPredictionParentWorkflow already ran the refresh
            # (and its ingest) as Step 1.
            assert self._job_input.graph_version_info
            graph_ver = (self._job_input.graph_version_info.graph_data_version)
            _apply_graph_version_to_table_configs(
                self._job_input.predictive_query.graph,
                self._job_input.graph_version_info,
            )
            self._job_input.config.graph_data_version = graph_ver

        # Persist graph data vesion for this job in ML-jobs-DB
        await workflow.execute_activity(
            JobOutputActivities.save_generate_pred_table_graph_version,
            args=[self._job_input.generate_pred_table_job_id,
                  graph_ver], schedule_to_close_timeout=timedelta(minutes=10))
        self.check_cancelled()

        log.info("GeneratePredictionTableWorkflow._generate_table: Calling "
                 "generate_prediction_table activity")
        graph_key = self._job_input.graph_data_compute_input.graph_key
        snapshot_id = DataSnapshot.GRAPH.canonicalize(
            f'{_graph_key_to_v2(graph_key)}@{graph_ver}')
        # gen_tt dispatches to the prediction-table body based on
        # job_input type.
        path, response, _ = await run_pooled_activity(
            snapshot_id=snapshot_id,
            activity=SnapshotActivity.GEN_TT,
            activity_fn=SnapshotActivities.gen_tt,
            args=[
                snapshot_id,
                self._job_input,
                self._ml_job_settings,
                self._home_dir,
                is_child_job,
            ],
            schedule_to_close_timeout=timedelta(
                hours=DATA_BACKEND_MATERIALIZE_TIMEOUT_HOURS),
            heartbeat_timeout=HEARTBEAT_TIMEOUT_COMPUTE_ACTIVITY,
            retry_policy=COMPUTE_ACTIVITY_RETRY_POLICY,
            resource_hint=getattr(self, '_resource_hint', None),
        )

        log.info("Finish prediction table generation")
        return GeneratePredTableResult(path, response)

    @workflow.run
    async def run(
        self,
        job_input: BatchPredictionJobInput,
        ml_job_settings: MLJobSettings,
        # True if generate-pred-table job is executed as a standalone
        # job, i.e., decoupled from any training job. False if this is
        # triggered as child workflow(job) of BatchPredictionWorkflow(Job).
        standalone_mode: bool,
        home_dir: str,
        is_child_job: Optional[bool] = False,
        resource_hint: Optional[ResourceHint] = None,
    ) -> GeneratePredTableResult:
        # Pool integration note: see TrainingWorkflow._run for the
        # rationale behind not opening a session-level lease at this
        # outer point.
        return await self.run_until_canceled(
            self._run(job_input, ml_job_settings, standalone_mode, home_dir,
                      is_child_job, resource_hint))

    async def _run(
        self,
        job_input: BatchPredictionJobInput,
        ml_job_settings: MLJobSettings,
        standalone_mode: bool,
        home_dir: str,
        is_child_job: Optional[bool] = False,
        resource_hint: Optional[ResourceHint] = None,
    ) -> GeneratePredTableResult:
        self._job_input = job_input
        self._ml_job_settings = ml_job_settings
        self._standalone_mode = standalone_mode
        self._home_dir = home_dir
        self._resource_hint = resource_hint
        assert job_input.generate_pred_table_job_id
        job_id = GeneratePredictionTableJobID(
            job_input.generate_pred_table_job_id)
        log.info(f"GeneratePredictionTableWorkflow._run: received "
                 f"standalone_mode={standalone_mode}, job_id={job_id}")

        # Init: create job entry in SQL DB, and persist job inputs
        await init_job_generic(job_input, ml_job_settings, job_id)

        if standalone_mode:
            assert job_input.id == job_id
            await enqueue_job(job_input.id)
        else:  # child mode
            assert job_id.is_child_of(job_input.id)

        log.info("Start prediction table generation")
        await self._save_job(
            GeneratePredTableJobOutput(JobStatus.RUNNING,
                                       JobEventLogEntry.generating_pred_table())
        )

        try:
            result = await self._generate_table(is_child_job)
        except Exception as ex:
            log.exception(
                f'GeneratePredictionTableWorkflow failed for job id={job_id}')
            if is_cancelled_exception(ex):
                job_output = GeneratePredTableJobOutput(
                    JobStatus.CANCELLED,
                    JobEventLogEntry('Generating Prediction Data', now_utc(),
                                     'Generating pred table is canceled.'),
                    workflow_end_state=WorkflowExecutionStatus.CANCELED,
                    failure_info=JobFailureInfo(
                        message='Generating prediction table is canceled.',
                        stacktrace=traceback.format_exc()))
            else:
                # We only save a ValidationResponse if it is a failure other
                # than job cancelled. Extract user-friendly message from
                # PublicException if available.
                validation_response = ValidationResponse(errors=[
                    ValidationError(
                        title="Uncaught error in generating prediction table",
                        message=maybe_get_public_message(ex) or (
                            "An internal error occurred while generating the "
                            "prediction table. Please retry the job. If the "
                            "problem continues, contact Kumo support."),
                    )
                ])
                await save_validation_response(
                    validation_response,
                    home_dir,
                    job_id,
                )
                job_output = GeneratePredTableJobOutput.failed(
                    validation=validation_response, exception=ex,
                    workflow_end_state=WorkflowExecutionStatus.FAILED)
            await self._save_job(job_output)
            raise
        else:
            failure_info = None
            if result.validation.ok:
                status = JobStatus.DONE
                msg = 'Successfully completed generating prediction table.'
            else:
                status = JobStatus.FAILED

                await save_validation_response(
                    result.validation,
                    home_dir,
                    job_id,
                )

                failure_info = extract_failure_info(
                    validation=result.validation)
                msg = failure_info.message

            await self._save_job(
                GeneratePredTableJobOutput(
                    status,
                    JobEventLogEntry('Generating Prediction Data', now_utc(),
                                     msg),
                    workflow_end_state=WorkflowExecutionStatus.COMPLETED,
                    pred_table_path=result.pred_table_path,
                    validation=result.validation,
                    failure_info=failure_info,
                ))
            return result
        finally:
            if self._standalone_mode:
                # enqueue only happens in this mode
                await dequeue_job(self._job_input.id)
