# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO(Kiriti, Manan): Surface any errors in Table ingestion workflow
# through validation framework and ensure these errors are shown in the UI.
from datetime import timedelta
from typing import List, Optional

from temporalio import workflow
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import ActivityError, WorkflowAlreadyStartedError
from temporalio.workflow import ChildWorkflowCancellationType

from io_utils import logging
from io_utils.dataclasses import kumo_dataclass
from kumo.data.backend_cache import AsyncDataTaskStatus, TableDataTaskResult
from kumo.data.common import TableRefreshStage
from kumo.validation.validation import ValidationResponse
from rest.backends import (
    DATA_BACKEND_INGEST_TIMEOUT_HOURS,
    DATA_BACKEND_IO_TIMEOUT_HOURS,
)
from rest.backends.data_backend.activities import DataBackendActivities
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.utils import compose_table_ingest_wf_id
from rest.backends.data_backend.workflows.mixins import TableChildWorkflowMixin
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.table_v2 import TableDefinition
from rest.constants import (
    CHILD_WF_WAITING_RETRY_POLICY,
    HEARTBEAT_TIMEOUT_ASYNC_ACTIVITY,
    VALIDATE_STATS_RETRY_POLICY,
)
from statedb.config.common import TableName
from statedb.config.table import TableDataVersion
from statedb.config.timestamp import now_utc
from temporalio_commons.error_utils import (
    extract_public_exception,
    is_cancelled_exception,
)

from .compute_workflows import TableIngestionWorkflow

log = logging.getLogger(__name__)

_INTERNAL_COL_STATS_FAILURE_MESSAGE = (
    'Kumo encountered an internal error while computing column statistics. '
    'Please retry. If the problem continues, contact Kumo support.')


@kumo_dataclass
class TableRefreshWorkflowInput:
    connector_spec: ConnectorExecutionRef
    table: TableDefinition
    data_version: TableDataVersion
    job_id: Optional[str] = None
    base_tables_args_list: Optional[List[TableDefinition]] = None
    base_tables_connector_spec: Optional[ConnectorExecutionRef] = None
    alias: Optional[TableName] = None
    defer_col_stats_validation: bool = False
    # Set by GraphRefreshWorkflow, which infers units once for all tables;
    # default keeps a standalone refresh inferring on its own.
    timestamp_units_inferred_by_parent: bool = False
    # Set by GraphRefreshWorkflow: every table of a graph reads in one
    # process, so each read divides the machine by however many can run at
    # once. Unset outside a graph refresh, where the read owns the process.
    graph_table_count: Optional[int] = None


@workflow.defn
class TableRefreshWorkflow(TableChildWorkflowMixin):
    """
    Orchestration workflow for end-to-end refresh of a given table at a given
    data version, including ingesting source table data, computing column
    stats in multiple steps.

    Clients who started TableRefreshWorkflow may not necessarily wait for its
    full completion, instead they can simply wait on partial completion
    of a specific `TableRefreshStage`.
    """
    @staticmethod
    def compose_wf_id(table: TableDefinition,
                      data_version: TableDataVersion) -> str:
        return f'refresh-{table.wf_key()}@ver={data_version}'

    @workflow.run
    async def run(
        self,
        ctx: DataBackendContext,
        input: TableRefreshWorkflowInput,
    ) -> TableDataTaskResult:
        """
        Args:
            input: Immutable typed workflow input. Its optional job ID
                correlates refresh state and child workflows; base-table
                fields describe legacy view refreshes.
        """
        self._ctx = ctx
        self._connector_spec = input.connector_spec
        self._table = input.table
        self._data_version = input.data_version
        self._job_id = input.job_id
        self._defer_col_stats_validation = input.defer_col_stats_validation
        self._base_tables_args_list = input.base_tables_args_list
        self._base_tables_connector = input.base_tables_connector_spec
        self._alias = input.alias
        self._graph_table_count = input.graph_table_count
        log.info(f"TableRefreshWorkflow.run() received job_id={input.job_id}, "
                 f"for table={input.table.get_source_table_name()}")

        parent = workflow.info().parent
        if parent:  # Parent workflow automatically register for notification.
            await self.register_watcher(parent.workflow_id, parent.run_id)

        if self._job_id is None:
            self._job_id = self._compose_job_id(input.table, input.data_version)

        refresh_status = TableDataTaskResult(
            status=AsyncDataTaskStatus.NOT_STARTED,
            validation=ValidationResponse(),
            stage=TableRefreshStage.INGEST_TABLE,
            data_version=self._data_version,
            created_at=now_utc(),
            updated_at=now_utc(),
        )
        await self._save_result(refresh_status, notify_watcher=False)

        # Skip when graph refresh already inferred units for all tables.
        if not input.timestamp_units_inferred_by_parent:
            await self._update_table_timestamp_units()

        # Step 1: table ingestion. In diskgraph mode, graph-level
        # orchestration is responsible for colstats stages and final
        # completion artifacts.
        refresh_status.status = AsyncDataTaskStatus.IN_PROGRESS
        await self._save_result(refresh_status, notify_watcher=False)

        ingest_result = await self._run_ingestion()
        ingest_result.created_at = now_utc()
        assert ingest_result.status.completed
        ingest_result.stage = TableRefreshStage.INGEST_TABLE

        # Post-ingest validation (IngestionValidator): empty-table error,
        # small-table warning, and per-table size-cap check. The ingest
        # activity itself skips these, so dispatch them here. Forward
        # ``existing_errors`` to suppress duplicate empty-table errors when
        # the ingest activity already reported one, and forward
        # ``connector_spec`` so the validator skips the local-path
        # ``is_empty`` sniff for deferred-download connectors, whose Kumo
        # materialize path isn't populated until graph-level download.
        if ingest_result.ok:
            ingest_validation = await workflow.execute_activity(
                DataBackendActivities.validate_table_ingest,
                args=[
                    self._table,
                    self._data_version,
                    self._ctx._config,
                    list(ingest_result.validation.errors),
                    self._connector_spec,
                ],
                retry_policy=VALIDATE_STATS_RETRY_POLICY,
                schedule_to_close_timeout=timedelta(
                    hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
            )
            ingest_result.validation.merge(ingest_validation)

        # Standalone refresh owns its col-stats lifecycle. Graph refresh passes
        # ``defer_col_stats_validation`` and computes/saves stats at graph
        # scope after all child table ingests complete.
        if ingest_result.ok and not self._defer_col_stats_validation:
            try:
                col_stats_written = await workflow.execute_activity(
                    DataBackendActivities.compute_standalone_table_col_stats,
                    args=[
                        self._connector_spec,
                        self._table,
                        self._data_version,
                    ],
                    retry_policy=VALIDATE_STATS_RETRY_POLICY,
                    schedule_to_close_timeout=timedelta(
                        hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
                )
            except ActivityError as error:
                if is_cancelled_exception(error):
                    raise
                public_error = extract_public_exception(error)
                ingest_result.status = AsyncDataTaskStatus.FAILED
                ingest_result.stage = TableRefreshStage.COMPUTE_MIN_COL_STATS
                ingest_result.error = (public_error.get_user_message()
                                       if public_error is not None else
                                       _INTERNAL_COL_STATS_FAILURE_MESSAGE)
                ingest_result.updated_at = now_utc()
                await workflow.execute_activity(
                    DataBackendActivities.save_colstat_task_result,
                    args=[self._table, self._data_version, ingest_result],
                    schedule_to_close_timeout=timedelta(
                        hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
                )
                await self._save_result(ingest_result)
                return ingest_result

            if col_stats_written:
                col_stats_validation = await workflow.execute_activity(
                    DataBackendActivities.validate_table_col_stats,
                    args=[
                        self._connector_spec.connector_id,
                        self._table,
                        self._data_version,
                    ],
                    retry_policy=VALIDATE_STATS_RETRY_POLICY,
                    schedule_to_close_timeout=timedelta(
                        hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
                )
                ingest_result.validation.merge(col_stats_validation)

                if ingest_result.ok:
                    await self._save_colstat_task_result(
                        TableRefreshStage.COMPUTE_MIN_COL_STATS)
                    await self._save_colstat_task_result(
                        TableRefreshStage.COMPUTE_FULL_COL_STATS)
                    ingest_result.stage = TableRefreshStage.COMPLETED
            elif ingest_result.ok:
                # Direct/deferred connectors intentionally have no standalone
                # staged Parquet to scan. Their graph refresh owns any later
                # graph-scope stats, so a successful table refresh is still
                # terminal here.
                ingest_result.col_stats_computed = False
                ingest_result.stage = TableRefreshStage.COMPLETED

        await self._save_result(ingest_result)
        return ingest_result

    async def _update_table_timestamp_units(self) -> None:
        self._table = await workflow.execute_activity(
            DataBackendActivities.update_table_timestamp_units,
            args=[
                self._table,
                self._connector_spec,
                self._base_tables_args_list,
                self._base_tables_connector,
            ],
            schedule_to_close_timeout=timedelta(
                hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
        )
        log.info(f"After table refresh, table has Kumo backend "
                 f"representation {self._table}")

    # Run table ingestion as child workflow; or, if an ingestion workflow for
    # the same table & data version is already running, execute an activity to
    # wait for its result.
    async def _run_ingestion(self) -> TableDataTaskResult:
        ingest_wf_id = self._compose_table_ingest_wf_id()
        while True:
            try:
                child_wf_handle = await workflow.start_child_workflow(
                    TableIngestionWorkflow.run,
                    id=ingest_wf_id,
                    args=[
                        self._connector_spec,
                        self._table,
                        self._job_id,
                        self._data_version,
                        self._graph_table_count,
                    ],
                    execution_timeout=timedelta(
                        hours=DATA_BACKEND_INGEST_TIMEOUT_HOURS),
                    id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
                    cancellation_type=ChildWorkflowCancellationType.TRY_CANCEL,
                )
                # Keep this lifecycle log stable for ingest observability.
                log.info(f'Started table ingest child workflow for '
                         f'{self._connector_spec.connector_id}.'
                         f'{self._table.get_source_table_name()}: '
                         f'{ingest_wf_id}')
                return await child_wf_handle
            except WorkflowAlreadyStartedError as err:
                log.info(f'Table ingest workflow {ingest_wf_id} has '
                         f'already started running. '
                         f'Waiting for its completion before '
                         f'starting a new run.')
                await workflow.execute_activity(
                    DataBackendActivities.wait_on_workflow,
                    args=[self._ctx, ingest_wf_id, err.run_id],
                    activity_id='wait_for_ingest',
                    schedule_to_close_timeout=timedelta(
                        hours=DATA_BACKEND_INGEST_TIMEOUT_HOURS),
                    heartbeat_timeout=HEARTBEAT_TIMEOUT_ASYNC_ACTIVITY,
                    retry_policy=CHILD_WF_WAITING_RETRY_POLICY,
                )

    async def _save_result(self, result: TableDataTaskResult,
                           notify_watcher: bool = True):
        await workflow.execute_activity(
            DataBackendActivities.save_table_refresh_result,
            args=[self._table, self._data_version, result],
            schedule_to_close_timeout=timedelta(
                hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
        )
        if notify_watcher:
            await self.notify_watcher(result)

    async def _save_colstat_task_result(self, stage: TableRefreshStage):
        task_result = TableDataTaskResult(
            status=AsyncDataTaskStatus.DONE,
            validation=ValidationResponse(),
            stage=stage,
            data_version=self._data_version,
            created_at=now_utc(),
            updated_at=now_utc(),
        )
        await workflow.execute_activity(
            DataBackendActivities.save_colstat_task_result,
            args=[self._table, self._data_version, task_result],
            schedule_to_close_timeout=timedelta(
                hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
        )

    @staticmethod
    def _compose_job_id(table: TableDefinition, data_version: str) -> str:
        """Create the refresh job identifier used to correlate state."""
        suffix = workflow.info().run_id[-6:]
        return (f'{table.fingerprint().fpid()}-'
                f'{table._dedupe_hash()}-'
                f'{data_version}__{suffix}')

    def _compose_table_ingest_wf_id(self) -> str:
        return compose_table_ingest_wf_id(self._table.fingerprint(),
                                          self._data_version)
