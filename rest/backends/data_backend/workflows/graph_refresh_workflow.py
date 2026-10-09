# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from typing import Any, Optional

from kumo_core.diskgraph.snapshot import DiskgraphIngestResult
from kumo_core.types.schema import VersionedSourceTableID
from temporalio import workflow
from temporalio.workflow import (
    ChildWorkflowCancellationType,
    ParentClosePolicy,
)

from io_utils import kumo_dataclass, logging
from kumo.data import get_persist_dir
from kumo.data.backend_cache import (
    AsyncDataTaskStatus,
    TableDataTaskResult,
    TableRefreshStage,
)
from kumo.validation import ValidationResponse, ValidationWarning
from pool.dataclasses import ResourceHint
from rest.backends import (
    DATA_BACKEND_INGEST_TIMEOUT_HOURS,
    DATA_BACKEND_IO_TIMEOUT_HOURS,
)
from rest.backends.common.activities import save_validation_response
from rest.backends.data_backend.activities import (
    DataBackendActivities,
    GraphTableStatsResult,
)
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.utils import data_backend_cache
from rest.backends.data_backend.workflows import (
    diskgraph_ingest_workflow,
    mixins,
    workflow_utils,
)
from rest.config.data_backend import GraphDataVersionInfo
from rest.config.graph_v2 import GraphDataComputeInput
from rest.config.table_v2 import TableDefinition
from rest.constants import VALIDATE_STATS_RETRY_POLICY
from statedb.config.common import TableID, now_utc
from statedb.utils.errors import GraphRefreshError, rethrow_or_wrap

from .table_refresh_workflow import (
    TableRefreshWorkflow,
    TableRefreshWorkflowInput,
)

log = logging.getLogger(__name__)

WorkflowID = str


@kumo_dataclass
class GraphRefreshResult:
    graph_version_info: GraphDataVersionInfo
    validation: ValidationResponse


@workflow.defn
class GraphRefreshWorkflow(mixins.TableParentWorkflowMixin):
    @staticmethod
    def compose_versioned_wf_id(
        graph_key: str,
        graph_data_version: str,
    ) -> str:
        return f'refresh-graph:name={graph_key}@ver={graph_data_version}'

    @staticmethod
    def compose_wf_id(
            input: GraphDataComputeInput,
            graph_version_info: Optional[GraphDataVersionInfo] = None) -> str:
        if graph_version_info:
            return GraphRefreshWorkflow.compose_versioned_wf_id(
                str(input.graph_key),
                graph_version_info.graph_data_version,
            )

        if workflow._Runtime.maybe_current() is not None:
            # Use workflow package's timestamp to be deterministic, i.e., the
            # timestamp would be recorded by Temporal server and fetched during
            # retry/replay of workflow execution.
            now = workflow.now().timestamp()
        else:
            # Use real timestamp as we are not inside workflow code.
            now = now_utc().timestamp()
        return f'refresh-graph:name={input.graph_key}@ts={now}'

    @workflow.run
    async def run(
        self,
        ctx: DataBackendContext,
        input: GraphDataComputeInput,
        graph_version_info: Optional[GraphDataVersionInfo] = None,
        job_id: Optional[str] = None,
        resource_hint: Optional[ResourceHint] = None,
    ) -> tuple[GraphRefreshResult, GraphDataComputeInput]:
        self._ctx = ctx
        self._input = input
        self._job_id = job_id
        self._resource_hint = resource_hint
        log.info("GraphRefreshWorkflow.run() received job_id=%s", job_id)

        return await self._run_graph_refresh(input, graph_version_info, job_id)

    async def _run_graph_refresh(
        self,
        input: GraphDataComputeInput,
        graph_version_info: Optional[GraphDataVersionInfo],
        job_id: Optional[str],
    ) -> tuple[GraphRefreshResult, GraphDataComputeInput]:
        if graph_version_info is None:
            self._graph_version_info = await workflow.execute_activity(
                DataBackendActivities.refresh_graph_data_version,
                args=[input, job_id],
                schedule_to_close_timeout=timedelta(
                    hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
            )
        else:
            # Graph data version info is already determined but we have to
            # save the version info under the job_id as it's expected in later
            # stages of the job.
            await workflow.execute_activity(
                DataBackendActivities.save_graph_data_version,
                args=[input, graph_version_info, job_id],
                schedule_to_close_timeout=timedelta(
                    hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
            )
            self._graph_version_info = graph_version_info

        # TODO(Dulloor): separate table refresh workflows from view refresh
        if self._job_id is None:
            # Compose a unique job_id; run_id suffix distinguishes
            # concurrent refreshes of the same graph version.
            suffix = workflow.info().run_id[-6:]
            job_id = (f'{input.graph_display_name}-'
                      f'{input.graph_key}-'
                      f'{self._graph_version_info.graph_data_version}__'
                      f'{suffix}')
            self._job_id = job_id

        # NOTE: Update the timestamp units for all tables and views.
        # This needs to be handled here because if we do this in table
        # refresh, the table will be updated and the fingerprint will
        # change compared to the table data in here, as a result, we will
        # fail the _validate_graph_table_stats_and_save check later.
        await self._update_tables_timestamp_units()

        # TODO(Dulloor): start table or view refresh
        # Kick off table refresh child wf for each of tables in the graph.
        await self._start_refresh_tables()

        # Wait until all table ingestions are done (but not entire wf)
        try:
            validation = await self.wait_completion_at_stage(
                TableRefreshStage.INGEST_TABLE)
        except Exception as e:
            msg = ("Kumo encountered an error while ingesting the graph. "
                   "This might be due to a problem "
                   "with the upstream data source. Please check your "
                   "connector configuration and data source.")
            rethrow_or_wrap(e, GraphRefreshError, msg)

        if not validation.ok:
            return GraphRefreshResult(self._graph_version_info,
                                      validation), self._input

        # Validate graph stats (total # rows, total size of all tables).
        # The result also carries total_source_size_bytes for pool admission.
        ingest_stats = await self._validate_graph_table_stats_and_save(
            stage=TableRefreshStage.INGEST_TABLE)
        validation.merge(ingest_stats.validation)
        if not validation.ok:
            return GraphRefreshResult(self._graph_version_info,
                                      validation), self._input

        # Generate col stats once at graph level using diskgraph ingest and
        # write them into existing cache paths.
        pkey_duplicate_counts = await self._run_diskgraph_colstats_pipeline(
            source_size_bytes_by_table=ingest_stats.source_size_bytes_by_table,
            source_size_bytes=ingest_stats.total_source_size_bytes,
        )

        table_validations = (
            await self._validate_table_col_stats_after_diskgraph(
                table_data_versions=self._get_table_data_versions(),
                pkey_duplicate_counts=pkey_duplicate_counts,
            ))
        colstats_validation = ValidationResponse()
        for table_validation in table_validations.values():
            colstats_validation.merge(table_validation)
        validation.merge(colstats_validation)
        if not validation.ok:
            return GraphRefreshResult(self._graph_version_info,
                                      validation), self._input
        await self._save_diskgraph_colstat_task_results(
            self._get_table_data_versions(), table_validations)

        # run validations that involve the graph as a whole
        validation.merge((await self._validate_graph_table_stats_and_save(
            stage=TableRefreshStage.COMPLETED,
            extra_validation=colstats_validation)).validation)
        return (GraphRefreshResult(self._graph_version_info,
                                   validation), self._input)

    async def _start_refresh_table(self, wf_id: str, table_id: TableID,
                                   table: TableDefinition,
                                   parent_close_policy: ParentClosePolicy):
        alias = self._input.table_aliases[table_id]
        table_data_version = self._graph_version_info.table_data_versions[alias]
        connector_spec = self._input.connectors[table.get_connector_id()]
        log.info(f"_start_refresh_table: passing job_id={self._job_id}, "
                 f"to TableRefreshWorkflow for table={alias}")
        await self.start_child_refresh(
            TableRefreshWorkflow.run, self._ctx, args=[
                self._ctx,
                TableRefreshWorkflowInput(
                    connector_spec=connector_spec,
                    table=table,
                    data_version=table_data_version,
                    job_id=self._job_id,
                    alias=alias,
                    defer_col_stats_validation=True,
                    timestamp_units_inferred_by_parent=True,
                    graph_table_count=len(self._input.tables),
                ),
            ], child_wf_id=wf_id, child_id=table_id,
            cancellation_type=ChildWorkflowCancellationType.TRY_CANCEL,
            parent_close_policy=parent_close_policy)

    async def _update_tables_timestamp_units(self) -> None:
        for table_id, table in self._input.tables.items():
            connector_spec = self._input.connectors[table.get_connector_id()]
            table_with_timestamp_unit_updated = (
                await workflow.execute_activity(
                    DataBackendActivities.update_table_timestamp_units,
                    args=[
                        table,
                        connector_spec,
                        None,
                        None,
                    ],
                    schedule_to_close_timeout=timedelta(
                        hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
                ))
            self._input.tables[table_id] = table_with_timestamp_unit_updated
            log.info(
                f"Updated timestamp units for {table_id=} from {table=} to "
                f"{table_with_timestamp_unit_updated=}")

    async def _start_refresh_tables(self):
        # Use ABANDON policy when we need the table refresh workflow to
        # continue running (computing full col stats) after this
        # GraphRefreshWorkflow finishes.
        parent_close_policy = ParentClosePolicy.ABANDON
        for table_id, table in self._input.tables.items():
            alias = self._input.table_aliases[table_id]
            data_version = self._graph_version_info.table_data_versions[alias]
            wf_id = TableRefreshWorkflow.compose_wf_id(table, data_version)
            await self._start_refresh_table(wf_id, table_id, table,
                                            parent_close_policy)
            log.info("Started Table refresh for %s: %s", table_id, wf_id)

    def _get_table_data_versions(self) -> dict[str, str]:
        table_data_versions = {}
        for alias, _ in self._all_graph_tables_by_alias().items():
            table_data_versions[alias] = str(
                self._graph_version_info.table_data_versions[alias])
        return table_data_versions

    def _all_graph_tables_by_alias(self) -> dict[str, TableDefinition]:
        tables_by_alias: dict[str, TableDefinition] = {}
        for table_id, table in self._input.tables.items():
            tables_by_alias[self._input.table_aliases[table_id]] = table
        for table_id, table in self._input.base_tables.items():
            tables_by_alias.setdefault(self._input.table_aliases[table_id],
                                       table)
        return tables_by_alias

    async def _run_diskgraph_colstats_pipeline(
        self,
        source_size_bytes_by_table: dict[str, int],
        source_size_bytes: int,
    ) -> Optional[dict[str, int]]:
        table_data_versions = self._get_table_data_versions()
        data_persist_dir = get_persist_dir()

        # Source-file download runs INSIDE ``DiskgraphIngestWorkflow`` on the
        # pool task queue (folded into the daemon's ``ensure_snapshot`` on the
        # snapshot's owning node), so source files land on the same node
        # ``ensure_snapshot`` reads from. We just pass ``connector_specs``
        # through; the child workflow does the dispatch.
        wf_id = diskgraph_ingest_workflow.DiskgraphIngestWorkflow.compose_wf_id(
            self._input.graph_key,
            self._graph_version_info.graph_data_version,
        )
        use_ingest_stats_v2 = workflow.patched('diskgraph-ingest-stats-v2')
        if use_ingest_stats_v2:
            wf_id = f'{wf_id}:stats-v2'
        child_task = await workflow_utils.start_child_data_wf(
            diskgraph_ingest_workflow.DiskgraphIngestWorkflow.run,
            self._ctx,
            wf_id=wf_id,
            args=[
                self._input.graph_definition,
                self._input.graph_display_name,
                data_persist_dir,
                table_data_versions,
                self._input.graph_key,
                self._graph_version_info.graph_data_version,
                self._all_graph_tables_by_alias(),
                self._input.connectors,
                source_size_bytes,
                self._resource_hint,
                source_size_bytes_by_table,
            ],
            execution_timeout=timedelta(
                hours=DATA_BACKEND_INGEST_TIMEOUT_HOURS),
            parent_close_policy=ParentClosePolicy.REQUEST_CANCEL,
        )
        result: DiskgraphIngestResult | dict[str, Any] = await child_task
        if isinstance(result, dict):
            result = DiskgraphIngestResult(**result)
        return (result.pkey_duplicate_counts if use_ingest_stats_v2 else None)

    async def _validate_table_col_stats_after_diskgraph(
        self,
        table_data_versions: dict[str, str],
        pkey_duplicate_counts: Optional[dict[str, int]],
    ) -> dict[str, ValidationResponse]:
        table_validations = {}
        for table_id, table in self._input.tables.items():
            alias = self._input.table_aliases[table_id]
            args = [
                table.get_connector_id(),
                table,
                table_data_versions[alias],
            ]
            if pkey_duplicate_counts is not None:
                args.append(pkey_duplicate_counts.get(alias, 0))
            table_validation = await workflow.execute_activity(
                DataBackendActivities.validate_table_col_stats,
                args=args,
                retry_policy=VALIDATE_STATS_RETRY_POLICY,
                schedule_to_close_timeout=timedelta(
                    hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
            )
            table_validations[alias] = table_validation
        return table_validations

    async def _save_diskgraph_colstat_task_results(
        self,
        table_data_versions: dict[str, str],
        table_validations: dict[str, ValidationResponse],
    ) -> None:
        for table_id, table in self._input.tables.items():
            alias = self._input.table_aliases[table_id]
            data_version = table_data_versions[alias]
            min_colstat_result = TableDataTaskResult(
                status=AsyncDataTaskStatus.DONE,
                stage=TableRefreshStage.COMPUTE_MIN_COL_STATS,
                validation=ValidationResponse(),
                data_version=data_version,
                created_at=now_utc(),
                updated_at=now_utc(),
            )
            await workflow.execute_activity(
                DataBackendActivities.save_colstat_task_result,
                args=[table, data_version, min_colstat_result],
                schedule_to_close_timeout=timedelta(minutes=20),
            )

            full_colstat_result = TableDataTaskResult(
                status=AsyncDataTaskStatus.DONE,
                stage=TableRefreshStage.COMPUTE_FULL_COL_STATS,
                validation=ValidationResponse(),
                data_version=data_version,
                created_at=now_utc(),
                updated_at=now_utc(),
            )
            await workflow.execute_activity(
                DataBackendActivities.save_colstat_task_result,
                args=[table, data_version, full_colstat_result],
                schedule_to_close_timeout=timedelta(minutes=20),
            )

            completed_result = TableDataTaskResult(
                status=AsyncDataTaskStatus.DONE,
                stage=TableRefreshStage.COMPLETED,
                validation=table_validations[alias],
                data_version=data_version,
                created_at=now_utc(),
                updated_at=now_utc(),
            )
            await workflow.execute_activity(
                DataBackendActivities.save_table_refresh_result,
                args=[table, data_version, completed_result],
                schedule_to_close_timeout=timedelta(
                    hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
            )

    async def _validate_graph_table_stats_and_save(
        self,
        stage: TableRefreshStage,
        extra_validation: Optional[ValidationResponse] = None,
    ) -> GraphTableStatsResult:
        versioned_table_dict = {}
        aliases = {}
        for tables in (self._input.tables, self._input.base_tables):
            for table_id, table_args in tables.items():
                alias = self._input.table_aliases[table_id]
                t_id = VersionedSourceTableID(
                    table_args.get_connector_id(),
                    table_args.get_source_table_name(),
                    self._graph_version_info.table_data_versions[alias])
                versioned_table_dict.setdefault(t_id, table_args)
                aliases.setdefault(t_id, []).append(alias)

        result: GraphTableStatsResult = await workflow.execute_activity(
            DataBackendActivities.validate_graph_table_stats,
            args=[
                self._ctx,
                versioned_table_dict,
                stage,
                list(self._input.connectors.values()),
                aliases,
            ],
            retry_policy=VALIDATE_STATS_RETRY_POLICY,
            schedule_to_close_timeout=timedelta(
                hours=DATA_BACKEND_IO_TIMEOUT_HOURS),
        )

        if extra_validation is not None:
            result.validation.merge(extra_validation)
        if stage == TableRefreshStage.COMPLETED:
            for edge in self._input.graph_definition.edges:
                if edge.src_table != edge.dst_table:
                    continue
                pkey = self._input.graph_definition.tables[edge.dst_table].pkey
                result.validation.warnings.append(
                    ValidationWarning(
                        title='Self-relation omitted',
                        message=(
                            f'Self-relation {edge.src_table}.{edge.fkey} -> '
                            f'{edge.dst_table}.{pkey} is not supported and was '
                            'omitted from the graph snapshot.'),
                    ))
        graph_dir = data_backend_cache().graph_cache.graph_dir(
            self._input.graph_key, self._graph_version_info.graph_data_version)
        if stage == TableRefreshStage.INGEST_TABLE:
            validation_identifier = 'table_stats'
        else:
            assert (stage in [
                TableRefreshStage.COMPUTE_MIN_COL_STATS,
                TableRefreshStage.COMPLETED
            ])
            validation_identifier = 'table_col_stats'

        await save_validation_response(result.validation, graph_dir,
                                       validation_identifier)

        return result
