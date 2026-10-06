# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
from collections.abc import Awaitable
from datetime import timedelta
from typing import Dict, List, Mapping, Optional, Tuple, Union, cast

import kumoapi.data_snapshot as snapshot_interfaces
import kumoapi.graph as graph_interfaces
from kumo_core.types import ConnectorType, TableStatType
from kumo_core.types.schema import FPID
from kumoapi.common import JobStatus
from temporalio.client import (
    WorkflowExecutionDescription,
    WorkflowExecutionStatus,
    WorkflowFailureError,
)

from io_utils import logging
from io_utils.json import load_many_json
from kumo.config.graph import GraphConfig
from kumo.connector.connector import ConnectorError
from kumo.data import get_active_cache
from kumo.data.backend_cache import (
    AsyncDataTaskStatus,
    TableDataTaskResult,
    TableRefreshStage,
)
from kumo.data.cache.exact_stats import ExactStats
from kumo.graph import Graph
from rest.backends.data_backend.activities import DataBackendActivities
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.data_backend import DataBackend
from rest.backends.data_backend.workflows import (
    GraphRefreshWorkflow,
    TableRefreshWorkflow,
)
from rest.config.data_backend import (
    ConnectorExecutionRef,
    GraphDataVersionInfo,
)
from rest.config.error_handling import root_cause_message
from rest.config.graph import Edge, EdgeHealth, EdgeHealthStats
from rest.config.graph_v2 import (
    GraphDataComputeInput,
    GraphDefinition,
    GraphSnapshotRequest,
    GraphSnapshotResponse,
    GraphSnapshotStatsResponse,
)
from rest.config.table_v2 import (
    ADLSSourceTable,
    GCSSourceTable,
    S3SourceTable,
    SnapshotStat,
    TableDBEntryV2,
    TableDefinition,
    TableSnapshotColumnStats,
    TableSnapshotResource,
)
from rest.deps.context import _current_user_ctx
from rest.driver.error_constants import GRAPH_SNAPSHOT_NOT_FOUND_MESSAGE
from rest.driver.table_logic import (
    TableLogic,
    _table_read_error_v2,
    canonicalize_custom_sql_source,
)
from rest.utils.table import ColStatsSet, get_full_time_range, get_table_stats
from service_framework.deploy_env import send_slack_message
from service_framework.deploy_stage import DeployStage
from statedb.config.common import (
    ConnectorLink,
    DataSnapshot,
    GraphIDv2,
    ProjectID,
    TableID,
    TableIDv2,
    TableName,
    now_utc,
)
from statedb.config.table import TableDataVersion
from statedb.db.auth import AuthPrincipal
from statedb.db.services import StateServices
from statedb.rest.connector_authorization import CONNECTOR_GRAPH_ACTIONS
from statedb.utils.errors import (
    TableDoesNotExistError,
    TableSnapshotDoesNotExistError,
)
from temporalio_commons.temporal_client import TemporalClient

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _PreparedTableRefresh:
    table_entry: TableDBEntryV2
    data_version: TableDataVersion
    execution_ref: ConnectorExecutionRef | None


_TABLE_REFRESH_BATCH_CLEANUP_TIMEOUT_SECONDS = 30.0


def _consume_background_future(future: asyncio.Future) -> None:
    """Consume a detached cleanup result without logging sensitive detail."""
    try:
        future.result()
    except BaseException:
        pass


async def get_graph_snapshot_info(
    graph_snapshot_id: snapshot_interfaces.GraphSnapshotID
) -> Tuple[GraphIDv2, GraphDataVersionInfo]:
    """
    Helper function to lookup persisted graph data version for the given
    graph data snapshot ID.
    """
    graph_fpid, data_ver = DataSnapshot.GRAPH.parse_id(graph_snapshot_id)
    graph_id = GraphIDv2.compose(fpid=graph_fpid)
    cache = get_active_cache()
    version_info_path = cache.graph_cache.version_info_path(graph_id, data_ver)
    graph_version_info = None
    for attempt in range(5):
        graph_version_info = (await
                              load_many_json.coroutine([version_info_path],
                                                       GraphDataVersionInfo))[0]
        if graph_version_info:
            break
        if attempt < 4:
            await asyncio.sleep(0.5)
    if not graph_version_info:
        raise ValueError(
            GRAPH_SNAPSHOT_NOT_FOUND_MESSAGE.format(graph_snapshot_id))
    assert graph_version_info.graph_data_version == data_ver
    return graph_id, graph_version_info


def _hydrate_graph_config_source_from_graph_def(
    graph_config: GraphConfig,
    graph_def: GraphDefinition,
) -> GraphConfig:
    """Backfill GraphConfig table source metadata from GraphDefinition.

    In diskgraph mode we derive ingest paths from GraphConfig source metadata.
    Some Graph object construction paths can produce sparse GraphConfig tables
    (e.g. missing data_source/source_name), so we copy canonical source fields
    from persisted GraphDefinition when absent.
    """
    for table_name, table_def in graph_def.tables.items():
        try:
            table_cfg = graph_config[table_name]
        except KeyError:
            continue

        source_table = table_def.source_table
        connector_id = getattr(source_table, 'connector_id', None)
        source_table_name = table_def.get_source_table_name()

        if connector_id and (not table_cfg.data_source
                             or table_cfg.data_source == 'default'):
            table_cfg.data_source = str(connector_id)
        if source_table_name and not table_cfg.source_name:
            table_cfg.source_name = str(source_table_name)

    return graph_config


def _build_ingest_graph_config(
    graph_obj: Graph,
    name_to_ver: Mapping[TableName, str],
    graph_def: GraphDefinition,
) -> GraphConfig:
    """Pin per-table data versions and build the hydrated GraphConfig for DG ingest.

    Pinning ``_data_version`` before ``graph_obj.config()`` avoids blocking
    Connector.get_data_version calls on the event loop; hydration backfills
    source metadata from the GraphDefinition so diskgraph ingest can resolve
    source paths. Callers build ``name_to_ver`` from their own version source
    (resolved versions for refresh, snapshot versions for snapshotting); the
    key set covers every table, so the lenient lookup never skips one.
    """
    for table in graph_obj.tables:
        ver = name_to_ver.get(TableName(table.table_name))
        if ver is not None:
            table._data_version = ver
    return _hydrate_graph_config_source_from_graph_def(graph_obj.config(),
                                                       graph_def)


class DataSnapshotApiDriver:
    """RPC service for table/graph data snapshot APIs."""
    def __init__(
        self,
        data_backend_ctx: DataBackendContext,
    ):
        self._data_backend_ctx = data_backend_ctx
        self._temporal_client = TemporalClient(data_backend_ctx._config)
        self._state_services: Optional[StateServices] = None

    def _configure_state_services(self, state_services: StateServices) -> None:
        self._state_services = state_services

    def _require_state_services(self) -> StateServices:
        if self._state_services is None:
            raise RuntimeError(
                "DataSnapshotApiDriver StateServices is not configured")
        return self._state_services

    def _table_service(self):
        return self._require_state_services().tables

    def _connector_service(self):
        return self._require_state_services().connectors

    def _graph_service(self):
        return self._require_state_services().graphs

    async def require_project_access(
        self,
        project_id: ProjectID,
        principal: AuthPrincipal,
    ) -> None:
        await self._require_state_services().projects.get_project(
            project_id,
            principal,
        )

    async def _get_graph_snapshot_info(
        self,
        graph_snapshot_id: snapshot_interfaces.GraphSnapshotID,
    ) -> Tuple[GraphIDv2, GraphDataVersionInfo]:
        return await get_graph_snapshot_info(graph_snapshot_id)

    async def _get_table_entry(
        self,
        project_id: ProjectID,
        fpid: FPID,
    ) -> Optional[TableDBEntryV2]:
        return await self._table_service().get_table_entry(project_id, fpid)

    async def _get_many_table_entries(
        self,
        project_id: ProjectID,
        table_ids: List[FPID],
    ) -> List[TableDBEntryV2]:
        table_db_entries_map = await self._table_service().get_table_entries(
            project_id, table_ids)
        return [
            table_db_entries_map[table_id] for table_id in table_ids
            if table_id in table_db_entries_map
        ]

    async def _get_required_table_entry(
        self,
        project_id: ProjectID,
        fpid: FPID,
    ) -> TableDBEntryV2:
        table_db_entry = await self._get_table_entry(project_id, fpid)
        if table_db_entry is None:
            raise KeyError(f"Table with id {fpid} not found in database")
        return table_db_entry

    async def _get_many_table_definitions(
        self,
        project_id: ProjectID,
        table_ids: List[TableIDv2],
    ) -> List[Union[TableDefinition, Exception]]:
        table_fpids = []
        for table_id in table_ids:
            _, _, table_fpid = TableIDv2.parse(table_id)
            assert table_fpid is not None
            table_fpids.append(table_fpid)

        table_db_entries_map = await self._table_service().get_table_entries(
            project_id, table_fpids)
        table_defs: List[Union[TableDefinition, Exception]] = []
        for table_id, table_fpid in zip(table_ids, table_fpids):
            table_db_entry = table_db_entries_map.get(table_fpid)
            if table_db_entry is None:
                table_defs.append(TableDoesNotExistError(table_id))
                continue
            table_defs.append(table_db_entry.config)
        return table_defs

    async def _get_graph_by_fpid(
        self,
        project_id: ProjectID,
        graph_fpid: FPID,
    ) -> Tuple[GraphIDv2, GraphDefinition]:
        graph_id = GraphIDv2.compose(fpid=graph_fpid)
        graph_def = await self._graph_service().get_graph_definition(
            project_id, graph_fpid)
        if graph_def is None:
            raise KeyError(f'graph-{graph_fpid} is not found.')
        return graph_id, graph_def

    async def _get_graph_data_compute_input(
        self,
        graph_id: GraphIDv2,
        project_id: ProjectID,
    ) -> GraphDataComputeInput:
        return await self._graph_service().get_graph_data_compute_input(
            graph_id,
            project_id=project_id,
            user_authorization=_current_user_ctx.get(),
        )

    async def _get_state_graph_edge_stats(
        self,
        graph_id: GraphIDv2,
        graph_version_info: GraphDataVersionInfo,
        project_id: ProjectID,
    ) -> EdgeHealthStats:
        data_compute_input = await self._graph_service().get_graph_cache_input(
            graph_id, project_id=project_id)

        link_stats = await DataBackend.get_edge_stats(
            self._data_backend_ctx,
            data_compute_input,
            graph_version_info,
        )
        edge_stats = EdgeHealthStats(is_ready=link_stats is not None,
                                     link_stats=link_stats)

        pquery_id, _ = GraphIDv2.parse(graph_id)
        if edge_stats.link_stats:
            for edge in edge_stats.link_stats:
                if edge.col_a:
                    edge.col_a.table_id = TableIDv2.update(
                        TableIDv2(edge.col_a.table_id), pquery_id, False)
                if edge.col_b:
                    edge.col_b.table_id = TableIDv2.update(
                        TableIDv2(edge.col_b.table_id), pquery_id, False)
        return edge_stats

    async def create_graph_snapshot(
        self,
        graph_fpid: FPID,
        refresh_source: bool,
        base_graph_name: Optional[str] = None,
        *,
        project_id: ProjectID,
    ) -> snapshot_interfaces.GraphSnapshotID:
        """Create a graph data snapshot for an existing graph: the data
        snapshot corresponds to a specific graph data version.

        Args:
            graph_fpid (FPID): Fingerprint ID of the graph definition.
            refresh_source (bool): If true, force refreshing all tables in this
            graph, i.e., sync with the data source to snapshot latest source
            data version.  If false, if any table in the graph has been already
            successfully ingested (snapshoted) in the past, the most recent
            snapshot (data version) of that table will be reused.
        Returns:
            api.GraphSnapshotID: resource ID for the graph data snapshot.
        snapshot.
        """
        await self.require_project_access(project_id, _current_user_ctx.get())
        graph_id, graph_def = await self._get_graph_by_fpid(
            project_id, graph_fpid)

        # Create a snapshot for each table in this graph.
        table_snapshot_ids = await self._create_many_table_snapshots(
            list(graph_def.tables.values()),
            refresh_source=refresh_source,
            project_id=project_id,
            defer_col_stats_validation=True,
        )

        table_versions: Dict[TableID, TableDataVersion] = {}
        for table_name, table_snapshot_id in zip(graph_def.tables.keys(),
                                                 table_snapshot_ids):
            fpid, data_ver = DataSnapshot.TABLE.parse_id(table_snapshot_id)
            table_id = TableIDv2.compose(None, table_name, fpid)
            table_versions[table_id] = TableDataVersion(data_ver)

        # Trigger graph refresh workflow with fixed table versions.
        # This is slightly different from GraphLogic.start_graph_refresh()
        # by computing and saving graph data vesion even before launching the
        # GraphRefreshWorkflow.
        data_compute_input = (await self._get_graph_data_compute_input(
            graph_id, project_id))
        # Pin the per-table data versions instead of mutating the (now
        # immutable) TableDefinition. `refresh_graph_data_version` reads
        # from `pinned_data_versions` to override the latest-version fetch.
        data_compute_input.pinned_data_versions = dict(table_versions)
        version_info = await DataBackend.refresh_graph_data_version_in_activity(
            self._data_backend_ctx,
            data_compute_input,
        )
        graph_entry = await self._graph_service().get_graph_entry(
            project_id, graph_fpid)
        if graph_entry and graph_entry.alias:
            version_info.graph_name = graph_entry.alias
        elif base_graph_name:
            version_info.graph_name = f"{base_graph_name} (modified)"
        else:
            version_info.graph_name = None
        # Re-save version_info to persist graph_name. This is necessary for
        # single-table graphs (no column_groups) where start_refresh_graph
        # returns early and never calls save_graph_data_version.
        await DataBackendActivities.save_graph_data_version(
            data_compute_input, version_info)
        connector_ids = tuple(data_compute_input.connectors)
        fresh_references = await self._connector_service(
        ).fresh_execution_refs_for_workflow(
            _current_user_ctx.get(),
            tuple(data_compute_input.connectors.values()),
            CONNECTOR_GRAPH_ACTIONS,
        )
        data_compute_input.connectors = dict(
            zip(connector_ids, fresh_references, strict=True))
        await DataBackend.start_refresh_graph(
            self._data_backend_ctx,
            data_compute_input,
            version_info,
        )
        return DataSnapshot.GRAPH.format_id(graph_fpid,
                                            version_info.graph_data_version)

    async def get_graph_snapshot(
        self,
        request: GraphSnapshotRequest,
        project_id: ProjectID,
    ) -> GraphSnapshotResponse:
        await self.require_project_access(project_id, _current_user_ctx.get())
        # Get data versions of tables in this graph.
        graph_snapshot_id = request.graph_snapshot_id
        graph_def = request.graph_definiton
        graph_version = request.graph_version

        # If graph defintion and graph version are available, they are
        # the preferred method to fetch a version of the graph snapshot
        # because we can always, easily, infer the graph ID from its
        # definition.
        if graph_def and graph_version:
            graph_fpid = graph_def.fpid
            graph_snapshot_id = DataSnapshot.GRAPH.format_id(
                graph_fpid, graph_version)
            graph_id, version_info = await self._get_graph_snapshot_info(
                graph_snapshot_id)
        else:
            if not graph_snapshot_id:
                raise ValueError(
                    "Please provide a valid graph snapshot ID or a" +
                    " valid graph definition and graph version" +
                    " to fetch a graph snapshot.")
            graph_id, version_info = await self._get_graph_snapshot_info(
                graph_snapshot_id)
            graph_fpid, _ = DataSnapshot.GRAPH.parse_id(graph_snapshot_id)

            _, graph_def = await self._get_graph_by_fpid(project_id, graph_fpid)

        table_snapshot_ids: Dict[FPID, TableDataVersion] = {}
        table_ids: Dict[
            graph_interfaces.TableName,
            snapshot_interfaces.TableSnapshotID,
        ] = {}
        for table_name, table_def in graph_def.tables.items():
            id = DataSnapshot.TABLE.format_id(
                table_def.fpid, version_info.table_data_versions[table_name])
            table_ids[table_name] = id
            fpid, data_ver = DataSnapshot.TABLE.parse_id(id)
            table_snapshot_ids[fpid] = TableDataVersion(data_ver)

        table_snapshots = await self._get_many_kumo_table_snapshots(
            project_id, table_snapshot_ids)

        graph_ingest_state = _merge_table_ingest_job_states(table_snapshots)
        if graph_def.col_groups:
            refresh_wf_id = GraphRefreshWorkflow.compose_versioned_wf_id(
                str(graph_id),
                version_info.graph_data_version,
            )
            graph_refresh_state = await self._graph_refresh_job_state(
                refresh_wf_id)
            if graph_refresh_state is not None:
                graph_ingest_state = _merge_graph_ingest_job_states(
                    graph_ingest_state,
                    graph_refresh_state,
                )

        return GraphSnapshotResponse(
            snapshot_interfaces.GraphSnapshotResource(
                id=graph_snapshot_id,
                table_ids=table_ids,
                stages={
                    snapshot_interfaces.GraphSnapshotStage.INGEST:
                    graph_ingest_state,
                    # TODO: add compute_edge_stats stage?
                },
                graph_name=version_info.graph_name,
            ),
            graph_def,
        )

    async def _graph_refresh_job_state(
        self,
        workflow_id: str,
    ) -> Optional[snapshot_interfaces.JobState]:
        description = await self._temporal_client.get_workflow_desc(workflow_id)
        if description is None or description.status is None:
            return None

        status = {
            WorkflowExecutionStatus.RUNNING: JobStatus.RUNNING,
            WorkflowExecutionStatus.CONTINUED_AS_NEW: JobStatus.RUNNING,
            WorkflowExecutionStatus.COMPLETED: JobStatus.DONE,
            WorkflowExecutionStatus.FAILED: JobStatus.FAILED,
            WorkflowExecutionStatus.TIMED_OUT: JobStatus.FAILED,
            WorkflowExecutionStatus.CANCELED: JobStatus.CANCELLED,
            WorkflowExecutionStatus.TERMINATED: JobStatus.CANCELLED,
        }.get(description.status)
        if status is None:
            raise ValueError(
                f'Unsupported WorkflowExecutionStatus {description.status}')

        state = snapshot_interfaces.JobState(
            status=status,
            start_time=description.start_time,
            end_time=description.close_time,
        )
        if description.status == WorkflowExecutionStatus.FAILED:
            handle = await self._temporal_client.get_workflow_handle(
                workflow_id,
                run_id=description.run_id,
            )
            try:
                await handle.result()
            except WorkflowFailureError as error:
                state.error = root_cause_message(error)
            else:
                state.error = 'Graph refresh failed without an error message.'
        elif description.status == WorkflowExecutionStatus.TIMED_OUT:
            state.error = 'Graph refresh timed out.'
        return state

    async def get_ui_edge_health(
        self,
        graph_snapshot_id: snapshot_interfaces.GraphSnapshotID,
        project_id: ProjectID,
    ) -> EdgeHealthStats:
        """
        Gets edge health statistics from the backend in form of EdgeHealthStats
        to show in the UI

        Args:
            graph_snapshot_id (api.GraphSnapshotID): The snapshot ID we want to
                use to get edge stats for

        Returns:
            EdgeHealthStats: Edge health stats for each edge
        """
        # Get edge stats from the backend
        await self.require_project_access(project_id, _current_user_ctx.get())
        graph_id, graph_version_info = await self._get_graph_snapshot_info(
            graph_snapshot_id)
        state_edge_stats = await self._get_state_graph_edge_stats(
            graph_id, graph_version_info, project_id)
        return state_edge_stats

    async def get_edge_health(
        self,
        graph_snapshot_id: snapshot_interfaces.GraphSnapshotID,
        project_id: ProjectID,
    ) -> graph_interfaces.EdgeHealthResponse:
        """
        Gets edge health statistics from the backend by doing the following
         - Get raw edge stats from the backend
         - Re-generate the edges from the col groups in the backend struct
         - Make sure stats are ready
         - If ready, process the raw backend edge stats into public struct

        Args:
            graph_snapshot_id (api.GraphSnapshotID): The snapshot ID we want to
                use to get edge stats for

        Returns:
            EdgeHealthResponse: Edge health stats for each edge
        """
        await self.require_project_access(project_id, _current_user_ctx.get())
        # Get edge stats from the backend
        v1_stats: EdgeHealthStats = await self.get_ui_edge_health(
            graph_snapshot_id, project_id)

        # Get information about the graph definition
        graph_fpid, _ = DataSnapshot.GRAPH.parse_id(graph_snapshot_id)
        _, graph_def = await self._get_graph_by_fpid(project_id, graph_fpid)
        edges = graph_def.edges

        # If not ready, early return
        if not v1_stats.is_ready:
            return graph_interfaces.EdgeHealthResponse(is_ready=False,
                                                       statistics={})
        assert v1_stats.link_stats is not None

        # If ready, process into public-facing api class
        statistics: Dict[str, graph_interfaces.EdgeHealthStatistics] = {}
        for stat in v1_stats.link_stats:
            assert isinstance(stat, EdgeHealth)
            # Create edge with col_a as the source (may not be accurate)
            edge = Edge(
                src_table=stat.col_a.table_name,
                fkey=stat.col_b.column_name,
                dst_table=stat.col_b.table_name,
            )
            # Make sure this is the right direction
            use_reversed_edge = False
            if edge not in edges:
                # If not, use the reverse edge
                edge = Edge(
                    src_table=stat.col_b.table_name,
                    fkey=stat.col_a.column_name,
                    dst_table=stat.col_a.table_name,
                )
                use_reversed_edge = True

            # Extract values, make sure they're all specified
            absolute_src_in_dst: Optional[float] = (stat.matches.a_in_b
                                                    if not use_reversed_edge
                                                    else stat.matches.b_in_a)
            absolute_dst_in_src: Optional[float] = (stat.matches.b_in_a
                                                    if not use_reversed_edge
                                                    else stat.matches.a_in_b)
            assert absolute_src_in_dst is not None
            assert absolute_dst_in_src is not None

            percent_src_in_dst: Optional[float] = (stat.percents.a_in_b
                                                   if not use_reversed_edge else
                                                   stat.percents.b_in_a)
            percent_dst_in_src: Optional[float] = (stat.percents.b_in_a
                                                   if not use_reversed_edge else
                                                   stat.percents.a_in_b)
            assert percent_src_in_dst is not None
            assert percent_dst_in_src is not None

            # Add the edge health into the statistics dictionary
            stats = graph_interfaces.EdgeHealthStatistics(
                absolute_match_stats=graph_interfaces.EdgeMatches(
                    src_in_dst=absolute_src_in_dst,
                    dst_in_src=absolute_dst_in_src,
                ),
                percent_match_stats=graph_interfaces.EdgeMatches(
                    src_in_dst=percent_src_in_dst,
                    dst_in_src=percent_dst_in_src,
                ),
                total_num_edges=stat.edge_count,
            )
            statistics[edge.fully_qualified_name] = stats

        return graph_interfaces.EdgeHealthResponse(is_ready=True,
                                                   statistics=statistics)

    async def create_table_snapshot(
        self,
        table_definition: TableDefinition,
        refresh_source: bool,
        *,
        project_id: ProjectID,
    ) -> snapshot_interfaces.TableSnapshotID:
        """Create a TableSnapshot resource that represents a specific data
        version of the table, as well as an associated table refresh workflow.

        Args:
            table_definition (TableDefinition): Definition of the table.
            refresh_source (bool, optional): Whether to force refreshing source
            table's data version, i.e., sync with data source to check latest
            version. If false, reuse the source table's most recent
            (successfully) ingested data version, if available.

        Returns:
            api.TableSnapshotID: _description_
        """
        await self.require_project_access(project_id, _current_user_ctx.get())
        table_definition = await self._canonicalize_custom_sql_name(
            table_definition, project_id)
        table_db_entry = await self._require_reachable_table_entry(
            table_definition, project_id)
        data_ver = await self._trigger_state_services_table_refresh(
            table_db_entry,
            refresh_source=refresh_source,
            project_id=project_id,
        )
        return DataSnapshot.TABLE.format_id(table_definition.fpid, data_ver)

    async def _create_many_table_snapshots(
        self,
        table_definitions: List[TableDefinition],
        *,
        refresh_source: bool,
        project_id: ProjectID,
        defer_col_stats_validation: bool = False,
    ) -> List[snapshot_interfaces.TableSnapshotID]:
        if not table_definitions:
            return []

        table_definitions = [
            await self._canonicalize_custom_sql_name(table_definition,
                                                     project_id)
            for table_definition in table_definitions
        ]
        table_fpids = [
            table_definition.fpid for table_definition in table_definitions
        ]
        table_entries_by_fpid = await self._table_service().get_table_entries(
            project_id, table_fpids)
        table_entries: List[TableDBEntryV2] = []
        for table_definition in table_definitions:
            table_entry = table_entries_by_fpid.get(table_definition.fpid)
            if table_entry is None:
                table_entry = await self._ensure_table_entry_with_state_services(
                    table_definition, project_id)
            table_entries.append(table_entry)

        data_versions = await self._trigger_many_state_services_table_refresh(
            table_entries,
            refresh_source=refresh_source,
            project_id=project_id,
            defer_col_stats_validation=defer_col_stats_validation,
        )
        return [
            DataSnapshot.TABLE.format_id(table_definition.fpid, data_ver)
            for table_definition, data_ver in zip(
                table_definitions,
                data_versions,
            )
        ]

    async def _canonicalize_custom_sql_name(
        self,
        table_definition: TableDefinition,
        project_id: ProjectID,
    ) -> TableDefinition:
        """Assign a Custom SQL table's server-canonical synthetic name + parsed
        referenced tables before this snapshot-create path computes its ``fpid``
        or persists it.

        Mirrors ``TableDriver.create_full_table`` so a SQL table snapshotted
        here dedupes onto the same fingerprint entry the table-create path
        produces, rather than persisting a divergent entry under the SDK's
        provisional ``md5(raw sql)`` name. Resolved authoritatively from
        credential-free connector metadata: a SQL table whose warehouse can't be
        typed fails loudly rather than persisting a non-canonical name.
        """
        if not table_definition.is_sql:
            return table_definition
        connector = await self._connector_service().get_for_project(
            ProjectID(project_id), table_definition.get_connector_id(),
            _current_user_ctx.get())
        connector_type = ConnectorType(connector.config.type.value)
        return canonicalize_custom_sql_source(table_definition, connector_type)

    async def _require_reachable_table_entry(
        self,
        table_definition: TableDefinition,
        project_id: ProjectID,
    ) -> TableDBEntryV2:
        """Return the project's entry for a definition that is durably
        reachable, either under a table name or through a saved graph.

        Raises ``ValueError`` when the project holds no entry for the
        definition, or holds a nameless one that no saved graph references.
        """
        entry = await self._table_service().get_table_entry(
            project_id, table_definition.fpid)
        if entry is None:
            raise ValueError(
                'This table is not saved in this project, so there is nothing '
                'to snapshot. Save the table with table.save(name), or save '
                'the graph it belongs to, and then request the snapshot.')
        if entry.alias:
            return cast(TableDBEntryV2, entry)
        graphs = await self._graph_service().list_graphs(project_id)
        if any(entry.id in graph.table_name_to_dbid.values()
               for graph in graphs):
            return cast(TableDBEntryV2, entry)
        raise ValueError(
            'This table is stored without a name and no saved graph uses it, '
            'so it cannot be snapshotted. Give it a name with '
            'table.save(name), and then request the snapshot.')

    async def _ensure_table_entry_with_state_services(
        self,
        table_definition: TableDefinition,
        project_id: ProjectID,
    ) -> TableDBEntryV2:
        table_service = self._table_service()

        table_db_entry = await table_service.get_table_entry(
            project_id, table_definition.fpid)
        if table_db_entry is not None:
            return table_db_entry

        await self._validate_new_table_with_state_services(
            table_definition, project_id)
        table_db_entry = await table_service.save_table_definition(
            project_id, table_definition)

        # Verify the saved definition through the StateServices read API. This
        # mirrors the legacy readback before refresh without opening StateDB
        # storage.
        table_def = await table_service.get_table_definition(
            project_id, table_definition.fpid)
        if table_def is None:
            raise TableDoesNotExistError(
                TableIDv2.compose(None, TableName('NA'), table_definition.fpid))
        return table_db_entry

    async def _validate_new_table_with_state_services(
        self,
        table_definition: TableDefinition,
        project_id: ProjectID,
    ) -> None:
        self._connector_service()

        source_table = table_definition.source_table
        is_file_source = isinstance(
            source_table, (S3SourceTable, GCSSourceTable, ADLSSourceTable))
        if is_file_source and source_table.connector_id is None:
            raise ConnectorError(
                "File source tables require a project-owned connector_id.")

        await TableLogic(
            self._data_backend_ctx,
            state_services=self._state_services,
            project_id=project_id,
        ).validate_table(table_definition)

    async def _trigger_state_services_table_refresh(
        self,
        table_entry: TableDBEntryV2,
        *,
        refresh_source: bool = True,
        project_id: ProjectID,
    ) -> TableDataVersion:
        data_versions = await self._trigger_many_state_services_table_refresh(
            [table_entry],
            refresh_source=refresh_source,
            project_id=project_id,
        )
        return data_versions[0]

    async def _trigger_many_state_services_table_refresh(
        self,
        table_entries: List[TableDBEntryV2],
        *,
        refresh_source: bool = True,
        project_id: ProjectID,
        defer_col_stats_validation: bool = False,
    ) -> List[TableDataVersion]:
        connector_service = self._connector_service()
        issued_refs: list[ConnectorExecutionRef] = []
        try:
            for table_entry in table_entries:
                table_def = table_entry.config
                connector_id = table_def.get_connector_id()
                execution_ref = (await
                                 connector_service.execution_ref_for_project(
                                     _current_user_ctx.get(), project_id,
                                     connector_id, CONNECTOR_GRAPH_ACTIONS))
                issued_refs.append(execution_ref)
        except BaseException:
            await self._run_table_refresh_cleanup(
                self._revoke_table_refresh_authorizations(issued_refs),
                label='issued capability revocation',
            )
            raise

        prepare_batch = asyncio.gather(
            *(self._prepare_state_services_table_refresh_with_connector(
                table_entry,
                execution_ref,
                refresh_source=refresh_source,
            ) for table_entry, execution_ref in zip(table_entries, issued_refs)
              ), return_exceptions=True)
        try:
            prepared_results = await asyncio.shield(prepare_batch)
        except BaseException:
            settled = await self._settle_table_refresh_batch(
                prepare_batch,
                label='table refresh preparation',
            )
            prepared = [
                result for result in (settled or [])
                if isinstance(result, _PreparedTableRefresh)
            ]
            await self._run_table_refresh_cleanup(
                self._revoke_table_refresh_authorizations(issued_refs + [
                    result.execution_ref
                    for result in prepared if result.execution_ref is not None
                ]),
                label='cancelled preparation capability revocation',
            )
            raise
        preparation_error = next((result for result in prepared_results
                                  if isinstance(result, BaseException)), None)
        prepared = [
            result for result in prepared_results
            if isinstance(result, _PreparedTableRefresh)
        ]
        workflow_refs = [
            result.execution_ref for result in prepared
            if result.execution_ref is not None
        ]
        if preparation_error is not None:
            await self._run_table_refresh_cleanup(
                self._revoke_table_refresh_authorizations(issued_refs +
                                                          workflow_refs),
                label='failed preparation capability revocation',
            )
            log.error('Preparing a table refresh batch failed (error_type=%s)',
                      type(preparation_error).__name__)
            raise ConnectorError(
                'Unable to prepare all table refreshes; please retry.'
            ) from None

        launchable = [
            result for result in prepared if result.execution_ref is not None
        ]
        start_batch = asyncio.gather(
            *(DataBackend.start_refresh_table(
                self._data_backend_ctx,
                result.execution_ref,
                result.table_entry.config,
                result.data_version,
                alias=(TableName(result.table_entry.alias)
                       if result.table_entry.alias else None),
                defer_col_stats_validation=defer_col_stats_validation,
            ) for result in launchable), return_exceptions=True)
        try:
            start_results = await asyncio.shield(start_batch)
        except BaseException:
            settled = await self._settle_table_refresh_batch(
                start_batch,
                label='table refresh start',
            )
            started_workflow_ids = [
                result for result in (settled or []) if isinstance(result, str)
            ]
            await self._run_table_refresh_cleanup(
                self._rollback_table_refresh_batch(
                    started_workflow_ids,
                    issued_refs + workflow_refs,
                ),
                label='cancelled table refresh rollback',
            )
            raise
        start_error = next(
            (result
             for result in start_results if isinstance(result, BaseException)),
            None)
        if start_error is not None:
            started_workflow_ids = [
                result for result in start_results if isinstance(result, str)
            ]
            await self._run_table_refresh_cleanup(
                self._rollback_table_refresh_batch(
                    started_workflow_ids,
                    issued_refs + workflow_refs,
                ),
                label='failed table refresh rollback',
            )
            log.error('Starting a table refresh batch failed (error_type=%s)',
                      type(start_error).__name__)
            raise ConnectorError(
                'Unable to start all table refreshes; partial starts were '
                'rolled back. Please retry.') from None
        return [result.data_version for result in prepared]

    async def _settle_table_refresh_batch(
        self,
        batch: asyncio.Future,
        *,
        label: str,
    ) -> list[object] | None:
        """Bound how long cancellation waits for shielded batch operations."""
        try:
            return await asyncio.wait_for(
                asyncio.shield(batch),
                timeout=_TABLE_REFRESH_BATCH_CLEANUP_TIMEOUT_SECONDS,
            )
        except BaseException as error:
            log.error(
                'Settling %s during cleanup failed (error_type=%s)',
                label,
                type(error).__name__,
            )
            if not batch.done():
                batch.cancel()
                batch.add_done_callback(_consume_background_future)
            return None

    async def _run_table_refresh_cleanup(
        self,
        cleanup: Awaitable[object],
        *,
        label: str,
    ) -> None:
        """Run bounded shielded cleanup without replacing a primary error."""
        task = asyncio.ensure_future(cleanup)
        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=_TABLE_REFRESH_BATCH_CLEANUP_TIMEOUT_SECONDS,
            )
        except BaseException as error:
            log.error(
                'Table refresh %s failed (error_type=%s)',
                label,
                type(error).__name__,
            )
            if not task.done():
                task.cancel()
                task.add_done_callback(_consume_background_future)

    async def _prepare_state_services_table_refresh_with_connector(
        self,
        table_entry: TableDBEntryV2,
        execution_ref: ConnectorExecutionRef,
        *,
        refresh_source: bool = True,
    ) -> _PreparedTableRefresh:
        table_def = table_entry.config
        connector_id = table_def.get_connector_id()
        source_table_name = str(table_def.get_source_table_name())
        src_table_data_ver = (
            await get_active_cache().table_version_cache.get.coroutine(
                connector_id, source_table_name))

        # A version probe binds its capability to the one-activity workflow;
        # the later table-refresh workflow must receive a fresh capability. A
        # Custom SQL table's data_version is a content hash over the SQL +
        # output cols + referenced-table versions; the name-keyed
        # table_version_cache cannot represent those, so a SQL table must always
        # (re)compute (which also consumes the version ref's capability) rather
        # than serve a synthetic-name cache hit.
        force_source_refresh = refresh_source or src_table_data_ver is None
        version_ref_consumed = table_def.is_sql or force_source_refresh
        if version_ref_consumed:
            try:
                src_table_data_ver = await DataBackend.get_table_data_version_in_activity(
                    self._data_backend_ctx,
                    execution_ref,
                    source_table_name,
                    refresh_table_version=force_source_refresh,
                    table_def=table_def,
                )
            except (FileNotFoundError, ConnectorError) as err:
                raise _table_read_error_v2(table_def, err)
        data_ver = TableDataVersion(src_table_data_ver)
        refresh_result = (await DataBackend.get_many_table_refresh_results([
            (table_def, data_ver)
        ]))[0]
        if refresh_result:
            refresh_wf_id = TableRefreshWorkflow.compose_wf_id(
                table_def, data_ver)
            if not refresh_result.status.completed:
                tclient = self._data_backend_ctx.temporal_client()
                wf_desc = await tclient.get_workflow_desc(refresh_wf_id)
                if (wf_desc
                        and wf_desc.status == WorkflowExecutionStatus.RUNNING):
                    log.info("Table refresh workflow is already running: %s",
                             refresh_wf_id)
                    return _PreparedTableRefresh(table_entry, data_ver, None)

            if (refresh_result.status == AsyncDataTaskStatus.DONE
                    and refresh_result.stage == TableRefreshStage.COMPLETED
                    and not refresh_source):
                log.info("Table refresh is already completed, wf: %s",
                         refresh_wf_id)
                return _PreparedTableRefresh(table_entry, data_ver, None)

            # Forced re-ingest over an existing (terminal or in-flight) result:
            # synchronously reset all refresh markers to in-progress *before*
            # launching the workflow. The relaunched TableRefreshWorkflow only
            # resets the overall marker, asynchronously, and never the
            # per-stage col-stat markers — so without this a snapshot waiter
            # keyed on (table, data_version) can observe the previous refresh's
            # stale DONE markers and return before the re-ingest has even reset
            # them, surfacing column_stats=None. Resetting here, in the request
            # path, makes "in progress" durable before this call returns so the
            # waiter correctly waits the re-ingest out to COMPLETED. Force
            # re-ingest semantics are unchanged: the workflow still relaunches
            # below.
            if refresh_source:
                await DataBackend.reset_table_refresh_markers(
                    table_def, data_ver)

        if version_ref_consumed:
            execution_ref, = await self._connector_service(
            ).fresh_execution_refs_for_workflow(
                _current_user_ctx.get(),
                (execution_ref, ),
                CONNECTOR_GRAPH_ACTIONS,
            )
        return _PreparedTableRefresh(table_entry, data_ver, execution_ref)

    async def _revoke_table_refresh_authorizations(
        self,
        references: List[ConnectorExecutionRef],
    ) -> None:
        tokens = sorted({
            reference.authorization_token
            for reference in references if reference.authorization_token
        })
        if not tokens:
            return
        try:
            service = self._require_state_services().connector_authorizations
            results = await asyncio.gather(
                *(service.revoke(token) for token in tokens),
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            log.error(
                'Preparing table refresh capability revocation failed '
                '(error_type=%s)',
                type(error).__name__,
            )
            return
        for result in results:
            if isinstance(result, BaseException):
                log.error(
                    'Revoking a table refresh capability failed '
                    '(error_type=%s)',
                    type(result).__name__)

    async def _rollback_table_refresh_batch(
        self,
        workflow_ids: List[str],
        references: List[ConnectorExecutionRef],
    ) -> None:
        temporal_client = self._data_backend_ctx.temporal_client()

        async def terminate(workflow_id: str) -> None:
            # Use the wrapper so the deployment suffix is applied before the
            # handle is constructed.  The start path returns the logical ID.
            handle = None
            for attempt in range(3):
                try:
                    if handle is None:
                        handle = await temporal_client.get_workflow_handle(
                            workflow_id)
                    await handle.terminate(
                        reason='Rolling back partial table refresh batch')
                    return
                except asyncio.CancelledError:
                    raise
                except BaseException as error:
                    if attempt == 2:
                        log.error(
                            'Rolling back table refresh workflow %s '
                            'failed (error_type=%s)', workflow_id,
                            type(error).__name__)
                        return
                    await asyncio.sleep(0.05 * (2**attempt))

        # Revoke authority before potentially slow provider termination.  This
        # is intentionally redundant with the Temporal terminate interceptor
        # and guarantees that a bounded/failed termination cannot leave a
        # confirmed partial start authorized.
        await self._revoke_table_refresh_authorizations(references)
        await asyncio.gather(*(terminate(workflow_id)
                               for workflow_id in sorted(set(workflow_ids))))

    async def get_table_snapshot(
        self,
        id: str,
        project_id: ProjectID,
    ) -> snapshot_interfaces.TableSnapshotResource:
        await self.require_project_access(project_id, _current_user_ctx.get())
        fpid, data_ver = DataSnapshot.TABLE.parse_id(id)
        return await self._get_kumo_table_snapshot(project_id, fpid,
                                                   TableDataVersion(data_ver))

    async def get_table_definition(
        self,
        id: snapshot_interfaces.TableSnapshotID,
        project_id: ProjectID,
    ) -> TableDefinition:
        await self.require_project_access(project_id, _current_user_ctx.get())
        fpid, _ = DataSnapshot.TABLE.parse_id(id)
        table_db_entry = await self._get_table_entry(project_id, fpid)
        if table_db_entry is None:
            raise TableSnapshotDoesNotExistError(id)
        return table_db_entry.config

    async def _get_graph_snapshot_stats(
        self,
        snapshot_id: snapshot_interfaces.GraphSnapshotID,
        project_id: ProjectID,
    ) -> Optional[GraphSnapshotStatsResponse]:
        """
        Aggregate summary statistics across all tables in a graph snapshot.
        This uses the backend cache to avoid expensive table snapshot fetches.
        Args:
            snapshot_id (GraphSnapshotID): The ID of the graph snapshot.
        Returns:
            Optional[GraphSnapshotStatsResponse]: Statistics including total
                rows and size, or None if unavailable.
        """
        # Assemble list of table ids and versions
        graph_snapshot = await self.get_graph_snapshot(
            GraphSnapshotRequest(graph_snapshot_id=snapshot_id), project_id)
        table_ids = []
        table_versions = []
        for (table_name, table_snapshot_id) in (
                graph_snapshot.snapshot_resource.table_ids.items()):
            fpid, table_version = DataSnapshot.TABLE.parse_id(table_snapshot_id)
            table_ids.append(
                TableIDv2.compose(None, TableName(table_name), fpid))
            table_versions.append(table_version)

        # Assemble table definitions
        tables = await self._get_many_table_definitions(project_id, table_ids)

        if any(isinstance(t, Exception) for t in tables):
            return None

        # Fetch table stats
        cache = get_active_cache()

        stats = await cache.get_many_min_exact_stats.coroutine(
            [(table.source_table_id.make_versioned_id(version),
              table.fingerprint())
             for version, table in zip(table_versions, tables)],
            skip_col_stats=True)

        try:
            total_rows = sum(({} if stat is None else stat.table_stats
                              )[TableStatType.NUM_ROWS].value for stat in stats)
        except KeyError:
            total_rows = None

        try:
            total_size_bytes = sum(({} if stat is None else stat.table_stats
                                    )[TableStatType.SOURCE_SIZE].value
                                   for stat in stats)
        except KeyError:
            total_size_bytes = None

        return GraphSnapshotStatsResponse(total_rows=total_rows,
                                          total_size_bytes=total_size_bytes)

    async def _get_kumo_table_snapshot(
            self, project_id: ProjectID, fpid: FPID,
            data_ver: TableDataVersion) -> TableSnapshotResource:
        table_db_entry = await self._get_required_table_entry(project_id, fpid)

        table_db_entries = [table_db_entry]
        table_versions = {fpid: data_ver}
        resolved_table = (table_db_entry.config, data_ver)
        refresh_results = await DataBackend.get_many_table_refresh_results(
            [resolved_table])
        workflow_descs = await self._resolve_snapshot_table_states(
            table_db_entries,
            table_versions,
            [resolved_table],
            refresh_results,
        )
        resolved_def, _ = resolved_table
        table_refresh_result = refresh_results[0]
        preloaded_wf_desc = workflow_descs[0]

        # Construct the response Resource object:
        snapshot_id = DataSnapshot.TABLE.format_id(fpid, data_ver)
        connector_id = resolved_def.get_connector_id()
        resource = TableSnapshotResource(
            id=snapshot_id, table_definition=table_db_entry.config, stages={
                snapshot_interfaces.TableSnapshotStage.INGEST:
                snapshot_interfaces.JobState(),
                snapshot_interfaces.TableSnapshotStage.MIN_COL_STATS:
                snapshot_interfaces.JobState(),
                snapshot_interfaces.TableSnapshotStage.FULL_COL_STATS:
                snapshot_interfaces.JobState(),
            }, name_alias=table_db_entry.alias,
            timestamps=table_db_entry.timestamps,
            connector=ConnectorLink(name=str(connector_id), id=connector_id))

        log.info('Table snapshot for %s.%s@%s has refresh result: %s',
                 connector_id, resolved_def.cache_key(), data_ver,
                 table_refresh_result)

        # There could be a delay between kicking off table refresh workflow
        # (upon creation of table snapshot resource) and writing the table
        # refresh result file (which happens at the beginning of workflow).
        # To resolve this, we check whether the table refresh workflow has
        # been registered with temporal:
        if not table_refresh_result:
            wf_desc = preloaded_wf_desc
            if wf_desc is None:
                refresh_wf_id = TableRefreshWorkflow.compose_wf_id(
                    resolved_def, data_ver)
                wf_desc = await self._temporal_client.get_workflow_desc(
                    refresh_wf_id)
            if not wf_desc:
                # There must be a TableRefreshWorkflow execution associated
                # with this snapshot. Otherwise, most likely the table snapshot
                # id is wrong/mispelled:
                raise KeyError(f'Table snapshot {resource.id} is not found.')

            assert wf_desc.status is not None
            if wf_desc.status == WorkflowExecutionStatus.RUNNING:
                # Table refresh workflow is running but hasn't written
                # table_refresh_result yet
                time_elapsed = now_utc() - wf_desc.start_time
                is_prod = (self._data_backend_ctx._config.deploy_mode ==
                           DeployStage.PROD)

                # Typically the delay is a few seconds. If this takes too long,
                # asynchronously send an alert Kumo to investigate:
                if time_elapsed > timedelta(minutes=10) and is_prod:
                    warning_msg = (
                        f'[WARNING] TableRefreshWorkflow id={wf_desc.id}, '
                        f'run_id={wf_desc.run_id} has been running for: '
                        f'{time_elapsed.total_seconds()} without saving table '
                        f'refresh result, please investigate!')
                    asyncio.ensure_future(
                        send_slack_message.coroutine(warning_msg))
                return resource

            wf_failed = (wf_desc.status in [
                WorkflowExecutionStatus.TIMED_OUT,
                WorkflowExecutionStatus.FAILED
            ])
            wf_canceled = (wf_desc.status in [
                WorkflowExecutionStatus.TERMINATED,
                WorkflowExecutionStatus.CANCELED
            ])
            if wf_canceled or wf_failed:
                ingest_stage = resource.stages[
                    snapshot_interfaces.TableSnapshotStage.INGEST]
                ingest_stage.end_time = wf_desc.close_time
                ingest_stage.status = (JobStatus.CANCELLED
                                       if wf_desc.status in [
                                           WorkflowExecutionStatus.CANCELED,
                                           WorkflowExecutionStatus.TERMINATED
                                       ] else JobStatus.FAILED)

                # Set colstat stage as CANCELED because ingest didn't succeed:
                resource.stages[snapshot_interfaces.TableSnapshotStage.
                                MIN_COL_STATS].status = JobStatus.CANCELLED
                resource.stages[snapshot_interfaces.TableSnapshotStage.
                                FULL_COL_STATS].status = JobStatus.CANCELLED
                return resource

        # Populate INGEST job state:
        ingest_task_result = (await DataBackend.get_many_table_refresh_results(
            tables=[(resolved_def, data_ver)],
            stage=TableRefreshStage.INGEST_TABLE,
        ))[0]
        ingest_stage = resource.stages[
            snapshot_interfaces.TableSnapshotStage.INGEST]
        _update_snapshot_job_state(ingest_stage, ingest_task_result)
        _merge_snapshot_job_state_warnings(ingest_stage, table_refresh_result)
        if ingest_task_result is None or not ingest_task_result.ok:
            return resource

        # table stats can be added after ingestion
        table_stats = get_active_cache().get_table_stats(
            resolved_def.fingerprint(),
            [TableStatType.SOURCE_SIZE, TableStatType.NUM_ROWS], data_ver)
        exact_stats = ExactStats({
            k: v.config()
            for k, v in table_stats.items()
        }, {})
        resource.stats = get_table_stats(exact_stats)

        # MIN_COL_STATS should finish before full column stats; check this
        # state first. Populate MIN_COL_STATS job state:
        min_colstat_task_result = (
            await DataBackend.get_many_table_refresh_results(
                tables=[(resolved_def, data_ver)],
                stage=TableRefreshStage.COMPUTE_MIN_COL_STATS,
            ))[0]
        min_colstat_stage = resource.stages[
            snapshot_interfaces.TableSnapshotStage.MIN_COL_STATS]
        _update_snapshot_job_state(min_colstat_stage, min_colstat_task_result)
        if min_colstat_task_result is None or not min_colstat_task_result.ok:
            if (min_colstat_task_result is not None
                    and min_colstat_task_result.status.completed):
                resource.stages[snapshot_interfaces.TableSnapshotStage.
                                FULL_COL_STATS].status = JobStatus.CANCELLED
            return resource

        # FULL_COL_STATS can be checked iff MIN_COL_STATS has succeeded:
        full_colstat_task_result = (
            await DataBackend.get_many_table_refresh_results(
                tables=[(resolved_def, data_ver)],
                stage=TableRefreshStage.COMPUTE_FULL_COL_STATS,
            ))[0]
        full_colstat_stage = resource.stages[
            snapshot_interfaces.TableSnapshotStage.FULL_COL_STATS]
        _update_snapshot_job_state(full_colstat_stage, full_colstat_task_result)

        # Fetch minimum column stats:
        resource.column_stats = []

        min_stats_dict = ColStatsSet.MINIMAL.get_stats_types(resolved_def)
        stats_dict = await DataBackend.get_min_col_stats(
            connector_spec=None,
            table=resolved_def,
            stats_dict=min_stats_dict,
            approx_sample_size=-1,  # unused.
            compute_missing_stats=False,
            # min_colstat_task_result already checked above
            check_refresh_status=False,
            table_data_version=data_ver)

        # time range must wait for min col stats
        min_col_stats = {
            k.name: {
                k: v.config()
                for k, v in v.items()
            }
            for k, v in stats_dict.items() if k.name == resolved_def.time_col
        }
        exact_stats = ExactStats({}, min_col_stats)
        resource.time_range = get_full_time_range(resolved_def, exact_stats)

        if full_colstat_task_result and full_colstat_task_result.ok:
            full_stats_dict = (
                ColStatsSet.FULL_ENCODER.get_stats_types(resolved_def))

            stats_dict.update(await DataBackend.get_min_col_stats(
                connector_spec=None,
                table=resolved_def,
                stats_dict=full_stats_dict,
                approx_sample_size=-1,  # unused.
                compute_missing_stats=False,
                # full_colstat_task_result already checked above
                check_refresh_status=False,
                table_data_version=data_ver))

        for col in stats_dict:
            stats = stats_dict[col]
            resource.column_stats.append(
                TableSnapshotColumnStats(
                    column_name=col.name,
                    stats={
                        # We must convert to Dict[str, Any] and consumers
                        # must convert back to Dict[ColStatType, Stats]
                        str(k.value):
                        SnapshotStat(**dataclasses.asdict(s.config()))
                        for k, s in stats.items()
                    },
                ))

        return resource

    async def _get_many_kumo_table_snapshots(
        self,
        project_id: ProjectID,
        table_versions: Dict[FPID, TableDataVersion],
    ) -> List[snapshot_interfaces.TableSnapshotResource]:
        """
        Get many TableSnapshot resources that represents a specific data
        version of the table, as well as an associated table refresh workflow.

        Note: Unlike `_get_kumo_table_snapshot`, this method does
        not fetch the min and full column stats for each table snapshot.
        It is currently only used when fetching a graph snapshot resource
        because a graph snapshot resource doesn't need table stats.

        Args:
            table_versions (Dict[FPID, TableDataVersion]): A mapping between
                the table FPID and table version for each table resource.

        Returns:
            List[snapshot_interfaces.TableSnapshotResource]
        """
        # Fetch all table DB entries to construct table responses
        table_ids = list(table_versions.keys())
        table_db_entries = await self._get_many_table_entries(
            project_id, table_ids)
        resolved_tables = [(table.config, table_versions[table.id])
                           for table in table_db_entries]
        resolved_refresh_results = (
            await DataBackend.get_many_table_refresh_results(resolved_tables))
        resolved_workflow_descs = await self._resolve_snapshot_table_states(
            table_db_entries,
            table_versions,
            resolved_tables,
            resolved_refresh_results,
        )

        # There could be a delay between kicking off table refresh workflow
        # (upon creation of table snapshot resource) and writing the table
        # refresh result file (which happens at the beginning of workflow).
        # To resolve this, we check whether the table refresh workflow has
        # been registered with temporal:
        resources: List[snapshot_interfaces.TableSnapshotResource] = []
        for (refresh_result, (t_def, t_ver), table_db_entry,
             preloaded_wf_desc) in zip(resolved_refresh_results,
                                       resolved_tables, table_db_entries,
                                       resolved_workflow_descs):
            # Construct response Resource object array
            data_ver = t_ver
            snapshot_id = DataSnapshot.TABLE.format_id(table_db_entry.id,
                                                       data_ver)
            resource = snapshot_interfaces.TableSnapshotResource(
                id=snapshot_id,
                table_definition=table_db_entry.config,
                stages={
                    snapshot_interfaces.TableSnapshotStage.INGEST:
                    snapshot_interfaces.JobState(),
                    snapshot_interfaces.TableSnapshotStage.MIN_COL_STATS:
                    snapshot_interfaces.JobState(),
                    snapshot_interfaces.TableSnapshotStage.FULL_COL_STATS:
                    snapshot_interfaces.JobState(),
                },
            )

            if not refresh_result:
                wf_desc = preloaded_wf_desc
                if wf_desc is None:
                    refresh_wf_id = TableRefreshWorkflow.compose_wf_id(
                        t_def, data_ver)
                    wf_desc = await self._temporal_client.get_workflow_desc(
                        refresh_wf_id)
                if not wf_desc:
                    # There must be a TableRefreshWorkflow execution associated
                    # with this snapshot. Otherwise, most likely
                    # the table snapshot id is wrong/mispelled:
                    raise KeyError(
                        f'Table snapshot {resource.id} is not found.')

                assert wf_desc.status is not None
                if wf_desc.status == WorkflowExecutionStatus.RUNNING:
                    # Table refresh workflow is running but hasn't written
                    # table_refresh_result yet
                    time_elapsed = now_utc() - wf_desc.start_time
                    is_prod = (self._data_backend_ctx._config.deploy_mode ==
                               DeployStage.PROD)

                    # Typically the delay is a few seconds. If this takes
                    # too long, asynchronously send an alert
                    # Kumo to investigate:
                    if time_elapsed > timedelta(minutes=10) and is_prod:
                        warning_msg = (
                            f'[WARNING] TableRefreshWorkflow id={wf_desc.id}, '
                            f'run_id={wf_desc.run_id} has been running for: '
                            f'{time_elapsed.total_seconds()} without saving '
                            f'table refresh result, please investigate!')
                        asyncio.ensure_future(
                            send_slack_message.coroutine(warning_msg))
                    resources.append(resource)
                    continue

                wf_failed = (wf_desc.status in [
                    WorkflowExecutionStatus.TIMED_OUT,
                    WorkflowExecutionStatus.FAILED
                ])
                wf_canceled = (wf_desc.status in [
                    WorkflowExecutionStatus.TERMINATED,
                    WorkflowExecutionStatus.CANCELED
                ])
                if wf_canceled or wf_failed:
                    ingest_stage = resource.stages[
                        snapshot_interfaces.TableSnapshotStage.INGEST]
                    ingest_stage.end_time = wf_desc.close_time
                    ingest_stage.status = (
                        JobStatus.CANCELLED if wf_desc.status in [
                            WorkflowExecutionStatus.CANCELED,
                            WorkflowExecutionStatus.TERMINATED
                        ] else JobStatus.FAILED)

                    # Set colstat stage as CANCELED because
                    # ingest didn't succeed:
                    resource.stages[snapshot_interfaces.TableSnapshotStage.
                                    MIN_COL_STATS].status = JobStatus.CANCELLED
                    resource.stages[snapshot_interfaces.TableSnapshotStage.
                                    FULL_COL_STATS].status = JobStatus.CANCELLED
                    resources.append(resource)
                    continue
            else:
                resources.append(resource)

        # Populate INGEST job state:
        ingest_task_results = (await DataBackend.get_many_table_refresh_results(
            resolved_tables,
            TableRefreshStage.INGEST_TABLE,
        ))
        for (ingest_result, refresh_result,
             resource) in zip(ingest_task_results, resolved_refresh_results,
                              resources):
            ingest_stage = resource.stages[
                snapshot_interfaces.TableSnapshotStage.INGEST]
            _update_snapshot_job_state(ingest_stage, ingest_result)
            _merge_snapshot_job_state_warnings(ingest_stage, refresh_result)

        return resources

    async def _resolve_snapshot_table_states(
        self,
        table_db_entries: List[TableDBEntryV2],
        table_versions: Dict[FPID, TableDataVersion],
        resolved_tables: List[Tuple[TableDefinition, TableDataVersion]],
        refresh_results: List[Optional[TableDataTaskResult]],
    ) -> List[Optional[WorkflowExecutionDescription]]:
        """Verify that every immutable table snapshot exists.

        A snapshot is represented either by a persisted refresh result or by an
        existing Temporal workflow. Connector identity is part of the immutable
        table definition, so there is no alternate-connector candidate search.
        """
        workflow_descs: List[Optional[WorkflowExecutionDescription]] = [
            None
        ] * len(resolved_tables)
        missing_indexes = [
            index for index, result in enumerate(refresh_results)
            if result is None
        ]
        missing_workflows = await asyncio.gather(*[
            self._temporal_client.get_workflow_desc(
                TableRefreshWorkflow.compose_wf_id(*resolved_tables[index]))
            for index in missing_indexes
        ])
        for index, workflow_desc in zip(missing_indexes, missing_workflows):
            if workflow_desc is None:
                table_entry = table_db_entries[index]
                snapshot_id = DataSnapshot.TABLE.format_id(
                    table_entry.id, table_versions[table_entry.id])
                raise KeyError(f'Table snapshot {snapshot_id} is not found.')
            workflow_descs[index] = workflow_desc
        return workflow_descs


def _update_snapshot_job_state(job_state: snapshot_interfaces.JobState,
                               stage_result: Optional[TableDataTaskResult]):
    if (not stage_result
            or stage_result.status == AsyncDataTaskStatus.NOT_STARTED):
        job_state.status = JobStatus.NOT_STARTED
    elif stage_result.status == AsyncDataTaskStatus.IN_PROGRESS:
        job_state.status = JobStatus.RUNNING
    elif stage_result.status.completed:
        job_state.start_time = stage_result.created_at
        job_state.end_time = stage_result.updated_at
        job_state.status = JobStatus(stage_result.status.name)
        # Populate error/warning.
        job_state.error = (stage_result.error
                           or stage_result.validation.error_message() or None)
        if job_state.error:
            job_state.status = JobStatus.FAILED
        job_state.warnings = [
            f'{w.title}: {w.message}' for w in stage_result.validation.warnings
        ]


def _merge_snapshot_job_state_warnings(
    job_state: snapshot_interfaces.JobState,
    refresh_result: Optional[TableDataTaskResult],
) -> None:
    if refresh_result is None:
        return
    for warning in refresh_result.validation.warnings:
        message = f'{warning.title}: {warning.message}'
        if message not in job_state.warnings:
            job_state.warnings.append(message)


def _merge_table_ingest_job_states(
    table_snapshots: List[snapshot_interfaces.TableSnapshotResource]
) -> snapshot_interfaces.JobState:
    """Merge per-table INGEST job states into one graph-level JobState.

    The merged status is RUNNING until a table fails or is cancelled or
    all tables are DONE. A terminal merged state spans from the earliest
    table start_time to the latest table end_time, independent of table
    iteration order, and aggregates per-table errors and warnings.
    """
    job_states = [
        t.stages[snapshot_interfaces.TableSnapshotStage.INGEST]
        for t in table_snapshots
    ]

    failed = any(s.status == snapshot_interfaces.JobStatus.FAILED
                 for s in job_states)
    canceled = any(s.status == snapshot_interfaces.JobStatus.CANCELLED
                   for s in job_states)
    all_done = all(s.status == snapshot_interfaces.JobStatus.DONE
                   for s in job_states)

    if not all_done and not failed and not canceled:
        return snapshot_interfaces.JobState(
            snapshot_interfaces.JobStatus.RUNNING)

    if failed:
        status = snapshot_interfaces.JobStatus.FAILED
    elif canceled:
        status = snapshot_interfaces.JobStatus.CANCELLED
    else:
        assert all_done
        status = snapshot_interfaces.JobStatus.DONE

    end_time = None
    start_time = None
    errors = []
    warnings = []
    for table_snapshot_id, job_state in zip([t.id for t in table_snapshots],
                                            job_states):
        if end_time is None:
            end_time = job_state.end_time
        elif job_state.end_time:
            end_time = max(end_time, job_state.end_time)

        if start_time is None:
            start_time = job_state.start_time
        elif job_state.start_time:
            start_time = min(start_time, job_state.start_time)

        if job_state.error:
            errors.append(job_state.error)
        elif job_state.status in [snapshot_interfaces.JobStatus.CANCELLED]:
            errors.append(
                f'{table_snapshot_id} ended in {job_state.status} state')
        if job_state.warnings:
            warning_msg = '\n'.join(job_state.warnings)
            warnings.append(f'{table_snapshot_id} has warnings: {warning_msg}')

    error = '\n'.join(errors) if errors else None
    return snapshot_interfaces.JobState(status, start_time, end_time, warnings,
                                        error)


def _merge_graph_ingest_job_states(
    table_state: snapshot_interfaces.JobState,
    graph_refresh_state: snapshot_interfaces.JobState,
) -> snapshot_interfaces.JobState:
    """Include direct DiskGraph ingest in the public graph INGEST stage."""
    if table_state.status != JobStatus.DONE:
        return table_state

    start_times = [
        value for value in (
            table_state.start_time,
            graph_refresh_state.start_time,
        ) if value is not None
    ]
    end_times = [
        value for value in (
            table_state.end_time,
            graph_refresh_state.end_time,
        ) if value is not None
    ]
    graph_refresh_state.start_time = min(start_times) if start_times else None
    graph_refresh_state.end_time = max(end_times) if end_times else None
    graph_refresh_state.warnings = [
        *table_state.warnings,
        *graph_refresh_state.warnings,
    ]
    return graph_refresh_state
