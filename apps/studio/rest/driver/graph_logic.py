# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
from collections import defaultdict
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import kumoapi.graph as graph_api
from deprecated import deprecated
from kumo_core.types import ConnectorType, TableStatType
from kumo_core.types.schema import FPID
from kumoapi.source_table import UnavailableSourceTable
from kumoapi.table import TableDefinition as ApiTableDefinition

from async_utils.instrument import measure_blocking_code
from async_utils.sync import blocking_method
from io_utils import logging
from kumo.config.predictive_query import PredictiveQueryV2Config
from kumo.connector import Connector
from kumo.data import get_active_cache
from kumo.data.backend_cache import ExactStats
from kumo.graph import Column, ForeignKey, PrimaryKey
from kumo.graph.graph import Graph
from kumo.pquery import PredictiveQuery, PredictiveQueryV2
from kumo.utils import batch_ops
from kumo.validation import ValidationException
from monitoring.tracing import TraceOptions, trace
from rest.backends.data_backend import connector_activity_registry
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.data_backend import DataBackend
from rest.config.data_backend import (
    ConnectorExecutionRef,
    ConnectorOpenSpec,
    ConnectorSpec,
    GraphDataVersionInfo,
)
from rest.config.graph import EdgeHealthStats, GraphResponse
from rest.config.graph_v2 import (
    CanonicalGraphDBEntry,
    ColumnKey,
    ColumnKeyGroup,
    GraphDataComputeInput,
    GraphDBEntryV2,
    GraphDefinition,
    ListResponse,
)
from rest.config.table_v2 import (
    TableDBEntryV2,
    TableDefinition,
    TableReferenceMetadata,
    TableResourceResponse,
    column_def_to_foreign_key,
)
from rest.deps.context import _current_user_ctx
from rest.driver import table_logic
from rest.driver.error_constants import (
    CANNOT_INFER_LINKS_WITH_EDGES_MESSAGE,
    COLUMN_GROUP_MISSING_COLUMN_MESSAGE,
    COLUMN_GROUP_MULTIPLE_PKEYS_MESSAGE,
    COLUMN_GROUP_NO_PKEY_MESSAGE,
    INVALID_GRAPH_PARAMETERS_MESSAGE,
    INVALID_GRAPH_PARAMETERS_TITLE,
)
from rest.utils.driver_common import get_connector_from_spec
from rest.utils.ui_to_sdk_utils import (
    generate_conn_graph_or_table_sdk_script,
    make_stype_and_dtype_sdk_compat,
)
from statedb.config.common import (
    ConnectorID,
    GraphID,
    GraphIDv1,
    GraphIDv2,
    PredictiveQueryID,
    ProjectID,
    TableID,
    TableIDv2,
    TableName,
)
from statedb.config.table import TableDataVersion
from statedb.db.services import StateServices
from statedb.utils.errors import GraphDoesNotExistError, TableDoesNotExistError

log = logging.getLogger(__name__)


async def canonicalize_graph_custom_sql_tables(
    graph: GraphDefinition,
    project_id: ProjectID,
    state_services: StateServices,
    principal: Any,
) -> GraphDefinition:
    """Route every embedded Custom SQL table through the same server-
    authoritative ``connector_type`` derivation + synthetic-name canonicalization
    the standalone table-create path applies, before the graph's fingerprint is
    taken or any embedded definition is persisted.

    A graph create embeds full ``TableDefinition``s and persists them directly
    (``save_graph_definition`` -> ``_save_table_definition_in_tx``), bypassing
    ``TableDriver.create_full_table``. Without this pass an embedded Custom SQL
    table would persist with an unstamped (client-omitted) ``connector_type`` and
    under the SDK's provisional ``md5(raw sql)`` name, and a file/blob connector
    that has no SQL engine would never be rejected.

    The connector *type* resolves from credential-free connector metadata
    (``get_for_project`` -> ``config.type.value`` — no Temporal activity or
    credentials, only the warehouse dialect), so
    ``canonicalize_custom_sql_source`` derives each SQL table's server-owned
    ``connector_type`` (rejecting a non-warehouse) and stamps the canonical
    synthetic name + parsed referenced tables. This is a cheap
    correctness/security gate applied by every graph-persistence entry
    point (``GraphDriver.create_full_graph_v2`` and
    ``GraphLogic.create_full_graph_def_v2``); it must run unconditionally and NOT
    be gated by ``skip_validation`` (which only gates the expensive
    table-size/target/permission checks). A no-op for a graph with no Custom SQL
    tables.
    """
    tables = graph._tables()
    if not any(table_def.is_sql for table_def in tables.values()):
        return graph
    connectors = state_services.connectors
    canonicalized: Dict[str, TableDefinition] = {}
    changed = False
    for table_name, table_def in tables.items():
        if not table_def.is_sql:
            canonicalized[table_name] = table_def
            continue
        connector = await connectors.get_for_project(
            project_id, table_def.get_connector_id(), principal)
        connector_type = ConnectorType(connector.config.type.value)
        canonical = table_logic.canonicalize_custom_sql_source(
            table_def, connector_type)
        canonicalized[table_name] = canonical
        changed = changed or canonical is not table_def
    if not changed:
        return graph
    return dataclasses.replace(graph, tables=canonicalized)


class GraphLogic:
    def __init__(
        self,
        data_backend_ctx: DataBackendContext,
        *,
        state_services: Optional[StateServices] = None,
        project_id: ProjectID,
    ):
        self._data_backend_ctx = data_backend_ctx
        self._state_services = state_services
        self._project_id = project_id
        self._tbl_logic = table_logic.TableLogic(
            data_backend_ctx,
            state_services=state_services,
            project_id=project_id,
        )

    @property
    def _graph_service(self):
        return self._require_state_services().graphs

    @property
    def _table_service(self):
        return self._require_state_services().tables

    @property
    def _pquery_service(self):
        return self._require_state_services().pqueries

    def _require_state_services(self) -> StateServices:
        if self._state_services is None:
            raise RuntimeError("GraphLogic requires StateServices")
        return self._state_services

    def _require_graph_service(self):
        return self._graph_service

    def _require_table_service(self):
        return self._table_service

    def _run_state_service(self, coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        close = getattr(coro, "close", None)
        if close is not None:
            close()
        raise RuntimeError(
            "GraphLogic synchronous graph storage helpers require "
            "StateServices and must be called outside a running event loop; "
            "async callers should use async GraphLogic/StateServices methods.")

    def _run_async_graph_logic(self, coro):
        return self._run_state_service(coro)

    async def _resolve_graph_fpid_from_state_services(
        self,
        graph_id: GraphIDv2,
    ) -> Optional[FPID]:
        pquery_id, graph_fpid = GraphIDv2.parse(graph_id)
        if graph_fpid is not None:
            return graph_fpid
        if pquery_id is None:
            return None

        pquery_def = await self._pquery_service.get_definition_by_id(pquery_id)
        return pquery_def.graph_fpid if pquery_def is not None else None

    async def _resolve_graph_fpids_from_state_services(
        self,
        graph_ids: Sequence[GraphIDv2],
    ) -> Dict[GraphIDv2, FPID]:
        result: Dict[GraphIDv2, FPID] = {}
        graph_ids_by_pquery_id: Dict[PredictiveQueryID,
                                     List[GraphIDv2]] = (defaultdict(list))
        for graph_id in graph_ids:
            pquery_id, graph_fpid = GraphIDv2.parse(graph_id)
            if graph_fpid is not None:
                result[graph_id] = graph_fpid
                continue
            if pquery_id is None:
                continue
            graph_ids_by_pquery_id[pquery_id].append(graph_id)

        if not graph_ids_by_pquery_id:
            return result

        pquery_defs_by_id = (
            await self._get_pquery_definitions_by_ids_from_state_services(
                graph_ids_by_pquery_id.keys()))

        for pquery_id, pquery_graph_ids in graph_ids_by_pquery_id.items():
            pquery_def = pquery_defs_by_id.get(pquery_id)
            if pquery_def is None:
                continue
            for graph_id in pquery_graph_ids:
                result[graph_id] = pquery_def.graph_fpid
        return result

    async def _get_pquery_definitions_by_ids_from_state_services(
        self,
        pquery_ids: Iterable[PredictiveQueryID],
    ) -> Dict[PredictiveQueryID, Any]:
        unique_ids = list(dict.fromkeys(pquery_ids))
        if not unique_ids:
            return {}
        return await self._pquery_service.get_definitions_by_ids(unique_ids)

    async def _get_graph_entries_by_fpid_from_state_services(
        self,
        graph_fpids: Iterable[FPID],
    ) -> Dict[FPID, GraphDBEntryV2]:
        fpids = set(graph_fpids)
        return await self._graph_service.get_graph_entries(
            self._project_id, fpids)

    async def _get_graph_entry_v2_from_state_services(
        self,
        graph_id: GraphIDv2,
    ) -> Optional[GraphDBEntryV2]:
        graph_fpid = await self._resolve_graph_fpid_from_state_services(graph_id
                                                                        )
        if graph_fpid is None:
            return None
        return await self._graph_service.get_graph_entry(
            self._project_id, graph_fpid)

    async def _get_many_graph_entries_v2_from_state_services(
        self,
        ids: List[GraphIDv2],
    ) -> List[Optional[CanonicalGraphDBEntry]]:
        gid_to_fpid = await self._resolve_graph_fpids_from_state_services(ids)

        entries_by_fpid = (await
                           self._get_graph_entries_by_fpid_from_state_services(
                               gid_to_fpid.values()))

        result: List[Optional[CanonicalGraphDBEntry]] = []
        for gid in ids:
            graph_fpid = gid_to_fpid.get(gid)
            if graph_fpid is None:
                result.append(None)
                continue
            graph_entry = entries_by_fpid.get(graph_fpid)
            if graph_entry is None:
                result.append(None)
                continue
            pquery_id, _ = GraphIDv2.parse(gid)
            result.append(graph_entry.as_canonical_entry(pquery_id))
        return result

    async def _make_graph_from_definition_with_state_services(
        self,
        graph_def: GraphDefinition,
        name: str,
    ) -> Graph:
        specs = await self._get_connector_specs_for_graph_def(graph_def)
        return _build_graph_from_definition_with_specs(
            graph_def,
            name,
            specs.values(),
        )

    async def _get_connector_specs_for_graph_def(
        self,
        graph_def: GraphDefinition,
    ) -> Dict[ConnectorID, ConnectorOpenSpec]:
        connector_service = self._require_state_services().connectors
        return {
            connector_id:
            await
            connector_service.config_for_project(_current_user_ctx.get(),
                                                 self._project_id, connector_id)
            for connector_id in connector_service.connector_ids_for_graph(
                graph_def)
        }

    async def create_full_graph_def_v2(
        self,
        graph_def: GraphDefinition,
        *,
        alias: Optional[str] = None,
        force_rename: bool = False,
        skip_validation: bool = False,
        created_by_name: str = "",
    ) -> Graph:
        r"""Create full graph definition in the V2 object model; this creates
        the graph and all its tables. No-op if any objects already exist, as
        determined by their fingerprint.

        A graph can be set with a name `alias` as an additional identifier.
        However, unless `force_rename` is set, a ValueError will be raised
        when:
            1) the graph entry already has a different name alias; and
            2) there is a name collision, i.e., another graph has the same
                `alias`
        """
        # Validate + canonicalize embedded Custom SQL tables before the graph's
        # fingerprint is computed or any definition is persisted. This runs
        # unconditionally (independent of ``skip_validation``): it is the cheap
        # connector_type correctness/security gate the table-create path applies,
        # and this pquery-driven path must not persist an unvalidated one.
        graph_def = await canonicalize_graph_custom_sql_tables(
            graph_def, self._project_id, self._require_state_services(),
            _current_user_ctx.get())
        # NOTE `fpid` is a cached property and may perform some computation:
        graph_fpid = graph_def.fpid
        graph_service = self._require_graph_service()
        existing_entry = await graph_service.get_graph_entry(
            self._project_id, graph_fpid)
        await graph_service.save_graph_definition(
            self._project_id,
            graph_def,
            alias=alias,
            force_rename=force_rename,
            created_by_name=created_by_name,
        )
        if existing_entry is not None:
            return
        graph = await self._make_graph_from_definition_with_state_services(
            graph_def,
            "NA",
        )
        if not skip_validation:
            await self.validate_graph(graph)
        return graph

    async def _make_graph_from_graph_def_v2(
            self, graph_def: GraphDefinition) -> Graph:
        return await self._make_graph_from_definition_with_state_services(
            graph_def=graph_def,
            name="NA",
        )

    @deprecated("Will be obsolete after db migration.")
    async def infer_links(
        self,
        graph_def: GraphDefinition,
    ) -> GraphDefinition:
        if graph_def.col_groups is not None and len(graph_def.col_groups) > 0:
            raise ValueError(CANNOT_INFER_LINKS_WITH_EDGES_MESSAGE)

        graph = await self._make_graph_from_graph_def_v2(graph_def)
        groups = graph.infer_col_groups()
        for grp in groups:
            pkey_table = TableIDv2.parse(grp.pkey.table + '-NA')[1]
            cols = [ColumnKey(table_name=pkey_table, col_name=grp.pkey.col)]
            for fkey in grp.fkeys:
                fkey_table = TableIDv2.parse(fkey.table + '-NA')[1]
                cols.append(ColumnKey(table_name=fkey_table, col_name=fkey.col))
            group = ColumnKeyGroup(columns=tuple(cols))

            graph_def.col_groups.append(group)

        return graph_def

    @deprecated("Will be obsolete after db migration.")
    async def validate_graph_definition(self, graph_def: GraphDefinition):
        graph: Graph = await self.make_graph_from_definition.coroutine(
            graph_def=graph_def,
            name="NA",
        )
        await self.validate_graph(graph)

    @deprecated("Will be obsolete after db migration.")
    async def validate_graph(self, graph: Graph):
        try:
            await graph.validate.coroutine(raise_on_error=True,
                                           ignore_encoders=True,
                                           enforce_connected=True)
        except ValidationException as e:
            raise ValidationException(
                title=INVALID_GRAPH_PARAMETERS_TITLE,
                message=INVALID_GRAPH_PARAMETERS_MESSAGE.format(e),
            )

    @trace(TraceOptions(trace_params=True))
    async def get_many_data_compute_inputs(
        self, graph_ids: List[GraphIDv2]
    ) -> List[Union[GraphDataComputeInput, Exception]]:
        return await self._require_graph_service(
        ).get_many_graph_data_compute_inputs(
            graph_ids,
            project_id=self._project_id,
            user_authorization=_current_user_ctx.get(),
        )

    @trace(TraceOptions(trace_params=True))
    async def get_graph_data_compute_input(
        self,
        graph_id: GraphIDv2,
    ) -> GraphDataComputeInput:
        return await self._require_graph_service().get_graph_data_compute_input(
            graph_id,
            project_id=self._project_id,
            user_authorization=_current_user_ctx.get(),
        )

    async def _get_num_rows(self, table_ids: List[TableID]) -> Optional[int]:
        tables_and_exc = await self._table_logic.get_many_table_definitions(
            table_ids)
        tables = batch_ops.raise_errors(
            tables_and_exc, "Failed to fetch some tables in the graph")
        cache = get_active_cache()
        maybe_stats = await cache.get_many_min_latest_exact_stats.coroutine(
            tables,
            fetch_column_stats=False,
        )
        stats, nones = batch_ops.remove_nones(maybe_stats)
        return self._sum_num_rows(stats, nones)

    @staticmethod
    def _sum_num_rows(
        stats: List[ExactStats],
        nones: List[int],
    ) -> Optional[int]:
        """Take the sum of num rows in all tables."""
        # Return None if some of the stats have not yet been computed.
        if len(nones) > 0:
            return None

        count = 0
        for stat in stats:
            if TableStatType.NUM_ROWS not in stat.table_stats:
                # This should never happen, since NUM_ROWS should always be
                # computed. If this happens, then there is a bug.
                log.error("TableStatType.NUM_ROWS is missing")
                return None
            count += int(stat.get_table_stat(TableStatType.NUM_ROWS, 0))

        return count

    def _get_table_adj_list(
            self, graph: CanonicalGraphDBEntry) -> Dict[TableID, Set[TableID]]:
        """Get table adjacency list of a graph."""
        adj_list: Dict[TableID, Set[TableID]] = defaultdict(set)
        for col_group in graph.col_groups:
            connected_table_ids: Set[TableID] = {
                graph.table_name_to_id[col.table_name]
                for col in col_group.columns
            }
            for table_id in connected_table_ids:
                for adj_table_id in connected_table_ids:
                    if table_id != adj_table_id:
                        adj_list[table_id].add(adj_table_id)
        return adj_list

    def get_bfs_tree(self, graph_id: GraphID,
                     root_table_name: str) -> Dict[TableID, List[TableID]]:
        return self._run_async_graph_logic(
            self.async_get_bfs_tree(graph_id, root_table_name))

    async def async_get_bfs_tree(
        self,
        graph_id: GraphID,
        root_table_name: str,
    ) -> Dict[TableID, List[TableID]]:
        """Convert graph to a BFS tree, starting traverse from the root."""
        bfs_tree: Dict[TableID, List[TableID]] = defaultdict(list)
        graph = await self.async_get_graph_entry(graph_id)
        root_table_id = graph.table_name_to_id[TableName(root_table_name)]
        adj_dict = self._get_table_adj_list(graph)
        queue = [root_table_id]
        visited = set([root_table_id])
        while queue:
            table_name = queue.pop(0)
            neighbors = adj_dict[table_name]
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    bfs_tree[table_name].append(neighbor)
        return bfs_tree

    async def get_graph(self, graph_id: GraphID) -> Optional[GraphResponse]:
        graph_service = self._graph_service
        if graph_service is not None and isinstance(graph_id, GraphIDv2):
            pquery_id, _ = GraphIDv2.parse(graph_id)
            graph_entry = await self._get_graph_entry_v2_from_state_services(
                graph_id)
            if graph_entry is None:
                return None
            graph = graph_entry.as_canonical_entry(pquery_id)
            table_ids = list(graph.table_name_to_id.values())
            table_ids_have_fpids = all(
                isinstance(table_id, TableIDv2)
                and TableIDv2.parse(table_id)[2] is not None
                for table_id in table_ids)
            num_rows = (await self._get_num_rows(table_ids)
                        if table_ids_have_fpids else None)
            num_links = sum(len(grp.columns) - 1 for grp in graph.col_groups)
            return GraphResponse(
                id=graph_id,
                name=graph.display_name,
                table_ids=table_ids,
                col_groups=graph.column_groups,
                num_tables=len(graph.table_name_to_id),
                num_links=num_links,
                timestamps=graph.timestamps,
                num_rows=num_rows,
                description=graph.display_description,
            )

        try:
            graph = await self.async_get_graph_entry(graph_id)
        except GraphDoesNotExistError:
            return None

        table_ids = list(graph.table_name_to_id.values())
        num_rows = await self._get_num_rows(table_ids)

        # This assumes that each column group has exactly 1 primary key,
        # and all fkeys are linked to that pkey (star graph)
        num_links = sum(len(grp.columns) - 1 for grp in graph.col_groups)

        return GraphResponse(id=graph_id, name=graph.display_name,
                             table_ids=table_ids,
                             col_groups=graph.column_groups, num_tables=len(
                                 graph.table_name_to_id), num_links=num_links,
                             timestamps=graph.timestamps, num_rows=num_rows,
                             description=graph.display_description)

    def contains_view(self, id: GraphID) -> bool:
        # Kumo views are no longer supported.
        assert self.exists(id)
        return False

    async def async_contains_view(self, id: GraphID) -> bool:
        # Kumo views are no longer supported.
        assert await self.async_exists(id)
        return False

    async def get_graph_def(self, id: GraphID) -> Optional[GraphDefinition]:
        assert isinstance(id, GraphIDv2)
        graph_service = self._require_graph_service()
        graph_fpid = await self._resolve_graph_fpid_from_state_services(id)
        if graph_fpid is None:
            return None
        return await graph_service.get_graph_definition(self._project_id,
                                                        graph_fpid)

    @deprecated("Will be obsolete after db migration.")
    async def get_edge_stats(
        self, id: GraphID,
        graph_version_info: GraphDataVersionInfo | None = None
    ) -> EdgeHealthStats:
        edge_stats = await self._get_edge_stats(id, graph_version_info)
        if isinstance(id, GraphIDv1):
            return edge_stats

        assert isinstance(id, GraphIDv2)
        pquery_id, _ = GraphIDv2.parse(id)
        if edge_stats.link_stats:
            for edge in edge_stats.link_stats:
                if edge.col_a:
                    edge.col_a.table_id = TableIDv2.update(
                        TableIDv2(edge.col_a.table_id), pquery_id, False)
                if edge.col_b:
                    edge.col_b.table_id = TableIDv2.update(
                        TableIDv2(edge.col_b.table_id), pquery_id, False)
        return edge_stats

    async def get_table_resources_in_graph(
        self,
        id: GraphID,
    ) -> List[TableResourceResponse]:
        """Return the tables in a graph as v2 ``TableResourceResponse``
        (each carrying a ``TableDefinition`` plus stats).

        Used by callers that need table identity, column count, and
        data-scale stats without a v1 downgrade.
        """
        if not isinstance(id, GraphIDv2):
            raise GraphDoesNotExistError(id)
        graph_service = self._require_graph_service()
        table_service = self._require_table_service()
        graph_fpid = await self._resolve_graph_fpid_from_state_services(id)
        if graph_fpid is None:
            raise GraphDoesNotExistError(id)
        graph_entry = await graph_service.get_graph_entry(
            self._project_id, graph_fpid)
        if graph_entry is None:
            raise GraphDoesNotExistError(id)
        entries_by_fpid = await table_service.get_table_entries(
            self._project_id, graph_entry.table_name_to_dbid.values())
        resolved: List[TableDBEntryV2] = []
        for table_fpid in graph_entry.table_name_to_dbid.values():
            entry = entries_by_fpid.get(table_fpid)
            if entry is None:
                raise TableDoesNotExistError(table_fpid)
            resolved.append(entry)
        return await self._table_logic._get_many_table_response_v2(
            resolved, fetch_column_stats=False)

    async def _get_edge_stats(
            self, graph_id: GraphID,
            graph_version_info: GraphDataVersionInfo | None) -> EdgeHealthStats:
        data_compute_input = await self.get_graph_data_compute_input(graph_id)
        link_stats = await DataBackend.get_edge_stats(self._data_backend_ctx,
                                                      data_compute_input,
                                                      graph_version_info)

        return EdgeHealthStats(is_ready=link_stats is not None,
                               link_stats=link_stats)

    def exists(self, id: GraphID) -> bool:
        return self._run_async_graph_logic(self.async_exists(id))

    async def async_exists(self, id: GraphID) -> bool:
        if isinstance(id, GraphIDv1):
            raise RuntimeError(
                "GraphLogic StateServices graph storage does not support "
                "v1 graph IDs.")

        assert isinstance(id, GraphIDv2)
        return await self.async_get_graph_entry_v2(id) is not None

    def _get_graph_entry_v1(self, id: GraphIDv1) -> None:
        raise RuntimeError(
            "GraphLogic StateServices graph storage does not support "
            "v1 graph IDs.")

    async def async_get_graph_entry_v2(
        self,
        id: GraphIDv2,
    ) -> Optional[GraphDBEntryV2]:
        return await self._get_graph_entry_v2_from_state_services(id)

    def get_many_graph_entries(
        self,
        graph_ids: Sequence[GraphID],
    ) -> List[Optional[CanonicalGraphDBEntry]]:
        return self._run_async_graph_logic(
            self.async_get_many_graph_entries(graph_ids))

    async def async_get_many_graph_entries(
        self,
        graph_ids: Sequence[GraphID],
    ) -> List[Optional[CanonicalGraphDBEntry]]:
        v1_ids, v2_ids = _split_graph_id_versions(graph_ids)
        if v1_ids:
            raise RuntimeError(
                "GraphLogic StateServices graph storage does not support "
                "v1 graph IDs.")

        v1_result: List[Optional[CanonicalGraphDBEntry]] = []
        v2_result = await self.async_get_many_graph_entries_v2(v2_ids)

        # Merge together
        return _merge_graph_id_versions(graph_ids, v1_result, v2_result)

    def get_graph_entry(self, graph_id: GraphID) -> CanonicalGraphDBEntry:
        return self._run_async_graph_logic(self.async_get_graph_entry(graph_id))

    async def async_get_graph_entry(
        self,
        graph_id: GraphID,
    ) -> CanonicalGraphDBEntry:
        entry = (await self.async_get_many_graph_entries([graph_id]))[0]
        if entry is None:
            raise GraphDoesNotExistError(graph_id)
        return entry

    def _get_many_graph_entries_v1(
            self,
            ids: List[GraphIDv1]) -> List[Optional[CanonicalGraphDBEntry]]:
        if ids:
            raise RuntimeError(
                "GraphLogic StateServices graph storage does not support "
                "v1 graph IDs.")
        return []

    async def async_get_many_graph_entries_v2(
        self,
        ids: List[GraphIDv2],
    ) -> List[Optional[CanonicalGraphDBEntry]]:
        r"""Multi-get operation to fetch Graph DB entries from StateDB.
        Note that the return type is :obj:`CanonicalGraphDBEntry`, a
        unification of V1 and V2 entries for easier typing and downstream
        logic."""

        return await self._get_many_graph_entries_v2_from_state_services(ids)

    @trace(TraceOptions(trace_params=True))
    @measure_blocking_code
    async def get_graph_object(self, id: GraphID) -> Optional[Graph]:
        r"""Returns a Kumo backend Graph object, created from the DB entry
        associated with the input graph ID. Specifically, this function:
            * Creates a `GraphDataComputeInput` object from `id`, fetching
                information from the DB
            * Casts `CreateGraphDataInput` to a `Graph`
            * Returns the `Graph` or `None` if no graph exists.
        """
        try:
            graph = await self.get_graph_data_compute_input(id)
        except GraphDoesNotExistError:
            return None
        return await self._make_graph_from_graph_data_compute_input(graph)

    def list_graph_names_v2(self) -> List[ListResponse]:
        return self._run_async_graph_logic(self.async_list_graph_names_v2())

    async def async_list_graph_names_v2(self) -> List[ListResponse]:
        r"""Returns an array of `ListReponse`s. Used for
        listing all the graph template names in the new
        graph page.

        Return:
            `List[ListResponse]` -- ListResponse is a class
            with minimal graph template metadata. Currently only
            contains `name`.
        """
        db_entries = await self._require_graph_service().list_named_graphs(
            self._project_id)

        list_response: List[ListResponse] = []

        for entry in db_entries:
            if entry.alias:
                num_tables = len(entry.table_name_to_dbid)
                num_links = sum(
                    len(grp.columns) - 1 for grp in entry.col_groups)
                list_response.append(
                    ListResponse(
                        id=f'graph-{entry.id}',
                        name=entry.alias,
                        timestamps=entry.timestamps,
                        num_tables=num_tables,
                        num_links=num_links,
                        created_by_name=entry.created_by_name,
                    ))

        return list_response

    async def start_graph_refresh(
        self, graph_id: GraphID,
        table_versions: Mapping[TableID,
                                TableDataVersion] = {}) -> Optional[str]:
        data_compute_intput = await self.get_graph_data_compute_input(graph_id)
        if table_versions:
            data_compute_intput.pinned_data_versions = dict(table_versions)
        return await DataBackend.start_refresh_graph(self._data_backend_ctx,
                                                     data_compute_intput)

    @blocking_method()
    def pquery_from_config(self, pquery_config: PredictiveQueryV2Config,
                           graph: Graph) -> PredictiveQuery:
        return PredictiveQueryV2.from_config(pquery_config, graph=graph)

    def delete_full_graph_def_v2(self, id: GraphIDv2) -> bool:
        return self._run_async_graph_logic(
            self.async_delete_full_graph_def_v2(id))

    async def async_delete_full_graph_def_v2(self, id: GraphIDv2) -> bool:
        graph_fpid = await self._resolve_graph_fpid_from_state_services(id)
        if graph_fpid is None:
            return False
        graph_service = self._require_graph_service()
        return await graph_service.decrement_or_delete_graph_definition(
            self._project_id, graph_fpid)

    @trace(TraceOptions(trace_params=True))
    async def _make_graph_from_graph_data_compute_input(
        self,
        graph_data_compute_input: GraphDataComputeInput,
    ) -> Graph:
        connector_specs = await self._get_connector_specs_for_graph_def(
            graph_data_compute_input.graph_definition)
        return _build_graph_from_definition_with_specs(
            graph_data_compute_input.graph_definition,
            graph_data_compute_input.graph_display_name,
            connector_specs.values(),
        )

    @deprecated("Will be obsolete after db migration.")
    async def get_graph_script_for_sdk(self, graph_id: GraphIDv2) -> str:
        """
        The method that handles the logic to generate a fully executable Python SDK script.
        1. Make sure graph exists. If not, throw `GraphDoesNotExistError`.
        2. Generate the connector and table configs for each table and unique connector.
        3. Generate the graph edge configs so that they are compatible with the SDK setup.
        4. Use all the configs to generate the final executable SDK file.
        5. Do some minor string parsing of the final file to make the script compatible
            with SDK implementations.

        Args:
            graph_id (GraphIDv2): The graph ID used to
            generate the metadata and script.
        Response:
            str: The complete python script as a string.
        """
        try:
            _, fpid = GraphIDv2.parse(graph_id)
            assert fpid is not None
        except BaseException:
            raise ValueError(f'Table resource id is malformed '
                             f': {graph_id}')

        graph_service = self._require_graph_service()
        db_entry = await graph_service.get_graph_entry(self._project_id, fpid)
        if not db_entry:
            raise GraphDoesNotExistError(graph_id)

        table_name_to_dbid = db_entry.table_name_to_dbid

        # Handle edge case where the same table is used multiple times in the graph
        _assert_no_duplicate_tables(table_name_to_dbid)

        table_and_connector_confs = await self._table_logic.get_many_table_configs_for_sdk(
            db_entry.table_name_to_dbid)

        connector_configs, table_configs = table_and_connector_confs

        edge_configs = get_edge_configs_for_sdk(
            db_entry.col_groups,
            table_configs,
        )
        final_script = generate_conn_graph_or_table_sdk_script(
            connector_configs,
            table_configs,
            edge_configs,
            project_id=self._project_id,
        )
        return make_stype_and_dtype_sdk_compat(final_script)

    @deprecated("Will be obsolete after db migration.")
    async def get_graph_metadata_for_sdk(
        self,
        graph_fpid: FPID,
    ) -> Tuple[List[ConnectorSpec], List[TableReferenceMetadata],
               List[graph_api.Edge]]:
        """
        Generates the connector(s), tables, and edges metadata for UI to
        SDK feature.
        Args:
            graph_id (FPID): The graph FPID used to
            generate the connector, table, and edges metadata.
        Response:
            Tuple[List[ConnectorSpec], List[TableReferenceMetadata],
               List[graph_api.Edge]]
        """
        graph_service = self._require_graph_service()
        db_entry = await graph_service.get_graph_entry(self._project_id,
                                                       graph_fpid)
        if not db_entry:
            raise GraphDoesNotExistError(GraphIDv2.compose(None, graph_fpid))
        table_name_to_dbid = db_entry.table_name_to_dbid

        # Handle edge case where the same table is used multiple times in the graph
        _assert_no_duplicate_tables(table_name_to_dbid)

        table_and_connector_confs = await self._table_logic.get_many_table_configs_for_sdk(
            table_name_to_dbid)
        connector_configs, table_configs = table_and_connector_confs
        edge_configs = get_edge_configs_for_sdk(db_entry.col_groups,
                                                table_configs)
        return (connector_configs, table_configs, edge_configs)

    @property
    def _table_logic(self) -> 'table_logic.TableLogic':
        return table_logic.TableLogic(
            self._data_backend_ctx,
            state_services=self._state_services,
            project_id=self._project_id,
        )


# ---------------------------------------------------------------------------
# V2-native graph helpers.
#
# These mirror `_add_tables_to_graph`, `_get_all_pkeys`, `_make_col_index`,
# `_make_fkeys`, `_convert_cols` but operate on `TableDefinition` directly
# — no v1 downgrade, no view branch (v2 doesn't model views). v2-only
# `make_graph_from_args_and_tables` callers dispatch here so that the graph
# helpers chain doesn't need `to_legacy_create_table_args`. The per-graph
# alias comes from the `TableID.table_name` dict key (v2 stores no alias
# on the `TableDefinition` itself).
# ---------------------------------------------------------------------------


def _make_col_index_v2(
    tables: Dict[TableName, TableDefinition], ) -> Dict[ColumnKey, Any]:
    """Map ColumnKey -> v2 Column for every column across all tables."""
    result: Dict[ColumnKey, Any] = {}
    for tname, definition in tables.items():
        for col in definition.cols:
            result[ColumnKey(table_name=tname, col_name=col.name)] = col
    return result


def _make_fkeys_v2(
    col_index: Dict[ColumnKey, Any],
    col_groups: List[ColumnKeyGroup],
    tables: Dict[TableName, TableDefinition],
) -> Dict[TableName, List[ForeignKey]]:
    """Calculate per-table foreign-key lists from name-keyed col_groups.

    Mirrors :func:`_make_fkeys` but operates entirely on ``TableName`` and
    ``ColumnKey`` (name-keyed). No ``TableID`` / ``ColumnID`` involved.
    """
    try:
        resolved = graph_api.resolve_foreign_keys(
            {
                str(name): definition
                for name, definition in tables.items()
            },
            col_groups,
        )
    except graph_api.GraphSpecError as error:
        if error.code == 'no_primary_key':
            rendered = ', '.join(f'{column.table_name}.{column.col_name}'
                                 for column in error.columns)
            message = COLUMN_GROUP_NO_PKEY_MESSAGE.format(rendered)
        elif error.code == 'multiple_primary_keys':
            first, second = error.columns[:2]
            message = COLUMN_GROUP_MULTIPLE_PKEYS_MESSAGE.format(
                f'{first.table_name}.{first.col_name}',
                f'{second.table_name}.{second.col_name}',
            )
        elif error.code == 'missing_column' and error.column is not None:
            key = ColumnKey(
                table_name=error.column.table_name,
                col_name=error.column.col_name,
            )
            message = COLUMN_GROUP_MISSING_COLUMN_MESSAGE.format(
                key, key.table_name)
        else:
            message = str(error)
        raise ValidationException(message=message) from error

    result: Dict[TableName, List[ForeignKey]] = defaultdict(list)
    for link in sorted(resolved, key=lambda edge: (edge.src_table, edge.fkey)):
        # The hosted graph builder has historically ignored relationships
        # within the primary-key table. Keep that adapter behavior distinct
        # from API/SDK graph reconstruction, which supports self-edges.
        if link.src_table == link.dst_table:
            continue
        source_key = ColumnKey(table_name=link.src_table, col_name=link.fkey)
        destination = tables[TableName(link.dst_table)]
        assert destination.pkey is not None
        result[TableName(link.src_table)].append(
            column_def_to_foreign_key(
                col_index[source_key],
                dst_connector_id=destination.get_connector_id(),
                dst_table_name=TableName(link.dst_table),
                dst_col_name=destination.pkey,
            ))
    return result


def _convert_cols_v2(cols: Any, pkey: Optional[PrimaryKey],
                     fkeys: List[ForeignKey]) -> List[Column]:
    """v2 mirror of :func:`_convert_cols`; takes v2 ``Column`` inputs."""
    result: List[Column] = []
    names_to_exclude: Set[str] = set()
    if pkey:
        names_to_exclude.add(pkey.name)
    for fkey in fkeys:
        names_to_exclude.add(fkey.name)
    for col in cols:
        if col.name in names_to_exclude:
            continue
        result.append(col.to_feature_column())
    return result


def _add_tables_to_graph_v2(
    graph: Graph,
    tables: Dict[TableName, TableDefinition],
    col_groups: List[ColumnKeyGroup],
    ignore_errors: bool = False,
) -> None:
    """Add tables to a Kumo Graph from name-keyed v2 inputs.

    No views (v2 doesn't model views). Per-graph alias is the dict key.
    """
    col_index = _make_col_index_v2(tables)
    links = _make_fkeys_v2(col_index, col_groups, tables)

    for tname, definition in tables.items():
        pkey = definition.primary_key_col()
        time_col = definition.time_column()
        end_time_col = definition.end_time_column()
        fkeys = links[tname]
        cols = _convert_cols_v2(definition.cols, pkey, fkeys)
        # Thread the SQL text so the built table is ``is_sql=True`` (mirrors
        # ``table_args_to_kumo_table``). Only the ``is_sql`` marker is later
        # serialized into ``TableConfig`` for the validate/train activity.
        custom_sql = (definition.source_table.sql
                      if definition.is_sql else None)
        graph.add(
            name=tname,
            data_source=definition.get_connector_id(),
            pkey=pkey,
            time_col=time_col,
            end_time_col=end_time_col,
            fkeys=fkeys,
            cols=cols,
            source_name=str(definition.get_source_table_name()),
            custom_sql=custom_sql,
            ignore_errors=ignore_errors,
        )


def _build_graph_from_definition_with_specs(
    graph_def: GraphDefinition,
    name: str,
    connector_specs: Iterable[ConnectorOpenSpec],
) -> Graph:
    """Build a backend graph from service-resolved connector specs."""
    connector_specs_dict = {spec.connector_id: spec for spec in connector_specs}

    graph = Graph(name)
    created_connectors: List[Connector] = []
    try:
        for connector_spec in connector_specs_dict.values():
            connector = get_connector_from_spec(connector_spec)
            created_connectors.append(connector)
            graph.connect(connector)
        _add_tables_to_graph_v2(
            graph,
            graph_def.tables,
            graph_def.col_groups,
            ignore_errors=True,
        )
        return graph
    except BaseException:
        try:
            _close_connectors(created_connectors)
        except BaseException:
            pass
        raise


def _close_connectors(connectors: Iterable[Connector]) -> None:
    """Close activity-local connector capabilities, including partial builds."""
    first_error: BaseException | None = None
    for connector in reversed(list(connectors)):
        try:
            connector.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
            log.warning(
                'Activity graph connector cleanup failed '
                '(cleanup_error_type=%s)',
                type(error).__name__)
    if first_error is not None:
        raise first_error


def _close_graph_connectors(graph: Graph) -> None:
    """Best-effort cleanup for an activity-local graph's connectors."""
    _close_connectors(graph.data_source_dict.values())


@connector_activity_registry.connector_authorized_activity
def infer_graph_links_activity(
    graph_def: GraphDefinition,
    connector_refs: Dict[ConnectorID, ConnectorExecutionRef],
) -> List[ColumnKeyGroup]:
    graph = _build_graph_from_definition_with_specs(graph_def, 'NA',
                                                    connector_refs.values())
    primary_error: BaseException | None = None
    try:
        inferred: List[ColumnKeyGroup] = []
        for group in graph.infer_col_groups():
            pkey_table = TableIDv2.parse(group.pkey.table + '-NA')[1]
            columns = [
                ColumnKey(table_name=pkey_table, col_name=group.pkey.col)
            ]
            for fkey in group.fkeys:
                fkey_table = TableIDv2.parse(fkey.table + '-NA')[1]
                columns.append(
                    ColumnKey(table_name=fkey_table, col_name=fkey.col))
            inferred.append(ColumnKeyGroup(columns=tuple(columns)))
        return inferred
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            _close_graph_connectors(graph)
        except BaseException:
            if primary_error is None:
                raise


@connector_activity_registry.connector_authorized_activity
async def validate_graph_definition_activity(
    graph_def: GraphDefinition,
    connector_refs: Dict[ConnectorID, ConnectorExecutionRef],
) -> None:
    graph = _build_graph_from_definition_with_specs(graph_def, 'NA',
                                                    connector_refs.values())
    primary_error: BaseException | None = None
    try:
        await graph.validate.coroutine(
            raise_on_error=True,
            ignore_encoders=True,
            enforce_connected=True,
        )
    except ValidationException as error:
        primary_error = error
        raise ValidationException(
            title=INVALID_GRAPH_PARAMETERS_TITLE,
            message=INVALID_GRAPH_PARAMETERS_MESSAGE.format(error),
        ) from error
    except ValueError as error:
        # ``graph.validate`` reports structural user-input failures (e.g. a
        # table with no key, or a float-typed primary key) via
        # ``io_utils.logging.warn_or_raise``, which raises a bare
        # ``ValueError``. That type is not in the data-backend activity
        # allowlist, so surfacing it raw masks the actionable, Kumo-authored
        # message as a generic InternalError. Re-raise as the allowlisted
        # ValidationException carrying the same message.
        primary_error = error
        raise ValidationException(
            title=INVALID_GRAPH_PARAMETERS_TITLE,
            message=INVALID_GRAPH_PARAMETERS_MESSAGE.format(error),
        ) from error
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            _close_graph_connectors(graph)
        except BaseException:
            if primary_error is None:
                raise


def _split_graph_id_versions(
        graph_ids: Sequence[GraphID]
) -> Tuple[List[GraphIDv1], List[GraphIDv2]]:
    # Split into v1 and v2
    v1_ids: List[GraphIDv1] = []
    v2_ids: List[GraphIDv2] = []
    for id in graph_ids:
        if isinstance(id, GraphIDv1):
            v1_ids.append(id)
        else:
            assert isinstance(id, GraphIDv2)
            v2_ids.append(id)

    return v1_ids, v2_ids


T = TypeVar('T')


def _merge_graph_id_versions(graph_ids: Sequence[GraphID], v1_result: List[T],
                             v2_result: List[T]) -> List[T]:
    # Merge together
    result: List[T] = []
    v1_idx = 0
    v2_idx = 0
    for id in graph_ids:
        if isinstance(id, GraphIDv1):
            result.append(v1_result[v1_idx])
            v1_idx += 1
        else:
            result.append(v2_result[v2_idx])
            v2_idx += 1

    return result


def get_edge_configs_for_sdk(
    col_groups: List[ColumnKeyGroup],
    table_configs: List[TableReferenceMetadata],
) -> List[graph_api.Edge]:
    """
    Generates the SDK compatible edge config for a graph. An edge in the SDK
    is represented as `("src_table": src_table, "fkey": col.col_name,
    "dst_table": dst_table)`.

    Args:
        col_groups (List[ColumnKeyGroup]): Column groups in a graph.
        table_configs (List[TableConfig]): A list of the generated
        table configs.

    Returns:
        List[EdgeConfig]: An array of SDK compatible edges.
    """

    # new_table_conf serves as a dictionary to be able to fetch
    # table configs both based on source table and alias name.
    # Because aliases are not part of a table definition, we
    # sometimes fetch a table under the wrong name. This should
    # handle the differences and how they are named in a graph.
    new_table_conf: Dict[str, Dict[str, Any]] = {}
    for table in table_configs:
        if table.table_alias_in_graph:
            new_table_conf[table.table_alias_in_graph] = table.table_config
        else:
            new_table_conf[table.source_table_name] = table.table_config

    tables = {
        name:
        ApiTableDefinition(
            cols=config['cols'],
            source_table=UnavailableSourceTable(config['table_name']),
            pkey=config['pkey'],
        )
        for name, config in new_table_conf.items()
    }
    return [
        graph_api.Edge(
            new_table_conf[edge.src_table]['table_name'],
            edge.fkey,
            new_table_conf[edge.dst_table]['table_name'],
        ) for edge in graph_api.resolve_foreign_keys(tables, col_groups)
    ]


def _assert_no_duplicate_tables(table_name_to_dbid: Dict[TableName, FPID]):
    if len(set(table_name_to_dbid.keys())) != len(
            set(table_name_to_dbid.values())):
        value_to_keys = {}
        for key, value in table_name_to_dbid.items():
            if value in value_to_keys:
                value_to_keys[value].append(key)
            else:
                value_to_keys[value] = [key]
        duplicate_keys = [
            key for keys in value_to_keys.values() if len(keys) > 1
            for key in keys
        ]
        raise ValueError(
            "Kumo cannot generate SDK code for graphs that contain multiple "
            "tables that point to the same source table. "
            f"The duplicate tables are: {duplicate_keys}.")
