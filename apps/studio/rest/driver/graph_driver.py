# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple

import kumoapi.graph as graph_api
from kumo_core.types.schema import FPID
from kumoapi.graph import ColumnKeyGroup as ApiColumnKeyGroup

from io_utils import logging
from kumo.validation import ValidationException
from kumo.validation.validation import ValidationError, ValidationResponse
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.data_backend import DataBackend
from rest.config.data_backend import ConnectorSpec, GraphDataVersionInfo
from rest.config.graph import EdgeHealthStats
from rest.config.graph_v2 import (
    GraphDataComputeInput,
    GraphDefinition,
    ListResponse,
)
from rest.config.table_v2 import (
    TableDefinition,
    TableReferenceMetadata,
    TableResourceResponse,
)
from rest.deps.context import _current_user_ctx
from rest.driver.error_constants import CANNOT_INFER_LINKS_WITH_EDGES_MESSAGE
from rest.driver.graph_logic import (
    _assert_no_duplicate_tables,
    canonicalize_graph_custom_sql_tables,
    get_edge_configs_for_sdk,
    infer_graph_links_activity,
    validate_graph_definition_activity,
)
from rest.driver.table_driver import TableDriver
from rest.driver.table_logic import TableLogic
from rest.settings import DataScaleLimits
from rest.utils.routers import ValidateNameResponse
from rest.utils.ui_to_sdk_utils import (
    generate_conn_graph_or_table_sdk_script,
    make_stype_and_dtype_sdk_compat,
)
from statedb.config.common import (
    ID,
    ConnectorID,
    GraphIDv2,
    ProjectID,
    TableIDv2,
    TableName,
)
from statedb.config.timestamp import ResourceTimestamps
from statedb.db.services import StateServices
from statedb.rest.connector_authorization import CONNECTOR_GRAPH_ACTIONS
from statedb.rest.graph_v2 import GraphDBEntryV2
from statedb.rest.resource_alias import graph_alias_conflict_message
from statedb.utils.errors import (
    GraphDoesNotExistError,
    GraphExistsInProjectError,
    TableDoesNotExistError,
)

log = logging.getLogger(__name__)

# Upper bound on concurrent per-table warehouse validations during graph
# creation: high enough that wide graphs stay within upstream proxy deadlines
# (kumo-ai/kumo#30053), low enough not to flood a warehouse with parallel
# metadata probes.
_MAX_CONCURRENT_TABLE_VALIDATIONS = 8


class GraphDriver:
    """Supports operations on graph objects.
    This class exclusively performes operations on the v2 object models.
    """
    def __init__(
        self,
        data_backend_ctx: DataBackendContext,
        data_scale_limits: DataScaleLimits,
        table_driver: TableDriver,
    ):
        """
        Initialize the GraphDriver.

        Args:
            data_backend_ctx: Context for data backend operations
            data_scale_limits: Limits for data scale validation
        """
        self._data_backend_ctx = data_backend_ctx
        self._data_scale_limits = data_scale_limits
        self._table_driver = table_driver
        self._state_services: Optional[StateServices] = None

    def _configure_state_services(self, state_services: StateServices) -> None:
        self._state_services = state_services

    def _graph_service(self):
        return self._require_state_services().graphs

    def _project_service(self):
        return self._require_state_services().projects

    def _table_service(self):
        return self._require_state_services().tables

    def _connector_service(self):
        return self._require_state_services().connectors

    def _require_state_services(self) -> StateServices:
        if self._state_services is None:
            raise RuntimeError("GraphDriver requires StateServices")
        return self._state_services

    def _require_graph_service(self):
        return self._graph_service()

    def _require_project_service(self):
        return self._project_service()

    def _require_table_service(self):
        return self._table_service()

    def _require_connector_service(self):
        return self._connector_service()

    async def _require_project_access(self,
                                      project_id: ProjectID | str) -> None:
        """Authorize a project before any graph repository or cache read."""
        await self._require_project_service().get_project(
            ProjectID(project_id),
            _current_user_ctx.get(),
        )

    async def get_graph_v2(
        self,
        graph_fpid: FPID,
        project_id: ProjectID,
    ) -> Tuple[
            graph_api.GraphResource,
            ResourceTimestamps,
            Dict[
                TableName,
                FPID,
            ],
    ]:
        """
        Retrieve a graph v2 resource by its FPID.

        Args:
            graph_fpid: The FPID of the graph to retrieve

        Returns:
            A tuple containing:
                - :class:`graph_api.GraphResource`: The graph resource
                - :class:`ResourceTimestamps`: Timestamps for the resource
                - :class:`Dict[TableName, FPID]`: Mapping of table names to
                FPIDs

        Raises:
            KeyError: If the graph with the given FPID is not found
        """
        await self._require_project_access(project_id)
        graph_service = self._require_graph_service()
        graph_entry = await graph_service.get_graph_entry(
            project_id, graph_fpid)
        if graph_entry is None:
            raise KeyError(f'Graph {graph_fpid} is not found')
        graph_def = await graph_service.get_graph_definition(
            project_id, graph_fpid)
        if graph_def is None:
            raise KeyError(f'Graph {graph_fpid} is not found')
        table_name_to_dbid: Dict[TableName, FPID] = {
            TableName(table_name): FPID(fpid)
            for table_name, fpid in graph_entry.table_name_to_dbid.items()
        }
        return (
            graph_api.GraphResource(
                id=GraphIDv2.public(graph_fpid),
                graph=graph_def,
                name_alias=graph_entry.alias,
            ),
            graph_entry.timestamps,
            table_name_to_dbid,
        )

    async def get_graph_v2_by_alias(
        self,
        graph_alias: ID,
        project_id: ProjectID,
    ) -> Tuple[
            graph_api.GraphResource,
            ResourceTimestamps,
            Dict[TableName, FPID],
    ]:
        """
        Retrieve a graph v2 resource by its alias.

        Args:
            graph_alias: The alias ID of the graph to retrieve

        Returns:
            A tuple containing:
                - :class:`graph_api.GraphResource`: The graph resource
                - :class:`ResourceTimestamps`: Timestamps for the resource
                - :class:`Dict[TableName, FPID]`: Mapping of table names to
                FPIDs

        Raises:
            KeyError: If the graph with the given alias is not found
        """
        await self._require_project_access(project_id)
        graph_service = self._require_graph_service()
        graph_entry = await graph_service.get_graph_entry_by_alias(
            project_id, graph_alias)
        if not graph_entry:
            raise KeyError(f'Graph alias {graph_alias} is not found')
        return await self.get_graph_v2(FPID(graph_entry.id), project_id)

    async def resolve_project_graph(
        self,
        project_id: str,
        graph_id_or_name: str,
    ) -> GraphDBEntryV2:
        """Resolve a canonical graph ID or project-local alias."""
        project_id = ProjectID(project_id)
        await self._require_project_access(project_id)
        try:
            _, fpid = GraphIDv2.parse(graph_id_or_name)
            if fpid is None:
                raise ValueError("Invalid graph ID passed.")
            graph_entry = await self._require_graph_service().get_graph_entry(
                project_id, fpid)
        except ValueError:
            graph_entry = await self._require_graph_service(
            ).get_graph_entry_by_alias(project_id, graph_id_or_name)
        if graph_entry is None:
            raise KeyError(
                f"Could not find graph '{graph_id_or_name}' in project '{project_id}'."
            )
        return graph_entry

    async def list_project_graph_names(
        self,
        project_id: str,
    ) -> List[ListResponse]:
        """List graph names scoped to a project using project-local aliases."""
        project_id = ProjectID(project_id)
        await self._require_project_access(project_id)
        entries = await self._require_graph_service().list_named_graphs(
            project_id)
        return [
            ListResponse(
                name=entry.alias,
                id=GraphIDv2.public(entry.id),
                timestamps=entry.timestamps,
            ) for entry in entries
        ]

    async def validate_project_graph_name(
        self,
        project_id: str,
        name: str,
    ) -> ValidateNameResponse:
        """Validate that a project-local graph display name is available."""
        try:
            ID(name)
        except ValueError as e:
            return ValidateNameResponse(is_valid=False, error_msg=str(e))

        typed_project_id = ProjectID(project_id)
        await self._require_project_access(typed_project_id)
        existing = await self._require_graph_service().get_graph_entry_by_alias(
            typed_project_id, name)
        if existing is not None:
            return ValidateNameResponse(
                is_valid=False,
                error_msg=graph_alias_conflict_message(name),
            )
        return ValidateNameResponse(is_valid=True, error_msg=None)

    async def list_graph_names_v2(
        self,
        project_id: ProjectID,
    ) -> List[ListResponse]:
        """
        List all graph names for v2 graphs.

        Returns:
            :class:`List[ListResponse]`: List of graph name responses
        """
        await self._require_project_access(project_id)
        graphs = await self._require_graph_service().list_named_graphs(
            project_id)
        return [
            ListResponse(
                id=f'graph-{graph.id}',
                name=graph.alias,
                timestamps=graph.timestamps,
                num_tables=len(graph.table_name_to_dbid),
                num_links=sum(len(grp.columns) - 1 for grp in graph.col_groups),
                created_by_name=graph.created_by_name,
            ) for graph in graphs if graph.alias
        ]

    async def suggest_links_v2(
        self,
        graph_def: GraphDefinition,
        project_id: ProjectID,
    ) -> List[ApiColumnKeyGroup]:
        """
        Suggest links for a graph definition by inferring column relationships.

        Args:
            graph_def: The graph definition to suggest links for

        Returns:
            :class:`List[ColumnKeyGroup]`: List of suggested column key groups
        """
        await self._require_project_access(project_id)
        new_graph_def = await self._infer_links(graph_def, project_id)
        return list(new_graph_def.col_groups)

    async def create_full_graph_v2(
        self,
        graph: GraphDefinition,
        *,
        project_id: str,
        alias: Optional[str] = None,
        force_rename: bool = False,
        skip_validation: bool = False,
    ) -> GraphIDv2:
        """
        Creates a graph_v2 object in the DB cascadingly creating tables
        if they do not exists.

        Args:
            graph: Definition of the Graph to create
            alias: Optional alias to use for the graph. If this
                is supplied, then an entry in the graph_v2_alias map is also
                created, enabling this graph to be queried by this alias in the
                future.
            force_rename: Boolean indicating, if the graph already
                exists with another alias, should this function call force
                rename to the new alias provided.
            skip_validation: Boolean indicating whether to skip graph
                validation after creation. Defaults to False.
            project_id: Optional project ID to scope the graph to.

        Returns:
            :class:`GraphIDv2`: ID of the created graph.

        Raises:
            ValidationException: If graph validation fails and skip_validation
                is False. Per-table validation failures are aggregated into a
                single exception that reports every failing table (each error
                title prefixed with ``Table '<name>':``, sorted by table
                name), rather than only the first failure.
        """
        if project_id is None:
            raise ValueError('project_id is required')
        alias = alias or None
        project_service = self._require_project_service()
        await project_service.validate_project_for_creation(
            project_id,
            _current_user_ctx.get(),
        )
        # Validate + canonicalize embedded Custom SQL tables before the graph's
        # fingerprint is used (below) or any definition is persisted. This runs
        # unconditionally: it is the cheap connector_type correctness/security
        # gate, independent of the expensive ``skip_validation`` checks.
        graph = await canonicalize_graph_custom_sql_tables(
            graph, ProjectID(project_id), self._require_state_services(),
            _current_user_ctx.get())
        if alias and force_rename is False:
            name_validation = await self.validate_project_graph_name(
                project_id, alias)
            if not name_validation.is_valid:
                existing_named_graph = await self._require_graph_service(
                ).get_graph_entry_by_alias(ProjectID(project_id), alias)
                if (existing_named_graph is None
                        or existing_named_graph.id != graph.fpid):
                    raise GraphExistsInProjectError(
                        name_validation.error_msg
                        or (f"Graph with name {alias} already "
                            "exists in this project."))

        table_resources = await self._create_full_graph_def_v2_from_state_services(
            graph,
            alias=alias,
            force_rename=force_rename,
            skip_validation=skip_validation,
            project_id=ProjectID(project_id),
        )

        total_size = 0
        for resource in table_resources:
            stats = resource.stats
            assert stats is not None
            total_size += stats.size_bytes
            self._data_scale_limits.validate_table(
                resource.name_alias or str(resource.id),
                num_cols=len(resource.table.cols),
                size_bytes=stats.size_bytes,
                raise_on_error=True,
            )

        # Validate overall graph-level scale limits
        self._data_scale_limits.validate_graph(
            total_size_bytes=total_size,
            raise_on_error=True,
        )

        # Return the public GraphIDv2
        graph_id = GraphIDv2.public(graph.fpid)

        return graph_id

    async def _create_full_graph_def_v2_from_state_services(
        self,
        graph: GraphDefinition,
        *,
        alias: Optional[str],
        force_rename: bool,
        skip_validation: bool,
        project_id: ProjectID,
    ) -> List[TableResourceResponse]:
        graph_service = self._require_graph_service()
        table_service = self._require_table_service()

        existing_entry = await graph_service.get_graph_entry(
            project_id, graph.fpid)
        await self._validate_graph_tables_for_creation(
            graph,
            project_id,
            validate_target=(existing_entry is None and not skip_validation),
        )
        if existing_entry is None and not skip_validation:
            await self._validate_graph_definition(graph, project_id)
        principal = _current_user_ctx.get()
        if principal is None:
            raise RuntimeError('Authenticated principal is required')
        graph_entry = await graph_service.save_graph_definition(
            project_id,
            graph,
            alias=alias,
            force_rename=force_rename,
            created_by=str(principal.user_id),
            created_by_name=principal.user_name,
        )
        table_entries_map = await table_service.get_table_entries(
            project_id, graph_entry.table_name_to_dbid.values())
        table_entries = []
        for table_fpid in graph_entry.table_name_to_dbid.values():
            table_entry = table_entries_map.get(table_fpid)
            if table_entry is None:
                raise TableDoesNotExistError(table_fpid)
            table_entries.append(table_entry)
        return await self._table_driver._table_resource_responses_from_entries(
            table_entries, fetch_column_stats=False)

    async def _validate_graph_tables_for_creation(
        self,
        graph: GraphDefinition,
        project_id: ProjectID,
        *,
        validate_target: bool,
    ) -> None:
        connector_service = self._require_connector_service()
        principal = _current_user_ctx.get()
        for table_def in graph.tables.values():
            await connector_service.execution_ref_for_project(
                principal,
                project_id,
                table_def.get_connector_id(),
                CONNECTOR_GRAPH_ACTIONS,
            )
        if not validate_target:
            return

        # Each ``validate_table`` call performs multiple warehouse
        # round-trips (connector open plus metadata, size, and permissions
        # probes), so validating tables one at a time scales request latency
        # with table count and pushes wide graphs past upstream proxy
        # deadlines (kumo-ai/kumo#30053). Validate concurrently, bounded so
        # one graph cannot flood a warehouse, and report every failing table
        # at once instead of stopping at the first.
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_TABLE_VALIDATIONS)

        async def _validate(table_def: TableDefinition) -> ValidationResponse:
            async with semaphore:
                return await self._table_driver.validate_table(
                    table_def,
                    project_id=str(project_id),
                )

        responses = await asyncio.gather(
            *(_validate(table_def) for table_def in graph.tables.values()),
            return_exceptions=True)
        for response in responses:
            if isinstance(response, BaseException):
                raise response
        failures = [(str(table_name), response)
                    for table_name, response in zip(graph.tables, responses)
                    if response.errors]
        if failures:
            failures.sort(key=lambda failure: failure[0])
            errors = [
                ValidationError(
                    title=f"Table '{table_name}': {error.title}",
                    message=error.message,
                ) for table_name, response in failures
                for error in response.errors
            ]
            raise ValidationException(message=ValidationResponse(
                errors=errors).error_message())

    async def delete_graph_alias(
        self,
        graph_id_or_alias: str,
        project_id: str,
    ) -> str:
        """Clear a graph alias by canonical id or project-local alias."""
        await self._require_project_access(project_id)
        graph_service = self._require_graph_service()
        try:
            _, fpid = GraphIDv2.parse(graph_id_or_alias)
        except ValueError:
            fpid = None
        if fpid is not None:
            entry = await graph_service.clear_graph_alias_by_id(
                ProjectID(project_id), fpid)
        else:
            entry = await graph_service.clear_graph_alias(
                ProjectID(project_id), graph_id_or_alias)
        if entry is None:
            raise GraphDoesNotExistError(graph_id_or_alias)
        return str(GraphIDv2.public(entry.id))

    async def refresh_graph_v2(
        self,
        graph_id: GraphIDv2,
        project_id: ProjectID,
    ) -> None:
        """Kick off an async refresh workflow for a v2 graph by id.

        Mirrors the v0 ``refresh_graph`` flow: verify the graph exists, then
        start the async graph-refresh workflow.
        """
        await self._require_project_access(project_id)
        _, fpid = GraphIDv2.parse(graph_id)
        if fpid is None:
            raise GraphDoesNotExistError(graph_id)
        graph_entry = await self._require_graph_service().get_graph_entry(
            project_id, fpid)
        if graph_entry is None:
            raise GraphDoesNotExistError(graph_id)

        execution_input = await self._get_graph_data_compute_input(
            graph_id, project_id)
        await self._start_graph_refresh(execution_input)

    async def _start_graph_refresh(
        self,
        data_compute_input: GraphDataComputeInput,
    ) -> Optional[str]:
        return await DataBackend.start_refresh_graph(
            self._data_backend_ctx,
            data_compute_input,
        )

    async def infer_links(
        self,
        graph: GraphDefinition,
        project_id: ProjectID,
    ) -> GraphDefinition:
        """
        Infer links for a graph definition by analyzing table relationships.

        This method analyzes the tables in the graph definition and
        automatically infers potential links between columns based on data
        types, column names, and existing relationships. The inferred links
        are added to the graph's column key groups
        (:attr:`GraphDefinition.col_groups`).

        Args:
            graph: The graph definition to infer links for

        Returns:
            :class:`GraphDefinition`: A new graph definition with inferred
            links added to the :attr:`col_groups` attribute. The returned
            graph will contain the same tables and configuration as the input,
            but with additional column key groups representing the inferred
            relationships between tables.

        Note:
            The inference process examines column names, data types, and
            existing relationships to suggest meaningful links between tables.
            This can help automate the process of defining graph structure and
            reduce manual configuration requirements.
        """
        await self._require_project_access(project_id)
        return await self._infer_links(graph, project_id)

    async def validate_graph(
        self,
        graph_definition: graph_api.GraphDefinition,
        project_id: ProjectID,
    ) -> ValidationResponse:
        """
        This method performs comprehensive validation of a graph definition by
        validating both individual table metadata and the overall graph
        structure. It validates table configurations, column definitions,
        and ensures the graph definition is consistent and properly formed.

        Args:
            graph_definition: The graph definition to validate

        Returns:
            :class:`ValidationResponse`: A validation response object
            containing any validation errors found during the process. If
            validation passes, the response will contain an empty errors list.
            If validation fails, the response will contain one or more
            :class:`ValidationError` objects with details about what failed
            validation.

        Note:
            This method validates both table-level metadata (through
            :class:`TableLogic.validate_table`) and graph-level structure
            (through :class:`GraphLogic.validate_graph_definition`). Any
            validation failures are captured and returned as part of the
            response rather than raising exceptions.
        """
        await self._require_project_access(project_id)
        graph_def = GraphDefinition(**dataclasses.asdict(graph_definition))

        try:
            await self._validate_graph_definition(graph_def, project_id)
        except Exception as e:
            # TODO(manan): fix the wide net here:
            return ValidationResponse(errors=[
                ValidationError(
                    title="Graph Validation Failure",
                    message=str(e),
                )
            ])
        return ValidationResponse()

    async def get_edge_stats(
        self,
        graph_id: GraphIDv2,
        project_id: ProjectID,
    ) -> EdgeHealthStats:
        """
        Get edge statistics and health metrics for a v2 graph.

        This method analyzes the relationships and connections between tables
        in the specified graph and returns comprehensive statistics about
        edge health, connectivity, and performance metrics. The statistics
        help assess the overall health and quality of the graph structure.

        Args:
            graph_id: The ID of the graph to get edge stats for

        Returns:
            :class:`EdgeHealthStats`: An object containing detailed statistics
            about the graph's edges, including health metrics, connectivity
            information, and performance indicators. This can be used to
            monitor graph quality and identify potential issues with table
            relationships.

        Note:
            Edge statistics provide insights into how well tables are connected
            and can help identify orphaned tables, missing relationships, or
            performance bottlenecks in the graph structure.
        """
        await self._require_project_access(project_id)
        return await self._get_edge_stats_impl(graph_id, project_id)

    async def graph_to_sdk(
        self,
        graph_id: GraphIDv2,
        project_id: ProjectID,
    ) -> str:
        """
        Used to generate a fully executable Python SDK script. The script contains:
            1. Connector(s) config(s). If there are tables from multiple connectors,
            we load the config for each connector.
            2. Table configs structured as Python dictionaries.
            3. Graph edge configs.
            4. Using the aforementioned configs to setup a graph in the SDK.
            using the metadata provided in the config.
        This method is part of the UI to SDK feature and will be called in the UI.
        Args:
            graph_id (GraphIDv2): The graph ID for the graph that we will be
            generating the config and script.
        Response:
            str: The complete python script as a string.
        """
        await self._require_project_access(project_id)
        return await self._get_graph_script_for_sdk_impl(graph_id, project_id)

    async def get_graph_metadata_for_sdk(
        self,
        graph_fpid: FPID,
        project_id: ProjectID,
    ) -> Tuple[List[ConnectorSpec], List[TableReferenceMetadata],
               List[graph_api.Edge]]:
        """
        Get graph metadata for SDK script generation.

        This method retrieves connector specs, table metadata, and edges
        for a graph, which can be used to generate SDK scripts.

        Args:
            graph_fpid: The FPID of the graph

        Returns:
            Tuple containing connector specs, table metadata, and edges
        """
        await self._require_project_access(project_id)
        return await self._get_graph_metadata_for_sdk_impl(
            graph_fpid, project_id)

    async def _execution_refs_for_graph(
        self,
        graph_def: GraphDefinition,
        project_id: ProjectID,
    ):
        connector_service = self._require_connector_service()
        connector_ids = connector_service.connector_ids_for_graph(graph_def)
        return {
            connector_id:
            await connector_service.execution_ref_for_project(
                _current_user_ctx.get(), project_id, connector_id,
                CONNECTOR_GRAPH_ACTIONS)
            for connector_id in connector_ids
        }

    async def _infer_links(
        self,
        graph_def: GraphDefinition,
        project_id: ProjectID,
    ) -> GraphDefinition:
        """Implementation of link inference for SQL DB."""
        if graph_def.col_groups and len(graph_def.col_groups) > 0:
            raise ValueError(CANNOT_INFER_LINKS_WITH_EDGES_MESSAGE)

        references = await self._execution_refs_for_graph(graph_def, project_id)
        groups = await DataBackend._execute_single_activity(
            self._data_backend_ctx,
            infer_graph_links_activity,
            [graph_def, references],
            f'infer-graph-links:{project_id}',
        )
        graph_def.col_groups.extend(groups)

        return graph_def

    async def _validate_graph_definition(
        self,
        graph_def: GraphDefinition,
        project_id: ProjectID,
    ):
        """Implementation of graph definition validation for SQL DB."""
        references = await self._execution_refs_for_graph(graph_def, project_id)
        await DataBackend._execute_single_activity(
            self._data_backend_ctx,
            validate_graph_definition_activity,
            [graph_def, references],
            f'validate-graph:{project_id}',
        )

    async def _get_graph_data_compute_input(
        self,
        graph_id: GraphIDv2,
        project_id: ProjectID,
    ) -> GraphDataComputeInput:
        """Build GraphDataComputeInput from graph_id.
        Most of this method consists of converting between different data
        models to enable execution of workflows in data backend which requires
        GraphDataComputeInput as input.
        """
        return await self._require_graph_service().get_graph_data_compute_input(
            graph_id,
            project_id=project_id,
            user_authorization=_current_user_ctx.get(),
        )

    async def _get_edge_stats_impl(
        self,
        graph_id: GraphIDv2,
        project_id: ProjectID,
        graph_version_info: Optional[GraphDataVersionInfo] = None,
    ) -> EdgeHealthStats:
        """Implementation of get_edge_stats for SQL DB."""
        data_compute_input = (await self._get_graph_data_compute_input(
            graph_id, project_id))
        return await self._get_edge_stats_from_compute_input(
            graph_id, data_compute_input, graph_version_info)

    async def _get_edge_stats_from_compute_input(
        self,
        graph_id: GraphIDv2,
        data_compute_input: GraphDataComputeInput,
        graph_version_info: Optional[GraphDataVersionInfo] = None,
    ) -> EdgeHealthStats:
        link_stats = await DataBackend.get_edge_stats(self._data_backend_ctx,
                                                      data_compute_input,
                                                      graph_version_info)

        edge_stats = EdgeHealthStats(is_ready=link_stats is not None,
                                     link_stats=link_stats)

        # Update table IDs for v2
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

    async def _get_graph_sdk_metadata_from_state_services(
        self,
        graph_fpid: FPID,
        project_id: ProjectID,
    ) -> Tuple[List[ConnectorSpec], List[TableReferenceMetadata],
               List[graph_api.Edge]]:
        graph_service = self._require_graph_service()
        table_service = self._require_table_service()

        graph_entry = await graph_service.get_graph_entry(
            project_id, graph_fpid)
        if graph_entry is None:
            raise GraphDoesNotExistError(GraphIDv2.compose(None, graph_fpid))

        table_name_to_dbid = graph_entry.table_name_to_dbid
        _assert_no_duplicate_tables(table_name_to_dbid)

        table_entries = await table_service.get_table_entries(
            project_id, table_name_to_dbid.values())
        connector_configs, table_configs = (
            await self._get_many_table_configs_for_sdk_from_entries(
                project_id,
                table_name_to_dbid,
                table_entries,
            ))
        edge_configs = get_edge_configs_for_sdk(graph_entry.col_groups,
                                                table_configs)
        return connector_configs, table_configs, edge_configs

    async def _get_many_table_configs_for_sdk_from_entries(
        self,
        project_id: ProjectID,
        table_name_to_dbid: Dict[TableName, FPID],
        db_entries,
    ) -> Tuple[List[ConnectorSpec], List[TableReferenceMetadata]]:
        table_fpids: Set[FPID] = set()
        table_fpid_to_name_in_graph: Dict[FPID, str] = {}
        for name, fpid in table_name_to_dbid.items():
            table_fpid_to_name_in_graph[fpid] = str(name)
            table_fpids.add(fpid)

        table_logic = TableLogic(
            self._data_backend_ctx,
            state_services=self._state_services,
            project_id=project_id,
        )
        table_confs: List[TableReferenceMetadata] = []
        connector_confs_set: Set[Tuple[Tuple[str, Any], ...]] = set()
        connector_spec_arr: List[ConnectorSpec] = []
        for fpid, table_entry in db_entries.items():
            if fpid not in table_fpids:
                continue
            table_def = table_entry.config
            table_id = TableIDv2.public(
                pquery_id=None,
                table_name=TableName("NA"),
                fpid=fpid,
            )
            connector_conf, src_tbl_conf = (
                table_logic.get_source_table_and_connector_metadata(
                    table_entry,
                    table_id,
                    table_fpid_to_name_in_graph,
                ))

            table_metadata = {
                "cols": table_def.cols,
                "pkey": table_def.pkey,
                "time_col": table_def.time_col,
                "end_time_col": table_def.end_time_col,
            }
            table_config = TableReferenceMetadata(
                source_table_name=src_tbl_conf["table_name"],
                table_fpid=fpid,
                table_config={
                    **src_tbl_conf,
                    **table_metadata,
                },
                table_alias_in_graph=table_fpid_to_name_in_graph[fpid],
            )
            if table_entry.alias:
                table_config.table_alias = table_entry.alias

            table_confs.append(table_config)
            connector_confs_set.add(connector_conf)

        for config in connector_confs_set:
            config_as_dict = dict(config)
            if "connector_id" in config_as_dict:
                connector_id = ConnectorID(config_as_dict["connector_id"])
            elif "name" in config_as_dict:
                connector_id = ConnectorID(config_as_dict["name"])
            else:
                raise ValueError(f"Connector with config {config_as_dict} does "
                                 "not have a valied connector ID.")
            connector = await self._require_connector_service(
            ).source_table_connector_for_project(
                project_id,
                connector_id,
                _current_user_ctx.get(),
            )
            config_as_dict["connector_id"] = str(connector_id)
            if connector is None:
                config_as_dict["is_project_upload"] = True
            else:
                config_as_dict["lookup_name"] = connector.name
            connector_spec_arr.append(
                ConnectorSpec(
                    connector_id=connector_id,
                    connector_config=config_as_dict,
                ))

        return connector_spec_arr, table_confs

    async def _get_graph_script_for_sdk_impl(
        self,
        graph_id: GraphIDv2,
        project_id: ProjectID,
    ) -> str:
        """Implementation of get_graph_script_for_sdk for SQL DB."""
        try:
            _, fpid = GraphIDv2.parse(graph_id)
            assert fpid is not None
        except BaseException:
            raise ValueError(f"Table resource id is malformed: {graph_id}")

        metadata = await self._get_graph_sdk_metadata_from_state_services(
            fpid, project_id)
        connector_configs, table_configs, edge_configs = metadata
        final_script = generate_conn_graph_or_table_sdk_script(
            connector_configs,
            table_configs,
            edge_configs,  # type: ignore
            project_id=project_id,
        )
        return make_stype_and_dtype_sdk_compat(final_script)

    async def _get_graph_metadata_for_sdk_impl(
        self,
        graph_fpid: FPID,
        project_id: ProjectID,
    ) -> Tuple[List[ConnectorSpec], List[TableReferenceMetadata],
               List[graph_api.Edge]]:
        """Implementation of get_graph_metadata_for_sdk for SQL DB."""
        metadata = await self._get_graph_sdk_metadata_from_state_services(
            graph_fpid, project_id)
        return metadata
