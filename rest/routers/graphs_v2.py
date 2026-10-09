# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from typing import List

import kumoapi.graph as api
from fastapi import APIRouter, Depends
from fastapi.params import Query
from typing_extensions import Annotated

from kumo.validation import ValidationResponse
from rest.config.desc import (
    _force_graph_rename_desc,
    _skip_graph_validation_desc,
)
from rest.config.graph import EdgeHealthStats
from rest.config.graph_v2 import (
    GraphDefinition,
    GraphResourceResponse,
    ListResponse,
    build_graph_definition,
)
from rest.config.status import EmptyResponse, IDResponse
from rest.deps.authentication import Authentication
from rest.driver.graph_driver import GraphDriver
from rest.utils.project_scope import require_project_id
from rest.utils.public import wrap_driver_errors
from rest.utils.routers import (
    StrictQueryAPIRoute,
    ValidateNameResponse,
    wrap_validation_exceptions,
)
from statedb.config.common import (
    _FP_SUFFIX_SEPARATOR,
    ID,
    GraphIDv2,
    ProjectID,
)
from statedb.utils.errors import GraphDoesNotExistError


class GraphRouter:
    """
    V2 graph resources.
    """
    def __init__(
        self,
        auth: Authentication,
        graph_driver: GraphDriver,
    ) -> None:
        self._graph_driver = graph_driver
        self._auth = auth

        self.router = APIRouter(
            prefix="/v1/graphs",
            route_class=StrictQueryAPIRoute,
            tags=["API v1 (Under Development): Graphs"],
            dependencies=[Depends(auth.authenticate_request_header)],
        )

        self.router.add_api_route(
            "/",
            self.create_graph_v2,
            response_model=IDResponse,
            methods=['POST'],
            summary="Create a new graph with its full definition.",
            responses={
                HTTPStatus.UNPROCESSABLE_ENTITY: {
                    'description':
                    "Unable to create a new Graph as the its definition is "
                    "invalid."
                }
            },
        )

        self.router.add_api_route(
            "/{graph_alias}",
            self.delete_graph_v2,
            response_model=IDResponse,
            methods=['DELETE'],
        )

        self.router.add_api_route(
            "/{graph_id_or_alias}",
            self.get_graph_v2,
            response_model=GraphResourceResponse,
            methods=['GET'],
            summary="Get a graph with its full definition by id or alias",
            responses={
                HTTPStatus.NOT_FOUND: {
                    'description':
                    "Unable to find the graph with the given id or name alias."
                }
            },
        )

        self.router.add_api_route(
            "/validate",
            self.validate,
            response_model=ValidationResponse,
            methods=['POST'],
        )

        self.router.add_api_route(
            "/utils/validate_name",
            self.validate_name_v2,
            response_model=ValidateNameResponse,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/",
            self.list_graph_names_v2,
            response_model=List[ListResponse],
            methods=['GET'],
            summary="List all V2 graphs; used for new UI.",
        )

        self.router.add_api_route(
            "/suggest_links",
            self.suggest_links_v2,
            response_model=List[api.ColumnKeyGroup],
            methods=['POST'],
        )

        self.router.add_api_route(
            "/{graph_id}/edge_stats",
            self.get_edge_stats,
            response_model=EdgeHealthStats,
            methods=['GET'],
            summary="Info about edge stats such as cols, link type, etc.",
        )

        self.router.add_api_route(
            "/{graph_id}/refresh",
            self.refresh_graph_v2,
            response_model=EmptyResponse,
            methods=['POST'],
            summary="Refresh a single graph by id.",
            responses={
                HTTPStatus.NOT_FOUND: {
                    'description': "Unable to find the graph with the given id."
                }
            },
        )

        self.router.add_api_route(
            "/{graph_id}/sdk",
            self.graph_to_sdk,
            response_model=str,
            methods=['GET'],
            summary="Fully executable SDK script to load a graph.",
        )

    async def validate_name_v2(
        self,
        name: str,
        project_id: str,
    ) -> ValidateNameResponse:
        """Validates whether a graph name can be used to create a new graph.
        Names must be valid GraphIDs, and should be unique.
        """
        return await self._graph_driver.validate_project_graph_name(
            project_id=project_id,
            name=name,
        )

    async def create_graph_v2(
        self,
        graph: api.GraphDefinition,
        project_id: Annotated[str, Query()],
        name_alias: Annotated[str | None, Query(max_length=64)] = None,
        force_rename: Annotated[
            bool, Query(description=_force_graph_rename_desc)] = False,
        skip_validation: Annotated[
            bool, Query(description=_skip_graph_validation_desc)] = False,
    ) -> IDResponse:
        """
        Create a GraphDefinition from api.GraphDefinition.

        Args:
            graph (api.GraphDefinition):
                Immutable metadata definition for a graph.
            name_alias (str):
                Optional human-friendly name alias for the graph.
            force_rename (bool): Check if
                - the graph entry already has a different name alias; or
                - there is a name collision, i.e., another graph has the same
                `alias`
            skip_validation (bool): Skip graph validation after creation.

        Returns:
            IDResponse of type GraphIDv2 for the created graph.
        """
        project_id = require_project_id(project_id)
        with wrap_validation_exceptions():
            name_alias = ID(name_alias) if name_alias else None

            graph_def = build_graph_definition(graph)

            graph_id = await self._graph_driver.create_full_graph_v2(
                graph_def,
                alias=name_alias,
                force_rename=force_rename,
                skip_validation=skip_validation,
                project_id=project_id,
            )

            return IDResponse(id=graph_id)

    async def delete_graph_v2(
        self,
        graph_alias: str,
        project_id: Annotated[str, Query()],
    ) -> IDResponse:
        """Clear a graph display name by canonical ID or project-local alias.

        The response preserves the legacy contract by echoing the submitted
        identifier.
        """
        with wrap_validation_exceptions():
            await self._graph_driver.delete_graph_alias(graph_alias,
                                                        project_id=project_id)
        return IDResponse(graph_alias)

    async def get_graph_v2(
        self,
        graph_id_or_alias: str,
        project_id: str,
    ) -> GraphResourceResponse:
        """Get a single, V2 graph by ID or alias.

        Args:
            graph_id_or_alias (str):
            project_id (str): Optional project scope. When supplied, resolves
                the graph via the project entry and enriches the response with
                project-scoped metadata.

        Returns:
            GraphResource: A config object containing
            graph resource ID, immutable graph metadata definition,
            and graph alias.
        """
        project_id = require_project_id(project_id)

        async def _get_graph_response(id: str):
            with wrap_validation_exceptions(), wrap_driver_errors():
                if _FP_SUFFIX_SEPARATOR in id:
                    if not id.startswith('graph-') or id.startswith('NA-'):
                        raise ValueError(
                            f'Graph resource id is malformed: {id}')
                    try:
                        fpid = GraphIDv2(id).fpid
                        assert fpid is not None
                    except BaseException:
                        raise ValueError(
                            f"Graph resource id is malformed: {id}")
                    (
                        graph_def,
                        tstamps,
                        table_mapping,
                    ) = await self._graph_driver.get_graph_v2(
                        fpid, ProjectID(project_id))
                else:
                    alias = ID(id)
                    (
                        graph_def,
                        tstamps,
                        table_mapping,
                    ) = await self._graph_driver.get_graph_v2_by_alias(
                        alias, ProjectID(project_id))

                num_links = sum(
                    len(group.columns) - 1
                    for group in graph_def.graph.col_groups)
                num_tables = len(graph_def.graph.tables)
                return GraphResourceResponse(
                    id=graph_def.id,
                    graph=graph_def.graph,
                    name_alias=graph_def.name_alias,
                    timestamps=tstamps,
                    table_name_to_fpid_mapping=table_mapping,
                    num_links=num_links,
                    num_tables=num_tables,
                )

        with wrap_validation_exceptions(
                not_exists_error=GraphDoesNotExistError), wrap_driver_errors():
            entry = await self._graph_driver.resolve_project_graph(
                project_id, graph_id_or_alias)
            response = await _get_graph_response(str(GraphIDv2.public(entry.id))
                                                 )
            response.created_by = entry.created_by
            response.created_by_name = entry.created_by_name
            return response

    async def validate(
        self,
        request: api.GraphDefinition,
        project_id: Annotated[str, Query()],
    ) -> ValidationResponse:
        with wrap_validation_exceptions(), wrap_driver_errors():
            return await self._graph_driver.validate_graph(
                request,
                ProjectID(project_id),
            )

    async def list_graph_names_v2(
        self,
        project_id: str,
    ) -> List[ListResponse]:
        """Get a list of all v2 graph names.

        Returns:
            List[ListResponse]: A config object containing
            graph template alias (name).
        """
        with wrap_validation_exceptions():
            return await self._graph_driver.list_project_graph_names(project_id)

    async def suggest_links_v2(
        self,
        request: GraphDefinition,
        project_id: Annotated[str, Query()],
    ) -> List[api.ColumnKeyGroup]:
        """Get a list of all potential graph links.

        Returns:
            List[ColumnKeyGroup]: A list of columns that
            should be linked together.
        """
        return await self._graph_driver.suggest_links_v2(
            request, ProjectID(project_id))

    async def get_edge_stats(
        self,
        graph_id: str,
        project_id: ProjectID,
    ) -> EdgeHealthStats:
        """Get edge stats for a v2 graph.

        Args:
            graph_id(str): String that's
            either graph-{fpid} or NA-{fpid}.
        Returns:
            EdgeHealthStats: A config object containing
            edge stats.
        """
        with wrap_validation_exceptions(
                not_exists_error=GraphDoesNotExistError, ):
            return await self._graph_driver.get_edge_stats(
                GraphIDv2.from_public(graph_id), project_id)

    async def refresh_graph_v2(
        self,
        graph_id: str,
        project_id: ProjectID,
    ) -> EmptyResponse:
        """Refresh a single v2 graph by id.

        Args:
            graph_id(str): String that's either graph-{fpid} or NA-{fpid}.
        """
        with wrap_validation_exceptions(
                not_exists_error=GraphDoesNotExistError):
            await self._graph_driver.refresh_graph_v2(
                GraphIDv2.from_public(graph_id), project_id)
        return EmptyResponse()

    async def graph_to_sdk(
        self,
        graph_id: GraphIDv2,
        project_id: Annotated[str, Query()],
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
        generated_script = ""
        with wrap_validation_exceptions(), wrap_driver_errors():
            if _FP_SUFFIX_SEPARATOR not in graph_id:
                raise ValueError(f"Graph resource id is malformed : {graph_id}")
            generated_script = await self._graph_driver.graph_to_sdk(
                graph_id, ProjectID(require_project_id(project_id)))
        if generated_script == "":
            raise ValueError(f"SDK script for graph with ID {graph_id}" +
                             "could not be generated.")
        return generated_script
