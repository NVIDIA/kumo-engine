# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from http import HTTPStatus
from typing import Callable

import kumoapi.graph as api
from fastapi import APIRouter, Depends
from fastapi.params import Query
from typing_extensions import Annotated

from kumo.validation import ValidationResponse
from rest.config.desc import _force_graph_rename_desc
from rest.config.graph_v2 import GraphDefinition
from rest.config.status import IDResponse
from rest.deps.authentication import Authentication
from rest.driver.graph_driver import GraphDriver
from rest.utils.public import wrap_driver_errors
from rest.utils.routers import wrap_validation_exceptions
from statedb.config.common import (
    _FP_SUFFIX_SEPARATOR,
    ID,
    GraphIDv2,
    ProjectID,
)


class GraphRouter:
    def __init__(
        self,
        api_key_auth: Callable,
        graph_driver: GraphDriver,
        auth: Authentication,
    ) -> None:
        self._auth = auth
        self._graph_driver = graph_driver

        self.router = APIRouter(
            prefix="/graphs",
            tags=["API v1 (Under Development): Graphs"],
            dependencies=[Depends(api_key_auth)],
        )

        self.router.add_api_route(
            "",
            self.create_graph,
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
            "/infer_links",
            self.infer_links,
            response_model=api.GraphDefinition,
            methods=['POST'],
            summary="Infer links in a graph.",
        )

        self.router.add_api_route(
            "/{graph_id_or_alias}",
            self.get_graph,
            response_model=api.GraphResource,
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

        # TODO: list graphs? delete graph? maybe later on

    async def create_graph(
        self,
        graph: api.GraphDefinition,
        project_id: Annotated[str, Query()],
        name_alias: Annotated[str | None, Query(max_length=64)] = None,
        force_rename: Annotated[
            bool, Query(description=_force_graph_rename_desc)] = False,
        skip_validation: Annotated[
            bool,
            Query(description="Skip graph validation after creation")] = False,
    ) -> IDResponse:
        with wrap_driver_errors():
            name_alias = ID(name_alias) if name_alias else None
            graph_def = GraphDefinition(**dataclasses.asdict(graph))
            graph_id = await self._graph_driver.create_full_graph_v2(
                graph_def,
                alias=name_alias,
                force_rename=force_rename,
                skip_validation=skip_validation,
                project_id=ProjectID(project_id),
            )
            return IDResponse(id=graph_id)

    async def infer_links(
        self,
        request: api.GraphDefinition,
        project_id: Annotated[str, Query()],
    ) -> api.GraphDefinition:
        with wrap_driver_errors():
            graph_def = GraphDefinition(**dataclasses.asdict(request))
            return await self._graph_driver.infer_links(
                graph_def,
                ProjectID(project_id),
            )

    async def get_graph(
        self,
        graph_id_or_alias: str,
        project_id: Annotated[str, Query()],
    ) -> api.GraphResource:
        with wrap_driver_errors(), wrap_validation_exceptions():
            if _FP_SUFFIX_SEPARATOR in graph_id_or_alias:
                if not graph_id_or_alias.startswith('graph-'):
                    raise ValueError(f'Graph resource id is malformed '
                                     f': {graph_id_or_alias}')
                try:
                    fpid = GraphIDv2.parse(graph_id_or_alias)[1]
                    assert fpid is not None
                except BaseException:
                    raise ValueError(f'Graph resource id is malformed '
                                     f': {graph_id_or_alias}')
                graph_def, _, _ = await self._graph_driver.get_graph_v2(
                    fpid, ProjectID(project_id))
            else:
                alias = ID(graph_id_or_alias)
                graph_def, _, _ = (await
                                   self._graph_driver.get_graph_v2_by_alias(
                                       alias, ProjectID(project_id)))
            return graph_def

    async def validate(
        self,
        request: api.GraphValidationRequest,
        project_id: Annotated[str, Query()],
    ) -> ValidationResponse:
        with wrap_driver_errors():
            return await self._graph_driver.validate_graph(
                request.graph_definition,
                ProjectID(project_id),
            )
