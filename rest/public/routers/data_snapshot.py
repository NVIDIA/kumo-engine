# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Callable

import kumoapi.data_snapshot as snapshot_interfaces
from fastapi import APIRouter, Depends
from fastapi.params import Query
from kumoapi.graph import EdgeHealthResponse
from kumoapi.table import TableDefinition as ApiTableDefinition
from typing_extensions import Annotated

from io_utils import logging
from io_utils.json import from_json_dict, to_json_dict
from rest.config.graph_v2 import GraphSnapshotRequest
from rest.config.status import IDResponse
from rest.config.table_v2 import TableDefinition, TableSnapshotResource
from rest.deps.authentication import Authentication
from rest.driver.data_snapshot_api_driver import DataSnapshotApiDriver
from rest.utils.public import wrap_driver_errors
from rest.utils.routers import StrictQueryAPIRoute, wrap_validation_exceptions
from statedb.config.common import GraphIDv2, ProjectID

log = logging.getLogger(__name__)

_refresh_source_desc = """
Whether to force refreshing source table data snapshot, i.e., syncing with
external data source and fetching latest data version.
"""


def _public_table_snapshot(
    snapshot: TableSnapshotResource,
) -> snapshot_interfaces.TableSnapshotResource:
    r"""Deep-project an internal snapshot onto its five public fields."""
    payload = to_json_dict(snapshot)
    return from_json_dict(
        {
            'id': payload['id'],
            'table_definition': payload['table_definition'],
            'stats': payload['stats'],
            'column_stats': payload['column_stats'],
            'stages': payload['stages'],
        },
        snapshot_interfaces.TableSnapshotResource,
    )


class TableSnapshotRouter:
    def __init__(
        self,
        api_key_auth: Callable,
        driver: DataSnapshotApiDriver,
        auth: Authentication,
    ) -> None:
        self._driver = driver
        self._auth = auth

        self.router = APIRouter(
            prefix="/tablesnapshots",
            tags=["API v1 (Under Development): Table data snapshots"],
            dependencies=[Depends(api_key_auth)],
        )

        self.router.add_api_route(
            "",
            self.create,
            response_model=IDResponse,
            methods=['POST'],
            summary="Create a table data snapshot.",
        )

        self.router.add_api_route(
            '/{id}',
            self.get,
            response_model=snapshot_interfaces.TableSnapshotResource,
            methods=['GET'],
            summary='Get a table snapshot.',
        )

    async def create(
        self,
        table_definition: ApiTableDefinition,
        project_id: Annotated[str, Query()],
        refresh_source: Annotated[
            bool, Query(description=_refresh_source_desc)] = False,
    ) -> IDResponse:
        """Create a TableSnapshotResource."""
        with wrap_driver_errors():
            id = await self._driver.create_table_snapshot(
                TableDefinition(**dataclasses.asdict(table_definition)),
                refresh_source,
                project_id=ProjectID(project_id),
            )
        return IDResponse(id)

    # TODO(siyang): Add query params for sample, col stats, data uri, etc.
    async def get(
        self,
        id: snapshot_interfaces.TableSnapshotID,
        project_id: Annotated[str, Query()],
    ) -> snapshot_interfaces.TableSnapshotResource:
        with wrap_driver_errors():
            snapshot = await self._driver.get_table_snapshot(
                id, ProjectID(project_id))
            return _public_table_snapshot(snapshot)


class GraphSnapshotRouter:
    def __init__(
        self,
        api_key_auth: Callable,
        driver: DataSnapshotApiDriver,
        auth: Authentication,
    ) -> None:
        self._driver = driver
        self._auth = auth

        self.router = APIRouter(
            prefix="/graphsnapshots",
            tags=["API v1 (Under Development): Graph data snapshots"],
            dependencies=[Depends(api_key_auth)],
            route_class=StrictQueryAPIRoute,
        )

        self.router.add_api_route(
            "",
            self.create,
            response_model=IDResponse,
            methods=['POST'],
            summary="Create a graph data snapshot.",
        )

        self.router.add_api_route(
            '/{id}',
            self.get,
            response_model=snapshot_interfaces.GraphSnapshotResource,
            methods=['GET'],
            summary='Get a graph snapshot.',
        )

        self.router.add_api_route(
            '/edge_health/{id}',
            self.get_edge_health,
            response_model=EdgeHealthResponse,
            methods=['GET'],
            summary=(
                'Get edge health statistics corresponding to a graph snapshot.'
            ),
        )

    async def create(
        self,
        graph_id: Annotated[str,
                            Query(description='Graph resource identifier')],
        project_id: Annotated[str, Query()],
        refresh_source: Annotated[
            bool, Query(description=_refresh_source_desc)] = False,
    ) -> IDResponse:
        """Create a GraphSnapshotResource."""
        with wrap_driver_errors(), wrap_validation_exceptions():
            graph_fpid = GraphIDv2.from_public(graph_id).fpid
            assert graph_fpid is not None
            graph_snapshot_id = await self._driver.create_graph_snapshot(
                graph_fpid,
                refresh_source,
                project_id=ProjectID(project_id),
            )
        return IDResponse(graph_snapshot_id)

    async def get(
        self,
        id: snapshot_interfaces.GraphSnapshotID,
        project_id: Annotated[str, Query()],
    ) -> snapshot_interfaces.GraphSnapshotResource:
        with wrap_driver_errors(), wrap_validation_exceptions():
            graph_snapshot = await self._driver.get_graph_snapshot(
                GraphSnapshotRequest(graph_snapshot_id=id, ),
                ProjectID(project_id),
            )
            return graph_snapshot.snapshot_resource

    async def get_edge_health(
        self,
        id: snapshot_interfaces.GraphSnapshotID,
        project_id: Annotated[str, Query()],
    ) -> EdgeHealthResponse:
        with wrap_driver_errors(), wrap_validation_exceptions():
            return await self._driver.get_edge_health(id, ProjectID(project_id))
