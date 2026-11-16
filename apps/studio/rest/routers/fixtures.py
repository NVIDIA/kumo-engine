# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from dataclasses import dataclass
from http import HTTPStatus
from typing import List, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import HTTPException

from async_utils.sync import blocking_func
from common_utils.string import validate_id, validate_max_length
from io_utils import fsutil, logging
from io_utils.json import from_json_dict, save_json
from rest.config.api_command import FixtureDefinition
from rest.config.status import EmptyResponse
from rest.deps.authentication import (
    Authentication,
    get_user_authorization_from_request,
)
from rest.driver.rest_api_driver import RestApiDriver
from rest.routers.fixture_utils import fixture_to_pquery_resources_v2
from rest.routers.graphs_v2 import GraphRouter
from rest.routers.tables_v2 import TableRouter
from statedb.config.common import PredictiveQueryID, ProjectID
from statedb.rest.connector_authorization import CONNECTOR_GRAPH_ACTIONS

log = logging.getLogger(__name__)


@dataclass
class SavePQueryArgs:
    project_id: ProjectID
    output_dir: Optional[str] = None
    pquery_ids: Optional[List[PredictiveQueryID]] = None
    fixture_name: Optional[str] = None


@dataclass
class LoadJsonArgs:
    file_path: str
    project_id: ProjectID


class FixturesResource:
    def __init__(
        self,
        auth: Authentication,
        driver: RestApiDriver,
        table_router_v2: TableRouter,
        graph_router_v2: GraphRouter,
    ) -> None:
        self._driver = driver
        self._table_router_v2 = table_router_v2
        self._graph_router_v2 = graph_router_v2
        self.router = APIRouter(
            prefix="/v0/fixtures",
            tags=["Fixtures"],
            dependencies=[Depends(auth.authenticate_request_header)],
            responses={HTTPStatus.NOT_FOUND: {
                "description": "Not found"
            }},
        )

        self.router.add_api_route(
            '/save_json',
            self.save_json,
            response_model=str,
            response_description='Successfully saved the fixtures as json.',
            methods=['POST'],
        )

        self.router.add_api_route('/load_json', self.load_json_file,
                                  response_model=EmptyResponse,
                                  methods=['POST'])

    async def save_json(self, args: SavePQueryArgs, request: Request) -> str:
        r"""Save predictive-query metadata as a credential-free fixture.

        Identities and project-owned connectors are intentionally not exported.
        The fixture retains connector IDs in its table definitions, and loading
        requires those connectors to exist in the exact requested project.

        Parameters:
            output_dir (str, optional): The dir for output json file. If it is
            not specified the json file is saved to the configured fixture
            directory.

            pquery_ids: Zero or more predictive-query IDs. If omitted, all
            queries visible in the exact project are included.

        Returns: The fixture JSON file path.

        Errors:
            HTTPStatus.NOT_FOUND: If any specified predictive query does not
            exist.
        """
        db_dump = await self._driver.dump_as_fixtures_v2(
            args.pquery_ids,
            args.project_id,
            get_user_authorization_from_request(request),
        )
        if not db_dump.queries:
            raise HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY,
                                detail='No pquery available to save.')
        check_and_raise_error_for_multiple_resources(db_dump)
        if args.fixture_name:
            file_name = f'{args.fixture_name}.json'
        else:
            query_names = '_'.join(
                sorted([q.pquery_name for q in db_dump.queries]))
            file_name = f'{db_dump.queries[0].graph_fpid}-{query_names}.json'
        output_dir = (args.output_dir
                      or self._driver._settings.fixture_export_path)

        file_path = os.path.join(output_dir, file_name)
        return await save_json_to_file_path.coroutine(db_dump, file_path)

    async def load_json_file(self, request_args: LoadJsonArgs,
                             request: Request) -> EmptyResponse:
        @blocking_func()
        def _load_fixture(path: str) -> FixtureDefinition:
            return from_json_dict(json.loads(fsutil.read_binary(path)),
                                  FixtureDefinition)

        fixture = await _load_fixture.coroutine(request_args.file_path)
        principal = get_user_authorization_from_request(request)
        connector_service = self._driver._require_state_services().connectors
        connector_ids = set()
        for graph in fixture.graphs:
            connector_ids.update(
                connector_service.connector_ids_for_graph(graph))
        for connector_id in connector_ids:
            await connector_service.execution_ref_for_project(
                principal, request_args.project_id, connector_id,
                CONNECTOR_GRAPH_ACTIONS)

        pquery_resources = await fixture_to_pquery_resources_v2.coroutine(
            fixture)
        for pquery_resource in pquery_resources:
            await self._driver.pquery_driver.create_predictive_query_v2(
                pquery_resource, project_id=request_args.project_id)

        return EmptyResponse()


@blocking_func()
def save_json_to_file_path(db_dump, file_path: str) -> str:
    save_json(db_dump, file_path, compressed=False)
    return fsutil.abspath(file_path)


def check_and_raise_error_for_multiple_resources(db_dump: FixtureDefinition):
    entity_name = ""
    if len(db_dump.graphs) > 1:
        entity_name = "Graph"
    elif len(db_dump.queries) > 1:
        entity_name = "Query"
    if entity_name:
        raise HTTPException(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            detail=(f'Cannot create fixture for Predictive Query containing '
                    f'more than one {entity_name}'))


def validate_name(fixture_name: str):
    try:
        validate_id(fixture_name)
        validate_max_length(fixture_name, 64)
    except Exception as e:
        if ("Identifier may only contain alphanumeric characters "
                "and underscores" in str(e)):
            raise HTTPException(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                detail=(f'Fixture name should only contain '
                        f'alphanumeric characters and '
                        f'underscores and must begin with '
                        f'a letter. Got: "{fixture_name}"'))
        if ("Identifier must be fewer than" in str(e)):
            raise HTTPException(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                detail=(f"Fixture name must be fewer than 64 "
                        f"characters long. Got: {fixture_name}"))
