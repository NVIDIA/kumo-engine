# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from http import HTTPStatus
from typing import Callable

import kumoapi.table as api
from fastapi import APIRouter, Depends
from fastapi.params import Query
from kumo_core.types.schema import FPID
from typing_extensions import Annotated

from kumo.validation import ValidationResponse
from rest.config.desc import _force_table_rename_desc
from rest.config.status import IDResponse
from rest.config.table_v2 import TableDefinition
from rest.deps.authentication import Authentication
from rest.driver.table_driver import TableDriver
from rest.utils.public import wrap_driver_errors
from rest.utils.routers import wrap_validation_exceptions
from statedb.config.common import (
    _FP_SUFFIX_SEPARATOR,
    ID,
    ProjectID,
    TableIDv2,
)


class TableRouter:
    def __init__(
        self,
        api_key_auth: Callable,
        table_driver: TableDriver,
        auth: Authentication,
    ) -> None:
        # Use the predictive query API driver, as there is no reason to further
        # create a table API driver (the internal REST API has no use for this
        # concept):
        self._auth = auth
        self._table_driver = table_driver

        self.router = APIRouter(
            prefix="/tables",
            tags=["API v1 (Under Development): Tables"],
            dependencies=[Depends(api_key_auth)],
        )

        self.router.add_api_route(
            "",
            self.create_table,
            response_model=IDResponse,
            methods=['POST'],
            summary="Create a new table with its full definition.",
            responses={
                HTTPStatus.UNPROCESSABLE_ENTITY: {
                    'description':
                    "A table name is required or the table definition is "
                    "invalid."
                }
            },
        )

        self.router.add_api_route(
            "/{table_id_or_alias}",
            self.get_table,
            response_model=api.TableResource,
            methods=['GET'],
            summary="Get a table with its full definition by id or alias",
            responses={
                HTTPStatus.NOT_FOUND: {
                    'description':
                    "Unable to find the table with the given id or name alias."
                }
            },
        )

        self.router.add_api_route(
            "/infer_metadata",
            self.infer_metadata,
            response_model=api.TableMetadataResponse,
            methods=['POST'],
        )

        self.router.add_api_route(
            "/resolve_sql",
            self.resolve_sql,
            response_model=api.SqlTableResolveResponse,
            methods=['POST'],
        )

        self.router.add_api_route(
            "/validate",
            self.validate,
            response_model=ValidationResponse,
            methods=['POST'],
        )

    async def create_table(
        self,
        table: api.TableDefinition,
        project_id: Annotated[str, Query()],
        name_alias: Annotated[str, Query(min_length=1, max_length=64)],
        force_rename: Annotated[
            bool, Query(description=_force_table_rename_desc)] = False,
    ) -> IDResponse:
        """Create a table under a human-friendly alias."""
        with wrap_driver_errors():
            name_alias = ID(name_alias)
            table_def = TableDefinition(**dataclasses.asdict(table))
            table_id = await self._table_driver.create_full_table(
                table_def,
                alias=name_alias,
                force_rename=force_rename,
                project_id=ProjectID(project_id),
            )
            return IDResponse(id=table_id)

    async def get_table(
        self,
        table_id_or_alias: str,
        project_id: Annotated[str, Query()],
    ) -> api.TableResource:
        with wrap_driver_errors(), wrap_validation_exceptions():
            if _FP_SUFFIX_SEPARATOR in table_id_or_alias:
                if not table_id_or_alias.startswith('table-'):
                    raise ValueError(f'Table resource id is malformed '
                                     f': {table_id_or_alias}')
                try:
                    table_id_or_alias = f'NA.{table_id_or_alias}'
                    fpid = (TableIDv2.parse(
                        TableIDv2.from_public(table_id_or_alias))[2])
                    assert fpid is not None
                    fpid = FPID(fpid)
                except BaseException:
                    raise ValueError(f'Table resource id is malformed'
                                     f': {table_id_or_alias}')
                # TODO(manan): fix this later.
                out = await self._table_driver.get_table(fpid, project_id)
            else:
                alias = ID(table_id_or_alias)
                # TODO(manan): fix this later.
                out = await self._table_driver.get_table_by_name(
                    alias, project_id)
            return api.TableResource(
                id=out.id,
                table=out.table,
                name_alias=out.name_alias,
            )

    async def infer_metadata(
        self,
        request: api.TableMetadataRequest,
        project_id: Annotated[str, Query()],
    ) -> api.TableMetadataResponse:
        r"""Infers the metadata of this Kumo table, given a (potentially
        incomplete) table definition."""
        with wrap_driver_errors():
            return await self._table_driver.infer_table_metadata(
                request, project_id=project_id)

    async def resolve_sql(
        self,
        request: api.SqlTableResolveRequest,
        project_id: Annotated[str, Query()],
    ) -> api.SqlTableResolveResponse:
        """Resolve and validate a Custom SQL table authored through the SDK."""
        with wrap_driver_errors():
            return await self._table_driver.resolve_sql(request,
                                                        project_id=project_id)

    async def validate(
        self,
        request: api.TableValidationRequest,
        project_id: Annotated[str, Query()],
    ) -> ValidationResponse:
        """Validate a table definition.

        Well-formed requests return semantic results in ``ValidationResponse``;
        malformed request bodies return a structured 422.
        """
        with wrap_driver_errors():
            return await self._table_driver.validate_table(
                request.table_definition, project_id=project_id)
