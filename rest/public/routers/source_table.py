# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from http import HTTPStatus
from typing import Callable
from urllib.parse import urlsplit

import kumoapi.source_table as api
from fastapi import APIRouter, Depends, Query, Request

from common_utils.string import validate_connector_relative_source_name
from kumo.connector.connector import ConnectorError
from rest.deps.authentication import (
    Authentication,
    get_user_authorization_from_request,
)
from rest.driver.connector_api_driver import ConnectorApiDriver
from rest.utils.public import wrap_driver_errors
from rest.utils.routers import wrap_validation_exceptions
from statedb.config.common import ConnectorID, ProjectID
from statedb.utils.errors import ConnectorDoesNotExistError


class SourceTableRouter:
    """Project-scoped discovery for an existing connector.

    There is deliberately no group, raw root directory, credential, or
    connectorless variant. The project path is the authorization and ownership
    boundary, and the connector service verifies the exact connector belongs
    to it before any external access occurs.

    A missing connector is a 404 on every route, matching the parent
    ``GET .../connectors/{connector_id}`` route. Columns and sample reads also
    return 404 when an exact physical source is missing or cannot be
    distinguished from one hidden by provider access controls.
    ``ConnectorDoesNotExistError`` is a ``ValueError`` and exceptions unwind
    inner-first, so the inner
    ``wrap_validation_exceptions(not_exists_error=...)`` must own that
    mapping; a bare inner wrapper would emit a 422 before the outer
    ``wrap_driver_errors`` entry, kept as documentation of intent, could fire.
    """
    def __init__(
        self,
        api_key_auth: Callable,
        driver: ConnectorApiDriver,
        auth: Authentication,
    ) -> None:
        del auth
        self._driver = driver
        self.router = APIRouter(
            prefix=("/projects/{project_id}/connectors/{connector_id}"),
            tags=["API v1: Project Connector Discovery"],
            dependencies=[Depends(api_key_auth)],
        )
        self.router.add_api_route(
            "/source-tables",
            self.list_source_tables,
            response_model=api.SourceTableListResponse,
            methods=["GET"],
        )
        self.router.add_api_route(
            "/source-table-config",
            self.get_source_table_config,
            response_model=api.SourceTableConfigResponse,
            methods=["GET"],
        )
        self.router.add_api_route(
            "/columns",
            self.get_columns,
            response_model=api.SourceTableDataResponse,
            methods=["GET"],
            responses={
                HTTPStatus.NOT_FOUND: {
                    "description":
                    "The connector or requested physical source was not found."
                },
            },
        )
        self.router.add_api_route(
            "/sample",
            self.get_sample,
            response_model=api.SourceTableDataResponse,
            methods=["GET"],
            responses={
                HTTPStatus.NOT_FOUND: {
                    "description":
                    "The connector or requested physical source was not found."
                },
            },
        )

    @staticmethod
    def _principal(request: Request):
        return get_user_authorization_from_request(request)

    @staticmethod
    def _source_name(table_name: str) -> str:
        """Preserve full URIs for connector-aware validation in the driver."""
        if urlsplit(table_name).scheme:
            return table_name
        validate_connector_relative_source_name(table_name)
        return table_name

    async def list_source_tables(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        request: Request,
    ) -> api.SourceTableListResponse:
        with wrap_driver_errors({
                ConnectorDoesNotExistError: HTTPStatus.NOT_FOUND,
                ConnectorError: HTTPStatus.UNPROCESSABLE_ENTITY,
        }), wrap_validation_exceptions(
                not_exists_error=ConnectorDoesNotExistError):
            table_names = await self._driver.get_project_source_table_names(
                project_id,
                connector_id,
                self._principal(request),
            )
        return api.SourceTableListResponse(table_names=table_names)

    async def get_columns(
            self,
            project_id: ProjectID,
            connector_id: ConnectorID,
            request: Request,
            table_name: str = Query(),
    ) -> api.SourceTableDataResponse:
        """Return columns, or 404 for a missing or unavailable physical source."""
        with wrap_driver_errors({
                ConnectorDoesNotExistError: HTTPStatus.NOT_FOUND,
                ConnectorError: HTTPStatus.UNPROCESSABLE_ENTITY,
        }), wrap_validation_exceptions(
                not_exists_error=ConnectorDoesNotExistError):
            source_table_name = self._source_name(table_name)
            response = await self._driver.get_project_source_columns(
                project_id,
                connector_id,
                self._principal(request),
                source_table_name,
                skip_cache=False,
            )
        return api.SourceTableDataResponse(
            table_name=source_table_name,
            cols=response.source_cols,
        )

    async def get_sample(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        request: Request,
        table_name: str = Query(),
        sample_rows: int = Query(default=10, ge=1, le=1000),
    ) -> api.SourceTableDataResponse:
        """Return sample rows, or 404 for a missing or unavailable source."""
        with wrap_driver_errors({
                ConnectorDoesNotExistError: HTTPStatus.NOT_FOUND,
                ConnectorError: HTTPStatus.UNPROCESSABLE_ENTITY,
        }), wrap_validation_exceptions(
                not_exists_error=ConnectorDoesNotExistError):
            source_table_name = self._source_name(table_name)
            response = await self._driver.get_project_sample_rows(
                project_id,
                connector_id,
                self._principal(request),
                source_table_name,
                num_rows=sample_rows,
                serialize_to_json=True,
                require_stats=False,
            )
        return api.SourceTableDataResponse(
            table_name=source_table_name,
            cols=response.columns,
            sample_rows=response.sample_df_serialized,
        )

    async def get_source_table_config(
            self,
            project_id: ProjectID,
            connector_id: ConnectorID,
            request: Request,
            table_name: str = Query(),
    ) -> api.SourceTableConfigResponse:
        """Resolve one source config; invalid warehouse coordinates return 422."""
        principal = self._principal(request)
        with wrap_driver_errors({
                ConnectorDoesNotExistError: HTTPStatus.NOT_FOUND,
                ConnectorError: HTTPStatus.UNPROCESSABLE_ENTITY,
        }), wrap_validation_exceptions(
                not_exists_error=ConnectorDoesNotExistError):
            source_table_name = self._source_name(table_name)
            return await self._driver.get_project_source_table_config(
                project_id,
                connector_id,
                principal,
                source_table_name,
            )
