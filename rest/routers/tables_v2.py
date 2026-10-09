# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from http import HTTPStatus
from typing import List, Optional

import kumoapi.table as api
from fastapi import APIRouter, Depends, Query
from typing_extensions import Annotated

from kumo.data.backend_cache import TableDataTaskResult
from kumo.validation import ValidationResponse
from rest.config.graph_v2 import ListResponse
from rest.config.status import EmptyResponse, IDResponse
from rest.config.table import ColStatsResponse
from rest.config.table_v2 import TableDefinition, TableResourceResponse
from rest.deps.authentication import Authentication
from rest.driver.table_driver import TableDriver
from rest.utils.project_scope import require_project_id
from rest.utils.public import wrap_driver_errors
from rest.utils.routers import (
    StrictQueryAPIRoute,
    ValidateNameResponse,
    wrap_validation_exceptions,
)
from statedb.config.common import (
    ID,
    ConnectorID,
    LastIngestedTimestamp,
    TableIDv2,
    TableName,
)
from statedb.utils.errors import (
    ConnectorDoesNotExistError,
    TableDoesNotExistError,
)


class TableRouter:
    def __init__(
        self,
        auth: Authentication,
        table_driver: TableDriver,
    ) -> None:
        self._auth = auth
        self._table_driver = table_driver

        self.router = APIRouter(
            prefix="/v1/tables",
            route_class=StrictQueryAPIRoute,
            tags=["API v1 (Under Development): Tables"],
            dependencies=[Depends(auth.authenticate_request_header)],
            responses={HTTPStatus.NOT_FOUND: {
                "description": "Not found"
            }},
        )

        self.router.add_api_route(
            "/",
            self.create_table_v2,
            response_model=IDResponse,
            methods=['POST'],
        )

        self.router.add_api_route(
            "/{table_alias}",
            self.delete_table_v2_alias,
            response_model=IDResponse,
            methods=['DELETE'],
        )

        self.router.add_api_route(
            "/{table_id_or_name}",
            self.get_table_v2,
            response_model=TableResourceResponse,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/",
            self.list_table_names,
            response_model=List[ListResponse],
            methods=['GET'],
            summary="List all of the tables " + "for the Tables page.",
            response_description="A list of all the names and " +
            "creation dates for all table templates.",
        )

        self.router.add_api_route(
            "/validate",
            self.validate_table_v2,
            response_model=ValidationResponse,
            methods=['POST'],
            summary="Validate a V2 table.",
        )

        self.router.add_api_route(
            "/resolve_sql",
            self.resolve_sql,
            response_model=api.SqlTableResolveResponse,
            methods=['POST'],
            summary="Resolve a Custom SQL query into a table definition.",
        )

        self.router.add_api_route(
            "/infer_metadata",
            self.infer_metadata,
            response_model=api.TableMetadataResponse,
            methods=['POST'],
            summary="Infer stats-based metadata for a table from its data "
            "source.",
        )

        self.router.add_api_route(
            "/{table_id}/col_stats",
            self.get_col_stats,
            response_model=ColStatsResponse,
            methods=['GET'],
            summary="Get exact column stats for the specified table",
        )

        self.router.add_api_route(
            "/{connector_id}/list_tables",
            self.list_v2_tables_in_connector,
            response_model=List[TableResourceResponse],
            methods=['GET'],
            summary="Get all tables in a connector.",
        )

        self.router.add_api_route(
            "/{table_id}/refresh",
            self.refresh_table_v2,
            response_model=EmptyResponse,
            methods=['POST'],
        )

        self.router.add_api_route(
            "/{table_id}/last_refreshed",
            self.get_last_ingested_timestamp,
            response_model=LastIngestedTimestamp,
            methods=['GET'],
            summary="Get the time at which the table was last ingested.",
        )

        self.router.add_api_route(
            "/{table_name}/validate_name",
            self.validate_v2_table_name,
            response_model=ValidateNameResponse,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{table_id}/sdk",
            self.table_to_sdk,
            response_model=str,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{table_id}/validation",
            self.get_vals,
            response_model=Optional[TableDataTaskResult],
            methods=['GET'],
            summary="If table is ingested, get any warnings or errors.",
        )

    async def create_table_v2(
        self,
        table: api.TableDefinition,
        project_id: Annotated[str, Query()],
        name_alias: Annotated[str, Query(min_length=1, max_length=64)],
        force_rename: Annotated[bool, Query()] = False,
    ):
        """
        Create a TableDefinition from UI.

        Args:
            table (api.TableDefinition):
                Immutable metadata definition for a table.
            name_alias (str): Human-friendly name for the table.
        """
        project_id = require_project_id(project_id)
        with wrap_validation_exceptions():
            name_alias = ID(name_alias)

            table_def = TableDefinition(**dataclasses.asdict(table))

            table_id = await self._table_driver.create_full_table(
                table_def,
                alias=name_alias,
                force_rename=force_rename,
                project_id=project_id,
            )

            return IDResponse(table_id)

    async def delete_table_v2_alias(
        self,
        table_alias: str,
        project_id: Annotated[str, Query()],
    ) -> IDResponse:
        """Clear a table's project-local alias.

        Canonical IDs are not accepted, and the fingerprint resource is
        preserved.
        """
        project_id = require_project_id(project_id)
        with wrap_driver_errors():
            tableid = await self._table_driver.delete_table_alias(
                TableName(table_alias),
                project_id=project_id,
            )
            return IDResponse(tableid)

    async def get_table_v2(
        self,
        table_id_or_name: str,
        project_id: str,
    ) -> TableResourceResponse:
        """
        Get a single (V2) Kumo table by ID.

        Args:

            table_id [TableIDv2]: The unique ID of the table composed of
            3 componenents formatted as pquery_id.table_name-fpid.hex_fp.
            Currently, for pquery_id and table_name we use `NA`.

        Return:
            TableResourceResponse:
                A config object containing table template metadata.
        """
        project_id = require_project_id(project_id)

        with wrap_validation_exceptions(
                not_exists_error=TableDoesNotExistError), wrap_driver_errors():
            entry = await self._table_driver.resolve_project_table(
                project_id, table_id_or_name)
            return await self._table_driver.get_table(entry.id, project_id)

    async def list_table_names(
        self,
        project_id: str,
    ) -> List[ListResponse]:
        """Get a list of all v2 table names.

        Returns:
            List[ListResponse]: A config object containing
            table template name and timestamps.
        """
        with wrap_validation_exceptions():

            return await self._table_driver.list_project_table_names(project_id)

    async def validate_table_v2(
        self,
        request: api.TableDefinition,
        project_id: Annotated[str, Query()],
    ) -> ValidationResponse:
        """Validate a table definition.

        Well-formed requests return semantic results in ``ValidationResponse``;
        malformed request bodies return a structured 422.
        """
        with wrap_driver_errors():
            return await self._table_driver.validate_table(
                request, project_id=project_id)

    async def resolve_sql(
        self,
        request: api.SqlTableResolveRequest,
        project_id: Annotated[str, Query()],
    ) -> api.SqlTableResolveResponse:
        with wrap_driver_errors():
            return await self._table_driver.resolve_sql(request,
                                                        project_id=project_id)

    async def infer_metadata(
        self,
        request: api.TableMetadataRequest,
        project_id: Annotated[str, Query()],
    ) -> api.TableMetadataResponse:
        r"""Infers the metadata of this Kumo table, given a (potentially
        incomplete) table definition.

        This is the same stats-based inference the public SDK endpoint
        (``POST /tables/infer_metadata``) dispatches: it delegates to
        :meth:`TableDriver.infer_table_metadata`, so a table authored from
        this route's response gets the same default semantic types as an
        SDK-authored table on the identical source. The UI table-create
        wizard calls this route to align its column-type defaults with the
        SDK path (the connector ``columns``/``sample`` previews it renders
        first are stats-free and may disagree; see kumo-ai/kumo#30131).
        """
        with wrap_driver_errors():
            return await self._table_driver.infer_table_metadata(
                request, project_id=project_id)

    async def get_col_stats(
        self,
        table_id: str,
        project_id: str,
    ) -> ColStatsResponse:
        """
        Get exact column stats for the specified table.
        This includes the column semantic type as well as associated stats
        like counts and distribution summary statistics.

        Args:
            table_id (TableIDv2): The unique ID of the table.
            Usually `NA.NA-fpid`.
        """
        with wrap_validation_exceptions(
                not_exists_error=TableDoesNotExistError):

            entry = await self._table_driver.resolve_project_table(
                project_id, table_id)
            table_id = TableIDv2.public(
                pquery_id=None,
                table_name=TableName("NA"),
                fpid=entry.id,
            )
            return await self._table_driver.get_col_stats_and_sample_rows(
                table_id,
                project_id=project_id,
            )

    async def list_v2_tables_in_connector(
        self,
        project_id: str,
        connector_id: Optional[ConnectorID] = None,
        fetch_column_stats: bool = True,
    ) -> List[TableResourceResponse]:
        # NOTE (jana): potentially not currently used in the UI and can be removed
        with wrap_validation_exceptions(
                not_exists_error=ConnectorDoesNotExistError):
            return await self._table_driver.list_tables(
                project_id=project_id,
                connector_id=connector_id,
                fetch_column_stats=fetch_column_stats,
            )

    async def refresh_table_v2(
        self,
        table_id: TableIDv2,
        project_id: str,
    ) -> EmptyResponse:
        """
        Refresh and re-ingest (if not up to date)
        a V2 table.
        This usually refreshes derived information about the table that may go
        out of sync with the data source, such as the list of columns or table
        statistics.
        Args:
            table_id (TableIDv2): The unique ID of the table.
        """
        with wrap_validation_exceptions(
                not_exists_error=TableDoesNotExistError):

            project_id = require_project_id(project_id)
            entry = await self._table_driver.resolve_project_table(
                project_id, str(table_id))
            table_id = TableIDv2.public(
                pquery_id=None,
                table_name=TableName("NA"),
                fpid=entry.id,
            )
            await self._table_driver.refresh_table(table_id,
                                                   project_id=project_id)
        return EmptyResponse()

    async def get_last_ingested_timestamp(
        self,
        table_id: str,
        project_id: str,
    ) -> LastIngestedTimestamp:
        """Retrieve the time at which this table was last ingested.

        Args:
            table_id (TableIDv2): The unique ID of the table. Usually
                `NA.NA-fpid`.

        Returns:
            LastIngestedTimestamp: Timestamp if successfully ingested, else
            a wrapper around None.
        """
        with wrap_validation_exceptions(
                not_exists_error=TableDoesNotExistError):

            entry = await self._table_driver.resolve_project_table(
                project_id, table_id)
            table_id = TableIDv2.public(
                pquery_id=None,
                table_name=TableName("NA"),
                fpid=entry.id,
            )
            ts = await self._table_driver.get_table_last_ingested_timestamp(
                table_id, project_id)
            return LastIngestedTimestamp(ts)

    async def validate_v2_table_name(
        self,
        table_name: str,
        project_id: str,
    ) -> ValidateNameResponse:
        """
        Validate that the `table_name` doesn't already exist in the DB.
        (Only check V2 tables.)

        Args:
            table_name (str): The table name the user would like to use when
            creating a new table.
        Response:
            ValidateNameResponse (is_valid: bool, error_msg: str)
        """
        return await self._table_driver.validate_project_table_name(
            project_id=project_id,
            name=table_name,
        )

    async def table_to_sdk(
        self,
        table_id: TableIDv2,
        project_id: str,
    ) -> str:
        """
        Used to generate a fully executable Python SDK script. The script contains:
            1. Complete table config structured as Python dictionary.
            2. Connector setup.
            3. Load the desired table in the sdk via `kumo.Table.from_source_table()`
            using the metadata provided in the config.
        This method is part of the UI to SDK feature and will be called in the UI.

        Args:
            table_id (TableIDv2): The table ID for the table that we will be
            generating the config and script.
        Response:
            str: The complete python script as a string.

        A table ID that is not present in the project returns HTTP 404.
        """
        with wrap_driver_errors(), wrap_validation_exceptions(
                not_exists_error=TableDoesNotExistError):
            entry = await self._table_driver.resolve_project_table(
                project_id, table_id)
            canonical_id = TableIDv2.public(
                pquery_id=None,
                table_name=TableName("NA"),
                fpid=entry.id,
            )
            script = await self._table_driver.table_to_sdk(
                canonical_id, project_id)
        if not script:
            raise ValueError("Could not generate SDK script for table " +
                             f"with ID {table_id}.")
        return script

    async def get_vals(
        self,
        table_id: str,
        project_id: str,
        data_version: Optional[str] = None,
    ) -> Optional[TableDataTaskResult]:
        with wrap_driver_errors(), wrap_validation_exceptions(
                not_exists_error=TableDoesNotExistError):

            entry = await self._table_driver.resolve_project_table(
                project_id, table_id)
            table_id = TableIDv2.public(
                pquery_id=None,
                table_name=TableName("NA"),
                fpid=entry.id,
            )
            return await self._table_driver.get_table_validation(
                table_id, project_id, data_version)
