# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus

import kumoapi.data_source as api
import kumoapi.source_table as source_api
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from common_utils.string import normalize_connector_source_name
from kumo.connector.connector import ConnectorError
from rest.config.status import EmptyResponse, RestStatusCode
from rest.deps.authentication import (
    Authentication,
    get_user_authorization_from_request,
)
from rest.driver.connector_api_driver import ConnectorApiDriver
from rest.utils.public import wrap_driver_errors
from rest.utils.routers import wrap_validation_exceptions
from statedb.config.common import ConnectorID, ProjectID
from statedb.utils.errors import (
    ConnectorDoesNotExistError,
    ConnectorInUseError,
    ConnectorNameConflictError,
    ConnectorValidationError,
    IdentityDoesNotExistError,
    IdentityForbiddenError,
)


class ConnectorResource:
    """Project-scoped connector lifecycle and validation.

    A missing connector is a 404 on all four discovery routes
    (``source-tables``, ``source-table-config``, ``columns``, ``sample``),
    matching the sibling ``GET /{connector_id}`` route and the public
    API v1 discovery routes. Columns and sample reads also return 404 when an
    exact physical source is missing or cannot be distinguished from one
    hidden by provider access controls. ``ConnectorDoesNotExistError`` is a
    ``ValueError`` and exceptions unwind inner-first, so the inner
    ``wrap_validation_exceptions(not_exists_error=...)`` must own that
    mapping; a bare inner wrapper would emit a 422 before the outer
    ``wrap_driver_errors`` entry, kept as documentation of intent, could
    fire.
    """
    def __init__(self, auth: Authentication,
                 driver: ConnectorApiDriver) -> None:
        self._driver = driver
        self.router = APIRouter(
            prefix='/v0/projects/{project_id}/connectors',
            tags=['Connectors'],
            dependencies=[Depends(auth.authenticate_request_header)],
        )
        self.router.add_api_route('/', self.create_connector, methods=['POST'],
                                  response_model=api.ConnectorResponse)
        self.router.add_api_route('/', self.list_connectors, methods=['GET'],
                                  response_model=list[api.ConnectorResponse])
        self.router.add_api_route('/{connector_id}', self.get_connector,
                                  methods=['GET'],
                                  response_model=api.ConnectorResponse)
        self.router.add_api_route('/{connector_id}', self.update_connector,
                                  methods=['PATCH'],
                                  response_model=api.ConnectorResponse)
        self.router.add_api_route('/{connector_id}', self.delete_connector,
                                  methods=['DELETE'],
                                  response_model=api.ConnectorResponse)
        self.router.add_api_route('/{connector_id}/validate',
                                  self.validate_connector, methods=['POST'],
                                  response_model=api.ConnectorResponse)
        self.router.add_api_route(
            '/{connector_id}/source-tables', self.list_source_tables,
            methods=['GET'], response_model=source_api.SourceTableListResponse)
        self.router.add_api_route(
            '/{connector_id}/source-table-config', self.get_source_table_config,
            methods=['GET'],
            response_model=source_api.SourceTableConfigResponse)
        self.router.add_api_route(
            '/{connector_id}/columns', self.get_columns, methods=['GET'],
            response_model=source_api.SourceTableDataResponse, responses={
                HTTPStatus.NOT_FOUND: {
                    'description':
                    'The connector or requested physical source was not found.'
                },
            })
        self.router.add_api_route(
            '/{connector_id}/sample', self.get_sample, methods=['GET'],
            response_model=source_api.SourceTableDataResponse, responses={
                HTTPStatus.NOT_FOUND: {
                    'description':
                    'The connector or requested physical source was not found.'
                },
            })
        self.router.add_api_route('/{connector_id}/download-urls',
                                  self.get_download_urls, methods=['GET'],
                                  response_model=list[str],
                                  include_in_schema=False)
        self.router.add_api_route('/uploads/start', self.start_file_upload,
                                  methods=['POST'],
                                  response_model=api.StartFileUploadResponse)
        self.router.add_api_route(
            '/uploads/connector', self.get_upload_connector, methods=['GET'],
            response_model=api.ProjectUploadConnectorResponse)
        self.router.add_api_route('/uploads/part', self.upload_file_part,
                                  methods=['PUT'], include_in_schema=False)
        self.router.add_api_route('/uploads/complete',
                                  self.complete_file_upload, methods=['POST'],
                                  response_model=EmptyResponse)
        self.router.add_api_route('/uploads/delete', self.delete_uploaded_file,
                                  methods=['POST'],
                                  response_model=EmptyResponse)

    @staticmethod
    def _principal(request: Request):
        return get_user_authorization_from_request(request)

    @staticmethod
    def _source_name(table_name: str) -> str:
        """Normalize a rooted-relative or rootless S3 source coordinate."""
        return normalize_connector_source_name(table_name)

    async def create_connector(
        self,
        project_id: ProjectID,
        args: api.CreateConnectorArgs,
        request: Request,
    ) -> api.ConnectorResponse:
        with wrap_driver_errors({
                IdentityForbiddenError: HTTPStatus.FORBIDDEN,
                IdentityDoesNotExistError: HTTPStatus.NOT_FOUND,
                ConnectorNameConflictError: HTTPStatus.CONFLICT,
                ConnectorValidationError: HTTPStatus.UNPROCESSABLE_ENTITY,
                ValueError: HTTPStatus.BAD_REQUEST,
        }):
            principal = self._principal(request)
            response = await self._driver.create_project_connector(
                project_id, args, principal)
            return response

    async def list_connectors(
        self,
        project_id: ProjectID,
        request: Request,
    ) -> list[api.ConnectorResponse]:
        with wrap_driver_errors():
            return await self._driver.list_project_connectors(
                project_id, self._principal(request))

    async def get_connector(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        request: Request,
    ) -> api.ConnectorResponse:
        with wrap_driver_errors({
                ConnectorDoesNotExistError:
                HTTPStatus.NOT_FOUND,
        }):
            return await self._driver.get_project_connector(
                project_id, connector_id, self._principal(request))

    async def update_connector(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        args: api.UpdateConnectorMetadataArgs,
        request: Request,
    ) -> api.ConnectorResponse:
        with wrap_driver_errors({ValueError: HTTPStatus.CONFLICT}):
            return await self._driver.update_project_connector_metadata(
                project_id, connector_id, args, self._principal(request))

    async def delete_connector(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        request: Request,
    ) -> api.ConnectorResponse:
        with wrap_driver_errors({
                ConnectorDoesNotExistError: HTTPStatus.NOT_FOUND,
                ConnectorInUseError: HTTPStatus.CONFLICT,
        }):
            return await self._driver.delete_project_connector(
                project_id, connector_id, self._principal(request))

    async def validate_connector(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        request: Request,
    ) -> api.ConnectorResponse:
        principal = self._principal(request)
        with wrap_driver_errors({
                ConnectorDoesNotExistError: HTTPStatus.NOT_FOUND,
                ValueError: HTTPStatus.UNPROCESSABLE_ENTITY,
        }):
            await self._driver.validate_project_connector(
                project_id, connector_id, principal)
            return await self._driver.get_project_connector(
                project_id, connector_id, principal)

    async def list_source_tables(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        request: Request,
    ) -> source_api.SourceTableListResponse:
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
        return source_api.SourceTableListResponse(table_names=table_names)

    async def get_columns(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        request: Request,
        table_name: str = Query(default=''),
        query: str | None = None,
    ) -> source_api.SourceTableDataResponse:
        """Return columns, or 404 for a missing or unavailable physical source."""
        with wrap_driver_errors({
                ConnectorDoesNotExistError: HTTPStatus.NOT_FOUND,
                ConnectorError: HTTPStatus.UNPROCESSABLE_ENTITY,
        }), wrap_validation_exceptions(
                not_exists_error=ConnectorDoesNotExistError):
            # A Custom SQL ``query`` discovers columns from the query's result
            # schema; it has no physical source table to normalize.
            source_table_name = ('' if query is not None else
                                 self._source_name(table_name))
            response = await self._driver.get_project_source_columns(
                project_id,
                connector_id,
                self._principal(request),
                source_table_name,
                skip_cache=False,
                query=query,
            )
        return source_api.SourceTableDataResponse(
            table_name=source_table_name,
            cols=response.source_cols,
        )

    async def get_source_table_config(
            self,
            project_id: ProjectID,
            connector_id: ConnectorID,
            request: Request,
            table_name: str = Query(),
    ) -> source_api.SourceTableConfigResponse:
        """Resolve one source config; invalid warehouse coordinates return 422."""
        with wrap_driver_errors({
                ConnectorDoesNotExistError: HTTPStatus.NOT_FOUND,
                ConnectorError: HTTPStatus.UNPROCESSABLE_ENTITY,
        }), wrap_validation_exceptions(
                not_exists_error=ConnectorDoesNotExistError):
            source_table_name = self._source_name(table_name)
            return await self._driver.get_project_source_table_config(
                project_id,
                connector_id,
                self._principal(request),
                source_table_name,
            )

    async def get_download_urls(
            self,
            project_id: ProjectID,
            connector_id: ConnectorID,
            request: Request,
            table_name: str = Query(),
    ) -> list[str]:
        with wrap_driver_errors({
                ConnectorDoesNotExistError: HTTPStatus.NOT_FOUND,
                ConnectorError: HTTPStatus.UNPROCESSABLE_ENTITY,
        }), wrap_validation_exceptions():
            return await self._driver.get_project_source_download_urls(
                project_id, connector_id, self._principal(request),
                self._source_name(table_name))

    async def get_sample(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        request: Request,
        table_name: str = Query(default=''),
        sample_rows: int = Query(default=10, ge=1, le=1000),
        query: str | None = None,
    ) -> source_api.SourceTableDataResponse:
        """Return sample rows, or 404 for a missing or unavailable source."""
        with wrap_driver_errors({
                ConnectorDoesNotExistError: HTTPStatus.NOT_FOUND,
                ConnectorError: HTTPStatus.UNPROCESSABLE_ENTITY,
        }), wrap_validation_exceptions(
                not_exists_error=ConnectorDoesNotExistError):
            # A Custom SQL ``query`` returns a bounded preview of the query
            # result; it has no physical source table to normalize.
            source_table_name = ('' if query is not None else
                                 self._source_name(table_name))
            response = await self._driver.get_project_sample_rows(
                project_id,
                connector_id,
                self._principal(request),
                source_table_name,
                num_rows=sample_rows,
                serialize_to_json=True,
                require_stats=False,
                query=query,
            )
        return source_api.SourceTableDataResponse(
            table_name=source_table_name,
            cols=response.columns,
            sample_rows=response.sample_df_serialized,
        )

    async def _project_upload_connector(
        self,
        project_id: ProjectID,
        request: Request,
    ) -> ConnectorID:
        return await self._driver.get_or_create_project_upload_connector(
            project_id, self._principal(request))

    async def start_file_upload(
        self,
        project_id: ProjectID,
        upload_request: api.StartFileUploadRequest,
        request: Request,
    ) -> api.StartFileUploadResponse:
        with wrap_driver_errors({ValueError: HTTPStatus.BAD_REQUEST}):
            connector_id = await self._project_upload_connector(
                project_id, request)
            result = await self._driver.start_file_upload(
                project_id, connector_id, upload_request)
        result.presigned_part_urls = {
            part_number: self._product_upload_part_url(project_id, url)
            for part_number, url in result.presigned_part_urls.items()
        }
        return result

    @staticmethod
    def _product_upload_part_url(project_id: ProjectID, url: str) -> str:
        project_prefix = (f'/v0/projects/{project_id}/connectors/uploads/part')
        if url.startswith('?'):
            return project_prefix + url
        return url

    async def get_upload_connector(
        self,
        project_id: ProjectID,
        request: Request,
    ) -> api.ProjectUploadConnectorResponse:
        connector_id = await self._project_upload_connector(project_id, request)
        return api.ProjectUploadConnectorResponse(
            connector_id=str(connector_id))

    async def upload_file_part(
        self,
        project_id: ProjectID,
        temp_upload_path: str,
        upload_id: str,
        part_number: int,
        request: Request,
    ) -> JSONResponse:
        with wrap_validation_exceptions():
            body = await request.body()
            etag = await self._driver.upload_file_part(project_id,
                                                       self._principal(request),
                                                       temp_upload_path,
                                                       upload_id, part_number,
                                                       body)
        return JSONResponse(content={'etag': etag}, headers={'ETag': etag})

    async def complete_file_upload(
        self,
        project_id: ProjectID,
        upload_request: api.CompleteFileUploadRequest,
        request: Request,
    ) -> EmptyResponse:
        connector_id = await self._project_upload_connector(project_id, request)
        with wrap_validation_exceptions():
            await self._driver.complete_file_upload(project_id, connector_id,
                                                    upload_request)
        return EmptyResponse()

    async def delete_uploaded_file(
        self,
        project_id: ProjectID,
        upload_request: api.DeleteUploadedFileRequest,
        request: Request,
    ) -> EmptyResponse:
        connector_id = await self._project_upload_connector(project_id, request)
        with wrap_validation_exceptions():
            deleted = await self._driver.delete_uploaded_file(
                project_id, connector_id, upload_request)
            if not deleted:
                raise HTTPException(
                    HTTPStatus.NOT_FOUND,
                    RestStatusCode.UPLOADED_FILE_DOES_NOT_EXIST.value)
        return EmptyResponse()
