# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from typing import Callable, List

import kumoapi.data_source as api
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from kumo.connector.connector import ConnectorError
from rest.config.status import EmptyResponse, RestStatusCode
from rest.deps.authentication import get_user_authorization_from_request
from rest.driver.connector_api_driver import ConnectorApiDriver
from rest.utils.public import wrap_driver_errors
from rest.utils.routers import wrap_validation_exceptions
from statedb.config.common import ConnectorID, ProjectID
from statedb.db.connectors import ConnectorAlreadyExistsError
from statedb.utils.errors import (
    ConnectorDoesNotExistError,
    ConnectorInUseError,
    ConnectorNameConflictError,
    ConnectorValidationError,
    IdentityDoesNotExistError,
    IdentityForbiddenError,
)


class ConnectorRouter:
    def __init__(
        self,
        api_key_auth: Callable,
        driver: ConnectorApiDriver,
    ) -> None:
        self._driver = driver
        self.router = APIRouter(
            prefix='/projects/{project_id}/connectors',
            tags=['API v1: Project Connectors'],
            dependencies=[Depends(api_key_auth)],
        )
        self.router.add_api_route('', self.create_connector,
                                  response_model=api.ConnectorResponse,
                                  methods=['POST'])
        self.router.add_api_route('', self.list_connectors,
                                  response_model=List[api.ConnectorResponse],
                                  methods=['GET'])
        self.router.add_api_route('/{connector_id}', self.get_connector,
                                  response_model=api.ConnectorResponse,
                                  methods=['GET'])
        self.router.add_api_route('/{connector_id}', self.update_connector,
                                  response_model=api.ConnectorResponse,
                                  methods=['PATCH'])
        self.router.add_api_route('/{connector_id}', self.delete_connector,
                                  response_model=api.ConnectorResponse,
                                  methods=['DELETE'])
        self.router.add_api_route('/{connector_id}/validate',
                                  self.validate_connector,
                                  response_model=api.ConnectorResponse,
                                  methods=['POST'])
        self.router.add_api_route('/uploads/start', self.start_file_upload,
                                  response_model=api.StartFileUploadResponse,
                                  methods=['POST'])
        self.router.add_api_route(
            '/uploads/connector', self.get_upload_connector,
            response_model=api.ProjectUploadConnectorResponse, methods=['GET'])
        self.router.add_api_route('/uploads/part', self.upload_file_part,
                                  methods=['PUT'], include_in_schema=False)
        self.router.add_api_route('/uploads/complete',
                                  self.complete_file_upload, methods=['POST'])
        self.router.add_api_route('/uploads/delete', self.delete_uploaded_file,
                                  methods=['POST'])

    @staticmethod
    def _principal(request: Request):
        return get_user_authorization_from_request(request)

    async def create_connector(
        self,
        project_id: ProjectID,
        connector_resource: api.CreateConnectorArgs,
        request: Request,
    ) -> api.ConnectorResponse:
        with wrap_driver_errors({
                IdentityForbiddenError: HTTPStatus.FORBIDDEN,
                IdentityDoesNotExistError: HTTPStatus.NOT_FOUND,
                ConnectorAlreadyExistsError: HTTPStatus.CONFLICT,
                ConnectorNameConflictError: HTTPStatus.CONFLICT,
                ConnectorValidationError: HTTPStatus.UNPROCESSABLE_ENTITY,
                ValueError: HTTPStatus.BAD_REQUEST,
                ConnectorError: HTTPStatus.UNPROCESSABLE_ENTITY,
        }):
            return await self._driver.create_project_connector(
                project_id, connector_resource, self._principal(request))

    async def list_connectors(
        self,
        project_id: ProjectID,
        request: Request,
    ) -> List[api.ConnectorResponse]:
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
        connector_resource: api.UpdateConnectorMetadataArgs,
        request: Request,
    ) -> api.ConnectorResponse:
        with wrap_driver_errors({
                ConnectorDoesNotExistError: HTTPStatus.NOT_FOUND,
                ValueError: HTTPStatus.CONFLICT,
        }):
            return await self._driver.update_project_connector_metadata(
                project_id, connector_id, connector_resource,
                self._principal(request))

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
                IdentityForbiddenError: HTTPStatus.FORBIDDEN,
                ConnectorError: HTTPStatus.UNPROCESSABLE_ENTITY,
                ValueError: HTTPStatus.UNPROCESSABLE_ENTITY,
        }):
            await self._driver.validate_project_connector(
                project_id, connector_id, principal)
            return await self._driver.get_project_connector(
                project_id, connector_id, principal)

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
            part_number: self._public_upload_part_url(project_id, url)
            for part_number, url in result.presigned_part_urls.items()
        }
        return result

    async def get_upload_connector(
        self,
        project_id: ProjectID,
        request: Request,
    ) -> api.ProjectUploadConnectorResponse:
        connector_id = await self._project_upload_connector(project_id, request)
        return api.ProjectUploadConnectorResponse(
            connector_id=str(connector_id))

    @staticmethod
    def _public_upload_part_url(project_id: ProjectID, url: str) -> str:
        public_prefix = f'/v1/projects/{project_id}/connectors/uploads/part'
        if url.startswith('?'):
            return public_prefix + url
        return url

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
            etag = await self._driver.upload_file_part(
                project_id, self._principal(request), temp_upload_path,
                upload_id, part_number, body)
        return JSONResponse(content={'etag': etag}, headers={'ETag': etag})

    async def complete_file_upload(
        self,
        project_id: ProjectID,
        upload_request: api.CompleteFileUploadRequest,
        request: Request,
    ) -> EmptyResponse:
        connector_id = await self._project_upload_connector(project_id, request)
        with wrap_validation_exceptions():
            await self._driver.complete_file_upload(
                project_id, connector_id, upload_request)
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
