# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from typing import List, Optional

from kumoapi.data_source import (
    CompleteFileUploadRequest,
    ConnectorResponse,
    ConnectorType,
    CreateConnectorArgs,
    DeleteUploadedFileRequest,
    ProjectUploadConnectorResponse,
    StartFileUploadRequest,
    StartFileUploadResponse,
)
from kumoapi.json_serde import to_json_dict

from kumoai_studio.client.endpoints import ConnectorEndpoints, HTTPMethod
from kumoai_studio.client.transport import Transport
from kumoai_studio.client.utils import parse_response, raise_on_error


class ConnectorAPI:
    r"""Typed API for project-owned connectors and managed uploads.

    Methods use the active SDK project unless an exact ``project_id`` is
    supplied by an advanced caller. External connector responses contain safe
    configuration and an immutable identity ID, never identity credentials.
    """
    def __init__(self, client: Transport) -> None:
        self._client = client

    @staticmethod
    def _project_id(project_id: Optional[str]) -> str:
        if project_id is not None:
            return project_id
        from kumoai_studio import _require_project, global_state
        _require_project()
        assert global_state._project_id is not None
        return global_state._project_id

    def start_file_upload(
        self,
        req: StartFileUploadRequest,
        project_id: Optional[str] = None,
    ) -> StartFileUploadResponse:
        r"""Start a managed upload in one project.

        Args:
            req: File metadata used to allocate upload parts.
            project_id: Exact target project, or ``None`` for the active
                project.

        Returns:
            Upload ID and presigned or project-relative part destinations.
        """
        project_id = self._project_id(project_id)
        response = self._client._request(
            ConnectorEndpoints.upload(project_id, 'start', HTTPMethod.POST),
            json=to_json_dict(req),
        )
        raise_on_error(response)
        return parse_response(StartFileUploadResponse, response)

    def get_upload_connector(
        self,
        project_id: Optional[str] = None,
    ) -> ProjectUploadConnectorResponse:
        r"""Return the hidden managed-upload connector for one project.

        Args:
            project_id: Exact project, or ``None`` for the active project.

        Returns:
            The project-specific upload connector reference. It is not backed
            by an external identity and cannot be used by another project.
        """
        project_id = self._project_id(project_id)
        response = self._client._request(
            ConnectorEndpoints.upload(project_id, 'connector', HTTPMethod.GET))
        raise_on_error(response)
        return parse_response(ProjectUploadConnectorResponse, response)

    def delete_file_upload(
        self,
        req: DeleteUploadedFileRequest,
        project_id: Optional[str] = None,
    ) -> None:
        r"""Delete an uploaded table or incomplete upload from one project.

        Args:
            req: Upload deletion request.
            project_id: Exact project, or ``None`` for the active project.
        """
        project_id = self._project_id(project_id)
        response = self._client._request(
            ConnectorEndpoints.upload(project_id, 'delete', HTTPMethod.POST),
            json=to_json_dict(req),
        )
        raise_on_error(response)

    def complete_file_upload(
        self,
        req: CompleteFileUploadRequest,
        project_id: Optional[str] = None,
    ) -> None:
        r"""Finalize an upload after all parts have been written.

        Args:
            req: Completed part metadata and final table name.
            project_id: Exact project, or ``None`` for the active project.
        """
        project_id = self._project_id(project_id)
        response = self._client._request(
            ConnectorEndpoints.upload(project_id, 'complete', HTTPMethod.POST),
            json=to_json_dict(req),
        )
        raise_on_error(response)

    def create(
        self,
        args: CreateConnectorArgs,
        project_id: Optional[str] = None,
    ) -> ConnectorResponse:
        r"""Create an external connector in one project.

        An absent ``args.identity_id`` requests atomic server-side identity
        resolution. The persisted response always contains the selected
        identity ID.

        Args:
            args: Connector display name, immutable target configuration,
                optional identity ID, and description.
            project_id: Exact project, or ``None`` for the active project.

        Returns:
            The persisted connector with its immutable connector and identity
            IDs.
        """
        project_id = self._project_id(project_id)
        response = self._client._request(
            ConnectorEndpoints.collection(project_id, HTTPMethod.POST),
            json=to_json_dict(args),
        )
        raise_on_error(response)
        return parse_response(ConnectorResponse, response)

    def create_if_not_exist(
        self,
        args: CreateConnectorArgs,
        project_id: Optional[str] = None,
    ) -> ConnectorResponse:
        r"""Atomically create or return an identical project connector.

        The server owns name uniqueness, idempotency, identity resolution, and
        conflict handling in one transaction. The SDK deliberately does not
        list before POST, which avoids a race between concurrent creators.
        With an omitted S3 identity, the server resolves the canonical ambient
        identity. A different target or explicit identity returns a conflict.

        Args:
            args: Desired connector definition.
            project_id: Exact project, or ``None`` for the active project.

        Returns:
            The new or existing matching connector.
        """
        return self.create(args, project_id=self._project_id(project_id))

    def get(
        self,
        connector_id: str,
        project_id: Optional[str] = None,
    ) -> Optional[ConnectorResponse]:
        r"""Get a connector by immutable ID within one project.

        Args:
            connector_id: Immutable connector ID.
            project_id: Exact project, or ``None`` for the active project.

        Returns:
            The connector, or ``None`` if it is absent or not visible in that
            project.
        """
        project_id = self._project_id(project_id)
        response = self._client._request(
            ConnectorEndpoints.item(project_id, connector_id, HTTPMethod.GET))
        if response.status_code == HTTPStatus.NOT_FOUND:
            return None
        raise_on_error(response)
        return parse_response(ConnectorResponse, response)

    def get_by_name(
        self,
        name: str,
        project_id: Optional[str] = None,
    ) -> Optional[ConnectorResponse]:
        r"""Get a connector by normalized display name within one project.

        Args:
            name: Project-local connector display name.
            project_id: Exact project, or ``None`` for the active project.

        Returns:
            The matching connector, or ``None``. Other projects are never
            searched.
        """
        from kumoai_studio import _normalize_name
        normalized_name = _normalize_name(name)
        matches = [
            connector for connector in self.list(project_id=project_id)
            if _normalize_name(connector.name) == normalized_name
        ]
        if len(matches) > 1:
            raise RuntimeError(
                f"Server returned duplicate connector name {name!r} in one "
                "project.")
        return matches[0] if matches else None

    def list(
        self,
        connector_type: Optional[ConnectorType] = None,
        project_id: Optional[str] = None,
    ) -> List[ConnectorResponse]:
        r"""List external connectors in one project.

        Args:
            connector_type: Optional provider-type filter.
            project_id: Exact project, or ``None`` for the active project.

        Returns:
            Safe connector definitions visible in the project.
        """
        project_id = self._project_id(project_id)
        response = self._client._request(
            ConnectorEndpoints.collection(project_id, HTTPMethod.GET))
        raise_on_error(response)
        connectors = parse_response(List[ConnectorResponse], response)
        if connector_type is not None:
            connectors = [
                connector for connector in connectors
                if connector.config.type == connector_type
            ]
        return connectors

    def delete_if_exists(
        self,
        connector_id: str,
        project_id: Optional[str] = None,
    ) -> bool:
        r"""Delete a connector by immutable ID if it exists in one project.

        .. warning::

            Active jobs and artifact exports that reference the connector may
            fail after deletion. Provider calls already in progress may still
            finish.

        Args:
            connector_id: Immutable connector ID.
            project_id: Exact project, or ``None`` for the active project.

        Returns:
            ``True`` if the connector was deleted; ``False`` if it was absent.
        """
        project_id = self._project_id(project_id)
        response = self._client._request(
            ConnectorEndpoints.item(project_id, connector_id,
                                    HTTPMethod.DELETE))
        if response.status_code == HTTPStatus.NOT_FOUND:
            return False
        raise_on_error(response)
        return True
