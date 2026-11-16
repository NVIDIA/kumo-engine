# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

from kumoapi.source_table import (
    SourceTableConfigResponse,
    SourceTableDataResponse,
    SourceTableListResponse,
)

from kumoai_studio.client.endpoints import SourceTableEndpoints
from kumoai_studio.client.transport import Transport
from kumoai_studio.client.utils import parse_response, raise_on_error


class SourceTableAPI:
    r"""Inspect source tables through project-owned connectors.

    These low-level methods accept an immutable connector ID and always
    authorize it in one project. Most users should start with a typed connector
    such as :class:`~kumoai_studio.connector.S3Connector` and call ``table()``, but
    this API is useful when an application stores connector IDs directly.

    For object stores, ``list_tables()`` is shallow. Pass an arbitrary-depth
    connector-relative coordinate to ``get_table_config()``, ``get_columns()``,
    or ``get_sample()`` to inspect an exact file or dataset directory.
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

    def list_tables(
        self,
        connector_id: str,
        project_id: Optional[str] = None,
    ) -> SourceTableListResponse:
        r"""List tables visible through a project-owned connector.

        Args:
            connector_id: Immutable connector ID.
            project_id: Exact project, or ``None`` for the active project.
        """
        response = self._client._request(
            SourceTableEndpoints.list_tables(self._project_id(project_id),
                                             connector_id))
        raise_on_error(response)
        return parse_response(SourceTableListResponse, response)

    def get_table_config(
        self,
        connector_id: str,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> SourceTableConfigResponse:
        r"""Return the persisted source configuration for one table.

        Args:
            connector_id: Immutable connector ID.
            table_name: Provider-native table coordinate relative to the
                connector root. Object-store coordinates may contain any
                number of path components.
            project_id: Exact project, or ``None`` for the active project.
        """
        response = self._client._request(
            SourceTableEndpoints.get_table_config(self._project_id(project_id),
                                                  connector_id),
            params={'table_name': table_name},
        )
        raise_on_error(response)
        return parse_response(SourceTableConfigResponse, response)

    def get_columns(
        self,
        connector_id: str,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> SourceTableDataResponse:
        r"""Return column metadata for one source table.

        Args:
            connector_id: Immutable connector ID.
            table_name: Provider-native table coordinate relative to the
                connector root. Object-store coordinates may contain any
                number of path components.
            project_id: Exact project, or ``None`` for the active project.
        """
        response = self._client._request(
            SourceTableEndpoints.get_columns(self._project_id(project_id),
                                             connector_id),
            params={'table_name': table_name},
        )
        raise_on_error(response)
        return parse_response(SourceTableDataResponse, response)

    def get_sample(
        self,
        connector_id: str,
        table_name: str,
        sample_rows: int,
        project_id: Optional[str] = None,
    ) -> SourceTableDataResponse:
        r"""Return column metadata and sample rows for one source table.

        Args:
            connector_id: Immutable connector ID.
            table_name: Provider-native table coordinate relative to the
                connector root. Object-store coordinates may contain any
                number of path components.
            sample_rows: Maximum number of rows to return.
            project_id: Exact project, or ``None`` for the active project.
        """
        response = self._client._request(
            SourceTableEndpoints.get_sample(self._project_id(project_id),
                                            connector_id),
            params={
                'table_name': table_name,
                'sample_rows': sample_rows
            },
        )
        raise_on_error(response)
        return parse_response(SourceTableDataResponse, response)

    def _get_download_urls(
        self,
        connector_id: str,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> List[str]:
        r"""Return server-authorized download references for Parquet data."""
        response = self._client._request(
            SourceTableEndpoints._get_download_urls(
                self._project_id(project_id), connector_id),
            params={'table_name': table_name},
        )
        raise_on_error(response)
        return parse_response(List[str], response)
