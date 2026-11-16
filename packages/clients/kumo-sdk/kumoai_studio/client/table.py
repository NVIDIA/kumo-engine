# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from typing import Any, Dict, Optional, Tuple

import requests
from kumoapi.common import ValidationResponse
from kumoapi.data_snapshot import TableSnapshotID, TableSnapshotResource
from kumoapi.json_serde import to_json_dict
from kumoapi.table import (
    SqlTableResolveRequest,
    SqlTableResolveResponse,
    TableDefinition,
    TableMetadataRequest,
    TableMetadataResponse,
    TableResource,
    TableValidationRequest,
)

from kumoai_studio.client.endpoints import TableEndpoints
from kumoai_studio.client.transport import Transport
from kumoai_studio.client.utils import (
    parse_id_response,
    parse_response,
    raise_on_error,
)

TableID = str
_METADATA_TIMEOUT = (10, 330)


class TableAPI:
    r"""Typed API definition for Kumo table definition."""
    def __init__(self, client: Transport) -> None:
        self._client = client

    def create_table(
        self,
        table_def: TableDefinition,
        *,
        name_alias: str,
        force_rename: bool = False,
        project_id: Optional[str] = None,
    ) -> TableID:
        r"""Creates a named Table (metadata definition) resource in Kumo."""
        params: Dict[str, Any] = {
            'force_rename': force_rename,
            'name_alias': name_alias,
        }
        if project_id is not None:
            params['project_id'] = project_id
        resp = self._client._request(
            TableEndpoints.create,
            params=params,
            json=to_json_dict(table_def),
        )
        raise_on_error(resp)
        return parse_id_response(resp)

    def get_table_if_exists(
        self,
        table_id_or_name: str,
        project_id: Optional[str] = None,
    ) -> Optional[TableResource]:
        r"""Fetches a connector given its ID."""
        params = ({
            'project_id': project_id
        } if project_id is not None else None)
        resp = self._client._request(
            TableEndpoints.get.with_id(table_id_or_name), params=params)
        if resp.status_code == HTTPStatus.NOT_FOUND:
            return None

        raise_on_error(resp)
        return parse_response(TableResource, resp)

    def create_snapshot(
        self,
        table_definition: TableDefinition,
        *,
        refresh_source: bool = False,
        project_id: Optional[str] = None,
    ) -> TableSnapshotID:
        params: Dict[str, Any] = {'refresh_source': refresh_source}
        if project_id is not None:
            params['project_id'] = project_id
        resp = self._client._request(TableEndpoints.create_snapshot,
                                     params=params,
                                     json=to_json_dict(table_definition))
        raise_on_error(resp)
        return TableSnapshotID(parse_id_response(resp))

    def get_snapshot(
        self,
        snapshot_id: TableSnapshotID,
        project_id: Optional[str] = None,
    ) -> TableSnapshotResource:
        r"""Fetch a table snapshot and its currently available statistics.

        Snapshot-level statistics become available after the ``INGEST`` stage
        succeeds. Until then, or when ingestion fails, ``resource.stats`` is
        :obj:`None`. Once populated, ``resource.stats.num_rows`` is the table
        row count and ``resource.stats.size_bytes`` is its size in bytes.

        Args:
            snapshot_id: The table snapshot identifier returned by
                :meth:`create_snapshot`.
            project_id: The immutable ID of the project that owns the table.
                Uses the server's default project context when omitted.

        Returns:
            The typed snapshot resource, including stage state, optional
            snapshot-level statistics, and any computed column statistics.
        """
        params = ({
            'project_id': project_id
        } if project_id is not None else None)
        resp = self._client._request(
            TableEndpoints.get_snapshot.with_id(snapshot_id), params=params)
        raise_on_error(resp)
        return parse_response(TableSnapshotResource, resp)

    def infer_metadata(
        self,
        request: TableMetadataRequest,
        project_id: Optional[str] = None,
    ) -> TableMetadataResponse:
        params = ({
            'project_id': project_id
        } if project_id is not None else None)
        response = self._client._request(TableEndpoints.infer_metadata,
                                         params=params,
                                         json=to_json_dict(request))
        raise_on_error(response)
        return parse_response(TableMetadataResponse, response)

    def validate_table(
        self,
        request: TableValidationRequest,
        project_id: Optional[str] = None,
    ) -> ValidationResponse:
        params = ({
            'project_id': project_id
        } if project_id is not None else None)
        response = self._client._request(TableEndpoints.validate,
                                         params=params,
                                         json=to_json_dict(request))
        raise_on_error(response)
        return parse_response(ValidationResponse, response)

    def resolve_sql(
        self,
        request: SqlTableResolveRequest,
        project_id: Optional[str] = None,
        timeout: Optional[Tuple[float, float]] = None,
    ) -> SqlTableResolveResponse:
        params = ({
            'project_id': project_id
        } if project_id is not None else None)
        try:
            response = self._client._request(
                TableEndpoints.resolve_sql,
                params=params,
                json=to_json_dict(request),
                timeout=timeout or _METADATA_TIMEOUT,
            )
        except requests.Timeout:
            raise TimeoutError(
                "Warehouse SQL validation timed out; completion could not be "
                "confirmed.") from None
        raise_on_error(response)
        return parse_response(SqlTableResolveResponse, response)
