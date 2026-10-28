# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from typing import Any, Dict, Optional, get_args

from kumoapi.common import ValidationResponse
from kumoapi.data_snapshot import GraphSnapshotID, GraphSnapshotResource
from kumoapi.graph import (
    EdgeHealthResponse,
    GraphDefinition,
    GraphResource,
    GraphValidationRequest,
)
from kumoapi.json_serde import to_json_dict
from kumoapi.typing import TimestampPrecision

from kumoai_studio.client.endpoints import GraphEndpoints
from kumoai_studio.client.transport import Transport
from kumoai_studio.client.utils import (
    parse_id_response,
    parse_response,
    raise_on_error,
)

GraphID = str


class GraphAPI:
    r"""Typed API definition for Kumo graph definition."""
    def __init__(self, client: Transport) -> None:
        self._client = client

    def create_graph(
        self,
        graph_def: GraphDefinition,
        *,
        name_alias: Optional[str] = None,
        force_rename: bool = False,
        project_id: Optional[str] = None,
        skip_validation: bool = False,
    ) -> GraphID:
        r"""Creates a Graph (metadata definition) resource object in Kumo."""
        params: Dict[str, Any] = {
            'force_rename': force_rename,
            'skip_validation': skip_validation,
        }
        if name_alias:
            params['name_alias'] = name_alias
        if project_id is not None:
            params['project_id'] = project_id
        resp = self._client._request(
            GraphEndpoints.create,
            params=params,
            json=to_json_dict(graph_def),
        )
        raise_on_error(resp)
        return parse_id_response(resp)

    def get_graph_if_exists(
        self,
        graph_id_or_name: str,
        project_id: Optional[str] = None,
    ) -> Optional[GraphResource]:
        params = ({
            'project_id': project_id
        } if project_id is not None else None)
        resp = self._client._request(
            GraphEndpoints.get.with_id(graph_id_or_name), params=params)
        if resp.status_code == HTTPStatus.NOT_FOUND:
            return None

        raise_on_error(resp)
        return parse_response(GraphResource, resp)

    def create_snapshot(
        self,
        graph_id: GraphID,
        *,
        refresh_source: bool = False,
        project_id: Optional[str] = None,
        timestamp_precision: TimestampPrecision = 'seconds',
    ) -> GraphSnapshotID:
        if timestamp_precision not in get_args(TimestampPrecision):
            raise ValueError(f"timestamp_precision must be one of "
                             f"{get_args(TimestampPrecision)}, got "
                             f"{timestamp_precision!r}")
        params: Dict[str, Any] = {
            'graph_id': graph_id,
            'refresh_source': refresh_source
        }
        if project_id is not None:
            params['project_id'] = project_id
        if timestamp_precision != 'seconds':
            params['timestamp_precision'] = timestamp_precision
        resp = self._client._request(GraphEndpoints.create_snapshot,
                                     params=params)
        raise_on_error(resp)
        # Servers that predate index clocks send no header and build seconds.
        built_precision = resp.headers.get('X-Kumo-Timestamp-Precision',
                                           'seconds')
        if built_precision != timestamp_precision:
            raise RuntimeError(
                f"Requested a {timestamp_precision} snapshot but the server "
                f"built a {built_precision} one; upgrade the server to use "
                f"this clock.")
        return GraphSnapshotID(parse_id_response(resp))

    def get_snapshot(
        self,
        snapshot_id: GraphSnapshotID,
        project_id: Optional[str] = None,
    ) -> GraphSnapshotResource:
        params = ({
            'project_id': project_id
        } if project_id is not None else None)
        resp = self._client._request(
            GraphEndpoints.get_snapshot.with_id(snapshot_id), params=params)
        raise_on_error(resp)
        return parse_response(GraphSnapshotResource, resp)

    def get_snapshot_if_exists(
        self,
        snapshot_id: GraphSnapshotID,
        project_id: Optional[str] = None,
    ) -> Optional[GraphSnapshotResource]:
        params = ({
            'project_id': project_id
        } if project_id is not None else None)
        resp = self._client._request(
            GraphEndpoints.get_snapshot.with_id(snapshot_id), params=params)
        if resp.status_code == HTTPStatus.NOT_FOUND:
            return None

        raise_on_error(resp)
        return parse_response(GraphSnapshotResource, resp)

    def get_edge_stats(
        self,
        graph_snapshot_id: GraphSnapshotID,
        project_id: Optional[str] = None,
    ) -> EdgeHealthResponse:
        r"""Fetches edge statistics given a snapshot id"""
        params = ({
            'project_id': project_id
        } if project_id is not None else None)
        resp = self._client._request(
            GraphEndpoints.get_edge_stats.with_id(graph_snapshot_id),
            params=params)
        raise_on_error(resp)
        return parse_response(EdgeHealthResponse, resp)

    def validate_graph(
        self,
        request: GraphValidationRequest,
        project_id: Optional[str] = None,
    ) -> ValidationResponse:
        params = ({
            'project_id': project_id
        } if project_id is not None else None)
        response = self._client._request(GraphEndpoints.validate,
                                         params=params,
                                         json=to_json_dict(request))
        raise_on_error(response)
        return parse_response(ValidationResponse, response)

    def infer_links(
        self,
        graph: GraphDefinition,
        project_id: Optional[str] = None,
    ) -> GraphDefinition:
        params = ({
            'project_id': project_id
        } if project_id is not None else None)
        resp = self._client._request(GraphEndpoints.infer_links, params=params,
                                     json=to_json_dict(graph))
        raise_on_error(resp)
        return parse_response(GraphDefinition, resp)
