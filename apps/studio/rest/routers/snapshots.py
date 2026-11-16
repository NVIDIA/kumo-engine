# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from datetime import datetime
from http import HTTPStatus
from typing import Any, Optional

import kumoapi.table as table_interfaces
from fastapi import APIRouter, Depends
from kumo_core.types import ColStatType
from kumoapi.data_snapshot import GraphSnapshotID, TableSnapshotID
from kumoapi.typing import Stype

from kumo.stats.stats import Stats
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.graph import EdgeHealthStats
from rest.config.graph_v2 import (
    GraphSnapshotRequest,
    GraphSnapshotResponse,
    GraphSnapshotStatsResponse,
)
from rest.config.table import ColStatsResponse, ColumnStats, ColumnSummary
from rest.config.table_v2 import TableSnapshotResponse
from rest.deps.authentication import Authentication, RequestContext, get_ctx
from rest.driver.data_snapshot_api_driver import DataSnapshotApiDriver
from rest.utils.project_scope import require_project_id
from rest.utils.routers import nullable_response, wrap_validation_exceptions
from statedb.config.common import ProjectID
from statedb.utils.errors import (
    GraphSnapshotDoesNotExistError,
    TableSnapshotDoesNotExistError,
)

log = logging.getLogger(__name__)

_TIMESTAMP_VALUE_STATS = {ColStatType.MIN, ColStatType.MAX}


def _coerce_snapshot_datetime(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        return value


def _snapshot_stat_from_payload(payload: Any, stat_type: ColStatType,
                                stype: Stype) -> Stats:
    if isinstance(payload, dict) and "value" in payload and "approx" in payload:
        stat = Stats.from_config(payload)
    elif hasattr(payload, "value") and hasattr(payload, "approx"):
        stat = Stats.from_config(payload)
    else:
        stat = Stats(payload, approx=False)

    if stype == Stype.timestamp and stat_type in _TIMESTAMP_VALUE_STATS:
        stat.value = _coerce_snapshot_datetime(stat.value)
    return stat


class SnapshotResource:
    def __init__(
        self,
        auth: Authentication,
        driver: DataSnapshotApiDriver,
        ml_job_backend: MLJobBackend,
    ) -> None:
        self._driver = driver
        self._auth = auth
        self._ml_job_backend = ml_job_backend

        self.router = APIRouter(
            prefix="/v0/snapshots",
            tags=["Snapshots"],
            dependencies=[Depends(auth.authenticate_request_header)],
            responses={HTTPStatus.NOT_FOUND: {
                "description": "Not found"
            }},
        )

        self.router.add_api_route(
            "/graph",
            self.get_graph_snapshot,
            response_model=GraphSnapshotResponse,
            methods=['POST'],
            summary='Get a snapshot of a graph',
            response_description='Snapshot of a graph',
        )

        self.router.add_api_route(
            "/graph/{id}/edge_stats",
            self.get_edge_stats,
            response_model=EdgeHealthStats,
            methods=['GET'],
            summary='Get edge health statistics for a graph snapshot',
            response_description='Edge health statistics for a graph snapshot',
        )

        self.router.add_api_route(
            "/graph/{id}/stats",
            self.get_graph_snapshot_stats,
            response_model=GraphSnapshotStatsResponse,
            methods=['GET'],
            summary='Get statistics (rows and size) for a graph snapshot',
            response_description=(
                'Statistics including total rows and size in bytes'),
            responses=nullable_response(),
        )

        self.router.add_api_route(
            "/table/{id}",
            self.get_table_snapshot,
            response_model=TableSnapshotResponse,
            methods=['GET'],
            summary='Get a snapshot of a table',
            response_description='Snapshot of a table',
        )

        self.router.add_api_route(
            "/table/{id}/col_stats",
            self.get_col_stats,
            response_model=ColStatsResponse,
            methods=['GET'],
            summary='Get column statistics information for a table snapshot',
            response_description='Table statistics for a table snapshot',
        )

    async def get_graph_snapshot(
            self,
            request: GraphSnapshotRequest,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> GraphSnapshotResponse:
        project_id = ProjectID(require_project_id(project_id))
        with wrap_validation_exceptions(GraphSnapshotDoesNotExistError):
            await self._driver.require_project_access(project_id,
                                                      ctx.user_authorization)
            snapshot_resource = await self._driver.get_graph_snapshot(
                request, project_id)
            return snapshot_resource

    async def get_graph_snapshot_stats(
        self,
        id: GraphSnapshotID,
        project_id: str,
        ctx: RequestContext = Depends(get_ctx),
    ) -> Optional[GraphSnapshotStatsResponse]:
        project_id = ProjectID(require_project_id(project_id))
        with wrap_validation_exceptions(GraphSnapshotDoesNotExistError):
            await self._driver.require_project_access(project_id,
                                                      ctx.user_authorization)
            return await self._driver._get_graph_snapshot_stats(id, project_id)

    async def get_edge_stats(
            self,
            id: GraphSnapshotID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> EdgeHealthStats:
        project_id = ProjectID(require_project_id(project_id))
        with wrap_validation_exceptions(GraphSnapshotDoesNotExistError):
            await self._driver.require_project_access(project_id,
                                                      ctx.user_authorization)
            return await self._driver.get_ui_edge_health(id, project_id)

    async def get_table_snapshot(
            self,
            id: TableSnapshotID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> TableSnapshotResponse:
        project_id = ProjectID(require_project_id(project_id))
        with wrap_validation_exceptions(TableSnapshotDoesNotExistError):
            await self._driver.require_project_access(project_id,
                                                      ctx.user_authorization)
            snapshot = await self._driver.get_table_snapshot(id, project_id)
            table_definition = snapshot.table_definition
            if table_definition is None:
                table_definition = await self._driver.get_table_definition(
                    id, project_id)

            return TableSnapshotResponse(snapshot, table_definition)

    async def get_col_stats(
            self,
            id: TableSnapshotID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> ColStatsResponse:
        with wrap_validation_exceptions(TableSnapshotDoesNotExistError):
            table = await self.get_table_snapshot(id, project_id, ctx)
            raw_col_stats = table.snapshot.column_stats

            def get_col_stats(
                    col: table_interfaces.Column) -> ColumnStats | None:
                if raw_col_stats is None:
                    return None
                matching_stats = [
                    stats for stats in raw_col_stats or []
                    if stats.column_name == col.name
                ]
                if len(matching_stats) > 0:
                    stype = Stype(col.stype)
                    stats_dict = {}
                    for key, value in matching_stats[0].stats.items():
                        stat_type = ColStatType(key)
                        stats_dict[stat_type] = _snapshot_stat_from_payload(
                            value, stat_type, stype)
                    try:
                        colstats = ColumnStats.from_stats_dict(
                            stype=stype,
                            stats_dict=stats_dict,
                        )
                    except Exception as e:
                        # If we cannot get the `ColumnStats` object, for
                        # whatever reason, do not break and simply return
                        # `None` (rendered as "loading..." in the UI).
                        log.exception(e)
                        return None
                    return colstats
                return None

            available = raw_col_stats is not None
            columns = []
            approx = False
            rows_used = -1
            for col in table.definition.cols:
                statistics = get_col_stats(col)
                if statistics is not None:
                    approx = approx or statistics.approx
                    if rows_used < 0:
                        rows_used = statistics.counts.total
                columns.append(
                    ColumnSummary(
                        name=col.name,
                        dtype=col.dtype,
                        stype=col.stype,
                        sample_rows=[],
                        statistics=statistics,
                    ))

            return ColStatsResponse(
                statistics_available=available,
                approx=approx,
                rows_used=rows_used,
                columns=columns,
            )
