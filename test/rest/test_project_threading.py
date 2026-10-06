# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for project_id threading through StateServices.

Run with:
    pytest test/rest/test_project_threading.py -v
"""
from __future__ import annotations

from test.statedb.db.connector_fixtures import seed_connector
from typing import List, Optional

import pytest
from kumo_core.types.schema import FPID
from kumoapi.source_table import FileType, S3SourceTable
from kumoapi.table import Column
from kumoapi.typing import Dtype, Stype

from rest.config.connector import FileConnectorResourceConfig
from rest.config.graph_v2 import GraphDBEntryV2
from rest.config.table_v2 import TableDBEntryV2, TableDefinition
from rest.deps.authentication import Permissions, UserAuthorization
from rest.deps.context import _current_user_ctx
from rest.driver.graph_driver import GraphDriver
from rest.driver.table_driver import TableDriver
from statedb.config.common import (
    ConnectorID,
    GraphIDv2,
    GroupID,
    TableIDv2,
    TableName,
)
from statedb.config.timestamp import get_current_timestamps
from statedb.db import StateDB, StateServices
from statedb.rest.connector_db import InternalConnectorDBEntry
from statedb.rest.group import KumoGroupCreateSpec
from statedb.utils.errors import ProjectNotFoundError

pytestmark = pytest.mark.asyncio

_GRAPH_FPID1 = FPID("c" * 32)


def _make_user(
    user_id: str,
    admin: bool = False,
    group_ids: Optional[List[str]] = None,
) -> UserAuthorization:
    perms = [
        Permissions.READ.compose(),
        Permissions.WRITE.compose(),
    ]
    if admin:
        perms.append(Permissions.ADMIN.compose())
    return UserAuthorization(
        user_id=user_id,
        access_token="tok",
        permissions=perms,
        kumo_group_ids=tuple(group_ids or []),
    )


async def _state_services(tmp_path) -> StateServices:
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    return StateServices(state_db)


async def _create_group(services: StateServices,
                        admin: UserAuthorization) -> GroupID:
    suffix = str(GroupID.generate())
    created = await services.groups.create_kumo_groups(
        [KumoGroupCreateSpec(display_name=f"Group {suffix}", )], admin)
    return created[0].id


async def _create_project(
    services: StateServices,
    user_auth: UserAuthorization,
    name: str = "Test Project",
    kumo_group_id: GroupID | None = None,
    is_private: bool = False,
):
    group_id = kumo_group_id or await _create_group(services, user_auth)
    project = await services.projects.create_project(
        name=name,
        group_id=str(group_id),
        is_private=is_private,
        user_authorization=user_auth,
    )
    connector_id = ConnectorID.generate()
    connector_name = f'test-{str(connector_id)[-8:]}'
    await seed_connector(
        services.state_db,
        InternalConnectorDBEntry(
            id=connector_id,
            project_id=project.id,
            name=connector_name,
            normalized_name=connector_name,
            config=FileConnectorResourceConfig(
                name=connector_name,
                root_dir=f'/tmp/{project.id}/{connector_name}',
            ),
            timestamps=get_current_timestamps(),
        ),
    )
    table = TableDefinition(
        cols=[Column(name='id', stype=Stype.ID, dtype=Dtype.string)],
        source_table=S3SourceTable(
            connector_id=connector_id,
            source={
                'mode': 'relative',
                'path': 'transactions'
            },
            file_type=FileType.PARQUET,
        ),
        pkey='id',
    )
    await services.tables.upsert_table_entry(
        TableDBEntryV2(
            id=table.fpid,
            project_id=project.id,
            alias=None,
            config=table,
            col_dtypes={'id': Dtype.string},
            timestamps=get_current_timestamps(),
        ))
    await services.graphs.upsert_graph_entry(
        GraphDBEntryV2(
            id=_GRAPH_FPID1,
            project_id=project.id,
            table_name_to_dbid={},
            col_groups=[],
            timestamps=get_current_timestamps(),
        ))
    return project


async def _table_by_source(
    services: StateServices,
    project_id,
    source_name: str,
) -> TableDBEntryV2:
    for entry in await services.tables.list_tables(project_id):
        if str(entry.config.get_source_table_name()) == source_name:
            return entry
    raise AssertionError(f"Missing table source {source_name}")


class TestCanonicalProjectOwnership:
    async def test_graph_rejects_cross_project_table(self, tmp_path):
        services = await _state_services(tmp_path)
        admin = _make_user("alice", admin=True)
        first = await _create_project(services, admin, name="First")
        second = await _create_project(services, admin, name="Second")
        foreign = await _table_by_source(services, second.id, "transactions")
        foreign_fpid = foreign.id

        graph = GraphDBEntryV2(
            id=FPID("1" * 32),
            project_id=first.id,
            table_name_to_dbid={"foreign": foreign_fpid},
            col_groups=[],
            timestamps=get_current_timestamps(),
        )
        with pytest.raises(ValueError, match="belong to its project"):
            await services.graphs.upsert_graph_entry(graph)

    async def test_inaccessible_project_denies_id_alias_reads_and_mutations(
            self, tmp_path):
        from unittest.mock import MagicMock

        services = await _state_services(tmp_path)
        admin = _make_user("alice", admin=True)
        outsider = _make_user("mallory")
        project = await _create_project(services, admin, name="Private",
                                        is_private=True)
        table = await _table_by_source(services, project.id, "transactions")
        table.alias = "secret-table"
        await services.tables.upsert_table_entry(table)
        graph = await services.graphs.get_graph_entry(project.id, _GRAPH_FPID1)
        assert graph is not None
        graph.alias = "secret-graph"
        await services.graphs.upsert_graph_entry(graph)

        table_driver = TableDriver(MagicMock(), MagicMock())
        graph_driver = GraphDriver(MagicMock(), MagicMock(), table_driver)
        table_driver._configure_state_services(services)
        graph_driver._configure_state_services(services)
        table_id = TableIDv2.public(
            pquery_id=None,
            table_name=TableName("NA"),
            fpid=table.id,
        )
        graph_id = GraphIDv2.public(graph.id)
        token = _current_user_ctx.set(outsider)
        try:
            denied_calls = (
                table_driver.resolve_project_table(project.id, table_id),
                table_driver.resolve_project_table(project.id, "secret-table"),
                table_driver.delete_table_alias("secret-table", project.id),
                graph_driver.resolve_project_graph(project.id, graph_id),
                graph_driver.resolve_project_graph(project.id, "secret-graph"),
                graph_driver.delete_graph_alias("secret-graph", project.id),
            )
            for call in denied_calls:
                with pytest.raises(ProjectNotFoundError):
                    await call
        finally:
            _current_user_ctx.reset(token)

        assert await services.tables.get_table_entry_by_alias(
            project.id, "secret-table") is not None
        assert await services.graphs.get_graph_entry_by_alias(
            project.id, "secret-graph") is not None
