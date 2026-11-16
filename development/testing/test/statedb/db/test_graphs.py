# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import pytest
from kumo_core.types.schema import FPID
from kumoapi.graph import ColumnKeyGroup
from kumoapi.typing import Dtype, Stype

from rest.config.connector import FileConnectorResourceConfig
from rest.config.graph_v2 import GraphDBEntryV2
from rest.config.table_v2 import (
    Column,
    DataSourceType,
    S3SourceTable,
    TableDBEntryV2,
    TableDefinition,
)
from statedb.config.common import ConnectorID, GroupID, ProjectID, TableName
from statedb.config.timestamp import ResourceTimestamps
from statedb.db import StateDB
from statedb.db.connectors import StateDBConnectorRepository
from statedb.db.graphs import StateDBGraphRepository
from statedb.db.repositories import JsonRepository
from statedb.db.tables import StateDBTableRepository
from statedb.rest.connector_db import InternalConnectorDBEntry
from statedb.rest.group import GroupStatus, KumoGroupDBEntry
from statedb.rest.project import ProjectDBEntry

PROJECT_A = ProjectID('project_' + '1' * 32)
PROJECT_B = ProjectID('project_' + '2' * 32)
GRAPH_FPID = FPID('1' * 32)
CONNECTOR_A = ConnectorID('connector_' + '1' * 32)
CONNECTOR_B = ConnectorID('connector_' + '2' * 32)


def _table_definition(connector_id: ConnectorID) -> TableDefinition:
    return TableDefinition(
        cols=[Column(name='id', stype=Stype.ID, dtype=Dtype.int64)],
        source_table=S3SourceTable(
            connector_id=connector_id,
            source={
                'mode': 'relative',
                'path': 'account.parquet'
            },
            file_type='PARQUET',
            data_source_type=DataSourceType.S3,
        ),
    )


TABLE_A = _table_definition(CONNECTOR_A).fpid
TABLE_B = _table_definition(CONNECTOR_B).fpid


def _entry(project_id: ProjectID, alias: str,
           ref_count: int = 1) -> GraphDBEntryV2:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return GraphDBEntryV2(
        id=GRAPH_FPID,
        project_id=project_id,
        table_name_to_dbid={
            TableName('ACCOUNT'):
            TABLE_A if project_id == PROJECT_A else TABLE_B
        },
        col_groups=[ColumnKeyGroup(columns=())],
        alias=alias,
        ref_count=ref_count,
        timestamps=ResourceTimestamps(created_at=now, updated_at=now),
    )


async def _seed_projects_connectors_and_tables(db: StateDB) -> None:
    projects = JsonRepository('projects', ProjectDBEntry)
    groups = JsonRepository('groups', KumoGroupDBEntry)
    connectors = StateDBConnectorRepository()
    tables = StateDBTableRepository()
    async with db.write_tx() as tx:
        group_id = GroupID('group_' + 'a' * 32)
        await groups.upsert(
            tx,
            str(group_id),
            KumoGroupDBEntry(
                id=group_id,
                slug='graph-group',
                display_name='Graph group',
                normalized_display_name='graph group',
                status=GroupStatus.ACTIVE,
                timestamps=ResourceTimestamps(
                    created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                ),
            ),
        )
        for project_id, connector_id, table_id in ((PROJECT_A, CONNECTOR_A,
                                                    TABLE_A),
                                                   (PROJECT_B, CONNECTOR_B,
                                                    TABLE_B)):
            await projects.upsert(
                tx,
                str(project_id),
                ProjectDBEntry(
                    id=project_id,
                    group_id=group_id,
                    slug=f"project-{str(project_id)[-8:]}",
                    display_name=str(project_id),
                    normalized_display_name=str(project_id),
                    timestamps=ResourceTimestamps(
                        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    ),
                ),
            )
            await connectors.create(
                tx,
                InternalConnectorDBEntry(
                    id=connector_id,
                    project_id=project_id,
                    name='uploads',
                    normalized_name='uploads',
                    config=FileConnectorResourceConfig(
                        name='uploads', root_dir=f'/tmp/{project_id}'),
                    timestamps=ResourceTimestamps(
                        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    ),
                ),
            )
            definition = _table_definition(connector_id)
            await tables.upsert(
                tx,
                table_id,
                TableDBEntryV2(
                    id=table_id,
                    project_id=project_id,
                    config=definition,
                    col_dtypes={},
                    timestamps=ResourceTimestamps(
                        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    ),
                ),
            )


@pytest.mark.asyncio
async def test_graph_repository_is_project_scoped(tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    await _seed_projects_connectors_and_tables(db)
    repo = StateDBGraphRepository()
    first = _entry(PROJECT_A, 'Graph')
    second = _entry(PROJECT_B, 'graph')
    async with db.write_tx() as tx:
        await repo.upsert(tx, GRAPH_FPID, first)
        await repo.upsert(tx, GRAPH_FPID, second)
    async with db.read_tx() as tx:
        assert await repo.get(tx, PROJECT_A, GRAPH_FPID) == first
        assert await repo.get(tx, PROJECT_B, GRAPH_FPID) == second
        assert await repo.get_by_alias(tx, PROJECT_A, 'GRAPH') == first
        assert await repo.list_named(tx, PROJECT_A) == [first]
    await db.close()


@pytest.mark.asyncio
async def test_graph_alias_uniqueness_is_per_project(tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    await _seed_projects_connectors_and_tables(db)
    repo = StateDBGraphRepository()
    first = _entry(PROJECT_A, 'Graph')
    collision = _entry(PROJECT_A, 'GRAPH')
    collision.id = FPID('2' * 32)
    other_project = _entry(PROJECT_B, 'GRAPH')
    other_project.id = collision.id
    async with db.write_tx() as tx:
        await repo.upsert(tx, first.id, first)
        with pytest.raises(ValueError) as error:
            await repo.upsert(tx, collision.id, collision)
        assert str(error.value) == (
            'Graph name GRAPH is already used in this project.')
        await repo.upsert(tx, other_project.id, other_project)
    await db.close()


@pytest.mark.asyncio
async def test_graph_alias_uses_one_nfkc_key_for_lookup_and_collision(
        tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    await _seed_projects_connectors_and_tables(db)
    repo = StateDBGraphRepository()
    fullwidth = _entry(PROJECT_A, 'Ｇｒａｐｈ')
    collision = _entry(PROJECT_A, ' graph ')
    collision.id = FPID('2' * 32)

    async with db.write_tx() as tx:
        await repo.upsert(tx, fullwidth.id, fullwidth)
        assert await repo.get_by_alias(tx, PROJECT_A, 'GRAPH') == fullwidth
        with pytest.raises(ValueError, match='already used'):
            await repo.upsert(tx, collision.id, collision)

    await db.close()


@pytest.mark.asyncio
async def test_graph_delete_cannot_cross_projects(tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    await _seed_projects_connectors_and_tables(db)
    repo = StateDBGraphRepository()
    first = _entry(PROJECT_A, 'a')
    second = _entry(PROJECT_B, 'b')
    async with db.write_tx() as tx:
        await repo.upsert(tx, GRAPH_FPID, first)
        await repo.upsert(tx, GRAPH_FPID, second)
        deleted = await repo.delete(tx, PROJECT_A, GRAPH_FPID)
        assert deleted == first
    async with db.read_tx() as tx:
        assert await repo.get(tx, PROJECT_A, GRAPH_FPID) is None
        assert await repo.get(tx, PROJECT_B, GRAPH_FPID) == second
    await db.close()


@pytest.mark.asyncio
async def test_graph_repository_rejects_table_from_another_project(
        tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    await _seed_projects_connectors_and_tables(db)
    repo = StateDBGraphRepository()
    entry = _entry(PROJECT_B, 'cross-project')
    entry.table_name_to_dbid = {TableName('ACCOUNT'): TABLE_A}
    async with db.write_tx() as tx:
        with pytest.raises(ValueError, match='must all belong'):
            await repo.upsert(tx, entry.id, entry)
    await db.close()
