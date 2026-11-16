# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import pytest
from kumoapi.data_source import UploadFileType
from kumoapi.typing import Dtype, Stype

from rest.config.table_v2 import (
    Column,
    DataSourceType,
    S3SourceTable,
    TableDefinition,
)
from statedb.config.common import GroupID, ProjectID, SourceTableName
from statedb.config.timestamp import ResourceTimestamps
from statedb.db import StateDB
from statedb.db.connector_authorizations import ConnectorAuthorizationService
from statedb.db.identities import ConnectorService
from statedb.db.repositories import JsonRepository
from statedb.db.services import TableService
from statedb.db.uploaded_files import UploadedFileService
from statedb.rest.group import GroupStatus, KumoGroupDBEntry
from statedb.rest.project import ProjectDBEntry

PROJECT_A = ProjectID('project_' + 'a' * 32)
PROJECT_B = ProjectID('project_' + 'b' * 32)


def _timestamps() -> ResourceTimestamps:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return ResourceTimestamps(created_at=now, updated_at=now)


async def _seed_projects(db: StateDB) -> None:
    projects = JsonRepository('projects', ProjectDBEntry)
    groups = JsonRepository('groups', KumoGroupDBEntry)
    async with db.write_tx() as tx:
        group_id = GroupID('group_' + 'a' * 32)
        await groups.upsert(
            tx,
            str(group_id),
            KumoGroupDBEntry(
                id=group_id,
                slug='upload-group',
                display_name='Upload group',
                normalized_display_name='upload group',
                status=GroupStatus.ACTIVE,
                timestamps=_timestamps(),
            ),
        )
        for project_id in (PROJECT_A, PROJECT_B):
            await projects.upsert(
                tx,
                str(project_id),
                ProjectDBEntry(
                    id=project_id,
                    group_id=group_id,
                    slug=f"project-{str(project_id)[-8:]}",
                    display_name=str(project_id),
                    normalized_display_name=str(project_id),
                    timestamps=_timestamps(),
                ),
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('file_type', 'suffix'),
    ((UploadFileType.CSV, '.csv'), (UploadFileType.PARQUET, '.parquet')),
)
async def test_upload_metadata_connector_and_table_are_project_isolated(
    tmp_path,
    file_type: UploadFileType,
    suffix: str,
) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / 'state.db'))
    await _seed_projects(db)
    connectors = ConnectorService(db, ConnectorAuthorizationService(db))
    uploads = UploadedFileService(db)
    tables = TableService(db)
    connector_a = await connectors.register_project_upload_connector(
        PROJECT_A, '/data/project-a/uploads')
    connector_b = await connectors.register_project_upload_connector(
        PROJECT_B, '/data/project-b/uploads')
    source_name = SourceTableName(f'customers-{file_type.value.lower()}')

    uploaded_a = await uploads.record_completed(
        PROJECT_A,
        connector_a,
        source_name,
        file_type,
        f'/data/project-a/uploads/{source_name}{suffix}',
    )
    uploaded_b = await uploads.record_completed(
        PROJECT_B,
        connector_b,
        source_name,
        file_type,
        f'/data/project-b/uploads/{source_name}{suffix}',
    )
    assert uploaded_a.project_id == PROJECT_A
    assert uploaded_b.project_id == PROJECT_B
    assert uploaded_a.id != uploaded_b.id
    assert await uploads.get_completed(PROJECT_A, connector_a,
                                       source_name) == uploaded_a
    assert await uploads.get_completed(PROJECT_B, connector_b,
                                       source_name) == uploaded_b
    assert await uploads.get(PROJECT_B, connector_a, source_name,
                             file_type) is None

    with pytest.raises(ValueError, match='project upload connector'):
        await uploads.record_completed(
            PROJECT_B,
            connector_a,
            source_name,
            file_type,
            f'/data/project-b/uploads/{source_name}{suffix}',
        )
    with pytest.raises(ValueError, match='outside the project upload root'):
        await uploads.record_completed(
            PROJECT_A,
            connector_a,
            SourceTableName('escaped'),
            file_type,
            f'/data/project-b/uploads/escaped{suffix}',
        )

    definition = TableDefinition(
        cols=[Column(name='id', stype=Stype.ID, dtype=Dtype.int64)],
        source_table=S3SourceTable(
            connector_id=connector_a,
            source={'mode': 'relative', 'path': str(source_name)},
            file_type=file_type.value,
            data_source_type=DataSourceType.S3,
        ),
    )
    await tables.save_table_definition(PROJECT_A, definition, alias='upload')
    with pytest.raises(ValueError, match='not active in project'):
        await tables.save_table_definition(PROJECT_B, definition,
                                           alias='stolen-upload')
    await db.close()
