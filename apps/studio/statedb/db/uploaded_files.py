# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unicodedata

from kumoapi.data_source import UploadFileType

from statedb.config.common import (
    ConnectorID,
    ProjectID,
    SourceTableName,
    UploadedFileID,
)
from statedb.db.repositories import JsonRepository
from statedb.db.runtime import StateDB
from statedb.rest.connector_db import is_active_project_upload_connector
from statedb.rest.project import ProjectDBEntry, ProjectStatus
from statedb.rest.uploaded_file import UploadedFileDBEntry
from statedb.utils.errors import ProjectNotFoundError


def normalize_uploaded_file_name(value: str) -> str:
    return unicodedata.normalize('NFKC', value.strip()).casefold()


class UploadedFileService:
    def __init__(self, state_db: StateDB) -> None:
        self._state_db = state_db
        self._uploads = JsonRepository('uploaded_files', UploadedFileDBEntry)
        self._projects = JsonRepository('projects', ProjectDBEntry)

    @staticmethod
    async def _require_project_upload_connector(tx, project_id, connector_id):
        from statedb.db.connectors import StateDBConnectorRepository

        connector = await StateDBConnectorRepository().get(tx, connector_id)
        if (connector is None or connector.project_id != project_id
                or not is_active_project_upload_connector(connector)):
            raise ValueError(
                'Uploaded file connector must be an active project upload '
                'connector')
        return connector

    async def record_completed(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        source_table_name: SourceTableName,
        file_type: UploadFileType,
        path: str,
    ) -> UploadedFileDBEntry:
        from statedb.config.timestamp import (
            get_current_timestamps,
            update_timestamps,
        )

        normalized_name = normalize_uploaded_file_name(source_table_name)
        async with self._state_db.write_tx() as tx:
            project = await self._projects.get(tx, str(project_id))
            if (project is None
                    or project.payload.status != ProjectStatus.ACTIVE):
                raise ProjectNotFoundError(
                    f"Active project '{project_id}' not found")
            connector = await self._require_project_upload_connector(
                tx, project_id, connector_id)
            root = connector.config.root_dir.rstrip('/')
            if path != root and not path.startswith(root + '/'):
                raise ValueError(
                    'Uploaded file path is outside the project upload root')
            existing = await self._uploads.find_one_by_fields(
                tx,
                {
                    'project_id': str(project_id),
                    'normalized_source_table_name': normalized_name,
                },
            )
            if existing is not None:
                entry = existing.payload
                if entry.connector_id != connector_id:
                    raise ValueError(
                        'Uploaded file belongs to another project connector')
                if entry.file_type != file_type:
                    raise ValueError(
                        'An uploaded file with this name already exists with '
                        'another file type')
                entry.path = path
                entry.timestamps = update_timestamps(entry.timestamps)
            else:
                entry = UploadedFileDBEntry(
                    id=UploadedFileID.generate(),
                    project_id=project_id,
                    connector_id=connector_id,
                    source_table_name=source_table_name,
                    normalized_source_table_name=normalized_name,
                    file_type=file_type,
                    path=path,
                    timestamps=get_current_timestamps(),
                )
            await self._uploads.upsert(tx, str(entry.id), entry)
            return entry

    async def get(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        source_table_name: SourceTableName,
        file_type: UploadFileType,
    ) -> UploadedFileDBEntry | None:
        async with self._state_db.read_tx() as tx:
            row = await self._uploads.find_one_by_fields(
                tx,
                {
                    'project_id':
                    str(project_id),
                    'connector_id':
                    str(connector_id),
                    'normalized_source_table_name':
                    normalize_uploaded_file_name(source_table_name),
                    'file_type':
                    file_type,
                },
            )
        return row.payload if row is not None else None

    async def get_completed(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        source_table_name: SourceTableName,
    ) -> UploadedFileDBEntry | None:
        """Return completed upload metadata without exposing its stored path.

        Callers use only ``file_type``. The lookup independently rechecks the
        active project and exact hidden upload connector so authorization
        cannot race project archival or connector tombstoning.
        """
        async with self._state_db.read_tx() as tx:
            project = await self._projects.get(tx, str(project_id))
            if (project is None
                    or project.payload.status != ProjectStatus.ACTIVE):
                raise ProjectNotFoundError(
                    f"Active project '{project_id}' not found")
            await self._require_project_upload_connector(
                tx, project_id, connector_id)
            row = await self._uploads.find_one_by_fields(
                tx,
                {
                    'project_id':
                    str(project_id),
                    'connector_id':
                    str(connector_id),
                    'normalized_source_table_name':
                    normalize_uploaded_file_name(source_table_name),
                },
            )
        return row.payload if row is not None else None

    async def delete(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        source_table_name: SourceTableName,
        file_type: UploadFileType,
    ) -> bool:
        async with self._state_db.write_tx() as tx:
            project = await self._projects.get(tx, str(project_id))
            if (project is None
                    or project.payload.status != ProjectStatus.ACTIVE):
                raise ProjectNotFoundError(
                    f"Active project '{project_id}' not found")
            await self._require_project_upload_connector(
                tx, project_id, connector_id)
            row = await self._uploads.find_one_by_fields(
                tx,
                {
                    'project_id':
                    str(project_id),
                    'connector_id':
                    str(connector_id),
                    'normalized_source_table_name':
                    normalize_uploaded_file_name(source_table_name),
                    'file_type':
                    file_type,
                },
            )
            if row is None:
                return False
            await self._uploads.delete(tx, row.id)
            return True

    async def list_for_project(
        self,
        project_id: ProjectID,
    ) -> list[UploadedFileDBEntry]:
        async with self._state_db.read_tx() as tx:
            rows = await self._uploads.list_by_field(tx, 'project_id',
                                                     str(project_id))
        return [row.payload for row in rows]
