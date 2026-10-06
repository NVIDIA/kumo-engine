# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from statedb.config.common import ArtifactExportJobID, ProjectID, UserID
from statedb.config.timestamp import get_current_timestamps, update_timestamps
from statedb.db.auth import AuthPrincipal
from statedb.db.identities import ConnectorService
from statedb.db.repositories import JsonRepository
from statedb.db.runtime import StateDB
from statedb.rest.artifact_export import ArtifactExportJobDBEntry
from statedb.utils.errors import ArtifactExportJobNotFoundError


class ArtifactExportJobService:
    """Persist immutable request identity; Temporal owns execution state."""

    def __init__(self, state_db: StateDB,
                 connectors: ConnectorService | None = None) -> None:
        self._state_db = state_db
        self._connectors = connectors
        self._jobs = JsonRepository('artifact_export_jobs',
                                    ArtifactExportJobDBEntry)

    async def create(
        self,
        *,
        id: ArtifactExportJobID,
        project_id: ProjectID,
        created_by: UserID,
        request_id: str,
        request_digest: str,
    ) -> tuple[ArtifactExportJobDBEntry, bool]:
        if not request_id:
            raise ValueError('request_id is required')
        if not request_digest:
            raise ValueError('request_digest is required')
        entry = ArtifactExportJobDBEntry(
            id=id,
            project_id=project_id,
            created_by=created_by,
            request_id=request_id,
            request_digest=request_digest,
            dispatch_sealed=False,
            cancel_requested=False,
            timestamps=get_current_timestamps(),
        )
        async with self._state_db.write_tx() as tx:
            inserted = await self._jobs.insert_if_absent(tx, str(id), entry)
            row = await self._jobs.get(tx, str(id))
            if row is None:
                raise RuntimeError('Artifact export request insert was lost')
            existing = row.payload
            if (existing.project_id != project_id
                    or existing.created_by != created_by
                    or existing.request_id != request_id
                    or existing.request_digest != request_digest):
                raise ValueError(
                    'Artifact export request_id was already used with a '
                    'different payload')
            return existing, inserted

    async def get(self,
                  id: ArtifactExportJobID) -> ArtifactExportJobDBEntry | None:
        async with self._state_db.read_tx() as tx:
            row = await self._jobs.get(tx, str(id))
            return row.payload if row is not None else None

    async def seal_dispatch(
        self,
        id: ArtifactExportJobID,
        *,
        project_id: ProjectID,
    ) -> ArtifactExportJobDBEntry:
        async with self._state_db.write_tx() as tx:
            row = await self._jobs.get(tx, str(id), for_update=True)
            if row is None:
                raise ArtifactExportJobNotFoundError(
                    'Artifact export job not found')
            entry = row.payload
            if entry.project_id != project_id:
                raise ValueError('Artifact export workflow identity mismatch')
            if entry.dispatch_sealed:
                return entry
            entry = replace(
                entry,
                dispatch_sealed=True,
                timestamps=update_timestamps(entry.timestamps),
            )
            await self._jobs.upsert(tx, str(id), entry)
            return entry

    async def request_cancel(
        self,
        id: ArtifactExportJobID,
        *,
        project_id: ProjectID,
    ) -> ArtifactExportJobDBEntry:
        async with self._state_db.write_tx() as tx:
            row = await self._jobs.get(tx, str(id), for_update=True)
            if row is None:
                raise ArtifactExportJobNotFoundError(
                    'Artifact export job not found')
            entry = row.payload
            if entry.project_id != project_id:
                raise ValueError('Artifact export workflow identity mismatch')
            if entry.dispatch_sealed and entry.cancel_requested:
                return entry
            entry = replace(
                entry,
                dispatch_sealed=True,
                cancel_requested=True,
                timestamps=update_timestamps(entry.timestamps),
            )
            await self._jobs.upsert(tx, str(id), entry)
            return entry

    async def get_for_project(
        self,
        id: ArtifactExportJobID,
        project_id: ProjectID,
        principal: AuthPrincipal | None,
    ) -> ArtifactExportJobDBEntry:
        if self._connectors is None:
            raise RuntimeError('Project access service is not configured')
        await self._connectors.require_project_access(principal, project_id)
        async with self._state_db.read_tx() as tx:
            row = await self._jobs.get(tx, str(id))
            if row is None or row.payload.project_id != project_id:
                raise ArtifactExportJobNotFoundError(
                    'Artifact export job not found')
            return row.payload
