# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable

from kumo_core.types.schema import FPID

from rest.config.graph_v2 import GraphDBEntryV2
from statedb.config.common import ID, ProjectID
from statedb.db.repositories import JsonRepository
from statedb.db.runtime import StateTx
from statedb.rest.project import ProjectDBEntry, ProjectStatus
from statedb.rest.resource_alias import (
    graph_alias_conflict_message,
    normalize_resource_alias,
)


def _graph_key(project_id: ProjectID | str, fpid: FPID | str) -> str:
    return f'{project_id}:{fpid}'


class StateDBGraphRepository:
    def __init__(self) -> None:
        self._graphs = JsonRepository('graphs', GraphDBEntryV2)
        self._projects = JsonRepository('projects', ProjectDBEntry)

    async def _require_active_project(
        self,
        tx: StateTx,
        project_id: ProjectID,
    ) -> None:
        project = await self._projects.get(tx, str(project_id))
        if project is None or project.payload.status != ProjectStatus.ACTIVE:
            raise ValueError(f"Graph project '{project_id}' is not active")

    async def _validate_persistence_invariants(
        self,
        tx: StateTx,
        entry: GraphDBEntryV2,
    ) -> None:
        from statedb.db.tables import StateDBTableRepository

        await self._require_active_project(tx, entry.project_id)
        tables = await StateDBTableRepository().get_many(
            tx, entry.project_id, entry.table_name_to_dbid.values())
        missing = set(entry.table_name_to_dbid.values()) - set(tables)
        if missing:
            raise ValueError('Graph tables must all belong to its project: ' +
                             ', '.join(sorted(map(str, missing))))

    async def get(self, tx: StateTx, project_id: ProjectID,
                  fpid: FPID) -> GraphDBEntryV2 | None:
        return await self._graphs.get_payload(tx, _graph_key(project_id, fpid))

    async def get_many(self, tx: StateTx, project_id: ProjectID,
                       fpids: Iterable[FPID]) -> dict[FPID, GraphDBEntryV2]:
        fpid_values = list(dict.fromkeys(FPID(str(fpid)) for fpid in fpids))
        rows = await self._graphs.list_by_ids(
            tx, [_graph_key(project_id, fpid) for fpid in fpid_values])
        return {row.payload.id: row.payload for row in rows}

    async def get_by_alias(self, tx: StateTx, project_id: ProjectID,
                           alias: ID | str) -> GraphDBEntryV2 | None:
        row = await self._graphs.find_one_by_fields(
            tx, {
                'project_id': str(project_id),
                'normalized_alias': normalize_resource_alias(str(alias)),
            })
        return row.payload if row else None

    async def list(self, tx: StateTx,
                   project_id: ProjectID) -> list[GraphDBEntryV2]:
        rows = await self._graphs.list_by_fields(
            tx, {'project_id': str(project_id)})
        return [row.payload for row in rows]

    async def list_named(self, tx: StateTx,
                         project_id: ProjectID) -> list[GraphDBEntryV2]:
        return [
            entry for entry in await self.list(tx, project_id) if entry.alias
        ]

    async def upsert(self, tx: StateTx, fpid: FPID, entry: GraphDBEntryV2, *,
                     force_rename: bool = False) -> None:
        if entry.id != fpid:
            raise ValueError('Graph entry ID does not match its fingerprint')
        entry.__post_init__()
        await self._validate_persistence_invariants(tx, entry)
        if entry.alias:
            other = await self.get_by_alias(tx, entry.project_id, entry.alias)
            if other is not None and other.id != fpid:
                if not force_rename:
                    raise ValueError(graph_alias_conflict_message(entry.alias))
                other.alias = None
                other.__post_init__()
                await self._graphs.upsert(
                    tx, _graph_key(entry.project_id, other.id), other)
        await self._graphs.upsert(tx, _graph_key(entry.project_id, fpid), entry)

    async def delete(self, tx: StateTx, project_id: ProjectID,
                     fpid: FPID) -> GraphDBEntryV2 | None:
        await self._require_active_project(tx, project_id)
        entry = await self.get(tx, project_id, fpid)
        if entry is not None:
            await self._graphs.delete(tx, _graph_key(project_id, fpid))
        return entry

    async def decrement_or_delete(self, tx: StateTx, project_id: ProjectID,
                                  fpid: FPID) -> bool:
        await self._require_active_project(tx, project_id)
        entry = await self.get(tx, project_id, fpid)
        if entry is None:
            return False
        if entry.ref_count <= 1:
            await self._graphs.delete(tx, _graph_key(project_id, fpid))
            return True
        entry.ref_count -= 1
        await self.upsert(tx, fpid, entry)
        return True
