# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable

from kumo_core.types.schema import FPID

from rest.config.graph_v2 import GraphDBEntryV2
from rest.config.table_v2 import TableDBEntryV2
from statedb.config.common import ID, ConnectorID, ProjectID, TableName
from statedb.db.connector_admission import require_connectors_for_admission
from statedb.db.graphs import StateDBGraphRepository
from statedb.db.pqueries import StateDBPQueryRepository
from statedb.db.repositories import JsonRepository
from statedb.db.runtime import StateTx
from statedb.rest.project import ProjectDBEntry, ProjectStatus
from statedb.rest.resource_alias import normalize_resource_alias


def _table_key(project_id: ProjectID | str, fpid: FPID | str) -> str:
    return f'{project_id}:{fpid}'


class StateDBTableRepository:
    def __init__(self) -> None:
        self._tables = JsonRepository('tables', TableDBEntryV2)
        self._projects = JsonRepository('projects', ProjectDBEntry)
        self._graphs = StateDBGraphRepository()
        self._pqueries = StateDBPQueryRepository()

    async def _require_active_project(
        self,
        tx: StateTx,
        project_id: ProjectID,
    ) -> None:
        project = await self._projects.get(tx, str(project_id))
        if project is None or project.payload.status != ProjectStatus.ACTIVE:
            raise ValueError(f"Table project '{project_id}' is not active")

    async def _validate_persistence_invariants(
        self,
        tx: StateTx,
        entry: TableDBEntryV2,
    ) -> None:
        connector_id = entry.config.get_connector_id()
        await require_connectors_for_admission(tx, entry.project_id,
                                               [connector_id])

    async def get(self, tx: StateTx, project_id: ProjectID,
                  fpid: FPID) -> TableDBEntryV2 | None:
        return await self._tables.get_payload(tx, _table_key(project_id, fpid))

    async def get_many(self, tx: StateTx, project_id: ProjectID,
                       fpids: Iterable[FPID]) -> dict[FPID, TableDBEntryV2]:
        fpid_values = list(dict.fromkeys(FPID(str(fpid)) for fpid in fpids))
        rows = await self._tables.list_by_ids(
            tx, [_table_key(project_id, fpid) for fpid in fpid_values])
        return {row.payload.id: row.payload for row in rows}

    async def get_by_alias(
            self, tx: StateTx, project_id: ProjectID,
            alias: ID | TableName | str) -> TableDBEntryV2 | None:
        row = await self._tables.find_one_by_fields(
            tx, {
                'project_id': str(project_id),
                'normalized_alias': normalize_resource_alias(str(alias)),
            })
        return row.payload if row is not None else None

    async def get_many_by_alias(
        self,
        tx: StateTx,
        project_id: ProjectID,
        aliases: Iterable[ID | TableName | str],
    ) -> dict[ID | TableName | str, TableDBEntryV2]:
        requested = list(aliases)
        wanted = {normalize_resource_alias(str(alias)) for alias in requested}
        if not wanted:
            return {}
        rows = await self._tables.list_by_fields(
            tx, {'project_id': str(project_id)})
        entries_by_normalized_alias = {
            row.payload.normalized_alias: row.payload
            for row in rows if row.payload.alias is not None
            and row.payload.normalized_alias in wanted
        }
        return {
            alias: entries_by_normalized_alias[normalized]
            for alias in requested
            if (normalized := normalize_resource_alias(str(alias))
                ) in entries_by_normalized_alias
        }

    async def list(self, tx: StateTx,
                   project_id: ProjectID) -> list[TableDBEntryV2]:
        rows = await self._tables.list_by_fields(
            tx, {'project_id': str(project_id)})
        return [row.payload for row in rows]

    async def list_named(self, tx: StateTx,
                         project_id: ProjectID) -> list[TableDBEntryV2]:
        return [
            entry for entry in await self.list(tx, project_id) if entry.alias
        ]

    async def list_by_connector_id(
        self,
        tx: StateTx,
        project_id: ProjectID,
        connector_id: ConnectorID | str,
    ) -> list[TableDBEntryV2]:
        return await self._list_by_connector_id(tx, project_id,
                                                str(connector_id),
                                                named_only=True)

    async def _graph_pinned_fpids(
        self,
        tx: StateTx,
        project_id: ProjectID,
    ) -> set[FPID]:
        """Table fingerprints held by a graph a user can still get to.

        ``GraphDBEntryV2.table_name_to_dbid`` is the only place the state DB
        stores a table fingerprint, so graphs are the sole way a table stays
        alive without an alias of its own.

        Every graph pins, named or not. ``graph-<fpid>`` is a read handle that
        does not need an alias, so deleting a table an unnamed graph lists
        leaves that graph readable and broken -- the next read raises
        ``KeyError`` on the missing fingerprint.

        That alone would move the wedge up a level, since the only graph
        delete route is keyed by alias and cannot reach an unnamed graph.
        Connector delete therefore collects an unreachable graph together with
        the tables under it, rather than treating the tables as free on their
        own.
        """
        graphs = await self._graphs.list(tx, project_id)
        return {
            fpid
            for graph in graphs
            for fpid in graph.table_name_to_dbid.values()
        }

    async def unreachable_graphs(
        self,
        tx: StateTx,
        project_id: ProjectID,
    ) -> list[GraphDBEntryV2]:
        """Graphs no request can name: no alias, and no stored query holds one.

        An aliased graph is listed and deletable. An unnamed one is readable by
        id but has no delete route, so it is residue once nothing queries it.
        """
        graphs = await self._graphs.list(tx, project_id)
        unnamed = [graph for graph in graphs if not graph.alias]
        if not unnamed:
            return []
        queried = await self._pqueries.list_referenced_graph_fpids(tx)
        return [graph for graph in unnamed if graph.id not in queried]

    async def partition_connector_references(
        self,
        tx: StateTx,
        project_id: ProjectID,
        connector_id: ConnectorID | str,
    ) -> tuple[list[TableDBEntryV2], list[TableDBEntryV2],
               list[GraphDBEntryV2]]:
        """Split a connector's table references into live and orphaned.

        A row is *live* while the user can still see and remove it: it carries
        a project-local alias (so it is listed and ``DELETE /v1/tables/{alias}``
        reaches it), or a graph the user can still reach pins its fingerprint.

        Anything else is unreachable. It is absent from the Tables list, which
        is alias-only, and both delete routes are keyed by alias, so no request
        can name it. Counting such a row as a live reference makes the
        connector permanently undeletable with nothing the user can remove to
        unblock it. It is residue of an earlier write, not a reason to keep a
        connector alive.

        An unreachable graph is residue on the same terms and is returned with
        the rows it pins, for the caller to delete in this transaction. Leaving
        it would keep the wedge, one level up: it has no delete route. Deleting
        its tables and not the graph would leave it readable by ``graph-<fpid>``
        and broken.
        """
        entries = await self._list_by_connector_id(tx, project_id,
                                                   str(connector_id),
                                                   named_only=False)
        if not entries:
            return [], [], []
        stale = await self.unreachable_graphs(tx, project_id)
        stale_by_fpid = {graph.id: graph for graph in stale}
        stale_ids = set(stale_by_fpid)
        # Keep a table while any graph which survives this collection pins it.
        # Set subtraction is not sufficient: a stale and a reachable graph can
        # share a table fingerprint.
        reachable_pinned = {
            fpid
            for graph in await self._graphs.list(tx, project_id)
            if graph.id not in stale_ids
            for fpid in graph.table_name_to_dbid.values()
        }
        live: list[TableDBEntryV2] = []
        orphaned: list[TableDBEntryV2] = []
        for entry in entries:
            reachable = bool(entry.alias) or entry.id in reachable_pinned
            (live if reachable else orphaned).append(entry)
        dropped = {entry.id for entry in orphaned}
        collect = [
            graph for graph in stale_by_fpid.values()
            if dropped & set(graph.table_name_to_dbid.values())
        ]
        return live, orphaned, collect

    async def _list_by_connector_id(
        self,
        tx: StateTx,
        project_id: ProjectID,
        connector_id: str,
        *,
        named_only: bool,
    ) -> list[TableDBEntryV2]:
        fields = (
            'connector_id',
            'bigquery_connector_id',
            'snowflake_connector_id',
            'databricks_connector_id',
        )
        entries = await self.list(tx, project_id)
        return [
            entry for entry in entries
            if (not named_only or bool(entry.alias)) and any(
                str(getattr(entry.config.source_table, field, '') or '') ==
                connector_id for field in fields)
        ]

    async def alias_exists(self, tx: StateTx, project_id: ProjectID,
                           alias: ID | TableName | str) -> bool:
        return await self.get_by_alias(tx, project_id, alias) is not None

    async def _unname_or_delete(
        self,
        tx: StateTx,
        entry: TableDBEntryV2,
    ) -> None:
        """Drop ``entry``'s alias, or the whole row once nothing reaches it.

        The alias holds one reference, so releasing it decrements
        ``ref_count`` exactly as ``clear_table_alias`` does. Leaving the count
        alone would keep the row looking claimed by a name it no longer has.

        What survives the release still decides: a remaining count means
        another resource claimed this fingerprint, and a graph pins it
        whether or not that graph is reachable, because a graph is readable
        by ``graph-<fpid>`` without an alias and would break on a missing
        table.
        """
        key = _table_key(entry.project_id, entry.id)
        entry.alias = None
        entry.ref_count = max(entry.ref_count - 1, 0)
        if entry.ref_count > 0:
            entry.__post_init__()
            await self._tables.upsert(tx, key, entry)
            return
        if entry.id in await self._graph_pinned_fpids(tx, entry.project_id):
            entry.__post_init__()
            await self._tables.upsert(tx, key, entry)
            return
        await self._tables.delete(tx, key)

    async def upsert(self, tx: StateTx, fpid: FPID, entry: TableDBEntryV2, *,
                     force_rename: bool = False) -> None:
        if entry.id != fpid:
            raise ValueError('Table entry ID does not match its fingerprint')
        entry.__post_init__()
        existing = await self.get(tx, entry.project_id, fpid)
        connector_id = entry.config.get_connector_id()
        existing_connector_id = (existing.config.get_connector_id()
                                 if existing is not None else None)
        if (existing is None or connector_id != existing_connector_id
                or entry.ref_count > existing.ref_count):
            await self._validate_persistence_invariants(tx, entry)
        else:
            # Metadata cleanup and reference-count decrements do not create a
            # new live connector use. They must remain possible after an
            # identity is unbound so an administrator can drain references
            # and decommission the connector.
            await self._require_active_project(tx, entry.project_id)
        if entry.alias:
            other = await self.get_by_alias(tx, entry.project_id, entry.alias)
            if other is not None and other.id != fpid:
                if not force_rename:
                    raise ValueError(
                        f'The name {other.alias} is already used by another '
                        'table in this project.')
                # Taking the alias unnames `other`. Keep the row only while
                # something still reaches it -- a saved graph pinning its
                # fingerprint, or a reference count another resource holds.
                # An unnamed row nothing reaches is invisible in every listing
                # and has no delete handle, yet still counts as a live
                # connector reference, which is how retried table adds make a
                # connector permanently undeletable.
                await self._unname_or_delete(tx, other)
        await self._tables.upsert(tx, _table_key(entry.project_id, fpid), entry)

    async def delete(self, tx: StateTx, project_id: ProjectID,
                     fpid: FPID) -> TableDBEntryV2 | None:
        await self._require_active_project(tx, project_id)
        entry = await self.get(tx, project_id, fpid)
        if entry is not None:
            await self._tables.delete(tx, _table_key(project_id, fpid))
        return entry

    async def decrement_or_delete(self, tx: StateTx, project_id: ProjectID,
                                  fpid: FPID) -> bool:
        await self._require_active_project(tx, project_id)
        entry = await self.get(tx, project_id, fpid)
        if entry is None:
            return False
        if entry.ref_count <= 1:
            await self._tables.delete(tx, _table_key(project_id, fpid))
            return True
        entry.ref_count -= 1
        await self.upsert(tx, fpid, entry)
        return True
