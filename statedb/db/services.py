# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sqlite3
from collections.abc import Callable, Iterable
from copy import deepcopy

import asyncpg
from kumo_core.types.schema import FPID
from kumoapi.data_source import derive_resource_slug, validate_resource_slug

from rest.config.data_backend import ConnectorExecutionRef
from rest.config.graph_v2 import GraphDataComputeInput, GraphDefinition
from rest.config.table_v2 import TableDefinition
from rest.utils.routers import ValidateNameResponse
from statedb.config.common import (
    APIKeyID,
    ConnectorID,
    GraphIDv2,
    GroupID,
    PredictiveQueryID,
    ProjectID,
    TableName,
)
from statedb.config.predictive_query import PQueryDefinition
from statedb.db.api_keys import APIKeyRepository, APIKeyVerificationCandidate
from statedb.db.auth import AuthPrincipal
from statedb.db.direct_connect_auth import DirectConnectAuthService
from statedb.db.graphs import StateDBGraphRepository
from statedb.db.group_relations import lock_group_relations
from statedb.db.groups import GroupService
from statedb.db.identities import ConnectorService, IdentityService
from statedb.db.pqueries import StateDBPQueryRepository
from statedb.db.repositories import JsonRepository, _json_value, _placeholder
from statedb.db.runtime import StateDB, StateDBSchemaStatus, StateTx
from statedb.db.tables import StateDBTableRepository
from statedb.db.uploaded_files import UploadedFileService
from statedb.rest.api_key import (
    APIKeyMetadataRecord,
    APIKeySecretRecord,
    APIKeyType,
)
from statedb.rest.connector_authorization import CONNECTOR_GRAPH_ACTIONS
from statedb.rest.graph_v2 import GraphDBEntryV2
from statedb.rest.group import GroupStatus, KumoGroupDBEntry
from statedb.rest.project import ProjectDBEntry, ProjectStatus
from statedb.rest.table_v2 import TableDBEntryV2


async def _save_table_definition_in_tx(
    tables: StateDBTableRepository,
    tx: StateTx,
    project_id: ProjectID,
    table_definition: TableDefinition,
    *,
    alias: str | None = None,
    force_rename: bool = False,
    increment_ref_count: bool = True,
    created_by: str = "",
    created_by_name: str = "",
) -> TableDBEntryV2:
    from rest.config.table_v2 import TableDBEntryV2
    from rest.utils.driver_common import get_current_timestamps
    from statedb.utils.errors import TableDefinitionAlreadyExists

    fpid = table_definition.fpid
    existing = await tables.get(tx, project_id, fpid)
    if existing is not None:
        entry = deepcopy(existing)
        if entry.alias and alias and entry.alias != alias:
            if not force_rename:
                raise TableDefinitionAlreadyExists(
                    f"This table definition already exists with name "
                    f"{entry.alias}, and cannot be created under a different "
                    "name.")
        if force_rename:
            entry.alias = alias
        else:
            entry.alias = alias or entry.alias
        if increment_ref_count:
            entry.ref_count += 1
        if created_by and not entry.created_by:
            entry.created_by = created_by
        if created_by_name and not entry.created_by_name:
            entry.created_by_name = created_by_name
        await tables.upsert(tx, fpid, entry, force_rename=force_rename)
        return entry

    col_dtypes = {
        col.name: col.dtype
        for col in table_definition.cols if col.dtype
    }
    entry = TableDBEntryV2(
        id=fpid,
        project_id=project_id,
        alias=alias,
        config=table_definition,
        col_dtypes=col_dtypes,
        timestamps=get_current_timestamps(),
        created_by=created_by,
        created_by_name=created_by_name,
    )
    await tables.upsert(tx, fpid, entry, force_rename=force_rename)
    return entry


class APIKeyNameAlreadyExistsError(ValueError):
    pass


class TableService:
    def __init__(self, state_db: StateDB) -> None:
        self._state_db = state_db
        self._tables = StateDBTableRepository()

    @property
    def state_db(self) -> StateDB:
        """The underlying StateDB (e.g. to late-bind other in-process services
        such as MLJobBackend to the same DB)."""
        return self._state_db

    async def get_table_entry(
        self,
        project_id: ProjectID,
        fpid: "FPID | str",
    ) -> TableDBEntryV2 | None:
        from kumo_core.types.schema import FPID

        async with self._state_db.read_tx() as tx:
            return await self._tables.get(tx, project_id, FPID(str(fpid)))

    async def get_table_entry_by_alias(
        self,
        project_id: ProjectID,
        alias: "TableName | str",
    ) -> TableDBEntryV2 | None:
        async with self._state_db.read_tx() as tx:
            return await self._tables.get_by_alias(tx, project_id, alias)

    async def get_table_entries_by_aliases(
        self,
        project_id: ProjectID,
        aliases: "Iterable[TableName | str]",
    ) -> dict[TableName | str, TableDBEntryV2]:
        async with self._state_db.read_tx() as tx:
            return await self._tables.get_many_by_alias(tx, project_id, aliases)

    async def get_table_entries(
        self,
        project_id: ProjectID,
        fpids: "Iterable[FPID | str]",
    ) -> dict[FPID, TableDBEntryV2]:
        from kumo_core.types.schema import FPID

        async with self._state_db.read_tx() as tx:
            return await self._tables.get_many(
                tx, project_id, [FPID(str(fpid)) for fpid in fpids])

    async def get_table_definition(
        self,
        project_id: ProjectID,
        fpid: "FPID | str",
    ) -> TableDefinition | None:
        entry = await self.get_table_entry(project_id, fpid)
        return entry.config if entry is not None else None

    async def list_tables(
        self,
        project_id: ProjectID,
        connector_id: "ConnectorID | None" = None,
    ) -> list[TableDBEntryV2]:
        async with self._state_db.read_tx() as tx:
            if connector_id is None:
                return await self._tables.list(tx, project_id)
            return await self._tables.list_by_connector_id(
                tx, project_id, str(connector_id))

    async def list_named_tables(
        self,
        project_id: ProjectID,
    ) -> list[TableDBEntryV2]:
        async with self._state_db.read_tx() as tx:
            return await self._tables.list_named(tx, project_id)

    async def table_alias_exists(
        self,
        project_id: ProjectID,
        alias: "TableName | str",
    ) -> bool:
        async with self._state_db.read_tx() as tx:
            return await self._tables.alias_exists(tx, project_id, alias)

    @staticmethod
    def _table_uses_connector_id(
        table: TableDBEntryV2,
        connector_id: "ConnectorID",
    ) -> bool:
        if table.alias is None or table.alias == "":
            return False
        source_table = table.config.source_table
        for field_name in (
                "connector_id",
                "bigquery_connector_id",
                "snowflake_connector_id",
                "databricks_connector_id",
        ):
            table_connector_id = getattr(source_table, field_name, None)
            if (table_connector_id
                    and ConnectorID(str(table_connector_id)) == connector_id):
                return True
        return False

    async def upsert_table_entry(
        self,
        entry: TableDBEntryV2,
        *,
        force_rename: bool = False,
    ) -> None:
        async with self._state_db.write_tx() as tx:
            await self._tables.upsert(
                tx,
                entry.id,
                entry,
                force_rename=force_rename,
            )

    async def save_table_definition(
        self,
        project_id: ProjectID,
        table_definition: TableDefinition,
        *,
        alias: str | None = None,
        force_rename: bool = False,
        increment_ref_count: bool = True,
        created_by: str = "",
        created_by_name: str = "",
    ) -> TableDBEntryV2:
        async with self._state_db.write_tx() as tx:
            return await _save_table_definition_in_tx(
                self._tables,
                tx,
                project_id,
                table_definition,
                alias=alias,
                force_rename=force_rename,
                increment_ref_count=increment_ref_count,
                created_by=created_by,
                created_by_name=created_by_name,
            )

    async def clear_table_alias(
        self,
        project_id: ProjectID,
        alias: "TableName | str",
    ) -> TableDBEntryV2 | None:
        async with self._state_db.write_tx() as tx:
            entry = await self._tables.get_by_alias(tx, project_id, alias)
            if entry is None:
                return None
            entry = deepcopy(entry)
            entry.alias = None
            if entry.ref_count <= 1:
                await self._tables.delete(tx, project_id, entry.id)
                return entry
            entry.ref_count -= 1
            await self._tables.upsert(tx, entry.id, entry)
            return entry

    async def delete_table_entry(
        self,
        project_id: ProjectID,
        fpid: "FPID | str",
    ) -> TableDBEntryV2 | None:
        from kumo_core.types.schema import FPID

        async with self._state_db.write_tx() as tx:
            return await self._tables.delete(tx, project_id, FPID(str(fpid)))

    async def decrement_or_delete_table_entry(
        self,
        project_id: ProjectID,
        fpid: "FPID | str",
    ) -> bool:
        from kumo_core.types.schema import FPID

        async with self._state_db.write_tx() as tx:
            return await self._tables.decrement_or_delete(
                tx, project_id, FPID(str(fpid)))


class GraphService:
    def __init__(
        self,
        state_db: StateDB,
        connector_service: "ConnectorService",
    ) -> None:
        self._state_db = state_db
        self._connectors = connector_service
        self._graphs = StateDBGraphRepository()
        self._pqueries = StateDBPQueryRepository()
        self._tables = StateDBTableRepository()

    async def get_graph_entry(
        self,
        project_id: ProjectID,
        fpid: "FPID | str",
    ) -> GraphDBEntryV2 | None:
        from kumo_core.types.schema import FPID

        async with self._state_db.read_tx() as tx:
            return await self._graphs.get(tx, project_id, FPID(str(fpid)))

    async def get_graph_entry_by_alias(
        self,
        project_id: ProjectID,
        alias: str,
    ) -> GraphDBEntryV2 | None:
        async with self._state_db.read_tx() as tx:
            return await self._graphs.get_by_alias(tx, project_id, alias)

    async def get_graph_entries(
        self,
        project_id: ProjectID,
        fpids: "Iterable[FPID | str]",
    ) -> dict[FPID, GraphDBEntryV2]:
        from kumo_core.types.schema import FPID

        async with self._state_db.read_tx() as tx:
            return await self._graphs.get_many(
                tx, project_id, [FPID(str(fpid)) for fpid in fpids])

    async def get_graph_definition(
        self,
        project_id: ProjectID,
        fpid: "FPID | str",
    ) -> GraphDefinition | None:
        from kumo_core.types.schema import FPID

        from rest.config.graph_v2 import GraphDefinition

        async with self._state_db.read_tx() as tx:
            entry = await self._graphs.get(tx, project_id, FPID(str(fpid)))
            if entry is None:
                return None
            table_entries = await self._tables.get_many(
                tx, project_id, entry.table_name_to_dbid.values())
            if any(table_fpid not in table_entries
                   for table_fpid in entry.table_name_to_dbid.values()):
                return None
        tables = {
            table_name: table_entries[table_fpid].config
            for table_name, table_fpid in entry.table_name_to_dbid.items()
        }
        return GraphDefinition(tables, entry.col_groups)

    async def get_graph_definitions(
        self,
        project_id: ProjectID,
        fpids: "Iterable[FPID | str]",
    ) -> dict[FPID, GraphDefinition]:
        from kumo_core.types.schema import FPID

        from rest.config.graph_v2 import GraphDefinition

        graph_fpids = [FPID(str(fpid)) for fpid in fpids]
        if not graph_fpids:
            return {}
        async with self._state_db.read_tx() as tx:
            graph_entries = await self._graphs.get_many(tx, project_id,
                                                        graph_fpids)
            table_fpids = {
                table_fpid
                for graph_entry in graph_entries.values()
                for table_fpid in graph_entry.table_name_to_dbid.values()
            }
            table_entries = await self._tables.get_many(tx, project_id,
                                                        table_fpids)

        definitions: dict[FPID, GraphDefinition] = {}
        for graph_fpid in graph_fpids:
            graph_entry = graph_entries.get(graph_fpid)
            if graph_entry is None:
                continue
            if any(table_fpid not in table_entries
                   for table_fpid in graph_entry.table_name_to_dbid.values()):
                continue
            tables = {
                table_name: table_entries[table_fpid].config
                for table_name, table_fpid in
                graph_entry.table_name_to_dbid.items()
            }
            definitions[graph_fpid] = GraphDefinition(tables,
                                                      graph_entry.col_groups)
        return definitions

    async def get_many_graph_data_compute_inputs(
        self,
        graph_ids: Iterable[GraphIDv2],
        *,
        project_id: ProjectID,
        user_authorization: AuthPrincipal | None,
    ) -> list[GraphDataComputeInput | Exception]:
        from rest.config.graph_v2 import GraphDataComputeInput, GraphDefinition
        from statedb.config.common import ConnectorID, TableID, TableName
        from statedb.utils.errors import (
            GraphDoesNotExistError,
            TableDoesNotExistError,
        )

        graph_ids = list(graph_ids)
        if not graph_ids:
            return []

        graph_fpids_by_id: dict[GraphIDv2, FPID] = {}
        graph_ids_by_pquery_id: dict[PredictiveQueryID, list[GraphIDv2]] = {}
        for graph_id in graph_ids:
            pquery_id, graph_fpid = GraphIDv2.parse(graph_id)
            if graph_fpid is not None:
                graph_fpids_by_id[graph_id] = graph_fpid
            elif pquery_id is not None:
                graph_ids_by_pquery_id.setdefault(pquery_id,
                                                  []).append(graph_id)

        async with self._state_db.read_tx() as tx:
            if graph_ids_by_pquery_id:
                pquery_definitions = await self._pqueries.get_definitions_by_ids(
                    tx, graph_ids_by_pquery_id)
                for pquery_id, pquery_graph_ids in (
                        graph_ids_by_pquery_id.items()):
                    definition = pquery_definitions.get(pquery_id)
                    if definition is not None:
                        for graph_id in pquery_graph_ids:
                            graph_fpids_by_id[graph_id] = definition.graph_fpid

            graph_entries = await self._graphs.get_many(
                tx, project_id, graph_fpids_by_id.values())
            table_fpids = {
                table_fpid
                for graph_entry in graph_entries.values()
                for table_fpid in graph_entry.table_name_to_dbid.values()
            }
            table_entries = await self._tables.get_many(tx, project_id,
                                                        table_fpids)

            pending: list[tuple[GraphDataComputeInput, set[ConnectorID]]
                          | Exception] = []
            for graph_id in graph_ids:
                graph_fpid = graph_fpids_by_id.get(graph_id)
                graph_entry = (graph_entries.get(graph_fpid)
                               if graph_fpid is not None else None)
                if graph_entry is None:
                    pending.append(GraphDoesNotExistError(graph_id))
                    continue

                pquery_id, _ = GraphIDv2.parse(graph_id)
                canonical_entry = graph_entry.as_canonical_entry(pquery_id)
                tables: dict[TableID, TableDefinition] = {}
                table_aliases: dict[TableID, TableName] = {}
                tables_by_name: dict[TableName, TableDefinition] = {}
                missing_table = None
                for table_name, table_id in (
                        canonical_entry.table_name_to_id.items()):
                    table_entry = table_entries.get(
                        graph_entry.table_name_to_dbid[table_name])
                    if table_entry is None:
                        missing_table = TableDoesNotExistError(table_id)
                        break
                    tables[table_id] = table_entry.config
                    table_aliases[table_id] = TableName(table_name)
                    tables_by_name[TableName(table_name)] = table_entry.config
                if missing_table is not None:
                    pending.append(missing_table)
                    continue

                graph_definition = GraphDefinition(
                    tables=tables_by_name,
                    col_groups=canonical_entry.col_groups,
                )
                connector_ids = self._connectors.connector_ids_for_graph(
                    graph_definition)
                pending.append((
                    GraphDataComputeInput(
                        graph_display_name=canonical_entry.display_name,
                        graph_key=canonical_entry.dedup_key,
                        tables=tables,
                        base_tables={},
                        connectors={},
                        graph_definition=graph_definition,
                        table_aliases=table_aliases,
                    ),
                    connector_ids,
                ))

        result: list[GraphDataComputeInput | Exception] = []
        execution_refs: dict[ConnectorID, ConnectorExecutionRef] = {}
        execution_ref_errors: dict[ConnectorID, Exception] = {}
        for item in pending:
            if isinstance(item, Exception):
                result.append(item)
                continue
            compute_input, connector_ids = item
            try:
                for connector_id in connector_ids:
                    if (connector_id in execution_refs
                            or connector_id in execution_ref_errors):
                        continue
                    try:
                        execution_refs[connector_id] = await (
                            self._connectors.execution_ref_for_project(
                                user_authorization,
                                project_id,
                                connector_id,
                                CONNECTOR_GRAPH_ACTIONS,
                            ))
                    except Exception as error:
                        execution_ref_errors[connector_id] = error
                connector_error = next(
                    (execution_ref_errors[connector_id]
                     for connector_id in connector_ids
                     if connector_id in execution_ref_errors),
                    None,
                )
                if connector_error is not None:
                    raise connector_error
                connectors = {
                    connector_id: execution_refs[connector_id]
                    for connector_id in connector_ids
                }
            except Exception as error:
                result.append(error)
                continue
            compute_input.connectors = connectors
            result.append(compute_input)
        return result

    async def get_graph_data_compute_input(
        self,
        graph_id: GraphIDv2,
        *,
        project_id: ProjectID,
        user_authorization: AuthPrincipal | None,
    ) -> GraphDataComputeInput:
        from rest.config.graph_v2 import GraphDataComputeInput

        result = (await self.get_many_graph_data_compute_inputs(
            [graph_id],
            project_id=project_id,
            user_authorization=user_authorization,
        ))[0]
        if isinstance(result, GraphDataComputeInput):
            return result
        raise result

    async def get_graph_cache_input(
        self,
        graph_id: GraphIDv2,
        *,
        project_id: ProjectID,
    ) -> GraphDataComputeInput:
        """Build graph cache coordinates without resolving a connector.

        Reads of already-computed graph statistics do not open a warehouse and
        therefore must not decrypt an identity credential or turn a later
        identity-binding change into a cached-data ACL.
        """
        from rest.config.graph_v2 import GraphDataComputeInput
        from statedb.config.common import TableID, TableName
        from statedb.utils.errors import GraphDoesNotExistError

        pquery_id, graph_fpid = GraphIDv2.parse(graph_id)
        if graph_fpid is None and pquery_id is not None:
            async with self._state_db.read_tx() as tx:
                pquery = await self._pqueries.get_definition_by_id(
                    tx, pquery_id)
            graph_fpid = pquery.graph_fpid if pquery is not None else None
        if graph_fpid is None:
            raise GraphDoesNotExistError(graph_id)

        graph_entry = await self.get_graph_entry(project_id, graph_fpid)
        graph_definition = await self.get_graph_definition(
            project_id, graph_fpid)
        if graph_entry is None or graph_definition is None:
            raise GraphDoesNotExistError(graph_id)
        canonical_entry = graph_entry.as_canonical_entry(pquery_id)
        tables: dict[TableID, TableDefinition] = {
            canonical_entry.table_name_to_id[table_name]: table_definition
            for table_name, table_definition in graph_definition.tables.items()
        }
        table_aliases = {
            table_id: TableName(table_name)
            for table_name, table_id in
            canonical_entry.table_name_to_id.items()
        }
        return GraphDataComputeInput(
            graph_display_name=canonical_entry.display_name,
            graph_key=canonical_entry.dedup_key,
            tables=tables,
            base_tables={},
            connectors={},
            graph_definition=graph_definition,
            table_aliases=table_aliases,
        )

    async def get_graph_execution_input(
        self,
        graph_id: GraphIDv2,
        *,
        project_id: ProjectID,
        user_authorization: AuthPrincipal | None,
    ) -> GraphDataComputeInput:
        from rest.config.graph_v2 import GraphDataComputeInput

        result = (await self.get_many_graph_data_compute_inputs(
            [graph_id],
            project_id=project_id,
            user_authorization=user_authorization,
        ))[0]
        if isinstance(result, GraphDataComputeInput):
            return result
        raise result

    async def list_graphs(
        self,
        project_id: ProjectID,
    ) -> list[GraphDBEntryV2]:
        async with self._state_db.read_tx() as tx:
            return await self._graphs.list(tx, project_id)

    async def list_named_graphs(
        self,
        project_id: ProjectID,
    ) -> list[GraphDBEntryV2]:
        async with self._state_db.read_tx() as tx:
            return await self._graphs.list_named(tx, project_id)

    async def upsert_graph_entry(
        self,
        entry: GraphDBEntryV2,
        *,
        force_rename: bool = False,
    ) -> None:
        async with self._state_db.write_tx() as tx:
            await self._graphs.upsert(
                tx,
                entry.id,
                entry,
                force_rename=force_rename,
            )

    async def save_graph_definition(
        self,
        project_id: ProjectID,
        graph_definition: GraphDefinition,
        *,
        alias: str | None = None,
        force_rename: bool = False,
        increment_ref_count: bool = True,
        created_by: str = "",
        created_by_name: str = "",
    ) -> GraphDBEntryV2:
        from rest.config.graph_v2 import GraphDBEntryV2
        from rest.driver.error_constants import GRAPH_RENAME_MESSAGE
        from rest.utils.driver_common import get_current_timestamps
        from statedb.config.common import TableName
        from statedb.utils.errors import GraphDefinitionAlreadyExists

        graph_fpid = graph_definition.fpid
        async with self._state_db.write_tx() as tx:
            existing = await self._graphs.get(tx, project_id, graph_fpid)
            if existing is not None:
                entry = deepcopy(existing)
                if entry.alias and alias and entry.alias != alias:
                    if not force_rename:
                        raise GraphDefinitionAlreadyExists(
                            GRAPH_RENAME_MESSAGE.format(entry.alias))
                if force_rename:
                    entry.alias = alias
                else:
                    entry.alias = alias or entry.alias
                if increment_ref_count:
                    entry.ref_count += 1
                if created_by_name and not entry.created_by_name:
                    entry.created_by_name = created_by_name
                if created_by and not entry.created_by:
                    entry.created_by = created_by
                await self._graphs.upsert(
                    tx,
                    graph_fpid,
                    entry,
                    force_rename=force_rename,
                )
                return entry

            table_name_to_fpid = {}
            for table_name, table_definition in graph_definition.tables.items():
                table_entry = await _save_table_definition_in_tx(
                    self._tables,
                    tx,
                    project_id,
                    table_definition,
                    increment_ref_count=True,
                )
                table_name_to_fpid[TableName(table_name)] = table_entry.id

            entry = GraphDBEntryV2(
                id=graph_fpid,
                project_id=project_id,
                table_name_to_dbid=table_name_to_fpid,
                col_groups=graph_definition.col_groups,
                timestamps=get_current_timestamps(),
                alias=alias,
                created_by=created_by,
                created_by_name=created_by_name,
            )
            await self._graphs.upsert(
                tx,
                graph_fpid,
                entry,
                force_rename=force_rename,
            )
            return entry

    async def clear_graph_alias(
        self,
        project_id: ProjectID,
        alias: str,
    ) -> GraphDBEntryV2 | None:
        async with self._state_db.write_tx() as tx:
            entry = await self._graphs.get_by_alias(tx, project_id, alias)
            if entry is None:
                return None
            entry = deepcopy(entry)
            entry.alias = None
            await self._graphs.upsert(tx, entry.id, entry)
            return entry

    async def clear_graph_alias_by_id(
        self,
        project_id: ProjectID,
        fpid: "FPID | str",
    ) -> GraphDBEntryV2 | None:
        """Atomically clear an alias by fingerprint, or return ``None`` if
        the graph is missing or already has no alias.
        """
        async with self._state_db.write_tx() as tx:
            entry = await self._graphs.get(tx, project_id, FPID(str(fpid)))
            if entry is None or entry.alias is None:
                return None
            entry = deepcopy(entry)
            entry.alias = None
            await self._graphs.upsert(tx, entry.id, entry)
            return entry

    async def delete_graph_entry(
        self,
        project_id: ProjectID,
        fpid: "FPID | str",
    ) -> GraphDBEntryV2 | None:
        from kumo_core.types.schema import FPID

        async with self._state_db.write_tx() as tx:
            return await self._graphs.delete(tx, project_id, FPID(str(fpid)))

    async def decrement_or_delete_graph_entry(
        self,
        project_id: ProjectID,
        fpid: "FPID | str",
    ) -> bool:
        from kumo_core.types.schema import FPID

        async with self._state_db.write_tx() as tx:
            return await self._graphs.decrement_or_delete(
                tx, project_id, FPID(str(fpid)))

    async def decrement_or_delete_graph_definition(
        self,
        project_id: ProjectID,
        fpid: "FPID | str",
    ) -> bool:
        from kumo_core.types.schema import FPID

        async with self._state_db.write_tx() as tx:
            entry = await self._graphs.get(tx, project_id, FPID(str(fpid)))
            if entry is None:
                return False
            if entry.ref_count > 1:
                entry = deepcopy(entry)
                entry.ref_count -= 1
                await self._graphs.upsert(tx, entry.id, entry)
                return True
            for table_fpid in entry.table_name_to_dbid.values():
                await self._tables.decrement_or_delete(tx, project_id,
                                                       table_fpid)
            await self._graphs.delete(tx, project_id, entry.id)
            return True


class APIKeyService:
    def __init__(self, state_db: StateDB) -> None:
        self._state_db = state_db
        self._repository = APIKeyRepository()

    async def list_metadata(
        self,
        *,
        type: APIKeyType | None = None,
        owner_user_id: str | None = None,
        search: str | None = None,
    ) -> list[APIKeyMetadataRecord]:
        async with self._state_db.read_tx() as tx:
            return await self._repository.list_metadata(
                tx,
                type=type,
                owner_user_id=owner_user_id,
                search=search,
            )

    async def get_metadata(
        self,
        api_key_id: APIKeyID,
    ) -> APIKeyMetadataRecord | None:
        async with self._state_db.read_tx() as tx:
            return await self._repository.get_metadata(tx, api_key_id)

    async def create_unique(
        self,
        *,
        metadata: APIKeyMetadataRecord,
        secret: APIKeySecretRecord,
    ) -> None:
        async with self._state_db.write_tx() as tx:
            existing_name = await self._repository.find_by_owner_name(
                tx,
                owner_user_id=metadata.owner_user_id,
                name=metadata.name,
            )
            if existing_name is not None:
                raise APIKeyNameAlreadyExistsError(metadata.name)
            await self._repository.create(
                tx,
                metadata=metadata,
                secret=secret,
            )

    async def delete_if(
        self,
        api_key_id: APIKeyID,
        predicate: Callable[[APIKeyMetadataRecord], bool],
    ) -> APIKeyMetadataRecord | None:
        async with self._state_db.write_tx() as tx:
            record = await self._repository.get_metadata(tx, api_key_id)
            if record is None or not predicate(record):
                return None
            await self._repository.delete(tx, api_key_id)
            return record

    async def find_verification_candidate(
        self,
        *,
        owner_user_hash: str,
        name: str,
    ) -> APIKeyVerificationCandidate | None:
        async with self._state_db.read_tx() as tx:
            return await self._repository.find_verification_candidate(
                tx,
                owner_user_hash=owner_user_hash,
                name=name,
            )

    async def update_last_used_time(
        self,
        api_key_id: APIKeyID,
        last_used_time,
    ) -> None:
        async with self._state_db.write_tx() as tx:
            current_secret = await self._repository.get_secret(tx, api_key_id)
            if current_secret is None:
                return
            current_secret.last_used_time = last_used_time
            await self._repository.set_secret(tx, current_secret)


class PQueryService:
    def __init__(self, state_db: StateDB) -> None:
        self._state_db = state_db
        self._repository = StateDBPQueryRepository()

    async def get_definition(
        self,
        fpid: FPID,
    ) -> PQueryDefinition | None:
        async with self._state_db.read_tx() as tx:
            return await self._repository.get_definition(tx, fpid)

    async def get_definition_required(
        self,
        fpid: FPID,
    ) -> PQueryDefinition:
        async with self._state_db.read_tx() as tx:
            return await self._repository.get_definition_required(tx, fpid)

    async def get_definitions(
        self,
        fpids: "Iterable[FPID]",
    ) -> dict[FPID, PQueryDefinition]:
        async with self._state_db.read_tx() as tx:
            return await self._repository.get_definitions(tx, fpids)

    async def list_definitions(self) -> list[PQueryDefinition]:
        async with self._state_db.read_tx() as tx:
            return await self._repository.list_definitions(tx)

    async def add_definition(self, definition: PQueryDefinition) -> FPID:
        async with self._state_db.write_tx() as tx:
            return await self._repository.add_definition(tx, definition)

    async def delete_definition(self, fpid: FPID) -> bool:
        async with self._state_db.write_tx() as tx:
            return await self._repository.delete_definition(tx, fpid)

    async def get_alias(self, name: str) -> FPID | None:
        async with self._state_db.read_tx() as tx:
            return await self._repository.get_alias(tx, name)

    async def get_aliases(self, names: "Iterable[str]") -> dict[str, FPID]:
        async with self._state_db.read_tx() as tx:
            return await self._repository.get_aliases(tx, names)

    async def resolve_fpids(
        self,
        pquery_ids: "Iterable[PredictiveQueryID]",
    ) -> dict[PredictiveQueryID, FPID]:
        async with self._state_db.read_tx() as tx:
            return await self._repository.resolve_fpids(tx, pquery_ids)

    async def list_aliases(self, name_pattern: str | None = None) -> list[str]:
        async with self._state_db.read_tx() as tx:
            if name_pattern is None:
                return await self._repository.list_aliases(tx)
            return await self._repository.list_aliases_matching(
                tx, name_pattern)

    async def set_alias(self, name: str, fpid: FPID) -> None:
        async with self._state_db.write_tx() as tx:
            await self._repository.set_alias(tx, name, fpid)

    async def delete_alias(self, name: str) -> None:
        async with self._state_db.write_tx() as tx:
            await self._repository.delete_alias(tx, name)

    async def get_definition_by_id(
        self,
        pquery_id: PredictiveQueryID,
    ) -> PQueryDefinition | None:
        async with self._state_db.read_tx() as tx:
            return await self._repository.get_definition_by_id(tx, pquery_id)

    async def get_definitions_by_ids(
        self,
        pquery_ids: "Iterable[PredictiveQueryID]",
    ) -> dict[PredictiveQueryID, PQueryDefinition]:
        async with self._state_db.read_tx() as tx:
            return await self._repository.get_definitions_by_ids(tx, pquery_ids)


class ProjectService:
    def __init__(self, state_db: StateDB) -> None:
        self._state_db = state_db
        self._projects = JsonRepository("projects", ProjectDBEntry)
        self._groups = JsonRepository("groups", KumoGroupDBEntry)

    @staticmethod
    def _require_principal(
            user_authorization: AuthPrincipal | None) -> AuthPrincipal:
        if user_authorization is None:
            from statedb.utils.errors import ProjectForbiddenError
            raise ProjectForbiddenError('Authenticated principal is required')
        return user_authorization

    @staticmethod
    def _require_group_membership(
        group_id: GroupID,
        user_authorization: AuthPrincipal,
    ) -> None:
        from statedb.utils.errors import ProjectForbiddenError

        if (not user_authorization.is_admin
                and group_id not in user_authorization.kumo_group_ids):
            if getattr(user_authorization, "external_membership_unavailable",
                       False):
                from statedb.utils import errors as statedb_errors
                raise statedb_errors.ExternalMembershipUnavailableError(
                    "External group membership is unavailable; please retry")
            raise ProjectForbiddenError("Not a member of this group")

    @staticmethod
    def _require_mutation_access(
        project: ProjectDBEntry,
        user_authorization: AuthPrincipal,
    ) -> None:
        from statedb.utils.errors import ProjectForbiddenError

        if not ProjectService._project_visible_to_auth(project,
                                                       user_authorization):
            membership_could_authorize = (not project.is_private
                                          or project.created_by == str(
                                              user_authorization.user_id))
            if (membership_could_authorize
                    and getattr(user_authorization,
                                "external_membership_unavailable", False)):
                from statedb.utils import errors as statedb_errors
                raise statedb_errors.ExternalMembershipUnavailableError(
                    "External group membership is unavailable; please retry")
            raise ProjectForbiddenError("Project mutation access denied")

    @staticmethod
    def _require_owner_or_admin(
        project: ProjectDBEntry,
        user_authorization: AuthPrincipal,
    ) -> None:
        from statedb.utils.errors import ProjectForbiddenError

        ProjectService._require_mutation_access(project, user_authorization)
        if (not user_authorization.is_admin
                and project.created_by != str(user_authorization.user_id)):
            raise ProjectForbiddenError(
                "Only the project owner or an administrator may change "
                "project administration settings")

    @staticmethod
    def _existing_project(
        project: ProjectDBEntry | None,
        project_id: str,
    ) -> ProjectDBEntry:
        from statedb.utils.errors import ProjectNotFoundError

        if project is None:
            raise ProjectNotFoundError(f"Project '{project_id}' not found")
        return project

    @staticmethod
    def _require_active_project(project: ProjectDBEntry) -> None:
        from statedb.utils.errors import ProjectNotAvailableError

        if project.status != ProjectStatus.ACTIVE:
            raise ProjectNotAvailableError(
                f"Project '{project.id}' is archived and cannot be mutated")

    @staticmethod
    def _current_user_id(user_authorization: AuthPrincipal) -> str:
        return str(user_authorization.user_id)

    @staticmethod
    def _current_user_name(user_authorization: AuthPrincipal) -> str:
        return user_authorization.user_name

    @classmethod
    def _project_visible_to_auth(
        cls,
        project: ProjectDBEntry,
        user_authorization: AuthPrincipal,
    ) -> bool:
        return cls._project_visible_to_user(
            project,
            user_id=str(user_authorization.user_id),
            group_ids={
                str(group_id)
                for group_id in user_authorization.kumo_group_ids
            },
            is_admin=user_authorization.is_admin,
        )

    async def _list_visible_projects_in_tx(
        self,
        tx,
        user_authorization: AuthPrincipal,
        *,
        group_id: str | None = None,
        display_name: str | None = None,
        slug: str | None = None,
        exclude_project_id: str | None = None,
    ) -> list[ProjectDBEntry]:
        clauses: list[str] = []
        args: list[object] = []
        if group_id is not None:
            args.append(group_id)
            clauses.append(f"{_json_value(tx, 'group_id')} = "
                           f"{_placeholder(tx, len(args))}")
        if display_name is not None:
            args.append(display_name)
            clauses.append(f"{_json_value(tx, 'display_name')} = "
                           f"{_placeholder(tx, len(args))}")
        if slug is not None:
            args.append(slug)
            clauses.append(f"{_json_value(tx, 'slug')} = "
                           f"{_placeholder(tx, len(args))}")
        if exclude_project_id is not None:
            args.append(exclude_project_id)
            clauses.append(f"id <> {_placeholder(tx, len(args))}")

        if not user_authorization.is_admin:
            group_ids = [
                str(group_id) for group_id in user_authorization.kumo_group_ids
            ]
            if group_ids:
                placeholders = []
                for user_group_id in group_ids:
                    args.append(user_group_id)
                    placeholders.append(_placeholder(tx, len(args)))
                clauses.append(f"{_json_value(tx, 'group_id')} IN "
                               f"({', '.join(placeholders)})")
            else:
                clauses.append('1 = 0')

        rows = await self._projects.list_where_sql(
            tx,
            " AND ".join(clauses) if clauses else "1 = 1",
            args,
        )
        return [
            row.payload for row in rows
            if self._project_visible_to_auth(row.payload, user_authorization)
        ]

    async def _get_project_in_tx(
        self,
        tx,
        project_id: str,
        user_authorization: AuthPrincipal,
        *,
        require_active: bool = False,
    ) -> ProjectDBEntry:
        from statedb.utils.errors import ProjectNotFoundError

        row = await self._projects.get(tx, str(project_id))
        project = self._existing_project(row.payload if row else None,
                                         str(project_id))
        if not self._project_visible_to_auth(project, user_authorization):
            raise ProjectNotFoundError(f"Project '{project_id}' not found")
        if require_active:
            self._require_active_project(project)
        return project

    async def list_visible_project_ids(
        self,
        user_authorization: AuthPrincipal,
    ) -> list[str]:
        return [
            str(project.id) for project in await
            self._list_visible_projects_allow_partial(user_authorization)
        ]

    async def _list_visible_projects_allow_partial(
        self,
        user_authorization: AuthPrincipal,
    ) -> list[ProjectDBEntry]:
        async with self._state_db.read_tx() as tx:
            return await self._list_visible_projects_in_tx(
                tx, user_authorization)

    async def create_project(
        self,
        *,
        name: str,
        group_id: str,
        slug: str | None = None,
        description: str = "",
        is_private: bool = False,
        user_authorization: AuthPrincipal | None,
    ) -> ProjectDBEntry:
        from rest.utils.driver_common import get_current_timestamps
        from statedb.config.common import GroupID, ProjectID
        from statedb.db.identities import normalize_display_name
        from statedb.utils.errors import (
            GroupNotFoundError,
            ProjectNameConflictError,
            ResourceSlugConflictError,
        )

        user_authorization = self._require_principal(user_authorization)
        group_id = GroupID(group_id)
        display_name = name.strip()
        normalized_name = normalize_display_name(display_name)
        if not normalized_name:
            raise ValueError('Project name cannot be empty')
        if len(display_name) > 64:
            raise ValueError('Project name cannot exceed 64 characters')
        resource_slug = (validate_resource_slug(slug) if slug is not None else
                         derive_resource_slug(display_name))
        try:
            async with self._state_db.write_tx() as tx:
                await lock_group_relations(tx, [group_id])
                group_row = await self._groups.get(tx, str(group_id))
                if (group_row is None
                        or group_row.payload.status != GroupStatus.ACTIVE):
                    raise GroupNotFoundError("Group not found.")
                self._require_group_membership(group_id, user_authorization)
                if await self._projects.exists_by_fields(
                        tx, {'normalized_display_name': normalized_name}):
                    raise ProjectNameConflictError(
                        'Project name is not available')
                if (await self._projects.exists_by_fields(
                        tx, {'normalized_display_name': resource_slug})
                        or await self._projects.exists_by_fields(
                            tx, {'slug': normalized_name})):
                    raise ProjectNameConflictError(
                        'Project name conflicts with an existing slug')
                project_id = ProjectID.generate()
                entry = ProjectDBEntry(
                    id=project_id,
                    slug=resource_slug,
                    display_name=display_name,
                    description=description,
                    group_id=group_id,
                    is_private=is_private,
                    normalized_display_name=normalized_name,
                    created_by=str(user_authorization.user_id),
                    created_by_name=user_authorization.user_name,
                    status=ProjectStatus.ACTIVE,
                    timestamps=get_current_timestamps(),
                )
                await self._projects.upsert(tx, str(project_id), entry)
        except (sqlite3.IntegrityError, asyncpg.UniqueViolationError) as error:
            async with self._state_db.read_tx() as tx:
                if await self._projects.exists_by_fields(
                        tx, {'slug': resource_slug}):
                    raise ResourceSlugConflictError(
                        f"Project slug '{resource_slug}' is already in use."
                    ) from error
            raise ProjectNameConflictError(
                'Project name is not available') from error
        return entry

    async def list_visible_projects(
        self,
        user_authorization: AuthPrincipal | None,
    ) -> list[ProjectDBEntry]:
        user_authorization = self._require_principal(user_authorization)
        async with self._state_db.read_tx() as tx:
            return await self._list_visible_projects_in_tx(
                tx, user_authorization)

    async def list_projects_in_group(
        self,
        group_id: GroupID,
        user_authorization: AuthPrincipal | None,
    ) -> list[ProjectDBEntry]:
        from statedb.config.common import GroupID

        user_authorization = self._require_principal(user_authorization)
        group_id = GroupID(group_id)
        self._require_group_membership(group_id, user_authorization)
        async with self._state_db.read_tx() as tx:
            return await self._list_visible_projects_in_tx(
                tx, user_authorization, group_id=str(group_id))

    async def get_project(
        self,
        project_id: "ProjectID | str",
        user_authorization: AuthPrincipal | None,
    ) -> ProjectDBEntry:
        from statedb.utils.errors import ProjectNotFoundError

        user_authorization = self._require_principal(user_authorization)
        async with self._state_db.read_tx() as tx:
            row = await self._projects.get(tx, str(project_id))
        project = self._existing_project(row.payload if row else None,
                                         str(project_id))
        if not self._project_visible_to_user(
                project,
                user_id=str(user_authorization.user_id),
                group_ids={
                    str(group_id)
                    for group_id in user_authorization.kumo_group_ids
                },
                is_admin=user_authorization.is_admin,
        ):
            raise ProjectNotFoundError(f"Project '{project_id}' not found")
        return project

    async def get_project_by_slug(
        self,
        slug: str,
        user_authorization: AuthPrincipal | None,
    ) -> ProjectDBEntry:
        """Resolve a slug only within the caller's visible project scope."""
        from statedb.utils.errors import ProjectNotFoundError

        user_authorization = self._require_principal(user_authorization)
        slug = validate_resource_slug(slug, allow_reserved=True)
        async with self._state_db.read_tx() as tx:
            projects = await self._list_visible_projects_in_tx(
                tx, user_authorization, slug=slug)
        if not projects:
            raise ProjectNotFoundError('Project not found')
        return projects[0]

    async def archive_project(
        self,
        project_id: "ProjectID | str",
        user_authorization: AuthPrincipal | None,
    ) -> ProjectDBEntry:
        from kumoapi.data_source import DEFAULT_PROJECT_ID

        from statedb.config.timestamp import update_timestamps

        user_authorization = self._require_principal(user_authorization)
        if str(project_id) == DEFAULT_PROJECT_ID:
            raise ValueError('The canonical Default project cannot be archived')
        async with self._state_db.write_tx() as tx:
            row = await self._projects.get(tx, str(project_id))
            project = self._existing_project(row.payload if row else None,
                                             str(project_id))
            # Connector creation locks the owning group's relations around
            # its final active-project/binding recheck and insert. Taking the
            # same lock prevents archival from racing that final decision.
            await lock_group_relations(tx, [project.group_id])
            row = await self._projects.get(tx, str(project_id))
            project = self._existing_project(row.payload if row else None,
                                             str(project_id))
            self._require_owner_or_admin(project, user_authorization)
            self._require_active_project(project)
            project.status = ProjectStatus.ARCHIVED
            project.timestamps = update_timestamps(project.timestamps)
            await self._projects.upsert(tx, str(project.id), project)
        return project

    async def update_project(
        self,
        project_id: "ProjectID | str",
        *,
        name: str | None = None,
        description: str | None = None,
        is_private: bool | None = None,
        user_authorization: AuthPrincipal | None,
    ) -> ProjectDBEntry:
        from kumoapi.data_source import DEFAULT_PROJECT_ID

        from statedb.config.timestamp import update_timestamps
        from statedb.db.identities import normalize_display_name
        from statedb.utils.errors import ProjectNameConflictError

        user_authorization = self._require_principal(user_authorization)
        if str(project_id) == DEFAULT_PROJECT_ID:
            if name is not None and name.strip() != 'Default':
                raise ValueError(
                    'The canonical Default project cannot be renamed')
            if is_private:
                raise ValueError(
                    'The canonical Default project must remain shared')
        try:
            async with self._state_db.write_tx() as tx:
                row = await self._projects.get(tx, str(project_id))
                project = self._existing_project(row.payload if row else None,
                                                 str(project_id))
                if name is not None or is_private is not None:
                    self._require_owner_or_admin(project, user_authorization)
                else:
                    self._require_mutation_access(project, user_authorization)
                self._require_active_project(project)
                if name is not None:
                    display_name = name.strip()
                    normalized_name = normalize_display_name(display_name)
                    if not normalized_name:
                        raise ValueError('Project name cannot be empty')
                    if len(display_name) > 64:
                        raise ValueError(
                            'Project name cannot exceed 64 characters')
                    if await self._projects.exists_by_fields(
                            tx, {'normalized_display_name': normalized_name},
                            exclude_id=str(project.id)):
                        raise ProjectNameConflictError(
                            'Project name is not available')
                    if await self._projects.exists_by_fields(
                            tx, {'slug': normalized_name},
                            exclude_id=str(project.id)):
                        raise ProjectNameConflictError(
                            'Project name conflicts with an existing slug')
                    project.display_name = display_name
                    project.normalized_display_name = normalized_name
                if description is not None:
                    project.description = description
                if is_private is not None:
                    project.is_private = is_private
                project.timestamps = update_timestamps(project.timestamps)
                await self._projects.upsert(tx, str(project.id), project)
        except (sqlite3.IntegrityError, asyncpg.UniqueViolationError) as error:
            raise ProjectNameConflictError(
                'Project name is not available') from error
        return project

    async def validate_project_for_creation(
        self,
        project_id: "ProjectID | str",
        user_authorization: AuthPrincipal | None,
    ) -> ProjectDBEntry:
        from statedb.utils.errors import ProjectNotAvailableError

        project = await self.get_project(project_id, user_authorization)
        if project.status == ProjectStatus.ARCHIVED:
            raise ProjectNotAvailableError(
                f"Project '{project_id}' is archived "
                "and cannot accept new resources.")
        return project

    async def validate_project_name(
        self,
        *,
        name: str,
        exclude_project_id: str | None = None,
        user_authorization: AuthPrincipal | None,
    ) -> ValidateNameResponse:
        from rest.utils.routers import ValidateNameResponse

        if not name or not name.strip():
            return ValidateNameResponse(
                is_valid=False,
                error_msg="Project name cannot be empty.",
            )

        if len(name.strip()) > 64:
            return ValidateNameResponse(
                is_valid=False,
                error_msg="Project name cannot exceed 64 characters.",
            )

        self._require_principal(user_authorization)
        from statedb.db.identities import normalize_display_name

        async with self._state_db.read_tx() as tx:
            matching = await self._projects.exists_by_fields(
                tx,
                {
                    'normalized_display_name': normalize_display_name(name),
                },
                exclude_id=exclude_project_id,
            )
        if matching:
            return ValidateNameResponse(
                is_valid=False, error_msg='Project name is not available.')

        return ValidateNameResponse(is_valid=True, error_msg=None)

    @staticmethod
    def _project_visible_to_user(
        project: ProjectDBEntry,
        *,
        user_id: str,
        group_ids: set[str],
        is_admin: bool,
    ) -> bool:
        if is_admin:
            return True
        if str(project.group_id) not in group_ids:
            return False
        if project.is_private:
            return project.created_by == user_id
        return True


class StateDBStatusService:
    def __init__(self, state_db: StateDB) -> None:
        self._state_db = state_db

    async def get_schema_status(self) -> StateDBSchemaStatus:
        return await self._state_db.get_schema_status()

    async def is_healthy(self) -> bool:
        return (await self.get_schema_status()).ok


class StateServices:
    def __init__(self, state_db: StateDB) -> None:
        from statedb.db.artifact_exports import ArtifactExportJobService
        from statedb.db.ml_jobs import MLJobService

        self.state_db = state_db
        self.status = StateDBStatusService(state_db)
        self.api_keys = APIKeyService(state_db)
        from statedb.db import connector_authorizations
        self.connector_authorizations = (
            connector_authorizations.ConnectorAuthorizationService(state_db))
        self.connectors = ConnectorService(state_db,
                                           self.connector_authorizations)
        self.artifact_exports = ArtifactExportJobService(
            state_db, self.connectors)
        self.identities = IdentityService(state_db,
                                          self.connector_authorizations)
        self.direct_connect_auth = DirectConnectAuthService(state_db)
        self.groups = GroupService(state_db)
        self.graphs = GraphService(state_db, self.connectors)
        self.ml_jobs = MLJobService(state_db)
        self.pqueries = PQueryService(state_db)
        self.projects = ProjectService(state_db)
        self.tables = TableService(state_db)
        self.uploaded_files = UploadedFileService(state_db)
