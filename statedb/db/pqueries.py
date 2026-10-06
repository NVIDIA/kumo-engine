# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections.abc import Iterable

from kumo_core.types.schema import FPID

from io_utils import kumo_dataclass
from statedb.config.common import PredictiveQueryID
from statedb.config.predictive_query import PQueryDefinition
from statedb.db.repositories import (
    JsonRepository,
    _in_predicate,
    _json_value,
    _placeholder,
)
from statedb.db.runtime import StateTx


@kumo_dataclass
class PQueryAliasRecord:
    name: str
    fpid: FPID


def _literal_regex_prefix(pattern: str) -> str:
    if pattern.startswith("^"):
        pattern = pattern[1:]
    prefix: list[str] = []
    escaped = False
    for char in pattern:
        if escaped:
            prefix.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in ".^$*+?{}[]|()":
            break
        prefix.append(char)
    return "".join(prefix)


class StateDBPQueryRepository:
    def __init__(self) -> None:
        self._definitions = JsonRepository(
            "pquery_definitions",
            PQueryDefinition,
        )
        self._aliases = JsonRepository("pquery_aliases", PQueryAliasRecord)

    async def get_definition(
        self,
        tx: StateTx,
        fpid: FPID,
    ) -> PQueryDefinition | None:
        return await self._definitions.get_payload(tx, fpid)

    async def get_definition_required(
        self,
        tx: StateTx,
        fpid: FPID,
    ) -> PQueryDefinition:
        definition = await self.get_definition(tx, fpid)
        if definition is None:
            raise KeyError(f"PQueryDefinition id={fpid} is not found in DB.")
        return definition

    async def get_definitions(
        self,
        tx: StateTx,
        fpids: Iterable[FPID],
    ) -> dict[FPID, PQueryDefinition]:
        fpid_values = [str(fpid) for fpid in fpids]
        rows = await self._definitions.list_by_ids(tx, fpid_values)
        ordered_rows = JsonRepository.rows_in_id_order(rows, fpid_values)
        return {FPID(row.id): row.payload for row in ordered_rows}

    async def list_definitions(self, tx: StateTx) -> list[PQueryDefinition]:
        return [row.payload for row in await self._definitions.list_all(tx)]

    async def list_referenced_graph_fpids(self, tx: StateTx) -> set[FPID]:
        """Every graph fingerprint a stored query holds.

        Definitions are keyed by fingerprint alone, with no project column, so
        this spans projects. A graph fingerprint is content-addressed, so a
        match in another project can only over-retain, never under-retain.
        """
        return {
            definition.graph_fpid
            for definition in await self.list_definitions(tx)
        }

    async def add_definition(
        self,
        tx: StateTx,
        definition: PQueryDefinition,
    ) -> FPID:
        fpid = definition.fpid
        if await self.get_definition(tx, fpid) is None:
            await self._definitions.upsert(tx, fpid, definition)
        return fpid

    async def delete_definition(self, tx: StateTx, fpid: FPID) -> bool:
        if await self.get_definition(tx, fpid) is None:
            return False
        await self._definitions.delete(tx, fpid)
        return True

    async def _get_definition_fpid_by_name(
        self,
        tx: StateTx,
        name: str,
    ) -> FPID | None:
        rows = await self._definitions.list_by_field(tx, "pquery_name", name)
        if len(rows) == 1:
            return FPID(rows[0].id)
        return None

    async def get_alias(self, tx: StateTx, name: str) -> FPID | None:
        definition_fpid = await self._get_definition_fpid_by_name(tx, name)
        if definition_fpid is not None:
            return definition_fpid
        row = await self._aliases.get(tx, name)
        return FPID(row.payload.fpid) if row else None

    async def get_aliases(
        self,
        tx: StateTx,
        names: Iterable[str],
    ) -> dict[str, FPID]:
        unique_names = list(dict.fromkeys(str(name) for name in names))
        if not unique_names:
            return {}

        args: list[object] = []
        definition_rows = await self._definitions.list_where_sql(
            tx,
            _in_predicate(tx, _json_value(tx, "pquery_name"), unique_names,
                          args),
            args,
        )
        definition_fpids_by_name: dict[str, list[FPID]] = {}
        for row in definition_rows:
            if row.payload.pquery_name is not None:
                definition_fpids_by_name.setdefault(row.payload.pquery_name,
                                                    []).append(FPID(row.id))
        aliases_by_name = {
            name: fpids[0]
            for name, fpids in definition_fpids_by_name.items()
            if len(fpids) == 1
        }

        missing_names = [
            name for name in unique_names if name not in aliases_by_name
        ]
        alias_rows = await self._aliases.list_by_ids(tx, missing_names)
        aliases_by_name.update(
            {row.payload.name: FPID(row.payload.fpid)
             for row in alias_rows})
        return {
            name: aliases_by_name[name]
            for name in unique_names if name in aliases_by_name
        }

    async def list_aliases(self, tx: StateTx) -> list[str]:
        definition_names = {
            row.payload.pquery_name
            for row in await self._definitions.list_all(tx)
            if row.payload.pquery_name is not None
        }
        alias_names = {
            row.payload.name
            for row in await self._aliases.list_all(tx)
        }
        return sorted(definition_names | alias_names)

    async def list_aliases_matching(
        self,
        tx: StateTx,
        name_pattern: str,
    ) -> list[str]:
        pattern = re.compile(name_pattern)
        prefix = _literal_regex_prefix(name_pattern)
        if prefix:
            placeholder = _placeholder(tx, 1)
            definition_rows = await self._definitions.list_where_sql(
                tx,
                f"{_json_value(tx, 'pquery_name')} LIKE {placeholder}",
                [f"{prefix}%"],
            )
            alias_rows = await self._aliases.list_where_sql(
                tx,
                f"id LIKE {placeholder}",
                [f"{prefix}%"],
            )
        else:
            definition_rows = await self._definitions.list_all(tx)
            alias_rows = await self._aliases.list_all(tx)
        names = {
            row.payload.pquery_name
            for row in definition_rows if row.payload.pquery_name is not None
        } | {row.payload.name
             for row in alias_rows}
        return sorted(name for name in names if pattern.match(name))

    async def set_alias(self, tx: StateTx, name: str, fpid: FPID) -> None:
        await self._aliases.upsert(tx, name,
                                   PQueryAliasRecord(name=name, fpid=fpid))

    async def delete_alias(self, tx: StateTx, name: str) -> None:
        await self._aliases.delete(tx, name)

    async def resolve_fpid(
        self,
        tx: StateTx,
        pquery_id: PredictiveQueryID,
    ) -> FPID | None:
        _, fpid = PredictiveQueryID.parse(pquery_id)
        if fpid is not None:
            return fpid
        return await self.get_alias(tx, str(pquery_id))

    async def resolve_fpids(
        self,
        tx: StateTx,
        pquery_ids: Iterable[PredictiveQueryID],
    ) -> dict[PredictiveQueryID, FPID]:
        unique_ids = list(dict.fromkeys(pquery_ids))
        resolved: dict[PredictiveQueryID, FPID] = {}
        alias_ids: list[PredictiveQueryID] = []
        for pquery_id in unique_ids:
            _, fpid = PredictiveQueryID.parse(pquery_id)
            if fpid is not None:
                resolved[pquery_id] = fpid
            else:
                alias_ids.append(pquery_id)

        aliases = await self.get_aliases(tx,
                                         [str(pq_id) for pq_id in alias_ids])
        for pquery_id in alias_ids:
            fpid = aliases.get(str(pquery_id))
            if fpid is not None:
                resolved[pquery_id] = fpid
        return resolved

    async def get_definition_by_id(
        self,
        tx: StateTx,
        pquery_id: PredictiveQueryID,
    ) -> PQueryDefinition | None:
        fpid = await self.resolve_fpid(tx, pquery_id)
        return await self.get_definition(tx, fpid) if fpid else None

    async def get_definitions_by_ids(
        self,
        tx: StateTx,
        pquery_ids: Iterable[PredictiveQueryID],
    ) -> dict[PredictiveQueryID, PQueryDefinition]:
        pquery_id_to_fpid = await self.resolve_fpids(tx, pquery_ids)
        definitions_by_fpid = await self.get_definitions(
            tx, pquery_id_to_fpid.values())
        return {
            pquery_id: definitions_by_fpid[fpid]
            for pquery_id, fpid in pquery_id_to_fpid.items()
            if fpid in definitions_by_fpid
        }
