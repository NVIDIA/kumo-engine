# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Generic, Mapping, TypeVar

from pydantic import TypeAdapter

from statedb.db.json_storage import (
    encode_timestamp,
    payload_dict_for_storage,
    payload_json_for_storage,
    utcnow,
)
from statedb.db.row_parsing import row_datetime, row_str
from statedb.db.runtime import StateTx
from statedb.db.schema import validate_identifier

PayloadT = TypeVar("PayloadT")


def _placeholder(tx: StateTx, position: int) -> str:
    return f"${position}" if tx.backend == "postgres" else "?"


def _json_value(tx: StateTx, field_name: str | tuple[str, ...]) -> str:
    fields = (field_name,) if isinstance(field_name, str) else field_name
    if not fields:
        raise ValueError("JSON field path cannot be empty")
    for field in fields:
        validate_identifier(field)
    if tx.backend == "postgres":
        if len(fields) == 1:
            return f"payload->>'{fields[0]}'"
        path = ",".join(fields)
        return f"payload#>>'{{{path}}}'"
    return f"json_extract(payload, '$.{'.'.join(fields)}')"


def _encode_json_value(tx: StateTx, value: object) -> object:
    if isinstance(value, Enum):
        value = value.value
    if tx.backend == "postgres" and isinstance(value, bool):
        return "true" if value else "false"
    return value


def _in_predicate(
    tx: StateTx,
    expression: str,
    values: Iterable[object],
    args: list[object],
) -> str:
    placeholders: list[str] = []
    for value in values:
        args.append(value)
        placeholders.append(_placeholder(tx, len(args)))
    if not placeholders:
        return "1 = 0"
    return f"{expression} IN ({', '.join(placeholders)})"


def _field_predicates(
    tx: StateTx,
    fields: Mapping[str | tuple[str, ...], object],
    args: list[object],
) -> list[str]:
    clauses: list[str] = []
    for field_name, value in fields.items():
        value_sql = _json_value(tx, field_name)
        if value is None:
            clauses.append(f"{value_sql} IS NULL")
        else:
            args.append(_encode_json_value(tx, value))
            clauses.append(f"{value_sql} = {_placeholder(tx, len(args))}")
    return clauses


@dataclass(frozen=True)
class StateRow(Generic[PayloadT]):
    id: str
    payload: PayloadT
    created_at: datetime
    updated_at: datetime


_PAYLOAD_ONLY_PHYSICAL_FIELDS = frozenset(("id", "timestamps", "created_at",
                                           "updated_at"))


class JsonRepository(Generic[PayloadT]):
    def __init__(
        self,
        table_name: str,
        payload_type: type[PayloadT],
        *,
        store_payload_only: bool = False,
    ) -> None:
        validate_identifier(table_name)
        self._table_name = table_name
        self._payload_type = payload_type
        # _payload_type is fixed per instance; build the TypeAdapter once and
        # reuse it across row loads instead of constructing one per row.
        self._payload_adapter: TypeAdapter[PayloadT] = TypeAdapter(payload_type)
        self._store_payload_only = store_payload_only

    async def get(
        self,
        tx: StateTx,
        id: str,
        *,
        for_update: bool = False,
    ) -> StateRow[PayloadT] | None:
        id_placeholder = _placeholder(tx, 1)
        lock_clause = ("FOR UPDATE"
                       if for_update and tx.backend == "postgres" else "")
        row = await tx.fetchrow(
            f"""
            SELECT payload, created_at, updated_at
            FROM {self._table_name}
            WHERE id = {id_placeholder}
            {lock_clause}
            """,
            id,
        )
        return self._row_from_db_row(id, row) if row is not None else None

    async def get_payload(self, tx: StateTx, id: str) -> PayloadT | None:
        row = await self.get(tx, id)
        return row.payload if row is not None else None

    @staticmethod
    def rows_in_id_order(
        rows: Iterable[StateRow[PayloadT]],
        ids: Iterable[object],
    ) -> list[StateRow[PayloadT]]:
        rows_by_id = {row.id: row for row in rows}
        return [
            rows_by_id[id] for value in ids if (id := str(value)) in rows_by_id
        ]

    async def list_all(self, tx: StateTx) -> list[StateRow[PayloadT]]:
        rows = await tx.fetch(
            f"""
            SELECT id, payload, created_at, updated_at
            FROM {self._table_name}
            ORDER BY id
            """)
        return [self._row_from_list_row(row) for row in rows]

    async def list_by_field(
        self,
        tx: StateTx,
        field_name: str | tuple[str, ...],
        value: object,
    ) -> list[StateRow[PayloadT]]:
        value_placeholder = _placeholder(tx, 1)
        rows = await tx.fetch(
            f"""
            SELECT id, payload, created_at, updated_at
            FROM {self._table_name}
            WHERE {_json_value(tx, field_name)} = {value_placeholder}
            ORDER BY id
            """,
            _encode_json_value(tx, value),
        )
        return [self._row_from_list_row(row) for row in rows]

    async def list_by_fields(
        self,
        tx: StateTx,
        fields: Mapping[str | tuple[str, ...], object],
        *,
        not_null_fields: Sequence[str | tuple[str, ...]] = (),
        order_by: str = "id",
    ) -> list[StateRow[PayloadT]]:
        args: list[object] = []
        clauses = _field_predicates(tx, fields, args)
        for field_name in not_null_fields:
            clauses.append(f"{_json_value(tx, field_name)} IS NOT NULL")
        return await self.list_where_sql(
            tx,
            " AND ".join(clauses) if clauses else "1 = 1",
            args,
            order_by=order_by,
        )

    async def list_by_ids(
        self,
        tx: StateTx,
        ids: Sequence[str],
        *,
        fields: Mapping[str | tuple[str, ...], object] | None = None,
        order_by: str = "id",
    ) -> list[StateRow[PayloadT]]:
        if not ids:
            return []
        args: list[object] = []
        clauses = [_in_predicate(tx, "id", ids, args)]
        clauses.extend(_field_predicates(tx, fields or {}, args))
        return await self.list_where_sql(
            tx,
            " AND ".join(clauses),
            args,
            order_by=order_by,
        )

    async def list_where_sql(
        self,
        tx: StateTx,
        where_sql: str,
        args: Sequence[object] = (),
        *,
        order_by: str = "id",
        limit: int | None = None,
    ) -> list[StateRow[PayloadT]]:
        if order_by != "id":
            validate_identifier(order_by)
        limit_sql = ""
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            limit_sql = f" LIMIT {limit}"
        rows = await tx.fetch(
            f"""
            SELECT id, payload, created_at, updated_at
            FROM {self._table_name}
            WHERE {where_sql}
            ORDER BY {order_by}
            {limit_sql}
            """,
            *args,
        )
        return [self._row_from_list_row(row) for row in rows]

    async def exists_by_fields(
        self,
        tx: StateTx,
        fields: Mapping[str | tuple[str, ...], object],
        *,
        exclude_id: str | None = None,
    ) -> bool:
        args: list[object] = []
        clauses = _field_predicates(tx, fields, args)
        if exclude_id is not None:
            args.append(exclude_id)
            clauses.append(f"id <> {_placeholder(tx, len(args))}")
        row = await tx.fetchrow(
            f"""
            SELECT id
            FROM {self._table_name}
            WHERE {" AND ".join(clauses)}
            LIMIT 1
            """,
            *args,
        )
        return row is not None

    async def find_one_by_field(
        self,
        tx: StateTx,
        field_name: str | tuple[str, ...],
        value: object,
    ) -> StateRow[PayloadT] | None:
        rows = await self.list_by_field(tx, field_name, value)
        if len(rows) > 1:
            raise ValueError(
                f"Expected at most one {self._table_name} row for "
                f"{field_name}={value!r}; found {len(rows)}")
        return rows[0] if rows else None

    async def find_one_by_fields(
        self,
        tx: StateTx,
        fields: Mapping[str | tuple[str, ...], object],
    ) -> StateRow[PayloadT] | None:
        rows = await self.list_by_fields(tx, fields)
        if len(rows) > 1:
            raise ValueError(
                f"Expected at most one {self._table_name} row for "
                f"{dict(fields)!r}; found {len(rows)}")
        return rows[0] if rows else None

    async def upsert(
        self,
        tx: StateTx,
        id: str,
        payload: PayloadT,
        *,
        extra_fields: Mapping[str, object] | None = None,
    ) -> None:
        payload_json = payload_json_for_storage(
            self._payload_dict_for_storage(payload, extra_fields))
        created_at_value, updated_at_value = self._timestamps_for_storage(
            payload)
        created_at = encode_timestamp(tx, created_at_value)
        updated_at = encode_timestamp(tx, updated_at_value)
        if tx.backend == "postgres":
            await tx.execute(
                f"""
                INSERT INTO {self._table_name}
                    (id, payload, created_at, updated_at)
                VALUES ($1, $2::jsonb, $3, $4)
                ON CONFLICT (id) DO UPDATE SET
                    payload = EXCLUDED.payload,
                    updated_at = EXCLUDED.updated_at
                """,
                id,
                payload_json,
                created_at,
                updated_at,
            )
            return
        await tx.execute(
            f"""
            INSERT INTO {self._table_name}
                (id, payload, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                payload = excluded.payload,
                updated_at = excluded.updated_at
            """,
            id,
            payload_json,
            created_at,
            updated_at,
        )

    async def insert_if_absent(
        self,
        tx: StateTx,
        id: str,
        payload: PayloadT,
    ) -> bool:
        """Atomically insert without mutating an existing immutable row."""
        payload_json = payload_json_for_storage(
            self._payload_dict_for_storage(payload, None))
        created_at_value, updated_at_value = self._timestamps_for_storage(
            payload)
        created_at = encode_timestamp(tx, created_at_value)
        updated_at = encode_timestamp(tx, updated_at_value)
        if tx.backend == "postgres":
            row = await tx.fetchrow(
                f"""
                INSERT INTO {self._table_name}
                    (id, payload, created_at, updated_at)
                VALUES ($1, $2::jsonb, $3, $4)
                ON CONFLICT (id) DO NOTHING
                RETURNING id
                """,
                id,
                payload_json,
                created_at,
                updated_at,
            )
            return row is not None
        row = await tx.fetchrow(
            f"""
            INSERT INTO {self._table_name}
                (id, payload, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (id) DO NOTHING
            RETURNING id
            """,
            id,
            payload_json,
            created_at,
            updated_at,
        )
        return row is not None

    async def delete(self, tx: StateTx, id: str) -> None:
        id_placeholder = _placeholder(tx, 1)
        await tx.execute(
            f"DELETE FROM {self._table_name} WHERE id = {id_placeholder}",
            id,
        )

    def _row_from_db_row(self, id: str,
                         row: object) -> StateRow[PayloadT]:
        created_at = row_datetime(row, "created_at", 1)
        updated_at = row_datetime(row, "updated_at", 2)
        return StateRow(
            id=id,
            payload=self._payload_from_json(
                id,
                row_str(row, "payload", 0),
                created_at,
                updated_at,
            ),
            created_at=created_at,
            updated_at=updated_at,
        )

    def _row_from_list_row(self, row: object) -> StateRow[PayloadT]:
        row_id = row_str(row, "id", 0)
        payload = row_str(row, "payload", 1)
        created_at = row_datetime(row, "created_at", 2)
        updated_at = row_datetime(row, "updated_at", 3)
        return StateRow(
            id=row_id,
            payload=self._payload_from_json(
                row_id,
                payload,
                created_at,
                updated_at,
            ),
            created_at=created_at,
            updated_at=updated_at,
        )

    def _payload_dict_for_storage(
        self,
        payload: PayloadT,
        extra_fields: Mapping[str, object] | None,
    ) -> dict[str, object]:
        return payload_dict_for_storage(
            payload,
            extra_fields=extra_fields,
            omit_fields=(
                _PAYLOAD_ONLY_PHYSICAL_FIELDS
                if self._store_payload_only else ()
            ),
        )

    def _timestamps_for_storage(
        self,
        payload: PayloadT,
    ) -> tuple[datetime, datetime]:
        timestamps = (
            getattr(payload, "timestamps", None) if self._store_payload_only else None
        )
        created_at = getattr(timestamps, "created_at", None)
        updated_at = getattr(timestamps, "updated_at", None)
        if isinstance(created_at, datetime) and isinstance(updated_at, datetime):
            return created_at, updated_at
        now = utcnow()
        return now, now

    def _payload_from_json(
        self,
        row_id: str,
        payload: str,
        created_at: datetime,
        updated_at: datetime,
    ) -> PayloadT:
        payload_dict = json.loads(payload)
        if self._store_payload_only:
            payload_dict.setdefault("id", row_id)
            payload_dict.setdefault(
                "timestamps",
                {
                    "created_at": created_at,
                    "updated_at": updated_at,
                },
            )
        return self._payload_adapter.validate_python(payload_dict)
