# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pydantic import ValidationError

from statedb.config.common import APIKeyID
from statedb.db.repositories import (
    JsonRepository,
    StateRow,
    _json_value,
    _placeholder,
)
from statedb.db.row_parsing import row_str
from statedb.db.runtime import StateTx
from statedb.rest.api_key import (
    APIKeyMetadataRecord,
    APIKeySecretRecord,
    APIKeyType,
)


@dataclass(frozen=True)
class APIKeyVerificationCandidate:
    metadata: APIKeyMetadataRecord
    secret: APIKeySecretRecord


def _active_type_filter(tx: StateTx, start_index: int) -> tuple[str, list[str]]:
    placeholders = [
        _placeholder(tx, start_index),
        _placeholder(tx, start_index + 1),
    ]
    return (
        f"{_json_value(tx, 'type')} IN ({', '.join(placeholders)})",
        [APIKeyType.ADMIN.value, APIKeyType.USER.value],
    )


class APIKeyRepository:
    def __init__(self) -> None:
        self._metadata = JsonRepository(
            "api_keys",
            APIKeyMetadataRecord,
            store_payload_only=True,
        )
        self._secrets = JsonRepository(
            "api_key_secrets",
            APIKeySecretRecord,
            store_payload_only=True,
        )

    async def get_metadata(
        self,
        tx: StateTx,
        api_key_id: APIKeyID,
    ) -> APIKeyMetadataRecord | None:
        try:
            return await self._metadata.get_payload(tx, api_key_id)
        except ValidationError:
            return None

    async def get_secret(
        self,
        tx: StateTx,
        api_key_id: APIKeyID,
    ) -> APIKeySecretRecord | None:
        return await self._secrets.get_payload(tx, api_key_id)

    async def list_metadata(
        self,
        tx: StateTx,
        *,
        type: APIKeyType | None = None,
        owner_user_id: str | None = None,
        search: str | None = None,
    ) -> list[APIKeyMetadataRecord]:
        filters: list[str] = []
        args: list[object] = []
        if type is not None:
            args.append(type.value)
            filters.append(
                f"{_json_value(tx, 'type')} = {_placeholder(tx, len(args))}")
        else:
            active_type_filter, active_type_args = _active_type_filter(
                tx,
                len(args) + 1)
            filters.append(active_type_filter)
            args.extend(active_type_args)
        if owner_user_id is not None:
            args.append(owner_user_id)
            filters.append(
                f"{_json_value(tx, 'owner_user_id')} = {_placeholder(tx, len(args))}"
            )
        search_term = search.strip().lower(
        ) if search and search.strip() else None
        if search_term is not None:
            search_fields = (
                "id",
                "name",
                "owner_display_name",
                "value_preview",
            )
            like_clauses: list[str] = []
            for field_name in search_fields:
                args.append(f"%{search_term}%")
                expression = "id" if field_name == "id" else _json_value(
                    tx, field_name)
                like_clauses.append(
                    f"LOWER({expression}) LIKE {_placeholder(tx, len(args))}")
            filters.append(f"({' OR '.join(like_clauses)})")
        rows = await self._metadata.list_where_sql(
            tx,
            " AND ".join(filters) or "1 = 1",
            args,
        )
        return [row.payload for row in rows]

    async def find_by_owner_name(
        self,
        tx: StateTx,
        *,
        owner_user_id: str,
        name: str,
    ) -> APIKeyMetadataRecord | None:
        rows = await tx.fetch(
            f"""
            SELECT id
            FROM api_keys
            WHERE {_json_value(tx, 'owner_user_id')} = {_placeholder(tx, 1)}
              AND {_json_value(tx, 'name')} = {_placeholder(tx, 2)}
              AND {_json_value(tx, 'type')} IN ({_placeholder(tx, 3)}, {_placeholder(tx, 4)})
            ORDER BY id
            """,
            owner_user_id,
            name,
            APIKeyType.ADMIN.value,
            APIKeyType.USER.value,
        )
        metadata_rows = await self._metadata_rows_by_id(tx, rows)
        if not metadata_rows:
            return None
        return metadata_rows[0].payload

    async def find_verification_candidate(
        self,
        tx: StateTx,
        *,
        owner_user_hash: str,
        name: str,
    ) -> APIKeyVerificationCandidate | None:
        rows = await tx.fetch(
            f"""
            SELECT id
            FROM api_keys
            WHERE {_json_value(tx, 'owner_user_hash')} = {_placeholder(tx, 1)}
              AND {_json_value(tx, 'name')} = {_placeholder(tx, 2)}
              AND {_json_value(tx, 'type')} IN ({_placeholder(tx, 3)}, {_placeholder(tx, 4)})
            ORDER BY id
            """,
            owner_user_hash,
            name,
            APIKeyType.ADMIN.value,
            APIKeyType.USER.value,
        )
        metadata_rows = await self._metadata_rows_by_id(tx, rows)
        if len(metadata_rows) != 1:
            return None
        metadata = metadata_rows[0].payload
        secret = await self.get_secret(tx, metadata.id)
        if secret is None:
            return None
        return APIKeyVerificationCandidate(metadata=metadata, secret=secret)

    async def create(
        self,
        tx: StateTx,
        *,
        metadata: APIKeyMetadataRecord,
        secret: APIKeySecretRecord,
    ) -> None:
        await self._metadata.upsert(tx, metadata.id, metadata)
        await self._secrets.upsert(tx, secret.id, secret)

    async def set_secret(
        self,
        tx: StateTx,
        secret: APIKeySecretRecord,
    ) -> None:
        await self._secrets.upsert(tx, secret.id, secret)

    async def delete(self, tx: StateTx, api_key_id: APIKeyID) -> None:
        await self._metadata.delete(tx, api_key_id)
        await self._secrets.delete(tx, api_key_id)

    async def _metadata_rows_by_id(
        self,
        tx: StateTx,
        rows: Sequence[object],
    ) -> list[StateRow[APIKeyMetadataRecord]]:
        ids = [APIKeyID(row_str(row, "id", 0)) for row in rows]
        try:
            metadata_rows = await self._metadata.list_by_ids(tx, ids)
        except ValidationError:
            return []
        return JsonRepository.rows_in_id_order(metadata_rows, ids)
