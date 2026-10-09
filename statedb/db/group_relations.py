# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib

from statedb.config.common import GroupID
from statedb.db.runtime import StateTx


def _group_relation_lock_key(group_id: GroupID | str) -> int:
    digest = hashlib.blake2b(
        f'kumo-group-relations:{group_id}'.encode('utf-8'),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder='big', signed=True)


async def lock_group_relations(
    tx: StateTx,
    group_ids: list[GroupID | str] | tuple[GroupID | str, ...],
) -> None:
    """Serialize relation changes with group deletion on Postgres.

    SQLite write transactions already serialize writers. PostgreSQL needs an
    explicit stable lock because these JSON-backed relations do not have
    foreign keys that can lock the owning group row for us. Group locks are
    last in the global order: connector-use, identity mutation, then sorted
    group relations.
    """
    if tx.backend != 'postgres':
        return
    for group_id in sorted({str(group_id) for group_id in group_ids}):
        await tx.fetchrow(
            'SELECT pg_advisory_xact_lock($1)',
            _group_relation_lock_key(group_id),
        )
