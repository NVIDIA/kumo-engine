# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib

from statedb.config.common import IdentityID
from statedb.db.runtime import StateTx


def _identity_mutation_lock_key(identity_id: IdentityID | str) -> int:
    digest = hashlib.blake2b(
        f'kumo-identity-mutation:{identity_id}'.encode('utf-8'),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder='big', signed=True)


async def lock_identity_mutations(
    tx: StateTx,
    identity_ids: list[IdentityID | str] | tuple[IdentityID | str, ...],
) -> None:
    """Serialize one identity's update, binding replacement, and deletion.

    The global advisory-lock order is connector-use locks, then identity
    mutation locks sorted by ID, then group-relation locks sorted by ID.
    Callers must not acquire an earlier class after a later class.
    """
    if tx.backend != 'postgres':
        return
    for identity_id in sorted({str(value) for value in identity_ids}):
        await tx.fetchrow(
            'SELECT pg_advisory_xact_lock($1)',
            _identity_mutation_lock_key(identity_id),
        )
