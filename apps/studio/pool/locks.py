# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-snapshot asyncio lock manager.

Operations on the same snapshot (placement, lease release, eviction) must
serialize; operations on different snapshots run in parallel. This manager
hands out one ``asyncio.Lock`` per snapshot id, lazily created on first use.

Locks are never reclaimed. The number of locks is bounded by the number of
snapshots that have ever been addressed in this process, which is small
relative to the bounded pool size.
"""
import asyncio
import contextlib
from typing import AsyncIterator, Dict


class PerSnapshotLocks:
    """Hands out one ``asyncio.Lock`` per snapshot id."""
    def __init__(self) -> None:
        self._locks: Dict[str, asyncio.Lock] = {}
        # Serializes lock creation so concurrent ``get``/``acquire`` for the
        # same new snapshot id do not create two locks.
        self._meta_lock = asyncio.Lock()

    async def get(self, snapshot_id: str) -> asyncio.Lock:
        """Return the lock for ``snapshot_id``, creating it if needed."""
        lock = self._locks.get(snapshot_id)
        if lock is not None:
            return lock
        async with self._meta_lock:
            lock = self._locks.get(snapshot_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[snapshot_id] = lock
        return lock

    @contextlib.asynccontextmanager
    async def acquire(self, snapshot_id: str) -> AsyncIterator[None]:
        """Acquire the per-snapshot lock for the duration of the context."""
        lock = await self.get(snapshot_id)
        async with lock:
            yield
