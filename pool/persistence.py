# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pool state persistence — ``pool_state.json.gz`` on the dataplane.

Whole-file write-through under a global ``asyncio.Lock``; the in-memory
``PoolState`` is authoritative while the controller is up. Mirrors the
existing REST-DB json-file pattern as a P0; Postgres migration to follow.

State is stored gzip-compressed at ``<path>.gz``. pool_state is highly
repetitive (71-char snapshot ids repeated per record, shared S3 path prefixes,
ISO timestamps), so gzip shrinks the real dg0 file ~11x (37.5 KB -> 3.3 KB).
The ``.gz`` suffix keeps the on-disk format honest (``.json`` = plain JSON,
``.json.gz`` = gzipped). ``load`` is backward-compatible with a pre-existing
uncompressed ``<path>`` and migrates it to ``<path>.gz`` on the next save.
"""
import asyncio
from typing import Callable

from io_utils import fsutil, logging
from io_utils.json import load_json_async, save_json
from pool import metrics
from pool.dataclasses import PoolState

log = logging.getLogger(__name__)


class PoolStatePersistence:
    def __init__(self, path: str,
                 empty_state_factory: Callable[[], PoolState]) -> None:
        # ``empty_state_factory`` is supplied by the caller because the
        # ``PoolState.placement_state`` variant depends on deployment mode.
        self._path = path                 # legacy uncompressed path (read-compat)
        self._gz_path = f'{path}.gz'       # canonical gzip-compressed path
        self._empty_state_factory = empty_state_factory
        self._lock = asyncio.Lock()

    @property
    def path(self) -> str:
        return self._gz_path

    async def load(self) -> PoolState:
        from_legacy = False
        try:
            async with self._lock:
                # Prefer the compressed file; fall back to a pre-existing
                # uncompressed ``.json`` (written before compression was enabled).
                if await asyncio.to_thread(fsutil.exists, self._gz_path):
                    state = await load_json_async(self._gz_path, PoolState,
                                                  compressed=True,
                                                  raise_on_missing=True)
                else:
                    state = await load_json_async(self._path, PoolState,
                                                  compressed=False,
                                                  raise_on_missing=True)
                    from_legacy = True
        except FileNotFoundError:
            # First-ever boot (or operator-initiated deletion). Persist an
            # empty state eagerly so write-permission errors surface now,
            # not on the first RPC.
            log.info(
                'pool_state not found at %s or %s; starting with empty state '
                'and persisting it.', self._gz_path, self._path)
            state = self._empty_state_factory()
            await self.save(state)
            return state

        if from_legacy:
            # One-time migration: write the compressed file and drop the stale
            # uncompressed one. Done here (not deferred to the next mutation) so
            # the on-disk format is normalized immediately on boot. ``save``
            # re-acquires the lock, so this runs after the read lock is released.
            log.info('Migrating legacy uncompressed pool_state %s -> %s.',
                     self._path, self._gz_path)
            await self.save(state)
            await asyncio.to_thread(fsutil.rm, self._path, True)
            source = self._gz_path
        else:
            source = self._gz_path

        log.info(
            'Loaded pool_state from %s: placement=%s, %d snapshots, '
            '%d dataplane_backups.', source, state.placement_state.type,
            len(state.snapshots), len(state.dataplane_backups))
        return state

    async def save(self, state: PoolState) -> None:
        # Time lock-wait + I/O: under contention the caller pays both, and the
        # save-latency scaling signal should reflect the full write-through.
        with metrics.time_state_save():
            async with self._lock:
                await save_json.coroutine(state, self._gz_path, compressed=True)
