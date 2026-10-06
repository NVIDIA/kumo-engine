# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``pool.locks.PerSnapshotLocks``."""
import asyncio

import pytest  # pants: no-infer-dep

from pool.locks import PerSnapshotLocks


def test_get_returns_same_lock_for_same_snapshot():
    async def _run() -> None:
        locks = PerSnapshotLocks()
        a = await locks.get('snap-1')
        b = await locks.get('snap-1')
        assert a is b

    asyncio.run(_run())


def test_get_returns_distinct_locks_for_different_snapshots():
    async def _run() -> None:
        locks = PerSnapshotLocks()
        a = await locks.get('snap-1')
        b = await locks.get('snap-2')
        assert a is not b

    asyncio.run(_run())


def test_concurrent_create_is_deduped():
    """Two coroutines racing for a previously-unseen snapshot id must end up
    with the same ``asyncio.Lock`` instance — otherwise critical sections
    for the same snapshot could run concurrently.
    """
    async def _run() -> None:
        locks = PerSnapshotLocks()
        results = await asyncio.gather(
            *[locks.get('snap-new') for _ in range(32)])
        assert all(r is results[0] for r in results)

    asyncio.run(_run())


@pytest.mark.asyncio
async def test_acquire_serializes_same_snapshot():
    """``acquire`` for the same snapshot serializes; execution_order matches
    lock entry order, not coroutine dispatch order.
    """
    locks = PerSnapshotLocks()
    order: list = []

    async def worker(name: str, delay: float) -> None:
        async with locks.acquire('snap-1'):
            order.append(('enter', name))
            await asyncio.sleep(delay)
            order.append(('exit', name))

    await asyncio.gather(worker('a', 0.02), worker('b', 0.0))
    # No interleaving: enter/exit must pair for each name before the next
    # name enters.
    assert order == [('enter', 'a'), ('exit', 'a'), ('enter', 'b'),
                     ('exit', 'b')]


@pytest.mark.asyncio
async def test_acquire_parallel_for_different_snapshots():
    """Operations on different snapshots run in parallel — we hold one
    lock and another coroutine can still acquire a different lock.
    """
    locks = PerSnapshotLocks()
    other_acquired = asyncio.Event()

    async def holder() -> None:
        async with locks.acquire('snap-a'):
            await other_acquired.wait()

    async def other() -> None:
        async with locks.acquire('snap-b'):
            other_acquired.set()

    await asyncio.wait_for(asyncio.gather(holder(), other()), timeout=1.0)
