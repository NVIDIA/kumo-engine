# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Singlehost placement strategy: one statically pre-provisioned
worker, all placements route to the single :data:`SINGLEHOST_WORKER_UID`
queue.  ``provision()`` / ``release()`` are no-ops.
"""
from collections.abc import Iterable
from typing import Optional

from pool.dataclasses import (
    PlacementRequest,
    PoolState,
    ProvisionedHandle,
    SnapshotActivity,
)
from pool.strategy import PlacementStrategy, ReconcileDelta, RouteDecision
from pool.task_queues import SINGLEHOST_DAEMON_TQ

SINGLEHOST_WORKER_UID = 'singlehost'


class SinglehostPlacementStrategy:
    """Concrete :class:`PlacementStrategy` for singlehost deployments."""

    def __init__(self, task_queue: str = SINGLEHOST_DAEMON_TQ) -> None:
        """Route every local activity to the configured static worker queue.

        The historical default remains the unsuffixed queue for direct users
        and tests.  A deployed controller supplies its deploy-environment
        scoped name, matching the worker it starts.
        """
        self._task_queue = task_queue

    async def route(
        self,
        request: PlacementRequest,
        warm_location_uid: Optional[str],
        snapshot_size_estimate_bytes: int = 0,
        *,
        home_worker_uid: Optional[str] = None,
        home_gpu_in_use: int = 0,
        overflow_viable: bool = False,
    ) -> RouteDecision:
        # Singlehost has one static worker and no overflow pool, so the
        # sibling-fanout inputs don't apply.
        del (request, warm_location_uid, snapshot_size_estimate_bytes,
             home_worker_uid, home_gpu_in_use, overflow_viable)
        return RouteDecision(
            path='durable',
            pool_worker_uid=SINGLEHOST_WORKER_UID,
            provisioned_new_worker=False,
            reason='SINGLEHOST_STATIC',
        )

    async def provision(
        self,
        request: PlacementRequest,
        decision: RouteDecision,
    ) -> ProvisionedHandle:
        del decision
        return ProvisionedHandle(
            kind='durable',
            pool_worker_uid=SINGLEHOST_WORKER_UID,
            task_queue=_task_queue_for_activity(request.activity, self._task_queue),
        )

    async def wait_provision_ready(
        self,
        handle: ProvisionedHandle,
        job_id: str,
    ) -> ProvisionedHandle:
        del job_id  # local execution; nothing to wait for.
        return handle

    async def release(
        self,
        request: PlacementRequest,
        handle: Optional[ProvisionedHandle],
    ) -> None:
        del request, handle  # static workers; no-op cleanup.

    async def reconcile(self, state: PoolState) -> ReconcileDelta:
        del state  # static topology; no rebind / retire flows.
        return ReconcileDelta(
            observed_pool_worker_uids={SINGLEHOST_WORKER_UID},
            live_pool_worker_uids={SINGLEHOST_WORKER_UID},
        )

    async def revalidate_retired_workers(
            self, uids: Iterable[str]) -> set[str]:
        return set(uids)

    async def delete_workers(self, uids: Iterable[str]) -> set[str]:
        # static worker; nothing to tear down — trivially "deleted".
        return set(uids)


def _task_queue_for_activity(activity: SnapshotActivity, task_queue: str) -> str:
    del activity
    return task_queue


# Static Protocol-conformance check.
_: PlacementStrategy = SinglehostPlacementStrategy()
