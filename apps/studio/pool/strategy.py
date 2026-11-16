# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Placement strategy seam between deployment-agnostic controller core and
per-deployment placement (singlehost / testing / k8s). Concrete strategies
implement the methods on :class:`PlacementStrategy`."""
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Protocol, Set, runtime_checkable

from pool.dataclasses import PlacementRequest, PoolState, ProvisionedHandle

__all__ = [
    'ActivityPollerWaiter',
    'PlacementStrategy',
    'ProvisionedHandle',
    'ReconcileDelta',
    'RouteDecision',
]

ActivityPollerWaiter = Callable[[str, str, float], Awaitable[None]]


@dataclass(frozen=True)
class RouteDecision:
    """Pure-function output of :meth:`PlacementStrategy.route`."""
    path: Literal['durable', 'overflow']
    pool_worker_uid: Optional[str]
    provisioned_new_worker: bool
    reason: str
    snapshot_size_estimate_bytes: int = 0
    tier_ssd_gib: int = 0
    tier_gpu_count: int = 1


@dataclass
class ReconcileDelta:
    """Output of :meth:`PlacementStrategy.reconcile`; controller core applies
    these atomically under ``state_lock``.

    ``reconcile`` no longer deletes pool workers inline. Instead it reports
    the uids it wants torn down — ``retired_uids`` (an expected worker whose
    controller heartbeat status has reached DEAD), ``incompatible_uids`` (a
    worker whose persisted tier is no longer configured), and
    ``unexpected_uids`` (a live worker that no longer maps to any
    lease/local-copy). The controller marks all three sets ``DRAINING`` under
    ``state_lock`` (so :meth:`route` can no longer pick them) and only *then*
    calls
    :meth:`PlacementStrategy.delete_workers` outside the lock to perform the
    actual K8s deletes. This closes the teardown route-race: a worker can never
    be selected for a new placement in the window between "decided to delete"
    and "deleted".
    """
    retired_uids: Set[str] = field(default_factory=set)
    # Workers with exact persisted SSD/GPU tiers removed from current config.
    # Liveness cannot veto their replacement: a live old-tier pod is still
    # incompatible with the controller's current provisioning contract.
    incompatible_uids: Set[str] = field(default_factory=set)
    # Live workers that are no longer expected (no lease / no local copy).
    # Carried separately from ``retired_uids`` because they get no
    # orphan-snapshot / DEAD-state bookkeeping; the controller re-checks them
    # under the lock, marks the survivors DRAINING, and scrubs their
    # (lease-less) placements before delete.
    unexpected_uids: Set[str] = field(default_factory=set)
    # Every worker object observed in the strategy's external substrate during
    # this reconcile, regardless of whether it is healthy or expected. The
    # controller uses this ground truth to retain a DRAINING tombstone until a
    # later observation confirms that the external worker is gone.
    observed_pool_worker_uids: Set[str] = field(default_factory=set)
    live_pool_worker_uids: Set[str] = field(default_factory=set)
    orphan_snapshots: Dict[str, str] = field(default_factory=dict)
    live_activity_pod_uids: Set[str] = field(default_factory=set)


@runtime_checkable
class PlacementStrategy(Protocol):
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
        """Pure function of state + request; no side effects.

        ``home_worker_uid`` is the worker already assigned to this
        snapshot's placement (if one exists), regardless of whether it is
        READY yet — set when a concurrent sibling placed the snapshot first
        and is still provisioning it. ``home_gpu_in_use`` is the lease-based
        count of durable GPU activities already on that home (valid before it
        is READY), so a strategy can keep routing to the home until its GPU
        slot is full. ``overflow_viable`` is True when an L2 dataplane backup
        exists, so an eph pod can bootstrap from it. Together they let a
        strategy keep sibling fan-out on the single durable home and overflow
        only the GPU-contended excess, never minting a competing durable
        worker (see backlog #17)."""
        ...

    async def provision(
        self,
        request: PlacementRequest,
        decision: RouteDecision,
    ) -> ProvisionedHandle:
        """Realize ``decision``. Idempotent on (request, decision)."""
        ...

    async def wait_provision_ready(
        self,
        handle: ProvisionedHandle,
        job_id: str,
    ) -> ProvisionedHandle:
        """Ensure the provisioned resource is ready to serve activities.

        Called by the controller *after* persisting ``handle`` to pool_state,
        so ``release_lease`` can always clean up even if this raises.
        Returns a new handle with ``provision_state='ready'`` and enriched
        ``debug_info``; the caller must replace the stored handle in pool_state.
        Persisted activity handles may be checked again after a retry. Static
        and daemon-backed strategies return ready handles unchanged.
        """
        ...

    async def release(
        self,
        request: PlacementRequest,
        handle: Optional[ProvisionedHandle],
    ) -> None:
        """Clean up resources after a lease is released.

        Durable strategies are typically a no-op; overflow strategies delete
        the activity pod. ``handle`` may be absent when creation completed
        before persistence, so implementations must also be able to identify
        resources from ``request``.
        """
        ...

    async def reconcile(self, state: PoolState) -> ReconcileDelta:
        """On controller start + periodic sweep.

        Pure observation + reporting: it must NOT delete pool workers inline.
        It returns the uids it wants torn down in
        :attr:`ReconcileDelta.retired_uids` (expected workers whose controller
        heartbeat status has reached DEAD),
        :attr:`ReconcileDelta.incompatible_uids` (persisted tier removed from
        config), or :attr:`ReconcileDelta.unexpected_uids`; the controller
        marks them ``DRAINING`` under ``state_lock`` before calling
        :meth:`delete_workers`."""
        ...

    async def revalidate_retired_workers(
            self, uids: Iterable[str]) -> Set[str]:
        """Return candidates still safe to retire on a fresh observation."""
        ...

    async def delete_workers(self, uids: Iterable[str]) -> Set[str]:
        """Tear down the given pool-worker uids.

        Called by the controller *after* it has marked those uids
        ``DRAINING`` under ``state_lock`` (so :meth:`route` can no longer
        select them). Must be resilient per-worker: one wedged uid must not
        prevent the others from being cleaned up. Returns the uids whose
        teardown call completed. The controller retains their DRAINING
        inventory tombstones until a later reconcile no longer observes the
        external worker; a uid whose teardown failed is omitted and retried on
        the next reconcile tick.
        Static-topology strategies (singlehost / testing) have nothing to tear
        down and trivially return all input uids."""
        ...
