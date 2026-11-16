# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K8s placement strategy adapter.

This module is the thin controller-facing wrapper around Grayson's Block F
``K8sPoolWorkerManager`` contract.
"""
import dataclasses
import hashlib
import json
import math
import os
import time
import urllib.parse
import uuid
from collections.abc import Collection, Iterable
from typing import Callable, Dict, List, Mapping, Optional, Protocol

from io_utils import logging
from pool.dataclasses import (
    K8sPlacementState,
    PlacementRequest,
    PoolState,
    PoolWorkerUID,
    ProvisionedHandle,
    PvcState,
    SnapshotActivity,
    SnapshotLocalCopyState,
    WorkerInventory,
    WorkerStatus,
    expected_pool_worker_uids,
)
from pool.errors import (
    DiskgraphCapacityError,
    ExistingWorkerUnavailableError,
    HomeStaleError,
)
from pool.images import activity_image_uri
from pool.k8s_pod import activity_pod_env_vars, daemon_pod_env_vars
from pool.storage_mode import PoolStorageMode
from pool.strategy import ActivityPollerWaiter, ReconcileDelta, RouteDecision
from pool.task_queues import (
    ACTIVITY_TQ_PREFIX,
    POOL_WORKER_UID_PREFIX_DURABLE,
    POOL_WORKER_UID_PREFIX_EPHNODE,
)
from pool.task_queues import activity_queue as activity_task_queue
from pool.task_queues import daemon_dispatch_queue as daemon_task_queue
from pool.timeouts import (
    ACTIVITY_POD_PROVISION_TIMEOUT_SEC,
    ACTIVITY_POLLER_READY_TIMEOUT_SEC,
    POOL_WORKER_PROVISION_TIMEOUT_SEC,
)
from pool.worker_manager import (
    K8sPoolWorkerManager,
    PoolWorkerInfo,
    PoolWorkerNotFoundError,
    PoolWorkerNotReadyError,
    ProvisionTimeoutError,
    PvcSpec,
    ResourceSpec,
)

log = logging.getLogger(__name__)

STORAGE_MODE_ACTIVITY_ON_PVC = PoolStorageMode.ACTIVITY_ON_PVC.value
STORAGE_MODE_ACTIVITY_ON_EMPTYDIR = PoolStorageMode.ACTIVITY_ON_EMPTYDIR.value
POOL_DAEMON_COMMAND = ['python3']
POOL_DAEMON_ARGS = ['-m', 'pool.worker_main']
POOL_ACTIVITY_COMMAND = ['python3']
POOL_ACTIVITY_ARGS = ['-m', 'pool.worker_main']
# Default fraction of physical SSD usable for snapshots. The complement
# (1 − utilization) is implicitly reserved for staging tables, temp files,
# and OS overhead. Matches pool_ssd_max_utilization None default.
DEFAULT_SSD_MAX_UTILIZATION = 0.85
OVERFLOW_EPHEMERAL_STORAGE_MULTIPLIER = 1.2
_ACTIVITY_POLLER_PROBE_TIMEOUT_SEC = 1.0

# Runtime gate for routing GPU-busy overflow to the EPHEMERAL tier - DEFAULT
# ENABLED (opt-OUT). This is the SAME flag that gates the init-time ephemeral
# canary (POOL_INIT_EPHEMERAL_CANARY_ENABLED / init_canary.EPH_CANARY_ENABLED):
# the ephemeral tier is a runtime overflow spill target, and on a normal
# multi-GPU deployment that runs a dedicated ``pool-type=ephemeral`` GPU node an
# overflow eph Job schedules fine, so the tier stays ON by default. On a
# single-GPU / no-ephemeral-node deployment there is no such node for an eph Job
# to land on: routing an overflow to the eph tier there would make
# ``provision`` mint a fire-and-forget eph Job that stays permanently Pending -
# the same silent stall #29865 fixes at startup, just moved to job time. Such
# clusters opt OUT by setting POOL_INIT_EPHEMERAL_CANARY_ENABLED=false (the
# single-GPU / no-ephemeral-nodepool escape hatch): the router then never routes
# to the eph tier and the GPU-busy + L2-backup case serializes on the home /
# warm worker instead (the no-backup case already does). Parsed identically to
# init_canary.EPH_CANARY_ENABLED so one env var governs both the init canary and
# this runtime route; threaded through the constructor (not read in the routing
# hot path).
EPH_TIER_ENABLED = os.environ.get('POOL_INIT_EPHEMERAL_CANARY_ENABLED',
                                  'true').strip().lower() not in ('0', 'false',
                                                                  'no', 'off')

WorkerInventoryProvider = Callable[[], Mapping[str, WorkerInventory]]
WarmLocationProvider = Callable[[str], Optional[str]]
SnapshotSizeProvider = Callable[[str], Optional[int]]
CapacityTierProvider = Callable[[str], Optional['CapacityTier']]

# Clusters whose Alloy DaemonSet ships logs to the configured Loki service.
_LOKI_EXPLORE_CLUSTERS = {'testing-uw4'}
_GRAFANA_BASE = os.environ.get('POOL_GRAFANA_BASE_URL', '').rstrip('/')
_GRAFANA_LOKI_DATASOURCE = 'grafanacloud-logs'

# K8s truncates pod names to 63 chars: 57 chars of deployment name
# + ``-`` + 5-char random hash. Match a comfortably-short prefix
# (well under the truncation point) so the regex resolves even when
# the deployment name is long enough to be truncated.
_DAEMON_POD_NAME_PREFIX_LEN = 44


def _loki_selector(cluster: str, extra_matchers: List[str]) -> str:
    matchers = [f'cluster="{cluster}"', *extra_matchers]
    return '{' + ', '.join(matchers) + '}'


def _build_loki_explore_url(cluster: Optional[str],
                            extra_matchers: List[str]) -> Optional[str]:
    """Grafana Cloud Loki Explore deep-link for ``cluster``+matchers.

    Returns ``None`` for clusters outside :data:`_LOKI_EXPLORE_CLUSTERS`
    so callers omit the URL from ``debug_info``.
    """
    if (not _GRAFANA_BASE or cluster is None
            or cluster not in _LOKI_EXPLORE_CLUSTERS):
        return None
    expr = _loki_selector(cluster, extra_matchers)
    left = json.dumps(
        {
            'datasource': _GRAFANA_LOKI_DATASOURCE,
            'queries': [{
                'refId': 'A',
                'expr': expr
            }],
            'range': {
                'from': 'now-6h',
                'to': 'now'
            },
        },
        separators=(',', ':'),
    )
    return f'{_GRAFANA_BASE}/explore?left={urllib.parse.quote(left)}'


@dataclasses.dataclass(frozen=True, order=True)
class CapacityTier:
    """Durable worker capacity class.

    Field order is load-bearing: sorted tiers order by
    ``(tier_ssd_gib, tier_gpu_count)``, and tier selection relies on that to
    pick the smallest SSD tier that satisfies snapshot size and GPU
    requirements.
    """
    tier_ssd_gib: int
    tier_gpu_count: int = 1

    @property
    def bytes(self) -> int:
        return self.tier_ssd_gib * (1 << 30)


ManagerPvcSpecFactory = Callable[[int], PvcSpec]
ManagerResourceSpecFactory = Callable[[PlacementRequest],
                                      Optional[ResourceSpec]]
PoolWorkerUIDFactory = Callable[[], str]


class K8sPlacementContext(Protocol):
    """Controller-owned state providers consumed by the k8s strategy."""
    @property
    def worker_inventory(self) -> Mapping[str, WorkerInventory]:
        raise NotImplementedError

    def warm_location_uid(self, snapshot_id: str) -> Optional[str]:
        raise NotImplementedError

    def snapshot_size_bytes(self, snapshot_id: str) -> Optional[int]:
        raise NotImplementedError

    def reserved_size_bytes_for_worker(self, pool_worker_uid: str) -> int:
        """Sum of estimated/recorded size_bytes for RESERVED placements on
        this worker — bytes promised to in-flight ingests that haven't landed
        on SSD yet and won't appear in the worker's heartbeat pvc_free_bytes.
        """
        raise NotImplementedError

    def capacity_tier_for_worker(
            self, pool_worker_uid: str) -> Optional['CapacityTier']:
        """Persisted worker capacity tier, available before daemon heartbeat."""
        raise NotImplementedError


def durable_pool_worker_uid() -> PoolWorkerUID:
    return f'{POOL_WORKER_UID_PREFIX_DURABLE}{uuid.uuid4().hex}'


# 16 hex chars = 64 bits of entropy = collision-safe across the per-tenant
# overflow workload while keeping the uid short enough for env vars / logs.
_DETERMINISTIC_EPH_UID_HEX_LEN = 16


def deterministic_ephemeral_pool_worker_uid(snapshot_id: str,
                                            job_id: str) -> PoolWorkerUID:
    """Stable eph uid derived from ``(snapshot_id, job_id)``.

    The controller does not persist eph uids in ``pool_state.json``
    (``PoolStorageMode.ACTIVITY_ON_EMPTYDIR.is_pvc_backed == False``),
    so this string is observable only on the eph pod's
    ``POOL_WORKER_UID`` env var and on heartbeats it emits.
    Making it deterministic means retries of the same caller-side
    workflow report the same uid — useful in logs/tracing and a
    prerequisite for the deterministic Job name on the manager side
    (see :meth:`K8sPoolWorkerManager.provision_ephemeral_activity_pod`
    reclaim-always semantics for why determinism matters here).
    """
    digest = hashlib.sha256(
        f'{snapshot_id}|{job_id}'.encode('utf-8')).hexdigest()
    return (f'{POOL_WORKER_UID_PREFIX_EPHNODE}'
            f'{digest[:_DETERMINISTIC_EPH_UID_HEX_LEN]}')


def build_k8s_placement_strategy(
    manager: K8sPoolWorkerManager,
    context: K8sPlacementContext,
    *,
    capacity_tiers: list[CapacityTier],
    activity_poller_waiter: ActivityPollerWaiter,
    ssd_max_utilization: float = DEFAULT_SSD_MAX_UTILIZATION,
    manager_pvc_spec_factory: Optional[ManagerPvcSpecFactory] = None,
    activity_resources_factory: Optional[ManagerResourceSpecFactory] = None,
    pool_worker_uid_factory: Optional[PoolWorkerUIDFactory] = None,
) -> 'K8sPlacementStrategy':
    """Wire ``K8sPlacementStrategy`` to controller-owned state providers."""
    return K8sPlacementStrategy(
        manager=manager,
        worker_inventory=lambda: context.worker_inventory,
        warm_location=context.warm_location_uid,
        snapshot_size_bytes=context.snapshot_size_bytes,
        reserved_size_bytes_for_worker=context.reserved_size_bytes_for_worker,
        capacity_tier_for_worker=context.capacity_tier_for_worker,
        capacity_tiers=capacity_tiers,
        ssd_max_utilization=ssd_max_utilization,
        manager_pvc_spec_factory=manager_pvc_spec_factory,
        activity_resources_factory=activity_resources_factory,
        pool_worker_uid_factory=pool_worker_uid_factory,
        activity_poller_waiter=activity_poller_waiter,
    )


class K8sPlacementStrategy:
    """Concrete k8s implementation of the placement strategy seam.

    Controller core owns leases and lifecycle state. This class owns only k8s
    routing/provisioning choices and calls into ``K8sPoolWorkerManager`` for
    the actual pod/PVC operations.
    """
    def __init__(
        self,
        manager: K8sPoolWorkerManager,
        worker_inventory: WorkerInventoryProvider,
        warm_location: WarmLocationProvider,
        snapshot_size_bytes: SnapshotSizeProvider,
        reserved_size_bytes_for_worker: Callable[[str], int],
        *,
        capacity_tiers: list[CapacityTier],
        activity_poller_waiter: ActivityPollerWaiter,
        ssd_max_utilization: float = DEFAULT_SSD_MAX_UTILIZATION,
        manager_pvc_spec_factory: Optional[ManagerPvcSpecFactory] = None,
        activity_resources_factory: Optional[ManagerResourceSpecFactory] = None,
        pool_worker_uid_factory: Optional[PoolWorkerUIDFactory] = None,
        capacity_tier_for_worker: Optional[CapacityTierProvider] = None,
        eph_tier_enabled: bool = EPH_TIER_ENABLED,
    ) -> None:
        self._manager = manager
        self._worker_inventory = worker_inventory
        self._warm_location = warm_location
        self._snapshot_size_bytes = snapshot_size_bytes
        self._reserved_size_bytes_for_worker = reserved_size_bytes_for_worker
        self._configured_capacity_tiers: list[CapacityTier] = sorted(
            capacity_tiers)
        self._available_capacity_tiers: list[CapacityTier] = list(
            self._configured_capacity_tiers)
        self._ssd_max_utilization = ssd_max_utilization
        self._manager_pvc_spec_factory: ManagerPvcSpecFactory = (
            manager_pvc_spec_factory
            or (lambda ssd_gib: PvcSpec(size_gib=ssd_gib)))
        self._activity_resources_factory = (activity_resources_factory
                                            or (lambda request: None))
        self._pool_worker_uid_factory = (pool_worker_uid_factory
                                         or durable_pool_worker_uid)
        self._capacity_tier_provider = (capacity_tier_for_worker
                                        or (lambda _uid: None))
        self._activity_poller_waiter = activity_poller_waiter
        # Opt-out gate for routing overflow to the ephemeral tier at runtime
        # (POOL_INIT_EPHEMERAL_CANARY_ENABLED). Enabled by default; set the flag
        # to false on a single-GPU / no-ephemeral-node deployment so the router
        # does not mint an eph Job that would hang Pending there (see
        # :data:`EPH_TIER_ENABLED`).
        self._eph_tier_enabled = eph_tier_enabled

    def restrict_to_available_capacity(
        self,
        available_cache_tiers: Collection[tuple[int, int]],
        *,
        ephemeral_available: bool,
    ) -> None:
        """Apply init-canary capability results before RPC serving starts.

        Existing durable workers remain eligible through inventory-based reuse;
        only creation of workers in an unavailable configured tier is blocked.
        """
        available = set(available_cache_tiers)
        unavailable = {(tier.tier_ssd_gib, tier.tier_gpu_count)
                       for tier in self._configured_capacity_tiers} - available
        self._available_capacity_tiers = [
            tier for tier in self._configured_capacity_tiers
            if (tier.tier_ssd_gib, tier.tier_gpu_count) in available
        ]
        eph_was_enabled = self._eph_tier_enabled
        self._eph_tier_enabled = (self._eph_tier_enabled
                                  and ephemeral_available)
        if unavailable or (eph_was_enabled and not self._eph_tier_enabled):
            log.warning(
                'Init canary restricted new placement capacity: '
                'unavailable_cache_tiers=%s ephemeral_available=%s',
                sorted(unavailable), self._eph_tier_enabled)

    async def start(self) -> None:
        """Start strategy-owned background tasks (delegated to the
        worker manager's kubeconfig refresh loop). Called once during
        controller startup; safe to invoke multiple times."""
        await self._manager.start()

    async def stop(self) -> None:
        """Cancel strategy-owned background tasks. Idempotent."""
        await self._manager.stop()

    async def route(self, request: PlacementRequest,
                    warm_location_uid: Optional[str],
                    snapshot_size_estimate_bytes: int = 0, *,
                    home_worker_uid: Optional[str] = None,
                    home_gpu_in_use: int = 0,
                    overflow_viable: bool = False) -> RouteDecision:
        # Effective size: exact persisted size for WARM/COLD snapshots;
        # controller-computed estimate (source × multiplier + buffer) for MISSING.
        persisted_size = self._snapshot_size_bytes(request.snapshot_id)
        size_bytes = persisted_size or snapshot_size_estimate_bytes
        minimum_size_bytes = self._minimum_admitted_size_bytes(
            request,
            persisted_size=persisted_size,
            target_size_bytes=size_bytes,
        )
        if (warm_location_uid is not None
                and not self._worker_meets_durable_hint(warm_location_uid,
                                                        request)):
            log.info(
                'Warm worker %s for snapshot %s is below min_ssd_gib=%d or '
                'min_gpu_slots=%d; selecting another durable worker.',
                warm_location_uid,
                request.snapshot_id,
                request.resource_hint.min_ssd_gib,
                self._min_gpu_slots(request),
            )
            warm_location_uid = None
        if warm_location_uid is not None:
            if self._worker_can_run_now(warm_location_uid, request):
                return RouteDecision(
                    path='durable',
                    pool_worker_uid=warm_location_uid,
                    provisioned_new_worker=False,
                    reason='warm cache hit',
                    snapshot_size_estimate_bytes=size_bytes,
                )
            warm_inventory = self._worker_inventory().get(warm_location_uid)
            warm_status = (warm_inventory.status
                           if warm_inventory is not None else None)
            if warm_status is WorkerStatus.STALE:
                raise HomeStaleError(
                    f'Warm worker {warm_location_uid} for snapshot '
                    f'{request.snapshot_id} is STALE; wait for the FSM to '
                    'settle and re-route.')
            if warm_status in (WorkerStatus.DEAD, WorkerStatus.DRAINING):
                log.info(
                    'Warm worker %s for snapshot %s is %s; treating as no warm '
                    'location and selecting a fresh durable worker.',
                    warm_location_uid, request.snapshot_id,
                    warm_status.value if warm_status is not None else None)
            elif overflow_viable and self._eph_tier_enabled:
                return RouteDecision(
                    path='overflow',
                    pool_worker_uid=None,
                    provisioned_new_worker=False,
                    reason='warm worker lacks activity capacity',
                    snapshot_size_estimate_bytes=size_bytes,
                )
            elif overflow_viable:
                # An L2 backup exists to overflow from, but the ephemeral tier
                # is opted out (POOL_INIT_EPHEMERAL_CANARY_ENABLED=false; the
                # tier is enabled by default). Routing to eph here would mint a
                # fire-and-forget eph Job that hangs Pending forever on a
                # deployment with no ephemeral node pool - the same silent stall
                # #29865 fixes at startup, just moved to job time. So serialize
                # on the warm durable worker instead of overflowing to an eph
                # Job.
                log.info(
                    'GPU consumer %s on warm snapshot %s: worker %s lacks '
                    'activity capacity and an L2 backup exists, but the '
                    'ephemeral tier is disabled (POOL_INIT_EPHEMERAL_CANARY_'
                    'ENABLED off); serializing on the warm durable worker '
                    'instead of overflowing to an eph Job.',
                    request.activity.value, request.snapshot_id,
                    warm_location_uid)
                return RouteDecision(
                    path='durable',
                    pool_worker_uid=warm_location_uid,
                    provisioned_new_worker=False,
                    reason='ephemeral tier disabled: serialize on warm worker',
                    snapshot_size_estimate_bytes=size_bytes,
                )
            else:
                log.warning(
                    'GPU consumer %s on warm snapshot %s: worker %s lacks '
                    'activity capacity and no L2 backup exists; serializing on '
                    'the warm durable worker.', request.activity.value,
                    request.snapshot_id, warm_location_uid)
                return RouteDecision(
                    path='durable',
                    pool_worker_uid=warm_location_uid,
                    provisioned_new_worker=False,
                    reason='warm worker busy, no backup: serialize on home',
                    snapshot_size_estimate_bytes=size_bytes,
                )

        # Sibling of an existing placement (backlog #17): a concurrent caller
        # already placed this snapshot on ``home_worker_uid``. NEVER mint a
        # second durable worker here — minting is reserved for the first
        # placement (the ``placement is None`` path below), so the
        # single-durable-host invariant is structural: mint-new ⟺ first touch.
        # Route to the home unless it can't serve this activity *and* we can
        # overflow to eph instead:
        #   * non-GPU (ENSURE/GEN_TT/MATERIALIZE_*) → always co-locate on the
        #     home (a producer just coalesces onto the single ingest/restore).
        #   * GPU (TRAIN/PREDICT) with the home's GPU slot free → durable on
        #     home (runs once it warms); the slot being free is judged from
        #     leases (``home_gpu_in_use``), valid before the home is READY.
        #   * GPU with the home's slot already taken + an L2 backup → overflow
        #     to an eph pod for parallelism.
        #   * GPU, slot taken, no backup → reuse the home (serialize on it),
        #     never herd; warn since a consumer here means X was never produced.
        if home_worker_uid is not None:
            # Health gate on the assigned home BEFORE the capacity check
            # (``_home_can_serve`` stays purely about GPU-slot / co-location
            # capacity). The home's controller heartbeat status decides:
            #   * STALE → raise HomeStaleError. The home was healthy and is
            #     mid self-heal (heartbeats lapsed 60-300s); the caller waits
            #     for the FSM to settle and re-routes rather than minting a
            #     competing durable worker that would split the snapshot.
            #   * DEAD / DRAINING → treat as no home (fall through to the
            #     mint-fresh path): the home is gone / being torn down, so a
            #     fresh durable worker is the only way forward.
            #   * READY or no-inventory-yet → keep the existing home logic.
            #     A home with no inventory is the in-flight / cold-fanout case
            #     (a sibling just minted it and it hasn't reported) — NOT an
            #     unhealthy worker — so siblings still co-locate on the single
            #     home (backlog #17) instead of herding new durable workers.
            home_inventory = self._worker_inventory().get(home_worker_uid)
            home_status = (home_inventory.status
                           if home_inventory is not None else None)
            if home_status is WorkerStatus.STALE:
                raise HomeStaleError(
                    f'Home worker {home_worker_uid} for snapshot '
                    f'{request.snapshot_id} is STALE; wait for the FSM to '
                    'settle and re-route.')
            # Disk gate: if the requested size or minimum physical tier no
            # longer fits the home, bypass it and continue through normal tier
            # selection. Disk-full recovery sets min_ssd_gib above the failed
            # tier, so it cannot co-locate onto that worker again.
            # No-inventory-yet homes are exempt — their capacity is unknown
            # (in-flight / cold-fanout), so siblings still co-locate.
            home_tier = self._worker_capacity_tier(home_worker_uid)
            home_total_bytes = (home_tier.bytes if home_tier is not None else
                                home_inventory.state.pvc_total_bytes
                                if home_inventory is not None else 0)
            home_usable_bytes = home_total_bytes * self._ssd_max_utilization
            home_required_bytes = size_bytes
            if (minimum_size_bytes < size_bytes
                    and not self._available_tier_fits(
                        size_bytes,
                        self._min_gpu_slots(request),
                        request.resource_hint.min_ssd_gib,
                    )):
                home_required_bytes = minimum_size_bytes
            home_too_small = (
                (home_required_bytes > 0 and home_usable_bytes > 0
                 and home_required_bytes > home_usable_bytes)
                or (request.resource_hint.min_ssd_gib > 0
                    and home_total_bytes < request.resource_hint.min_ssd_gib *
                    (1 << 30)))
            if home_too_small:
                log.info(
                    'Home worker %s for snapshot %s cannot hold the '
                    'required %d bytes with min_ssd_gib=%d '
                    '(usable capacity %d); bypassing the home and '
                    'minting/reusing a larger durable worker.', home_worker_uid,
                    request.snapshot_id, home_required_bytes,
                    request.resource_hint.min_ssd_gib, int(home_usable_bytes))
            # DEAD / DRAINING home → fall through to the mint-fresh path below
            # (the home is gone / being torn down). READY or no-inventory-yet
            # (in-flight / cold-fanout) home → run the existing home logic.
            elif home_status not in (WorkerStatus.DEAD, WorkerStatus.DRAINING):
                if self._home_can_serve(request, home_worker_uid,
                                        home_gpu_in_use):
                    return RouteDecision(
                        path='durable',
                        pool_worker_uid=home_worker_uid,
                        provisioned_new_worker=False,
                        reason='reuse home (free slot / co-locate)',
                        snapshot_size_estimate_bytes=size_bytes,
                    )
                if overflow_viable and self._eph_tier_enabled:
                    return RouteDecision(
                        path='overflow',
                        pool_worker_uid=None,
                        provisioned_new_worker=False,
                        reason='home GPU slot busy; overflow to eph',
                        snapshot_size_estimate_bytes=size_bytes,
                    )
                if overflow_viable:
                    # An L2 backup exists to overflow from, but the ephemeral
                    # tier is opted out (POOL_INIT_EPHEMERAL_CANARY_ENABLED=
                    # false; the tier is enabled by default). Routing to eph
                    # here would mint a fire-and-forget eph Job that hangs
                    # Pending forever on a deployment with no ephemeral node
                    # pool - the same silent stall #29865 fixes at startup, just
                    # moved to job time. So serialize on the home worker
                    # instead, exactly as the no-backup case below does.
                    log.info(
                        'GPU consumer %s on snapshot %s: home %s GPU-busy with '
                        'an L2 backup, but the ephemeral tier is disabled '
                        '(POOL_INIT_EPHEMERAL_CANARY_ENABLED off); serializing '
                        'on home instead of overflowing to an eph Job.',
                        request.activity.value, request.snapshot_id,
                        home_worker_uid)
                    return RouteDecision(
                        path='durable',
                        pool_worker_uid=home_worker_uid,
                        provisioned_new_worker=False,
                        reason='ephemeral tier disabled: serialize on home',
                        snapshot_size_estimate_bytes=size_bytes,
                    )
                log.warning(
                    'GPU consumer %s on snapshot %s: home %s GPU-busy and no '
                    'L2 backup to overflow from; serializing on home. Caller '
                    'should ensure the snapshot exists before consuming.',
                    request.activity.value, request.snapshot_id,
                    home_worker_uid)
                return RouteDecision(
                    path='durable',
                    pool_worker_uid=home_worker_uid,
                    provisioned_new_worker=False,
                    reason='home GPU busy, no backup: serialize on home',
                    snapshot_size_estimate_bytes=size_bytes,
                )
            if not home_too_small:
                log.info(
                    'Home worker %s for snapshot %s is %s; treating as no '
                    'home and minting/reusing a fresh durable worker.',
                    home_worker_uid, request.snapshot_id, home_status)

        candidate = self._best_existing_durable_candidate(
            request, size_bytes=size_bytes)
        provisioned_new_worker = candidate is None
        if candidate is None:
            try:
                chosen_tier = self._choose_tier_for_snapshot(
                    size_bytes,
                    self._min_gpu_slots(request),
                    request.resource_hint.min_ssd_gib,
                    request.snapshot_id,
                    minimum_size_bytes=minimum_size_bytes,
                )
            except DiskgraphCapacityError:
                # Init-canary can make every suitable tier unavailable for
                # new workers while a previously created worker remains
                # healthy. Use it if its live free space reaches the 1.5x
                # streaming floor.
                fallback_candidate = (self._best_existing_durable_candidate(
                    request,
                    size_bytes=minimum_size_bytes,
                ) if minimum_size_bytes > 0 else None)
                if fallback_candidate is None:
                    raise
                candidate = fallback_candidate
                chosen_tier = self._worker_capacity_tier(candidate)
                provisioned_new_worker = False
            else:
                candidate = self._pool_worker_uid_factory()
        else:
            chosen_tier = self._worker_capacity_tier(candidate)
        admitted_size_bytes = size_bytes
        if chosen_tier is not None:
            admitted_size_bytes = min(
                size_bytes,
                int(chosen_tier.bytes * self._ssd_max_utilization),
            )
        using_minimum_fallback = (size_bytes > 0
                                  and admitted_size_bytes < size_bytes)
        return RouteDecision(
            path='durable',
            pool_worker_uid=candidate,
            provisioned_new_worker=provisioned_new_worker,
            reason=(
                'reuse existing durable worker' if not provisioned_new_worker
                else 'largest available tier meets minimum ingest capacity' if
                using_minimum_fallback else 'no durable worker with capacity'),
            snapshot_size_estimate_bytes=admitted_size_bytes,
            tier_ssd_gib=chosen_tier.tier_ssd_gib
            if chosen_tier is not None else 0,
            tier_gpu_count=chosen_tier.tier_gpu_count
            if chosen_tier is not None else 1,
        )

    async def provision(self, request: PlacementRequest,
                        decision: RouteDecision) -> ProvisionedHandle:
        activity = request.activity.value

        if decision.path == 'overflow':
            if not self._eph_tier_enabled:
                raise RuntimeError(
                    'Persisted overflow placement cannot be provisioned: '
                    'ephemeral capacity is unavailable for this controller '
                    'process.')
            activity_image = activity_image_uri(request.activity)
            size_bytes = self._required_snapshot_size(request.snapshot_id)
            # Deterministic uid keeps worker heartbeats stable for this
            # placement and lets retries reclaim the same Job.
            pool_worker_uid = deterministic_ephemeral_pool_worker_uid(
                snapshot_id=request.snapshot_id,
                job_id=request.job_id,
            )
            await self._manager.provision_ephemeral_activity_pod(
                job_id=request.job_id,
                snapshot_id=request.snapshot_id,
                activity=activity,
                image=activity_image,
                env=activity_pod_env_vars(
                    pool_worker_uid=pool_worker_uid, job_id=request.job_id,
                    storage_mode=PoolStorageMode.ACTIVITY_ON_EMPTYDIR,
                    require_gpu=self._needs_gpu(request.activity)),
                command=POOL_ACTIVITY_COMMAND,
                args=POOL_ACTIVITY_ARGS,
                resources=self._activity_resources_factory(request),
                ephemeral_storage_bytes=self._overflow_storage_bytes(
                    size_bytes),
            )
            return ProvisionedHandle(
                kind='overflow',
                pool_worker_uid=None,
                task_queue=activity_task_queue(request.job_id),
                provision_state='provisioning',
            )

        if decision.pool_worker_uid is None:
            raise RuntimeError(
                'Durable placement requires planned pool_worker_uid.')
        pool_worker_uid = decision.pool_worker_uid
        size_bytes = decision.snapshot_size_estimate_bytes
        chosen_tier = self._tier_from_decision(decision)
        if chosen_tier is None:
            chosen_tier = self._worker_capacity_tier(pool_worker_uid)
        inventory = self._worker_inventory().get(pool_worker_uid)
        existing_worker_is_routable = (inventory is not None
                                       and inventory.status.is_routable())
        if (chosen_tier is None and self._available_capacity_tiers
                != self._configured_capacity_tiers
                and not existing_worker_is_routable):
            raise RuntimeError(
                'Durable placement capacity tier is unknown while init-canary '
                'has restricted cache capacity; refusing to reclaim '
                f'non-routable worker {pool_worker_uid}.')
        if chosen_tier is None:
            chosen_tier = self._choose_tier_for_snapshot(
                size_bytes,
                self._min_gpu_slots(request),
                request.resource_hint.min_ssd_gib,
                request.snapshot_id,
                minimum_size_bytes=self._minimum_admitted_size_bytes(
                    request,
                    persisted_size=self._snapshot_size_bytes(
                        request.snapshot_id),
                    target_size_bytes=size_bytes,
                ),
            )
        self._validate_tier_for_request(
            request,
            chosen_tier,
            target_size_bytes=size_bytes,
        )
        existing_only = chosen_tier not in self._available_capacity_tiers
        if existing_only and not existing_worker_is_routable:
            tier_key = (chosen_tier.tier_ssd_gib, chosen_tier.tier_gpu_count)
            raise RuntimeError(
                f'Durable cache tier {tier_key} '
                'is unavailable for new provisioning in this controller '
                'process.')
        pvc_spec = self._manager_pvc_spec_factory(chosen_tier.tier_ssd_gib)
        pvc_spec.gpu_count = chosen_tier.tier_gpu_count
        try:
            info = await self._manager.provision_pool_worker(
                pool_worker_uid=pool_worker_uid,
                pvc_spec=pvc_spec,
                env=daemon_pod_env_vars(pool_worker_uid=pool_worker_uid),
                require_existing_resources=existing_only,
            )
            if info.pool_worker_uid != pool_worker_uid:
                raise RuntimeError(
                    'K8sPoolWorkerManager returned unexpected worker uid: '
                    f'planned={pool_worker_uid} actual={info.pool_worker_uid}')
            if existing_only:
                matches_current_envelope = (
                    await
                    self._manager.registered_worker_matches_current_cache_tier(
                        info, pvc_spec))
                if not matches_current_envelope:
                    raise PoolWorkerNotReadyError(
                        f'pool worker {pool_worker_uid!r} does not match the '
                        'current cache-tier envelope.')

            if request.activity == SnapshotActivity.ENSURE_SNAPSHOT:
                # Daemon already registered by provision_pool_worker above —
                # no activity Job needed, handle is immediately ready.
                return ProvisionedHandle(
                    kind='durable',
                    pool_worker_uid=pool_worker_uid,
                    task_queue=daemon_task_queue(pool_worker_uid),
                    provision_state='ready',
                    debug_info=self._daemon_debug_info(pool_worker_uid),
                )

            await self._manager.provision_activity_pod(
                pool_worker_uid=pool_worker_uid,
                job_id=request.job_id,
                snapshot_id=request.snapshot_id,
                activity=activity,
                image=activity_image_uri(request.activity),
                env=activity_pod_env_vars(
                    pool_worker_uid=pool_worker_uid, job_id=request.job_id,
                    storage_mode=PoolStorageMode.ACTIVITY_ON_PVC,
                    require_gpu=request.activity in {
                        SnapshotActivity.TRAIN,
                        SnapshotActivity.PREDICT,
                    }),
                command=POOL_ACTIVITY_COMMAND,
                args=POOL_ACTIVITY_ARGS,
                resources=self._activity_resources_factory(request),
                require_existing_resources=existing_only,
            )
        except (PoolWorkerNotFoundError, PoolWorkerNotReadyError) as exc:
            if not existing_only:
                raise
            raise ExistingWorkerUnavailableError(
                f'Existing worker {pool_worker_uid} in a canary-unavailable '
                'cache tier no longer has reusable live resources.', ) from exc

        return ProvisionedHandle(
            kind='durable',
            pool_worker_uid=pool_worker_uid,
            task_queue=activity_task_queue(request.job_id),
            provision_state='provisioning',
        )

    async def wait_provision_ready(
        self,
        handle: ProvisionedHandle,
        job_id: str,
    ) -> ProvisionedHandle:
        if not handle.task_queue.startswith(ACTIVITY_TQ_PREFIX):
            return handle
        job_name = self._manager.activity_job_name(job_id, handle.kind)
        info = await self._manager.wait_for_activity_job_pod_running(
            job_name=job_name,
            timeout=ACTIVITY_POD_PROVISION_TIMEOUT_SEC,
        )
        deadline = time.monotonic() + ACTIVITY_POLLER_READY_TIMEOUT_SEC
        last_poller_timeout: Optional[ProvisionTimeoutError] = None
        # Temporal keeps recent pollers, so only accept a match while the same
        # pod remains the current Kubernetes attempt.
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timeout_error = ProvisionTimeoutError(
                    f'activity Job {job_name!r} did not register a poller '
                    f'from its current pod within '
                    f'{ACTIVITY_POLLER_READY_TIMEOUT_SEC:.0f}s')
                raise timeout_error from last_poller_timeout
            try:
                await self._activity_poller_waiter(
                    handle.task_queue,
                    info.pod_name,
                    min(_ACTIVITY_POLLER_PROBE_TIMEOUT_SEC, remaining),
                )
            except ProvisionTimeoutError as error:
                last_poller_timeout = error
                poller_ready = False
            else:
                poller_ready = True

            remaining = deadline - time.monotonic()
            if remaining > 0:
                confirmed = await self._manager.wait_for_activity_job_pod_running(
                    job_name=job_name,
                    timeout=remaining,
                )
                if poller_ready and confirmed.pod_name == info.pod_name:
                    return dataclasses.replace(
                        handle,
                        provision_state='ready',
                        debug_info={
                            **handle.debug_info,
                            **self._activity_debug_info(info.pod_name)
                        },
                    )
                info = confirmed

    def _build_debug_info(self, pod_name: str,
                          loki_pod_matcher: str) -> Dict[str, str]:
        """Carrier for PlacementResult.debug_info.

        ``loki_pod_matcher`` is the LogQL label expression that pins
        Grafana's view to this specific pod / Deployment — Alloy
        labelmaps differ between Jobs (``service_name=<pod>``) and
        Deployments (``k8s_deployment_name=<name>``).
        """
        debug_info = {
            'pod_name': pod_name,
            'k8s_namespace': self._manager.namespace,
        }
        url = _build_loki_explore_url(
            cluster=self._manager.kube_context,
            extra_matchers=[loki_pod_matcher],
        )
        if url is not None:
            debug_info['grafana_logs_url'] = url
        return debug_info

    def _activity_debug_info(self, pod_name: str) -> Dict[str, str]:
        return self._build_debug_info(
            pod_name=pod_name,
            loki_pod_matcher=f'service_name="{pod_name}"',
        )

    def _daemon_debug_info(self, pool_worker_uid: str) -> Dict[str, str]:
        # Daemon pod names carry a per-restart suffix; Alloy preserves
        # ``service_name`` (= pod name) but the pod name truncates the
        # deployment name to fit k8s's 63-char cap. Regex-match a
        # comfortably-short prefix of the deployment name so the URL
        # works regardless of truncation or restart suffix.
        deployment_name = self._manager.deployment_name_for_uid(pool_worker_uid)
        prefix = deployment_name[:_DAEMON_POD_NAME_PREFIX_LEN]
        return self._build_debug_info(
            pod_name=deployment_name,
            loki_pod_matcher=f'service_name=~"{prefix}.+"',
        )

    async def release(self, request: PlacementRequest,
                      handle: Optional[ProvisionedHandle]) -> None:
        del handle
        await self._manager.delete_activity_pods(
            job_id=request.job_id,
            wait_for_deletion=True,
        )

    async def reconcile(self, state: PoolState) -> ReconcileDelta:
        # Retire-on-DEAD requires two independent signals: the controller's
        # heartbeat-derived view is DEAD and Kubernetes has no plausibly-live
        # or recently-transitioned daemon row. Pending/Terminating rows
        # veto retirement for the normal daemon-provision window, but age out
        # so a dead node cannot pin its data-bearing PVC forever.
        workers = await self._manager.list_pool_workers()
        # Recreate/rebind can transiently expose more than one pod row for the
        # same logical worker UID. Kubernetes list order is not a safety
        # contract, so aggregate all rows: any Registered duplicate vetoes
        # destructive retirement of the UID's shared PVC.
        observed_uids = {worker.pool_worker_uid for worker in workers}
        registered_uids = {
            worker.pool_worker_uid
            for worker in workers if worker.status == 'Registered'
        }
        retirement_veto_uids = self._retirement_veto_uids(workers)
        expected_uids = self._expected_pool_worker_uids(state)
        inventory = self._worker_inventory()
        live_uids: set[str] = set()
        retired: set[str] = set()
        incompatible: set[str] = set()

        placement_state = state.placement_state
        if isinstance(placement_state, K8sPlacementState):
            configured_tiers = {
                (tier.tier_ssd_gib, tier.tier_gpu_count)
                for tier in self._configured_capacity_tiers
            }
            incompatible = {
                uid
                for uid, pvc in placement_state.pvcs.items()
                if (pvc.tier_ssd_gib > 0 and pvc.tier_gpu_count > 0
                    and (pvc.tier_ssd_gib, pvc.tier_gpu_count)
                    not in configured_tiers)
            }

        # Toleration changes are a maintenance operation, like removing a
        # capacity tier. Reuse the controller's existing retirement path.
        # A rollout can expose two pod generations sharing one PVC: do not
        # retire that UID if any current-config incarnation still exists.
        current_toleration_uids = {
            worker.pool_worker_uid for worker in workers
            if worker.extra_tolerations_match
        }
        stale_toleration_uids = {
            worker.pool_worker_uid for worker in workers
            if not worker.extra_tolerations_match
        } - current_toleration_uids
        if stale_toleration_uids:
            log.warning(
                'Retiring pool workers after extra tolerations changed: %s. '
                'This maintenance operation deletes their activity Jobs '
                'and cache PVCs.', sorted(stale_toleration_uids))
        if isinstance(placement_state, K8sPlacementState):
            # Untracked UIDs must remain in the ordinary unexpected-worker
            # path: the controller only applies incompatibility to tracked PVCs.
            incompatible.update(stale_toleration_uids & placement_state.pvcs.keys())

        for uid in sorted(expected_uids):
            is_registered = uid in registered_uids
            if is_registered:
                live_uids.add(uid)
            inv = inventory.get(uid)
            if (inv is not None and inv.status == WorkerStatus.DEAD
                    and uid not in retirement_veto_uids):
                # Don't delete inline: report the uid as retired so the
                # controller can flip it to DRAINING + scrub its
                # lease/local/PVC state under state_lock before
                # delete_workers() tears it down
                # (teardown route-race fix). The actual delete happens in
                # delete_workers() with force_on_timeout=True.
                retired.add(uid)

        # Collect unexpected pool workers (live in K8s but no longer mapped
        # to any lease/local-copy). Don't delete inline: report them so the
        # controller flips them to DRAINING under state_lock before
        # delete_workers() tears them down (teardown route-race fix). The
        # per-worker resilience that used to live here now lives in
        # delete_workers().
        unexpected: set[str] = {
            uid
            for uid in observed_uids if uid not in expected_uids
        }
        # A prior teardown attempt can finish after the daemon pod disappears
        # or can fail after controller state has already been scrubbed. Keep
        # retrying controller-owned DRAINING entries until delete_workers()
        # succeeds and a later reconcile confirms that K8s no longer observes
        # the worker.
        unexpected.update(uid for uid, inv in inventory.items()
                          if inv.status == WorkerStatus.DRAINING)
        unexpected.difference_update(incompatible)

        return ReconcileDelta(
            observed_pool_worker_uids=observed_uids,
            live_pool_worker_uids=live_uids,
            retired_uids=retired,
            incompatible_uids=incompatible,
            unexpected_uids=unexpected,
            orphan_snapshots=self._orphan_snapshots_for_retired_workers(
                state, retired),
        )

    async def revalidate_retired_workers(self, uids: Iterable[str]) -> set[str]:
        """Return candidates still safe after a fresh K8s observation."""
        candidates = set(uids)
        if not candidates:
            return set()
        workers = await self._manager.list_pool_workers()
        return candidates - self._retirement_veto_uids(workers)

    @staticmethod
    def _retirement_veto_uids(workers: Iterable[PoolWorkerInfo]) -> set[str]:
        """UIDs with a live or still-within-grace daemon pod row."""
        now = time.time()
        vetoes: set[str] = set()
        for worker in workers:
            if worker.status in {'Registered', 'Running'}:
                vetoes.add(worker.pool_worker_uid)
                continue
            if worker.status == 'Pending':
                status_since = worker.created_at
            elif worker.status == 'Terminating':
                # An old pod can have started terminating only moments ago;
                # creation time would incorrectly expire that handoff window.
                status_since = worker.deletion_started_at
            else:
                continue
            if (status_since is None
                    or now - status_since < POOL_WORKER_PROVISION_TIMEOUT_SEC):
                vetoes.add(worker.pool_worker_uid)
        return vetoes

    async def delete_workers(self, uids: Iterable[str]) -> set[str]:
        """Tear down each uid via the worker manager; return those torn down.

        Per-worker try/except so a single wedged PVC (CSI / etcd hiccup,
        kubelet not reaping a Terminating pod that pins the PVC finalizer)
        can't take down the whole teardown pass — the bad worker is omitted
        from the returned set and retried on the next reconcile tick while the
        good cleanups go through. ``force_on_timeout=True`` because the
        controller has already committed these workers to teardown (heartbeat
        DEAD, no longer expected, or persisted tier removed from config).
        """
        deleted: set[str] = set()
        for uid in sorted(uids):
            try:
                await self._manager.delete_pool_worker(
                    uid,
                    force_on_timeout=True,
                )
                deleted.add(uid)
            except Exception:
                log.exception(
                    'delete_pool_worker failed for uid=%s; will retry on '
                    'the next reconcile tick.', uid)
        return deleted

    def _orphan_snapshots_for_retired_workers(
        self,
        state: PoolState,
        retired_uids: set[str],
    ) -> dict[str, str]:
        retired = set(retired_uids)
        return {
            local.snapshot_id: local.pool_worker_uid
            for local in state.local_copies.values()
            if (local.pool_worker_uid in retired
                and local.local_state != SnapshotLocalCopyState.EVICTED)
        }

    def _expected_pool_worker_uids(self, state: PoolState) -> set[str]:
        return expected_pool_worker_uids(state)

    def _worker_can_run_now(self, pool_worker_uid: str,
                            request: PlacementRequest) -> bool:
        inventory = self._worker_inventory().get(pool_worker_uid)
        if inventory is None or not inventory.status.is_routable():
            return False
        if not self._worker_meets_durable_hint(pool_worker_uid, request):
            return False
        if self._needs_gpu(request.activity):
            return (inventory.gpu_slots_total > inventory.gpu_slots_in_use)
        return True

    def _worker_meets_durable_hint(
        self,
        pool_worker_uid: str,
        request: PlacementRequest,
    ) -> bool:
        inventory = self._worker_inventory().get(pool_worker_uid)
        tier = self._worker_capacity_tier(pool_worker_uid)
        gpu_count = (tier.tier_gpu_count if tier is not None else
                     inventory.gpu_slots_total if inventory is not None else 1)
        total_bytes = (
            tier.bytes if tier is not None else
            inventory.state.pvc_total_bytes if inventory is not None else 0)
        return (gpu_count >= self._min_gpu_slots(request)
                and total_bytes >= request.resource_hint.min_ssd_gib *
                (1 << 30))

    def _home_can_serve(self, request: PlacementRequest, home_worker_uid: str,
                        home_gpu_in_use: int) -> bool:
        """Whether the snapshot's already-assigned home worker can serve this
        activity — judged from in-flight leases (``home_gpu_in_use``) so it
        holds even before the home is READY (unlike ``_worker_can_run_now``,
        which gates on READY + heartbeat). Non-GPU activities always co-locate;
        a GPU activity needs a free GPU slot. ``gpu_slots_total`` defaults to 1
        (POOL_GPU_CONCURRENCY) when the home hasn't registered an inventory yet.
        """
        if not self._needs_gpu(request.activity):
            return True
        inventory = self._worker_inventory().get(home_worker_uid)
        tier = (None if inventory is not None else
                self._worker_capacity_tier(home_worker_uid))
        gpu_total = (inventory.gpu_slots_total if inventory is not None else
                     max(tier.tier_gpu_count, 1) if tier is not None else 1)
        if gpu_total < self._min_gpu_slots(request):
            return False
        total_bytes = (inventory.state.pvc_total_bytes if inventory is not None
                       else tier.bytes if tier is not None else 0)
        if total_bytes < request.resource_hint.min_ssd_gib * (1 << 30):
            return False
        return home_gpu_in_use < gpu_total

    def starting_worker_can_fit(
        self,
        request: PlacementRequest,
        pvc: PvcState,
        *,
        size_bytes: int,
        gpu_slots_in_use: int,
    ) -> bool:
        """Potential capacity only: never route without a real heartbeat.

        Called under the controller state lock. Subtract reservations just as
        for live inventory so an in-flight ingest cannot promise its bytes to
        a second snapshot. The heartbeat must confirm actual free space later.
        """
        if CapacityTier(
                pvc.tier_ssd_gib,
                pvc.tier_gpu_count) not in self._configured_capacity_tiers:
            # Reconcile retires removed tiers; do not wait for their capacity.
            return False
        if pvc.tier_ssd_gib < request.resource_hint.min_ssd_gib:
            return False
        if pvc.tier_gpu_count < self._min_gpu_slots(request):
            return False
        if (self._needs_gpu(request.activity)
                and gpu_slots_in_use >= pvc.tier_gpu_count):
            return False
        usable = int(pvc.tier_ssd_gib * (1 << 30) * self._ssd_max_utilization)
        reserved = self._reserved_size_bytes_for_worker(pvc.pool_worker_uid)
        return usable > reserved and size_bytes <= usable - reserved

    def _best_existing_durable_candidate(
            self, request: PlacementRequest,
            size_bytes: int = 0) -> Optional[PoolWorkerUID]:
        best_uid: Optional[PoolWorkerUID] = None
        best_available: int = -1
        unknown_size_min_total_bytes = (request.resource_hint.min_ssd_gib *
                                        (1 << 30) if size_bytes <= 0 else 0)
        for inventory in self._worker_inventory().values():
            if not inventory.status.is_routable():
                continue
            if inventory.gpu_slots_total < self._min_gpu_slots(request):
                continue
            candidate_tier = self._capacity_tier_provider(
                inventory.pool_worker_uid)
            candidate_total_bytes = (candidate_tier.bytes
                                     if candidate_tier is not None else
                                     inventory.state.pvc_total_bytes)
            if (candidate_total_bytes
                    < request.resource_hint.min_ssd_gib * (1 << 30)):
                continue
            if (unknown_size_min_total_bytes > 0
                    and candidate_total_bytes < unknown_size_min_total_bytes):
                continue
            if self._needs_gpu(
                    request.activity) and (inventory.gpu_slots_total
                                           <= inventory.gpu_slots_in_use):
                continue
            # available = pvc_free − dead_zone − in_flight_reserved, where
            # dead_zone = pvc_total × (1 − utilization) is the fraction of the
            # PVC implicitly reserved for staging tables, tmp files, and OS.
            dead_zone = int(inventory.state.pvc_total_bytes *
                            (1 - self._ssd_max_utilization))
            reserved = self._reserved_size_bytes_for_worker(
                inventory.pool_worker_uid)
            available = max(
                0,
                inventory.state.pvc_free_bytes - dead_zone - reserved,
            )
            if size_bytes > 0 and size_bytes > available:
                continue
            if available > best_available:
                best_available = available
                best_uid = inventory.pool_worker_uid
        return best_uid

    def _minimum_admitted_size_bytes(
        self,
        request: PlacementRequest,
        *,
        persisted_size: Optional[int],
        target_size_bytes: int,
    ) -> int:
        """Minimum usable SSD capacity for this placement.

        Fresh ingests prefer the controller's normal 3x source-size target.
        If no eligible tier reaches that target, streaming ingest can proceed
        with 1.5x the source estimate. Restores never use this fallback: a
        persisted snapshot must fit at its known size.
        """
        if persisted_size:
            return target_size_bytes
        if request.source_size_bytes <= 0:
            return 0
        return (request.source_size_bytes * 3 + 1) // 2

    def _available_tier_fits(
        self,
        size_bytes: int,
        min_gpu_slots: int,
        min_ssd_gib: int,
    ) -> bool:
        return any(tier.tier_gpu_count >= min_gpu_slots and tier.tier_ssd_gib >=
                   min_ssd_gib and int(tier.bytes *
                                       self._ssd_max_utilization) >= size_bytes
                   for tier in self._available_capacity_tiers)

    def _validate_tier_for_request(
        self,
        request: PlacementRequest,
        tier: CapacityTier,
        *,
        target_size_bytes: int,
    ) -> None:
        """Reject stale or pinned decisions that bypass current constraints."""
        min_gpu_slots = self._min_gpu_slots(request)
        min_ssd_gib = request.resource_hint.min_ssd_gib
        if (tier.tier_gpu_count < min_gpu_slots
                or tier.tier_ssd_gib < min_ssd_gib):
            raise DiskgraphCapacityError(
                f'DiskGraph tier ({tier.tier_ssd_gib} GiB, '
                f'{tier.tier_gpu_count} GPUs) does not satisfy '
                f'min_ssd_gib={min_ssd_gib} and '
                f'min_gpu_slots={min_gpu_slots}.')
        persisted_size = self._snapshot_size_bytes(request.snapshot_id)
        minimum_size_bytes = self._minimum_admitted_size_bytes(
            request,
            persisted_size=persisted_size,
            target_size_bytes=target_size_bytes,
        )
        usable_bytes = int(tier.bytes * self._ssd_max_utilization)
        if minimum_size_bytes > usable_bytes:
            raise DiskgraphCapacityError(
                f'DiskGraph tier {tier.tier_ssd_gib} GiB has '
                f'{usable_bytes / (1 << 30):.1f} GiB usable SSD, below the '
                f'{minimum_size_bytes / (1 << 30):.1f} GiB minimum for '
                f'snapshot {request.snapshot_id}.')

    def _choose_tier_for_snapshot(
        self,
        size_bytes: int,
        min_gpu_slots: int,
        min_ssd_gib: int,
        snapshot_id: str,
        *,
        minimum_size_bytes: int = 0,
    ) -> CapacityTier:
        """Choose the preferred tier, or the largest safe fallback tier."""
        candidates = [
            tier for tier in self._available_capacity_tiers
            if (tier.tier_gpu_count >= min_gpu_slots
                and tier.tier_ssd_gib >= min_ssd_gib)
        ]
        if not candidates:
            configured_max_ssd_gib = max(
                (tier.tier_ssd_gib for tier in self._configured_capacity_tiers
                 if tier.tier_gpu_count >= min_gpu_slots),
                default=0,
            )
            available_max_ssd_gib = max(
                (tier.tier_ssd_gib for tier in self._available_capacity_tiers
                 if tier.tier_gpu_count >= min_gpu_slots),
                default=0,
            )
            raise DiskgraphCapacityError(
                f'No available DiskGraph tier satisfies '
                f'min_ssd_gib={min_ssd_gib} and '
                f'min_gpu_slots={min_gpu_slots} for snapshot {snapshot_id}; '
                f'the configured maximum SSD tier is '
                f'{configured_max_ssd_gib} GiB and the currently available '
                f'maximum is {available_max_ssd_gib} GiB.')
        if size_bytes <= 0:
            return max(candidates)
        for tier in candidates:
            if int(tier.bytes * self._ssd_max_utilization) >= size_bytes:
                return tier
        largest_candidate = max(candidates)
        largest_usable = int(largest_candidate.bytes *
                             self._ssd_max_utilization)
        if minimum_size_bytes > 0 and largest_usable >= minimum_size_bytes:
            return largest_candidate
        configured_max_usable = max(
            (int(tier.bytes * self._ssd_max_utilization)
             for tier in self._configured_capacity_tiers
             if (tier.tier_gpu_count >= min_gpu_slots
                 and tier.tier_ssd_gib >= min_ssd_gib)),
            default=0,
        )
        available_max_usable = max(
            (int(tier.bytes * self._ssd_max_utilization)
             for tier in candidates),
            default=0,
        )
        raise DiskgraphCapacityError(
            f'DiskGraph snapshot {snapshot_id} needs '
            f'{size_bytes / (1 << 30):.1f} GiB of usable SSD for the '
            f'preferred placement and at least '
            f'{minimum_size_bytes / (1 << 30):.1f} GiB to ingest; the '
            f'configured maximum usable capacity is '
            f'{configured_max_usable / (1 << 30):.1f} GiB and the currently '
            f'available maximum is {available_max_usable / (1 << 30):.1f} '
            f'GiB for '
            f'min_gpu_slots={min_gpu_slots} and '
            f'min_ssd_gib={min_ssd_gib}.')

    def _worker_capacity_tier(self,
                              pool_worker_uid: str) -> Optional[CapacityTier]:
        tier = self._capacity_tier_provider(pool_worker_uid)
        if tier is not None:
            return tier
        inventory = self._worker_inventory().get(pool_worker_uid)
        if inventory is not None:
            ssd_gib = inventory.state.pvc_total_bytes >> 30
            if ssd_gib > 0:
                return CapacityTier(
                    tier_ssd_gib=ssd_gib,
                    tier_gpu_count=max(inventory.gpu_slots_total, 1),
                )
        return None

    @staticmethod
    def _tier_from_decision(decision: RouteDecision) -> Optional[CapacityTier]:
        if decision.tier_ssd_gib <= 0:
            return None
        return CapacityTier(
            tier_ssd_gib=decision.tier_ssd_gib,
            tier_gpu_count=max(decision.tier_gpu_count, 1),
        )

    @staticmethod
    def _min_gpu_slots(request: PlacementRequest) -> int:
        return max(request.resource_hint.min_gpu_slots, 1)

    @staticmethod
    def _needs_gpu(activity: SnapshotActivity) -> bool:
        return activity in {SnapshotActivity.TRAIN, SnapshotActivity.PREDICT}

    def _required_snapshot_size(self, snapshot_id: str) -> int:
        size_bytes = self._snapshot_size_bytes(snapshot_id)
        if size_bytes is None or size_bytes <= 0:
            raise RuntimeError(
                'Overflow placement requires recorded snapshot size for '
                f'snapshot={snapshot_id}.')
        return size_bytes

    @staticmethod
    def _overflow_storage_bytes(size_bytes: int) -> int:
        return math.ceil(size_bytes * OVERFLOW_EPHEMERAL_STORAGE_MULTIPLIER)
