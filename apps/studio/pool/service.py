# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pool controller AutoRPC service.

The server-side ``PoolController`` is a single-replica service that
manages diskgraph snapshot placement, leases, per-node PVC lifecycle,
and node inventory. Callers reach it via
``pool.client.get_controller_client()``.
"""
import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    TYPE_CHECKING,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    Optional,
    Tuple,
)

from temporalio.api.enums.v1 import TaskQueueType
from temporalio.api.taskqueue.v1 import TaskQueue
from temporalio.api.workflowservice.v1 import DescribeTaskQueueRequest
from temporalio.client import WorkflowExecutionStatus
from temporalio.service import RPCError

from common_utils.time import now_utc
from io_utils import fsutil, logging
from pool import metrics
from pool.daemon_dispatch import build_temporal_snapshot_deleter
from pool.dataclasses import (
    DataplaneBackup,
    JobID,
    K8sPlacementState,
    PlacementRequest,
    PlacementResult,
    PlacementStatus,
    PoolState,
    PoolStats,
    PoolStatsFilter,
    PoolWorkerStats,
    ProvisionedHandle,
    PvcState,
    ResourceHint,
    SinglehostPlacementState,
    SnapshotActivity,
    SnapshotComplianceRecord,
    SnapshotID,
    SnapshotLifecyclePolicy,
    SnapshotLocalCopyRecord,
    SnapshotLocalCopyState,
    SnapshotPlacement,
    SnapshotState,
    WorkerEvent,
    WorkerEventType,
    WorkerInventory,
    WorkerStateSnapshot,
    WorkerStatus,
    WorkerStatusReport,
    expected_pool_worker_uids,
)
from pool.errors import (
    ExistingWorkerUnavailableError,
    InsufficientDataplaneCapacityError,
    SnapshotEvictionInProgressError,
)
from pool.images import daemon_image_uri
from pool.k8s_strategy import K8sPlacementStrategy
from pool.locks import PerSnapshotLocks
from pool.persistence import PoolStatePersistence
from pool.storage_mode import PoolStorageMode
from pool.strategy import (
    ActivityPollerWaiter,
    PlacementStrategy,
    ReconcileDelta,
    RouteDecision,
)
from pool.strategy_singlehost import SinglehostPlacementStrategy
from pool.task_queues import activity_worker_identity
from pool.timeouts import POOL_WORKER_STARTUP_GRACE_SEC
from pool.worker_manager import PvcSpec, ResourceSpec, SecretKeyRef
from pool.worker_manager.types import ProvisionTimeoutError
from service_framework._raw_deploy_config import (
    DiskgraphConfig,
    DiskgraphNodeType,
    K8sPoolStrategyConfig,
    PoolPodResources,
)
from service_framework.deploy_env_config import DeployEnvConfig, deploy_env
from statedb.config.common import DataSnapshot
from statedb.utils.errors import DiskgraphError
from temporalio_commons.auto_rpc import auto_rpc_service, rpc_handler
from temporalio_commons.auto_rpc_exception import is_retryable
from temporalio_commons.temporal_client import TemporalClient

if TYPE_CHECKING:
    from pool.k8s_strategy import CapacityTier
    from pool.worker_manager import K8sPoolWorkerManager, PoolWorkerInfo

log = logging.getLogger(__name__)

# Periodic heartbeat cadence and READY -> STALE -> DEAD grace windows.
POOL_HEARTBEAT_INTERVAL_SECS = 30
WORKER_STALE_THRESHOLD_SECS = 60
# Heartbeat age past which an expected worker is declared DEAD and retired by
# reconcile (no relaunch, no node-death/self-heal discrimination — see
# K8sPlacementStrategy.reconcile). Sized to comfortably exceed the worst-case
# legitimate self-heal of a daemon pod (full recreate: terminationGrace +
# reschedule + warm-image start + daemon init + local-SSD index scan + first
# registration), so a pod/container crash on a live node clears the STALE
# window via a fresh heartbeat before this fires and is a no-op for retirement.
# Routing stops at STALE (60s) regardless; only retirement waits the full
# window. Replaces the old 120s value + the separate 4-min reconcile rebind
# timer (both removed with the rebind machinery).
WORKER_DEAD_THRESHOLD_SECS = 300
CONTROLLER_MAINTENANCE_INTERVAL_SECS = POOL_HEARTBEAT_INTERVAL_SECS
LEASE_DEFENSIVE_SWEEP_INTERVAL_SECS = 5 * 60
REACTIVE_EVICTION_WINDOW_SECS = 5 * 60
# Grace window before terminal tombstone records (EVICTED local copies,
# deleted compliance records) are GC'd from pool_state. Long enough to stay
# visible on the dashboard / for late events to settle; short enough that
# pool_state tracks the active working set, not cumulative history.
TOMBSTONE_GC_GRACE_SECS = 15 * 60
DEFAULT_K8S_POOL_STORAGE_CLASS = 'topolvm-ssd'

DataplaneDeleter = Callable[[DataplaneBackup], Awaitable[None]]
LocalSnapshotDeleter = Callable[[str, str], Awaitable[bool]]
UtcClock = Callable[[], datetime]
MonotonicClock = Callable[[], float]


class WorkflowLeaseState(str, Enum):
    """Result of looking up the workflow that owns a lease.

    Used by the periodic ``_dangling_lease_sweep_once`` to decide
    whether to force-release a lease the caller can no longer release
    itself (e.g. workflow hit ``run_timeout`` and ``finally`` never ran).
    """
    RUNNING = 'RUNNING'
    TERMINAL = 'TERMINAL'
    NOT_FOUND = 'NOT_FOUND'


WorkflowStatusChecker = Callable[[str], Awaitable[WorkflowLeaseState]]


async def _noop_workflow_status_checker(workflow_id: str) -> WorkflowLeaseState:
    del workflow_id
    return WorkflowLeaseState.RUNNING


def build_temporal_workflow_status_checker(
    config: DeployEnvConfig,
    temporal_client: Optional[TemporalClient] = None,
) -> WorkflowStatusChecker:
    """Temporal-backed lookup mapping workflow_id -> ``WorkflowLeaseState``."""
    temporal = temporal_client or TemporalClient(config)

    async def check(workflow_id: str) -> WorkflowLeaseState:
        desc = await temporal.get_workflow_desc(workflow_id)
        if desc is None:
            return WorkflowLeaseState.NOT_FOUND
        if desc.status == WorkflowExecutionStatus.RUNNING:
            return WorkflowLeaseState.RUNNING
        return WorkflowLeaseState.TERMINAL

    return check


def build_temporal_activity_poller_waiter(
    config: DeployEnvConfig,
    temporal_client: Optional[TemporalClient] = None,
    *,
    poll_interval_secs: float = 1.0,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> ActivityPollerWaiter:
    """Wait for the selected pod's poller, ignoring stale pollers on its queue."""
    temporal = temporal_client or TemporalClient(config)

    async def wait(task_queue: str, pod_name: str, timeout: float) -> None:
        expected_identity = activity_worker_identity(pod_name,
                                                     config.instance_id)
        fully_qualified_queue = config.task_queue(task_queue)
        deadline = monotonic_fn() + timeout
        last_rpc_error: Optional[str] = None
        while monotonic_fn() < deadline:
            try:
                client = await temporal.get()
                response = await client.workflow_service.describe_task_queue(
                    DescribeTaskQueueRequest(
                        namespace=config.temporal_namespace,
                        task_queue=TaskQueue(name=fully_qualified_queue),
                        task_queue_type=(
                            TaskQueueType.TASK_QUEUE_TYPE_ACTIVITY),
                        report_pollers=True,
                    ))
                if any(poller.identity == expected_identity
                       for poller in response.pollers):
                    return
                last_rpc_error = None
            except RPCError as error:
                if not is_retryable(error):
                    raise
                last_rpc_error = getattr(error.status, 'name',
                                         str(error.status))
            remaining = deadline - monotonic_fn()
            if remaining > 0:
                await asyncio.sleep(min(poll_interval_secs, remaining))
        suffix = (f'; last Temporal RPC status={last_rpc_error}'
                  if last_rpc_error else '')
        raise ProvisionTimeoutError(
            f'activity pod {pod_name!r} reached Running but did not register '
            f'its Temporal ACTIVITY poller on {fully_qualified_queue!r} '
            f'within {timeout:.0f}s{suffix}')

    return wait


async def _default_dataplane_deleter(backup: DataplaneBackup) -> None:
    await asyncio.to_thread(fsutil.rm, backup.dataplane_path)


async def _missing_local_snapshot_deleter(snapshot_id: str,
                                          pool_worker_uid: str) -> bool:
    del snapshot_id, pool_worker_uid
    raise NotImplementedError(
        'Local snapshot eviction requested, but no daemon local-delete '
        'dispatcher is configured.')


def _canonicalize_snapshot_id(snapshot_id: SnapshotID) -> SnapshotID:
    try:
        return DataSnapshot.GRAPH.canonicalize(snapshot_id)
    except ValueError:
        return snapshot_id


def _placement_key(snapshot_id: str, job_id: str) -> str:
    return f'{snapshot_id}:{job_id}'


def _local_copy_key(snapshot_id: str, pool_worker_uid: str) -> str:
    return f'{snapshot_id}:{pool_worker_uid}'


@auto_rpc_service
class PoolController:
    """Single-replica pool controller service.

    Construct via :func:`create_pool_controller`, which loads persisted state
    before returning a ready-to-run instance. Direct construction is
    supported for tests that want to inject a pre-built state.

    **Atomicity contract for state-mutating handlers:**

    Placement persists a strategy plan and logical lease before external
    provisioning. Retries that find an open but not-yet-provisioned placement
    finish the same plan instead of routing again. Snapshot materialization is
    sequenced by the calling workflow on the returned task queue.

        Other simple state mutations use mutate-then-persist under the
        controller state lock. The global write lock inside
        ``PoolStatePersistence`` serializes full-file dataplane writes across
        different snapshots.

    The :meth:`_mutate_and_persist` helper encapsulates the common simple
    mutation path.
    """
    def __init__(
        self,
        state: PoolState,
        persistence: PoolStatePersistence,
        strategy: Optional[PlacementStrategy] = None,
        workflow_status_checker: Optional[WorkflowStatusChecker] = None,
        dataplane_deleter: Optional[DataplaneDeleter] = None,
        local_snapshot_deleter: Optional[LocalSnapshotDeleter] = None,
        data_plane_storage_root: Optional[str] = None,
        lifecycle_policy: Optional[SnapshotLifecyclePolicy] = None,
        now_utc_fn: Optional[UtcClock] = None,
        time_fn: Optional[MonotonicClock] = None,
        lease_sweep_interval_secs: float = LEASE_DEFENSIVE_SWEEP_INTERVAL_SECS,
        worker_stale_threshold_secs: float = WORKER_STALE_THRESHOLD_SECS,
        worker_dead_threshold_secs: float = WORKER_DEAD_THRESHOLD_SECS,
        maintenance_interval_secs: float = CONTROLLER_MAINTENANCE_INTERVAL_SECS,
        tombstone_gc_grace_secs: float = TOMBSTONE_GC_GRACE_SECS,
        ingest_space_multiplier: int = 4,
        snapshot_estimate_buffer_bytes: int = 0,
        worker_startup_grace_secs: float = POOL_WORKER_STARTUP_GRACE_SEC,
    ) -> None:
        self._state = state
        self._persistence = persistence
        self._strategy = strategy
        self._workflow_status_checker = (workflow_status_checker
                                         if workflow_status_checker is not None
                                         else _noop_workflow_status_checker)
        self._dataplane_deleter = (dataplane_deleter if dataplane_deleter
                                   is not None else _default_dataplane_deleter)
        self._local_snapshot_deleter = (local_snapshot_deleter
                                        if local_snapshot_deleter is not None
                                        else _missing_local_snapshot_deleter)
        self._data_plane_storage_root = data_plane_storage_root
        self._lifecycle_policy = lifecycle_policy or SnapshotLifecyclePolicy()
        if self._lifecycle_policy.min_ssd_idle_secs < 0:
            raise ValueError('min_ssd_idle_secs must be non-negative')
        if self._lifecycle_policy.max_ssd_idle_secs < 0:
            raise ValueError('max_ssd_idle_secs must be non-negative')
        if self._lifecycle_policy.max_dataplane_retention_secs < 0:
            raise ValueError(
                'max_dataplane_retention_secs must be non-negative')
        if (self._lifecycle_policy.min_ssd_idle_secs
                > self._lifecycle_policy.max_ssd_idle_secs):
            raise ValueError('min_ssd_idle_secs must be <= max_ssd_idle_secs')
        if (self._lifecycle_policy.max_ssd_idle_secs
                > self._lifecycle_policy.max_dataplane_retention_secs):
            raise ValueError(
                'max_ssd_idle_secs must be <= '
                'max_dataplane_retention_secs (L2 compliance sweep waits '
                'for L1 to drain; violating this ordering causes L2 deletion '
                'to block indefinitely and silently violate the dataplane '
                'retention SLA)')
        # ssd_max_utilization range (0, 1] is enforced by the dataclass field.
        self._now_utc = now_utc_fn if now_utc_fn is not None else now_utc
        self._time = time_fn if time_fn is not None else time.time
        self._lease_sweep_interval_secs = lease_sweep_interval_secs
        self._worker_stale_threshold_secs = worker_stale_threshold_secs
        self._worker_dead_threshold_secs = worker_dead_threshold_secs
        self._maintenance_interval_secs = maintenance_interval_secs
        self._tombstone_gc_grace_secs = tombstone_gc_grace_secs
        self._ingest_space_multiplier = ingest_space_multiplier
        self._snapshot_estimate_buffer_bytes = snapshot_estimate_buffer_bytes
        self._state_lock = asyncio.Lock()
        self._worker_startup_grace_secs = worker_startup_grace_secs
        # Set-and-replace, never clear: a heartbeat between checking capacity
        # and awaiting this captured event must not be lost.
        self._worker_capacity_changed = asyncio.Event()
        self._snapshot_locks = PerSnapshotLocks()
        self._placement_provision_locks = PerSnapshotLocks()
        self._startup_expected_worker_uids = expected_pool_worker_uids(state)
        # Rebuilt from daemon status reports on startup; not persisted.
        # Empty at boot time — the first ~30s after startup will return
        # errors for ``place_snapshot`` until enough reports arrive. See
        # design doc: callers are Temporal workflows and retry transparently.
        #
        # TODO(startup-reconcile): consider a proactive startup-poll — the k8s
        # strategy knows which workers had PVCs at last shutdown (via
        # ``self._state.placement_state.pvcs``); fan out a
        # ``report_status`` pull to each one's daemon task queue. Shrinks
        # the "empty inventory" window from one heartbeat-interval to a
        # single AutoRPC round-trip and stops bouncing early placement
        # requests back to the caller. Orthogonal to the heartbeat
        # protocol — still works purely as a fallback if some workers
        # don't respond to the pull. Strategy-owned (k8s only);
        # singlehost has no worker set to poll.
        self._worker_inventory: Dict[str, WorkerInventory] = {}
        # Teardown calls can return before a worker observed at the start of the
        # reconcile disappears from Kubernetes. Keep its DRAINING inventory
        # tombstone until a later reconcile confirms absence, closing the late
        # heartbeat resurrection window without persisting runtime inventory.
        self._delete_succeeded_pending_absence: set[str] = set()
        # Fast mirror of persisted ``SnapshotPlacement.lease_activities`` for
        # leases created in this process. Restart recovery reads persisted
        # activity metadata so durable GPU reservations do not under-count.
        self._lease_activity: Dict[Tuple[str, str], SnapshotActivity] = {}
        self._maintenance_task: Optional[asyncio.Task[None]] = None
        # Persisted suspect timestamps prevent restart loops from resetting a
        # dead worker's countdown. This boot-local floor still gives healthy
        # daemons time to heartbeat once RPC serving begins; canary/startup
        # time must never consume that opportunity.
        self._silent_worker_dead_not_before: Optional[float] = None
        self._last_dangling_lease_sweep_at: Optional[float] = None
        self._reactive_eviction_requested_at: Optional[float] = None

    # --- Accessors used by tests and maintenance helpers; not exposed via
    # AutoRPC. ---

    @property
    def state(self) -> PoolState:
        return self._state

    @property
    def worker_inventory(self) -> Dict[str, WorkerInventory]:
        return self._worker_inventory

    def _set_placement_strategy(self, strategy: PlacementStrategy) -> None:
        """Install the deployment-specific placement strategy.

        ``create_pool_controller`` uses this after constructing the controller
        so strategy factories can close over controller-owned state providers
        such as worker inventory and snapshot size.
        """
        self._strategy = strategy

    def _snapshot_size_bytes(self, snapshot_id: str) -> Optional[int]:
        """Recorded size from the placement; if absent (COLD: evicted, lives on
        the dataplane), the size persisted on the L2 backup. None if neither."""
        placement = self._state.snapshots.get(snapshot_id)
        if placement is not None and placement.size_bytes > 0:
            return placement.size_bytes
        backup = self._state.dataplane_backups.get(snapshot_id)
        if backup is not None and backup.size_bytes > 0:
            return backup.size_bytes
        return None

    async def _ensure_backup_size(self, snapshot_id: str) -> None:
        """Backfill a COLD snapshot's size when unrecorded (a legacy backup
        written before sizes were stored): scan its dataplane dir and persist it
        onto the backup. Off the placement locks — ``fsutil.dirsize`` blocks."""
        if self._snapshot_size_bytes(snapshot_id) is not None:
            return
        backup = self._state.dataplane_backups.get(snapshot_id)
        if backup is None:
            return
        size_bytes = await asyncio.to_thread(fsutil.dirsize,
                                             backup.dataplane_path)
        if size_bytes <= 0:
            return

        def _backfill_size(state: PoolState) -> None:
            current = state.dataplane_backups.get(snapshot_id)
            if current is not None and current.size_bytes <= 0:
                current.size_bytes = size_bytes

        await self._mutate_and_persist(snapshot_id, _backfill_size)

    # --- Internal helpers for placement, lifecycle, and reconciliation. ---

    def _warm_location_uid(self, snapshot_id: str) -> Optional[str]:
        # Worker uid holding a usable local copy (WARM or
        # EVICTION_CANDIDATE on a READY placement), else None.
        placement = self._state.snapshots.get(snapshot_id)
        if placement is None or placement.state != SnapshotState.READY:
            return None
        local = self._state.local_copies.get(
            _local_copy_key(snapshot_id, placement.worker))
        if (local is not None and local.local_state not in {
                SnapshotLocalCopyState.WARM,
                SnapshotLocalCopyState.EVICTION_CANDIDATE,
        }):
            return None
        return placement.worker

    def _raise_if_current_copy_evicting(
            self, snapshot_id: str,
            placement: Optional[SnapshotPlacement]) -> None:
        """Block placement while its current local copy is being deleted."""
        if placement is None:
            return
        local = self._state.local_copies.get(
            _local_copy_key(snapshot_id, placement.worker))
        if (local is not None
                and local.local_state == SnapshotLocalCopyState.EVICTING):
            raise SnapshotEvictionInProgressError(
                f'Snapshot {snapshot_id} is being evicted from worker '
                f'{placement.worker}; retry placement after deletion '
                'finishes.')

    def _validate_l2_backup(self, snapshot_id: str) -> None:
        backup = self._state.dataplane_backups.get(snapshot_id)
        if backup is None:
            raise AssertionError(
                f'L2 backup check requested for {snapshot_id}, but no '
                'dataplane backup is recorded.')

    def _lookup_lease_handle(self, placement: Optional[SnapshotPlacement],
                             job_id: JobID) -> Optional[ProvisionedHandle]:
        # Handle for an existing lease, looked up by recorded path.
        # No cross-map fallback (avoids cross-shard contamination).
        if placement is None or job_id not in placement.leases:
            return None
        path = placement.lease_paths.get(job_id)
        if path == 'durable':
            return placement.durable_handles.get(job_id)
        if path == 'overflow':
            return placement.overflow_handles.get(job_id)
        return None

    def _handle_worker_routable(
        self,
        handle: ProvisionedHandle,
        request: PlacementRequest,
    ) -> bool:
        """Whether the worker backing ``handle`` is still serviceable for the
        ``place_snapshot`` fast-path short-circuit.

        A ready handle is re-served without re-routing UNLESS the controller's
        heartbeat-derived inventory shows its worker has gone non-READY
        (STALE/DEAD/DRAINING). Treated as serviceable:

        * Overflow handles (no durable ``pool_worker_uid``): the eph pod is its
          own throwaway worker with no controller inventory to age; re-routing
          an overflow cache-hit would mint a fresh eph Job and orphan the old.
        * A durable worker with no inventory entry yet (just minted / mid
          daemon cold-start, hasn't heartbeat): NOT unhealthy — same in-flight
          case route()'s home path treats as benign. Falling through here would
          re-route a perfectly good fresh placement.

        Only an inventory entry that EXISTS and is non-routable forces a
        cache-miss fall-through (so a STALE home raises HomeStaleError and a
        DEAD/DRAINING home re-routes).
        """
        if handle.kind != 'durable' or handle.pool_worker_uid is None:
            return True
        inventory = self._worker_inventory.get(handle.pool_worker_uid)
        if inventory is not None and not inventory.status.is_routable():
            return False
        placement_state = self._state.placement_state
        pvc_state = (placement_state.pvcs.get(handle.pool_worker_uid) if
                     isinstance(placement_state, K8sPlacementState) else None)
        total_ssd_gib = (pvc_state.tier_ssd_gib if pvc_state is not None else
                         inventory.state.pvc_total_bytes >> 30
                         if inventory is not None else 0)
        gpu_count = (pvc_state.tier_gpu_count if pvc_state is not None else
                     inventory.gpu_slots_total if inventory is not None else 1)
        if (request.resource_hint.min_ssd_gib > 0
                and total_ssd_gib < request.resource_hint.min_ssd_gib):
            return False
        if gpu_count < request.resource_hint.min_gpu_slots:
            return False
        return True

    def _pinned_decision_for_retry(
        self, placement: Optional[SnapshotPlacement], job_id: JobID,
        resource_hint: Optional[ResourceHint] = None
    ) -> Optional[RouteDecision]:
        # Reconstruct prior route decision so retry doesn't re-route
        # (gpu_slots_in_use shifts across attempts → could flip path).
        if placement is None or job_id not in placement.leases:
            return None
        path = placement.lease_paths.get(job_id)
        if path is None:
            return None
        tier_ssd_gib = 0
        tier_gpu_count = 1
        if path == 'durable':
            # Don't pin a durable retry onto a worker whose inventory shows it
            # has gone non-routable: if its home went STALE/DEAD/DRAINING since
            # the lease was committed, fall through to route() so the
            # node-death path runs (STALE → HomeStaleError, DEAD/DRAINING →
            # mint fresh). A still-READY home — or one with no inventory entry
            # yet (just minted / mid daemon cold-start, not unhealthy) — keeps
            # the pin so the retry re-attaches to the half-provisioned PVC/pod
            # rather than minting a fresh one.
            inventory = (self._worker_inventory.get(placement.worker)
                         if placement.worker else None)
            if inventory is not None and not inventory.status.is_routable():
                return None
            placement_state = self._state.placement_state
            if isinstance(placement_state, K8sPlacementState):
                pvc_state = placement_state.pvcs.get(placement.worker)
                if pvc_state is not None:
                    tier_ssd_gib = pvc_state.tier_ssd_gib
                    tier_gpu_count = pvc_state.tier_gpu_count
            if (resource_hint is not None and tier_ssd_gib > 0
                    and tier_ssd_gib < resource_hint.min_ssd_gib):
                return None
            if (resource_hint is not None and tier_gpu_count > 0
                    and tier_gpu_count < resource_hint.min_gpu_slots):
                return None
        return RouteDecision(
            path=path,
            pool_worker_uid=(placement.worker if path == 'durable' else None),
            provisioned_new_worker=False,
            reason='pinned-on-retry',
            snapshot_size_estimate_bytes=placement.size_bytes,
            tier_ssd_gib=tier_ssd_gib,
            tier_gpu_count=tier_gpu_count,
        )

    def _open_lease_count(self, snapshot_id: str) -> int:
        placement = self._state.snapshots.get(snapshot_id)
        return len(placement.leases) if placement is not None else 0

    def _gpu_slots_in_use_for_worker(self, pool_worker_uid: str) -> int:
        # Count durable GPU leases on this worker. Overflow leases run
        # on eph nodes and don't count. Source-of-truth is lease_paths
        # (handle-map membership is unreliable mid-provision).
        gpu_activities = {SnapshotActivity.TRAIN, SnapshotActivity.PREDICT}
        count = 0
        for snapshot_id, placement in self._state.snapshots.items():
            if placement.worker != pool_worker_uid:
                continue
            for job_id in placement.leases:
                if placement.lease_paths.get(job_id) != 'durable':
                    continue
                activity = placement.lease_activities.get(job_id)
                if activity is None:
                    activity = self._lease_activity.get((snapshot_id, job_id))
                if activity is None:
                    # Activity is unknowable for pre-lease_activities durable
                    # leases, so count every legacy durable lease as one GPU
                    # slot. This may transiently serialize CPU-only legacy
                    # work after upgrade, but avoids GPU oversubscription.
                    count += 1
                    continue
                if activity in gpu_activities:
                    count += 1
        return count

    def _persisted_gpu_slots_total_for_worker(self, pool_worker_uid: str,
                                              fallback: int = 1) -> int:
        """GPU capacity from persisted PVC state.

        Fresh workers commit ``PvcState`` under the placement state lock before
        K8s provisioning starts, so their first daemon heartbeat should observe
        the chosen tier. ``fallback`` only covers legacy/pre-tier workers or
        otherwise missing PVC bookkeeping.
        """
        placement_state = self._state.placement_state
        if isinstance(placement_state, K8sPlacementState):
            pvc_state = placement_state.pvcs.get(pool_worker_uid)
            if pvc_state is not None:
                return max(pvc_state.tier_gpu_count, 1)
        return max(fallback, 1)

    def _persisted_worker_state(self,
                                pool_worker_uid: str) -> WorkerStateSnapshot:
        """Conservative capacity view before a worker's first heartbeat.

        Persisted PVC tier metadata proves total capacity, but used bytes are
        unknown. Report the entire tier as used so routing/metrics never treat
        an unverified worker as having free disk while still preserving the
        pool's configured capacity in dashboards.
        """
        total_bytes = 0
        placement_state = self._state.placement_state
        if isinstance(placement_state, K8sPlacementState):
            pvc_state = placement_state.pvcs.get(pool_worker_uid)
            if pvc_state is not None:
                total_bytes = max(pvc_state.tier_ssd_gib, 0) * (1 << 30)
        return WorkerStateSnapshot(
            ssd_total_bytes=total_bytes,
            ssd_used_bytes=total_bytes,
        )

    def _silent_worker_suspects(self) -> Dict[str, float]:
        """Return the persisted map that authoritatively marks synthetic
        silent-worker inventory entries."""
        placement_state = self._state.placement_state
        if isinstance(placement_state, K8sPlacementState):
            return placement_state.silent_worker_suspect_since
        return {}

    def _silent_worker_status(self, suspect_since: float,
                              now: float) -> WorkerStatus:
        """Compute synthetic STALE/DEAD status from one shared policy."""
        # ``None`` means RPC-serving grace has not started. Fail closed as
        # STALE; startup-only reconcile must never retire a worker before it has
        # had an opportunity to heartbeat into this process.
        dead_grace_elapsed = (self._silent_worker_dead_not_before is not None
                              and now >= self._silent_worker_dead_not_before)
        if (dead_grace_elapsed
                and now - suspect_since >= self._worker_dead_threshold_secs):
            return WorkerStatus.DEAD
        return WorkerStatus.STALE

    def _worker_uids_with_persisted_refs(self) -> set[str]:
        worker_uids = {
            placement.worker
            for placement in self._state.snapshots.values() if placement.worker
        }
        worker_uids.update(
            local.pool_worker_uid
            for local in self._state.local_copies.values()
            if local.local_state != SnapshotLocalCopyState.EVICTED)
        return worker_uids

    def _worker_inventory_for_stats(self) -> Dict[str, WorkerInventory]:
        inventory = dict(PoolControllerPlacementContext(self).worker_inventory)
        placement_state = self._state.placement_state
        if not isinstance(placement_state, K8sPlacementState):
            return inventory

        referenced_workers = self._worker_uids_with_persisted_refs()
        for pool_worker_uid, pvc_state in placement_state.pvcs.items():
            if pool_worker_uid in inventory:
                continue
            if pool_worker_uid not in referenced_workers:
                continue
            # Synthetic stats for persisted workers that have not heartbeated
            # after restart. SSD usage is unknown, so report zero free space
            # instead of an optimistic "100% free"; health=STALE is the
            # signal that observability should treat these values as degraded.
            inventory[pool_worker_uid] = WorkerInventory(
                pool_worker_uid=pool_worker_uid,
                state=self._persisted_worker_state(pool_worker_uid),
                status=WorkerStatus.STALE,
                gpu_slots_total=max(pvc_state.tier_gpu_count, 1),
                gpu_slots_in_use=self._gpu_slots_in_use_for_worker(
                    pool_worker_uid),
            )
        return inventory

    def _placement_status(
            self, snapshot_id: str,
            placement: Optional[SnapshotPlacement]) -> PlacementStatus:
        # WARM = on PVC, COLD = in L2 backup, MISSING = neither.
        if placement is not None and placement.state == SnapshotState.READY:
            return PlacementStatus.WARM
        if snapshot_id in self._state.dataplane_backups:
            return PlacementStatus.COLD
        return PlacementStatus.MISSING

    def _placement_result(
        self,
        snapshot_id: str,
        task_queue: str,
        status: PlacementStatus,
        debug_info: Optional[Dict[str, str]] = None,
        pool_worker_uid: Optional[str] = None,
    ) -> PlacementResult:
        # dataplane_path only set for COLD (activity restores from L2).
        # pool_worker_uid is the durable worker the placement landed on (None
        # for overflow); the workflow-side ``run_pooled_activity`` polls its
        # health after a failure to decide STALE-wait / READY-retry / reroute.
        backup = self._state.dataplane_backups.get(snapshot_id)
        tier_ssd_gib = 0
        if pool_worker_uid is not None:
            placement_state = self._state.placement_state
            if isinstance(placement_state, K8sPlacementState):
                pvc_state = placement_state.pvcs.get(pool_worker_uid)
                if pvc_state is not None:
                    tier_ssd_gib = pvc_state.tier_ssd_gib
                if tier_ssd_gib <= 0:
                    inventory = self._worker_inventory.get(pool_worker_uid)
                    if inventory is not None:
                        tier_ssd_gib = inventory.state.pvc_total_bytes >> 30
        return PlacementResult(
            snapshot_id=snapshot_id,
            task_queue=task_queue,
            pool_worker_uid=pool_worker_uid,
            status=status,
            dataplane_path=(backup.dataplane_path
                            if status == PlacementStatus.COLD
                            and backup is not None else None),
            tier_ssd_gib=tier_ssd_gib,
            debug_info=debug_info or {},
        )

    def _ensure_compliance_record(self,
                                  snapshot_id: str) -> SnapshotComplianceRecord:
        record = self._state.snapshot_compliance.get(snapshot_id)
        if record is None:
            record = SnapshotComplianceRecord(
                snapshot_id=snapshot_id,
                max_dataplane_retention_secs=(
                    self._lifecycle_policy.max_dataplane_retention_secs),
            )
            self._state.snapshot_compliance[snapshot_id] = record
        return record

    def _sync_lifecycle_from_leases(
        self,
        snapshot_id: str,
        *,
        final_release_at=None,
    ) -> None:
        """Derive lifecycle protection clocks from the logical lease table."""
        count = self._open_lease_count(snapshot_id)
        record = self._ensure_compliance_record(snapshot_id)
        record.active_consumer_count = count
        if count > 0:
            record.compliance_delete_at = None
            record.expired_at = None
            record.deleted_at = None
            for local in self._state.local_copies.values():
                if (local.snapshot_id == snapshot_id and local.local_state
                        != SnapshotLocalCopyState.EVICTED):
                    local.evictable_under_pressure_at = None
                    local.force_evict_at = None
            return

        if final_release_at is None:
            return

        record.last_released_at = final_release_at
        record.compliance_delete_at = (final_release_at + timedelta(
            seconds=self._lifecycle_policy.max_dataplane_retention_secs))
        record.expired_at = None
        record.deleted_at = None
        for local in self._state.local_copies.values():
            if (local.snapshot_id != snapshot_id
                    or local.local_state == SnapshotLocalCopyState.EVICTED):
                continue
            local.last_released_at = final_release_at
            local.evictable_under_pressure_at = (
                final_release_at +
                timedelta(seconds=self._lifecycle_policy.min_ssd_idle_secs))
            local.force_evict_at = (
                final_release_at +
                timedelta(seconds=self._lifecycle_policy.max_ssd_idle_secs))

    def _record_local_copy_ready(
        self,
        snapshot_id: str,
        pool_worker_uid: str,
        *,
        size_bytes: Optional[int] = None,
        observed_at=None,
    ) -> None:
        observed_at = observed_at or self._now_utc()
        key = _local_copy_key(snapshot_id, pool_worker_uid)
        local = self._state.local_copies.get(key)
        if (local is not None
                and local.local_state == SnapshotLocalCopyState.EVICTING):
            # EVICTING is a committed delete. It is entered only with no open
            # leases, and place_snapshot blocks new authorized work until the
            # delete completes. A positive report in this window therefore
            # describes the old copy (or orphaned work) and must not reopen
            # warm placement while the external delete is in flight.
            log.debug(
                'Ignoring READY report during committed snapshot eviction: '
                'snapshot=%s worker=%s', snapshot_id, pool_worker_uid)
            return
        if local is None:
            local = SnapshotLocalCopyRecord(
                snapshot_id=snapshot_id,
                pool_worker_uid=pool_worker_uid,
                min_ssd_idle_secs=(self._lifecycle_policy.min_ssd_idle_secs),
                max_ssd_idle_secs=(self._lifecycle_policy.max_ssd_idle_secs),
            )
            self._state.local_copies[key] = local
        local.local_state = SnapshotLocalCopyState.WARM
        local.evicted_at = None
        if size_bytes is not None:
            local.size_bytes = size_bytes

        record = self._ensure_compliance_record(snapshot_id)
        if record.active_consumer_count > 0:
            local.evictable_under_pressure_at = None
            local.force_evict_at = None
        else:
            local.last_released_at = record.last_released_at or observed_at
            local.evictable_under_pressure_at = (
                local.last_released_at +
                timedelta(seconds=self._lifecycle_policy.min_ssd_idle_secs))
            local.force_evict_at = (
                local.last_released_at +
                timedelta(seconds=self._lifecycle_policy.max_ssd_idle_secs))
            if record.last_released_at is None:
                record.last_released_at = observed_at
                record.compliance_delete_at = (observed_at + timedelta(
                    seconds=self._lifecycle_policy.max_dataplane_retention_secs)
                                               )

    def _has_live_local_copy(self, snapshot_id: str) -> bool:
        return any(
            local.snapshot_id == snapshot_id and local.local_state in {
                SnapshotLocalCopyState.WARM,
                SnapshotLocalCopyState.EVICTION_CANDIDATE,
                SnapshotLocalCopyState.EVICTING,
            } for local in self._state.local_copies.values())

    def _worker_under_ssd_pressure(self, pool_worker_uid: str) -> bool:
        inventory = self._worker_inventory.get(pool_worker_uid)
        if inventory is None or inventory.state.pvc_total_bytes <= 0:
            return False
        # Bond pressure to the configured logical cap (fraction of physical),
        # not the raw physical disk: a worker is "full" once used exceeds
        # total * ssd_max_utilization, leaving the complement as headroom.
        free_floor = (inventory.state.pvc_total_bytes *
                      (1.0 - self._lifecycle_policy.ssd_max_utilization))
        return inventory.state.pvc_free_bytes < free_floor

    def _total_pool_disk_bytes(self) -> int:
        """Total physical PVC capacity across known workers (0 if none has
        reported yet)."""
        return sum(inv.state.pvc_total_bytes
                   for inv in self._worker_inventory.values())

    def _reject_if_exceeds_disk_capacity(self, source_size_bytes: int) -> None:
        """Require the fixed host to have at least 1.5x the source estimate.

        A fixed host cannot try the preferred 3x tier and then move elsewhere,
        so it uses the same streaming-ingest floor as the largest K8s tier.
        Capacity and unknown source sizes are allowed through when they cannot
        be checked here; the ingest's local no-space classification remains
        the final correctness guard.
        """
        total = self._total_pool_disk_bytes()
        if total <= 0 or source_size_bytes <= 0:
            return
        minimum_bytes = (source_size_bytes * 3 + 1) // 2
        if minimum_bytes > total:

            def _gib(n: int) -> str:
                return f'{n / (1 << 30):.1f} GiB'

            raise DiskgraphError(
                f'Estimated source size {_gib(source_size_bytes)} requires at '
                f'least {_gib(minimum_bytes)} of local SSD for streaming '
                f'DiskGraph ingest, but this fixed host has {_gib(total)}. '
                f'Use a host with more disk or a smaller dataset.')

    def _active_dataplane_demand_bytes(self, snapshot_id: str,
                                       pool_worker_uid: str) -> int:
        return sum(
            max(placement.size_bytes, 0)
            for candidate_id, placement in self._state.snapshots.items()
            if candidate_id != snapshot_id and placement.leases and
            placement.state != SnapshotState.READY and candidate_id not in self.
            _state.dataplane_backups and placement.worker == pool_worker_uid)

    def _reject_if_insufficient_local_dataplane_capacity(
            self, snapshot_id: str, pool_worker_uid: str,
            estimate_bytes: int) -> None:
        root = self._data_plane_storage_root
        try:
            if (root is None or not fsutil.is_local(root)
                    or estimate_bytes <= 0):
                return
        except ValueError:
            return

        inventory = self._worker_inventory.get(pool_worker_uid)
        if inventory is None:
            return
        state = inventory.state
        same_filesystem = state.dataplane_same_filesystem_as_ssd
        if state.dataplane_total_bytes <= 0 or same_filesystem is None:
            return

        # Capacity and filesystem identity come from the selected writer's
        # mount namespace; an equal path on another K8s node is not evidence
        # of a shared capacity domain.
        active_bytes = self._active_dataplane_demand_bytes(
            snapshot_id, pool_worker_uid)
        copies_per_snapshot = 2 if same_filesystem else 1
        reported_free = state.dataplane_free_bytes
        if same_filesystem:
            reported_free = min(reported_free, state.pvc_free_bytes)
        effective_free = max(reported_free - active_bytes * copies_per_snapshot,
                             0)
        required_bytes = estimate_bytes * copies_per_snapshot
        if required_bytes <= effective_free:
            return

        raise InsufficientDataplaneCapacityError(
            f'Fresh DiskGraph ingestion requires an estimated '
            f'{required_bytes} bytes on local data_plane_storage_root '
            f'{root!r}, but only {effective_free} bytes are available after '
            f'active ingests. Free or expand that volume, configure object '
            f'storage for the dataplane root, or reduce the dataset size.')

    def _request_reactive_eviction(self) -> None:
        """Signal an immediate local-retention sweep after ENOSPC."""
        self._reactive_eviction_requested_at = self._time()

    def _reactive_eviction_requested(self) -> bool:
        if self._reactive_eviction_requested_at is None:
            return False
        return (self._time() - self._reactive_eviction_requested_at
                <= REACTIVE_EVICTION_WINDOW_SECS)

    def _should_dispatch_local_eviction(self, local: SnapshotLocalCopyRecord,
                                        sweep_now) -> bool:
        force_elapsed = (local.force_evict_at is not None
                         and sweep_now >= local.force_evict_at)
        return (force_elapsed or self._reactive_eviction_requested()
                or self._worker_under_ssd_pressure(local.pool_worker_uid))

    def _mark_local_copy_evicted(self, local: SnapshotLocalCopyRecord,
                                 sweep_now) -> None:
        local.local_state = SnapshotLocalCopyState.EVICTED
        local.evicted_at = sweep_now
        placement = self._state.snapshots.get(local.snapshot_id)
        if (placement is not None and placement.worker == local.pool_worker_uid
                and not placement.leases):
            del self._state.snapshots[local.snapshot_id]

    async def _cleanup_orphan_snapshot(self, snapshot_id: str,
                                       pool_worker_uid: str) -> bool:
        async with self._state_lock:
            async with self._snapshot_locks.acquire(snapshot_id):
                if self._open_lease_count(snapshot_id) > 0:
                    log.info(
                        'Skipping orphan snapshot cleanup for snapshot=%s '
                        'worker=%s; active lease exists.', snapshot_id,
                        pool_worker_uid)
                    return False

        deleted = await self._local_snapshot_deleter(snapshot_id,
                                                     pool_worker_uid)
        sweep_now = self._now_utc()
        async with self._state_lock:
            async with self._snapshot_locks.acquire(snapshot_id):
                if self._open_lease_count(snapshot_id) > 0:
                    log.info(
                        'Skipping orphan snapshot cleanup state update for '
                        'snapshot=%s worker=%s; active lease exists.',
                        snapshot_id, pool_worker_uid)
                    return False
                local_key = _local_copy_key(snapshot_id, pool_worker_uid)
                local = self._state.local_copies.get(local_key)
                if local is None:
                    local = SnapshotLocalCopyRecord(
                        snapshot_id=snapshot_id,
                        pool_worker_uid=pool_worker_uid,
                    )
                    self._state.local_copies[local_key] = local
                self._mark_local_copy_evicted(local, sweep_now)
                await self._persistence.save(self._state)

        if not deleted:
            log.info(
                'Orphan snapshot cleanup found no local directory for '
                'snapshot=%s worker=%s; marked evicted.', snapshot_id,
                pool_worker_uid)
        return True

    def _expected_worker_uids(self) -> set[str]:
        """Pool-worker uids the controller still expects to exist, re-derived
        under ``state_lock`` from live ``self._state`` (the strategy computed
        its delta on an unlocked deepcopy). Same definition the strategy uses
        — see :func:`expected_pool_worker_uids`. Used to drop freshly-leased
        workers from the delete set before tearing them down.
        """
        return expected_pool_worker_uids(self._state)

    async def _refresh_silent_expected_workers(self) -> None:
        """Seed bounded liveness for persisted workers with no heartbeat.

        ``worker_inventory`` is in-memory only. After a controller restart,
        persisted K8s PVC/snapshot state may still reference a worker whose
        daemon will either re-heartbeat shortly or never return. Treat that
        silence as STALE during the normal dead-threshold grace window, then
        expose it as DEAD so strategy reconcile can use the existing retire
        path. A real heartbeat clears the suspect timer.
        """
        placement_state = self._state.placement_state
        if not isinstance(placement_state, K8sPlacementState):
            return

        now = self._time()
        async with self._state_lock:
            expected = (self._expected_worker_uids()
                        & self._startup_expected_worker_uids)
            suspects = placement_state.silent_worker_suspect_since
            suspects_changed = False
            for pool_worker_uid in list(suspects):
                if pool_worker_uid not in expected:
                    del suspects[pool_worker_uid]
                    suspects_changed = True
                    # Membership in ``suspects`` proves this inventory entry is
                    # synthetic. Remove both sides together so a worker that is
                    # no longer expected cannot age into a phantom DEAD metric.
                    self._worker_inventory.pop(pool_worker_uid, None)

            for pool_worker_uid in sorted(expected):
                inventory = self._worker_inventory.get(pool_worker_uid)
                if inventory is not None and pool_worker_uid not in suspects:
                    continue

                suspect_since = suspects.get(pool_worker_uid)
                if suspect_since is None:
                    suspect_since = now
                    suspects[pool_worker_uid] = suspect_since
                    suspects_changed = True

                status = self._silent_worker_status(suspect_since, now)
                state = (inventory.state if inventory is not None else
                         self._persisted_worker_state(pool_worker_uid))
                self._worker_inventory[pool_worker_uid] = WorkerInventory(
                    pool_worker_uid=pool_worker_uid,
                    state=state,
                    status=status,
                    gpu_slots_total=(self._persisted_gpu_slots_total_for_worker(
                        pool_worker_uid)),
                    gpu_slots_in_use=self._gpu_slots_in_use_for_worker(
                        pool_worker_uid),
                    last_heartbeat=suspect_since,
                )

            if suspects_changed:
                await self._persistence.save(self._state)

    def _apply_reconcile_worker_delta(self, delta: ReconcileDelta) -> bool:
        """Apply controller-owned effects from strategy worker reconciliation.

        The strategy owns K8s-specific side effects such as deleting
        pool-worker resources. Controller core only reflects the resulting
        worker state in its own bookkeeping: retire local/lease state for UIDs
        the strategy has declared terminal (DEAD heartbeat) or incompatible
        with the current configured tier set.
        """
        mutated = False
        retired_at = self._now_utc()

        for pool_worker_uid in (delta.retired_uids |
                                delta.incompatible_uids):
            self._mark_worker_draining(pool_worker_uid)
            if self._scrub_deleted_worker_state(pool_worker_uid, retired_at):
                mutated = True

        return mutated

    def _mark_worker_draining(self, pool_worker_uid: str) -> None:
        """Install the non-routable tombstone before external teardown.

        Unexpected workers discovered during startup may not have sent a
        heartbeat into this controller process yet. They still need an
        inventory tombstone: otherwise a teardown call that returns before the
        observed pod disappears leaves a window where its first late heartbeat
        can register it as READY and recreate scrubbed placement state.
        """
        inventory = self._worker_inventory.get(pool_worker_uid)
        if inventory is None:
            inventory = WorkerInventory(
                pool_worker_uid=pool_worker_uid,
                state=self._persisted_worker_state(pool_worker_uid),
                status=WorkerStatus.DRAINING,
                gpu_slots_total=self._persisted_gpu_slots_total_for_worker(
                    pool_worker_uid),
                gpu_slots_in_use=self._gpu_slots_in_use_for_worker(
                    pool_worker_uid),
                last_heartbeat=self._time(),
            )
            self._worker_inventory[pool_worker_uid] = inventory
        else:
            inventory.status = WorkerStatus.DRAINING

    def _event_matches_current_worker(self, state: PoolState, snapshot_id: str,
                                      pool_worker_uid: str) -> bool:
        """Whether a local/PVC event may mutate this snapshot placement."""
        placement = state.snapshots.get(snapshot_id)
        return placement is None or placement.worker == pool_worker_uid

    def _local_snapshot_report_is_current_locked(
            self, pool_worker_uid: str,
            worker_storage_mode: PoolStorageMode) -> bool:
        """Whether a local-state report still has valid worker ownership.

        Caller holds ``state_lock``. Re-checking here, immediately before the
        persisted mutation, closes the window where reconcile commits a worker
        to DRAINING after ``report_status``'s entry check but before the report
        acquires the lock.

        Activity reports do not create worker inventory. If their daemon
        inventory has already been pruned, persisted PVC bookkeeping is the
        remaining ownership proof. Teardown scrubs that bookkeeping together
        with placement/local state, so a delayed activity report cannot recreate
        a copy for an already-deleted worker. Non-K8s strategies retain their
        existing behavior.
        """
        inventory = self._worker_inventory.get(pool_worker_uid)
        if (inventory is not None
                and inventory.status == WorkerStatus.DRAINING):
            return False
        if (worker_storage_mode == PoolStorageMode.ACTIVITY_ON_PVC
                and inventory is None):
            placement_state = self._state.placement_state
            if (isinstance(placement_state, K8sPlacementState)
                    and pool_worker_uid not in placement_state.pvcs):
                return False
        return True

    def _scrub_deleted_worker_state(self, pool_worker_uid: str,
                                    retired_at) -> bool:
        """Clear controller state pointing at a worker committed for deletion:
        its tracked PVC, any ``SnapshotPlacement`` homed on it, and any
        non-evicted local copy.

        Dropping the placement is what makes the teardown route-race fix hold
        for the *home* path: ``route`` derives ``home_worker_uid`` from
        ``placement.worker`` and co-locates new activities there WITHOUT a
        READY gate (see ``K8sPlacementStrategy._home_can_serve``), so a
        placement left pointing at a draining worker would let a fresh
        activity land on it mid-teardown. Called for both retired (daemon
        gone) and unexpected (no lease / no local copy) deletes; an
        unexpected worker only has lease-less placements + evicted copies, so
        this just removes the now-defunct entries. Returns whether any state
        changed.
        """
        mutated = False
        placement_state = self._state.placement_state
        if isinstance(placement_state, K8sPlacementState):
            if placement_state.pvcs.pop(pool_worker_uid, None) is not None:
                mutated = True
            if (placement_state.silent_worker_suspect_since.pop(
                    pool_worker_uid, None) is not None):
                mutated = True
            self._startup_expected_worker_uids.discard(pool_worker_uid)

        affected_snapshot_ids: set[str] = set()
        for snapshot_id, placement in list(self._state.snapshots.items()):
            if placement.worker == pool_worker_uid:
                if placement.leases:
                    placement.leases.clear()
                    placement.last_lease_released = retired_at
                del self._state.snapshots[snapshot_id]
                affected_snapshot_ids.add(snapshot_id)
                mutated = True

        for local in self._state.local_copies.values():
            if (local.pool_worker_uid == pool_worker_uid
                    and local.local_state != SnapshotLocalCopyState.EVICTED):
                affected_snapshot_ids.add(local.snapshot_id)
                self._mark_local_copy_evicted(local, retired_at)
                mutated = True

        for snapshot_id in affected_snapshot_ids:
            self._sync_lifecycle_from_leases(snapshot_id,
                                             final_release_at=retired_at)
        return mutated

    async def _replan_unavailable_worker(
        self,
        request: PlacementRequest,
        pool_worker_uid: str,
        inventory_token: Optional[WorkerInventory],
    ) -> bool:
        """Retire a failed pinned worker after pod and heartbeat fencing.

        ``ExistingWorkerUnavailableError`` already carries the no-create
        reclaim's fresh PVC/Deployment failure; revalidate concurrent pod and
        heartbeat evidence here before mutating persisted placement state.
        """
        assert self._strategy is not None
        safe_to_retire = await self._strategy.revalidate_retired_workers(
            {pool_worker_uid})
        if pool_worker_uid not in safe_to_retire:
            return False

        async with self._state_lock:
            placement = self._state.snapshots.get(request.snapshot_id)
            if (placement is None or placement.worker != pool_worker_uid
                    or request.job_id not in placement.leases):
                return True
            # A heartbeat after the failed provision proves the worker is
            # still participating. Let the prompt failure surface rather than
            # racing that fresh evidence with destructive cleanup.
            if (self._worker_inventory.get(pool_worker_uid)
                    is not inventory_token):
                return False
            self._mark_worker_draining(pool_worker_uid)
            self._scrub_deleted_worker_state(pool_worker_uid, self._now_utc())
            await self._persistence.save(self._state)

        deleted = await self._strategy.delete_workers({pool_worker_uid})
        if pool_worker_uid in deleted:
            async with self._state_lock:
                self._delete_succeeded_pending_absence.add(pool_worker_uid)
        return True

    def _record_heartbeat_ready_snapshot(self, snapshot_id: str,
                                         pool_worker_uid: str,
                                         observed_at) -> bool:
        placement_before = deepcopy(self._state.snapshots.get(snapshot_id))
        local_key = _local_copy_key(snapshot_id, pool_worker_uid)
        local_before = deepcopy(self._state.local_copies.get(local_key))
        compliance_before = deepcopy(
            self._state.snapshot_compliance.get(snapshot_id))

        placement = self._state.snapshots.get(snapshot_id)
        if placement is None:
            placement = SnapshotPlacement(
                worker=pool_worker_uid,
                state=SnapshotState.READY,
            )
            self._state.snapshots[snapshot_id] = placement
        elif placement.worker != pool_worker_uid:
            log.warning(
                'Preserving current snapshot home while recording a physical '
                'copy reported by a non-current worker: '
                'snapshot=%s current_worker=%s event_worker=%s', snapshot_id,
                placement.worker, pool_worker_uid)
        else:
            placement.state = SnapshotState.READY

        self._record_local_copy_ready(
            snapshot_id,
            pool_worker_uid,
            observed_at=observed_at,
        )

        return (placement_before != self._state.snapshots.get(snapshot_id)
                or local_before != self._state.local_copies.get(local_key)
                or compliance_before
                != self._state.snapshot_compliance.get(snapshot_id))

    async def _reconcile_missing_local_copies_from_first_heartbeat(
            self, pool_worker_uid: str, present_snapshots: set[str]) -> None:
        """Apply negative startup-scan evidence from a worker's first report."""
        sweep_now = self._now_utc()
        local_keys = [
            key for key, local in self._state.local_copies.items()
            if (local.pool_worker_uid == pool_worker_uid
                and local.local_state != SnapshotLocalCopyState.EVICTED
                and local.snapshot_id not in present_snapshots)
        ]
        for key in local_keys:
            local = self._state.local_copies[key]
            async with self._state_lock:
                async with self._snapshot_locks.acquire(local.snapshot_id):
                    current = self._state.local_copies.get(key)
                    if (current is None or current.local_state
                            == SnapshotLocalCopyState.EVICTED
                            or current.snapshot_id in present_snapshots):
                        continue
                    self._mark_local_copy_evicted(current, sweep_now)
                    self._log_open_placements_for_missing_local_copy(
                        current.snapshot_id, pool_worker_uid)
                    await self._persistence.save(self._state)

    def _log_open_placements_for_missing_local_copy(
            self, snapshot_id: str, pool_worker_uid: str) -> None:
        placement = self._state.snapshots.get(snapshot_id)
        if placement is None or placement.worker != pool_worker_uid:
            return
        for job_id in placement.leases:
            log.warning(
                'Active placement points at a local copy missing from daemon '
                'startup scan: snapshot=%s job=%s worker=%s. The workflow is '
                'responsible for retrying place_snapshot/ensure_snapshot.',
                snapshot_id, job_id, pool_worker_uid)

    async def _reconcile_ready_snapshots_from_heartbeat(
            self, pool_worker_uid: str, ready_snapshots: list[str]) -> None:
        """Apply positive READY evidence from daemon heartbeat inventory.

        Heartbeats are authoritative for complete local snapshots discovered
        during daemon startup scan or periodic inventory. This intentionally
        only adds/refreshes READY state; absence from a heartbeat is a separate
        policy because transient daemon/reporting gaps must not evict data.
        """
        observed_at = self._now_utc()
        for snapshot_id in ready_snapshots:
            async with self._state_lock:
                async with self._snapshot_locks.acquire(snapshot_id):
                    if not self._local_snapshot_report_is_current_locked(
                            pool_worker_uid, PoolStorageMode.DAEMON_ON_PVC):
                        log.warning(
                            'Dropping heartbeat READY mutation after worker '
                            'teardown committed: snapshot=%s worker=%s',
                            snapshot_id, pool_worker_uid)
                        continue
                    if self._record_heartbeat_ready_snapshot(
                            snapshot_id, pool_worker_uid, observed_at):
                        await self._persistence.save(self._state)

    def _sweep_worker_health(self, now: Optional[float] = None) -> None:
        """Age in-memory worker inventory through READY -> STALE -> DEAD.

        Worker inventory is not persisted; this helper only updates the
        controller's current liveness view used by placement/stats.
        """
        now = self._time() if now is None else now
        suspects = self._silent_worker_suspects()
        for inventory in self._worker_inventory.values():
            suspect_since = suspects.get(inventory.pool_worker_uid)
            if suspect_since is not None:
                inventory.status = self._silent_worker_status(
                    suspect_since, now)
                continue
            # Never age a controller-owned status back to READY: DRAINING
            # (scale-down committed) is set deliberately by reconcile, and
            # heartbeat age must not override it — doing so could re-expose a
            # draining worker to routing. See WorkerStatus.is_controller_owned.
            if inventory.status.is_controller_owned():
                continue
            age = now - inventory.last_heartbeat
            if age >= self._worker_dead_threshold_secs:
                inventory.status = WorkerStatus.DEAD
            elif age >= self._worker_stale_threshold_secs:
                inventory.status = WorkerStatus.STALE
            else:
                inventory.status = WorkerStatus.READY

    async def _reconcile_once(
            self, *, refresh_silent_workers: bool = True) -> ReconcileDelta:
        """Ask the active placement strategy to reconcile external reality.

        P0 applies orphan snapshot cleanup because that is a controller-owned
        daemon delete effect. Retire (DEAD), incompatible-tier, and unexpected
        teardown actions stay strategy/K8s-specific follow-up work.
        """
        if self._strategy is None:
            return ReconcileDelta()
        if refresh_silent_workers:
            await self._refresh_silent_expected_workers()
        delta = await self._strategy.reconcile(deepcopy(self._state))
        log.info(
            'placement strategy reconcile: retired=%d incompatible=%d '
            'unexpected=%d live=%d orphans=%d',
            len(delta.retired_uids),
            len(delta.incompatible_uids),
            len(delta.unexpected_uids),
            len(delta.live_pool_worker_uids),
            len(delta.orphan_snapshots),
        )
        # A pod appearing after the first observation vetoes state/PVC cleanup.
        revalidated_retired_uids = (await
                                    self._strategy.revalidate_retired_workers(
                                        delta.retired_uids))
        # Phase 1 (under state_lock): reflect the reconcile result in
        # controller bookkeeping AND flip every uid we're about to delete to a
        # non-routable status BEFORE any K8s delete happens.
        #
        # ``unexpected_uids`` need an airtight re-check: ``reconcile`` decided
        # they were no longer expected (no lease / no local copy) on a deepcopy
        # snapshot taken WITHOUT the lock. Between that snapshot and acquiring
        # the lock here a concurrent ``place_snapshot`` may have reserved a
        # fresh lease on one of them — re-derive the expected set under the
        # lock and drop any such freshly-leased uid from the delete set so it
        # isn't torn down out from under a live placement.
        #
        # ``retired_uids`` also need a health re-check. A daemon heartbeat can
        # arrive after the strategy snapshots DEAD but before this apply phase;
        # only workers that are still DEAD and pass external revalidation are
        # committed to deletion. Once committed, they become DRAINING
        # synchronously before this block next awaits. That DRAINING tombstone
        # remains until teardown succeeds and a later external observation
        # confirms the worker is absent.
        #
        # ``route`` needs ``state_lock``; the warm and durable-candidate paths
        # gate on READY, and the home path is closed by scrubbing the
        # placement that points at the worker (below). So once a uid is
        # committed for deletion here it is both non-routable AND unreachable
        # via an existing placement while ``delete_workers`` runs unlocked
        # below (teardown route-race fix).
        async with self._state_lock:
            confirmed_absent = (self._delete_succeeded_pending_absence -
                                delta.observed_pool_worker_uids)
            for pool_worker_uid in confirmed_absent:
                inventory = self._worker_inventory.get(pool_worker_uid)
                if (inventory is not None
                        and inventory.status == WorkerStatus.DRAINING):
                    self._worker_inventory.pop(pool_worker_uid, None)
                self._delete_succeeded_pending_absence.discard(pool_worker_uid)

            expected_now = self._expected_worker_uids()
            retired_to_delete = {
                uid
                for uid in revalidated_retired_uids
                if (inventory := self._worker_inventory.get(uid)) is not None
                and inventory.status == WorkerStatus.DEAD
            }
            placement_state = self._state.placement_state
            incompatible_to_delete = {
                uid
                for uid in delta.incompatible_uids
                if (isinstance(placement_state, K8sPlacementState)
                    and uid in placement_state.pvcs
                    and uid not in confirmed_absent
                    and uid not in self._delete_succeeded_pending_absence)
            }
            unexpected_to_delete = {
                uid
                for uid in delta.unexpected_uids
                if (uid not in expected_now and uid not in confirmed_absent
                    and uid not in self._delete_succeeded_pending_absence)
            }
            orphan_snapshots_to_delete = {
                snapshot_id: pool_worker_uid
                for snapshot_id, pool_worker_uid in
                delta.orphan_snapshots.items()
                if (pool_worker_uid not in delta.retired_uids
                    or pool_worker_uid in retired_to_delete)
            }
            effective_delta = replace(
                delta,
                retired_uids=retired_to_delete,
                incompatible_uids=incompatible_to_delete,
                unexpected_uids=unexpected_to_delete,
                orphan_snapshots=orphan_snapshots_to_delete,
            )
            uids_to_delete = (retired_to_delete | incompatible_to_delete |
                              unexpected_to_delete)
            mutated = self._apply_reconcile_worker_delta(effective_delta)
            # Retired/incompatible uids were flipped to DRAINING + scrubbed by
            # the delta application above; do the same for unexpected uids:
            # flip them DRAINING (non-routable) and scrub any state pointing at
            # them — especially a lingering placement.worker the home path
            # would otherwise co-locate onto (see _scrub_deleted_worker_state).
            scrub_now = self._now_utc()
            for pool_worker_uid in unexpected_to_delete:
                self._mark_worker_draining(pool_worker_uid)
                if self._scrub_deleted_worker_state(pool_worker_uid, scrub_now):
                    mutated = True
            if mutated:
                await self._persistence.save(self._state)

        # Phase 2 (no lock): perform the actual K8s teardown. A successful call
        # is not sufficient proof that a worker observed at the start of this
        # reconcile is gone: the DEAD-node force path can proceed while its pod
        # is still terminating. Keep DRAINING in that case and prune only after
        # a later reconcile no longer observes the worker. A failed teardown is
        # omitted from ``deleted`` and remains DRAINING for retry next tick.
        if uids_to_delete:
            deleted = await self._strategy.delete_workers(uids_to_delete)
            if deleted:
                async with self._state_lock:
                    for pool_worker_uid in deleted:
                        if (pool_worker_uid
                                in effective_delta.observed_pool_worker_uids):
                            self._delete_succeeded_pending_absence.add(
                                pool_worker_uid)
                        else:
                            self._worker_inventory.pop(pool_worker_uid, None)

        for snapshot_id, pool_worker_uid in sorted(
                effective_delta.orphan_snapshots.items()):
            try:
                await self._cleanup_orphan_snapshot(snapshot_id,
                                                    pool_worker_uid)
            except Exception:
                log.exception(
                    'Orphan snapshot cleanup failed for snapshot=%s worker=%s; '
                    'will retry on next reconcile.', snapshot_id,
                    pool_worker_uid)
        return effective_delta

    async def _maintenance_once(self) -> ReconcileDelta:
        """Run one controller-owned maintenance tick.

        This is the production call site for heartbeat health aging,
        dangling-lease cleanup, lifecycle sweeps, and the strategy reconcile
        hook.
        """
        self._sweep_worker_health()
        maintenance_now = self._time()
        if self._dangling_lease_sweep_due(maintenance_now):
            await self._dangling_lease_sweep_once()
            self._last_dangling_lease_sweep_at = maintenance_now
        await self._lifecycle_sweeps_once()
        delta = await self._reconcile_once()
        metrics.update_state_gauges(self._state,
                                    self._worker_inventory_for_stats(),
                                    self._lease_activity)
        return delta

    def _dangling_lease_sweep_due(self, maintenance_now: float) -> bool:
        return (self._last_dangling_lease_sweep_at is None
                or maintenance_now - self._last_dangling_lease_sweep_at
                >= self._lease_sweep_interval_secs)

    async def _dangling_lease_sweep_once(self) -> int:
        """Release open leases whose owning workflow is terminal or gone.

        Normal release flows through ``release_snapshot`` from the workflow's
        ``finally``. When a workflow hits ``run_timeout`` Temporal terminates
        without running ``finally``, so the lease leaks. This periodic sweep
        reaps those leases by asking Temporal whether the owning workflow is
        still RUNNING.

        Keys leases on the parent Temporal workflow_id recorded at lease
        commit (``SnapshotPlacement.lease_workflow_ids``), not on ``job_id``
        directly: sharded batch prediction creates one lease per shard with
        a distinct ``job_id``, but all shards share the parent workflow_id
        — which is what Temporal knows about.

        Idempotent — release goes through ``_release_lease`` which is itself
        idempotent on ``(snapshot_id, job_id)``.
        """
        # Group leases by parent workflow_id first so each workflow_id is
        # queried against Temporal at most once per sweep. Sharded jobs
        # collapse to one entry here; without batching a 5-shard BP would
        # issue 5 identical ``get_workflow_desc`` RPCs per 5-min sweep.
        leases_by_wf: dict[str, list[tuple[str, JobID]]] = {}
        for snapshot_id, placement in self._state.snapshots.items():
            for job_id in placement.leases:
                # Fall back to ``job_id`` only for legacy lease records
                # persisted before ``lease_workflow_ids`` existed; new
                # leases always populate the map at commit.
                workflow_id = placement.lease_workflow_ids.get(job_id, job_id)
                leases_by_wf.setdefault(workflow_id, []).append(
                    (snapshot_id, job_id))

        released = 0
        for workflow_id, leases in leases_by_wf.items():
            try:
                workflow_state = await self._workflow_status_checker(workflow_id
                                                                     )
            except Exception:
                log.exception(
                    'Dangling lease sweep: workflow status check failed '
                    'for workflow_id=%s (covers %d lease(s)); leaving '
                    'leases open until next sweep.', workflow_id, len(leases))
                continue

            if workflow_state == WorkflowLeaseState.RUNNING:
                continue

            for snapshot_id, job_id in leases:
                log.info(
                    'Dangling lease sweep: force-releasing lease for '
                    'snapshot=%s job=%s workflow_id=%s workflow_state=%s.',
                    snapshot_id, job_id, workflow_id, workflow_state.value)
                try:
                    await self._release_lease(snapshot_id, job_id)
                except Exception:
                    log.exception(
                        'Dangling lease sweep: release failed for snapshot=%s '
                        'job=%s; leaving the lease open until the next sweep.',
                        snapshot_id, job_id)
                else:
                    released += 1
        return released

    async def _lifecycle_sweeps_once(self) -> None:
        """Run the controller periodic L2/L1 cleanup pass."""
        sweep_now = self._now_utc()
        compliance_deletes: list[tuple[str, DataplaneBackup]] = []
        local_deletes: list[tuple[str, str]] = []
        mutated = False

        async with self._state_lock:
            for record in list(self._state.snapshot_compliance.values()):
                if self._prepare_compliance_sweep(record, sweep_now,
                                                  compliance_deletes):
                    mutated = True

            for local in list(self._state.local_copies.values()):
                if self._prepare_local_copy_sweep(local, sweep_now,
                                                  local_deletes):
                    mutated = True

            if self._gc_tombstone_records(sweep_now):
                mutated = True

            if mutated:
                await self._persistence.save(self._state)
            self._reactive_eviction_requested_at = None

        for snapshot_id, backup in compliance_deletes:
            await self._run_compliance_delete(snapshot_id, backup, sweep_now)

        for snapshot_id, pool_worker_uid in local_deletes:
            await self._run_local_copy_delete(snapshot_id, pool_worker_uid,
                                              sweep_now)

    def _gc_tombstone_records(self, sweep_now: datetime) -> bool:
        """Drop terminal tombstone records past the GC grace horizon so
        pool_state tracks the active working set, not cumulative history.

        Without this, ``local_copies`` (EVICTED) and ``snapshot_compliance``
        (``deleted_at`` set) records accumulate forever — they are only ever
        flipped to terminal, never removed — making the whole-file ``save()``
        grow linearly with lifetime ingest count (the global-lock throughput
        ceiling + state-file leak; see backlog P0). Caller holds
        ``self._state_lock``.
        """
        horizon = timedelta(seconds=self._tombstone_gc_grace_secs)
        mutated = False
        gc_local = 0
        gc_compliance = 0

        for key, local in list(self._state.local_copies.items()):
            if local.local_state != SnapshotLocalCopyState.EVICTED:
                continue
            if local.evicted_at is None:
                # Defensive: an EVICTED record with no evicted_at (legacy / a
                # state file from before the field was stamped) would otherwise
                # never become GC-eligible — the exact leak this guards against.
                # Backfill so it ages out one grace window from now.
                local.evicted_at = sweep_now
                mutated = True
                continue
            if sweep_now - local.evicted_at >= horizon:
                del self._state.local_copies[key]
                gc_local += 1
                mutated = True

        # A deleted compliance record means the L2 backup is gone. Re-check
        # there's no live lease or local copy (a snapshot re-requested after
        # deletion would have recreated them) before dropping the tombstone.
        for snapshot_id, record in list(
                self._state.snapshot_compliance.items()):
            if (record.deleted_at is not None
                    and sweep_now - record.deleted_at >= horizon
                    and self._open_lease_count(snapshot_id) == 0
                    and not self._has_live_local_copy(snapshot_id)):
                del self._state.snapshot_compliance[snapshot_id]
                gc_compliance += 1
                mutated = True

        # Observability parity with the L1/L2 delete counters: operators can
        # tell "GC pruned N" from "nothing to prune" / "GC silently broken".
        metrics.inc_sweep_delete('tombstone_local', gc_local)
        metrics.inc_sweep_delete('tombstone_compliance', gc_compliance)
        return mutated

    def _prepare_compliance_sweep(
        self,
        record: SnapshotComplianceRecord,
        sweep_now,
        compliance_deletes: list[tuple[str, DataplaneBackup]],
    ) -> bool:
        mutated = False
        clock_elapsed = (record.compliance_delete_at is not None
                         and sweep_now >= record.compliance_delete_at)
        if (not clock_elapsed or self._open_lease_count(record.snapshot_id) > 0
                or self._has_live_local_copy(record.snapshot_id)):
            return mutated

        if record.deleted_at is not None:
            return mutated

        if record.expired_at is None:
            record.expired_at = sweep_now
            mutated = True

        backup = self._state.dataplane_backups.get(record.snapshot_id)
        if backup is not None:
            compliance_deletes.append((record.snapshot_id, deepcopy(backup)))
            return mutated

        record.deleted_at = sweep_now
        return True

    async def _run_compliance_delete(
        self,
        snapshot_id: str,
        backup: DataplaneBackup,
        sweep_now,
    ) -> None:
        try:
            await self._dataplane_deleter(backup)
        except Exception:
            log.exception(
                'Dataplane delete failed for snapshot=%s path=%s; '
                'will retry on next sweep.', snapshot_id, backup.dataplane_path)
            return

        async with self._state_lock:
            record = self._state.snapshot_compliance.get(snapshot_id)
            current_backup = self._state.dataplane_backups.get(snapshot_id)
            if (record is None or record.expired_at is None
                    or record.deleted_at is not None or current_backup != backup
                    or self._open_lease_count(snapshot_id) > 0
                    or self._has_live_local_copy(snapshot_id)):
                return

            del self._state.dataplane_backups[snapshot_id]
            record.deleted_at = sweep_now
            await self._persistence.save(self._state)
            metrics.inc_sweep_delete('dataplane')

    def _prepare_local_copy_sweep(
        self,
        local: SnapshotLocalCopyRecord,
        sweep_now,
        local_deletes: list[tuple[str, str]],
    ) -> bool:
        mutated = False
        if local.local_state == SnapshotLocalCopyState.EVICTING:
            if self._open_lease_count(local.snapshot_id) > 0:
                return mutated
            if local.snapshot_id in self._state.dataplane_backups:
                local_deletes.append((local.snapshot_id, local.pool_worker_uid))
            return mutated

        if local.local_state == SnapshotLocalCopyState.WARM:
            if self._open_lease_count(local.snapshot_id) > 0:
                return mutated
            if (local.evictable_under_pressure_at is not None
                    and sweep_now >= local.evictable_under_pressure_at):
                local.local_state = SnapshotLocalCopyState.EVICTION_CANDIDATE
                mutated = True

        if local.local_state != SnapshotLocalCopyState.EVICTION_CANDIDATE:
            return mutated
        if self._open_lease_count(local.snapshot_id) > 0:
            return mutated
        if not self._should_dispatch_local_eviction(local, sweep_now):
            return mutated
        if local.snapshot_id not in self._state.dataplane_backups:
            return mutated

        local.local_state = SnapshotLocalCopyState.EVICTING
        local_deletes.append((local.snapshot_id, local.pool_worker_uid))
        return True

    async def _run_local_copy_delete(
        self,
        snapshot_id: str,
        pool_worker_uid: str,
        sweep_now,
    ) -> None:
        async with self._state_lock:
            async with self._snapshot_locks.acquire(snapshot_id):
                local = self._state.local_copies.get(
                    _local_copy_key(snapshot_id, pool_worker_uid))
                if (local is None or local.local_state
                        != SnapshotLocalCopyState.EVICTING):
                    return

        try:
            deleted = await self._local_snapshot_deleter(
                snapshot_id, pool_worker_uid)
        except Exception:
            log.exception(
                'Daemon local snapshot delete failed for snapshot=%s '
                'worker=%s; will retry on next sweep.', snapshot_id,
                pool_worker_uid)
            await self._finish_local_copy_delete(snapshot_id, pool_worker_uid,
                                                 sweep_now, deleted=False,
                                                 failed=True)
            return

        await self._finish_local_copy_delete(snapshot_id, pool_worker_uid,
                                             sweep_now, deleted=deleted,
                                             failed=False)

    async def _finish_local_copy_delete(
        self,
        snapshot_id: str,
        pool_worker_uid: str,
        sweep_now,
        *,
        deleted: bool,
        failed: bool,
    ) -> None:
        async with self._state_lock:
            async with self._snapshot_locks.acquire(snapshot_id):
                local = self._state.local_copies.get(
                    _local_copy_key(snapshot_id, pool_worker_uid))
                if (local is None or local.local_state
                        != SnapshotLocalCopyState.EVICTING):
                    return

                if failed:
                    # The external delete may have partially removed files.
                    # Keep the placement barrier active and rejoin/retry the
                    # deterministic delete on the next lifecycle sweep.
                    await self._persistence.save(self._state)
                    return

                if not deleted:
                    log.info(
                        'Daemon local snapshot delete found no directory for '
                        'snapshot=%s worker=%s; marking evicted.', snapshot_id,
                        pool_worker_uid)
                self._mark_local_copy_evicted(local, sweep_now)
                await self._persistence.save(self._state)
                # Count only physical deletions; deleted=False is an
                # idempotent no-op (daemon found no directory), not an
                # eviction performed.
                if deleted:
                    metrics.inc_sweep_delete('local')

    async def _maintenance_loop(self, interval_secs: float) -> None:
        while True:
            try:
                await self._maintenance_once()
            except Exception:
                log.exception('Pool controller maintenance tick failed.')
            await asyncio.sleep(interval_secs)

    def _start_maintenance_loop(self,
                                interval_secs: Optional[float] = None) -> None:
        if (self._maintenance_task is not None
                and not self._maintenance_task.done()):
            return
        if interval_secs is None:
            interval_secs = self._maintenance_interval_secs
        if self._silent_worker_dead_not_before is None:
            self._silent_worker_dead_not_before = (
                self._time() + self._worker_stale_threshold_secs)
        self._maintenance_task = asyncio.create_task(
            self._maintenance_loop(interval_secs))

    async def _mutate_and_persist(
            self, snapshot_id: str, mutate_fn: Callable[[PoolState],
                                                        None]) -> None:
        """Apply ``mutate_fn`` to ``self._state`` and persist under the
        controller state lock. See class docstring's §Atomicity contract.
        Block B may add variants (e.g., multi-snapshot mutations) but must
        preserve the mutate-then-persist ordering.
        """
        async with self._state_lock:
            async with self._snapshot_locks.acquire(snapshot_id):
                mutate_fn(self._state)
                await self._persistence.save(self._state)

    async def _apply_worker_event(self, pool_worker_uid: str,
                                  event: WorkerEvent, *,
                                  worker_storage_mode: PoolStorageMode) -> None:
        """Apply daemon lifecycle events that affect persisted pool state.

        ``worker_storage_mode`` is the reporter's own storage contract.
        Local lifecycle events from a non-PVC-backed reporter (overflow eph
        pod) are dropped — see :meth:`PoolStorageMode.is_pvc_backed`.
        Dataplane side-effects (SNAPSHOT_PERSIST_UPLOADED) are still honored
        regardless of reporter storage mode: the L2 backup is durable even
        when the reporter's local copy is not.
        """
        if event.type == WorkerEventType.SNAPSHOT_ENSURE_NO_SPACE:
            await self._handle_ensure_snapshot_no_space(
                pool_worker_uid, event, worker_storage_mode=worker_storage_mode)
            return

        if event.type == WorkerEventType.SNAPSHOT_ENSURE_FAILED:
            await self._handle_ensure_snapshot_failed(pool_worker_uid, event)
            return

        if event.snapshot_id is None:
            log.warning(
                'Ignoring worker event without snapshot_id: type=%s '
                'worker=%s', event.type.value, pool_worker_uid)
            return

        snapshot_id = event.snapshot_id

        if event.type == WorkerEventType.SSD_CACHE_CORRUPTED:
            if worker_storage_mode != PoolStorageMode.DAEMON_ON_PVC:
                log.debug(
                    'Dropping SSD corruption from non-daemon worker: '
                    'snapshot=%s worker=%s storage_mode=%s', snapshot_id,
                    pool_worker_uid, worker_storage_mode.value)
                return

            def invalidate_local_copy(state: PoolState) -> None:
                local_key = _local_copy_key(snapshot_id, pool_worker_uid)
                local = state.local_copies.get(local_key)
                placement = state.snapshots.get(snapshot_id)
                if local is None:
                    # Preserve an exact tombstone for legacy current-home
                    # state that predates local-copy records. Do not invent
                    # state for an unrelated reporter.
                    if (placement is None
                            or placement.worker != pool_worker_uid):
                        return
                    local = SnapshotLocalCopyRecord(
                        snapshot_id=snapshot_id,
                        pool_worker_uid=pool_worker_uid,
                        size_bytes=placement.size_bytes,
                    )
                    state.local_copies[local_key] = local
                self._mark_local_copy_evicted(local, event.timestamp)
                if (placement is not None
                        and placement.worker == pool_worker_uid
                        and placement.leases):
                    placement.state = SnapshotState.RESERVED

            await self._mutate_and_persist(snapshot_id, invalidate_local_copy)
            return

        if event.type == WorkerEventType.SNAPSHOT_PERSIST_UPLOADED:
            dataplane_path = event.dataplane_path
            if dataplane_path is None:
                log.warning(
                    'Ignoring dataplane-ready event without dataplane_path: '
                    'snapshot=%s worker=%s', snapshot_id, pool_worker_uid)
                return

            def record_backup(state: PoolState) -> None:
                # Record the size so it survives eviction: the placement's
                # (set by SNAPSHOT_READY), else the event's.
                placement = state.snapshots.get(snapshot_id)
                size_bytes = (placement.size_bytes if placement is not None
                              and placement.size_bytes > 0 else
                              (event.size_bytes or 0))
                state.dataplane_backups[snapshot_id] = DataplaneBackup(
                    snapshot_id=snapshot_id,
                    dataplane_path=dataplane_path,
                    size_bytes=size_bytes,
                )
                record = self._ensure_compliance_record(snapshot_id)
                record.expired_at = None
                record.deleted_at = None

            await self._mutate_and_persist(snapshot_id, record_backup)
            return

        if event.type == WorkerEventType.SNAPSHOT_READY:
            if not worker_storage_mode.is_pvc_backed:
                # Overflow eph pod: its emptyDir-backed copy is not a
                # durable local-copy identity. Recording the eph UID
                # into placement.worker / local_copies would let
                # _expected_pool_worker_uids feed it to the reconcile
                # loop, which would then provision a daemon Deployment
                # + PVC with the eph-prefix UID — the genesis loop that
                # corrupted antest-349a2f pool_state.json.
                log.debug(
                    'Dropping SNAPSHOT_READY from non-PVC-backed worker: '
                    'snapshot=%s worker=%s storage_mode=%s', snapshot_id,
                    pool_worker_uid, worker_storage_mode.value)
                return

            def record_local_ready(state: PoolState) -> None:
                if not self._local_snapshot_report_is_current_locked(
                        pool_worker_uid, worker_storage_mode):
                    log.warning(
                        'Dropping SNAPSHOT_READY after worker ownership was '
                        'removed: snapshot=%s worker=%s storage_mode=%s',
                        snapshot_id, pool_worker_uid, worker_storage_mode.value)
                    return
                placement = state.snapshots.get(snapshot_id)
                if placement is None:
                    if worker_storage_mode.is_activity_scoped:
                        backup = state.dataplane_backups.get(snapshot_id)
                        if (backup is not None and backup.size_bytes <= 0
                                and event.size_bytes is not None):
                            backup.size_bytes = event.size_bytes
                        self._record_local_copy_ready(
                            snapshot_id,
                            pool_worker_uid,
                            size_bytes=event.size_bytes,
                            observed_at=event.timestamp,
                        )
                        log.warning(
                            'Recorded physical SNAPSHOT_READY copy from PVC '
                            'activity worker without recreating a placement: '
                            'snapshot=%s worker=%s', snapshot_id,
                            pool_worker_uid)
                        return
                    placement = SnapshotPlacement(worker=pool_worker_uid)
                    state.snapshots[snapshot_id] = placement
                elif self._event_matches_current_worker(state, snapshot_id,
                                                        pool_worker_uid):
                    placement.worker = pool_worker_uid
                else:
                    backup = state.dataplane_backups.get(snapshot_id)
                    if (backup is not None and backup.size_bytes <= 0
                            and event.size_bytes is not None):
                        backup.size_bytes = event.size_bytes
                    self._record_local_copy_ready(
                        snapshot_id,
                        pool_worker_uid,
                        size_bytes=event.size_bytes,
                        observed_at=event.timestamp,
                    )
                    log.warning(
                        'Preserving current snapshot home while recording a '
                        'physical SNAPSHOT_READY copy from a non-current '
                        'worker: '
                        'snapshot=%s current_worker=%s event_worker=%s',
                        snapshot_id, placement.worker, pool_worker_uid)
                    return

                if event.size_bytes is not None:
                    placement.size_bytes = event.size_bytes
                    # Also fill the backup's size if one exists (e.g. a COLD
                    # snapshot being restored), so a legacy 0 self-heals.
                    backup = state.dataplane_backups.get(snapshot_id)
                    if backup is not None:
                        backup.size_bytes = event.size_bytes
                placement.state = SnapshotState.READY
                self._record_local_copy_ready(
                    snapshot_id,
                    pool_worker_uid,
                    size_bytes=event.size_bytes,
                    observed_at=event.timestamp,
                )

            await self._mutate_and_persist(snapshot_id, record_local_ready)
            return

        if (event.type in {
                WorkerEventType.SNAPSHOT_INGEST_STARTED,
                WorkerEventType.SNAPSHOT_RESTORE_STARTED,
        } and not worker_storage_mode.is_pvc_backed):
            log.debug(
                'Dropping %s from non-PVC-backed worker: '
                'snapshot=%s worker=%s storage_mode=%s', event.type.value,
                snapshot_id, pool_worker_uid, worker_storage_mode.value)
            return

        if event.type == WorkerEventType.SNAPSHOT_INGEST_STARTED:

            def record_ingesting(state: PoolState) -> None:
                placement = state.snapshots.get(snapshot_id)
                if (placement is not None
                        and self._event_matches_current_worker(
                            state, snapshot_id, pool_worker_uid)):
                    placement.state = SnapshotState.INGESTING

            await self._mutate_and_persist(snapshot_id, record_ingesting)
            return

        if event.type == WorkerEventType.SNAPSHOT_RESTORE_STARTED:

            def record_restoring(state: PoolState) -> None:
                placement = state.snapshots.get(snapshot_id)
                if (placement is not None
                        and self._event_matches_current_worker(
                            state, snapshot_id, pool_worker_uid)):
                    placement.state = SnapshotState.RESTORING

            await self._mutate_and_persist(snapshot_id, record_restoring)
            return

        log.warning('Ignoring unsupported worker event: type=%s worker=%s',
                    event.type.value, pool_worker_uid)

    async def _handle_ensure_snapshot_no_space(
            self, pool_worker_uid: str, event: WorkerEvent, *,
            worker_storage_mode: PoolStorageMode) -> None:
        if event.snapshot_id is None or event.job is None:
            log.warning(
                'Ignoring no-space ensure_snapshot event without snapshot/job: '
                'worker=%s snapshot=%s job=%s',
                pool_worker_uid,
                event.snapshot_id,
                event.job.job_id if event.job is not None else None,
            )
            return

        if worker_storage_mode.is_pvc_backed:
            placement = self._state.snapshots.get(event.snapshot_id)
            if not self._event_matches_current_worker(
                    self._state, event.snapshot_id, pool_worker_uid):
                current_worker = (placement.worker
                                  if placement is not None else None)
                log.warning(
                    'Ignoring no-space ensure_snapshot event from '
                    'non-current worker: snapshot=%s job=%s '
                    'current_worker=%s event_worker=%s',
                    event.snapshot_id,
                    event.job.job_id,
                    current_worker,
                    pool_worker_uid,
                )
                return

        log.info(
            'ensure_snapshot reported no local SSD space: '
            'snapshot=%s job=%s worker=%s storage_mode=%s; releasing lease.',
            event.snapshot_id,
            event.job.job_id,
            pool_worker_uid,
            worker_storage_mode.value,
        )
        await self._release_lease(event.snapshot_id, event.job.job_id)
        if worker_storage_mode.is_pvc_backed:
            self._request_reactive_eviction()

    async def _handle_ensure_snapshot_failed(self, pool_worker_uid: str,
                                             event: WorkerEvent) -> None:
        if event.snapshot_id is None or event.job is None:
            log.warning(
                'Ignoring failed ensure_snapshot event without snapshot/job: '
                'worker=%s snapshot=%s job=%s',
                pool_worker_uid,
                event.snapshot_id,
                event.job.job_id if event.job is not None else None,
            )
            return

        log.warning(
            'ensure_snapshot failed: snapshot=%s job=%s worker=%s; releasing '
            'lease and clearing non-ready placement progress.',
            event.snapshot_id,
            event.job.job_id,
            pool_worker_uid,
        )
        snapshot_id = event.snapshot_id
        await self._release_lease(snapshot_id, event.job.job_id)

        def clear_failed_progress(state: PoolState) -> None:
            placement = state.snapshots.get(snapshot_id)
            if (placement is not None and placement.worker == pool_worker_uid
                    and placement.state != SnapshotState.READY):
                del state.snapshots[snapshot_id]

        await self._mutate_and_persist(snapshot_id, clear_failed_progress)

    def _starting_worker_grace_remaining(
        self,
        request: PlacementRequest,
        decision: RouteDecision,
    ) -> float:
        """Called under state lock, immediately before committing a route."""
        state = self._state.placement_state
        if (not isinstance(self._strategy, K8sPlacementStrategy)
                or not isinstance(state, K8sPlacementState)
                or decision.path != 'durable'
                or not decision.provisioned_new_worker):
            return 0
        remaining = 0.0
        for uid, pvc in state.pvcs.items():
            # Known STALE/DEAD/DRAINING workers are not new startups. READY
            # workers were already considered by route() with real free space.
            if uid in self._worker_inventory:
                continue
            age = max(0.0, (self._now_utc() - pvc.created_at).total_seconds())
            grace = self._worker_startup_grace_secs - age
            if grace <= 0:
                continue
            if self._strategy.starting_worker_can_fit(
                    request, pvc,
                    size_bytes=decision.snapshot_size_estimate_bytes,
                    gpu_slots_in_use=self._gpu_slots_in_use_for_worker(uid)):
                remaining = max(remaining, grace)
        return remaining

    @asynccontextmanager
    async def _reserve_placement_route(
        self,
        request: PlacementRequest,
    ) -> AsyncIterator[Tuple[Optional[SnapshotPlacement], RouteDecision, int]]:
        """Yield a route with locks held through the caller's reservation.

        Check, route and commit are one critical section. On a startup wait we
        release BOTH state and snapshot locks, then reroute from fresh state on
        every notification. No waiter owns a worker or reserves speculative
        capacity. The existing per-(snapshot, job) provision lock still makes
        duplicate calls for the same placement idempotent.
        """
        assert self._strategy is not None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._worker_startup_grace_secs
        while True:
            async with self._state_lock:
                async with self._snapshot_locks.acquire(request.snapshot_id):
                    # Capture BEFORE reading/routing: even a report arriving
                    # during an await in route cannot cause a missed wakeup.
                    changed = self._worker_capacity_changed
                    placement = self._state.snapshots.get(request.snapshot_id)
                    self._raise_if_current_copy_evicting(
                        request.snapshot_id, placement)
                    snapshot_size_estimate = (
                        request.source_size_bytes *
                        self._ingest_space_multiplier +
                        self._snapshot_estimate_buffer_bytes
                        if request.source_size_bytes > 0 else 0)
                    decision = self._pinned_decision_for_retry(
                        placement, request.job_id, request.resource_hint)
                    if decision is None:
                        home_worker_uid = (placement.worker
                                           if placement is not None
                                           and placement.worker else None)
                        # Preserve same-snapshot coalescing/GPU overflow. The
                        # grace only applies when routing needs a NEW daemon.
                        home_gpu_in_use = (
                            self._gpu_slots_in_use_for_worker(home_worker_uid)
                            if home_worker_uid is not None else 0)
                        # Singlehost MISSING ingestion has a fixed-host 1.5x
                        # guard. WARM reuse/COLD restore already fit the disk;
                        # k8s admission is per tier, never summed across nodes.
                        if (isinstance(self._strategy,
                                       SinglehostPlacementStrategy)
                                and self._placement_status(
                                    request.snapshot_id,
                                    placement) == PlacementStatus.MISSING):
                            self._reject_if_exceeds_disk_capacity(
                                request.source_size_bytes)
                        decision = await self._strategy.route(
                            request, warm_location_uid=self._warm_location_uid(
                                request.snapshot_id),
                            snapshot_size_estimate_bytes=snapshot_size_estimate,
                            home_worker_uid=home_worker_uid,
                            home_gpu_in_use=home_gpu_in_use,
                            overflow_viable=(request.snapshot_id
                                             in self._state.dataplane_backups))
                    wait_secs = min(
                        deadline - loop.time(),
                        self._starting_worker_grace_remaining(
                            request, decision))
                    if wait_secs <= 0:
                        yield placement, decision, snapshot_size_estimate
                        return
            log.info(
                'Waiting up to %.1fs for starting pool capacity: '
                'snapshot=%s job=%s', wait_secs, request.snapshot_id,
                request.job_id)
            try:
                await asyncio.wait_for(changed.wait(), timeout=wait_secs)
            except TimeoutError:
                # Reroute once more before scale-out; do not extend the budget
                # for repeated heartbeats, new workers or spurious wakeups.
                deadline = loop.time()

    # --- RPC handlers ---

    @rpc_handler
    @metrics.observe_place_snapshot
    async def place_snapshot(self,
                             request: PlacementRequest) -> PlacementResult:
        """Place ``request.snapshot_id`` for ``request.job_id`` and return
        the task queue that should serve this activity execution.

        Caller contract:

        - Dispatch activities on the returned task queue.
        - Pair this call with :meth:`release_lease` on job completion
          (success or failure). Placement atomically acquires a lease
          that protects the snapshot from eviction until released.
        - On failure (no capacity, controller still warming up): raises.
          Callers are Temporal workflows; standard activity retry
          handles transient cases.

        Happy path (first call from a workflow):
          1. fast-path lookup MISS (no handle yet)
          2. retry-pin MISS (no prior lease) → call route() → durable
          3. create new SnapshotPlacement, append lease, persist
          4. provision() returns ProvisionedHandle
          5. store handle in durable_handles[job_id], return WARM/COLD

        Edge cases handled inline:
          - fast-path HIT: Temporal retry after a successful call
          - retry-pin HIT: phase 2 cancelled before storing handle
          - overflow path: route() picks eph pod (needs warm copy)
          - lease released concurrently mid-phase-2 → orphan handle
        """
        request = replace(
            request,
            snapshot_id=_canonicalize_snapshot_id(request.snapshot_id),
        )
        if self._strategy is None:
            raise NotImplementedError(
                'place_snapshot requires a PlacementStrategy. Real K8s and '
                'singlehost strategies land in follow-up PRs; tests inject a '
                'mock strategy for Block B controller-core behavior.')

        # Backfill a legacy COLD backup's size before the locks (it may scan
        # the dataplane), so routing/overflow read a known size.
        await self._ensure_backup_size(request.snapshot_id)

        # Per-(snapshot, job) lock: serializes concurrent calls for
        # the same job.
        placement_key = _placement_key(request.snapshot_id, request.job_id)
        async with self._placement_provision_locks.acquire(placement_key):
            # ----- PHASE 1: route + reserve lease (under lock) -----
            # State lock spans route() so sibling shards see each
            # other's leases when computing gpu_slots_in_use.
            handle_to_validate: Optional[ProvisionedHandle] = None
            ready_handle_to_replace: Optional[ProvisionedHandle] = None
            async with self._state_lock:
                async with self._snapshot_locks.acquire(request.snapshot_id):
                    placement = self._state.snapshots.get(request.snapshot_id)

                    # Fast path: handle already provisioned and ready.
                    handle = self._lookup_lease_handle(placement,
                                                       request.job_id)
                    if handle is not None:
                        self._raise_if_current_copy_evicting(
                            request.snapshot_id, placement)
                        if handle.provision_state == 'ready':
                            if self._handle_worker_routable(handle, request):
                                # A persisted ready handle still needs its
                                # exact activity poller revalidated. That may
                                # call external services, so do it outside the
                                # state locks.
                                handle_to_validate = handle
                            else:
                                # Do not reuse a durable handle whose worker
                                # became STALE/DEAD/DRAINING after it was
                                # minted. Release its deterministic Job before
                                # route() chooses a recovered or fresh worker;
                                # the runner's re-place loop depends on this.
                                ready_handle_to_replace = handle
                        else:
                            # The pod or poller is not ready yet. Check it
                            # outside the state locks.
                            handle_to_validate = handle

            if handle_to_validate is not None:
                handle_to_validate = await self._wait_provision_ready(
                    request, handle_to_validate)
                status = await self._persist_ready_handle(
                    request, handle_to_validate)
                return self._placement_result(
                    request.snapshot_id, handle_to_validate.task_queue, status,
                    debug_info=handle_to_validate.debug_info,
                    pool_worker_uid=handle_to_validate.pool_worker_uid)

            async with self._reserve_placement_route(request) as (
                    placement, decision, snapshot_size_estimate):
                if (decision.path == 'durable'
                        and decision.pool_worker_uid is None):
                    raise RuntimeError('Durable placement requires planned '
                                       'pool_worker_uid.')
                if decision.path == 'overflow':
                    # Overflow needs L2 backup to bootstrap from.
                    self._validate_l2_backup(request.snapshot_id)

                decision_reservation_bytes = max(
                    decision.snapshot_size_estimate_bytes,
                    snapshot_size_estimate,
                )
                if decision.tier_ssd_gib > 0:
                    tier_usable_bytes = int(
                        decision.tier_ssd_gib * (1 << 30) *
                        self._lifecycle_policy.ssd_max_utilization)
                    decision_reservation_bytes = min(
                        decision_reservation_bytes,
                        tier_usable_bytes,
                    )
                candidate_reservation_bytes = max(
                    placement.size_bytes if placement is not None else 0,
                    decision_reservation_bytes,
                )
                reservation_worker_uid = (decision.pool_worker_uid if (
                    decision.path == 'durable'
                    and self._placement_status(request.snapshot_id, placement)
                    == PlacementStatus.MISSING) else None)
                if (reservation_worker_uid is not None and
                    (placement is None
                     or placement.worker != reservation_worker_uid
                     or candidate_reservation_bytes > placement.size_bytes)):
                    self._reject_if_insufficient_local_dataplane_capacity(
                        request.snapshot_id,
                        reservation_worker_uid,
                        candidate_reservation_bytes,
                    )

                # Commit lease + decision to state.
                # placement is None → controller never saw this
                # snapshot before (no prior place, no daemon report).
                # placement is not None → snapshot already known
                # (prior place call, heartbeat, or SNAPSHOT_READY).
                if placement is None:
                    # First-ever placement is always durable
                    # (overflow needs a pre-existing warm copy),
                    # so pool_worker_uid is guaranteed set.
                    if decision.pool_worker_uid is None:
                        raise AssertionError(
                            'First-placement RouteDecision is missing '
                            'pool_worker_uid; overflow on first '
                            'placement is impossible.')
                    placement = SnapshotPlacement(
                        worker=decision.pool_worker_uid,
                        size_bytes=(decision.snapshot_size_estimate_bytes
                                    or snapshot_size_estimate),
                    )
                    self._state.snapshots[request.snapshot_id] = placement
                elif (not placement.leases
                      and placement.state != SnapshotState.READY):
                    placement.size_bytes = max(
                        placement.size_bytes,
                        decision.snapshot_size_estimate_bytes
                        or snapshot_size_estimate,
                    )

                if reservation_worker_uid is not None:
                    # Reserve the admitted candidate on its selected writer
                    # before provision releases the controller lock.
                    placement.size_bytes = candidate_reservation_bytes
                    placement.worker = reservation_worker_uid

                if (decision.path == 'durable'
                        and decision.pool_worker_uid is not None
                        and decision.tier_ssd_gib > 0):
                    placement_state = self._state.placement_state
                    if isinstance(placement_state, K8sPlacementState):
                        pvc_state = placement_state.pvcs.setdefault(
                            decision.pool_worker_uid,
                            PvcState(
                                pool_worker_uid=decision.pool_worker_uid,
                                tier_ssd_gib=decision.tier_ssd_gib,
                                tier_gpu_count=max(decision.tier_gpu_count, 1),
                                created_at=self._now_utc(),
                            ),
                        )
                        if pvc_state.tier_ssd_gib <= 0:
                            pvc_state.tier_ssd_gib = decision.tier_ssd_gib
                        if (pvc_state.tier_ssd_gib == decision.tier_ssd_gib):
                            pvc_state.tier_gpu_count = max(
                                pvc_state.tier_gpu_count,
                                decision.tier_gpu_count,
                                1,
                            )
                        else:
                            log.warning(
                                'Durable worker %s persisted SSD tier '
                                '%s GiB does not match route decision '
                                '%s GiB; preserving existing gpu_count=%s '
                                'instead of applying decision gpu_count=%s.',
                                decision.pool_worker_uid,
                                pvc_state.tier_ssd_gib,
                                decision.tier_ssd_gib,
                                pvc_state.tier_gpu_count,
                                decision.tier_gpu_count,
                            )

                if request.job_id not in placement.leases:
                    placement.leases.append(request.job_id)
                # lease_paths anchors the pin-on-retry above and
                # drives _gpu_slots_in_use_for_worker.
                placement.lease_paths[request.job_id] = decision.path
                # lease_workflow_ids keys the dangling-lease sweep so
                # it can group leases by parent workflow without
                # parsing ``job_id``. Required even when
                # ``job_id == workflow_id`` (single-shot path) so the
                # sweep doesn't have to special-case missing entries.
                placement.lease_workflow_ids[
                    request.job_id] = request.workflow_id
                placement.lease_activities[request.job_id] = request.activity

                self._lease_activity[(request.snapshot_id,
                                      request.job_id)] = request.activity
                self._sync_lifecycle_from_leases(request.snapshot_id)
                await self._persistence.save(self._state)

            # ----- PHASE 2: provision K8s (no lock) -----
            # provision() uses deterministic names → retry re-attaches.
            # PRESERVE THE LEASE ON FAILURE. The lease is what makes
            # ``_pinned_decision_for_retry`` return the same uid on the
            # next attempt, so ``strategy.provision`` reclaims the
            # half-provisioned PVC + Deployment instead of minting a
            # fresh one (which would orphan the old PVC). Whether the
            # exception is a Cancel (workflow-side abort) or a generic
            # provision failure (transient K8s API blip, image pull
            # delay, etc.), the right action is the same: surface to
            # the caller, let them retry; if they give up entirely,
            # the dangling-lease sweep reaps the lease + the orphan
            # resources via reconcile.
            handle_to_release: Optional[ProvisionedHandle] = None
            if ready_handle_to_replace is not None:
                # Clean up the stale placement before reprovisioning its
                # deterministic resources.
                await self._release_strategy_handle(ready_handle_to_replace,
                                                    request.snapshot_id,
                                                    request.job_id)
            inventory_token = (self._worker_inventory.get(
                decision.pool_worker_uid) if decision.pool_worker_uid else None)
            try:
                provisioned_handle = await self._strategy.provision(
                    request, decision)
            except ExistingWorkerUnavailableError as exc:
                if (decision.pool_worker_uid is not None
                        and await self._replan_unavailable_worker(
                            request, decision.pool_worker_uid,
                            inventory_token)):
                    raise RuntimeError(
                        f'Unavailable worker {decision.pool_worker_uid} was '
                        'retired; retrying placement on a fresh worker.'
                    ) from exc
                raise

            # Store handle back into placement.
            async with self._state_lock:
                async with self._snapshot_locks.acquire(request.snapshot_id):
                    placement = self._state.snapshots.get(request.snapshot_id)
                    if (placement is None
                            or request.job_id not in placement.leases):
                        # Lease released concurrently; orphan the handle.
                        handle_to_release = provisioned_handle
                        status = PlacementStatus.MISSING
                    else:
                        if provisioned_handle.pool_worker_uid:
                            # Durable handles carry worker uid; learn
                            # it here if route() minted a new worker.
                            placement.worker = provisioned_handle.pool_worker_uid
                        if provisioned_handle.kind == 'durable':
                            # Per-job: task_queue keyed by job_id.
                            placement.durable_handles[
                                request.job_id] = provisioned_handle
                        else:
                            # Per-job: distinct eph Job per call.
                            placement.overflow_handles[
                                request.job_id] = provisioned_handle
                        status = self._placement_status(request.snapshot_id,
                                                        placement)
                        await self._persistence.save(self._state)

            # Tear down orphaned handle outside the lock (K8s calls).
            if handle_to_release is not None:
                await self._release_strategy_handle(handle_to_release,
                                                    request.snapshot_id,
                                                    request.job_id)
            else:
                # Wait for the pod and its Temporal poller. The persisted
                # handle lets release_lease clean up if readiness fails.
                provisioned_handle = await self._wait_provision_ready(
                    request, provisioned_handle)
                status = await self._persist_ready_handle(
                    request, provisioned_handle)

            return self._placement_result(
                request.snapshot_id,
                provisioned_handle.task_queue,
                status,
                debug_info=provisioned_handle.debug_info,
                pool_worker_uid=provisioned_handle.pool_worker_uid,
            )

    async def _persist_ready_handle(
        self,
        request: PlacementRequest,
        handle: ProvisionedHandle,
    ) -> PlacementStatus:
        """Persist the ready handle before returning the task queue."""
        async with self._state_lock:
            async with self._snapshot_locks.acquire(request.snapshot_id):
                placement = self._state.snapshots.get(request.snapshot_id)
                if placement is not None:
                    handles = (placement.durable_handles if handle.kind
                               == 'durable' else placement.overflow_handles)
                    if request.job_id in handles:
                        handles[request.job_id] = handle
                        await self._persistence.save(self._state)
                return self._placement_status(request.snapshot_id, placement)

    async def _wait_provision_ready(
        self,
        request: PlacementRequest,
        handle: ProvisionedHandle,
    ) -> ProvisionedHandle:
        """Wait for readiness; release the placement if the wait fails."""
        assert self._strategy is not None
        try:
            return await self._strategy.wait_provision_ready(
                handle, request.job_id)
        except (Exception, asyncio.CancelledError):
            try:
                await self._release_lease_while_placement_locked(
                    request.snapshot_id, request.job_id)
            except Exception:
                log.exception(
                    'Failed to clean up unready placement: snapshot=%s '
                    'job=%s', request.snapshot_id, request.job_id)
            raise

    @rpc_handler
    @metrics.count_rpc_errors('release_lease')
    async def release_lease(self, snapshot_id: str, job_id: str) -> None:
        """Release the lease ``job_id`` holds on ``snapshot_id``.

        Paired with :meth:`place_snapshot`: callers must invoke this on job
        completion (success or failure) so the snapshot becomes eligible for
        eviction. Idempotent — releasing a lease that does not exist is a
        no-op, so ``try/finally`` retry loops are safe.
        """
        snapshot_id = _canonicalize_snapshot_id(snapshot_id)
        await self._release_lease(snapshot_id, job_id)

    async def _release_lease(self, snapshot_id: str, job_id: str) -> None:
        placement_key = _placement_key(snapshot_id, job_id)
        async with self._placement_provision_locks.acquire(placement_key):
            await self._release_lease_while_placement_locked(
                snapshot_id, job_id)

    async def _release_lease_while_placement_locked(self, snapshot_id: str,
                                                    job_id: str) -> None:
        async with self._state_lock:
            async with self._snapshot_locks.acquire(snapshot_id):
                placement = self._state.snapshots.get(snapshot_id)
                if placement is None or job_id not in placement.leases:
                    return
                handle_to_release = self._lookup_lease_handle(placement, job_id)

        if self._strategy is not None:
            await self._release_strategy_handle(handle_to_release, snapshot_id,
                                                job_id)

        async with self._state_lock:
            async with self._snapshot_locks.acquire(snapshot_id):
                await self._release_lease_locked(snapshot_id, job_id)

    async def _release_lease_locked(self, snapshot_id: str,
                                    job_id: str) -> None:
        placement = self._state.snapshots.get(snapshot_id)
        if placement is None or job_id not in placement.leases:
            return
        if self._strategy is not None:
            # Pop from the map matching this lease's recorded route decision.
            # No cross-map fallback (e.g. an overflow lease that never got a
            # handle because section 2 was cancelled — releasing it must NOT
            # accidentally tear down a sibling shard's durable activity pod).
            lease_path = placement.lease_paths.get(job_id)
            if lease_path == 'overflow':
                placement.overflow_handles.pop(job_id, None)
            elif lease_path == 'durable':
                placement.durable_handles.pop(job_id, None)

        released_at = self._now_utc()
        placement.leases.remove(job_id)
        placement.lease_paths.pop(job_id, None)
        placement.lease_workflow_ids.pop(job_id, None)
        placement.lease_activities.pop(job_id, None)
        self._lease_activity.pop((snapshot_id, job_id), None)
        if not placement.leases:
            placement.last_lease_released = released_at
        self._sync_lifecycle_from_leases(snapshot_id,
                                         final_release_at=released_at)
        await self._persistence.save(self._state)

    async def _release_strategy_handle(self,
                                       handle: Optional[ProvisionedHandle],
                                       snapshot_id: str, job_id: str) -> None:
        assert self._strategy is not None
        request = PlacementRequest(
            snapshot_id=snapshot_id,
            job_id=job_id,
            # Synthetic request for strategy.release(); workflow_id is not
            # consumed by any strategy.release path (release keys on
            # snapshot_id + job_id), so this is a stable filler.
            workflow_id=job_id,
            activity=SnapshotActivity.GEN_TT,
            resource_hint=ResourceHint(),
        )
        await self._strategy.release(request, handle)

    @rpc_handler
    @metrics.count_rpc_errors('report_status')
    async def report_status(self, report: WorkerStatusReport) -> None:
        """Record a worker's liveness + state snapshot, with an optional
        lifecycle event.

        Called by each worker's daemon on every heartbeat interval, and
        ad-hoc when a lifecycle event occurs (snapshot ready, job
        started/finished, etc.). ``report.state`` is the source of
        truth for what's on the worker — the controller reconciles its
        own records against it. ``report.event``, when set, is
        an immediate notification of a specific transition; omit it on
        pure heartbeats.

        Fire-and-forget from the caller's side: no return value, no
        caller obligations beyond sending the next report on schedule.
        Missed heartbeats past the grace window mark the worker
        STALE/DEAD and stop routing to it.
        """
        now = self._time()
        pool_worker_uid = report.worker_name
        state = report.state
        is_activity_scoped = report.worker_storage_mode.is_activity_scoped
        existing = self._worker_inventory.get(pool_worker_uid)
        is_draining_report = (existing is not None
                              and existing.status == WorkerStatus.DRAINING)
        silent_suspects = self._silent_worker_suspects()
        was_silent_suspect = pool_worker_uid in silent_suspects
        silent_suspect_cleared = False
        is_first_worker_report = (not is_activity_scoped and
                                  (pool_worker_uid not in self._worker_inventory
                                   or was_silent_suspect))
        gpu_slots_total = 0
        gpu_slots_in_use = 0
        if not is_activity_scoped:
            # (1) Daemon inventory update — best-effort, in-memory.
            # A worker the reconcile path has committed to tear down
            # (DRAINING) must not be flipped back to READY by a heartbeat that
            # arrives before its daemon is actually scaled down — that would
            # re-expose it to routing in the delete window. Every other status
            # (READY/STALE/DEAD) is heartbeat-derived, so a fresh heartbeat is
            # allowed to bring a worker back to READY (a self-healed daemon
            # re-registering within the DEAD window clears its own staleness).
            preserved_status = (WorkerStatus.DRAINING if existing is not None
                                and existing.status == WorkerStatus.DRAINING
                                else WorkerStatus.READY)
            is_draining_report = preserved_status == WorkerStatus.DRAINING
            gpu_slots_total = self._persisted_gpu_slots_total_for_worker(
                pool_worker_uid)
            gpu_slots_in_use = self._gpu_slots_in_use_for_worker(
                pool_worker_uid)
            self._worker_inventory[pool_worker_uid] = WorkerInventory(
                pool_worker_uid=pool_worker_uid,
                state=state,
                status=preserved_status,
                gpu_slots_total=gpu_slots_total,
                gpu_slots_in_use=gpu_slots_in_use,
                last_heartbeat=now,
            )
            silent_suspect_cleared = (silent_suspects.pop(
                pool_worker_uid, None) is not None)
        log.debug(
            'report_status from %s: %d ready, %d in-progress, '
            'pvc_free=%d/%d, '
            'gpu=%d/%d, event=%s', pool_worker_uid, len(state.ready_snapshots),
            len(state.in_progress_snapshots), state.pvc_free_bytes,
            state.pvc_total_bytes, gpu_slots_in_use, gpu_slots_total,
            report.event.type.value if report.event else 'none')

        if silent_suspect_cleared:
            async with self._state_lock:
                await self._persistence.save(self._state)

        if is_draining_report:
            # A completed L2 upload remains durable and must stay visible to
            # compliance deletion even if worker teardown won the race. Other
            # late local events are ignored so they cannot recreate scrubbed
            # placement/PVC state. This is intentionally fail-closed: once
            # teardown commits, even a still-alive worker cannot rejoin with
            # warm L1 data; any reusable snapshot is restored from L2 instead.
            if (report.event is not None and report.event.type
                    == WorkerEventType.SNAPSHOT_PERSIST_UPLOADED):
                await self._apply_worker_event(
                    pool_worker_uid,
                    report.event,
                    worker_storage_mode=report.worker_storage_mode,
                )
            return

        if is_first_worker_report:
            await self._reconcile_missing_local_copies_from_first_heartbeat(
                pool_worker_uid,
                set(state.ready_snapshots)
                | set(state.in_progress_snapshots),
            )

        if not is_activity_scoped:
            # Positive heartbeat reconciliation catches missed SNAPSHOT_READY
            # events and daemon startup-scan discoveries. We only reconcile
            # snapshots reported as READY here; treating missing entries as evicted
            # needs additional worker-failure/rebind policy.
            await self._reconcile_ready_snapshots_from_heartbeat(
                pool_worker_uid, state.ready_snapshots)

        if report.event is not None:
            await self._apply_worker_event(
                pool_worker_uid, report.event,
                worker_storage_mode=report.worker_storage_mode)

        if not is_activity_scoped:
            self._worker_capacity_changed.set()
            self._worker_capacity_changed = asyncio.Event()

    @rpc_handler
    @metrics.count_rpc_errors('get_pool_stats')
    async def get_pool_stats(
            self, filter: Optional[PoolStatsFilter] = None) -> PoolStats:
        """Return a point-in-time snapshot of pool-wide aggregate stats.

        Read-only; safe to poll at high frequency. Intended for monitoring
        dashboards and Prometheus exporters.
        """
        stats_filter = filter or PoolStatsFilter()
        allowed_workers = (set(stats_filter.pool_worker_uids) if
                           stats_filter.pool_worker_uids is not None else None)
        inventory = list(self._worker_inventory_for_stats().values())
        if allowed_workers is not None:
            inventory = [
                inv for inv in inventory
                if inv.pool_worker_uid in allowed_workers
            ]
        snapshot_items = list(self._state.snapshots.items())
        if allowed_workers is not None:
            snapshot_items = [
                item for item in snapshot_items
                if item[1].worker in allowed_workers
            ]
        snapshots = [snapshot for _, snapshot in snapshot_items]
        snapshot_ids = {snapshot_id for snapshot_id, _ in snapshot_items}
        local_copies = [
            deepcopy(local)
            for _key, local in sorted(self._state.local_copies.items())
        ]
        snapshot_compliance = [
            deepcopy(record)
            for _key, record in sorted(self._state.snapshot_compliance.items())
        ]
        if allowed_workers is not None:
            local_copies = [
                local for local in local_copies
                if local.pool_worker_uid in allowed_workers
            ]
            local_copy_snapshot_ids = {
                local.snapshot_id
                for local in local_copies
            }
            snapshot_compliance = [
                record for record in snapshot_compliance
                if record.snapshot_id in snapshot_ids
                or record.snapshot_id in local_copy_snapshot_ids
            ]
        dataplane_backups = len(self._state.dataplane_backups)
        if allowed_workers is not None:
            dataplane_backups = sum(
                1 for snapshot_id in self._state.dataplane_backups
                if snapshot_id in snapshot_ids)
        active_lease_counts_by_worker: Dict[str, int] = {}
        for placement in snapshots:
            if not placement.worker:
                continue
            active_lease_counts_by_worker[placement.worker] = (
                active_lease_counts_by_worker.get(placement.worker, 0) +
                len(placement.leases))

        live_local_snapshot_ids_by_worker: Dict[str, set[str]] = {}
        for local in self._state.local_copies.values():
            if local.local_state == SnapshotLocalCopyState.EVICTED:
                continue
            if allowed_workers is not None and (local.pool_worker_uid
                                                not in allowed_workers):
                continue
            live_local_snapshot_ids_by_worker.setdefault(
                local.pool_worker_uid, set()).add(local.snapshot_id)

        snapshot_ids_by_worker: Dict[str, set[str]] = {}
        for snapshot_id, snapshot in snapshot_items:
            if not snapshot.worker:
                continue
            snapshot_ids_by_worker.setdefault(snapshot.worker,
                                              set()).add(snapshot_id)
        for pool_worker_uid, local_snapshot_ids in (
                live_local_snapshot_ids_by_worker.items()):
            snapshot_ids_by_worker.setdefault(pool_worker_uid,
                                              set()).update(local_snapshot_ids)

        durable_workers = [
            PoolWorkerStats(
                pool_worker_uid=inv.pool_worker_uid,
                health=inv.status,
                pvc_free_bytes=inv.state.pvc_free_bytes,
                pvc_total_bytes=inv.state.pvc_total_bytes,
                gpu_slots_total=self._persisted_gpu_slots_total_for_worker(
                    inv.pool_worker_uid, inv.gpu_slots_total),
                gpu_slots_in_use=self._gpu_slots_in_use_for_worker(
                    inv.pool_worker_uid),
                active_leases=active_lease_counts_by_worker.get(
                    inv.pool_worker_uid, 0),
                snapshot_ids=(sorted(
                    snapshot_ids_by_worker.get(inv.pool_worker_uid, set()))
                              if stats_filter.include_snapshots else []),
            )
            for inv in sorted(inventory, key=lambda item: item.pool_worker_uid)
        ]
        return PoolStats(
            num_nodes=len(inventory),
            num_healthy_nodes=sum(1 for inv in inventory
                                  if inv.status == WorkerStatus.READY),
            num_snapshots=len(snapshots),
            total_pvc_bytes=sum(inv.state.pvc_total_bytes for inv in inventory),
            free_pvc_bytes=sum(inv.state.pvc_free_bytes for inv in inventory),
            active_leases=sum(len(snapshot.leases) for snapshot in snapshots),
            dataplane_backups=dataplane_backups,
            durable_workers=durable_workers,
            snapshots=([deepcopy(snapshot) for snapshot in snapshots]
                       if stats_filter.include_snapshots else []),
            local_copies=(local_copies
                          if stats_filter.include_snapshots else []),
            snapshot_compliance=(snapshot_compliance
                                 if stats_filter.include_snapshots else []),
        )


class PoolControllerPlacementContext:
    """Adapter exposing controller-owned placement providers to strategies.

    This lives outside the AutoRPC service class so these provider methods do
    not become part of the RPC surface.
    """
    def __init__(self, controller: PoolController) -> None:
        self._controller = controller

    @property
    def worker_inventory(self) -> Dict[str, WorkerInventory]:
        """Inventory with ``gpu_slots_in_use`` overridden from lease state.

        The controller knows in_use authoritatively from approved leases.
        Callers must invoke this under ``_state_lock`` so concurrent shards
        see each other's lease appends; ``place_snapshot`` holds the lock
        across both ``strategy.route()`` and the lease append for exactly
        this reason. Without the lock-hold, N shards each read ``in_use=0``
        and all route to the same warm worker.
        """
        base = self._controller.worker_inventory
        synthesized: Dict[str, WorkerInventory] = {}
        for pool_worker_uid, inventory in base.items():
            in_use = self._controller._gpu_slots_in_use_for_worker(
                pool_worker_uid)
            total = self._controller._persisted_gpu_slots_total_for_worker(
                pool_worker_uid, inventory.gpu_slots_total)
            synthesized[pool_worker_uid] = replace(
                inventory,
                gpu_slots_total=total,
                gpu_slots_in_use=in_use,
            )
        return synthesized

    def warm_location_uid(self, snapshot_id: str) -> Optional[str]:
        return self._controller._warm_location_uid(snapshot_id)

    def snapshot_size_bytes(self, snapshot_id: str) -> Optional[int]:
        return self._controller._snapshot_size_bytes(snapshot_id)

    def capacity_tier_for_worker(self, pool_worker_uid: str):
        from pool.k8s_strategy import CapacityTier

        placement_state = self._controller._state.placement_state
        if not isinstance(placement_state, K8sPlacementState):
            return None
        pvc_state = placement_state.pvcs.get(pool_worker_uid)
        if pvc_state is None or pvc_state.tier_ssd_gib <= 0:
            return None
        return CapacityTier(
            tier_ssd_gib=pvc_state.tier_ssd_gib,
            tier_gpu_count=max(pvc_state.tier_gpu_count, 1),
        )

    def reserved_size_bytes_for_worker(self, pool_worker_uid: str) -> int:
        """Bytes for in-flight placements (RESERVED, INGESTING, RESTORING) on
        this worker that may not yet appear in the daemon's pvc_free_bytes
        heartbeat — subtract when computing effective available space.
        INGESTING/RESTORING are included because the daemon writes data before
        the next heartbeat, so pvc_free_bytes lags behind real usage.
        """
        _IN_FLIGHT = {
            SnapshotState.RESERVED,
            SnapshotState.INGESTING,
            SnapshotState.RESTORING,
        }
        return sum(p.size_bytes
                   for p in self._controller._state.snapshots.values()
                   if p.worker == pool_worker_uid and p.state in _IN_FLIGHT)


PlacementStrategyFactory = Callable[[PoolController], PlacementStrategy]

_POOL_STORAGE_CLASS_NAME_ENV = 'POOL_STORAGE_CLASS_NAME'

# Substrate wiring stamped onto pool pods (see K8sPoolWorkerManager). The chart
# sets these on the pool-controller for runtime-created worker pods.
_POOL_WORKER_ENV_CONFIGMAP_ENV = 'POOL_WORKER_ENV_CONFIGMAP'
_POOL_WORKER_TEMPORAL_CERT_SECRET_ENV = 'POOL_WORKER_TEMPORAL_CERT_SECRET'
_POOL_WORKER_TENANT_SECRET_SECRET_ENV = 'POOL_WORKER_TENANT_SECRET_SECRET'
_POOL_WORKER_TENANT_SECRET_KEY_ENV = 'POOL_WORKER_TENANT_SECRET_KEY'
_POOL_WORKER_STATEDB_SECRET_ENV = 'POOL_WORKER_STATEDB_SECRET'
_POOL_WORKER_STATEDB_CA_SECRET_ENV = 'POOL_WORKER_STATEDB_CA_SECRET'
_POOL_WORKER_STATEDB_CA_SECRET_KEY_ENV = 'POOL_WORKER_STATEDB_CA_SECRET_KEY'
_POOL_WORKER_GCP_APPLICATION_CREDENTIALS_SECRET_ENV = (
    'POOL_WORKER_GCP_APPLICATION_CREDENTIALS_SECRET')
_POOL_WORKER_GCP_APPLICATION_CREDENTIALS_KEY_ENV = (
    'POOL_WORKER_GCP_APPLICATION_CREDENTIALS_KEY')
_POOL_WORKER_POD_LABELS_ENV = 'POOL_WORKER_POD_LABELS_JSON'
_POOL_WORKER_SERVICE_ACCOUNT_ENV = 'POOL_WORKER_SERVICE_ACCOUNT'
_POOL_WORKER_FS_GROUP_ENV = 'POOL_WORKER_FS_GROUP'


def _configured_tenant_secret_key_ref() -> Optional[SecretKeyRef]:
    """Tenant secret-key coordinates from the chart, or None if unset.

    Both halves (Secret name + data key) are required to reference a
    value; a half-set pair is a chart bug, so warn and treat as unset
    rather than rendering a broken secretKeyRef into every pool pod.
    """
    from pool.worker_manager import SecretKeyRef
    secret_name = os.environ.get(_POOL_WORKER_TENANT_SECRET_SECRET_ENV,
                                 '').strip()
    key = os.environ.get(_POOL_WORKER_TENANT_SECRET_KEY_ENV, '').strip()
    if not (secret_name or key):
        return None
    if not (secret_name and key):
        # Static message only: CodeQL's clear-text-logging heuristic treats
        # any value derived from *SECRET* identifiers as sensitive.
        log.warning(
            'Ignoring half-configured tenant secret ref: set BOTH '
            'POOL_WORKER_TENANT_SECRET_SECRET and '
            'POOL_WORKER_TENANT_SECRET_KEY (exactly one is currently set); '
            'pool pods will not receive the tenant secret key.')
        return None
    return SecretKeyRef(secret_name=secret_name, key=key)


def _k8s_pool_storage_class() -> str:
    return (os.environ.get(_POOL_STORAGE_CLASS_NAME_ENV, '').strip()
            or DEFAULT_K8S_POOL_STORAGE_CLASS)


def _configured_pool_worker_service_account() -> Optional[str]:
    """ServiceAccount for runtime-created workers, or non-chart inheritance."""
    return (os.environ.get(_POOL_WORKER_SERVICE_ACCOUNT_ENV, '').strip()
            or None)


def _configured_pool_worker_fs_group() -> Optional[int]:
    """Runtime PVC group; an explicit empty value delegates to the platform."""
    if _POOL_WORKER_FS_GROUP_ENV not in os.environ:
        return 999
    value = os.environ[_POOL_WORKER_FS_GROUP_ENV].strip()
    return int(value) if value else None


def _diskgraph_config(config: DeployEnvConfig) -> DiskgraphConfig:
    data_compute = config.data_compute
    if not isinstance(data_compute, DiskgraphConfig):
        raise ValueError('Pool controller requires diskgraph data_compute; got '
                         f'{type(data_compute).__name__}.')
    return data_compute


def _k8s_pool_config(
        config: DeployEnvConfig) -> Optional[K8sPoolStrategyConfig]:
    """Return the k8s-specific pool sub-config, or None if unset.

    Returns None (rather than fabricating an empty default) because
    K8sPoolStrategyConfig now has a required field (pool_diskgraph_node_types)
    and can't be constructed bare. Callers that only read optional sub-fields
    (heartbeat thresholds, ssd utilization) treat None as "all defaults".
    """
    return _diskgraph_config(config).k8s_pool


def _configured_lifecycle_policy(
        config: DeployEnvConfig) -> SnapshotLifecyclePolicy:
    defaults = SnapshotLifecyclePolicy()
    dg = _diskgraph_config(config)
    k8s = _k8s_pool_config(config)
    # Preserve explicit ordering validation, but keep an omitted pressure
    # window within a shorter configured hard-expiry TTL.
    # This makes max-only configuration safe without changing the 24h default.
    max_ssd_idle_secs = (defaults.max_ssd_idle_secs if dg.pool_max_ssd_idle_secs
                         is None else dg.pool_max_ssd_idle_secs)
    min_ssd_idle_secs = (min(defaults.min_ssd_idle_secs, max_ssd_idle_secs)
                         if dg.pool_min_ssd_idle_secs is None else
                         dg.pool_min_ssd_idle_secs)
    return SnapshotLifecyclePolicy(
        min_ssd_idle_secs=min_ssd_idle_secs,
        max_ssd_idle_secs=max_ssd_idle_secs,
        max_dataplane_retention_secs=(defaults.max_dataplane_retention_secs
                                      if dg.pool_max_dataplane_retention_secs
                                      is None else
                                      dg.pool_max_dataplane_retention_secs),
        ssd_max_utilization=(defaults.ssd_max_utilization if k8s is None
                             or k8s.pool_ssd_max_utilization is None else
                             k8s.pool_ssd_max_utilization),
    )


def _configured_lease_sweep_interval_secs(config: DeployEnvConfig) -> float:
    value = _diskgraph_config(config).pool_lease_sweep_interval_secs
    return LEASE_DEFENSIVE_SWEEP_INTERVAL_SECS if value is None else value


def _configured_worker_stale_threshold_secs(config: DeployEnvConfig) -> float:
    k8s = _k8s_pool_config(config)
    value = k8s.pool_worker_stale_threshold_secs if k8s else None
    return WORKER_STALE_THRESHOLD_SECS if value is None else value


def _configured_worker_dead_threshold_secs(config: DeployEnvConfig) -> float:
    k8s = _k8s_pool_config(config)
    value = k8s.pool_worker_dead_threshold_secs if k8s else None
    return WORKER_DEAD_THRESHOLD_SECS if value is None else value


def _configured_maintenance_interval_secs(config: DeployEnvConfig) -> float:
    value = _diskgraph_config(config).pool_maintenance_interval_secs
    return CONTROLLER_MAINTENANCE_INTERVAL_SECS if value is None else value


def _configured_tombstone_gc_grace_secs(config: DeployEnvConfig) -> float:
    value = _diskgraph_config(config).pool_tombstone_gc_grace_secs
    return TOMBSTONE_GC_GRACE_SECS if value is None else value


# Subset of RFC 1123 (k8s label / namespace name shape) with length
# capped at 48 so the derived ``kumo-dataplane-<id>`` namespace stays
# under k8s's 63-char cap. Fail-fast here surfaces a clear message
# rather than a cryptic kubectl ``invalid DNS name`` error mid-flight.
_K8S_TENANT_RE = re.compile(r'^[a-z0-9][a-z0-9-]{0,47}$')


def _configured_k8s_pool_names(config: DeployEnvConfig, ) -> tuple[str, str]:
    configured_tenant = os.environ.get('POOL_WORKER_TENANT', '').strip()
    tenant = configured_tenant or re.sub(
        r'[^a-z0-9-]', '-', config.id_suffix.lower()).lstrip('-')
    if not _K8S_TENANT_RE.match(tenant):
        raise ValueError(
            f'Diskgraph k8s pool tenant must match {_K8S_TENANT_RE.pattern} '
            f'— RFC 1123 lowercase alphanumeric + "-", starting with '
            f'alphanumeric, ≤48 chars (so the derived "kumo-dataplane-<id>" '
            f'namespace stays under k8s\'s 63-char cap). Derived tenant '
            f'{tenant!r} from {config.deploy_mode.value} '
            f'id_suffix={config.id_suffix!r}.')
    namespace = os.environ.get('POD_NAMESPACE') or f'kumo-dataplane-{tenant}'
    return tenant, namespace


def _empty_state_for_strategy(config: DeployEnvConfig) -> PoolState:
    diskgraph = _diskgraph_config(config)
    if diskgraph.pool_strategy == 'singlehost':
        return PoolState(placement_state=SinglehostPlacementState())
    if diskgraph.pool_strategy == 'k8s':
        return PoolState(placement_state=K8sPlacementState())
    raise ValueError(f'Unknown diskgraph pool strategy: '
                     f'{diskgraph.pool_strategy}')


def _normalize_state_for_strategy(config: DeployEnvConfig,
                                  state: PoolState) -> Optional[PoolState]:
    diskgraph = _diskgraph_config(config)
    if (diskgraph.pool_strategy != 'singlehost'
            or state.placement_state.type == 'singlehost'):
        return None
    if not (config.deploy_mode.is_dev or config.deploy_mode.is_test):
        log.warning(
            'Loaded pool_state.json with placement=%s for singlehost pool '
            'outside local/test deploy mode; leaving persisted state '
            'unchanged.',
            state.placement_state.type,
        )
        return None

    log.warning(
        'Loaded pool_state.json with placement=%s for singlehost pool; '
        'rewriting placement state and dropping %d placement records, '
        '%d local copies, and %d compliance records. Preserving %d '
        'dataplane backups.',
        state.placement_state.type,
        len(state.snapshots),
        len(state.local_copies),
        len(state.snapshot_compliance),
        len(state.dataplane_backups),
    )
    return PoolState(
        placement_state=SinglehostPlacementState(),
        dataplane_backups=dict(state.dataplane_backups),
    )


def _scrub_unbacked_k8s_placement_ghosts(state: PoolState) -> bool:
    """Remove legacy placement homes with no persisted ownership evidence.

    A lease, a PVC record, or a non-evicted local-copy record is sufficient to
    retain a placement. A lease-less placement whose worker has none of those
    cannot be healed by silent-worker retirement (the worker is not expected)
    and causes routing to co-locate forever onto a ghost UID. Startup is the
    safe point to remove it before daemon heartbeats reconcile physical copies.
    """
    placement_state = state.placement_state
    if not isinstance(placement_state, K8sPlacementState):
        return False

    removed = False
    for snapshot_id, placement in list(state.snapshots.items()):
        pool_worker_uid = placement.worker
        if placement.leases or not pool_worker_uid:
            continue
        if pool_worker_uid in placement_state.pvcs:
            continue
        local = state.local_copies.get(
            _local_copy_key(snapshot_id, pool_worker_uid))
        if (local is not None
                and local.local_state != SnapshotLocalCopyState.EVICTED):
            continue
        log.warning(
            'Removing unbacked legacy snapshot placement during startup: '
            'snapshot=%s worker=%s', snapshot_id, pool_worker_uid)
        del state.snapshots[snapshot_id]
        removed = True
    return removed


def _configured_strategy_factory(
        config: DeployEnvConfig) -> PlacementStrategyFactory:
    diskgraph = _diskgraph_config(config)
    if diskgraph.pool_strategy == 'singlehost':
        from pool.strategy_singlehost import SinglehostPlacementStrategy
        from pool.task_queues import SINGLEHOST_DAEMON_TQ

        # Unlike the static worker UID, the Temporal queue must be scoped to
        # this devserver.  Otherwise two local controllers sharing Temporal
        # can dispatch placement work to either process.
        task_queue = config.task_queue(SINGLEHOST_DAEMON_TQ)
        return lambda _: SinglehostPlacementStrategy(task_queue=task_queue)
    if diskgraph.pool_strategy == 'k8s':
        return _configured_k8s_strategy_factory(config, diskgraph)
    raise ValueError(f'Unknown diskgraph pool strategy: '
                     f'{diskgraph.pool_strategy}')


def _pool_resource_spec(
    resources: PoolPodResources,
    *,
    gpu_count: int = 0,
) -> Optional[ResourceSpec]:
    """Translate CPU/memory config without changing GPU placement policy."""
    if not resources.requests and not resources.limits and not gpu_count:
        return None
    return ResourceSpec(
        cpu_request=resources.requests.get('cpu'),
        cpu_limit=resources.limits.get('cpu'),
        memory_request=resources.requests.get('memory'),
        memory_limit=resources.limits.get('memory'),
        gpu_count=gpu_count,
    )


def _build_configured_k8s_manager(
    config: DeployEnvConfig,
    diskgraph: DiskgraphConfig,
):
    """Construct the k8s pool worker manager + its tier/PVC inputs.

    Shared by :func:`_configured_k8s_strategy_factory` (which wraps it in
    a placement strategy) and :func:`run_configured_init_canary` (which
    validates those exact tiers / PVC sizing at startup) so both agree on
    the manager, the cache capacity tiers, and the PVC-spec factory without
    duplicating the construction.

    Returns ``(manager, sizes_gib, capacity_tiers, manager_pvc_spec)``.
    """
    from kubernetes.client import V1Toleration

    from pool.k8s_strategy import (
        POOL_DAEMON_ARGS,
        POOL_DAEMON_COMMAND,
        CapacityTier,
    )
    from pool.worker_manager import K8sPoolWorkerManager
    from pool.worker_manager import PvcSpec as WorkerManagerPvcSpec

    tenant, namespace = _configured_k8s_pool_names(config)
    # PIN the Python kubernetes client to the right cluster context. The
    # kubectl ``current-context`` is mutable global state (devserver / ``aws
    # eks update-kubeconfig`` flip it), which has silently bound the
    # controller to the wrong cluster before. Override via
    # ``POOL_K8S_CLUSTER``; in-cluster runs ignore this because
    # ``load_incluster_config`` succeeds first.
    kube_context = os.environ.get('POOL_K8S_CLUSTER', 'testing-uw4')
    if diskgraph.k8s_pool is None:
        raise ValueError(
            "pool_strategy='k8s' requires data_compute.k8s_pool with "
            'pool_diskgraph_node_types.')
    k8s_pool = diskgraph.k8s_pool
    node_types: dict[str, DiskgraphNodeType] = dict(
        k8s_pool.pool_diskgraph_node_types)
    # Capacity tier is (ssd_gib, gpu_count). Config keys remain opaque platform
    # names; duplicate capacity tuples are intentionally interchangeable.
    capacity_tiers = sorted({
        CapacityTier(tier_ssd_gib=nt.ssd_gib, tier_gpu_count=nt.gpu_count)
        for nt in node_types.values()
    })
    sizes_gib: list[int] = sorted({nt.ssd_gib for nt in node_types.values()})
    largest_ssd_gib = max(sizes_gib)
    manager = K8sPoolWorkerManager(
        namespace=namespace,
        tenant=tenant,
        storage_class=_k8s_pool_storage_class(),
        daemon_image=daemon_image_uri(),
        daemon_command=POOL_DAEMON_COMMAND,
        daemon_args=POOL_DAEMON_ARGS,
        daemon_resources=_pool_resource_spec(
            k8s_pool.pool_resources.effective_daemon()),
        pool_extra_tolerations=[
            V1Toleration(key=t.key, operator=t.operator or 'Equal',
                         value=t.value, effect=t.effect,
                         toleration_seconds=t.toleration_seconds)
            for t in k8s_pool.pool_extra_tolerations
        ],
        kube_context=kube_context,
        # Fallback for rebind path (PVC already exists; size is ignored by K8s).
        default_storage_request=f'{largest_ssd_gib}Gi',
        # Keep runtime data-plane identity separate from the controller's
        # Kubernetes-management identity. None preserves POD_SERVICE_ACCOUNT
        # or default resolution for non-chart deployments.
        service_account=_configured_pool_worker_service_account(),
        runtime_fs_group=_configured_pool_worker_fs_group(),
        # Substrate wiring for the runtime-created pool pods. Secret/ConfigMap
        # names come from the chart (env); the Temporal cert dir comes from the
        # deploy config (empty for Temporal Cloud => no cert mount, correct).
        pod_env_configmap=(os.environ.get(_POOL_WORKER_ENV_CONFIGMAP_ENV,
                                          '').strip() or None),
        temporal_cert_secret=(os.environ.get(
            _POOL_WORKER_TEMPORAL_CERT_SECRET_ENV, '').strip() or None),
        temporal_cert_dir=(config.tls_cert_dir or None),
        tenant_secret_key_ref=_configured_tenant_secret_key_ref(),
        # Password auth only: the Secret name comes from the chart, the mount
        # path from the deploy config, so both halves agree with what the app
        # resolves dsn_secret_name against.
        statedb_secret=(os.environ.get(_POOL_WORKER_STATEDB_SECRET_ENV,
                                       '').strip() or None),
        statedb_ca_secret=(os.environ.get(_POOL_WORKER_STATEDB_CA_SECRET_ENV,
                                          '').strip() or None),
        statedb_ca_secret_key=(os.environ.get(
            _POOL_WORKER_STATEDB_CA_SECRET_KEY_ENV, '').strip() or None),
        secrets_mount_path=(deploy_env().secrets.mount_path or None),
        gcp_application_credentials_secret=(os.environ.get(
            _POOL_WORKER_GCP_APPLICATION_CREDENTIALS_SECRET_ENV,
            '').strip() or None),
        gcp_application_credentials_secret_key=(os.environ.get(
            _POOL_WORKER_GCP_APPLICATION_CREDENTIALS_KEY_ENV,
            '').strip() or None),
        gcp_application_credentials_path=(os.environ.get(
            'GOOGLE_APPLICATION_CREDENTIALS', '').strip() or None),
        pod_template_labels=json.loads(
            os.environ.get(_POOL_WORKER_POD_LABELS_ENV, '{}')),
    )

    def manager_pvc_spec(chosen_ssd_gib: int) -> WorkerManagerPvcSpec:
        return WorkerManagerPvcSpec(size_gib=chosen_ssd_gib)

    return manager, sizes_gib, capacity_tiers, manager_pvc_spec


async def _registered_cache_tier_keys(
    controller: PoolController,
    manager: 'K8sPoolWorkerManager',
    workers: Iterable['PoolWorkerInfo'],
    *,
    configured_tiers: Iterable['CapacityTier'],
    pvc_spec_factory: Callable[[int], PvcSpec],
) -> set[tuple[int, int]]:
    """Configured tiers proven by current Registered pod/PVC envelopes."""
    configured = {(tier.tier_ssd_gib, tier.tier_gpu_count)
                  for tier in configured_tiers}
    placement_state = controller._state.placement_state
    if not isinstance(placement_state, K8sPlacementState):
        return set()
    result: set[tuple[int, int]] = set()
    for worker in workers:
        if worker.status != 'Registered':
            continue
        pvc_state = placement_state.pvcs.get(worker.pool_worker_uid)
        if (pvc_state is None or pvc_state.tier_ssd_gib <= 0
                or pvc_state.tier_gpu_count <= 0):
            continue
        key = (pvc_state.tier_ssd_gib, pvc_state.tier_gpu_count)
        if key not in configured or key in result:
            continue
        spec = pvc_spec_factory(pvc_state.tier_ssd_gib)
        spec.gpu_count = pvc_state.tier_gpu_count
        if await manager.registered_worker_matches_current_cache_tier(
                worker, spec):
            result.add(key)
    return result


async def run_configured_init_canary(
    config: DeployEnvConfig,
    controller: PoolController,
) -> None:
    """Apply init-time node-pool capability validation to the controller.

    No-op for the singlehost / testing strategies (no external node pools
    to validate). For the ``k8s`` strategy, a Registered daemon with an exact
    persisted tier and matching live pod/PVC envelope is accepted as proof;
    unproven tiers run the active canary. Unavailable capabilities are removed
    from new placement before RPC serving.
    """
    diskgraph = _diskgraph_config(config)
    if diskgraph.pool_strategy != 'k8s':
        log.info(
            'Init canary: pool_strategy=%s has no external node pools '
            'to validate; skipping.', diskgraph.pool_strategy)
        return
    from pool.k8s_strategy import K8sPlacementStrategy
    from pool.worker_manager.init_canary import run_init_canary
    manager, _sizes_gib, capacity_tiers, manager_pvc_spec = (
        _build_configured_k8s_manager(config, diskgraph))
    canary_pvc_specs = []
    for tier in capacity_tiers:
        spec = manager_pvc_spec(tier.tier_ssd_gib)
        spec.gpu_count = tier.tier_gpu_count
        canary_pvc_specs.append(spec)

    async def registered_cache_tier_keys() -> set[tuple[int, int]]:
        workers = await manager.list_pool_workers()
        return await _registered_cache_tier_keys(
            controller,
            manager,
            workers,
            configured_tiers=capacity_tiers,
            pvc_spec_factory=manager_pvc_spec,
        )

    result = await run_init_canary(
        manager,
        cache_tier_specs=canary_pvc_specs,
        pvc_spec_factory=manager_pvc_spec,
        prevalidated_cache_tiers=registered_cache_tier_keys,
    )
    strategy = controller._strategy
    if not isinstance(strategy, K8sPlacementStrategy):
        raise RuntimeError(
            "pool_strategy='k8s' constructed a non-K8s placement strategy")
    strategy.restrict_to_available_capacity(
        result.available_cache_tiers,
        ephemeral_available=result.ephemeral_available,
    )


def _configured_k8s_strategy_factory(
    config: DeployEnvConfig,
    diskgraph: DiskgraphConfig,
) -> PlacementStrategyFactory:
    from pool.k8s_strategy import build_k8s_placement_strategy

    # Shared with the init canary: the manager (cluster-pinned), the cache
    # capacity tiers, and the PVC-spec factory. See
    # ``_build_configured_k8s_manager`` for the kube-context pinning and
    # tier-collapsing rationale.
    if diskgraph.k8s_pool is None:
        raise ValueError(
            "pool_strategy='k8s' requires data_compute.k8s_pool with "
            'pool_diskgraph_node_types.')
    k8s_pool = diskgraph.k8s_pool
    manager, _sizes_gib, capacity_tiers, manager_pvc_spec = (
        _build_configured_k8s_manager(config, diskgraph))

    def activity_resources(request: PlacementRequest) -> Optional[ResourceSpec]:
        """Shared by cache and ephemeral pods; TRAIN/PREDICT still get 1 GPU."""
        resources = k8s_pool.pool_resources
        if request.activity == SnapshotActivity.TRAIN:
            return _pool_resource_spec(resources.effective_activity('training'),
                                       gpu_count=1)
        if request.activity == SnapshotActivity.PREDICT:
            return _pool_resource_spec(
                resources.effective_activity('prediction'), gpu_count=1)
        return _pool_resource_spec(resources.effective_activity(None))

    from pool.k8s_strategy import DEFAULT_SSD_MAX_UTILIZATION
    ssd_max_utilization = k8s_pool.pool_ssd_max_utilization or DEFAULT_SSD_MAX_UTILIZATION
    activity_poller_waiter = build_temporal_activity_poller_waiter(config)
    return lambda controller: build_k8s_placement_strategy(
        manager,
        PoolControllerPlacementContext(controller),
        capacity_tiers=capacity_tiers,
        ssd_max_utilization=ssd_max_utilization,
        manager_pvc_spec_factory=manager_pvc_spec,
        activity_resources_factory=activity_resources,
        activity_poller_waiter=activity_poller_waiter,
    )


async def create_pool_controller(
    config: DeployEnvConfig,
    *,
    strategy_factory: Optional[PlacementStrategyFactory] = None,
    workflow_status_checker: Optional[WorkflowStatusChecker] = None,
    local_snapshot_deleter: Optional[LocalSnapshotDeleter] = None,
    start_maintenance_loop: bool = True,
) -> PoolController:
    """Construct a controller with persisted state loaded.

    Reads ``pool_state.json`` from the configured dataplane storage root
    (or initializes an empty state on first boot / after data loss) and
    returns a ready-to-run controller. Callers run it under AutoRPC with
    ``temporalio_commons.auto_rpc.build_server`` — see the Block H image
    entry point for the wiring.

    ``strategy_factory`` is the deployment seam: it receives the constructed
    controller so the strategy can read controller-owned providers without
    controller core importing a concrete Block F/Block K implementation.
    Tests can override it; production wiring derives it from
    :func:`_configured_strategy_factory`.
    """
    persistence = PoolStatePersistence(
        config.pool_state_path,
        empty_state_factory=(
            lambda: PoolState(placement_state=K8sPlacementState())
            if strategy_factory is not None else _empty_state_for_strategy(
                config)),
    )
    state = await persistence.load()
    should_scrub_k8s_ghosts = (strategy_factory is not None or
                               _diskgraph_config(config).pool_strategy == 'k8s')
    state_changed = (should_scrub_k8s_ghosts
                     and _scrub_unbacked_k8s_placement_ghosts(state))
    if strategy_factory is None:
        normalized_state = _normalize_state_for_strategy(config, state)
        if normalized_state is not None:
            state = normalized_state
            state_changed = True
    if state_changed:
        await persistence.save(state)
    effective_strategy_factory: PlacementStrategyFactory = (
        strategy_factory if strategy_factory is not None else
        _configured_strategy_factory(config))
    controller = PoolController(
        state=state,
        persistence=persistence,
        workflow_status_checker=(
            workflow_status_checker if workflow_status_checker is not None else
            build_temporal_workflow_status_checker(config)),
        local_snapshot_deleter=(local_snapshot_deleter
                                if local_snapshot_deleter is not None else
                                build_temporal_snapshot_deleter(config)),
        data_plane_storage_root=config.data_plane_storage_root,
        lifecycle_policy=_configured_lifecycle_policy(config),
        lease_sweep_interval_secs=_configured_lease_sweep_interval_secs(config),
        worker_stale_threshold_secs=(
            _configured_worker_stale_threshold_secs(config)),
        worker_dead_threshold_secs=(
            _configured_worker_dead_threshold_secs(config)),
        maintenance_interval_secs=_configured_maintenance_interval_secs(config),
        tombstone_gc_grace_secs=_configured_tombstone_gc_grace_secs(config),
        ingest_space_multiplier=_diskgraph_config(
            config).ingest_space_multiplier,
        snapshot_estimate_buffer_bytes=(
            (_diskgraph_config(config).snapshot_estimate_buffer_gib or 0) *
            (1 << 30)),
    )
    controller._set_placement_strategy(effective_strategy_factory(controller))
    # Strategy-owned background tasks (e.g. K8sPoolWorkerManager's
    # kubeconfig refresh loop, needed off-cluster to dodge the
    # kubernetes-client v24 exec-token cache bug). No-op for strategies
    # that don't define it.
    strategy = controller._strategy
    start_fn = getattr(strategy, 'start', None)
    if start_fn is not None and asyncio.iscoroutinefunction(start_fn):
        try:
            await start_fn()
        except Exception:
            log.exception(
                'Strategy startup hook failed; controller will boot '
                'anyway and the periodic maintenance loop is the backstop.')
    try:
        # Startup reconciliation may clean already-orphaned external objects,
        # but persisted workers must not age before RPC serving starts.
        await controller._reconcile_once(refresh_silent_workers=False)
    except Exception:
        log.exception('Placement strategy startup reconcile failed; serving on '
                      'loaded pool state. Periodic maintenance will retry.')
    if start_maintenance_loop:
        controller._start_maintenance_loop()
    return controller
