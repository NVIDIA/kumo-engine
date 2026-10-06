# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prometheus metrics for the pool controller.

Metric instances are module-level singletons registered with the default
``prometheus_client`` registry. Each controller process is single-replica and
single-tenant.

Gauges are recomputed in full on every controller maintenance tick
(:meth:`PoolController._maintenance_once`) by :func:`update_state_gauges`, which
sets a value for *every* enum member so a state that drops to zero reports ``0``
rather than leaving a stale series. Counters and histograms are incremented at
their event sites.

``pool.controller_main`` starts the ``prometheus_client`` HTTP exposition
endpoint at boot (see block C3).
"""
import functools
from contextlib import contextmanager
from typing import Awaitable, Callable, Iterator, Mapping, Tuple, TypeVar

from prometheus_client import Counter, Gauge, Histogram

from pool.dataclasses import (
    PlacementResult,
    PoolState,
    SnapshotActivity,
    SnapshotLocalCopyState,
    SnapshotState,
    WorkerInventory,
    WorkerStatus,
)

# --- Gauges (recomputed each tick) ---------------------------------------

POOL_SNAPSHOTS = Gauge(
    'pool_snapshots_total',
    'Number of snapshot placements by lifecycle state.',
    ['state'],
)
POOL_WORKERS = Gauge(
    'pool_workers_total',
    'Number of pool workers by health status.',
    ['status'],
)
POOL_ACTIVE_LEASES = Gauge(
    'pool_active_leases_total',
    'Number of open logical leases across all snapshots.',
    [],
)
POOL_LOCAL_COPIES = Gauge(
    'pool_local_copies_total',
    'Number of L1 local-SSD copies by state.',
    ['state'],
)
POOL_DATAPLANE_BACKUPS = Gauge(
    'pool_dataplane_backups_total',
    'Number of L2 dataplane backups tracked by the controller.',
    [],
)
# Per-worker SSD/PVC capacity from the latest heartbeat. used% / free are
# derived in the dashboard (free = total - used).
POOL_WORKER_SSD_TOTAL_BYTES = Gauge(
    'pool_worker_ssd_total_bytes',
    "Worker SSD/PVC total capacity (bytes), from the worker's last heartbeat.",
    ['pool_worker_uid'],
)
POOL_WORKER_SSD_USED_BYTES = Gauge(
    'pool_worker_ssd_used_bytes',
    "Worker SSD/PVC used bytes, from the worker's last heartbeat.",
    ['pool_worker_uid'],
)
# In-flight (leased) activities by placement target and type, derived from the
# controller's open leases — one open lease == one in-flight activity. Named
# "activities", not "jobs": in Kumo a *job* is a training/prediction workflow,
# whereas these are the Temporal activities that lease a snapshot. ``activity``
# is the SnapshotActivity (ENSURE_SNAPSHOT / GEN_TT / MATERIALIZE_*).
# ``pool_worker_uid`` is the durable home worker the activity runs on, or the
# literal "overflow" for leases routed to ephemeral spill pods (which don't run
# on — and shouldn't be charged to — the snapshot's home worker).
POOL_ACTIVE_ACTIVITIES = Gauge(
    'pool_active_activities_total',
    'In-flight (leased) activities by worker (or "overflow") and type.',
    ['pool_worker_uid', 'activity'],
)
# Open leases split by placement route: durable (on the snapshot's home PVC
# worker) vs overflow (ephemeral emptyDir activity pod on a spill node).
POOL_LEASES_BY_ROUTE = Gauge(
    'pool_leases_by_route_total',
    'Open leases by placement route (durable=home worker, overflow=ephemeral).',
    ['route'],
)
# GPU slot occupancy per worker (heartbeat).
POOL_WORKER_GPU_SLOTS_TOTAL = Gauge(
    'pool_worker_gpu_slots_total',
    'Total GPU slots on a worker (heartbeat).',
    ['pool_worker_uid'],
)
POOL_WORKER_GPU_SLOTS_IN_USE = Gauge(
    'pool_worker_gpu_slots_in_use',
    'GPU slots currently in use on a worker (heartbeat).',
    ['pool_worker_uid'],
)

# --- Histograms ----------------------------------------------------------

POOL_PLACEMENT_DURATION = Histogram(
    'pool_placement_duration_seconds',
    'Wall time of place_snapshot, labelled by placement outcome.',
    ['result'],
)
# No tenant label: persistence is tenant-agnostic and runs in its own module.
POOL_STATE_SAVE_DURATION = Histogram(
    'pool_state_save_duration_seconds',
    'Wall time of a full pool_state write-through.',
)

# --- Counters ------------------------------------------------------------

# prometheus_client appends ``_total`` to counter sample names, so the exposed
# series are ``pool_lifecycle_sweep_deletes_total`` / ``pool_controller_errors_total``.
POOL_LIFECYCLE_SWEEP_DELETES = Counter(
    'pool_lifecycle_sweep_deletes',
    'Deletions performed by the lifecycle sweeps, by layer.',
    ['type'],
)
POOL_CONTROLLER_ERRORS = Counter(
    'pool_controller_errors',
    'Exceptions raised out of controller RPC handlers, by call.',
    ['call'],
)


def update_state_gauges(
    state: PoolState,
    worker_inventory: Mapping[str, WorkerInventory],
    lease_activity: Mapping[Tuple[str, str], SnapshotActivity],
) -> None:
    """Recompute every state gauge.

    ``worker_inventory`` is the controller's live, non-persisted worker table
    (rebuilt from heartbeats); ``lease_activity`` is the controller's in-process
    ``(snapshot_id, job_id) -> SnapshotActivity`` lease mirror. Persisted
    ``SnapshotPlacement.lease_activities`` is preferred when present so restart
    recovery keeps active-activity gauges accurate before new leases arrive.

    Sets a value for every enum member so dropped-to-zero states report ``0``.
    """
    by_snapshot_state = {s: 0 for s in SnapshotState}
    for placement in state.snapshots.values():
        by_snapshot_state[placement.state] = (
            by_snapshot_state.get(placement.state, 0) + 1)
    for snapshot_state, count in by_snapshot_state.items():
        POOL_SNAPSHOTS.labels(state=snapshot_state.value).set(count)

    by_local_state = {s: 0 for s in SnapshotLocalCopyState}
    for local in state.local_copies.values():
        by_local_state[local.local_state] = (
            by_local_state.get(local.local_state, 0) + 1)
    for local_state, count in by_local_state.items():
        POOL_LOCAL_COPIES.labels(state=local_state.value).set(count)

    by_worker_status = {s: 0 for s in WorkerStatus}
    for worker in worker_inventory.values():
        by_worker_status[worker.status] = (
            by_worker_status.get(worker.status, 0) + 1)
    for worker_status, count in by_worker_status.items():
        POOL_WORKERS.labels(status=worker_status.value).set(count)

    open_leases = sum(
        len(placement.leases) for placement in state.snapshots.values())
    POOL_ACTIVE_LEASES.set(open_leases)
    POOL_DATAPLANE_BACKUPS.set(len(state.dataplane_backups))

    # Per-worker SSD/GPU capacity. Workers are ephemeral, so clear stale
    # per-uid series and re-set from the current inventory rather than leaving
    # a departed worker's series frozen.
    for gauge in (POOL_WORKER_SSD_TOTAL_BYTES, POOL_WORKER_SSD_USED_BYTES,
                  POOL_WORKER_GPU_SLOTS_TOTAL, POOL_WORKER_GPU_SLOTS_IN_USE):
        gauge.clear()
    for uid, worker in worker_inventory.items():
        st = worker.state
        POOL_WORKER_SSD_TOTAL_BYTES.labels(pool_worker_uid=uid).set(
            st.ssd_total_bytes)
        POOL_WORKER_SSD_USED_BYTES.labels(pool_worker_uid=uid).set(
            st.ssd_used_bytes)
        POOL_WORKER_GPU_SLOTS_TOTAL.labels(pool_worker_uid=uid).set(
            worker.gpu_slots_total)
        POOL_WORKER_GPU_SLOTS_IN_USE.labels(pool_worker_uid=uid).set(
            worker.gpu_slots_in_use)

    # In-flight (leased) activities by placement target + type, from the
    # controller's own lease bookkeeping (one open lease == one in-flight
    # activity). The worker heartbeat can't report this — activities run in
    # separate pods. Durable leases are charged to the snapshot's home worker;
    # overflow leases run on ephemeral spill pods and bucket under "overflow"
    # rather than the home worker. Series clear each tick (active set shrinks
    # to zero between workloads; departed workers must not freeze).
    POOL_ACTIVE_ACTIVITIES.clear()
    by_worker_activity: dict[tuple[str, str], int] = {}
    for snapshot_id, placement in state.snapshots.items():
        for job_id in placement.leases:
            activity = placement.lease_activities.get(job_id)
            if activity is None:
                activity = lease_activity.get((snapshot_id, job_id))
            if activity is None:
                continue
            durable = placement.lease_paths.get(job_id) == 'durable'
            worker = placement.worker if durable else 'overflow'
            key = (worker, activity.value)
            by_worker_activity[key] = by_worker_activity.get(key, 0) + 1
    for (worker, activity_type), count in by_worker_activity.items():
        POOL_ACTIVE_ACTIVITIES.labels(pool_worker_uid=worker,
                                      activity=activity_type).set(count)

    by_route: dict[str, int] = {'durable': 0, 'overflow': 0}
    for placement in state.snapshots.values():
        for route in placement.lease_paths.values():
            by_route[route] = by_route.get(route, 0) + 1
    for route, count in by_route.items():
        POOL_LEASES_BY_ROUTE.labels(route=route).set(count)


def observe_placement(result: str, duration_secs: float) -> None:
    POOL_PLACEMENT_DURATION.labels(result=result).observe(duration_secs)


def inc_sweep_delete(delete_type: str, count: int = 1) -> None:
    if count:
        POOL_LIFECYCLE_SWEEP_DELETES.labels(type=delete_type).inc(count)


def inc_error(call: str) -> None:
    POOL_CONTROLLER_ERRORS.labels(call=call).inc()


_T = TypeVar('_T')


def count_rpc_errors(
    call: str,
) -> Callable[[Callable[..., Awaitable[_T]]], Callable[..., Awaitable[_T]]]:
    """Decorate a ``PoolController`` async RPC handler so any exception out of
    it increments ``pool_controller_errors_total{call=...}`` then re-raises.

    Applied *below* ``@rpc_handler`` so auto_rpc sees a signature-preserving
    wrapper (``functools.wraps``)."""
    def decorator(
        func: Callable[..., Awaitable[_T]], ) -> Callable[..., Awaitable[_T]]:
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs) -> _T:
            try:
                return await func(self, *args, **kwargs)
            # Only real failures count. CancelledError is BaseException, not
            # Exception, so cooperative cancellation / shutdown propagates
            # uncounted and never trips the errors-rising alert.
            except Exception:
                inc_error(call)
                raise

        return wrapper

    return decorator


def observe_place_snapshot(
    func: Callable[..., Awaitable[PlacementResult]],
) -> Callable[..., Awaitable[PlacementResult]]:
    """Decorate ``PoolController.place_snapshot`` to record
    ``pool_placement_duration_seconds{result=...}`` (result = the returned
    ``PlacementStatus`` value, or ``error``) and count errors. Uses
    ``self._time`` so test clocks drive the histogram deterministically."""
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs) -> PlacementResult:
        start = self._time()
        try:
            result = await func(self, *args, **kwargs)
        # CancelledError (BaseException) propagates without touching either
        # metric — it's neither an error nor a completed placement.
        except Exception:
            inc_error('place_snapshot')
            observe_placement('error', self._time() - start)
            raise
        observe_placement(result.status.value, self._time() - start)
        return result

    return wrapper


@contextmanager
def time_state_save() -> Iterator[None]:
    with POOL_STATE_SAVE_DURATION.time():
        yield
