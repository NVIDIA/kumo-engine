# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataclasses for pool controller state and RPC payloads."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional, Union

from pydantic import Field

from common_utils.time import now_utc
from io_utils import kumo_dataclass
from pool.storage_mode import PoolStorageMode

SnapshotID = str
JobID = str
PoolWorkerUID = str

DEFAULT_MIN_SSD_IDLE_SECS = 4 * 60 * 60
DEFAULT_MAX_SSD_IDLE_SECS = 24 * 60 * 60
DEFAULT_MAX_DATAPLANE_RETENTION_SECS = 7 * 24 * 60 * 60
# Fraction of a worker's physical SSD the pool may use before it's "full".
# The complement (here 15%) is headroom for un-accounted use the controller's
# per-snapshot size bookkeeping doesn't track — ingest staging/tmp, the
# diskgraph index intermediates, logs — plus ops/alerting margin. Expressed as
# a fraction of physical so it auto-scales with the worker instance type.
DEFAULT_SSD_MAX_UTILIZATION = 0.85


@kumo_dataclass
class SnapshotLifecyclePolicy:
    """Controller-wide three-timer lifecycle defaults.

    All three timers start at final lease release:

    - ``min_ssd_idle_secs``: L1 becomes evictable under SSD pressure.
    - ``max_ssd_idle_secs``: L1 is force-evicted even without pressure.
    - ``max_dataplane_retention_secs``: L2 dataplane backup must be deleted.

    ``ssd_max_utilization`` is the orthogonal *capacity* knob (not a timer): the
    fraction of a worker's physical SSD the controller treats as usable when
    deciding SSD pressure. A worker is "under pressure" once used exceeds
    ``physical_total * ssd_max_utilization``, so the complement is headroom for
    un-accounted spillover/tmp + ops margin. It is the configurable replacement
    for the old hardcoded 10%-free pressure threshold.

    Required ordering:
    ``min_ssd_idle_secs <= max_ssd_idle_secs <=
    max_dataplane_retention_secs``. The ordering is load-bearing because the
    compliance sweep waits for live L1 copies to drain before deleting L2; if
    L2 expires before L1's force-evict clock, dataplane deletion can block
    past the customer-facing retention SLA.

    ``max_dataplane_retention_secs`` is the customer-facing compliance
    guarantee. The two SSD timers are deploy-env tuning knobs for cache
    thrash prevention and node scale-down economics.
    """
    min_ssd_idle_secs: int = DEFAULT_MIN_SSD_IDLE_SECS
    max_ssd_idle_secs: int = DEFAULT_MAX_SSD_IDLE_SECS
    max_dataplane_retention_secs: int = (DEFAULT_MAX_DATAPLANE_RETENTION_SECS)
    ssd_max_utilization: float = Field(default=DEFAULT_SSD_MAX_UTILIZATION,
                                       gt=0.0, le=1.0)


class SnapshotActivity(str, Enum):
    # Daemon-hosted; must route to the PVC-hosting worker's daemon queue.
    ENSURE_SNAPSHOT = 'ENSURE_SNAPSHOT'
    # Activity-scoped, single pod on the PVC-hosting worker (writes target
    # tables under {snap}/targets/{job_id}/).
    GEN_TT = 'GEN_TT'
    # Activity-scoped, single pod on the PVC-hosting worker (custom-TT gen:
    # builds the durable staging bundle from the customer parquet). Needs the
    # parent graph index resident, so it is placed via the pool like GEN_TT so
    # ``_ensure_snapshot`` restores the base snapshot before the wheel gen opens
    # ``<data_root>/<graph>/<ver>/index_metadata.json``.
    GEN_CUSTOM_TT = 'GEN_CUSTOM_TT'
    # Activity-scoped, single pod on the PVC-hosting worker (graph-format
    # conversion of the target table for training / prediction).
    MATERIALIZE_TRAINING_TASK = 'MATERIALIZE_TRAINING_TASK'
    MATERIALIZE_PREDICTION_TASK = 'MATERIALIZE_PREDICTION_TASK'
    # Activity-scoped, overflow-eligible: spills onto ephemeral nodes with
    # emptyDir when PVC-hosting worker's GPU capacity is exhausted.
    TRAIN = 'TRAIN'
    PREDICT = 'PREDICT'


class PlacementStatus(str, Enum):
    # Snapshot is on the target node's PVC; dispatch the activity immediately.
    WARM = 'WARM'
    # Dataplane backup exists but no PVC entry; ensure_snapshot(dataplane_path).
    COLD = 'COLD'
    # No PVC entry, no dataplane backup; ensure_snapshot ingests from source.
    # No IN_PROGRESS state: concurrent requests both see COLD/MISSING and
    # serialize on the daemon's per-snapshot lock.
    MISSING = 'MISSING'


@kumo_dataclass
class ResourceHint:
    """Opaque sizing hint forwarded to the worker manager.

    ``min_gpu_slots`` is the durable-home capacity requested by the workflow.
    It defaults to 1 to preserve current single-GPU behavior. A single
    TRAIN/PREDICT pod still requests one GPU; this hint only affects which
    durable cache capacity tier is selected for the snapshot home.

    ``min_ssd_gib`` is a hard lower bound on the worker's physical SSD tier.
    It is normally zero. A local disk-full retry sets it above the tier that
    failed so placement cannot return to the same host size.
    """
    min_gpu_slots: int = 1
    min_ssd_gib: int = 0

    def __post_init__(self) -> None:
        if self.min_gpu_slots < 1:
            raise ValueError(
                f'min_gpu_slots must be >= 1, got {self.min_gpu_slots}')
        if self.min_ssd_gib < 0:
            raise ValueError(
                f'min_ssd_gib must be >= 0, got {self.min_ssd_gib}')


@kumo_dataclass
class PlacementRequest:
    """Input to :meth:`PoolController.place_snapshot`.

    One request represents one activity placement and one logical lease.
    Batch workloads fan out by issuing one request per child workflow/job_id;
    the request intentionally carries no parallelism or estimated job duration.

    ``source_size_bytes`` carries the estimated total source bytes. Zero means
    the source size is unknown. For WARM/COLD snapshots the exact persisted
    ``SnapshotPlacement.size_bytes`` is used instead.

    ``workflow_id`` is the parent Temporal workflow_id (always
    ``workflow.info().workflow_id`` of the caller). It is passed as a
    structured field — rather than parsed back out of ``job_id`` — so the
    controller's dangling-lease sweep can group leases by their owning
    workflow without relying on a string encoding of the shard suffix.
    Sharded callers (sibling activities under one workflow) use distinct
    ``job_id`` values but share the same ``workflow_id``.
    """
    snapshot_id: SnapshotID
    job_id: JobID
    workflow_id: str
    activity: SnapshotActivity
    resource_hint: ResourceHint
    source_size_bytes: int = 0


@kumo_dataclass
class PlacementResult:
    """Placement implies a lease pinning ``job_id`` until
    ``release_lease(snapshot_id, job_id)``.
    """
    snapshot_id: SnapshotID
    task_queue: str
    status: PlacementStatus
    # SSD capacity selected for the durable placement. Zero denotes overflow.
    tier_ssd_gib: int
    # Populated only when ``status == COLD``.
    dataplane_path: Optional[str] = None
    # The durable pool worker this placement landed on, when known. Used by
    # ``run_pooled_activity`` to poll that worker's health (via get_pool_stats)
    # after an activity failure so it can decide STALE-wait / READY-retry /
    # DEAD-reroute. ``None`` for overflow (eph) placements — those run on a
    # throwaway eph pod with no durable worker to poll, so the runner re-places
    # from scratch.
    pool_worker_uid: Optional[PoolWorkerUID] = None
    # Strategy-provided observability metadata. Callers forward
    # opaquely to ``workflow.upsert_memo({f"pool_{activity}": ...})``
    # so URL-shaped values render as clickable links in Temporal
    # Cloud UI. Common keys (strategy-specific):
    #   k8s   → "pod_name", "k8s_namespace", "grafana_logs_url"
    #   single → "pid", "log_file"
    debug_info: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProvisionedHandle:
    """Output of :meth:`PlacementStrategy.provision`; persisted on
    :class:`SnapshotPlacement` so :meth:`PlacementStrategy.release` and
    :meth:`PlacementStrategy.wait_provision_ready` can be called later.

    ``provision()`` returns immediately after creating the underlying
    resource (k8s Job / daemon Deployment).  The caller must invoke
    ``wait_provision_ready()`` before dispatching activities — and must
    persist this handle to ``SnapshotPlacement`` *before* that call so
    that ``release_lease`` can clean up the resource even if the wait
    raises or the controller crashes mid-wait.
    """

    # --- generic fields (all strategies) ---
    kind: Literal['durable', 'overflow']
    task_queue: str
    pool_worker_uid: Optional[PoolWorkerUID] = None
    provision_state: Literal['provisioning', 'ready'] = 'ready'
    # Observability metadata propagated into PlacementResult.debug_info.
    # Populated at provision() time where possible; enriched with pod_name
    # by wait_provision_ready() once the pod is Running.
    debug_info: Dict[str, str] = field(default_factory=dict)

    # --- k8s-strategy-specific fields; None for singlehost / testing ---
    # (no fields currently — cleanup uses job_id label selector via
    #  delete_activity_pods; wait uses job_id to derive k8s job name)


class SnapshotState(str, Enum):
    """Lifecycle state of a snapshot's placement on a per-node PVC.

    ::

        RESERVED  →  INGESTING  →  READY  →  (evicted: placement removed)
                  \\→  RESTORING →  ↗

    ``RESERVED``:  placement created by the controller (atomic reservation
                   on the target node); ``ensure_snapshot`` activity not yet
                   started on the daemon.
    ``INGESTING``: daemon reports it has begun a fresh ingest from source.
    ``RESTORING``: daemon reports it has begun restoring from dataplane backup.
    ``READY``:     snapshot subdirectory is on the PVC and serving jobs.

    There is no explicit terminal state — eviction removes the
    ``SnapshotPlacement`` entry entirely rather than marking it evicted.
    """
    RESERVED = 'RESERVED'
    INGESTING = 'INGESTING'
    RESTORING = 'RESTORING'
    READY = 'READY'


class RetentionMode(str, Enum):
    EXPLICIT_TTL = 'EXPLICIT_TTL'
    DEFAULT = 'DEFAULT'
    JOB_BOUND = 'JOB_BOUND'


class SnapshotLocalCopyState(str, Enum):
    WARM = 'WARM'
    EVICTION_CANDIDATE = 'EVICTION_CANDIDATE'
    EVICTING = 'EVICTING'
    EVICTED = 'EVICTED'


@kumo_dataclass
class SnapshotComplianceRecord:
    """L2 dataplane retention state for one snapshot.

    Logical leases drive ``active_consumer_count``. While the count is
    positive, ``compliance_delete_at`` is ``None`` and the L2 idle clock is
    paused. The clock starts from ``last_released_at`` when the count reaches
    zero. ``expired_at`` and ``deleted_at`` derive the retention state: both
    unset means active, only ``expired_at`` set means delete pending, and
    ``deleted_at`` set is the durable delete tombstone.
    """
    snapshot_id: SnapshotID
    retention_mode: RetentionMode = RetentionMode.DEFAULT
    active_consumer_count: int = 0
    last_released_at: Optional[datetime] = None
    compliance_delete_at: Optional[datetime] = None
    # Creation-time snapshot of the policy for audit/debug only. Enforcement
    # reads the live ``SnapshotLifecyclePolicy`` so tuning applies to existing
    # records (kumo#29047).
    max_dataplane_retention_secs: int = (DEFAULT_MAX_DATAPLANE_RETENTION_SECS)
    expired_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None


@kumo_dataclass
class SnapshotLocalCopyRecord:
    """L1 local-SSD retention state for one snapshot on one worker."""
    snapshot_id: SnapshotID
    pool_worker_uid: PoolWorkerUID
    local_state: SnapshotLocalCopyState = SnapshotLocalCopyState.WARM
    size_bytes: int = 0
    last_released_at: Optional[datetime] = None
    evictable_under_pressure_at: Optional[datetime] = None
    force_evict_at: Optional[datetime] = None
    # Creation-time snapshot of the policy for audit/debug only. Enforcement
    # reads the live ``SnapshotLifecyclePolicy`` so tuning applies to existing
    # records (kumo#29047).
    min_ssd_idle_secs: int = DEFAULT_MIN_SSD_IDLE_SECS
    max_ssd_idle_secs: int = DEFAULT_MAX_SSD_IDLE_SECS
    evicted_at: Optional[datetime] = None


@kumo_dataclass
class SnapshotPlacement:
    """One snapshot's placement on a specific worker's PVC.
    The snapshot id is carried as the ``PoolState.snapshots`` dict key.
    """
    worker: PoolWorkerUID
    state: SnapshotState = SnapshotState.RESERVED
    # Job IDs pinning this placement (empty => evictable after the L1 window).
    leases: List[JobID] = field(default_factory=list)
    last_lease_released: Optional[datetime] = None
    size_bytes: int = 0
    # Route decision recorded per-lease in section 1 of ``place_snapshot``.
    # Required for retry correctness: when section 2 (provision) is cancelled
    # by Temporal's start_to_close_timeout, the next attempt must look up the
    # handle in the right map (``durable_handles`` vs ``overflow_handles``).
    # Without this, retries fall back to a sibling shard's handle and dispatch
    # to the wrong task_queue (see commit log for the cross-shard bug).
    lease_paths: Dict[JobID, Literal['durable',
                                     'overflow']] = field(default_factory=dict)
    # Parent Temporal workflow_id per lease (set from ``PlacementRequest``
    # on lease commit). The dangling-lease sweep groups leases by this map
    # so each workflow is queried against Temporal at most once per sweep,
    # without parsing ``job_id`` strings.
    lease_workflow_ids: Dict[JobID, str] = field(default_factory=dict)
    # Activity per lease. Used to recover durable GPU reservations after
    # controller restart; legacy states without entries are counted
    # conservatively by the controller when needed.
    lease_activities: Dict[JobID,
                           SnapshotActivity] = field(default_factory=dict)
    # Per-job handles for durable placements. Each durable shard provisions
    # its own activity pod and ``task_queue`` (k8s_strategy returns
    # ``task_queue=activity_task_queue(job_id)``), so handles are per-job
    # rather than shared across the snapshot.
    durable_handles: Dict[JobID,
                          ProvisionedHandle] = field(default_factory=dict)
    # Per-job handles for overflow placements (``emptyDir`` activity
    # pods on non-PVC nodes). Each call to ``place_snapshot`` that
    # routes to overflow records its handle here keyed by ``job_id``.
    # ``release_lease`` pops the entry for its job_id and passes it to
    # ``strategy.release`` to delete the corresponding k8s Job — without
    # per-job keying, concurrent overflow pods would orphan as the
    # latest handle would clobber previous ones.
    overflow_handles: Dict[JobID,
                           ProvisionedHandle] = field(default_factory=dict)


@kumo_dataclass
class PvcState:
    """Controller's bookkeeping for a worker's per-node PVC; K8s itself owns
    the PVC resource."""
    pool_worker_uid: PoolWorkerUID
    tier_ssd_gib: int = 0
    tier_gpu_count: int = 1
    created_at: datetime = field(default_factory=now_utc)


class WorkerStatus(str, Enum):
    READY = 'READY'
    # Heartbeats lapsed past the stale threshold; deprioritized / debug signal.
    STALE = 'STALE'
    # Heartbeats lapsed past the dead threshold and ground truth agrees.
    DEAD = 'DEAD'
    # Scale-down in progress; not routable, waiting for in-flight work to drain.
    DRAINING = 'DRAINING'

    def is_routable(self) -> bool:
        """Whether a worker in this status may receive a new placement.

        Only READY is routable. Centralizes the policy that every routing
        path (warm reuse, durable-candidate, home co-locate) must enforce so
        a worker mid-teardown (DRAINING) or unhealthy (STALE/DEAD) is never
        selected.
        """
        return self is WorkerStatus.READY

    def is_controller_owned(self) -> bool:
        """Whether this status is owned by the controller's lifecycle logic
        and must NOT be overwritten by heartbeat-age sweeping.

        DRAINING (scale-down committed) is set deliberately by reconcile;
        ageing it via heartbeat would re-expose a draining worker to routing.
        It is the only heartbeat-immune status: READY/STALE/DEAD are all
        derived purely from heartbeat age.
        """
        return self is WorkerStatus.DRAINING


@kumo_dataclass
class ActiveJob:
    job_id: JobID
    # Free-form label: "ingest", "train", "predict".
    job_type: str
    snapshot_id: SnapshotID


@kumo_dataclass
class WorkerStateSnapshot:
    """Point-in-time inventory carried in every :class:`WorkerStatusReport`.
    The controller reconciles in-memory state against this on every receipt.
    """
    ssd_total_bytes: int
    ssd_used_bytes: int
    ready_snapshots: List[SnapshotID] = field(default_factory=list)
    in_progress_snapshots: List[SnapshotID] = field(default_factory=list)
    # Writer-observed local dataplane capacity. Zero/None means unknown so
    # older workers and transient probe failures remain compatible.
    dataplane_total_bytes: int = 0
    dataplane_used_bytes: int = 0
    dataplane_same_filesystem_as_ssd: Optional[bool] = None

    @property
    def pvc_total_bytes(self) -> int:
        return self.ssd_total_bytes

    @property
    def pvc_free_bytes(self) -> int:
        return max(self.ssd_total_bytes - self.ssd_used_bytes, 0)

    @property
    def dataplane_free_bytes(self) -> int:
        return max(self.dataplane_total_bytes - self.dataplane_used_bytes, 0)


class WorkerEventType(str, Enum):
    # RESERVED -> INGESTING (fresh ingest begins).
    SNAPSHOT_INGEST_STARTED = 'SNAPSHOT_INGEST_STARTED'
    # RESERVED -> RESTORING (dataplane restore begins).
    SNAPSHOT_RESTORE_STARTED = 'SNAPSHOT_RESTORE_STARTED'
    # Actual size exceeded estimate while ingestion is in progress.
    SNAPSHOT_SIZE_UPDATE = 'SNAPSHOT_SIZE_UPDATE'
    # INGESTING|RESTORING -> READY; final size reported with the event.
    SNAPSHOT_READY = 'SNAPSHOT_READY'
    # ensure_snapshot ran out of local SSD space. Controller releases the
    # workflow lease and triggers an immediate local-retention sweep; workflow
    # retry re-enters place_snapshot with the original PlacementRequest.
    SNAPSHOT_ENSURE_NO_SPACE = 'SNAPSHOT_ENSURE_NO_SPACE'
    # ensure_snapshot failed for a non-capacity reason. Controller releases the
    # workflow lease and clears non-ready placement progress; no eviction trigger.
    SNAPSHOT_ENSURE_FAILED = 'SNAPSHOT_ENSURE_FAILED'
    # Async persist upload completed after a fresh ingest. Controller records
    # the dataplane backup only after this event; before then the local copy is
    # not eligible for eviction.
    SNAPSHOT_PERSIST_UPLOADED = 'SNAPSHOT_PERSIST_UPLOADED'
    # The worker detected a corrupt local SSD copy and invalidated its route.
    SSD_CACHE_CORRUPTED = 'SSD_CACHE_CORRUPTED'
    GPU_JOB_STARTED = 'GPU_JOB_STARTED'
    GPU_JOB_COMPLETED = 'GPU_JOB_COMPLETED'


@kumo_dataclass
class WorkerEvent:
    """Only fields relevant to ``type`` are populated; rest are ``None``."""
    type: WorkerEventType
    snapshot_id: Optional[SnapshotID] = None
    job: Optional[ActiveJob] = None
    size_bytes: Optional[int] = None
    # For SNAPSHOT_PERSIST_UPLOADED, the controller-owned L2 prefix that now
    # contains a complete snapshot backup.
    dataplane_path: Optional[str] = None
    # Set by the daemon to preserve transition time across controller queueing.
    timestamp: datetime = field(default_factory=now_utc)


@kumo_dataclass
class WorkerStatusReport:
    """Unified daemon -> controller status. ``state`` is always fresh and
    drives reconciliation; ``event`` is set only on lifecycle transitions
    so the controller can react without waiting for the next heartbeat.
    """
    worker_name: PoolWorkerUID
    state: WorkerStateSnapshot
    worker_storage_mode: PoolStorageMode = PoolStorageMode.DAEMON_ON_PVC
    event: Optional[WorkerEvent] = None


@kumo_dataclass
class WorkerInventory:
    """Controller's record of a worker, rebuilt from status reports."""
    pool_worker_uid: PoolWorkerUID
    state: WorkerStateSnapshot
    status: WorkerStatus = WorkerStatus.READY
    # Controller-owned GPU capacity/occupancy. The daemon heartbeat only
    # reports disk/snapshot state; durable GPU totals come from persisted
    # capacity tier metadata and in-use slots come from open leases.
    gpu_slots_total: int = 0
    gpu_slots_in_use: int = 0
    # Epoch seconds; transitions ``status`` to STALE past the stale threshold
    # and to DEAD past the dead threshold (when ground truth agrees).
    last_heartbeat: float = 0.0


@kumo_dataclass
class DataplaneBackup:
    """L2 backup metadata; ``dataplane_path`` is anything fsutil understands."""
    snapshot_id: SnapshotID
    dataplane_path: str
    # Materialized size, recorded at ingest so a COLD snapshot keeps its size
    # after its placement is GC'd. 0 if not yet recorded (legacy backup).
    size_bytes: int = 0


@kumo_dataclass
class PoolStatsFilter:
    pool_worker_uids: Optional[List[PoolWorkerUID]] = None
    include_snapshots: bool = False
    include_leases: bool = False
    include_placements: bool = False


@kumo_dataclass
class PoolWorkerStats:
    pool_worker_uid: PoolWorkerUID
    health: WorkerStatus
    pvc_free_bytes: int
    pvc_total_bytes: int
    gpu_slots_total: int = 0
    gpu_slots_in_use: int = 0
    active_leases: int = 0
    snapshot_ids: List[SnapshotID] = field(default_factory=list)


@kumo_dataclass
class PoolStats:
    num_nodes: int = 0
    num_healthy_nodes: int = 0
    num_snapshots: int = 0
    total_pvc_bytes: int = 0
    free_pvc_bytes: int = 0
    active_leases: int = 0
    dataplane_backups: int = 0
    durable_workers: List[PoolWorkerStats] = field(default_factory=list)
    snapshots: List[SnapshotPlacement] = field(default_factory=list)
    local_copies: List[SnapshotLocalCopyRecord] = field(default_factory=list)
    snapshot_compliance: List[SnapshotComplianceRecord] = field(
        default_factory=list)


@kumo_dataclass
class K8sPlacementState:
    """K8s-strategy-owned slice of :class:`PoolState`; opaque to controller core."""
    type: Literal['k8s'] = 'k8s'
    # One PVC per worker; created lazily on first placement, deleted when last.
    pvcs: Dict[PoolWorkerUID, PvcState] = field(default_factory=dict)
    # Epoch seconds for persisted workers that existed at controller startup
    # but have not heartbeated in this process. Persisted so repeated
    # controller restarts cannot reset the DEAD retirement countdown forever.
    silent_worker_suspect_since: Dict[PoolWorkerUID,
                                      float] = field(default_factory=dict)


@kumo_dataclass
class SinglehostPlacementState:
    """Singlehost has no durable strategy state today; this empty variant
    keeps the ``placement_state`` discriminator consistent across deployments."""
    type: Literal['singlehost'] = 'singlehost'


# pydantic discriminates on ``type`` for deserialization. New placement
# variants are added here.
PlacementState = Union[K8sPlacementState, SinglehostPlacementState]


@kumo_dataclass
class PoolState:
    """Persisted pool state at ``pool_state.json``.

    Single-typed atomicity: deployment-agnostic state (snapshots,
    dataplane_backups) and strategy-owned state (placement_state, opaque to
    core) land in one ``save()`` under one lock. ``worker_inventory`` is
    rebuilt from heartbeats; not persisted.
    """
    # Required — ``create_pool_controller(config)`` picks the right variant.
    placement_state: PlacementState
    snapshots: Dict[SnapshotID, SnapshotPlacement] = field(default_factory=dict)
    dataplane_backups: Dict[SnapshotID,
                            DataplaneBackup] = field(default_factory=dict)
    snapshot_compliance: Dict[SnapshotID, SnapshotComplianceRecord] = field(
        default_factory=dict)
    local_copies: Dict[str,
                       SnapshotLocalCopyRecord] = field(default_factory=dict)


def expected_pool_worker_uids(state: PoolState) -> set[str]:
    """Pool-worker uids the controller still expects to exist.

    A uid is "expected" if it holds a non-evicted local copy or backs a
    placement with an open lease. Shared by the strategy's ``reconcile``
    (computed on an unlocked deepcopy) and the controller's under-lock
    re-check so the "still needed" rule has a single definition across the
    controller/strategy seam — divergence here decides whether a live-leased
    worker gets torn down.
    """
    expected: set[str] = set()
    expected.update(local.pool_worker_uid
                    for local in state.local_copies.values()
                    if local.local_state != SnapshotLocalCopyState.EVICTED)
    expected.update(placement.worker for placement in state.snapshots.values()
                    if placement.leases and placement.worker)
    return expected
