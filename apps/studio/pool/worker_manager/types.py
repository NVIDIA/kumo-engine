# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared types for pool-worker managers.

Prototype phase — dataclass fields are intentionally minimal and are
expected to churn once controller-side design (Block B / An) settles.
Keep this module import-cheap: no kubernetes, no asyncio.

"""
from typing import Literal, Optional

from io_utils import kumo_dataclass

# ---------------------------------------------------------------------------
# Lifecycle status
# ---------------------------------------------------------------------------

# Orchestration-level (k8s pod phase) lifecycle. Distinct from pool-worker
# HEALTH (READY / STALE / DEAD) which is owned by the pool controller and
# combines this with heartbeat freshness and k8s node.Ready. The pool
# controller composes both views into its own inventory record.
PodStatus = Literal['Pending', 'Running', 'Registered', 'Succeeded',
                    'Terminating', 'Failed']

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@kumo_dataclass
class SecretKeyRef:
    """One value inside a k8s Secret (a pod-spec ``secretKeyRef``).

    ``secret_name`` is the Secret object, ``key`` the data entry within
    it — the same two coordinates a ``V1SecretKeySelector`` takes.
    """
    secret_name: str
    key: str


@kumo_dataclass
class PvcSpec:
    """Caller-supplied configuration for the daemon's per-node cache
    PVC. Currently capacity only — drives ``requests.storage`` on the
    PVC so the WFFC CSI provisioner can bind a cache volume in compatible
    node topology. ``gpu_count`` pins fresh
    daemons to the matching cache-node GPU-count label. Named to leave room
    for future fields (storage class override, access modes, custom
    annotations) without another rename.

    ``None`` fields mean "use manager defaults" (e.g. the manager's
    ``default_storage_request`` ctor arg when ``snapshot_size_bytes``
    is unset).

    Distinct from :class:`ResourceSpec`, which configures the
    container-level ``V1Container.resources`` knobs (cpu / memory /
    gpu); this one is exclusively about the daemon's PVC.

    ``size_gib``: the exact PVC capacity in GiB, applied directly with no
    headroom. The controller picks the smallest usable-SSD size that fits the
    snapshot estimate from the sizes in
    ``DiskgraphConfig.k8s_pool.pool_diskgraph_node_types`` and sets
    ``size_gib`` to it. The controller also stamps it as the daemon pod's
    ``kumo.ai/diskgraph-ssd-size`` and ``kumo.ai/diskgraph-gpu-count``
    nodeSelectors so the pod lands on a cache node of that capacity. They stay
    on rebind because portable storage may constrain only a zone.
    """
    size_gib: int
    gpu_count: int = 1


@kumo_dataclass
class ResourceSpec:
    """Container resource request / limit knobs for a daemon pod or
    activity pod. Fields left ``None`` are not emitted into the
    rendered ``V1ResourceRequirements`` — caller picks per-pod which
    knobs to set.

    QoS shorthand:
      - All fields ``None`` + ``gpu_count == 0`` → BestEffort QoS
        (no scheduler reservation; first to evict under node pressure).
      - Any ``*_request`` set, no matching ``*_limit`` → Burstable.
      - All four ``cpu_*`` / ``memory_*`` set with request==limit →
        Guaranteed (kubelet won't evict ahead of any Burstable pod
        on the same node).

    GPU semantics: ``gpu_count == 0`` omits the ``nvidia.com/gpu``
    resource entirely (no NVIDIA device-plugin allocation, no
    scheduler bin-packing on GPU capacity). ``> 0`` adds it to BOTH
    requests and limits — extended resources require requests==limits;
    K8s rejects mismatches. Negative values are rejected at
    construction time.

    Resource quantity strings follow K8s conventions (e.g. ``"2"``,
    ``"100m"``, ``"4Gi"``) — manager doesn't parse them, just passes
    them through to the K8s API.
    """
    cpu_request: Optional[str] = None
    cpu_limit: Optional[str] = None
    memory_request: Optional[str] = None
    memory_limit: Optional[str] = None
    gpu_count: int = 0

    def __post_init__(self) -> None:
        if self.gpu_count < 0:
            raise ValueError(f'gpu_count must be >= 0, got {self.gpu_count}')


@kumo_dataclass
class PoolWorkerInfo:
    """Orchestration view of one pool worker = (node + daemon pod + PVC).

    ``pool_worker_uid`` is a stable identity assigned at provisioning
    time and reusable across daemon-pod recoveries on the same node
    (design §5). ``node_name`` is populated once the scheduler binds the
    daemon pod; until then it is ``None``.

    Prototype: expect additional fields (e.g. last-heartbeat, snapshot
    inventory) once the controller-side design is finalized.
    """
    pool_worker_uid: str
    pvc_name: str
    # Optional: pre-scheduling, the daemon pod is not yet bound to a
    # name (provision_pool_worker constructs PoolWorkerInfo only after
    # the pod has a name, but get_status / list_pool_workers can in
    # principle observe pods mid-construction).
    daemon_pod_name: Optional[str]
    node_name: Optional[str]
    status: PodStatus
    # Kubernetes-owned wall-clock timestamps. Optional for compatibility with
    # older callers and defensive handling of incomplete API objects.
    created_at: Optional[float] = None
    deletion_started_at: Optional[float] = None
    # Live pod fingerprint versus this manager's configured extras, independent
    # of readiness. A mismatch requires the same maintenance teardown as a
    # removed capacity tier; admission-added tolerations do not affect it.
    extra_tolerations_match: bool = True


@kumo_dataclass
class ActivityPodInfo:
    """Orchestration view of a disposable activity pod.

    ``pool_worker_uid`` is set for cache-pool pods (pinned to a specific
    pool worker) and ``None`` for ephemeral-pool pods (no per-node PVC
    affinity — any eph-pool node will do).

    ``job_id`` is the controller's per-job tracking key, populated
    from the pod's ``kumo.ai/job-id`` annotation. (The corresponding
    ``job-id-hash`` label carries the SHA-256-truncated digest used
    for K8s-side selectors — raw Temporal workflow_ids would violate
    label-value constraints.) Pool controller tracks activity pods
    as ``(pool_worker_uid, job_id)`` tuples for orphan cleanup and
    rebind relaunch (design §6.1).
    """
    pod_name: str
    pool_worker_uid: Optional[str]
    job_id: Optional[str]
    node_name: Optional[str]
    status: PodStatus


# ---------------------------------------------------------------------------
# Exception taxonomy
# ---------------------------------------------------------------------------
# Callers (pool controller) inspect types to decide retry vs. surface.
# Keep this list small and stable.


class PoolWorkerManagerError(Exception):
    """Base exception for all pool-worker-manager operations.

    Retryability convention: errors that should be retried by the
    AutoRPC framework also inherit :class:`RuntimeError`, which the
    framework's ``is_retryable`` predicate
    (``temporalio_commons.auto_rpc_exception``) accepts as a retryable
    marker without forcing this module to depend on temporalio_commons
    (types.py is intentionally import-cheap). Deterministic errors
    (``InvalidPvcSpecError``, ``NodePoolNotFoundError``,
    ``K8sPermissionError``, ``UnclassifiedApiError``) inherit only from
    ``PoolWorkerManagerError`` and are therefore non-retryable by
    default.
    """


class ProvisionTimeoutError(PoolWorkerManagerError, RuntimeError):
    """Provisioning did not reach Registered within timeout.

    Retryable. Typical causes: Karpenter slow, image pull slow, daemon
    slow to bind task queue.
    """


class ProvisionTerminalError(PoolWorkerManagerError):
    """Activity pod entered a terminal-broken Pending state (e.g. ImagePullBackOff).

    Not retryable — the underlying cause (bad image, missing secret, bad env
    ref) must be fixed before retry will succeed.

    Intentionally does NOT inherit RuntimeError (unlike ProvisionTimeoutError).
    AutoRPC's is_retryable() returns True for RuntimeError subclasses; by
    keeping ProvisionTerminalError outside that chain, AutoRPC wraps it in
    _AutoRpcNonRetriableError, which propagates non_retryable=True through the
    AutoRPC → Temporal boundary so Temporal stops retrying immediately.
    """


class QuotaExceededError(PoolWorkerManagerError, RuntimeError):
    """Infra refused due to quota / capacity.

    Retryable after a longer backoff. Typical causes: EC2 capacity
    unavailable, NodePool limit hit, CSI storage capacity exhausted.
    """


class InvalidPvcSpecError(PoolWorkerManagerError):
    """PvcSpec cannot be satisfied. Not retryable."""


class NodePoolNotFoundError(PoolWorkerManagerError):
    """Configured Karpenter NodePool is missing. Not retryable."""


class K8sPermissionError(PoolWorkerManagerError):
    """K8s apiserver returned 403 for a reason other than quota — the
    caller lacks RBAC permission for the operation, or the bound IRSA
    role can't satisfy the underlying AWS action.

    Not retryable. Resolution is structural (RBAC binding fix, IRSA
    role attach, IAM policy update).
    """


class PoolWorkerNotFoundError(PoolWorkerManagerError):
    """``pool_worker_uid`` does not exist. Caller may treat as
    already-released.
    """


class PoolWorkerNotReadyError(PoolWorkerManagerError):
    """``pool_worker_uid`` exists but its daemon pod is not Registered.

    Raised by ``provision_activity_pod`` when the target pool worker is
    mid-recovery, mid-rebind, or otherwise not yet serving — fast-fail
    so the caller doesn't mount a half-ready PVC. Caller decides whether
    to wait and retry, or schedule the activity on a different pool
    worker.
    """


class PVCBindTimeoutError(PoolWorkerManagerError, RuntimeError):
    """The per-node PVC didn't bind to a PV within the provisioning
    window — typically a CSI capacity miss (the LV couldn't be carved
    out of the cache pool's VG, or no eligible cache node has enough
    free GiB).

    Retryable on a different instance type / capacity hint per design
    §8.3 — distinct from ``ProvisionTimeoutError`` so the pool
    controller can route this signal to a different recovery branch
    (resize the request, switch instance family, etc.).
    """


class ActivityPodNotFoundError(PoolWorkerManagerError):
    """An activity pod / Job referenced by name doesn't exist.

    Reserved in the taxonomy for callers writing against design §4.1's
    method shape (``get_activity_pod_status(pod_id)``-style lookups).
    The current label-selector-based interface doesn't raise this, but
    keeping it in the taxonomy lets controller code compile cleanly.
    """


class UnclassifiedApiError(PoolWorkerManagerError):
    """K8s ``ApiException`` that ``_translate_api_error`` could not map
    to a more specific kumo error class.

    Wrapping the original ``ApiException`` (instead of re-raising it
    directly) avoids a self-referential ``__cause__`` chain when the
    caller does ``raise _translate_api_error(exc) from exc``. The
    original ``ApiException`` is preserved on this exception's
    ``__cause__`` so callers retain HTTP status / body for inspection.
    """


class ApiServerThrottledError(PoolWorkerManagerError, RuntimeError):
    """K8s API server returned 429 Too Many Requests.

    Always retryable with exponential backoff. Caller should NOT retry
    immediately — the apiserver is asking for breathing room.
    """


class ApiServerTransientError(PoolWorkerManagerError, RuntimeError):
    """K8s API server returned a transient 5xx (apiserver / etcd /
    admission webhook flake).

    Retryable. Distinct from QuotaExceededError so callers can use a
    tighter retry budget for transient flakes vs. capacity-driven
    failures.
    """


class PoolWorkerAlreadyExistsError(PoolWorkerManagerError):
    """``create_*`` returned 409 Conflict — a uid-tagged resource
    already exists.

    When this surfaces depends on which path
    ``provision_pool_worker`` is on:

    - **Fresh-mint** (caller passed no ``pool_worker_uid``, manager
      generated one) — 409 propagates as
      ``PoolWorkerAlreadyExistsError``. With a 10-char uuid hex,
      collision is astronomically unlikely, so a 409 here means a
      real bug (label collision, leftover state from a prior tenant,
      etc.) that the caller should see explicitly.
    - **Reclaim** (caller passed an explicit ``pool_worker_uid``) —
      the manager **does** auto-recover: 409 from PVC and/or
      Deployment create is treated as "already there from a prior
      attempt" (controller crash mid-provision, daemon-pod recovery
      on the same uid) and the call continues to
      ``_wait_for_daemon_registered``. This error does NOT surface
      on the reclaim path.

    Activity Job provisioners own a separate deterministic-name reclaim
    contract: they consume this error and attach to the existing Job. Other
    call sites, including delete operations, surface it without auto-recovery;
    they must define an equivalent explicit reclaim contract before consuming
    a bare 409. The manager does not infer recovery intent from the status code
    alone.
    """


class PoolWorkerCleanupFailedError(PoolWorkerManagerError, RuntimeError):
    """``delete_pool_worker`` could not clear all uid-tagged resources
    within its budget.

    Three failure modes surface this:

    1. ``_wait_for_daemon_pod_gone`` timed out — the daemon pod is
       still mounted, and we refuse to delete the PVC under a
       still-mounted kubelet (would wedge the CSI attacher
       finalizer). Caller can retry, or pass ``force_on_timeout``
       through ``delete_pool_worker``'s DEAD-node path (design §10).
    2. ``_try_delete_pvc`` exhausted its bounded poll without the PVC
       disappearing. CSI still owns cleanup; the caller must retry or
       repair it rather than bypassing volume-protection finalizers.
    3. ``delete_activity_pods`` timed out while waiting for the Job and its
       pods to be deleted before reusing its name.

    Cleanup is idempotent and therefore retryable. Persistent daemon/PVC
    failures should still route the uid into a retired/quarantined set and
    alert; re-provisioning against half-deleted state risks contention.
    """
