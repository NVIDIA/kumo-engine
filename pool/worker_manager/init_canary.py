# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Init-time node-pool validation ("canary") for the Diskgraph pool.

Why this exists
---------------
The pool controller's cache / ephemeral node pools are external infra it
does not own: the platform (kumo-delivery NodePools + kumo-control CSI
config) is responsible for delivering nodes that carry the right
``kumo.ai/diskgraph-*`` labels, the dedicated pool taint, an
``nvidia.com/gpu`` device plugin, and — for the cache pool — compatible
capacity and topology for the StorageClass to bind a PVC. If any of that is misconfigured,
nothing fails at deploy: the controller boots happily and the breakage
only surfaces the first time a user triggers a job (Karpenter never
provisions a matching node, the daemon Pod sits Pending, the activity
times out). On scale-to-zero / CSP-managed pools there is no node at
deploy at all, so a read-only conformance probe can't see the problem
either.

This module exercises the pool INFRA envelope each node pool must provide
in a short-lived **canary** pod per node pool, run once at controller
startup (``pool.controller_main.main`` calls :func:`run_init_canary` after
``create_pool_controller`` and before ``worker.run()``). A failed capability
is removed from runtime placement. Startup raises only when no cache tier is
available, preserving a "not healthy at init" signal for a controller with no
durable placement capacity.

For a genuine label or taint misconfiguration, the canary stays Pending and
does not provision a node. A too-short ``POOL_CANARY_GRACE_SECS`` can still
mark a correctly configured slow GPU + storage tier unavailable just as its node
comes up. Tune the grace to the cluster's cold-provision time (node launch,
kubelet registration, device-plugin advertisement, CSI binding, and the cold
pull of the daemon image).

What each canary validates
--------------------------
- **cache** (one per distinct ``(ssd_gib, gpu_count)`` tier): validates the cache
  NodePool's infra envelope — label + taint + a bound cache PVC + a GPU.
  The cache PVC may use node-local storage (for example TopoLVM on EKS)
  or zonal block storage (Azure Disk on AKS). A GPU is used by the
  activity pods that run on that pool (the canary's ``nvidia.com/gpu``
  request matches the configured tier count and exercises the device plugin).
  nodeSelector = the daemon's cache
  selector for the tier (``diskgraph-pool-type=cache`` +
  ``diskgraph-ssd-size=<gib>`` + ``diskgraph-gpu-count=<n>``),
  tolerations = the built-in cache pool toleration plus configured extras, and a
  PVC on the cache StorageClass sized to the daemon's PVC size for that
  tier. ``WaitForFirstConsumer`` binds the volume in compatible
  topology and validates that the requested cache volume can be provisioned. Note: the canary
  validates the pool envelope, it does not literally mirror the cache
  daemon's own resource requests — that daemon runs with ``gpu_count=0``
  (the k8s.py default), so it requests no GPU itself.
- **ephemeral** (one, OPT-OUT): label + taint + GPU + scheduling only. Eph
  disk is ``ephemeral-storage`` sized per-snapshot at runtime (not knowable
  at init), so the eph disk dimension is validated at job time, not here.
  DEFAULT ENABLED (``POOL_INIT_EPHEMERAL_CANARY_ENABLED``): the eph tier is a
  runtime overflow spill target validated at init on deployments that run a
  dedicated ephemeral pool. A single-GPU / no-ephemeral-node deployment sets
  ``POOL_INIT_EPHEMERAL_CANARY_ENABLED=false`` to avoid an unnecessary probe;
  a failed enabled probe disables runtime ephemeral overflow for that process.

Cost control
------------
Provisioning a GPU node and cache volume takes time and incurs cost, so a success
is recorded as an ANNOTATION on the pool controller's own Deployment
(``kumo.ai/diskgraph-pools-validated=<config-hash>@<epoch>``), keyed by a
hash of the pool-infra dimensions a canary actually exercises (cache tiers,
labels, StorageClass, the cache/eph pool taints and extra tolerations). On
start, a fresh matching record (within :data:`CANARY_RESULT_TTL_SECS`)
SKIPS re-validation. The hash changes when any of those keyed dimensions
change, forcing re-validation; the TTL forces periodic re-validation to
catch infra drift. Only complete active validation is recorded; process-local
proof from an existing Registered worker is never persisted as global success.

The annotation lives on the controller's own Deployment (not a ConfigMap)
deliberately: it is covered by the controller SA's existing
``deployments [get,patch]`` grant and needs no extra RBAC. The Deployment
name is read from ``POOL_CONTROLLER_DEPLOYMENT_NAME`` (default
``kumo-poolcontroller``); if that Deployment can't be read/patched the
canary still runs (caching is best-effort) — it just won't dedupe across
restarts.

Cleanup safety
--------------
Every canary resource is labelled ``app=dg-pool-canary`` and reconciled on
start (any leftover canary pods/PVCs from a crashed prior run are deleted
before a fresh validation), so a controller crash mid-validation can't leak
canary pods / PVCs / nodes.

The Helm chart's readiness probe watches the sentinel created by
``pool.controller_main`` only after this canary and controller construction
complete. This gates startup readiness; it is not a steady-state Temporal
worker health check.
"""
import asyncio
import hashlib
import json
import os
import time
from collections.abc import Collection
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional, Union, cast

from kubernetes.client.models import (
    V1Container,
    V1Deployment,
    V1ObjectMeta,
    V1PersistentVolumeClaim,
    V1PersistentVolumeClaimList,
    V1PersistentVolumeClaimSpec,
    V1PersistentVolumeClaimVolumeSource,
    V1Pod,
    V1PodList,
    V1PodSpec,
    V1ResourceRequirements,
    V1Volume,
    V1VolumeMount,
)
from kubernetes.client.rest import ApiException

from io_utils import logging
from pool.worker_manager.k8s import (
    DISKGRAPH_GPU_COUNT_LABEL,
    DISKGRAPH_POOL_TYPE_LABEL,
    DISKGRAPH_SSD_SIZE_LABEL,
    LABEL_TENANT,
    NVME_MOUNT_PATH,
    K8sPoolWorkerManager,
    _PoolScheduling,
    _runtime_container_security_context,
    _runtime_pod_security_context,
    _toleration_key,
    _unschedulable_reason,
)
from pool.worker_manager.types import PoolWorkerManagerError, PvcSpec

log = logging.getLogger(__name__)

# Grace per canary before init fails. The budget must cover a full cold
# path on a scale-to-zero pool: GPU node provision (minutes on
# Karpenter / a CSP-managed autoscaler: EC2 launch + kubelet register +
# nvidia device-plugin advertise), CSI binding the PVC once that node
# exists (WaitForFirstConsumer), AND the kubelet pulling the canary's image
# onto the fresh node. The canary runs the daemon image — the same image the
# real daemon pulls, so it is guaranteed pullable on the pool nodes (a tiny
# pause image is deliberately NOT used: it could be unreachable in a
# locked-down BYOC registry and fail a healthy pool). That image is large
# (~16GB) and can take minutes to pull cold, so 600s covers node-provision +
# a cold image pull while still surfacing a genuine misconfig at deploy.
# NOTE: there is no fast-fail — an Unschedulable pod is the expected
# transient state while the autoscaler provisions, so a real label/taint
# misconfig ALSO takes the full grace to surface; the bump from 300s trades
# slightly slower misconfig feedback for not CrashLooping (and churning a
# just-provisioned node) on a healthy-but-slow cold image pull.
# Within the window, Pending / Unschedulable / PVC-Pending is tolerated and
# logged as progress; only PAST grace do we fail. Override via
# POOL_CANARY_GRACE_SECS (raise it on clusters with slow cold image pulls).
CANARY_GRACE_SECS = int(os.environ.get('POOL_CANARY_GRACE_SECS', '600'))

# Re-validation TTL. A recorded success older than this is treated as stale
# (infra may have drifted) and the canary re-runs. 24h balances catching
# drift against the per-provision node cost. Override via
# POOL_CANARY_RESULT_TTL_HOURS.
CANARY_RESULT_TTL_SECS = int(
    os.environ.get('POOL_CANARY_RESULT_TTL_HOURS', '24')) * 3600

# Poll cadence while waiting for a canary to schedule / its PVC to bind.
_CANARY_POLL_INTERVAL_SECS = 5.0

# Bound on how long we wait for leftover canary pods / PVCs to actually
# disappear before launching fresh canaries (reconcile-on-start), and on the
# best-effort cleanup wait for a canary pod to terminate before its PVC is
# deleted. The controller is a SINGLE replica, so a leftover can only be a
# crashed prior run's resource (never a concurrent writer); these are short
# API-object deletions, not node provisions, so ~30s is generous. Bounded so
# the controller can't hang forever waiting on a wedged finalizer.
_RECONCILE_WAIT_SECS = 30.0

# Gate: when False (POOL_INIT_CANARY_ENABLED=false / 0 / no), the canary is
# skipped entirely and the controller boots without validating its pools.
# DEFAULT ON for the k8s strategy. Operators turn it off to bypass validation
# (for example, to bring up the controller before its NodePools exist). Single
# source of truth for the chart contract.
CANARY_ENABLED = os.environ.get(
    'POOL_INIT_CANARY_ENABLED', 'true').strip().lower() not in (
        '0', 'false', 'no', 'off')

# Independent gate for the EPHEMERAL canary - DEFAULT ENABLED (opt-OUT).
#
# The ephemeral tier is a runtime OVERFLOW spill target: a fresh
# TRAIN / PREDICT / MATERIALIZE co-locates on the CACHE node (the activity pod
# mounts the pool worker's PVC, which pins it to the daemon's node), and the eph
# tier is reached when an ALREADY-materialized snapshot's cache worker is
# GPU-exhausted (``service.overflow_viable`` is gated on a prior dataplane
# backup existing). A deployment that runs a dedicated ``pool-type=ephemeral``
# GPU node uses that tier, so the eph canary validates it at init BY DEFAULT.
#
# But on a single-GPU / no-ephemeral-node deployment there is no such node. An
# enabled probe times out and disables runtime ephemeral overflow for this
# process. Such a cluster can avoid that unnecessary probe with
# POOL_INIT_EPHEMERAL_CANARY_ENABLED=false (the single-GPU /
# no-ephemeral-nodepool escape hatch), which skips this init canary AND disables
# runtime overflow to the eph tier.
#
# The controller runs under a NAMESPACED Role with no cluster-scoped ``nodes``
# RBAC (by design - see helm/kumo-helm role-workload-manager), so it cannot
# auto-detect whether an ephemeral node pool exists. Hence an explicit opt-OUT
# rather than node-inventory autodetection: the eph canary runs UNLESS
# POOL_INIT_EPHEMERAL_CANARY_ENABLED is falsey. The CACHE canary - the tier
# every run actually uses - stays governed by CANARY_ENABLED above and is
# unaffected.
EPH_CANARY_ENABLED = os.environ.get(
    'POOL_INIT_EPHEMERAL_CANARY_ENABLED', 'true').strip().lower() not in (
        '0', 'false', 'no', 'off')

# Canary identity label — narrows list/delete to canary resources only so
# reconcile-on-start can't touch daemon / activity pods.
CANARY_APP_LABEL_VALUE = 'dg-pool-canary'
_LABEL_APP = 'app'
# Which pool a given canary targets (cache tier vs eph), for diagnostics.
_LABEL_CANARY_POOL = 'kumo.ai/canary-pool'
_CANARY_SPEC_HASH_ANNOTATION = 'kumo.ai/pool-canary-spec-hash'

# Result cache: an annotation on the controller's OWN Deployment (covered by
# the existing ``deployments [get,patch]`` grant — no ConfigMap RBAC needed).
# Value is ``<config-hash>@<epoch-seconds>``.
RESULT_ANNOTATION_KEY = 'kumo.ai/diskgraph-pools-validated'
CONTROLLER_DEPLOYMENT_NAME = os.environ.get(
    'POOL_CONTROLLER_DEPLOYMENT_NAME', 'kumo-poolcontroller')

# Canary container resources: cache canaries request the configured tier GPU
# count; eph canaries request one GPU. Tiny CPU / memory keeps the canary easy
# to fit once the GPU dimension is satisfied. The container is a no-op sleep —
# we only need it Running (and, for cache, its PVC Bound); no app workload.
_CANARY_CPU = '100m'
_CANARY_MEMORY = '128Mi'
_GPU_RESOURCE_KEY = 'nvidia.com/gpu'
# Long enough that the kubelet keeps it Running through the grace window;
# delete-on-success / reconcile-on-start reclaim it well before this fires.
_CANARY_SLEEP_SECS = 1800


class InitCanaryValidationError(PoolWorkerManagerError):
    """An init canary could not be scheduled (and, for cache, its PVC
    bound) within the grace window — the node pool can't deliver a
    schedulable node with the required GPU + disk.

    The message names the offending tier and the exact labels / GPU / disk it
    expected, classified (where the scheduler / PVC events allow) into label /
    GPU / disk misconfig. The startup orchestrator contains an individual tier
    failure when another cache tier is usable, and propagates an aggregate
    error when no cache tier remains available.
    """


@dataclass(frozen=True)
class InitCanaryResult:
    """Pool capabilities proven usable for this controller process."""
    available_cache_tiers: frozenset[tuple[int, int]]
    ephemeral_available: bool


PrevalidatedCacheTierSource = Union[
    Collection[tuple[int, int]],
    Callable[[], Awaitable[Collection[tuple[int, int]]]],
]


def _canary_pod_name(pool: str) -> str:
    return f'dg-pool-canary-{pool}'


def _canary_pvc_name(pool: str) -> str:
    return f'dg-pool-canary-pvc-{pool}'


def _canary_labels(tenant: str, pool: str) -> Dict[str, str]:
    return {
        _LABEL_APP: CANARY_APP_LABEL_VALUE,
        LABEL_TENANT: tenant,
        _LABEL_CANARY_POOL: pool,
    }


def _taint_str(tol) -> str:
    """``key=value:effect`` for a V1Toleration — a stable, hashable
    rendering of the pool taint the canary tolerates."""
    return f'{tol.key}={tol.value}:{tol.effect}'


def pool_config_hash(
    *,
    tiers_gib: List[int],
    tier_specs: Optional[List[str]] = None,
    storage_class: str,
    cache_node_selector_keys: List[str],
    cache_taint: str,
    eph_taint: str,
    eph_canary_enabled: bool = True,
    extra_tolerations: Optional[List[str]] = None,
) -> str:
    """Stable hash of the pool-infra dimensions a canary actually exercises.

    Keyed on: the set of cache tier specs, the StorageClass, the label keys used
    to select cache nodes, the cache- and eph-pool taint VALUES the
    canary tolerates, and whether the eph canary is enabled. A change to any
    of these invalidates a cached success and forces re-validation - e.g.
    re-pointing a pool's dedicated taint must re-prove the pool, since a
    TTL-fresh cached success would otherwise wrongly SKIP re-validation.
    ``eph_canary_enabled`` is keyed so that toggling the (opt-out) eph canary
    forces a re-validation that matches its current mode, rather than
    inheriting a cached success recorded while it was skipped. The eph pool
    carries no size dimension, so it contributes only its taint (beyond the
    constant selector keys). The canary's own image is deliberately NOT keyed:
    it is just the no-op ``sleep`` container, not a pool-infra dimension, so a
    kumo image bump must not force an unnecessary GPU/storage re-validation when
    the node-pool infra is unchanged.
    """
    dimensions = {
        'tiers_gib': sorted(tiers_gib),
        'tier_specs': sorted(tier_specs or []),
        'storage_class': storage_class,
        'cache_node_selector_keys': sorted(cache_node_selector_keys),
        'cache_taint': cache_taint,
        'eph_taint': eph_taint,
        'eph_canary_enabled': eph_canary_enabled,
    }
    # Preserve the old hash when no extras are configured: an image upgrade
    # alone must not force GPU/PVC provisioning. Include every extra field,
    # including operator and tolerationSeconds; ordering is not meaningful.
    if extra_tolerations:
        dimensions['extra_tolerations'] = sorted(set(extra_tolerations))
    payload = json.dumps(dimensions, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]


def _build_canary_pod(
    *,
    name: str,
    namespace: str,
    labels: Dict[str, str],
    sched: _PoolScheduling,
    service_account: str,
    image: str,
    pvc_name: Optional[str],
    gpu_count: int = 1,
) -> V1Pod:
    """A short-lived GPU canary mirroring a daemon / eph pod's schedule.

    ``pvc_name`` set (cache) → mounts that PVC so WaitForFirstConsumer
    binds it on the chosen node (validates disk). ``None`` (eph) → no
    volume; validates label + taint + GPU + scheduling only.
    """
    volume_mounts: List[V1VolumeMount] = []
    volumes: List[V1Volume] = []
    if pvc_name is not None:
        volume_mounts.append(
            V1VolumeMount(mount_path=NVME_MOUNT_PATH, name='ssd'))
        volumes.append(
            V1Volume(
                name='ssd',
                persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
                    claim_name=pvc_name),
            ))
    pod = V1Pod(
        metadata=V1ObjectMeta(name=name, namespace=namespace, labels=labels),
        spec=V1PodSpec(
            # One-shot: never restart. The canary is deleted on success;
            # on failure it stays Pending (no node) and we crash init.
            restart_policy='Never',
            security_context=_runtime_pod_security_context(),
            service_account_name=service_account,
            node_selector=sched.node_selector,
            tolerations=sched.tolerations,
            containers=[
                V1Container(
                    name='canary',
                    image=image,
                    security_context=_runtime_container_security_context(),
                    # No app workload — just hold the node long enough to
                    # prove it scheduled with a GPU + (cache) bound PVC.
                    command=['sleep', str(_CANARY_SLEEP_SECS)],
                    resources=V1ResourceRequirements(
                        requests={
                            'cpu': _CANARY_CPU,
                            'memory': _CANARY_MEMORY,
                            _GPU_RESOURCE_KEY: str(gpu_count),
                        },
                        limits={_GPU_RESOURCE_KEY: str(gpu_count)},
                    ),
                    volume_mounts=volume_mounts,
                ),
            ],
            volumes=volumes,
        ),
    )
    # Fingerprint our requested spec before API defaulting/admission. A
    # surviving canary must not prove a different scheduling configuration;
    # tolerance ordering/duplicates and admission additions are immaterial.
    assert pod.spec is not None and pod.metadata is not None
    spec: dict[str, object] = dict(pod.spec.to_dict())
    spec['tolerations'] = sorted({
        json.dumps(_toleration_key(t)) for t in sched.tolerations
    })
    pod.metadata.annotations = {
        _CANARY_SPEC_HASH_ANNOTATION: hashlib.sha256(
            json.dumps(spec, sort_keys=True).encode()).hexdigest()[:32]
    }
    return pod


def _build_canary_pvc(
    *,
    name: str,
    namespace: str,
    labels: Dict[str, str],
    storage_class: str,
    storage_size: str,
) -> V1PersistentVolumeClaim:
    return V1PersistentVolumeClaim(
        metadata=V1ObjectMeta(name=name, namespace=namespace, labels=labels),
        spec=V1PersistentVolumeClaimSpec(
            # Matches the daemon PVC (ReadWriteOnce on the cache
            # StorageClass) so the WaitForFirstConsumer bind exercises the
            # same CSI binding path the real daemon would.
            access_modes=['ReadWriteOnce'],
            storage_class_name=storage_class,
            resources=V1ResourceRequirements(
                requests={'storage': storage_size}),
        ),
    )


def _pvc_requested_size(pvc: V1PersistentVolumeClaim) -> Optional[str]:
    """The PVC's requested ``storage`` size, or None if not set."""
    spec = pvc.spec
    if spec is None or spec.resources is None or spec.resources.requests is None:
        return None
    return spec.resources.requests.get('storage')


def _pvc_is_reusable(
    pvc: V1PersistentVolumeClaim, requested_size: str) -> bool:
    """A leftover canary PVC is safe to reuse only when it is NOT Terminating
    and its requested size matches exactly. PVC size is immutable, so a
    same-tier leftover bound at an OLD size must NOT be accepted as a valid
    canary — that would record a success for a size that was never proven.
    """
    meta = pvc.metadata
    if meta is not None and meta.deletion_timestamp is not None:
        return False
    return _pvc_requested_size(pvc) == requested_size


class _CanaryRunner:
    """Drives the canary lifecycle against one manager's CoreV1Api.

    Stateless beyond the injected manager; all k8s sync calls go through
    ``asyncio.to_thread`` to match the manager's threading discipline.
    """
    def __init__(self, manager: K8sPoolWorkerManager) -> None:
        self._manager = manager
        self._core = manager.core_v1_api
        self._apps = manager.apps_v1_api
        self._namespace = manager.namespace
        self._tenant = manager._tenant
        self._service_account = manager._service_account
        self._image = manager._daemon_image

    async def reconcile_leftovers(self) -> None:
        """Delete any canary pods / PVCs from a crashed prior run AND wait
        for them to actually be gone before fresh canaries are created.

        Selected strictly by the ``app=dg-pool-canary`` label so daemon /
        activity resources are never touched. The controller is a SINGLE
        replica, so a leftover can only be a crashed prior run's resource —
        never a concurrent writer. Deleting AND waiting (rather than just
        issuing the delete) matters because a same-tier leftover PVC bound
        at an OLD size would otherwise be silently reused: PVC size is
        immutable, so a fresh canary would validate a size that was never
        actually proven. Pods are deleted first, then PVCs (a PVC can't be
        deleted cleanly while a pod still mounts it). Best-effort: a 404
        means it was already reaped. On wait timeout we log a clear warning
        naming what survived and continue (``_create_pvc`` / ``_create_pod``
        harden against a surviving leftover on the create path).
        """
        selector = f'{_LABEL_APP}={CANARY_APP_LABEL_VALUE}'
        try:
            pods = cast(
                V1PodList,
                await asyncio.to_thread(
                    self._core.list_namespaced_pod,
                    namespace=self._namespace,
                    label_selector=selector,
                ),
            )
            for pod in (pods.items or []):
                if pod.metadata and pod.metadata.name:
                    await self._delete_pod(pod.metadata.name)
        except ApiException as exc:
            log.warning('canary reconcile: listing leftover pods failed '
                        '(status=%s); continuing', exc.status)
        try:
            pvcs = cast(
                V1PersistentVolumeClaimList,
                await asyncio.to_thread(
                    self._core.list_namespaced_persistent_volume_claim,
                    namespace=self._namespace,
                    label_selector=selector,
                ),
            )
            for pvc in (pvcs.items or []):
                if pvc.metadata and pvc.metadata.name:
                    await self._delete_pvc(pvc.metadata.name)
        except ApiException as exc:
            log.warning('canary reconcile: listing leftover PVCs failed '
                        '(status=%s); continuing', exc.status)

        # Poll until no canary pods AND no canary PVCs remain, so leftovers
        # are actually gone before fresh canaries are created. Bounded so a
        # wedged finalizer can't hang the controller forever.
        deadline = time.monotonic() + _RECONCILE_WAIT_SECS
        remaining_pods: List[str] = []
        remaining_pvcs: List[str] = []
        while time.monotonic() < deadline:
            remaining_pods = await self._list_leftover_names(
                self._core.list_namespaced_pod, selector)
            remaining_pvcs = await self._list_leftover_names(
                self._core.list_namespaced_persistent_volume_claim, selector)
            if not remaining_pods and not remaining_pvcs:
                return
            await asyncio.sleep(_CANARY_POLL_INTERVAL_SECS)
        log.warning(
            'canary reconcile: leftover canary resources still present after '
            '%.0fs (pods=%s, pvcs=%s); continuing — create path will harden '
            'against any survivor', _RECONCILE_WAIT_SECS, remaining_pods,
            remaining_pvcs)

    async def _list_leftover_names(self, list_fn, selector: str) -> List[str]:
        """Names of canary resources still matching ``selector`` (best-effort:
        a list failure returns ``[]`` so the reconcile wait isn't wedged by a
        transient API error)."""
        try:
            result = await asyncio.to_thread(
                list_fn, namespace=self._namespace, label_selector=selector)
        except ApiException as exc:
            log.warning('canary reconcile: re-listing leftovers failed '
                        '(status=%s); treating as none remaining', exc.status)
            return []
        return [
            item.metadata.name for item in (result.items or [])
            if item.metadata and item.metadata.name
        ]

    async def _delete_pod(self, name: str) -> None:
        try:
            await asyncio.to_thread(
                self._core.delete_namespaced_pod,
                name=name,
                namespace=self._namespace,
                grace_period_seconds=0,
            )
        except ApiException as exc:
            if exc.status != 404:
                log.warning('canary: delete pod %s returned %s', name,
                            exc.status)

    async def _delete_pvc(self, name: str) -> None:
        try:
            await asyncio.to_thread(
                self._core.delete_namespaced_persistent_volume_claim,
                name=name,
                namespace=self._namespace,
            )
        except ApiException as exc:
            if exc.status != 404:
                log.warning('canary: delete PVC %s returned %s', name,
                            exc.status)

    async def _read_pod(self, name: str) -> Optional[V1Pod]:
        try:
            return cast(
                V1Pod,
                await asyncio.to_thread(
                    self._core.read_namespaced_pod,
                    name=name,
                    namespace=self._namespace,
                ),
            )
        except ApiException as exc:
            if exc.status == 404:
                return None
            raise

    async def _read_pvc(self, name: str) -> Optional[V1PersistentVolumeClaim]:
        try:
            return cast(
                V1PersistentVolumeClaim,
                await asyncio.to_thread(
                    self._core.read_namespaced_persistent_volume_claim,
                    name=name,
                    namespace=self._namespace,
                ),
            )
        except ApiException as exc:
            if exc.status == 404:
                return None
            raise

    async def _read_pvc_phase(self, name: str) -> Optional[str]:
        pvc = await self._read_pvc(name)
        if pvc is None:
            return None
        return pvc.status.phase if pvc.status else None

    async def _wait_for_gone(
        self,
        read_fn: Callable[[str], Awaitable[Optional[object]]],
        name: str,
    ) -> bool:
        """Poll (bounded) until ``read_fn(name)`` returns None (the object is
        gone). Returns True if it disappeared within
        :data:`_RECONCILE_WAIT_SECS`, False on timeout. Used for both canary
        pods (``_read_pod``) and PVCs (``_read_pvc``)."""
        deadline = time.monotonic() + _RECONCILE_WAIT_SECS
        while time.monotonic() < deadline:
            if await read_fn(name) is None:
                return True
            await asyncio.sleep(_CANARY_POLL_INTERVAL_SECS)
        return False

    async def run_one(
        self,
        *,
        pool: str,
        sched: _PoolScheduling,
        pvc_size: Optional[str],
        storage_class: str,
        expectation: str,
        gpu_count: int = 1,
    ) -> None:
        """Launch one canary, wait out the grace window, classify+raise on
        failure, delete on success.

        ``pvc_size`` set → cache canary (also waits for PVC Bound);
        ``None`` → eph canary (scheduling + GPU only). ``expectation`` is
        a human string naming the labels / GPU / disk this canary expects,
        embedded in the failure message.
        """
        labels = _canary_labels(self._tenant, pool)
        pod_name = _canary_pod_name(pool)
        pvc_name = _canary_pvc_name(pool) if pvc_size is not None else None

        # The creates live INSIDE the try so the finally cleans up whatever
        # was created even when a create itself raises (e.g. the stale-
        # leftover guards in _create_pvc / _create_pod): a fresh PVC must not
        # leak — and hold CSI capacity — across crashloop retries
        # until the next start's reconcile_leftovers().
        try:
            if pvc_name is not None:
                assert pvc_size is not None
                await self._create_pvc(
                    _build_canary_pvc(
                        name=pvc_name,
                        namespace=self._namespace,
                        labels=labels,
                        storage_class=storage_class,
                        storage_size=pvc_size,
                    ),
                    storage_size=pvc_size,
                )
            await self._create_pod(
                _build_canary_pod(
                    name=pod_name,
                    namespace=self._namespace,
                    labels=labels,
                    sched=sched,
                    service_account=self._service_account,
                    image=self._image,
                    pvc_name=pvc_name,
                    gpu_count=gpu_count,
                ))

            await self._wait_for_ready(
                pool=pool,
                pod_name=pod_name,
                pvc_name=pvc_name,
                expectation=expectation,
            )
        finally:
            # Delete-on-success AND on-failure: on success we reclaim the
            # node/volume (reclaim Delete releases the CSI volume). On a
            # label/taint misconfig nothing was provisioned (Pending), so
            # deletion is a no-op cleanup of the API objects. On a healthy
            # but slow pool that provisioned a node just as grace expired,
            # this deletion leaves the node unused and the autoscaler
            # consolidates it — so tune POOL_CANARY_GRACE_SECS to the
            # cluster's cold-provision time to avoid churning real nodes.
            # Either way no canary API state leaks.
            #
            # Order mirrors delete_pool_worker: delete the pod, WAIT for it to
            # be gone, THEN delete the PVC — deleting the PVC under a still-
            # mounted kubelet would wedge the CSI attacher / pvc-protection
            # finalizer. Cleanup is BEST-EFFORT: a cleanup error or timeout
            # must NOT mask the validation result, so we log and continue
            # rather than letting it raise over a real success/failure.
            try:
                await self._delete_pod(pod_name)
                if pvc_name is not None:
                    if not await self._wait_for_gone(self._read_pod, pod_name):
                        log.warning(
                            'canary[%s] cleanup: pod %s still present after '
                            '%.0fs; deleting PVC anyway (best-effort)', pool,
                            pod_name, _RECONCILE_WAIT_SECS)
                    await self._delete_pvc(pvc_name)
                    # Wait for the PVC to actually be GONE: until then CSI
                    # can still hold capacity, so exiting/retrying with the
                    # PVC merely Terminating can starve the next attempt.
                    if not await self._wait_for_gone(self._read_pvc, pvc_name):
                        log.warning(
                            'canary[%s] cleanup: PVC %s still present after '
                            '%.0fs; continuing — its capacity is released '
                            'once the deletion finalizes', pool, pvc_name,
                            _RECONCILE_WAIT_SECS)
            except Exception:  # noqa: BLE001 — best-effort cleanup
                log.warning(
                    'canary[%s] cleanup encountered an error; ignoring so the '
                    'validation result is not masked', pool, exc_info=True)

    async def _create_pod(self, pod: V1Pod) -> None:
        assert pod.metadata is not None and pod.metadata.name is not None
        name = pod.metadata.name
        try:
            await asyncio.to_thread(
                self._core.create_namespaced_pod,
                namespace=self._namespace, body=pod)
            return
        except ApiException as exc:
            if exc.status != 409:
                raise
        # 409: the controller is a SINGLE replica, so this can only be a
        # leftover from a crashed prior run (never a concurrent writer).
        # The fixed per-tier name can outlive configuration changes. Reuse
        # requires the current requested spec, not just a live old pod.
        existing = await self._read_pod(name)
        if existing is None:
            # Raced away between create and read — recreate.
            await asyncio.to_thread(
                self._core.create_namespaced_pod,
                namespace=self._namespace, body=pod)
            return
        meta = existing.metadata
        terminating = meta is not None and meta.deletion_timestamp is not None
        expected_hash = (pod.metadata.annotations or {}).get(
            _CANARY_SPEC_HASH_ANNOTATION)
        actual_hash = ((meta.annotations or {}).get(_CANARY_SPEC_HASH_ANNOTATION)
                       if meta else None)
        if not terminating and expected_hash and actual_hash == expected_hash:
            log.info('canary pod %s already has the current spec; reusing', name)
            return
        reason = 'Terminating' if terminating else 'using an outdated spec'
        log.info('canary pod %s is %s; waiting for deletion then recreating',
                 name, reason)
        if not terminating:
            await self._delete_pod(name)
        if not await self._wait_for_gone(self._read_pod, name):
            raise InitCanaryValidationError(
                f'canary pod {name!r} still {reason} after '
                f'{_RECONCILE_WAIT_SECS:.0f}s; cannot create a fresh '
                'canary — refusing to validate with a stale pod')
        await asyncio.to_thread(
            self._core.create_namespaced_pod,
            namespace=self._namespace, body=pod)

    async def _create_pvc(
        self,
        pvc: V1PersistentVolumeClaim,
        *,
        storage_size: str,
    ) -> None:
        assert pvc.metadata is not None and pvc.metadata.name is not None
        name = pvc.metadata.name
        try:
            await asyncio.to_thread(
                self._core.create_namespaced_persistent_volume_claim,
                namespace=self._namespace, body=pvc)
            return
        except ApiException as exc:
            if exc.status != 409:
                raise
        # 409: SINGLE-replica controller, so this is a leftover from a crashed
        # prior run. PVC size is IMMUTABLE, so a same-tier leftover bound at an
        # OLD size would otherwise be "reused" and validated as Bound for a
        # size that was never actually proven (a FALSE success). Only reuse
        # when the leftover is NOT Terminating AND its requested size matches
        # exactly; otherwise delete it, wait for it gone, and recreate.
        existing = await self._read_pvc(name)
        if existing is None:
            await asyncio.to_thread(
                self._core.create_namespaced_persistent_volume_claim,
                namespace=self._namespace, body=pvc)
            return
        if _pvc_is_reusable(existing, storage_size):
            log.info('canary PVC %s already exists at the requested size '
                     '%s and is not Terminating; reusing', name, storage_size)
            return
        log.info('canary PVC %s is a STALE leftover (Terminating or a '
                 'different requested size than %s); deleting and recreating '
                 'so we validate the correct size', name, storage_size)
        await self._delete_pvc(name)
        if not await self._wait_for_gone(self._read_pvc, name):
            raise InitCanaryValidationError(
                f'stale canary PVC {name!r} did not disappear within '
                f'{_RECONCILE_WAIT_SECS:.0f}s after delete; cannot create a '
                f'fresh PVC at the requested size {storage_size} — refusing '
                f'to proceed (failing validation rather than recording a '
                f'false success for an unvalidated size)')
        await asyncio.to_thread(
            self._core.create_namespaced_persistent_volume_claim,
            namespace=self._namespace, body=pvc)

    async def _wait_for_ready(
        self,
        *,
        pool: str,
        pod_name: str,
        pvc_name: Optional[str],
        expectation: str,
    ) -> None:
        deadline = time.monotonic() + CANARY_GRACE_SECS
        last_phase = 'unknown'
        last_unsched: Optional[str] = None
        last_pvc_phase: Optional[str] = None
        while time.monotonic() < deadline:
            pod = await self._read_pod(pod_name)
            if pod is not None:
                last_phase = (pod.status.phase
                              if pod.status and pod.status.phase else 'Pending')
                last_unsched = _unschedulable_reason(pod)
                if last_phase in ('Running', 'Succeeded'):
                    # Pod placed on a node — only NOW can a
                    # WaitForFirstConsumer PVC bind, so check disk readiness
                    # only after scheduling succeeds (avoids polling the PVC
                    # every iteration while the pod is still Pending).
                    if pvc_name is not None:
                        last_pvc_phase = await self._read_pvc_phase(pvc_name)
                    if pvc_name is None or last_pvc_phase == 'Bound':
                        log.info(
                            'canary[%s] OK: pod=%s phase=%s%s', pool, pod_name,
                            last_phase,
                            f' pvc={last_pvc_phase}' if pvc_name else '')
                        return
                # Within grace, Pending / Unschedulable / PVC-Pending is
                # expected (node still provisioning) — log progress.
                log.info(
                    'canary[%s] waiting: phase=%s%s%s', pool, last_phase,
                    f' pvc={last_pvc_phase}' if pvc_name else '',
                    f' scheduler="{last_unsched}"' if last_unsched else '')
            await asyncio.sleep(_CANARY_POLL_INTERVAL_SECS)

        raise InitCanaryValidationError(
            self._classify_failure(
                pool=pool,
                expectation=expectation,
                last_phase=last_phase,
                last_unsched=last_unsched,
                pvc_name=pvc_name,
                last_pvc_phase=last_pvc_phase,
            ))

    def _classify_failure(
        self,
        *,
        pool: str,
        expectation: str,
        last_phase: str,
        last_unsched: Optional[str],
        pvc_name: Optional[str],
        last_pvc_phase: Optional[str],
    ) -> str:
        """Turn the terminal pod / PVC state into an actionable message.

        Distinguishes, from the scheduler's ``PodScheduled=False`` message
        and the PVC phase: (a) no node matches selector/taint (label/pool
        misconfig), (b) insufficient GPU, (c) PVC unbound / insufficient
        storage capacity or topology. Names the tier + the exact expected
        labels / size.
        """
        # Head is deliberately NEUTRAL about the root cause — the per-cause
        # text below distinguishes "no schedulable node" (misconfig) from
        # "node came up but the image was still pulling" (cold-pull, NOT a
        # pool fault), so the head must not pre-judge it as undeliverable.
        head = (f'Diskgraph pool init canary FAILED for pool {pool!r} after '
                f'{CANARY_GRACE_SECS}s. Expected: {expectation}.')
        msg = (last_unsched or '').lower()

        # Classify the ROOT cause. An Unschedulable pod (PodScheduled=False)
        # is the root: its WaitForFirstConsumer PVC can't bind as a
        # CONSEQUENCE of having no node, so we read the scheduler message
        # for GPU / label-or-taint signals BEFORE concluding disk. Only when
        # the pod actually scheduled (no Unschedulable condition) but the PVC
        # still didn't bind is it a genuine CSI/storage failure.
        gpu_miss = ('insufficient nvidia.com/gpu' in msg
                    or 'insufficient gpu' in msg)
        label_miss = (("didn't match" in msg
                       and ('affinity' in msg or 'selector' in msg))
                      or 'untolerated taint' in msg or 'had taint' in msg)
        if gpu_miss:
            cause = ('INSUFFICIENT GPU — nodes matched the selector/taint but '
                     'none advertise nvidia.com/gpu (NVIDIA device plugin '
                     'missing / not Ready, or non-GPU instance type)')
        elif label_miss:
            cause = ('NO NODE matches the required label / taint — the cache '
                     'or ephemeral NodePool is missing, mislabelled, or its '
                     'dedicated taint differs from what the controller '
                     'tolerates (LABEL / POOL MISCONFIG)')
        elif last_phase == 'unknown':
            # The pod read returned None on EVERY poll: it was never observed
            # via the API. This is NOT a node / disk problem (no node was ever
            # asked for) — the create likely failed silently, the pod was
            # deleted out-of-band, or we hit API eventual-consistency.
            cause = ('the canary pod was NEVER observed via the API '
                     '(creation failed silently, the pod was deleted '
                     'out-of-band, or API eventual-consistency) — this is NOT '
                     'a node / disk problem; check controller RBAC (pods '
                     'get/create) and API server health')
        elif last_phase == 'Failed':
            # Pod scheduled onto a node but the container did not stay
            # Running — crashed / OOMKilled / bad image entrypoint. The node
            # delivered; the canary container itself failed.
            cause = ('the canary container did NOT stay Running on a '
                     'scheduled node (crashed / OOMKilled / bad image '
                     'entrypoint) — the node scheduled but the container '
                     'failed; check the canary image and resource requests')
        elif pvc_name is not None and last_pvc_phase != 'Bound':
            # Pod was NOT reported Unschedulable for GPU/label reasons, yet
            # the PVC never bound → disk capacity / StorageClass.
            if 'insufficient' in msg and ('storage' in msg or 'topolvm' in msg):
                cause = ('PVC unbound: CSI reports INSUFFICIENT STORAGE for '
                         'the requested cache volume size')
            else:
                cause = (f'PVC {pvc_name!r} never reached Bound '
                         f'(last phase: {last_pvc_phase}) — the WFFC '
                         f'StorageClass lacks compatible topology/capacity, '
                         f'or its CSI provisioner is misconfigured')
        elif last_unsched:
            cause = (f'pod is Unschedulable ({last_phase}) — scheduler could '
                     f'not place it on any node')
        else:
            cause = (f'pod SCHEDULED (no PodScheduled=False condition) but '
                     f'never reached Running (last phase: {last_phase}) — a '
                     f'node WAS delivered, so this is NOT a pool/label/taint '
                     f'misconfig. Most likely the ~16GB daemon image is still '
                     f'being pulled onto the fresh node and did not finish '
                     f'within the {CANARY_GRACE_SECS}s grace (cold image pull), '
                     f'or the node is still finishing provisioning. Raise '
                     f'POOL_CANARY_GRACE_SECS for clusters with slow cold '
                     f'image pulls before suspecting the node pool')
        msg_out = f'{head} Cause: {cause}. Scheduler said: {last_unsched!r}.'
        if pool == 'ephemeral':
            # The ephemeral tier is ENABLED by default, so this failure is
            # expected on a cluster that has no dedicated ephemeral GPU node.
            # Name the exact escape hatch so an operator hitting it knows how to
            # opt out (and that opting out also disables the runtime overflow
            # route, not just this init canary).
            msg_out += (
                ' NOTE: the ephemeral tier is ENABLED by default; if this '
                'cluster has no dedicated pool-type=ephemeral GPU node (e.g. a '
                'single-GPU deployment that never uses the ephemeral overflow '
                'tier), opt OUT by setting POOL_INIT_EPHEMERAL_CANARY_ENABLED='
                'false - this skips the ephemeral init canary AND disables '
                'runtime overflow to the ephemeral tier (kumo-ai/kumo#29865).')
        return msg_out


def _parse_result_annotation(value: str) -> Optional[tuple[str, float]]:
    """Parse ``<config-hash>@<epoch-seconds>``; None if malformed."""
    if '@' not in value:
        return None
    cfg_hash, _, validated_at_raw = value.partition('@')
    if not cfg_hash or not validated_at_raw:
        return None
    try:
        return cfg_hash, float(validated_at_raw)
    except ValueError:
        return None


async def _read_cached_result(
    runner: _CanaryRunner) -> Optional[tuple[str, float]]:
    """Read ``(config_hash, validated_at_epoch)`` from the controller
    Deployment's validation annotation, or None if absent / unreadable /
    malformed (best-effort; failure to read just forces re-validation)."""
    try:
        dep = cast(
            V1Deployment,
            await asyncio.to_thread(
                runner._apps.read_namespaced_deployment,
                name=CONTROLLER_DEPLOYMENT_NAME,
                namespace=runner._namespace,
            ),
        )
    except ApiException as exc:
        if exc.status != 404:
            log.warning(
                'canary: reading controller Deployment %r for cached result '
                'failed (status=%s); will re-validate',
                CONTROLLER_DEPLOYMENT_NAME, exc.status)
        return None
    annotations = ((dep.metadata.annotations or {})
                   if dep.metadata else {})
    value = annotations.get(RESULT_ANNOTATION_KEY)
    if not value:
        return None
    return _parse_result_annotation(value)


async def _record_success(runner: _CanaryRunner, config_hash: str) -> None:
    """Stamp the validation result on the controller Deployment as an
    annotation (covered by the existing ``deployments [get,patch]`` grant).
    Best-effort: a failure just means the next start re-validates."""
    value = f'{config_hash}@{int(time.time())}'
    body = {'metadata': {'annotations': {RESULT_ANNOTATION_KEY: value}}}
    try:
        await asyncio.to_thread(
            runner._apps.patch_namespaced_deployment,
            name=CONTROLLER_DEPLOYMENT_NAME,
            namespace=runner._namespace,
            body=body,
        )
    except ApiException as exc:
        log.warning(
            'canary: recording success on controller Deployment %r failed '
            '(status=%s); next start will re-validate',
            CONTROLLER_DEPLOYMENT_NAME, exc.status)


async def run_init_canary(
    manager: K8sPoolWorkerManager,
    *,
    pvc_spec_factory: Callable[[int], PvcSpec],
    cache_tiers_gib: Optional[List[int]] = None,
    cache_tier_specs: Optional[List[PvcSpec]] = None,
    prevalidated_cache_tiers: Optional[PrevalidatedCacheTierSource] = None,
) -> InitCanaryResult:
    """Validate Diskgraph node-pool capabilities at controller startup.

    Launches one cache canary per distinct ``(ssd_gib, gpu_count)`` tier, each
    mirroring the real daemon pod's scheduling + GPU + cache disk requirements.
    The eph canary is OPT-OUT (:data:`EPH_CANARY_ENABLED` /
    ``POOL_INIT_EPHEMERAL_CANARY_ENABLED``, default enabled): it validates the
    dedicated ephemeral GPU pool at init. A single-GPU / no-ephemeral-node
    deployment sets ``POOL_INIT_EPHEMERAL_CANARY_ENABLED=false`` to skip it so
    controller readiness is not gated on an ephemeral node it does not have.
    No-op when :data:`CANARY_ENABLED` is
    False (``POOL_INIT_CANARY_ENABLED=false``). Skips when a fresh
    validation annotation (within :data:`CANARY_RESULT_TTL_SECS`) on the
    controller Deployment matches the current pool-config hash. That hash
    is keyed on the cache tiers, StorageClass, cache-node selector keys,
    and the cache- and eph-pool taint VALUES — so re-pointing a pool taint
    forces re-validation. An expected failure disables only that capability;
    startup raises :class:`InitCanaryValidationError` when no cache tier is
    usable. Unexpected errors continue to propagate.

    Args:
        manager: the same ``K8sPoolWorkerManager`` the controller's k8s
            strategy uses — the canary borrows its CoreV1Api, namespace,
            service account, image, selectors, taints, and StorageClass.
        cache_tiers_gib: legacy SSD-only tier input. Used when
            ``cache_tier_specs`` is omitted.
        cache_tier_specs: exact cache capacity tiers from
            ``pool_diskgraph_node_types``. Preferred for GPU-aware validation.
        pvc_spec_factory: ``Callable[[int], PvcSpec]`` — the SAME factory
            the strategy passes to ``provision_pool_worker``, so the
            canary's PVC is sized exactly as the daemon's for that tier.
        prevalidated_cache_tiers: exact cache tiers already proven by a
            Registered daemon and its persisted PVC capacity, or an async
            loader for that evidence. The loader runs only after the global
            disable and fresh-result bypasses.
    """
    if cache_tier_specs is None:
        cache_tier_specs = [
            pvc_spec_factory(tier_gib)
            for tier_gib in sorted(set(cache_tiers_gib or []))
        ]
    tier_specs_by_key = {
        (spec.size_gib, spec.gpu_count): spec for spec in cache_tier_specs
    }
    tier_specs = [
        tier_specs_by_key[key] for key in sorted(tier_specs_by_key)
    ]
    all_cache_tiers = frozenset(tier_specs_by_key)
    if not CANARY_ENABLED:
        log.info(
            'Diskgraph pool init canary: DISABLED via '
            'POOL_INIT_CANARY_ENABLED; controller readiness is decoupled '
            'from pool provisioning. Pool misconfig will only surface at '
            'first job.')
        return InitCanaryResult(
            available_cache_tiers=all_cache_tiers,
            ephemeral_available=EPH_CANARY_ENABLED,
        )
    if not tier_specs:
        raise InitCanaryValidationError(
            'Diskgraph pool init canary: no cache tiers configured '
            '(pool_diskgraph_node_types is empty) — cannot validate the '
            'cache pool.')
    tiers = sorted({spec.size_gib for spec in tier_specs})
    tier_keys = [
        f'{spec.size_gib}Gi/{spec.gpu_count}gpu' for spec in tier_specs
    ]
    storage_class = manager.storage_class
    runner = _CanaryRunner(manager)

    # Sample the eph scheduling (reused for the eph canary below) and the
    # first cache tier's scheduling so the hash is keyed on the actual taint
    # VALUES the canary tolerates.
    eph_sched = manager.eph_canary_scheduling()
    cache_sched0 = manager.cache_canary_scheduling(
        tier_specs[0].size_gib, tier_specs[0].gpu_count)

    config_hash = pool_config_hash(
        tiers_gib=tiers,
        tier_specs=tier_keys,
        storage_class=storage_class,
        cache_node_selector_keys=[
            DISKGRAPH_POOL_TYPE_LABEL,
            DISKGRAPH_SSD_SIZE_LABEL,
            DISKGRAPH_GPU_COUNT_LABEL,
        ],
        cache_taint=_taint_str(cache_sched0.tolerations[0]),
        eph_taint=_taint_str(eph_sched.tolerations[0]),
        eph_canary_enabled=EPH_CANARY_ENABLED,
        extra_tolerations=[
            json.dumps(_toleration_key(t))
            for t in (cache_sched0.tolerations[1:] + eph_sched.tolerations[1:])
        ],
    )

    cached = await _read_cached_result(runner)
    if cached is not None:
        cached_hash, validated_at = cached
        age = time.time() - validated_at
        if cached_hash == config_hash and 0 <= age < CANARY_RESULT_TTL_SECS:
            log.info(
                'Diskgraph pool init canary: SKIP — fresh matching result '
                '(config_hash=%s, age=%.0fs < TTL %ds).', config_hash, age,
                CANARY_RESULT_TTL_SECS)
            return InitCanaryResult(
                available_cache_tiers=all_cache_tiers,
                ephemeral_available=EPH_CANARY_ENABLED,
            )
        log.info(
            'Diskgraph pool init canary: cached result stale/mismatched '
            '(cached_hash=%s vs %s, age=%.0fs); re-validating.', cached_hash,
            config_hash, age)

    log.info(
        'Diskgraph pool init canary: validating %d cache tier(s) %s%s '
        '(grace=%ds each, storage_class=%s).', len(tier_specs),
        tier_keys,
        ' + eph pool' if EPH_CANARY_ENABLED else '',
        CANARY_GRACE_SECS, storage_class)

    # Reconcile leftovers from any crashed prior validation before launching.
    await runner.reconcile_leftovers()

    # Cache canaries — label + taint + GPU + disk (PVC sized per tier).
    # Tiers are validated SEQUENTIALLY on purpose: each provisions one
    # GPU node and cache volume at a time, limiting concurrent validation.
    # Parallelizing (asyncio.gather) would burst all N cache tiers + the eph
    # canary as N+1 simultaneous GPU nodes — deliberately avoided.
    if callable(prevalidated_cache_tiers):
        prevalidated = await prevalidated_cache_tiers()
    else:
        prevalidated = prevalidated_cache_tiers
    available_cache_tiers = set(prevalidated or ()) & all_cache_tiers
    actively_validated_cache_tiers: set[tuple[int, int]] = set()
    cache_failures: list[str] = []
    for spec in tier_specs:
        tier_gib = spec.size_gib
        gpu_count = spec.gpu_count
        tier_key = (tier_gib, gpu_count)
        if tier_key in available_cache_tiers:
            log.info(
                'Diskgraph pool init canary: SKIP cache tier %s — a '
                'Registered daemon already proves this exact persisted tier.',
                tier_key,
            )
            continue
        sched = manager.cache_canary_scheduling(tier_gib, gpu_count)
        pvc_size = manager.cache_pvc_size_str(spec)
        expectation = (
            f'cache tier {tier_gib} GiB: node labelled '
            f'{DISKGRAPH_POOL_TYPE_LABEL}=cache, '
            f'{DISKGRAPH_SSD_SIZE_LABEL}={tier_gib}, '
            f'{DISKGRAPH_GPU_COUNT_LABEL}={gpu_count}, tolerating taint '
            f'{[_taint_str(t) for t in sched.tolerations]}, with '
            f'nvidia.com/gpu>={gpu_count} and a {pvc_size} PVC bindable on '
            f'StorageClass {storage_class!r}')
        try:
            await runner.run_one(
                pool=f'cache-{tier_gib}-gpu{gpu_count}',
                sched=sched,
                pvc_size=pvc_size,
                storage_class=storage_class,
                expectation=expectation,
                gpu_count=gpu_count,
            )
        except InitCanaryValidationError as exc:
            cache_failures.append(str(exc))
            log.error(
                'Diskgraph pool init canary: cache tier %s is unavailable; '
                'new placements will not use it. %s', tier_key, exc)
        else:
            available_cache_tiers.add(tier_key)
            actively_validated_cache_tiers.add(tier_key)

    if not available_cache_tiers:
        detail = ' | '.join(cache_failures) or 'no tier produced usable proof'
        raise InitCanaryValidationError(
            'Diskgraph pool init canary: no cache tier is available; refusing '
            f'to start without durable placement capacity. {detail}')

    # Ephemeral canary — label + taint + GPU + scheduling only. Eph disk is
    # ephemeral-storage sized per-snapshot at runtime, so the disk dimension
    # is validated at job time, not here. Reuse the eph_sched sampled above.
    #
    # OPT-OUT: the eph tier is validated at init BY DEFAULT (see
    # EPH_CANARY_ENABLED), for the deployments that run a dedicated ephemeral
    # pool. A single-GPU / no-ephemeral-node deployment (which never uses the
    # eph tier) sets POOL_INIT_EPHEMERAL_CANARY_ENABLED=false to skip it; an
    # enabled probe failure disables runtime overflow for this process.
    ephemeral_available = False
    if EPH_CANARY_ENABLED:
        eph_expectation = (
            f'ephemeral pool: node labelled '
            f'{DISKGRAPH_POOL_TYPE_LABEL}=ephemeral, '
            f'tolerating taints {[_taint_str(t) for t in eph_sched.tolerations]}, '
            f'with nvidia.com/gpu>=1 (eph disk is '
            f'ephemeral-storage sized per-job at runtime, validated at job '
            f'time)')
        try:
            await runner.run_one(
                pool='ephemeral',
                sched=eph_sched,
                pvc_size=None,
                storage_class=storage_class,
                expectation=eph_expectation,
            )
        except InitCanaryValidationError as exc:
            log.error(
                'Diskgraph pool init canary: ephemeral capacity is unavailable; '
                'runtime overflow will be disabled. %s', exc)
        else:
            ephemeral_available = True
    else:
        log.info(
            'Diskgraph pool init canary: SKIP ephemeral pool - opted out via '
            'POOL_INIT_EPHEMERAL_CANARY_ENABLED=false, so controller readiness '
            'is not gated on a dedicated ephemeral GPU node (single-GPU / '
            'no-ephemeral-nodepool escape hatch). Unset the flag or set it true '
            'to validate the ephemeral pool at init.')

    if (actively_validated_cache_tiers == set(all_cache_tiers)
            and (not EPH_CANARY_ENABLED or ephemeral_available)):
        await _record_success(runner, config_hash)
        log.info('Diskgraph pool init canary: all pools validated; recorded '
                 'success (config_hash=%s).', config_hash)

    return InitCanaryResult(
        available_cache_tiers=frozenset(available_cache_tiers),
        ephemeral_available=ephemeral_available,
    )
