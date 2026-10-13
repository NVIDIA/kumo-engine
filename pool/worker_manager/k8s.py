# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes-backed pool-worker manager.

A *pool worker* is the triplet (k8s node, daemon pod, per-node PVC).
``K8sPoolWorkerManager`` owns the k8s-API operations for:

- Pool-worker lifecycle: ``provision_pool_worker`` / ``delete_pool_worker``
  (PVC + daemon Deployment; node provisioning delegated to Karpenter).
- Activity-pod lifecycle: ``provision_activity_pod`` (cache pool, pinned
  to a pool worker via PVC) and ``provision_ephemeral_activity_pod``
  (eph pool, emptyDir on RAID0 NVMe). Both are wrapped as ``batch/v1
  Job`` so the K8s Job controller respawns crashed pods within
  ``backoffLimit`` (design §6.1 + §9 row 1).
- Status / listing: ``get_status``, ``list_pool_workers``,
  ``list_activity_pods``.

Snapshot ingest, restore, deletion, heartbeat tracking, and rebind
orchestration are handled by controller code.

"""
import asyncio
import contextlib
import copy
import dataclasses
import hashlib
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, cast

from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from kubernetes.client.models import (
    V1Affinity,
    V1Capabilities,
    V1ConfigMapEnvSource,
    V1Container,
    V1Deployment,
    V1DeploymentSpec,
    V1DeploymentStrategy,
    V1EmptyDirVolumeSource,
    V1EnvFromSource,
    V1EnvVar,
    V1EnvVarSource,
    V1Job,
    V1JobSpec,
    V1KeyToPath,
    V1LabelSelector,
    V1ObjectFieldSelector,
    V1ObjectMeta,
    V1PersistentVolumeClaim,
    V1PersistentVolumeClaimSpec,
    V1PersistentVolumeClaimVolumeSource,
    V1Pod,
    V1PodAffinity,
    V1PodAffinityTerm,
    V1PodAntiAffinity,
    V1PodDisruptionBudget,
    V1PodDisruptionBudgetSpec,
    V1PodSecurityContext,
    V1PodSpec,
    V1PodTemplateSpec,
    V1ResourceRequirements,
    V1SeccompProfile,
    V1SecretKeySelector,
    V1SecretVolumeSource,
    V1SecurityContext,
    V1Toleration,
    V1Volume,
    V1VolumeMount,
)
from kubernetes.client.rest import ApiException

from io_utils import logging
from io_utils.errors import mark_exception_restricted
from pool.task_queues import ACTIVITY_WORKER_POD_NAME_ENV
from pool.timeouts import (
    ACTIVITY_JOB_DELETE_TIMEOUT_SEC,
    POOL_WORKER_PROVISION_TIMEOUT_SEC,
)
from pool.worker_manager.types import (
    ActivityPodInfo,
    ApiServerThrottledError,
    ApiServerTransientError,
    InvalidPvcSpecError,
    K8sPermissionError,
    NodePoolNotFoundError,
    PodStatus,
    PoolWorkerAlreadyExistsError,
    PoolWorkerCleanupFailedError,
    PoolWorkerInfo,
    PoolWorkerNotFoundError,
    PoolWorkerNotReadyError,
    ProvisionTerminalError,
    ProvisionTimeoutError,
    PVCBindTimeoutError,
    PvcSpec,
    QuotaExceededError,
    ResourceSpec,
    SecretKeyRef,
    UnclassifiedApiError,
)

log = logging.getLogger(__name__)

# k8s Job activeDeadlineSeconds: safety net for Jobs the controller forgot
# entirely (crashed without storing a handle, or dangling-lease sweep missed).
# Default 48 h — long enough never to interfere with legitimate training runs
# while still cleaning up truly orphaned Jobs. Override via
# POOL_ACTIVITY_JOB_DEADLINE_HOURS for environments with unusually long
# training budgets.
_ACTIVITY_JOB_DEADLINE_SECS = (
    int(os.environ.get('POOL_ACTIVITY_JOB_DEADLINE_HOURS', '48')) * 3600)
# Pod waiting.reason values that indicate the pod will never start without
# operator intervention — fail-fast immediately on first detection.
# ErrImagePull is intentionally excluded (transient; kubelet retries it →
# eventual ImagePullBackOff).
_IMMEDIATE_TERMINAL_REASONS = frozenset([
    'CrashLoopBackOff',
    'CreateContainerConfigError',
    'CreateContainerError',
    'RunContainerError',
])

# ImagePullBackOff can be transient (ECR auth refresh, brief network blip).
# Allow it to retry for this long before treating it as terminal.  Kubelet
# is already doing exponential backoff internally; we just bound the total
# wait so a genuinely bad image tag doesn't block the slot indefinitely.
# Override via POOL_IMAGE_PULL_BACKOFF_GRACE_MINS.
_IMAGE_PULL_BACKOFF_GRACE_SECS = (
    int(os.environ.get('POOL_IMAGE_PULL_BACKOFF_GRACE_MINS', '10')) * 60)


def _runtime_pod_security_context(
        fs_group: Optional[int] = None) -> V1PodSecurityContext:
    """Restricted-compatible baseline, with an optional CSI volume group."""
    kwargs: Dict[str, Any] = {
        'seccomp_profile': V1SeccompProfile(type='RuntimeDefault'),
    }
    if fs_group is not None:
        kwargs.update(fs_group=fs_group,
                      fs_group_change_policy='OnRootMismatch')
    return V1PodSecurityContext(**kwargs)


def _runtime_container_security_context() -> V1SecurityContext:
    # run_as_non_root is an assertion, not a UID: Pod Security Admission runs
    # before the image is pulled, so `restricted` rejects any pod that does not
    # declare it. The kubelet then verifies it at container start, which
    # requires a numeric USER (see docker/Dockerfile.kumo_backend_{cpu,gpu}).
    #
    # Deliberately no run_as_user: pinning one would be rejected by OpenShift,
    # whose SCC assigns a UID from the namespace's own range.
    return V1SecurityContext(
        allow_privilege_escalation=False,
        privileged=False,
        run_as_non_root=True,
        read_only_root_filesystem=os.environ.get(
            'POOL_WORKER_READ_ONLY_ROOT_FILESYSTEM', 'true').strip().lower()
        not in ('0', 'false', 'no', 'off'),
        capabilities=V1Capabilities(drop=['ALL']),
    )


# ---------------------------------------------------------------------------
# Labels and conventions
# ---------------------------------------------------------------------------

LABEL_APP = 'app'
LABEL_APP_VALUE_DAEMON = 'diskgraph-daemon'
LABEL_APP_VALUE_ACTIVITY = 'diskgraph-activity-pod'
LABEL_APP_VALUE_ACTIVITY_EPH = 'diskgraph-eph-activity-pod'
LABEL_TENANT = 'tenant'
# Bare label keys per design §6.1 canonical schema. Stamped on daemon
# pod / PVC / activity pods so all three can be found via a single
# selector when reconciling or deleting.
LABEL_POOL_WORKER_UID = 'pool-worker-uid'
# Per-job tracking key. Activity pods carry this so the controller can
# index them as (pool_worker_uid, job_id) tuples for orphan cleanup
# and rebind relaunch (design §6.1).
#
# Holds a SHA-256-derived hex digest of the raw caller-supplied job_id
# (Temporal workflow_id), NOT the raw value. Raw Temporal workflow_ids
# routinely contain characters K8s rejects in label values (`:` and
# `@` aren't in the [A-Za-z0-9._-] charset; total length can exceed
# the 63-char cap), so the manager hashes on stamp + selector. The
# unhashed value is preserved on :data:`ANNOTATION_JOB_ID` so
# operators can recover it via ``kubectl describe`` and so
# :func:`_pod_to_activity_pod_info` can return the raw form to
# callers transparently.
LABEL_JOB_ID_HASH = 'job-id-hash'
# Human-readable annotation carrying the raw, unhashed job_id
# (Temporal workflow_id, e.g.
# ``dg-ingest:NA-abc123@graph-data-version-456:01HXYZ789``).
# Annotations have no K8s-imposed charset / length constraints, so
# the raw form survives. Stamped on Job metadata AND on the pod
# template metadata so it propagates onto every spawned pod and
# shows up in ``kubectl describe pod``.
ANNOTATION_JOB_ID = 'kumo.ai/job-id'
# Records Kumo's configured extras separately from admission-added tolerations.
_ANNOTATION_POOL_EXTRA_TOLERATIONS_HASH = 'kumo.ai/pool-extra-tolerations-hash'
# Snapshot identity (which snapshot this activity reads/writes).
#
# Holds the truncated SHA-256 hex of the raw snapshot_id, NOT the raw
# value — same K8s-label-constraint reason as :data:`LABEL_JOB_ID_HASH`
# (snapshot ids may carry uri/scheme-like segments or exceed the
# 63-char cap). Raw value preserved on :data:`ANNOTATION_SNAPSHOT_ID`.
LABEL_SNAPSHOT_ID_HASH = 'snapshot-id-hash'
# Human-readable annotation carrying the raw, unhashed snapshot_id.
ANNOTATION_SNAPSHOT_ID = 'kumo.ai/snapshot-id'
# Activity name: 'train' / 'predict' / 'ingest' / etc. Used by the
# controller for activity-class filtering during reconciliation.
LABEL_ACTIVITY = 'activity'
# DG pool-role label, namespaced + diskgraph-specific so it doubles as the
# BYOC contract label a customer stamps on their own cache/eph NodePools (we
# do NOT reuse kumo's internal `kumo.ai/instance-type` convention for DG — we
# don't expect customer infra to adopt it). Used two ways: (1) the daemon /
# activity pod `nodeSelector` that targets cache vs eph nodes; (2) a label on
# the pods themselves to narrow list/delete queries to one pool. The same key
# is consumed by kumo-control (lvmd/CSI nodeSelector) and stamped by
# kumo-delivery NodePools — all DG components agree on it.
DISKGRAPH_POOL_TYPE_LABEL = 'kumo.ai/diskgraph-pool-type'
POOL_TYPE_CACHE = 'cache'
POOL_TYPE_EPHEMERAL = 'ephemeral'

_CPU_DSHM_SIZE_LIMIT = '8Gi'

# Cache-volume mount point inside daemon / activity containers. Matches design
# §6.3 / §7.2 / §8.5 — the canonical path the daemon image expects to
# read snapshots from (DISK_ROOT env var also points here).
NVME_MOUNT_PATH = '/mnt/localssd'

# Daemon scratch mount point. Ingest writes parquet temp / spill here;
# the emptyDir uses the node's kubelet ephemeral-storage filesystem.
# EKS cache nodes put that filesystem on dedicated EBS, while AKS uses
# the node pool's OS/temporary disk. Activity pods do not mount it.
EBS_SCRATCH_MOUNT_PATH = '/mnt/ebs-scratch'

# Default PVC request per design §8.4 (original single-pool formula with 1% /
# 50 GiB headroom against a ~1.8 TiB local-NVMe tier).
DEFAULT_STORAGE_REQUEST = '1750Gi'

_GCP_APPLICATION_CREDENTIALS_VOLUME = 'gcp-application-credentials'

# Cadence for polling pod status while waiting for Registered.
_POLL_INTERVAL_SECS = 2.0

# Headroom multiplier when sizing a PVC from ``snapshot_size_bytes``.
_STORAGE_HEADROOM = 1.25

# Cloud-/autoscaler-neutral SIZE label for cache nodes — the second dimension
# on top of the cache pool role. Its value is the cache PVC capacity in GiB
# (e.g. "1750"). The platform stamps it on each
# single-instance-type cache NodePool (kumo-delivery
# apps/kumo-dataplane/lib/nodepool-cache.libsonnet). The controller picks the
# smallest size that fits the snapshot and selects it by this label — so it
# picks a tier with enough storage on any cloud/autoscaler, independent of
# whether the CSI provisioner uses local or network-block storage.
# Eph nodes need no size label; their sizing rides the standard
# ephemeral-storage request.
DISKGRAPH_SSD_SIZE_LABEL = 'kumo.ai/diskgraph-ssd-size'
DISKGRAPH_GPU_COUNT_LABEL = 'kumo.ai/diskgraph-gpu-count'


def _resolve_service_account(service_account: Optional[str]) -> str:
    """SA for the manager AND the daemon/activity/canary pods it launches.

    Explicit argument wins; otherwise POD_SERVICE_ACCOUNT (downward-API
    env injected by the OSS Helm chart) so launched pods inherit the
    controller's own SA; otherwise the legacy SaaS default
    (``kumo-temporal``). ``None`` and ``''`` both mean unset
    — empty strings fall through, matching the POD_NAMESPACE sites.
    """
    return (service_account or os.environ.get('POD_SERVICE_ACCOUNT')
            or 'kumo-temporal')


@dataclasses.dataclass(frozen=True)
class _PoolScheduling:
    """The nodeSelector + tolerations a pod must carry to
    land on one Diskgraph node pool. Built by
    :meth:`K8sPoolWorkerManager.cache_canary_scheduling` /
    :meth:`eph_canary_scheduling` so the init canary reproduces the real
    daemon / eph pod's scheduling envelope exactly.
    """
    node_selector: Dict[str, str]
    tolerations: List[V1Toleration]


class K8sPoolWorkerManager:
    """Pool-worker manager backed by the Kubernetes API.

    Safe for concurrent use — each method issues a fresh k8s request
    and carries no in-process state beyond constructor config. All
    public methods are async; sync API calls are dispatched via
    ``asyncio.to_thread``.
    """
    def __init__(
        self,
        namespace: str,
        tenant: str,
        storage_class: str,
        daemon_image: str,
        daemon_command: Optional[List[str]] = None,
        daemon_args: Optional[List[str]] = None,
        pool_taint_key: str = 'dedicated',
        service_account: Optional[str] = None,
        runtime_fs_group: Optional[int] = 999,
        daemon_resources: Optional[ResourceSpec] = None,
        default_storage_request: str = DEFAULT_STORAGE_REQUEST,
        ebs_scratch_mount_path: str = EBS_SCRATCH_MOUNT_PATH,
        # Substrate wiring stamped onto pool pods (daemon/activity/eph) so they
        # match the chart-managed static workers. All optional: when unset (the
        # managed-SaaS case) the pods are unchanged. BYOC sets the shared env,
        # optional Temporal mTLS Secret.
        pod_env_configmap: Optional[str] = None,
        temporal_cert_secret: Optional[str] = None,
        temporal_cert_dir: Optional[str] = None,
        tenant_secret_key_ref: Optional[SecretKeyRef] = None,
        statedb_secret: Optional[str] = None,
        statedb_ca_secret: Optional[str] = None,
        statedb_ca_secret_key: Optional[str] = None,
        secrets_mount_path: Optional[str] = None,
        gcp_application_credentials_secret: Optional[str] = None,
        gcp_application_credentials_secret_key: Optional[str] = None,
        gcp_application_credentials_path: Optional[str] = None,
        pod_template_labels: Optional[Dict[str, str]] = None,
        pool_extra_tolerations: Optional[List[V1Toleration]] = None,
        load_kube_config: bool = True,
        kube_context: Optional[str] = None,
        core_v1_api: Optional[k8s_client.CoreV1Api] = None,
        apps_v1_api: Optional[k8s_client.AppsV1Api] = None,
        batch_v1_api: Optional[k8s_client.BatchV1Api] = None,
        policy_v1_api: Optional[k8s_client.PolicyV1Api] = None,
    ) -> None:
        """Construct the manager.

        Args:
            namespace: k8s namespace for all resources (tenant
                dataplane, e.g. ``kumo-dataplane-{tenant}``). May also
                be the chart's own release namespace when
                ``POD_NAMESPACE`` steers
                :func:`pool.service._configured_k8s_pool_names` (OSS/
                BYOC single-namespace deploys).
            tenant: tenant identifier; stamped on resources as a label.
            storage_class: ``WaitForFirstConsumer`` CSI StorageClass
                for cache PVCs (for example TopoLVM on EKS or zonal
                Azure Disk on AKS).
            daemon_image: container image for the daemon pod.
            daemon_command: optional ``command`` (entrypoint) override
                for the daemon container. ``None`` uses the image's
                default ENTRYPOINT.
            daemon_args: optional ``args`` for the daemon container.
            pool_taint_key: shared taint key on cache + eph NodePools.
                Default ``"dedicated"`` matches the per-tenant taint
                kumo-delivery applies to both Diskgraph NodePools (see
                ``apps/kumo-dataplane/lib/nodepool-{cache,eph}.libsonnet``).
                Toleration values are derived per-pool from ``tenant``:
                cache pods tolerate ``dedicated=diskgraph-<tenant>``,
                eph pods tolerate ``dedicated=diskgraph-eph-<tenant>``
                — same prefixes the libsonnet uses for
                ``dedicatedTaintValue``.

                NB: ``dedicated`` is reused across other kumo dataplane
                NodePools (spark, trainer, graphstore — each with its
                own ``dedicated=<workload>-<tenant>`` taint), so the key
                is *not* what isolates Diskgraph nodes from other
                workloads — the **value** (``diskgraph-<tenant>`` /
                ``diskgraph-eph-<tenant>``) does. This is fine as long
                as no other workload installs a wildcard
                ``key=dedicated, operator=Exists`` toleration that
                could match a Diskgraph value. The one intentional
                wildcard tolerator is TopoLVM's ``lvmd`` / ``node``
                DaemonSet (kumo-control side), which uses
                ``dedicated Exists`` because it must run on every
                tenant's cache node regardless of the per-tenant taint
                value to register the driver in ``CSINode`` and publish
                ``CSIStorageCapacity`` for the local NVMe Volume Group.
            pool_extra_tolerations: additional Kubernetes tolerations for all
                runtime pool pods and canaries. They do not replace the built-in
                tenant tolerations or alter node selectors. None and [] both
                preserve existing tolerations.
            service_account: ServiceAccount the daemon, activity pods,
                and ephemeral activity pods all run under. When
                ``None`` (the default), resolved by
                :func:`_resolve_service_account`: the
                ``POD_SERVICE_ACCOUNT`` env var (downward-API,
                injected by the OSS Helm chart) so launched pods
                inherit the controller's own SA, falling back to
                ``kumo-temporal`` otherwise. The ``kumo-temporal``
                fallback aligns with the rest of the kumo dataplane
                workloads so IRSA + RBAC bindings can be reused 1:1.
            runtime_fs_group: group applied to daemon and cache-activity pods
                for root-owned CSI volumes. The stock image uses 999. Set
                ``None`` on OpenShift so its SCC can assign a namespace-valid
                supplemental group.
            daemon_resources: optional ``ResourceSpec`` for the daemon
                container's cpu / memory / gpu request + limit knobs.
                ``None`` → default Burstable QoS spec
                (``cpu_request='1'``, ``memory_request='2Gi'``, no
                limits) — small footprint since the daemon isn't on
                the GPU/IO hot path; activity pods do the heavy
                lifting. Pass a fully-populated ``ResourceSpec`` (all
                cpu_* + memory_* set with request==limit) to promote
                the daemon to Guaranteed QoS so kubelet won't evict
                it ahead of any Burstable pod under node pressure.
            default_storage_request: fallback PVC size when
                ``PvcSpec.snapshot_size_bytes`` is unset.
            ebs_scratch_mount_path: path inside the daemon container
                where its scratch ``emptyDir`` is mounted (default
                ``/mnt/ebs-scratch``). It uses the node's kubelet
                ephemeral-storage filesystem. Activity pods do not
                see this mount.
            load_kube_config: when True, load in-cluster config (or
                ``~/.kube/config`` for laptop runs). Set False when the
                api clients are injected directly.
            core_v1_api / apps_v1_api / batch_v1_api / policy_v1_api:
                optional pre-built clients (tests inject mocks). When
                all four are None, new clients are constructed; passing
                any subset is treated as "construct the rest".
        """
        self._namespace = namespace
        self._tenant = tenant
        self._storage_class = storage_class
        self._daemon_image = daemon_image
        self._daemon_command = daemon_command
        self._daemon_args = daemon_args
        self._pool_taint_key = pool_taint_key
        # Toleration values are tenant-derived per the kumo-delivery
        # convention (`dedicatedTaintValue` in nodepool-{cache,eph}
        # libsonnet): cache nodes get `diskgraph-<tenant>`, eph nodes
        # get `diskgraph-eph-<tenant>`. Pods must match these on the
        # `dedicated` key to schedule.
        self._cache_pool_taint_value = f'diskgraph-{tenant}'
        self._eph_pool_taint_value = f'diskgraph-eph-{tenant}'
        self._pool_extra_tolerations = copy.deepcopy(pool_extra_tolerations
                                                     or [])
        self._service_account = _resolve_service_account(service_account)
        self._runtime_fs_group = runtime_fs_group
        # Default daemon resource shape (Burstable QoS) — small CPU /
        # memory request, no limits — historical behavior. Caller
        # promotes to Guaranteed by passing a fully-populated
        # ResourceSpec.
        self._daemon_resources = daemon_resources or ResourceSpec(
            cpu_request='1', memory_request='2Gi')
        self._default_storage_request = default_storage_request
        self._ebs_scratch_mount_path = ebs_scratch_mount_path
        self._pod_env_configmap = pod_env_configmap
        self._temporal_cert_secret = temporal_cert_secret
        self._temporal_cert_dir = temporal_cert_dir
        self._tenant_secret_key_ref = tenant_secret_key_ref
        self._statedb_secret = statedb_secret
        self._statedb_ca_secret = statedb_ca_secret
        self._statedb_ca_secret_key = statedb_ca_secret_key or 'ca.pem'
        self._secrets_mount_path = secrets_mount_path
        self._gcp_application_credentials_secret = (
            gcp_application_credentials_secret)
        self._gcp_application_credentials_secret_key = (
            gcp_application_credentials_secret_key)
        self._gcp_application_credentials_path = (
            gcp_application_credentials_path)
        self._pod_template_labels = dict(pod_template_labels or {})

        self._kube_context = kube_context
        self._load_kube_config_on_init = load_kube_config
        self._refresh_task: Optional['asyncio.Task[None]'] = None
        if (core_v1_api is not None and apps_v1_api is not None
                and batch_v1_api is not None and policy_v1_api is not None):
            self._core = core_v1_api
            self._apps = apps_v1_api
            self._batch = batch_v1_api
            self._policy = policy_v1_api
        else:
            if load_kube_config:
                _load_kube_config(context=kube_context)
            self._core = core_v1_api or k8s_client.CoreV1Api()
            self._apps = apps_v1_api or k8s_client.AppsV1Api()
            self._batch = batch_v1_api or k8s_client.BatchV1Api()
            self._policy = policy_v1_api or k8s_client.PolicyV1Api()
            # Log the actual cluster server URL so we can immediately
            # tell if the client is bound to the wrong cluster (testing
            # vs prod, dev-uw2 vs testing-uw4). This has bitten us
            # repeatedly when relying on kubectl's mutable current-context.
            try:
                host = k8s_client.Configuration.get_default_copy().host
                log.info(
                    'K8sPoolWorkerManager bound to k8s API server: %s '
                    '(context=%s, namespace=%s)', host, kube_context
                    or '<incluster-or-current>', namespace)
            except Exception:
                log.exception(
                    'K8sPoolWorkerManager could not determine the k8s '
                    'API server it is bound to.')

    @property
    def namespace(self) -> str:
        """Tenant namespace this manager operates in."""
        return self._namespace

    @property
    def kube_context(self) -> Optional[str]:
        """kubectl context (cluster name) the manager is bound to.

        ``None`` when running in-cluster against the current namespace
        (no explicit context was passed).
        """
        return self._kube_context

    @staticmethod
    def deployment_name_for_uid(pool_worker_uid: str) -> str:
        """Daemon Deployment name for the given pool_worker_uid.

        Exposed so consumers (the placement strategy, log helpers)
        don't have to re-derive the naming convention. The uid
        already carries its ``pool-worker-`` prefix; this just
        prepends ``diskgraph-daemon-``.
        """
        return _deployment_name_for_uid(pool_worker_uid)

    @staticmethod
    def activity_job_name(job_id: str, kind: str) -> str:
        """K8s Job name for the activity pod backing this placement.

        Exposed so :class:`K8sPlacementStrategy` can derive the name
        for ``wait_for_activity_job_pod_running`` without duplicating
        the hashing convention.
        """
        if kind == 'overflow':
            return f'diskgraph-eph-activity-{_hash_job_id(job_id)}'
        return f'diskgraph-activity-{_hash_job_id(job_id)}'

    async def start(self) -> None:
        """Start the background kubeconfig refresh task.

        ``kubernetes`` client v24 caches the exec-provider token but has
        a known refresh bug under long-lived processes — the cached
        token outlives the underlying ``aws eks get-token`` expiry and
        the API server returns 401. Off-cluster runs (devserver) are
        the only place this bites because in-cluster service account
        tokens come from kubelet and don't share this code path.
        Periodically reload kubeconfig so a fresh exec invocation gets
        called; this rebuilds the API clients against new credentials.
        """
        if (self._refresh_task is not None and not self._refresh_task.done()
                or not self._load_kube_config_on_init):
            return
        self._refresh_task = asyncio.create_task(
            self._kube_config_refresh_loop(), name='k8s-kubeconfig-refresh')

    async def stop(self) -> None:
        """Cancel the background kubeconfig refresh task. Idempotent."""
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._refresh_task
            self._refresh_task = None

    async def _kube_config_refresh_loop(self) -> None:
        # ``aws eks get-token`` tokens are 15 min; refresh well inside
        # that window so an in-flight API call always has a fresh token.
        interval_secs = 5 * 60
        while True:
            await asyncio.sleep(interval_secs)
            try:
                await asyncio.to_thread(_load_kube_config,
                                        context=self._kube_context)
                self._core = k8s_client.CoreV1Api()
                self._apps = k8s_client.AppsV1Api()
                self._batch = k8s_client.BatchV1Api()
                self._policy = k8s_client.PolicyV1Api()
                log.info(
                    'K8sPoolWorkerManager refreshed kubeconfig '
                    '(context=%s).', self._kube_context
                    or '<incluster-or-current>')
            except Exception:
                log.exception(
                    'K8sPoolWorkerManager kubeconfig refresh failed; '
                    'will retry in %ds.', interval_secs)

    # ------------------------------------------------------------------
    # Pool-worker lifecycle
    # ------------------------------------------------------------------

    async def provision_pool_worker(
        self,
        pool_worker_uid: Optional[str] = None,
        pvc_spec: Optional[PvcSpec] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = POOL_WORKER_PROVISION_TIMEOUT_SEC,
        require_existing_resources: bool = False,
    ) -> PoolWorkerInfo:
        """Provision a fresh pool worker (PVC + daemon Deployment).

        Ordering (design §8, option a): PVC is created first so the
        scheduler sees the PVC + its CSI capacity requirement when
        placing the daemon pod. If no existing cache-pool node has
        enough capacity, Karpenter provisions a new one.

        ``env``: optional ``Dict[str, str]`` of env vars stamped onto
        the daemon container. Same contract as
        :meth:`provision_activity_pod`'s ``env`` — manager is a faithful
        translator, caller composes the keys the daemon image expects
        (e.g. ``POOL_WORKER_UID``, ``DISK_ROOT``). ``None`` or empty
        dict → no V1EnvVar entries.

        ``pool_worker_uid`` has dual-use semantics:

        - ``None`` (default) — mint a fresh 10-char uid and create
          PVC + Deployment from scratch. A 409 from either create
          would indicate a uuid hex collision (astronomically
          unlikely) and propagates as
          :class:`PoolWorkerAlreadyExistsError`.
        - explicit string — **reclaim** an existing uid. Used in two
          scenarios: (a) daemon-pod recovery, where the controller
          re-binds the old PVC + node to a new daemon-pod incarnation;
          (b) controller crash recovery, where the controller restarts
          mid-provision and re-issues the call with the same uid. In
          both cases, a 409 from the create paths is treated as
          "already there from the prior attempt" — the call continues
          to ``_wait_for_daemon_registered`` rather than failing. If
          the wait then fails, rollback is intentionally **skipped**:
          the caller asked to reuse what was there, so the manager
          doesn't nuke it. Caller decides whether to
          ``delete_pool_worker(uid)`` + retry from scratch.

        ``require_existing_resources`` is the no-create reclaim path used when
        an existing worker belongs to a tier disabled by the init canary. It
        requires an explicit uid, verifies that the PVC and Deployment exist
        and are not terminating both before and after the registration wait,
        and never creates a PVC, Deployment, or PDB.

        Raises:
            ProvisionTimeoutError: daemon did not reach Registered.
            QuotaExceededError: infra refused capacity.
            InvalidPvcSpecError: PVC spec cannot be satisfied.
            NodePoolNotFoundError: a referenced NodePool / NodeClass CRD
                is missing (404).
            PoolWorkerAlreadyExistsError: 409 on a fresh-mint create
                (uid collision) — never raised on the reclaim path,
                where 409 is recovery signal.
        """
        # When the caller passes an explicit uid they're either (a)
        # reusing PVC + node from a daemon-pod recovery on the same
        # uid, or (b) recovering from a controller crash mid-provision.
        # Both cases are reclaim semantics: a 409 from create_* means
        # the resource is already there from the prior attempt, and we
        # should continue to _wait_for_daemon_registered rather than
        # treating it as a hard failure. Fresh-mint uids should never
        # collide (10-char hex from uuid4), so a 409 there is a real
        # bug and we propagate.
        is_reclaim = pool_worker_uid is not None

        uid = pool_worker_uid or uuid.uuid4().hex[:10]
        pvc_name = _pvc_name_for_uid(uid)
        deployment_name = _deployment_name_for_uid(uid)
        if require_existing_resources:
            if pool_worker_uid is None:
                raise ValueError('require_existing_resources needs an '
                                 'explicit pool_worker_uid.')
            await self._require_live_pool_worker_resources(uid)
            info = await self._wait_for_daemon_registered(uid, timeout)
            await self._require_live_pool_worker_resources(uid)
            return info

        storage_size = (self._storage_size_from_pvc_spec(pvc_spec) if pvc_spec
                        is not None else self._default_storage_request)
        # SSD-size label for the daemon's nodeSelector (= the chosen PVC size).
        # None on the rebind path (pvc_spec unset): the existing PVC's bound PV
        # pins the daemon to its node, so no size selector is applied.
        ssd_size_gib = pvc_spec.size_gib if pvc_spec is not None else None
        gpu_count = pvc_spec.gpu_count if pvc_spec is not None else None

        # Step 1: create PVC. On reclaim, 409 means the prior attempt
        # already created it — continue.
        existing_pvc_bound = False
        try:
            await asyncio.to_thread(self._create_pvc_sync, pvc_name, uid,
                                    storage_size)
        except PoolWorkerAlreadyExistsError:
            if not is_reclaim:
                raise
            existing_pvc_bound = await asyncio.to_thread(
                self._pvc_is_bound_sync, pvc_name)
            log.info(
                'provision_pool_worker: PVC %s already exists for '
                'uid=%s; reclaiming bound=%s', pvc_name, uid,
                existing_pvc_bound)

        try:
            try:
                # A Bound portable PV may constrain only a zone, so preserve
                # the known SSD/GPU tier in the Deployment's node selector.
                await asyncio.to_thread(self._create_daemon_deployment_sync,
                                        deployment_name, pvc_name, uid, env,
                                        ssd_size_gib, gpu_count)
            except PoolWorkerAlreadyExistsError:
                if not is_reclaim:
                    raise
                log.info(
                    'provision_pool_worker: Deployment %s already '
                    'exists for uid=%s; reclaiming', deployment_name, uid)
            # Create the PodDisruptionBudget right after the Deployment
            # exists — it does NOT need to wait for Registered, and
            # protecting the daemon pod the moment it can be scheduled
            # is strictly safer. Reclaim-idempotent (409 → already there).
            await asyncio.to_thread(self._create_daemon_pdb_sync, uid)
            info = await self._wait_for_daemon_registered(uid, timeout)
        except Exception:
            # On the reclaim path, the caller explicitly asked to
            # reuse the uid's resources — if reclaim fails, propagate
            # without nuking what they were trying to reclaim. Caller
            # decides whether to delete_pool_worker(uid) + retry, or
            # to inspect / repair manually.
            if is_reclaim:
                raise
            # Fresh-provision rollback. If create-Deployment succeeded
            # but `_wait_for_daemon_registered` raised, the Deployment
            # at replicas=1 would otherwise stay alive forever creating
            # and failing pods against the (about-to-be-deleted) PVC.
            # Delete the Deployment FIRST so no new pods can come up,
            # then the PVC. Both calls are best-effort: swallow any
            # secondary failure so the original exception (the one the
            # caller actually cares about) is what propagates.
            try:
                await asyncio.to_thread(
                    self._apps.delete_namespaced_deployment,
                    name=deployment_name,
                    namespace=self._namespace,
                    propagation_policy='Background',
                    grace_period_seconds=0,
                )
            except ApiException as dep_exc:
                if dep_exc.status != 404:
                    log.warning(
                        'provision_pool_worker rollback: deployment '
                        'cleanup returned %s for uid=%s', dep_exc.status, uid)
            try:
                await self._try_delete_pvc(pvc_name, timeout=5.0)
            except Exception as pvc_exc:
                log.warning(
                    'provision_pool_worker rollback: PVC cleanup '
                    'failed for uid=%s: %s', uid, pvc_exc)
            try:
                await asyncio.to_thread(
                    self._policy.delete_namespaced_pod_disruption_budget,
                    name=_pdb_name_for_uid(uid),
                    namespace=self._namespace,
                )
            except ApiException as pdb_exc:
                if pdb_exc.status != 404:
                    log.warning(
                        'provision_pool_worker rollback: PDB cleanup '
                        'returned %s for uid=%s', pdb_exc.status, uid)
            raise

        return info

    async def _require_live_pool_worker_resources(
        self,
        pool_worker_uid: str,
    ) -> None:
        """Require the reclaim substrate to be live and not tearing down."""
        pvc_name = _pvc_name_for_uid(pool_worker_uid)
        deployment_name = _deployment_name_for_uid(pool_worker_uid)
        try:
            pvc = cast(
                V1PersistentVolumeClaim,
                await asyncio.to_thread(
                    self._core.read_namespaced_persistent_volume_claim,
                    name=pvc_name,
                    namespace=self._namespace,
                ),
            )
        except ApiException as exc:
            if exc.status == 404:
                _mark_api_error_restricted(exc)
                raise PoolWorkerNotFoundError(
                    f'PVC {pvc_name} for pool worker {pool_worker_uid} '
                    'does not exist.') from exc
            raise _translate_api_error(exc) from exc

        try:
            deployment = cast(
                V1Deployment,
                await asyncio.to_thread(
                    self._apps.read_namespaced_deployment,
                    name=deployment_name,
                    namespace=self._namespace,
                ),
            )
        except ApiException as exc:
            if exc.status == 404:
                _mark_api_error_restricted(exc)
                raise PoolWorkerNotFoundError(
                    f'Deployment {deployment_name} for pool worker '
                    f'{pool_worker_uid} does not exist.') from exc
            raise _translate_api_error(exc) from exc

        if (pvc.metadata is None
                or pvc.metadata.deletion_timestamp is not None):
            raise PoolWorkerNotReadyError(
                f'PVC {pvc_name} for pool worker {pool_worker_uid} is '
                'terminating.')
        if (deployment.metadata is None
                or deployment.metadata.deletion_timestamp is not None):
            raise PoolWorkerNotReadyError(
                f'Deployment {deployment_name} for pool worker '
                f'{pool_worker_uid} is terminating.')
        if (deployment.spec is not None
                and deployment.spec.replicas is not None
                and deployment.spec.replicas <= 0):
            raise PoolWorkerNotReadyError(
                f'Deployment {deployment_name} for pool worker '
                f'{pool_worker_uid} is scaled down for teardown.')

    async def delete_pool_worker(
        self,
        pool_worker_uid: str,
        grace_period_seconds: int = 30,
        scale_down_timeout: float = 120,
        pvc_delete_timeout: float = 60,
        force_on_timeout: bool = False,
    ) -> None:
        """Tear down a pool worker per design §5.3 in this order:

        1. **Delete the PodDisruptionBudget** first — it is only an
           eviction guard, and once teardown is committed we want the
           node reclaimable. Removing it up front means a later wedge
           (stuck daemon pod / wedged PVC finalizer that aborts a later
           step) can't leave the ``maxUnavailable:0`` PDB orphaned and
           pin the node from consolidation forever; the still-running
           daemon pod keeps the node alive until it is drained in step 3.
        2. **Force-delete activity pods** on this pool worker (grace=0)
           and wait for their shared-PVC mounts to terminate.
        3. **Scale daemon Deployment to 0** + wait for the daemon pod
           to actually terminate.
        4. **Delete the PVC**, with a bounded poll for finalizer
           clearance (`pvc_delete_timeout`, design §5.3 footnote).
        5. **Delete the (now empty) Deployment** so the API-server
           record is gone.

        Idempotent end-to-end: any step that finds the resource
        already gone (404) is treated as success.

        ``force_on_timeout``: default ``False`` raises
        :class:`PoolWorkerCleanupFailedError` if the daemon pod
        doesn't terminate within ``scale_down_timeout`` (refusing to
        delete the PVC under a still-mounted kubelet that would wedge
        the CSI attacher finalizer). Pass ``True`` for the design
        §10 DEAD-node teardown path: the caller has decided the node
        is gone and the volume must be ripped out regardless of the
        kubelet's mount state.
        """
        deployment_name = _deployment_name_for_uid(pool_worker_uid)
        pvc_name = _pvc_name_for_uid(pool_worker_uid)

        # Step 1: delete the PodDisruptionBudget first. It is only the
        # eviction guard; removing it before the steps that can wedge (the
        # daemon scale-down wait, the PVC finalizer poll) guarantees we never
        # leave the maxUnavailable:0 PDB orphaned and pinning the node from
        # consolidation forever if a later step raises. The daemon pod still
        # blocks consolidation until it is drained in step 3. Best-effort:
        # a 404 means it was never created or already cleaned up.
        try:
            await asyncio.to_thread(
                self._policy.delete_namespaced_pod_disruption_budget,
                name=_pdb_name_for_uid(pool_worker_uid),
                namespace=self._namespace,
            )
        except ApiException as exc:
            if exc.status != 404:
                _mark_api_error_restricted(exc)
                raise

        # Step 2: force-delete activity Jobs pinned to this pool worker and
        # fence their pods before deleting the shared PVC.
        await self.delete_activity_pods(
            pool_worker_uid=pool_worker_uid,
            grace_period_seconds=0,
            wait_for_deletion=True,
        )

        # Step 3: scale daemon Deployment to 0 + wait for pod to die.
        try:
            await asyncio.to_thread(
                self._apps.patch_namespaced_deployment_scale,
                name=deployment_name,
                namespace=self._namespace,
                body={'spec': {
                    'replicas': 0
                }},
            )
        except ApiException as exc:
            if exc.status == 404:
                log.info('delete_pool_worker: deployment %s already gone',
                         deployment_name)
            else:
                _mark_api_error_restricted(exc)
                raise
        await self._wait_for_daemon_pod_gone(
            pool_worker_uid,
            timeout=scale_down_timeout,
            force_on_timeout=force_on_timeout,
        )

        # Step 4: delete PVC with bounded wait for finalizer clearance.
        await self._try_delete_pvc(
            pvc_name,
            timeout=pvc_delete_timeout,
        )

        # Step 5: delete the (now-zero-replica) Deployment object.
        try:
            await asyncio.to_thread(
                self._apps.delete_namespaced_deployment,
                name=deployment_name,
                namespace=self._namespace,
                grace_period_seconds=grace_period_seconds,
            )
        except ApiException as exc:
            if exc.status != 404:
                _mark_api_error_restricted(exc)
                raise

    # ------------------------------------------------------------------
    # Activity-pod lifecycle
    # ------------------------------------------------------------------
    # Activity pods are wrapped as `batch/v1 Job`s per design §6.1 + §9
    # row 1. The Job's `backoffLimit > 0` makes the K8s Job controller
    # respawn a fresh pod on crash/OOM (the pod template has
    # `restartPolicy: Never`). Outside callers see ActivityPodInfo
    # records — the Job is the unit of identity, the underlying pod is
    # an implementation detail.

    async def provision_activity_pod(
        self,
        pool_worker_uid: str,
        job_id: str,
        snapshot_id: str,
        activity: str,
        image: str,
        command: Optional[List[str]] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        resources: Optional[ResourceSpec] = None,
        backoff_limit: int = 2,
        require_existing_resources: bool = False,
    ) -> None:
        """Provision a cache-pool activity pod (wrapped as a K8s Job).

        Required pod affinity pins the activity to the pool worker
        daemon's host. The shared ``ReadWriteOnce`` PVC then supplies
        the same cache data, while PV affinity enforces any additional
        storage topology constraint.

        Validates that the target pool worker is Registered before
        creating the Job — fast-fails rather than letting the Job's
        pod sit in Pending while a half-ready PVC waits to bind.

        ``env`` contains application env vars supplied by the caller. The
        manager is otherwise agnostic to the daemon image's env contract, but
        also injects the downward-API pod name used in the Temporal worker
        identity. ``snapshot_id`` / ``job_id`` / ``pool_worker_uid`` /
        ``activity`` stay as separate parameters because they stamp K8s labels
        used for filtering and orphan cleanup.

        ``resources``: optional ``ResourceSpec`` carrying cpu / memory
        / gpu request + limit knobs for the activity container. See
        :class:`ResourceSpec` for QoS semantics + GPU contract. ``None``
        (default) → BestEffort CPU-only activity, no resource entries
        emitted at all (byte-for-byte compat with pre-``ResourceSpec``
        manifests).

        ``require_existing_resources`` rechecks that the target PVC and
        Deployment exist and are not terminating immediately before the Job is
        created or reclaimed.

        **Deterministic reclaim semantics**: the Job name is derived from
        ``job_id`` so a retry after a slow image pull, a Karpenter cold-start,
        or a controller crash lands on the same K8s Job. A 409 from
        ``create_namespaced_job`` therefore means a prior attempt created the
        Job; this method attaches to it instead of failing or creating another
        Job. The manager does not roll it back. Once the handle is persisted,
        the controller owns cleanup if readiness fails or the lease is
        released.

        Mirrors :meth:`provision_ephemeral_activity_pod`; see that method's
        docstring for the overflow-specific scheduling rationale.

        Returns once the k8s Job has been created (or reclaimed).  The
        caller must follow up with :meth:`wait_for_activity_job_pod_running`
        — but must first persist the handle to pool_state so that
        ``release_lease`` can clean up the Job even if the wait raises.

        Raises:
            PoolWorkerNotFoundError: ``pool_worker_uid`` doesn't exist.
            PoolWorkerNotReadyError: pool worker exists but its daemon
                is not yet Registered (mid-provision, mid-rebind, or STALE).
        """
        worker_info = await self._get_pool_worker_info(pool_worker_uid)
        if worker_info is None:
            raise PoolWorkerNotFoundError(
                f'pool worker {pool_worker_uid!r} not found')
        _require_current_extra_tolerations(worker_info)
        if worker_info.status != 'Registered':
            raise PoolWorkerNotReadyError(
                f'pool worker {pool_worker_uid!r} is not Registered '
                f'(current status: {worker_info.status!r})')

        # job_id is hashed for the Job name too: Job names are
        # DNS-1123 subdomains ([a-z0-9-]), which raw Temporal
        # workflow_ids (`:` / `@` / uppercase) routinely violate.
        # The hash form is K8s-safe and the raw value is recoverable
        # from the ``kumo.ai/job-id`` annotation on the Job / pod.
        #
        # Name is deterministic in ``job_id`` so a Temporal retry after a
        # lapsed placement heartbeat or controller crash reclaims the same Job
        # rather than creating an untracked activity attempt.
        job_name = self.activity_job_name(job_id, 'durable')
        pvc_name = _pvc_name_for_uid(pool_worker_uid)
        if require_existing_resources:
            await self._require_live_pool_worker_resources(pool_worker_uid)

        try:
            await asyncio.to_thread(
                self._create_activity_job_sync,
                job_name=job_name,
                pool_worker_uid=pool_worker_uid,
                pvc_name=pvc_name,
                job_id=job_id,
                snapshot_id=snapshot_id,
                activity=activity,
                image=image,
                command=command,
                args=args,
                env=env,
                resources=resources,
                backoff_limit=backoff_limit,
            )
        except PoolWorkerAlreadyExistsError:
            log.info(
                'provision_activity_pod: Job %s already exists for '
                'job_id=%s pool_worker_uid=%s; reclaiming', job_name, job_id,
                pool_worker_uid)

    async def provision_ephemeral_activity_pod(
        self,
        job_id: str,
        snapshot_id: str,
        activity: str,
        image: str,
        ephemeral_storage_bytes: int,
        command: Optional[List[str]] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        resources: Optional[ResourceSpec] = None,
        backoff_limit: int = 2,
    ) -> None:
        """Provision an ephemeral-pool activity pod (wrapped as a Job).

        emptyDir-backed; no PVC, no pool-worker pinning. Karpenter
        provisions an eph-pool node sized to fit the requested
        ``ephemeral_storage_bytes``.

        ``env``: see :meth:`provision_activity_pod` — the caller owns the
        application env contract and the manager adds only the pod identity it
        requires for exact Temporal poller matching.

        ``resources``: see :meth:`provision_activity_pod` — same
        contract. Eph-pool nodes can host multiple activity pods (no
        PVC-RWO pinning), so scheduler arbitration via these resource
        entries matters more here than on the cache pool.

        Returns once the k8s Job has been created (or reclaimed).  The
        caller must follow up with :meth:`wait_for_activity_job_pod_running`
        — but must first persist the handle to pool_state so that
        ``release_lease`` can clean up the Job even if the wait raises.

        **Deterministic reclaim semantics**: the Job name is derived from
        ``job_id`` so a retry after an RPC timeout during a Karpenter
        cold-start, or after a controller crash, lands on the same K8s Job. A
        409 from ``create_namespaced_job`` therefore means a prior attempt
        created the Job; this method attaches to it instead of starting
        another cold node. The manager does not roll it back. Once the handle
        is persisted, the controller owns cleanup if readiness fails or the
        lease is released.

        Mirrors the cache-pool activity Job reclaim contract; see
        :meth:`provision_activity_pod` for the shared behavior.
        """
        # See cache-pool variant — Job names are DNS-1123 subdomains,
        # so embed the hashed form and keep the raw value on the
        # ``kumo.ai/job-id`` annotation. Name is deterministic in
        # ``job_id`` so retries converge on the same K8s Job.
        job_name = self.activity_job_name(job_id, 'overflow')

        try:
            await asyncio.to_thread(
                self._create_eph_activity_job_sync,
                job_name=job_name,
                job_id=job_id,
                snapshot_id=snapshot_id,
                activity=activity,
                image=image,
                command=command,
                args=args,
                env=env,
                resources=resources,
                ephemeral_storage_bytes=ephemeral_storage_bytes,
                backoff_limit=backoff_limit,
            )
        except PoolWorkerAlreadyExistsError:
            log.info(
                'provision_ephemeral_activity_pod: Job %s already '
                'exists for job_id=%s; reclaiming', job_name, job_id)

    async def delete_activity_pods(
        self,
        pool_worker_uid: Optional[str] = None,
        pool_type: Optional[str] = None,
        job_id: Optional[str] = None,
        grace_period_seconds: Optional[int] = None,
        wait_for_deletion: bool = False,
        deletion_timeout: float = ACTIVITY_JOB_DELETE_TIMEOUT_SEC,
    ) -> None:
        """Delete activity Jobs and any matching orphan pods.

        Filters compose with AND. A successful empty collection is treated as
        success. ``wait_for_deletion=True`` waits until matching Jobs and pods
        are gone before the Job name can be reused.

        - ``pool_worker_uid=X``: delete activity Jobs pinned to that
          cache-pool pool worker (used during rebind / pool-worker
          teardown).
        - ``pool_type='ephemeral'``: delete eph-pool activity Jobs.
        - ``job_id=Y``: delete activity Jobs belonging to one workflow
          job (used for orphan cleanup on ``release_lease`` —
          design §6.1).
        - all None: delete every activity Job in the tenant.

        ``grace_period_seconds`` overrides the K8s default (30s) on the Job
        collection deletion and each explicit matching-pod deletion.
        Pass ``0`` to force-terminate immediately (used by
        ``delete_pool_worker`` step 2 per design §5.3, where the node
        has already been tainted so the pods are abandoned anyway).
        ``None`` keeps K8s's default behavior.
        """
        selector = self._activity_label_selector(pool_worker_uid, pool_type,
                                                 job_id)

        kwargs: Dict[str, Any] = {
            'namespace': self._namespace,
            'label_selector': selector,
            'propagation_policy': ('Foreground'
                                   if wait_for_deletion else 'Background'),
        }
        if grace_period_seconds is not None:
            kwargs['grace_period_seconds'] = grace_period_seconds

        try:
            await asyncio.to_thread(
                self._batch.delete_collection_namespaced_job,
                **kwargs,
            )
        except ApiException as exc:
            raise _translate_api_error(exc) from exc
        try:
            pods = await asyncio.to_thread(
                self._core.list_namespaced_pod,
                namespace=self._namespace,
                label_selector=selector,
            )
            for pod in pods.items:
                pod_name = pod.metadata.name if pod.metadata else None
                if pod_name is None:
                    continue
                pod_delete_kwargs: Dict[str, Any] = {
                    'name': pod_name,
                    'namespace': self._namespace,
                }
                if grace_period_seconds is not None:
                    pod_delete_kwargs[
                        'grace_period_seconds'] = grace_period_seconds
                try:
                    await asyncio.to_thread(
                        self._core.delete_namespaced_pod,
                        **pod_delete_kwargs,
                    )
                except ApiException as exc:
                    if exc.status != 404:
                        raise
        except ApiException as exc:
            raise _translate_api_error(exc) from exc

        if not wait_for_deletion:
            return

        # Foreground keeps the owner Job until its blocking dependents are gone.
        # Explicit pod deletion also covers orphans left by an older
        # background-deletion attempt before deterministic-name reuse.
        deadline = time.monotonic() + deletion_timeout
        while time.monotonic() < deadline:
            try:
                jobs = await asyncio.to_thread(
                    self._batch.list_namespaced_job,
                    namespace=self._namespace,
                    label_selector=selector,
                )
                pods = await asyncio.to_thread(
                    self._core.list_namespaced_pod,
                    namespace=self._namespace,
                    label_selector=selector,
                )
            except ApiException as exc:
                raise _translate_api_error(exc) from exc
            if not jobs.items and not pods.items:
                return
            await asyncio.sleep(_POLL_INTERVAL_SECS)
        raise PoolWorkerCleanupFailedError(
            'activity Jobs or pods were not deleted within '
            f'{deletion_timeout:.0f}s for selector={selector!r}')

    # ------------------------------------------------------------------
    # Status / listing
    # ------------------------------------------------------------------

    async def get_status(self, pool_worker_uid: str) -> PoolWorkerInfo:
        """Return the pool worker's current orchestration-level status.

        Raises:
            PoolWorkerNotFoundError: no daemon pod with this uid.
        """
        info = await self._get_pool_worker_info(pool_worker_uid)
        if info is None:
            raise PoolWorkerNotFoundError(
                f'pool worker {pool_worker_uid!r} not found')
        return info

    async def list_pool_workers(self) -> List[PoolWorkerInfo]:
        """Return all pool workers in this tenant (cache-pool only —
        ephemeral pool has no pool-worker concept).
        """
        pods = await asyncio.to_thread(
            self._core.list_namespaced_pod,
            namespace=self._namespace,
            label_selector=self._daemon_label_selector(),
        )
        results: List[PoolWorkerInfo] = []
        extras_hash = self._cache_extra_tolerations_hash()
        for pod in pods.items:
            labels = (pod.metadata.labels or {}) if pod.metadata else {}
            uid = labels.get(LABEL_POOL_WORKER_UID)
            if uid is None:
                continue
            results.append(_pod_to_pool_worker_info(pod, uid, extras_hash))
        return results

    async def registered_worker_matches_current_cache_tier(
            self, worker: PoolWorkerInfo, spec: PvcSpec) -> bool:
        """Whether a Registered worker still matches this cache tier.

        Persisted tier metadata identifies a candidate, but an old or rebound
        daemon may use a stale scheduling or storage envelope. A missing live
        object is ordinary reconciliation drift and returns ``False``; other
        API failures remain unexpected and propagate.
        """
        if (worker.status != 'Registered' or not worker.daemon_pod_name
                or not worker.pvc_name):
            return False
        try:
            pod = cast(
                V1Pod,
                await asyncio.to_thread(
                    self._core.read_namespaced_pod,
                    name=worker.daemon_pod_name,
                    namespace=self._namespace,
                ),
            )
            pvc = cast(
                V1PersistentVolumeClaim,
                await asyncio.to_thread(
                    self._core.read_namespaced_persistent_volume_claim,
                    name=worker.pvc_name,
                    namespace=self._namespace,
                ),
            )
        except ApiException as exc:
            if exc.status == 404:
                return False
            raise _translate_api_error(exc) from exc

        scheduling = self.cache_canary_scheduling(
            spec.size_gib, spec.gpu_count)
        pod_spec = pod.spec
        if (_derive_pod_status(pod) != 'Registered' or pod_spec is None
                or pod_spec.node_selector != scheduling.node_selector):
            return False
        expected_tolerations = {
            _toleration_key(t)
            for t in scheduling.tolerations
        }
        actual_tolerations = {
            _toleration_key(t)
            for t in (pod_spec.tolerations or [])
        }
        # Admission may add tolerations, including another effect for a key
        # configured in extras. Require our entries without rejecting those
        # additions; retain the existing exact check for our tenant taint key.
        if not expected_tolerations.issubset(actual_tolerations):
            return False
        if {t
                for t in actual_tolerations
                if t[0] == self._pool_taint_key} != {
                    t
                    for t in expected_tolerations
                    if t[0] == self._pool_taint_key
                }:
            return False
        # A subset check alone misses removed extras. The pod's fingerprint
        # records what Kumo configured, not what admission later injected.
        annotations = (pod.metadata.annotations or {}) if pod.metadata else {}
        if annotations.get(_ANNOTATION_POOL_EXTRA_TOLERATIONS_HASH) != (
                self._cache_extra_tolerations_hash()):
            return False
        if not any(
                volume.persistent_volume_claim is not None
                and volume.persistent_volume_claim.claim_name
                == worker.pvc_name for volume in (pod_spec.volumes or [])):
            return False

        pvc_metadata = pvc.metadata
        pvc_spec = pvc.spec
        pvc_status = pvc.status
        if (pvc_metadata is None or pvc_metadata.deletion_timestamp is not None
                or pvc_spec is None or pvc_status is None):
            return False
        storage_request = ((pvc_spec.resources.requests or {}).get('storage')
                           if pvc_spec.resources is not None else None)
        return (pvc_spec.storage_class_name == self._storage_class
                and set(pvc_spec.access_modes or []) == {'ReadWriteOnce'}
                and str(storage_request) == self.cache_pvc_size_str(spec)
                and pvc_status.phase == 'Bound'
                and bool(pvc_spec.volume_name))

    async def list_activity_pods(
        self,
        pool_worker_uid: Optional[str] = None,
        pool_type: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> List[ActivityPodInfo]:
        """List activity pods, filtered by uid / pool type / job id.

        Lists the underlying pods (children of activity Jobs) so
        callers see accurate Running/Succeeded/Failed status. Filters
        compose with AND.
        """
        selector = self._activity_label_selector(pool_worker_uid, pool_type,
                                                 job_id)

        pods = await asyncio.to_thread(
            self._core.list_namespaced_pod,
            namespace=self._namespace,
            label_selector=selector,
        )
        return [_pod_to_activity_pod_info(p) for p in pods.items]

    # ------------------------------------------------------------------
    # Internals: sync helpers (run via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _create_pvc_sync(
        self,
        pvc_name: str,
        uid: str,
        storage_size: str,
    ) -> None:
        pvc = V1PersistentVolumeClaim(
            metadata=V1ObjectMeta(
                name=pvc_name,
                namespace=self._namespace,
                labels=self._daemon_labels(uid),
            ),
            spec=V1PersistentVolumeClaimSpec(
                # ReadWriteOnce: PVC is mounted on a single NODE but
                # multiple PODS on that node can share the mount (daemon
                # + concurrent activity pods, per design §6.2).
                # ReadWriteOncePod would be too strict — only one pod
                # at a time, blocking daemon ↔ activity-pod coexistence.
                access_modes=['ReadWriteOnce'],
                storage_class_name=self._storage_class,
                resources=V1ResourceRequirements(
                    requests={'storage': storage_size}, ),
            ),
        )
        try:
            self._core.create_namespaced_persistent_volume_claim(
                namespace=self._namespace, body=pvc)
        except ApiException as exc:
            raise _translate_api_error(exc) from exc

    def _pvc_is_bound_sync(self, pvc_name: str) -> bool:
        pvc = cast(
            V1PersistentVolumeClaim,
            self._core.read_namespaced_persistent_volume_claim(
                name=pvc_name,
                namespace=self._namespace,
            ),
        )
        return (pvc.status.phase if pvc.status else None) == 'Bound'

    # --- substrate wiring -------------------------------------------------
    # Pool pods are created at runtime by the controller, so (unlike the
    # chart-managed static workers) the chart can't mount their substrate.
    # The manager stamps the same wiring here, parameterized by config so it
    # stays substrate-agnostic: envFrom the shared env ConfigMap (DISABLE_AUTH
    # + flags), the Temporal mTLS cert files (self-hosted), optional Postgres
    # files, and static GCP ADC. Each piece is opt-in; unset is a no-op.

    def _substrate_env_from(self) -> List[V1EnvFromSource]:
        if not self._pod_env_configmap:
            return []
        return [
            V1EnvFromSource(config_map_ref=V1ConfigMapEnvSource(
                name=self._pod_env_configmap))
        ]

    def _tenant_secret_env(self) -> List[V1EnvVar]:
        """Per-tenant secret-key env for runtime-created pool pods.

        Mirrors the chart's delivery to the chart-managed workloads
        (FERNET_SECRET_NAME redirects the app's Fernet lookup to an env
        the secret-provider chain resolves).  Without it, any activity
        that touches credential encryption on a non-preprod stage dies
        with SecretNotFoundException, because the tenant key reaches spawned
        pods only through this env.
        """
        if self._tenant_secret_key_ref is None:
            return []
        ref = V1EnvVarSource(secret_key_ref=V1SecretKeySelector(
            name=self._tenant_secret_key_ref.secret_name,
            key=self._tenant_secret_key_ref.key))
        return [
            V1EnvVar(name='FERNET_SECRET_NAME',
                     value='credentials_encryption_key'),
            V1EnvVar(name='credentials_encryption_key', value_from=ref),
            V1EnvVar(name='API_KEY_PEPPER', value_from=ref),
            V1EnvVar(name='PASSWORD_RESET_TOKEN_PEPPER', value_from=ref),
        ]

    def _substrate_volumes(self) -> List[V1Volume]:
        vols = [
            V1Volume(name='runtime-tmp', empty_dir=V1EmptyDirVolumeSource()),
            V1Volume(name='runtime-cache', empty_dir=V1EmptyDirVolumeSource()),
        ]
        if self._temporal_cert_secret and self._temporal_cert_dir:
            vols.append(
                V1Volume(
                    name='temporal-mtls', secret=V1SecretVolumeSource(
                        secret_name=self._temporal_cert_secret,
                        default_mode=420)))
        # Password auth points metadata_storage at dsn_secret_name, which the
        # app resolves as a FILE under secrets.mount_path. Runtime pods open
        # StateDB just like the chart-managed ones, so without this mount every
        # spawned pod dies with SecretNotFoundException. IAM auth needs nothing
        # here: its connection details are inline in the deploy config.
        if self._statedb_secret and self._secrets_mount_path:
            vols.append(
                V1Volume(
                    name='kumo-secrets', secret=V1SecretVolumeSource(
                        secret_name=self._statedb_secret,
                        default_mode=420)))
        # A CA the image does not already trust (Cloud SQL's per-instance CA,
        # self-hosted, internal PKI). Mounted in a subdirectory so it does not
        # shadow the connection Secret's keys.
        if self._statedb_ca_secret and self._secrets_mount_path:
            vols.append(
                V1Volume(
                    name='kumo-statedb-ca',
                    secret=V1SecretVolumeSource(
                        secret_name=self._statedb_ca_secret,
                        # Remapped to a fixed filename, matching the chart, so
                        # tls.sslrootcert is one constant everywhere.
                        items=[
                            V1KeyToPath(key=self._statedb_ca_secret_key,
                                        path='ca.pem')
                        ],
                        default_mode=420)))
        if (self._gcp_application_credentials_secret
                and self._gcp_application_credentials_secret_key
                and self._gcp_application_credentials_path):
            vols.append(
                V1Volume(
                    name=_GCP_APPLICATION_CREDENTIALS_VOLUME,
                    secret=V1SecretVolumeSource(
                        secret_name=(
                            self._gcp_application_credentials_secret),
                        items=[
                            V1KeyToPath(
                                key=self._gcp_application_credentials_secret_key,
                                path=os.path.basename(
                                    self._gcp_application_credentials_path))
                        ],
                        default_mode=292)))
        return vols

    def _substrate_volume_mounts(self) -> List[V1VolumeMount]:
        mounts = [
            V1VolumeMount(name='runtime-tmp', mount_path='/tmp'),
            V1VolumeMount(name='runtime-cache',
                          mount_path='/home/kumouser/.cache'),
        ]
        if self._temporal_cert_secret and self._temporal_cert_dir:
            mounts.append(
                V1VolumeMount(name='temporal-mtls',
                              mount_path=self._temporal_cert_dir,
                              read_only=True))
        if self._statedb_secret and self._secrets_mount_path:
            mounts.append(
                V1VolumeMount(name='kumo-secrets',
                              mount_path=self._secrets_mount_path,
                              read_only=True))
        if self._statedb_ca_secret and self._secrets_mount_path:
            mounts.append(
                V1VolumeMount(name='kumo-statedb-ca',
                              mount_path=f'{self._secrets_mount_path}/ca',
                              read_only=True))
        if (self._gcp_application_credentials_secret
                and self._gcp_application_credentials_secret_key
                and self._gcp_application_credentials_path):
            mounts.append(
                V1VolumeMount(
                    name=_GCP_APPLICATION_CREDENTIALS_VOLUME,
                    mount_path=os.path.dirname(
                        self._gcp_application_credentials_path),
                    read_only=True))
        return mounts

    def _apply_substrate_wiring(self, workload: 'V1Deployment | V1Job') -> None:
        """Append the substrate envFrom + volumes/mounts to the pool pod's
        single container, in place on ``workload``'s pod template."""
        spec = workload.spec
        if (spec is None or spec.template is None
                or spec.template.spec is None):
            return
        pod_spec = spec.template.spec
        containers = pod_spec.containers
        if not containers:
            return
        env_from = self._substrate_env_from()
        volumes = self._substrate_volumes()
        mounts = self._substrate_volume_mounts()
        container = containers[0]
        if env_from:
            container.env_from = (container.env_from or []) + env_from
        if volumes:
            pod_spec.volumes = (pod_spec.volumes or []) + volumes
            container.volume_mounts = (container.volume_mounts or []) + mounts

    def _create_daemon_deployment_sync(
        self,
        deployment_name: str,
        pvc_name: str,
        uid: str,
        env: Optional[Dict[str, str]],
        ssd_size_gib: Optional[int] = None,
        gpu_count: Optional[int] = None,
    ) -> None:
        labels = self._daemon_labels(uid)
        extras_hash = self._cache_extra_tolerations_hash()
        dep = V1Deployment(
            metadata=V1ObjectMeta(
                name=deployment_name,
                namespace=self._namespace,
                labels=labels,
            ),
            spec=V1DeploymentSpec(
                replicas=1,
                # Recreate strategy: design §7.2 + §9 row 2 — the old
                # daemon pod is fully terminated before a replacement
                # spawns, so we never have two daemons heartbeating on
                # the same node during a rollout.
                strategy=V1DeploymentStrategy(type='Recreate'),
                selector=V1LabelSelector(match_labels={
                    LABEL_APP: LABEL_APP_VALUE_DAEMON,
                    LABEL_POOL_WORKER_UID: uid,
                }),
                template=V1PodTemplateSpec(
                    metadata=V1ObjectMeta(
                        labels=self._runtime_pod_labels(labels),
                        annotations={
                            _ANNOTATION_POOL_EXTRA_TOLERATIONS_HASH:
                            extras_hash,
                        } if extras_hash else None,
                        # NB: voluntary-disruption protection for the
                        # node hosting this daemon is provided by a
                        # PodDisruptionBudget (maxUnavailable:0) created
                        # alongside the Deployment — see
                        # `_create_daemon_pdb_sync`. The PDB is cloud-
                        # portable and needs no node-level RBAC.
                    ),
                    spec=V1PodSpec(
                        restart_policy='Always',
                        # worker_main configures Temporal's daemon shutdown
                        # grace to zero so TERM immediately cancels the active
                        # activity. Keep enough pod grace for the native
                        # subprocess supervisor's 10s TERM + 5s KILL/reap
                        # sequence before kubelet sends SIGKILL.
                        termination_grace_period_seconds=30,
                        # Azure Disk and other CSI PVCs are root-owned at
                        # creation; stock Kumo images write as gid 999.
                        security_context=_runtime_pod_security_context(
                            self._runtime_fs_group),
                        service_account_name=self._service_account,
                        node_selector=self._cache_node_selector(
                            ssd_size_gib, gpu_count),
                        tolerations=self._pool_tolerations(
                            self._cache_pool_taint_value),
                        # Structural guardrail: at most one daemon pod per
                        # node (design §7.2). The diskgraph-ssd-size
                        # nodeSelector already picks a big-enough pool, so no
                        # node affinity is needed.
                        affinity=V1Affinity(
                            pod_anti_affinity=_daemon_anti_affinity(), ),
                        containers=[
                            V1Container(
                                name='daemon',
                                image=self._daemon_image,
                                security_context=(
                                    _runtime_container_security_context()),
                                command=self._daemon_command,
                                args=self._daemon_args,
                                env=(_env_dict_to_v1envvar_list(env) +
                                     self._tenant_secret_env()),
                                resources=_compose_resource_requirements(
                                    self._daemon_resources),
                                volume_mounts=[
                                    V1VolumeMount(
                                        mount_path=NVME_MOUNT_PATH,
                                        name='ssd',
                                    ),
                                    # Daemon ingest temp/spill on
                                    # kubelet ephemeral storage.
                                    # Activity pods do not mount it.
                                    V1VolumeMount(
                                        mount_path=(
                                            self._ebs_scratch_mount_path),
                                        name='ebs-scratch',
                                    ),
                                ],
                            ),
                        ],
                        volumes=[
                            V1Volume(
                                name='ssd',
                                persistent_volume_claim=(
                                    V1PersistentVolumeClaimVolumeSource(
                                        claim_name=pvc_name, )),
                            ),
                            # emptyDir uses the cache node's kubelet
                            # ephemeral-storage path. EKS provisions
                            # dedicated EBS for that path; other
                            # platforms must size their node-pool disk.
                            V1Volume(
                                name='ebs-scratch',
                                empty_dir=V1EmptyDirVolumeSource(),
                            ),
                        ],
                    ),
                ),
            ),
        )
        self._apply_substrate_wiring(dep)
        try:
            self._apps.create_namespaced_deployment(namespace=self._namespace,
                                                    body=dep)
        except ApiException as exc:
            raise _translate_api_error(exc) from exc

    def _create_daemon_pdb_sync(self, uid: str) -> None:
        # A PodDisruptionBudget with maxUnavailable:0 is the cloud-
        # portable replacement for the node `do-not-disrupt`
        # annotation: it blocks every *voluntary* eviction of the
        # daemon pod (Karpenter consolidation/drift, cluster-autoscaler
        # scale-down, `kubectl drain`) for as long as the daemon pod
        # exists — keeping the node alive while the pool worker is in
        # use. Unlike the old approach it needs no cluster-scoped node
        # RBAC; it lives entirely in the tenant namespace.
        pdb = V1PodDisruptionBudget(
            metadata=V1ObjectMeta(
                name=_pdb_name_for_uid(uid),
                namespace=self._namespace,
                labels=self._daemon_labels(uid),
            ),
            spec=V1PodDisruptionBudgetSpec(
                max_unavailable=0,
                selector=V1LabelSelector(match_labels={
                    LABEL_APP: LABEL_APP_VALUE_DAEMON,
                    LABEL_POOL_WORKER_UID: uid,
                }),
            ),
        )
        try:
            self._policy.create_namespaced_pod_disruption_budget(
                namespace=self._namespace, body=pdb)
        except ApiException as exc:
            if exc.status == 409:
                # Reclaim-idempotent: a prior provision attempt (or a
                # rebind on the same uid) already created it.
                log.info(
                    '_create_daemon_pdb_sync: PDB %s already exists for '
                    'uid=%s; treating as already-created',
                    _pdb_name_for_uid(uid), uid)
                return
            raise _translate_api_error(exc) from exc

    def _create_activity_job_sync(
        self,
        job_name: str,
        pool_worker_uid: str,
        pvc_name: str,
        job_id: str,
        snapshot_id: str,
        activity: str,
        image: str,
        command: Optional[List[str]],
        args: Optional[List[str]],
        env: Optional[Dict[str, str]],
        resources: Optional[ResourceSpec],
        backoff_limit: int,
    ) -> None:
        extra_mounts, extra_volumes = _dshm_volume(resources)
        labels = {
            LABEL_APP: LABEL_APP_VALUE_ACTIVITY,
            LABEL_TENANT: self._tenant,
            DISKGRAPH_POOL_TYPE_LABEL: POOL_TYPE_CACHE,
            LABEL_POOL_WORKER_UID: pool_worker_uid,
            LABEL_JOB_ID_HASH: _hash_job_id(job_id),
            LABEL_SNAPSHOT_ID_HASH: _hash_snapshot_id(snapshot_id),
            LABEL_ACTIVITY: activity,
        }
        # Annotations carry data K8s labels can't hold (raw Temporal
        # workflow_id with `:` / `@` / >63-char composite form, plus
        # snapshot ids that may carry uri-like segments). Stamped on
        # both the Job and the pod template so the annotations
        # propagate to the spawned pod and show up under
        # ``kubectl describe pod``.
        annotations = {
            ANNOTATION_JOB_ID: job_id,
            ANNOTATION_SNAPSHOT_ID: snapshot_id,
        }
        job = V1Job(
            metadata=V1ObjectMeta(
                name=job_name,
                namespace=self._namespace,
                labels=labels,
                annotations=annotations,
            ),
            spec=V1JobSpec(
                backoff_limit=backoff_limit,
                active_deadline_seconds=_ACTIVITY_JOB_DEADLINE_SECS,
                template=V1PodTemplateSpec(
                    metadata=V1ObjectMeta(
                        labels=self._runtime_pod_labels(labels),
                        annotations=annotations,
                    ),
                    spec=V1PodSpec(
                        # Crashed container is NOT restarted in place
                        # — Job spawns a fresh pod (design §6.1).
                        restart_policy='Never',
                        termination_grace_period_seconds=30,
                        security_context=_runtime_pod_security_context(
                            self._runtime_fs_group),
                        service_account_name=self._service_account,
                        # The daemon affinity is required for zonal RWO volumes
                        # such as Azure Disk: PV affinity alone selects a zone,
                        # not the daemon's host, and can produce Multi-Attach.
                        affinity=_daemon_pod_affinity(pool_worker_uid),
                        # ssd_size_gib=None: the affinity selects the daemon,
                        # so no size selector is needed.
                        node_selector=self._cache_node_selector(None),
                        tolerations=self._pool_tolerations(
                            self._cache_pool_taint_value),
                        containers=[
                            V1Container(
                                name='activity',
                                image=image,
                                security_context=(
                                    _runtime_container_security_context()),
                                command=command,
                                args=args,
                                env=(_env_dict_to_v1envvar_list(env) +
                                     [_activity_worker_pod_name_env()] +
                                     self._tenant_secret_env()),
                                resources=_compose_resource_requirements(
                                    resources),
                                volume_mounts=[
                                    V1VolumeMount(
                                        mount_path=NVME_MOUNT_PATH,
                                        name='ssd',
                                    ),
                                    *extra_mounts,
                                ],
                            ),
                        ],
                        volumes=[
                            V1Volume(
                                name='ssd',
                                # The daemon and its activities share this RWO
                                # volume on one host (design §6.2).
                                persistent_volume_claim=(
                                    V1PersistentVolumeClaimVolumeSource(
                                        claim_name=pvc_name, )),
                            ),
                            *extra_volumes,
                        ],
                    ),
                ),
            ),
        )
        self._apply_substrate_wiring(job)
        try:
            self._batch.create_namespaced_job(namespace=self._namespace,
                                              body=job)
        except ApiException as exc:
            raise _translate_api_error(exc) from exc

    def _create_eph_activity_job_sync(
        self,
        job_name: str,
        job_id: str,
        snapshot_id: str,
        activity: str,
        image: str,
        command: Optional[List[str]],
        args: Optional[List[str]],
        env: Optional[Dict[str, str]],
        resources: Optional[ResourceSpec],
        ephemeral_storage_bytes: int,
        backoff_limit: int,
    ) -> None:
        labels = {
            LABEL_APP: LABEL_APP_VALUE_ACTIVITY_EPH,
            LABEL_TENANT: self._tenant,
            DISKGRAPH_POOL_TYPE_LABEL: POOL_TYPE_EPHEMERAL,
            LABEL_JOB_ID_HASH: _hash_job_id(job_id),
            LABEL_SNAPSHOT_ID_HASH: _hash_snapshot_id(snapshot_id),
            LABEL_ACTIVITY: activity,
        }
        # See cache-pool variant for rationale — raw job_id and
        # snapshot_id preserved on annotations, hashed onto labels.
        annotations = {
            ANNOTATION_JOB_ID: job_id,
            ANNOTATION_SNAPSHOT_ID: snapshot_id,
        }
        eph_storage = f'{ephemeral_storage_bytes}'
        extra_mounts, extra_volumes = _dshm_volume(resources)
        job = V1Job(
            metadata=V1ObjectMeta(
                name=job_name,
                namespace=self._namespace,
                labels=labels,
                annotations=annotations,
            ),
            spec=V1JobSpec(
                backoff_limit=backoff_limit,
                active_deadline_seconds=_ACTIVITY_JOB_DEADLINE_SECS,
                template=V1PodTemplateSpec(
                    metadata=V1ObjectMeta(
                        labels=self._runtime_pod_labels(labels),
                        annotations=annotations,
                    ),
                    spec=V1PodSpec(
                        restart_policy='Never',
                        termination_grace_period_seconds=30,
                        security_context=_runtime_pod_security_context(),
                        service_account_name=self._service_account,
                        node_selector=self._eph_node_selector(),
                        tolerations=self._pool_tolerations(
                            self._eph_pool_taint_value),
                        containers=[
                            V1Container(
                                name='activity',
                                image=image,
                                security_context=(
                                    _runtime_container_security_context()),
                                command=command,
                                args=args,
                                env=(_env_dict_to_v1envvar_list(env) +
                                     [_activity_worker_pod_name_env()] +
                                     self._tenant_secret_env()),
                                # request = ceil(snapshot×1.2) is the scheduling
                                # floor only; deliberately NO limit so the
                                # dedicated/disposable eph node's full instance-store
                                # is usable rather than the pod being kubelet-evicted
                                # at the request while the node sits empty (backlog
                                # #18). Node-level eviction is the out-of-disk backstop.
                                resources=_compose_resource_requirements(
                                    resources,
                                    ephemeral_storage_request=eph_storage,
                                ),
                                volume_mounts=[
                                    V1VolumeMount(
                                        mount_path=NVME_MOUNT_PATH,
                                        name='ssd',
                                    ),
                                    *extra_mounts,
                                ],
                            ),
                        ],
                        volumes=[
                            V1Volume(
                                name='ssd',
                                # No size_limit — see the ephemeral-storage
                                # request rationale above (backlog #18).
                                empty_dir=V1EmptyDirVolumeSource(),
                            ),
                            *extra_volumes,
                        ],
                    ),
                ),
            ),
        )
        self._apply_substrate_wiring(job)
        try:
            self._batch.create_namespaced_job(namespace=self._namespace,
                                              body=job)
        except ApiException as exc:
            raise _translate_api_error(exc) from exc

    # ------------------------------------------------------------------
    # Internals: async helpers
    # ------------------------------------------------------------------

    async def _wait_for_daemon_registered(
        self,
        uid: str,
        timeout: float,
    ) -> PoolWorkerInfo:
        """Poll the daemon pod until Registered / Failed / timeout.

        **Scope of this method**: it returns based purely on the
        K8s-level pod status (phase=Running + all container statuses
        Ready). This class does NOT track Temporal heartbeat state —
        the **pool controller** owns the heartbeat layer and is
        responsible for upgrading a pool worker's status from
        "container Ready" to "actually serving" (and handling the
        STALE / DEAD transitions per design §10).

        The container's readiness probe SHOULD be wired to the
        Temporal task-queue binding so that container.ready becomes
        a tight proxy for "queue bound" — but that's a daemon-image
        concern, not this manager's. Until then, callers must treat
        a Registered return value as "k8s thinks it's healthy",
        not "Temporal-bound".
        """
        deadline = time.monotonic() + timeout
        last: Optional[PoolWorkerInfo] = None
        while time.monotonic() < deadline:
            info = await self._get_pool_worker_info(uid)
            last = info
            if info is not None:
                _require_current_extra_tolerations(info)
                if info.status == 'Registered':
                    return info
                if info.status == 'Failed':
                    raise ProvisionTimeoutError(
                        f'pool worker {uid!r} daemon pod entered Failed '
                        f'state during provisioning')
            await asyncio.sleep(_POLL_INTERVAL_SECS)

        last_status = last.status if last else 'unknown'
        # Disambiguate PVC-bind stuck (CSI capacity miss, retryable on
        # different instance type per §8.3) from a generic timeout
        # (image pull, daemon slow, etc.) so the pool controller can
        # route to the right recovery branch. We only raise the
        # specialised error when the PVC's phase is the well-known
        # "stuck on scheduling" state (``Pending``); any other phase
        # (Bound, Lost, unknown) falls through to the generic timeout.
        try:
            # Cast: asyncio.to_thread returns Awaitable[Unknown] per
            # pyright; the kubernetes client's read_namespaced_pvc
            # actually returns V1PersistentVolumeClaim. Narrow it so
            # the .status / .metadata accesses below type-check.
            pvc = cast(
                V1PersistentVolumeClaim,
                await asyncio.to_thread(
                    self._core.read_namespaced_persistent_volume_claim,
                    name=_pvc_name_for_uid(uid),
                    namespace=self._namespace,
                ),
            )
            pvc_phase = (pvc.status.phase if pvc.status else None)
            if pvc_phase == 'Pending':
                pvc_disp = (pvc.metadata.name if pvc.metadata
                            and pvc.metadata.name else _pvc_name_for_uid(uid))
                raise PVCBindTimeoutError(
                    f'PVC {pvc_disp!r} stuck in Pending '
                    f'after {timeout:.0f}s — likely a CSI capacity '
                    f'miss; retry on a different instance type '
                    f'(design §8.3)')
        except ApiException:
            # PVC unreadable — fall through to generic timeout.
            pass
        raise ProvisionTimeoutError(
            f'pool worker {uid!r} did not reach Registered within '
            f'{timeout:.0f}s (last status: {last_status})')

    async def wait_for_activity_job_pod_running(
        self,
        job_name: str,
        timeout: float,
    ) -> ActivityPodInfo:
        """Wait for the newest live attempt of a serial activity Job.

        Jobs use the default ``parallelism=1`` and retain earlier terminal
        pods. Prefer Pending, Running, or Registered pods over terminal pods,
        then select the newest pod within that class so an earlier failed
        attempt cannot hide its replacement.

        ``Running`` or ``Registered`` is ready for the exact-poller check.
        ``Succeeded`` is terminal because the activity exited before its
        Temporal poller became ready. ``Failed`` is checked against Job status
        to distinguish an exhausted Job from an attempt Kubernetes may replace.

        Deterministic container errors such as CrashLoopBackOff and
        CreateContainerConfigError fail immediately. Unschedulable and
        ErrImagePull remain under Kubernetes retry control. ImagePullBackOff
        gets a per-pod grace window because each replacement attempt starts a
        new image pull.
        """
        deadline = time.monotonic() + timeout
        image_pull_backoff_at: Dict[str, float] = {}
        last_phase = 'unknown'
        while time.monotonic() < deadline:
            try:
                pods = await asyncio.to_thread(
                    self._core.list_namespaced_pod,
                    namespace=self._namespace,
                    label_selector=f'job-name={job_name}',
                )
            except ApiException as exc:
                raise _translate_api_error(exc) from exc

            non_terminating_pods = [
                pod for pod in pods.items
                if _derive_pod_status(pod) != 'Terminating'
            ]
            if not non_terminating_pods:
                await self._require_activity_job_status(job_name)
                if pods.items:
                    last_phase = 'Terminating'
            else:
                pod = max(non_terminating_pods,
                          key=_activity_attempt_priority_key)
                state = _derive_pod_status(pod)
                last_phase = state
                if state in {'Running', 'Registered'}:
                    return _pod_to_activity_pod_info(pod)

                if (state == 'Pending' and pod.status
                        and pod.status.container_statuses):
                    for cs in pod.status.container_statuses:
                        if not (cs.state and cs.state.waiting
                                and cs.state.waiting.reason):
                            continue
                        reason = cs.state.waiting.reason
                        if reason in _IMMEDIATE_TERMINAL_REASONS:
                            raise ProvisionTerminalError(
                                f'activity Job {job_name!r} pod is stuck: '
                                f'{reason} (container {cs.name!r})')
                        if reason == 'ImagePullBackOff':
                            now = time.monotonic()
                            pod_key = ((pod.metadata.uid or pod.metadata.name
                                        or '<unknown>')
                                       if pod.metadata else '<unknown>')
                            first_seen = image_pull_backoff_at.get(pod_key)
                            if first_seen is None:
                                image_pull_backoff_at[pod_key] = now
                                log.warning(
                                    'activity Job %r: ImagePullBackOff '
                                    '(grace window %.0fs)', job_name,
                                    _IMAGE_PULL_BACKOFF_GRACE_SECS)
                            elif (now - first_seen
                                  > _IMAGE_PULL_BACKOFF_GRACE_SECS):
                                raise ProvisionTerminalError(
                                    f'activity Job {job_name!r} pod stuck '
                                    f'in ImagePullBackOff for >'
                                    f'{_IMAGE_PULL_BACKOFF_GRACE_SECS:.0f}s '
                                    f'(container {cs.name!r})')
                elif state == 'Succeeded':
                    raise ProvisionTerminalError(
                        f'activity Job {job_name!r} completed before its '
                        'Temporal poller became ready')
                elif state == 'Failed':
                    await self._require_activity_job_status(job_name)
            await asyncio.sleep(_POLL_INTERVAL_SECS)

        raise ProvisionTimeoutError(
            f'activity Job {job_name!r} pod did not reach Running '
            f'within {timeout:.0f}s (last phase: {last_phase})')

    async def _require_activity_job_status(self, job_name: str) -> V1Job:
        """Return active Job status or raise for a terminal/missing Job."""
        try:
            job = cast(V1Job, await asyncio.to_thread(
                self._batch.read_namespaced_job_status,
                name=job_name,
                namespace=self._namespace,
            ))
            conditions = (
                (job.status.conditions if job.status else None) or [])
            if any(condition.type == 'Failed'
                   and str(condition.status).lower() == 'true'
                   for condition in conditions):
                raise ProvisionTimeoutError(
                    f'activity Job {job_name!r} exhausted its pod attempts '
                    'in Failed state during provisioning')
            if any(condition.type == 'Complete'
                   and str(condition.status).lower() == 'true'
                   for condition in conditions):
                raise ProvisionTerminalError(
                    f'activity Job {job_name!r} completed before its '
                    'Temporal poller became ready')
            return job
        except ApiException as exc:
            if exc.status == 404:
                _mark_api_error_restricted(exc)
                raise ProvisionTimeoutError(
                    f'activity Job {job_name!r} disappeared before its pod '
                    'reached Running') from exc
            raise _translate_api_error(exc) from exc

    async def _get_pool_worker_info(
        self,
        uid: str,
    ) -> Optional[PoolWorkerInfo]:
        pods = await asyncio.to_thread(
            self._core.list_namespaced_pod,
            namespace=self._namespace,
            label_selector=(f'{LABEL_APP}={LABEL_APP_VALUE_DAEMON},'
                            f'{LABEL_POOL_WORKER_UID}={uid}'),
        )
        if not pods.items:
            return None
        # Deployment has replicas=1 plus hostname anti-affinity, so normally
        # exactly one pod matches. A Recreate rollout or rebind can briefly
        # expose both generations; select the newest so callers see the live
        # pod rather than a tombstoned predecessor.
        pod = max(pods.items, key=_pod_recency_key)
        return _pod_to_pool_worker_info(
            pod, uid, self._cache_extra_tolerations_hash())

    async def _try_delete_pvc(
        self,
        pvc_name: str,
        timeout: float = 60.0,
    ) -> None:
        """Delete the PVC and ensure it's actually gone.

        Issue ``Delete`` and poll up to ``timeout`` for the API object
        to disappear normally. CSI owns its protection finalizers; if
        they do not clear, fail visibly so cleanup can be retried or
        repaired without orphaning the underlying volume.

        Idempotent: a 404 at any step is success.
        """
        try:
            await asyncio.to_thread(
                self._core.delete_namespaced_persistent_volume_claim,
                name=pvc_name,
                namespace=self._namespace,
            )
        except ApiException as exc:
            if exc.status == 404:
                log.info('_try_delete_pvc: PVC %s already deleted', pvc_name)
                return
            # Non-404 (5xx, 403 RBAC, etc.) propagates: silently
            # dropping it would orphan the PVC while the caller's
            # teardown reports success.
            _mark_api_error_restricted(exc)
            raise

        # Stage 1: bounded poll for normal removal.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                await asyncio.to_thread(
                    self._core.read_namespaced_persistent_volume_claim,
                    name=pvc_name,
                    namespace=self._namespace,
                )
            except ApiException as exc:
                if exc.status == 404:
                    return
                _mark_api_error_restricted(exc)
                raise
            await asyncio.sleep(_POLL_INTERVAL_SECS)

        raise PoolWorkerCleanupFailedError(
            f'PVC {pvc_name!r} still exists {timeout:.0f}s after '
            'deletion; CSI cleanup is incomplete and must succeed '
            'before this worker can be retired')

    async def _wait_for_daemon_pod_gone(
        self,
        uid: str,
        timeout: float,
        force_on_timeout: bool = False,
    ) -> None:
        """Poll until the daemon pod for ``uid`` is reaped (or timeout).

        Used during teardown after scaling the Deployment to 0 — we
        want to know the pod actually went away before deleting the
        PVC, otherwise the kubelet may still hold the volume mount and
        deleting the PVC under it wedges the CSI attacher finalizer.

        On timeout: raise :class:`PoolWorkerCleanupFailedError` so the
        caller (``delete_pool_worker`` → controller) can route the uid
        into a retired/quarantined set instead of silently proceeding
        to a PVC delete that would wedge.

        Pass ``force_on_timeout=True`` for the design §10 DEAD-node
        path: the caller has decided the node is gone and the volume
        must be ripped out regardless of the kubelet's mount state.
        In that case the timeout is logged and we return normally.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pods = await asyncio.to_thread(
                self._core.list_namespaced_pod,
                namespace=self._namespace,
                label_selector=(f'{LABEL_APP}={LABEL_APP_VALUE_DAEMON},'
                                f'{LABEL_POOL_WORKER_UID}={uid}'),
            )
            if not pods.items:
                return
            await asyncio.sleep(_POLL_INTERVAL_SECS)
        if force_on_timeout:
            log.warning(
                '_wait_for_daemon_pod_gone: pod for uid=%s still '
                'present after %.0fs; force_on_timeout=True (DEAD '
                'node path), proceeding to PVC delete anyway', uid, timeout)
            return
        raise PoolWorkerCleanupFailedError(
            f'daemon pod for uid={uid!r} did not terminate within '
            f'{timeout:.0f}s; refusing to delete PVC under a still-'
            f'mounted kubelet (would wedge the CSI attacher '
            f'finalizer). Caller can retry, or pass '
            f'force_on_timeout=True for the DEAD-node teardown path.')

    # ------------------------------------------------------------------
    # Pure helpers (no I/O)
    # ------------------------------------------------------------------

    def _daemon_labels(self, uid: str) -> Dict[str, str]:
        return {
            LABEL_APP: LABEL_APP_VALUE_DAEMON,
            LABEL_TENANT: self._tenant,
            LABEL_POOL_WORKER_UID: uid,
        }

    def _runtime_pod_labels(self,
                            owned_labels: Dict[str, str]) -> Dict[str, str]:
        # Routing/cleanup labels must not be overridden by chart-global labels.
        return {**self._pod_template_labels, **owned_labels}

    def _activity_label_selector(self, pool_worker_uid: Optional[str],
                                 pool_type: Optional[str],
                                 job_id: Optional[str]) -> str:
        selector_parts = [
            f'{LABEL_APP} in ({LABEL_APP_VALUE_ACTIVITY},'
            f'{LABEL_APP_VALUE_ACTIVITY_EPH})',
            f'{LABEL_TENANT}={self._tenant}',
        ]
        if pool_worker_uid is not None:
            selector_parts.append(f'{LABEL_POOL_WORKER_UID}={pool_worker_uid}')
        if pool_type is not None:
            selector_parts.append(f'{DISKGRAPH_POOL_TYPE_LABEL}={pool_type}')
        if job_id is not None:
            selector_parts.append(f'{LABEL_JOB_ID_HASH}={_hash_job_id(job_id)}')
        return ','.join(selector_parts)

    def _daemon_label_selector(self) -> str:
        return (f'{LABEL_APP}={LABEL_APP_VALUE_DAEMON},'
                f'{LABEL_TENANT}={self._tenant}')

    def _storage_size_from_pvc_spec(self, spec: PvcSpec) -> str:
        return f'{spec.size_gib}Gi'

    # ------------------------------------------------------------------
    # Init-canary scheduling primitives (mirror the daemon / eph pods)
    # ------------------------------------------------------------------
    # The init canary (``pool.worker_manager.init_canary``) builds short-
    # lived pods that must reproduce the EXACT scheduling envelope of the
    # real daemon / eph activity pods so a node-pool misconfig fails at
    # deploy rather than at first job. These thin accessors expose the
    # private selector/taint/storage-class state without duplicating the
    # label keys or taint-value derivation in the canary module.

    def cache_canary_scheduling(
            self, ssd_size_gib: int,
            gpu_count: Optional[int] = 1) -> '_PoolScheduling':
        """nodeSelector + tolerations the daemon uses for a cache tier."""
        return _PoolScheduling(
            node_selector=self._cache_node_selector(ssd_size_gib, gpu_count),
            tolerations=self._pool_tolerations(self._cache_pool_taint_value),
        )

    def eph_canary_scheduling(self) -> '_PoolScheduling':
        """nodeSelector + tolerations the eph activity pods use."""
        return _PoolScheduling(
            node_selector=self._eph_node_selector(),
            tolerations=self._pool_tolerations(self._eph_pool_taint_value),
        )

    def _pool_tolerations(self, tenant_taint_value: str) -> List[V1Toleration]:
        """Keep the tenant toleration first, then append unique configured extras."""
        tolerations = [
            V1Toleration(key=self._pool_taint_key, operator='Equal',
                         value=tenant_taint_value, effect='NoSchedule')
        ]
        seen = {_toleration_key(tolerations[0])}
        for extra in self._pool_extra_tolerations:
            key = _toleration_key(extra)
            if key not in seen:
                tolerations.append(copy.deepcopy(extra))
                seen.add(key)
        return tolerations

    def _cache_extra_tolerations_hash(self) -> Optional[str]:
        extras = self._pool_tolerations(self._cache_pool_taint_value)[1:]
        if not extras:
            # Preserve legacy pod metadata and reuse when no extras are set.
            return None
        payload = '\n'.join(
            sorted(json.dumps(_toleration_key(t)) for t in extras))
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    @property
    def storage_class(self) -> str:
        """CSI StorageClass used for cache PVCs."""
        return self._storage_class

    @property
    def core_v1_api(self) -> k8s_client.CoreV1Api:
        """CoreV1Api this manager is bound to (shared with the canary)."""
        return self._core

    @property
    def apps_v1_api(self) -> k8s_client.AppsV1Api:
        """AppsV1Api this manager is bound to (shared with the canary)."""
        return self._apps

    def cache_pvc_size_str(self, spec: PvcSpec) -> str:
        """PVC ``requests.storage`` string the daemon uses for a tier."""
        return self._storage_size_from_pvc_spec(spec)

    def _cache_node_selector(self, ssd_size_gib: Optional[int],
                             gpu_count: Optional[int] = None) -> Dict[str, str]:
        """``nodeSelector`` for cache-pool pods.

        Always pins the cache pool role via ``kumo.ai/diskgraph-pool-type``.
        Set capacity labels whenever their values are known. ``None`` keeps
        legacy/default-size rebinds role-only; activity pod affinity pins
        activity pods to the daemon's exact host.
        """
        selector = {DISKGRAPH_POOL_TYPE_LABEL: POOL_TYPE_CACHE}
        if ssd_size_gib is not None:
            selector[DISKGRAPH_SSD_SIZE_LABEL] = str(ssd_size_gib)
        if gpu_count is not None:
            selector[DISKGRAPH_GPU_COUNT_LABEL] = str(gpu_count)
        return selector

    def _eph_node_selector(self) -> Dict[str, str]:
        """``nodeSelector`` for ephemeral-pool activity pods: the pool role
        via ``kumo.ai/diskgraph-pool-type``. Sizing is portable via the pod's
        standard ``ephemeral-storage`` request, so eph needs no size label."""
        return {DISKGRAPH_POOL_TYPE_LABEL: POOL_TYPE_EPHEMERAL}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _toleration_key(
        t: V1Toleration) -> tuple[str, str, str, str, Optional[int]]:
    """Compare all toleration fields, accounting for Kubernetes defaulting."""
    return (t.key or '', t.operator or 'Equal', t.value or '', t.effect
            or '', t.toleration_seconds)


def _load_kube_config(context: Optional[str] = None) -> None:
    """Load in-cluster config when running inside a pod; else kubeconfig.

    ``context``: explicit kubeconfig context name (e.g.
    ``arn:aws:eks:us-west-2:123456789012:cluster/example-cluster``). When
    set, the Python kubernetes client is pinned to that cluster
    regardless of the operator's current kubectl context. Off-cluster
    runs (devserver) MUST pass this — relying on the global
    current-context has bitten us repeatedly (devserver's own setup
    flips it to dev-uw2, k8s ApiException 404 follows). ``None``
    falls back to the kubeconfig default for backward compat.
    """
    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config(context=context)


def _pvc_name_for_uid(uid: str) -> str:
    return f'diskgraph-pvc-{uid}'


def _deployment_name_for_uid(uid: str) -> str:
    return f'diskgraph-daemon-{uid}'


def _pdb_name_for_uid(uid: str) -> str:
    return f'diskgraph-pdb-{uid}'


def _daemon_anti_affinity() -> V1PodAntiAffinity:
    """Hard antiaffinity: at most one daemon pod per node (design §7.2)."""
    return V1PodAntiAffinity(
        required_during_scheduling_ignored_during_execution=[
            V1PodAffinityTerm(
                label_selector=V1LabelSelector(
                    match_labels={LABEL_APP: LABEL_APP_VALUE_DAEMON}, ),
                topology_key='kubernetes.io/hostname',
            ),
        ], )


def _daemon_pod_affinity(uid: str) -> V1Affinity:
    """Pin a durable activity to its daemon's host."""
    return V1Affinity(
        pod_affinity=V1PodAffinity(
            required_during_scheduling_ignored_during_execution=[
                V1PodAffinityTerm(
                    label_selector=V1LabelSelector(match_labels={
                        LABEL_APP: LABEL_APP_VALUE_DAEMON,
                        LABEL_POOL_WORKER_UID: uid,
                    }),
                    topology_key='kubernetes.io/hostname',
                ),
            ], ), )


def _pod_recency_key(pod: V1Pod) -> tuple[float, str]:
    metadata = pod.metadata
    return (
        metadata.creation_timestamp.timestamp()
        if metadata and metadata.creation_timestamp else 0.0,
        metadata.name if metadata and metadata.name else '',
    )


def _activity_attempt_priority_key(pod: V1Pod) -> tuple[bool, float, str]:
    return (
        _derive_pod_status(pod) in {'Pending', 'Running', 'Registered'},
        *_pod_recency_key(pod),
    )


def _require_current_extra_tolerations(worker: PoolWorkerInfo) -> None:
    # Final guard when startup reconciliation failed or a daemon appeared
    # after its observation. Never create an activity pinned to stale policy.
    if not worker.extra_tolerations_match:
        raise PoolWorkerNotReadyError(
            f'pool worker {worker.pool_worker_uid!r} has outdated extra '
            'tolerations; maintenance cleanup must retire its daemon and '
            'cache PVC before it can be replaced')


def _pod_to_pool_worker_info(
        pod: V1Pod, uid: str,
        expected_extra_tolerations_hash: Optional[str] = None) -> PoolWorkerInfo:
    metadata = pod.metadata
    annotations = (metadata.annotations or {}) if metadata else {}
    return PoolWorkerInfo(
        pool_worker_uid=uid,
        pvc_name=_pvc_name_for_uid(uid),
        daemon_pod_name=metadata.name if metadata else None,
        node_name=(pod.spec.node_name if pod.spec else None) or None,
        status=_derive_pod_status(pod),
        created_at=(metadata.creation_timestamp.timestamp()
                    if metadata and metadata.creation_timestamp else None),
        deletion_started_at=(metadata.deletion_timestamp.timestamp()
                             if metadata and metadata.deletion_timestamp else
                             None),
        extra_tolerations_match=(
            annotations.get(_ANNOTATION_POOL_EXTRA_TOLERATIONS_HASH)
            == expected_extra_tolerations_hash),
    )


def _pod_to_activity_pod_info(pod: V1Pod) -> ActivityPodInfo:
    labels = (pod.metadata.labels or {}) if pod.metadata else {}
    # job_id is read from the annotation, NOT the label: the label
    # holds the truncated SHA-256 hash so K8s-side filtering stays
    # well-formed, while the annotation preserves the raw Temporal
    # workflow_id for callers (and for `kubectl describe pod`).
    annotations = ((pod.metadata.annotations or {}) if pod.metadata else {})
    return ActivityPodInfo(
        pod_name=pod.metadata.name if pod.metadata else '',
        pool_worker_uid=labels.get(LABEL_POOL_WORKER_UID),
        job_id=annotations.get(ANNOTATION_JOB_ID),
        node_name=(pod.spec.node_name if pod.spec else None) or None,
        status=_derive_pod_status(pod),
    )


def _derive_pod_status(pod: V1Pod) -> PodStatus:
    """Map k8s pod phase + metadata to :data:`PodStatus`.

    "Registered" heuristic: phase=Running AND all container statuses
    ``ready``. Replaceable with a Temporal task-queue check once
    daemon readiness is wired to task-queue binding.
    """
    if pod.metadata and pod.metadata.deletion_timestamp is not None:
        return 'Terminating'
    phase = pod.status.phase if pod.status else 'Pending'
    if phase == 'Pending':
        return 'Pending'
    if phase == 'Running':
        cs = pod.status.container_statuses if pod.status else None
        if cs and all(c.ready for c in cs):
            return 'Registered'
        return 'Running'
    if phase == 'Succeeded':
        # One-shot activity pods that ran to completion. Distinct from
        # 'Terminating' so design §6.5 cleanup callers can filter on
        # `status == 'Succeeded'` to find finished-and-reapable pods
        # (vs pods currently being deleted, which carry deletion_
        # timestamp and are mapped to 'Terminating' above).
        return 'Succeeded'
    if phase == 'Failed':
        return 'Failed'
    return 'Pending'


def _unschedulable_reason(  # pyright: ignore[reportUnusedFunction]
        pod: V1Pod) -> Optional[str]:
    # NOTE: not dead — imported and called by
    # ``pool.worker_manager.init_canary``. The pragma suppresses pyright's
    # reportUnusedFunction false positive: pyright treats this underscore-
    # prefixed module-level helper as private and flags it because nothing
    # in THIS file references it; the cross-module import does not count.
    """Return the scheduler's human-readable reason a pod can't be placed.

    Reads the ``PodScheduled`` condition: when the scheduler fails to
    find a node it sets ``PodScheduled=False`` with
    ``reason=Unschedulable`` and a ``message`` enumerating the
    predicates that rejected every node (e.g. ``0/3 nodes are
    available: 3 node(s) didn't match Pod's node affinity/selector``,
    ``... Insufficient nvidia.com/gpu``, ``... had untolerated taint``,
    or a CSI ``Insufficient <storage>`` predicate). Returns that
    message, or ``None`` when the pod has no such condition (still being
    evaluated, or already scheduled). Used by the init canary to
    classify a stuck-Pending pod into label/GPU/disk misconfig.
    """
    if not pod.status or not pod.status.conditions:
        return None
    for cond in pod.status.conditions:
        if cond.type == 'PodScheduled' and cond.status == 'False':
            return cond.message or cond.reason or 'Unschedulable'
    return None


# 64-bit truncated SHA-256 hex. At 16 hex chars the value sits well
# under the 63-char K8s label cap and uses only the [0-9a-f] subset of
# the allowed [A-Za-z0-9._-] charset. Collision probability per the
# birthday bound: ~1% at 2^32 distinct ids — far above any realistic
# per-tenant lifetime workload.
_LABEL_HASH_HEX_LEN = 16


def _hash_job_id(job_id: str) -> str:
    """Stable, K8s-label-safe hex digest of a raw ``job_id``.

    K8s label values are constrained to ``[A-Za-z0-9._-]`` and a
    63-char max, which raw Temporal workflow_ids
    (e.g. ``dg-ingest:NA-abc123@graph-data-version-456:01HXYZ789``)
    violate on both axes. The manager hashes on every write and read
    path so the K8s side stays well-formed; the unhashed value lives
    on the :data:`ANNOTATION_JOB_ID` annotation for human
    readability and for :func:`_pod_to_activity_pod_info` to return
    to callers.

    Returns the first :data:`_LABEL_HASH_HEX_LEN` hex chars of
    SHA-256(``job_id``).
    """
    return hashlib.sha256(
        job_id.encode('utf-8')).hexdigest()[:_LABEL_HASH_HEX_LEN]


def _hash_snapshot_id(snapshot_id: str) -> str:
    """Stable, K8s-label-safe hex digest of a raw ``snapshot_id``.

    Mirrors :func:`_hash_job_id` — see that function's docstring for
    rationale. Raw value lives on :data:`ANNOTATION_SNAPSHOT_ID`.
    The two helpers stay distinct (rather than one generic
    ``_hash_label_value``) so the call sites self-document which
    field is being normalised.
    """
    return hashlib.sha256(
        snapshot_id.encode('utf-8')).hexdigest()[:_LABEL_HASH_HEX_LEN]


def _dshm_volume(
    resources: Optional[ResourceSpec],
) -> tuple[List[V1VolumeMount], List[V1Volume]]:
    """tmpfs ``/dev/shm`` volume + mount for activity pods.

    PyTorch DataLoader workers ship sampled batches to the main
    process via POSIX shared memory; the container default 64 MiB
    ``/dev/shm`` exhausts within a few prefetched batches. The dshm
    emptyDir supplies the required capacity.

    That transfer is CPU-side, so CPU-only ``train`` / ``predict``
    pods need it too (kumo-ai/kumo#29547). GPU pods are capped by the
    ``memory_limit`` ``ResourceSpec`` gives them; CPU pods may be
    BestEffort with none, hence the explicit ceiling.
    """
    is_gpu = resources is not None and resources.gpu_count > 0
    return (
        [V1VolumeMount(mount_path='/dev/shm', name='dshm')],
        [
            V1Volume(
                name='dshm',
                empty_dir=V1EmptyDirVolumeSource(
                    medium='Memory',
                    size_limit=None if is_gpu else _CPU_DSHM_SIZE_LIMIT,
                ),
            )
        ],
    )


def _activity_worker_pod_name_env() -> V1EnvVar:
    """Downward-API pod name used to make the Temporal poller identity unique."""
    return V1EnvVar(
        name=ACTIVITY_WORKER_POD_NAME_ENV,
        value_from=V1EnvVarSource(field_ref=V1ObjectFieldSelector(
            field_path='metadata.name')),
    )


def _env_dict_to_v1envvar_list(
    env: Optional[Dict[str, str]], ) -> List[V1EnvVar]:
    """Translate a caller-provided ``Dict[str, str]`` of env vars to
    the ``List[V1EnvVar]`` shape ``V1Container.env`` expects.

    Returns ``[]`` for ``None`` or empty input — V1Container accepts
    that. The manager is intentionally agnostic to which keys the
    daemon image expects: composing the right env contract (e.g.
    SNAPSHOT_ID / TEMPORAL_TASK_QUEUE / POOL_WORKER_UID) is the
    strategy / controller layer's job, not this manager's.
    """
    if not env:
        return []
    return [V1EnvVar(name=k, value=v) for k, v in env.items()]


_GPU_RESOURCE_KEY = 'nvidia.com/gpu'


def _compose_resource_requirements(
    spec: Optional[ResourceSpec] = None,
    *,
    ephemeral_storage_request: Optional[str] = None,
    ephemeral_storage_limit: Optional[str] = None,
) -> Optional[V1ResourceRequirements]:
    """Build a ``V1ResourceRequirements`` from a ``ResourceSpec`` plus
    optional ``ephemeral-storage`` entries (eph activity pods stamp
    those alongside cpu/memory/gpu so all four resource families share
    one ``V1ResourceRequirements``).

    See :class:`ResourceSpec` for QoS / GPU semantics. Returns ``None``
    when no fields would be set — the caller passes that through to
    ``V1Container.resources=None`` so the manifest field is omitted
    entirely (preserves byte-for-byte compat with BestEffort-QoS pods
    that opt out of all knobs).
    """
    spec = spec or ResourceSpec()
    requests: Dict[str, str] = {}
    limits: Dict[str, str] = {}
    if spec.cpu_request is not None:
        requests['cpu'] = spec.cpu_request
    if spec.cpu_limit is not None:
        limits['cpu'] = spec.cpu_limit
    if spec.memory_request is not None:
        requests['memory'] = spec.memory_request
    if spec.memory_limit is not None:
        limits['memory'] = spec.memory_limit
    if ephemeral_storage_request is not None:
        requests['ephemeral-storage'] = ephemeral_storage_request
    if ephemeral_storage_limit is not None:
        limits['ephemeral-storage'] = ephemeral_storage_limit
    if spec.gpu_count > 0:
        gpu_str = str(spec.gpu_count)
        requests[_GPU_RESOURCE_KEY] = gpu_str
        limits[_GPU_RESOURCE_KEY] = gpu_str
    if not requests and not limits:
        return None
    return V1ResourceRequirements(requests=requests, limits=limits)


def _mark_api_error_restricted(exc: ApiException) -> str:
    status = exc.status
    safe_message = ('Kubernetes API operation failed '
                    f'(status={status if status is not None else "unknown"})')
    mark_exception_restricted(exc, safe_message=safe_message)
    return safe_message


def _translate_api_error(exc: ApiException) -> Exception:
    """Best-effort translation from k8s ``ApiException`` to our taxonomy.

    Only confident mappings are translated; everything else is returned
    as-is for the caller to handle. The pool controller can broaden
    this as we gather production signal.

    Retry-classification mappings (the controller's retry layer needs
    these distinctions): 429 → ApiServerThrottledError (back off),
    5xx → ApiServerTransientError (retry with budget), 409 →
    PoolWorkerAlreadyExistsError (likely controller crash recovery —
    the manager doesn't auto-recover; caller decides).
    """
    status = exc.status
    body = (exc.body or '').lower() if isinstance(exc.body, str) else ''
    safe_message = _mark_api_error_restricted(exc)
    if status == 429:
        return ApiServerThrottledError(safe_message)
    if status is not None and 500 <= status < 600:
        return ApiServerTransientError(safe_message)
    if status == 409:
        return PoolWorkerAlreadyExistsError(safe_message)
    if status == 403:
        # 403 with 'exceeded quota' is an objectcount/ResourceQuota
        # rejection — transient under retry once usage drops. Everything
        # else 403 (RBAC denial, IRSA can't satisfy AWS action, admission
        # webhook block) is deterministic: needs an operator fix.
        if 'exceeded quota' in body:
            return QuotaExceededError(safe_message)
        return K8sPermissionError(safe_message)
    if status == 404 and ('nodepool' in body or 'nodeclass' in body):
        return NodePoolNotFoundError(safe_message)
    if status == 422:
        return InvalidPvcSpecError(safe_message)
    # No specific mapping. Returning ``exc`` here would mean callers do
    # ``raise exc from exc``, setting ``exc.__cause__ = exc`` — a
    # self-referential chain that infinite-recurses Temporal's failure
    # serializer (``temporalio/converter.py:_error_to_failure``). Wrap
    # in a fresh exception instead; original ApiException is preserved
    # via the caller's ``raise ... from exc``.
    return UnclassifiedApiError(safe_message)
