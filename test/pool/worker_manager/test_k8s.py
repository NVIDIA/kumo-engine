# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :mod:`pool.worker_manager.k8s` — K8sPoolWorkerManager
and its module-level helpers.

Uses ``MagicMock(spec=...)`` to inject fake CoreV1Api / AppsV1Api /
BatchV1Api clients. No cluster, no kubeconfig, no real Kubernetes —
runs in <1 s and validates the spec we hand to the K8s API plus the
control-flow around it.
"""
# Stub injection for laptop runs (opt-in via
# KUMO_WORKER_MANAGER_TEST_STUBS=1). Must run before ``import pool.*``.
import os as _os  # noqa: E402,I100
import sys as _sys  # noqa: E402

if _os.environ.get("KUMO_WORKER_MANAGER_TEST_STUBS") == "1":
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "_stubs"))

import asyncio  # noqa: E402
import datetime  # noqa: E402
from typing import Optional  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402
from kubernetes import client as k8s_client  # noqa: E402
from kubernetes.client.models import (
    V1ContainerState,
    V1ContainerStateWaiting,
    V1ContainerStatus,
    V1Deployment,
    V1DeploymentSpec,
    V1JobList,
    V1LabelSelector,
    V1ObjectMeta,
    V1PersistentVolumeClaim,
    V1PersistentVolumeClaimSpec,
    V1PersistentVolumeClaimStatus,
    V1PersistentVolumeClaimVolumeSource,
    V1Pod,
    V1PodCondition,
    V1PodList,
    V1PodSpec,
    V1PodStatus,
    V1PodTemplateSpec,
    V1ResourceRequirements,
    V1Toleration,
    V1Volume,
)
from kubernetes.client.rest import ApiException

from common_utils.error_diagnostics import (
    exception_message_should_be_logged_at_error,
    safe_exception_message_for_log,
)
from pool.task_queues import ACTIVITY_WORKER_POD_NAME_ENV
from pool.worker_manager.k8s import (
    ANNOTATION_JOB_ID,
    ANNOTATION_SNAPSHOT_ID,
    DISKGRAPH_GPU_COUNT_LABEL,
    DISKGRAPH_POOL_TYPE_LABEL,
    DISKGRAPH_SSD_SIZE_LABEL,
    LABEL_ACTIVITY,
    LABEL_APP,
    LABEL_APP_VALUE_ACTIVITY,
    LABEL_APP_VALUE_ACTIVITY_EPH,
    LABEL_APP_VALUE_DAEMON,
    LABEL_JOB_ID_HASH,
    LABEL_POOL_WORKER_UID,
    LABEL_SNAPSHOT_ID_HASH,
    LABEL_TENANT,
    POOL_TYPE_CACHE,
    POOL_TYPE_EPHEMERAL,
    K8sPoolWorkerManager,
    _daemon_anti_affinity,
    _derive_pod_status,
    _dshm_volume,
    _hash_job_id,
    _hash_snapshot_id,
    _pdb_name_for_uid,
    _pod_to_activity_pod_info,
    _pod_to_pool_worker_info,
    _resolve_service_account,
    _translate_api_error,
)
from pool.worker_manager.types import (
    ApiServerThrottledError,
    ApiServerTransientError,
    InvalidPvcSpecError,
    K8sPermissionError,
    NodePoolNotFoundError,
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

NAMESPACE = "diskgraph-test"
TENANT = "test"
POD_TEMPLATE_LABELS = {
    "azure.workload.identity/use": "true",
    LABEL_APP: "caller-must-not-override",
}

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_manager(
    core: Optional[MagicMock] = None,
    apps: Optional[MagicMock] = None,
    batch: Optional[MagicMock] = None,
    policy: Optional[MagicMock] = None,
    pod_template_labels: Optional[dict[str, str]] = None,
    storage_class: str = "topolvm-ssd",
) -> K8sPoolWorkerManager:
    core = core or MagicMock(spec=k8s_client.CoreV1Api)
    if batch is None:
        batch = MagicMock(spec=k8s_client.BatchV1Api)
        batch.list_namespaced_job.return_value = V1JobList(items=[])
    return K8sPoolWorkerManager(
        namespace=NAMESPACE,
        tenant=TENANT,
        storage_class=storage_class,
        daemon_image="kumo/daemon:test",
        daemon_command=["/usr/bin/daemon"],
        daemon_args=["--mode=test"],
        pod_template_labels=pod_template_labels,
        load_kube_config=False,
        core_v1_api=core,
        apps_v1_api=apps or MagicMock(spec=k8s_client.AppsV1Api),
        batch_v1_api=batch,
        policy_v1_api=policy or MagicMock(spec=k8s_client.PolicyV1Api),
    )


def _make_pod(
    name: str,
    *,
    phase: str = "Running",
    container_ready: bool = True,
    node_name: str = "node-1",
    deletion_timestamp=None,
    labels: Optional[dict] = None,
    annotations: Optional[dict] = None,
    creation_timestamp=None,
    waiting_reason: Optional[str] = None,
    conditions: Optional[list] = None,
) -> V1Pod:
    cs_state = (V1ContainerState(waiting=V1ContainerStateWaiting(
        reason=waiting_reason)) if waiting_reason else None)
    return V1Pod(
        metadata=V1ObjectMeta(
            name=name,
            namespace=NAMESPACE,
            labels=labels or {},
            annotations=annotations or {},
            deletion_timestamp=deletion_timestamp,
            creation_timestamp=creation_timestamp,
        ),
        spec=V1PodSpec(containers=[], node_name=node_name),
        status=V1PodStatus(
            phase=phase,
            container_statuses=[
                V1ContainerStatus(
                    name="c",
                    ready=container_ready,
                    image="kumo/x:test",
                    image_id="",
                    restart_count=0,
                    state=cs_state,
                ),
            ],
            conditions=conditions,
        ),
    )


def _api_exc(status: int, body: str = "") -> ApiException:
    exc = ApiException(status=status)
    exc.body = body
    return exc


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_derive_pod_status_pending():
    pod = _make_pod("p", phase="Pending", container_ready=False)
    assert _derive_pod_status(pod) == "Pending"


def test_derive_pod_status_running_container_not_ready():
    pod = _make_pod("p", phase="Running", container_ready=False)
    assert _derive_pod_status(pod) == "Running"


def test_derive_pod_status_running_all_ready_is_registered():
    pod = _make_pod("p", phase="Running", container_ready=True)
    assert _derive_pod_status(pod) == "Registered"


def test_derive_pod_status_failed():
    pod = _make_pod("p", phase="Failed", container_ready=False)
    assert _derive_pod_status(pod) == "Failed"


def test_derive_pod_status_terminating_via_deletion_timestamp():
    pod = _make_pod(
        "p",
        phase="Running",
        deletion_timestamp=datetime.datetime(2026, 1, 1),
    )
    assert _derive_pod_status(pod) == "Terminating"


def test_derive_pod_status_succeeded():
    # Activity Jobs that ran to completion surface as Succeeded — kept
    # distinct from Terminating so design §6.5 cleanup callers can
    # filter on `status == 'Succeeded'` to find finished-but-reapable
    # pods (vs in-progress-deletion pods, which carry a deletion_
    # timestamp and map to 'Terminating').
    pod = _make_pod("p", phase="Succeeded", container_ready=True)
    assert _derive_pod_status(pod) == "Succeeded"


def test_pod_to_pool_worker_info_populates_fields():
    created_at = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    pod = _make_pod(
        "daemon-abc-yyy",
        node_name="ip-10-0-1-1",
        creation_timestamp=created_at,
    )
    info = _pod_to_pool_worker_info(pod, "abc")
    assert info.pool_worker_uid == "abc"
    assert info.pvc_name == "diskgraph-pvc-abc"
    assert info.daemon_pod_name == "daemon-abc-yyy"
    assert info.node_name == "ip-10-0-1-1"
    assert info.status == "Registered"
    assert info.created_at == created_at.timestamp()
    assert info.deletion_started_at is None


def test_pod_to_pool_worker_info_maps_deletion_timestamp():
    deleted_at = datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc)
    pod = _make_pod("daemon-abc-yyy", deletion_timestamp=deleted_at)

    info = _pod_to_pool_worker_info(pod, "abc")

    assert info.status == "Terminating"
    assert info.deletion_started_at == deleted_at.timestamp()


def test_pod_to_activity_pod_info_reads_labels():
    # job_id now lives on the annotation, not the label — the label
    # carries the (opaque) hash, the annotation preserves the raw
    # value for callers / `kubectl describe`.
    pod = _make_pod(
        "act-1",
        labels={
            LABEL_POOL_WORKER_UID: "abc",
            LABEL_JOB_ID_HASH: _hash_job_id("job-9"),
        },
        annotations={ANNOTATION_JOB_ID: "job-9"},
    )
    info = _pod_to_activity_pod_info(pod)
    assert info.pod_name == "act-1"
    assert info.pool_worker_uid == "abc"
    assert info.job_id == "job-9"


def test_pod_to_activity_pod_info_eph_has_no_pool_worker_uid():
    pod = _make_pod(
        "eph-1",
        labels={LABEL_JOB_ID_HASH: _hash_job_id("j")},
        annotations={ANNOTATION_JOB_ID: "j"},
    )
    info = _pod_to_activity_pod_info(pod)
    assert info.pool_worker_uid is None
    assert info.job_id == "j"


def test_dshm_volume_bounds_cpu_activities():
    for resources in (None, ResourceSpec(gpu_count=0)):
        _, volumes = _dshm_volume(resources)
        empty_dir = volumes[0].empty_dir
        assert empty_dir is not None
        assert empty_dir.size_limit == '8Gi'


def test_dshm_volume_emits_unbounded_memory_emptydir_for_gpu_activity():
    mounts, volumes = _dshm_volume(ResourceSpec(gpu_count=1))
    assert len(mounts) == 1
    assert mounts[0].mount_path == '/dev/shm'
    assert mounts[0].name == 'dshm'
    assert len(volumes) == 1
    assert volumes[0].name == 'dshm'
    # medium=Memory makes this a tmpfs (RAM-backed) so torch
    # multiprocessing has enough room for prefetched batches; no
    # size limit preserves the default-half-node-memory behavior.
    assert volumes[0].empty_dir is not None
    assert volumes[0].empty_dir.medium == 'Memory'
    assert volumes[0].empty_dir.size_limit is None


def test_deployment_name_for_uid_static_method():
    # Exposed so the placement strategy can derive the daemon's
    # k8s_deployment_name (the stable label that survives pod restart)
    # without re-implementing the format string.
    assert K8sPoolWorkerManager.deployment_name_for_uid(
        'pool-worker-abc123') == 'diskgraph-daemon-pool-worker-abc123'


def test_manager_namespace_and_kube_context_properties():
    mgr = _make_manager()
    # ``_make_manager`` constructs with namespace=NAMESPACE and no
    # explicit kube_context (kubeconfig is loaded in-cluster style).
    assert mgr.namespace == NAMESPACE
    assert mgr.kube_context is None


def test_resolve_service_account_precedence(monkeypatch):
    # explicit arg > POD_SERVICE_ACCOUNT env > legacy default
    monkeypatch.delenv('POD_SERVICE_ACCOUNT', raising=False)
    assert _resolve_service_account(None) == 'kumo-temporal'
    monkeypatch.setenv('POD_SERVICE_ACCOUNT', 'kumo')
    assert _resolve_service_account(None) == 'kumo'
    assert _resolve_service_account('explicit-sa') == 'explicit-sa'


def test_daemon_anti_affinity_uses_hostname_topology():
    aa = _daemon_anti_affinity()
    terms = aa.required_during_scheduling_ignored_during_execution
    assert terms is not None
    term = terms[0]
    assert term.topology_key == "kubernetes.io/hostname"
    assert term.label_selector is not None
    assert term.label_selector.match_labels == {
        LABEL_APP: LABEL_APP_VALUE_DAEMON,
    }


def test_storage_size_from_pvc_spec_uses_size_gib_directly():
    mgr = _make_manager()
    # PvcSpec carries the exact tier GiB; no headroom applied.
    assert mgr._storage_size_from_pvc_spec(PvcSpec(size_gib=900)) == "900Gi"
    assert mgr._storage_size_from_pvc_spec(PvcSpec(size_gib=1800)) == "1800Gi"


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------


def test_translate_quota_exceeded():
    out = _translate_api_error(_api_exc(403, "exceeded quota: storage"))
    assert isinstance(out, QuotaExceededError)


def test_translate_403_rbac_is_permission_error():
    out = _translate_api_error(
        _api_exc(
            403,
            'pods is forbidden: User "system:serviceaccount:kumo-dataplane-x:'
            'kumo-temporal" cannot create resource "pods" in API group ""'))
    assert isinstance(out, K8sPermissionError)
    # Not retryable: K8sPermissionError must NOT inherit RuntimeError so
    # AutoRPC's ``is_retryable`` falls through to non-retryable.
    assert not isinstance(out, RuntimeError)


def test_translate_403_quota_takes_precedence_over_rbac():
    # If the body contains 'exceeded quota', that wins — quota is the
    # transient 403 case.
    out = _translate_api_error(
        _api_exc(403, "pods exceeded quota: requests.cpu used 100 limit 100"))
    assert isinstance(out, QuotaExceededError)
    assert not isinstance(out, K8sPermissionError)


def test_translate_nodeclass_not_found():
    out = _translate_api_error(
        _api_exc(404, "ec2nodeclass.karpenter.k8s.aws not found"), )
    assert isinstance(out, NodePoolNotFoundError)


def test_translate_invalid_resource_returns_invalidhint():
    out = _translate_api_error(_api_exc(422, "invalid resource request"))
    assert isinstance(out, InvalidPvcSpecError)


def test_translate_unknown_wraps_in_unclassified_api_error():
    # Unmapped codes (e.g. 400 Bad Request) get wrapped in a fresh
    # UnclassifiedApiError with a safe status-only message.
    # Wrapping (instead of re-raising the ApiException directly)
    # avoids a self-referential __cause__ chain when callers do
    # ``raise translate(exc) from exc`` — Temporal's failure
    # serializer infinite-recurses on cycles. __cause__ itself is
    # set by the caller's ``raise ... from exc``, not by this helper.
    exc = _api_exc(400, "bad request")
    out = _translate_api_error(exc)
    assert isinstance(out, UnclassifiedApiError)
    assert out is not exc
    assert str(out) == 'Kubernetes API operation failed (status=400)'


@pytest.mark.parametrize(
    ('status', 'body', 'expected_type'),
    [
        (403, 'pods is forbidden provider-secret-canary', K8sPermissionError),
        (429, 'throttled provider-secret-canary', ApiServerThrottledError),
        (503, 'webhook provider-secret-canary', ApiServerTransientError),
        (400, 'bad request provider-secret-canary', UnclassifiedApiError),
    ],
)
def test_translate_api_error_keeps_provider_response_restricted(
        status, body, expected_type):
    exc = _api_exc(status, body)

    translated = _translate_api_error(exc)

    assert isinstance(translated, expected_type)
    assert 'provider-secret-canary' not in str(translated)
    assert exception_message_should_be_logged_at_error(translated)
    assert not exception_message_should_be_logged_at_error(exc)
    assert safe_exception_message_for_log(exc) == (
        f'Kubernetes API operation failed (status={status})')


# ---------------------------------------------------------------------------
# provision_pool_worker (async, mocked k8s)
# ---------------------------------------------------------------------------


def test_provision_pool_worker_creates_pvc_then_deployment():
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    # First poll returns Registered daemon pod.
    core.list_namespaced_pod.return_value = V1PodList(
        items=[
            _make_pod(
                "diskgraph-daemon-uid-xxx",
                labels={
                    LABEL_APP: LABEL_APP_VALUE_DAEMON,
                    LABEL_POOL_WORKER_UID: "uid",
                },
            ),
        ], )

    mgr = _make_manager(core=core, apps=apps)
    info = asyncio.run(
        mgr.provision_pool_worker(pool_worker_uid="uid", timeout=5.0), )

    assert info.status == "Registered"
    assert info.pool_worker_uid == "uid"
    assert info.pvc_name == "diskgraph-pvc-uid"
    assert core.create_namespaced_persistent_volume_claim.called
    assert apps.create_namespaced_deployment.called
    pvc_call = core.create_namespaced_persistent_volume_claim.call_args
    pvc_body = pvc_call.kwargs["body"]
    # PVC must be RWO so daemon + activity pods can co-mount on the
    # same node (design §6.2). RWOP would block this.
    assert pvc_body.spec.access_modes == ["ReadWriteOnce"]
    assert pvc_body.spec.storage_class_name == "topolvm-ssd"
    assert pvc_body.metadata.labels[LABEL_POOL_WORKER_UID] == "uid"
    daemon_spec = (apps.create_namespaced_deployment.call_args.kwargs["body"].
                   spec.template.spec)
    assert daemon_spec.security_context.fs_group == 999
    assert daemon_spec.security_context.fs_group_change_policy == "OnRootMismatch"
    assert daemon_spec.security_context.seccomp_profile.type == "RuntimeDefault"
    daemon_security = daemon_spec.containers[0].security_context
    assert daemon_security.privileged is False
    assert daemon_security.allow_privilege_escalation is False
    assert daemon_security.run_as_non_root is True
    assert daemon_security.capabilities.drop == ["ALL"]


def test_provision_pool_worker_rolls_back_pvc_on_deployment_failure():
    # Fresh-mint path: pool_worker_uid omitted → uid is generated, so
    # this exercises the rollback branch. The reclaim path
    # (explicit pool_worker_uid) intentionally skips rollback —
    # covered by test_provision_pool_worker_reclaim_skips_rollback_on_wait_failure.
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    apps.create_namespaced_deployment.side_effect = _api_exc(
        422,
        "invalid pod spec",
    )

    mgr = _make_manager(core=core, apps=apps)
    with pytest.raises(InvalidPvcSpecError):
        asyncio.run(mgr.provision_pool_worker(timeout=5.0))

    # PVC was created, then cleanup deletion was attempted.
    assert core.create_namespaced_persistent_volume_claim.called
    assert core.delete_namespaced_persistent_volume_claim.called


def test_provision_pool_worker_timeout_raises():
    """Generic provision timeout — daemon pod stuck Pending while
    PVC is already Bound. Should surface ProvisionTimeoutError, not
    PVCBindTimeoutError (PVC is fine, daemon is the holdup)."""
    from kubernetes.client.models import (
        V1PersistentVolumeClaim,
        V1PersistentVolumeClaimStatus,
    )
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    core.list_namespaced_pod.return_value = V1PodList(
        items=[
            _make_pod(
                "p",
                phase="Pending",
                container_ready=False,
                labels={
                    LABEL_APP: LABEL_APP_VALUE_DAEMON,
                    LABEL_POOL_WORKER_UID: "uid",
                },
            ),
        ], )
    # PVC is Bound — so timeout disambiguator falls through to the
    # generic ProvisionTimeoutError, not PVCBindTimeoutError.
    core.read_namespaced_persistent_volume_claim.return_value = (
        V1PersistentVolumeClaim(
            metadata=V1ObjectMeta(name="diskgraph-pvc-uid"),
            status=V1PersistentVolumeClaimStatus(phase="Bound"),
        ))

    mgr = _make_manager(core=core, apps=apps)
    with pytest.raises(ProvisionTimeoutError):
        asyncio.run(
            mgr.provision_pool_worker(
                pool_worker_uid="uid",
                timeout=0.1,
            ), )


def test_provision_pool_worker_pvc_bind_timeout_raises_specialised():
    """When the PVC stays Pending past the timeout (CSI capacity
    miss), surface PVCBindTimeoutError so the controller can route
    to a different recovery branch (resize / different instance type
    per design §8.3)."""
    from kubernetes.client.models import (
        V1PersistentVolumeClaim,
        V1PersistentVolumeClaimStatus,
    )
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    # No daemon pod yet; PVC stuck in Pending.
    core.list_namespaced_pod.return_value = V1PodList(items=[])
    core.read_namespaced_persistent_volume_claim.return_value = (
        V1PersistentVolumeClaim(
            metadata=V1ObjectMeta(name="diskgraph-pvc-uid"),
            status=V1PersistentVolumeClaimStatus(phase="Pending"),
        ))

    mgr = _make_manager(core=core, apps=apps)
    with pytest.raises(PVCBindTimeoutError, match="Pending"):
        asyncio.run(
            mgr.provision_pool_worker(
                pool_worker_uid="uid",
                timeout=0.1,
            ), )


def test_daemon_deployment_spec_has_required_properties():
    """The daemon Deployment must carry: Recreate strategy,
    `dedicated=diskgraph-<tenant>` toleration (matches kumo-delivery's
    nodepool-cache.libsonnet `dedicatedTaintValue`), hostname
    antiaffinity, the PVC volume, the constructor's command/args, AND
    must NOT carry a pod-level do-not-disrupt annotation (daemon-pod
    disruption protection is now a namespaced PodDisruptionBudget with
    maxUnavailable:0, not a node annotation)."""
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod(
            "p",
            labels={
                LABEL_APP: LABEL_APP_VALUE_DAEMON,
                LABEL_POOL_WORKER_UID: "uid",
            },
        ),
    ])

    mgr = _make_manager(core=core, apps=apps)
    asyncio.run(mgr.provision_pool_worker(pool_worker_uid="uid", timeout=5.0), )

    body = apps.create_namespaced_deployment.call_args.kwargs["body"]
    pod_spec = body.spec.template.spec
    # Recreate strategy: design §7.2 + §9 row 2.
    assert body.spec.strategy.type == "Recreate"
    assert pod_spec.termination_grace_period_seconds == 30
    # Pod template MUST NOT carry do-not-disrupt annotation — daemon-pod
    # disruption protection is now a namespaced PodDisruptionBudget
    # (maxUnavailable:0), not a node annotation.
    pod_annotations = body.spec.template.metadata.annotations or {}
    assert "karpenter.sh/do-not-disrupt" not in pod_annotations
    # Tolerates the cache-pool taint: kumo-delivery stamps
    # `dedicated=diskgraph-<tenant>` on cache NodePools, derived from
    # the manager's `tenant` ctor param.
    tolerations = pod_spec.tolerations
    assert any(t.key == "dedicated" and t.value == f"diskgraph-{TENANT}"
               and t.effect == "NoSchedule" for t in tolerations)
    # podAntiAffinity on hostname for at-most-one-daemon-per-node.
    aa = pod_spec.affinity.pod_anti_affinity
    term = aa.required_during_scheduling_ignored_during_execution[0]
    assert term.topology_key == "kubernetes.io/hostname"
    # No nodeAffinity: node sizing is handled by the cache-node-type
    # nodeSelector (cloud-neutral), not a Karpenter instance-local-nvme floor.
    assert pod_spec.affinity.node_affinity is None
    # Container picks up custom command/args from the constructor.
    container = pod_spec.containers[0]
    assert container.command == ["/usr/bin/daemon"]
    assert container.args == ["--mode=test"]
    # Daemon container has both NVMe and EBS scratch mounts.
    mount_paths = {m.mount_path: m.name for m in container.volume_mounts}
    assert mount_paths == {
        "/mnt/localssd": "ssd",
        "/mnt/ebs-scratch": "ebs-scratch",
    }
    # Pod has both volumes: NVMe PVC + EBS-backed emptyDir scratch.
    # The EBS volume itself is attached at the node level via the
    # cache EC2NodeClass blockDeviceMappings + userData; the daemon
    # pod only declares the emptyDir, kubelet places it on the
    # /var/lib/kubelet mount point that backs the EBS volume.
    vols_by_name = {v.name: v for v in pod_spec.volumes}
    pvc_vol = vols_by_name["ssd"]
    assert (pvc_vol.persistent_volume_claim.claim_name == "diskgraph-pvc-uid")
    ebs_vol = vols_by_name["ebs-scratch"]
    assert ebs_vol.empty_dir is not None
    assert ebs_vol.persistent_volume_claim is None
    assert ebs_vol.ephemeral is None


def test_provision_pool_worker_creates_pdb_max_unavailable_zero():
    """After the daemon Deployment is created, the manager creates a
    namespaced PodDisruptionBudget (maxUnavailable:0) selecting the
    daemon pod — the cloud-portable replacement for the old node
    disruption annotation. Disruption protection lives entirely
    in the tenant namespace now (no cluster-scoped node RBAC)."""
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    policy = MagicMock(spec=k8s_client.PolicyV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod(
            "p",
            labels={
                LABEL_APP: LABEL_APP_VALUE_DAEMON,
                LABEL_POOL_WORKER_UID: "uid",
            },
        ),
    ])

    mgr = _make_manager(core=core, apps=apps, policy=policy)
    asyncio.run(mgr.provision_pool_worker(pool_worker_uid="uid", timeout=5.0), )

    policy.create_namespaced_pod_disruption_budget.assert_called_once()
    body = (
        policy.create_namespaced_pod_disruption_budget.call_args.kwargs["body"])
    assert body.metadata.name == _pdb_name_for_uid("uid")
    assert body.metadata.name == "diskgraph-pdb-uid"
    assert body.spec.max_unavailable == 0
    assert body.spec.selector.match_labels == {
        LABEL_APP: LABEL_APP_VALUE_DAEMON,
        LABEL_POOL_WORKER_UID: "uid",
    }


def test_provision_pool_worker_pdb_already_exists_is_idempotent():
    """A 409 from create_namespaced_pod_disruption_budget (the PDB was
    already created by a prior provision attempt / rebind on the same
    uid) is swallowed — the reclaim provision must NOT raise."""
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    policy = MagicMock(spec=k8s_client.PolicyV1Api)
    policy.create_namespaced_pod_disruption_budget.side_effect = _api_exc(
        409, "already exists")
    core.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod(
            "p",
            labels={
                LABEL_APP: LABEL_APP_VALUE_DAEMON,
                LABEL_POOL_WORKER_UID: "uid",
            },
        ),
    ])

    mgr = _make_manager(core=core, apps=apps, policy=policy)
    # Explicit uid → reclaim path; the pre-existing PDB is recovery
    # signal, not an error.
    asyncio.run(mgr.provision_pool_worker(pool_worker_uid="uid", timeout=5.0), )


def test_provision_pool_worker_rollback_deletes_pdb():
    """Fresh-mint provision where the daemon never reaches Registered:
    the PDB was created (right after the Deployment), so the rollback
    path must best-effort delete it alongside the Deployment + PVC."""
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    policy = MagicMock(spec=k8s_client.PolicyV1Api)
    # No daemon pod ever shows up → _wait_for_daemon_registered times
    # out and raises. Fresh-mint (no explicit uid) → rollback runs.
    core.list_namespaced_pod.return_value = V1PodList(items=[])
    core.read_namespaced_persistent_volume_claim.side_effect = _api_exc(
        404, "not found")

    mgr = _make_manager(core=core, apps=apps, policy=policy)
    with pytest.raises(Exception):
        asyncio.run(mgr.provision_pool_worker(timeout=0.3))

    # PDB create happened (after Deployment create), so rollback issues
    # a best-effort delete.
    policy.create_namespaced_pod_disruption_budget.assert_called_once()
    policy.delete_namespaced_pod_disruption_budget.assert_called_once()
    # PVC + Deployment rollback still happen.
    assert core.delete_namespaced_persistent_volume_claim.called
    assert apps.delete_namespaced_deployment.called


# ---------------------------------------------------------------------------
# delete_pool_worker
# ---------------------------------------------------------------------------


def _empty_pool_worker_lookup_mocks() -> tuple:
    """Helpers for delete_pool_worker tests: returns Mocks pre-wired so
    ``_get_pool_worker_info`` returns ``None`` and PVC reads 404
    (avoids Mock truthiness pitfalls + lets PVC poll terminate fast).
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[])
    core.read_namespaced_persistent_volume_claim.side_effect = (_api_exc(
        404, "not found"))
    return core, apps


def test_delete_pool_worker_deletes_deployment_and_pvc():
    """delete_pool_worker patches the Deployment scale to 0, deletes
    the PVC, and finally deletes the Deployment object (design §5.3
    sequence)."""
    core, apps = _empty_pool_worker_lookup_mocks()

    mgr = _make_manager(core=core, apps=apps)
    asyncio.run(mgr.delete_pool_worker("uid"))

    apps.patch_namespaced_deployment_scale.assert_called_once()
    apps.delete_namespaced_deployment.assert_called_once()
    core.delete_namespaced_persistent_volume_claim.assert_called_once()
    scale_kwargs = (apps.patch_namespaced_deployment_scale.call_args.kwargs)
    assert scale_kwargs["name"] == "diskgraph-daemon-uid"
    assert scale_kwargs["body"] == {"spec": {"replicas": 0}}
    dep_kwargs = apps.delete_namespaced_deployment.call_args.kwargs
    assert dep_kwargs["name"] == "diskgraph-daemon-uid"
    pvc_kwargs = (
        core.delete_namespaced_persistent_volume_claim.call_args.kwargs)
    assert pvc_kwargs["name"] == "diskgraph-pvc-uid"


def test_delete_pool_worker_idempotent_when_deployment_already_gone():
    core, apps = _empty_pool_worker_lookup_mocks()
    apps.delete_namespaced_deployment.side_effect = _api_exc(
        404,
        "not found",
    )
    apps.patch_namespaced_deployment_scale.side_effect = _api_exc(
        404,
        "not found",
    )

    mgr = _make_manager(core=core, apps=apps)
    # Should NOT raise — already-gone is success.
    asyncio.run(mgr.delete_pool_worker("uid"))


def test_delete_pool_worker_step_2_fences_activity_pod_deletion():
    """Design §5.3 specifies a force-delete (grace=0) for activity
    pods and a foreground fence before deleting their shared PVC."""
    core, apps = _empty_pool_worker_lookup_mocks()
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    batch.list_namespaced_job.return_value = V1JobList(items=[])

    mgr = _make_manager(core=core, apps=apps, batch=batch)
    asyncio.run(mgr.delete_pool_worker("uid"))

    batch.delete_collection_namespaced_job.assert_called_once()
    kwargs = batch.delete_collection_namespaced_job.call_args.kwargs
    assert kwargs["grace_period_seconds"] == 0
    assert kwargs["propagation_policy"] == "Foreground"
    batch.list_namespaced_job.assert_called_once()


def test_delete_pool_worker_propagates_non_404_from_activity_cleanup():
    """A 5xx / 403 during step 2 must NOT be silently swallowed —
    that would leak orphan activity Jobs while the caller's
    teardown reports success."""
    core, apps = _empty_pool_worker_lookup_mocks()
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    batch.delete_collection_namespaced_job.side_effect = _api_exc(
        500,
        "internal server error",
    )

    mgr = _make_manager(core=core, apps=apps, batch=batch)
    with pytest.raises(ApiServerTransientError):
        asyncio.run(mgr.delete_pool_worker("uid"))


def test_delete_pool_worker_propagates_404_from_activity_cleanup():
    """An empty collection succeeds; HTTP 404 instead signals an API or
    namespace failure and must propagate through the manager taxonomy."""
    core, apps = _empty_pool_worker_lookup_mocks()
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    batch.delete_collection_namespaced_job.side_effect = _api_exc(
        404,
        "not found",
    )

    mgr = _make_manager(core=core, apps=apps, batch=batch)
    with pytest.raises(UnclassifiedApiError):
        asyncio.run(mgr.delete_pool_worker("uid"))


def test_try_delete_pvc_propagates_non_404_on_initial_delete():
    """The initial delete call must propagate non-404 errors (5xx,
    403, etc.) — silently dropping them would leave an orphan PVC
    while the teardown caller reports success."""
    core, apps = _empty_pool_worker_lookup_mocks()
    core.delete_namespaced_persistent_volume_claim.side_effect = (_api_exc(
        503, "service unavailable"))

    mgr = _make_manager(core=core, apps=apps)
    with pytest.raises(ApiException):
        asyncio.run(mgr.delete_pool_worker("uid"))


# ---------------------------------------------------------------------------
# get_status / list_pool_workers / _get_pool_worker_info
# ---------------------------------------------------------------------------


def test_get_status_raises_pool_worker_not_found_on_empty():
    core = MagicMock(spec=k8s_client.CoreV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[])

    mgr = _make_manager(core=core)
    with pytest.raises(PoolWorkerNotFoundError, match="uid"):
        asyncio.run(mgr.get_status("uid"))


def test_get_status_picks_newest_pod_during_rollover():
    """During Recreate rollover or rebind, ``list_namespaced_pod`` may
    transiently return 2 pods for the same uid. The newer one is the
    live state — verify _get_pool_worker_info picks it.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    older = _make_pod(
        "p-old",
        node_name="node-old",
        labels={
            LABEL_APP: LABEL_APP_VALUE_DAEMON,
            LABEL_POOL_WORKER_UID: "uid",
        },
        creation_timestamp=datetime.datetime(2026, 1, 1, 0, 0, 0),
    )
    newer = _make_pod(
        "p-new",
        node_name="node-new",
        labels={
            LABEL_APP: LABEL_APP_VALUE_DAEMON,
            LABEL_POOL_WORKER_UID: "uid",
        },
        creation_timestamp=datetime.datetime(2026, 1, 2, 0, 0, 0),
    )
    # Order reversed to make sure picker doesn't just take items[0].
    core.list_namespaced_pod.return_value = V1PodList(items=[older, newer])

    mgr = _make_manager(core=core)
    info = asyncio.run(mgr.get_status("uid"))
    assert info.daemon_pod_name == "p-new"
    assert info.node_name == "node-new"


def test_list_pool_workers_filters_to_daemons_with_uid_label():
    core = MagicMock(spec=k8s_client.CoreV1Api)
    core.list_namespaced_pod.return_value = V1PodList(
        items=[
            _make_pod(
                "p1",
                labels={
                    LABEL_APP: LABEL_APP_VALUE_DAEMON,
                    LABEL_POOL_WORKER_UID: "uid1",
                },
            ),
            _make_pod(
                "p2",
                labels={
                    LABEL_APP: LABEL_APP_VALUE_DAEMON,
                    LABEL_POOL_WORKER_UID: "uid2",
                },
            ),
            # No uid label → should be skipped.
            _make_pod("p3", labels={LABEL_APP: LABEL_APP_VALUE_DAEMON}),
        ], )

    mgr = _make_manager(core=core)
    workers = asyncio.run(mgr.list_pool_workers())

    assert {w.pool_worker_uid for w in workers} == {"uid1", "uid2"}
    # Selector includes the daemon app label and tenant.
    selector = (core.list_namespaced_pod.call_args.kwargs["label_selector"])
    assert f"{LABEL_APP}={LABEL_APP_VALUE_DAEMON}" in selector
    assert f"{LABEL_TENANT}={TENANT}" in selector


def _current_registered_cache_worker(mgr: K8sPoolWorkerManager):
    worker = PoolWorkerInfo(
        pool_worker_uid="uid",
        pvc_name="diskgraph-pvc-uid",
        daemon_pod_name="daemon-uid",
        node_name="node-1",
        status="Registered",
    )
    scheduling = mgr.cache_canary_scheduling(800, 1)
    pod = _make_pod(
        "daemon-uid",
        labels={
            LABEL_APP: LABEL_APP_VALUE_DAEMON,
            LABEL_TENANT: TENANT,
            LABEL_POOL_WORKER_UID: "uid",
        },
    )
    assert pod.spec is not None
    pod.spec.node_selector = scheduling.node_selector
    pod.spec.tolerations = scheduling.tolerations
    pod.spec.volumes = [
        V1Volume(
            name="ssd",
            persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
                claim_name=worker.pvc_name),
        )
    ]
    pvc = V1PersistentVolumeClaim(
        metadata=V1ObjectMeta(name=worker.pvc_name, namespace=NAMESPACE),
        spec=V1PersistentVolumeClaimSpec(
            access_modes=["ReadWriteOnce"],
            storage_class_name="topolvm-ssd",
            volume_name="pv-uid",
            resources=V1ResourceRequirements(requests={"storage": "800Gi"}),
        ),
        status=V1PersistentVolumeClaimStatus(phase="Bound"),
    )
    return worker, pod, pvc


def test_registered_worker_current_cache_tier_requires_live_envelope():
    core = MagicMock(spec=k8s_client.CoreV1Api)
    mgr = _make_manager(core=core)
    worker, pod, pvc = _current_registered_cache_worker(mgr)
    core.read_namespaced_pod.return_value = pod
    core.read_namespaced_persistent_volume_claim.return_value = pvc

    assert asyncio.run(
        mgr.registered_worker_matches_current_cache_tier(
            worker, PvcSpec(size_gib=800, gpu_count=1)))


@pytest.mark.parametrize(
    "stale_dimension",
    [
        "portable-rebind-selector",
        "non-registered-pod",
        "old-taint",
        "missing-pvc-volume",
        "old-storage-class",
        "old-size",
        "old-access-mode",
        "pending-pvc",
        "unbound-pvc",
    ],
)
def test_registered_worker_old_envelope_does_not_prevalidate_tier(
        stale_dimension):
    core = MagicMock(spec=k8s_client.CoreV1Api)
    portable_rebind = stale_dimension == "portable-rebind-selector"
    storage_class = "managed-csi" if portable_rebind else "topolvm-ssd"
    mgr = _make_manager(core=core, storage_class=storage_class)
    worker, pod, pvc = _current_registered_cache_worker(mgr)
    spec = PvcSpec(size_gib=800, gpu_count=1)
    assert pod.spec is not None
    assert pod.status is not None
    assert pvc.spec is not None
    assert pvc.status is not None
    if portable_rebind:
        worker.node_name = pod.spec.node_name = "cache-800-gpu1"
        pod.spec.node_selector = {
            DISKGRAPH_POOL_TYPE_LABEL: POOL_TYPE_CACHE,
        }
        pvc.spec.storage_class_name = storage_class
        pvc.spec.resources.requests["storage"] = "3400Gi"
        spec = PvcSpec(size_gib=3400, gpu_count=4)
    elif stale_dimension == "non-registered-pod":
        pod.status.container_statuses[0].ready = False
    elif stale_dimension == "old-taint":
        pod.spec.tolerations = [
            V1Toleration(
                key="dedicated",
                operator="Equal",
                value="diskgraph-old-tenant",
                effect="NoSchedule",
            )
        ]
    elif stale_dimension == "missing-pvc-volume":
        pod.spec.volumes = []
    elif stale_dimension == "old-storage-class":
        pvc.spec.storage_class_name = "topolvm-old"
    elif stale_dimension == "old-size":
        pvc.spec.resources.requests["storage"] = "400Gi"
    elif stale_dimension == "old-access-mode":
        pvc.spec.access_modes = ["ReadWriteMany"]
    elif stale_dimension == "pending-pvc":
        pvc.status.phase = "Pending"
    else:
        pvc.spec.volume_name = None
    core.read_namespaced_pod.return_value = pod
    core.read_namespaced_persistent_volume_claim.return_value = pvc

    assert not asyncio.run(
        mgr.registered_worker_matches_current_cache_tier(worker, spec))


def test_registered_worker_missing_live_object_does_not_prevalidate_tier():
    core = MagicMock(spec=k8s_client.CoreV1Api)
    mgr = _make_manager(core=core)
    worker, _pod, _pvc = _current_registered_cache_worker(mgr)
    core.read_namespaced_pod.side_effect = _api_exc(404)

    assert not asyncio.run(
        mgr.registered_worker_matches_current_cache_tier(
            worker, PvcSpec(size_gib=800, gpu_count=1)))


def test_registered_worker_unexpected_api_error_propagates():
    core = MagicMock(spec=k8s_client.CoreV1Api)
    mgr = _make_manager(core=core)
    worker, _pod, _pvc = _current_registered_cache_worker(mgr)
    core.read_namespaced_pod.side_effect = _api_exc(403)

    with pytest.raises(K8sPermissionError):
        asyncio.run(
            mgr.registered_worker_matches_current_cache_tier(
                worker, PvcSpec(size_gib=800, gpu_count=1)))


# ---------------------------------------------------------------------------
# provision_activity_pod (cache pool, Job-wrapped)
# ---------------------------------------------------------------------------


def _registered_daemon_for(uid: str) -> V1PodList:
    return V1PodList(items=[
        _make_pod(
            f"daemon-{uid}",
            labels={
                LABEL_APP: LABEL_APP_VALUE_DAEMON,
                LABEL_POOL_WORKER_UID: uid,
            },
        ),
    ])


def test_provision_activity_pod_raises_not_found_when_uid_missing():
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[])

    mgr = _make_manager(core=core, batch=batch)
    with pytest.raises(PoolWorkerNotFoundError):
        asyncio.run(
            mgr.provision_activity_pod(
                pool_worker_uid="ghost",
                job_id="j1",
                snapshot_id="s1",
                activity="train",
                image="kumo/activity:test",
            ))
    # Job MUST NOT be created if the pool worker isn't found.
    assert not batch.create_namespaced_job.called


def test_provision_activity_pod_raises_not_ready_when_daemon_pending():
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    # Daemon is Pending — not yet Registered.
    core.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod(
            "daemon-uid",
            phase="Pending",
            container_ready=False,
            labels={
                LABEL_APP: LABEL_APP_VALUE_DAEMON,
                LABEL_POOL_WORKER_UID: "uid",
            },
        ),
    ])

    mgr = _make_manager(core=core, batch=batch)
    with pytest.raises(PoolWorkerNotReadyError, match="Pending"):
        asyncio.run(
            mgr.provision_activity_pod(
                pool_worker_uid="uid",
                job_id="j1",
                snapshot_id="s1",
                activity="train",
                image="kumo/activity:test",
            ))
    assert not batch.create_namespaced_job.called


def test_provision_activity_pod_translates_env_dict_and_stamps_labels():
    """The manager is agnostic to the daemon image's env contract:
    whatever ``env`` dict the caller passes is faithfully translated
    to ``V1EnvVar`` entries on the container, and labels are stamped
    from the dedicated method params (snapshot_id / pool_worker_uid /
    job_id / activity / pool-type).
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)

    daemon_pods = _registered_daemon_for("uid")
    activity_pod = _make_pod(
        "act-pod-1",
        labels={
            LABEL_APP: LABEL_APP_VALUE_ACTIVITY,
            LABEL_POOL_WORKER_UID: "uid",
            LABEL_JOB_ID_HASH: _hash_job_id("j7"),
        },
        annotations={ANNOTATION_JOB_ID: "j7"},
    )
    core.list_namespaced_pod.side_effect = [
        daemon_pods,
        V1PodList(items=[activity_pod]),
    ]

    # Caller composes the env dict — manager doesn't peek at keys.
    caller_env = {
        "POOL_WORKER_UID": "uid",
        "POOL_TASK_QUEUE": "diskgraph-activity-tq-j7",
        "SNAPSHOT_ID": "snap-7",
        "DISK_ROOT": "/mnt/localssd",
    }

    mgr = _make_manager(core=core, batch=batch,
                        pod_template_labels=POD_TEMPLATE_LABELS)
    asyncio.run(
        mgr.provision_activity_pod(
            pool_worker_uid="uid",
            job_id="j7",
            snapshot_id="snap-7",
            activity="predict",
            image="kumo/activity:v2",
            env=caller_env,
        ))
    info = asyncio.run(
        mgr.wait_for_activity_job_pod_running(
            job_name=mgr.activity_job_name("j7", "durable"),
            timeout=5.0,
        ))

    assert batch.create_namespaced_job.called
    job_body = batch.create_namespaced_job.call_args.kwargs["body"]
    assert job_body.spec.backoff_limit == 2
    template_spec = job_body.spec.template.spec
    assert template_spec.security_context.seccomp_profile.type == "RuntimeDefault"
    assert template_spec.restart_policy == "Never"
    assert template_spec.termination_grace_period_seconds == 30
    assert template_spec.security_context.fs_group == 999
    assert template_spec.security_context.fs_group_change_policy == "OnRootMismatch"
    container = template_spec.containers[0]
    assert container.security_context.privileged is False
    assert container.security_context.allow_privilege_escalation is False
    assert container.security_context.run_as_non_root is True
    assert container.security_context.capabilities.drop == ["ALL"]
    assert container.image == "kumo/activity:v2"

    # Caller env stays intact; the manager adds only the downward-API pod name
    # used to identify this exact Temporal poller.
    actual_env = {
        e.name: e.value
        for e in container.env if e.value_from is None
    }
    assert actual_env == caller_env
    identity_env = next(e for e in container.env
                        if e.name == ACTIVITY_WORKER_POD_NAME_ENV)
    assert identity_env.value_from.field_ref.field_path == "metadata.name"

    # Azure Disk PV affinity is zonal, so the activity also pins to this
    # worker's daemon host to avoid a cross-node RWO Multi-Attach.
    affinity_term = (template_spec.affinity.pod_affinity.
                     required_during_scheduling_ignored_during_execution[0])
    assert affinity_term.topology_key == "kubernetes.io/hostname"
    assert affinity_term.label_selector.match_labels == {
        LABEL_APP: LABEL_APP_VALUE_DAEMON,
        LABEL_POOL_WORKER_UID: "uid",
    }

    # PVC mount remains the shared data path.
    pvc_vol = template_spec.volumes[0]
    assert (pvc_vol.persistent_volume_claim.claim_name == "diskgraph-pvc-uid")
    # Labels still derived from method params (used for K8s
    # selector queries — see delete_activity_pods). Both job-id and
    # snapshot-id are stored hashed; raw values live on annotations.
    labels = job_body.metadata.labels
    assert labels[LABEL_APP] == LABEL_APP_VALUE_ACTIVITY
    assert labels[LABEL_POOL_WORKER_UID] == "uid"
    assert labels[LABEL_JOB_ID_HASH] == _hash_job_id("j7")
    assert labels[LABEL_SNAPSHOT_ID_HASH] == _hash_snapshot_id("snap-7")
    assert labels[LABEL_ACTIVITY] == "predict"
    assert labels[DISKGRAPH_POOL_TYPE_LABEL] == POOL_TYPE_CACHE
    pod_labels = job_body.spec.template.metadata.labels
    assert pod_labels["azure.workload.identity/use"] == "true"
    assert pod_labels[LABEL_APP] == LABEL_APP_VALUE_ACTIVITY
    # Raw values preserved on annotations for human readability,
    # propagated to the pod template so spawned pods inherit them.
    annotations = job_body.metadata.annotations
    assert annotations[ANNOTATION_JOB_ID] == "j7"
    assert annotations[ANNOTATION_SNAPSHOT_ID] == "snap-7"
    pod_annotations = job_body.spec.template.metadata.annotations
    assert pod_annotations[ANNOTATION_JOB_ID] == "j7"
    assert pod_annotations[ANNOTATION_SNAPSHOT_ID] == "snap-7"
    assert info.pool_worker_uid == "uid"
    assert info.job_id == "j7"


def test_provision_eph_activity_pod_uses_emptydir_and_eph_storage():
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    activity_pod = _make_pod(
        "eph-act-1",
        labels={
            LABEL_APP: LABEL_APP_VALUE_ACTIVITY_EPH,
            LABEL_JOB_ID_HASH: "j-eph",
        },
    )
    core.list_namespaced_pod.return_value = V1PodList(items=[activity_pod], )

    mgr = _make_manager(core=core, batch=batch,
                        pod_template_labels=POD_TEMPLATE_LABELS)
    asyncio.run(
        mgr.provision_ephemeral_activity_pod(
            job_id="j-eph",
            snapshot_id="snap-eph",
            activity="ingest",
            image="kumo/activity:eph",
            ephemeral_storage_bytes=10 * (1 << 30),
        ))

    job_body = batch.create_namespaced_job.call_args.kwargs["body"]
    template_spec = job_body.spec.template.spec
    assert template_spec.security_context.fs_group is None
    assert template_spec.security_context.seccomp_profile.type == "RuntimeDefault"
    assert template_spec.termination_grace_period_seconds == 30
    # Targets ephemeral capacity; the tenant-specific toleration gates access.
    assert template_spec.node_selector == {
        DISKGRAPH_POOL_TYPE_LABEL: POOL_TYPE_EPHEMERAL
    }
    # Toleration matches kumo-delivery's nodepool-eph.libsonnet
    # `dedicated=diskgraph-eph-<tenant>` taint.
    tol = template_spec.tolerations[0]
    assert tol.key == "dedicated"
    assert tol.value == f"diskgraph-eph-{TENANT}"
    # Volume is emptyDir, NOT persistentVolumeClaim.
    vol = template_spec.volumes[0]
    assert vol.empty_dir is not None
    assert vol.persistent_volume_claim is None
    # ephemeral-storage request is the scheduling floor only; deliberately NO
    # limit and NO emptyDir sizeLimit, so the pod may use the dedicated eph
    # node's full instance-store instead of being evicted at the request
    # while the node sits empty (backlog #18).
    container = template_spec.containers[0]
    assert container.security_context.privileged is False
    assert container.security_context.allow_privilege_escalation is False
    assert container.security_context.run_as_non_root is True
    assert container.security_context.capabilities.drop == ["ALL"]
    assert container.resources.requests["ephemeral-storage"] == str(10 *
                                                                    (1 << 30))
    assert "ephemeral-storage" not in (container.resources.limits or {})
    assert vol.empty_dir.size_limit is None
    # No pool-worker-uid label on eph activity Jobs.
    assert LABEL_POOL_WORKER_UID not in job_body.metadata.labels
    assert (job_body.spec.template.metadata.
            labels["azure.workload.identity/use"] == "true")


# ---------------------------------------------------------------------------
# delete_activity_pods / list_activity_pods (label-selector tests)
# ---------------------------------------------------------------------------


def test_delete_activity_pods_deletes_jobs_and_matching_orphans():
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    core = MagicMock(spec=k8s_client.CoreV1Api)
    core.list_namespaced_pod.return_value = V1PodList(
        items=[_make_pod('orphan-activity')])
    mgr = _make_manager(batch=batch, core=core)
    asyncio.run(mgr.delete_activity_pods())

    batch.delete_collection_namespaced_job.assert_called_once()
    kwargs = batch.delete_collection_namespaced_job.call_args.kwargs
    assert kwargs["namespace"] == NAMESPACE
    # Background propagation lets Job deletion return while matching pods are
    # deleted explicitly below.
    assert kwargs["propagation_policy"] == "Background"
    # Default selector includes both cache + eph activity app values.
    selector = kwargs["label_selector"]
    assert LABEL_APP_VALUE_ACTIVITY in selector
    assert LABEL_APP_VALUE_ACTIVITY_EPH in selector
    assert f"{LABEL_TENANT}={TENANT}" in selector
    core.list_namespaced_pod.assert_called_once_with(namespace=NAMESPACE,
                                                     label_selector=selector)
    core.delete_namespaced_pod.assert_called_once_with(name='orphan-activity',
                                                       namespace=NAMESPACE)


def test_delete_activity_pods_filters_and_waits_for_deletion(monkeypatch):
    monkeypatch.setattr('pool.worker_manager.k8s._POLL_INTERVAL_SECS', 0)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    batch.list_namespaced_job.side_effect = [
        MagicMock(items=[MagicMock()]),
        MagicMock(items=[])
    ]
    core = MagicMock(spec=k8s_client.CoreV1Api)
    core.list_namespaced_pod.side_effect = [
        V1PodList(items=[_make_pod('orphan-activity')]),
        V1PodList(items=[_make_pod('orphan-activity')]),
        V1PodList(items=[]),
    ]
    mgr = _make_manager(batch=batch, core=core)
    asyncio.run(
        mgr.delete_activity_pods(
            pool_worker_uid="uid-1",
            pool_type="ephemeral",
            job_id="job-7",
            grace_period_seconds=0,
            wait_for_deletion=True,
            deletion_timeout=5,
        ))

    kwargs = batch.delete_collection_namespaced_job.call_args.kwargs
    selector = kwargs["label_selector"]
    assert f"{LABEL_POOL_WORKER_UID}=uid-1" in selector
    assert f"{DISKGRAPH_POOL_TYPE_LABEL}=ephemeral" in selector
    assert (f"{LABEL_JOB_ID_HASH}={_hash_job_id('job-7')}" in selector)
    assert "job-id-hash=job-7" not in selector
    assert kwargs['propagation_policy'] == 'Foreground'
    assert kwargs['grace_period_seconds'] == 0
    assert batch.list_namespaced_job.call_count == 2
    assert core.list_namespaced_pod.call_count == 3
    core.delete_namespaced_pod.assert_called_once_with(name='orphan-activity',
                                                       namespace=NAMESPACE,
                                                       grace_period_seconds=0)


def test_list_activity_pods_filter_by_uid():
    core = MagicMock(spec=k8s_client.CoreV1Api)
    core.list_namespaced_pod.return_value = V1PodList(
        items=[
            _make_pod(
                "act-1",
                labels={
                    LABEL_APP: LABEL_APP_VALUE_ACTIVITY,
                    LABEL_POOL_WORKER_UID: "uid-1",
                    LABEL_JOB_ID_HASH: "j",
                },
            ),
        ], )
    mgr = _make_manager(core=core)
    pods = asyncio.run(mgr.list_activity_pods(pool_worker_uid="uid-1"))

    assert len(pods) == 1
    assert pods[0].pool_worker_uid == "uid-1"
    selector = (core.list_namespaced_pod.call_args.kwargs["label_selector"])
    assert f"{LABEL_POOL_WORKER_UID}=uid-1" in selector


def test_list_activity_pods_no_filter_covers_both_cache_and_eph():
    """Default selector must use ``app in (cache, eph)`` so callers
    listing without filters see both pool variants — design §6.1."""
    core = MagicMock(spec=k8s_client.CoreV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[])
    mgr = _make_manager(core=core)
    asyncio.run(mgr.list_activity_pods())

    selector = (core.list_namespaced_pod.call_args.kwargs["label_selector"])
    assert (f"{LABEL_APP} in "
            f"({LABEL_APP_VALUE_ACTIVITY},{LABEL_APP_VALUE_ACTIVITY_EPH})"
            ) in selector


# ---------------------------------------------------------------------------
# Coverage for §4 contract notes: ordering, idempotency, parameter passthrough
# ---------------------------------------------------------------------------


def test_delete_pool_worker_follows_design_5_3_sequence():
    """§5.3 teardown sequence: delete the PodDisruptionBudget FIRST, then
    scale Deployment to 0 → delete PVC → delete Deployment object. The PDB
    (the eviction guard) is removed up front so a later wedge (stuck pod /
    wedged PVC finalizer that aborts a subsequent step) can't orphan the
    ``maxUnavailable:0`` PDB and pin the node from consolidation forever; the
    still-running daemon pod keeps the node alive until it is drained. Among
    the remaining steps, scale-to-0 MUST precede PVC delete (PVC is in use by
    the daemon pod), and PVC delete MUST precede the final Deployment delete
    (so ``delete_pool_worker`` returns only after the volume is released).
    """
    core, apps = _empty_pool_worker_lookup_mocks()
    policy = MagicMock(spec=k8s_client.PolicyV1Api)
    sequence = []
    apps.patch_namespaced_deployment_scale.side_effect = (
        lambda *a, **kw: sequence.append("scale-to-0"))
    core.delete_namespaced_persistent_volume_claim.side_effect = (
        lambda *a, **kw: sequence.append("pvc"))
    apps.delete_namespaced_deployment.side_effect = (
        lambda *a, **kw: sequence.append("deployment"))
    policy.delete_namespaced_pod_disruption_budget.side_effect = (
        lambda *a, **kw: sequence.append("pdb"))

    mgr = _make_manager(core=core, apps=apps, policy=policy)
    asyncio.run(mgr.delete_pool_worker("uid"))

    assert sequence == [
        "pdb", "scale-to-0", "pvc", "deployment"
    ], (f"expected pdb → scale-to-0 → pvc → deployment; got {sequence}")
    # PDB is deleted by its uid-derived name so Karpenter can reclaim
    # the now-empty node.
    pdb_kwargs = policy.delete_namespaced_pod_disruption_budget.call_args.kwargs
    assert pdb_kwargs["name"] == "diskgraph-pdb-uid"


def test_delete_pool_worker_propagates_grace_period_seconds():
    """Custom ``grace_period_seconds`` (used for SIGKILL on DEAD
    nodes — design §10) must reach the final K8s deployment delete
    call."""
    core, apps = _empty_pool_worker_lookup_mocks()
    mgr = _make_manager(core=core, apps=apps)

    asyncio.run(mgr.delete_pool_worker("uid", grace_period_seconds=0))

    kwargs = apps.delete_namespaced_deployment.call_args.kwargs
    assert kwargs["grace_period_seconds"] == 0


def test_delete_pool_worker_deletes_pdb():
    """Teardown deletes the daemon's PodDisruptionBudget (by its
    uid-derived name) so Karpenter consolidation can reclaim the
    now-empty node — the namespaced replacement for the old node
    disruption-annotation removal."""
    core, apps = _empty_pool_worker_lookup_mocks()
    policy = MagicMock(spec=k8s_client.PolicyV1Api)

    mgr = _make_manager(core=core, apps=apps, policy=policy)
    asyncio.run(mgr.delete_pool_worker("uid"))

    policy.delete_namespaced_pod_disruption_budget.assert_called_once()
    kwargs = policy.delete_namespaced_pod_disruption_budget.call_args.kwargs
    assert kwargs["name"] == "diskgraph-pdb-uid"


def test_delete_pool_worker_pdb_already_gone_is_idempotent():
    """A 404 from the PDB delete (it was never created, or already
    cleaned up by a prior teardown) must NOT raise — teardown is
    idempotent end-to-end."""
    core, apps = _empty_pool_worker_lookup_mocks()
    policy = MagicMock(spec=k8s_client.PolicyV1Api)
    policy.delete_namespaced_pod_disruption_budget.side_effect = _api_exc(
        404, "not found")

    mgr = _make_manager(core=core, apps=apps, policy=policy)
    # Should NOT raise.
    asyncio.run(mgr.delete_pool_worker("uid"))


def test_provision_pool_worker_with_explicit_uid_uses_it_in_resource_names():
    """The rebind path (§9 row 3) round-trips the same
    ``pool_worker_uid`` through delete → provision. Verify the explicit
    uid appears in PVC + Deployment names so the rebind reuses
    predictable resource names.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    policy = MagicMock(spec=k8s_client.PolicyV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod(
            "p", labels={
                LABEL_APP: LABEL_APP_VALUE_DAEMON,
                LABEL_POOL_WORKER_UID: "rebind-uid",
            }),
    ])

    mgr = _make_manager(core=core, apps=apps, policy=policy)
    info = asyncio.run(
        mgr.provision_pool_worker(
            pool_worker_uid="rebind-uid",
            timeout=5.0,
        ))

    assert info.pool_worker_uid == "rebind-uid"
    pvc_body = (
        core.create_namespaced_persistent_volume_claim.call_args.kwargs["body"])
    assert pvc_body.metadata.name == "diskgraph-pvc-rebind-uid"
    dep_body = apps.create_namespaced_deployment.call_args.kwargs["body"]
    assert dep_body.metadata.name == "diskgraph-daemon-rebind-uid"
    # The PDB is named from the same uid so the rebind protects the
    # daemon under a predictable, idempotent name.
    pdb_body = (
        policy.create_namespaced_pod_disruption_budget.call_args.kwargs["body"])
    assert pdb_body.metadata.name == "diskgraph-pdb-rebind-uid"


@pytest.mark.parametrize(
    "phase,container_ready",
    [
        ("Pending", False),
        ("Running", False),  # Pod Running but container not Ready yet
        ("Failed", False),
    ],
)
def test_provision_activity_pod_raises_not_ready_for_non_registered_phases(
    phase,
    container_ready,
):
    """§4 contract: ``provision_activity_pod requires READY pool worker``.
    Any non-Registered status must fast-fail with PoolWorkerNotReadyError
    — not just ``Pending``.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod(
            "daemon-uid",
            phase=phase,
            container_ready=container_ready,
            labels={
                LABEL_APP: LABEL_APP_VALUE_DAEMON,
                LABEL_POOL_WORKER_UID: "uid",
            },
        ),
    ])

    mgr = _make_manager(core=core, batch=batch)
    with pytest.raises(PoolWorkerNotReadyError):
        asyncio.run(
            mgr.provision_activity_pod(
                pool_worker_uid="uid",
                job_id="j",
                snapshot_id="s",
                activity="train",
                image="kumo/activity:test",
            ))
    assert not batch.create_namespaced_job.called


def test_provision_eph_activity_pod_translates_env_dict():
    """Eph variant: caller-supplied ``env`` dict is faithfully
    translated to ``V1EnvVar`` entries. Same caller-owns-the-contract
    pattern as the cache pool's provision_activity_pod.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod(
            "eph",
            labels={
                LABEL_APP: LABEL_APP_VALUE_ACTIVITY_EPH,
                LABEL_JOB_ID_HASH: "j-eph",
            },
        ),
    ])
    caller_env = {
        "POOL_TASK_QUEUE": "diskgraph-activity-tq-j-eph",
        "SNAPSHOT_ID": "snap-eph",
        "DISK_ROOT": "/mnt/localssd",
    }
    mgr = _make_manager(core=core, batch=batch)
    asyncio.run(
        mgr.provision_ephemeral_activity_pod(
            job_id="j-eph",
            snapshot_id="snap-eph",
            activity="ingest",
            image="kumo/activity:eph",
            ephemeral_storage_bytes=10 * (1 << 30),
            env=caller_env,
        ))
    body = batch.create_namespaced_job.call_args.kwargs["body"]
    container = body.spec.template.spec.containers[0]
    actual_env = {
        e.name: e.value
        for e in container.env if e.value_from is None
    }
    assert actual_env == caller_env
    identity_env = next(e for e in container.env
                        if e.name == ACTIVITY_WORKER_POD_NAME_ENV)
    assert identity_env.value_from.field_ref.field_path == "metadata.name"


def test_create_pvc_translates_quota_error_to_typed_exception():
    """If the K8s API rejects PVC creation with a quota error,
    ``provision_pool_worker`` must surface ``QuotaExceededError`` (not
    the bare ApiException) so callers can route to backoff retry.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    core.create_namespaced_persistent_volume_claim.side_effect = (_api_exc(
        403, "exceeded quota: storage"))

    mgr = _make_manager(core=core, apps=apps)
    with pytest.raises(QuotaExceededError):
        asyncio.run(
            mgr.provision_pool_worker(
                pool_worker_uid="uid",
                timeout=2.0,
            ))
    # Deployment never gets created if PVC creation failed.
    assert not apps.create_namespaced_deployment.called


def test_translate_api_error_chains_via_from_exc():
    """Sanity: when our typed errors wrap an ApiException, callers
    can still get at the original via ``__cause__`` (set by
    ``raise X from exc``).
    """
    api_exc = _api_exc(403, "exceeded quota")
    typed = _translate_api_error(api_exc)
    # _translate_api_error itself returns the typed instance; the
    # ``from exc`` happens at the raise site (e.g., _create_pvc_sync).
    # Verify the caller-side chain by re-raising:
    try:
        raise typed from api_exc
    except QuotaExceededError as e:
        assert e.__cause__ is api_exc


def test_storage_size_from_pvc_spec_formats_gib_string():
    """_storage_size_from_pvc_spec emits the exact tier GiB as a K8s
    quantity string — the controller picks the tier, no headroom applied here.
    """
    mgr = _make_manager()
    assert mgr._storage_size_from_pvc_spec(PvcSpec(size_gib=1750)) == "1750Gi"


def test_provision_then_delete_then_provision_same_uid_smoke():
    """Rebind-shape round-trip (§9 row 3): the API must support
    provision(uid=X) → delete(uid=X) → provision(uid=X) without
    error. End-to-end on a real cluster is C02; here we just confirm
    the client-side calls succeed in sequence.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    registered_pod = _make_pod(
        "p", labels={
            LABEL_APP: LABEL_APP_VALUE_DAEMON,
            LABEL_POOL_WORKER_UID: "rebind-uid",
        })
    # The teardown's `_wait_for_daemon_pod_gone` polls list-pods
    # until empty — feed it an empty result during the delete window
    # then go back to a Registered pod for the second provision.
    core.list_namespaced_pod.side_effect = [
        V1PodList(items=[registered_pod]),  # 1st provision wait
        V1PodList(items=[]),  # delete: no activity pod
        V1PodList(items=[]),  # delete: activity fence clear
        V1PodList(items=[]),  # delete: daemon pod gone
        V1PodList(items=[registered_pod]),  # 2nd provision wait
    ]
    # PVC reads always 404 — short-circuits the _try_delete_pvc poll.
    core.read_namespaced_persistent_volume_claim.side_effect = _api_exc(
        404,
        "not found",
    )

    mgr = _make_manager(core=core, apps=apps)

    asyncio.run(
        mgr.provision_pool_worker(
            pool_worker_uid="rebind-uid",
            timeout=5.0,
        ), )
    asyncio.run(mgr.delete_pool_worker("rebind-uid"))
    asyncio.run(
        mgr.provision_pool_worker(
            pool_worker_uid="rebind-uid",
            timeout=5.0,
        ), )

    # Two PVC creates + two Deployment creates over the round-trip.
    assert (core.create_namespaced_persistent_volume_claim.call_count == 2)
    assert apps.create_namespaced_deployment.call_count == 2
    # Both reference the same PVC name (predictable rebind).
    pvc_names = [
        c.kwargs["body"].metadata.name
        for c in core.create_namespaced_persistent_volume_claim.call_args_list
    ]
    assert pvc_names == [
        "diskgraph-pvc-rebind-uid",
        "diskgraph-pvc-rebind-uid",
    ]


# ---------------------------------------------------------------------------
# Regression coverage for pod lifecycle fixes (PR 28396)
# ---------------------------------------------------------------------------


def test_derive_pod_status_terminating_takes_precedence_over_succeeded():
    """A pod that finished AND is being deleted (deletion_timestamp set,
    phase=Succeeded) must surface as 'Terminating', not 'Succeeded'.

    Pins the branch order in `_derive_pod_status` — if someone reorders
    so the phase==Succeeded check runs first, an in-progress-deletion
    Succeeded pod would erroneously appear as a reapable-completed pod
    to design §6.5 cleanup callers.
    """
    pod = _make_pod(
        "p",
        phase="Succeeded",
        container_ready=True,
        deletion_timestamp=datetime.datetime(2026, 1, 1),
    )
    assert _derive_pod_status(pod) == "Terminating"


def test_pod_to_pool_worker_info_returns_none_daemon_pod_name_when_metadata_missing(
):  # noqa: E501
    """`_pod_to_pool_worker_info` must return `None` (not `''`) for
    daemon_pod_name when `pod.metadata` is absent — pinning the
    Optional[str] contract from PoolWorkerInfo so callers can use
    `is None` instead of comparing against an empty-string sentinel.
    """
    pod = V1Pod(metadata=None, spec=None, status=None)
    info = _pod_to_pool_worker_info(pod, "uid-x")
    assert info.daemon_pod_name is None
    assert info.node_name is None


# ---------------------------------------------------------------------------
# Regression coverage for Siyang's review (#1, #2, #3, #5, #6a)
# ---------------------------------------------------------------------------


def test_translate_api_error_429_throttled():
    out = _translate_api_error(_api_exc(429, "too many requests"))
    assert isinstance(out, ApiServerThrottledError)


def test_translate_api_error_5xx_transient():
    # 500 / 502 / 503 / 504 all map to the same transient bucket so
    # the controller can apply one retry policy.
    for status in (500, 502, 503, 504):
        out = _translate_api_error(_api_exc(status, "boom"))
        assert isinstance(out, ApiServerTransientError), (
            f"status {status} should be ApiServerTransientError, "
            f"got {type(out).__name__}")


def test_translate_api_error_409_already_exists():
    out = _translate_api_error(_api_exc(409, "already exists"))
    assert isinstance(out, PoolWorkerAlreadyExistsError)


def test_try_delete_pvc_timeout_preserves_csi_finalizers():
    # A slow CSI delete must fail visibly without bypassing the
    # provisioner's volume-protection finalizers.
    core, apps = _empty_pool_worker_lookup_mocks()
    pvc_obj = MagicMock()
    core.read_namespaced_persistent_volume_claim.side_effect = None
    core.read_namespaced_persistent_volume_claim.return_value = pvc_obj

    mgr = _make_manager(core=core, apps=apps)
    with pytest.raises(PoolWorkerCleanupFailedError):
        asyncio.run(mgr.delete_pool_worker("uid", pvc_delete_timeout=0.5))
    core.patch_namespaced_persistent_volume_claim.assert_not_called()


def test_wait_for_daemon_pod_gone_raises_on_timeout_by_default():
    # delete_pool_worker → step 3 scales Deployment to 0 → polls for
    # the daemon pod to reap. If timeout elapses with the pod still
    # present, raise so we don't proceed to PVC delete under a
    # still-mounted kubelet (would wedge the CSI attacher finalizer).
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    daemon_pod = _make_pod(
        "p",
        node_name="node-X",
        labels={
            LABEL_APP: LABEL_APP_VALUE_DAEMON,
            LABEL_POOL_WORKER_UID: "uid",
        },
    )
    # First list call (in _get_pool_worker_info) returns the daemon
    # so we know its node_name. Subsequent calls (the wait poll)
    # always return the pod alive — never reaped.
    core.list_namespaced_pod.side_effect = lambda **kwargs: V1PodList(items=[
    ] if "diskgraph-activity-pod" in kwargs["label_selector"] else [daemon_pod])

    mgr = _make_manager(core=core, apps=apps)
    with pytest.raises(PoolWorkerCleanupFailedError):
        asyncio.run(mgr.delete_pool_worker("uid", scale_down_timeout=0.5))


def test_wait_for_daemon_pod_gone_with_force_on_timeout_returns():
    # The DEAD-node teardown path (design §10) passes force_on_timeout
    # so the manager rips the volume out even with a still-mounted
    # kubelet. Calling _wait_for_daemon_pod_gone directly with
    # force_on_timeout=True should return None instead of raising.
    core = MagicMock(spec=k8s_client.CoreV1Api)
    daemon_pod = _make_pod(
        "p",
        labels={
            LABEL_APP: LABEL_APP_VALUE_DAEMON,
            LABEL_POOL_WORKER_UID: "uid",
        },
    )
    core.list_namespaced_pod.return_value = V1PodList(items=[daemon_pod])

    mgr = _make_manager(core=core)
    # Should return None (not raise) when force_on_timeout=True even
    # though the pod is never reaped.
    result = asyncio.run(
        mgr._wait_for_daemon_pod_gone(
            "uid",
            timeout=0.3,
            force_on_timeout=True,
        ))
    assert result is None


# ---------------------------------------------------------------------------
# Layer 3: fail-fast on terminal Pending states
# ---------------------------------------------------------------------------


def test_wait_for_activity_job_pod_running_immediate_terminal_on_crash_loop():
    """CrashLoopBackOff is immediately terminal — no grace window."""
    core = MagicMock(spec=k8s_client.CoreV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod('act', phase='Pending', container_ready=False,
                  waiting_reason='CrashLoopBackOff'),
    ])
    mgr = _make_manager(core=core)
    with pytest.raises(ProvisionTerminalError, match='CrashLoopBackOff'):
        asyncio.run(
            mgr.wait_for_activity_job_pod_running(
                job_name='diskgraph-activity-abc', timeout=5.0))
    assert core.list_namespaced_pod.call_count == 1  # fail on first poll


def test_wait_for_activity_job_pod_running_image_pull_backoff_grace_window(
        monkeypatch):
    """Each Job attempt gets its own transient image-pull grace."""
    monkeypatch.setattr(
        'pool.worker_manager.k8s._IMAGE_PULL_BACKOFF_GRACE_SECS', 0)
    core = MagicMock(spec=k8s_client.CoreV1Api)
    old = _make_pod('old', phase='Pending', container_ready=False,
                    waiting_reason='ImagePullBackOff')
    replacement = _make_pod('replacement', phase='Pending',
                            container_ready=False,
                            waiting_reason='ImagePullBackOff')
    core.list_namespaced_pod.side_effect = [
        V1PodList(items=[old]),
        V1PodList(items=[replacement]),
        V1PodList(items=[replacement]),
    ]
    mgr = _make_manager(core=core)
    with pytest.raises(ProvisionTerminalError, match='ImagePullBackOff'):
        asyncio.run(
            mgr.wait_for_activity_job_pod_running(
                job_name='diskgraph-activity-abc', timeout=5.0))
    assert core.list_namespaced_pod.call_count == 3


def test_wait_for_cache_activity_pod_allows_gpu_handoff_to_finish(monkeypatch):
    """A released lease can precede termination of the previous GPU pod."""
    monkeypatch.setattr('pool.worker_manager.k8s._POLL_INTERVAL_SECS', 0)
    core = MagicMock(spec=k8s_client.CoreV1Api)
    core.list_namespaced_pod.side_effect = [
        V1PodList(items=[
            _make_pod(
                'act',
                phase='Pending',
                container_ready=False,
                labels={DISKGRAPH_POOL_TYPE_LABEL: POOL_TYPE_CACHE},
                conditions=[
                    V1PodCondition(
                        type='PodScheduled',
                        status='False',
                        reason='Unschedulable',
                        message=('0/1 nodes are available: '
                                 'Insufficient nvidia.com/gpu.'),
                    ),
                ],
            ),
        ]),
        V1PodList(items=[
            _make_pod(
                'act',
                phase='Running',
                labels={DISKGRAPH_POOL_TYPE_LABEL: POOL_TYPE_CACHE},
                annotations={ANNOTATION_JOB_ID: 'job-1'},
            ),
        ]),
    ]
    mgr = _make_manager(core=core)

    info = asyncio.run(
        mgr.wait_for_activity_job_pod_running(
            job_name='diskgraph-activity-abc',
            timeout=5.0,
        ))

    assert info.pod_name == 'act'
    assert core.list_namespaced_pod.call_count == 2


def test_wait_for_cache_activity_gpu_shortage_times_out_retryably(monkeypatch):
    monkeypatch.setattr('pool.worker_manager.k8s._POLL_INTERVAL_SECS', 0)
    core = MagicMock(spec=k8s_client.CoreV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod(
            'act',
            phase='Pending',
            container_ready=False,
            labels={DISKGRAPH_POOL_TYPE_LABEL: POOL_TYPE_CACHE},
            conditions=[
                V1PodCondition(
                    type='PodScheduled',
                    status='False',
                    reason='Unschedulable',
                    message=('0/1 nodes are available: '
                             'Insufficient nvidia.com/gpu.'),
                ),
            ],
        ),
    ])
    mgr = _make_manager(core=core)

    with pytest.raises(ProvisionTimeoutError, match='did not reach Running'):
        asyncio.run(
            mgr.wait_for_activity_job_pod_running(
                job_name='diskgraph-activity-abc',
                timeout=0.01,
            ))


def test_wait_for_ephemeral_activity_pod_ignores_transient_unschedulable(
        monkeypatch):
    """Ephemeral pods may be Unschedulable while Karpenter scales capacity."""
    monkeypatch.setattr('pool.worker_manager.k8s._POLL_INTERVAL_SECS', 0)
    core = MagicMock(spec=k8s_client.CoreV1Api)
    core.list_namespaced_pod.side_effect = [
        V1PodList(items=[
            _make_pod(
                'act',
                phase='Pending',
                container_ready=False,
                labels={DISKGRAPH_POOL_TYPE_LABEL: POOL_TYPE_EPHEMERAL},
                conditions=[
                    V1PodCondition(
                        type='PodScheduled',
                        status='False',
                        reason='Unschedulable',
                        message='0/1 nodes are available: scale-up pending.',
                    ),
                ],
            ),
        ]),
        V1PodList(items=[
            _make_pod(
                'act',
                phase='Running',
                labels={DISKGRAPH_POOL_TYPE_LABEL: POOL_TYPE_EPHEMERAL},
                annotations={ANNOTATION_JOB_ID: 'job-1'},
            ),
        ]),
    ]
    mgr = _make_manager(core=core)

    info = asyncio.run(
        mgr.wait_for_activity_job_pod_running(
            job_name='diskgraph-eph-activity-abc',
            timeout=5.0,
        ))

    assert info.pod_name == 'act'
    assert core.list_namespaced_pod.call_count == 2


def test_wait_for_activity_job_waits_for_job_retry_after_failed_pod(
        monkeypatch):
    monkeypatch.setattr('pool.worker_manager.k8s._POLL_INTERVAL_SECS', 0)
    core = MagicMock(spec=k8s_client.CoreV1Api)
    failed = _make_pod('old-failed', phase='Failed')
    running = _make_pod('new-running', phase='Running')
    core.list_namespaced_pod.side_effect = [
        V1PodList(items=[failed]),
        V1PodList(items=[failed, running]),
    ]
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    batch.read_namespaced_job_status.return_value = MagicMock(status=MagicMock(
        conditions=[]))
    mgr = _make_manager(core=core, batch=batch)

    info = asyncio.run(
        mgr.wait_for_activity_job_pod_running(job_name='diskgraph-activity-abc',
                                              timeout=5.0))

    assert info.pod_name == 'new-running'
    batch.read_namespaced_job_status.assert_called_once()


@pytest.mark.parametrize('remaining_pod', [
    None,
    _make_pod('terminating', deletion_timestamp=datetime.datetime(2026, 1, 1)),
    _make_pod('failed', phase='Failed'),
])
def test_wait_for_activity_job_fails_fast_when_persisted_job_is_gone(
        remaining_pod):
    core = MagicMock(spec=k8s_client.CoreV1Api)
    core.list_namespaced_pod.return_value = V1PodList(
        items=[] if remaining_pod is None else [remaining_pod])
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    batch.read_namespaced_job_status.side_effect = _api_exc(404, 'not found')
    mgr = _make_manager(core=core, batch=batch)

    with pytest.raises(ProvisionTimeoutError, match='disappeared'):
        asyncio.run(
            mgr.wait_for_activity_job_pod_running(
                job_name='diskgraph-activity-missing', timeout=5.0))


@pytest.mark.parametrize(
    ('condition_type', 'error_type', 'message'),
    [
        ('Failed', ProvisionTimeoutError, 'exhausted its pod attempts'),
        ('Complete', ProvisionTerminalError, 'completed before its Temporal'),
    ],
)
def test_wait_for_activity_job_fails_fast_for_terminal_job_without_pods(
        condition_type, error_type, message):
    core = MagicMock(spec=k8s_client.CoreV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[])
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    batch.read_namespaced_job_status.return_value = MagicMock(status=MagicMock(
        conditions=[MagicMock(type=condition_type, status='True')]))
    mgr = _make_manager(core=core, batch=batch)

    with pytest.raises(error_type, match=message):
        asyncio.run(
            mgr.wait_for_activity_job_pod_running(
                job_name='diskgraph-activity-terminal', timeout=5.0))


def test_provision_activity_pod_does_not_rollback_on_wait_failure():
    """The manager leaves a failed Job for controller-owned cleanup."""
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    # Daemon pod is Registered so the precondition check passes.
    core.list_namespaced_pod.side_effect = [
        # _get_pool_worker_info call:
        V1PodList(items=[
            _make_pod(
                "p",
                labels={
                    LABEL_APP: LABEL_APP_VALUE_DAEMON,
                    LABEL_POOL_WORKER_UID: "uid",
                },
            )
        ]),
        # _wait_for_activity_job_pod_running poll: pod stays Failed.
        V1PodList(items=[_make_pod("act-1", phase="Failed")]),
    ]
    batch.create_namespaced_job.return_value = MagicMock()
    batch.read_namespaced_job_status.return_value = MagicMock(status=MagicMock(
        conditions=[MagicMock(type='Failed', status='True')]))

    mgr = _make_manager(core=core, apps=apps, batch=batch)
    asyncio.run(
        mgr.provision_activity_pod(
            pool_worker_uid="uid",
            job_id="job-1",
            snapshot_id="snap-1",
            activity="materialize",
            image="img:test",
        ))
    with pytest.raises(ProvisionTimeoutError):
        asyncio.run(
            mgr.wait_for_activity_job_pod_running(
                job_name=mgr.activity_job_name("job-1", "durable"),
                timeout=2.0,
            ))

    # The controller owns cleanup after readiness fails.
    batch.delete_namespaced_job.assert_not_called()


def test_provision_ephemeral_activity_pod_does_not_rollback_on_wait_failure():
    """The manager leaves a completed Job for controller-owned cleanup."""
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    core.list_namespaced_pod.return_value = V1PodList(
        items=[_make_pod("eph-1", phase="Succeeded")])
    batch.create_namespaced_job.return_value = MagicMock()

    mgr = _make_manager(core=core, apps=apps, batch=batch)
    asyncio.run(
        mgr.provision_ephemeral_activity_pod(
            job_id="job-1",
            snapshot_id="snap-1",
            activity="materialize",
            image="img:test",
            ephemeral_storage_bytes=10 * (1 << 30),
        ))
    with pytest.raises(ProvisionTerminalError,
                       match='completed before its Temporal poller'):
        asyncio.run(
            mgr.wait_for_activity_job_pod_running(
                job_name=mgr.activity_job_name("job-1", "overflow"),
                timeout=2.0,
            ))

    batch.delete_namespaced_job.assert_not_called()


def test_provision_pool_worker_rolls_back_deployment_on_wait_failure():
    # Existing test covers create-Deployment-fails path. This pins
    # the wait-fails path: create succeeded, _wait_for_daemon_registered
    # raised → rollback must delete BOTH Deployment and PVC, with
    # Deployment first so no new pods come up against the soon-to-be-
    # deleted PVC.
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    # _wait_for_daemon_registered polls list_namespaced_pod; never
    # returns Registered → ProvisionTimeoutError after timeout.
    core.list_namespaced_pod.return_value = V1PodList(items=[])

    mgr = _make_manager(core=core, apps=apps)
    with pytest.raises(ProvisionTimeoutError):
        asyncio.run(mgr.provision_pool_worker(timeout=0.3))

    # Both rollback paths fired: Deployment then PVC.
    apps.delete_namespaced_deployment.assert_called_once()
    core.delete_namespaced_persistent_volume_claim.assert_called_once()


# ---------------------------------------------------------------------------
# Regression coverage for Siyang's review #6b — reclaim semantics
# ---------------------------------------------------------------------------


def _live_pool_worker_resources(uid="existing-uid"):
    return (
        V1PersistentVolumeClaim(
            metadata=V1ObjectMeta(name=f"diskgraph-pvc-{uid}"), ),
        V1Deployment(
            metadata=V1ObjectMeta(name=f"diskgraph-daemon-{uid}"),
            spec=V1DeploymentSpec(
                replicas=1,
                selector=V1LabelSelector(),
                template=V1PodTemplateSpec(),
            ),
        ),
    )


def _assert_no_pool_worker_resources_created(core, apps, policy):
    core.create_namespaced_persistent_volume_claim.assert_not_called()
    apps.create_namespaced_deployment.assert_not_called()
    policy.create_namespaced_pod_disruption_budget.assert_not_called()


@pytest.mark.parametrize(
    "resource_state, expected_error",
    [
        ("missing-pvc", PoolWorkerNotFoundError),
        ("terminating-deployment", PoolWorkerNotReadyError),
        ("scaled-down-deployment", PoolWorkerNotReadyError),
    ],
)
def test_existing_only_activity_rechecks_resources_after_worker_reclaim(
        resource_state, expected_error):
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    policy = MagicMock(spec=k8s_client.PolicyV1Api)
    pvc, deployment = _live_pool_worker_resources()
    core.read_namespaced_persistent_volume_claim.return_value = pvc
    apps.read_namespaced_deployment.return_value = deployment
    core.list_namespaced_pod.return_value = _registered_daemon_for(
        "existing-uid")
    manager = _make_manager(core=core, apps=apps, batch=batch, policy=policy)

    asyncio.run(
        manager.provision_pool_worker(
            pool_worker_uid="existing-uid",
            timeout=2.0,
            require_existing_resources=True,
        ), )

    if resource_state == "missing-pvc":
        core.read_namespaced_persistent_volume_claim.side_effect = _api_exc(404)
    elif resource_state == "terminating-deployment":
        assert deployment.metadata is not None
        deployment.metadata.deletion_timestamp = datetime.datetime(2026, 1, 1)
    else:
        assert deployment.spec is not None
        deployment.spec.replicas = 0

    with pytest.raises(expected_error, match="existing-uid"):
        asyncio.run(
            manager.provision_activity_pod(
                pool_worker_uid="existing-uid",
                job_id="job-1",
                snapshot_id="snapshot-1",
                activity="train",
                image="kumo/activity:test",
                require_existing_resources=True,
            ), )

    batch.create_namespaced_job.assert_not_called()
    _assert_no_pool_worker_resources_created(core, apps, policy)


@pytest.mark.parametrize(
    "resource_state, expected_error",
    [
        ("missing-deployment", PoolWorkerNotFoundError),
        ("terminating-pvc", PoolWorkerNotReadyError),
    ],
)
def test_provision_pool_worker_existing_only_rejects_before_wait(
        resource_state, expected_error):
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    policy = MagicMock(spec=k8s_client.PolicyV1Api)
    pvc, deployment = _live_pool_worker_resources()
    if resource_state == "missing-deployment":
        core.read_namespaced_persistent_volume_claim.return_value = pvc
        apps.read_namespaced_deployment.side_effect = _api_exc(404)
    else:
        assert pvc.metadata is not None
        pvc.metadata.deletion_timestamp = datetime.datetime(2026, 1, 1)
        core.read_namespaced_persistent_volume_claim.return_value = pvc
        apps.read_namespaced_deployment.return_value = deployment

    mgr = _make_manager(core=core, apps=apps, policy=policy)
    with pytest.raises(expected_error, match="existing-uid"):
        asyncio.run(
            mgr.provision_pool_worker(
                pool_worker_uid="existing-uid",
                timeout=2.0,
                require_existing_resources=True,
            ), )

    core.list_namespaced_pod.assert_not_called()
    _assert_no_pool_worker_resources_created(core, apps, policy)


@pytest.mark.parametrize(
    "resource_state, expected_error",
    [
        ("missing-pvc", PoolWorkerNotFoundError),
        ("terminating-deployment", PoolWorkerNotReadyError),
    ],
)
def test_provision_pool_worker_existing_only_rejects_after_wait(
        resource_state, expected_error):
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    policy = MagicMock(spec=k8s_client.PolicyV1Api)
    pvc, deployment = _live_pool_worker_resources()
    core.list_namespaced_pod.return_value = _registered_daemon_for(
        "existing-uid")
    if resource_state == "missing-pvc":
        core.read_namespaced_persistent_volume_claim.side_effect = [
            pvc,
            _api_exc(404),
        ]
        apps.read_namespaced_deployment.return_value = deployment
    else:
        assert deployment.metadata is not None
        terminating_deployment = V1Deployment(
            metadata=V1ObjectMeta(
                name="diskgraph-daemon-existing-uid",
                deletion_timestamp=datetime.datetime(2026, 1, 1),
            ), )
        core.read_namespaced_persistent_volume_claim.return_value = pvc
        apps.read_namespaced_deployment.side_effect = [
            deployment,
            terminating_deployment,
        ]

    mgr = _make_manager(core=core, apps=apps, policy=policy)
    with pytest.raises(expected_error, match="existing-uid"):
        asyncio.run(
            mgr.provision_pool_worker(
                pool_worker_uid="existing-uid",
                timeout=2.0,
                require_existing_resources=True,
            ), )

    core.list_namespaced_pod.assert_called()
    _assert_no_pool_worker_resources_created(core, apps, policy)


def test_provision_pool_worker_reclaim_recovers_from_409_on_pvc_create():
    """Reclaim path: caller passes an explicit uid + the PVC create
    returns 409 (resource already exists from a prior attempt). The
    manager should log + continue to _wait_for_daemon_registered
    rather than failing — controller crash recovery contract.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    # PVC create surfaces 409 (translated to PoolWorkerAlreadyExistsError).
    core.create_namespaced_persistent_volume_claim.side_effect = (_api_exc(
        409, "persistentvolumeclaims already exists"))
    # Deployment create succeeds (or also 409 — either is fine).
    apps.create_namespaced_deployment.return_value = MagicMock()
    # _wait_for_daemon_registered finds a Registered daemon.
    registered_pod = _make_pod(
        "p",
        labels={
            LABEL_APP: LABEL_APP_VALUE_DAEMON,
            LABEL_POOL_WORKER_UID: "existing-uid",
        },
    )
    core.list_namespaced_pod.return_value = V1PodList(items=[registered_pod])

    mgr = _make_manager(core=core, apps=apps)
    info = asyncio.run(
        mgr.provision_pool_worker(
            pool_worker_uid="existing-uid",
            timeout=2.0,
        ), )
    assert info.pool_worker_uid == "existing-uid"
    assert info.status == "Registered"


def test_provision_pool_worker_reclaim_recovers_from_409_on_deployment_create():
    """Same shape — but the 409 surfaces from Deployment create
    (PVC was already created and survived the prior attempt; the
    Deployment was too).
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    # PVC create succeeds.
    core.create_namespaced_persistent_volume_claim.return_value = (MagicMock())
    apps.create_namespaced_deployment.side_effect = (_api_exc(
        409, "deployments already exists"))
    registered_pod = _make_pod(
        "p",
        labels={
            LABEL_APP: LABEL_APP_VALUE_DAEMON,
            LABEL_POOL_WORKER_UID: "existing-uid",
        },
    )
    core.list_namespaced_pod.return_value = V1PodList(items=[registered_pod])

    mgr = _make_manager(core=core, apps=apps)
    info = asyncio.run(
        mgr.provision_pool_worker(
            pool_worker_uid="existing-uid",
            timeout=2.0,
        ), )
    assert info.status == "Registered"


def test_provision_pool_worker_reclaim_skips_rollback_on_wait_failure():
    """Reclaim path: if _wait_for_daemon_registered raises, rollback
    is intentionally NOT performed — caller asked to reuse the uid's
    resources, so the manager doesn't nuke them. Caller decides
    whether to delete_pool_worker(uid) + retry.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    # Both creates raise 409 (full reclaim — both resources existed).
    core.create_namespaced_persistent_volume_claim.side_effect = (_api_exc(
        409, "already exists"))
    apps.create_namespaced_deployment.side_effect = (_api_exc(
        409, "already exists"))
    # Daemon never reaches Registered → ProvisionTimeoutError.
    core.list_namespaced_pod.return_value = V1PodList(items=[])

    mgr = _make_manager(core=core, apps=apps)
    with pytest.raises(ProvisionTimeoutError):
        asyncio.run(
            mgr.provision_pool_worker(pool_worker_uid="existing-uid",
                                      timeout=0.3))

    # Rollback skipped on reclaim — neither delete API was called.
    apps.delete_namespaced_deployment.assert_not_called()
    core.delete_namespaced_persistent_volume_claim.assert_not_called()


def test_provision_pool_worker_fresh_mint_propagates_409():
    """Fresh-mint path (no uid passed): 10-char uuid hex collisions
    are astronomically unlikely, so a 409 there means a real bug —
    propagate as PoolWorkerAlreadyExistsError instead of silently
    auto-recovering against someone else's resources.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    core.create_namespaced_persistent_volume_claim.side_effect = (_api_exc(
        409, "already exists"))

    mgr = _make_manager(core=core, apps=apps)
    with pytest.raises(PoolWorkerAlreadyExistsError):
        # No pool_worker_uid passed → fresh-mint path.
        asyncio.run(mgr.provision_pool_worker(timeout=2.0))


# ---------------------------------------------------------------------------
# Cache activity Job: deterministic naming + reclaim contract
# ---------------------------------------------------------------------------


def test_provision_activity_pod_job_name_is_deterministic_in_job_id():
    """Cache Job name is now derived solely from hash(job_id) — no
    random suffix. Two calls with the same job_id MUST land on the
    same Job name so that a retry (e.g., Temporal heartbeat lapsed on
    the controller's place_snapshot activity during a slow image-pull)
    hits 409 → reclaim instead of spawning a fresh K8s Job. Without
    this, every retry creates a new Pending pod pinned to the same
    cache pool worker's RWO PVC — the 24-pod fan-out bug.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    core.list_namespaced_pod.side_effect = [
        _registered_daemon_for("uid"),
        V1PodList(items=[
            _make_pod(
                "act",
                labels={
                    LABEL_APP: LABEL_APP_VALUE_ACTIVITY,
                    LABEL_POOL_WORKER_UID: "uid",
                    LABEL_JOB_ID_HASH: _hash_job_id("j-det"),
                },
            )
        ]),
    ]
    mgr = _make_manager(core=core, batch=batch)
    asyncio.run(
        mgr.provision_activity_pod(
            pool_worker_uid="uid",
            job_id="j-det",
            snapshot_id="snap",
            activity="ingest",
            image="img",
        ))
    asyncio.run(
        mgr.wait_for_activity_job_pod_running(
            job_name=mgr.activity_job_name("j-det", "durable"),
            timeout=2.0,
        ))
    name_first = batch.create_namespaced_job.call_args.kwargs[
        "body"].metadata.name

    # Second call — fresh mock state but same job_id.
    core2 = MagicMock(spec=k8s_client.CoreV1Api)
    batch2 = MagicMock(spec=k8s_client.BatchV1Api)
    core2.list_namespaced_pod.side_effect = [
        _registered_daemon_for("uid"),
        V1PodList(items=[
            _make_pod(
                "act",
                labels={
                    LABEL_APP: LABEL_APP_VALUE_ACTIVITY,
                    LABEL_POOL_WORKER_UID: "uid",
                    LABEL_JOB_ID_HASH: _hash_job_id("j-det"),
                },
            )
        ]),
    ]
    mgr2 = _make_manager(core=core2, batch=batch2)
    asyncio.run(
        mgr2.provision_activity_pod(
            pool_worker_uid="uid",
            job_id="j-det",
            snapshot_id="snap",
            activity="ingest",
            image="img",
        ))
    asyncio.run(
        mgr2.wait_for_activity_job_pod_running(
            job_name=mgr2.activity_job_name("j-det", "durable"),
            timeout=2.0,
        ))
    name_second = batch2.create_namespaced_job.call_args.kwargs[
        "body"].metadata.name

    assert name_first == name_second
    assert name_first == f"diskgraph-activity-{_hash_job_id('j-det')}"


def test_provision_activity_pod_recovers_from_409_on_create():
    """Reclaim-always contract (cache pool): Job create returns 409 (a
    prior attempt already created the Job — e.g. Temporal retried the
    place_snapshot activity while the previous controller call was
    still waiting on a slow image pull). The manager should log +
    continue to the pod-running wait rather than failing. Deterministic
    naming makes 409 the expected retry signal, not an error.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    batch.create_namespaced_job.side_effect = (_api_exc(
        409, "jobs.batch already exists"))
    core.list_namespaced_pod.side_effect = [
        _registered_daemon_for("uid"),
        # Pod from the prior attempt is now Running.
        V1PodList(items=[
            _make_pod(
                "act",
                phase="Running",
                labels={
                    LABEL_APP: LABEL_APP_VALUE_ACTIVITY,
                    LABEL_POOL_WORKER_UID: "uid",
                    LABEL_JOB_ID_HASH: _hash_job_id("j-reclaim"),
                },
            )
        ]),
    ]

    mgr = _make_manager(core=core, batch=batch)
    asyncio.run(
        mgr.provision_activity_pod(
            pool_worker_uid="uid",
            job_id="j-reclaim",
            snapshot_id="snap",
            activity="materialize",
            image="img",
        ))
    info = asyncio.run(
        mgr.wait_for_activity_job_pod_running(
            job_name=mgr.activity_job_name("j-reclaim", "durable"),
            timeout=2.0,
        ))
    assert info.pod_name == "act"
    # Manager must NOT roll back a Job it found in-flight.
    batch.delete_namespaced_job.assert_not_called()


# ---------------------------------------------------------------------------
# Eph activity Job: deterministic naming + reclaim contract
# ---------------------------------------------------------------------------


def test_provision_ephemeral_activity_pod_job_name_is_deterministic_in_job_id():
    """Eph Job name is now derived solely from hash(job_id) — no random
    suffix. Two calls with the same job_id MUST land on the same Job
    name so that a retry (after Temporal RPC timeout / controller
    crash during Karpenter cold-start) hits 409 → reclaim instead of
    spawning a fresh K8s Job and a fresh node-up.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod(
            "eph",
            labels={
                LABEL_APP: LABEL_APP_VALUE_ACTIVITY_EPH,
                LABEL_JOB_ID_HASH: _hash_job_id("j-det"),
            },
        )
    ])

    mgr = _make_manager(core=core, batch=batch)
    asyncio.run(
        mgr.provision_ephemeral_activity_pod(
            job_id="j-det",
            snapshot_id="snap",
            activity="ingest",
            image="img",
            ephemeral_storage_bytes=1 * (1 << 30),
        ))
    name_first = batch.create_namespaced_job.call_args.kwargs[
        "body"].metadata.name

    # Second call — fresh mock state but same job_id.
    core2 = MagicMock(spec=k8s_client.CoreV1Api)
    batch2 = MagicMock(spec=k8s_client.BatchV1Api)
    core2.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod(
            "eph",
            labels={
                LABEL_APP: LABEL_APP_VALUE_ACTIVITY_EPH,
                LABEL_JOB_ID_HASH: _hash_job_id("j-det"),
            },
        )
    ])
    mgr2 = _make_manager(core=core2, batch=batch2)
    asyncio.run(
        mgr2.provision_ephemeral_activity_pod(
            job_id="j-det",
            snapshot_id="snap",
            activity="ingest",
            image="img",
            ephemeral_storage_bytes=1 * (1 << 30),
        ))
    name_second = batch2.create_namespaced_job.call_args.kwargs[
        "body"].metadata.name

    assert name_first == name_second
    assert name_first == f"diskgraph-eph-activity-{_hash_job_id('j-det')}"


def test_provision_ephemeral_activity_pod_recovers_from_409_on_create():
    """Reclaim-always contract: Job create returns 409 (a prior attempt
    already created the Job, e.g. the controller timed out during
    Karpenter cold-start and Temporal retried the activity). The
    manager should log + continue to the pod-running wait rather than
    failing. Deterministic naming makes 409 the expected retry signal,
    not an error.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    batch.create_namespaced_job.side_effect = (_api_exc(
        409, "jobs.batch already exists"))
    # Pod from the prior attempt is now Running on the freshly
    # provisioned Karpenter node.
    core.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod(
            "eph",
            phase="Running",
            labels={
                LABEL_APP: LABEL_APP_VALUE_ACTIVITY_EPH,
                LABEL_JOB_ID_HASH: _hash_job_id("j-reclaim"),
            },
        )
    ])

    mgr = _make_manager(core=core, batch=batch)
    asyncio.run(
        mgr.provision_ephemeral_activity_pod(
            job_id="j-reclaim",
            snapshot_id="snap",
            activity="materialize",
            image="img",
            ephemeral_storage_bytes=1 * (1 << 30),
        ))
    info = asyncio.run(
        mgr.wait_for_activity_job_pod_running(
            job_name=mgr.activity_job_name("j-reclaim", "overflow"),
            timeout=2.0,
        ))
    assert info.pod_name == "eph"
    # Manager must NOT roll back a Job it found in-flight.
    batch.delete_namespaced_job.assert_not_called()


# ---------------------------------------------------------------------------
# Regression coverage for the env-as-dict refactor (Siyang follow-up)
# ---------------------------------------------------------------------------


def test_provision_activity_pod_with_no_env_dict_only_adds_pod_identity():
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    core.list_namespaced_pod.side_effect = [
        _registered_daemon_for("uid"),
        V1PodList(items=[
            _make_pod(
                "act",
                labels={
                    LABEL_APP: LABEL_APP_VALUE_ACTIVITY,
                    LABEL_POOL_WORKER_UID: "uid",
                    LABEL_JOB_ID_HASH: "j-noenv",
                },
            )
        ]),
    ]

    mgr = _make_manager(core=core, batch=batch)
    asyncio.run(
        mgr.provision_activity_pod(
            pool_worker_uid="uid",
            job_id="j-noenv",
            snapshot_id="snap",
            activity="train",
            image="kumo/activity:test",
            # env intentionally omitted — exercise the default-None path.
        ))
    asyncio.run(
        mgr.wait_for_activity_job_pod_running(
            job_name=mgr.activity_job_name("j-noenv", "durable"),
            timeout=2.0,
        ))
    body = batch.create_namespaced_job.call_args.kwargs["body"]
    container = body.spec.template.spec.containers[0]
    assert [env.name for env in container.env] == [ACTIVITY_WORKER_POD_NAME_ENV]
    assert container.env[0].value_from.field_ref.field_path == "metadata.name"


def test_delete_pool_worker_passes_force_on_timeout_through():
    """delete_pool_worker(force_on_timeout=True) must thread the
    kwarg into _wait_for_daemon_pod_gone so the §10 DEAD-node path
    is reachable from the public API. If the wiring drops, the
    daemon-still-mounted case below would raise instead of returning.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    daemon_pod = _make_pod(
        "p",
        node_name="node-X",
        labels={
            LABEL_APP: LABEL_APP_VALUE_DAEMON,
            LABEL_POOL_WORKER_UID: "uid",
        },
    )
    # Daemon pod NEVER reaps — _wait_for_daemon_pod_gone would
    # normally raise PoolWorkerCleanupFailedError on timeout. With
    # force_on_timeout=True it should log + return so the rest of
    # delete_pool_worker (PVC delete, Deployment delete, PDB delete)
    # can run.
    core.list_namespaced_pod.side_effect = lambda **kwargs: V1PodList(items=[
    ] if "diskgraph-activity-pod" in kwargs["label_selector"] else [daemon_pod])
    core.read_namespaced_persistent_volume_claim.side_effect = _api_exc(
        404,
        "not found",
    )

    mgr = _make_manager(core=core, apps=apps)
    # Should NOT raise — force_on_timeout=True overrides the default
    # PoolWorkerCleanupFailedError.
    asyncio.run(
        mgr.delete_pool_worker(
            "uid",
            scale_down_timeout=0.3,
            force_on_timeout=True,
        ))


def test_provision_ephemeral_activity_pod_with_no_env_only_adds_pod_identity():
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod(
            "eph", labels={
                LABEL_APP: LABEL_APP_VALUE_ACTIVITY_EPH,
                LABEL_JOB_ID_HASH: "j-eph-noenv",
            }),
    ])

    mgr = _make_manager(core=core, batch=batch)
    asyncio.run(
        mgr.provision_ephemeral_activity_pod(
            job_id="j-eph-noenv",
            snapshot_id="snap",
            activity="ingest",
            image="kumo/activity:eph",
            ephemeral_storage_bytes=1 * (1 << 30),
            # env intentionally omitted.
        ))
    body = batch.create_namespaced_job.call_args.kwargs["body"]
    container = body.spec.template.spec.containers[0]
    assert [env.name for env in container.env] == [ACTIVITY_WORKER_POD_NAME_ENV]
    assert container.env[0].value_from.field_ref.field_path == "metadata.name"


# ---------------------------------------------------------------------------
# Follow-up: kumo-temporal SA default + daemon env-dict support
# ---------------------------------------------------------------------------


def test_default_service_account_is_kumo_temporal_for_all_pods():
    """Daemon Deployment, activity Job, and eph activity Job all run
    under the same default SA (``kumo-temporal``) so they share IRSA +
    RBAC bindings with the rest of the dataplane workloads (mlworkflow
    et al.). Pins the new default — accidentally changing it back to
    ``diskgraph-daemon`` would silently break secret access on every
    pod the manager spawns.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    daemon_pods = _registered_daemon_for("uid")
    activity_pod = _make_pod(
        "act-pod",
        labels={
            LABEL_APP: LABEL_APP_VALUE_ACTIVITY,
            LABEL_POOL_WORKER_UID: "uid",
            LABEL_JOB_ID_HASH: "j-sa",
        },
    )
    eph_pod = _make_pod(
        "eph-pod",
        labels={
            LABEL_APP: LABEL_APP_VALUE_ACTIVITY_EPH,
            LABEL_JOB_ID_HASH: "j-sa-eph",
        },
    )
    # Calls in order: provision_pool_worker (registered daemon),
    # provision_activity_pod (daemon-status check + activity-pod
    # ready), provision_ephemeral_activity_pod (eph-pod ready).
    core.list_namespaced_pod.side_effect = [
        daemon_pods,
        daemon_pods,
        V1PodList(items=[activity_pod]),
        V1PodList(items=[eph_pod]),
    ]

    mgr = _make_manager(core=core, apps=apps, batch=batch)
    asyncio.run(mgr.provision_pool_worker(
        pool_worker_uid="uid",
        timeout=2.0,
    ))
    asyncio.run(
        mgr.provision_activity_pod(
            pool_worker_uid="uid",
            job_id="j-sa",
            snapshot_id="snap",
            activity="train",
            image="kumo/activity:test",
        ))
    asyncio.run(
        mgr.wait_for_activity_job_pod_running(
            job_name=mgr.activity_job_name("j-sa", "durable"),
            timeout=2.0,
        ))
    asyncio.run(
        mgr.provision_ephemeral_activity_pod(
            job_id="j-sa-eph",
            snapshot_id="snap",
            activity="ingest",
            image="kumo/activity:eph",
            ephemeral_storage_bytes=1 * (1 << 30),
        ))
    asyncio.run(
        mgr.wait_for_activity_job_pod_running(
            job_name=mgr.activity_job_name("j-sa-eph", "overflow"),
            timeout=2.0,
        ))

    dep_spec = (apps.create_namespaced_deployment.call_args.kwargs["body"].spec.
                template.spec)
    assert dep_spec.service_account_name == "kumo-temporal"

    # Both Job creates land on the same mock; inspect each call's body.
    job_specs = [
        c.kwargs["body"].spec.template.spec
        for c in batch.create_namespaced_job.call_args_list
    ]
    assert len(job_specs) == 2
    for spec in job_specs:
        assert spec.service_account_name == "kumo-temporal"


def test_provision_pool_worker_translates_env_dict_for_daemon():
    """Daemon container picks up the caller-supplied ``env`` dict
    verbatim — same faithful-translator contract the activity pod
    already follows. The manager doesn't peek at keys; whatever the
    daemon image needs (e.g. POOL_WORKER_UID, DISK_ROOT) is the
    caller's responsibility to compose.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    core.list_namespaced_pod.return_value = _registered_daemon_for("uid")

    caller_env = {
        "POOL_WORKER_UID": "uid",
        "DISK_ROOT": "/mnt/localssd",
        "EBS_SCRATCH_ROOT": "/mnt/ebs-scratch",
    }
    mgr = _make_manager(core=core, apps=apps)
    asyncio.run(
        mgr.provision_pool_worker(
            pool_worker_uid="uid",
            env=caller_env,
            timeout=2.0,
        ))

    dep_body = apps.create_namespaced_deployment.call_args.kwargs["body"]
    container = dep_body.spec.template.spec.containers[0]
    actual_env = {e.name: e.value for e in container.env}
    assert actual_env == caller_env


def test_provision_pool_worker_with_no_env_dict_uses_empty_env():
    """``env=None`` (default) → daemon V1Container.env is the empty
    list, not a manager-derived hardcoded set. Mirrors the regression
    coverage already in place for the activity-pod path so the same
    contract holds across all three components.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    core.list_namespaced_pod.return_value = _registered_daemon_for("uid")

    mgr = _make_manager(core=core, apps=apps)
    asyncio.run(
        mgr.provision_pool_worker(
            pool_worker_uid="uid",
            # env intentionally omitted.
            timeout=2.0,
        ))

    dep_body = apps.create_namespaced_deployment.call_args.kwargs["body"]
    container = dep_body.spec.template.spec.containers[0]
    assert container.env == []


def test_pool_taint_values_are_derived_from_tenant_for_both_pools():
    """Pool taint tolerations are tenant-derived per kumo-delivery's
    `dedicatedTaintValue` convention: cache pods tolerate
    `dedicated=diskgraph-<tenant>`, eph pods tolerate
    `dedicated=diskgraph-eph-<tenant>`. Pin the format-string
    derivation against a non-default tenant string so a future
    refactor that hardcodes the prefix or drops the per-pool split
    fails loudly.
    """
    custom_tenant = "acme-prod"
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)

    daemon_pods = _registered_daemon_for("uid")
    activity_pod = _make_pod(
        "act-pod",
        labels={
            LABEL_APP: LABEL_APP_VALUE_ACTIVITY,
            LABEL_POOL_WORKER_UID: "uid",
            LABEL_JOB_ID_HASH: "j-tnt",
        },
    )
    eph_pod = _make_pod(
        "eph-pod",
        labels={
            LABEL_APP: LABEL_APP_VALUE_ACTIVITY_EPH,
            LABEL_JOB_ID_HASH: "j-tnt-eph",
        },
    )
    core.list_namespaced_pod.side_effect = [
        daemon_pods,
        daemon_pods,
        V1PodList(items=[activity_pod]),
        V1PodList(items=[eph_pod]),
    ]
    mgr = K8sPoolWorkerManager(
        namespace=NAMESPACE,
        tenant=custom_tenant,
        storage_class="topolvm-ssd",
        daemon_image="kumo/daemon:test",
        daemon_command=["/usr/bin/daemon"],
        daemon_args=["--mode=test"],
        load_kube_config=False,
        core_v1_api=core,
        apps_v1_api=apps,
        batch_v1_api=batch,
        policy_v1_api=MagicMock(spec=k8s_client.PolicyV1Api),
    )
    asyncio.run(mgr.provision_pool_worker(
        pool_worker_uid="uid",
        timeout=2.0,
    ))
    asyncio.run(
        mgr.provision_activity_pod(
            pool_worker_uid="uid",
            job_id="j-tnt",
            snapshot_id="snap",
            activity="train",
            image="kumo/activity:test",
        ))
    asyncio.run(
        mgr.wait_for_activity_job_pod_running(
            job_name=mgr.activity_job_name("j-tnt", "durable"),
            timeout=2.0,
        ))
    asyncio.run(
        mgr.provision_ephemeral_activity_pod(
            job_id="j-tnt-eph",
            snapshot_id="snap",
            activity="ingest",
            image="kumo/activity:eph",
            ephemeral_storage_bytes=1 * (1 << 30),
        ))
    asyncio.run(
        mgr.wait_for_activity_job_pod_running(
            job_name=mgr.activity_job_name("j-tnt-eph", "overflow"),
            timeout=2.0,
        ))

    dep_tols = (apps.create_namespaced_deployment.call_args.kwargs["body"].spec.
                template.spec.tolerations)
    assert any(t.key == "dedicated" and t.value == f"diskgraph-{custom_tenant}"
               for t in dep_tols)
    job_calls = batch.create_namespaced_job.call_args_list
    assert len(job_calls) == 2
    cache_act_tols = (
        job_calls[0].kwargs["body"].spec.template.spec.tolerations)
    eph_act_tols = (job_calls[1].kwargs["body"].spec.template.spec.tolerations)
    assert any(t.key == "dedicated" and t.value == f"diskgraph-{custom_tenant}"
               for t in cache_act_tols)
    assert any(
        t.key == "dedicated" and t.value == f"diskgraph-eph-{custom_tenant}"
        for t in eph_act_tols)


# ---------------------------------------------------------------------------
# gpu_count parameter — `nvidia.com/gpu` extended-resource request/limit
# ---------------------------------------------------------------------------


def test_provision_activity_pod_with_gpu_count_zero_omits_gpu_resource():
    """``gpu_count=0`` (default) → CPU-only activity. The cache
    activity container's `resources` field stays unset, byte-for-byte
    matching the pre-`gpu_count` behavior — no ``nvidia.com/gpu``
    request, no scheduler arbitration, no NVIDIA device-plugin
    allocation.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    core.list_namespaced_pod.side_effect = [
        _registered_daemon_for("uid"),
        V1PodList(items=[
            _make_pod(
                "act",
                labels={
                    LABEL_APP: LABEL_APP_VALUE_ACTIVITY,
                    LABEL_POOL_WORKER_UID: "uid",
                    LABEL_JOB_ID_HASH: "j-cpu",
                },
            )
        ]),
    ]
    mgr = _make_manager(core=core, batch=batch)
    asyncio.run(
        mgr.provision_activity_pod(
            pool_worker_uid="uid",
            job_id="j-cpu",
            snapshot_id="snap",
            activity="ingest",
            image="kumo/activity:test",
            # gpu_count omitted — defaults to 0.
        ))
    body = batch.create_namespaced_job.call_args.kwargs["body"]
    container = body.spec.template.spec.containers[0]
    assert container.resources is None


def test_provision_activity_pod_with_gpu_count_positive_sets_gpu_resource():
    """``gpu_count=N`` (>0) stamps ``nvidia.com/gpu: '<N>'`` on BOTH
    requests AND limits — extended resources in K8s reject mismatched
    values, so the helper must mirror them.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    core.list_namespaced_pod.side_effect = [
        _registered_daemon_for("uid"),
        V1PodList(items=[
            _make_pod(
                "act",
                labels={
                    LABEL_APP: LABEL_APP_VALUE_ACTIVITY,
                    LABEL_POOL_WORKER_UID: "uid",
                    LABEL_JOB_ID_HASH: "j-gpu",
                },
            )
        ]),
    ]
    mgr = _make_manager(core=core, batch=batch)
    asyncio.run(
        mgr.provision_activity_pod(
            pool_worker_uid="uid",
            job_id="j-gpu",
            snapshot_id="snap",
            activity="train",
            image="kumo/activity:gpu",
            resources=ResourceSpec(gpu_count=2),
        ))
    body = batch.create_namespaced_job.call_args.kwargs["body"]
    container = body.spec.template.spec.containers[0]
    assert container.resources is not None
    assert container.resources.requests == {"nvidia.com/gpu": "2"}
    assert container.resources.limits == {"nvidia.com/gpu": "2"}


def test_provision_eph_activity_pod_with_gpu_count_zero_keeps_storage_only():
    """``gpu_count=0`` on the eph variant: ``ephemeral-storage`` is on
    requests (the scheduling floor) but deliberately NOT on limits — the
    dedicated eph node's full instance-store is usable (backlog #18).
    ``nvidia.com/gpu`` is absent.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod(
            "eph", labels={
                LABEL_APP: LABEL_APP_VALUE_ACTIVITY_EPH,
                LABEL_JOB_ID_HASH: "j-eph-cpu",
            }),
    ])
    eph_storage = 2 * (1 << 30)
    mgr = _make_manager(core=core, batch=batch)
    asyncio.run(
        mgr.provision_ephemeral_activity_pod(
            job_id="j-eph-cpu",
            snapshot_id="snap",
            activity="ingest",
            image="kumo/activity:eph",
            ephemeral_storage_bytes=eph_storage,
            # gpu_count omitted — defaults to 0.
        ))
    body = batch.create_namespaced_job.call_args.kwargs["body"]
    container = body.spec.template.spec.containers[0]
    assert container.resources.requests == {
        "ephemeral-storage": str(eph_storage),
    }
    # No ephemeral-storage limit (backlog #18); no other resources here.
    assert "ephemeral-storage" not in (container.resources.limits or {})


def test_provision_eph_activity_pod_with_gpu_count_positive_adds_gpu():
    """``gpu_count>0`` on the eph variant: ``ephemeral-storage`` is on
    requests only (scheduling floor, no limit — backlog #18), while
    ``nvidia.com/gpu`` is stamped on BOTH requests + limits.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod(
            "eph", labels={
                LABEL_APP: LABEL_APP_VALUE_ACTIVITY_EPH,
                LABEL_JOB_ID_HASH: "j-eph-gpu",
            }),
    ])
    eph_storage = 2 * (1 << 30)
    mgr = _make_manager(core=core, batch=batch)
    asyncio.run(
        mgr.provision_ephemeral_activity_pod(
            job_id="j-eph-gpu",
            snapshot_id="snap",
            activity="train",
            image="kumo/activity:eph-gpu",
            ephemeral_storage_bytes=eph_storage,
            resources=ResourceSpec(gpu_count=1),
        ))
    body = batch.create_namespaced_job.call_args.kwargs["body"]
    container = body.spec.template.spec.containers[0]
    assert container.resources.requests == {
        "ephemeral-storage": str(eph_storage),
        "nvidia.com/gpu": "1",
    }
    # ephemeral-storage intentionally absent from limits (backlog #18);
    # GPU stays request==limit.
    assert container.resources.limits == {
        "nvidia.com/gpu": "1",
    }


def test_resource_spec_rejects_negative_gpu_count():
    """Negative gpu_count is rejected at ResourceSpec construction
    via ``__post_init__`` — fail-fast on misconfig before any K8s
    API call.
    """
    with pytest.raises(ValueError, match="gpu_count must be >= 0"):
        ResourceSpec(gpu_count=-1)


# ---------------------------------------------------------------------------
# cpu / memory request + limit on daemon, activity, eph activity pods
# ---------------------------------------------------------------------------


def test_daemon_with_cpu_memory_limits_emits_guaranteed_qos_shape():
    """Setting both ``daemon_cpu_limit`` and ``daemon_memory_limit``
    (typically equal to the matching requests) promotes the daemon to
    Guaranteed QoS — kubelet won't evict it ahead of any Burstable
    pod under node pressure.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    core.list_namespaced_pod.return_value = _registered_daemon_for("uid")

    mgr = K8sPoolWorkerManager(
        namespace=NAMESPACE,
        tenant=TENANT,
        storage_class="topolvm-ssd",
        daemon_image="kumo/daemon:test",
        daemon_command=["/usr/bin/daemon"],
        daemon_args=["--mode=test"],
        daemon_resources=ResourceSpec(
            cpu_request="2",
            cpu_limit="2",
            memory_request="4Gi",
            memory_limit="4Gi",
        ),
        load_kube_config=False,
        core_v1_api=core,
        apps_v1_api=apps,
        batch_v1_api=MagicMock(spec=k8s_client.BatchV1Api),
        policy_v1_api=MagicMock(spec=k8s_client.PolicyV1Api),
    )
    asyncio.run(mgr.provision_pool_worker(
        pool_worker_uid="uid",
        timeout=2.0,
    ))

    dep_body = apps.create_namespaced_deployment.call_args.kwargs["body"]
    container = dep_body.spec.template.spec.containers[0]
    assert container.resources.requests == {"cpu": "2", "memory": "4Gi"}
    assert container.resources.limits == {"cpu": "2", "memory": "4Gi"}


def test_provision_activity_pod_with_cpu_memory_emits_resource_entries():
    """Caller-supplied cpu/memory request + limit show up on the
    activity container as the matching ``V1ResourceRequirements``
    entries. Combined with ``gpu_count`` they produce a Guaranteed
    QoS pod (modulo the GPU extended resource which is already
    request==limit by construction).
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    core.list_namespaced_pod.side_effect = [
        _registered_daemon_for("uid"),
        V1PodList(items=[
            _make_pod(
                "act",
                labels={
                    LABEL_APP: LABEL_APP_VALUE_ACTIVITY,
                    LABEL_POOL_WORKER_UID: "uid",
                    LABEL_JOB_ID_HASH: "j-cm",
                },
            )
        ]),
    ]
    mgr = _make_manager(core=core, batch=batch)
    asyncio.run(
        mgr.provision_activity_pod(
            pool_worker_uid="uid",
            job_id="j-cm",
            snapshot_id="snap",
            activity="train",
            image="kumo/activity:gpu",
            resources=ResourceSpec(
                cpu_request="8",
                cpu_limit="8",
                memory_request="32Gi",
                memory_limit="32Gi",
                gpu_count=1,
            ),
        ))
    body = batch.create_namespaced_job.call_args.kwargs["body"]
    container = body.spec.template.spec.containers[0]
    assert container.resources.requests == {
        "cpu": "8",
        "memory": "32Gi",
        "nvidia.com/gpu": "1",
    }
    assert container.resources.limits == {
        "cpu": "8",
        "memory": "32Gi",
        "nvidia.com/gpu": "1",
    }


def test_provision_eph_activity_pod_with_cpu_memory_emits_resource_entries():
    """Eph variant carries cpu/memory/gpu/ephemeral-storage on a
    single ``V1ResourceRequirements`` — verifies the unified helper
    composes all four cleanly.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod(
            "eph", labels={
                LABEL_APP: LABEL_APP_VALUE_ACTIVITY_EPH,
                LABEL_JOB_ID_HASH: "j-eph-cm",
            }),
    ])
    eph_storage = 4 * (1 << 30)
    mgr = _make_manager(core=core, batch=batch)
    asyncio.run(
        mgr.provision_ephemeral_activity_pod(
            job_id="j-eph-cm",
            snapshot_id="snap",
            activity="train",
            image="kumo/activity:eph-gpu",
            ephemeral_storage_bytes=eph_storage,
            resources=ResourceSpec(
                cpu_request="4",
                cpu_limit="8",
                memory_request="16Gi",
                memory_limit="32Gi",
                gpu_count=1,
            ),
        ))
    body = batch.create_namespaced_job.call_args.kwargs["body"]
    container = body.spec.template.spec.containers[0]
    assert container.resources.requests == {
        "cpu": "4",
        "memory": "16Gi",
        "ephemeral-storage": str(eph_storage),
        "nvidia.com/gpu": "1",
    }
    # ephemeral-storage intentionally absent from limits (backlog #18);
    # cpu/memory/gpu limits unchanged.
    assert container.resources.limits == {
        "cpu": "8",
        "memory": "32Gi",
        "nvidia.com/gpu": "1",
    }


# ---------------------------------------------------------------------------
# job_id hashing: K8s-label-illegal Temporal workflow_ids round-trip
# ---------------------------------------------------------------------------

# Real Temporal workflow_id shape: contains ':' (twice) and '@', plus
# total length above the 63-char K8s label-value cap when composed.
# Both characters are outside the [A-Za-z0-9._-] label charset, so the
# manager MUST hash this before stamping it as a label, and MUST hash
# it again before building any selector. The raw form must survive on
# the kumo.ai/job-id annotation.
_RAW_TEMPORAL_JOB_ID = (
    "dg-ingest:NA-abc123def@graph-data-version-456:01HXYZ789")


def test_hash_job_id_meets_k8s_label_value_constraints():
    """Sanity: the hash is short enough, charset-clean, and stable."""
    h = _hash_job_id(_RAW_TEMPORAL_JOB_ID)
    assert len(h) == 16  # _JOB_ID_HASH_HEX_LEN
    # K8s label values must match [A-Za-z0-9._-]; the hex subset is a
    # strict subset of that.
    assert all(c in "0123456789abcdef" for c in h)
    # Stable across calls — the selector path depends on this.
    assert _hash_job_id(_RAW_TEMPORAL_JOB_ID) == h


def test_provision_activity_pod_hashes_temporal_workflow_id():
    """Cache-pool activity Job stamps the hash on the label, the raw
    value on the annotation, and uses the hash in the Job name —
    nowhere should the raw ':'/'@' form leak into a K8s field with
    charset constraints (label value, Job name).
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    core.list_namespaced_pod.side_effect = [
        _registered_daemon_for("uid"),
        V1PodList(items=[
            _make_pod(
                "act",
                labels={
                    LABEL_APP: LABEL_APP_VALUE_ACTIVITY,
                    LABEL_POOL_WORKER_UID: "uid",
                    LABEL_JOB_ID_HASH: _hash_job_id(_RAW_TEMPORAL_JOB_ID),
                },
                annotations={ANNOTATION_JOB_ID: _RAW_TEMPORAL_JOB_ID},
            )
        ]),
    ]
    mgr = _make_manager(core=core, batch=batch)
    asyncio.run(
        mgr.provision_activity_pod(
            pool_worker_uid="uid",
            job_id=_RAW_TEMPORAL_JOB_ID,
            snapshot_id="snap",
            activity="ingest",
            image="kumo/activity:test",
        ))
    info = asyncio.run(
        mgr.wait_for_activity_job_pod_running(
            job_name=mgr.activity_job_name(_RAW_TEMPORAL_JOB_ID, "durable"),
            timeout=2.0,
        ))

    job_body = batch.create_namespaced_job.call_args.kwargs["body"]
    # Job name must be DNS-1123 subdomain compliant — no ':' / '@'.
    job_name = job_body.metadata.name
    assert ":" not in job_name and "@" not in job_name
    assert _hash_job_id(_RAW_TEMPORAL_JOB_ID) in job_name
    assert _RAW_TEMPORAL_JOB_ID not in job_name

    # Label = hash, raw value gone from labels entirely.
    labels = job_body.metadata.labels
    assert labels[LABEL_JOB_ID_HASH] == _hash_job_id(_RAW_TEMPORAL_JOB_ID)
    assert _RAW_TEMPORAL_JOB_ID not in labels.values()

    # Annotation = raw, on both Job and pod template.
    assert (job_body.metadata.annotations[ANNOTATION_JOB_ID] ==
            _RAW_TEMPORAL_JOB_ID)
    pod_template = job_body.spec.template.metadata
    assert (pod_template.annotations[ANNOTATION_JOB_ID] == _RAW_TEMPORAL_JOB_ID)

    # Caller sees the raw value back via ActivityPodInfo.
    assert info.job_id == _RAW_TEMPORAL_JOB_ID


def test_provision_ephemeral_activity_pod_hashes_temporal_workflow_id():
    """Eph variant: same hash-on-label / raw-on-annotation contract."""
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod(
            "eph",
            labels={
                LABEL_APP: LABEL_APP_VALUE_ACTIVITY_EPH,
                LABEL_JOB_ID_HASH: _hash_job_id(_RAW_TEMPORAL_JOB_ID),
            },
            annotations={ANNOTATION_JOB_ID: _RAW_TEMPORAL_JOB_ID},
        )
    ])
    mgr = _make_manager(core=core, batch=batch)
    asyncio.run(
        mgr.provision_ephemeral_activity_pod(
            job_id=_RAW_TEMPORAL_JOB_ID,
            snapshot_id="snap",
            activity="ingest",
            image="kumo/activity:eph",
            ephemeral_storage_bytes=1 * (1 << 30),
        ))
    info = asyncio.run(
        mgr.wait_for_activity_job_pod_running(
            job_name=mgr.activity_job_name(_RAW_TEMPORAL_JOB_ID, "overflow"),
            timeout=2.0,
        ))
    body = batch.create_namespaced_job.call_args.kwargs["body"]
    assert ":" not in body.metadata.name
    assert "@" not in body.metadata.name
    assert (body.metadata.labels[LABEL_JOB_ID_HASH] == _hash_job_id(
        _RAW_TEMPORAL_JOB_ID))
    assert (
        body.metadata.annotations[ANNOTATION_JOB_ID] == _RAW_TEMPORAL_JOB_ID)
    assert info.job_id == _RAW_TEMPORAL_JOB_ID


def test_delete_activity_pods_hashes_job_id_in_selector():
    """delete_activity_pods(job_id=raw) must hash the value into the
    label selector, otherwise K8s rejects the request as malformed.
    """
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    mgr = _make_manager(batch=batch)
    asyncio.run(mgr.delete_activity_pods(job_id=_RAW_TEMPORAL_JOB_ID))
    selector = (batch.delete_collection_namespaced_job.call_args.
                kwargs["label_selector"])
    assert (f"{LABEL_JOB_ID_HASH}={_hash_job_id(_RAW_TEMPORAL_JOB_ID)}"
            in selector)
    # Raw chars must never appear in the selector string.
    assert ":" not in selector
    assert "@" not in selector


def test_list_activity_pods_hashes_job_id_in_selector():
    """Same contract on the read path — list_activity_pods(job_id=raw)
    hashes before building the selector.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[])
    mgr = _make_manager(core=core)
    asyncio.run(mgr.list_activity_pods(job_id=_RAW_TEMPORAL_JOB_ID))
    selector = (core.list_namespaced_pod.call_args.kwargs["label_selector"])
    assert (f"{LABEL_JOB_ID_HASH}={_hash_job_id(_RAW_TEMPORAL_JOB_ID)}"
            in selector)
    assert ":" not in selector and "@" not in selector


# ---------------------------------------------------------------------------
# snapshot_id hashing: same constraint, same shape
# ---------------------------------------------------------------------------

# A snapshot id that would violate K8s label-value constraints if
# stamped raw: contains a uri-like ':' / '/' and uppercase chars.
_RAW_SNAPSHOT_ID = "s3://example-snapshots/v2:graph-NA-abc123:ts=01HZZZ"


def test_provision_activity_pod_hashes_snapshot_id():
    """Cache-pool activity Job: snapshot_id is hashed on the label
    and preserved raw on the annotation. Same K8s-safety rationale
    as job_id.
    """
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    core.list_namespaced_pod.side_effect = [
        _registered_daemon_for("uid"),
        V1PodList(items=[
            _make_pod(
                "act",
                labels={
                    LABEL_APP: LABEL_APP_VALUE_ACTIVITY,
                    LABEL_POOL_WORKER_UID: "uid",
                    LABEL_JOB_ID_HASH: _hash_job_id("j-snap"),
                    LABEL_SNAPSHOT_ID_HASH: _hash_snapshot_id(_RAW_SNAPSHOT_ID),
                },
                annotations={
                    ANNOTATION_JOB_ID: "j-snap",
                    ANNOTATION_SNAPSHOT_ID: _RAW_SNAPSHOT_ID,
                },
            )
        ]),
    ]
    mgr = _make_manager(core=core, batch=batch)
    asyncio.run(
        mgr.provision_activity_pod(
            pool_worker_uid="uid",
            job_id="j-snap",
            snapshot_id=_RAW_SNAPSHOT_ID,
            activity="ingest",
            image="kumo/activity:test",
        ))

    job_body = batch.create_namespaced_job.call_args.kwargs["body"]
    labels = job_body.metadata.labels
    assert (
        labels[LABEL_SNAPSHOT_ID_HASH] == _hash_snapshot_id(_RAW_SNAPSHOT_ID))
    # Raw ':' / '/' must NEVER leak into label values.
    assert _RAW_SNAPSHOT_ID not in labels.values()
    for v in labels.values():
        assert ":" not in v and "/" not in v

    # Annotation carries the raw value, on both Job and pod template.
    assert (job_body.metadata.annotations[ANNOTATION_SNAPSHOT_ID] ==
            _RAW_SNAPSHOT_ID)
    pod_template_annotations = job_body.spec.template.metadata.annotations
    assert (
        pod_template_annotations[ANNOTATION_SNAPSHOT_ID] == _RAW_SNAPSHOT_ID)


def test_provision_ephemeral_activity_pod_hashes_snapshot_id():
    """Eph variant: same contract."""
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod(
            "eph",
            labels={
                LABEL_APP: LABEL_APP_VALUE_ACTIVITY_EPH,
                LABEL_JOB_ID_HASH: _hash_job_id("j-eph-snap"),
                LABEL_SNAPSHOT_ID_HASH: _hash_snapshot_id(_RAW_SNAPSHOT_ID),
            },
            annotations={
                ANNOTATION_JOB_ID: "j-eph-snap",
                ANNOTATION_SNAPSHOT_ID: _RAW_SNAPSHOT_ID,
            },
        )
    ])
    mgr = _make_manager(core=core, batch=batch)
    asyncio.run(
        mgr.provision_ephemeral_activity_pod(
            job_id="j-eph-snap",
            snapshot_id=_RAW_SNAPSHOT_ID,
            activity="ingest",
            image="kumo/activity:eph",
            ephemeral_storage_bytes=1 * (1 << 30),
        ))
    body = batch.create_namespaced_job.call_args.kwargs["body"]
    labels = body.metadata.labels
    assert (
        labels[LABEL_SNAPSHOT_ID_HASH] == _hash_snapshot_id(_RAW_SNAPSHOT_ID))
    assert _RAW_SNAPSHOT_ID not in labels.values()
    assert (
        body.metadata.annotations[ANNOTATION_SNAPSHOT_ID] == _RAW_SNAPSHOT_ID)
    assert (body.spec.template.metadata.annotations[ANNOTATION_SNAPSHOT_ID] ==
            _RAW_SNAPSHOT_ID)


def test_hash_snapshot_id_charset_and_length():
    """Sanity: the snapshot_id hash also meets K8s label-value
    constraints (≤63 chars, hex-only charset, stable)."""
    h = _hash_snapshot_id(_RAW_SNAPSHOT_ID)
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)
    assert _hash_snapshot_id(_RAW_SNAPSHOT_ID) == h


# ---------------------------------------------------------------------------
# Cache-node-type node targeting (cloud-/autoscaler-neutral).
# ---------------------------------------------------------------------------


def test_daemon_uses_capacity_selectors_and_no_node_affinity():
    """The daemon pins the cache pool role via `kumo.ai/diskgraph-pool-type`
    plus the chosen SSD/GPU capacity labels, with NO node affinity."""
    from pool.worker_manager import PvcSpec

    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod(
            "p",
            labels={
                LABEL_APP: LABEL_APP_VALUE_DAEMON,
                LABEL_POOL_WORKER_UID: "uid",
            },
        ),
    ])

    mgr = _make_manager(core=core, apps=apps)
    # Explicit uid (deterministic readiness mock) AND a pvc_spec so the sized
    # fresh-provision path stamps the ssd-size label (= the PVC size).
    asyncio.run(
        mgr.provision_pool_worker(
            pool_worker_uid="uid",
            pvc_spec=PvcSpec(size_gib=1750, gpu_count=4),
            timeout=5.0,
        ))

    pod_spec = apps.create_namespaced_deployment.call_args.kwargs["body"] \
        .spec.template.spec
    assert pod_spec.node_selector == {
        DISKGRAPH_POOL_TYPE_LABEL: POOL_TYPE_CACHE,
        DISKGRAPH_SSD_SIZE_LABEL: "1750",
        DISKGRAPH_GPU_COUNT_LABEL: "4",
    }
    assert pod_spec.affinity.node_affinity is None
    # Isolation + structural guardrail preserved.
    assert any(t.key == "dedicated" and t.value == f"diskgraph-{TENANT}"
               for t in pod_spec.tolerations)
    assert pod_spec.affinity.pod_anti_affinity is not None


def test_daemon_rebind_keeps_role_omits_ssd_size():
    """On the rebind path (no pvc_spec), the bound PV pins the daemon to its
    node, so the size selector is omitted (a stale size could wrongly exclude
    the original node) — but the cache pool role is still pinned."""
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    core.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod(
            "p",
            labels={
                LABEL_APP: LABEL_APP_VALUE_DAEMON,
                LABEL_POOL_WORKER_UID: "uid",
            },
        ),
    ])

    mgr = _make_manager(core=core, apps=apps)
    asyncio.run(mgr.provision_pool_worker(pool_worker_uid="uid", timeout=5.0))

    pod_spec = apps.create_namespaced_deployment.call_args.kwargs["body"] \
        .spec.template.spec
    assert pod_spec.node_selector == {
        DISKGRAPH_POOL_TYPE_LABEL: POOL_TYPE_CACHE
    }
    assert pod_spec.affinity.node_affinity is None


def test_daemon_rebind_bound_pvc_keeps_capacity_selectors_with_pvc_spec():
    """A Bound portable PV cannot replace known SSD/GPU selectors."""
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    core.create_namespaced_persistent_volume_claim.side_effect = _api_exc(
        409, "already exists")
    pvc = MagicMock()
    pvc.status.phase = "Bound"
    core.read_namespaced_persistent_volume_claim.return_value = pvc
    core.list_namespaced_pod.return_value = V1PodList(items=[
        _make_pod(
            "p",
            labels={
                LABEL_APP: LABEL_APP_VALUE_DAEMON,
                LABEL_POOL_WORKER_UID: "uid",
            },
        ),
    ])

    mgr = _make_manager(core=core, apps=apps)
    asyncio.run(
        mgr.provision_pool_worker(
            pool_worker_uid="uid",
            pvc_spec=PvcSpec(size_gib=1750, gpu_count=4),
            timeout=5.0,
        ))

    pod_spec = apps.create_namespaced_deployment.call_args.kwargs["body"] \
        .spec.template.spec
    assert pod_spec.node_selector == {
        DISKGRAPH_POOL_TYPE_LABEL: POOL_TYPE_CACHE,
        DISKGRAPH_SSD_SIZE_LABEL: "1750",
        DISKGRAPH_GPU_COUNT_LABEL: "4",
    }


def test_cache_activity_pod_targets_cache_role_only():
    """Cache activity pods use the cache pool and daemon affinity."""
    core = MagicMock(spec=k8s_client.CoreV1Api)
    batch = MagicMock(spec=k8s_client.BatchV1Api)

    activity_pod = _make_pod(
        "act-pod-1",
        labels={
            LABEL_APP: LABEL_APP_VALUE_ACTIVITY,
            LABEL_POOL_WORKER_UID: "uid",
            LABEL_JOB_ID_HASH: _hash_job_id("j1"),
        },
        annotations={ANNOTATION_JOB_ID: "j1"},
    )
    core.list_namespaced_pod.side_effect = [
        _registered_daemon_for("uid"),
        V1PodList(items=[activity_pod]),
    ]

    mgr = _make_manager(core=core, batch=batch)
    asyncio.run(
        mgr.provision_activity_pod(
            pool_worker_uid="uid",
            job_id="j1",
            snapshot_id="s1",
            activity="train",
            image="kumo/activity:test",
        ))

    template_spec = batch.create_namespaced_job.call_args.kwargs["body"] \
        .spec.template.spec
    assert template_spec.node_selector == {
        DISKGRAPH_POOL_TYPE_LABEL: POOL_TYPE_CACHE
    }
    assert any(t.key == "dedicated" and t.value == f"diskgraph-{TENANT}"
               for t in template_spec.tolerations)


def _make_wired_manager(
        apps: Optional[MagicMock] = None) -> K8sPoolWorkerManager:
    """Manager configured with Temporal mTLS and shared env."""
    return K8sPoolWorkerManager(
        namespace=NAMESPACE,
        tenant=TENANT,
        storage_class="topolvm-ssd",
        daemon_image="kumo/daemon:test",
        daemon_command=["/usr/bin/daemon"],
        daemon_args=["--mode=test"],
        load_kube_config=False,
        pod_env_configmap="kumo-env",
        temporal_cert_secret="temporal-mtls",
        temporal_cert_dir="/etc/temporal/certs",
        tenant_secret_key_ref=SecretKeyRef(secret_name="kumo-acme-secret-key",
                                           key="credentials_encryption_key"),
        pod_template_labels=POD_TEMPLATE_LABELS,
        core_v1_api=MagicMock(spec=k8s_client.CoreV1Api),
        apps_v1_api=apps or MagicMock(spec=k8s_client.AppsV1Api),
        batch_v1_api=MagicMock(spec=k8s_client.BatchV1Api),
        policy_v1_api=MagicMock(spec=k8s_client.PolicyV1Api),
    )


def _built_daemon_pod_spec(mgr: K8sPoolWorkerManager, apps: MagicMock):
    """Run the daemon Deployment builder (which applies substrate wiring) and
    return the pod spec that would be created. Uses the daemon builder rather
    than provision_activity_pod so the test doesn't depend on a live daemon
    pod lookup."""
    mgr._create_daemon_deployment_sync("dep-uid", "pvc-uid", "uid",
                                       {"DISK_ROOT": "/mnt/localssd"},
                                       ssd_size_gib=800)
    return apps.create_namespaced_deployment.call_args.kwargs["body"] \
        .spec.template.spec


def _make_gcp_wired_manager(
    *,
    apps: Optional[MagicMock] = None,
    batch: Optional[MagicMock] = None,
) -> K8sPoolWorkerManager:
    """Manager configured with a selected key from a customer Secret."""
    return K8sPoolWorkerManager(
        namespace=NAMESPACE,
        tenant=TENANT,
        storage_class="topolvm-ssd",
        daemon_image="kumo/daemon:test",
        daemon_command=["/usr/bin/daemon"],
        daemon_args=["--mode=test"],
        load_kube_config=False,
        pod_env_configmap="kumo-config",
        gcp_application_credentials_secret="gcp-application-credentials",
        gcp_application_credentials_secret_key="credentials.json",
        gcp_application_credentials_path="/customer/adc/runtime-key.json",
        core_v1_api=MagicMock(spec=k8s_client.CoreV1Api),
        apps_v1_api=apps or MagicMock(spec=k8s_client.AppsV1Api),
        batch_v1_api=batch or MagicMock(spec=k8s_client.BatchV1Api),
        policy_v1_api=MagicMock(spec=k8s_client.PolicyV1Api),
    )


def _assert_gcp_credentials_wiring(spec: V1PodSpec) -> None:
    assert spec.containers is not None
    container = spec.containers[0]
    assert container.env_from is not None
    assert any(source.config_map_ref.name == "kumo-config"
               for source in container.env_from)
    assert spec.volumes is not None
    volume = next(v for v in spec.volumes
                  if v.name == "gcp-application-credentials")
    assert volume.secret.secret_name == "gcp-application-credentials"
    assert volume.secret.default_mode == 292
    assert [(item.key, item.path) for item in volume.secret.items] == [
        ("credentials.json", "runtime-key.json"),
    ]
    assert container.volume_mounts is not None
    mount = next(m for m in container.volume_mounts
                 if m.name == "gcp-application-credentials")
    assert mount.mount_path == "/customer/adc"
    assert mount.read_only is True


def test_daemon_pod_mounts_selected_gcp_credentials_key():
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    spec = _built_daemon_pod_spec(_make_gcp_wired_manager(apps=apps), apps)

    _assert_gcp_credentials_wiring(spec)


@pytest.mark.parametrize("pool_type", ["cache", "ephemeral"])
def test_activity_jobs_mount_selected_gcp_credentials_key(pool_type):
    batch = MagicMock(spec=k8s_client.BatchV1Api)
    mgr = _make_gcp_wired_manager(batch=batch)

    if pool_type == "cache":
        mgr._create_activity_job_sync(
            "job-cache",
            "uid",
            "pvc-uid",
            "job-id",
            "snapshot-id",
            "train",
            "kumo/activity:test",
            None,
            None,
            None,
            None,
            2,
        )
    else:
        mgr._create_eph_activity_job_sync(
            "job-ephemeral",
            "job-id",
            "snapshot-id",
            "predict",
            "kumo/activity:test",
            None,
            None,
            None,
            None,
            10 * (1 << 30),
            2,
        )

    spec = batch.create_namespaced_job.call_args.kwargs[
        "body"].spec.template.spec
    _assert_gcp_credentials_wiring(spec)


def test_daemon_pod_substrate_wiring_when_configured():
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    spec = _built_daemon_pod_spec(_make_wired_manager(apps=apps), apps)
    labels = apps.create_namespaced_deployment.call_args.kwargs[
        "body"].spec.template.metadata.labels
    assert labels["azure.workload.identity/use"] == "true"
    container = spec.containers[0]
    assert any(e.config_map_ref and e.config_map_ref.name == "kumo-env"
               for e in (container.env_from or []))
    vol_secrets = {
        v.name: v.secret.secret_name
        for v in spec.volumes if v.secret
    }
    assert vol_secrets.get("temporal-mtls") == "temporal-mtls"
    mount_paths = {m.name: m.mount_path for m in container.volume_mounts}
    assert mount_paths.get("temporal-mtls") == "/etc/temporal/certs"
    assert all(v.name != "gcp-application-credentials" for v in spec.volumes)
    assert all(m.name != "gcp-application-credentials"
               for m in container.volume_mounts)


def test_daemon_pod_mounts_statedb_secret_when_password_auth():
    """Password auth resolves dsn_secret_name as a file under
    secrets.mount_path. Runtime pool pods open StateDB too, so the Secret has
    to be mounted there or every spawned pod fails to connect."""
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    mgr = K8sPoolWorkerManager(
        namespace=NAMESPACE,
        tenant=TENANT,
        storage_class="topolvm-ssd",
        daemon_image="kumo/daemon:test",
        daemon_command=["/usr/bin/daemon"],
        daemon_args=["--mode=test"],
        load_kube_config=False,
        statedb_secret="kumo-statedb",
        secrets_mount_path="/var/run/secrets/kumo",
        core_v1_api=MagicMock(spec=k8s_client.CoreV1Api),
        apps_v1_api=apps,
        batch_v1_api=MagicMock(spec=k8s_client.BatchV1Api),
        policy_v1_api=MagicMock(spec=k8s_client.PolicyV1Api),
    )
    spec = _built_daemon_pod_spec(mgr, apps)
    vol_secrets = {
        v.name: v.secret.secret_name
        for v in spec.volumes if v.secret
    }
    assert vol_secrets.get("kumo-secrets") == "kumo-statedb"
    mount_paths = {
        m.name: m.mount_path
        for m in spec.containers[0].volume_mounts
    }
    assert mount_paths.get("kumo-secrets") == "/var/run/secrets/kumo"


def test_daemon_pod_has_no_statedb_mount_under_iam_auth():
    """With no StateDB Secret configured — the IAM path, whose connection
    details are inline in the deploy config — neither the connection Secret nor
    the CA is mounted into pool pods."""
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    spec = _built_daemon_pod_spec(_make_wired_manager(apps=apps), apps)
    for name in ("kumo-secrets", "kumo-statedb-ca"):
        assert all(v.name != name for v in spec.volumes)
        assert all(m.name != name for m in spec.containers[0].volume_mounts)


def test_daemon_pod_tenant_secret_env_when_configured():
    """Runtime-created pool pods get the per-tenant secret-key env
    (FERNET_SECRET_NAME redirect + secretKeyRef values), mirroring the
    chart's delivery to chart-managed workloads. Without it, credential
    encryption on a non-preprod stage dies with SecretNotFoundException."""
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    spec = _built_daemon_pod_spec(_make_wired_manager(apps=apps), apps)
    env = {e.name: e for e in spec.containers[0].env}
    assert env["FERNET_SECRET_NAME"].value == "credentials_encryption_key"
    for name in ("credentials_encryption_key", "API_KEY_PEPPER",
                 "PASSWORD_RESET_TOKEN_PEPPER"):
        ref = env[name].value_from.secret_key_ref
        assert ref.name == "kumo-acme-secret-key"
        assert ref.key == "credentials_encryption_key"


def test_daemon_pod_tenant_secret_env_absent_when_unconfigured():
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    spec = _built_daemon_pod_spec(_make_manager(apps=apps), apps)
    env_names = {e.name for e in spec.containers[0].env}
    assert "FERNET_SECRET_NAME" not in env_names
    assert "credentials_encryption_key" not in env_names


def test_daemon_pod_no_substrate_wiring_by_default():
    """Managed-SaaS default (no wiring configured) leaves pool pods unchanged:
    no env-from, no temporal/secret mounts."""
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    spec = _built_daemon_pod_spec(_make_manager(apps=apps), apps)
    assert not (spec.containers[0].env_from or [])
    vol_names = {v.name for v in (spec.volumes or [])}
    assert "temporal-mtls" not in vol_names


def test_daemon_pod_mounts_statedb_ca_beside_the_connection_secret():
    """A CA the image does not already trust (Cloud SQL's per-instance CA, or an
    internal PKI) has to reach pool pods too, at the path tls.sslrootcert names.
    It mounts in a subdirectory so it cannot shadow the connection Secret."""
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    mgr = K8sPoolWorkerManager(
        namespace=NAMESPACE,
        tenant=TENANT,
        storage_class="topolvm-ssd",
        daemon_image="kumo/daemon:test",
        daemon_command=["/usr/bin/daemon"],
        daemon_args=["--mode=test"],
        load_kube_config=False,
        statedb_secret="kumo-statedb",
        statedb_ca_secret="cloudsql-ca",
        statedb_ca_secret_key="ca.crt",
        secrets_mount_path="/var/run/secrets/kumo",
        core_v1_api=MagicMock(spec=k8s_client.CoreV1Api),
        apps_v1_api=apps,
        batch_v1_api=MagicMock(spec=k8s_client.BatchV1Api),
        policy_v1_api=MagicMock(spec=k8s_client.PolicyV1Api),
    )
    spec = _built_daemon_pod_spec(mgr, apps)
    vol_secrets = {
        v.name: v.secret.secret_name
        for v in spec.volumes if v.secret
    }
    assert vol_secrets.get("kumo-statedb-ca") == "cloudsql-ca"
    mount_paths = {
        m.name: m.mount_path
        for m in spec.containers[0].volume_mounts
    }
    assert mount_paths.get("kumo-statedb-ca") == "/var/run/secrets/kumo/ca"
    # The connection Secret still owns the parent directory.
    assert mount_paths.get("kumo-secrets") == "/var/run/secrets/kumo"
    # The source key is remapped to a fixed filename, so tls.sslrootcert is the
    # same constant whatever the operator's Secret happens to call it.
    ca_vol = next(v for v in spec.volumes if v.name == "kumo-statedb-ca")
    assert [(i.key, i.path)
            for i in ca_vol.secret.items] == [("ca.crt", "ca.pem")]
