# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :mod:`pool.worker_manager.init_canary` — the init-time
node-pool validation canary.

Mocks CoreV1Api (no cluster) and the grace-loop clock / sleep so the
tests run in <1s. Validates: per-tier canary spec (nodeSelector /
toleration / GPU / PVC size), success-path delete, unschedulable-past-
grace failure classification (label vs GPU), PVC-unbound-past-grace disk
failure, and cached-fresh-result skip.
"""
# Stub injection for laptop runs — mirrors test_k8s.py.
import os as _os  # noqa: E402,I100
import sys as _sys  # noqa: E402

if _os.environ.get("KUMO_WORKER_MANAGER_TEST_STUBS") == "1":
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "_stubs"))

import asyncio  # noqa: E402
from dataclasses import FrozenInstanceError  # noqa: E402
from typing import Optional  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402
from kubernetes import client as k8s_client  # noqa: E402
from kubernetes.client.models import (
    V1Deployment,
    V1ObjectMeta,
    V1PersistentVolumeClaim,
    V1PersistentVolumeClaimStatus,
    V1Pod,
    V1PodCondition,
    V1PodList,
    V1PodSpec,
    V1PodStatus,
)
from kubernetes.client.rest import ApiException

import pool.worker_manager.init_canary as canary_mod
from pool.worker_manager.init_canary import (
    CANARY_APP_LABEL_VALUE,
    RESULT_ANNOTATION_KEY,
    InitCanaryValidationError,
    _build_canary_pod,
    _build_canary_pvc,
    _CanaryRunner,
    _pvc_is_reusable,
    _read_cached_result,
    _record_success,
    pool_config_hash,
    run_init_canary,
)
from pool.worker_manager.k8s import (
    DISKGRAPH_GPU_COUNT_LABEL,
    DISKGRAPH_POOL_TYPE_LABEL,
    DISKGRAPH_SSD_SIZE_LABEL,
    POOL_TYPE_CACHE,
    POOL_TYPE_EPHEMERAL,
    K8sPoolWorkerManager,
    _unschedulable_reason,
)
from pool.worker_manager.types import PvcSpec

NAMESPACE = "kumo-dataplane-acme"
TENANT = "acme"
STORAGE_CLASS = "topolvm-ssd"
_GPU_KEY = "nvidia.com/gpu"


def _make_manager(
    core: MagicMock,
    apps: Optional[MagicMock] = None,
) -> K8sPoolWorkerManager:
    return K8sPoolWorkerManager(
        namespace=NAMESPACE,
        tenant=TENANT,
        storage_class=STORAGE_CLASS,
        daemon_image="reg/kumo-backend-cpu:1.0",
        load_kube_config=False,
        core_v1_api=core,
        apps_v1_api=apps or MagicMock(spec=k8s_client.AppsV1Api),
        batch_v1_api=MagicMock(spec=k8s_client.BatchV1Api),
        policy_v1_api=MagicMock(spec=k8s_client.PolicyV1Api),
    )


def _apps_no_cache() -> MagicMock:
    """AppsV1Api whose controller-Deployment read returns no validation
    annotation (404) → no cached result, forces a fresh validation."""
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    apps.read_namespaced_deployment.side_effect = _api_exc(404)
    return apps


def _apps_with_annotation(value: str) -> MagicMock:
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    apps.read_namespaced_deployment.return_value = V1Deployment(
        metadata=V1ObjectMeta(
            name="kumo-poolcontroller",
            annotations={RESULT_ANNOTATION_KEY: value},
        ))
    return apps


def _api_exc(status: int, body: str = "") -> ApiException:
    exc = ApiException(status=status)
    exc.body = body
    return exc


def _pvc_spec_factory(tier_gib: int) -> PvcSpec:
    return PvcSpec(size_gib=tier_gib)


def _running_pod() -> V1Pod:
    return V1Pod(
        metadata=V1ObjectMeta(name="p", namespace=NAMESPACE),
        spec=V1PodSpec(containers=[]),
        status=V1PodStatus(phase="Running"),
    )


def _unschedulable_pod(message: str) -> V1Pod:
    return V1Pod(
        metadata=V1ObjectMeta(name="p", namespace=NAMESPACE),
        spec=V1PodSpec(containers=[]),
        status=V1PodStatus(
            phase="Pending",
            conditions=[
                V1PodCondition(
                    type="PodScheduled",
                    status="False",
                    reason="Unschedulable",
                    message=message,
                ),
            ],
        ),
    )


def _bound_pvc() -> V1PersistentVolumeClaim:
    return V1PersistentVolumeClaim(
        metadata=V1ObjectMeta(name="pvc", namespace=NAMESPACE),
        status=V1PersistentVolumeClaimStatus(phase="Bound"),
    )


def _pending_pvc() -> V1PersistentVolumeClaim:
    return V1PersistentVolumeClaim(
        metadata=V1ObjectMeta(name="pvc", namespace=NAMESPACE),
        status=V1PersistentVolumeClaimStatus(phase="Pending"),
    )


@pytest.fixture(autouse=True)
def _fast_grace(monkeypatch):  # pyright: ignore[reportUnusedFunction]
    """Shrink the grace window + poll interval so the wait loop iterates a
    few times then fails fast, using the REAL clock / sleep (patching the
    global ``time.monotonic`` would break the asyncio event loop, which
    uses it for scheduling). The failure-path tests cover ~0.1s of real
    polling; the success-path tests return on the first poll."""
    monkeypatch.setattr(canary_mod, "CANARY_GRACE_SECS", 0.1)
    monkeypatch.setattr(canary_mod, "_CANARY_POLL_INTERVAL_SECS", 0.01)
    # Bound the reconcile / best-effort cleanup waits too: the success-path
    # mocks keep returning a Running pod (it never "disappears"), so the
    # post-delete _wait_for_gone(pod) would otherwise spin for the full 30s.
    monkeypatch.setattr(canary_mod, "_RECONCILE_WAIT_SECS", 0.05)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_unschedulable_reason_reads_pod_scheduled_condition():
    pod = _unschedulable_pod("0/3 nodes are available: 3 Insufficient gpu.")
    assert "Insufficient gpu" in (_unschedulable_reason(pod) or "")


def test_unschedulable_reason_none_when_no_condition():
    assert _unschedulable_reason(_running_pod()) is None


def _hash(**overrides) -> str:
    kw = dict(
        tiers_gib=[800, 1750],
        storage_class="sc",
        cache_node_selector_keys=["a"],
        cache_taint="dedicated=diskgraph-acme:NoSchedule",
        eph_taint="dedicated=diskgraph-eph-acme:NoSchedule",
    )
    kw.update(overrides)
    return pool_config_hash(**kw)  # type: ignore[arg-type]


def test_pool_config_hash_changes_with_tiers():
    base = _hash(tiers_gib=[800, 1750])
    changed = _hash(tiers_gib=[800])
    assert base != changed
    # Order-independent.
    assert base == _hash(tiers_gib=[1750, 800])


def test_pool_config_hash_stable_for_same_inputs():
    assert _hash() == _hash()


def test_pool_config_hash_changes_with_cache_taint():
    assert _hash() != _hash(cache_taint="dedicated=other:NoSchedule")


def test_pool_config_hash_changes_with_eph_taint():
    assert _hash() != _hash(eph_taint="dedicated=other-eph:NoSchedule")


def test_pool_config_hash_changes_with_eph_canary_enabled():
    # Toggling the opt-out eph canary must force re-validation so a cached
    # success recorded in the other mode isn't reused (kumo-ai/kumo#29865).
    assert _hash(eph_canary_enabled=False) != _hash(eph_canary_enabled=True)


def _classify(**overrides) -> str:
    """Invoke ``_CanaryRunner._classify_failure`` with sensible defaults."""
    runner = _CanaryRunner(_make_manager(MagicMock(spec=k8s_client.CoreV1Api)))
    kw = dict(
        pool="cache-800",
        expectation="some expectation",
        last_phase="unknown",
        last_unsched=None,
        pvc_name="dg-pool-canary-pvc-cache-800",
        last_pvc_phase=None,
    )
    kw.update(overrides)
    return runner._classify_failure(**kw)  # type: ignore[arg-type]


def test_classify_never_observed_pod_is_not_a_disk_failure():
    """FIX 2 regression: a pod that was never observed (last_phase stays
    'unknown') with a pvc_name set must NOT be classified as a disk /
    StorageClass failure — it should say the pod was never observed."""
    msg = _classify(
        last_phase="unknown",
        pvc_name="dg-pool-canary-pvc-cache-800",
        last_pvc_phase=None,
    )
    assert "never" in msg.lower() and "observed" in msg.lower()
    # Must NOT misattribute to disk / StorageClass.
    assert "StorageClass" not in msg
    assert "NVMe" not in msg


def test_classify_failed_phase_is_container_failure():
    """FIX 11: last_phase='Failed' → the container failed on a scheduled
    node, not a provisioning / image-pull stall."""
    msg = _classify(last_phase="Failed", pvc_name=None, last_pvc_phase=None)
    assert "container" in msg.lower()
    assert "OOMKilled" in msg or "crashed" in msg.lower()
    # Distinct from the cold-image-pull else branch.
    assert "image is still" not in msg


def test_classify_scheduled_not_running_is_cold_pull_not_misconfig():
    """A pod that SCHEDULED (no PodScheduled=False) but never reached
    Running is a cold image pull / slow provision on a delivered node — it
    must NOT be misattributed to a pool/label/taint misconfig, and must
    point operators at POOL_CANARY_GRACE_SECS for slow cold pulls."""
    msg = _classify(last_phase="Pending", last_unsched=None, pvc_name=None,
                    last_pvc_phase=None)
    assert "image" in msg.lower() and "pull" in msg.lower()
    assert "POOL_CANARY_GRACE_SECS" in msg
    # The node WAS delivered → not a misconfig, and the head must not claim
    # the pool cannot deliver a schedulable node.
    assert "not a pool" in msg.lower()
    assert "cannot deliver a" not in msg


# ---------------------------------------------------------------------------
# Canary pod spec mirrors the daemon / eph requirements
# ---------------------------------------------------------------------------


def test_cache_canary_pod_spec_has_selector_toleration_gpu_and_pvc():
    core = MagicMock(spec=k8s_client.CoreV1Api)
    mgr = _make_manager(core)
    sched = mgr.cache_canary_scheduling(800)
    pod = _build_canary_pod(
        name="dg-pool-canary-cache-800",
        namespace=NAMESPACE,
        labels={"app": CANARY_APP_LABEL_VALUE},
        sched=sched,
        service_account="kumo-temporal",
        image="img:test",
        pvc_name="dg-pool-canary-pvc-cache-800",
    )
    spec = pod.spec
    assert spec is not None
    assert spec.security_context.seccomp_profile.type == "RuntimeDefault"
    container_security = spec.containers[0].security_context
    assert container_security.privileged is False
    assert container_security.allow_privilege_escalation is False
    assert container_security.run_as_non_root is True
    assert container_security.capabilities.drop == ["ALL"]
    # nodeSelector mirrors the daemon's cache selector for the tier.
    assert spec.node_selector == {
        DISKGRAPH_POOL_TYPE_LABEL: POOL_TYPE_CACHE,
        DISKGRAPH_SSD_SIZE_LABEL: "800",
        DISKGRAPH_GPU_COUNT_LABEL: "1",
    }
    # Toleration mirrors the dedicated cache-pool taint.
    assert spec.tolerations is not None
    tol = spec.tolerations[0]
    assert tol.key == "dedicated"
    assert tol.value == f"diskgraph-{TENANT}"
    assert tol.effect == "NoSchedule"
    # GPU requested (validates device plugin).
    resources = spec.containers[0].resources
    assert resources is not None and resources.requests is not None
    assert resources.requests[_GPU_KEY] == "1"
    # PVC mounted (forces WaitForFirstConsumer bind → validates disk).
    assert spec.volumes is not None
    pvc_src = spec.volumes[0].persistent_volume_claim
    assert pvc_src is not None
    assert pvc_src.claim_name == "dg-pool-canary-pvc-cache-800"


def test_cache_canary_pod_spec_can_target_multi_gpu_tier():
    core = MagicMock(spec=k8s_client.CoreV1Api)
    mgr = _make_manager(core)
    sched = mgr.cache_canary_scheduling(3400, gpu_count=4)
    pod = _build_canary_pod(
        name="dg-pool-canary-cache-3400-gpu4",
        namespace=NAMESPACE,
        labels={"app": CANARY_APP_LABEL_VALUE},
        sched=sched,
        service_account="kumo-temporal",
        image="img:test",
        pvc_name="dg-pool-canary-pvc-cache-3400-gpu4",
        gpu_count=4,
    )
    spec = pod.spec
    assert spec is not None
    assert spec.node_selector == {
        DISKGRAPH_POOL_TYPE_LABEL: POOL_TYPE_CACHE,
        DISKGRAPH_SSD_SIZE_LABEL: "3400",
        DISKGRAPH_GPU_COUNT_LABEL: "4",
    }
    resources = spec.containers[0].resources
    assert resources is not None and resources.requests is not None
    assert resources.requests[_GPU_KEY] == "4"
    assert resources.limits is not None
    assert resources.limits[_GPU_KEY] == "4"


def test_eph_canary_pod_spec_has_selector_toleration_gpu_no_pvc():
    core = MagicMock(spec=k8s_client.CoreV1Api)
    mgr = _make_manager(core)
    sched = mgr.eph_canary_scheduling()
    pod = _build_canary_pod(
        name="dg-pool-canary-ephemeral",
        namespace=NAMESPACE,
        labels={"app": CANARY_APP_LABEL_VALUE},
        sched=sched,
        service_account="kumo-temporal",
        image="img:test",
        pvc_name=None,
    )
    spec = pod.spec
    assert spec is not None
    assert spec.node_selector == {
        DISKGRAPH_POOL_TYPE_LABEL: POOL_TYPE_EPHEMERAL
    }
    assert spec.tolerations is not None
    assert spec.tolerations[0].value == f"diskgraph-eph-{TENANT}"
    resources = spec.containers[0].resources
    assert resources is not None and resources.requests is not None
    assert resources.requests[_GPU_KEY] == "1"
    # Eph canary mounts no PVC (disk validated at job time).
    assert not spec.volumes


def test_cache_canary_pvc_sized_from_factory():
    """The canary PVC request must equal the daemon's PVC size for the
    tier (reuses the manager's pvc-size string)."""
    core = MagicMock(spec=k8s_client.CoreV1Api)
    mgr = _make_manager(core)
    assert mgr.cache_pvc_size_str(_pvc_spec_factory(1750)) == "1750Gi"


# ---------------------------------------------------------------------------
# run_init_canary — success / failure / cache paths
# ---------------------------------------------------------------------------


def _core_with_no_cache_result() -> MagicMock:
    core = MagicMock(spec=k8s_client.CoreV1Api)
    # reconcile_leftovers lists nothing.
    core.list_namespaced_pod.return_value = V1PodList(items=[])
    core.list_namespaced_persistent_volume_claim.return_value = MagicMock(
        items=[])
    return core


def _expected_config_hash(tiers, *, eph_enabled=False) -> str:
    return pool_config_hash(
        tiers_gib=tiers,
        tier_specs=[f"{tier}Gi/1gpu" for tier in sorted(set(tiers))],
        storage_class=STORAGE_CLASS,
        cache_node_selector_keys=[
            DISKGRAPH_POOL_TYPE_LABEL,
            DISKGRAPH_SSD_SIZE_LABEL,
            DISKGRAPH_GPU_COUNT_LABEL,
        ],
        cache_taint=f"dedicated=diskgraph-{TENANT}:NoSchedule",
        eph_taint=f"dedicated=diskgraph-eph-{TENANT}:NoSchedule",
        eph_canary_enabled=eph_enabled,
    )


def test_success_path_creates_and_deletes_canary_and_records_result():
    core = _core_with_no_cache_result()
    apps = _apps_no_cache()
    # Pod reads: Running immediately; PVC reads: Bound.
    core.read_namespaced_pod.return_value = _running_pod()
    core.read_namespaced_persistent_volume_claim.return_value = _bound_pvc()
    mgr = _make_manager(core, apps)

    asyncio.run(
        run_init_canary(mgr, cache_tiers_gib=[800],
                        pvc_spec_factory=_pvc_spec_factory))

    # Default (POOL_INIT_EPHEMERAL_CANARY_ENABLED enabled): the cache canary
    # (tier 800) AND the eph canary run - the eph tier is validated at init by
    # default (opt-out, kumo-ai/kumo#29865), so 2 pods are created and the eph
    # canary IS among them.
    assert core.create_namespaced_pod.call_count == 2
    created_names = [
        kw["body"].metadata.name
        for _, kw in core.create_namespaced_pod.call_args_list
    ]
    assert "dg-pool-canary-ephemeral" in created_names
    # Only the cache tier creates a PVC (the eph canary has no PVC).
    assert core.create_namespaced_persistent_volume_claim.call_count == 1
    # Success deletes both pods + the single cache PVC.
    assert core.delete_namespaced_pod.call_count == 2
    assert core.delete_namespaced_persistent_volume_claim.call_count == 1
    # Records success as an annotation on the controller Deployment; the hash
    # reflects the enabled eph canary.
    apps.patch_namespaced_deployment.assert_called_once()
    _, kw = apps.patch_namespaced_deployment.call_args
    ann = kw["body"]["metadata"]["annotations"][RESULT_ANNOTATION_KEY]
    assert ann.startswith(_expected_config_hash([800], eph_enabled=True) + "@")


def test_eph_canary_opt_out_skips_ephemeral_pool(monkeypatch):
    """With POOL_INIT_EPHEMERAL_CANARY_ENABLED=false (the single-GPU /
    no-ephemeral-nodepool opt-out), only the cache canary runs - the eph canary
    is skipped so controller readiness is not gated on an ephemeral node the
    deployment does not have (kumo-ai/kumo#29865)."""
    monkeypatch.setattr(canary_mod, "EPH_CANARY_ENABLED", False)
    core = _core_with_no_cache_result()
    apps = _apps_no_cache()
    core.read_namespaced_pod.return_value = _running_pod()
    core.read_namespaced_persistent_volume_claim.return_value = _bound_pvc()
    mgr = _make_manager(core, apps)

    asyncio.run(
        run_init_canary(mgr, cache_tiers_gib=[800],
                        pvc_spec_factory=_pvc_spec_factory))

    # Only the cache canary (tier 800) runs → 1 pod, eph NOT among them.
    assert core.create_namespaced_pod.call_count == 1
    created_names = [
        kw["body"].metadata.name
        for _, kw in core.create_namespaced_pod.call_args_list
    ]
    assert "dg-pool-canary-ephemeral" not in created_names
    # Success deletes the single cache pod + its PVC.
    assert core.delete_namespaced_pod.call_count == 1
    assert core.delete_namespaced_persistent_volume_claim.call_count == 1
    # Opting the eph canary out changes the config hash (forces re-validation
    # rather than reusing an enabled-eph cached success).
    _, kw = apps.patch_namespaced_deployment.call_args
    ann = kw["body"]["metadata"]["annotations"][RESULT_ANNOTATION_KEY]
    assert ann.startswith(_expected_config_hash([800], eph_enabled=False) + "@")
    assert not ann.startswith(
        _expected_config_hash([800], eph_enabled=True) + "@")


def _run_mocked_canary(monkeypatch, run_one, *,
                       tier_specs: Optional[list[PvcSpec]] = None,
                       prevalidated=None):
    monkeypatch.setattr(canary_mod, "EPH_CANARY_ENABLED", False)
    monkeypatch.setattr(canary_mod._CanaryRunner, "run_one", run_one)
    apps = _apps_no_cache()
    result = asyncio.run(
        run_init_canary(
            _make_manager(_core_with_no_cache_result(), apps),
            cache_tiers_gib=[800] if tier_specs is None else None,
            cache_tier_specs=tier_specs,
            pvc_spec_factory=_pvc_spec_factory,
            prevalidated_cache_tiers=prevalidated,
        ))
    return result, apps


async def _fail_if_called(*_args, **_kwargs):
    raise AssertionError("unexpected call")


def test_partial_cache_failure_keeps_other_proven_tiers(monkeypatch):
    calls: list[str] = []

    async def run_one(_runner, **kwargs):
        calls.append(kwargs["pool"])
        if kwargs["pool"] == "cache-800-gpu1":
            raise InitCanaryValidationError("capacity unavailable")

    result, apps = _run_mocked_canary(
        monkeypatch,
        run_one,
        tier_specs=[PvcSpec(400, 1),
                    PvcSpec(800, 1),
                    PvcSpec(3400, 4)],
        prevalidated={(400, 1), (999, 9)},
    )

    assert isinstance(result.available_cache_tiers, frozenset)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "ephemeral_available", True)
    assert result.available_cache_tiers == frozenset({(400, 1), (3400, 4)})
    assert calls == ["cache-800-gpu1", "cache-3400-gpu4"]
    assert not apps.patch_namespaced_deployment.called


def test_registered_tier_proof_is_not_persisted_as_global_success(monkeypatch):
    result, apps = _run_mocked_canary(monkeypatch, _fail_if_called,
                                      prevalidated={(800, 1)})

    assert result.available_cache_tiers == frozenset({(800, 1)})
    assert not apps.patch_namespaced_deployment.called


def test_unexpected_canary_error_still_propagates(monkeypatch):
    async def run_one(_runner, **_kwargs):
        raise RuntimeError("programming error")

    with pytest.raises(RuntimeError, match="programming error"):
        _run_mocked_canary(monkeypatch, run_one)


def test_unschedulable_label_misconfig_fails_with_tier_and_labels():
    core = _core_with_no_cache_result()
    apps = _apps_no_cache()
    core.read_namespaced_pod.return_value = _unschedulable_pod(
        "0/2 nodes are available: 2 node(s) didn't match Pod's node affinity/"
        "selector.")
    core.read_namespaced_persistent_volume_claim.return_value = _pending_pvc()
    mgr = _make_manager(core, apps)

    with pytest.raises(InitCanaryValidationError) as ei:
        asyncio.run(
            run_init_canary(mgr, cache_tiers_gib=[800],
                            pvc_spec_factory=_pvc_spec_factory))
    msg = str(ei.value)
    # Names the tier + the exact expected labels.
    assert "cache-800" in msg
    assert f"{DISKGRAPH_SSD_SIZE_LABEL}=800" in msg
    # An Unschedulable pod with a label-miss message is classified as a
    # LABEL / POOL MISCONFIG (the root cause), not a disk failure — even
    # though its WaitForFirstConsumer PVC also never bound.
    assert "LABEL / POOL MISCONFIG" in msg
    assert "didn't match" in msg
    assert "no cache tier is available" in msg
    # No success recorded.
    assert not apps.patch_namespaced_deployment.called


def test_insufficient_gpu_classified_as_gpu_failure():
    core = _core_with_no_cache_result()
    apps = _apps_no_cache()
    # Cache canary pod is Unschedulable for GPU; its PVC never binds as a
    # consequence — but the root cause is classified as GPU, not disk.
    core.read_namespaced_pod.return_value = _unschedulable_pod(
        "0/1 nodes are available: 1 Insufficient nvidia.com/gpu.")
    core.read_namespaced_persistent_volume_claim.return_value = _pending_pvc()
    mgr = _make_manager(core, apps)

    with pytest.raises(InitCanaryValidationError) as ei:
        asyncio.run(
            run_init_canary(mgr, cache_tiers_gib=[800],
                            pvc_spec_factory=_pvc_spec_factory))
    msg = str(ei.value)
    assert "INSUFFICIENT GPU" in msg
    assert "nvidia.com/gpu" in msg


def test_eph_gpu_failure_disables_ephemeral_and_names_gpu_cause(
        monkeypatch, caplog):
    """Cache tier passes (Running + Bound); eph canary hits Insufficient
    GPU → message classified as INSUFFICIENT GPU (no PVC to mislead it), and
    the eph failure names the POOL_INIT_EPHEMERAL_CANARY_ENABLED=false opt-out.

    The eph canary is enabled by default; pin it explicitly so the test is
    independent of the ambient env.

    Branch the pod read on name (not a finite side_effect list, which the
    cache canary's polling would exhaust → StopIteration wedging the loop):
    the cache canary pod is always Running; the eph canary pod is always
    Unschedulable on GPU.
    """
    monkeypatch.setattr(canary_mod, "EPH_CANARY_ENABLED", True)
    core = _core_with_no_cache_result()
    apps = _apps_no_cache()

    def _read_pod(name, namespace):
        if "ephemeral" in name:
            return _unschedulable_pod(
                "0/1 nodes are available: 1 Insufficient nvidia.com/gpu.")
        return _running_pod()  # cache canary → ready

    core.read_namespaced_pod.side_effect = _read_pod
    core.read_namespaced_persistent_volume_claim.return_value = _bound_pvc()
    mgr = _make_manager(core, apps)

    result = asyncio.run(
        run_init_canary(mgr, cache_tiers_gib=[800],
                        pvc_spec_factory=_pvc_spec_factory))

    assert result.available_cache_tiers == frozenset({(800, 1)})
    assert result.ephemeral_available is False
    assert not apps.patch_namespaced_deployment.called
    assert "ephemeral capacity is unavailable" in caplog.text
    assert "INSUFFICIENT GPU" in caplog.text
    assert "POOL_INIT_EPHEMERAL_CANARY_ENABLED=false" in caplog.text


def test_eph_failure_message_names_optout_but_cache_does_not():
    """The opt-out escape hatch (POOL_INIT_EPHEMERAL_CANARY_ENABLED=false) is
    appended ONLY to an ephemeral-pool failure - it is the fix for a missing
    ephemeral node pool, not for a cache-tier failure, so the cache failure
    message must NOT mention it."""
    label_miss = (
        "0/1 nodes are available: 1 node(s) didn't match Pod's node affinity/"
        "selector.")
    eph_msg = _classify(
        pool="ephemeral",
        last_phase="Pending",
        last_unsched=label_miss,
        pvc_name=None,
        last_pvc_phase=None,
    )
    assert "POOL_INIT_EPHEMERAL_CANARY_ENABLED=false" in eph_msg

    cache_msg = _classify(
        pool="cache-800",
        last_phase="Pending",
        last_unsched=label_miss,
        pvc_name="dg-pool-canary-pvc-cache-800",
        last_pvc_phase=None,
    )
    assert "POOL_INIT_EPHEMERAL_CANARY_ENABLED" not in cache_msg


def test_pvc_unbound_classified_as_disk_failure():
    """Cache canary pod schedules + runs but its PVC never binds → disk
    (TopoLVM capacity) failure naming the tier."""
    core = _core_with_no_cache_result()
    apps = _apps_no_cache()
    core.read_namespaced_pod.return_value = _running_pod()
    # TopoLVM reports insufficient capacity on the PVC; phase stays Pending.
    core.read_namespaced_persistent_volume_claim.return_value = _pending_pvc()
    mgr = _make_manager(core, apps)

    with pytest.raises(InitCanaryValidationError) as ei:
        asyncio.run(
            run_init_canary(mgr, cache_tiers_gib=[1750],
                            pvc_spec_factory=_pvc_spec_factory))
    msg = str(ei.value)
    assert "cache-1750" in msg
    assert "PVC" in msg and "Bound" in msg
    # Names the expected PVC size + StorageClass.
    assert "1750Gi" in msg
    assert STORAGE_CLASS in msg


def test_cached_fresh_result_skips_validation():
    import time as _t
    core = MagicMock(spec=k8s_client.CoreV1Api)
    # Fresh annotation matching the current config hash. Stamp a few seconds
    # in the PAST so ``:.0f`` rounding can't land the timestamp in the future
    # (which would make age < 0 and fail the ``0 <= age`` clock-skew guard).
    apps = _apps_with_annotation(
        f"{_expected_config_hash([800], eph_enabled=True)}@{_t.time() - 5:.0f}")
    mgr = _make_manager(core, apps)

    result = asyncio.run(
        run_init_canary(mgr, cache_tiers_gib=[800],
                        pvc_spec_factory=_pvc_spec_factory,
                        prevalidated_cache_tiers=_fail_if_called))
    # No canary launched.
    assert not core.create_namespaced_pod.called
    assert not core.create_namespaced_persistent_volume_claim.called
    assert result.available_cache_tiers == frozenset({(800, 1)})
    assert result.ephemeral_available is True


def test_stale_cached_result_revalidates():
    core = _core_with_no_cache_result()
    # Mismatched hash + ancient timestamp → re-validate.
    apps = _apps_with_annotation("deadbeef@1")
    core.read_namespaced_pod.return_value = _running_pod()
    core.read_namespaced_persistent_volume_claim.return_value = _bound_pvc()
    mgr = _make_manager(core, apps)

    asyncio.run(
        run_init_canary(mgr, cache_tiers_gib=[800],
                        pvc_spec_factory=_pvc_spec_factory))
    # Re-validated: canaries launched.
    assert core.create_namespaced_pod.called


def test_matching_hash_but_expired_ttl_revalidates(monkeypatch):
    """FIX 12: a cached annotation whose hash MATCHES the current config but
    whose age >= CANARY_RESULT_TTL_SECS must force re-validation (the
    'stale' test above only exercises a mismatched hash)."""
    import time as _t
    monkeypatch.setattr(canary_mod, "CANARY_RESULT_TTL_SECS", 100)
    core = _core_with_no_cache_result()
    # Matching hash, but stamped well beyond the TTL.
    expired_at = _t.time() - 1000
    apps = _apps_with_annotation(
        f"{_expected_config_hash([800], eph_enabled=True)}@{expired_at:.0f}")
    core.read_namespaced_pod.return_value = _running_pod()
    core.read_namespaced_persistent_volume_claim.return_value = _bound_pvc()
    mgr = _make_manager(core, apps)

    asyncio.run(
        run_init_canary(mgr, cache_tiers_gib=[800],
                        pvc_spec_factory=_pvc_spec_factory))
    # TTL expired despite the matching hash → re-validated.
    assert core.create_namespaced_pod.called


def test_empty_tiers_raises():
    core = MagicMock(spec=k8s_client.CoreV1Api)
    mgr = _make_manager(core)
    with pytest.raises(InitCanaryValidationError, match="no cache tiers"):
        asyncio.run(
            run_init_canary(mgr, cache_tiers_gib=[],
                            pvc_spec_factory=_pvc_spec_factory))


def test_reconcile_leftovers_uses_canary_label_only():
    """Reconcile must select by app=dg-pool-canary so daemon/activity pods
    are never deleted."""
    core = _core_with_no_cache_result()
    apps = _apps_no_cache()
    core.read_namespaced_pod.return_value = _running_pod()
    core.read_namespaced_persistent_volume_claim.return_value = _bound_pvc()
    mgr = _make_manager(core, apps)
    asyncio.run(
        run_init_canary(mgr, cache_tiers_gib=[800],
                        pvc_spec_factory=_pvc_spec_factory))
    # The reconcile list call filters on the canary app label.
    selectors = [
        kw.get("label_selector")
        for _, kw in core.list_namespaced_pod.call_args_list
    ]
    assert any(CANARY_APP_LABEL_VALUE in (s or "") for s in selectors)


def test_disabled_gate_skips_everything(monkeypatch):
    """POOL_INIT_CANARY_ENABLED=false → no validation, no k8s calls."""
    monkeypatch.setattr(canary_mod, "CANARY_ENABLED", False)
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    mgr = _make_manager(core, apps)

    result = asyncio.run(
        run_init_canary(mgr, cache_tiers_gib=[800],
                        pvc_spec_factory=_pvc_spec_factory,
                        prevalidated_cache_tiers=_fail_if_called))
    assert not core.create_namespaced_pod.called
    assert not apps.read_namespaced_deployment.called
    assert result.available_cache_tiers == frozenset({(800, 1)})
    assert result.ephemeral_available is True


# ---------------------------------------------------------------------------
# FIX A: recorded timestamp must never be in the future (int() truncates,
# not :.0f which rounds half-up) so a fresh result isn't rejected by the
# ``0 <= age`` guard on a fast restart.
# ---------------------------------------------------------------------------


def _apps_roundtrip_annotation() -> MagicMock:
    """AppsV1Api whose patch stores the annotation and whose subsequent read
    returns it — so _record_success → _read_cached_result round-trips."""
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    store: dict = {}

    def _patch(name, namespace, body):
        store.update(body["metadata"]["annotations"])
        return V1Deployment(metadata=V1ObjectMeta(name=name))

    def _read(name, namespace):
        return V1Deployment(
            metadata=V1ObjectMeta(name=name, annotations=dict(store)))

    apps.patch_namespaced_deployment.side_effect = _patch
    apps.read_namespaced_deployment.side_effect = _read
    return apps


def test_record_success_then_read_gives_non_negative_age():
    """FIX A: a freshly recorded success, read back immediately, must yield
    age >= 0 — i.e. the stamped epoch is not in the future, so the
    ``0 <= age`` clock-skew guard in run_init_canary doesn't reject it on a
    fast restart and force a needless GPU-node re-validation."""
    import time as _t
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = _apps_roundtrip_annotation()
    runner = _CanaryRunner(_make_manager(core, apps))

    asyncio.run(_record_success(runner, "cfg123"))
    cached = asyncio.run(_read_cached_result(runner))
    assert cached is not None
    cached_hash, validated_at = cached
    assert cached_hash == "cfg123"
    # int() truncates, so the stamp is <= now (never the future).
    age = _t.time() - validated_at
    assert age >= 0


# ---------------------------------------------------------------------------
# FIX C: a stale leftover PVC at a DIFFERENT requested size must trigger
# delete+recreate, never a silent "Bound" reuse / false success.
# ---------------------------------------------------------------------------


def _canary_pvc(size: str, *,
                terminating: bool = False) -> V1PersistentVolumeClaim:
    pvc = _build_canary_pvc(
        name="dg-pool-canary-pvc-cache-800",
        namespace=NAMESPACE,
        labels={"app": CANARY_APP_LABEL_VALUE},
        storage_class=STORAGE_CLASS,
        storage_size=size,
    )
    if terminating:
        assert pvc.metadata is not None
        pvc.metadata.deletion_timestamp = "2024-01-01T00:00:00Z"
    return pvc


def test_pvc_is_reusable_size_match_and_mismatch():
    """FIX C decision: a same-size, non-Terminating leftover is reusable; a
    different size or a Terminating leftover is NOT (would be a false
    success on an unvalidated / soon-gone PVC)."""
    assert _pvc_is_reusable(_canary_pvc("800Gi"), "800Gi")
    # Different requested size → NOT reusable (PVC size is immutable).
    assert not _pvc_is_reusable(_canary_pvc("400Gi"), "800Gi")
    # Terminating → NOT reusable even at the matching size.
    assert not _pvc_is_reusable(_canary_pvc("800Gi", terminating=True), "800Gi")


def test_create_pvc_size_mismatch_deletes_and_recreates():
    """FIX C: on a 409 whose existing PVC has a DIFFERENT requested size, the
    runner deletes the stale PVC, waits for it gone, then recreates at the
    correct size — it must NOT silently reuse the old-sized PVC."""
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    runner = _CanaryRunner(_make_manager(core, apps))

    # First create raises 409 (leftover exists); recreate after delete is OK.
    core.create_namespaced_persistent_volume_claim.side_effect = [
        _api_exc(409), None
    ]
    # The existing leftover is bound at the WRONG size (400Gi vs 800Gi).
    core.read_namespaced_persistent_volume_claim.side_effect = [
        _canary_pvc("400Gi"),  # 409 read: stale, different size
        _api_exc(404),  # _wait_for_gone(pvc): now gone
    ]

    asyncio.run(runner._create_pvc(_canary_pvc("800Gi"), storage_size="800Gi"))

    # Stale PVC deleted, then recreated at the correct size (2 creates total).
    assert core.delete_namespaced_persistent_volume_claim.call_count == 1
    assert core.create_namespaced_persistent_volume_claim.call_count == 2


def test_create_pod_failure_still_deletes_pvc():
    """A create-path failure (pod create raises AFTER the PVC was created)
    must still run the cleanup and delete the fresh PVC. Previously the
    creates ran BEFORE the try, so such a failure leaked the PVC — holding
    TopoLVM VG capacity across crashloop retries — until the next process
    start's reconcile_leftovers()."""
    core = MagicMock(spec=k8s_client.CoreV1Api)
    mgr = _make_manager(core)
    runner = _CanaryRunner(mgr)
    core.create_namespaced_pod.side_effect = _api_exc(500)
    # Cleanup reads: the pod was never created → 404; the PVC is gone right
    # after its delete → 404 (so the wait-for-gone returns immediately).
    core.read_namespaced_pod.side_effect = _api_exc(404)
    core.read_namespaced_persistent_volume_claim.side_effect = _api_exc(404)

    with pytest.raises(ApiException):
        asyncio.run(
            runner.run_one(
                pool="cache-800",
                sched=mgr.cache_canary_scheduling(800),
                pvc_size="800Gi",
                storage_class=STORAGE_CLASS,
                expectation="exp",
            ))

    # The PVC was created, then cleaned up despite the pod-create failure.
    assert core.create_namespaced_persistent_volume_claim.call_count == 1
    assert core.delete_namespaced_persistent_volume_claim.call_count == 1


def test_cleanup_waits_for_pvc_gone_after_delete():
    """After deleting the canary PVC, the cleanup must poll until the PVC
    is actually GONE (TopoLVM only releases the VG capacity once the
    deletion finalizes) instead of returning with it still Terminating."""
    core = MagicMock(spec=k8s_client.CoreV1Api)
    mgr = _make_manager(core)
    runner = _CanaryRunner(mgr)
    deleted = {"pod": False, "pvc": False}
    pvc_reads_after_delete: list = []

    def _delete_pod(name, namespace, grace_period_seconds=None):
        deleted["pod"] = True

    def _delete_pvc(name, namespace):
        deleted["pvc"] = True

    def _read_pod(name, namespace):
        if deleted["pod"]:
            raise _api_exc(404)
        return _running_pod()

    def _read_pvc(name, namespace):
        if deleted["pvc"]:
            pvc_reads_after_delete.append(name)
            raise _api_exc(404)
        return _bound_pvc()

    core.delete_namespaced_pod.side_effect = _delete_pod
    core.delete_namespaced_persistent_volume_claim.side_effect = _delete_pvc
    core.read_namespaced_pod.side_effect = _read_pod
    core.read_namespaced_persistent_volume_claim.side_effect = _read_pvc

    asyncio.run(
        runner.run_one(
            pool="cache-800",
            sched=mgr.cache_canary_scheduling(800),
            pvc_size="800Gi",
            storage_class=STORAGE_CLASS,
            expectation="exp",
        ))

    assert deleted["pvc"]
    # The cleanup re-read the PVC after its delete: the wait-for-gone ran.
    assert pvc_reads_after_delete


def test_create_pvc_same_size_reuses():
    """FIX C: a 409 whose existing PVC matches the requested size and is not
    Terminating is a safe reuse — no delete, no recreate."""
    core = MagicMock(spec=k8s_client.CoreV1Api)
    apps = MagicMock(spec=k8s_client.AppsV1Api)
    runner = _CanaryRunner(_make_manager(core, apps))

    core.create_namespaced_persistent_volume_claim.side_effect = [_api_exc(409)]
    core.read_namespaced_persistent_volume_claim.return_value = _canary_pvc(
        "800Gi")

    asyncio.run(runner._create_pvc(_canary_pvc("800Gi"), storage_size="800Gi"))

    assert not core.delete_namespaced_persistent_volume_claim.called
    # Only the initial (409) create attempt — no recreate.
    assert core.create_namespaced_persistent_volume_claim.call_count == 1
