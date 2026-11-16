# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the K8s placement strategy adapter."""
import asyncio
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import pytest

from pool.dataclasses import (
    K8sPlacementState,
    PlacementRequest,
    PoolState,
    ProvisionedHandle,
    PvcState,
    ResourceHint,
    SnapshotActivity,
    SnapshotLocalCopyRecord,
    SnapshotLocalCopyState,
    SnapshotPlacement,
    WorkerInventory,
    WorkerStateSnapshot,
    WorkerStatus,
)
from pool.errors import (
    DiskgraphCapacityError,
    ExistingWorkerUnavailableError,
    HomeStaleError,
)
from pool.k8s_pod import _FORWARDED_DEPLOY_ENV_KEYS
from pool.k8s_strategy import (
    POOL_ACTIVITY_ARGS,
    POOL_ACTIVITY_COMMAND,
    POOL_WORKER_UID_PREFIX_DURABLE,
    POOL_WORKER_UID_PREFIX_EPHNODE,
    STORAGE_MODE_ACTIVITY_ON_EMPTYDIR,
    STORAGE_MODE_ACTIVITY_ON_PVC,
    CapacityTier,
    K8sPlacementStrategy,
    ManagerPvcSpecFactory,
    activity_task_queue,
    build_k8s_placement_strategy,
    daemon_task_queue,
    deterministic_ephemeral_pool_worker_uid,
)
from pool.strategy import RouteDecision
from pool.timeouts import (
    ACTIVITY_POD_PROVISION_TIMEOUT_SEC,
    POOL_WORKER_PROVISION_TIMEOUT_SEC,
)
from pool.worker_manager import (
    PoolWorkerNotFoundError,
    PoolWorkerNotReadyError,
    ProvisionTimeoutError,
    PvcSpec,
)


@dataclass
class FakePoolWorkerInfo:
    pool_worker_uid: str
    status: str = 'Registered'
    created_at: Optional[float] = None
    deletion_started_at: Optional[float] = None
    extra_tolerations_match: bool = True


@dataclass
class FakeActivityPodInfo:
    pod_name: str


class FakeManager:
    # Mirror the read-only surface K8sPoolWorkerManager exposes so the
    # placement strategy can build observability metadata without
    # branching on test vs real manager.
    namespace = 'kumo-dataplane-fake'
    kube_context = 'example-cluster'

    def __init__(self) -> None:
        self.provision_pool_worker_calls: list[dict[str, Any]] = []
        self.provision_pool_worker_require_existing_resources: list[bool] = []
        self.provision_activity_pod_require_existing_resources: list[bool] = []
        self.delete_pool_worker_calls: list[dict[str, Any]] = []
        self.provision_activity_pod_calls: list[dict[str, Any]] = []
        self.provision_ephemeral_activity_pod_calls: list[dict[str, Any]] = []
        self.provision_pool_worker_envs: list[Optional[Mapping[str, str]]] = []
        self.wait_for_activity_job_pod_running_calls: list[dict[str, Any]] = []
        self.activity_running_pod_names: list[str] = []
        self.delete_activity_pods_calls: list[dict[str, Any]] = []
        self.live_workers: list[FakePoolWorkerInfo] = []
        self.next_pool_worker_uid = 'worker-new'
        self.provision_pool_worker_error: Optional[Exception] = None
        self.provision_activity_pod_error: Optional[Exception] = None
        self.registered_worker_matches_current_cache_tier_result = True
        self.registered_worker_matches_current_cache_tier_calls: list[
            tuple[str, PvcSpec]] = []
        # uid -> exception to raise from delete_pool_worker for that uid.
        self.delete_pool_worker_errors: dict[str, Exception] = {}

    @staticmethod
    def deployment_name_for_uid(pool_worker_uid: str) -> str:
        return f'diskgraph-daemon-{pool_worker_uid}'

    async def provision_pool_worker(
        self,
        pool_worker_uid: Optional[str] = None,
        pvc_spec: Any = None,
        env: Optional[Mapping[str, str]] = None,
        timeout: float = 600,
        require_existing_resources: bool = False,
    ) -> FakePoolWorkerInfo:
        self.provision_pool_worker_envs.append(env)
        self.provision_pool_worker_require_existing_resources.append(
            require_existing_resources)
        self.provision_pool_worker_calls.append({
            'pool_worker_uid': pool_worker_uid,
            'pvc_spec': pvc_spec,
            'timeout': timeout,
        })
        if self.provision_pool_worker_error is not None:
            raise self.provision_pool_worker_error
        return FakePoolWorkerInfo(
            pool_worker_uid=pool_worker_uid or self.next_pool_worker_uid)

    async def registered_worker_matches_current_cache_tier(
        self,
        worker: FakePoolWorkerInfo,
        spec: PvcSpec,
    ) -> bool:
        self.registered_worker_matches_current_cache_tier_calls.append(
            (worker.pool_worker_uid, spec))
        return self.registered_worker_matches_current_cache_tier_result

    async def delete_pool_worker(
        self,
        pool_worker_uid: str,
        grace_period_seconds: int = 30,
        scale_down_timeout: float = 120,
        pvc_delete_timeout: float = 60,
        force_on_timeout: bool = False,
    ) -> None:
        self.delete_pool_worker_calls.append({
            'pool_worker_uid':
            pool_worker_uid,
            'grace_period_seconds':
            grace_period_seconds,
            'scale_down_timeout':
            scale_down_timeout,
            'pvc_delete_timeout':
            pvc_delete_timeout,
            'force_on_timeout':
            force_on_timeout,
        })
        err = self.delete_pool_worker_errors.get(pool_worker_uid)
        if err is not None:
            raise err

    async def provision_activity_pod(
        self,
        pool_worker_uid: str,
        job_id: str,
        snapshot_id: str,
        activity: str,
        image: str,
        env: Optional[Mapping[str, str]] = None,
        command: Optional[Sequence[str]] = None,
        args: Optional[Sequence[str]] = None,
        resources: Any = None,
        backoff_limit: int = 2,
        timeout: float = 300,
        require_existing_resources: bool = False,
    ) -> FakeActivityPodInfo:
        self.provision_activity_pod_require_existing_resources.append(
            require_existing_resources)
        if self.provision_activity_pod_error is not None:
            raise self.provision_activity_pod_error
        self.provision_activity_pod_calls.append({
            'pool_worker_uid': pool_worker_uid,
            'job_id': job_id,
            'snapshot_id': snapshot_id,
            'activity': activity,
            'image': image,
            'env': env,
            'command': command,
            'args': args,
            'resources': resources,
            'backoff_limit': backoff_limit,
            'timeout': timeout,
        })
        return FakeActivityPodInfo(pod_name=f'pod-{job_id}')

    async def provision_ephemeral_activity_pod(
        self,
        job_id: str,
        snapshot_id: str,
        activity: str,
        image: str,
        env: Optional[Mapping[str, str]] = None,
        command: Optional[Sequence[str]] = None,
        args: Optional[Sequence[str]] = None,
        resources: Any = None,
        ephemeral_storage_bytes: int = 0,
        backoff_limit: int = 2,
        timeout: float = 300,
    ) -> FakeActivityPodInfo:
        self.provision_ephemeral_activity_pod_calls.append({
            'job_id':
            job_id,
            'snapshot_id':
            snapshot_id,
            'activity':
            activity,
            'image':
            image,
            'env':
            env,
            'command':
            command,
            'args':
            args,
            'resources':
            resources,
            'ephemeral_storage_bytes':
            ephemeral_storage_bytes,
            'backoff_limit':
            backoff_limit,
            'timeout':
            timeout,
        })
        return FakeActivityPodInfo(pod_name=f'eph-pod-{job_id}')

    async def list_pool_workers(self) -> list[FakePoolWorkerInfo]:
        return self.live_workers

    @staticmethod
    def activity_job_name(job_id: str, kind: str) -> str:
        return f'{kind}-{job_id}'

    async def wait_for_activity_job_pod_running(
        self,
        job_name: str,
        timeout: float,
    ) -> FakeActivityPodInfo:
        self.wait_for_activity_job_pod_running_calls.append({
            'job_name': job_name,
            'timeout': timeout,
        })
        if self.activity_running_pod_names:
            return FakeActivityPodInfo(
                pod_name=self.activity_running_pod_names.pop(0))
        return FakeActivityPodInfo(pod_name=f'pod-{job_name}')

    async def delete_activity_pods(self, **kwargs: Any) -> None:
        self.delete_activity_pods_calls.append(kwargs)


@pytest.fixture(autouse=True)
def clean_forwarded_deploy_env(monkeypatch):
    for key in _FORWARDED_DEPLOY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


class FakeK8sContext:
    def __init__(
        self,
        inventory: dict[str, WorkerInventory],
        warm: dict[str, str],
        sizes: dict[str, int],
        capacity_tiers: Optional[dict[str, CapacityTier]] = None,
    ) -> None:
        self._inventory = inventory
        self._warm = warm
        self._sizes = sizes
        self._capacity_tiers = capacity_tiers or {}

    @property
    def worker_inventory(self) -> dict[str, WorkerInventory]:
        return self._inventory

    def warm_location_uid(self, snapshot_id: str) -> Optional[str]:
        return self._warm.get(snapshot_id)

    def snapshot_size_bytes(self, snapshot_id: str) -> Optional[int]:
        return self._sizes.get(snapshot_id)

    def reserved_size_bytes_for_worker(self, pool_worker_uid: str) -> int:
        return 0

    def capacity_tier_for_worker(
            self, pool_worker_uid: str) -> Optional[CapacityTier]:
        return self._capacity_tiers.get(pool_worker_uid)


def _inventory(
    pool_worker_uid: str,
    *,
    status: WorkerStatus = WorkerStatus.READY,
    pvc_total_bytes: int = 2_000,
    pvc_free_bytes: int = 1_000,
    gpu_slots_total: int = 4,
    gpu_slots_in_use: int = 0,
) -> WorkerInventory:
    return WorkerInventory(
        pool_worker_uid=pool_worker_uid,
        status=status,
        last_heartbeat=1.0,
        gpu_slots_total=gpu_slots_total,
        gpu_slots_in_use=gpu_slots_in_use,
        state=WorkerStateSnapshot(
            ssd_total_bytes=pvc_total_bytes,
            ssd_used_bytes=max(pvc_total_bytes - pvc_free_bytes, 0),
        ),
    )


def _request(
    *,
    snapshot_id: str = 'snap-1',
    job_id: str = 'job-1',
    activity: SnapshotActivity = SnapshotActivity.TRAIN,
    source_size_bytes: int = 0,
    min_gpu_slots: int = 1,
    min_ssd_gib: int = 0,
) -> PlacementRequest:
    return PlacementRequest(
        snapshot_id=snapshot_id,
        job_id=job_id,
        workflow_id=job_id,
        activity=activity,
        resource_hint=ResourceHint(
            min_gpu_slots=min_gpu_slots,
            min_ssd_gib=min_ssd_gib,
        ),
        source_size_bytes=source_size_bytes,
    )


def _expected_activity_env(
    pool_worker_uid: str, job_id: str = 'job-1',
    storage_mode: str = (STORAGE_MODE_ACTIVITY_ON_PVC)
) -> dict[str, str]:
    return {
        'POOL_WORKER_UID': pool_worker_uid,
        'POOL_TASK_QUEUE': activity_task_queue(job_id),
        'POOL_STORAGE_MODE': storage_mode,
        'POOL_GPU_CONCURRENCY': '1',
        'POOL_MAX_CONCURRENT_ACTIVITIES': '1',
        'KUMO_REQUIRE_GPU': 'true',
        'KUMO_ACCELERATOR': 'gpu',
    }


def _default_hint_factory(ssd_gib: int) -> PvcSpec:
    return PvcSpec(size_gib=ssd_gib)


async def _accept_activity_poller(
    task_queue: str,
    pod_name: str,
    timeout: float,
) -> None:
    del task_queue, pod_name, timeout


def _strategy(
    manager: FakeManager,
    inventory: dict[str, WorkerInventory],
    warm: dict[str, str],
    sizes: dict[str, int],
    *,
    ssd_max_utilization: float = 1.0,
    capacity_tiers: Optional[list[CapacityTier]] = None,
    hint_factory: ManagerPvcSpecFactory = _default_hint_factory,
    activity_resources_factory=None,
    eph_tier_enabled: bool = True,
    activity_poller_waiter=_accept_activity_poller,
) -> K8sPlacementStrategy:
    capacity_tiers = capacity_tiers or [CapacityTier(tier_ssd_gib=1750)]
    return K8sPlacementStrategy(
        manager,  # type: ignore[arg-type]
        worker_inventory=lambda: inventory,
        warm_location=lambda snapshot_id: warm.get(snapshot_id),
        snapshot_size_bytes=lambda snapshot_id: sizes.get(snapshot_id),
        reserved_size_bytes_for_worker=lambda uid: 0,
        capacity_tiers=capacity_tiers,
        ssd_max_utilization=ssd_max_utilization,
        manager_pvc_spec_factory=hint_factory,  # type: ignore[arg-type]
        activity_resources_factory=activity_resources_factory,
        pool_worker_uid_factory=lambda: 'worker-new',
        eph_tier_enabled=eph_tier_enabled,
        activity_poller_waiter=activity_poller_waiter,
    )


def test_k8s_strategy_warm_available_worker_uses_durable_activity_pod():
    manager = FakeManager()
    poller_calls: list[tuple[str, str, float]] = []

    async def wait_for_poller(task_queue: str, pod_name: str,
                              timeout: float) -> None:
        poller_calls.append((task_queue, pod_name, timeout))

    inventory = {'worker-a': _inventory('worker-a', gpu_slots_in_use=0)}
    strategy = _strategy(
        manager,
        inventory,
        warm={'snap-1': 'worker-a'},
        sizes={},
        activity_poller_waiter=wait_for_poller,
    )

    async def drive():
        req = _request()
        decision = await strategy.route(req, warm_location_uid='worker-a')
        handle = await strategy.provision(req, decision)
        ready = await strategy.wait_provision_ready(handle, req.job_id)
        await strategy.release(req, ready)
        return decision, ready

    decision, handle = asyncio.run(drive())

    assert decision.path == 'durable'
    assert decision.pool_worker_uid == 'worker-a'
    assert handle.task_queue == activity_task_queue('job-1')
    assert handle.pool_worker_uid == 'worker-a'
    assert manager.provision_pool_worker_calls == [{
        'pool_worker_uid':
        'worker-a',
        'pvc_spec':
        PvcSpec(size_gib=1750),
        'timeout':
        600,
    }]
    assert manager.provision_pool_worker_envs == [{
        'POOL_WORKER_UID':
        'worker-a',
        'POOL_TASK_QUEUE':
        'diskgraph-daemon-worker-a',
        'POOL_STORAGE_MODE':
        'daemon-on-pvc',
        'POOL_GPU_CONCURRENCY':
        '1',
        'POOL_MAX_CONCURRENT_ACTIVITIES':
        '0',
    }]
    assert len(manager.provision_activity_pod_calls) == 1
    assert manager.provision_activity_pod_calls[0]['pool_worker_uid'] == (
        'worker-a')
    assert manager.provision_activity_pod_calls[0]['activity'] == 'TRAIN'
    assert '/kumo-backend-gpu:' in manager.provision_activity_pod_calls[0][
        'image']
    assert manager.provision_activity_pod_calls[0][
        'command'] == POOL_ACTIVITY_COMMAND
    assert manager.provision_activity_pod_calls[0]['args'] == POOL_ACTIVITY_ARGS
    assert manager.provision_activity_pod_calls[0][
        'env'] == _expected_activity_env('worker-a')
    assert manager.provision_ephemeral_activity_pod_calls == []
    assert handle.provision_state == 'ready'
    assert manager.wait_for_activity_job_pod_running_calls[0] == {
        'job_name': 'durable-job-1',
        'timeout': ACTIVITY_POD_PROVISION_TIMEOUT_SEC,
    }
    assert poller_calls == [(
        activity_task_queue('job-1'),
        'pod-durable-job-1',
        1.0,
    )]
    assert manager.delete_activity_pods_calls == [{
        'job_id': 'job-1',
        'wait_for_deletion': True,
    }]


def test_warm_worker_without_inventory_keeps_default_hint():
    strategy = _strategy(
        FakeManager(),
        {},
        warm={'snap-1': 'worker-unreported'},
        sizes={'snap-1': 100},
    )

    decision = asyncio.run(
        strategy.route(
            _request(activity=SnapshotActivity.GEN_TT),
            warm_location_uid='worker-unreported',
        ))

    assert decision.pool_worker_uid == 'worker-unreported'
    assert not decision.provisioned_new_worker


@pytest.mark.parametrize('stale_a_poller', [False, True])
def test_k8s_readiness_follows_current_activity_attempt(stale_a_poller):
    manager = FakeManager()
    manager.activity_running_pod_names = ['pod-a', 'pod-b', 'pod-b']
    poller_pods = []

    async def wait_for_poller(task_queue: str, pod_name: str,
                              timeout: float) -> None:
        del task_queue, timeout
        poller_pods.append(pod_name)
        if pod_name == 'pod-a' and not stale_a_poller:
            raise ProvisionTimeoutError('no poller')

    strategy = _strategy(
        manager,
        inventory={'worker-a': _inventory('worker-a')},
        warm={'snap-1': 'worker-a'},
        sizes={},
        activity_poller_waiter=wait_for_poller,
    )
    handle = ProvisionedHandle(
        kind='durable',
        pool_worker_uid='worker-a',
        task_queue=activity_task_queue('job-1'),
        provision_state='ready',
    )

    ready = asyncio.run(strategy.wait_provision_ready(handle, 'job-1'))

    assert ready.debug_info['pod_name'] == 'pod-b'
    assert poller_pods == ['pod-a', 'pod-b']
    assert len(manager.wait_for_activity_job_pod_running_calls) == 3


def test_k8s_strategy_warm_gpu_busy_uses_overflow_with_l2_check():
    manager = FakeManager()
    inventory = {
        'worker-a': _inventory(
            'worker-a',
            gpu_slots_total=4,
            gpu_slots_in_use=4,
        )
    }
    strategy = _strategy(
        manager,
        inventory,
        warm={'snap-1': 'worker-a'},
        sizes={'snap-1': 1000},
        eph_tier_enabled=True,
    )

    async def drive():
        req = _request()
        decision = await strategy.route(
            req,
            warm_location_uid='worker-a',
            overflow_viable=True,
        )
        handle = await strategy.provision(req, decision)
        return decision, handle

    decision, handle = asyncio.run(drive())

    assert decision.path == 'overflow'
    assert decision.pool_worker_uid is None
    assert decision.provisioned_new_worker is False
    assert handle.task_queue == activity_task_queue('job-1')
    assert handle.pool_worker_uid is None
    assert manager.provision_activity_pod_calls == []
    assert len(manager.provision_ephemeral_activity_pod_calls) == 1
    assert manager.provision_ephemeral_activity_pod_calls[0]['job_id'] == (
        'job-1')
    assert manager.provision_ephemeral_activity_pod_calls[0][
        'snapshot_id'] == 'snap-1'
    assert manager.provision_ephemeral_activity_pod_calls[0][
        'activity'] == 'TRAIN'
    assert manager.provision_ephemeral_activity_pod_calls[0][
        'command'] == POOL_ACTIVITY_COMMAND
    assert manager.provision_ephemeral_activity_pod_calls[0][
        'args'] == POOL_ACTIVITY_ARGS
    assert manager.provision_ephemeral_activity_pod_calls[0][
        'ephemeral_storage_bytes'] == 1200
    env = manager.provision_ephemeral_activity_pod_calls[0]['env']
    assert env['POOL_WORKER_UID'].startswith(POOL_WORKER_UID_PREFIX_EPHNODE)
    # Deterministic in (snapshot_id, job_id) so a retry of the same
    # workflow reports the same uid via heartbeats and lands on the
    # same K8s Job (paired with the manager's reclaim-always semantics).
    assert env['POOL_WORKER_UID'] == deterministic_ephemeral_pool_worker_uid(
        snapshot_id='snap-1', job_id='job-1')
    assert env == _expected_activity_env(
        env['POOL_WORKER_UID'],
        storage_mode=STORAGE_MODE_ACTIVITY_ON_EMPTYDIR,
    )


def test_k8s_strategy_warm_gpu_busy_backup_eph_disabled_serializes_on_warm():
    """Symmetric to the home-branch gate: when the ephemeral tier is opted out
    (``POOL_INIT_EPHEMERAL_CANARY_ENABLED=false``; the tier is enabled by
    default), a GPU consumer that hits its warm location but whose warm worker
    cannot run now must NOT overflow to eph even though an L2 backup exists -
    routing to eph would mint a fire-and-forget eph Job that hangs Pending
    forever on a deployment with no ephemeral node pool (the #29865 stall moved
    to job time). Instead it serializes on the warm durable worker, and
    provisions no eph Job."""
    manager = FakeManager()
    inventory = {
        'worker-a': _inventory(
            'worker-a',
            gpu_slots_total=4,
            gpu_slots_in_use=4,
        )
    }
    strategy = _strategy(
        manager,
        inventory,
        warm={'snap-1': 'worker-a'},
        sizes={'snap-1': 1000},
        eph_tier_enabled=False,
    )

    async def drive():
        req = _request()
        decision = await strategy.route(
            req,
            warm_location_uid='worker-a',
            overflow_viable=True,
        )
        handle = await strategy.provision(req, decision)
        return decision, handle

    decision, _ = asyncio.run(drive())

    assert decision.path == 'durable'
    assert decision.pool_worker_uid == 'worker-a'  # the warm worker, not eph
    assert decision.provisioned_new_worker is False
    assert decision.reason == 'ephemeral tier disabled: serialize on warm worker'
    # No eph Job minted: the overflow route is never taken with the tier off.
    assert manager.provision_ephemeral_activity_pod_calls == []


def test_k8s_strategy_warm_gpu_busy_respects_canary_ephemeral_availability():
    """With the ephemeral tier enabled (``eph_tier_enabled=True``), the same
    warm-branch inputs preserve the original behavior: a GPU consumer whose
    warm worker lacks activity capacity and has an L2 backup overflows to an
    eph pod."""
    manager = FakeManager()
    inventory = {
        'worker-a': _inventory(
            'worker-a',
            gpu_slots_total=4,
            gpu_slots_in_use=4,
        )
    }
    strategy = _strategy(
        manager,
        inventory,
        warm={'snap-1': 'worker-a'},
        sizes={'snap-1': 1000},
        eph_tier_enabled=True,
    )
    strategy.restrict_to_available_capacity(
        {(1750, 1)}, ephemeral_available=True)

    async def drive():
        req = _request()
        overflow_decision = await strategy.route(
            req,
            warm_location_uid='worker-a',
            overflow_viable=True,
        )
        handle = await strategy.provision(req, overflow_decision)
        strategy.restrict_to_available_capacity(
            {(1750, 1)}, ephemeral_available=False)
        durable_decision = await strategy.route(
            req, warm_location_uid='worker-a', overflow_viable=True)
        return overflow_decision, handle, durable_decision

    decision, handle, durable_decision = asyncio.run(drive())

    assert decision.path == 'overflow'
    assert decision.pool_worker_uid is None
    assert handle.pool_worker_uid is None
    assert manager.provision_pool_worker_calls == []
    assert len(manager.provision_ephemeral_activity_pod_calls) == 1
    assert durable_decision.path == 'durable'
    assert durable_decision.reason == (
        'ephemeral tier disabled: serialize on warm worker')


def test_canary_availability_blocks_new_tier_but_keeps_existing_reuse():
    manager = FakeManager()
    inventory: dict[str, WorkerInventory] = {}
    strategy = _strategy(
        manager,
        inventory,
        warm={},
        sizes={},
        capacity_tiers=[
            CapacityTier(tier_ssd_gib=800, tier_gpu_count=1),
            CapacityTier(tier_ssd_gib=3400, tier_gpu_count=4),
        ],
    )
    strategy.restrict_to_available_capacity(
        frozenset({(800, 1)}),
        ephemeral_available=True,
    )
    compatible_request = _request(
        activity=SnapshotActivity.GEN_TT,
        min_gpu_slots=1,
    )
    compatible_decision = asyncio.run(
        strategy.route(
            compatible_request,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=100,
        ))
    asyncio.run(strategy.provision(compatible_request, compatible_decision))
    assert compatible_decision.provisioned_new_worker is True
    assert compatible_decision.tier_ssd_gib == 800
    assert compatible_decision.tier_gpu_count == 1
    assert manager.provision_pool_worker_calls[0]['pvc_spec'] == PvcSpec(
        size_gib=800, gpu_count=1)

    request = _request(
        activity=SnapshotActivity.GEN_TT,
        min_gpu_slots=4,
    )

    with pytest.raises(DiskgraphCapacityError,
                       match='min_gpu_slots=4') as exc:
        asyncio.run(
            strategy.route(
                request,
                warm_location_uid=None,
                snapshot_size_estimate_bytes=100,
            ))
    assert 'configured maximum SSD tier is 3400 GiB' in str(exc.value)
    assert 'currently available maximum is 0 GiB' in str(exc.value)

    existing_size = 3400 * (1 << 30)
    inventory['worker-existing'] = _inventory(
        'worker-existing',
        pvc_total_bytes=existing_size,
        pvc_free_bytes=existing_size,
        gpu_slots_total=4,
    )
    decision = asyncio.run(
        strategy.route(
            request,
            warm_location_uid='worker-existing',
            snapshot_size_estimate_bytes=100,
        ))
    asyncio.run(strategy.provision(request, decision))
    assert decision.pool_worker_uid == 'worker-existing'
    assert decision.provisioned_new_worker is False
    assert manager.provision_pool_worker_calls[-1]['pvc_spec'] == PvcSpec(
        size_gib=3400, gpu_count=4)
    assert manager.provision_pool_worker_require_existing_resources == [
        False,
        True,
    ]
    assert manager.provision_activity_pod_require_existing_resources == [
        False,
        True,
    ]


def test_canary_allows_unknown_size_reuse_of_existing_worker():
    worker_size = 900 * (1 << 30)
    manager = FakeManager()
    inventory = {
        'worker-small':
        _inventory(
            'worker-small',
            pvc_total_bytes=worker_size,
            pvc_free_bytes=worker_size,
            gpu_slots_total=4,
        ),
    }
    strategy = _strategy(
        manager,
        inventory,
        warm={},
        sizes={},
        capacity_tiers=[
            CapacityTier(tier_ssd_gib=900, tier_gpu_count=4),
            CapacityTier(tier_ssd_gib=1800, tier_gpu_count=4),
        ],
    )
    strategy.restrict_to_available_capacity(
        {(900, 4)}, ephemeral_available=True)

    request = _request(
        activity=SnapshotActivity.GEN_TT,
        min_gpu_slots=4,
    )

    decision = asyncio.run(
        strategy.route(
            request,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=0,
        ))

    assert decision.pool_worker_uid == 'worker-small'
    assert decision.tier_ssd_gib == 900


def test_canary_restriction_reports_configured_and_available_size_maxima():
    strategy = _strategy(
        FakeManager(),
        {},
        warm={},
        sizes={},
        capacity_tiers=[
            CapacityTier(tier_ssd_gib=900, tier_gpu_count=4),
            CapacityTier(tier_ssd_gib=1800, tier_gpu_count=4),
        ],
    )
    strategy.restrict_to_available_capacity(
        {(900, 4)}, ephemeral_available=True)

    with pytest.raises(DiskgraphCapacityError,
                       match='preferred placement') as exc:
        asyncio.run(
            strategy.route(
                _request(
                    activity=SnapshotActivity.GEN_TT,
                    min_gpu_slots=4,
                    source_size_bytes=1000 * (1 << 30),
                ),
                warm_location_uid=None,
                snapshot_size_estimate_bytes=1000 * (1 << 30),
            ))

    assert 'configured maximum usable capacity is 1800.0 GiB' in str(exc.value)
    assert 'currently available maximum is 900.0 GiB' in str(exc.value)


def test_canary_reuses_existing_worker_at_streaming_minimum():
    gib = 1 << 30
    inventory = {
        'worker-existing':
        _inventory(
            'worker-existing',
            pvc_total_bytes=900 * gib,
            pvc_free_bytes=900 * gib,
        ),
    }
    strategy = _strategy(
        FakeManager(),
        inventory,
        warm={},
        sizes={},
        capacity_tiers=[
            CapacityTier(tier_ssd_gib=900),
            CapacityTier(tier_ssd_gib=1800),
        ],
    )
    strategy.restrict_to_available_capacity(set(), ephemeral_available=True)

    decision = asyncio.run(
        strategy.route(
            _request(
                activity=SnapshotActivity.GEN_TT,
                source_size_bytes=500 * gib,
            ),
            warm_location_uid=None,
            snapshot_size_estimate_bytes=1500 * gib,
        ))

    assert decision.pool_worker_uid == 'worker-existing'
    assert not decision.provisioned_new_worker
    assert decision.snapshot_size_estimate_bytes == 900 * gib


@pytest.mark.parametrize('manager_error', [
    PoolWorkerNotFoundError('PVC disappeared'),
    PoolWorkerNotReadyError('Deployment is terminating'),
])
@pytest.mark.parametrize('failure_point', ['pool-worker', 'activity'])
def test_canary_unavailable_tier_resource_race_fails_actionably(
        manager_error, failure_point):
    manager = FakeManager()
    if failure_point == 'pool-worker':
        manager.provision_pool_worker_error = manager_error
    else:
        manager.provision_activity_pod_error = manager_error
    inventory = {
        'worker-unavailable':
        _inventory(
            'worker-unavailable',
            pvc_total_bytes=3400 * (1 << 30),
            pvc_free_bytes=3400 * (1 << 30),
            gpu_slots_total=4,
        )
    }
    strategy = _strategy(
        manager,
        inventory,
        warm={},
        sizes={},
        capacity_tiers=[CapacityTier(800, 1), CapacityTier(3400, 4)],
    )
    strategy.restrict_to_available_capacity(
        {(800, 1)}, ephemeral_available=True)
    decision = RouteDecision(
        path='durable',
        pool_worker_uid='worker-unavailable',
        provisioned_new_worker=False,
        reason='pinned-on-retry',
        snapshot_size_estimate_bytes=100,
        tier_ssd_gib=3400,
        tier_gpu_count=4,
    )

    with pytest.raises(
            ExistingWorkerUnavailableError,
            match=r'canary-unavailable.*no longer has reusable live resources',
    ) as exc:
        asyncio.run(
            strategy.provision(
                _request(activity=SnapshotActivity.GEN_TT),
                decision,
            ))

    assert exc.value.__cause__ is manager_error
    assert manager.provision_pool_worker_require_existing_resources == [True]
    assert manager.provision_activity_pod_calls == []

    strategy.restrict_to_available_capacity(
        {(800, 1), (3400, 4)}, ephemeral_available=True)
    with pytest.raises(type(manager_error)) as available_exc:
        asyncio.run(strategy.provision(
            _request(activity=SnapshotActivity.GEN_TT), decision))
    assert available_exc.value is manager_error


def test_canary_availability_blocks_pinned_ephemeral_retry():
    manager = FakeManager()
    strategy = _strategy(manager, {}, warm={}, sizes={'snap-1': 1000})
    strategy.restrict_to_available_capacity(
        {(1750, 1)}, ephemeral_available=False)
    decision = RouteDecision(
        path='overflow',
        pool_worker_uid=None,
        provisioned_new_worker=False,
        reason='persisted retry',
    )

    with pytest.raises(RuntimeError, match='ephemeral capacity is unavailable'):
        asyncio.run(strategy.provision(_request(), decision))

    assert manager.provision_ephemeral_activity_pod_calls == []


def test_canary_availability_blocks_pinned_new_tier_but_not_existing_worker():
    manager = FakeManager()
    inventory: dict[str, WorkerInventory] = {}
    strategy = _strategy(
        manager,
        inventory,
        warm={},
        sizes={},
        capacity_tiers=[CapacityTier(800, 1), CapacityTier(3400, 4)],
    )
    strategy.restrict_to_available_capacity(
        {(800, 1)}, ephemeral_available=True)
    decision = RouteDecision(
        path='durable',
        pool_worker_uid='worker-pinned',
        provisioned_new_worker=True,
        reason='persisted retry',
        snapshot_size_estimate_bytes=100,
        tier_ssd_gib=3400,
        tier_gpu_count=4,
    )

    with pytest.raises(RuntimeError, match='cache tier .* is unavailable'):
        asyncio.run(strategy.provision(_request(), decision))
    assert manager.provision_pool_worker_calls == []

    inventory['worker-pinned'] = _inventory(
        'worker-pinned', gpu_slots_total=4)
    manager.registered_worker_matches_current_cache_tier_result = False
    with pytest.raises(
            ExistingWorkerUnavailableError,
            match='canary-unavailable',
    ) as exc:
        asyncio.run(strategy.provision(_request(), decision))
    assert isinstance(exc.value.__cause__, PoolWorkerNotReadyError)
    assert 'current cache-tier envelope' in str(exc.value.__cause__)
    assert manager.provision_activity_pod_calls == []

    manager.registered_worker_matches_current_cache_tier_result = True
    asyncio.run(strategy.provision(_request(), decision))
    assert manager.provision_pool_worker_calls[-1]['pool_worker_uid'] == (
        'worker-pinned')
    assert manager.registered_worker_matches_current_cache_tier_calls == [
        ('worker-pinned', PvcSpec(size_gib=3400, gpu_count=4)),
        ('worker-pinned', PvcSpec(size_gib=3400, gpu_count=4)),
    ]


def test_canary_availability_blocks_pinned_retry_with_unknown_tier():
    manager = FakeManager()
    inventory: dict[str, WorkerInventory] = {}
    strategy = _strategy(
        manager,
        inventory,
        warm={},
        sizes={},
        capacity_tiers=[CapacityTier(800, 1), CapacityTier(3400, 4)],
    )
    strategy.restrict_to_available_capacity(
        {(800, 1)}, ephemeral_available=True)
    decision = RouteDecision(
        path='durable',
        pool_worker_uid='worker-legacy-pinned',
        provisioned_new_worker=False,
        reason='pinned-on-retry',
        snapshot_size_estimate_bytes=100,
    )

    with pytest.raises(RuntimeError, match='capacity tier is unknown'):
        asyncio.run(strategy.provision(_request(), decision))

    assert manager.provision_pool_worker_calls == []

    inventory['worker-legacy-pinned'] = _inventory('worker-legacy-pinned')
    asyncio.run(strategy.provision(_request(), decision))
    assert manager.provision_pool_worker_calls[0]['pool_worker_uid'] == (
        'worker-legacy-pinned')


def test_k8s_strategy_warm_non_gpu_activity_uses_gpu_host_image():
    manager = FakeManager()
    inventory = {
        'worker-a': _inventory(
            'worker-a',
            gpu_slots_total=4,
            gpu_slots_in_use=4,
        )
    }
    strategy = _strategy(manager, inventory, warm={'snap-1': 'worker-a'},
                         sizes={})

    async def drive():
        req = _request(activity=SnapshotActivity.GEN_TT)
        decision = await strategy.route(req, warm_location_uid='worker-a')
        handle = await strategy.provision(req, decision)
        return decision, handle

    decision, handle = asyncio.run(drive())

    assert decision.path == 'durable'
    assert decision.pool_worker_uid == 'worker-a'
    assert handle.pool_worker_uid == 'worker-a'
    assert len(manager.provision_activity_pod_calls) == 1
    assert manager.provision_activity_pod_calls[0]['activity'] == 'GEN_TT'
    assert '/kumo-backend-gpu:' in manager.provision_activity_pod_calls[0][
        'image']
    assert manager.provision_ephemeral_activity_pod_calls == []


def test_k8s_strategy_overflow_requires_recorded_snapshot_size():
    manager = FakeManager()
    inventory = {
        'worker-a': _inventory(
            'worker-a',
            gpu_slots_total=4,
            gpu_slots_in_use=4,
        )
    }
    strategy = _strategy(
        manager,
        inventory,
        warm={'snap-1': 'worker-a'},
        sizes={},
        eph_tier_enabled=True,
    )

    async def drive():
        req = _request()
        decision = await strategy.route(
            req,
            warm_location_uid='worker-a',
            overflow_viable=True,
        )
        await strategy.provision(req, decision)

    with pytest.raises(RuntimeError, match='recorded snapshot size'):
        asyncio.run(drive())

    assert manager.provision_ephemeral_activity_pod_calls == []


def test_k8s_strategy_cold_snapshot_uses_best_existing_worker_and_daemon():
    manager = FakeManager()
    inventory = {
        'worker-a': _inventory('worker-a', pvc_free_bytes=250),
        'worker-b': _inventory('worker-b', pvc_free_bytes=900),
    }
    strategy = _strategy(manager, inventory, warm={}, sizes={})

    async def drive():
        req = _request(activity=SnapshotActivity.GEN_TT)
        decision = await strategy.route(
            req,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=1,
        )
        handle = await strategy.provision(req, decision)
        return decision, handle

    decision, handle = asyncio.run(drive())

    assert decision.path == 'durable'
    assert decision.pool_worker_uid == 'worker-b'
    assert handle.pool_worker_uid == 'worker-b'
    assert handle.task_queue == activity_task_queue('job-1')
    assert manager.provision_activity_pod_calls[0]['pool_worker_uid'] == (
        'worker-b')
    assert manager.provision_activity_pod_calls[0]['job_id'] == 'job-1'
    assert manager.provision_activity_pod_calls[0]['snapshot_id'] == 'snap-1'
    assert manager.provision_activity_pod_calls[0]['activity'] == 'GEN_TT'


def test_k8s_strategy_cold_routing_excludes_non_ready_workers():
    manager = FakeManager()
    inventory = {
        'worker-stale':
        _inventory(
            'worker-stale',
            status=WorkerStatus.STALE,
            pvc_free_bytes=2_000,
        ),
        'worker-dead':
        _inventory(
            'worker-dead',
            status=WorkerStatus.DEAD,
            pvc_free_bytes=3_000,
        ),
        # DRAINING worker (scale-down committed) must also be excluded from
        # routing even though it has the most free space (teardown race fix).
        'worker-draining':
        _inventory(
            'worker-draining',
            status=WorkerStatus.DRAINING,
            pvc_free_bytes=9_000,
        ),
        'worker-ready':
        _inventory('worker-ready', pvc_free_bytes=250),
    }
    strategy = _strategy(manager, inventory, warm={}, sizes={})

    async def drive():
        req = _request(activity=SnapshotActivity.GEN_TT)
        decision = await strategy.route(
            req,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=1,
        )
        handle = await strategy.provision(req, decision)
        return decision, handle

    decision, handle = asyncio.run(drive())

    assert decision.pool_worker_uid == 'worker-ready'
    assert handle.pool_worker_uid == 'worker-ready'
    assert manager.provision_pool_worker_calls == [{
        'pool_worker_uid':
        'worker-ready',
        'pvc_spec':
        PvcSpec(size_gib=1750),
        'timeout':
        600,
    }]
    assert manager.provision_activity_pod_calls[0]['pool_worker_uid'] == (
        'worker-ready')


def test_k8s_strategy_warm_stale_worker_waits_for_fsm():
    manager = FakeManager()
    inventory = {
        'worker-a':
        _inventory(
            'worker-a',
            status=WorkerStatus.STALE,
            gpu_slots_total=4,
            gpu_slots_in_use=0,
        )
    }
    strategy = _strategy(
        manager,
        inventory,
        warm={'snap-1': 'worker-a'},
        sizes={'snap-1': 1000},
    )

    async def drive():
        req = _request(activity=SnapshotActivity.TRAIN)
        return await strategy.route(req, warm_location_uid='worker-a')

    with pytest.raises(HomeStaleError):
        asyncio.run(drive())

    assert manager.provision_activity_pod_calls == []
    assert manager.provision_ephemeral_activity_pod_calls == []


def test_k8s_strategy_warm_gpu_busy_without_l2_serializes_on_home():
    manager = FakeManager()
    inventory = {
        'worker-a': _inventory(
            'worker-a',
            gpu_slots_total=4,
            gpu_slots_in_use=4,
        )
    }
    strategy = _strategy(
        manager,
        inventory,
        warm={'snap-1': 'worker-a'},
        sizes={},
    )

    async def drive():
        req = _request(activity=SnapshotActivity.PREDICT)
        return await strategy.route(
            req,
            warm_location_uid='worker-a',
            overflow_viable=False,
        )

    decision = asyncio.run(drive())

    assert decision.path == 'durable'
    assert decision.pool_worker_uid == 'worker-a'
    assert decision.reason == 'warm worker busy, no backup: serialize on home'


def test_k8s_strategy_in_progress_home_gpu_free_routes_to_home():
    """Backlog #17: a GPU consumer (TRAIN/PREDICT) whose snapshot already has
    an in-flight home runs ON that home when its GPU slot is free — it does
    NOT overflow just because the home isn't READY yet, and never mints a 2nd
    durable worker. ``home_gpu_in_use=0`` (no GPU lease on the home yet, e.g.
    only a producer's ENSURE lease there). Models mixed producer+consumer:
    the consumer rides the home the producer is warming."""
    manager = FakeManager()
    strategy = _strategy(manager, inventory={}, warm={}, sizes={'snap-1': 1000})

    async def drive():
        req = _request(activity=SnapshotActivity.PREDICT)
        return await strategy.route(req, warm_location_uid=None,
                                    home_worker_uid='worker-home',
                                    home_gpu_in_use=0, overflow_viable=True)

    decision = asyncio.run(drive())

    assert decision.path == 'durable'
    assert decision.pool_worker_uid == 'worker-home'
    assert decision.provisioned_new_worker is False


def test_k8s_strategy_in_progress_home_gpu_busy_with_backup_overflows():
    """A GPU consumer overflows to eph only when the home's GPU slot is
    already taken by another TRAIN/PREDICT (``home_gpu_in_use >= gpu_total``,
    default 1), an L2 backup exists to bootstrap from, AND the ephemeral tier
    is enabled (``eph_tier_enabled=True`` /
    ``POOL_INIT_EPHEMERAL_CANARY_ENABLED``). This is the cold-fanout sibling:
    shard-0 holds the home's GPU, shards 1..N overflow."""
    manager = FakeManager()
    strategy = _strategy(manager, inventory={}, warm={}, sizes={'snap-1': 1000},
                         eph_tier_enabled=True)

    async def drive():
        req = _request(activity=SnapshotActivity.PREDICT)
        decision = await strategy.route(
            req, warm_location_uid=None, home_worker_uid='worker-home',
            home_gpu_in_use=1, overflow_viable=True)
        handle = await strategy.provision(req, decision)
        return decision, handle

    decision, handle = asyncio.run(drive())

    assert decision.path == 'overflow'
    assert decision.pool_worker_uid is None
    assert handle.pool_worker_uid is None
    assert manager.provision_pool_worker_calls == []
    assert len(manager.provision_ephemeral_activity_pod_calls) == 1


def test_k8s_strategy_home_gpu_busy_backup_eph_disabled_serializes_on_home():
    """When the ephemeral tier is opted out
    (``POOL_INIT_EPHEMERAL_CANARY_ENABLED=false``; the tier is enabled by
    default), a GPU consumer whose home GPU slot is taken must NOT overflow to
    eph even though an L2 backup exists - routing to eph would mint a
    fire-and-forget eph Job that hangs Pending forever on a deployment with no
    ephemeral node pool (the #29865 stall moved to job time). Instead it
    serializes on the home worker, exactly as the no-backup case does, and
    provisions no eph Job."""
    manager = FakeManager()
    strategy = _strategy(manager, inventory={}, warm={}, sizes={'snap-1': 1000},
                         eph_tier_enabled=False)

    async def drive():
        req = _request(activity=SnapshotActivity.PREDICT)
        decision = await strategy.route(
            req, warm_location_uid=None, home_worker_uid='worker-home',
            home_gpu_in_use=1, overflow_viable=True)
        handle = await strategy.provision(req, decision)
        return decision, handle

    decision, _ = asyncio.run(drive())

    assert decision.path == 'durable'
    assert decision.pool_worker_uid == 'worker-home'  # reuse home, not 'worker-new'
    assert decision.provisioned_new_worker is False
    assert decision.reason == 'ephemeral tier disabled: serialize on home'
    # No eph Job minted: the overflow route is never taken with the tier off.
    assert manager.provision_ephemeral_activity_pod_calls == []


def test_k8s_strategy_home_gpu_busy_backup_eph_enabled_overflows():
    """With the ephemeral tier enabled (``eph_tier_enabled=True``), the same
    inputs preserve the original behavior: a GPU consumer with the home GPU
    slot taken and an L2 backup overflows to an eph pod."""
    manager = FakeManager()
    strategy = _strategy(manager, inventory={}, warm={}, sizes={'snap-1': 1000},
                         eph_tier_enabled=True)

    async def drive():
        req = _request(activity=SnapshotActivity.PREDICT)
        decision = await strategy.route(req, warm_location_uid=None,
                                        home_worker_uid='worker-home',
                                        home_gpu_in_use=1, overflow_viable=True)
        handle = await strategy.provision(req, decision)
        return decision, handle

    decision, handle = asyncio.run(drive())

    assert decision.path == 'overflow'
    assert decision.pool_worker_uid is None
    assert handle.pool_worker_uid is None
    assert manager.provision_pool_worker_calls == []
    assert len(manager.provision_ephemeral_activity_pod_calls) == 1


def test_k8s_strategy_in_progress_home_gpu_busy_no_backup_serializes_on_home():
    """GPU consumer, home GPU slot taken, and NO L2 backup to overflow from
    (X was never produced) → serialize on the home rather than mint a 2nd
    durable worker. Never herds; lease TTL reaps it if X never materializes."""
    manager = FakeManager()
    strategy = _strategy(manager, inventory={}, warm={}, sizes={})

    async def drive():
        req = _request(activity=SnapshotActivity.PREDICT)
        return await strategy.route(req, warm_location_uid=None,
                                    home_worker_uid='worker-home',
                                    home_gpu_in_use=1, overflow_viable=False)

    decision = asyncio.run(drive())

    assert decision.path == 'durable'
    assert decision.pool_worker_uid == 'worker-home'  # reuse home, not 'worker-new'
    assert decision.provisioned_new_worker is False


def test_k8s_strategy_in_progress_home_producer_coalesces():
    """A producer (ENSURE_SNAPSHOT) — or any non-GPU activity — always
    co-locates on the in-flight home (it just needs X present), regardless of
    GPU occupancy or backup. It coalesces onto the single ingest/restore and
    never overflows or mints a 2nd worker."""
    manager = FakeManager()
    strategy = _strategy(manager, inventory={}, warm={}, sizes={})

    async def drive():
        req = _request(activity=SnapshotActivity.ENSURE_SNAPSHOT)
        return await strategy.route(req, warm_location_uid=None,
                                    home_worker_uid='worker-home',
                                    home_gpu_in_use=1, overflow_viable=True)

    decision = asyncio.run(drive())

    assert decision.path == 'durable'
    assert decision.pool_worker_uid == 'worker-home'
    assert decision.provisioned_new_worker is False


def test_k8s_strategy_route_stale_home_raises_home_stale_error():
    """Node-death resilience: a STALE home (was healthy, heartbeats lapsed but
    not yet DEAD) makes route raise HomeStaleError so the caller waits for the
    FSM to settle and re-routes rather than minting a competing durable worker
    that would split the snapshot."""
    manager = FakeManager()
    inventory = {
        'worker-home': _inventory('worker-home', status=WorkerStatus.STALE),
    }
    strategy = _strategy(manager, inventory=inventory, warm={},
                         sizes={'snap-1': 1000})

    async def drive():
        req = _request(activity=SnapshotActivity.PREDICT)
        return await strategy.route(req, warm_location_uid=None,
                                    home_worker_uid='worker-home',
                                    home_gpu_in_use=0, overflow_viable=True)

    with pytest.raises(HomeStaleError):
        asyncio.run(drive())


def test_k8s_strategy_route_dead_home_mints_fresh_worker():
    """A DEAD home is treated as no home: route falls through to the mint-fresh
    path (no candidates here → a fresh durable worker is provisioned)."""
    manager = FakeManager()
    inventory = {
        'worker-home': _inventory('worker-home', status=WorkerStatus.DEAD),
    }
    strategy = _strategy(manager, inventory=inventory, warm={},
                         sizes={'snap-1': 1000})

    async def drive():
        req = _request(activity=SnapshotActivity.PREDICT)
        return await strategy.route(req, warm_location_uid=None,
                                    home_worker_uid='worker-home',
                                    home_gpu_in_use=0, overflow_viable=True)

    decision = asyncio.run(drive())

    assert decision.path == 'durable'
    assert decision.pool_worker_uid == 'worker-new'  # minted, not the dead home
    assert decision.provisioned_new_worker is True


def test_k8s_strategy_route_draining_home_mints_fresh_worker():
    """A DRAINING home (scale-down committed) is treated as no home → mint
    fresh, same as DEAD."""
    manager = FakeManager()
    inventory = {
        'worker-home': _inventory('worker-home', status=WorkerStatus.DRAINING),
    }
    strategy = _strategy(manager, inventory=inventory, warm={},
                         sizes={'snap-1': 1000})

    async def drive():
        req = _request(activity=SnapshotActivity.PREDICT)
        return await strategy.route(req, warm_location_uid=None,
                                    home_worker_uid='worker-home',
                                    home_gpu_in_use=0, overflow_viable=True)

    decision = asyncio.run(drive())

    assert decision.path == 'durable'
    assert decision.pool_worker_uid == 'worker-new'
    assert decision.provisioned_new_worker is True


def test_k8s_strategy_route_missing_inventory_home_still_co_locates():
    """An in-flight / cold-fanout home (a sibling minted it, no inventory
    reported yet) is NOT treated as unhealthy: siblings still co-locate on the
    single home (backlog #17) rather than herding new durable workers."""
    manager = FakeManager()
    strategy = _strategy(manager, inventory={}, warm={}, sizes={'snap-1': 1000})

    async def drive():
        req = _request(activity=SnapshotActivity.PREDICT)
        return await strategy.route(req, warm_location_uid=None,
                                    home_worker_uid='worker-home',
                                    home_gpu_in_use=0, overflow_viable=True)

    decision = asyncio.run(drive())

    assert decision.path == 'durable'
    assert decision.pool_worker_uid == 'worker-home'
    assert decision.provisioned_new_worker is False


def test_k8s_strategy_min_reuse_threshold_is_strict():
    manager = FakeManager()
    inventory = {
        'worker-at-threshold':
        _inventory('worker-at-threshold', pvc_free_bytes=100),
        'worker-above-threshold':
        _inventory('worker-above-threshold', pvc_free_bytes=101),
    }
    strategy = _strategy(
        manager,
        inventory,
        warm={},
        sizes={},
    )

    async def drive():
        req = _request(activity=SnapshotActivity.GEN_TT)
        decision = await strategy.route(
            req,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=100,
        )
        handle = await strategy.provision(req, decision)
        return decision, handle

    decision, handle = asyncio.run(drive())

    assert decision.pool_worker_uid == 'worker-above-threshold'
    assert handle.pool_worker_uid == 'worker-above-threshold'
    assert manager.provision_pool_worker_calls == [{
        'pool_worker_uid':
        'worker-above-threshold',
        'pvc_spec':
        PvcSpec(size_gib=1750),
        'timeout':
        600,
    }]


def test_k8s_strategy_cold_gpu_routing_excludes_gpu_full_worker():
    manager = FakeManager()
    inventory = {
        'worker-big-but-busy':
        _inventory(
            'worker-big-but-busy',
            pvc_free_bytes=2_000,
            gpu_slots_total=4,
            gpu_slots_in_use=4,
        ),
        'worker-smaller-with-gpu':
        _inventory(
            'worker-smaller-with-gpu',
            pvc_free_bytes=250,
            gpu_slots_total=4,
            gpu_slots_in_use=1,
        ),
    }
    strategy = _strategy(manager, inventory, warm={}, sizes={})

    async def drive():
        req = _request(activity=SnapshotActivity.PREDICT)
        decision = await strategy.route(
            req,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=1,
        )
        handle = await strategy.provision(req, decision)
        return decision, handle

    decision, handle = asyncio.run(drive())

    assert decision.pool_worker_uid == 'worker-smaller-with-gpu'
    assert handle.pool_worker_uid == 'worker-smaller-with-gpu'
    assert manager.provision_pool_worker_calls == [{
        'pool_worker_uid':
        'worker-smaller-with-gpu',
        'pvc_spec':
        PvcSpec(size_gib=1750),
        'timeout':
        600,
    }]
    assert manager.provision_activity_pod_calls[0]['pool_worker_uid'] == (
        'worker-smaller-with-gpu')


def test_k8s_strategy_provisions_new_worker_when_no_candidate_exists():
    manager = FakeManager()
    manager.next_pool_worker_uid = 'worker-new'
    inventory = {
        'worker-a': _inventory('worker-a', pvc_free_bytes=50),
    }
    hints: list[int] = []

    def hint_factory(tier_ssd_gib: int) -> PvcSpec:
        hints.append(tier_ssd_gib)
        return PvcSpec(size_gib=tier_ssd_gib)

    strategy = _strategy(
        manager,
        inventory,
        warm={},
        sizes={'snap-1': 2048},
        hint_factory=hint_factory,
    )

    async def drive():
        req = _request()
        decision = await strategy.route(
            req,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=1,
        )
        handle = await strategy.provision(req, decision)
        return decision, handle

    decision, handle = asyncio.run(drive())

    assert decision.pool_worker_uid == 'worker-new'
    assert decision.provisioned_new_worker is True
    # Factory now receives chosen_ssd_gib (1750 = the only tier in _strategy's
    # default), not the raw snapshot size_bytes (2048). The tier is selected as
    # the smallest tier that fits snapshot_size + staging_reserve.
    assert hints == [1750]
    assert len(manager.provision_pool_worker_calls) == 1
    call = manager.provision_pool_worker_calls[0]
    assert call['pool_worker_uid'] == 'worker-new'
    assert isinstance(call['pvc_spec'], PvcSpec)
    assert call['pvc_spec'].size_gib == 1750
    assert manager.provision_activity_pod_calls[0]['pool_worker_uid'] == (
        'worker-new')
    assert manager.provision_activity_pod_calls[0][
        'env'] == _expected_activity_env('worker-new')
    assert handle.pool_worker_uid == 'worker-new'
    assert handle.task_queue == activity_task_queue('job-1')


def test_k8s_strategy_ensure_snapshot_runs_on_daemon_queue():
    manager = FakeManager()
    strategy = _strategy(manager, inventory={}, warm={}, sizes={'snap-1': 2048})

    async def drive():
        req = _request(activity=SnapshotActivity.ENSURE_SNAPSHOT)
        decision = await strategy.route(req, warm_location_uid=None)
        handle = await strategy.provision(req, decision)
        handle = await strategy.wait_provision_ready(handle, req.job_id)
        handle = await strategy.wait_provision_ready(handle, req.job_id)
        return decision, handle

    decision, handle = asyncio.run(drive())

    expected_queue = daemon_task_queue('worker-new')
    assert decision.path == 'durable'
    assert decision.pool_worker_uid == 'worker-new'
    assert decision.provisioned_new_worker is True
    assert handle.task_queue == expected_queue
    assert handle.pool_worker_uid == 'worker-new'
    assert manager.provision_pool_worker_calls == [{
        'pool_worker_uid':
        'worker-new',
        'pvc_spec':
        PvcSpec(size_gib=1750),
        'timeout':
        600,
    }]
    assert manager.provision_pool_worker_envs == [{
        'POOL_WORKER_UID':
        'worker-new',
        'POOL_TASK_QUEUE':
        expected_queue,
        'POOL_STORAGE_MODE':
        'daemon-on-pvc',
        'POOL_GPU_CONCURRENCY':
        '1',
        'POOL_MAX_CONCURRENT_ACTIVITIES':
        '0',
    }]
    assert manager.provision_activity_pod_calls == []
    assert manager.wait_for_activity_job_pod_running_calls == []
    assert manager.provision_ephemeral_activity_pod_calls == []


def test_k8s_strategy_default_new_worker_uid_has_durable_prefix():
    manager = FakeManager()
    strategy = K8sPlacementStrategy(
        manager,  # type: ignore[arg-type]
        worker_inventory=lambda: {},
        warm_location=lambda snapshot_id: None,
        snapshot_size_bytes=lambda snapshot_id: 2048,
        reserved_size_bytes_for_worker=lambda uid: 0,
        capacity_tiers=[CapacityTier(tier_ssd_gib=1750)],
        activity_poller_waiter=_accept_activity_poller,
    )

    async def drive():
        req = _request()
        decision = await strategy.route(
            req,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=1,
        )
        return await strategy.provision(req, decision)

    handle = asyncio.run(drive())

    assert handle.pool_worker_uid is not None
    assert handle.pool_worker_uid.startswith(POOL_WORKER_UID_PREFIX_DURABLE)


def test_k8s_strategy_rejects_durable_decision_without_planned_worker():
    manager = FakeManager()
    strategy = _strategy(manager, inventory={}, warm={}, sizes={'snap-1': 2048})
    req = _request()
    decision = RouteDecision(
        path='durable',
        pool_worker_uid=None,
        provisioned_new_worker=True,
        reason='invalid test decision',
    )

    with pytest.raises(RuntimeError, match='planned pool_worker_uid'):
        asyncio.run(strategy.provision(req, decision))

    assert manager.provision_pool_worker_calls == []


def test_k8s_strategy_reclaims_persisted_planned_worker_before_activity():
    manager = FakeManager()
    strategy = _strategy(manager, inventory={}, warm={}, sizes={'snap-1': 2048})
    req = _request()
    decision = RouteDecision(
        path='durable',
        pool_worker_uid='worker-planned',
        provisioned_new_worker=False,
        reason='test',
    )

    handle = asyncio.run(strategy.provision(req, decision))

    assert manager.provision_pool_worker_calls == [{
        'pool_worker_uid':
        'worker-planned',
        'pvc_spec':
        PvcSpec(size_gib=1750),
        'timeout':
        600,
    }]
    assert manager.provision_activity_pod_calls[0]['pool_worker_uid'] == (
        'worker-planned')
    assert manager.provision_activity_pod_calls[0][
        'env'] == _expected_activity_env('worker-planned')
    assert handle.pool_worker_uid == 'worker-planned'
    assert handle.task_queue == activity_task_queue(req.job_id)


def test_k8s_strategy_reconcile_reports_live_pool_worker_uids():
    manager = FakeManager()
    manager.live_workers = [
        FakePoolWorkerInfo('worker-a'),
        FakePoolWorkerInfo('worker-b'),
    ]
    strategy = _strategy(manager, inventory={}, warm={}, sizes={})
    state = PoolState(placement_state=K8sPlacementState())
    state.local_copies['snap-a:worker-a'] = SnapshotLocalCopyRecord(
        snapshot_id='snap-a',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.WARM,
    )
    state.snapshots['snap-b'] = SnapshotPlacement(
        worker='worker-b',
        leases=['job-b'],
    )

    delta = asyncio.run(strategy.reconcile(state))

    assert delta.live_pool_worker_uids == {'worker-a', 'worker-b'}
    assert delta.retired_uids == set()
    assert manager.delete_pool_worker_calls == []


@pytest.mark.parametrize(
    ('saved_tier', 'configured_tiers', 'expected_incompatible'),
    [
        ((400, 1), [(1024, 1)], {'worker-a'}),
        ((400, 1), [(400, 4)], {'worker-a'}),
        ((400, 1), [(400, 1), (1024, 1)], set()),
        ((1024, 1), [(1024, 1)], set()),
        ((0, 1), [(1024, 1)], set()),
        ((400, 0), [(1024, 1)], set()),
    ],
)
def test_k8s_strategy_reconcile_classifies_removed_persisted_tier(
        saved_tier, configured_tiers, expected_incompatible):
    manager = FakeManager()
    strategy = _strategy(
        manager,
        inventory={},
        warm={},
        sizes={},
        capacity_tiers=[CapacityTier(*tier) for tier in configured_tiers],
    )
    state = PoolState(placement_state=K8sPlacementState(pvcs={
        'worker-a': PvcState('worker-a', *saved_tier),
    }))

    delta = asyncio.run(strategy.reconcile(state))

    assert delta.incompatible_uids == expected_incompatible


def test_k8s_strategy_reconcile_leaves_expected_worker_without_dead_status():
    """An expected worker whose controller inventory is not DEAD (here: no
    inventory entry yet, e.g. mid daemon cold-start) is left alone — reconcile
    no longer relaunches it and does NOT retire it. Routing gates on READY and
    the workflow-side runner handles the transient window."""
    manager = FakeManager()
    strategy = _strategy(manager, inventory={}, warm={}, sizes={})
    state = PoolState(placement_state=K8sPlacementState())
    state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        leases=['job-1'],
    )

    delta = asyncio.run(strategy.reconcile(state))

    # No relaunch (the rebind machinery is gone), no retire, no delete.
    assert manager.provision_pool_worker_calls == []
    assert manager.delete_pool_worker_calls == []
    assert delta.retired_uids == set()
    assert delta.unexpected_uids == set()


def test_k8s_strategy_reconcile_leaves_stale_expected_worker():
    """A STALE (not yet DEAD) expected worker is left alone — only DEAD
    triggers retirement."""
    manager = FakeManager()
    inventory = {'worker-a': _inventory('worker-a', status=WorkerStatus.STALE)}
    strategy = _strategy(manager, inventory=inventory, warm={}, sizes={})
    state = PoolState(placement_state=K8sPlacementState())
    state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        leases=['job-1'],
    )

    delta = asyncio.run(strategy.reconcile(state))

    assert manager.delete_pool_worker_calls == []
    assert delta.retired_uids == set()
    assert delta.unexpected_uids == set()


def test_k8s_strategy_reconcile_retires_when_inventory_dead():
    """Retire-on-DEAD: an expected worker whose controller heartbeat status is
    DEAD is reported as retired (no relaunch). reconcile does not delete inline;
    it reports the uid for the controller to flip DEAD + scrub under state_lock
    before delete_workers tears it down."""
    manager = FakeManager()
    inventory = {'worker-a': _inventory('worker-a', status=WorkerStatus.DEAD)}
    strategy = _strategy(manager, inventory=inventory, warm={}, sizes={})
    state = PoolState(placement_state=K8sPlacementState())
    state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        leases=['job-1'],
    )

    delta = asyncio.run(strategy.reconcile(state))

    assert manager.provision_pool_worker_calls == []
    assert manager.delete_pool_worker_calls == []
    assert delta.live_pool_worker_uids == set()
    assert delta.retired_uids == {'worker-a'}
    assert delta.unexpected_uids == set()

    manager.live_workers = [FakePoolWorkerInfo('worker-a')]
    assert asyncio.run(
        strategy.revalidate_retired_workers(delta.retired_uids)) == set()
    manager.live_workers = []

    # The actual delete happens via delete_workers (called by the
    # controller after flipping the uid non-routable under state_lock).
    asyncio.run(strategy.delete_workers({'worker-a'}))
    assert manager.delete_pool_worker_calls == [{
        'pool_worker_uid': 'worker-a',
        'grace_period_seconds': 30,
        'scale_down_timeout': 120,
        'pvc_delete_timeout': 60,
        'force_on_timeout': True,
    }]


@pytest.mark.parametrize('pod_status', ['Running', 'Registered'])
def test_k8s_strategy_live_pod_row_always_vetoes_dead_retirement(pod_status):
    manager = FakeManager()
    manager.live_workers = [FakePoolWorkerInfo('worker-a', status=pod_status)]
    inventory = {'worker-a': _inventory('worker-a', status=WorkerStatus.DEAD)}
    strategy = _strategy(manager, inventory=inventory, warm={}, sizes={})
    state = PoolState(placement_state=K8sPlacementState())
    state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        leases=['job-1'],
    )

    delta = asyncio.run(strategy.reconcile(state))

    assert delta.observed_pool_worker_uids == {'worker-a'}
    assert delta.live_pool_worker_uids == (
        {'worker-a'} if pod_status == 'Registered' else set())
    assert delta.retired_uids == set()
    assert manager.delete_pool_worker_calls == []


@pytest.mark.parametrize(('pod_status', 'timestamp_field'), [
    ('Pending', 'created_at'),
    ('Terminating', 'deletion_started_at'),
])
def test_k8s_strategy_non_live_pod_row_veto_expires(
        monkeypatch, pod_status, timestamp_field):
    now = 2_000_000_000.0
    monkeypatch.setattr('pool.k8s_strategy.time.time', lambda: now)
    manager = FakeManager()
    inventory = {'worker-a': _inventory('worker-a', status=WorkerStatus.DEAD)}
    strategy = _strategy(manager, inventory=inventory, warm={}, sizes={})
    state = PoolState(placement_state=K8sPlacementState())
    state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        leases=['job-1'],
    )
    timestamps = {
        # Proves a freshly-Terminating old pod uses deletion time, not birth.
        'created_at': now - 10 * POOL_WORKER_PROVISION_TIMEOUT_SEC,
        'deletion_started_at': now - POOL_WORKER_PROVISION_TIMEOUT_SEC + 1,
    }
    if pod_status == 'Pending':
        timestamps['created_at'] = (
            now - POOL_WORKER_PROVISION_TIMEOUT_SEC + 1)
    manager.live_workers = [
        FakePoolWorkerInfo(
            'worker-a', status=pod_status,
            created_at=timestamps['created_at'],
            deletion_started_at=timestamps['deletion_started_at'])
    ]

    protected = asyncio.run(strategy.reconcile(state))

    assert protected.retired_uids == set()
    setattr(manager.live_workers[0], timestamp_field,
            now - POOL_WORKER_PROVISION_TIMEOUT_SEC)

    expired = asyncio.run(strategy.reconcile(state))

    assert expired.retired_uids == {'worker-a'}


@pytest.mark.parametrize('pod_status', ['Pending', 'Terminating'])
def test_k8s_strategy_missing_non_live_timestamp_fails_closed(pod_status):
    manager = FakeManager()
    manager.live_workers = [FakePoolWorkerInfo('worker-a', status=pod_status)]
    inventory = {'worker-a': _inventory('worker-a', status=WorkerStatus.DEAD)}
    strategy = _strategy(manager, inventory=inventory, warm={}, sizes={})
    state = PoolState(placement_state=K8sPlacementState())
    state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        leases=['job-1'],
    )

    delta = asyncio.run(strategy.reconcile(state))

    assert delta.retired_uids == set()


def test_k8s_strategy_failed_pod_row_does_not_veto_dead_retirement():
    manager = FakeManager()
    manager.live_workers = [FakePoolWorkerInfo('worker-a', status='Failed')]
    inventory = {'worker-a': _inventory('worker-a', status=WorkerStatus.DEAD)}
    strategy = _strategy(manager, inventory=inventory, warm={}, sizes={})
    state = PoolState(placement_state=K8sPlacementState())
    state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        leases=['job-1'],
    )

    delta = asyncio.run(strategy.reconcile(state))

    assert delta.retired_uids == {'worker-a'}


def test_k8s_strategy_any_recent_duplicate_vetoes_retirement(monkeypatch):
    now = 2_000_000_000.0
    monkeypatch.setattr('pool.k8s_strategy.time.time', lambda: now)
    manager = FakeManager()
    manager.live_workers = [
        FakePoolWorkerInfo(
            'worker-a',
            status='Terminating',
            deletion_started_at=(
                now - POOL_WORKER_PROVISION_TIMEOUT_SEC),
        ),
        FakePoolWorkerInfo(
            'worker-a',
            status='Pending',
            created_at=now - 1,
        ),
    ]
    inventory = {'worker-a': _inventory('worker-a', status=WorkerStatus.DEAD)}
    strategy = _strategy(manager, inventory=inventory, warm={}, sizes={})
    state = PoolState(placement_state=K8sPlacementState())
    state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        leases=['job-1'],
    )

    delta = asyncio.run(strategy.reconcile(state))

    assert delta.retired_uids == set()


def test_k8s_strategy_revalidation_vetoes_new_pending_pod(monkeypatch):
    now = 2_000_000_000.0
    monkeypatch.setattr('pool.k8s_strategy.time.time', lambda: now)
    manager = FakeManager()
    inventory = {'worker-a': _inventory('worker-a', status=WorkerStatus.DEAD)}
    strategy = _strategy(manager, inventory=inventory, warm={}, sizes={})
    state = PoolState(placement_state=K8sPlacementState())
    state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        leases=['job-1'],
    )
    delta = asyncio.run(strategy.reconcile(state))
    assert delta.retired_uids == {'worker-a'}

    manager.live_workers = [
        FakePoolWorkerInfo(
            'worker-a',
            status='Pending',
            created_at=now,
        )
    ]

    revalidated = asyncio.run(
        strategy.revalidate_retired_workers(delta.retired_uids))

    assert revalidated == set()
    manager.live_workers[0].created_at = (
        now - POOL_WORKER_PROVISION_TIMEOUT_SEC)

    revalidated_after_grace = asyncio.run(
        strategy.revalidate_retired_workers(delta.retired_uids))

    assert revalidated_after_grace == {'worker-a'}


def test_k8s_strategy_any_registered_duplicate_vetoes_retirement():
    """A replacement can briefly expose duplicate pod rows for one UID.

    List ordering must not allow a later Terminating row to hide a Registered
    row and authorize deletion of their shared PVC during a heartbeat outage.
    """
    manager = FakeManager()
    manager.live_workers = [
        FakePoolWorkerInfo('worker-a', status='Registered'),
        FakePoolWorkerInfo('worker-a', status='Terminating'),
    ]
    inventory = {'worker-a': _inventory('worker-a', status=WorkerStatus.DEAD)}
    strategy = _strategy(manager, inventory=inventory, warm={}, sizes={})
    state = PoolState(placement_state=K8sPlacementState())
    state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        leases=['job-1'],
    )

    delta = asyncio.run(strategy.reconcile(state))

    assert delta.observed_pool_worker_uids == {'worker-a'}
    assert delta.live_pool_worker_uids == {'worker-a'}
    assert delta.retired_uids == set()


def test_k8s_strategy_reconcile_reports_orphans_for_retired_worker():
    manager = FakeManager()
    inventory = {'worker-a': _inventory('worker-a', status=WorkerStatus.DEAD)}
    strategy = _strategy(manager, inventory=inventory, warm={}, sizes={})
    state = PoolState(placement_state=K8sPlacementState())
    state.local_copies['snap-1:worker-a'] = SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.WARM,
    )
    state.local_copies['snap-evicted:worker-a'] = SnapshotLocalCopyRecord(
        snapshot_id='snap-evicted',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.EVICTED,
    )

    delta = asyncio.run(strategy.reconcile(state))

    assert delta.retired_uids == {'worker-a'}
    assert delta.orphan_snapshots == {'snap-1': 'worker-a'}


def test_k8s_strategy_reconcile_reports_orphan_live_worker_as_unexpected():
    manager = FakeManager()
    manager.live_workers = [FakePoolWorkerInfo('worker-orphan')]
    strategy = _strategy(manager, inventory={}, warm={}, sizes={})
    state = PoolState(placement_state=K8sPlacementState())

    delta = asyncio.run(strategy.reconcile(state))

    # reconcile no longer deletes inline — it reports the unexpected uid.
    assert manager.delete_pool_worker_calls == []
    assert delta.unexpected_uids == {'worker-orphan'}
    assert delta.observed_pool_worker_uids == {'worker-orphan'}
    assert delta.live_pool_worker_uids == set()
    assert delta.retired_uids == set()

    asyncio.run(strategy.delete_workers(delta.unexpected_uids))
    assert manager.delete_pool_worker_calls == [{
        'pool_worker_uid': 'worker-orphan',
        'grace_period_seconds': 30,
        'scale_down_timeout': 120,
        'pvc_delete_timeout': 60,
        'force_on_timeout': True,
    }]


def test_k8s_strategy_retries_draining_worker_after_pod_disappears():
    """Scrubbed DRAINING state remains a teardown request even when no pod is
    left for list_pool_workers() to report."""
    manager = FakeManager()
    inventory = {
        'worker-a': _inventory('worker-a', status=WorkerStatus.DRAINING),
    }
    strategy = _strategy(manager, inventory=inventory, warm={}, sizes={})
    state = PoolState(placement_state=K8sPlacementState())

    delta = asyncio.run(strategy.reconcile(state))

    assert delta.observed_pool_worker_uids == set()
    assert delta.unexpected_uids == {'worker-a'}


def test_k8s_strategy_reconcile_reports_pvc_only_live_worker_as_unexpected():
    manager = FakeManager()
    manager.live_workers = [FakePoolWorkerInfo('worker-pvc-only')]
    strategy = _strategy(manager, inventory={}, warm={}, sizes={})
    state = PoolState(placement_state=K8sPlacementState(
        pvcs={
            'worker-pvc-only': PvcState(pool_worker_uid='worker-pvc-only'),
        }))

    delta = asyncio.run(strategy.reconcile(state))

    assert manager.delete_pool_worker_calls == []
    assert delta.unexpected_uids == {'worker-pvc-only'}
    assert delta.live_pool_worker_uids == set()
    assert delta.retired_uids == set()


def test_k8s_strategy_reconcile_reports_stale_home_only_worker_as_unexpected():
    manager = FakeManager()
    manager.live_workers = [FakePoolWorkerInfo('worker-home-only')]
    strategy = _strategy(manager, inventory={}, warm={}, sizes={})
    state = PoolState(placement_state=K8sPlacementState())
    state.snapshots['snap-home'] = SnapshotPlacement(worker='worker-home-only')

    delta = asyncio.run(strategy.reconcile(state))

    assert manager.delete_pool_worker_calls == []
    assert delta.unexpected_uids == {'worker-home-only'}
    assert delta.live_pool_worker_uids == set()
    assert delta.retired_uids == set()


def test_k8s_strategy_delete_workers_is_per_worker_resilient():
    """One wedged uid must not block the others from being torn down —
    delete_workers swallows per-worker failures and presses on (the bad
    uid is retried on the next reconcile tick)."""
    manager = FakeManager()
    manager.delete_pool_worker_errors = {
        'worker-bad': RuntimeError('CSI finalizer wedged'),
    }
    strategy = _strategy(manager, inventory={}, warm={}, sizes={})

    # Should NOT raise despite worker-bad failing; returns only the uids
    # actually torn down so the controller prunes those and retries the bad
    # one on the next tick.
    torn_down = asyncio.run(
        strategy.delete_workers(['worker-bad', 'worker-good']))

    attempted = {c['pool_worker_uid'] for c in manager.delete_pool_worker_calls}
    assert attempted == {'worker-bad', 'worker-good'}
    assert torn_down == {'worker-good'}


def test_k8s_strategy_activity_resources_factory_forwarded_to_durable_pod():
    """``activity_resources_factory`` should be invoked per request and
    its return value forwarded to ``provision_activity_pod`` so GPU
    capacity reservation reaches the K8s scheduler."""
    from pool.worker_manager import ResourceSpec

    manager = FakeManager()
    inventory = {'worker-a': _inventory('worker-a', gpu_slots_in_use=0)}
    seen: list[SnapshotActivity] = []

    def resources_factory(request):
        seen.append(request.activity)
        if request.activity in {
                SnapshotActivity.TRAIN, SnapshotActivity.PREDICT
        }:
            return ResourceSpec(gpu_count=1)
        return None

    strategy = _strategy(
        manager,
        inventory,
        warm={'snap-1': 'worker-a'},
        sizes={},
        activity_resources_factory=resources_factory,
    )

    async def drive():
        req = _request(activity=SnapshotActivity.TRAIN)
        decision = await strategy.route(req, warm_location_uid='worker-a')
        await strategy.provision(req, decision)

    asyncio.run(drive())

    assert seen == [SnapshotActivity.TRAIN]
    assert len(manager.provision_activity_pod_calls) == 1
    assert manager.provision_activity_pod_calls[0]['resources'] == (
        ResourceSpec(gpu_count=1))
    assert manager.provision_activity_pod_calls[0]['env'][
        'KUMO_REQUIRE_GPU'] == 'true'
    assert manager.provision_activity_pod_calls[0]['env'][
        'KUMO_ACCELERATOR'] == 'gpu'


def test_k8s_strategy_activity_resources_factory_forwarded_to_overflow_pod():
    """Same plumbing must apply to the overflow (ephemeral) path so
    GPU reservation works for overflow PREDICT shards."""
    from pool.worker_manager import ResourceSpec

    manager = FakeManager()
    # warm worker is GPU-saturated -> route forces overflow.
    inventory = {
        'worker-a': _inventory('worker-a', gpu_slots_total=1,
                               gpu_slots_in_use=1),
    }
    strategy = _strategy(
        manager,
        inventory,
        warm={'snap-1': 'worker-a'},
        sizes={'snap-1': 1000},
        activity_resources_factory=(lambda request: ResourceSpec(gpu_count=1)),
        eph_tier_enabled=True,
    )

    async def drive():
        req = _request(activity=SnapshotActivity.PREDICT)
        decision = await strategy.route(
            req,
            warm_location_uid='worker-a',
            overflow_viable=True,
        )
        assert decision.path == 'overflow'
        await strategy.provision(req, decision)

    asyncio.run(drive())

    assert len(manager.provision_ephemeral_activity_pod_calls) == 1
    assert manager.provision_ephemeral_activity_pod_calls[0]['resources'] == (
        ResourceSpec(gpu_count=1))
    assert manager.provision_ephemeral_activity_pod_calls[0]['env'][
        'KUMO_REQUIRE_GPU'] == 'true'
    assert manager.provision_ephemeral_activity_pod_calls[0]['env'][
        'KUMO_ACCELERATOR'] == 'gpu'


def test_build_k8s_placement_strategy_uses_controller_context_providers():
    manager = FakeManager()
    context = FakeK8sContext(
        inventory={'worker-a': _inventory('worker-a', gpu_slots_in_use=0)},
        warm={'snap-1': 'worker-a'},
        sizes={},
    )
    strategy = build_k8s_placement_strategy(
        manager,  # type: ignore[arg-type]
        context,
        capacity_tiers=[CapacityTier(tier_ssd_gib=1750)],
        activity_poller_waiter=_accept_activity_poller,
    )

    async def drive():
        req = _request()
        decision = await strategy.route(
            req,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=1,
        )
        handle = await strategy.provision(req, decision)
        return decision, handle

    decision, handle = asyncio.run(drive())

    assert decision.pool_worker_uid == 'worker-a'
    assert handle.pool_worker_uid == 'worker-a'


# ---------------------------------------------------------------------------
# Size-aware admission
# ---------------------------------------------------------------------------


def test_size_aware_admission_rejects_worker_with_insufficient_free_space():
    # Worker has 200 bytes free; snapshot needs 300 → must provision a new worker.
    manager = FakeManager()
    inventory = {'worker-a': _inventory('worker-a', pvc_free_bytes=200)}
    strategy = _strategy(manager, inventory, warm={}, sizes={})

    async def drive():
        req = _request(activity=SnapshotActivity.GEN_TT, )
        return await strategy.route(req, warm_location_uid=None,
                                    snapshot_size_estimate_bytes=300)

    decision = asyncio.run(drive())
    assert decision.provisioned_new_worker
    assert decision.pool_worker_uid == 'worker-new'


def test_size_aware_admission_picks_worker_with_most_available_space():
    # Two workers: A has 500 bytes free, B has 300 bytes free.
    # Both fit the 250-byte snapshot; strategy must pick A (max available).
    manager = FakeManager()
    inventory = {
        'worker-a': _inventory('worker-a', pvc_free_bytes=500),
        'worker-b': _inventory('worker-b', pvc_free_bytes=300),
    }
    strategy = _strategy(manager, inventory, warm={}, sizes={})

    async def drive():
        req = _request(activity=SnapshotActivity.GEN_TT, )
        return await strategy.route(req, warm_location_uid=None,
                                    snapshot_size_estimate_bytes=250)

    decision = asyncio.run(drive())
    assert not decision.provisioned_new_worker
    assert decision.pool_worker_uid == 'worker-a'


def test_size_aware_admission_accounts_for_reserved_in_flight_bytes():
    # Worker has 400 bytes free, but 300 bytes are already reserved by a
    # prior in-flight RESERVED placement. Effective available = 100 bytes.
    # Snapshot needs 200 → must provision a new worker.
    manager = FakeManager()
    inventory = {'worker-a': _inventory('worker-a', pvc_free_bytes=400)}
    strategy = K8sPlacementStrategy(
        manager,  # type: ignore[arg-type]
        worker_inventory=lambda: inventory,
        warm_location=lambda snapshot_id: None,
        snapshot_size_bytes=lambda snapshot_id: None,
        reserved_size_bytes_for_worker=lambda uid: 300
        if uid == 'worker-a' else 0,
        capacity_tiers=[CapacityTier(tier_ssd_gib=1750)],
        activity_poller_waiter=_accept_activity_poller,
        pool_worker_uid_factory=lambda: 'worker-new',
    )

    async def drive():
        req = _request(activity=SnapshotActivity.GEN_TT, )
        return await strategy.route(req, warm_location_uid=None,
                                    snapshot_size_estimate_bytes=200)

    decision = asyncio.run(drive())
    assert decision.provisioned_new_worker
    assert decision.pool_worker_uid == 'worker-new'


def test_size_aware_admission_raises_when_snapshot_too_large_for_any_worker():
    # No existing workers; fresh worker usable capacity is below the 1.5 GiB
    # streaming-ingest minimum for a 1 GiB source.
    manager = FakeManager()
    strategy = K8sPlacementStrategy(
        manager,  # type: ignore[arg-type]
        worker_inventory=lambda: {},
        warm_location=lambda snapshot_id: None,
        snapshot_size_bytes=lambda snapshot_id: None,
        reserved_size_bytes_for_worker=lambda uid: 0,
        capacity_tiers=[CapacityTier(tier_ssd_gib=1)],
        activity_poller_waiter=_accept_activity_poller,
        pool_worker_uid_factory=lambda: 'worker-new',
    )

    async def drive():
        req = _request(
            activity=SnapshotActivity.GEN_TT,
            source_size_bytes=1 * (1 << 30),
        )
        return await strategy.route(
            req,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=3 * (1 << 30),
        )

    with pytest.raises(DiskgraphCapacityError, match='at least 1.5 GiB'):
        asyncio.run(drive())


def test_tier_selection_picks_smallest_fitting_tier():
    # Two tiers: 900 GiB and 1800 GiB. A 500 GiB snapshot fits in 900 GiB
    # (900 - 0 staging = 900 >= 500) so the strategy picks the smaller tier,
    # not the larger one. This drives Karpenter to best-fit a g6e.8xlarge
    # instead of a g6e.16xlarge.
    manager = FakeManager()
    hints: list[int] = []

    def hint_factory(tier_ssd_gib: int) -> PvcSpec:
        hints.append(tier_ssd_gib)
        return PvcSpec(size_gib=tier_ssd_gib)

    strategy = K8sPlacementStrategy(
        manager,  # type: ignore[arg-type]
        worker_inventory=lambda: {},
        warm_location=lambda snapshot_id: None,
        snapshot_size_bytes=lambda snapshot_id: None,
        reserved_size_bytes_for_worker=lambda uid: 0,
        capacity_tiers=[
            CapacityTier(tier_ssd_gib=900),
            CapacityTier(tier_ssd_gib=1800),
        ],
        activity_poller_waiter=_accept_activity_poller,
        manager_pvc_spec_factory=hint_factory,
        pool_worker_uid_factory=lambda: 'worker-new',
    )

    async def drive():
        req = _request(activity=SnapshotActivity.GEN_TT, )
        decision = await strategy.route(
            req, warm_location_uid=None,
            snapshot_size_estimate_bytes=500 * (1 << 30))
        handle = await strategy.provision(req, decision)
        return decision, handle

    decision, _ = asyncio.run(drive())
    assert decision.provisioned_new_worker
    # Smallest fitting tier is 900 GiB, not 1800 GiB.
    assert hints == [900]


def test_tier_selection_prefers_three_times_source_size():
    gib = 1 << 30
    strategy = _strategy(
        FakeManager(),
        {},
        warm={},
        sizes={},
        capacity_tiers=[
            CapacityTier(tier_ssd_gib=900),
            CapacityTier(tier_ssd_gib=1800),
        ],
    )
    request = _request(
        activity=SnapshotActivity.GEN_TT,
        source_size_bytes=400 * gib,
    )

    decision = asyncio.run(
        strategy.route(
            request,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=1200 * gib,
        ))

    assert decision.tier_ssd_gib == 1800
    assert decision.snapshot_size_estimate_bytes == 1200 * gib


def test_tier_selection_uses_largest_tier_when_it_meets_only_streaming_minimum(
):
    gib = 1 << 30
    strategy = _strategy(
        FakeManager(),
        {},
        warm={},
        sizes={},
        capacity_tiers=[
            CapacityTier(tier_ssd_gib=900),
            CapacityTier(tier_ssd_gib=1200),
        ],
    )
    request = _request(
        activity=SnapshotActivity.GEN_TT,
        source_size_bytes=500 * gib,
    )

    decision = asyncio.run(
        strategy.route(
            request,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=1500 * gib,
        ))

    assert decision.tier_ssd_gib == 1200
    assert decision.snapshot_size_estimate_bytes == 1200 * gib
    assert decision.reason == (
        'largest available tier meets minimum ingest capacity')


def test_tier_selection_rejects_tier_below_streaming_minimum():
    gib = 1 << 30
    strategy = _strategy(
        FakeManager(),
        {},
        warm={},
        sizes={},
        capacity_tiers=[CapacityTier(tier_ssd_gib=900)],
    )
    request = _request(
        activity=SnapshotActivity.GEN_TT,
        source_size_bytes=800 * gib,
    )

    with pytest.raises(DiskgraphCapacityError,
                       match='at least 1200.0 GiB'):
        asyncio.run(
            strategy.route(
                request,
                warm_location_uid=None,
                snapshot_size_estimate_bytes=2400 * gib,
            ))


def test_min_ssd_gib_forces_strictly_larger_tier():
    strategy = _strategy(
        FakeManager(),
        {},
        warm={},
        sizes={},
        capacity_tiers=[
            CapacityTier(tier_ssd_gib=900),
            CapacityTier(tier_ssd_gib=1800),
        ],
    )

    decision = asyncio.run(
        strategy.route(
            _request(
                activity=SnapshotActivity.GEN_TT,
                min_ssd_gib=901,
            ),
            warm_location_uid=None,
            snapshot_size_estimate_bytes=1,
        ))

    assert decision.tier_ssd_gib == 1800


def test_k8s_strategy_home_too_small_for_estimate_mints_larger_worker():
    """A home below the requested capacity is not reused."""
    manager = FakeManager()
    # Home reported inventory: pvc_total (ssd_total_bytes) = 2_000.
    inventory = {'worker-home': _inventory('worker-home')}
    strategy = _strategy(manager, inventory, warm={}, sizes={})

    async def drive():
        req = _request(activity=SnapshotActivity.ENSURE_SNAPSHOT)
        return await strategy.route(
            req,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=3_000,  # > usable 2_000 × 1.0
            home_worker_uid='worker-home')

    decision = asyncio.run(drive())

    assert decision.path == 'durable'
    assert decision.pool_worker_uid == 'worker-new'
    assert decision.provisioned_new_worker is True


def test_k8s_strategy_home_fits_estimate_still_co_locates():
    """Control: an estimate that fits the home's usable PVC keeps the
    existing co-location behavior (no herd of new workers)."""
    manager = FakeManager()
    inventory = {'worker-home': _inventory('worker-home')}
    strategy = _strategy(manager, inventory, warm={}, sizes={})

    async def drive():
        req = _request(activity=SnapshotActivity.ENSURE_SNAPSHOT)
        return await strategy.route(
            req,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=1_500,  # <= usable 2_000 × 1.0
            home_worker_uid='worker-home')

    decision = asyncio.run(drive())

    assert decision.path == 'durable'
    assert decision.pool_worker_uid == 'worker-home'
    assert decision.provisioned_new_worker is False


def test_k8s_strategy_home_without_inventory_still_co_locates():
    """A home with no inventory yet (in-flight / cold-fanout) has unknown
    capacity — the disk gate must not bypass it (backlog #17: siblings keep
    co-locating on the single home)."""
    manager = FakeManager()
    strategy = _strategy(manager, inventory={}, warm={}, sizes={})

    async def drive():
        req = _request(activity=SnapshotActivity.ENSURE_SNAPSHOT)
        return await strategy.route(req, warm_location_uid=None,
                                    snapshot_size_estimate_bytes=3_000,
                                    home_worker_uid='worker-home')

    decision = asyncio.run(drive())

    assert decision.path == 'durable'
    assert decision.pool_worker_uid == 'worker-home'
    assert decision.provisioned_new_worker is False


def test_tier_selection_filters_by_requested_gpu_slots():
    manager = FakeManager()
    strategy = K8sPlacementStrategy(
        manager,  # type: ignore[arg-type]
        worker_inventory=lambda: {},
        warm_location=lambda snapshot_id: None,
        snapshot_size_bytes=lambda snapshot_id: None,
        reserved_size_bytes_for_worker=lambda uid: 0,
        capacity_tiers=[
            CapacityTier(tier_ssd_gib=2948, tier_gpu_count=4),
            CapacityTier(tier_ssd_gib=2948, tier_gpu_count=8),
        ],
        activity_poller_waiter=_accept_activity_poller,
        pool_worker_uid_factory=lambda: 'worker-new',
    )

    async def drive():
        req = _request(
            activity=SnapshotActivity.GEN_TT,
            min_gpu_slots=8,
        )
        decision = await strategy.route(
            req,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=100 * (1 << 30),
        )
        await strategy.provision(req, decision)
        return decision

    decision = asyncio.run(drive())
    assert decision.tier_ssd_gib == 2948
    assert decision.tier_gpu_count == 8
    pvc_spec = manager.provision_pool_worker_calls[0]['pvc_spec']
    assert pvc_spec == PvcSpec(size_gib=2948, gpu_count=8)


def test_tier_selection_rejects_when_no_gpu_tier_satisfies_request():
    strategy = K8sPlacementStrategy(
        FakeManager(),  # type: ignore[arg-type]
        worker_inventory=lambda: {},
        warm_location=lambda snapshot_id: None,
        snapshot_size_bytes=lambda snapshot_id: None,
        reserved_size_bytes_for_worker=lambda uid: 0,
        capacity_tiers=[
            CapacityTier(tier_ssd_gib=1750, tier_gpu_count=1),
            CapacityTier(tier_ssd_gib=2948, tier_gpu_count=2),
        ],
        activity_poller_waiter=_accept_activity_poller,
    )

    async def drive():
        req = _request(
            activity=SnapshotActivity.GEN_TT,
            min_gpu_slots=3,
        )
        return await strategy.route(
            req,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=100 * (1 << 30),
        )

    with pytest.raises(DiskgraphCapacityError, match='min_gpu_slots=3'):
        asyncio.run(drive())


def test_existing_worker_reuse_respects_requested_gpu_slots():
    inventory = {
        'worker-small':
        _inventory('worker-small', pvc_free_bytes=2_000, gpu_slots_total=1),
        'worker-large':
        _inventory('worker-large', pvc_free_bytes=2_000, gpu_slots_total=4),
    }
    strategy = K8sPlacementStrategy(
        FakeManager(),  # type: ignore[arg-type]
        worker_inventory=lambda: inventory,
        warm_location=lambda snapshot_id: None,
        snapshot_size_bytes=lambda snapshot_id: None,
        reserved_size_bytes_for_worker=lambda uid: 0,
        capacity_tiers=[CapacityTier(tier_ssd_gib=1750)],
        activity_poller_waiter=_accept_activity_poller,
    )

    async def drive():
        req = _request(
            activity=SnapshotActivity.GEN_TT,
            min_gpu_slots=4,
        )
        return await strategy.route(
            req,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=100,
        )

    decision = asyncio.run(drive())
    assert decision.pool_worker_uid == 'worker-large'


def test_unknown_size_reuses_existing_matching_worker():
    worker_size = 900 * (1 << 30)
    inventory = {
        'worker-small':
        _inventory(
            'worker-small',
            pvc_total_bytes=worker_size,
            pvc_free_bytes=worker_size,
            gpu_slots_total=4,
        ),
    }
    strategy = K8sPlacementStrategy(
        FakeManager(),  # type: ignore[arg-type]
        worker_inventory=lambda: inventory,
        warm_location=lambda snapshot_id: None,
        snapshot_size_bytes=lambda snapshot_id: None,
        reserved_size_bytes_for_worker=lambda uid: 0,
        capacity_tiers=[
            CapacityTier(tier_ssd_gib=900, tier_gpu_count=4),
            CapacityTier(tier_ssd_gib=1800, tier_gpu_count=4),
        ],
        activity_poller_waiter=_accept_activity_poller,
        pool_worker_uid_factory=lambda: 'worker-new',
    )

    async def drive():
        req = _request(
            activity=SnapshotActivity.GEN_TT,
            min_gpu_slots=4,
        )
        return await strategy.route(
            req,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=0,
        )

    decision = asyncio.run(drive())
    assert decision.pool_worker_uid == 'worker-small'
    assert not decision.provisioned_new_worker
    assert decision.tier_ssd_gib == 900
    assert decision.tier_gpu_count == 4


def test_unknown_size_new_worker_uses_largest_tier():
    strategy = _strategy(
        FakeManager(),
        {},
        warm={},
        sizes={},
        capacity_tiers=[
            CapacityTier(tier_ssd_gib=400, tier_gpu_count=4),
            CapacityTier(tier_ssd_gib=900, tier_gpu_count=4),
            CapacityTier(tier_ssd_gib=1800, tier_gpu_count=4),
        ],
    )

    decision = asyncio.run(
        strategy.route(
            _request(
                activity=SnapshotActivity.GEN_TT,
                min_gpu_slots=4,
            ),
            warm_location_uid=None,
        ))

    assert decision.provisioned_new_worker
    assert decision.tier_ssd_gib == 1800


def test_existing_worker_reuse_returns_capacity_tier_from_inventory():
    inventory = {
        'worker-large':
        WorkerInventory(
            pool_worker_uid='worker-large',
            status=WorkerStatus.READY,
            last_heartbeat=1.0,
            gpu_slots_total=4,
            state=WorkerStateSnapshot(
                ssd_total_bytes=2948 * (1 << 30),
                ssd_used_bytes=0,
            ),
        ),
    }
    strategy = K8sPlacementStrategy(
        FakeManager(),  # type: ignore[arg-type]
        worker_inventory=lambda: inventory,
        warm_location=lambda snapshot_id: None,
        snapshot_size_bytes=lambda snapshot_id: None,
        reserved_size_bytes_for_worker=lambda uid: 0,
        capacity_tiers=[CapacityTier(tier_ssd_gib=1750)],
        activity_poller_waiter=_accept_activity_poller,
    )

    async def drive():
        req = _request(
            activity=SnapshotActivity.GEN_TT,
            min_gpu_slots=4,
        )
        return await strategy.route(
            req,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=100,
        )

    decision = asyncio.run(drive())
    assert decision.pool_worker_uid == 'worker-large'
    assert decision.tier_ssd_gib == 2948
    assert decision.tier_gpu_count == 4


def test_existing_worker_reuse_prefers_persisted_capacity_tier_over_inventory():
    inventory = {
        'worker-large':
        WorkerInventory(
            pool_worker_uid='worker-large',
            status=WorkerStatus.READY,
            last_heartbeat=1.0,
            gpu_slots_total=4,
            state=WorkerStateSnapshot(
                # Live filesystem accounting can be slightly below the
                # configured PVC tier; persisted placement state is canonical.
                ssd_total_bytes=(2948 * (1 << 30)) - 1,
                ssd_used_bytes=0,
            ),
        ),
    }
    strategy = K8sPlacementStrategy(
        FakeManager(),  # type: ignore[arg-type]
        worker_inventory=lambda: inventory,
        warm_location=lambda snapshot_id: None,
        snapshot_size_bytes=lambda snapshot_id: None,
        reserved_size_bytes_for_worker=lambda uid: 0,
        capacity_tiers=[CapacityTier(tier_ssd_gib=1750)],
        activity_poller_waiter=_accept_activity_poller,
        capacity_tier_for_worker=lambda uid: CapacityTier(
            tier_ssd_gib=2948,
            tier_gpu_count=4,
        ) if uid == 'worker-large' else None,
    )

    async def drive():
        req = _request(
            activity=SnapshotActivity.GEN_TT,
            min_gpu_slots=4,
        )
        return await strategy.route(
            req,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=100,
        )

    decision = asyncio.run(drive())
    assert decision.pool_worker_uid == 'worker-large'
    assert decision.tier_ssd_gib == 2948
    assert decision.tier_gpu_count == 4


def test_unknown_size_reuse_uses_persisted_tier_when_inventory_bytes_drift():
    tier_bytes = 2948 * (1 << 30)
    inventory = {
        'worker-large':
        WorkerInventory(
            pool_worker_uid='worker-large',
            status=WorkerStatus.READY,
            last_heartbeat=1.0,
            gpu_slots_total=4,
            state=WorkerStateSnapshot(
                # Live filesystem accounting can drift below the configured
                # PVC tier; persisted placement state is canonical.
                ssd_total_bytes=tier_bytes - 1,
                ssd_used_bytes=0,
            ),
        ),
    }
    strategy = K8sPlacementStrategy(
        FakeManager(),  # type: ignore[arg-type]
        worker_inventory=lambda: inventory,
        warm_location=lambda snapshot_id: None,
        snapshot_size_bytes=lambda snapshot_id: None,
        reserved_size_bytes_for_worker=lambda uid: 0,
        capacity_tiers=[CapacityTier(tier_ssd_gib=2948, tier_gpu_count=4)],
        activity_poller_waiter=_accept_activity_poller,
        capacity_tier_for_worker=lambda uid: CapacityTier(
            tier_ssd_gib=2948,
            tier_gpu_count=4,
        ) if uid == 'worker-large' else None,
        pool_worker_uid_factory=lambda: 'worker-new',
    )

    async def drive():
        req = _request(
            activity=SnapshotActivity.GEN_TT,
            min_gpu_slots=4,
        )
        return await strategy.route(
            req,
            warm_location_uid=None,
            snapshot_size_estimate_bytes=0,
        )

    decision = asyncio.run(drive())
    assert decision.pool_worker_uid == 'worker-large'
    assert not decision.provisioned_new_worker
    assert decision.tier_ssd_gib == 2948
    assert decision.tier_gpu_count == 4
