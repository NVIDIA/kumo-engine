# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``pool.service.PoolController``.

We exercise the controller instance directly (bypassing AutoRPC transport).
Full AutoRPC roundtrip tests require a Temporal dev server and belong with the
Block E integration tests.
"""
import asyncio
import time
from dataclasses import replace
from datetime import timedelta
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Literal,
    Mapping,
    Optional,
    cast,
)
from unittest.mock import AsyncMock, Mock

import pytest  # pants: no-infer-dep

import pool.worker_manager.init_canary as canary_mod
from common_utils.time import now_utc
from io_utils.json import to_json
from pool import metrics
from pool.dataclasses import (
    ActiveJob,
    DataplaneBackup,
    K8sPlacementState,
    PlacementRequest,
    PlacementResult,
    PlacementStatus,
    PoolState,
    PoolStatsFilter,
    PoolWorkerStats,
    ProvisionedHandle,
    PvcState,
    ResourceHint,
    SnapshotActivity,
    SnapshotComplianceRecord,
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
)
from pool.errors import (
    ExistingWorkerUnavailableError,
    SnapshotEvictionInProgressError,
)
from pool.k8s_strategy import (
    CapacityTier,
    K8sPlacementStrategy,
    activity_task_queue,
)
from pool.k8s_strategy import \
    build_k8s_placement_strategy as _build_k8s_placement_strategy
from pool.persistence import PoolStatePersistence
from pool.service import (
    CONTROLLER_MAINTENANCE_INTERVAL_SECS,
    LEASE_DEFENSIVE_SWEEP_INTERVAL_SECS,
    POOL_HEARTBEAT_INTERVAL_SECS,
    REACTIVE_EVICTION_WINDOW_SECS,
    WORKER_DEAD_THRESHOLD_SECS,
    WORKER_STALE_THRESHOLD_SECS,
    PoolController,
    PoolControllerPlacementContext,
    WorkflowLeaseState,
    _build_configured_k8s_manager,
    _configured_k8s_pool_names,
    _configured_lifecycle_policy,
    _configured_pool_worker_fs_group,
    _configured_pool_worker_service_account,
    _configured_strategy_factory,
    _k8s_pool_storage_class,
    _normalize_state_for_strategy,
    _scrub_unbacked_k8s_placement_ghosts,
    build_temporal_activity_poller_waiter,
    create_pool_controller,
    run_configured_init_canary,
)
from pool.snapshot_paths import dataplane_snapshot_path
from pool.storage_mode import PoolStorageMode
from pool.strategy import PlacementStrategy, ReconcileDelta, RouteDecision
from pool.strategy_singlehost import (
    SINGLEHOST_WORKER_UID,
    SinglehostPlacementStrategy,
)
from pool.task_queues import activity_worker_identity
from pool.timeouts import POOL_WORKER_STARTUP_GRACE_SEC
from pool.worker_manager import PoolWorkerInfo, ProvisionTimeoutError, PvcSpec
from service_framework._raw_deploy_config import (
    DiskgraphConfig,
    DiskgraphNodeType,
    K8sPoolStrategyConfig,
)
from service_framework.deploy_env_config import DeployEnvConfig
from service_framework.deploy_stage import DeployStage
from statedb.utils.errors import DiskgraphError
from temporalio_commons.auto_rpc_exception import is_retryable


async def _accept_activity_poller(*_args) -> None:
    pass


def build_k8s_placement_strategy(*args, **kwargs):
    kwargs.setdefault('activity_poller_waiter', _accept_activity_poller)
    return _build_k8s_placement_strategy(*args, **kwargs)


class MockPlacementStrategy:
    def __init__(self, fail_provision: bool = False,
                 enqueue_daemon: bool = False,
                 requires_l2_backup_check: bool = False,
                 fail_release: bool = False, fail_reconcile: bool = False,
                 reconcile_delta: Optional[ReconcileDelta] = None) -> None:
        self.fail_provision = fail_provision
        self.enqueue_daemon = enqueue_daemon
        self.requires_l2_backup_check = requires_l2_backup_check
        self.fail_release = fail_release
        self.fail_reconcile = fail_reconcile
        self.reconcile_delta = reconcile_delta or ReconcileDelta()
        self.route_calls: list[tuple[PlacementRequest, Optional[str]]] = []
        # Parallel record of the sibling-fanout routing inputs (backlog #17),
        # keyed positionally to route_calls: (home_worker_uid, home_gpu_in_use,
        # overflow_viable).
        self.route_home_calls: list[tuple[Optional[str], int, bool]] = []
        self.provision_calls: list[tuple[PlacementRequest, RouteDecision]] = []
        self.wait_provision_ready_calls: list[ProvisionedHandle] = []
        self.wait_provision_ready_error: Optional[BaseException] = None
        self.release_calls: list[Optional[ProvisionedHandle]] = []
        self.reconcile_calls: list[PoolState] = []
        self.on_reconcile: Optional[Callable[[], None]] = None
        self.revalidated_retired_workers: Optional[set[str]] = None
        # uids the controller asked delete_workers to tear down (one
        # entry per call, each a sorted list of the uids in that call).
        self.delete_workers_calls: list[list[str]] = []
        # uids delete_workers reports as actually torn down — the controller
        # prunes their inventory entries. Default empty = "delete not yet
        # confirmed" so a flipped DRAINING/DEAD entry persists for assertions;
        # set per-test to exercise the prune path.
        self.delete_workers_returns: set[str] = set()

    async def route(self, request: PlacementRequest,
                    warm_location_uid: Optional[str],
                    snapshot_size_estimate_bytes: int = 0, *,
                    home_worker_uid: Optional[str] = None,
                    home_gpu_in_use: int = 0,
                    overflow_viable: bool = False) -> RouteDecision:
        self.route_calls.append((request, warm_location_uid))
        self.route_home_calls.append(
            (home_worker_uid, home_gpu_in_use, overflow_viable))
        # Mirror the real policy (backlog #17): keep siblings on the single
        # home; a GPU consumer overflows only when the home's GPU slot is
        # already taken (gpu_total=1 here) AND a backup exists; never mint a
        # 2nd durable worker.
        if home_worker_uid is not None:
            needs_gpu = request.activity in {
                SnapshotActivity.TRAIN, SnapshotActivity.PREDICT
            }
            home_can_serve = (not needs_gpu) or home_gpu_in_use < 1
            if home_can_serve or not overflow_viable:
                path, uid = 'durable', home_worker_uid
            else:
                path, uid = 'overflow', None
        elif self.requires_l2_backup_check:
            path, uid = 'overflow', None
        else:
            path, uid = 'durable', warm_location_uid or 'worker-a'
        return RouteDecision(
            path=path,
            pool_worker_uid=uid,
            provisioned_new_worker=False,
            reason='test',
        )

    async def provision(self, request: PlacementRequest,
                        decision: RouteDecision) -> ProvisionedHandle:
        self.provision_calls.append((request, decision))
        if self.fail_provision:
            raise RuntimeError('no capacity')
        return ProvisionedHandle(
            kind=decision.path,
            pool_worker_uid=decision.pool_worker_uid,
            task_queue=f'diskgraph-activity-tq-{request.job_id}',
        )

    async def wait_provision_ready(self, handle: ProvisionedHandle,
                                   job_id: str) -> ProvisionedHandle:
        del job_id
        self.wait_provision_ready_calls.append(handle)
        if self.wait_provision_ready_error is not None:
            raise self.wait_provision_ready_error
        return handle

    async def release(self, request: PlacementRequest,
                      handle: Optional[ProvisionedHandle]) -> None:
        del request
        self.release_calls.append(handle)
        if self.fail_release:
            raise RuntimeError('release failed')

    async def reconcile(self, state: PoolState) -> ReconcileDelta:
        self.reconcile_calls.append(state)
        if self.on_reconcile is not None:
            self.on_reconcile()
        if self.fail_reconcile:
            raise RuntimeError('reconcile failed')
        return self.reconcile_delta

    async def revalidate_retired_workers(self, uids) -> set[str]:
        candidates = set(uids)
        if self.revalidated_retired_workers is None:
            return candidates
        return candidates & self.revalidated_retired_workers

    async def delete_workers(self, uids) -> set[str]:
        self.delete_workers_calls.append(sorted(uids))
        return set(self.delete_workers_returns)


class _FakePoolWorkerInfo:
    def __init__(self, pool_worker_uid: str,
                 status: str = 'Registered',
                 extra_tolerations_match: bool = True) -> None:
        self.pool_worker_uid = pool_worker_uid
        self.status = status
        self.extra_tolerations_match = extra_tolerations_match


class _FakeActivityPodInfo:
    def __init__(self, pod_name: str) -> None:
        self.pod_name = pod_name


def test_configured_canary_applies_registered_evidence_and_result(monkeypatch):
    capacity_tiers = [CapacityTier(800, 1), CapacityTier(3400, 4)]
    placement_state = K8sPlacementState()
    placement_state.pvcs.update({
        uid: PvcState(uid, ssd_gib, gpu_count)
        for uid, ssd_gib, gpu_count in [
            ('worker-valid', 800, 1),
            ('worker-old-spec', 3400, 4),
            ('worker-running', 3400, 4),
            ('worker-unconfigured', 999, 1),
            ('worker-wrong-gpu', 800, 4),
            ('worker-incomplete', 800, 0),
        ]
    })
    workers = cast(list[PoolWorkerInfo], [
        _FakePoolWorkerInfo('worker-valid', 'Registered'),
        _FakePoolWorkerInfo('worker-old-spec', 'Registered'),
        _FakePoolWorkerInfo('worker-running', 'Running'),
        _FakePoolWorkerInfo('worker-unconfigured', 'Registered'),
        _FakePoolWorkerInfo('worker-wrong-gpu', 'Registered'),
        _FakePoolWorkerInfo('worker-incomplete', 'Registered'),
        _FakePoolWorkerInfo('worker-missing-state', 'Registered'),
    ])

    async def list_pool_workers():
        return workers

    async def registered_worker_matches_current_cache_tier(worker, spec):
        assert (worker.pool_worker_uid, spec.size_gib, spec.gpu_count) in {
            ('worker-valid', 800, 1),
            ('worker-old-spec', 3400, 4),
        }
        return worker.pool_worker_uid == 'worker-valid'

    manager = SimpleNamespace(
        list_pool_workers=list_pool_workers,
        registered_worker_matches_current_cache_tier=(
            registered_worker_matches_current_cache_tier),
    )
    strategy = object.__new__(K8sPlacementStrategy)
    restrictions = []
    monkeypatch.setattr(
        strategy,
        'restrict_to_available_capacity',
        lambda tiers, *, ephemeral_available: restrictions.append(
            (tiers, ephemeral_available)),
    )
    controller = cast(
        PoolController,
        SimpleNamespace(
            _state=PoolState(placement_state=placement_state),
            _strategy=strategy,
        ),
    )

    async def run_canary(_manager, *, cache_tier_specs, pvc_spec_factory,
                         prevalidated_cache_tiers):
        del pvc_spec_factory
        assert {(spec.size_gib, spec.gpu_count)
                for spec in cache_tier_specs} == {(800, 1), (3400, 4)}
        assert await prevalidated_cache_tiers() == {(800, 1)}
        return canary_mod.InitCanaryResult(
            available_cache_tiers=frozenset({(800, 1)}),
            ephemeral_available=False,
        )

    monkeypatch.setattr(
        'pool.service._diskgraph_config',
        lambda _config: SimpleNamespace(pool_strategy='k8s'),
    )
    monkeypatch.setattr(
        'pool.service._build_configured_k8s_manager',
        lambda _config, _diskgraph: (
            manager,
            [800, 3400],
            capacity_tiers,
            lambda ssd_gib: PvcSpec(size_gib=ssd_gib),
        ),
    )
    monkeypatch.setattr(canary_mod, 'run_init_canary', run_canary)

    asyncio.run(
        run_configured_init_canary(cast(DeployEnvConfig, object()), controller))

    assert restrictions == [(frozenset({(800, 1)}), False)]


class _FakeK8sManager:
    # Constants mirror K8sPoolWorkerManager's public surface so the
    # placement strategy can build observability metadata without
    # branching on test vs. real.
    namespace = 'kumo-dataplane-fake'
    kube_context = 'example-cluster'

    def __init__(self) -> None:
        self.provision_pool_worker_calls: list[dict[str, Any]] = []
        self.provision_pool_worker_envs: list[Optional[Mapping[str, str]]] = []
        self.provision_activity_pod_calls: list[dict[str, Any]] = []
        self.provision_ephemeral_activity_pod_calls: list[dict[str, Any]] = []
        self.delete_pool_worker_calls: list[dict[str, Any]] = []
        self.delete_activity_pods_calls: list[dict[str, Any]] = []
        self.live_workers: list[_FakePoolWorkerInfo] = []
        self.on_provision_pool_worker: Optional[Callable[
            [Optional[str]], Awaitable[None]]] = None

    @staticmethod
    def deployment_name_for_uid(pool_worker_uid: str) -> str:
        return f'diskgraph-daemon-{pool_worker_uid}'

    @staticmethod
    def activity_job_name(job_id: str, kind: str) -> str:
        return f'{kind}-{job_id}'

    async def wait_for_activity_job_pod_running(
            self, job_name: str, timeout: float) -> '_FakeActivityPodInfo':
        return _FakeActivityPodInfo(pod_name=f'pod-{job_name}')

    async def provision_pool_worker(
        self,
        pool_worker_uid: Optional[str] = None,
        pvc_spec: Any = None,
        env: Optional[Mapping[str, str]] = None,
        timeout: float = 600,
        require_existing_resources: bool = False,
    ) -> _FakePoolWorkerInfo:
        if self.on_provision_pool_worker is not None:
            await self.on_provision_pool_worker(pool_worker_uid)
        self.provision_pool_worker_envs.append(env)
        self.provision_pool_worker_calls.append({
            'pool_worker_uid': pool_worker_uid,
            'pvc_spec': pvc_spec,
            'timeout': timeout,
        })
        return _FakePoolWorkerInfo(pool_worker_uid or 'pool-worker-new')

    async def provision_activity_pod(self, **kwargs: Any) -> None:
        self.provision_activity_pod_calls.append(kwargs)

    async def registered_worker_matches_current_cache_tier(
            self, info: Any, pvc_spec: Any) -> bool:
        return True

    async def provision_ephemeral_activity_pod(self, **kwargs: Any) -> None:
        self.provision_ephemeral_activity_pod_calls.append(kwargs)

    async def list_pool_workers(self) -> list[_FakePoolWorkerInfo]:
        return self.live_workers

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

    async def delete_activity_pods(self, **kwargs: Any) -> None:
        self.delete_activity_pods_calls.append(kwargs)


class _CreateControllerConfig:
    def __init__(self, pool_state_path: str,
                 data_plane_storage_root: Optional[str] = None,
                 data_compute: Optional[DiskgraphConfig] = None) -> None:
        self.pool_state_path = pool_state_path
        self.data_plane_storage_root = (data_plane_storage_root
                                        or str(Path(pool_state_path).parent))
        # The pool controller only runs for diskgraph deployments, so the
        # config it reads always carries a real ``DiskgraphConfig``.
        self.data_compute = (DiskgraphConfig()
                             if data_compute is None else data_compute)


class _FakeClock:
    def __init__(self) -> None:
        self.current = now_utc()
        self.monotonic = 1_000_000.0

    def now_utc(self):
        return self.current

    def time(self) -> float:
        return self.monotonic

    def advance(self, seconds: int) -> None:
        self.current += timedelta(seconds=seconds)
        self.monotonic += seconds


@pytest.fixture(autouse=True)
def _clear_pod_namespace(monkeypatch):  # pyright: ignore[reportUnusedFunction]
    # The k8s_namespace tests assert the legacy derivation unless a test
    # sets POD_NAMESPACE itself; guarantee a clean slate even when the
    # test runner itself executes inside a pod with the downward API.
    monkeypatch.delenv('POD_NAMESPACE', raising=False)
    monkeypatch.delenv('POOL_WORKER_TENANT', raising=False)


def test_configured_k8s_pool_names_from_id_suffix():
    # DEV id_suffix carries leading dash + user prefix; the helper
    # strips the dash and uses the rest as the tenant slug.
    config = SimpleNamespace(
        id_suffix='-antest-0bdf7d',
        deploy_mode=SimpleNamespace(value='dev'),
    )

    assert _configured_k8s_pool_names(config) == (  # type: ignore[arg-type]
        'antest-0bdf7d',
        'kumo-dataplane-antest-0bdf7d',
    )


def test_configured_k8s_pool_names_normalizes_disallowed_chars():
    # Underscores etc. become hyphens; the result must still match
    # k8s RFC 1123 (lowercase alphanumeric + dash, leading alnum).
    config = SimpleNamespace(
        id_suffix='SIYANG_DEV_DK',
        deploy_mode=SimpleNamespace(value='dev'),
    )

    assert _configured_k8s_pool_names(config) == (  # type: ignore[arg-type]
        'siyang-dev-dk',
        'kumo-dataplane-siyang-dev-dk',
    )


def test_configured_k8s_pool_names_rejects_too_long_id_suffix():
    # >48 chars after normalization → derived namespace would breach
    # k8s's 63-char cap on object names.
    config = SimpleNamespace(
        id_suffix='a' * 60,
        deploy_mode=SimpleNamespace(value='dev'),
    )

    with pytest.raises(ValueError, match='RFC 1123'):
        _configured_k8s_pool_names(config)  # type: ignore[arg-type]


def test_configured_k8s_pool_names_pod_namespace_override(monkeypatch):
    # The OSS/BYOC chart injects POD_NAMESPACE via the downward API; the
    # pool controller must launch workers into its own namespace instead
    # of the legacy kumo-dataplane-<tenant> derivation.
    monkeypatch.setenv('POD_NAMESPACE', 'kumo')
    config = SimpleNamespace(
        id_suffix='-antest-0bdf7d',
        deploy_mode=SimpleNamespace(value='dev'),
    )

    assert _configured_k8s_pool_names(config) == (  # type: ignore[arg-type]
        'antest-0bdf7d',
        'kumo',
    )


def test_configured_k8s_pool_names_prod_instance(monkeypatch):
    monkeypatch.setenv('POD_NAMESPACE', 'kumo-acme')
    monkeypatch.setenv('POOL_WORKER_TENANT', 'acme')
    config = DeployEnvConfig(
        deploy_mode=DeployStage.PROD,
        instance_id='prod',
    )

    assert _configured_k8s_pool_names(config) == (
        'acme',
        'kumo-acme',
    )


def test_configured_k8s_pool_names_pod_namespace_empty_falls_back(monkeypatch):
    monkeypatch.setenv('POD_NAMESPACE', '')
    config = SimpleNamespace(
        id_suffix='-antest-0bdf7d',
        deploy_mode=SimpleNamespace(value='dev'),
    )

    assert _configured_k8s_pool_names(config) == (  # type: ignore[arg-type]
        'antest-0bdf7d',
        'kumo-dataplane-antest-0bdf7d',
    )


def _diskgraph_pool_config(
    pool_strategy: Literal['k8s', 'singlehost'] = 'singlehost',
    deploy_mode: DeployStage = DeployStage.DEV,
) -> DeployEnvConfig:
    return DeployEnvConfig(
        deploy_mode=deploy_mode,
        data_compute=DiskgraphConfig(pool_strategy=pool_strategy),
    )


def _poller_waiter(describe, **kwargs):
    temporal_client = SimpleNamespace(get=AsyncMock(
        return_value=SimpleNamespace(workflow_service=SimpleNamespace(
            describe_task_queue=describe))))
    config = SimpleNamespace(instance_id='dev-1',
                             temporal_namespace='test-namespace',
                             task_queue=lambda queue: f'{queue}-dev-1')
    return build_temporal_activity_poller_waiter(cast(DeployEnvConfig, config),
                                                 cast(Any, temporal_client),
                                                 poll_interval_secs=0, **kwargs)


@pytest.mark.asyncio
async def test_temporal_activity_poller_waiter_requires_selected_pod_identity():
    from temporalio.api.enums.v1 import TaskQueueType
    from temporalio.service import RPCError, RPCStatusCode

    expected_identity = activity_worker_identity('activity-pod-new', 'dev-1')
    describe = AsyncMock(side_effect=[
        RPCError(message='unavailable', status=RPCStatusCode.UNAVAILABLE,
                 raw_grpc_status=b''),
        SimpleNamespace(pollers=[SimpleNamespace(identity='stale-poller')]),
        SimpleNamespace(pollers=[SimpleNamespace(identity=expected_identity)]),
    ])
    waiter = _poller_waiter(describe)

    await waiter('diskgraph-activity-tq-job', 'activity-pod-new', 1)

    assert describe.await_count == 3
    request = describe.await_args_list[0].args[0]
    assert request.namespace == 'test-namespace'
    assert request.task_queue.name == 'diskgraph-activity-tq-job-dev-1'
    assert request.task_queue_type == TaskQueueType.TASK_QUEUE_TYPE_ACTIVITY
    assert request.report_pollers is True


@pytest.mark.asyncio
async def test_temporal_activity_poller_waiter_times_out_on_stale_poller():
    describe = AsyncMock(return_value=SimpleNamespace(
        pollers=[SimpleNamespace(identity='pool-worker-old-pod-dev-1')]))
    monotonic_values = iter([0.0, 0.0, 2.0, 2.0])
    waiter = _poller_waiter(
        describe,
        monotonic_fn=lambda: next(monotonic_values),
    )

    with pytest.raises(ProvisionTimeoutError,
                       match='did not register its Temporal ACTIVITY poller'):
        await waiter('diskgraph-activity-tq-job', 'activity-pod-new', 1)

    assert describe.await_count == 1


@pytest.mark.asyncio
async def test_temporal_activity_poller_waiter_propagates_nonretryable_rpc_error(
):
    from temporalio.service import RPCError, RPCStatusCode

    describe = AsyncMock(
        side_effect=RPCError(message='not found', status=RPCStatusCode.
                             NOT_FOUND, raw_grpc_status=b''))
    waiter = _poller_waiter(describe)

    with pytest.raises(RPCError):
        await waiter('diskgraph-activity-tq-job', 'activity-pod-new', 1)

    assert describe.await_count == 1


def test_configured_singlehost_strategy_uses_instance_scoped_queue():
    config = replace(_diskgraph_pool_config(), instance_id='devserver-train-a')
    strategy = _configured_strategy_factory(config)(cast(PoolController, None))
    request = PlacementRequest(
        snapshot_id='snapshot',
        job_id='job',
        workflow_id='workflow',
        activity=SnapshotActivity.GEN_TT,
        resource_hint=ResourceHint(),
    )

    async def provision():
        decision = await strategy.route(request, None)
        return await strategy.provision(request, decision)

    handle = asyncio.run(provision())
    assert handle.task_queue == 'diskgraph-daemon-tq-devserver-train-a'


def test_normalize_state_for_singlehost_drops_k8s_placement_state():
    state = PoolState(
        placement_state=K8sPlacementState(),
        snapshots={
            'snap-1':
            SnapshotPlacement(worker='worker-a', state=SnapshotState.READY,
                              size_bytes=123),
        },
        dataplane_backups={
            'snap-1':
            DataplaneBackup(snapshot_id='snap-1',
                            dataplane_path='s3://b/snap-1'),
        },
    )
    config = _diskgraph_pool_config(pool_strategy='singlehost')

    normalized = _normalize_state_for_strategy(config, state)

    assert normalized is not None
    assert normalized.placement_state.type == 'singlehost'
    assert normalized.snapshots == {}
    assert normalized.dataplane_backups == state.dataplane_backups


def test_normalize_state_for_k8s_leaves_state_unchanged():
    state = PoolState(placement_state=K8sPlacementState())
    config = _diskgraph_pool_config(pool_strategy='k8s')

    assert _normalize_state_for_strategy(config, state) is None


def test_normalize_state_for_singlehost_leaves_non_dev_state_unchanged():
    state = PoolState(placement_state=K8sPlacementState())
    config = _diskgraph_pool_config(
        pool_strategy='singlehost',
        deploy_mode=DeployStage.PROD,
    )

    assert _normalize_state_for_strategy(config, state) is None


def test_scrub_unbacked_k8s_placement_ghosts_is_conservative():
    state = PoolState(placement_state=K8sPlacementState())
    placement_state = state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    state.snapshots = {
        'ghost': SnapshotPlacement(worker='worker-ghost'),
        'leased': SnapshotPlacement(worker='worker-leased', leases=['job-1']),
        'pvc-backed': SnapshotPlacement(worker='worker-pvc'),
        'copy-backed': SnapshotPlacement(worker='worker-copy'),
        'evicted-only': SnapshotPlacement(worker='worker-evicted'),
    }
    placement_state.pvcs['worker-pvc'] = PvcState(pool_worker_uid='worker-pvc')
    state.local_copies['copy-backed:worker-copy'] = SnapshotLocalCopyRecord(
        snapshot_id='copy-backed',
        pool_worker_uid='worker-copy',
        local_state=SnapshotLocalCopyState.WARM,
    )
    state.local_copies['evicted-only:worker-evicted'] = (
        SnapshotLocalCopyRecord(
            snapshot_id='evicted-only',
            pool_worker_uid='worker-evicted',
            local_state=SnapshotLocalCopyState.EVICTED,
        ))

    assert _scrub_unbacked_k8s_placement_ghosts(state)

    assert set(state.snapshots) == {'leased', 'pvc-backed', 'copy-backed'}


def _make_controller(tmp_path, strategy: Optional[PlacementStrategy] = None,
                     workflow_status_checker=None, dataplane_deleter=None,
                     local_snapshot_deleter=None, data_plane_storage_root=None,
                     lifecycle_policy=None, now_utc_fn=None, time_fn=None,
                     worker_startup_grace_secs: float =
                     POOL_WORKER_STARTUP_GRACE_SEC) -> PoolController:
    persistence = PoolStatePersistence(
        str(tmp_path / 'pool_state.json'),
        empty_state_factory=lambda: PoolState(placement_state=K8sPlacementState(
        )),
    )
    return PoolController(state=PoolState(placement_state=K8sPlacementState()),
                          persistence=persistence, strategy=strategy,
                          workflow_status_checker=workflow_status_checker,
                          dataplane_deleter=dataplane_deleter,
                          local_snapshot_deleter=local_snapshot_deleter,
                          data_plane_storage_root=data_plane_storage_root,
                          lifecycle_policy=lifecycle_policy,
                          now_utc_fn=now_utc_fn, time_fn=time_fn,
                          worker_startup_grace_secs=worker_startup_grace_secs)


def _make_controller_with_state(
        tmp_path, state: PoolState,
        strategy: Optional[MockPlacementStrategy] = None,
        time_fn=None) -> PoolController:
    persistence = PoolStatePersistence(
        str(tmp_path / 'pool_state.json'),
        empty_state_factory=lambda: PoolState(placement_state=K8sPlacementState(
        )),
    )
    return PoolController(state=state, persistence=persistence,
                          strategy=strategy, time_fn=time_fn)


def _state_with_expected_worker(worker: str = 'worker-a') -> PoolState:
    state = PoolState(placement_state=K8sPlacementState())
    placement_state = state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    placement_state.pvcs[worker] = PvcState(pool_worker_uid=worker)
    state.snapshots['snap-1'] = SnapshotPlacement(
        worker=worker,
        state=SnapshotState.READY,
        leases=['job-1'],
        lease_paths={'job-1': 'durable'},
        durable_handles={
            'job-1': _handle(pool_worker_uid=worker),
        },
    )
    return state


def _arm_silent_worker_retirement(controller: PoolController,
                                  not_before: float) -> None:
    """Model maintenance having started RPC-serving heartbeat grace."""
    controller._silent_worker_dead_not_before = not_before


def _hist_count(metric, **labels) -> float:
    total = 0.0
    for family in metric.collect():
        for sample in family.samples:
            if (sample.name.endswith('_count') and all(
                    sample.labels.get(k) == v for k, v in labels.items())):
                total += sample.value
    return total


def test_maintenance_tick_updates_state_gauges_and_placement_histogram(
        tmp_path):
    controller = _make_controller(tmp_path, strategy=MockPlacementStrategy())
    placement_count_before = _hist_count(metrics.POOL_PLACEMENT_DURATION)
    asyncio.run(
        controller.place_snapshot.coroutine(
            _request(snapshot_id='snap-wire', job_id='job-wire')))
    asyncio.run(controller._maintenance_once())

    # _maintenance_once -> metrics.update_state_gauges
    snapshots_total = sum(
        metrics.POOL_SNAPSHOTS.labels(state=s.value)._value.get()
        for s in SnapshotState)
    assert snapshots_total == 1
    assert metrics.POOL_ACTIVE_LEASES._value.get() == 1

    # @observe_place_snapshot recorded exactly one placement observation.
    assert (_hist_count(
        metrics.POOL_PLACEMENT_DURATION) == placement_count_before + 1)


def test_active_activity_metrics_use_persisted_lease_activity_after_restart():
    state = PoolState(
        placement_state=K8sPlacementState(),
        snapshots={
            'snap-restart':
            SnapshotPlacement(
                worker='worker-a',
                leases=['job-1'],
                lease_paths={'job-1': 'durable'},
                lease_activities={'job-1': SnapshotActivity.PREDICT},
            )
        },
    )

    metrics.update_state_gauges(state, {}, {})

    assert metrics.POOL_ACTIVE_ACTIVITIES.labels(
        pool_worker_uid='worker-a',
        activity='PREDICT',
    )._value.get() == 1


def test_lifecycle_policy_rejects_max_ssd_above_max_dataplane(tmp_path):
    with pytest.raises(ValueError, match='max_ssd_idle_secs'):
        _make_controller(
            tmp_path,
            lifecycle_policy=SnapshotLifecyclePolicy(
                min_ssd_idle_secs=60,
                max_ssd_idle_secs=7200,
                max_dataplane_retention_secs=3600,
            ),
        )


ALT_GRAPH_SNAPSHOT_ID = 'NA-11111111111111111111111111111111@v1'
CANONICAL_GRAPH_SNAPSHOT_ID = ('graph-11111111111111111111111111111111@v1')


def _request(
    snapshot_id: str = 'snap-1',
    job_id: str = 'job-1',
    activity: SnapshotActivity = SnapshotActivity.TRAIN,
    workflow_id: Optional[str] = None,
    min_gpu_slots: int = 1,
) -> PlacementRequest:
    return PlacementRequest(
        snapshot_id=snapshot_id,
        job_id=job_id,
        workflow_id=workflow_id if workflow_id is not None else job_id,
        activity=activity,
        resource_hint=ResourceHint(min_gpu_slots=min_gpu_slots),
    )


def _handle(
    task_queue: str = 'diskgraph-activity-tq-job-1',
    pool_worker_uid: Optional[str] = 'worker-a',
    kind: str = 'durable',
) -> ProvisionedHandle:
    return ProvisionedHandle(
        kind=kind,  # type: ignore[arg-type]
        pool_worker_uid=pool_worker_uid,
        task_queue=task_queue,
    )


def _open_snapshot(
    controller: PoolController,
    snapshot_id: str = 'snap-1',
    job_id: str = 'job-1',
    worker: str = 'worker-a',
    *,
    state: SnapshotState = SnapshotState.READY,
    task_queue: str = 'diskgraph-activity-tq-job-1',
) -> SnapshotPlacement:
    placement = SnapshotPlacement(
        worker=worker,
        state=state,
        leases=[job_id],
        lease_paths={job_id: 'durable'},
        durable_handles={
            job_id: _handle(task_queue=task_queue, pool_worker_uid=worker),
        },
    )
    controller.state.snapshots[snapshot_id] = placement
    controller._sync_lifecycle_from_leases(snapshot_id)
    return placement


def test_dataplane_snapshot_path_uses_snapshot_prefix_and_safe_segment():
    assert dataplane_snapshot_path(
        's3://bucket/env',
        'snap/with/slash',
    ) == 's3://bucket/env/snapshots/snap%2Fwith%2Fslash'


def test_starting_worker_shared_by_concurrent_different_snapshots(
        tmp_path, monkeypatch):
    """The second route sees the first reservation before any heartbeat."""
    async def run():
        controller = _make_controller(tmp_path)
        manager = _FakeK8sManager()
        uids = iter(['worker-a', 'worker-b', 'worker-c'])
        strategy = build_k8s_placement_strategy(
            manager,  # type: ignore[arg-type]
            PoolControllerPlacementContext(controller),
            capacity_tiers=[CapacityTier(1000, 4)],
            pool_worker_uid_factory=lambda: next(uids),
        )
        controller._set_placement_strategy(strategy)
        first_provisioning = asyncio.Event()
        allow_provision = asyncio.Event()
        second_routed = asyncio.Event()
        original_route = strategy.route

        async def observe_route(request, **kwargs):
            decision = await original_route(request, **kwargs)
            if request.snapshot_id == 'snap-2':
                second_routed.set()
            return decision

        monkeypatch.setattr(strategy, 'route', observe_route)

        async def provision(_uid):
            first_provisioning.set()
            await allow_provision.wait()

        manager.on_provision_pool_worker = provision
        first = asyncio.create_task(
            controller.place_snapshot.coroutine(
                replace(_request(), source_size_bytes=10 << 30)))
        second = None
        try:
            await asyncio.wait_for(first_provisioning.wait(), 10)
            second = asyncio.create_task(
                controller.place_snapshot.coroutine(
                    replace(_request('snap-2', 'job-2'),
                            source_size_bytes=10 << 30)))
            await asyncio.wait_for(second_routed.wait(), 10)
            assert set(_k8s_pvcs(controller)) == {'worker-a'}
            assert 'snap-2' not in controller._state.snapshots
            # report_status must be able to acquire the controller lock while
            # another placement waits for capacity.
            await asyncio.wait_for(
                controller.report_status.coroutine(
                    _report('worker-a', ready_snapshots=[],
                            in_progress_snapshots=['snap-1'],
                            pvc_total_bytes=1000 << 30,
                            pvc_free_bytes=1000 << 30)), 1)
            allow_provision.set()
            results = await asyncio.wait_for(asyncio.gather(first, second), 1)
            assert {result.pool_worker_uid
                    for result in results} == {'worker-a'}
            assert set(_k8s_pvcs(controller)) == {'worker-a'}
            assert controller._state.snapshots['snap-2'].leases == ['job-2']
        finally:
            for task in [first, second]:
                if task is not None:
                    task.cancel()
            await asyncio.gather(*[t for t in [first, second] if t is not None],
                                 return_exceptions=True)

    asyncio.run(run())


def _k8s_pvcs(controller: PoolController) -> dict[str, PvcState]:
    state = controller.state.placement_state
    assert isinstance(state, K8sPlacementState)
    return state.pvcs


class _ObservedCapacityEvent(asyncio.Event):
    """Observe the wait boundary without replacing routing or admission."""
    def __init__(self):
        super().__init__()
        self.waiting = asyncio.Event()

    async def wait(self):
        self.waiting.set()
        return await super().wait()


def _controller_with_starting_worker(
        tmp_path, *, grace: float = POOL_WORKER_STARTUP_GRACE_SEC,
        clock: Optional[_FakeClock] = None):
    controller = _make_controller(tmp_path, worker_startup_grace_secs=grace,
                                  now_utc_fn=clock.now_utc if clock else None)
    manager = _FakeK8sManager()
    uids = (f'worker-new-{i}' for i in count())
    controller._set_placement_strategy(
        build_k8s_placement_strategy(
            manager,  # type: ignore[arg-type]
            PoolControllerPlacementContext(controller),
            capacity_tiers=[
                CapacityTier(50, 4), CapacityTier(1000, 1), CapacityTier(1000, 4)
            ],
            pool_worker_uid_factory=lambda: next(uids)))
    _k8s_pvcs(controller)['worker-starting'] = PvcState(
        pool_worker_uid='worker-starting', tier_ssd_gib=1000, tier_gpu_count=4,
        created_at=controller._now_utc())
    controller._state.snapshots['snap-starting'] = SnapshotPlacement(
        worker='worker-starting', size_bytes=100 << 30, leases=['job-starting'],
        lease_paths={'job-starting': 'durable'},
        lease_activities={'job-starting': SnapshotActivity.TRAIN})
    return controller, manager


def test_starting_worker_heartbeat_before_wait_is_not_lost(tmp_path):
    async def run():
        controller, _ = _controller_with_starting_worker(tmp_path)

        class HeartbeatBeforeWait(asyncio.Event):
            async def wait(self):
                # Deliver after capacity was checked and locks released, but
                # BEFORE awaiting the captured event. A clear() would lose it.
                await controller.report_status.coroutine(
                    _report('worker-starting', ready_snapshots=[],
                            in_progress_snapshots=['snap-starting'],
                            pvc_total_bytes=1000 << 30,
                            pvc_free_bytes=1000 << 30))
                return await super().wait()

        controller._worker_capacity_changed = HeartbeatBeforeWait()
        result = await asyncio.wait_for(
            controller.place_snapshot.coroutine(
                replace(_request(), source_size_bytes=10 << 30)), 10)
        assert result.pool_worker_uid == 'worker-starting'
        assert len(_k8s_pvcs(controller)) == 1

    asyncio.run(run())


@pytest.mark.parametrize('blocker', [
    'expired',
    'ssd_reserved',
    'gpu_reserved',
    'ssd_hint',
    'gpu_hint',
    'stale',
    'dead',
    'draining',
    'incompatible_tier',
])
def test_starting_worker_unsuitable_does_not_delay_scale_out(tmp_path, blocker):
    async def run():
        controller, _ = _controller_with_starting_worker(tmp_path)
        pvc = _k8s_pvcs(controller)['worker-starting']
        placement = controller._state.snapshots['snap-starting']
        if blocker == 'expired':
            pvc.created_at -= timedelta(seconds=91)
        elif blocker == 'ssd_reserved':
            placement.size_bytes = 840 << 30
        elif blocker == 'gpu_reserved':
            pvc.tier_gpu_count = 1
        elif blocker == 'ssd_hint':
            pvc.tier_ssd_gib = 50
        elif blocker == 'gpu_hint':
            pvc.tier_gpu_count = 1
        elif blocker == 'incompatible_tier':
            pvc.tier_ssd_gib = 900
        else:
            controller._worker_inventory['worker-starting'] = WorkerInventory(
                pool_worker_uid='worker-starting', state=_snapshot(),
                status=WorkerStatus(blocker.upper()), last_heartbeat=0)
        event = _ObservedCapacityEvent()
        controller._worker_capacity_changed = event
        request = replace(
            _request(), source_size_bytes=10 << 30, resource_hint=ResourceHint(
                min_ssd_gib=100 if blocker == 'ssd_hint' else 0,
                min_gpu_slots=2 if blocker == 'gpu_hint' else 1))
        result = await asyncio.wait_for(
            controller.place_snapshot.coroutine(request), 10)
        assert result.pool_worker_uid != 'worker-starting'
        assert not event.waiting.is_set()

    asyncio.run(run())


def test_starting_worker_wait_can_use_another_ready_worker(tmp_path):
    async def run():
        controller, _ = _controller_with_starting_worker(tmp_path)
        event = _ObservedCapacityEvent()
        controller._worker_capacity_changed = event
        task = asyncio.create_task(
            controller.place_snapshot.coroutine(
                replace(_request(), source_size_bytes=10 << 30)))
        try:
            await asyncio.wait_for(event.waiting.wait(), 10)
            await controller.report_status.coroutine(
                _report('worker-other', ready_snapshots=[],
                        pvc_total_bytes=1000 << 30, pvc_free_bytes=1000 << 30))
            result = await asyncio.wait_for(task, 10)
            assert result.pool_worker_uid == 'worker-other'
            assert 'worker-starting' not in controller._worker_inventory
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_starting_worker_wait_cancellation_leaves_no_reservation(tmp_path):
    async def run():
        controller, _ = _controller_with_starting_worker(tmp_path)
        event = _ObservedCapacityEvent()
        controller._worker_capacity_changed = event
        task = asyncio.create_task(
            controller.place_snapshot.coroutine(_request()))
        await asyncio.wait_for(event.waiting.wait(), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert set(controller._state.snapshots) == {'snap-starting'}
        assert set(_k8s_pvcs(controller)) == {'worker-starting'}
        assert not controller._state_lock.locked()
        async with controller._snapshot_locks.acquire('snap-1'):
            pass

    asyncio.run(run())


def test_starting_worker_grace_expires_and_retry_keeps_new_worker(tmp_path):
    async def run():
        controller, manager = _controller_with_starting_worker(
            tmp_path, grace=0.05)
        event = _ObservedCapacityEvent()
        controller._worker_capacity_changed = event
        first = await asyncio.wait_for(
            controller.place_snapshot.coroutine(_request()), 10)
        assert event.waiting.is_set()
        assert first.pool_worker_uid != 'worker-starting'
        retry = await asyncio.wait_for(
            controller.place_snapshot.coroutine(_request()), 10)
        assert retry.pool_worker_uid == first.pool_worker_uid
        assert len(_k8s_pvcs(controller)) == 2
        assert len(manager.provision_pool_worker_calls) == 1

    asyncio.run(run())


def test_starting_worker_waiters_recheck_reservations_after_wakeup(tmp_path):
    async def run():
        controller, _ = _controller_with_starting_worker(tmp_path)
        # Two 400GiB admissions cannot BOTH fit alongside the 100GiB already
        # reserved in an 850GiB usable tier, even before the next heartbeat.
        waiting = asyncio.Queue()

        class CountWaiters(asyncio.Event):
            async def wait(self):
                waiting.put_nowait(None)
                return await super().wait()

        controller._worker_capacity_changed = CountWaiters()
        tasks = [
            asyncio.create_task(
                controller.place_snapshot.coroutine(
                    replace(_request(f'snap-{i}', f'job-{i}'),
                            source_size_bytes=100 << 30))) for i in range(2)
        ]
        try:
            for _ in tasks:
                await asyncio.wait_for(waiting.get(), 10)
            await controller.report_status.coroutine(
                _report('worker-starting', ready_snapshots=[],
                        in_progress_snapshots=['snap-starting'],
                        pvc_total_bytes=1000 << 30, pvc_free_bytes=1000 << 30))
            results = await asyncio.wait_for(asyncio.gather(*tasks), 10)
            assert sum(r.pool_worker_uid == 'worker-starting'
                       for r in results) == 1
            assert len(_k8s_pvcs(controller)) == 2
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())


def test_starting_worker_wakeups_do_not_reset_request_budget(tmp_path):
    async def run():
        controller, _ = _controller_with_starting_worker(tmp_path, grace=0.05)
        waits = 0

        class RenewedStartup(asyncio.Event):
            async def wait(self):
                nonlocal waits
                waits += 1
                # Model more fresh startups appearing while unrelated workers
                # heartbeat. Worker age alone must not extend this request.
                _k8s_pvcs(controller)[
                    'worker-starting'].created_at = controller._now_utc()
                await controller.report_status.coroutine(
                    _report('worker-too-small', ready_snapshots=[],
                            pvc_total_bytes=1, pvc_free_bytes=1))
                controller._worker_capacity_changed = RenewedStartup()
                return await super().wait()

        controller._worker_capacity_changed = RenewedStartup()
        result = await asyncio.wait_for(
            controller.place_snapshot.coroutine(
                replace(_request(), source_size_bytes=10 << 30)), 10)
        assert waits > 1
        assert result.pool_worker_uid != 'worker-starting'

    asyncio.run(run())


def test_starting_worker_age_survives_controller_restart(tmp_path):
    clock = _FakeClock()
    controller, _ = _controller_with_starting_worker(tmp_path, clock=clock)
    asyncio.run(controller._persistence.save(controller._state))
    state = asyncio.run(controller._persistence.load())
    clock.advance(91)
    restarted = _make_controller_with_state(tmp_path, state)
    restarted._now_utc = clock.now_utc
    restarted._set_placement_strategy(
        build_k8s_placement_strategy(
            _FakeK8sManager(),  # type: ignore[arg-type]
            PoolControllerPlacementContext(restarted),
            capacity_tiers=[CapacityTier(1000, 4)],
            pool_worker_uid_factory=lambda: 'worker-new'))
    event = _ObservedCapacityEvent()
    restarted._worker_capacity_changed = event
    result = asyncio.run(restarted.place_snapshot.coroutine(_request()))
    assert result.pool_worker_uid == 'worker-new'
    assert not event.waiting.is_set()


def test_dataplane_snapshot_path_uses_diskgraph_service_layout():
    assert dataplane_snapshot_path(
        's3://bucket/env',
        'graph-1bf8d6a669f6a6caba68a2393f1044e3@v1',
    ) == ('s3://bucket/env/diskgraph_snapshots/'
          'NA-1bf8d6a669f6a6caba68a2393f1044e3/v1')


def _snapshot(**overrides: Any) -> WorkerStateSnapshot:
    # ``Dict[str, Any]`` is load-bearing: without it, pyright infers the
    # dict's value type as the union of the concrete literals, and the
    # unpacking into ``WorkerStateSnapshot(**defaults)`` then fails
    # type-checking on every field.
    pvc_total_bytes = overrides.pop('pvc_total_bytes', 1_000_000)
    pvc_free_bytes = overrides.pop('pvc_free_bytes', 200_000)
    defaults: Dict[str, Any] = dict(
        ssd_total_bytes=pvc_total_bytes,
        ssd_used_bytes=max(pvc_total_bytes - pvc_free_bytes, 0),
        ready_snapshots=['snap-1'],
        in_progress_snapshots=[],
    )
    defaults.update(overrides)
    return WorkerStateSnapshot(**defaults)


def _report(pool_worker_uid: str = 'worker-a', event: WorkerEvent | None = None,
            worker_storage_mode: PoolStorageMode = (
                PoolStorageMode.DAEMON_ON_PVC),
            **snapshot_overrides: Any) -> WorkerStatusReport:
    return WorkerStatusReport(worker_name=pool_worker_uid,
                              state=_snapshot(**snapshot_overrides),
                              worker_storage_mode=worker_storage_mode,
                              event=event)


def _seed_ready_local_copy(controller: PoolController,
                           worker: str = 'worker-a') -> None:
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker=worker, state=SnapshotState.READY)
    controller.state.local_copies[f'snap-1:{worker}'] = (
        SnapshotLocalCopyRecord(
            snapshot_id='snap-1',
            pool_worker_uid=worker,
            local_state=SnapshotLocalCopyState.WARM,
        ))


def _snapshot_ready_event(snapshot_id: str, timestamp,
                          size_bytes: int = 4096) -> WorkerEvent:
    return WorkerEvent(
        type=WorkerEventType.SNAPSHOT_READY,
        snapshot_id=snapshot_id,
        size_bytes=size_bytes,
        timestamp=timestamp,
    )


def _snapshot_dataplane_ready_event(snapshot_id: str, timestamp) -> WorkerEvent:
    return WorkerEvent(
        type=WorkerEventType.SNAPSHOT_PERSIST_UPLOADED,
        snapshot_id=snapshot_id,
        dataplane_path=f's3://bucket/{snapshot_id}',
        timestamp=timestamp,
    )


@pytest.fixture(autouse=True)
def stub_dataplane_dirsize(monkeypatch):
    """Stub the dataplane scan in ``_ensure_backup_size`` so unit tests never
    hit real S3. Tests that exercise the scan override this themselves."""
    from io_utils import fsutil
    monkeypatch.setattr(fsutil, 'dirsize', lambda path: 0)


def test_heartbeat_protocol_constants_match_design():
    assert POOL_HEARTBEAT_INTERVAL_SECS == 30
    assert CONTROLLER_MAINTENANCE_INTERVAL_SECS == POOL_HEARTBEAT_INTERVAL_SECS
    assert WORKER_STALE_THRESHOLD_SECS == 60
    # 300s: node-death-resilience threshold, aligned with k8s's default
    # node.kubernetes.io/unreachable:NoExecute toleration (retire-on-DEAD).
    assert WORKER_DEAD_THRESHOLD_SECS == 300
    assert WORKER_STALE_THRESHOLD_SECS < WORKER_DEAD_THRESHOLD_SECS


def test_report_status_records_new_node(tmp_path):
    controller = _make_controller(tmp_path)
    asyncio.run(controller.report_status.coroutine(_report()))
    inv = controller.worker_inventory['worker-a']
    assert inv.pool_worker_uid == 'worker-a'
    assert inv.state.pvc_free_bytes == 200_000
    assert inv.state.ready_snapshots == ['snap-1']
    assert inv.gpu_slots_total == 1
    assert inv.gpu_slots_in_use == 0
    assert inv.status == WorkerStatus.READY
    assert inv.last_heartbeat > 0


@pytest.mark.parametrize('ready_source', ['heartbeat', 'event'])
def test_ready_report_does_not_cancel_committed_eviction(
        tmp_path, ready_source):
    controller = _make_controller(tmp_path)
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a', state=SnapshotState.READY)
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.EVICTING,
    ))
    controller.worker_inventory['worker-a'] = WorkerInventory(
        pool_worker_uid='worker-a',
        state=_snapshot(ready_snapshots=[]),
    )
    report = (_report(
        ready_snapshots=['snap-1']) if ready_source == 'heartbeat' else _report(
            ready_snapshots=[],
            event=WorkerEvent(
                type=WorkerEventType.SNAPSHOT_READY,
                snapshot_id='snap-1',
            ),
        ))

    asyncio.run(controller.report_status.coroutine(report))

    assert controller.state.local_copies[
        'snap-1:worker-a'].local_state == SnapshotLocalCopyState.EVICTING
    assert controller._warm_location_uid('snap-1') is None


def test_report_status_ready_heartbeat_rebuilds_warm_snapshot_state(tmp_path):
    controller = _make_controller(tmp_path)

    asyncio.run(
        controller.report_status.coroutine(
            _report(ready_snapshots=['snap-ready'])))

    placement = controller.state.snapshots['snap-ready']
    local = controller.state.local_copies['snap-ready:worker-a']
    compliance = controller.state.snapshot_compliance['snap-ready']
    assert placement.worker == 'worker-a'
    assert placement.state == SnapshotState.READY
    assert local.pool_worker_uid == 'worker-a'
    assert local.local_state == SnapshotLocalCopyState.WARM
    assert local.evictable_under_pressure_at is not None
    assert local.force_evict_at is not None
    assert compliance.active_consumer_count == 0
    assert compliance.compliance_delete_at is not None

    loaded = asyncio.run(controller._persistence.load())
    assert loaded.snapshots['snap-ready'] == placement
    assert loaded.local_copies['snap-ready:worker-a'] == local
    assert loaded.snapshot_compliance['snap-ready'] == compliance


def test_report_status_ready_heartbeat_pauses_local_clock_while_leased(
        tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    asyncio.run(controller.place_snapshot.coroutine(_request()))

    asyncio.run(
        controller.report_status.coroutine(_report(ready_snapshots=['snap-1'])))

    local = controller.state.local_copies['snap-1:worker-a']
    compliance = controller.state.snapshot_compliance['snap-1']
    assert local.local_state == SnapshotLocalCopyState.WARM
    assert local.evictable_under_pressure_at is None
    assert local.force_evict_at is None
    assert compliance.active_consumer_count == 1
    assert compliance.compliance_delete_at is None


def test_report_status_ready_heartbeat_preserves_known_snapshot_size(tmp_path):
    controller = _make_controller(tmp_path)
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        state=SnapshotState.RESERVED,
        size_bytes=4096,
    )

    asyncio.run(controller.report_status.coroutine(_report()))

    placement = controller.state.snapshots['snap-1']
    assert placement.state == SnapshotState.READY
    assert placement.size_bytes == 4096


def test_report_status_absent_snapshot_does_not_evict_existing_warm_state(
        tmp_path):
    controller = _make_controller(tmp_path)
    asyncio.run(controller.report_status.coroutine(_report(ready_snapshots=[])))
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        state=SnapshotState.READY,
        size_bytes=4096,
    )
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.WARM,
    ))

    asyncio.run(controller.report_status.coroutine(_report(ready_snapshots=[])))

    assert controller.state.snapshots['snap-1'].state == SnapshotState.READY
    assert controller.state.local_copies[
        'snap-1:worker-a'].local_state == SnapshotLocalCopyState.WARM


def test_first_report_missing_known_local_copy_marks_evicted(tmp_path):
    controller = _make_controller(tmp_path)
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        state=SnapshotState.READY,
        size_bytes=4096,
    )
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.WARM,
    ))

    asyncio.run(
        controller.report_status.coroutine(
            _report(ready_snapshots=[], in_progress_snapshots=[])))

    assert 'snap-1' not in controller.state.snapshots
    local = controller.state.local_copies['snap-1:worker-a']
    assert local.local_state == SnapshotLocalCopyState.EVICTED
    assert local.evicted_at is not None

    loaded = asyncio.run(controller._persistence.load())
    assert loaded.local_copies[
        'snap-1:worker-a'].local_state == SnapshotLocalCopyState.EVICTED


def test_first_report_missing_active_local_copy_marks_evicted_without_dispatch(
        tmp_path):
    controller = _make_controller(tmp_path)
    placement = _open_snapshot(controller)
    placement.size_bytes = 4096
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.WARM,
    ))

    asyncio.run(
        controller.report_status.coroutine(
            _report(ready_snapshots=[], in_progress_snapshots=[])))

    assert controller.state.local_copies[
        'snap-1:worker-a'].local_state == SnapshotLocalCopyState.EVICTED
    placement = controller.state.snapshots['snap-1']
    assert placement.durable_handles['job-1'].task_queue == (
        'diskgraph-activity-tq-job-1')
    assert placement.leases == ['job-1']


def test_first_report_missing_inactive_local_copy_does_not_dispatch(tmp_path):
    controller = _make_controller(tmp_path)
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.WARM,
    ))

    asyncio.run(
        controller.report_status.coroutine(
            _report(ready_snapshots=[], in_progress_snapshots=[])))

    assert controller.state.local_copies[
        'snap-1:worker-a'].local_state == SnapshotLocalCopyState.EVICTED
    assert 'snap-1' not in controller.state.snapshots


def test_first_report_keeps_known_ready_local_copy(tmp_path):
    controller = _make_controller(tmp_path)
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        state=SnapshotState.READY,
    )
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.WARM,
    ))

    asyncio.run(
        controller.report_status.coroutine(
            _report(ready_snapshots=['snap-1'], in_progress_snapshots=[])))

    assert controller.state.snapshots['snap-1'].state == SnapshotState.READY
    assert controller.state.local_copies[
        'snap-1:worker-a'].local_state == SnapshotLocalCopyState.WARM


def test_first_report_keeps_known_in_progress_local_copy(tmp_path):
    controller = _make_controller(tmp_path)
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        state=SnapshotState.RESERVED,
    )
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.WARM,
    ))

    asyncio.run(
        controller.report_status.coroutine(
            _report(ready_snapshots=[], in_progress_snapshots=['snap-1'])))

    assert controller.state.snapshots['snap-1'].state == SnapshotState.RESERVED
    assert controller.state.local_copies[
        'snap-1:worker-a'].local_state == SnapshotLocalCopyState.WARM


def test_report_status_overwrites_previous_inventory(tmp_path):
    """A node's subsequent report fully replaces its prior inventory —
    stale entries (finished jobs, departed snapshots) must not leak.
    """
    controller = _make_controller(tmp_path)
    asyncio.run(
        controller.report_status.coroutine(
            _report(ready_snapshots=['snap-1', 'snap-2'], pvc_free_bytes=500)))
    asyncio.run(
        controller.report_status.coroutine(
            _report(ready_snapshots=['snap-2'], pvc_free_bytes=250)))
    inv = controller.worker_inventory['worker-a']
    assert inv.state.ready_snapshots == ['snap-2']
    assert inv.state.pvc_free_bytes == 250


def test_report_status_activity_scoped_does_not_overwrite_daemon_inventory(
        tmp_path):
    controller = _make_controller(tmp_path)
    asyncio.run(
        controller.report_status.coroutine(
            _report(ready_snapshots=['snap-1'], pvc_free_bytes=500)))

    asyncio.run(
        controller.report_status.coroutine(
            _report(
                worker_storage_mode=PoolStorageMode.ACTIVITY_ON_PVC,
                ready_snapshots=[],
                pvc_free_bytes=0,
            )))

    inv = controller.worker_inventory['worker-a']
    assert inv.state.ready_snapshots == ['snap-1']
    assert inv.state.pvc_free_bytes == 500


def test_report_status_is_independent_across_nodes(tmp_path):
    controller = _make_controller(tmp_path)
    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-a', ready_snapshots=[])))
    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-b', ready_snapshots=[])))
    assert set(controller.worker_inventory) == {'worker-a', 'worker-b'}


def test_worker_health_sweep_transitions_ready_stale_dead(tmp_path):
    controller = _make_controller(tmp_path)
    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-ready')))
    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-stale')))
    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-dead')))

    now = time.time()
    controller.worker_inventory['worker-ready'].last_heartbeat = (
        now - WORKER_STALE_THRESHOLD_SECS + 1)
    controller.worker_inventory['worker-stale'].last_heartbeat = (
        now - WORKER_STALE_THRESHOLD_SECS)
    controller.worker_inventory['worker-dead'].last_heartbeat = (
        now - WORKER_DEAD_THRESHOLD_SECS)

    controller._sweep_worker_health(now)

    assert controller.worker_inventory[
        'worker-ready'].status == WorkerStatus.READY
    assert controller.worker_inventory[
        'worker-stale'].status == WorkerStatus.STALE
    assert controller.worker_inventory[
        'worker-dead'].status == WorkerStatus.DEAD


def test_report_status_resets_stale_worker_to_ready(tmp_path):
    controller = _make_controller(tmp_path)
    asyncio.run(controller.report_status.coroutine(_report()))
    controller.worker_inventory['worker-a'].last_heartbeat = (
        time.time() - WORKER_STALE_THRESHOLD_SECS)
    controller._sweep_worker_health()
    assert controller.worker_inventory['worker-a'].status == WorkerStatus.STALE

    asyncio.run(controller.report_status.coroutine(_report()))

    assert controller.worker_inventory['worker-a'].status == WorkerStatus.READY


def test_report_status_local_ready_event_updates_snapshot_progress(tmp_path):
    controller = _make_controller(tmp_path)
    ready = WorkerEvent(type=WorkerEventType.SNAPSHOT_READY,
                        snapshot_id='snap-1', size_bytes=2048)

    asyncio.run(
        controller.report_status.coroutine(
            _report(event=ready, ready_snapshots=[])))
    placement = controller.state.snapshots['snap-1']
    assert placement.state == SnapshotState.READY
    assert placement.size_bytes == 2048


def test_report_status_ingest_started_event_transitions_to_ingesting(tmp_path):
    """SNAPSHOT_INGEST_STARTED transitions placement from RESERVED → INGESTING,
    making pool_snapshots_total{state='INGESTING'} non-zero during active ingest."""
    controller = _make_controller(tmp_path)
    controller.state.snapshots['snap-1'] = SnapshotPlacement(worker='worker-1')
    assert controller.state.snapshots['snap-1'].state == SnapshotState.RESERVED

    event = WorkerEvent(type=WorkerEventType.SNAPSHOT_INGEST_STARTED,
                        snapshot_id='snap-1')
    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-1', event=event,
                    ready_snapshots=[])))

    assert controller.state.snapshots['snap-1'].state == SnapshotState.INGESTING


def test_report_status_restore_started_event_transitions_to_restoring(tmp_path):
    """SNAPSHOT_RESTORE_STARTED transitions placement from RESERVED → RESTORING."""
    controller = _make_controller(tmp_path)
    controller.state.snapshots['snap-1'] = SnapshotPlacement(worker='worker-1')
    backup = DataplaneBackup(
        snapshot_id='snap-1',
        dataplane_path='s3://bucket/snap-1',
        size_bytes=4096,
    )
    controller.state.dataplane_backups['snap-1'] = backup

    event = WorkerEvent(type=WorkerEventType.SNAPSHOT_RESTORE_STARTED,
                        snapshot_id='snap-1')
    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-1', event=event,
                    ready_snapshots=[])))

    assert controller.state.snapshots['snap-1'].state == SnapshotState.RESTORING
    assert controller.state.dataplane_backups['snap-1'] == backup
    loaded = asyncio.run(controller._persistence.load())
    assert loaded.dataplane_backups['snap-1'] == backup


def test_report_status_ingest_started_no_placement_is_noop(tmp_path):
    """SNAPSHOT_INGEST_STARTED for an unknown snapshot does not create a placement."""
    controller = _make_controller(tmp_path)

    event = WorkerEvent(type=WorkerEventType.SNAPSHOT_INGEST_STARTED,
                        snapshot_id='snap-unknown')
    asyncio.run(
        controller.report_status.coroutine(
            _report(event=event, ready_snapshots=[])))

    assert 'snap-unknown' not in controller.state.snapshots


def test_report_status_local_ready_from_eph_pod_does_not_claim_placement(
        tmp_path):
    """SNAPSHOT_READY from an ACTIVITY_ON_EMPTYDIR (overflow eph) pod must
    not be recorded as a durable local copy. Otherwise the eph pod's UID
    would land in placement.worker / local_copies, and the next reconcile
    pass would call provision_pool_worker(eph-uid) → genesis loop that
    spawned the orphan EH-prefix daemon Deployments on antest-349a2f.
    """
    controller = _make_controller(tmp_path)
    ready = WorkerEvent(type=WorkerEventType.SNAPSHOT_READY,
                        snapshot_id='snap-1', size_bytes=2048)

    asyncio.run(
        controller.report_status.coroutine(
            _report(
                pool_worker_uid='pool-eh-worker-deadbeef',
                worker_storage_mode=PoolStorageMode.ACTIVITY_ON_EMPTYDIR,
                event=ready,
                ready_snapshots=[],
            )))

    assert controller.state.snapshots == {}
    assert controller.state.local_copies == {}


@pytest.mark.parametrize('event_type', [
    WorkerEventType.SNAPSHOT_INGEST_STARTED,
    WorkerEventType.SNAPSHOT_RESTORE_STARTED,
])
def test_report_status_started_event_from_eph_pod_does_not_dirty_home(
        tmp_path, event_type):
    controller = _make_controller(tmp_path)
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='durable-worker',
        state=SnapshotState.READY,
        size_bytes=2048,
    )
    event = WorkerEvent(type=event_type, snapshot_id='snap-1')

    asyncio.run(
        controller.report_status.coroutine(
            _report(
                pool_worker_uid='pool-eh-worker-deadbeef',
                worker_storage_mode=PoolStorageMode.ACTIVITY_ON_EMPTYDIR,
                event=event,
                ready_snapshots=[],
            )))

    placement = controller.state.snapshots['snap-1']
    assert placement.worker == 'durable-worker'
    assert placement.state == SnapshotState.READY
    assert placement.size_bytes == 2048


def test_report_status_local_ready_from_activity_on_pvc_records_placement(
        tmp_path):
    """SNAPSHOT_READY from an ACTIVITY_ON_PVC pod (durable cache-pool
    activity sharing the daemon's PVC) still records placement — the
    local copy IS durable, just written by the activity pod rather than
    the daemon."""
    controller = _make_controller(tmp_path)
    placement_state = controller.state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    placement_state.pvcs['worker-a'] = PvcState(pool_worker_uid='worker-a')
    controller.state.snapshots['snap-1'] = SnapshotPlacement(worker='worker-a')
    ready = WorkerEvent(type=WorkerEventType.SNAPSHOT_READY,
                        snapshot_id='snap-1', size_bytes=2048)

    asyncio.run(
        controller.report_status.coroutine(
            _report(
                worker_storage_mode=PoolStorageMode.ACTIVITY_ON_PVC,
                event=ready,
                ready_snapshots=[],
            )))

    placement = controller.state.snapshots['snap-1']
    assert placement.state == SnapshotState.READY
    assert placement.worker == 'worker-a'
    assert controller.state.local_copies[
        'snap-1:worker-a'].local_state == SnapshotLocalCopyState.WARM


def test_activity_on_pvc_ready_without_placement_tracks_physical_copy(tmp_path):
    controller = _make_controller(tmp_path)
    placement_state = controller.state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    placement_state.pvcs['worker-a'] = PvcState(pool_worker_uid='worker-a')
    assert 'worker-a' not in controller.worker_inventory
    controller.state.dataplane_backups['snap-1'] = DataplaneBackup(
        snapshot_id='snap-1',
        dataplane_path='s3://bucket/snap-1',
        size_bytes=0,
    )
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.EVICTED,
    ))
    ready = WorkerEvent(type=WorkerEventType.SNAPSHOT_READY,
                        snapshot_id='snap-1', size_bytes=4096)

    asyncio.run(
        controller.report_status.coroutine(
            _report(
                worker_storage_mode=PoolStorageMode.ACTIVITY_ON_PVC,
                event=ready,
                ready_snapshots=[],
            )))

    assert 'snap-1' not in controller.state.snapshots
    assert controller.state.dataplane_backups['snap-1'].size_bytes == 4096
    local = controller.state.local_copies['snap-1:worker-a']
    assert local.local_state == SnapshotLocalCopyState.WARM
    assert local.size_bytes == 4096
    assert local.evictable_under_pressure_at is not None
    assert local.force_evict_at is not None


def test_report_status_ready_from_stale_pvc_worker_does_not_steal_current_home(
        tmp_path):
    controller = _make_controller(tmp_path)
    placement_state = controller.state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    placement_state.pvcs['worker-current'] = PvcState(
        pool_worker_uid='worker-current')
    placement_state.pvcs['worker-old'] = PvcState(pool_worker_uid='worker-old')
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-current',
        state=SnapshotState.READY,
        size_bytes=2048,
    )
    ready = WorkerEvent(type=WorkerEventType.SNAPSHOT_READY,
                        snapshot_id='snap-1', size_bytes=4096)

    asyncio.run(
        controller.report_status.coroutine(
            _report(
                pool_worker_uid='worker-old',
                worker_storage_mode=PoolStorageMode.ACTIVITY_ON_PVC,
                event=ready,
                ready_snapshots=[],
            )))

    placement = controller.state.snapshots['snap-1']
    assert placement.worker == 'worker-current'
    assert placement.state == SnapshotState.READY
    assert placement.size_bytes == 2048
    old_copy = controller.state.local_copies['snap-1:worker-old']
    assert old_copy.local_state == SnapshotLocalCopyState.WARM
    assert old_copy.size_bytes == 4096


def test_report_status_heartbeat_ready_from_stale_worker_does_not_steal_home(
        tmp_path):
    controller = _make_controller(tmp_path)
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-current',
        state=SnapshotState.READY,
        size_bytes=2048,
    )

    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-old', event=None,
                    ready_snapshots=['snap-1'])))

    placement = controller.state.snapshots['snap-1']
    assert placement.worker == 'worker-current'
    assert placement.state == SnapshotState.READY
    assert placement.size_bytes == 2048
    assert controller.state.local_copies[
        'snap-1:worker-old'].local_state == SnapshotLocalCopyState.WARM


@pytest.mark.parametrize('event_type', [
    WorkerEventType.SNAPSHOT_INGEST_STARTED,
    WorkerEventType.SNAPSHOT_RESTORE_STARTED,
])
def test_report_status_started_event_from_stale_pvc_worker_does_not_dirty_home(
        tmp_path, event_type):
    controller = _make_controller(tmp_path)
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-current',
        state=SnapshotState.READY,
        size_bytes=2048,
    )
    event = WorkerEvent(type=event_type, snapshot_id='snap-1')

    asyncio.run(
        controller.report_status.coroutine(
            _report(
                pool_worker_uid='worker-old',
                worker_storage_mode=PoolStorageMode.ACTIVITY_ON_PVC,
                event=event,
                ready_snapshots=[],
            )))

    placement = controller.state.snapshots['snap-1']
    assert placement.worker == 'worker-current'
    assert placement.state == SnapshotState.READY
    assert placement.size_bytes == 2048


def test_report_status_dataplane_ready_event_records_dataplane_backup(tmp_path):
    controller = _make_controller(tmp_path)
    event = WorkerEvent(
        type=WorkerEventType.SNAPSHOT_PERSIST_UPLOADED,
        snapshot_id='snap-1',
        dataplane_path='s3://bucket/env/snapshots/snap-1',
        size_bytes=512,
    )

    asyncio.run(
        controller.report_status.coroutine(
            _report(event=event, ready_snapshots=[])))

    backup = controller.state.dataplane_backups['snap-1']
    assert backup.snapshot_id == 'snap-1'
    assert backup.dataplane_path == 's3://bucket/env/snapshots/snap-1'
    # Size captured from the persist event (no placement here).
    assert backup.size_bytes == 512
    assert 'snap-1' in controller.state.snapshot_compliance
    assert controller.state.snapshots == {}

    loaded = asyncio.run(controller._persistence.load())
    assert loaded.dataplane_backups['snap-1'] == backup


def test_ssd_corruption_invalidates_current_home_without_losing_active_lease(
        tmp_path):
    """Deleting corrupt L1 must remove WARM routing, not lease coordination."""
    controller = _make_controller(tmp_path)
    corrupted_at = now_utc()
    placement = _open_snapshot(controller)
    placement.size_bytes = 4096
    handle = placement.durable_handles['job-1']
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.WARM,
        size_bytes=4096,
    ))
    controller.state.dataplane_backups['snap-1'] = DataplaneBackup(
        snapshot_id='snap-1',
        dataplane_path='s3://bucket/snap-1',
    )
    controller.worker_inventory['worker-a'] = WorkerInventory(
        pool_worker_uid='worker-a', state=_snapshot())
    event = WorkerEvent(
        type=WorkerEventType.SSD_CACHE_CORRUPTED,
        snapshot_id='snap-1',
        timestamp=corrupted_at,
    )

    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-a', event=event,
                    ready_snapshots=[])))

    local = controller.state.local_copies['snap-1:worker-a']
    assert local.local_state == SnapshotLocalCopyState.EVICTED
    assert local.evicted_at == corrupted_at
    placement = controller.state.snapshots['snap-1']
    assert placement.state == SnapshotState.RESERVED
    assert placement.leases == ['job-1']
    assert placement.lease_paths == {'job-1': 'durable'}
    assert placement.durable_handles == {'job-1': handle}
    assert (controller._placement_status('snap-1',
                                         placement) == PlacementStatus.COLD)

    loaded = asyncio.run(controller._persistence.load())
    assert loaded.local_copies['snap-1:worker-a'] == local
    assert loaded.snapshots['snap-1'] == placement


def test_unleased_corruption_creates_idempotent_tombstone(tmp_path):
    ready_at = now_utc()
    controller = _make_controller(tmp_path)
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a', state=SnapshotState.READY, size_bytes=4096)
    corruption = WorkerEvent(
        type=WorkerEventType.SSD_CACHE_CORRUPTED,
        snapshot_id='snap-1',
        timestamp=ready_at,
    )

    asyncio.run(
        controller.report_status.coroutine(
            _report(
                event=corruption,
                ready_snapshots=[],
            )))

    local = controller.state.local_copies['snap-1:worker-a']
    assert local.local_state == SnapshotLocalCopyState.EVICTED
    assert local.evicted_at == ready_at
    assert 'snap-1' not in controller.state.snapshots
    expected_state = to_json(controller.state)

    # Duplicate corruption is idempotent.
    asyncio.run(
        controller.report_status.coroutine(
            _report(
                event=corruption,
                ready_snapshots=[],
            )))
    assert to_json(controller.state) == expected_state

    loaded = asyncio.run(controller._persistence.load())
    assert to_json(loaded) == expected_state


def test_ssd_corruption_from_other_worker_does_not_demote_current_home(
        tmp_path):
    controller = _make_controller(tmp_path)
    _seed_ready_local_copy(controller, worker='worker-current')
    controller.state.local_copies['snap-1:worker-old'] = (
        SnapshotLocalCopyRecord(
            snapshot_id='snap-1',
            pool_worker_uid='worker-old',
            local_state=SnapshotLocalCopyState.WARM,
        ))
    controller.worker_inventory['worker-old'] = WorkerInventory(
        pool_worker_uid='worker-old', state=_snapshot())
    event = WorkerEvent(
        type=WorkerEventType.SSD_CACHE_CORRUPTED,
        snapshot_id='snap-1',
    )

    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-old', event=event,
                    ready_snapshots=[])))

    assert (controller.state.local_copies['snap-1:worker-old'].local_state ==
            SnapshotLocalCopyState.EVICTED)
    assert (controller.state.local_copies['snap-1:worker-current'].local_state
            == SnapshotLocalCopyState.WARM)
    assert controller.state.snapshots['snap-1'].worker == 'worker-current'
    assert controller.state.snapshots['snap-1'].state == SnapshotState.READY


@pytest.mark.parametrize(('storage_mode', 'worker_status'), [
    (PoolStorageMode.ACTIVITY_ON_PVC, None),
    (PoolStorageMode.ACTIVITY_ON_EMPTYDIR, None),
    (PoolStorageMode.DAEMON_ON_PVC, WorkerStatus.DRAINING),
])
def test_ssd_corruption_from_non_authoritative_reporter_is_ignored(
        tmp_path, storage_mode, worker_status):
    controller = _make_controller(tmp_path)
    _seed_ready_local_copy(controller)
    if worker_status is not None:
        controller.worker_inventory['worker-a'] = WorkerInventory(
            pool_worker_uid='worker-a', state=_snapshot(), status=worker_status)
    expected_state = to_json(controller.state)

    asyncio.run(
        controller.report_status.coroutine(
            _report(
                worker_storage_mode=storage_mode,
                event=WorkerEvent(type=WorkerEventType.SSD_CACHE_CORRUPTED,
                                  snapshot_id='snap-1'),
                ready_snapshots=[],
            )))

    assert to_json(controller.state) == expected_state


def test_snapshot_size_bytes_falls_back_to_backup_for_cold(tmp_path):
    """COLD snapshot: the placement (and its size) is GC'd on eviction, so the
    size falls back to the L2 backup; a live placement size takes precedence."""
    controller = _make_controller(tmp_path)
    controller.state.dataplane_backups['snap-1'] = DataplaneBackup(
        snapshot_id='snap-1', dataplane_path='s3://bucket/snap-1',
        size_bytes=4096)
    assert controller._snapshot_size_bytes('snap-1') == 4096

    # A legacy zero-size backup is still treated as absent (fail-fast preserved).
    controller.state.dataplane_backups['snap-2'] = DataplaneBackup(
        snapshot_id='snap-2', dataplane_path='s3://bucket/snap-2', size_bytes=0)
    assert controller._snapshot_size_bytes('snap-2') is None

    # A live placement size takes precedence over the backup.
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a', size_bytes=8192)
    assert controller._snapshot_size_bytes('snap-1') == 8192


def test_snapshot_ready_backfills_existing_backup_size(tmp_path):
    """A backup's size is backfilled on the next SNAPSHOT_READY (e.g. a COLD
    snapshot being restored), so a legacy zero-size record self-heals."""
    controller = _make_controller(tmp_path)
    placement_state = controller.state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    placement_state.pvcs['worker-a'] = PvcState(pool_worker_uid='worker-a')
    controller.state.dataplane_backups['snap-1'] = DataplaneBackup(
        snapshot_id='snap-1', dataplane_path='s3://bucket/snap-1', size_bytes=0)

    asyncio.run(
        controller.report_status.coroutine(
            _report(
                worker_storage_mode=PoolStorageMode.ACTIVITY_ON_PVC,
                event=_snapshot_ready_event('snap-1', now_utc(),
                                            size_bytes=2048),
                ready_snapshots=[])))

    assert controller.state.dataplane_backups['snap-1'].size_bytes == 2048


def test_ensure_backup_size_backfills_legacy_cold_backup(tmp_path, monkeypatch):
    """A legacy COLD backup (size 0) gets its size scanned from the dataplane
    on demand and written back."""
    from io_utils import fsutil
    controller = _make_controller(tmp_path)
    controller.state.dataplane_backups['snap-1'] = DataplaneBackup(
        snapshot_id='snap-1', dataplane_path='s3://bucket/snap-1', size_bytes=0)

    scanned: list[str] = []

    def _fake_dirsize(path: str) -> int:
        scanned.append(path)
        return 7777

    monkeypatch.setattr(fsutil, 'dirsize', _fake_dirsize)

    asyncio.run(controller._ensure_backup_size('snap-1'))

    assert scanned == ['s3://bucket/snap-1']  # scanned the dataplane dir once
    assert controller.state.dataplane_backups['snap-1'].size_bytes == 7777
    assert controller._snapshot_size_bytes('snap-1') == 7777  # now resolvable


def test_ensure_backup_size_noop_when_known_or_no_backup(tmp_path, monkeypatch):
    """No scan when the size is already cached, or when there's no backup to
    scan (a MISSING snapshot) — keeps it off the hot path for the common case."""
    from io_utils import fsutil
    controller = _make_controller(tmp_path)
    scanned: list[str] = []
    monkeypatch.setattr(fsutil, 'dirsize', lambda p:
                        (scanned.append(p), 999)[1])

    # No backup recorded → nothing to scan.
    asyncio.run(controller._ensure_backup_size('missing-snap'))
    # Backup already carries a size → no scan.
    controller.state.dataplane_backups['snap-1'] = DataplaneBackup(
        snapshot_id='snap-1', dataplane_path='s3://bucket/snap-1',
        size_bytes=4096)
    asyncio.run(controller._ensure_backup_size('snap-1'))

    assert scanned == []


def test_report_status_dataplane_ready_resets_compliance_tombstone(tmp_path):
    controller = _make_controller(tmp_path)
    deleted_at = now_utc() - timedelta(hours=1)
    controller.state.snapshot_compliance['snap-1'] = (SnapshotComplianceRecord(
        snapshot_id='snap-1',
        expired_at=deleted_at,
        deleted_at=deleted_at,
    ))
    event = WorkerEvent(
        type=WorkerEventType.SNAPSHOT_PERSIST_UPLOADED,
        snapshot_id='snap-1',
        dataplane_path='s3://bucket/env/snapshots/snap-1',
    )

    asyncio.run(
        controller.report_status.coroutine(
            _report(event=event, ready_snapshots=[])))

    compliance = controller.state.snapshot_compliance['snap-1']
    assert compliance.expired_at is None
    assert compliance.deleted_at is None


def test_report_status_dataplane_ready_without_path_is_ignored(tmp_path):
    controller = _make_controller(tmp_path)
    event = WorkerEvent(type=WorkerEventType.SNAPSHOT_PERSIST_UPLOADED,
                        snapshot_id='snap-1')

    asyncio.run(
        controller.report_status.coroutine(
            _report(event=event, ready_snapshots=[])))

    assert controller.state.dataplane_backups == {}


def test_report_status_no_space_event_releases_lease_and_requests_reactive_sweep(
        tmp_path):
    deleted: list[tuple[str, str]] = []

    async def delete_local(snapshot_id: str, pool_worker_uid: str) -> bool:
        deleted.append((snapshot_id, pool_worker_uid))
        return True

    strategy = MockPlacementStrategy()
    controller = _make_controller(
        tmp_path,
        strategy=strategy,
        local_snapshot_deleter=delete_local,
    )
    controller.state.dataplane_backups['snap-1'] = DataplaneBackup(
        snapshot_id='snap-1',
        dataplane_path='s3://bucket/snap-1',
    )
    _open_snapshot(controller)
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.EVICTION_CANDIDATE,
    ))
    event = WorkerEvent(
        type=WorkerEventType.SNAPSHOT_ENSURE_NO_SPACE,
        snapshot_id='snap-1',
        job=ActiveJob(
            job_id='job-1',
            job_type='ensure_snapshot',
            snapshot_id='snap-1',
        ),
    )

    asyncio.run(
        controller.report_status.coroutine(
            _report(
                in_progress_snapshots=['snap-1'],
                ready_snapshots=[],
                event=event,
            )))

    placement = controller.state.snapshots['snap-1']
    assert placement.leases == []
    assert placement.last_lease_released is not None
    assert strategy.release_calls == [
        ProvisionedHandle(
            kind='durable',
            task_queue='diskgraph-activity-tq-job-1',
            pool_worker_uid='worker-a',
        )
    ]
    assert deleted == []
    assert controller.state.local_copies['snap-1:worker-a'].local_state == (
        SnapshotLocalCopyState.EVICTION_CANDIDATE)
    assert controller._reactive_eviction_requested_at is not None


def test_report_status_no_space_from_emptydir_releases_without_reactive_eviction(
        tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    placement = SnapshotPlacement(
        worker='worker-a',
        state=SnapshotState.READY,
        leases=['job-overflow'],
        lease_paths={'job-overflow': 'overflow'},
        overflow_handles={
            'job-overflow':
            ProvisionedHandle(
                kind='overflow',
                task_queue='diskgraph-eph-activity-tq-job-overflow',
                pool_worker_uid='pool-eh-worker-deadbeef',
            ),
        },
    )
    controller.state.snapshots['snap-1'] = placement
    controller._sync_lifecycle_from_leases('snap-1')
    event = WorkerEvent(
        type=WorkerEventType.SNAPSHOT_ENSURE_NO_SPACE,
        snapshot_id='snap-1',
        job=ActiveJob(
            job_id='job-overflow',
            job_type='ensure_snapshot',
            snapshot_id='snap-1',
        ),
    )

    asyncio.run(
        controller.report_status.coroutine(
            _report(
                pool_worker_uid='pool-eh-worker-deadbeef',
                worker_storage_mode=PoolStorageMode.ACTIVITY_ON_EMPTYDIR,
                event=event,
                ready_snapshots=[],
            )))

    placement = controller.state.snapshots['snap-1']
    assert placement.leases == []
    assert placement.last_lease_released is not None
    assert strategy.release_calls == [
        ProvisionedHandle(
            kind='overflow',
            task_queue='diskgraph-eph-activity-tq-job-overflow',
            pool_worker_uid='pool-eh-worker-deadbeef',
        )
    ]
    assert controller._reactive_eviction_requested_at is None


def test_report_status_no_space_from_stale_pvc_worker_is_ignored(tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    _open_snapshot(controller, worker='worker-current')
    event = WorkerEvent(
        type=WorkerEventType.SNAPSHOT_ENSURE_NO_SPACE,
        snapshot_id='snap-1',
        job=ActiveJob(
            job_id='job-1',
            job_type='ensure_snapshot',
            snapshot_id='snap-1',
        ),
    )

    asyncio.run(
        controller.report_status.coroutine(
            _report(
                pool_worker_uid='worker-old',
                worker_storage_mode=PoolStorageMode.ACTIVITY_ON_PVC,
                event=event,
                ready_snapshots=[],
            )))

    placement = controller.state.snapshots['snap-1']
    assert placement.worker == 'worker-current'
    assert placement.leases == ['job-1']
    assert strategy.release_calls == []
    assert controller._reactive_eviction_requested_at is None


def test_report_status_generic_ensure_failure_releases_lease_only(tmp_path):
    deleted: list[tuple[str, str]] = []

    async def delete_local(snapshot_id: str, pool_worker_uid: str) -> bool:
        deleted.append((snapshot_id, pool_worker_uid))
        return True

    strategy = MockPlacementStrategy()
    controller = _make_controller(
        tmp_path,
        strategy=strategy,
        local_snapshot_deleter=delete_local,
    )
    _open_snapshot(controller, state=SnapshotState.RESERVED)
    event = WorkerEvent(
        type=WorkerEventType.SNAPSHOT_ENSURE_FAILED,
        snapshot_id='snap-1',
        job=ActiveJob(
            job_id='job-1',
            job_type='ensure_snapshot',
            snapshot_id='snap-1',
        ),
    )

    asyncio.run(
        controller.report_status.coroutine(
            _report(
                in_progress_snapshots=['snap-1'],
                ready_snapshots=[],
                event=event,
            )))

    assert 'snap-1' not in controller.state.snapshots
    assert strategy.release_calls == [
        ProvisionedHandle(
            kind='durable',
            task_queue='diskgraph-activity-tq-job-1',
            pool_worker_uid='worker-a',
        )
    ]
    assert deleted == []
    assert controller._reactive_eviction_requested_at is None


def test_place_snapshot_requires_strategy(tmp_path):
    controller = _make_controller(tmp_path)
    with pytest.raises(NotImplementedError, match='PlacementStrategy'):
        asyncio.run(controller.place_snapshot.coroutine(_request()))


def test_snapshot_eviction_in_progress_error_is_retryable():
    assert is_retryable(SnapshotEvictionInProgressError('evicting')) is True


@pytest.mark.parametrize('reuse_existing_handle', [False, True])
def test_place_snapshot_retries_while_current_copy_is_evicting(
        tmp_path, reuse_existing_handle):
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    controller.state.dataplane_backups['snap-1'] = DataplaneBackup(
        snapshot_id='snap-1', dataplane_path='s3://bucket/snap-1')
    if reuse_existing_handle:
        _open_snapshot(controller)
        request = _request()
    else:
        controller.state.snapshots['snap-1'] = SnapshotPlacement(
            worker='worker-a', state=SnapshotState.READY)
        request = _request(job_id='job-2')
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.EVICTING,
    ))
    expected_leases = ['job-1'] if reuse_existing_handle else []

    with pytest.raises(SnapshotEvictionInProgressError):
        asyncio.run(controller.place_snapshot.coroutine(request))

    assert controller.state.snapshots['snap-1'].leases == expected_leases
    assert strategy.route_calls == []
    assert strategy.provision_calls == []


def test_place_snapshot_provisions_once_and_persists(tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    result = asyncio.run(controller.place_snapshot.coroutine(_request()))

    assert result.task_queue == 'diskgraph-activity-tq-job-1'
    assert result.status == PlacementStatus.MISSING
    assert len(strategy.route_calls) == 1
    assert len(strategy.provision_calls) == 1
    placement = controller.state.snapshots['snap-1']
    assert placement.leases == ['job-1']
    assert placement.durable_handles['job-1'].task_queue == (
        'diskgraph-activity-tq-job-1')
    compliance = controller.state.snapshot_compliance['snap-1']
    assert compliance.active_consumer_count == 1
    assert compliance.compliance_delete_at is None

    loaded = asyncio.run(controller._persistence.load())
    assert loaded.snapshots == controller.state.snapshots
    assert loaded.snapshot_compliance == controller.state.snapshot_compliance


def test_place_snapshot_and_release_lease_canonicalize_graph_snapshot_id(
        tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)

    asyncio.run(
        controller.place_snapshot.coroutine(
            _request(snapshot_id=ALT_GRAPH_SNAPSHOT_ID)))
    asyncio.run(
        controller.release_lease.coroutine(ALT_GRAPH_SNAPSHOT_ID, 'job-1'))

    assert strategy.route_calls[0][0].snapshot_id == CANONICAL_GRAPH_SNAPSHOT_ID
    placement = controller.state.snapshots[CANONICAL_GRAPH_SNAPSHOT_ID]
    assert placement.leases == []
    assert placement.last_lease_released is not None


def test_place_snapshot_provisions_outside_state_lock(tmp_path):
    observed_state_lock: list[bool] = []
    controller: PoolController

    class ObservingProvisionStrategy(MockPlacementStrategy):
        async def provision(self, request: PlacementRequest,
                            decision: RouteDecision) -> ProvisionedHandle:
            observed_state_lock.append(controller._state_lock.locked())
            return await super().provision(request, decision)

    strategy = ObservingProvisionStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)

    result = asyncio.run(controller.place_snapshot.coroutine(_request()))

    assert result.task_queue == 'diskgraph-activity-tq-job-1'
    assert observed_state_lock == [False]


def test_place_snapshot_retry_same_pair_is_idempotent(tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    first = asyncio.run(controller.place_snapshot.coroutine(_request()))
    second = asyncio.run(controller.place_snapshot.coroutine(_request()))

    assert second.task_queue == first.task_queue
    assert len(strategy.route_calls) == 1
    assert len(strategy.provision_calls) == 1
    assert controller.state.snapshots['snap-1'].leases == ['job-1']


def test_place_snapshot_result_carries_pool_worker_uid(tmp_path):
    """PlacementResult exposes the placed durable worker uid so the workflow
    runner can poll its health after a failure."""
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    result = asyncio.run(controller.place_snapshot.coroutine(_request()))
    assert result.pool_worker_uid == 'worker-a'


def test_place_snapshot_fast_path_short_circuits_for_ready_worker(tmp_path):
    """A ready worker is revalidated without routing or provisioning again."""
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    asyncio.run(controller.place_snapshot.coroutine(_request()))
    # Worker reports a fresh heartbeat → inventory READY.
    asyncio.run(
        controller.report_status.coroutine(_report(pool_worker_uid='worker-a')))

    asyncio.run(controller.place_snapshot.coroutine(_request()))

    # Fast path hit: no second route()/provision().
    assert len(strategy.route_calls) == 1
    assert len(strategy.provision_calls) == 1
    assert len(strategy.wait_provision_ready_calls) == 2


def test_place_snapshot_ready_revalidation_failure_cleans_up(tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    asyncio.run(controller.place_snapshot.coroutine(_request()))
    asyncio.run(
        controller.report_status.coroutine(_report(pool_worker_uid='worker-a')))
    strategy.wait_provision_ready_error = ProvisionTimeoutError('Job missing')

    with pytest.raises(ProvisionTimeoutError, match='Job missing'):
        asyncio.run(controller.place_snapshot.coroutine(_request()))

    placement = controller.state.snapshots['snap-1']
    assert placement.leases == []
    assert 'job-1' not in placement.durable_handles
    assert len(strategy.release_calls) == 1


def test_place_snapshot_fast_path_falls_through_for_dead_worker(tmp_path):
    """Replacing a dead worker releases its Job before provisioning."""
    events = []

    class ReplacementStrategy(MockPlacementStrategy):
        async def route(self, request: PlacementRequest,
                        warm_location_uid: Optional[str],
                        snapshot_size_estimate_bytes: int = 0, *,
                        home_worker_uid: Optional[str] = None,
                        home_gpu_in_use: int = 0,
                        overflow_viable: bool = False) -> RouteDecision:
            decision = await super().route(
                request,
                warm_location_uid,
                snapshot_size_estimate_bytes,
                home_worker_uid=home_worker_uid,
                home_gpu_in_use=home_gpu_in_use,
                overflow_viable=overflow_viable,
            )
            if len(self.route_calls) == 2:
                decision = replace(decision, pool_worker_uid='worker-b')
            return decision

        async def provision(self, request: PlacementRequest,
                            decision: RouteDecision) -> ProvisionedHandle:
            events.append(('provision', decision.pool_worker_uid))
            return await super().provision(request, decision)

        async def release(self, request: PlacementRequest,
                          handle: Optional[ProvisionedHandle]) -> None:
            await super().release(request, handle)
            events.append(
                ('release', handle.pool_worker_uid if handle else None))

    strategy = ReplacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    first = asyncio.run(controller.place_snapshot.coroutine(_request()))
    asyncio.run(
        controller.report_status.coroutine(_report(pool_worker_uid='worker-a')))
    # Age the worker to DEAD.
    controller.worker_inventory['worker-a'].last_heartbeat = (
        time.time() - WORKER_DEAD_THRESHOLD_SECS)
    controller._sweep_worker_health()
    assert controller.worker_inventory['worker-a'].status == WorkerStatus.DEAD

    second = asyncio.run(controller.place_snapshot.coroutine(_request()))

    assert first.pool_worker_uid == 'worker-a'
    assert second.pool_worker_uid == 'worker-b'
    assert len(strategy.route_calls) == 2
    assert events == [('provision', 'worker-a'), ('release', 'worker-a'),
                      ('provision', 'worker-b')]


def test_place_snapshot_retry_after_cancelled_provision_reuses_worker(tmp_path):
    class CancelOnceProvisionStrategy(MockPlacementStrategy):
        async def route(self, request: PlacementRequest,
                        warm_location_uid: Optional[str],
                        snapshot_size_estimate_bytes: int = 0, *,
                        home_worker_uid: Optional[str] = None,
                        home_gpu_in_use: int = 0,
                        overflow_viable: bool = False) -> RouteDecision:
            self.route_calls.append((request, warm_location_uid))
            self.route_home_calls.append(
                (home_worker_uid, home_gpu_in_use, overflow_viable))
            return RouteDecision(
                path='durable',
                pool_worker_uid='worker-planned',
                provisioned_new_worker=True,
                reason='test cold durable placement',
            )

        async def provision(self, request: PlacementRequest,
                            decision: RouteDecision) -> ProvisionedHandle:
            self.provision_calls.append((request, decision))
            if len(self.provision_calls) == 1:
                raise asyncio.CancelledError()
            return ProvisionedHandle(
                kind=decision.path,
                pool_worker_uid=decision.pool_worker_uid,
                task_queue=f'diskgraph-activity-tq-{request.job_id}',
            )

    strategy = CancelOnceProvisionStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(controller.place_snapshot.coroutine(_request()))

    placement = controller.state.snapshots['snap-1']
    assert placement.worker == 'worker-planned'
    assert placement.leases == ['job-1']
    assert placement.durable_handles == {}

    loaded = asyncio.run(controller._persistence.load())
    assert loaded.snapshots['snap-1'].worker == 'worker-planned'
    assert loaded.snapshots['snap-1'].leases == ['job-1']

    result = asyncio.run(controller.place_snapshot.coroutine(_request()))

    # Retry does NOT re-call route(): the lease's path was pinned in
    # ``lease_paths`` on attempt 1, and route() is not stable across
    # attempts (inventory.gpu_slots_in_use shifts as sibling leases
    # release). Re-routing could flip the path (e.g. overflow → durable)
    # and orphan the in-flight K8s resources from attempt 1. The pinned
    # decision is reconstructed and re-provisioned, with provision()
    # adopting in-flight resources via deterministic naming.
    assert result.task_queue == 'diskgraph-activity-tq-job-1'
    assert len(strategy.route_calls) == 1
    assert len(strategy.provision_calls) == 2
    assert strategy.provision_calls[1][1].pool_worker_uid == 'worker-planned'
    assert strategy.provision_calls[1][1].reason == 'pinned-on-retry'
    placement = controller.state.snapshots['snap-1']
    assert placement.leases == ['job-1']
    assert placement.durable_handles['job-1'].pool_worker_uid == (
        'worker-planned')


def test_concurrent_place_snapshot_retry_same_pair_is_idempotent(tmp_path):
    class BlockingProvisionStrategy(MockPlacementStrategy):
        def __init__(self) -> None:
            super().__init__()
            self.provision_started = asyncio.Event()
            self.allow_provision = asyncio.Event()

        async def provision(self, request: PlacementRequest,
                            decision: RouteDecision) -> ProvisionedHandle:
            self.provision_calls.append((request, decision))
            self.provision_started.set()
            await self.allow_provision.wait()
            return ProvisionedHandle(
                kind=decision.path,
                pool_worker_uid=decision.pool_worker_uid,
                task_queue=f'diskgraph-activity-tq-{request.job_id}',
            )

    async def drive() -> tuple[PoolController, BlockingProvisionStrategy,
                               list[PlacementResult]]:
        strategy = BlockingProvisionStrategy()
        controller = _make_controller(tmp_path, strategy=strategy)

        first = asyncio.create_task(
            controller.place_snapshot.coroutine(_request()))
        await strategy.provision_started.wait()

        second = asyncio.create_task(
            controller.place_snapshot.coroutine(_request()))
        await asyncio.sleep(0)

        assert len(strategy.route_calls) == 1
        assert len(strategy.provision_calls) == 1

        strategy.allow_provision.set()
        results = await asyncio.gather(first, second)
        return controller, strategy, list(results)

    controller, strategy, results = asyncio.run(drive())

    assert [result.task_queue for result in results] == [
        'diskgraph-activity-tq-job-1',
        'diskgraph-activity-tq-job-1',
    ]
    assert len(strategy.route_calls) == 1
    assert len(strategy.provision_calls) == 1
    placement = controller.state.snapshots['snap-1']
    assert placement.leases == ['job-1']
    assert placement.durable_handles['job-1'] is not None


def test_concurrent_place_snapshot_distinct_jobs_provision_separately(tmp_path):
    """Sibling fan-out: N concurrent ``place_snapshot`` calls with the
    same snapshot_id but distinct job_ids each provision their own handle.

    Models sharded batch prediction (``BatchPredictionJobWorkflow``'s
    ``asyncio.gather`` over N ``_run_shard`` coroutines): each shard
    passes a distinct ``shard_index`` into the placement coordinator so
    the controller sees distinct job_ids and routes each as its own
    placement.

    Regression guard for the placement fast-path: if all shards
    collapsed under the same job_id (the old default), only 1
    ``strategy.provision`` call would fire and the other N-1 shards
    would silently reuse one pod that processes a single activity
    before terminating.
    """
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)

    async def drive() -> list[PlacementResult]:
        return await asyncio.gather(*[
            controller.place_snapshot.coroutine(_request(job_id=f'job-{i}'))
            for i in range(5)
        ])

    results = asyncio.run(drive())

    assert {r.task_queue
            for r in results
            } == {f'diskgraph-activity-tq-job-{i}'
                  for i in range(5)}
    assert len(strategy.provision_calls) == 5
    assert set(controller.state.snapshots['snap-1'].leases) == {
        f'job-{i}'
        for i in range(5)
    }
    assert controller.state.snapshot_compliance[
        'snap-1'].active_consumer_count == 5


def test_concurrent_place_snapshot_cold_fanout_overflows_siblings(tmp_path):
    """Backlog #17: N concurrent ``place_snapshot`` for the same COLD
    snapshot (distinct shard job_ids; an L2 backup present) must place
    **1 durable home + (N-1) overflow eph**, NOT N durable workers (the
    cold-fanout herd).

    Models sharded batch prediction on an evicted-but-materialized
    snapshot. The per-snapshot lock serializes PHASE 1, so the first
    sibling records the placement (home worker); the controller then passes
    that ``home_worker_uid`` + ``overflow_viable=True`` to ``route()`` for
    every later sibling, which overflows instead of minting a competing
    durable worker.

    Pre-fix this provisioned 5 durable workers (5 full-PVC cache nodes for
    one BP job); the guard is ``durable == 1``.
    """
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    # COLD: materialized (L2 backup recorded) but not warm on any worker.
    controller.state.dataplane_backups['snap-1'] = DataplaneBackup(
        snapshot_id='snap-1',
        dataplane_path='s3://bucket/snap-1',
    )

    async def drive() -> list[PlacementResult]:
        return await asyncio.gather(*[
            controller.place_snapshot.coroutine(
                _request(job_id=f'job-{i}', activity=SnapshotActivity.PREDICT))
            for i in range(5)
        ])

    asyncio.run(drive())

    paths = [decision.path for _, decision in strategy.provision_calls]
    assert paths.count('durable') == 1, paths
    assert paths.count('overflow') == 4, paths
    # Exactly one sibling saw no home (the first to take the per-snapshot
    # lock); every later sibling saw the recorded home, the home's GPU slot
    # already taken by shard-0's durable PREDICT lease (home_gpu_in_use >= 1),
    # and a viable backup → overflow.
    homes = strategy.route_home_calls
    assert sum(1 for home, _, _ in homes if home is None) == 1
    assert sum(1 for home, _, _ in homes if home is not None) == 4
    assert all(viable for _, _, viable in homes)
    assert all(gpu_in_use >= 1 for home, gpu_in_use, _ in homes
               if home is not None)


def test_concurrent_place_snapshot_producer_coalesces_to_one_worker(tmp_path):
    """Backlog #17: N concurrent ``ENSURE_SNAPSHOT`` (producer) for the same
    not-yet-existing snapshot coalesce onto a SINGLE durable worker — they do
    not herd into N durable workers, and they do not overflow (a producer just
    needs X to exist, so it rides the one ingest). Non-GPU ⇒ the home always
    serves it.

    Pre-fix the no-backup fall-through minted a fresh worker per sibling.
    """
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)

    # No L2 backup: X is being produced for the first time.
    async def drive() -> list[PlacementResult]:
        return await asyncio.gather(*[
            controller.place_snapshot.coroutine(
                _request(job_id=f'ingest-{i}',
                         activity=SnapshotActivity.ENSURE_SNAPSHOT))
            for i in range(5)
        ])

    asyncio.run(drive())

    paths = [decision.path for _, decision in strategy.provision_calls]
    assert paths.count('durable') == 5, paths  # all durable, none overflow
    assert paths.count('overflow') == 0, paths
    durable_workers = {
        decision.pool_worker_uid
        for _, decision in strategy.provision_calls
    }
    assert len(durable_workers) == 1, durable_workers  # single home, no herd


def test_place_snapshot_records_lease_activity(tmp_path):
    """``_lease_activity`` map is populated by ``place_snapshot`` so the
    controller can derive ``gpu_slots_in_use`` from leases without waiting
    for a daemon heartbeat to refresh it."""
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)

    asyncio.run(
        controller.place_snapshot.coroutine(
            _request(activity=SnapshotActivity.TRAIN)))

    assert controller._lease_activity[('snap-1',
                                       'job-1')] == (SnapshotActivity.TRAIN)


def test_release_lease_clears_lease_activity_entry(tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    asyncio.run(controller.place_snapshot.coroutine(_request()))

    asyncio.run(controller.release_lease.coroutine('snap-1', 'job-1'))

    assert ('snap-1', 'job-1') not in controller._lease_activity


def test_gpu_slots_in_use_for_worker_counts_durable_train_predict_only(
        tmp_path):
    """Only TRAIN/PREDICT durable leases count toward GPU pressure;
    GEN_TT/MATERIALIZE are CPU-only and overflow leases run on eph nodes."""
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)

    asyncio.run(
        controller.place_snapshot.coroutine(
            _request(snapshot_id='snap-1', job_id='train-1',
                     activity=SnapshotActivity.TRAIN)))
    asyncio.run(
        controller.place_snapshot.coroutine(
            _request(snapshot_id='snap-1', job_id='predict-1',
                     activity=SnapshotActivity.PREDICT)))
    asyncio.run(
        controller.place_snapshot.coroutine(
            _request(snapshot_id='snap-1', job_id='gentt-1',
                     activity=SnapshotActivity.GEN_TT)))

    assert controller._gpu_slots_in_use_for_worker('worker-a') == 2
    # Worker that holds no leases reports zero.
    assert controller._gpu_slots_in_use_for_worker('worker-b') == 0


def test_gpu_slots_in_use_for_worker_excludes_overflow_leases(tmp_path):
    """Overflow leases (in ``overflow_handles``) run on ephemeral nodes —
    they don't consume the durable worker's GPU."""
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)

    asyncio.run(
        controller.place_snapshot.coroutine(
            _request(snapshot_id='snap-1', job_id='durable-train',
                     activity=SnapshotActivity.TRAIN)))
    asyncio.run(
        controller.place_snapshot.coroutine(
            _request(snapshot_id='snap-1', job_id='overflow-train',
                     activity=SnapshotActivity.TRAIN)))

    placement = controller.state.snapshots['snap-1']
    # Simulate the overflow path: re-route the second lease to overflow and
    # move its handle into overflow_handles.
    placement.lease_paths['overflow-train'] = 'overflow'
    placement.durable_handles.pop('overflow-train', None)
    placement.overflow_handles['overflow-train'] = ProvisionedHandle(
        kind='overflow', pool_worker_uid=None,
        task_queue='diskgraph-activity-tq-overflow-train')

    assert controller._gpu_slots_in_use_for_worker('worker-a') == 1


def test_placement_context_synthesizes_gpu_slots_in_use_from_leases(tmp_path):
    """``PoolControllerPlacementContext.worker_inventory`` must override
    inventory's ``gpu_slots_in_use`` with a freshly counted lease total so
    concurrent shards see fresh data rather than a heartbeat-lagged stale
    zero."""
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)

    # Seed inventory with a worker that *thinks* it has 0 GPU in use.
    controller._worker_inventory['worker-a'] = WorkerInventory(
        pool_worker_uid='worker-a',
        status=WorkerStatus.READY,
        last_heartbeat=1.0,
        gpu_slots_total=1,
        gpu_slots_in_use=0,
        state=WorkerStateSnapshot(
            ssd_total_bytes=2_000,
            ssd_used_bytes=0,
        ),
    )

    # Now place a TRAIN lease against worker-a.
    asyncio.run(
        controller.place_snapshot.coroutine(
            _request(snapshot_id='snap-1', job_id='train-1',
                     activity=SnapshotActivity.TRAIN)))

    context = PoolControllerPlacementContext(controller)
    synthesized = context.worker_inventory
    assert synthesized['worker-a'].gpu_slots_in_use == 1
    # The base inventory dict on the controller is untouched.
    assert controller._worker_inventory['worker-a'].gpu_slots_in_use == 0


def test_place_snapshot_different_jobs_share_snapshot_capacity_reservation(
        tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(
        tmp_path,
        strategy=strategy,
        data_plane_storage_root='/local/dataplane',
    )
    asyncio.run(
        controller.report_status.coroutine(
            _report(
                ready_snapshots=[],
                dataplane_total_bytes=10_000,
                dataplane_used_bytes=9_600,
                dataplane_same_filesystem_as_ssd=False,
            )))
    # The first 400-byte estimate exactly fills the reported free budget; the
    # second job must get its own lease without charging the snapshot again.
    request = replace(
        _request(),
        source_size_bytes=100,
    )

    job_1 = asyncio.run(controller.place_snapshot.coroutine(request))
    job_2 = asyncio.run(
        controller.place_snapshot.coroutine(replace(request, job_id='job-2')))

    assert job_1.task_queue == 'diskgraph-activity-tq-job-1'
    assert job_2.task_queue == 'diskgraph-activity-tq-job-2'
    assert controller.state.snapshots['snap-1'].size_bytes == 400
    assert set(controller.state.snapshots['snap-1'].leases) == {
        'job-1',
        'job-2',
    }
    assert controller.state.snapshot_compliance[
        'snap-1'].active_consumer_count == 2
    assert len(strategy.provision_calls) == 2


def test_place_snapshot_resets_compliance_tombstone(tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    deleted_at = now_utc() - timedelta(hours=1)
    controller.state.snapshot_compliance['snap-1'] = (SnapshotComplianceRecord(
        snapshot_id='snap-1',
        expired_at=deleted_at,
        deleted_at=deleted_at,
    ))

    asyncio.run(controller.place_snapshot.coroutine(_request()))

    compliance = controller.state.snapshot_compliance['snap-1']
    assert compliance.active_consumer_count == 1
    assert compliance.compliance_delete_at is None
    assert compliance.expired_at is None
    assert compliance.deleted_at is None


def test_place_snapshot_passes_warm_location_to_strategy(tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-warm', state=SnapshotState.READY)

    asyncio.run(controller.place_snapshot.coroutine(_request()))

    assert strategy.route_calls[0][1] == 'worker-warm'
    assert controller.state.snapshots['snap-1'].worker == 'worker-warm'


def test_place_snapshot_non_ready_home_reused_not_warm_hit(tmp_path):
    """A RESERVED (non-ready) placement is NOT treated as a warm cache hit
    (``warm_location_uid`` stays None), but its home worker IS reused rather
    than minting a second durable worker — the single-durable-host invariant
    (backlog #17). Pre-fix the controller ignored the reserved home and routed
    to a fresh ``worker-a``, forking the snapshot onto two durable workers."""
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-reserved',
        state=SnapshotState.RESERVED,
    )

    asyncio.run(controller.place_snapshot.coroutine(_request()))

    assert strategy.route_calls[0][1] is None  # not a warm hit
    # Reuses the in-flight reserved home (GPU slot free), not a fresh worker.
    assert controller.state.snapshots['snap-1'].worker == 'worker-reserved'


def test_place_snapshot_l2_check_blocks_when_backup_missing(tmp_path):
    strategy = MockPlacementStrategy(requires_l2_backup_check=True)
    controller = _make_controller(tmp_path, strategy=strategy)

    with pytest.raises(AssertionError, match='no dataplane backup'):
        asyncio.run(controller.place_snapshot.coroutine(_request()))

    assert len(strategy.route_calls) == 1
    assert strategy.provision_calls == []
    assert controller.state.snapshots == {}


def test_place_snapshot_l2_check_allows_valid_backup(tmp_path):
    strategy = MockPlacementStrategy(requires_l2_backup_check=True)
    controller = _make_controller(tmp_path, strategy=strategy)
    controller.state.dataplane_backups['snap-1'] = DataplaneBackup(
        snapshot_id='snap-1',
        dataplane_path='s3://bucket/snap-1',
    )
    # Overflow path only fires when a warm placement already exists.
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-warm', )

    result = asyncio.run(controller.place_snapshot.coroutine(_request()))

    assert result.task_queue == 'diskgraph-activity-tq-job-1'
    assert result.status == PlacementStatus.COLD
    assert len(strategy.route_calls) == 1
    assert len(strategy.provision_calls) == 1
    assert controller.state.snapshots['snap-1'].leases == ['job-1']


def test_release_lease_marks_lease_and_placement_released(tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    asyncio.run(controller.place_snapshot.coroutine(_request()))

    asyncio.run(controller.release_lease.coroutine('snap-1', 'job-1'))
    asyncio.run(controller.release_lease.coroutine('snap-1', 'job-1'))

    placement = controller.state.snapshots['snap-1']
    compliance = controller.state.snapshot_compliance['snap-1']
    assert placement.leases == []
    assert placement.last_lease_released is not None
    assert compliance.active_consumer_count == 0
    assert compliance.last_released_at == placement.last_lease_released
    assert compliance.compliance_delete_at is not None
    assert len(strategy.release_calls) == 1


def test_release_one_of_multiple_leases_keeps_compliance_clock_paused(tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    asyncio.run(controller.place_snapshot.coroutine(_request(job_id='job-1')))
    asyncio.run(controller.place_snapshot.coroutine(_request(job_id='job-2')))

    asyncio.run(controller.release_lease.coroutine('snap-1', 'job-1'))

    compliance = controller.state.snapshot_compliance['snap-1']
    assert compliance.active_consumer_count == 1
    assert compliance.compliance_delete_at is None

    asyncio.run(controller.release_lease.coroutine('snap-1', 'job-2'))

    compliance = controller.state.snapshot_compliance['snap-1']
    assert compliance.active_consumer_count == 0
    assert compliance.compliance_delete_at is not None


def test_snapshot_lifecycle_timeline_evicts_after_final_release_idle(tmp_path):
    deleted: list[tuple[str, str]] = []

    async def delete_local(snapshot_id: str, pool_worker_uid: str) -> bool:
        deleted.append((snapshot_id, pool_worker_uid))
        return True

    clock = _FakeClock()
    controller = _make_controller(
        tmp_path,
        strategy=MockPlacementStrategy(),
        local_snapshot_deleter=delete_local,
        now_utc_fn=clock.now_utc,
        time_fn=clock.time,
    )

    asyncio.run(
        controller.place_snapshot.coroutine(
            _request(snapshot_id='snap-X', job_id='job-1')))

    assert controller.state.snapshots['snap-X'].leases == ['job-1']
    compliance = controller.state.snapshot_compliance['snap-X']
    assert compliance.active_consumer_count == 1
    assert compliance.compliance_delete_at is None

    clock.advance(1)
    asyncio.run(
        controller.report_status.coroutine(
            _report(
                ready_snapshots=[],
                event=_snapshot_ready_event('snap-X', clock.now_utc()),
            )))
    asyncio.run(
        controller.report_status.coroutine(
            _report(
                ready_snapshots=[],
                event=_snapshot_dataplane_ready_event('snap-X',
                                                      clock.now_utc()),
            )))

    local = controller.state.local_copies['snap-X:worker-a']
    # kumo#29047: bump the live policy AFTER the records were created with the
    # default policy. Enforcement must use these new values, not the frozen
    # per-record snapshot.
    controller._lifecycle_policy = SnapshotLifecyclePolicy(
        min_ssd_idle_secs=4 * 60 * 60,
        max_ssd_idle_secs=8 * 60 * 60,
        max_dataplane_retention_secs=10 * 60 * 60,
    )
    assert local.local_state == SnapshotLocalCopyState.WARM
    assert local.evictable_under_pressure_at is None
    assert local.force_evict_at is None

    clock.advance(5 * 60)
    asyncio.run(controller.release_lease.coroutine('snap-X', 'job-1'))

    released_at = clock.current
    placement = controller.state.snapshots['snap-X']
    compliance = controller.state.snapshot_compliance['snap-X']
    local = controller.state.local_copies['snap-X:worker-a']
    assert placement.last_lease_released == released_at
    assert compliance.active_consumer_count == 0
    assert compliance.compliance_delete_at == (released_at +
                                               timedelta(seconds=10 * 60 * 60))
    assert local.evictable_under_pressure_at == (released_at +
                                                 timedelta(seconds=4 * 60 * 60))
    assert local.force_evict_at == (released_at +
                                    timedelta(seconds=8 * 60 * 60))

    clock.advance(9 * 60 * 60)
    asyncio.run(controller._lifecycle_sweeps_once())

    assert deleted == [('snap-X', 'worker-a')]
    assert 'snap-X' not in controller.state.snapshots
    local = controller.state.local_copies['snap-X:worker-a']
    assert local.local_state == SnapshotLocalCopyState.EVICTED
    assert local.evicted_at == clock.current


def test_lifecycle_policy_stamps_new_snapshot_records(tmp_path):
    clock = _FakeClock()
    controller = _make_controller(
        tmp_path,
        strategy=MockPlacementStrategy(),
        lifecycle_policy=SnapshotLifecyclePolicy(
            min_ssd_idle_secs=30,
            max_ssd_idle_secs=90,
            max_dataplane_retention_secs=120,
        ),
        now_utc_fn=clock.now_utc,
        time_fn=clock.time,
    )

    asyncio.run(
        controller.place_snapshot.coroutine(
            _request(snapshot_id='snap-policy', job_id='job-policy')))
    asyncio.run(
        controller.report_status.coroutine(
            _report(
                ready_snapshots=[],
                event=_snapshot_ready_event(
                    'snap-policy',
                    clock.now_utc(),
                ),
            )))
    asyncio.run(controller.release_lease.coroutine('snap-policy', 'job-policy'))

    released_at = clock.current
    compliance = controller.state.snapshot_compliance['snap-policy']
    local = controller.state.local_copies['snap-policy:worker-a']
    assert compliance.max_dataplane_retention_secs == 120
    assert compliance.compliance_delete_at == (released_at +
                                               timedelta(seconds=120))
    assert local.min_ssd_idle_secs == 30
    assert local.max_ssd_idle_secs == 90
    assert local.evictable_under_pressure_at == (released_at +
                                                 timedelta(seconds=30))
    assert local.force_evict_at == (released_at + timedelta(seconds=90))


def test_lifecycle_policy_bump_applies_to_existing_records(tmp_path):
    """kumo#29047: tuning retention/idle live must re-apply to records that
    predate the change. The per-record values frozen at creation are
    audit-only; enforcement reads the live policy."""
    clock = _FakeClock()
    controller = _make_controller(
        tmp_path,
        strategy=MockPlacementStrategy(),
        lifecycle_policy=SnapshotLifecyclePolicy(
            min_ssd_idle_secs=30,
            max_ssd_idle_secs=60,
            max_dataplane_retention_secs=120,
        ),
        now_utc_fn=clock.now_utc,
        time_fn=clock.time,
    )

    asyncio.run(
        controller.place_snapshot.coroutine(
            _request(snapshot_id='snap-bump', job_id='job-bump')))
    asyncio.run(
        controller.report_status.coroutine(
            _report(
                ready_snapshots=[],
                event=_snapshot_ready_event('snap-bump', clock.now_utc()),
            )))

    # Records were stamped with the aggressive creation-time policy.
    compliance = controller.state.snapshot_compliance['snap-bump']
    local = controller.state.local_copies['snap-bump:worker-a']
    assert compliance.max_dataplane_retention_secs == 120
    assert local.min_ssd_idle_secs == 30
    assert local.max_ssd_idle_secs == 60

    # Operator bumps the live policy (e.g. configMap rollover + controller
    # restart) to the relaxed dg0 defaults.
    controller._lifecycle_policy = SnapshotLifecyclePolicy(
        min_ssd_idle_secs=600,
        max_ssd_idle_secs=3600,
        max_dataplane_retention_secs=259200,
    )

    clock.advance(60)
    asyncio.run(controller.release_lease.coroutine('snap-bump', 'job-bump'))

    released_at = clock.current
    compliance = controller.state.snapshot_compliance['snap-bump']
    local = controller.state.local_copies['snap-bump:worker-a']
    # Enforcement clocks reflect the NEW policy, not the stamped values.
    assert compliance.compliance_delete_at == (released_at +
                                               timedelta(seconds=259200))
    assert local.evictable_under_pressure_at == (released_at +
                                                 timedelta(seconds=600))
    assert local.force_evict_at == (released_at + timedelta(seconds=3600))
    # The stamped per-record values are untouched (audit-only).
    assert compliance.max_dataplane_retention_secs == 120
    assert local.min_ssd_idle_secs == 30
    assert local.max_ssd_idle_secs == 60


def test_overlapping_train_and_gen_tt_start_idle_clock_after_last_release(
        tmp_path):
    deleted: list[tuple[str, str]] = []

    async def delete_local(snapshot_id: str, pool_worker_uid: str) -> bool:
        deleted.append((snapshot_id, pool_worker_uid))
        return True

    clock = _FakeClock()
    controller = _make_controller(
        tmp_path,
        strategy=MockPlacementStrategy(),
        local_snapshot_deleter=delete_local,
        lifecycle_policy=SnapshotLifecyclePolicy(
            min_ssd_idle_secs=60,
            max_ssd_idle_secs=120,
            max_dataplane_retention_secs=300,
        ),
        now_utc_fn=clock.now_utc,
        time_fn=clock.time,
    )

    async def place_both() -> None:
        await asyncio.gather(
            controller.place_snapshot.coroutine(
                _request(snapshot_id='snap-X', job_id='job-1')),
            controller.place_snapshot.coroutine(
                _request(
                    snapshot_id='snap-X',
                    job_id='job-2',
                    activity=SnapshotActivity.GEN_TT,
                )),
        )

    asyncio.run(place_both())
    assert controller.state.snapshot_compliance[
        'snap-X'].active_consumer_count == 2

    clock.advance(1)
    asyncio.run(
        controller.report_status.coroutine(
            _report(
                ready_snapshots=[],
                event=_snapshot_ready_event('snap-X', clock.now_utc()),
            )))
    asyncio.run(
        controller.report_status.coroutine(
            _report(
                ready_snapshots=[],
                event=_snapshot_dataplane_ready_event('snap-X',
                                                      clock.now_utc()),
            )))

    asyncio.run(controller.release_lease.coroutine('snap-X', 'job-1'))
    compliance = controller.state.snapshot_compliance['snap-X']
    local = controller.state.local_copies['snap-X:worker-a']
    assert compliance.active_consumer_count == 1
    assert compliance.compliance_delete_at is None
    assert local.evictable_under_pressure_at is None
    assert local.force_evict_at is None

    clock.advance(10 * 60)
    asyncio.run(controller._lifecycle_sweeps_once())
    assert deleted == []
    assert controller.state.local_copies[
        'snap-X:worker-a'].local_state == SnapshotLocalCopyState.WARM

    asyncio.run(controller.release_lease.coroutine('snap-X', 'job-2'))
    final_release_at = clock.current
    compliance = controller.state.snapshot_compliance['snap-X']
    local = controller.state.local_copies['snap-X:worker-a']
    assert compliance.active_consumer_count == 0
    assert compliance.compliance_delete_at == (final_release_at +
                                               timedelta(seconds=300))
    assert local.evictable_under_pressure_at == (final_release_at +
                                                 timedelta(seconds=60))
    assert local.force_evict_at == (final_release_at + timedelta(seconds=120))

    clock.advance(119)
    asyncio.run(controller._lifecycle_sweeps_once())
    assert deleted == []
    assert controller.state.local_copies['snap-X:worker-a'].local_state == (
        SnapshotLocalCopyState.EVICTION_CANDIDATE)

    clock.advance(1)
    asyncio.run(controller._lifecycle_sweeps_once())
    assert deleted == [('snap-X', 'worker-a')]
    assert controller.state.local_copies[
        'snap-X:worker-a'].local_state == SnapshotLocalCopyState.EVICTED


def test_full_flow_warm_gpu_busy_routes_second_gpu_job_to_overflow(tmp_path):
    manager = _FakeK8sManager()
    controller = _make_controller(tmp_path)
    controller._set_placement_strategy(
        build_k8s_placement_strategy(
            manager,  # type: ignore[arg-type]
            PoolControllerPlacementContext(controller),
            capacity_tiers=[CapacityTier(tier_ssd_gib=1750)],
        ))

    # Controller derives durable GPU usage authoritatively from leases, not
    # daemon-reported heartbeat usage.
    asyncio.run(
        controller.report_status.coroutine(
            _report(
                ready_snapshots=['snap-X'],
                event=_snapshot_ready_event('snap-X', now_utc(), 1000),
            )))
    asyncio.run(
        controller.report_status.coroutine(
            _report(
                ready_snapshots=['snap-X'],
                event=_snapshot_dataplane_ready_event('snap-X', now_utc()),
            )))

    job_1_result = asyncio.run(
        controller.place_snapshot.coroutine(
            _request(snapshot_id='snap-X', job_id='job-1')))

    assert job_1_result.task_queue == activity_task_queue('job-1')
    assert manager.provision_pool_worker_calls[0]['pool_worker_uid'] == (
        'worker-a')
    assert manager.provision_activity_pod_calls[0]['pool_worker_uid'] == (
        'worker-a')
    assert manager.provision_activity_pod_calls[0]['activity'] == 'TRAIN'
    assert manager.provision_activity_pod_calls[0]['env'][
        'POOL_TASK_QUEUE'] == activity_task_queue('job-1')
    assert manager.provision_ephemeral_activity_pod_calls == []

    asyncio.run(
        controller.report_status.coroutine(_report(ready_snapshots=['snap-X'
                                                                    ], )))

    job_2_result = asyncio.run(
        controller.place_snapshot.coroutine(
            _request(
                snapshot_id='snap-X',
                job_id='job-2',
                activity=SnapshotActivity.PREDICT,
            )))

    assert job_2_result.task_queue == activity_task_queue('job-2')
    assert len(manager.provision_pool_worker_calls) == 1
    assert len(manager.provision_activity_pod_calls) == 1
    assert len(manager.provision_ephemeral_activity_pod_calls) == 1
    overflow_call = manager.provision_ephemeral_activity_pod_calls[0]
    assert overflow_call['job_id'] == 'job-2'
    assert overflow_call['snapshot_id'] == 'snap-X'
    assert overflow_call['activity'] == 'PREDICT'
    assert overflow_call['ephemeral_storage_bytes'] == 1200
    assert overflow_call['env']['POOL_TASK_QUEUE'] == activity_task_queue(
        'job-2')

    placement = controller.state.snapshots['snap-X']
    assert placement.worker == 'worker-a'
    assert 'job-2' in placement.overflow_handles
    assert controller.state.snapshot_compliance[
        'snap-X'].active_consumer_count == 2


@pytest.mark.parametrize('readiness_error',
                         [ProvisionTimeoutError, asyncio.CancelledError])
def test_unready_activity_poller_releases_lease_and_job_before_retry(
        tmp_path, monkeypatch, readiness_error):
    manager = _FakeK8sManager()
    if readiness_error is ProvisionTimeoutError:
        monkeypatch.setattr(
            'pool.k8s_strategy.ACTIVITY_POLLER_READY_TIMEOUT_SEC', 0.01)

    async def reject_poller(task_queue: str, pod_name: str,
                            timeout: float) -> None:
        del task_queue, pod_name
        if readiness_error is ProvisionTimeoutError:
            await asyncio.sleep(timeout)
        raise readiness_error('poller did not register')

    controller = _make_controller(tmp_path)
    controller._set_placement_strategy(
        build_k8s_placement_strategy(
            manager,  # type: ignore[arg-type]
            PoolControllerPlacementContext(controller),
            capacity_tiers=[CapacityTier(tier_ssd_gib=1750)],
            pool_worker_uid_factory=lambda: 'worker-new',
            activity_poller_waiter=reject_poller,
        ))

    with pytest.raises(readiness_error):
        asyncio.run(
            controller.place_snapshot.coroutine(
                _request(snapshot_id='snap-new', job_id='job-new')))

    placement = controller.state.snapshots['snap-new']
    assert 'job-new' not in placement.leases
    assert 'job-new' not in placement.durable_handles
    assert manager.delete_activity_pods_calls == [
        {
            'job_id': 'job-new',
            'wait_for_deletion': True,
        },
    ]


def test_controller_persists_and_uses_configured_gpu_capacity(tmp_path):
    manager = _FakeK8sManager()
    controller = _make_controller(tmp_path)
    controller._set_placement_strategy(
        build_k8s_placement_strategy(
            manager,  # type: ignore[arg-type]
            PoolControllerPlacementContext(controller),
            capacity_tiers=[CapacityTier(tier_ssd_gib=2948, tier_gpu_count=4)],
            pool_worker_uid_factory=lambda: 'worker-a',
        ))

    asyncio.run(
        controller.place_snapshot.coroutine(
            _request(
                snapshot_id='snap-gpu',
                job_id='job-1',
                activity=SnapshotActivity.TRAIN,
                min_gpu_slots=4,
            )))

    placement_state = controller.state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    pvc_state = placement_state.pvcs['worker-a']
    assert pvc_state.tier_ssd_gib == 2948
    assert pvc_state.tier_gpu_count == 4
    assert manager.provision_pool_worker_calls[0]['pvc_spec'] == PvcSpec(
        size_gib=2948,
        gpu_count=4,
    )

    asyncio.run(
        controller.report_status.coroutine(_report(
            pool_worker_uid='worker-a', )))

    inventory = controller.worker_inventory['worker-a']
    assert inventory.gpu_slots_total == 4
    synthesized = PoolControllerPlacementContext(controller).worker_inventory
    assert synthesized['worker-a'].gpu_slots_in_use == 1


def test_controller_repairs_legacy_pvc_capacity_metadata(tmp_path):
    manager = _FakeK8sManager()
    controller = _make_controller(tmp_path)
    placement_state = controller.state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    placement_state.pvcs['worker-a'] = PvcState(pool_worker_uid='worker-a')
    controller._set_placement_strategy(
        build_k8s_placement_strategy(
            manager,  # type: ignore[arg-type]
            PoolControllerPlacementContext(controller),
            capacity_tiers=[CapacityTier(tier_ssd_gib=2948, tier_gpu_count=4)],
            pool_worker_uid_factory=lambda: 'worker-a',
        ))

    asyncio.run(
        controller.place_snapshot.coroutine(
            _request(
                snapshot_id='snap-gpu',
                job_id='job-1',
                activity=SnapshotActivity.PREDICT,
                min_gpu_slots=4,
            )))

    pvc_state = placement_state.pvcs['worker-a']
    assert pvc_state.tier_ssd_gib == 2948
    assert pvc_state.tier_gpu_count == 4


def test_controller_uses_persisted_gpu_capacity_before_heartbeat(tmp_path):
    manager = _FakeK8sManager()
    controller = _make_controller(tmp_path)
    controller._set_placement_strategy(
        build_k8s_placement_strategy(
            manager,  # type: ignore[arg-type]
            PoolControllerPlacementContext(controller),
            capacity_tiers=[CapacityTier(tier_ssd_gib=2948, tier_gpu_count=4)],
            pool_worker_uid_factory=lambda: 'worker-a',
        ))
    controller.state.dataplane_backups['snap-gpu'] = DataplaneBackup(
        snapshot_id='snap-gpu',
        dataplane_path='s3://bucket/snap-gpu',
        size_bytes=1000,
    )

    async def place(job_id: str) -> PlacementResult:
        return await controller.place_snapshot.coroutine(
            _request(
                snapshot_id='snap-gpu',
                job_id=job_id,
                activity=SnapshotActivity.PREDICT,
                min_gpu_slots=4,
            ))

    results = [asyncio.run(place(f'job-{idx}')) for idx in range(5)]

    assert [result.pool_worker_uid for result in results] == [
        'worker-a',
        'worker-a',
        'worker-a',
        'worker-a',
        None,
    ]
    placement = controller.state.snapshots['snap-gpu']
    assert placement.lease_paths == {
        'job-0': 'durable',
        'job-1': 'durable',
        'job-2': 'durable',
        'job-3': 'durable',
        'job-4': 'overflow',
    }
    assert len(manager.provision_activity_pod_calls) == 4
    assert len(manager.provision_ephemeral_activity_pod_calls) == 1
    assert 'worker-a' not in controller.worker_inventory

    stats = asyncio.run(
        controller.get_pool_stats.coroutine(
            PoolStatsFilter(pool_worker_uids=['worker-a'])))
    assert stats.num_nodes == 1
    assert stats.num_healthy_nodes == 0
    assert len(stats.durable_workers) == 1
    worker_stats = stats.durable_workers[0]
    assert worker_stats.pool_worker_uid == 'worker-a'
    assert worker_stats.health == WorkerStatus.STALE
    assert worker_stats.pvc_total_bytes == 2948 * (1 << 30)
    assert worker_stats.pvc_free_bytes == 0
    assert worker_stats.gpu_slots_total == 4
    assert worker_stats.gpu_slots_in_use == 4

    asyncio.run(controller.release_lease.coroutine('snap-gpu', 'job-0'))
    released_slot_result = asyncio.run(place('job-5'))

    assert released_slot_result.pool_worker_uid == 'worker-a'
    placement = controller.state.snapshots['snap-gpu']
    assert placement.lease_paths['job-5'] == 'durable'
    assert 'job-0' not in placement.leases
    assert controller._gpu_slots_in_use_for_worker('worker-a') == 4


def test_fresh_worker_heartbeat_observes_persisted_gpu_capacity(tmp_path):
    manager = _FakeK8sManager()
    controller = _make_controller(tmp_path)
    controller._set_placement_strategy(
        build_k8s_placement_strategy(
            manager,  # type: ignore[arg-type]
            PoolControllerPlacementContext(controller),
            capacity_tiers=[CapacityTier(tier_ssd_gib=2948, tier_gpu_count=4)],
            pool_worker_uid_factory=lambda: 'worker-a',
        ))
    controller.state.dataplane_backups['snap-gpu'] = DataplaneBackup(
        snapshot_id='snap-gpu',
        dataplane_path='s3://bucket/snap-gpu',
        size_bytes=1000,
    )

    async def heartbeat_during_provision(
            pool_worker_uid: Optional[str]) -> None:
        assert pool_worker_uid == 'worker-a'
        placement_state = controller.state.placement_state
        assert isinstance(placement_state, K8sPlacementState)
        assert placement_state.pvcs['worker-a'].tier_gpu_count == 4
        await controller.report_status.coroutine(
            _report(
                pool_worker_uid='worker-a',
                ready_snapshots=[],
                pvc_total_bytes=2948 * (1 << 30),
                pvc_free_bytes=2948 * (1 << 30),
            ))

    manager.on_provision_pool_worker = heartbeat_during_provision

    result = asyncio.run(
        controller.place_snapshot.coroutine(
            _request(
                snapshot_id='snap-gpu',
                job_id='job-0',
                activity=SnapshotActivity.PREDICT,
                min_gpu_slots=4,
            )))

    assert result.pool_worker_uid == 'worker-a'
    assert controller.worker_inventory['worker-a'].gpu_slots_total == 4


def test_legacy_durable_lease_without_activity_counts_conservatively(tmp_path):
    controller = _make_controller(tmp_path)
    controller.state.snapshots['snap-legacy'] = SnapshotPlacement(
        worker='worker-a',
        leases=['job-legacy'],
        lease_paths={'job-legacy': 'durable'},
    )

    assert controller._gpu_slots_in_use_for_worker('worker-a') == 1


@pytest.mark.parametrize('handleless', [False, True])
def test_release_failure_preserves_lease_for_retry(tmp_path, handleless):
    strategy = MockPlacementStrategy(fail_release=True)
    controller = _make_controller(tmp_path, strategy=strategy)
    if handleless:
        controller.state.snapshots['snap-1'] = SnapshotPlacement(
            worker='worker-a', leases=['job-1'],
            lease_paths={'job-1': 'durable'})
        asyncio.run(controller._persistence.save(controller.state))
    else:
        asyncio.run(controller.place_snapshot.coroutine(_request()))

    with pytest.raises(RuntimeError, match='release failed'):
        asyncio.run(controller.release_lease.coroutine('snap-1', 'job-1'))

    placement = controller.state.snapshots['snap-1']
    assert placement.leases == ['job-1']
    assert ('job-1' in placement.durable_handles) is not handleless
    assert len(strategy.release_calls) == 1
    assert (strategy.release_calls[0] is None) is handleless

    loaded = asyncio.run(controller._persistence.load())
    assert loaded.snapshots['snap-1'].leases == ['job-1']

    strategy.fail_release = False
    asyncio.run(controller.release_lease.coroutine('snap-1', 'job-1'))
    assert controller.state.snapshots['snap-1'].leases == []


def test_release_serializes_same_placement_reprovision(tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    asyncio.run(controller.place_snapshot.coroutine(_request()))
    release_started = asyncio.Event()
    finish_release = asyncio.Event()

    async def blocked_release(request, handle):
        release_started.set()
        await finish_release.wait()

    strategy.release = blocked_release

    async def exercise():
        release = asyncio.create_task(
            controller.release_lease.coroutine('snap-1', 'job-1'))
        await release_started.wait()
        replace = asyncio.create_task(
            controller.place_snapshot.coroutine(_request()))
        await asyncio.sleep(0)
        assert not replace.done()
        finish_release.set()
        await asyncio.gather(release, replace)

    asyncio.run(exercise())
    assert len(strategy.provision_calls) == 2


def test_release_missing_lease_is_noop(tmp_path):
    controller = _make_controller(tmp_path, strategy=MockPlacementStrategy())
    asyncio.run(controller.release_lease.coroutine('snap-1', 'job-1'))


def test_release_lease_without_placement_marks_lease_released(tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)

    asyncio.run(controller.release_lease.coroutine('snap-1', 'job-1'))

    assert controller.state.snapshots == {}
    assert strategy.release_calls == []


def test_place_snapshot_provision_failure_preserves_lease(tmp_path):
    # Provision failures must NOT release the lease: workflow may
    # retry place_snapshot and a fresh lease would orphan the
    # half-provisioned PVC/pod. Dangling-lease sweep cleans up if the
    # workflow gives up entirely.
    strategy = MockPlacementStrategy(fail_provision=True)
    controller = _make_controller(tmp_path, strategy=strategy)

    with pytest.raises(RuntimeError, match='no capacity'):
        asyncio.run(controller.place_snapshot.coroutine(_request()))

    assert controller.state.snapshots['snap-1'].leases == ['job-1']


def test_place_snapshot_replans_confirmed_unavailable_worker(tmp_path):
    class UnavailableOnceStrategy(MockPlacementStrategy):
        async def provision(self, request: PlacementRequest,
                            decision: RouteDecision) -> ProvisionedHandle:
            self.provision_calls.append((request, decision))
            if decision.pool_worker_uid == 'worker-stale':
                raise ExistingWorkerUnavailableError('resources disappeared')
            return ProvisionedHandle(
                kind=decision.path,
                pool_worker_uid=decision.pool_worker_uid,
                task_queue=f'diskgraph-activity-tq-{request.job_id}',
            )

    strategy = UnavailableOnceStrategy()
    strategy.delete_workers_returns = {'worker-stale'}
    state = _state_with_expected_worker('worker-stale')
    state.snapshots['snap-1'].durable_handles.clear()
    controller = _make_controller_with_state(tmp_path, state, strategy)

    with pytest.raises(RuntimeError, match='retired.*retrying'):
        asyncio.run(controller.place_snapshot.coroutine(_request()))

    assert strategy.delete_workers_calls == [['worker-stale']]
    assert 'snap-1' not in controller.state.snapshots
    placement_state = state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    assert 'worker-stale' not in placement_state.pvcs
    assert controller.worker_inventory['worker-stale'].status == (
        WorkerStatus.DRAINING)
    loaded = asyncio.run(controller._persistence.load())
    assert 'snap-1' not in loaded.snapshots

    result = asyncio.run(controller.place_snapshot.coroutine(_request()))

    assert result.pool_worker_uid == 'worker-a'
    assert strategy.provision_calls[1][1].reason == 'test'


def test_place_snapshot_replan_requires_no_live_pod_or_fresh_heartbeat(tmp_path):
    controller: PoolController

    class HeartbeatDuringRevalidationStrategy(MockPlacementStrategy):
        revalidation_count = 0

        async def provision(self, request: PlacementRequest,
                            decision: RouteDecision) -> ProvisionedHandle:
            raise ExistingWorkerUnavailableError('resources disappeared')

        async def revalidate_retired_workers(self, uids) -> set[str]:
            self.revalidation_count += 1
            if self.revalidation_count == 2:
                uid = next(iter(uids))
                await controller.report_status.coroutine(
                    _report(pool_worker_uid=uid, ready_snapshots=[]))
            return await super().revalidate_retired_workers(uids)

    strategy = HeartbeatDuringRevalidationStrategy()
    state = _state_with_expected_worker('worker-stale')
    state.snapshots['snap-1'].durable_handles.clear()
    controller = _make_controller_with_state(tmp_path, state, strategy)
    controller.worker_inventory['worker-stale'] = WorkerInventory(
        pool_worker_uid='worker-stale',
        state=WorkerStateSnapshot(ssd_total_bytes=0, ssd_used_bytes=0),
        status=WorkerStatus.READY,
        last_heartbeat=1.0,
    )

    strategy.revalidated_retired_workers = set()
    with pytest.raises(ExistingWorkerUnavailableError):
        asyncio.run(controller.place_snapshot.coroutine(_request()))
    strategy.revalidated_retired_workers = None
    with pytest.raises(ExistingWorkerUnavailableError):
        asyncio.run(controller.place_snapshot.coroutine(_request()))

    assert strategy.revalidation_count == 2
    assert state.snapshots['snap-1'].leases == ['job-1']
    placement_state = state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    assert 'worker-stale' in placement_state.pvcs
    assert strategy.delete_workers_calls == []


def test_place_snapshot_records_daemon_queue_without_dispatch(tmp_path):
    strategy = MockPlacementStrategy(enqueue_daemon=True)
    controller = _make_controller(tmp_path, strategy=strategy)

    result = asyncio.run(controller.place_snapshot.coroutine(_request()))

    assert result.task_queue == 'diskgraph-activity-tq-job-1'
    placement = controller.state.snapshots['snap-1']
    assert placement.durable_handles['job-1'] is not None
    assert len(strategy.release_calls) == 0


def test_place_snapshot_reuses_existing_provisioned_daemon_queue_placement(
        tmp_path):
    strategy = MockPlacementStrategy(enqueue_daemon=True)
    controller = _make_controller(tmp_path, strategy=strategy)
    _open_snapshot(controller)

    result = asyncio.run(controller.place_snapshot.coroutine(_request()))

    assert result.task_queue == 'diskgraph-activity-tq-job-1'
    assert strategy.route_calls == []
    assert strategy.provision_calls == []


def test_ready_event_records_local_copy_and_pauses_clock_while_leased(tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    asyncio.run(controller.place_snapshot.coroutine(_request()))

    ready = WorkerEvent(
        type=WorkerEventType.SNAPSHOT_READY,
        snapshot_id='snap-1',
        size_bytes=4096,
    )
    asyncio.run(controller.report_status.coroutine(_report(event=ready)))

    local = controller.state.local_copies['snap-1:worker-a']
    assert local.pool_worker_uid == 'worker-a'
    assert local.local_state == SnapshotLocalCopyState.WARM
    assert local.size_bytes == 4096
    assert local.evictable_under_pressure_at is None
    assert local.force_evict_at is None


def test_maintenance_once_returns_noop_without_strategy(tmp_path):
    controller = _make_controller(tmp_path)

    delta = asyncio.run(controller._maintenance_once())

    assert delta == ReconcileDelta()


def test_maintenance_once_calls_strategy_with_state_snapshot(tmp_path):
    # worker-a is reported unexpected (lease-less placement) → controller
    # flips it DRAINING under the lock before delete_workers tears it down.
    delta = ReconcileDelta(unexpected_uids={'worker-a'},
                           live_pool_worker_uids={'worker-c'})
    strategy = MockPlacementStrategy(reconcile_delta=delta)
    controller = _make_controller(tmp_path, strategy=strategy)
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        state=SnapshotState.READY,
    )
    asyncio.run(controller.report_status.coroutine(_report(ready_snapshots=[])))

    result = asyncio.run(controller._maintenance_once())

    assert result == delta
    assert len(strategy.reconcile_calls) == 1
    # reconcile() receives a deepcopy snapshot taken BEFORE the controller
    # scrubs the unexpected worker's lease-less placement, so it still sees
    # snap-1 and is a distinct object from the (now-scrubbed) live state.
    assert strategy.reconcile_calls[0] is not controller.state
    assert 'snap-1' in strategy.reconcile_calls[0].snapshots
    assert 'snap-1' not in controller.state.snapshots
    # delete_workers mock returns nothing torn down, so the DRAINING inventory
    # entry survives the prune and is observable here.
    assert controller.worker_inventory[
        'worker-a'].status == WorkerStatus.DRAINING


@pytest.mark.parametrize('delta', [
    ReconcileDelta(retired_uids={'worker-a'}),
    ReconcileDelta(incompatible_uids={'worker-a'}),
])
def test_maintenance_once_scrubs_worker_state_from_reconcile_delta(
        tmp_path, delta):
    strategy = MockPlacementStrategy(reconcile_delta=delta)
    controller = _make_controller(tmp_path, strategy=strategy)
    placement_state = controller.state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    placement_state.pvcs['worker-a'] = PvcState(pool_worker_uid='worker-a')
    asyncio.run(
        controller.report_status.coroutine(_report(ready_snapshots=['snap-1'])))
    asyncio.run(controller.place_snapshot.coroutine(_request()))
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.EVICTING,
    ))
    inventory = controller.worker_inventory['worker-a']
    inventory.last_heartbeat = (controller._time() - WORKER_DEAD_THRESHOLD_SECS)
    inventory.status = WorkerStatus.DEAD

    result = asyncio.run(controller._maintenance_once())

    assert result == delta
    assert controller.worker_inventory[
        'worker-a'].status == WorkerStatus.DRAINING
    assert 'worker-a' not in placement_state.pvcs
    assert 'snap-1' not in controller.state.snapshots
    assert controller.state.local_copies[
        'snap-1:worker-a'].local_state == SnapshotLocalCopyState.EVICTED
    assert controller.state.snapshot_compliance[
        'snap-1'].active_consumer_count == 0
    assert controller.state.snapshot_compliance[
        'snap-1'].compliance_delete_at is not None
    assert strategy.release_calls == []


@pytest.mark.parametrize('reason', ['capacity-tier', 'extra-tolerations'])
@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_maintenance_once_replaces_live_incompatible_worker(
        tmp_path, monkeypatch, reason, cleanup_fails):
    manager = _FakeK8sManager()
    manager.live_workers = [_FakePoolWorkerInfo(
        'worker-a', extra_tolerations_match=reason != 'extra-tolerations')]
    state = _state_with_expected_worker()
    placement_state = state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    placement_state.pvcs['worker-a'].tier_ssd_gib = (
        400 if reason == 'capacity-tier' else 512)
    state.dataplane_backups['snap-1'] = DataplaneBackup(
        snapshot_id='snap-1', dataplane_path='gs://example/snap-1',
        size_bytes=100 * (1 << 30))
    backups = dict(state.dataplane_backups)
    if cleanup_fails:
        monkeypatch.setattr(manager, 'delete_pool_worker',
                            AsyncMock(side_effect=RuntimeError('delete failed')))
    controller = _make_controller_with_state(tmp_path, state)
    controller._set_placement_strategy(
        build_k8s_placement_strategy(
            manager,  # type: ignore[arg-type]
            PoolControllerPlacementContext(controller),
            capacity_tiers=[CapacityTier(512), CapacityTier(1024)],
            pool_worker_uid_factory=lambda: 'worker-new',
        ))
    controller._worker_inventory['worker-a'] = WorkerInventory(
        pool_worker_uid='worker-a',
        status=WorkerStatus.READY,
        last_heartbeat=time.time(),
        state=_snapshot(),
    )

    asyncio.run(controller._reconcile_once(refresh_silent_workers=False))

    assert controller.worker_inventory[
        'worker-a'].status == WorkerStatus.DRAINING
    assert 'worker-a' not in placement_state.pvcs
    assert 'snap-1' not in controller.state.snapshots
    if not cleanup_fails:
        assert manager.delete_pool_worker_calls[0]['pool_worker_uid'] == 'worker-a'
    assert controller.state.dataplane_backups == backups
    assert asyncio.run(controller._persistence.load()) == controller.state

    # The old pod can keep heartbeating after failed Kubernetes cleanup.
    # Its report must not revive the drained worker or restore a stale home.
    asyncio.run(controller.report_status.coroutine(
        _report(ready_snapshots=['snap-1'])))
    assert controller.worker_inventory[
        'worker-a'].status == WorkerStatus.DRAINING
    assert 'snap-1' not in controller.state.snapshots

    asyncio.run(
        controller.place_snapshot.coroutine(
            replace(_request(), source_size_bytes=100 * (1 << 30))))

    assert manager.provision_pool_worker_calls[-1]['pvc_spec'].size_gib == 512
    assert placement_state.pvcs['worker-new'].tier_ssd_gib == 512


def test_reconcile_retires_expected_worker_that_never_heartbeats_after_restart(
        tmp_path):
    clock = [1000.0]
    manager = _FakeK8sManager()
    controller = _make_controller_with_state(tmp_path,
                                             _state_with_expected_worker(),
                                             time_fn=lambda: clock[0])
    controller._set_placement_strategy(
        build_k8s_placement_strategy(
            manager,  # type: ignore[arg-type]
            PoolControllerPlacementContext(controller),
            capacity_tiers=[CapacityTier(1750)],
        ))
    _arm_silent_worker_retirement(controller, clock[0])
    placement_state = controller.state.placement_state
    assert isinstance(placement_state, K8sPlacementState)

    first_delta = asyncio.run(controller._reconcile_once())

    assert first_delta.retired_uids == set()
    assert manager.delete_pool_worker_calls == []
    assert controller.worker_inventory['worker-a'].status == WorkerStatus.STALE
    assert placement_state.silent_worker_suspect_since['worker-a'] == 1000.0
    assert 'snap-1' in controller.state.snapshots
    clock[0] = 1030.0
    controller._sweep_worker_health()
    assert controller.worker_inventory['worker-a'].status == WorkerStatus.STALE

    clock[0] = 1000.0 + WORKER_DEAD_THRESHOLD_SECS
    second_delta = asyncio.run(controller._reconcile_once())

    assert second_delta.retired_uids == {'worker-a'}
    assert manager.delete_pool_worker_calls == [{
        'pool_worker_uid': 'worker-a',
        'grace_period_seconds': 30,
        'scale_down_timeout': 120,
        'pvc_delete_timeout': 60,
        'force_on_timeout': True,
    }]
    assert 'worker-a' not in placement_state.pvcs
    assert 'worker-a' not in controller.worker_inventory
    assert 'snap-1' not in controller.state.snapshots


def test_silent_worker_registered_in_k8s_is_not_destructively_retired(tmp_path):
    """A Temporal heartbeat outage must not delete a separately healthy pod
    and its data-bearing PVC."""
    clock = [1000.0]
    manager = _FakeK8sManager()
    manager.live_workers = [_FakePoolWorkerInfo('worker-a', 'Registered')]
    controller = _make_controller_with_state(
        tmp_path,
        _state_with_expected_worker(),
        time_fn=lambda: clock[0],
    )
    controller._set_placement_strategy(
        build_k8s_placement_strategy(
            manager,  # type: ignore[arg-type]
            PoolControllerPlacementContext(controller),
            capacity_tiers=[CapacityTier(1750)],
        ))
    _arm_silent_worker_retirement(controller, clock[0])
    placement_state = controller.state.placement_state
    assert isinstance(placement_state, K8sPlacementState)

    asyncio.run(controller._reconcile_once())
    clock[0] += WORKER_DEAD_THRESHOLD_SECS
    protected = asyncio.run(controller._reconcile_once())

    assert protected.observed_pool_worker_uids == {'worker-a'}
    assert protected.live_pool_worker_uids == {'worker-a'}
    assert protected.retired_uids == set()
    assert controller.worker_inventory['worker-a'].status == WorkerStatus.DEAD
    assert manager.delete_pool_worker_calls == []
    assert 'worker-a' in placement_state.pvcs
    assert 'snap-1' in controller.state.snapshots

    manager.live_workers = []
    retired = asyncio.run(controller._reconcile_once())

    assert retired.retired_uids == {'worker-a'}
    assert manager.delete_pool_worker_calls[0]['pool_worker_uid'] == 'worker-a'
    assert 'worker-a' not in placement_state.pvcs


def test_silent_worker_stats_use_persisted_pvc_capacity(tmp_path):
    state = _state_with_expected_worker()
    placement_state = state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    placement_state.pvcs['worker-a'].tier_ssd_gib = 1750
    controller = _make_controller_with_state(tmp_path, state)

    asyncio.run(controller._refresh_silent_expected_workers())
    stats = asyncio.run(controller.get_pool_stats.coroutine())

    assert controller.worker_inventory['worker-a'].status == WorkerStatus.STALE
    assert stats.total_pvc_bytes == 1750 * (1 << 30)
    assert stats.free_pvc_bytes == 0


def test_silent_worker_removed_when_no_longer_expected(tmp_path):
    state = _state_with_expected_worker()
    controller = _make_controller_with_state(tmp_path, state)
    placement_state = state.placement_state
    assert isinstance(placement_state, K8sPlacementState)

    asyncio.run(controller._refresh_silent_expected_workers())
    assert 'worker-a' in controller.worker_inventory
    assert 'worker-a' in placement_state.silent_worker_suspect_since

    state.snapshots.pop('snap-1')
    asyncio.run(controller._refresh_silent_expected_workers())
    controller._sweep_worker_health(controller._time() +
                                    WORKER_DEAD_THRESHOLD_SECS)
    stats = asyncio.run(controller.get_pool_stats.coroutine())

    assert 'worker-a' not in controller.worker_inventory
    assert 'worker-a' not in placement_state.silent_worker_suspect_since
    assert stats.num_nodes == 0


def test_restart_synthetic_stale_blocks_retry_until_first_heartbeat(tmp_path):
    state = _state_with_expected_worker()
    controller = _make_controller_with_state(tmp_path, state)
    placement = state.snapshots['snap-1']

    asyncio.run(controller._refresh_silent_expected_workers())

    assert controller.worker_inventory['worker-a'].status == WorkerStatus.STALE
    assert controller._pinned_decision_for_retry(placement, 'job-1') is None

    asyncio.run(
        controller.report_status.coroutine(_report(ready_snapshots=['snap-1'])))

    decision = controller._pinned_decision_for_retry(placement, 'job-1')
    assert decision is not None
    assert decision.pool_worker_uid == 'worker-a'


def test_reconcile_silent_worker_deadline_survives_controller_restart(tmp_path):
    clock = [1000.0]
    state = _state_with_expected_worker()
    manager_1 = _FakeK8sManager()
    controller_1 = _make_controller_with_state(tmp_path, state,
                                               time_fn=lambda: clock[0])
    controller_1._set_placement_strategy(
        build_k8s_placement_strategy(
            manager_1,  # type: ignore[arg-type]
            PoolControllerPlacementContext(controller_1),
            capacity_tiers=[CapacityTier(1750)],
        ))
    _arm_silent_worker_retirement(controller_1, clock[0])
    placement_state = state.placement_state
    assert isinstance(placement_state, K8sPlacementState)

    first_delta = asyncio.run(controller_1._reconcile_once())

    assert first_delta.retired_uids == set()
    assert placement_state.silent_worker_suspect_since['worker-a'] == 1000.0

    # Simulate controller process restart before another heartbeat arrives.
    # The second controller must preserve the original suspect timestamp
    # instead of restarting the DEAD countdown at the new process start time.
    clock[0] = 1000.0 + WORKER_DEAD_THRESHOLD_SECS
    manager_2 = _FakeK8sManager()
    controller_2 = _make_controller_with_state(tmp_path, state,
                                               time_fn=lambda: clock[0])
    controller_2._set_placement_strategy(
        build_k8s_placement_strategy(
            manager_2,  # type: ignore[arg-type]
            PoolControllerPlacementContext(controller_2),
            capacity_tiers=[CapacityTier(1750)],
        ))
    _arm_silent_worker_retirement(controller_2, clock[0])

    second_delta = asyncio.run(controller_2._reconcile_once())

    assert second_delta.retired_uids == {'worker-a'}
    assert manager_2.delete_pool_worker_calls[0]['pool_worker_uid'] == (
        'worker-a')
    assert 'worker-a' not in placement_state.silent_worker_suspect_since
    assert 'snap-1' not in controller_2.state.snapshots


def test_silent_worker_cannot_retire_before_serving_heartbeat_grace(tmp_path):
    async def drive() -> None:
        clock = [1000.0]
        state = _state_with_expected_worker()
        placement_state = state.placement_state
        assert isinstance(placement_state, K8sPlacementState)
        # Simulate a persisted deadline that expired while a slow init canary
        # was running and no AutoRPC worker existed to receive heartbeats.
        placement_state.silent_worker_suspect_since['worker-a'] = 0.0
        strategy = MockPlacementStrategy(reconcile_delta=ReconcileDelta(
            retired_uids={'worker-a'}))
        controller = _make_controller_with_state(
            tmp_path,
            state,
            strategy=strategy,
            time_fn=lambda: clock[0],
        )

        startup_delta = await controller._reconcile_once(
            refresh_silent_workers=False)
        assert startup_delta.retired_uids == set()
        assert controller.worker_inventory == {}
        assert strategy.delete_workers_calls == []

        controller._start_maintenance_loop(interval_secs=3600)
        while len(strategy.reconcile_calls) < 2:
            await asyncio.sleep(0)
        assert controller.worker_inventory[
            'worker-a'].status == WorkerStatus.STALE
        assert strategy.delete_workers_calls == []
        assert 'snap-1' in controller.state.snapshots

        maintenance_task = controller._maintenance_task
        assert maintenance_task is not None
        maintenance_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await maintenance_task

        clock[0] += POOL_HEARTBEAT_INTERVAL_SECS
        await controller.report_status.coroutine(_report())
        assert controller.worker_inventory[
            'worker-a'].status == WorkerStatus.READY
        assert 'worker-a' not in (placement_state.silent_worker_suspect_since)

    asyncio.run(drive())


def test_silent_worker_stays_stale_until_rpc_grace_is_started(tmp_path):
    clock = [1000.0 + WORKER_DEAD_THRESHOLD_SECS]
    state = _state_with_expected_worker()
    placement_state = state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    placement_state.silent_worker_suspect_since['worker-a'] = 0.0
    controller = _make_controller_with_state(
        tmp_path,
        state,
        time_fn=lambda: clock[0],
    )

    asyncio.run(controller._refresh_silent_expected_workers())

    assert controller._silent_worker_dead_not_before is None
    assert controller.worker_inventory['worker-a'].status == WorkerStatus.STALE


def test_reconcile_retires_worker_expected_only_by_local_copy(tmp_path):
    clock = [1000.0]
    manager = _FakeK8sManager()
    state = PoolState(placement_state=K8sPlacementState())
    placement_state = state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    placement_state.pvcs['worker-a'] = PvcState(pool_worker_uid='worker-a')
    state.local_copies['snap-1:worker-a'] = SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.WARM,
    )
    controller = _make_controller_with_state(tmp_path, state,
                                             time_fn=lambda: clock[0])
    controller._set_placement_strategy(
        build_k8s_placement_strategy(
            manager,  # type: ignore[arg-type]
            PoolControllerPlacementContext(controller),
            capacity_tiers=[CapacityTier(1750)],
        ))
    _arm_silent_worker_retirement(controller, clock[0])

    first_delta = asyncio.run(controller._reconcile_once())

    assert first_delta.retired_uids == set()
    assert controller.worker_inventory['worker-a'].status == WorkerStatus.STALE
    assert placement_state.silent_worker_suspect_since['worker-a'] == 1000.0

    clock[0] = 1000.0 + WORKER_DEAD_THRESHOLD_SECS
    second_delta = asyncio.run(controller._reconcile_once())

    assert second_delta.retired_uids == {'worker-a'}
    assert manager.delete_pool_worker_calls[0]['pool_worker_uid'] == 'worker-a'
    assert 'worker-a' not in placement_state.pvcs
    assert state.local_copies['snap-1:worker-a'].local_state == (
        SnapshotLocalCopyState.EVICTED)


def test_reconcile_retires_legacy_expected_worker_without_pvc_state(tmp_path):
    clock = [1000.0]
    manager = _FakeK8sManager()
    state = _state_with_expected_worker()
    placement_state = state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    placement_state.pvcs.clear()
    controller = _make_controller_with_state(
        tmp_path,
        state,
        time_fn=lambda: clock[0],
    )
    controller._set_placement_strategy(
        build_k8s_placement_strategy(
            manager,  # type: ignore[arg-type]
            PoolControllerPlacementContext(controller),
            capacity_tiers=[CapacityTier(1750)],
        ))
    _arm_silent_worker_retirement(controller, clock[0])

    asyncio.run(controller._reconcile_once())
    clock[0] = 1000.0 + WORKER_DEAD_THRESHOLD_SECS
    delta = asyncio.run(controller._reconcile_once())

    assert delta.retired_uids == {'worker-a'}
    assert manager.delete_pool_worker_calls[0]['pool_worker_uid'] == 'worker-a'
    assert 'snap-1' not in controller.state.snapshots


def test_report_status_clears_silent_worker_suspect(tmp_path):
    clock = [2000.0]
    manager = _FakeK8sManager()
    controller = _make_controller_with_state(tmp_path,
                                             _state_with_expected_worker(),
                                             time_fn=lambda: clock[0])
    controller._set_placement_strategy(
        build_k8s_placement_strategy(
            manager,  # type: ignore[arg-type]
            PoolControllerPlacementContext(controller),
            capacity_tiers=[CapacityTier(1750)],
        ))
    placement_state = controller.state.placement_state
    assert isinstance(placement_state, K8sPlacementState)

    asyncio.run(controller._reconcile_once())
    assert controller.worker_inventory['worker-a'].status == WorkerStatus.STALE
    assert placement_state.silent_worker_suspect_since['worker-a'] == 2000.0

    clock[0] = 2020.0
    asyncio.run(controller.report_status.coroutine(_report()))

    assert controller.worker_inventory['worker-a'].status == WorkerStatus.READY
    assert 'worker-a' not in placement_state.silent_worker_suspect_since

    clock[0] = 2000.0 + WORKER_DEAD_THRESHOLD_SECS
    delta = asyncio.run(controller._reconcile_once())

    assert delta.retired_uids == set()
    assert manager.delete_pool_worker_calls == []
    assert 'snap-1' in controller.state.snapshots


def test_first_heartbeat_after_silent_synthetic_inventory_cleans_missing_local(
        tmp_path):
    manager = _FakeK8sManager()
    state = PoolState(placement_state=K8sPlacementState())
    placement_state = state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    placement_state.pvcs['worker-a'] = PvcState(pool_worker_uid='worker-a')
    state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        state=SnapshotState.READY,
    )
    state.local_copies['snap-1:worker-a'] = SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.WARM,
    )
    controller = _make_controller_with_state(tmp_path, state)
    controller._set_placement_strategy(
        build_k8s_placement_strategy(
            manager,  # type: ignore[arg-type]
            PoolControllerPlacementContext(controller),
            capacity_tiers=[CapacityTier(1750)],
        ))

    asyncio.run(controller._reconcile_once())
    assert controller.worker_inventory['worker-a'].status == WorkerStatus.STALE

    asyncio.run(controller.report_status.coroutine(_report(ready_snapshots=[])))

    assert controller.worker_inventory['worker-a'].status == WorkerStatus.READY
    assert controller.state.local_copies[
        'snap-1:worker-a'].local_state == SnapshotLocalCopyState.EVICTED
    assert 'snap-1' not in controller.state.snapshots


def test_retired_worker_late_heartbeat_does_not_resurrect_snapshot(tmp_path):
    controller = _make_controller_with_state(tmp_path,
                                             _state_with_expected_worker())
    controller.worker_inventory['worker-a'] = WorkerInventory(
        pool_worker_uid='worker-a',
        state=_snapshot(ready_snapshots=['snap-1']),
        status=WorkerStatus.DEAD,
        last_heartbeat=0,
    )

    mutated = controller._apply_reconcile_worker_delta(
        ReconcileDelta(retired_uids={'worker-a'}))

    assert mutated
    assert controller.worker_inventory[
        'worker-a'].status == WorkerStatus.DRAINING
    assert 'snap-1' not in controller.state.snapshots

    asyncio.run(
        controller.report_status.coroutine(_report(ready_snapshots=['snap-1'])))

    assert controller.worker_inventory[
        'worker-a'].status == WorkerStatus.DRAINING
    assert 'snap-1' not in controller.state.snapshots


def test_reconcile_does_not_suspect_worker_created_after_process_start(
        tmp_path):
    manager = _FakeK8sManager()
    controller = _make_controller(tmp_path)
    controller._set_placement_strategy(
        build_k8s_placement_strategy(
            manager,  # type: ignore[arg-type]
            PoolControllerPlacementContext(controller),
            capacity_tiers=[CapacityTier(1750)],
        ))
    placement_state = controller.state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    placement_state.pvcs['worker-a'] = PvcState(pool_worker_uid='worker-a')
    _open_snapshot(controller, worker='worker-a')

    delta = asyncio.run(controller._reconcile_once())

    assert delta.retired_uids == set()
    assert 'worker-a' not in controller.worker_inventory
    assert manager.delete_pool_worker_calls == []
    assert 'snap-1' in controller.state.snapshots


def test_maintenance_once_marks_unexpected_worker_draining_before_delete(
        tmp_path):
    """Teardown route-race fix: an unexpected worker is flipped to DRAINING
    (non-routable) under state_lock BEFORE delete_workers tears it down, the
    lease-less placement that still points at it is scrubbed (so route()'s
    home path can't co-locate a new activity onto a worker mid-teardown), and
    delete_workers is called with that uid."""
    delta = ReconcileDelta(unexpected_uids={'worker-x'})
    strategy = MockPlacementStrategy(reconcile_delta=delta)
    controller = _make_controller(tmp_path, strategy=strategy)
    # worker-x is live in the inventory but not expected (no lease / local).
    controller._worker_inventory['worker-x'] = WorkerInventory(
        pool_worker_uid='worker-x',
        status=WorkerStatus.READY,
        last_heartbeat=time.time(),
        state=_snapshot(),
    )
    # A lease-less placement still homes on worker-x (lease released + local
    # copy evicted → unexpected, yet placement.worker == worker-x). The home
    # path would otherwise route a new activity here without a READY gate.
    controller._state.snapshots['snap-x'] = SnapshotPlacement(worker='worker-x')

    result = asyncio.run(controller._maintenance_once())

    assert result == delta
    # Non-routable before delete, the home-path placement is scrubbed, and the
    # strategy was asked to delete it.
    assert controller.worker_inventory[
        'worker-x'].status == WorkerStatus.DRAINING
    assert 'snap-x' not in controller.state.snapshots
    assert strategy.delete_workers_calls == [['worker-x']]


def test_queued_heartbeat_cannot_recreate_state_after_unexpected_scrub(
        tmp_path):
    """Admission is re-checked when the heartbeat finally acquires state_lock.

    Model another state write holding the lock while reconcile queues first and
    a daemon heartbeat queues second. The heartbeat has already refreshed the
    in-memory inventory, but its READY snapshot must not be applied after the
    preceding reconcile commits the worker to DRAINING and scrubs its state.
    """
    async def drive() -> tuple[PoolController, MockPlacementStrategy]:
        delta = ReconcileDelta(
            unexpected_uids={'worker-x'},
            observed_pool_worker_uids={'worker-x'},
        )
        strategy = MockPlacementStrategy(reconcile_delta=delta)
        controller = _make_controller(tmp_path, strategy=strategy)
        controller._worker_inventory['worker-x'] = WorkerInventory(
            pool_worker_uid='worker-x',
            status=WorkerStatus.READY,
            last_heartbeat=time.time(),
            state=_snapshot(),
        )

        await controller._state_lock.acquire()
        reconcile_task = asyncio.create_task(
            controller._reconcile_once(refresh_silent_workers=False))
        await asyncio.sleep(0)
        assert not reconcile_task.done()

        report_task = asyncio.create_task(
            controller.report_status.coroutine(
                _report(pool_worker_uid='worker-x',
                        ready_snapshots=['late-snapshot'])))
        await asyncio.sleep(0)
        assert not report_task.done()
        assert controller.worker_inventory[
            'worker-x'].status == WorkerStatus.READY

        controller._state_lock.release()
        await asyncio.gather(reconcile_task, report_task)
        return controller, strategy

    controller, strategy = asyncio.run(drive())

    assert strategy.delete_workers_calls == [['worker-x']]
    assert controller.worker_inventory[
        'worker-x'].status == WorkerStatus.DRAINING
    assert 'late-snapshot' not in controller.state.snapshots
    assert 'late-snapshot:worker-x' not in controller.state.local_copies


def test_deleted_worker_stays_draining_until_followup_observes_absence(
        tmp_path):
    """A teardown return is not proof that a previously observed pod is gone.
    Keep the tombstone through late heartbeats and prune it only after a later
    external observation confirms absence."""
    delta = ReconcileDelta(
        unexpected_uids={'worker-x'},
        observed_pool_worker_uids={'worker-x'},
    )
    strategy = MockPlacementStrategy(reconcile_delta=delta)
    strategy.delete_workers_returns = {'worker-x'}
    controller = _make_controller(tmp_path, strategy=strategy)
    controller._worker_inventory['worker-x'] = WorkerInventory(
        pool_worker_uid='worker-x',
        status=WorkerStatus.READY,
        last_heartbeat=time.time(),
        state=_snapshot(),
    )

    asyncio.run(controller._maintenance_once())

    assert strategy.delete_workers_calls == [['worker-x']]
    assert controller.worker_inventory[
        'worker-x'].status == WorkerStatus.DRAINING

    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-x',
                    ready_snapshots=['late-snapshot'])))

    assert controller.worker_inventory[
        'worker-x'].status == WorkerStatus.DRAINING
    assert 'late-snapshot' not in controller.state.snapshots

    # The delete already succeeded. As long as Kubernetes still observes the
    # worker, keep its DRAINING tombstone without re-running teardown.
    asyncio.run(controller._maintenance_once())

    assert strategy.delete_workers_calls == [['worker-x']]
    assert 'worker-x' in controller._delete_succeeded_pending_absence
    assert controller.worker_inventory[
        'worker-x'].status == WorkerStatus.DRAINING

    strategy.reconcile_delta = ReconcileDelta()
    asyncio.run(controller._maintenance_once())

    assert 'worker-x' not in controller.worker_inventory
    assert 'worker-x' not in controller._delete_succeeded_pending_absence
    assert strategy.delete_workers_calls == [['worker-x']]


def test_pending_absence_worker_does_not_block_new_unexpected_teardown(
        tmp_path):
    strategy = MockPlacementStrategy(reconcile_delta=ReconcileDelta(
        unexpected_uids={'worker-x'},
        observed_pool_worker_uids={'worker-x'},
    ))
    strategy.delete_workers_returns = {'worker-x'}
    controller = _make_controller(tmp_path, strategy=strategy)
    for uid in ('worker-x', 'worker-y'):
        controller._worker_inventory[uid] = WorkerInventory(
            pool_worker_uid=uid,
            status=WorkerStatus.READY,
            last_heartbeat=time.time(),
            state=_snapshot(),
        )

    asyncio.run(controller._reconcile_once(refresh_silent_workers=False))

    strategy.reconcile_delta = ReconcileDelta(
        unexpected_uids={'worker-x', 'worker-y'},
        observed_pool_worker_uids={'worker-x', 'worker-y'},
    )
    strategy.delete_workers_returns = {'worker-y'}
    asyncio.run(controller._reconcile_once(refresh_silent_workers=False))

    assert strategy.delete_workers_calls == [['worker-x'], ['worker-y']]
    assert controller._delete_succeeded_pending_absence == {
        'worker-x', 'worker-y'
    }


def test_activity_ready_after_absence_prune_cannot_recreate_phantom_copy(
        tmp_path):
    """A delayed activity event needs current PVC ownership after prune.

    Activity reports do not create daemon inventory. Once a successful teardown
    has been externally confirmed and its DRAINING tombstone/PVC bookkeeping are
    gone, a queued SNAPSHOT_READY must not create an expected-but-unreapable
    WARM copy for the deleted worker.
    """
    strategy = MockPlacementStrategy(reconcile_delta=ReconcileDelta(
        unexpected_uids={'worker-x'},
        observed_pool_worker_uids={'worker-x'},
    ))
    strategy.delete_workers_returns = {'worker-x'}
    controller = _make_controller(tmp_path, strategy=strategy)
    placement_state = controller.state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    placement_state.pvcs['worker-x'] = PvcState(pool_worker_uid='worker-x')
    controller._worker_inventory['worker-x'] = WorkerInventory(
        pool_worker_uid='worker-x',
        status=WorkerStatus.READY,
        last_heartbeat=time.time(),
        state=_snapshot(),
    )

    asyncio.run(controller._reconcile_once(refresh_silent_workers=False))

    assert controller.worker_inventory[
        'worker-x'].status == WorkerStatus.DRAINING
    assert 'worker-x' not in placement_state.pvcs
    assert 'worker-x' in controller._delete_succeeded_pending_absence

    strategy.reconcile_delta = ReconcileDelta()
    asyncio.run(controller._reconcile_once(refresh_silent_workers=False))

    assert 'worker-x' not in controller.worker_inventory
    assert 'worker-x' not in controller._delete_succeeded_pending_absence

    ready = WorkerEvent(
        type=WorkerEventType.SNAPSHOT_READY,
        snapshot_id='late-snapshot',
        size_bytes=4096,
    )
    asyncio.run(
        controller.report_status.coroutine(
            _report(
                pool_worker_uid='worker-x',
                worker_storage_mode=PoolStorageMode.ACTIVITY_ON_PVC,
                event=ready,
                ready_snapshots=[],
            )))

    assert 'late-snapshot' not in controller.state.snapshots
    assert 'late-snapshot:worker-x' not in controller.state.local_copies


def test_observed_unexpected_worker_without_heartbeat_gets_draining_tombstone(
        tmp_path):
    """Startup cleanup must quarantine a leaked pod before its first RPC.

    The K8s delete call may return while the pod is still observed. A synthetic
    DRAINING entry prevents its first late heartbeat from registering READY or
    recreating state before a later reconcile confirms pod absence.
    """
    delta = ReconcileDelta(
        unexpected_uids={'leaked-worker'},
        observed_pool_worker_uids={'leaked-worker'},
    )
    strategy = MockPlacementStrategy(reconcile_delta=delta)
    strategy.delete_workers_returns = {'leaked-worker'}
    controller = _make_controller(tmp_path, strategy=strategy)
    assert controller.worker_inventory == {}

    asyncio.run(controller._reconcile_once(refresh_silent_workers=False))

    assert controller.worker_inventory[
        'leaked-worker'].status == WorkerStatus.DRAINING
    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='leaked-worker',
                    ready_snapshots=['late-snapshot'])))
    assert controller.worker_inventory[
        'leaked-worker'].status == WorkerStatus.DRAINING
    assert 'late-snapshot' not in controller.state.snapshots

    strategy.reconcile_delta = ReconcileDelta()
    asyncio.run(controller._reconcile_once(refresh_silent_workers=False))
    assert 'leaked-worker' not in controller.worker_inventory


def test_failed_worker_teardown_stays_draining_and_retries(tmp_path):
    delta = ReconcileDelta(
        incompatible_uids={'worker-x'},
        observed_pool_worker_uids={'worker-x'},
    )
    strategy = MockPlacementStrategy(reconcile_delta=delta)
    controller = _make_controller(tmp_path, strategy=strategy)
    placement_state = controller.state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    placement_state.pvcs['worker-x'] = PvcState(pool_worker_uid='worker-x')
    controller._worker_inventory['worker-x'] = WorkerInventory(
        pool_worker_uid='worker-x',
        status=WorkerStatus.READY,
        last_heartbeat=time.time(),
        state=_snapshot(),
    )

    asyncio.run(controller._maintenance_once())
    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-x', ready_snapshots=['snap-late'])))
    strategy.reconcile_delta = ReconcileDelta(
        unexpected_uids={'worker-x'},
        observed_pool_worker_uids={'worker-x'},
    )
    asyncio.run(controller._maintenance_once())

    assert strategy.delete_workers_calls == [['worker-x'], ['worker-x']]
    assert controller.worker_inventory[
        'worker-x'].status == WorkerStatus.DRAINING
    assert 'snap-late' not in controller.state.snapshots


def test_maintenance_once_drops_freshly_leased_unexpected_worker(tmp_path):
    """Airtight re-check: reconcile flagged worker-a as unexpected on its
    (unlocked) snapshot, but a lease was reserved on it since. Under the lock
    the controller re-derives 'expected' and must NOT delete it."""
    delta = ReconcileDelta(unexpected_uids={'worker-a'})
    strategy = MockPlacementStrategy(reconcile_delta=delta)
    controller = _make_controller(tmp_path, strategy=strategy)
    controller._worker_inventory['worker-a'] = WorkerInventory(
        pool_worker_uid='worker-a',
        status=WorkerStatus.READY,
        last_heartbeat=time.time(),
        state=_snapshot(),
    )
    # A fresh lease landed on worker-a after reconcile snapshotted state.
    asyncio.run(controller.place_snapshot.coroutine(_request()))

    asyncio.run(controller._maintenance_once())

    # Re-check kept it alive: not deleted, still routable (READY).
    assert strategy.delete_workers_calls == [] or 'worker-a' not in (
        strategy.delete_workers_calls[0])
    assert controller.worker_inventory[
        'worker-a'].status != WorkerStatus.DRAINING


def test_maintenance_once_drops_retirement_when_worker_recovers_before_apply(
        tmp_path):
    deleted_snapshots: list[tuple[str, str]] = []

    async def delete_local(snapshot_id: str, pool_worker_uid: str) -> bool:
        deleted_snapshots.append((snapshot_id, pool_worker_uid))
        return True

    delta = ReconcileDelta(
        retired_uids={'worker-a'},
        orphan_snapshots={'snap-1': 'worker-a'},
    )
    strategy = MockPlacementStrategy(reconcile_delta=delta)
    controller = _make_controller(
        tmp_path,
        strategy=strategy,
        local_snapshot_deleter=delete_local,
    )
    asyncio.run(
        controller.report_status.coroutine(_report(ready_snapshots=['snap-1'])))
    inventory = controller.worker_inventory['worker-a']
    inventory.last_heartbeat = (controller._time() - WORKER_DEAD_THRESHOLD_SECS)
    inventory.status = WorkerStatus.DEAD

    def recover_before_apply() -> None:
        inventory.status = WorkerStatus.READY
        inventory.last_heartbeat = controller._time()

    strategy.on_reconcile = recover_before_apply

    result = asyncio.run(controller._maintenance_once())

    assert result.retired_uids == set()
    assert result.orphan_snapshots == {}
    assert strategy.delete_workers_calls == []
    assert deleted_snapshots == []
    assert controller.worker_inventory['worker-a'].status == WorkerStatus.READY
    assert controller.state.snapshots['snap-1'].worker == 'worker-a'
    assert controller.state.local_copies[
        'snap-1:worker-a'].local_state == SnapshotLocalCopyState.WARM


def test_maintenance_once_preserves_worker_when_k8s_revalidation_vetoes(
        tmp_path):
    strategy = MockPlacementStrategy(reconcile_delta=ReconcileDelta(
        retired_uids={'worker-a'}))
    strategy.revalidated_retired_workers = set()
    controller = _make_controller_with_state(tmp_path,
                                             _state_with_expected_worker(),
                                             strategy=strategy)
    controller.worker_inventory['worker-a'] = WorkerInventory(
        pool_worker_uid='worker-a', status=WorkerStatus.DEAD,
        last_heartbeat=1.0, state=_snapshot())

    result = asyncio.run(controller._maintenance_once())

    assert result.retired_uids == set()
    assert strategy.delete_workers_calls == []
    placement_state = controller.state.placement_state
    assert isinstance(placement_state, K8sPlacementState)
    assert 'worker-a' in placement_state.pvcs
    assert controller.state.snapshots['snap-1'].leases == ['job-1']


def test_maintenance_once_deletes_retired_worker_even_with_stale_lease(
        tmp_path):
    """Rebind-timeout retired workers are NOT subject to the freshly-leased
    re-check: the daemon is gone and any lease is dangling, so the worker is
    still retired (DRAINING + state cleared) and handed to delete_workers."""
    delta = ReconcileDelta(retired_uids={'worker-a'})
    strategy = MockPlacementStrategy(reconcile_delta=delta)
    controller = _make_controller(tmp_path, strategy=strategy)
    asyncio.run(
        controller.report_status.coroutine(_report(ready_snapshots=['snap-1'])))
    # Open lease on worker-a → it would look "expected", but retired wins.
    asyncio.run(controller.place_snapshot.coroutine(_request()))
    inventory = controller.worker_inventory['worker-a']
    inventory.last_heartbeat = (controller._time() - WORKER_DEAD_THRESHOLD_SECS)
    inventory.status = WorkerStatus.DEAD

    asyncio.run(controller._maintenance_once())

    assert controller.worker_inventory[
        'worker-a'].status == WorkerStatus.DRAINING
    assert strategy.delete_workers_calls == [['worker-a']]
    assert 'snap-1' not in controller.state.snapshots


def test_sweep_worker_health_does_not_resurrect_draining(tmp_path):
    """A DRAINING worker (scale-down committed) must not be aged back to
    READY by the heartbeat-age sweep, even if its heartbeat is fresh."""
    controller = _make_controller(tmp_path)
    controller._worker_inventory['worker-a'] = WorkerInventory(
        pool_worker_uid='worker-a',
        status=WorkerStatus.DRAINING,
        last_heartbeat=time.time(),  # fresh heartbeat
        state=_snapshot(),
    )

    controller._sweep_worker_health(time.time())

    assert controller.worker_inventory[
        'worker-a'].status == WorkerStatus.DRAINING


def test_report_status_does_not_resurrect_draining_worker(tmp_path):
    """A heartbeat arriving before the daemon is actually scaled down must not
    flip a DRAINING worker back to READY (which would re-expose it to
    routing in the delete window)."""
    controller = _make_controller(tmp_path)
    asyncio.run(controller.report_status.coroutine(_report()))
    controller.worker_inventory['worker-a'].status = WorkerStatus.DRAINING

    asyncio.run(controller.report_status.coroutine(_report()))

    assert controller.worker_inventory[
        'worker-a'].status == WorkerStatus.DRAINING


def test_draining_worker_still_records_completed_dataplane_upload(tmp_path):
    controller = _make_controller(tmp_path)
    asyncio.run(controller.report_status.coroutine(_report()))
    controller.worker_inventory['worker-a'].status = WorkerStatus.DRAINING
    uploaded = WorkerEvent(
        type=WorkerEventType.SNAPSHOT_PERSIST_UPLOADED,
        snapshot_id='snap-1',
        dataplane_path='s3://bucket/snap-1',
        size_bytes=4096,
    )

    asyncio.run(controller.report_status.coroutine(_report(event=uploaded)))

    backup = controller.state.dataplane_backups['snap-1']
    assert backup.dataplane_path == 's3://bucket/snap-1'
    assert backup.size_bytes == 4096
    assert controller.worker_inventory[
        'worker-a'].status == WorkerStatus.DRAINING


def test_late_activity_ready_does_not_recreate_scrubbed_worker_state(tmp_path):
    controller = _make_controller(tmp_path)
    asyncio.run(controller.report_status.coroutine(_report()))
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        state=SnapshotState.READY,
    )
    controller._apply_reconcile_worker_delta(
        ReconcileDelta(retired_uids={'worker-a'}))
    ready = WorkerEvent(
        type=WorkerEventType.SNAPSHOT_READY,
        snapshot_id='snap-1',
        size_bytes=4096,
    )

    asyncio.run(
        controller.report_status.coroutine(
            _report(
                worker_storage_mode=PoolStorageMode.ACTIVITY_ON_PVC,
                event=ready,
                ready_snapshots=['snap-1'],
            )))

    assert controller.state.snapshots == {}
    assert controller.state.local_copies[
        'snap-1:worker-a'].local_state == SnapshotLocalCopyState.EVICTED
    assert controller.worker_inventory[
        'worker-a'].status == WorkerStatus.DRAINING


def test_maintenance_once_cleans_orphan_snapshot_from_reconcile_delta(tmp_path):
    deleted: list[tuple[str, str]] = []

    async def delete_local(snapshot_id: str, pool_worker_uid: str) -> bool:
        deleted.append((snapshot_id, pool_worker_uid))
        return True

    delta = ReconcileDelta(orphan_snapshots={'snap-orphan': 'worker-c'})
    strategy = MockPlacementStrategy(reconcile_delta=delta)
    controller = _make_controller(
        tmp_path,
        strategy=strategy,
        local_snapshot_deleter=delete_local,
    )
    controller.state.snapshots['snap-orphan'] = SnapshotPlacement(
        worker='worker-c',
        state=SnapshotState.READY,
    )
    controller.state.local_copies['snap-orphan:worker-c'] = (
        SnapshotLocalCopyRecord(
            snapshot_id='snap-orphan',
            pool_worker_uid='worker-c',
            local_state=SnapshotLocalCopyState.WARM,
        ))

    result = asyncio.run(controller._maintenance_once())

    assert result == delta
    assert deleted == [('snap-orphan', 'worker-c')]
    assert 'snap-orphan' not in controller.state.snapshots
    assert controller.state.local_copies[
        'snap-orphan:worker-c'].local_state == SnapshotLocalCopyState.EVICTED


def test_maintenance_once_retries_orphan_snapshot_after_delete_failure(
        tmp_path):
    attempts: list[tuple[str, str]] = []

    async def delete_local(snapshot_id: str, pool_worker_uid: str) -> bool:
        attempts.append((snapshot_id, pool_worker_uid))
        raise RuntimeError('daemon unavailable')

    delta = ReconcileDelta(orphan_snapshots={'snap-orphan': 'worker-c'})
    strategy = MockPlacementStrategy(reconcile_delta=delta)
    controller = _make_controller(
        tmp_path,
        strategy=strategy,
        local_snapshot_deleter=delete_local,
    )
    controller.state.snapshots['snap-orphan'] = SnapshotPlacement(
        worker='worker-c',
        state=SnapshotState.READY,
    )
    controller.state.local_copies['snap-orphan:worker-c'] = (
        SnapshotLocalCopyRecord(
            snapshot_id='snap-orphan',
            pool_worker_uid='worker-c',
            local_state=SnapshotLocalCopyState.WARM,
        ))

    result = asyncio.run(controller._maintenance_once())

    assert result == delta
    assert attempts == [('snap-orphan', 'worker-c')]
    assert controller.state.snapshots['snap-orphan'].state == (
        SnapshotState.READY)
    assert controller.state.local_copies[
        'snap-orphan:worker-c'].local_state == SnapshotLocalCopyState.WARM


def test_maintenance_once_skips_orphan_snapshot_with_active_lease(tmp_path):
    deleted: list[tuple[str, str]] = []

    async def delete_local(snapshot_id: str, pool_worker_uid: str) -> bool:
        deleted.append((snapshot_id, pool_worker_uid))
        return True

    delta = ReconcileDelta(orphan_snapshots={'snap-1': 'worker-a'})
    strategy = MockPlacementStrategy(reconcile_delta=delta)
    controller = _make_controller(
        tmp_path,
        strategy=strategy,
        local_snapshot_deleter=delete_local,
    )
    _open_snapshot(controller)
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.WARM,
    ))

    result = asyncio.run(controller._maintenance_once())

    assert result == delta
    assert deleted == []
    assert controller.state.snapshots['snap-1'].state == SnapshotState.READY
    assert controller.state.local_copies[
        'snap-1:worker-a'].local_state == SnapshotLocalCopyState.WARM


def test_maintenance_once_orphan_delete_does_not_hold_state_lock(tmp_path):
    deleted: list[tuple[str, str]] = []
    observed_state_lock: list[bool] = []
    controller: PoolController

    async def delete_local(snapshot_id: str, pool_worker_uid: str) -> bool:
        observed_state_lock.append(controller._state_lock.locked())
        async with controller._state_lock:
            placement = controller.state.snapshots.setdefault(
                snapshot_id,
                SnapshotPlacement(worker=pool_worker_uid),
            )
            placement.leases.append('job-new')
        deleted.append((snapshot_id, pool_worker_uid))
        return True

    delta = ReconcileDelta(orphan_snapshots={'snap-1': 'worker-a'})
    strategy = MockPlacementStrategy(reconcile_delta=delta)
    controller = _make_controller(
        tmp_path,
        strategy=strategy,
        local_snapshot_deleter=delete_local,
    )
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        state=SnapshotState.READY,
    )
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.WARM,
    ))

    result = asyncio.run(asyncio.wait_for(controller._maintenance_once(), 1))

    assert result == delta
    assert observed_state_lock == [False]
    assert deleted == [('snap-1', 'worker-a')]
    assert controller.state.local_copies[
        'snap-1:worker-a'].local_state == SnapshotLocalCopyState.WARM


def test_dangling_lease_sweep_keeps_running_workflow_lease(tmp_path):
    seen: list[str] = []

    async def workflow_status(workflow_id: str) -> WorkflowLeaseState:
        seen.append(workflow_id)
        return WorkflowLeaseState.RUNNING

    strategy = MockPlacementStrategy()
    controller = _make_controller(
        tmp_path,
        strategy=strategy,
        workflow_status_checker=workflow_status,
    )
    asyncio.run(controller.place_snapshot.coroutine(_request()))

    released = asyncio.run(controller._dangling_lease_sweep_once())

    assert released == 0
    assert seen == ['job-1']
    assert controller.state.snapshots['snap-1'].leases == ['job-1']


def test_dangling_lease_sweep_continues_after_release_failure(tmp_path):
    async def workflow_status(workflow_id: str) -> WorkflowLeaseState:
        assert workflow_id in {'job-1', 'job-2'}
        return WorkflowLeaseState.TERMINAL

    strategy = MockPlacementStrategy()
    controller = _make_controller(
        tmp_path,
        strategy=strategy,
        workflow_status_checker=workflow_status,
    )
    asyncio.run(controller.place_snapshot.coroutine(_request()))
    asyncio.run(
        controller.place_snapshot.coroutine(
            _request(snapshot_id='snap-2', job_id='job-2')))

    original_release = strategy.release

    async def release(request, handle):
        if request.job_id == 'job-1':
            strategy.release_calls.append(handle)
            raise RuntimeError('release failed')
        await original_release(request, handle)

    strategy.release = release

    released = asyncio.run(controller._dangling_lease_sweep_once())

    assert released == 1
    assert controller.state.snapshots['snap-1'].leases == ['job-1']
    assert controller.state.snapshots['snap-2'].leases == []
    assert controller.state.snapshots['snap-2'].last_lease_released is not None
    assert controller.state.snapshot_compliance[
        'snap-2'].active_consumer_count == 0
    assert len(strategy.release_calls) == 2


def test_dangling_lease_sweep_releases_not_found_workflow_lease(tmp_path):
    async def workflow_status(workflow_id: str) -> WorkflowLeaseState:
        assert workflow_id == 'job-1'
        return WorkflowLeaseState.NOT_FOUND

    controller = _make_controller(
        tmp_path,
        strategy=MockPlacementStrategy(),
        workflow_status_checker=workflow_status,
    )
    asyncio.run(controller.place_snapshot.coroutine(_request()))

    released = asyncio.run(controller._dangling_lease_sweep_once())

    assert released == 1
    assert controller.state.snapshots['snap-1'].leases == []


def test_dangling_lease_sweep_groups_sharded_leases_by_workflow_id(tmp_path):
    """Sharded BP siblings share a parent ``workflow_id`` but have
    distinct ``job_id`` values. The sweep groups by the structured
    ``workflow_id`` field on the request (recorded into
    ``lease_workflow_ids`` at commit), so a 5-shard BP collapses to one
    Temporal status check per sweep instead of five.
    """
    seen: list[str] = []

    async def workflow_status(workflow_id: str) -> WorkflowLeaseState:
        seen.append(workflow_id)
        return WorkflowLeaseState.RUNNING

    controller = _make_controller(
        tmp_path,
        strategy=MockPlacementStrategy(),
        workflow_status_checker=workflow_status,
    )
    asyncio.run(
        controller.place_snapshot.coroutine(
            _request(job_id='wf-abc#shard-0', workflow_id='wf-abc')))
    asyncio.run(
        controller.place_snapshot.coroutine(
            _request(snapshot_id='snap-2', job_id='wf-abc#shard-1',
                     workflow_id='wf-abc')))

    released = asyncio.run(controller._dangling_lease_sweep_once())

    assert released == 0
    # Both shards roll up to one ``wf-abc`` status check (batched sweep).
    assert seen == ['wf-abc']
    # Both leases carry the parent workflow_id on the placement record.
    assert (controller.state.snapshots['snap-1'].
            lease_workflow_ids['wf-abc#shard-0'] == 'wf-abc')
    assert (controller.state.snapshots['snap-2'].
            lease_workflow_ids['wf-abc#shard-1'] == 'wf-abc')


def test_dangling_lease_sweep_keeps_lease_open_on_status_error(tmp_path):
    async def workflow_status(workflow_id: str) -> WorkflowLeaseState:
        del workflow_id
        raise RuntimeError('temporal unavailable')

    strategy = MockPlacementStrategy()
    controller = _make_controller(
        tmp_path,
        strategy=strategy,
        workflow_status_checker=workflow_status,
    )
    asyncio.run(controller.place_snapshot.coroutine(_request()))

    released = asyncio.run(controller._dangling_lease_sweep_once())

    assert released == 0
    assert controller.state.snapshots['snap-1'].leases == ['job-1']
    assert strategy.release_calls == []


def test_maintenance_once_runs_dangling_sweep_on_cadence(tmp_path):
    async def workflow_status(workflow_id: str) -> WorkflowLeaseState:
        assert workflow_id == 'job-1'
        return WorkflowLeaseState.TERMINAL

    strategy = MockPlacementStrategy()
    controller = _make_controller(
        tmp_path,
        strategy=strategy,
        workflow_status_checker=workflow_status,
    )
    asyncio.run(controller.place_snapshot.coroutine(_request()))

    controller._last_dangling_lease_sweep_at = time.time()
    asyncio.run(controller._maintenance_once())
    assert controller.state.snapshots['snap-1'].leases == ['job-1']

    controller._last_dangling_lease_sweep_at = (
        time.time() - LEASE_DEFENSIVE_SWEEP_INTERVAL_SECS)
    asyncio.run(controller._maintenance_once())
    assert controller.state.snapshots['snap-1'].leases == []


def test_lifecycle_sweeps_keep_l2_active_while_l1_live(tmp_path):
    controller = _make_controller(tmp_path)
    past = now_utc() - timedelta(seconds=1)
    controller.state.snapshot_compliance['snap-1'] = (SnapshotComplianceRecord(
        snapshot_id='snap-1',
        active_consumer_count=0,
        compliance_delete_at=past,
    ))
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.WARM,
        evictable_under_pressure_at=past,
    ))

    asyncio.run(controller._lifecycle_sweeps_once())

    compliance = controller.state.snapshot_compliance['snap-1']
    assert compliance.expired_at is None
    assert compliance.deleted_at is None
    assert controller.state.local_copies['snap-1:worker-a'].local_state == (
        SnapshotLocalCopyState.EVICTION_CANDIDATE)
    loaded = asyncio.run(controller._persistence.load())
    assert loaded.snapshot_compliance == controller.state.snapshot_compliance
    assert loaded.local_copies == controller.state.local_copies


def test_compliance_sweep_expires_only_after_l1_evicted(tmp_path):
    deleted: list[DataplaneBackup] = []

    async def delete_backup(backup: DataplaneBackup) -> None:
        deleted.append(backup)

    controller = _make_controller(tmp_path, dataplane_deleter=delete_backup)
    past = now_utc() - timedelta(seconds=1)
    backup = DataplaneBackup(
        snapshot_id='snap-1',
        dataplane_path='s3://bucket/snap-1',
    )
    controller.state.dataplane_backups['snap-1'] = backup
    controller.state.snapshot_compliance['snap-1'] = (SnapshotComplianceRecord(
        snapshot_id='snap-1',
        active_consumer_count=0,
        compliance_delete_at=past,
    ))
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.EVICTED,
    ))

    asyncio.run(controller._lifecycle_sweeps_once())

    assert deleted == [backup]
    assert controller.state.snapshot_compliance['snap-1'].deleted_at is not None
    assert 'snap-1' not in controller.state.dataplane_backups


def test_compliance_sweep_deletes_dataplane_backup_when_l1_clear(tmp_path):
    deleted: list[DataplaneBackup] = []
    controller_box: list[PoolController] = []

    async def delete_backup(backup: DataplaneBackup) -> None:
        assert not controller_box[0]._state_lock.locked()
        deleted.append(backup)

    controller = _make_controller(tmp_path, dataplane_deleter=delete_backup)
    controller_box.append(controller)
    past = now_utc() - timedelta(seconds=1)
    backup = DataplaneBackup(
        snapshot_id='snap-1',
        dataplane_path='s3://bucket/snap-1',
    )
    controller.state.dataplane_backups['snap-1'] = backup
    controller.state.snapshot_compliance['snap-1'] = (SnapshotComplianceRecord(
        snapshot_id='snap-1',
        active_consumer_count=0,
        compliance_delete_at=past,
        expired_at=past,
    ))

    asyncio.run(controller._lifecycle_sweeps_once())

    assert deleted == [backup]
    assert 'snap-1' not in controller.state.dataplane_backups
    assert controller.state.snapshot_compliance['snap-1'].deleted_at is not None


def test_compliance_sweep_keeps_expired_when_dataplane_delete_fails(tmp_path):
    async def delete_backup(backup: DataplaneBackup) -> None:
        del backup
        raise RuntimeError('s3 unavailable')

    controller = _make_controller(tmp_path, dataplane_deleter=delete_backup)
    past = now_utc() - timedelta(seconds=1)
    controller.state.dataplane_backups['snap-1'] = DataplaneBackup(
        snapshot_id='snap-1',
        dataplane_path='s3://bucket/snap-1',
    )
    controller.state.snapshot_compliance['snap-1'] = (SnapshotComplianceRecord(
        snapshot_id='snap-1',
        active_consumer_count=0,
        compliance_delete_at=past,
        expired_at=past,
    ))

    asyncio.run(controller._lifecycle_sweeps_once())

    assert 'snap-1' in controller.state.dataplane_backups
    compliance = controller.state.snapshot_compliance['snap-1']
    assert compliance.expired_at == past
    assert compliance.deleted_at is None


def test_local_retention_sweep_dispatches_daemon_delete_on_force(tmp_path):
    deleted: list[tuple[str, str]] = []
    controller_box: list[PoolController] = []

    async def delete_local(snapshot_id: str, pool_worker_uid: str) -> bool:
        controller = controller_box[0]
        assert not controller._state_lock.locked()
        assert controller._warm_location_uid(snapshot_id) is None
        assert controller.state.local_copies[
            f'{snapshot_id}:{pool_worker_uid}'].local_state == (
                SnapshotLocalCopyState.EVICTING)
        deleted.append((snapshot_id, pool_worker_uid))
        return True

    controller = _make_controller(tmp_path, local_snapshot_deleter=delete_local)
    controller_box.append(controller)
    past = now_utc() - timedelta(seconds=1)
    controller.state.dataplane_backups['snap-1'] = DataplaneBackup(
        snapshot_id='snap-1',
        dataplane_path='s3://bucket/snap-1',
    )
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        state=SnapshotState.READY,
    )
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.EVICTION_CANDIDATE,
        force_evict_at=past,
    ))

    asyncio.run(controller._lifecycle_sweeps_once())

    assert deleted == [('snap-1', 'worker-a')]
    assert 'snap-1' not in controller.state.snapshots
    local = controller.state.local_copies['snap-1:worker-a']
    assert local.local_state == SnapshotLocalCopyState.EVICTED
    assert local.evicted_at is not None


def test_local_retention_sweep_does_not_delete_candidate_without_pressure(
        tmp_path):
    deleted: list[tuple[str, str]] = []

    async def delete_local(snapshot_id: str, pool_worker_uid: str) -> bool:
        deleted.append((snapshot_id, pool_worker_uid))
        return True

    controller = _make_controller(tmp_path, local_snapshot_deleter=delete_local)
    controller.state.dataplane_backups['snap-1'] = DataplaneBackup(
        snapshot_id='snap-1',
        dataplane_path='s3://bucket/snap-1',
    )
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.EVICTION_CANDIDATE,
    ))

    asyncio.run(controller._lifecycle_sweeps_once())

    assert deleted == []
    assert controller.state.local_copies['snap-1:worker-a'].local_state == (
        SnapshotLocalCopyState.EVICTION_CANDIDATE)


def test_local_retention_sweep_reactive_trigger_deletes_candidate(tmp_path):
    deleted: list[tuple[str, str]] = []

    async def delete_local(snapshot_id: str, pool_worker_uid: str) -> bool:
        deleted.append((snapshot_id, pool_worker_uid))
        return True

    controller = _make_controller(tmp_path, local_snapshot_deleter=delete_local)
    controller.state.dataplane_backups['snap-1'] = DataplaneBackup(
        snapshot_id='snap-1',
        dataplane_path='s3://bucket/snap-1',
    )
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.EVICTION_CANDIDATE,
    ))

    controller._request_reactive_eviction()
    asyncio.run(controller._lifecycle_sweeps_once())

    assert deleted == [('snap-1', 'worker-a')]
    assert controller.state.local_copies[
        'snap-1:worker-a'].local_state == SnapshotLocalCopyState.EVICTED
    assert controller._reactive_eviction_requested_at is None


def test_local_retention_sweep_expired_reactive_trigger_does_not_delete(
        tmp_path):
    deleted: list[tuple[str, str]] = []

    async def delete_local(snapshot_id: str, pool_worker_uid: str) -> bool:
        deleted.append((snapshot_id, pool_worker_uid))
        return True

    controller = _make_controller(tmp_path, local_snapshot_deleter=delete_local)
    controller.state.dataplane_backups['snap-1'] = DataplaneBackup(
        snapshot_id='snap-1',
        dataplane_path='s3://bucket/snap-1',
    )
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.EVICTION_CANDIDATE,
    ))
    controller._reactive_eviction_requested_at = (
        time.time() - REACTIVE_EVICTION_WINDOW_SECS - 1)

    asyncio.run(controller._lifecycle_sweeps_once())

    assert deleted == []
    assert controller.state.local_copies['snap-1:worker-a'].local_state == (
        SnapshotLocalCopyState.EVICTION_CANDIDATE)
    assert controller._reactive_eviction_requested_at is None


def test_local_retention_sweep_reactive_trigger_respects_active_lease(tmp_path):
    deleted: list[tuple[str, str]] = []

    async def delete_local(snapshot_id: str, pool_worker_uid: str) -> bool:
        deleted.append((snapshot_id, pool_worker_uid))
        return True

    controller = _make_controller(tmp_path, local_snapshot_deleter=delete_local)
    controller.state.dataplane_backups['snap-1'] = DataplaneBackup(
        snapshot_id='snap-1',
        dataplane_path='s3://bucket/snap-1',
    )
    _open_snapshot(controller)
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.EVICTION_CANDIDATE,
    ))

    controller._request_reactive_eviction()
    asyncio.run(controller._lifecycle_sweeps_once())

    assert deleted == []
    assert controller.state.local_copies['snap-1:worker-a'].local_state == (
        SnapshotLocalCopyState.EVICTION_CANDIDATE)
    assert controller._reactive_eviction_requested_at is None


def test_local_retention_sweep_keeps_barrier_on_daemon_delete_failure(tmp_path):
    attempts: list[tuple[str, str]] = []

    async def delete_local(snapshot_id: str, pool_worker_uid: str) -> bool:
        attempts.append((snapshot_id, pool_worker_uid))
        raise RuntimeError('daemon unavailable')

    strategy = MockPlacementStrategy()
    controller = _make_controller(
        tmp_path,
        strategy=strategy,
        local_snapshot_deleter=delete_local,
    )
    past = now_utc() - timedelta(seconds=1)
    controller.state.dataplane_backups['snap-1'] = DataplaneBackup(
        snapshot_id='snap-1',
        dataplane_path='s3://bucket/snap-1',
    )
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a', state=SnapshotState.READY)
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.EVICTION_CANDIDATE,
        force_evict_at=past,
    ))

    asyncio.run(controller._lifecycle_sweeps_once())

    assert attempts == [('snap-1', 'worker-a')]
    assert controller.state.local_copies['snap-1:worker-a'].local_state == (
        SnapshotLocalCopyState.EVICTING)
    with pytest.raises(SnapshotEvictionInProgressError):
        asyncio.run(
            controller.place_snapshot.coroutine(_request(job_id='job-2')))
    assert strategy.route_calls == []


def test_placement_waits_for_inflight_eviction_then_retries_cold(tmp_path):
    async def drive() -> None:
        delete_started = asyncio.Event()
        allow_delete = asyncio.Event()

        async def delete_local(snapshot_id: str, pool_worker_uid: str) -> bool:
            assert (snapshot_id, pool_worker_uid) == ('snap-1', 'worker-a')
            delete_started.set()
            await allow_delete.wait()
            return True

        strategy = MockPlacementStrategy()
        controller = _make_controller(
            tmp_path,
            strategy=strategy,
            local_snapshot_deleter=delete_local,
        )
        past = now_utc() - timedelta(seconds=1)
        controller.state.dataplane_backups['snap-1'] = DataplaneBackup(
            snapshot_id='snap-1', dataplane_path='s3://bucket/snap-1')
        controller.state.snapshots['snap-1'] = SnapshotPlacement(
            worker='worker-a', state=SnapshotState.READY)
        controller.state.local_copies['snap-1:worker-a'] = (
            SnapshotLocalCopyRecord(
                snapshot_id='snap-1',
                pool_worker_uid='worker-a',
                local_state=SnapshotLocalCopyState.EVICTION_CANDIDATE,
                force_evict_at=past,
            ))

        sweep = asyncio.create_task(controller._lifecycle_sweeps_once())
        await delete_started.wait()

        with pytest.raises(SnapshotEvictionInProgressError):
            await controller.place_snapshot.coroutine(_request(job_id='job-2'))
        assert strategy.route_calls == []

        allow_delete.set()
        await sweep

        assert controller.state.local_copies[
            'snap-1:worker-a'].local_state == SnapshotLocalCopyState.EVICTED
        result = await controller.place_snapshot.coroutine(
            _request(job_id='job-2'))
        assert result.status == PlacementStatus.COLD
        assert len(strategy.route_calls) == 1

    asyncio.run(drive())


def test_local_retention_sweep_retries_persisted_evicting_state(tmp_path):
    deleted: list[tuple[str, str]] = []

    async def delete_local(snapshot_id: str, pool_worker_uid: str) -> bool:
        deleted.append((snapshot_id, pool_worker_uid))
        return True

    controller = _make_controller(tmp_path, local_snapshot_deleter=delete_local)
    controller.state.dataplane_backups['snap-1'] = DataplaneBackup(
        snapshot_id='snap-1',
        dataplane_path='s3://bucket/snap-1',
    )
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.EVICTING,
    ))

    asyncio.run(controller._lifecycle_sweeps_once())

    assert deleted == [('snap-1', 'worker-a')]
    assert controller.state.local_copies[
        'snap-1:worker-a'].local_state == SnapshotLocalCopyState.EVICTED


def test_start_maintenance_loop_runs_periodic_tick(tmp_path):
    async def drive() -> MockPlacementStrategy:
        strategy = MockPlacementStrategy()
        controller = _make_controller(tmp_path, strategy=strategy)
        controller._start_maintenance_loop(interval_secs=3600)
        await asyncio.sleep(0)
        assert controller._maintenance_task is not None
        controller._maintenance_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await controller._maintenance_task
        return strategy

    strategy = asyncio.run(drive())

    assert len(strategy.reconcile_calls) == 1


def test_create_pool_controller_runs_startup_reconcile_before_return(tmp_path):
    deleted: list[tuple[str, str]] = []

    async def delete_local(snapshot_id: str, pool_worker_uid: str) -> bool:
        deleted.append((snapshot_id, pool_worker_uid))
        return True

    async def drive(
    ) -> tuple[PoolController, MockPlacementStrategy, PlacementResult]:
        strategy = MockPlacementStrategy(reconcile_delta=ReconcileDelta(
            unexpected_uids={'leaked-worker'},
            orphan_snapshots={'snap-orphan': 'worker-a'}))

        def factory(controller: PoolController) -> MockPlacementStrategy:
            context = PoolControllerPlacementContext(controller)
            assert context.worker_inventory == {}
            assert context.warm_location_uid('missing-snapshot') is None
            assert context.snapshot_size_bytes('missing-snapshot') is None
            return strategy

        controller = await create_pool_controller(
            _CreateControllerConfig(  # type: ignore[arg-type]
                str(tmp_path / 'pool_state.json')),
            strategy_factory=factory,
            local_snapshot_deleter=delete_local,
            start_maintenance_loop=False,
        )
        result = await controller.place_snapshot.coroutine(_request())
        return controller, strategy, result

    controller, strategy, result = asyncio.run(drive())

    assert controller._maintenance_task is None
    assert deleted == [('snap-orphan', 'worker-a')]
    assert controller.state.local_copies[
        'snap-orphan:worker-a'].local_state == SnapshotLocalCopyState.EVICTED
    assert len(strategy.reconcile_calls) == 1
    assert strategy.delete_workers_calls == [['leaked-worker']]
    assert result.task_queue == 'diskgraph-activity-tq-job-1'
    assert len(strategy.route_calls) == 1
    assert len(strategy.provision_calls) == 1


def test_create_pool_controller_wires_lifecycle_policy(tmp_path):
    async def delete_local(snapshot_id: str, pool_worker_uid: str) -> bool:
        del snapshot_id, pool_worker_uid
        return True

    async def drive() -> PoolController:
        return await create_pool_controller(
            _CreateControllerConfig(  # type: ignore[arg-type]
                str(tmp_path / 'pool_state.json'),
                data_compute=DiskgraphConfig(
                    pool_min_ssd_idle_secs=11,
                    pool_max_ssd_idle_secs=22,
                    pool_max_dataplane_retention_secs=33,
                ),
            ),
            strategy_factory=lambda _: MockPlacementStrategy(),
            local_snapshot_deleter=delete_local,
            start_maintenance_loop=False,
        )

    controller = asyncio.run(drive())

    assert controller._lifecycle_policy == SnapshotLifecyclePolicy(
        min_ssd_idle_secs=11,
        max_ssd_idle_secs=22,
        max_dataplane_retention_secs=33,
    )


def test_configured_lifecycle_policy_clamps_implicit_min_to_max(tmp_path):
    policy = _configured_lifecycle_policy(
        _CreateControllerConfig(  # type: ignore[arg-type]
            str(tmp_path / 'pool_state.json'),
            data_compute=DiskgraphConfig(pool_max_ssd_idle_secs=3600),
        ))

    assert policy.min_ssd_idle_secs == 3600
    assert policy.max_ssd_idle_secs == 3600


def test_k8s_pool_storage_class_defaults_to_topolvm_ssd(monkeypatch):
    monkeypatch.delenv('POOL_STORAGE_CLASS_NAME', raising=False)

    assert _k8s_pool_storage_class() == 'topolvm-ssd'


def test_k8s_pool_storage_class_uses_env_override(monkeypatch):
    monkeypatch.setenv('POOL_STORAGE_CLASS_NAME', 'custom-local-ssd')

    assert _k8s_pool_storage_class() == 'custom-local-ssd'


def test_pool_worker_service_account_preserves_non_chart_inheritance(
        monkeypatch):
    monkeypatch.delenv('POOL_WORKER_SERVICE_ACCOUNT', raising=False)

    assert _configured_pool_worker_service_account() is None


def test_pool_worker_service_account_uses_env_override(monkeypatch):
    monkeypatch.setenv('POOL_WORKER_SERVICE_ACCOUNT', '  kumo-worker  ')

    assert _configured_pool_worker_service_account() == 'kumo-worker'


def test_pool_worker_fs_group_supports_platform_assignment(monkeypatch):
    monkeypatch.setenv('POOL_WORKER_FS_GROUP', '')

    assert _configured_pool_worker_fs_group() is None


def test_configured_manager_uses_worker_service_account(monkeypatch):
    manager_type = Mock()
    monkeypatch.setattr('pool.worker_manager.K8sPoolWorkerManager',
                        manager_type)
    monkeypatch.setenv('POOL_WORKER_SERVICE_ACCOUNT', 'kumo-worker')
    monkeypatch.setenv('POOL_WORKER_GCP_APPLICATION_CREDENTIALS_SECRET',
                       ' gcp-application-credentials ')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS',
                       ' /customer/adc/runtime-key.json ')
    monkeypatch.setenv('POOL_WORKER_GCP_APPLICATION_CREDENTIALS_KEY',
                       ' credentials.json ')
    diskgraph = DiskgraphConfig(
        pool_strategy='k8s',
        k8s_pool=K8sPoolStrategyConfig(
            pool_diskgraph_node_types={
                'test-node': DiskgraphNodeType(ssd_gib=256, gpu_count=1),
            }),
    )
    config = DeployEnvConfig(
        deploy_mode=DeployStage.DEV,
        instance_id='manager-test',
        data_compute=diskgraph,
    )

    manager, _, _, _ = _build_configured_k8s_manager(config, diskgraph)

    assert manager is manager_type.return_value
    assert manager_type.call_args.kwargs['service_account'] == 'kumo-worker'
    assert manager_type.call_args.kwargs['runtime_fs_group'] == 999
    assert manager_type.call_args.kwargs[
        'gcp_application_credentials_secret'
    ] == 'gcp-application-credentials'
    assert manager_type.call_args.kwargs[
        'gcp_application_credentials_secret_key'
    ] == 'credentials.json'
    assert manager_type.call_args.kwargs[
        'gcp_application_credentials_path'
    ] == '/customer/adc/runtime-key.json'


def test_create_pool_controller_tolerates_startup_reconcile_failure(tmp_path):
    async def delete_local(snapshot_id: str, pool_worker_uid: str) -> bool:
        del snapshot_id, pool_worker_uid
        return True

    async def drive(
    ) -> tuple[PoolController, MockPlacementStrategy, PlacementResult]:
        strategy = MockPlacementStrategy(fail_reconcile=True)

        controller = await create_pool_controller(
            _CreateControllerConfig(  # type: ignore[arg-type]
                str(tmp_path / 'pool_state.json')),
            strategy_factory=lambda _: strategy,
            local_snapshot_deleter=delete_local,
            start_maintenance_loop=False,
        )
        result = await controller.place_snapshot.coroutine(_request())
        return controller, strategy, result

    controller, strategy, result = asyncio.run(drive())

    assert controller._maintenance_task is None
    assert len(strategy.reconcile_calls) == 1
    assert result.task_queue == 'diskgraph-activity-tq-job-1'


def test_get_pool_stats_defaults_to_aggregate_only(tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    asyncio.run(controller.report_status.coroutine(_report()))
    asyncio.run(controller.place_snapshot.coroutine(_request()))

    stats = asyncio.run(controller.get_pool_stats.coroutine())

    assert stats.num_nodes == 1
    assert stats.num_healthy_nodes == 1
    assert stats.active_leases == 1
    assert stats.durable_workers == [
        PoolWorkerStats(
            pool_worker_uid='worker-a',
            health=WorkerStatus.READY,
            pvc_free_bytes=200_000,
            pvc_total_bytes=1_000_000,
            gpu_slots_total=1,
            gpu_slots_in_use=1,
            active_leases=1,
        )
    ]
    assert stats.snapshots == []
    assert stats.local_copies == []
    assert stats.snapshot_compliance == []


def test_get_pool_stats_does_not_mutate_worker_health(tmp_path):
    controller = _make_controller(tmp_path)
    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-ready')))
    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-dead')))
    controller.worker_inventory['worker-dead'].last_heartbeat = (
        time.time() - WORKER_DEAD_THRESHOLD_SECS)

    stats = asyncio.run(controller.get_pool_stats.coroutine())

    assert stats.num_nodes == 2
    assert stats.num_healthy_nodes == 2
    assert [worker.health for worker in stats.durable_workers] == [
        WorkerStatus.READY,
        WorkerStatus.READY,
    ]
    assert controller.worker_inventory[
        'worker-dead'].status == WorkerStatus.READY


def test_get_pool_stats_counts_active_leases(tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    asyncio.run(controller.report_status.coroutine(_report()))
    asyncio.run(controller.place_snapshot.coroutine(_request()))

    stats = asyncio.run(controller.get_pool_stats.coroutine(PoolStatsFilter()))

    assert stats.num_nodes == 1
    assert stats.num_healthy_nodes == 1
    assert stats.active_leases == 1
    assert stats.dataplane_backups == 0


def test_get_pool_stats_returns_snapshot_debug_when_requested(tmp_path):
    controller = _make_controller(tmp_path)
    controller.state.snapshots['snap-1'] = SnapshotPlacement(
        worker='worker-a',
        state=SnapshotState.READY,
        size_bytes=2048,
    )
    controller.state.local_copies['snap-1:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-1',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.WARM,
        size_bytes=2048,
    ))
    controller.state.snapshot_compliance['snap-1'] = (SnapshotComplianceRecord(
        snapshot_id='snap-1',
        active_consumer_count=1,
    ))

    stats = asyncio.run(
        controller.get_pool_stats.coroutine(
            PoolStatsFilter(include_snapshots=True)))

    assert stats.snapshots == [
        SnapshotPlacement(
            worker='worker-a',
            state=SnapshotState.READY,
            size_bytes=2048,
        )
    ]
    assert stats.local_copies == [
        SnapshotLocalCopyRecord(
            snapshot_id='snap-1',
            pool_worker_uid='worker-a',
            local_state=SnapshotLocalCopyState.WARM,
            size_bytes=2048,
        )
    ]
    assert stats.snapshot_compliance == [
        SnapshotComplianceRecord(
            snapshot_id='snap-1',
            active_consumer_count=1,
        )
    ]


def test_get_pool_stats_filters_by_pool_worker_uid(tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(tmp_path, strategy=strategy)
    controller.state.snapshots['snap-on-a'] = SnapshotPlacement(
        worker='worker-a', state=SnapshotState.READY)
    controller.state.snapshots['snap-on-b'] = SnapshotPlacement(
        worker='worker-b', state=SnapshotState.READY)
    controller.state.dataplane_backups['snap-on-a'] = DataplaneBackup(
        snapshot_id='snap-on-a',
        dataplane_path='s3://bucket/snap-on-a',
    )
    controller.state.dataplane_backups['snap-on-b'] = DataplaneBackup(
        snapshot_id='snap-on-b',
        dataplane_path='s3://bucket/snap-on-b',
    )
    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-a', ready_snapshots=[])))
    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-b', ready_snapshots=[])))
    asyncio.run(controller.place_snapshot.coroutine(_request()))

    stats = asyncio.run(
        controller.get_pool_stats.coroutine(
            PoolStatsFilter(
                pool_worker_uids=['worker-b'],
                include_snapshots=True,
                include_leases=True,
                include_placements=True,
            )))

    assert stats.num_nodes == 1
    assert stats.num_snapshots == 1
    assert stats.dataplane_backups == 1
    assert stats.active_leases == 0
    assert stats.durable_workers == [
        PoolWorkerStats(
            pool_worker_uid='worker-b',
            health=WorkerStatus.READY,
            pvc_free_bytes=200_000,
            pvc_total_bytes=1_000_000,
            gpu_slots_total=1,
            gpu_slots_in_use=0,
            active_leases=0,
            snapshot_ids=['snap-on-b'],
        )
    ]
    assert stats.snapshots == [
        SnapshotPlacement(
            worker='worker-b',
            state=SnapshotState.READY,
        )
    ]
    assert stats.local_copies == []
    assert stats.snapshot_compliance == []


def test_get_pool_stats_filter_scopes_local_copy_debug_rows(tmp_path):
    controller = _make_controller(tmp_path)
    controller.state.local_copies['snap-a:worker-a'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-a',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.WARM,
    ))
    controller.state.local_copies['snap-b:worker-b'] = (SnapshotLocalCopyRecord(
        snapshot_id='snap-b',
        pool_worker_uid='worker-b',
        local_state=SnapshotLocalCopyState.EVICTION_CANDIDATE,
    ))
    controller.state.snapshot_compliance['snap-a'] = (SnapshotComplianceRecord(
        snapshot_id='snap-a'))
    controller.state.snapshot_compliance['snap-b'] = (SnapshotComplianceRecord(
        snapshot_id='snap-b'))

    stats = asyncio.run(
        controller.get_pool_stats.coroutine(
            PoolStatsFilter(
                pool_worker_uids=['worker-b'],
                include_snapshots=True,
            )))

    assert stats.snapshots == []
    assert stats.local_copies == [
        SnapshotLocalCopyRecord(
            snapshot_id='snap-b',
            pool_worker_uid='worker-b',
            local_state=SnapshotLocalCopyState.EVICTION_CANDIDATE,
        )
    ]
    assert stats.snapshot_compliance == [
        SnapshotComplianceRecord(snapshot_id='snap-b')
    ]


def test_report_status_does_not_persist_worker_inventory(tmp_path):
    """Node inventory is intentionally not persisted — a controller
    restart rebuilds it from status reports. Verify ``pool_state.json``
    stays empty after a report_status.
    """
    import os
    controller = _make_controller(tmp_path)
    asyncio.run(controller.report_status.coroutine(_report(ready_snapshots=[])))
    assert not os.path.exists(str(tmp_path / 'pool_state.json'))


def test_mutate_and_persist_helper(tmp_path):
    """The internal helper used by Block B handlers acquires the
    per-snapshot lock, applies the mutate_fn, and persists.
    """
    controller = _make_controller(tmp_path)

    def mutate(state: PoolState) -> None:
        state.dataplane_backups['snap-1'] = DataplaneBackup(
            snapshot_id='snap-1', dataplane_path='s3://b/snap-1')

    asyncio.run(controller._mutate_and_persist('snap-1', mutate))
    assert 'snap-1' in controller.state.dataplane_backups
    # Persisted to disk too — a subsequent load should see the mutation.
    loaded = asyncio.run(controller._persistence.load())
    assert 'snap-1' in loaded.dataplane_backups


def _gc_controller(tmp_path, grace_secs=900):
    clock = _FakeClock()
    controller = _make_controller(tmp_path, now_utc_fn=clock.now_utc,
                                  time_fn=clock.time)
    controller._tombstone_gc_grace_secs = grace_secs
    return controller, clock


def test_gc_prunes_evicted_local_copy_past_grace(tmp_path):
    controller, clock = _gc_controller(tmp_path, grace_secs=900)
    controller.state.local_copies[
        'snap-old:worker-a'] = SnapshotLocalCopyRecord(
            snapshot_id='snap-old',
            pool_worker_uid='worker-a',
            local_state=SnapshotLocalCopyState.EVICTED,
            evicted_at=clock.now_utc() - timedelta(seconds=901),
        )
    assert controller._gc_tombstone_records(clock.now_utc()) is True
    assert 'snap-old:worker-a' not in controller.state.local_copies


def test_gc_retains_evicted_local_copy_within_grace(tmp_path):
    controller, clock = _gc_controller(tmp_path, grace_secs=900)
    controller.state.local_copies['snap-recent:worker-a'] = (
        SnapshotLocalCopyRecord(
            snapshot_id='snap-recent',
            pool_worker_uid='worker-a',
            local_state=SnapshotLocalCopyState.EVICTED,
            evicted_at=clock.now_utc() - timedelta(seconds=300),
        ))
    assert controller._gc_tombstone_records(clock.now_utc()) is False
    assert 'snap-recent:worker-a' in controller.state.local_copies


def test_gc_retains_active_local_copy(tmp_path):
    controller, clock = _gc_controller(tmp_path, grace_secs=900)
    # WARM copy with no evicted_at is never a tombstone, regardless of age.
    controller.state.local_copies[
        'snap-warm:worker-a'] = SnapshotLocalCopyRecord(
            snapshot_id='snap-warm',
            pool_worker_uid='worker-a',
            local_state=SnapshotLocalCopyState.WARM,
        )
    assert controller._gc_tombstone_records(clock.now_utc()) is False
    assert 'snap-warm:worker-a' in controller.state.local_copies


def test_gc_prunes_deleted_compliance_past_grace(tmp_path):
    controller, clock = _gc_controller(tmp_path, grace_secs=900)
    controller.state.snapshot_compliance['snap-x'] = SnapshotComplianceRecord(
        snapshot_id='snap-x',
        deleted_at=clock.now_utc() - timedelta(seconds=901),
    )
    assert controller._gc_tombstone_records(clock.now_utc()) is True
    assert 'snap-x' not in controller.state.snapshot_compliance


def test_gc_retains_deleted_compliance_with_live_local_copy(tmp_path):
    # Snapshot re-requested after deletion → a live local copy exists; the
    # tombstone GC must not drop the compliance record out from under it.
    controller, clock = _gc_controller(tmp_path, grace_secs=900)
    controller.state.snapshot_compliance['snap-x'] = SnapshotComplianceRecord(
        snapshot_id='snap-x',
        deleted_at=clock.now_utc() - timedelta(seconds=901),
    )
    controller.state.local_copies['snap-x:worker-a'] = SnapshotLocalCopyRecord(
        snapshot_id='snap-x',
        pool_worker_uid='worker-a',
        local_state=SnapshotLocalCopyState.WARM,
    )
    assert controller._gc_tombstone_records(clock.now_utc()) is False
    assert 'snap-x' in controller.state.snapshot_compliance


def test_lifecycle_sweep_runs_tombstone_gc(tmp_path):
    controller, clock = _gc_controller(tmp_path, grace_secs=900)
    controller.state.local_copies[
        'snap-old:worker-a'] = SnapshotLocalCopyRecord(
            snapshot_id='snap-old',
            pool_worker_uid='worker-a',
            local_state=SnapshotLocalCopyState.EVICTED,
            evicted_at=clock.now_utc() - timedelta(seconds=901),
        )
    asyncio.run(controller._lifecycle_sweeps_once())
    assert 'snap-old:worker-a' not in controller.state.local_copies


def test_gc_backfills_evicted_local_copy_without_evicted_at(tmp_path):
    # Legacy/defensive: an EVICTED record with no evicted_at (e.g. a state file
    # from before the field was stamped) must not leak — GC backfills evicted_at
    # so it ages out, rather than skipping it forever.
    controller, clock = _gc_controller(tmp_path, grace_secs=900)
    controller.state.local_copies['snap-legacy:w'] = SnapshotLocalCopyRecord(
        snapshot_id='snap-legacy',
        pool_worker_uid='w',
        local_state=SnapshotLocalCopyState.EVICTED,
    )  # evicted_at defaults to None

    # First pass backfills evicted_at (does not delete yet).
    assert controller._gc_tombstone_records(clock.now_utc()) is True
    assert (controller.state.local_copies['snap-legacy:w'].evicted_at ==
            clock.now_utc())

    # Once the grace window elapses from the backfilled stamp, it's GC'd.
    clock.advance(901)
    assert controller._gc_tombstone_records(clock.now_utc()) is True
    assert 'snap-legacy:w' not in controller.state.local_copies


# ---------------------------------------------------------------------------
# SSD pressure bonds to the configured logical cap (ssd_max_utilization),
# not the raw physical disk.
# ---------------------------------------------------------------------------


def test_ssd_pressure_bonds_to_configured_max_utilization(tmp_path):
    # ssd_max_utilization=0.85 ⇒ a worker is "under pressure" once free drops
    # below 15% of physical total.
    controller = _make_controller(
        tmp_path,
        lifecycle_policy=SnapshotLifecyclePolicy(ssd_max_utilization=0.85))

    # 20% free (> 15% floor) → not under pressure.
    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-a', pvc_total_bytes=1_000_000,
                    pvc_free_bytes=200_000)))
    assert controller._worker_under_ssd_pressure('worker-a') is False

    # 12% free (< 15% floor) → under pressure. (Re-report overwrites inventory.)
    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-a', pvc_total_bytes=1_000_000,
                    pvc_free_bytes=120_000)))
    assert controller._worker_under_ssd_pressure('worker-a') is True


def test_ssd_pressure_threshold_is_configurable(tmp_path):
    # A laxer cap (0.95 ⇒ pressure only when free < 5%) keeps the same
    # 12%-free worker out of pressure — proving the knob drives the threshold
    # (under the old hardcoded 10%-free constant this worker was never
    # under pressure regardless).
    controller = _make_controller(
        tmp_path,
        lifecycle_policy=SnapshotLifecyclePolicy(ssd_max_utilization=0.95))
    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-a', pvc_total_bytes=1_000_000,
                    pvc_free_bytes=120_000)))
    assert controller._worker_under_ssd_pressure('worker-a') is False


def test_ssd_max_utilization_out_of_range_rejected(tmp_path):
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match='ssd_max_utilization'):
            _make_controller(
                tmp_path, lifecycle_policy=SnapshotLifecyclePolicy(
                    ssd_max_utilization=bad))


def test_reject_if_exceeds_disk_capacity_noop_when_capacity_unknown(tmp_path):
    # No worker has reported PVC capacity yet -> total is 0, so the guard must
    # not false-reject while the controller is still warming up.
    controller = _make_controller(tmp_path)
    controller._reject_if_exceeds_disk_capacity(10**15)  # no raise


def test_reject_if_exceeds_disk_capacity_boundary(tmp_path):
    controller = _make_controller(tmp_path)
    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-a', pvc_total_bytes=3_000_000)))
    # A 2,000,000-byte source needs exactly 3,000,000 bytes at the 1.5x floor.
    controller._reject_if_exceeds_disk_capacity(2_000_000)  # == limit: ok
    with pytest.raises(DiskgraphError, match='streaming DiskGraph ingest'):
        controller._reject_if_exceeds_disk_capacity(2_000_001)  # over: reject


def test_place_snapshot_fails_fast_when_estimate_exceeds_disk(tmp_path):
    # The singlehost path computes its own on-SSD estimate and never calls
    # ensure_diskgraph_space, so the disk-capacity guard must fire here, before
    # routing/ingest. The guard is single-host only (k8s has its own ceiling),
    # so this uses the singlehost strategy.
    controller = _make_controller(tmp_path,
                                  strategy=SinglehostPlacementStrategy())
    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-a', pvc_total_bytes=3_000_000)))
    # A source just over 2,000,000 bytes needs more than the host's 3,000,000
    # bytes at the 1.5x streaming floor.
    req = replace(
        _request(snapshot_id='snap-too-big', job_id='job-too-big'),
        source_size_bytes=2_000_001,
    )
    with pytest.raises(DiskgraphError, match='streaming DiskGraph ingest'):
        asyncio.run(controller.place_snapshot.coroutine(req))


def test_place_snapshot_reuses_ready_snapshot_despite_oversized_estimate(
        tmp_path):
    # A READY snapshot already sits on the singlehost disk (WARM reuse — no
    # new disk or dataplane demand), so capacity admission must not reject it.
    # Matches the PlacementRequest contract: WARM/COLD
    # use the persisted SnapshotPlacement.size_bytes, not the estimate.
    controller = _make_controller(
        tmp_path,
        strategy=SinglehostPlacementStrategy(),
        data_plane_storage_root='/local/dataplane',
    )
    asyncio.run(
        controller.report_status.coroutine(
            _report(
                pool_worker_uid=SINGLEHOST_WORKER_UID,
                pvc_total_bytes=3_000_000,
                ready_snapshots=[],
                dataplane_total_bytes=10_000,
                dataplane_used_bytes=10_000,
                dataplane_same_filesystem_as_ssd=False,
            )))
    controller.state.snapshots['snap-warm'] = SnapshotPlacement(
        worker=SINGLEHOST_WORKER_UID,
        state=SnapshotState.READY,
        size_bytes=4096,
    )
    controller.state.local_copies[f'snap-warm:{SINGLEHOST_WORKER_UID}'] = (
        SnapshotLocalCopyRecord(
            snapshot_id='snap-warm',
            pool_worker_uid=SINGLEHOST_WORKER_UID,
            local_state=SnapshotLocalCopyState.WARM,
        ))
    # Same over-limit numbers as above: estimate 1_200_000 > limit 1_000_000.
    req = replace(
        _request(snapshot_id='snap-warm', job_id='job-new'),
        source_size_bytes=300_000,
    )

    result = asyncio.run(controller.place_snapshot.coroutine(req))

    assert result.status == PlacementStatus.WARM
    assert result.pool_worker_uid == SINGLEHOST_WORKER_UID


def test_place_snapshot_restores_cold_snapshot_despite_oversized_estimate(
        tmp_path, monkeypatch):
    # COLD: the local copy was evicted (placement record removed) but an L2
    # backup exists. Restore demand is the snapshot's actual size — it fit
    # before — so fresh-ingest disk and dataplane admission must not reject it.
    strategy = SinglehostPlacementStrategy()
    original_route = strategy.route

    async def route_with_exact_size(*args, **kwargs):
        return replace(
            await original_route(*args, **kwargs),
            snapshot_size_estimate_bytes=4096,
        )

    monkeypatch.setattr(strategy, 'route', route_with_exact_size)
    controller = _make_controller(
        tmp_path,
        strategy=strategy,
        data_plane_storage_root='/local/dataplane',
    )
    asyncio.run(
        controller.report_status.coroutine(
            _report(
                pool_worker_uid=SINGLEHOST_WORKER_UID,
                pvc_total_bytes=3_000_000,
                ready_snapshots=[],
                dataplane_total_bytes=10_000,
                dataplane_used_bytes=10_000,
                dataplane_same_filesystem_as_ssd=False,
            )))
    controller.state.dataplane_backups['snap-cold'] = DataplaneBackup(
        snapshot_id='snap-cold', dataplane_path='s3://b/snap-cold',
        size_bytes=4096)
    req = replace(
        _request(snapshot_id='snap-cold', job_id='job-cold'),
        source_size_bytes=300_000,
    )

    result = asyncio.run(controller.place_snapshot.coroutine(req))

    assert result.status == PlacementStatus.COLD
    assert result.dataplane_path == 's3://b/snap-cold'
    assert controller.state.snapshots['snap-cold'].size_bytes == 4096


def test_place_snapshot_fails_fast_when_placement_exists_but_not_ready(
        tmp_path):
    # A prior placement that never reached READY is still a fresh ingest and
    # remains subject to the fixed host's 1.5x minimum.
    controller = _make_controller(tmp_path,
                                  strategy=SinglehostPlacementStrategy())
    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid=SINGLEHOST_WORKER_UID,
                    pvc_total_bytes=3_000_000, ready_snapshots=[])))
    controller.state.snapshots['snap-retry'] = SnapshotPlacement(
        worker=SINGLEHOST_WORKER_UID,
        state=SnapshotState.INGESTING,
        size_bytes=800_000,
    )
    req = replace(
        _request(snapshot_id='snap-retry', job_id='job-retry'),
        source_size_bytes=2_000_001,
    )
    with pytest.raises(DiskgraphError, match='streaming DiskGraph ingest'):
        asyncio.run(controller.place_snapshot.coroutine(req))


def test_local_dataplane_admission_rejects_missing_snapshot_before_provision(
        tmp_path):
    controller = _make_controller(
        tmp_path,
        strategy=SinglehostPlacementStrategy(),
        data_plane_storage_root='/local/dataplane',
    )
    asyncio.run(
        controller.report_status.coroutine(
            _report(
                pool_worker_uid=SINGLEHOST_WORKER_UID,
                ready_snapshots=[],
                dataplane_total_bytes=10_000,
                dataplane_used_bytes=9_701,
                dataplane_same_filesystem_as_ssd=False,
            )))
    # 100 source bytes * default 4x = 400 bytes.
    request = replace(
        _request(snapshot_id='snap-dataplane-full', job_id='job-dataplane'),
        source_size_bytes=100,
    )

    with pytest.raises(DiskgraphError) as exc_info:
        asyncio.run(controller.place_snapshot.coroutine(request))

    assert type(
        exc_info.value).__name__ == ('InsufficientDataplaneCapacityError')
    assert exc_info.value.non_retryable is True
    assert '/local/dataplane' in str(exc_info.value)
    assert '400 bytes' in str(exc_info.value)
    assert '299 bytes' in str(exc_info.value)
    assert 'snap-dataplane-full' not in controller.state.snapshots


def test_local_dataplane_admission_counts_active_missing_snapshots_once(
        tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(
        tmp_path,
        strategy=strategy,
        data_plane_storage_root='/local/dataplane',
    )
    asyncio.run(
        controller.report_status.coroutine(
            _report(
                pool_worker_uid='worker-a',
                ready_snapshots=[],
                dataplane_total_bytes=10_000,
                dataplane_used_bytes=9_201,
                dataplane_same_filesystem_as_ssd=False,
            )))

    first = replace(
        _request(snapshot_id='snap-first', job_id='job-first'),
        source_size_bytes=100,
    )
    second = replace(
        _request(snapshot_id='snap-second', job_id='job-second'),
        source_size_bytes=100,
    )

    asyncio.run(controller.place_snapshot.coroutine(first))
    assert controller.state.snapshots['snap-first'].size_bytes == 400

    with pytest.raises(DiskgraphError) as exc_info:
        asyncio.run(controller.place_snapshot.coroutine(second))
    assert type(
        exc_info.value).__name__ == ('InsufficientDataplaneCapacityError')
    assert len(strategy.provision_calls) == 1

    controller.state.snapshots['snap-first'].leases.clear()
    controller.state.snapshots['snap-other-worker'] = SnapshotPlacement(
        worker='worker-b', leases=['job-other'], size_bytes=10_000)
    result = asyncio.run(controller.place_snapshot.coroutine(second))
    assert result.status == PlacementStatus.MISSING
    assert len(strategy.provision_calls) == 2


class _ReplacementWorkerStrategy(MockPlacementStrategy):
    def __init__(self, route_estimate_bytes: int = 0) -> None:
        super().__init__()
        self.route_estimate_bytes = route_estimate_bytes

    async def route(self, request: PlacementRequest,
                    warm_location_uid: Optional[str],
                    snapshot_size_estimate_bytes: int = 0, *,
                    home_worker_uid: Optional[str] = None,
                    home_gpu_in_use: int = 0,
                    overflow_viable: bool = False) -> RouteDecision:
        del snapshot_size_estimate_bytes, home_gpu_in_use, overflow_viable
        self.route_calls.append((request, warm_location_uid))
        self.route_home_calls.append((home_worker_uid, 0, False))
        return RouteDecision(
            path='durable',
            pool_worker_uid='worker-b',
            provisioned_new_worker=False,
            reason='replace non-routable home',
            snapshot_size_estimate_bytes=self.route_estimate_bytes,
        )


async def _cross_worker_reroute_controller(
        tmp_path, strategy: MockPlacementStrategy,
        replacement_used_bytes: int) -> PoolController:
    controller = _make_controller(
        tmp_path,
        strategy=strategy,
        data_plane_storage_root='/local/dataplane',
    )
    for worker, used_bytes in (('worker-a', 9_000), ('worker-b',
                                                     replacement_used_bytes)):
        await controller.report_status.coroutine(
            _report(
                pool_worker_uid=worker,
                ready_snapshots=[],
                dataplane_total_bytes=10_000,
                dataplane_used_bytes=used_bytes,
                dataplane_same_filesystem_as_ssd=False,
            ))
    controller.worker_inventory['worker-a'].status = WorkerStatus.DEAD
    controller.state.snapshots['snap-reroute'] = SnapshotPlacement(
        worker='worker-a',
        leases=['job-existing'],
        size_bytes=400,
    )
    return controller


def test_local_dataplane_admission_checks_leased_cross_worker_reroute(tmp_path):
    strategy = _ReplacementWorkerStrategy()
    controller = asyncio.run(
        _cross_worker_reroute_controller(tmp_path, strategy, 9_601))

    request = replace(
        _request(snapshot_id='snap-reroute', job_id='job-new'),
        source_size_bytes=100,
    )

    with pytest.raises(DiskgraphError) as exc_info:
        asyncio.run(controller.place_snapshot.coroutine(request))

    assert type(
        exc_info.value).__name__ == ('InsufficientDataplaneCapacityError')
    assert strategy.route_home_calls == [('worker-a', 0, False)]
    assert strategy.provision_calls == []


def test_local_dataplane_admission_checks_larger_same_worker_retry(tmp_path):
    strategy = MockPlacementStrategy()
    controller = _make_controller(
        tmp_path,
        strategy=strategy,
        data_plane_storage_root='/local/dataplane',
    )
    asyncio.run(
        controller.report_status.coroutine(
            _report(
                pool_worker_uid='worker-a',
                ready_snapshots=[],
                dataplane_total_bytes=10_000,
                dataplane_used_bytes=9_551,
                dataplane_same_filesystem_as_ssd=False,
            )))
    controller.state.snapshots['snap-retry'] = SnapshotPlacement(
        worker='worker-a',
        leases=['job-retry'],
        lease_paths={'job-retry': 'durable'},
        size_bytes=400,
    )
    request = replace(
        _request(snapshot_id='snap-retry', job_id='job-retry'),
        source_size_bytes=150,
    )

    with pytest.raises(DiskgraphError) as exc_info:
        asyncio.run(controller.place_snapshot.coroutine(request))

    assert type(
        exc_info.value).__name__ == ('InsufficientDataplaneCapacityError')
    assert controller.state.snapshots['snap-retry'].size_bytes == 400
    assert strategy.route_calls == []
    assert strategy.provision_calls == []


def test_local_dataplane_admission_reserves_cross_worker_reroute_before_provision(
        tmp_path):
    class BlockingReplacementWorkerStrategy(_ReplacementWorkerStrategy):
        def __init__(self) -> None:
            super().__init__(route_estimate_bytes=400)
            self.provision_started = asyncio.Event()
            self.allow_provision = asyncio.Event()

        async def provision(self, request: PlacementRequest,
                            decision: RouteDecision) -> ProvisionedHandle:
            self.provision_calls.append((request, decision))
            if request.snapshot_id == 'snap-reroute':
                self.provision_started.set()
                await self.allow_provision.wait()
            return ProvisionedHandle(
                kind=decision.path,
                pool_worker_uid=decision.pool_worker_uid,
                task_queue=f'diskgraph-activity-tq-{request.job_id}',
            )

    async def drive() -> None:
        strategy = BlockingReplacementWorkerStrategy()
        controller = await _cross_worker_reroute_controller(
            tmp_path, strategy, 8_501)
        reroute = replace(
            _request(snapshot_id='snap-reroute', job_id='job-reroute'),
            source_size_bytes=200,
        )
        second = replace(
            _request(snapshot_id='snap-second', job_id='job-second'),
            source_size_bytes=300,
        )

        first_task = asyncio.create_task(
            controller.place_snapshot.coroutine(reroute))
        await strategy.provision_started.wait()
        try:
            with pytest.raises(DiskgraphError) as exc_info:
                await controller.place_snapshot.coroutine(second)
            assert type(exc_info.value).__name__ == (
                'InsufficientDataplaneCapacityError')
            assert controller.state.snapshots['snap-reroute'].worker == (
                'worker-b')
            assert controller.state.snapshots['snap-reroute'].size_bytes == 800
            loaded = await controller._persistence.load()
            assert loaded.snapshots['snap-reroute'].worker == 'worker-b'
            assert loaded.snapshots['snap-reroute'].size_bytes == 800
            assert len(strategy.route_calls) == 2
            assert len(strategy.provision_calls) == 1
        finally:
            strategy.allow_provision.set()
            _ = await first_task

    asyncio.run(drive())


@pytest.mark.parametrize(('used_bytes', 'admitted'), [
    (9_201, False),
    (9_200, True),
])
def test_local_dataplane_admission_combines_shared_filesystem_demand(
        tmp_path, used_bytes, admitted):
    strategy = MockPlacementStrategy()
    controller = _make_controller(
        tmp_path,
        strategy=strategy,
        data_plane_storage_root='/local/dataplane',
    )
    asyncio.run(
        controller.report_status.coroutine(
            _report(
                pool_worker_uid='worker-a',
                ready_snapshots=[],
                pvc_total_bytes=10_000,
                pvc_free_bytes=10_000 - used_bytes,
                dataplane_total_bytes=10_000,
                dataplane_used_bytes=used_bytes,
                dataplane_same_filesystem_as_ssd=True,
            )))
    request = replace(
        _request(snapshot_id='snap-shared-fs', job_id='job-shared-fs'),
        source_size_bytes=100,
    )

    if admitted:
        result = asyncio.run(controller.place_snapshot.coroutine(request))
        assert result.status == PlacementStatus.MISSING
        assert len(strategy.provision_calls) == 1
    else:
        with pytest.raises(DiskgraphError) as exc_info:
            asyncio.run(controller.place_snapshot.coroutine(request))
        assert type(
            exc_info.value).__name__ == ('InsufficientDataplaneCapacityError')
        assert strategy.provision_calls == []


@pytest.mark.parametrize(
    ('data_plane_storage_root', 'snapshot_overrides', 'source_size_bytes'),
    [
        ('s3://bucket/dataplane', {
            'dataplane_total_bytes': 10_000,
            'dataplane_used_bytes': 10_000,
            'dataplane_same_filesystem_as_ssd': False,
        }, 100),
        ('/local/dataplane', {
            'dataplane_total_bytes': 0,
            'dataplane_used_bytes': 0,
            'dataplane_same_filesystem_as_ssd': None,
        }, 100),
        ('/local/dataplane', {
            'dataplane_total_bytes': 10_000,
            'dataplane_used_bytes': 10_000,
            'dataplane_same_filesystem_as_ssd': False,
        }, 0),
    ],
)
def test_local_dataplane_admission_bypasses_object_or_unknown_inputs(
        tmp_path, data_plane_storage_root, snapshot_overrides,
        source_size_bytes):
    strategy = MockPlacementStrategy()
    controller = _make_controller(
        tmp_path,
        strategy=strategy,
        data_plane_storage_root=data_plane_storage_root,
    )
    asyncio.run(
        controller.report_status.coroutine(
            _report(pool_worker_uid='worker-a', ready_snapshots=[],
                    **snapshot_overrides)))
    request = replace(
        _request(snapshot_id='snap-bypass', job_id='job-dataplane'),
        source_size_bytes=source_size_bytes,
    )

    result = asyncio.run(controller.place_snapshot.coroutine(request))

    assert result.status == PlacementStatus.MISSING
    assert len(strategy.provision_calls) == 1
