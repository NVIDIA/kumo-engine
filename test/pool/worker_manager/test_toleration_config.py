# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configured tolerations reach real pod builders; only K8s API I/O is mocked."""

import asyncio
import copy
from functools import partial
from test.pool.worker_manager.test_init_canary import _bound_pvc, _running_pod
from test.pool.worker_manager.test_k8s import _current_registered_cache_worker
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from kubernetes import client
from kubernetes.client.exceptions import ApiException
from pydantic import TypeAdapter

import pool.worker_manager.init_canary as canary
from pool.dataclasses import (
    K8sPlacementState,
    PoolState,
    PvcState,
    SnapshotPlacement,
)
from pool.k8s_strategy import CapacityTier, K8sPlacementStrategy
from pool.service import _build_configured_k8s_manager
from pool.worker_manager.k8s import K8sPoolWorkerManager
from pool.worker_manager.types import (
    PoolWorkerNotReadyError,
    PvcSpec,
    ResourceSpec,
)
from service_framework._raw_deploy_config import DiskgraphConfig
from service_framework.deploy_env_config import DeployEnvConfig
from service_framework.deploy_stage import DeployStage

EXTRAS = [
    dict(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'),
    dict(key='dedicated', value='customer-pool', effect='NoSchedule'),
    dict(key='maintenance', operator='Exists', effect='NoExecute',
         tolerationSeconds=60),
]


def configured_manager(monkeypatch, pool_config):
    core = MagicMock(spec=client.CoreV1Api)
    apps = MagicMock(spec=client.AppsV1Api)
    batch = MagicMock(spec=client.BatchV1Api)
    monkeypatch.setenv('POOL_WORKER_TENANT', 'test')
    monkeypatch.setattr(
        'pool.worker_manager.K8sPoolWorkerManager',
        partial(K8sPoolWorkerManager, load_kube_config=False, core_v1_api=core,
                apps_v1_api=apps, batch_v1_api=batch,
                policy_v1_api=MagicMock(spec=client.PolicyV1Api)))
    pool = {'pool_diskgraph_node_types': {'test': {'ssd_gib': 800}}}
    pool.update(pool_config)
    diskgraph = TypeAdapter(DiskgraphConfig).validate_python({'k8s_pool': pool})
    config = TypeAdapter(DeployEnvConfig).validate_python(
        dict(deploy_mode=DeployStage.DEV, instance_id='toleration-test',
             data_compute=diskgraph))
    manager, _, _, _ = _build_configured_k8s_manager(config, diskgraph)
    return manager, core, apps, batch


def toleration_fields(tolerations):
    # Normalize API-defaulted empty fields, but preserve every meaningful field.
    return [(t.key or '', t.operator or 'Equal', t.value or '', t.effect
             or '', t.toleration_seconds) for t in tolerations]


def built_in(ephemeral):
    return ('dedicated', 'Equal',
            'diskgraph-eph-test' if ephemeral else 'diskgraph-test',
            'NoSchedule', None)


EXPECTED_EXTRAS = [
    ('nvidia.com/gpu', 'Exists', '', 'NoSchedule', None),
    ('dedicated', 'Equal', 'customer-pool', 'NoSchedule', None),
    ('maintenance', 'Exists', '', 'NoExecute', 60),
]


def changed_extras(change):
    extras = copy.deepcopy(EXTRAS)
    if change == 'reordered':
        extras.reverse()
    elif change == 'duplicate':
        extras.append(copy.deepcopy(EXTRAS[0]))
    elif change == 'added':
        extras.append(dict(key='new', operator='Exists'))
    elif change == 'removed':
        extras.pop(0)
    elif change == 'removed-all':
        extras.clear()
    elif change == 'operator':
        extras[0]['operator'] = 'Equal'
    elif change == 'effect':
        extras[0]['effect'] = 'NoExecute'
    elif change == 'seconds':
        extras[2]['tolerationSeconds'] = 120
    return extras


@pytest.mark.parametrize('pool_config', [
    {},
    dict(pool_extra_tolerations=None),
    dict(pool_extra_tolerations=[]),
    dict(pool_extra_tolerations=EXTRAS),
])
@pytest.mark.parametrize(
    'kind',
    ['daemon', 'cache-activity', 'eph-activity', 'cache-canary', 'eph-canary'])
def test_configured_extras_reach_every_pool_pod(monkeypatch, pool_config, kind):
    manager, _, apps, batch = configured_manager(monkeypatch, pool_config)
    if kind == 'daemon':
        manager._create_daemon_deployment_sync('daemon', 'pvc', 'uid', None)
        spec = apps.create_namespaced_deployment.call_args.kwargs[
            'body'].spec.template.spec
    elif kind.endswith('activity'):
        kwargs: dict[str, Any] = dict(job_name='activity', job_id='job',
                                      snapshot_id='snap', activity='train',
                                      image='test/image:tag', command=None,
                                      args=None, env=None,
                                      resources=ResourceSpec(), backoff_limit=0)
        if kind == 'eph-activity':
            manager._create_eph_activity_job_sync(**kwargs,
                                                  ephemeral_storage_bytes=4096)
        else:
            manager._create_activity_job_sync(**kwargs, pool_worker_uid='uid',
                                              pvc_name='pvc')
        spec = batch.create_namespaced_job.call_args.kwargs[
            'body'].spec.template.spec
    else:
        scheduling = (manager.eph_canary_scheduling() if kind == 'eph-canary'
                      else manager.cache_canary_scheduling(800))
        spec = canary._build_canary_pod(
            name='canary', namespace='test', labels={}, sched=scheduling,
            service_account='worker', image='test/image:tag',
            pvc_name=None if kind == 'eph-canary' else 'pvc').spec
    expected = [built_in(kind.startswith('eph-'))]
    if pool_config.get('pool_extra_tolerations'):
        expected += EXPECTED_EXTRAS
    assert spec is not None
    assert toleration_fields(spec.tolerations) == expected
    assert spec.node_selector is not None
    assert spec.node_selector['kumo.ai/diskgraph-pool-type'] == (
        'ephemeral' if kind.startswith('eph-') else 'cache')
    assert spec.node_name is None


def test_duplicate_extras_do_not_duplicate_builtin_toleration(monkeypatch):
    extras = [
        dict(key='dedicated', value='diskgraph-test', effect='NoSchedule'),
        EXTRAS[0],
        EXTRAS[0],
    ]
    manager, _, apps, _ = configured_manager(monkeypatch,
                                             {'pool_extra_tolerations': extras})
    manager._create_daemon_deployment_sync('daemon', 'pvc', 'uid', None)
    spec = apps.create_namespaced_deployment.call_args.kwargs[
        'body'].spec.template.spec
    assert toleration_fields(
        spec.tolerations) == [built_in(False), EXPECTED_EXTRAS[0]]


@pytest.mark.parametrize('change', [
    'unchanged', 'reordered', 'duplicate', 'added', 'removed', 'removed-all',
    'operator', 'effect', 'seconds', 'missing-on-pod'
])
def test_registered_worker_proof_requires_current_extras(monkeypatch, change):
    manager, _, apps, _ = configured_manager(monkeypatch,
                                             {'pool_extra_tolerations': EXTRAS})
    worker, pod, pvc = _current_registered_cache_worker(manager)
    manager._create_daemon_deployment_sync('daemon', 'pvc', 'uid', None)
    template = apps.create_namespaced_deployment.call_args.kwargs[
        'body'].spec.template
    assert pod.metadata is not None
    pod.metadata.annotations = template.metadata.annotations
    # Match a real API response: omitted fields have defaults and admission
    # adds the standard node-unreachable toleration.
    assert pod.spec is not None
    pod.spec.tolerations = [
        client.V1Toleration(key=key, operator=operator, value=value,
                            effect=effect, toleration_seconds=seconds) for key,
        operator, value, effect, seconds in [built_in(False), *EXPECTED_EXTRAS]
    ]
    pod.spec.tolerations.append(
        client.V1Toleration(key='node.kubernetes.io/unreachable',
                            operator='Exists', effect='NoExecute',
                            toleration_seconds=300))
    pod.spec.tolerations.reverse()
    if change == 'missing-on-pod':
        pod.spec.tolerations = [
            t for t in pod.spec.tolerations if t.key != 'maintenance'
        ]
    current, core, _, _ = configured_manager(
        monkeypatch, {'pool_extra_tolerations': changed_extras(change)})
    core.read_namespaced_pod.return_value = pod
    core.read_namespaced_persistent_volume_claim.return_value = pvc
    matches = asyncio.run(
        current.registered_worker_matches_current_cache_tier(
            worker, PvcSpec(size_gib=800, gpu_count=1)))
    assert matches == (change in ('unchanged', 'reordered', 'duplicate'))


@pytest.mark.parametrize('extras', [
    [],
    [dict(key='dedicated', value='diskgraph-test', effect='NoSchedule')],
    [
        dict(key='node.kubernetes.io/not-ready', operator='Exists',
             effect='NoSchedule')
    ],
])
def test_admission_tolerations_do_not_invalidate_registered_worker(
        monkeypatch, extras):
    manager, core, apps, _ = configured_manager(
        monkeypatch, {'pool_extra_tolerations': extras})
    worker, pod, pvc = _current_registered_cache_worker(manager)
    manager._create_daemon_deployment_sync('daemon', 'pvc', 'uid', None)
    template = apps.create_namespaced_deployment.call_args.kwargs[
        'body'].spec.template
    assert pod.metadata is not None
    pod.metadata.annotations = template.metadata.annotations
    if len(template.spec.tolerations) == 1:
        # Omitted/no-effective-extras keeps the pre-feature pod metadata.
        assert not template.metadata.annotations
    assert pod.spec is not None
    assert pod.spec.tolerations is not None
    pod.spec.tolerations.extend([
        client.V1Toleration(key='node.kubernetes.io/memory-pressure',
                            operator='Exists', effect='NoSchedule'),
        client.V1Toleration(key='customer.example/admission', operator='Exists',
                            effect='NoSchedule'),
        client.V1Toleration(key='node.kubernetes.io/not-ready',
                            operator='Exists', effect='NoExecute',
                            toleration_seconds=300),
    ])
    core.read_namespaced_pod.return_value = pod
    core.read_namespaced_persistent_volume_claim.return_value = pvc
    assert asyncio.run(
        manager.registered_worker_matches_current_cache_tier(
            worker, PvcSpec(size_gib=800, gpu_count=1)))


@pytest.mark.parametrize('change', [
    'unchanged', 'reordered', 'duplicate', 'removed', 'operator', 'effect',
    'seconds'
])
def test_canary_cache_is_invalidated_by_effective_extra_changes(
        monkeypatch, change):
    monkeypatch.setattr(canary, 'CANARY_ENABLED', True)
    monkeypatch.setattr(canary, 'EPH_CANARY_ENABLED', False)
    monkeypatch.setattr(canary, '_RECONCILE_WAIT_SECS', 0.001)
    monkeypatch.setattr(canary, '_CANARY_POLL_INTERVAL_SECS', 0.001)

    def run(extras, annotations):
        manager, core, apps, _ = configured_manager(
            monkeypatch, {'pool_extra_tolerations': extras})
        apps.read_namespaced_deployment.return_value = client.V1Deployment(
            metadata=client.V1ObjectMeta(annotations=annotations))
        core.list_namespaced_pod.return_value = client.V1PodList(items=[])
        core.list_namespaced_persistent_volume_claim.return_value = (
            client.V1PersistentVolumeClaimList(items=[]))
        core.read_namespaced_pod.return_value = _running_pod()
        core.read_namespaced_persistent_volume_claim.return_value = _bound_pvc()
        asyncio.run(
            canary.run_init_canary(
                manager, cache_tier_specs=[PvcSpec(size_gib=800, gpu_count=1)],
                pvc_spec_factory=lambda size: PvcSpec(size_gib=size)))
        return core, apps

    _, original_apps = run(EXTRAS, {})
    annotations = original_apps.patch_namespaced_deployment.call_args.kwargs[
        'body']['metadata']['annotations']
    core, _ = run(changed_extras(change), annotations)
    expected_creates = 0 if change in ('unchanged', 'reordered',
                                       'duplicate') else 1
    assert core.create_namespaced_pod.call_count == expected_creates


def daemon_pod(manager, apps):
    _, pod, _ = _current_registered_cache_worker(manager)
    manager._create_daemon_deployment_sync('daemon', 'pvc', 'uid', None)
    template = apps.create_namespaced_deployment.call_args.kwargs[
        'body'].spec.template
    assert pod.metadata is not None
    pod.metadata.annotations = template.metadata.annotations
    return pod


@pytest.mark.parametrize('tracked', [True, False])
@pytest.mark.parametrize('change', [
    'unchanged', 'reordered', 'duplicate', 'added', 'removed', 'removed-all',
    'operator', 'effect', 'seconds',
])
@pytest.mark.parametrize('current_incarnation', [False, True])
def test_changed_extras_retire_existing_daemon(
        monkeypatch, change, tracked, current_incarnation):
    old, _, apps, _ = configured_manager(
        monkeypatch, {'pool_extra_tolerations': EXTRAS})
    pod = daemon_pod(old, apps)
    current, core, current_apps, _ = configured_manager(
        monkeypatch, {'pool_extra_tolerations': changed_extras(change)})
    pods = [pod]
    if current_incarnation:
        pods.append(daemon_pod(current, current_apps))
    core.list_namespaced_pod.return_value = client.V1PodList(items=pods)
    strategy = K8sPlacementStrategy(
        current, lambda: {}, lambda _: None, lambda _: 0, lambda _: 0,
        capacity_tiers=[CapacityTier(800)], activity_poller_waiter=AsyncMock())
    state = PoolState(placement_state=K8sPlacementState(pvcs={
        'uid': PvcState('uid', tier_ssd_gib=800),
    } if tracked else {}))
    if tracked:
        state.snapshots['snap'] = SnapshotPlacement(worker='uid', leases=['job'])

    delta = asyncio.run(strategy.reconcile(state))

    expected = ({'uid'} if tracked and not current_incarnation and change not in
                ('unchanged', 'reordered', 'duplicate') else set())
    assert delta.incompatible_uids == expected
    assert delta.unexpected_uids == (set() if tracked else {'uid'})


@pytest.mark.parametrize('extras', [
    None, [], [dict(key='dedicated', value='diskgraph-test', effect='NoSchedule')]
])
def test_legacy_daemon_remains_usable_without_effective_extras(
        monkeypatch, extras):
    manager, core, apps, _ = configured_manager(
        monkeypatch, {'pool_extra_tolerations': extras})
    pod = daemon_pod(manager, apps)
    assert pod.metadata is not None and not pod.metadata.annotations
    core.list_namespaced_pod.return_value = client.V1PodList(items=[pod])

    info = asyncio.run(manager._wait_for_daemon_registered('uid', timeout=1))

    assert info.pool_worker_uid == 'uid'
    assert info.extra_tolerations_match


@pytest.mark.parametrize('operation', ['reclaim-daemon', 'create-activity'])
def test_stale_daemon_cannot_receive_work_before_cleanup(monkeypatch, operation):
    old, _, apps, _ = configured_manager(
        monkeypatch, {'pool_extra_tolerations': EXTRAS})
    pod = daemon_pod(old, apps)
    current, core, _, batch = configured_manager(monkeypatch, {})
    core.list_namespaced_pod.return_value = client.V1PodList(items=[pod])

    with pytest.raises(PoolWorkerNotReadyError, match='tolerations'):
        if operation == 'reclaim-daemon':
            asyncio.run(current._wait_for_daemon_registered('uid', timeout=1))
        else:
            asyncio.run(current.provision_activity_pod(
                pool_worker_uid='uid', job_id='job', snapshot_id='snap',
                activity='train', image='test/image:tag'))
    batch.create_namespaced_job.assert_not_called()


@pytest.mark.parametrize('change', [
    'unchanged', 'reordered', 'duplicate', 'added', 'removed', 'removed-all',
    'legacy', 'selector', 'gpu-count', 'service-account', 'pvc',
])
def test_surviving_canary_requires_current_spec(monkeypatch, change):
    old, _, _, _ = configured_manager(
        monkeypatch, {'pool_extra_tolerations': EXTRAS})
    current, core, _, _ = configured_manager(
        monkeypatch, {'pool_extra_tolerations': changed_extras(change)})

    def build(manager):
        sched = manager.cache_canary_scheduling(800)
        if manager is current and change == 'selector':
            sched.node_selector['customer.example/pool'] = 'new'
        return canary._build_canary_pod(
            name='canary', namespace='test', labels={}, sched=sched,
            service_account=('new' if manager is current and change
                             == 'service-account' else 'worker'),
            image='test/image:tag',
            pvc_name='new' if manager is current and change == 'pvc' else 'pvc',
            gpu_count=4 if manager is current and change == 'gpu-count' else 1)

    existing, requested = build(old), build(current)
    assert existing.spec is not None and existing.spec.tolerations is not None
    existing.spec.tolerations.reverse()
    existing.spec.tolerations.append(client.V1Toleration(
        key='node.kubernetes.io/unreachable', operator='Exists',
        effect='NoExecute', toleration_seconds=300))
    if change == 'legacy':
        assert existing.metadata is not None
        existing.metadata.annotations = {}
    core.create_namespaced_pod.side_effect = [ApiException(status=409),
                                              None]
    core.read_namespaced_pod.side_effect = [existing,
                                            ApiException(status=404)]

    asyncio.run(canary._CanaryRunner(current)._create_pod(requested))

    reusable = change in ('unchanged', 'reordered', 'duplicate')
    assert core.create_namespaced_pod.call_count == (1 if reusable else 2)
    assert core.delete_namespaced_pod.call_count == (0 if reusable else 1)
    if not reusable:
        assert core.create_namespaced_pod.call_args.kwargs['body'] is requested


def test_stale_canary_cleanup_failure_cannot_prove_capacity(monkeypatch):
    manager, core, _, _ = configured_manager(monkeypatch, {})
    requested = canary._build_canary_pod(
        name='canary', namespace='test', labels={},
        sched=manager.cache_canary_scheduling(800),
        service_account='worker', image='test/image:tag', pvc_name='pvc')
    core.create_namespaced_pod.side_effect = ApiException(status=409)
    core.read_namespaced_pod.return_value = _running_pod()
    core.delete_namespaced_pod.side_effect = ApiException(status=403)
    monkeypatch.setattr(canary, '_RECONCILE_WAIT_SECS', 0.001)
    monkeypatch.setattr(canary, '_CANARY_POLL_INTERVAL_SECS', 0.001)

    with pytest.raises(canary.InitCanaryValidationError):
        asyncio.run(canary._CanaryRunner(manager)._create_pod(requested))
    assert core.create_namespaced_pod.call_count == 1
