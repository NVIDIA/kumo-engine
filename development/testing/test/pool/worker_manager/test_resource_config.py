# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real deploy-config/factory/pod-spec path; only Kubernetes API I/O is mocked."""

from functools import partial
from unittest.mock import MagicMock

import pytest
from kubernetes import client
from pydantic import TypeAdapter

from pool.dataclasses import PlacementRequest, ResourceHint, SnapshotActivity
from pool.k8s_strategy import K8sPlacementStrategy
from pool.service import PoolController, _configured_k8s_strategy_factory
from pool.worker_manager.k8s import K8sPoolWorkerManager
from service_framework._raw_deploy_config import DiskgraphConfig
from service_framework.deploy_env_config import DeployEnvConfig
from service_framework.deploy_stage import DeployStage

CONFIGURED = {
    'daemon': {
        'requests': {
            'cpu': '2'
        },
        'limits': {
            'memory': '16Gi'
        }
    },
    'activity_defaults': {
        'requests': {
            'cpu': '2',
            'memory': '8Gi'
        },
        'limits': {
            'cpu': '8',
            'memory': '32Gi'
        },
    },
    'training_overrides': {
        'requests': {
            'cpu': '4'
        },
        'limits': {
            'memory': '64Gi'
        }
    },
    'prediction_overrides': {
        'requests': {
            'memory': '16Gi'
        }
    },
}


def configured_strategy(monkeypatch, resources):
    apps = MagicMock(spec=client.AppsV1Api)
    batch = MagicMock(spec=client.BatchV1Api)
    monkeypatch.setattr(
        'pool.worker_manager.K8sPoolWorkerManager',
        partial(
            K8sPoolWorkerManager,
            load_kube_config=False,
            core_v1_api=MagicMock(spec=client.CoreV1Api),
            apps_v1_api=apps,
            batch_v1_api=batch,
            policy_v1_api=MagicMock(spec=client.PolicyV1Api),
        ))
    diskgraph = TypeAdapter(DiskgraphConfig).validate_python({
        'k8s_pool': {
            'pool_diskgraph_node_types': {
                'test': {
                    'ssd_gib': 800
                }
            },
            'pool_resources': resources,
        }
    })
    config = DeployEnvConfig(deploy_mode=DeployStage.DEV,
                             instance_id='resource-test',
                             data_compute=diskgraph)
    strategy = _configured_k8s_strategy_factory(config, diskgraph)(
        MagicMock(spec=PoolController))
    assert isinstance(strategy, K8sPlacementStrategy)
    return strategy, apps, batch


@pytest.mark.parametrize('configured', [False, True])
def test_configured_daemon_resources_reach_deployment(monkeypatch, configured):
    strategy, apps, _ = configured_strategy(monkeypatch,
                                            CONFIGURED if configured else {})
    strategy._manager._create_daemon_deployment_sync('daemon', 'pvc', 'uid',
                                                     None)
    pod = apps.create_namespaced_deployment.call_args.kwargs[
        'body'].spec.template.spec
    resources = pod.containers[0].resources
    assert resources.requests == {
        'cpu': '2' if configured else '1',
        'memory': '2Gi'
    }
    assert (resources.limits or {}) == ({
        'memory': '16Gi'
    } if configured else {})


@pytest.mark.parametrize('configured', [False, True])
@pytest.mark.parametrize('ephemeral', [False, True])
@pytest.mark.parametrize('activity', [
    activity for activity in SnapshotActivity
    if activity != SnapshotActivity.ENSURE_SNAPSHOT
])
def test_configured_activity_resources_reach_both_pod_types(
        monkeypatch, configured, ephemeral, activity):
    strategy, _, batch = configured_strategy(monkeypatch,
                                             CONFIGURED if configured else {})
    request = PlacementRequest(snapshot_id='snap', job_id='job',
                               workflow_id='wf', activity=activity,
                               resource_hint=ResourceHint())
    resources = strategy._activity_resources_factory(request)
    if ephemeral:
        strategy._manager._create_eph_activity_job_sync(
            job_name='activity', job_id='job', snapshot_id='snap',
            activity=activity.value, image='test/image:tag', command=None,
            args=None, env=None, resources=resources, backoff_limit=0,
            ephemeral_storage_bytes=4096)
    else:
        strategy._manager._create_activity_job_sync(
            job_name='activity', job_id='job', snapshot_id='snap',
            activity=activity.value, image='test/image:tag', command=None,
            args=None, env=None, resources=resources, backoff_limit=0,
            pool_worker_uid='uid', pvc_name='pvc')
    pod = batch.create_namespaced_job.call_args.kwargs[
        'body'].spec.template.spec
    actual = pod.containers[0].resources
    requests = {'cpu': '2', 'memory': '8Gi'} if configured else {}
    limits = {'cpu': '8', 'memory': '32Gi'} if configured else {}
    if activity in (SnapshotActivity.TRAIN, SnapshotActivity.PREDICT):
        requests['nvidia.com/gpu'] = limits['nvidia.com/gpu'] = '1'
        if configured and activity == SnapshotActivity.TRAIN:
            requests['cpu'], limits['memory'] = '4', '64Gi'
        if configured and activity == SnapshotActivity.PREDICT:
            requests['memory'] = '16Gi'
    if ephemeral:
        requests['ephemeral-storage'] = '4096'
    assert (actual.requests or {} if actual else {}) == requests
    assert (actual.limits or {} if actual else {}) == limits
    assert pod.node_name is None  # CPU/memory admission must go through scheduler.
    assert pod.containers[
        0].security_context.allow_privilege_escalation is False
