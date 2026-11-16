# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``pool.k8s_pod`` env-var helpers.

The (future) K8sPlacementStrategy uses these when constructing pod
manifests; the env-var keys + values must match what
:mod:`pool.worker_main` parses, which is what these tests pin.
"""
import pytest

from pool.k8s_pod import (
    _FORWARDED_CONNECTOR_IO_ENV_KEYS,
    _FORWARDED_DEPLOY_ENV_KEYS,
    activity_pod_env_vars,
    daemon_pod_env_vars,
    forwarded_deploy_env_vars,
    pool_worker_env_vars,
)
from pool.storage_mode import PoolStorageMode


@pytest.fixture(autouse=True)
def clean_forwarded_deploy_env(monkeypatch):
    for key in (*_FORWARDED_DEPLOY_ENV_KEYS, *_FORWARDED_CONNECTOR_IO_ENV_KEYS):
        monkeypatch.delenv(key, raising=False)


def test_pool_worker_env_vars_full_set():
    env = pool_worker_env_vars(
        pool_worker_uid='worker-a',
        task_queue='some-tq',
        storage_mode=PoolStorageMode.DAEMON_ON_PVC,
        gpu_concurrency=2,
        max_concurrent_activities=4,
    )
    assert env == {
        'POOL_WORKER_UID': 'worker-a',
        'POOL_TASK_QUEUE': 'some-tq',
        'POOL_STORAGE_MODE': 'daemon-on-pvc',
        'POOL_GPU_CONCURRENCY': '2',
        'POOL_MAX_CONCURRENT_ACTIVITIES': '4',
    }


def test_daemon_pod_env_vars_uses_daemon_queue():
    env = daemon_pod_env_vars(pool_worker_uid='worker-a')
    assert env == {
        'POOL_WORKER_UID': 'worker-a',
        'POOL_TASK_QUEUE': 'diskgraph-daemon-worker-a',
        'POOL_STORAGE_MODE': 'daemon-on-pvc',
        'POOL_GPU_CONCURRENCY': '1',
        'POOL_MAX_CONCURRENT_ACTIVITIES': '0',
    }
    assert 'POOL_HEARTBEAT_INTERVAL_SECS' not in env
    assert 'SSD_MOUNT_PATH' not in env
    # Daemon does not report authoritative GPU capacity; the controller owns
    # durable capacity from its selected tier, so no env var carries it.
    assert 'POOL_WORKER_GPU_SLOTS_TOTAL' not in env


def test_activity_pod_env_vars_uses_activity_queue():
    env = activity_pod_env_vars(pool_worker_uid='worker-a', job_id='job-xyz')
    assert env['POOL_TASK_QUEUE'] == 'diskgraph-activity-tq-job-xyz'
    assert env['POOL_STORAGE_MODE'] == 'activity-on-pvc'
    assert env['POOL_GPU_CONCURRENCY'] == '1'


def test_activity_pod_with_higher_gpu_concurrency():
    env = activity_pod_env_vars(pool_worker_uid='w-a', job_id='j-1',
                                gpu_concurrency=4)
    assert env['POOL_GPU_CONCURRENCY'] == '4'


def test_gpu_activity_pod_requires_and_selects_gpu():
    env = activity_pod_env_vars(
        pool_worker_uid='w-a',
        job_id='j-1',
        require_gpu=True,
    )
    assert env['KUMO_REQUIRE_GPU'] == 'true'
    assert env['KUMO_ACCELERATOR'] == 'gpu'

    cpu_env = activity_pod_env_vars(
        pool_worker_uid='w-a',
        job_id='j-2',
    )
    assert 'KUMO_REQUIRE_GPU' not in cpu_env
    assert 'KUMO_ACCELERATOR' not in cpu_env


def test_forwarded_deploy_env_vars_empty_when_unset():
    assert forwarded_deploy_env_vars() == {}


def test_forwarded_deploy_env_vars_inlines_config_file_to_json(
        tmp_path, monkeypatch):
    # Controller may set DEPLOY_ENV_CONFIG_FILE (a host-local path
    # that's not valid inside the pod) instead of forwarding
    # DEPLOY_ENV_CONFIG_JSON. Inline the file content so the pod's
    # deploy_env() resolves to the same structured config.
    config = '{"orchestration": {"type": "cloud"}}'
    config_file = tmp_path / 'deploy_env.json'
    config_file.write_text(config)
    monkeypatch.setenv('DEPLOY_ENV_CONFIG_FILE', str(config_file))
    monkeypatch.setenv('DEPLOY_STAGE', 'dev')

    env = forwarded_deploy_env_vars()

    assert env['DEPLOY_ENV_CONFIG_JSON'] == config
    assert env['DEPLOY_STAGE'] == 'dev'
    # FILE itself isn't forwarded — only its expanded content.
    assert 'DEPLOY_ENV_CONFIG_FILE' not in env


def test_forwarded_deploy_env_vars_existing_json_takes_precedence_over_file(
        tmp_path, monkeypatch):
    monkeypatch.setenv('DEPLOY_ENV_CONFIG_JSON',
                       '{"orchestration": {"type": "local"}}')
    config_file = tmp_path / 'unused.json'
    config_file.write_text('{"orchestration": {"type": "cloud"}}')
    monkeypatch.setenv('DEPLOY_ENV_CONFIG_FILE', str(config_file))

    env = forwarded_deploy_env_vars()

    # JSON env directly set wins; file is ignored.
    assert env[
        'DEPLOY_ENV_CONFIG_JSON'] == '{"orchestration": {"type": "local"}}'


def test_daemon_pod_env_vars_forwards_deploy_env(monkeypatch):
    monkeypatch.setenv('DEPLOY_ENV_CONFIG_JSON', '{"orchestration": {}}')
    monkeypatch.setenv('DEPLOY_STAGE', 'dev')
    monkeypatch.setenv('INSTANCE_ID', 'antest-3a37fb')
    monkeypatch.setenv('AWS_REGION', 'us-west-2')
    monkeypatch.setenv('CUSTOM_S3_DATA_PLANE_BUCKET',
                       's3://example-bucket/preprod/test')

    env = daemon_pod_env_vars(pool_worker_uid='worker-a')

    assert env['DEPLOY_ENV_CONFIG_JSON'] == '{"orchestration": {}}'
    assert env['DEPLOY_STAGE'] == 'dev'
    assert env['INSTANCE_ID'] == 'antest-3a37fb'
    assert env['AWS_REGION'] == 'us-west-2'
    assert env['CUSTOM_S3_DATA_PLANE_BUCKET'] == (
        's3://example-bucket/preprod/test')
    assert env['POOL_WORKER_UID'] == 'worker-a'
    assert env['POOL_STORAGE_MODE'] == PoolStorageMode.DAEMON_ON_PVC.value


def test_activity_pod_env_vars_forwards_deploy_env(monkeypatch):
    monkeypatch.setenv('DEPLOY_ENV_CONFIG_JSON', '{"orchestration": {}}')
    monkeypatch.setenv('DEPLOY_STAGE', 'dev')
    monkeypatch.setenv('INSTANCE_ID', 'antest-3a37fb')
    monkeypatch.setenv('AWS_REGION', 'us-west-2')
    monkeypatch.setenv('CUSTOM_S3_DATA_PLANE_BUCKET',
                       's3://example-bucket/preprod/test')

    pvc_env = activity_pod_env_vars(
        pool_worker_uid='worker-a',
        job_id='job-1',
    )
    emptydir_env = activity_pod_env_vars(
        pool_worker_uid='worker-b',
        job_id='job-2',
        storage_mode=PoolStorageMode.ACTIVITY_ON_EMPTYDIR,
    )

    for env in (pvc_env, emptydir_env):
        assert env['DEPLOY_ENV_CONFIG_JSON'] == '{"orchestration": {}}'
        assert env['DEPLOY_STAGE'] == 'dev'
        assert env['INSTANCE_ID'] == 'antest-3a37fb'
        assert env['AWS_REGION'] == 'us-west-2'
        assert env['CUSTOM_S3_DATA_PLANE_BUCKET'] == (
            's3://example-bucket/preprod/test')

    assert pvc_env['POOL_STORAGE_MODE'] == (
        PoolStorageMode.ACTIVITY_ON_PVC.value)
    assert emptydir_env['POOL_STORAGE_MODE'] == (
        PoolStorageMode.ACTIVITY_ON_EMPTYDIR.value)


def test_pool_pods_forward_only_nonsecret_gcp_connector_io_controls(
        monkeypatch):
    monkeypatch.setenv('GCP_STORAGE_TRANSFER_BUCKET', 'transfer-bucket')
    monkeypatch.setenv('GOOGLE_CLOUD_PROJECT', 'staging-project')
    forbidden_credentials = {
        'GOOGLE_APPLICATION_CREDENTIALS': '/tmp/service-account.json',
        'GOOGLE_CREDENTIALS_STR_JSON': 'private-key-material',
    }
    for name, value in forbidden_credentials.items():
        monkeypatch.setenv(name, value)

    for env in (
            daemon_pod_env_vars(pool_worker_uid='worker-a'),
            activity_pod_env_vars(pool_worker_uid='worker-a', job_id='job-1'),
    ):
        assert env['GCP_STORAGE_TRANSFER_BUCKET'] == 'transfer-bucket'
        assert env['GOOGLE_CLOUD_PROJECT'] == 'staging-project'
        assert forbidden_credentials.keys().isdisjoint(env)
        assert all(value not in env.values()
                   for value in forbidden_credentials.values())
