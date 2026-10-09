# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure isolation checks for concurrent devserver identities.

These exercise derivation only: no tmux, Docker, Temporal, or devserver is
started here.  The matching two-instance smoke test is recorded in the
multi-devserver validation document.
"""

import os

import pytest

from pool.task_queues import SINGLEHOST_DAEMON_TQ
from rest.devserver.cgroup import CgroupLimits, scope_argv
from rest.devserver.instance import DevserverInstance
from rest.devserver.main import (
    DEVSERVER_BASE_PORT,
    _devserver_runtime_env,
    generate_base_port,
)
from service_framework._raw_deploy_config import (
    DiskgraphConfig,
    FileStorageConfig,
)
from service_framework.deploy_env_config import (
    DeployEnvConfig,
    _resolve_data_compute_roots,
    _resolve_storage_roots,
)
from service_framework.deploy_stage import DeployStage
from tmux_tools import config


def test_legacy_instance_keeps_existing_resources(monkeypatch):
    monkeypatch.setenv('KUMO_HOME', '/tmp/kumo-home')
    instance = DevserverInstance(10000)

    assert instance.environment() == {}
    assert instance.data_root('/mnt/localssd') == '/mnt/localssd'
    assert instance.name == 'devserver'
    assert instance.docker_label == 'devserver_id=10000'
    assert instance.ui_node_modules_volume() is None
    assert instance.ports()['jaeger_ui'] == 16686
    assert instance.ports()['jaeger_otlp'] == 4317
    assert instance.ports()['prometheus'] == 9090


def test_two_instances_have_disjoint_ports_paths_and_names(monkeypatch):
    monkeypatch.setenv('KUMO_HOME', '/mnt/localssd/kumo-home')
    first = DevserverInstance(11000)
    second = DevserverInstance(12000)
    first_env = first.environment()
    second_env = second.environment()

    assert set(first.ports().values()).isdisjoint(second.ports().values())
    assert first_env['DB_STORAGE_ROOT'] != second_env['DB_STORAGE_ROOT']
    assert first_env['KUMO_HOME'] != second_env['KUMO_HOME']
    assert first.data_root('/mnt/localssd') != second.data_root('/mnt/localssd')
    assert first.docker_label != second.docker_label
    assert first.ui_node_modules_volume() != second.ui_node_modules_volume()
    assert first.name != second.name
    assert first.temporal_instance_id != second.temporal_instance_id
    assert (f'data-backend-task-queue-{first.temporal_instance_id}'
            != f'data-backend-task-queue-{second.temporal_instance_id}')


def test_ports_have_stable_offsets_and_no_module_global_state():
    first = DevserverInstance(11000)
    second = DevserverInstance(12000)

    assert first.ports()['rest_api'] == 11001
    assert first.ports()['public_rest_api'] == 11002
    assert first.ports()['jaeger_ui'] == 11014
    assert first.ports()['jaeger_otlp'] == 11015
    assert first.ports()['prometheus'] == 11016
    assert second.ports()['rest_api'] == 12001
    assert first.ports() == DevserverInstance(11000).ports()


def test_generate_base_port_precedence(monkeypatch):
    monkeypatch.setenv('PHASE1B_DEVSERVER_BASE_PORT', '12000')

    assert generate_base_port(base_port=13000) == 13000
    assert generate_base_port() == 12000

    monkeypatch.delenv('PHASE1B_DEVSERVER_BASE_PORT')
    assert generate_base_port() == DEVSERVER_BASE_PORT


def test_many_instances_have_no_resource_collisions(monkeypatch):
    monkeypatch.setenv('KUMO_HOME', '/mnt/localssd/kumo-home')
    instances = [DevserverInstance(port) for port in range(11000, 16000, 1000)]
    resources = []
    for instance in instances:
        env = instance.environment()
        resources.extend(f'port:{port}' for port in instance.ports().values())
        resources.extend([
            f'path:{env["DB_STORAGE_ROOT"]}',
            f'path:{env["KUMO_HOME"]}',
            f'path:{instance.data_root("/mnt/localssd")}',
            f'name:{instance.name}',
            f'name:{instance.docker_label}',
            f'name:ml-workflow-task-queue-{instance.temporal_instance_id}',
        ])
    assert len(resources) == len(set(resources))


def test_tmux_server_is_instance_specific():
    first = config.Driver(name='website', panels=[],
                          tmux_server_id=DevserverInstance(11000).name)
    second = config.Driver(name='website', panels=[],
                           tmux_server_id=DevserverInstance(12000).name)

    assert first.make_tmux_config().server_id == 'devserver-11000'
    assert second.make_tmux_config().server_id == 'devserver-12000'


def test_default_tmux_server_is_unchanged():
    driver = config.Driver(name='website', panels=[])

    assert driver.make_tmux_config().server_id == 'devserver'


def test_apply_environment_is_idempotent(monkeypatch):
    monkeypatch.setenv('KUMO_HOME', '/mnt/localssd/kumo-home')
    instance = DevserverInstance(11000)

    instance.apply_environment()
    first = {key: os.environ[key] for key in instance.environment()}
    instance.apply_environment()
    second = {key: os.environ[key] for key in instance.environment()}

    assert first == second
    assert os.environ['KUMO_HOME'] == '/mnt/localssd/kumo-home/devserver-11000'


def test_runtime_environment_includes_all_instance_paths(monkeypatch):
    monkeypatch.setenv('KUMO_HOME', '/mnt/localssd/kumo-home')
    env = DevserverInstance(11000).environment()

    assert env['INSTANCE_ID'] == 'devserver-11000'
    assert env['DEVSERVER_INSTANCE_SUFFIX'] == '/devserver-11000'
    assert env['DB_STORAGE_ROOT'].startswith(env['KUMO_HOME'] + '/')
    assert env['TMPDIR'].startswith('/tmp/kumo-dev-')
    assert len(env['TMPDIR']) == len('/tmp/kumo-dev-') + 12
    assert env['KUMO_AUTORPC_SERVER_NAME_PREFIX'] == 'devserver-11000'


def test_runtime_tmpdir_fits_torch_shared_memory_socket_limit():
    env = DevserverInstance(11000, child_id='a' * 48).environment({
        'KUMO_HOME':
        '/very/' + ('deeply-nested/' * 20) + 'kumo-home',
    })

    torch_manager_socket = os.path.join(env['TMPDIR'], 'torch-shm-dir-XXXXXXXX',
                                        'manager.sock')
    assert len(torch_manager_socket.encode()) < 108


def test_runtime_environment_preserves_operator_mise_roots():
    parent = {
        'HOME': '/home/developer',
        'KUMO_HOME': '/mnt/localssd/kumo-home',
        'XDG_CACHE_HOME': '/operator/cache',
        'XDG_DATA_HOME': '/operator/data',
    }

    env = DevserverInstance(11000).environment(parent)

    assert env['XDG_CACHE_HOME'].startswith(env['KUMO_HOME'] + '/')
    assert env['XDG_DATA_HOME'].startswith(env['KUMO_HOME'] + '/')
    assert env['MISE_CACHE_DIR'] == '/operator/cache/mise'
    assert env['MISE_DATA_DIR'] == '/operator/data/mise'


def test_runtime_environment_respects_explicit_mise_roots():
    parent = {
        'KUMO_HOME': '/mnt/localssd/kumo-home',
        'MISE_CACHE_DIR': '/shared/mise-cache',
        'MISE_DATA_DIR': '/shared/mise-data',
    }

    env = DevserverInstance(11000).environment(parent)

    assert env['MISE_CACHE_DIR'] == '/shared/mise-cache'
    assert env['MISE_DATA_DIR'] == '/shared/mise-data'


def test_apply_environment_creates_private_local_roots(tmp_path):
    env = {'KUMO_HOME': str(tmp_path / 'kumo-home')}
    instance = DevserverInstance(11000, child_id='train-a')

    instance.apply_environment(env)

    for name in ('KUMO_HOME', 'DB_STORAGE_ROOT', 'HF_HOME', 'KUMO_GRAFANA_HOME',
                 'TMPDIR', 'TORCH_HOME', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME',
                 'XDG_DATA_HOME'):
        assert os.path.isdir(env[name]), name


def test_child_identity_preserves_operator_machine_identity(monkeypatch):
    """A dynamic port must not become the Temporal/state identity."""
    monkeypatch.setenv('KUMO_HOME', '/mnt/localssd/kumo-home')
    monkeypatch.setenv('INSTANCE_ID', 'm128r2-20260711')

    train_a = DevserverInstance(11000, child_id='train-a')
    train_b = DevserverInstance(12000, child_id='train-b')

    assert train_a.environment()['INSTANCE_ID'] == 'm128r2-20260711-train-a'
    assert train_b.environment()['INSTANCE_ID'] == 'm128r2-20260711-train-b'
    assert train_a.temporal_instance_id == 'm128r2-20260711-train-a'
    assert train_a.name == 'devserver-train-a'
    assert train_a.environment()['KUMO_AUTORPC_SERVER_NAME_PREFIX'] == (
        'devserver-train-a')


def test_child_identity_is_stable_when_auto_port_allocation_retries(
        monkeypatch):
    monkeypatch.setenv('INSTANCE_ID', 'm128r2-20260711')

    first_attempt = DevserverInstance(11000, child_id='train-a')
    retry = DevserverInstance(12000, child_id='train-a')

    assert first_attempt.temporal_instance_id == retry.temporal_instance_id
    assert first_attempt.ports() != retry.ports()


def test_child_id_is_normalized_and_rejects_unsafe_path_content():
    assert DevserverInstance(11000, child_id='Train-A').child_id == 'train-a'
    with pytest.raises(ValueError, match='child_id'):
        DevserverInstance(11000, child_id='../train-a')


def test_runtime_environment_propagates_data_root_scope(monkeypatch):
    monkeypatch.setenv('DEVSERVER_INSTANCE_SUFFIX', '/devserver-11000')

    assert 'DEVSERVER_INSTANCE_SUFFIX=/devserver-11000 ' in _devserver_runtime_env(
    )


def test_cgroup_scope_is_managed_and_carries_limits_before_launch():
    limits = CgroupLimits(cpus=2.5, memory_gib=4)
    argv = scope_argv('devserver-11000', limits)

    assert argv == [
        'systemd-run',
        '--user',
        '--scope',
        '--unit=kumo-devserver-11000',
        '--property=CPUQuota=250%',
        '--property=MemoryMax=4G',
    ]


@pytest.mark.parametrize('cpus,memory_gib', [(0, 1), (1, 0), (-1, 1)])
def test_cgroup_limits_reject_non_positive_values(cpus, memory_gib):
    with pytest.raises(ValueError, match='positive'):
        CgroupLimits(cpus=cpus, memory_gib=memory_gib)


def test_cgroup_scope_rejects_path_names():
    with pytest.raises(ValueError, match='one path component'):
        scope_argv('devserver/11000', CgroupLimits(cpus=1, memory_gib=1))


def _resolved_paths(instance: DevserverInstance, monkeypatch) -> dict:
    monkeypatch.setenv('DEVSERVER_INSTANCE_SUFFIX', instance.suffix)
    storage = _resolve_storage_roots(
        FileStorageConfig(data_plane_storage_root='/srv/data',
                          artifact_root='/srv/artifacts'), DeployStage.DEV,
        instance.temporal_instance_id, False)
    data_compute = _resolve_data_compute_roots(
        DiskgraphConfig(data_root='/mnt/localssd/diskgraph'),
        instance.temporal_instance_id)
    return {
        'data_plane': storage.data_plane_storage_root,
        'control_plane': storage.control_plane_storage_root,
        'artifact': storage.artifact_root,
        'data_root': data_compute.data_root,
    }


def test_deploy_roots_are_disjoint_and_legacy_roots_are_unchanged(monkeypatch):
    legacy = _resolved_paths(DevserverInstance(10000), monkeypatch)
    first = _resolved_paths(DevserverInstance(11000), monkeypatch)
    second = _resolved_paths(DevserverInstance(12000), monkeypatch)

    assert legacy == {
        'data_plane': '/srv/data',
        'control_plane': '/srv/data',
        'artifact': '/srv/artifacts',
        'data_root': '/mnt/localssd/diskgraph',
    }
    assert set(first.values()).isdisjoint(second.values())
    assert first['data_root'].endswith('/devserver-11000')


@pytest.mark.parametrize('base_queue', [
    'data-backend-task-queue',
    'ml-workflow-task-queue',
    SINGLEHOST_DAEMON_TQ,
])
def test_temporal_task_queues_are_instance_scoped(base_queue):
    first = DeployEnvConfig(instance_id='devserver-11000',
                            deploy_mode=DeployStage.DEV)
    second = DeployEnvConfig(instance_id='devserver-12000',
                             deploy_mode=DeployStage.DEV)

    assert first.task_queue(base_queue) != second.task_queue(base_queue)
    assert first.task_queue(base_queue).endswith('-devserver-11000')


def test_temporal_workflow_ids_are_instance_scoped():
    first = DeployEnvConfig(instance_id='devserver-11000',
                            deploy_mode=DeployStage.DEV)
    second = DeployEnvConfig(instance_id='devserver-12000',
                             deploy_mode=DeployStage.DEV)

    assert first.workflow_id('graph-refresh') != second.workflow_id(
        'graph-refresh')
