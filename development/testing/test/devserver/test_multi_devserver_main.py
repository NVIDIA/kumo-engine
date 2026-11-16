# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the non-launching multi-devserver coordinator entrypoint."""

import socket
from dataclasses import replace
from pathlib import Path

import pytest

from rest.devserver.lifecycle import read_manifest
from rest.devserver.multi_main import (
    ChildLaunchSpec,
    MultiDevserverDependencies,
    MultiDevserverLaunchError,
    _port_available,
    _wait_for_expected_listeners,
    build_multi_devserver_launch,
    launch_multi_devservers,
    parse_multi_devserver_args,
    render_child_client_environment,
    render_launch_plan,
)


def _dependencies(*, unavailable_ports=()):
    unavailable = set(unavailable_ports)
    subprocess_calls = []

    def subprocess_runner(*args, **kwargs):
        subprocess_calls.append((args, kwargs))
        raise AssertionError('planning must not start a subprocess')

    return (MultiDevserverDependencies(
        is_port_available=lambda port: port not in unavailable,
        capacity_provider=lambda: (64, 512),
        python_executable='/test/python',
        main_script=Path('/repo/rest/devserver/main.py'),
        child_popen=subprocess_runner,
        wait_for_listeners=lambda _: True,
    ), subprocess_calls)


def test_port_availability_uses_a_real_bind_attempt() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert not _port_available(port)
    assert _port_available(port)


def test_readiness_requires_only_always_on_child_listeners() -> None:
    child = ChildLaunchSpec(
        devserver_id='train-a',
        base_port=11000,
        command=(),
        environment={},
    )
    # Optional Grafana/Jaeger/Prometheus addresses remain available, while
    # the three operational endpoints are listening.
    assert _wait_for_expected_listeners(
        child,
        lambda port: port not in {11000, 11001, 11002},
        timeout_seconds=0,
    )


def test_auto_builds_fenced_detached_specs_with_inherited_parent_identity():
    dependencies, subprocess_calls = _dependencies(unavailable_ports={11001})

    launch = build_multi_devserver_launch(
        cgroup_config='auto',
        nums='2',
        devserver_ids='train-a,train-b',
        parent_env={
            'INSTANCE_ID': 'm128r2-20260711',
            'KUMO_HOME': '/scratch/kumo',
        },
        manifest_path=Path('/state/manifest.json'),
        dependencies=dependencies,
    )

    first, second = launch.children
    assert [first.base_port, second.base_port] == [11100, 11200]
    assert first.environment['INSTANCE_ID'] == 'm128r2-20260711-train-a'
    assert second.environment['INSTANCE_ID'] == 'm128r2-20260711-train-b'
    assert first.environment['KUMO_HOME'] == '/scratch/kumo/devserver-train-a'
    assert first.command == (
        '/test/python',
        '-m',
        'rest.devserver.main',
        '--child-id',
        'train-a',
        '--base_port',
        '11100',
        '--detached',
        '--cgroup',
        '--cgroup-cpus',
        '28',
        '--cgroup-memory-gib',
        '230',
    )
    assert launch.manifest.phase == 'planned'
    assert launch.manifest.children['train-a']['resources']['launch_spec'][
        'owner_token'] == launch.manifest.owner_token
    assert subprocess_calls == []


def test_custom_config_keeps_explicit_ports_and_limits():
    dependencies, _ = _dependencies()

    launch = build_multi_devserver_launch(
        cgroup_config='{"small":{"cpus":4,"memory_gib":32}}',
        base_ports='15000',
        parent_env={'INSTANCE_ID': 'machine'},
        manifest_path=Path('/state/manifest.json'),
        dependencies=dependencies,
    )

    child = launch.children[0]
    assert child.base_port == 15000
    assert '--cgroup-cpus' in child.command
    assert child.command[child.command.index('--cgroup-cpus') + 1] == '4'
    assert child.command[child.command.index('--cgroup-memory-gib') + 1] == '32'
    assert child.environment['INSTANCE_ID'] == 'machine-small'


def test_unfenced_specs_have_no_cgroup_flags_or_capacity_lookup():
    capacity_calls = []
    dependencies = MultiDevserverDependencies(
        is_port_available=lambda _: True,
        capacity_provider=lambda: capacity_calls.append(True),
        python_executable='/test/python',
        main_script=Path('/repo/rest/devserver/main.py'),
        child_popen=lambda *args, **kwargs: None,
        wait_for_listeners=lambda _: True,
    )

    launch = build_multi_devserver_launch(
        nums=1,
        devserver_ids='ui-only',
        unfenced=True,
        parent_env={'INSTANCE_ID': 'machine'},
        manifest_path=Path('/state/manifest.json'),
        dependencies=dependencies,
    )

    assert '--cgroup' not in launch.children[0].command
    assert capacity_calls == []


def test_plan_output_contains_child_addresses_limits_and_raw_temporal_identity(
):
    dependencies, _ = _dependencies()
    launch = build_multi_devserver_launch(
        cgroup_config='{"train-a":{"cpus":4,"memory_gib":32}}',
        base_ports='15000',
        parent_env={'INSTANCE_ID': 'machine'},
        manifest_path=Path('/state/manifest.json'),
        dependencies=dependencies,
    )

    output = render_launch_plan(launch)

    assert 'Multi-devserver launch plan (planned; no process started)' in output
    assert 'train-a' in output
    assert 'http://localhost:15001/docs' in output
    assert 'raw INSTANCE_ID=machine-train-a' in output
    assert 'cgroup=4 CPUs, 32 GiB' in output


def test_client_environment_uses_one_running_manifest_owned_child(tmp_path):
    dependencies, _ = _dependencies()
    launch = build_multi_devserver_launch(
        cgroup_config='{"train-a":{"cpus":4,"memory_gib":32}}',
        base_ports='15000', parent_env={'INSTANCE_ID': 'machine'},
        manifest_path=tmp_path / 'manifest.json', dependencies=dependencies,
    )
    from rest.devserver.lifecycle import write_manifest_atomic
    write_manifest_atomic(tmp_path / 'manifest.json',
                          replace(launch.manifest, phase='running'))
    home = tmp_path / 'homes' / 'devserver-train-a'
    home.mkdir(parents=True)

    output = render_child_client_environment(
        'train-a', manifest_path=tmp_path / 'manifest.json',
        parent_env={'KUMO_HOME': str(tmp_path / 'homes')},
    )

    assert output == (
        'export KUMO_API_URL=http://localhost:15002; '
        f'export KUMO_HOME={home}')


def test_client_environment_rejects_unknown_or_not_running_child(tmp_path):
    dependencies, _ = _dependencies()
    launch = build_multi_devserver_launch(
        cgroup_config='{"train-a":{"cpus":4,"memory_gib":32}}',
        base_ports='15000', parent_env={'INSTANCE_ID': 'machine'},
        manifest_path=tmp_path / 'manifest.json', dependencies=dependencies,
    )
    from rest.devserver.lifecycle import write_manifest_atomic
    write_manifest_atomic(tmp_path / 'manifest.json', launch.manifest)

    with pytest.raises(MultiDevserverLaunchError, match='not running'):
        render_child_client_environment(
            'train-a', manifest_path=tmp_path / 'manifest.json',
            parent_env={'KUMO_HOME': str(tmp_path / 'homes')},
        )


def test_cli_inputs_default_to_environment_and_validate_unfenced(monkeypatch):
    monkeypatch.setenv('CGROUP_CONFIG', 'auto')
    monkeypatch.setenv('NUMS', '2')
    monkeypatch.setenv('DEVSERVER_IDS', 'train-a,train-b')
    monkeypatch.setenv('BASE_PORTS', '15000,15100')
    args = parse_multi_devserver_args([])

    assert args.cgroup_config == 'auto'
    assert args.nums == '2'
    assert args.devserver_ids == 'train-a,train-b'
    assert args.base_ports == '15000,15100'
    assert not args.unfenced

    with pytest.raises(SystemExit) as error:
        parse_multi_devserver_args(['--unfenced', '--cgroup-config', 'auto'])
    assert error.value.code == 2


def test_launch_persists_manifest_before_starting_only_owned_specs(tmp_path):
    dependencies, calls = _dependencies()
    launch = build_multi_devserver_launch(
        cgroup_config='{"train-a":{"cpus":4,"memory_gib":32}}',
        base_ports='15000',
        parent_env={'INSTANCE_ID': 'machine'},
        manifest_path=tmp_path / 'manifest.json',
        dependencies=dependencies,
    )
    dependencies = MultiDevserverDependencies(
        is_port_available=dependencies.is_port_available,
        capacity_provider=dependencies.capacity_provider,
        python_executable=dependencies.python_executable,
        main_script=dependencies.main_script,
        child_popen=lambda *args, **kwargs: calls.append((args, kwargs)),
        wait_for_listeners=lambda _: True,
    )

    manifest_path = tmp_path / 'manifest.json'
    launch_multi_devservers(launch, manifest_path=manifest_path,
                            dependencies=dependencies, cwd=tmp_path)

    assert len(calls) == 1
    assert calls[0][1]['env']['INSTANCE_ID'] == 'machine-train-a'
    assert read_manifest(manifest_path).phase == 'running'


def test_launch_retries_only_an_auto_port_block_that_loses_preflight(tmp_path):
    dependencies, calls = _dependencies()
    launch = build_multi_devserver_launch(
        cgroup_config='auto',
        nums='1',
        parent_env={'INSTANCE_ID': 'machine'},
        manifest_path=tmp_path / 'manifest.json',
        dependencies=dependencies,
    )
    # Planning saw 11000 free.  Simulate an unrelated listener arriving after
    # planning but before detached child launch.
    unavailable = {11001}
    dependencies = MultiDevserverDependencies(
        is_port_available=lambda port: port not in unavailable,
        capacity_provider=dependencies.capacity_provider,
        python_executable=dependencies.python_executable,
        main_script=dependencies.main_script,
        child_popen=lambda *args, **kwargs: calls.append((args, kwargs)),
        wait_for_listeners=lambda _: True,
    )

    manifest_path = tmp_path / 'manifest.json'
    launch_multi_devservers(launch, manifest_path=manifest_path,
                            dependencies=dependencies, cwd=tmp_path)

    command = calls[0][0][0]
    assert command[command.index('--base_port') + 1] == '11100'
    assert read_manifest(manifest_path).children['devserver-1']['resources'][
        'launch_spec']['base_port'] == 11100


def test_launch_does_not_change_an_explicit_port_or_cleanup_on_failure(
        tmp_path):
    dependencies, calls = _dependencies()
    launch = build_multi_devserver_launch(
        cgroup_config='{"train-a":{"cpus":4,"memory_gib":32}}',
        base_ports='15000',
        parent_env={'INSTANCE_ID': 'machine'},
        manifest_path=tmp_path / 'manifest.json',
        dependencies=dependencies,
    )
    dependencies = MultiDevserverDependencies(
        is_port_available=lambda port: port != 15001,
        capacity_provider=dependencies.capacity_provider,
        python_executable=dependencies.python_executable,
        main_script=dependencies.main_script,
        child_popen=lambda *args, **kwargs: calls.append((args, kwargs)),
        wait_for_listeners=lambda _: True,
    )

    manifest_path = tmp_path / 'manifest.json'
    with pytest.raises(MultiDevserverLaunchError, match='explicit base port'):
        launch_multi_devservers(launch, manifest_path=manifest_path,
                                dependencies=dependencies, cwd=tmp_path)

    assert calls == []
    assert read_manifest(manifest_path).phase == 'stopping'


def test_listener_failure_is_durable_and_existing_manifest_is_not_overwritten(
        tmp_path):
    dependencies, calls = _dependencies()
    launch = build_multi_devserver_launch(
        cgroup_config='{"train-a":{"cpus":4,"memory_gib":32}}',
        base_ports='15000',
        parent_env={'INSTANCE_ID': 'machine'},
        manifest_path=tmp_path / 'manifest.json',
        dependencies=dependencies,
    )
    dependencies = MultiDevserverDependencies(
        is_port_available=dependencies.is_port_available,
        capacity_provider=dependencies.capacity_provider,
        python_executable=dependencies.python_executable,
        main_script=dependencies.main_script,
        child_popen=lambda *args, **kwargs: calls.append((args, kwargs)),
        wait_for_listeners=lambda _: False,
    )
    manifest_path = tmp_path / 'manifest.json'
    with pytest.raises(MultiDevserverLaunchError, match='expected listener'):
        launch_multi_devservers(launch, manifest_path=manifest_path,
                                dependencies=dependencies, cwd=tmp_path)
    stopped = read_manifest(manifest_path)
    assert stopped.phase == 'stopping'
    assert len(calls) == 1

    # A recorded child that still verifies alive keeps the manifest intact.
    alive = replace(dependencies, resource_verifier=lambda *_: False)
    with pytest.raises(MultiDevserverLaunchError,
                       match='owner-verified recovery'):
        launch_multi_devservers(launch, manifest_path=manifest_path,
                                dependencies=alive, cwd=tmp_path)
    assert read_manifest(manifest_path) == stopped


def _write_stale_manifest(launch, manifest_path):
    from rest.devserver.lifecycle import write_manifest_atomic
    manifest_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_manifest_atomic(manifest_path, replace(launch.manifest,
                                                 phase='stopping'))


def _two_child_launch(tmp_path):
    dependencies, _ = _dependencies()
    return build_multi_devserver_launch(
        cgroup_config=('{"train-a":{"cpus":4,"memory_gib":32},'
                       '"train-b":{"cpus":4,"memory_gib":32}}'),
        base_ports='15000,15100',
        parent_env={'INSTANCE_ID': 'machine'},
        manifest_path=tmp_path / 'manifest.json',
        dependencies=dependencies,
    )


def test_stale_manifest_recovers_when_every_child_is_verified_dead(tmp_path):
    dependencies, _ = _dependencies()
    launch = _two_child_launch(tmp_path)
    manifest_path = tmp_path / 'manifest.json'
    _write_stale_manifest(launch, manifest_path)

    popened = []
    cleaned = []
    deps = replace(
        dependencies,
        child_popen=lambda *a, **k: popened.append((a, k)),
        wait_for_listeners=lambda _: True,
        resource_verifier=lambda *_: True,
        cleanup_child_residue=lambda instance: cleaned.append(instance),
    )
    launch_multi_devservers(launch, manifest_path=manifest_path,
                            dependencies=deps, cwd=tmp_path)

    assert read_manifest(manifest_path).phase == 'running'
    assert len(popened) == 2
    # Cleanup targets exactly the recorded child identities, nothing broader.
    assert sorted(instance.child_id for instance in cleaned) == [
        'train-a', 'train-b']
    assert sorted(instance.base_port for instance in cleaned) == [15000, 15100]


def test_stale_manifest_with_one_live_child_blocks_and_is_preserved(tmp_path):
    dependencies, _ = _dependencies()
    launch = _two_child_launch(tmp_path)
    manifest_path = tmp_path / 'manifest.json'
    _write_stale_manifest(launch, manifest_path)
    stopped = read_manifest(manifest_path)

    popened = []
    cleaned = []
    live = {'train-a'}
    deps = replace(
        dependencies,
        child_popen=lambda *a, **k: popened.append((a, k)),
        resource_verifier=lambda rid, *_: rid.split('/', 1)[0] not in live,
        cleanup_child_residue=lambda instance: cleaned.append(instance),
    )
    with pytest.raises(MultiDevserverLaunchError,
                       match='owner-verified recovery'):
        launch_multi_devservers(launch, manifest_path=manifest_path,
                                dependencies=deps, cwd=tmp_path)

    assert manifest_path.exists()
    assert read_manifest(manifest_path) == stopped
    assert popened == []
    assert cleaned == []


def test_stale_manifest_stays_when_verifier_raises(tmp_path):
    dependencies, _ = _dependencies()
    launch = _two_child_launch(tmp_path)
    manifest_path = tmp_path / 'manifest.json'
    _write_stale_manifest(launch, manifest_path)
    stopped = read_manifest(manifest_path)

    def raiser(*_):
        raise RuntimeError('probe exploded')

    popened = []
    deps = replace(
        dependencies,
        child_popen=lambda *a, **k: popened.append((a, k)),
        resource_verifier=raiser,
    )
    with pytest.raises(MultiDevserverLaunchError,
                       match='owner-verified recovery'):
        launch_multi_devservers(launch, manifest_path=manifest_path,
                                dependencies=deps, cwd=tmp_path)

    assert manifest_path.exists()
    assert read_manifest(manifest_path) == stopped
    assert popened == []


def test_default_recoverable_verifier_reads_scope_and_public_port():
    from rest.devserver.multi_main import _recoverable_resource_verifier

    scope_calls = []
    port_calls = []

    def scope(name):
        scope_calls.append(name)
        return False

    def port(number):
        port_calls.append(number)
        return False

    verifier = _recoverable_resource_verifier(scope, port)
    resource = {'base_port': 15000}

    assert verifier('train-a/launch_spec', resource, 'token') is True
    assert scope_calls == ['kumo-devserver-train-a.scope']
    # public_rest_api is base_port + 2.
    assert port_calls == [15002]

    # A live scope short-circuits to alive without a port probe.
    scope_calls.clear()
    port_calls.clear()
    verifier = _recoverable_resource_verifier(lambda _: True, port)
    assert verifier('train-a/launch_spec', resource, 'token') is False
    assert port_calls == []
