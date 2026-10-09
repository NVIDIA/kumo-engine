# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinator entrypoint for a planned multi-devserver launch."""

import argparse
import contextlib
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from rest.devserver.coordinator import (
    assign_auto_base_ports,
    resolve_auto_cgroup_limits,
)
from rest.devserver.instance import DevserverInstance
from rest.devserver.lifecycle import (
    DevserverLifecycleError,
    DevserverManifest,
    ResourceVerifier,
    StaleManifestStatus,
    coordinator_locks,
    inspect_stale_manifest,
    new_manifest,
    new_owner_token,
    read_manifest,
    write_manifest_atomic,
)
from rest.devserver.planning import (
    MultiDevserverLaunchPlan,
    build_multi_devserver_plan,
)


@dataclass(frozen=True)
class MultiDevserverDependencies:
    is_port_available: Callable[[int], bool]
    capacity_provider: Callable[[], tuple[float, float]]
    python_executable: str
    main_script: Path
    child_popen: Callable[..., object]
    wait_for_listeners: Optional[Callable[["ChildLaunchSpec"], bool]] = None
    # Stale-manifest recovery probes/actions; None resolves to the real ones.
    resource_verifier: Optional[ResourceVerifier] = None
    scope_is_active: Optional[Callable[[str], bool]] = None
    port_is_listening: Optional[Callable[[int], bool]] = None
    cleanup_child_residue: Optional[Callable[[DevserverInstance], None]] = None


@dataclass(frozen=True)
class ChildLaunchSpec:
    devserver_id: str
    base_port: int
    command: tuple[str, ...]
    environment: Mapping[str, str]
    auto_base_port: bool = False


@dataclass(frozen=True)
class MultiDevserverLaunch:
    plan: MultiDevserverLaunchPlan
    children: tuple[ChildLaunchSpec, ...]
    manifest: DevserverManifest


class MultiDevserverLaunchError(DevserverLifecycleError):
    """A child did not safely reach its expected ready state.

    The manifest is deliberately retained in ``stopping`` state.  A later
    operator/recovery command must prove ownership before stopping or deleting
    any recorded resource; the launcher never guesses which process to kill.
    """


def _cgroup_v2_min(filename: str,
                   parse: Callable[[Sequence[str]], float]) -> Optional[float]:
    """Most-restrictive cgroup-v2 limit for this process (leaf->root), or None.

    Walks up so a limit set on an ancestor scope/pod still counts; ``max`` at
    every level (unfenced) yields None.
    """
    try:
        with open('/proc/self/cgroup') as source:
            rel = source.read().split('0::', 1)[1].splitlines()[0].strip('/')
    except (OSError, IndexError):
        return None
    parts = rel.split('/') if rel else []
    best: Optional[float] = None
    for depth in range(len(parts), -1, -1):
        try:
            raw = (Path('/sys/fs/cgroup', *parts[:depth])
                   / filename).read_text().split()
        except OSError:
            continue
        if not raw or raw[0] == 'max':
            continue
        value = parse(raw)
        best = value if best is None else min(best, value)
    return best


def _host_capacity() -> tuple[float, float]:
    """Available (cpus, memory_gib), capped by this process's cgroup fence.

    cgroup-v2 ``cpu.max`` / ``memory.max`` cap the host affinity / MemTotal so
    CGROUP_CONFIG=auto divides the coordinator's real budget (e.g. inside a
    container/pod), not the bare host.
    """
    cpus = float(len(os.sched_getaffinity(0)))
    cpu_quota = _cgroup_v2_min('cpu.max', lambda r: float(r[0]) / float(r[1]))
    if cpu_quota is not None:
        cpus = min(cpus, cpu_quota)
    with open('/proc/meminfo') as source:
        values = dict(line.split(':', 1) for line in source if ':' in line)
    memory_gib = float(values['MemTotal'].split()[0]) / 1024**2
    mem_bytes = _cgroup_v2_min('memory.max', lambda r: float(r[0]))
    if mem_bytes is not None:
        memory_gib = min(memory_gib, mem_bytes / 1024**3)
    return cpus, memory_gib


def _port_available(port: int) -> bool:
    """Return whether this process can bind the IPv4 devserver address.

    Unprivileged ``lsof`` can hide root-owned listener descriptors.  The UI
    runs behind Docker's root-owned ``docker-proxy``, so checking with lsof
    can incorrectly report its port as free forever.  A bind attempt is
    independent of listener ownership and exactly models child startup.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(('0.0.0.0', port))
        except OSError:
            return False
    return True


def default_dependencies() -> MultiDevserverDependencies:
    return MultiDevserverDependencies(
        is_port_available=_port_available,
        capacity_provider=_host_capacity,
        python_executable=sys.executable,
        main_script=Path(__file__).with_name('main.py'),
        child_popen=subprocess.Popen,
    )


def _systemctl_scope_active(scope_name: str) -> bool:
    """Whether a ``--user`` scope reports ``active``.

    A missing/inactive scope exits non-zero with a benign status word; only an
    unexpected failure to even run ``systemctl`` propagates (fail-closed).
    """
    result = subprocess.run(['systemctl', '--user', 'is-active', scope_name],
                            capture_output=True, text=True)
    return result.stdout.strip() == 'active'


def _tcp_port_listening(port: int) -> bool:
    """Whether ``127.0.0.1:port`` accepts a connection.

    A refused connection is a definitive negative; any other socket error
    propagates so recovery fails closed.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        try:
            probe.connect(('127.0.0.1', port))
        except ConnectionRefusedError:
            return False
        return True


def _recoverable_resource_verifier(
    scope_is_active: Callable[[str], bool],
    port_is_listening: Callable[[int], bool],
) -> ResourceVerifier:
    """True means a recorded child is provably NOT running (safe to reclaim)."""

    def verify(resource_id: str, resource: Mapping[str, Any],
               owner_token: str) -> bool:
        child_id = resource_id.split('/', 1)[0]
        instance = DevserverInstance(int(resource['base_port']),
                                     child_id=child_id)
        if scope_is_active(f'kumo-{instance.name}.scope'):
            return False  # scope alive
        if port_is_listening(instance.ports()['public_rest_api']):
            return False  # listener present; occupancy is never ownership
        return True  # dead/recoverable

    return verify


def _cleanup_child_residue(instance: DevserverInstance) -> None:
    """Best-effort reclaim of one recorded child's residue only."""
    with contextlib.suppress(Exception):
        subprocess.run(
            ['systemctl', '--user', 'reset-failed', f'kumo-{instance.name}.scope'],
            capture_output=True)
    with contextlib.suppress(Exception):
        subprocess.run(['tmux', '-L', instance.name, 'kill-server'],
                       capture_output=True)
    with contextlib.suppress(Exception):
        listed = subprocess.run(
            ['docker', 'ps', '-aq', '--filter', f'label={instance.docker_label}'],
            capture_output=True, text=True)
        ids = listed.stdout.split()
        if ids:
            subprocess.run(['docker', 'rm', '-f', *ids], capture_output=True)


def _recover_or_block_stale_manifest(
        manifest_path: Path,
        dependencies: MultiDevserverDependencies) -> None:
    """Reclaim a manifest whose every child is verified dead, else fail closed."""
    existing = read_manifest(manifest_path)
    scope_is_active = dependencies.scope_is_active or _systemctl_scope_active
    port_is_listening = dependencies.port_is_listening or _tcp_port_listening
    cleanup = dependencies.cleanup_child_residue or _cleanup_child_residue
    verifier = dependencies.resource_verifier or _recoverable_resource_verifier(
        scope_is_active, port_is_listening)
    inspection = inspect_stale_manifest(existing, verifier)
    if inspection.status is not StaleManifestStatus.VERIFIED:
        raise MultiDevserverLaunchError(
            f'existing lifecycle manifest requires owner-verified recovery: '
            f'{manifest_path}: {inspection.reason}')
    for child_id, child in existing.children.items():
        launch_spec = child.get('resources', {}).get('launch_spec', {})
        base_port = launch_spec.get('base_port')
        if base_port is None:
            continue
        cleanup(DevserverInstance(int(base_port), child_id=child_id))
    manifest_path.unlink()


def _all_ports_available(child: ChildLaunchSpec,
                         is_port_available: Callable[[int], bool]) -> bool:
    return all(
        is_port_available(port) for port in DevserverInstance(
            child.base_port, child_id=child.devserver_id).ports().values())


def _wait_for_expected_listeners(
    child: ChildLaunchSpec,
    is_port_available: Callable[[int], bool],
    *,
    # Cold UI install/build can exceed a minute.
    timeout_seconds: float = 300,
    poll_seconds: float = 0.25,
) -> bool:
    """Wait until each child's always-on listeners bind.

    Occupancy is never treated as ownership, so recovery stays fail-closed.
    """
    # web_ui + both APIs are the always-on readiness contract; metrics/Grafana/
    # Jaeger/Prometheus may be absent (e.g. DEVSERVER_MONITORING=0) but are
    # still reserved.
    allocated = DevserverInstance(child.base_port,
                                  child_id=child.devserver_id).ports()
    expected_ports = {
        name: allocated[name]
        for name in ('web_ui', 'rest_api', 'public_rest_api')
    }
    deadline = time.monotonic() + timeout_seconds
    while True:
        if all(not is_port_available(port) for port in expected_ports.values()):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_seconds)


def _with_base_port(child: ChildLaunchSpec, base_port: int) -> ChildLaunchSpec:
    command = list(child.command)
    port_index = command.index('--base_port') + 1
    command[port_index] = str(base_port)
    return replace(child, base_port=base_port, command=tuple(command))


def _next_auto_port(
        child: ChildLaunchSpec, *, reserved_ports: set[int],
        is_port_available: Callable[[int], bool]) -> ChildLaunchSpec:
    """Choose a replacement only for a coordinator-auto-assigned block."""
    if not child.auto_base_port:
        raise MultiDevserverLaunchError(
            f'explicit base port {child.base_port} lost its complete port '
            'block before launch; refusing to choose a different address')
    # ``assign_auto_base_ports`` uses this stride.  Start after the original
    # allocation so a racing unrelated listener cannot cause an infinite retry.
    for base_port in range(child.base_port + 100, 65536, 100):
        candidate = _with_base_port(child, base_port)
        ports = set(
            DevserverInstance(candidate.base_port,
                              child_id=candidate.devserver_id).ports().values())
        if max(ports) > 65535:
            break
        if not ports & reserved_ports and _all_ports_available(
                candidate, is_port_available):
            return candidate
    raise MultiDevserverLaunchError(
        f'no replacement complete port block is available for auto child '
        f'{child.devserver_id!r}')


def _manifest_with_child_spec(manifest: DevserverManifest,
                              child: ChildLaunchSpec) -> DevserverManifest:
    """Persist an auto-retried command before starting it."""
    children = {
        child_id: dict(data)
        for child_id, data in manifest.children.items()
    }
    child_data = children[child.devserver_id]
    resources = dict(child_data['resources'])
    launch_spec = dict(resources['launch_spec'])
    launch_spec['command'] = list(child.command)
    launch_spec['base_port'] = child.base_port
    resources['launch_spec'] = launch_spec
    child_data['resources'] = resources
    children[child.devserver_id] = child_data
    return replace(manifest, children=children)


def build_multi_devserver_launch(
    *,
    cgroup_config: Optional[str] = None,
    nums: Optional[str] = None,
    devserver_ids: Optional[str] = None,
    base_ports: Optional[str] = None,
    unfenced: bool = False,
    parent_env: Mapping[str, str],
    manifest_path: Path,
    dependencies: MultiDevserverDependencies,
) -> MultiDevserverLaunch:
    plan = build_multi_devserver_plan(cgroup_config, nums=nums,
                                      devserver_ids=devserver_ids,
                                      base_ports=base_ports, unfenced=unfenced)
    plan = assign_auto_base_ports(plan,
                                  is_available=dependencies.is_port_available)
    if plan.cgroup_mode == 'auto':
        cpu_capacity, memory_gib = dependencies.capacity_provider()
        limits = resolve_auto_cgroup_limits(cpu_capacity=cpu_capacity,
                                            memory_gib=memory_gib,
                                            count=plan.nums)
        plan = MultiDevserverLaunchPlan(
            cgroup_mode=plan.cgroup_mode, devservers=tuple(
                type(child)(child.devserver_id, child.base_port, limit)
                for child, limit in zip(plan.devservers, limits)))

    owner_token = new_owner_token()
    auto_base_ports = base_ports is None or base_ports.strip().casefold(
    ) == 'auto'
    specs = []
    children = {}
    for child in plan.devservers:
        instance = DevserverInstance(child.base_port,
                                     child_id=child.devserver_id)
        env = dict(parent_env)
        env.update(instance.environment(env))
        command = [
            dependencies.python_executable, '-m', 'rest.devserver.main',
            '--child-id', child.devserver_id, '--base_port',
            str(child.base_port), '--detached'
        ]
        if child.cgroup_limits is not None:
            command.extend([
                '--cgroup', '--cgroup-cpus', f'{child.cgroup_limits.cpus:g}',
                '--cgroup-memory-gib', f'{child.cgroup_limits.memory_gib:g}'
            ])
        specs.append(
            ChildLaunchSpec(child.devserver_id, child.base_port, tuple(command),
                            env, auto_base_port=auto_base_ports))
        children[child.devserver_id] = {
            'resources': {
                'launch_spec': {
                    'owner_token': owner_token,
                    'command': list(command),
                    'base_port': child.base_port,
                },
            },
        }
    manifest = new_manifest(children, owner_token=owner_token)
    return MultiDevserverLaunch(plan, tuple(specs), manifest)


def render_launch_plan(launch: MultiDevserverLaunch) -> str:
    lines = ['Multi-devserver launch plan (planned; no process started)']
    for child, spec in zip(launch.plan.devservers, launch.children):
        limits = ('unfenced' if child.cgroup_limits is None else
                  f'cgroup={child.cgroup_limits.cpus:g} CPUs, '
                  f'{child.cgroup_limits.memory_gib:g} GiB')
        lines.extend([
            f'- {spec.devserver_id}: http://localhost:{spec.base_port + 1}/docs',
            f'  raw INSTANCE_ID={spec.environment["INSTANCE_ID"]}',
            f'  {limits}',
        ])
    return '\n'.join(lines)


def launch_multi_devservers(
    launch: MultiDevserverLaunch,
    *,
    manifest_path: Path,
    dependencies: MultiDevserverDependencies,
    cwd: Path,
) -> None:
    """Launch only planned children, leaving a durable fail-closed trail.

    An existing manifest is auto-recovered only when every recorded child is
    verified dead; an ambiguous or live one still blocks the launch.  Likewise
    a partial launch is recorded as ``stopping`` rather than being cleaned up
    by process name or port.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    child_ids = [child.devserver_id for child in launch.children]
    with coordinator_locks(manifest_path.parent, child_ids):
        if manifest_path.exists():
            # Auto-recover only when every recorded child is verified dead.
            _recover_or_block_stale_manifest(manifest_path, dependencies)
        write_manifest_atomic(manifest_path, launch.manifest)
        launching = replace(launch.manifest, phase='launching')
        write_manifest_atomic(manifest_path, launching)
        active_manifest = launching
        launched_children = []
        try:
            for child in launch.children:
                reserved_ports = set()
                for other in launch.children:
                    if other.devserver_id == child.devserver_id:
                        continue
                    reserved_ports.update(
                        DevserverInstance(
                            other.base_port,
                            child_id=other.devserver_id).ports().values())
                if not _all_ports_available(child,
                                            dependencies.is_port_available):
                    child = _next_auto_port(
                        child, reserved_ports=reserved_ports,
                        is_port_available=dependencies.is_port_available)
                    active_manifest = _manifest_with_child_spec(
                        active_manifest, child)
                    write_manifest_atomic(manifest_path, active_manifest)
                dependencies.child_popen(child.command,
                                         env=dict(child.environment),
                                         cwd=str(cwd), start_new_session=True)
                launched_children.append(child.devserver_id)
                listener_ready = dependencies.wait_for_listeners
                if listener_ready is None:
                    listener_ready = lambda spec: _wait_for_expected_listeners(
                        spec, dependencies.is_port_available)
                if not listener_ready(child):
                    raise MultiDevserverLaunchError(
                        f'child {child.devserver_id!r} did not bind every '
                        'expected listener before the readiness deadline')
        except BaseException as error:
            # Do not guess at teardown.  ``stopping`` makes the partial state
            # explicit and forces a later owner-verifying recovery operation.
            write_manifest_atomic(manifest_path,
                                  replace(active_manifest, phase='stopping'))
            if isinstance(error, MultiDevserverLaunchError):
                raise
            raise MultiDevserverLaunchError(
                f'launch failed after starting {launched_children}: {error}') \
                from error
        write_manifest_atomic(manifest_path,
                              replace(active_manifest, phase='running'))


def parse_multi_devserver_args(argv: Sequence[str]) -> argparse.Namespace:
    def env_optional(name: str) -> Optional[str]:
        value = os.environ.get(name)
        return value if value and value.strip() else None

    parser = argparse.ArgumentParser()
    parser.add_argument('--cgroup-config',
                        default=env_optional('CGROUP_CONFIG'))
    parser.add_argument('--nums', default=env_optional('NUMS'))
    parser.add_argument('--devserver-ids',
                        default=env_optional('DEVSERVER_IDS'))
    parser.add_argument('--base-ports', default=env_optional('BASE_PORTS'))
    parser.add_argument('--unfenced', action='store_true',
                        default=os.environ.get('UNFENCED') == '1')
    parser.add_argument(
        '--print-child-env',
        metavar='DEVSERVER_ID',
        help=('print shell exports for one running manifest-owned child; '
              'intended for external clients such as test harnesses'),
    )
    args = parser.parse_args(argv)
    if args.unfenced and args.cgroup_config:
        parser.error('UNFENCED is mutually exclusive with CGROUP_CONFIG')
    return args


def render_child_client_environment(
    devserver_id: str,
    *,
    manifest_path: Path,
    parent_env: Mapping[str, str],
) -> str:
    """Return the endpoint/home pair for one running owned child.

    External clients are not launched by a child devserver and consequently
    cannot inherit its ``KUMO_HOME``.  Resolving both values from the same
    manifest-owned child prevents a client from sending requests to one
    devserver while polling artifact paths belonging to another one.
    """
    # Constructing an instance performs the same normalized, safe child-ID
    # validation used for launch.  The base port is replaced below from the
    # durable launch spec.
    normalized_id = DevserverInstance(10000, child_id=devserver_id).child_id
    assert normalized_id is not None
    manifest = read_manifest(manifest_path)
    if manifest.phase != 'running':
        raise MultiDevserverLaunchError(
            f'devserver manifest is {manifest.phase!r}, not running: '
            f'{manifest_path}')
    child = manifest.children.get(normalized_id)
    if child is None:
        raise MultiDevserverLaunchError(
            f'devserver child {normalized_id!r} is not owned by manifest: '
            f'{manifest_path}')
    launch_spec = child['resources'].get('launch_spec')
    if not isinstance(launch_spec, Mapping):
        raise MultiDevserverLaunchError(
            f'devserver child {normalized_id!r} has no launch specification')
    command = launch_spec.get('command')
    if not isinstance(command, list):
        raise MultiDevserverLaunchError(
            f'devserver child {normalized_id!r} has an invalid launch command')
    try:
        child_arg = command[command.index('--child-id') + 1]
        base_port = int(command[command.index('--base_port') + 1])
    except (ValueError, IndexError) as error:
        raise MultiDevserverLaunchError(
            f'devserver child {normalized_id!r} launch command is incomplete') \
            from error
    if child_arg != normalized_id:
        raise MultiDevserverLaunchError(
            f'devserver manifest child {normalized_id!r} has mismatched '
            f'launch identity {child_arg!r}')
    instance = DevserverInstance(base_port, child_id=normalized_id)
    child_home = os.path.abspath(instance.environment(parent_env)['KUMO_HOME'])
    if not os.path.isdir(child_home):
        raise MultiDevserverLaunchError(
            f'devserver child {normalized_id!r} home is absent: {child_home}')
    public_url = f'http://localhost:{instance.ports()["public_rest_api"]}'
    return 'export KUMO_API_URL={}; export KUMO_HOME={}'.format(
        shlex.quote(public_url), shlex.quote(child_home))


def main(argv: Sequence[str] = ()) -> int:
    args = parse_multi_devserver_args(argv)
    manifest_path = Path(os.environ.get('KUMO_HOME', '.')) / \
        '.multi-devserver' / 'manifest.json'
    if args.print_child_env:
        print(render_child_client_environment(
            args.print_child_env, manifest_path=manifest_path,
            parent_env=os.environ))
        return 0
    launch = build_multi_devserver_launch(
        cgroup_config=args.cgroup_config, nums=args.nums,
        devserver_ids=args.devserver_ids, base_ports=args.base_ports,
        unfenced=args.unfenced, parent_env=os.environ,
        manifest_path=manifest_path,
        dependencies=default_dependencies())
    print(render_launch_plan(launch))
    if os.environ.get('PLAN_ONLY') == '1':
        return 0
    launch_multi_devservers(launch, manifest_path=manifest_path,
                            dependencies=default_dependencies(), cwd=Path.cwd())
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
