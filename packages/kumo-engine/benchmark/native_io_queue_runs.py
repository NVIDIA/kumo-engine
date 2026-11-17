# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observe an explicitly pinned candidate without changing its runtime or harness."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import stat
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from io_utils import fsutil

RUNTIME_PATHS = ('packages/diskgraph', 'packages/kumo-core/src',
                 'packages/kumo-engine/kumo/engine',
                 'packages/kumo-engine/test/wheel_gpu_smoke.py')
SOURCE_FILES = ('native_io_queue_runs.py', 'native_io_queue.py',
                'native_io_queue.c', 'native_io_queue_results.py', 'native_io_cpu.py')


def read_json(path: str) -> dict:
    return json.loads(fsutil.read_text(path))


def save_json(path: str, value: dict) -> None:
    fsutil.write_text(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


def durable_artifacts(spec: dict, root: str, controller_log: str | None = None) -> list[str]:
    prefix = spec['result'].removesuffix('.json')
    paths = list(spec['iops_paths'].values()) + list(spec['observer_paths'].values())
    paths += [spec['result'], prefix + '.log', prefix + '-resources.json',
              spec['runtime_prepared_dir'] + '/manifest.json']
    paths += [root + '/' + name for name in ('plan.json', 'progress.json', 'completion.json')]
    base = Path(root).resolve()
    synced = []
    if controller_log is not None:
        sys.stdout.flush()
        # Only the explicitly declared experiment log may live beside root.
        with fsutil.get_binary_handle(controller_log, 'rb') as handle:
            os.fsync(handle.fileno())
        synced.append(controller_log)
    for path in sorted(set(paths)):
        if not Path(path).resolve().is_relative_to(base):
            raise ValueError('Artifact durability is restricted to this owned experiment')
        if fsutil.exists(path):
            # fsutil has no durability primitive; fsync on a read fd also
            # handles the root collector's readable, immutable output files.
            with fsutil.get_binary_handle(path, 'rb') as handle:
                os.fsync(handle.fileno())
            synced.append(path)

    def sync_directory(path: str) -> None:
        for child in fsutil.listdir(path):
            if Path(child).is_symlink() or not Path(child).resolve().is_relative_to(base):
                raise ValueError('Owned artifact directory contains an external link')
            if fsutil.isdir(child):
                sync_directory(child)
        # POSIX directory-fd durability has no fsutil equivalent. This also
        # persists successful private-clone/work-directory removal metadata.
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    sync_directory(root)
    return synced


def validate_controller(root: str, path: str | None) -> None:
    metadata = os.fstat(1)
    if path is None:
        if stat.S_ISREG(metadata.st_mode):
            raise ValueError('Redirected controller stdout requires --controller-log')
        return
    if os.path.abspath(path) != root + '-controller.log' or Path(path).is_symlink():
        raise ValueError('Controller log must be the exact owned experiment companion path')
    declared = os.stat(path)
    if (not stat.S_ISREG(metadata.st_mode) or
            (metadata.st_dev, metadata.st_ino) != (declared.st_dev, declared.st_ino)):
        raise ValueError('Declared controller log is not the redirected stdout file')


def quiet_device(path: str, seconds: float, timeout: float) -> dict:
    from native_io_trace import device_identity
    if not all(math.isfinite(value) for value in (seconds, timeout)) or not 0 < seconds < timeout:
        raise ValueError('Device quiet interval must be positive and shorter than its timeout')
    device = device_identity(path)
    started = time.monotonic_ns()
    deadline = time.monotonic() + timeout
    previous, stable_since, samples = None, None, 0
    while time.monotonic() < deadline:
        counters = [int(value) for value in fsutil.read_text(device['sysfs'] + '/stat').split()]
        if len(counters) < 11 or any(value < 0 for value in counters):
            raise ValueError('Unsupported physical-device statistics')
        now = time.monotonic_ns()
        samples += 1
        if counters[8] or counters != previous:
            stable_since = now if counters[8] == 0 else None
        elif stable_since is not None and (now - stable_since) / 1e9 >= seconds:
            return {'device': device, 'started_ns': started, 'quiet_started_ns': stable_since,
                    'ended_ns': now, 'required_seconds': seconds, 'samples': samples,
                    'stable_counters': counters}
        previous = counters
        time.sleep(.05)
    raise TimeoutError('Physical device did not become quiet before payload release')


def file_hash(path: str) -> str:
    return hashlib.sha256(fsutil.read_binary(path)).hexdigest()


def git(root: str, *arguments: str) -> str:
    return subprocess.check_output(['git', '-C', root, *arguments], text=True).strip()


def pin_source(root: str, commit: str) -> dict:
    if git(root, 'rev-parse', 'HEAD') != commit or git(root, 'status', '--porcelain'):
        raise ValueError(f'Expected clean, committed source {commit}: {root}')
    return {'root': root, 'commit': commit}


def verify_pins(pins: dict) -> None:
    for key in ('candidate', 'tools'):
        pin_source(pins[key]['root'], pins[key]['commit'])
    for path, expected in pins['files'].items():
        if file_hash(path) != expected:
            raise ValueError(f'Pinned source or installed binary changed: {path}')


def observer_order(repetitions: int, backends: list[str], modes: list[str],
                   seed: int = 20261109) -> list[tuple[int, str, str]]:
    if repetitions < 1 or not backends or not modes:
        raise ValueError('Positive repetitions and nonempty controls are required')
    if len(set(backends)) != len(backends) or len(set(modes)) != len(modes):
        raise ValueError('Repeated controls would overwrite trial identities')
    rng = random.Random(seed)
    rows = []
    for repeat in range(repetitions):
        selected = list(backends)
        rng.shuffle(selected)
        for backend in selected:
            paired = list(modes)
            rng.shuffle(paired)
            rows.extend((repeat, backend, mode) for mode in paired)
    return rows


def observer_command(spec: dict, args: argparse.Namespace, mode: str) -> list[str]:
    if mode not in {'queue', 'cpu', 'offcpu'}:
        raise ValueError('No collector exists for the unobserved control')
    paths = spec['observer_paths']
    script = 'native_io_queue.py' if mode == 'queue' else 'native_io_cpu.py'
    command = [
        'taskset', '-c', args.collector_cpus, 'sudo', '-n', spec['python'],
        str(Path(__file__).with_name(script)), 'collect',
        '--candidate-root', args.candidate_root, '--gate', spec['iops_paths']['gate'],
        '--ready', paths['ready'], '--stop', spec['iops_paths']['stop'],
        '--output', paths['record'], '--events', paths['events'],
        '--device', args.device, '--timeout', str(spec['timeout'] + 600),
        '--drain-seconds', str(args.drain_seconds),
        '--bcc-python-path', args.bcc_python_path,
    ]
    if mode == 'queue':
        command += ['--request-capacity', str(args.request_capacity),
                    '--event-capacity-bytes', str(args.event_capacity_bytes),
                    '--quiet-seconds', str(args.quiet_seconds)]
    else:
        command += ['--mode', mode, '--frequency', str(args.frequency),
                    '--event-capacity-bytes', str(args.event_capacity_bytes)]
    return command


@contextmanager
def ram_log(path: str):
    # No fsutil equivalent exists for an anonymous, bounded RAM-only fd.
    # Pre-size and seal growth so observer output cannot spill onto storage.
    capacity = 64 * 1024**2
    fd = os.memfd_create('native-observer-log', os.MFD_ALLOW_SEALING)
    os.ftruncate(fd, capacity)
    # Stable Linux UAPI values (linux/fcntl.h); this Python build omits their
    # symbolic exports even though memfd sealing is available in the kernel.
    fcntl.fcntl(fd, 1033, 0x0004)  # F_ADD_SEALS, F_SEAL_GROW
    with os.fdopen(fd, 'r+b') as stream:
        try:
            yield stream
        finally:
            end = stream.tell()
            stream.seek(0)
            fsutil.write_binary(path, stream.read(end))
            if end >= capacity:
                raise RuntimeError('Bounded observer log filled its RAM capacity')


def validate_identity(ready: dict, gate: dict) -> None:
    if (ready['status'] != 'READY' or ready['gate_pid'] != gate['pid'] or
            ready['cgroup_id'] != gate['cgroup_id'] or
            ready['cgroup_path'] != gate['cgroup_path']):
        raise ValueError('Observer attached to a different target service')


def finish_observer(collector: subprocess.Popen, timeout: float) -> None:
    from native_io_iops import collector_group_alive, kill_collector
    try:
        collector.wait(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        kill_collector(collector)
        raise RuntimeError('Observer finalization timed out; reject this trial') from error
    if collector_group_alive(collector.pid):
        kill_collector(collector)
        raise RuntimeError('Observer left an owned process group behind')


def validate_observation(record: dict, row: dict, gate: dict,
                         events: bytes, mode: str, pins: dict, config: dict) -> None:
    # The tools tree supplies its own observer validators; the published
    # candidate's historical collector hashes must never be relabeled.
    if mode == 'queue':
        from native_io_queue_results import validate_collection
    else:
        from native_io_cpu import validate_collection
    validate_collection(record, row, gate, events)
    validate_identity(record['ready'], gate)
    script = 'native_io_queue.py' if mode == 'queue' else 'native_io_cpu.py'
    expected = pins['files'][str(Path(__file__).with_name(script))]
    if record['collector_sha256'] != expected:
        raise ValueError('Observer source differs from the tools source pin')
    if record['event_capacity_bytes'] != config['event_capacity_bytes']:
        raise ValueError('Observer RAM capacity differs from the requested control')
    if mode == 'queue':
        expected = pins['files'][str(Path(__file__).with_name('native_io_queue.c'))]
        if record['bpf_source_sha256'] != expected:
            raise ValueError('BPF source differs from the tools source pin')
        if (record['request_capacity'] != config['request_capacity'] or
                record['device']['path'] != config['device']):
            raise ValueError('Queue observer device/capacity differs from the requested control')
    elif record['frequency_hz_per_cpu'] != config['frequency'] or record['mode'] != mode:
        raise ValueError('Profile frequency/provider differs from the requested control')


def run_service(spec: dict, args: argparse.Namespace, mode: str, pins: dict) -> dict:
    # Imports follow the explicit candidate-path bootstrap in run(). No runtime
    # or observer implementation is copied into, or substituted in, that tree.
    from native_io_iops import (
        phase_rates,
        publish_json,
        service_command,
        stop_service,
        validate_result,
        wait_json,
    )
    paths, observed = spec['iops_paths'], spec['observer_paths']
    unit = 'dg-native-iops-' + uuid.uuid4().hex + '.service'
    collector = service = None
    exit_code = 1
    with ram_log(paths['service_log']) as service_log, ram_log(paths['trace_log']) as observer_log:
        try:
            service = subprocess.Popen(service_command(spec, unit, args.service_user),
                                       stdout=service_log, stderr=subprocess.STDOUT)
            gate = wait_json(paths['gate'], service, args.startup_timeout)
            if mode != 'none':
                collector = subprocess.Popen(observer_command(spec, args, mode),
                                             stdout=observer_log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                ready = wait_json(observed['ready'], collector, args.startup_timeout)
                validate_identity(ready, gate)
                if collector.poll() is not None:
                    raise RuntimeError('Observer exited before payload release')
            durable_artifacts(spec, args.output_dir, args.controller_log)
            boundary = quiet_device(args.device, args.quiet_seconds, args.quiet_timeout)
            publish_json(paths['release'], {'status': 'READY', 'pid': gate['pid'],
                                            'cgroup_id': gate['cgroup_id']})
            deadline = time.monotonic() + spec['timeout'] + 600
            while service.poll() is None:
                if collector is not None and collector.poll() is not None:
                    raise RuntimeError('Observer stopped while the payload was still running')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Observed payload exceeded its owned-service budget')
                time.sleep(.1)
            exit_code = service.wait(timeout=30)
        finally:
            try:
                stop_service(unit)
                if service is not None and service.poll() is None:
                    service.wait(timeout=30)
            finally:
                publish_json(paths['stop'], {'exit_code': exit_code,
                                             'result_path': spec['result']})
                if collector is not None:
                    finish_observer(collector, args.finalize_timeout)
    if exit_code or (collector is not None and collector.returncode):
        raise RuntimeError(f'Trial/observer failed; retained logs: {paths}')
    row = read_json(spec['result'])
    validate_result(row, spec)
    verify_pins(pins)
    if mode != 'none':
        record = read_json(observed['record'])
        validate_observation(record, row, read_json(paths['gate']),
                             fsutil.read_binary(observed['events']), mode, pins,
                             spec['observer_config'])
    row['queue_observer'] = {'status': 'PASS', 'mode': mode, 'paths': observed,
                             'pins': pins, 'service_unit': unit,
                             'gate': read_json(paths['gate']),
                             'pre_payload_quiet': boundary,
                             'application_rates': phase_rates(row, None)}
    if mode != 'none':
        row['queue_observer'].update(record_sha256=file_hash(observed['record']),
                                     events_sha256=file_hash(observed['events']))
    save_json(spec['result'], row)
    return row


def run(args: argparse.Namespace) -> None:
    args.template_results = os.path.abspath(args.template_results)
    args.output_dir = os.path.abspath(args.output_dir)
    args.build_provenance = os.path.abspath(args.build_provenance)
    args.python = os.path.abspath(args.python)  # Preserve the venv symlink.
    candidate = str(Path(args.candidate_root).resolve())
    args.candidate_root = candidate
    tools = str(Path(__file__).resolve().parents[3])
    if candidate == tools:
        raise ValueError('Published candidate and measurement tools must be separate trees')
    if any(Path(args.output_dir).is_relative_to(root) for root in (candidate, tools)):
        raise ValueError('Measurement outputs must stay outside both frozen source trees')
    pins = {'candidate': pin_source(candidate, args.candidate_commit),
            'tools': pin_source(tools, args.tools_commit)}
    bench = str(Path(candidate) / 'packages/kumo-engine/benchmark')
    sys.path.insert(0, bench)
    os.chdir(candidate)
    from native_io_iops import load_templates, new_spec
    from native_io_isolation import clone_prepared, verify_baseline
    from native_io_matrix import (
        source_provenance,
        validate_predictions,
        validate_work,
    )
    from native_io_resources import evict_files, owned_files

    templates = load_templates(args.template_results)
    by_backend = {row['backend']: row for row in templates}
    selected = args.backends.split(',')
    modes = args.observers.split(',')
    if set(selected) - by_backend.keys() or set(modes) - {'none', 'queue', 'cpu', 'offcpu'}:
        raise ValueError('Unknown backend or observer mode')
    build = read_json(args.build_provenance)
    subprocess.run(['git', '-C', candidate, 'merge-base', '--is-ancestor',
                    build['source_git_head'], 'HEAD'], check=True, capture_output=True)
    for path in RUNTIME_PATHS:
        if git(candidate, 'diff', build['source_git_head'], 'HEAD', '--', path):
            raise ValueError(f'Candidate runtime differs from installed build: {path}')
    if (build['native']['sha256'] != args.expected_native_sha256 or
            file_hash(build['native']['path']) != args.expected_native_sha256 or
            file_hash(build['ingest']['path']) != build['ingest']['sha256']):
        raise ValueError('Installed native/ingest does not match the pinned release')
    source = source_provenance(templates[0]['matrix_spec']['runner'])
    source['git_status'] = git(candidate, 'status', '--porcelain')
    if source['git_status']:
        raise ValueError('Published candidate changed while resolving its source pins')
    files = [str(Path(__file__).with_name(name)) for name in SOURCE_FILES]
    files += list(source['file_sha256']) + [
        args.build_provenance, build['native']['path'], build['ingest']['path']]
    files += [str(Path(bench) / name) for name in (
        'native_io_iops.py', 'native_io_phases.py', 'native_io_trace.py')]
    pins['files'] = {path: file_hash(path) for path in files}
    order = observer_order(args.repetitions, selected, modes)
    plan = {'candidate': pins['candidate'], 'tools': pins['tools'],
            'template_results': args.template_results, 'order_seed': 20261109,
            'order': order, 'pins': pins, 'argv': sys.argv, 'arguments': vars(args),
            'workload': 'Unmodified original full-step public fit and predictions',
            'qualification': 'Repeated pairs' if args.repetitions >= 5 else 'Exploratory'}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    validate_controller(args.output_dir, args.controller_log)
    if fsutil.exists(args.output_dir):
        raise FileExistsError('Each experiment must use a fresh output directory')
    fsutil.makedirs(args.output_dir)
    save_json(args.output_dir + '/plan.json', plan)
    rows = []
    args.python = str(Path(args.python).absolute())
    args.steps, args.pilot, args.phase_markers, args.trace_mode = None, False, '1', 'off'
    for repeat, backend, mode in order:
        verify_pins(pins)
        original_output = args.output_dir
        args.output_dir = f'{original_output}/{mode}'
        fsutil.makedirs(args.output_dir, exist_ok=True)
        spec = new_spec(by_backend[backend], args, repeat, source)
        args.output_dir = original_output
        prefix = spec['result'].removesuffix('.json')
        spec['observer_paths'] = {key: prefix + suffix for key, suffix in (
            ('ready', '-observer-ready.json'), ('record', '-observer.json'),
            ('events', '-observer.bin'))}
        spec['observer_config'] = {'mode': mode, 'frequency': args.frequency,
                                   'event_capacity_bytes': args.event_capacity_bytes,
                                   'request_capacity': args.request_capacity, 'device': args.device}
        prepared = spec['prepared_dir']
        if file_hash(prepared + '/manifest.json') != spec['prepared_manifest_sha256']:
            raise ValueError('Prepared manifest differs from the original measured input')
        manifest = read_json(prepared + '/manifest.json')
        expected = {item['path']: item['sha256'] for item in manifest['pristine_owned_files']}
        clone = clone_prepared(prepared, spec['runtime_prepared_dir'], expected_sha256=expected)
        spec['eviction'] = evict_files(owned_files(clone, spec['runtime_prepared_dir']))
        spec['runtime_manifest_sha256'] = file_hash(spec['runtime_prepared_dir'] + '/manifest.json')
        save_json(spec['iops_paths']['spec'], spec)
        try:
            row = run_service(spec, args, mode, pins)
        finally:
            try:
                verify_baseline(clone)
            finally:
                durable_artifacts(spec, args.output_dir, args.controller_log)
        validate_work([by_backend[backend], row])
        validate_predictions([by_backend[backend], row], 1e-6)
        rows.append(row)
        validate_work(rows)
        validate_predictions(rows, 1e-6)
        verify_pins(pins)
        fsutil.rm(spec['runtime_prepared_dir'])
        save_json(args.output_dir + '/progress.json', {
            'completed': len(rows), 'expected': len(order), 'last': spec['result']})
        print(f'PASS {len(rows)}/{len(order)} {repeat} {backend} {mode}', flush=True)
        durable_artifacts(spec, args.output_dir, args.controller_log)
    save_json(args.output_dir + '/completion.json', {
        'status': 'PASS', 'results': [row['matrix_spec']['result'] for row in rows],
        'pins': pins, 'workload_correctness': 'Exact original work and predictions'})
    durable_artifacts(spec, args.output_dir, args.controller_log)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('candidate-root', 'candidate-commit', 'tools-commit', 'template-results',
                   'output-dir', 'build-provenance', 'expected-native-sha256', 'python'):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--controller-log')
    parser.add_argument('--backends', default='mmap,buffered,direct')
    parser.add_argument('--observers', default='none,queue')
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--service-user', default='dmawhirter')
    parser.add_argument('--collector-cpus', default='0-4,10-14')
    parser.add_argument('--device', default='/dev/nvme0n1')
    parser.add_argument('--request-capacity', type=int, default=65536)
    parser.add_argument('--event-capacity-bytes', type=int, default=536870912)
    parser.add_argument('--startup-timeout', type=float, default=180.)
    parser.add_argument('--drain-seconds', type=float, default=10.)
    parser.add_argument('--finalize-timeout', type=float, default=300.)
    parser.add_argument('--quiet-seconds', type=float, default=1.)
    parser.add_argument('--quiet-timeout', type=float, default=30.)
    parser.add_argument('--bcc-python-path', default='/usr/lib/python3/dist-packages')
    parser.add_argument('--frequency', type=int, default=99)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
