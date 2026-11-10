# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the existing benchmark payload with attributed system-service I/O tracing."""

import argparse
import hashlib
import json
import math
import os
import pwd
import re
import statistics
import subprocess
import time
import uuid
from pathlib import Path

from native_io_isolation import clone_prepared, file_hash, verify_baseline
from native_io_matrix import (
    read_json,
    reuse_trial,
    run_cell,
    source_provenance,
    summarize,
    training_evidence,
    trial_order,
    validate_predictions,
    validate_work,
    write_json,
)
from native_io_resources import cgroup_path, evict_files, owned_files

from io_utils import fsutil


def publish_json(path: str, value: dict) -> None:
    temporary = path + '.writing'
    write_json(temporary, value)
    # fsutil has no atomic rename; these gates are explicitly local Linux files.
    os.replace(temporary, path)


def wait_json(path: str, process: subprocess.Popen | None, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if fsutil.exists(path):
            try:
                return read_json(path)
            except json.JSONDecodeError:
                pass
        if process is not None and process.poll() is not None:
            raise RuntimeError(f'Process exited before gate {path}: {process.returncode}')
        time.sleep(.05)
    raise TimeoutError(f'Timed out waiting for {path}')


def service_command(spec: dict, unit: str, user: str) -> list[str]:
    env = spec['iops_environment']
    command = [
        'sudo', '-n', 'systemd-run', '--quiet', '--wait', '--collect', '--pipe',
        '--service-type=exec', f'--unit={unit}', f'--uid={user}',
        f'--working-directory={Path.cwd()}',
        '-p', f'MemoryMax={spec["memory_bytes"]}', '-p', 'MemorySwapMax=0',
        '-p', 'IOAccounting=yes', '-p', 'KillMode=control-group',
        '-p', f'RuntimeMaxSec={spec["timeout"] + 600}',
    ]
    command.extend(f'--setenv={key}={value}' for key, value in sorted(env.items()))
    return command + [
        '--', 'taskset', '-c', spec['cpus'], spec['python'],
        str(Path(__file__).resolve()), 'cell', '--spec', spec['iops_paths']['spec'],
    ]


def stop_service(unit: str) -> None:
    if not re.fullmatch(r'dg-native-iops-[0-9a-f]{32}\.service', unit):
        raise ValueError('Refuse to stop an unrelated system service')
    try:
        subprocess.run(['sudo', '-n', 'systemctl', 'stop', unit],
                       capture_output=True, timeout=30, check=False)
    except subprocess.TimeoutExpired:
        subprocess.run(['sudo', '-n', 'systemctl', 'kill', '--kill-whom=all',
                        '--signal=KILL', unit], capture_output=True, timeout=10, check=False)
    state = subprocess.run(['sudo', '-n', 'systemctl', 'is-active', unit],
                           capture_output=True, text=True, timeout=30,
                           check=False).stdout.strip()
    if state not in {'inactive', 'failed', 'unknown'}:
        raise RuntimeError(f'System service did not stop: {unit}: {state}')


def ancestor_limits(group: str) -> list[dict]:
    limits = []
    path = Path(group)
    root = Path('/sys/fs/cgroup')
    if root not in path.parents:
        raise ValueError('Target cgroup is outside cgroup v2')
    while path != root:
        values = dict(path=str(path))
        for name in ('memory.max', 'memory.swap.max', 'cpu.max', 'cpuset.cpus.effective'):
            target = str(path / name)
            values[name.replace('.', '_')] = fsutil.read_text(target).strip()
        limits.append(values)
        path = path.parent
    return limits


def cpu_set(text: str) -> set[int]:
    cpus = set()
    for part in text.split(','):
        bounds = part.split('-')
        first = int(bounds[0])
        last = int(bounds[-1])
        if first < 0 or last < first or len(bounds) > 2:
            raise ValueError('Invalid CPU affinity')
        cpus.update(range(first, last + 1))
    return cpus


def run_gated_cell(spec: dict) -> None:
    paths = spec['iops_paths']
    group = cgroup_path()
    if os.getuid() != spec['iops_config']['uid']:
        raise RuntimeError('System service is running as the wrong user')
    if '/system.slice/dg-native-iops-' not in group:
        raise RuntimeError('IOPS payload is not in its requested system service')
    if not fsutil.exists(group + '/io.stat'):
        raise RuntimeError('System service has no cgroup I/O accounting')
    ancestors = ancestor_limits(group)
    if any(item['memory_max'] != 'max' and int(item['memory_max']) < spec['memory_bytes']
           for item in ancestors):
        raise RuntimeError('An ancestor cgroup imposes a lower memory budget')
    cpus = cpu_set(spec['cpus'])
    if set(os.sched_getaffinity(0)) != cpus:
        raise RuntimeError('System service has a different CPU affinity')
    for item in ancestors:
        quota, period = item['cpu_max'].split()
        if (quota != 'max' and int(quota) / int(period) < len(cpus)):
            raise RuntimeError('An ancestor cgroup imposes a lower CPU quota')
        if not cpus <= cpu_set(item['cpuset_cpus_effective']):
            raise RuntimeError('An ancestor cgroup excludes requested CPUs')
    gate = dict(pid=os.getpid(), cgroup_path=group,
                cgroup_id=os.stat(group).st_ino, monotonic_ns=time.perf_counter_ns(),
                spec_path=paths['spec'], prepared_dir=spec['runtime_prepared_dir'],
                ancestor_limits=ancestors)
    publish_json(paths['gate'], gate)
    release = wait_json(paths['release'], None, spec['iops_config']['startup_timeout'])
    if (release['pid'] != gate['pid'] or release['cgroup_id'] != gate['cgroup_id'] or
            release['status'] != 'READY'):
        raise RuntimeError('Trace release belongs to a different service')
    run_cell(spec)


def collector_command(spec: dict) -> list[str]:
    paths, config = spec['iops_paths'], spec['iops_config']
    command = [
        'taskset', '-c', config['collector_cpus'], 'sudo', '-n', spec['python'],
        str(Path(__file__).with_name('native_io_trace.py').resolve()), 'collect',
        '--gate', paths['gate'], '--ready', paths['ready'], '--stop', paths['stop'],
        '--output', paths['trace'], '--request-capacity', str(config['request_capacity']),
        '--drain-seconds', str(config['drain_seconds']),
    ]
    if config['device'] is not None:
        command.extend(['--device', config['device']])
    return command


def validate_ready(ready: dict, gate: dict) -> None:
    if (ready.get('status') != 'READY' or ready.get('gate_pid') != gate['pid'] or
            ready.get('cgroup_id') != gate['cgroup_id']):
        raise RuntimeError('Collector attached to a different service')


def stop_collector(collector: subprocess.Popen, drain_seconds: float) -> None:
    try:
        collector.wait(timeout=drain_seconds + 90)
    except subprocess.TimeoutExpired:
        collector.send_signal(2)
        try:
            collector.wait(timeout=30)
        except subprocess.TimeoutExpired as error:
            # Popen creates this unique session; kill its root collector descendants too.
            subprocess.run(['sudo', '-n', 'kill', '-KILL', '--', f'-{collector.pid}'],
                           capture_output=True, timeout=10, check=False)
            collector.wait(timeout=10)
            raise RuntimeError('Collector required forced termination; reject trial') from error


def validate_trace(spec: dict, row: dict) -> dict:
    from native_io_trace import validate_collection
    trace = read_json(spec['iops_paths']['trace'])
    gate = read_json(spec['iops_paths']['gate'])
    expected = spec['source_provenance']['file_sha256']
    for name, field in [('native_io_trace.py', 'collector_sha256'),
                        ('native_io_trace.c', 'bpf_source_sha256')]:
        path = str(Path(__file__).with_name(name).resolve())
        if trace[field] != expected[path]:
            raise ValueError('Collector source differs from the requested source pin')
    config = spec['iops_config']
    if trace['request_capacity'] != config['request_capacity']:
        raise ValueError('Collector request capacity differs from requested control')
    if config['device'] is not None and trace['device']['path'] != config['device']:
        raise ValueError('Collector device differs from requested physical device')
    validate_collection(trace, row, gate)
    validate_attribution(trace, row)
    return trace


def phase_rates(row: dict, trace: dict | None) -> dict:
    rates = {}
    for phase, interval in row['phase_intervals_ns'].items():
        seconds = (interval['end_ns'] - interval['start_ns']) / 1e9
        if not math.isfinite(seconds) or seconds <= 0 or seconds != row['phase_seconds'][phase]:
            raise ValueError('Workload interval differs from canonical phase duration')
        if phase == 'training':
            training = row['work']['training_interval']
            before, after = training['native_before'], training['native_after']
        elif phase == 'e2e':
            before, after = row['native_io']['fit']['before'], row['native_io']['predict']['after']
        else:
            before, after = row['native_io'][phase]['before'], row['native_io'][phase]['after']
        operations = None if row['backend'] == 'mmap' else after['io_ops'] - before['io_ops']
        item = dict(workload_seconds=seconds,
                    application_completed_read_sqes=operations,
                    application_read_requests_per_second=(operations / seconds
                                                          if operations is not None else None),
                    application_semantics='Native completed Read/Readv SQEs after page coalescing; '
                                          'mmap bypasses this counter',
                    physical=None)
        if trace is not None:
            marks = [mark for mark in trace['markers'] if mark['phase_id'] == {
                'fit': 1, 'training': 2, 'predict': 3, 'repeat': 4, 'reload': 5,
                'e2e': 6}[phase]]
            if (interval['start_ns'] < marks[0]['after_ns'] or
                    interval['end_ns'] > marks[1]['before_ns']):
                raise ValueError('Workload interval is outside its kernel phase markers')
            item['physical'] = dict(
                **trace['phases'][phase],
                workload_start_after_kernel_ns=interval['start_ns'] - marks[0]['after_ns'],
                kernel_end_after_workload_ns=marks[1]['after_ns'] - interval['end_ns'],
                semantics='Final device completions during the BPF marker window; '
                          'its own marker duration is the physical-rate denominator')
        rates[phase] = item
    return rates


def validate_attribution(trace: dict, row: dict) -> None:
    if row['regime'] != 'constrained':
        return
    snapshots = row['phase_io_devices']['training']
    device = trace['device']['io_stat_key']
    before = snapshots['before']['devices'].get(device, {})
    after = snapshots['after']['devices'].get(device, {})
    positive_io = any(after.get(key, 0) > before.get(key, 0) for key in ('rios', 'rbytes'))
    evidence = row['training_evidence']
    if evidence['has_pressure_evidence'] and evidence['pid_physical_read_bytes'] > 0:
        if not positive_io:
            raise ValueError('Constrained training reads lack target-device cgroup accounting')
        if any(trace['counts'][phase]['reads'] <= 0 for phase in ('total', 'training')):
            raise ValueError('Positive constrained cgroup reads have no attributed completions')


def run_service(spec: dict, user: str) -> dict:
    paths = spec['iops_paths']
    config = spec['iops_config']
    unit = 'dg-native-iops-' + uuid.uuid4().hex + '.service'
    collector = None
    service = None
    exit_code = 1
    with fsutil.get_binary_handle(paths['service_log'], 'wb') as service_log, \
            fsutil.get_binary_handle(paths['trace_log'], 'wb') as trace_log:
        try:
            service = subprocess.Popen(service_command(spec, unit, user),
                                       stdout=service_log, stderr=subprocess.STDOUT)
            gate = wait_json(paths['gate'], service, config['startup_timeout'])
            if config['trace_mode'] == 'on':
                collector = subprocess.Popen(collector_command(spec),
                                             stdout=trace_log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                ready = wait_json(paths['ready'], collector, config['startup_timeout'])
                validate_ready(ready, gate)
                if collector.poll() is not None:
                    raise RuntimeError('Collector exited before releasing the payload')
            publish_json(paths['release'], dict(status='READY', pid=gate['pid'],
                                                cgroup_id=gate['cgroup_id']))
            exit_code = service.wait(timeout=spec['timeout'] + 600)
        finally:
            try:
                stop_service(unit)
                if service is not None and service.poll() is None:
                    service.wait(timeout=30)
            finally:
                publish_json(paths['stop'], dict(exit_code=exit_code,
                                                 result_path=spec['result']))
                if collector is not None:
                    stop_collector(collector, config['drain_seconds'])
    if exit_code:
        raise RuntimeError(f'IOPS service failed: {exit_code}; {paths["service_log"]}')
    row = read_json(spec['result'])
    validate_result(row, spec)
    trace = None
    if config['trace_mode'] == 'on':
        if collector is None or collector.returncode:
            raise RuntimeError(f'IOPS collector failed; {paths["trace_log"]}')
        trace = validate_trace(spec, row)
    row['iops'] = dict(status='PASS' if trace is not None else 'CONTROL_NO_TRACE',
                       config=config, service_unit=unit,
                       gate=read_json(paths['gate']), trace=paths['trace'] if trace else None,
                       trace_sha256=file_hash(paths['trace']) if trace else None,
                       phase_rates=phase_rates(row, trace))
    write_json(spec['result'], row)
    return row


def load_templates(directory: str) -> list[dict]:
    rows = []
    for path in sorted(fsutil.listdir(directory, suffix='.json')):
        if fsutil.isdir(path):
            continue
        row = read_json(path)
        if 'matrix_spec' in row and 'phase_seconds' in row:
            rows.append(row)
    if not rows or len({(row['task'], row['regime']) for row in rows}) != 1:
        raise ValueError('Select one completed original task/regime group')
    repetitions = len({row['repeat'] for row in rows})
    summarize(rows, repetitions, 20261109)
    return rows


def new_spec(template: dict, args: argparse.Namespace, repeat: int,
             source: dict) -> dict:
    spec = dict(template['matrix_spec'])
    if (spec['workers'], spec['threads'], spec['torch_threads'], spec['loader_mode']) != (
            0, 4, 1, 'standalone'):
        raise ValueError('IOPS comparison requires workers0/Rayon4/Torch1/standalone')
    expected_memory = (64 if spec['regime'] == 'resident' else 4) * 1024**3
    if spec['memory_bytes'] != expected_memory:
        raise ValueError('IOPS comparison requires the original 64/4GiB limits')
    name = f'{spec["task"]}-{spec["regime"]}-{repeat}-{spec["backend"]}'
    output = str(Path(args.output_dir).resolve())
    paths = {key: f'{output}/{name}-{suffix}' for key, suffix in (
        ('spec', 'spec.json'), ('gate', 'gate.json'), ('ready', 'trace-ready.json'),
        ('release', 'run-ready.json'), ('stop', 'stop.json'), ('trace', 'trace.json'),
        ('service_log', 'service.log'), ('trace_log', 'trace.log'))}
    config = dict(trace_mode=args.trace_mode, phase_markers=args.phase_markers,
                  device=args.device, uid=pwd.getpwnam(args.service_user).pw_uid,
                  request_capacity=args.request_capacity, drain_seconds=args.drain_seconds,
                  startup_timeout=args.startup_timeout, collector_cpus=args.collector_cpus)
    spec.update(repeat=repeat, result=f'{output}/{name}.json',
                log=f'{output}/{name}.log', resources=f'{output}/{name}-resources.json',
                runtime_prepared_dir=f'{output}/{name}-prepared', eviction=None,
                source_provenance=source, pilot=args.pilot, cleanup_work=True,
                iops_config=config, iops_paths=paths)
    if args.steps is not None:
        if args.steps != spec['steps'] and not args.pilot:
            raise ValueError('A changed workload step count requires --pilot')
        spec['steps'] = args.steps
    if args.python is not None:
        spec['python'] = os.path.abspath(args.python)
    spec['expected_native_sha256'] = args.expected_native_sha256
    if args.build_provenance is not None:
        spec['build_provenance'] = str(Path(args.build_provenance).resolve())
        spec['build_provenance_sha256'] = file_hash(spec['build_provenance'])
    environment = {key: value for key, value in os.environ.items()
                   if key in {'PATH', 'PYTHONPATH', 'MPLCONFIGDIR', 'LD_LIBRARY_PATH',
                              'VIRTUAL_ENV', 'RUSTUP_TOOLCHAIN'}}
    environment.update(
        DG_EDGE_BACKEND=spec['backend'], DG_DIRECT_CACHE_GB=str(spec['direct_cache_gib']),
        DG_IO_RINGS=str(spec['io_rings']), DG_IO_QD=str(spec['io_qd']),
        DG_IO_BATCH_MULT=str(spec['io_batch_mult']), RAYON_NUM_THREADS=str(spec['threads']),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        PYTHONHASHSEED=str(spec['model_seed']), DG_IOPS_PHASE_MARKERS=args.phase_markers)
    spec['iops_environment'] = environment
    return spec


def validate_args(args: argparse.Namespace) -> None:
    if args.repetitions < 1 or (args.steps is not None and args.steps < 1):
        raise ValueError('Repetitions and optional steps must be positive')
    durations = (args.drain_seconds, args.startup_timeout)
    if args.request_capacity < 1 or any(not math.isfinite(value) or value <= 0 for value in durations):
        raise ValueError('Collector capacity and timeouts must be positive')
    if not re.fullmatch(r'[0-9a-f]{64}', args.expected_native_sha256):
        raise ValueError('Expected native SHA256 must be explicit')


def validate_resume(spec: dict, row: dict) -> None:
    saved = row.get('iops', {})
    traced = spec['iops_config']['trace_mode'] == 'on'
    expected_status = 'PASS' if traced else 'CONTROL_NO_TRACE'
    if saved.get('config') != spec['iops_config'] or saved.get('status') != expected_status:
        raise ValueError('Resume instrumentation configuration or status differs')
    validate_result(row, spec)
    if traced:
        if (saved.get('trace') != spec['iops_paths']['trace'] or
                saved.get('trace_sha256') != file_hash(spec['iops_paths']['trace'])):
            raise ValueError('Resume trace archive differs')
        trace = validate_trace(spec, row)
    else:
        trace = None
    if saved.get('phase_rates') != phase_rates(row, trace):
        raise ValueError('Resume phase rates differ from raw counter intervals')


def validate_result(row: dict, spec: dict) -> None:
    if row['provenance']['native_sha256'] != spec['expected_native_sha256']:
        raise ValueError('Loaded native library differs from requested SHA256')
    evidence = training_evidence(row, spec)
    if evidence != row['training_evidence']:
        raise ValueError('Saved training eligibility differs from raw snapshots')
    if not spec['pilot']:
        if evidence['interval_seconds'] < max(30, spec['min_training_seconds']):
            raise ValueError('Formal training interval is too short')
        if spec['regime'] == 'constrained' and not evidence['has_pressure_evidence']:
            raise ValueError('Formal training has no measured pressure')


def verify_source(source: dict) -> None:
    current = source_provenance(str(Path(__file__).resolve()))
    status = subprocess.run(['git', 'status', '--porcelain'], check=True,
                            capture_output=True, text=True).stdout
    if (current['git_sha'] != source['git_sha'] or
            current['tracked_diff_sha256'] != source['tracked_diff_sha256'] or
            status != source['git_status'] or
            any(file_hash(path) != digest for path, digest in source['file_sha256'].items())):
        raise ValueError('Benchmark source changed during the IOPS experiment')


def summarize_rates(rows: list[dict]) -> dict:
    summary = {}
    for backend in sorted({row['backend'] for row in rows}):
        selected = [row for row in rows if row['backend'] == backend]
        phases = {}
        for phase in selected[0]['iops']['phase_rates']:
            rates = [row['iops']['phase_rates'][phase] for row in selected]
            app = [rate['application_read_requests_per_second'] for rate in rates]
            physical = [rate['physical']['completed_read_requests_per_second']
                        for rate in rates if rate['physical'] is not None]
            phases[phase] = dict(
                application_read_requests_per_second=(statistics.median(app)
                                                      if all(x is not None for x in app) else None),
                physical_completed_read_requests_per_second=(statistics.median(physical)
                                                             if physical else None))
        summary[backend] = dict(repetitions=len(selected), phase_rate_medians=phases)
    return summary


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    templates = load_templates(args.template_results)
    if args.repetitions < 5 and not args.pilot:
        raise ValueError('Fewer than five pairs require --pilot')
    if (args.trace_mode == 'off' or args.phase_markers == '0') and not args.pilot:
        raise ValueError('Untraced controls cannot be formal IOPS trials')
    if args.trace_mode == 'on' and args.phase_markers != '1':
        raise ValueError('Block tracing requires phase markers')
    runner = templates[0]['matrix_spec']['runner']
    source = source_provenance(runner)
    if args.expected_source_commit is not None and source['git_sha'] != args.expected_source_commit:
        raise ValueError('Current source HEAD differs from requested signed commit')
    if not args.pilot and args.expected_source_commit is None:
        raise ValueError('Formal IOPS trials require --expected-source-commit')
    for name in ('native_io_iops.py', 'native_io_phases.py', 'native_io_trace.py',
                 'native_io_trace.c'):
        path = str(Path(__file__).with_name(name).resolve())
        source['file_sha256'][path] = file_hash(path)
    status = subprocess.run(['git', 'status', '--porcelain'], check=True,
                            capture_output=True, text=True).stdout
    source['git_status'] = status
    if not args.pilot and status:
        raise ValueError('Formal IOPS trials require committed, clean source')
    for template in templates:
        original = template['matrix_spec']['source_provenance']['file_sha256']
        if any(file_hash(path) != digest for path, digest in original.items()
               if path != template['matrix_spec']['runner']):
            raise ValueError('Canonical replay validator changed since original measurement')
    build = read_json(args.build_provenance or templates[0]['matrix_spec']['build_provenance'])
    if build['native']['sha256'] != args.expected_native_sha256:
        raise ValueError('Requested native pin differs from build provenance')
    subprocess.run(['git', 'merge-base', '--is-ancestor', build['source_git_head'], 'HEAD'],
                   check=True, capture_output=True)
    diff = subprocess.run(['git', 'diff', build['source_git_head'], '--',
                           'packages/diskgraph'], capture_output=True, check=True).stdout
    if diff:
        raise ValueError('Native source differs from the pinned installed build')
    mapped_paths = ['packages/kumo-core/src', 'packages/kumo-engine/kumo/engine',
                    'packages/kumo-engine/test/wheel_gpu_smoke.py']
    runtime_diff = subprocess.run(
        ['git', 'diff', build['source_git_head'], '--', *mapped_paths],
        capture_output=True, check=True).stdout
    if runtime_diff:
        raise ValueError('Engine/Core runtime differs from the pinned build source')
    source['build_compatibility'] = dict(
        native_build_commit=build['source_git_head'], source_head=source['git_sha'],
        native_tree_diff_sha256=hashlib.sha256(diff).hexdigest(),
        runtime_paths=mapped_paths,
        runtime_diff_sha256=hashlib.sha256(runtime_diff).hexdigest(),
        benchmark_changes='Explicit phase markers, intervals and system-service collector',
        original_runner_sha256=templates[0]['matrix_spec'][
            'source_provenance']['file_sha256'][runner],
        current_runner_sha256=file_hash(runner))
    fsutil.makedirs(str(Path(args.output_dir).resolve()), exist_ok=True)
    by_backend = {row['backend']: row for row in templates}
    rows = []
    for repeat, backend in trial_order(args.repetitions, 20261109):
        verify_source(source)
        spec = new_spec(by_backend[backend], args, repeat, source)
        if fsutil.exists(spec['result']):
            if not args.resume:
                raise FileExistsError(spec['result'])
            row = read_json(spec['result'])
            reuse_trial(row, spec, args.expected_native_sha256)
            validate_resume(spec, row)
            verify_source(source)
            if spec['steps'] == by_backend[backend]['matrix_spec']['steps']:
                validate_work([by_backend[backend], row])
                validate_predictions([by_backend[backend], row], 1e-6)
            rows.append(row)
            continue
        prepared = spec['prepared_dir']
        manifest = read_json(prepared + '/manifest.json')
        if file_hash(prepared + '/manifest.json') != spec['prepared_manifest_sha256']:
            raise ValueError('Prepared input no longer matches the measured original')
        expected = {item['path']: item['sha256'] for item in manifest['pristine_owned_files']}
        clone = clone_prepared(prepared, spec['runtime_prepared_dir'], expected_sha256=expected)
        spec['eviction'] = evict_files(owned_files(clone, spec['runtime_prepared_dir']))
        spec['runtime_manifest_sha256'] = file_hash(spec['runtime_prepared_dir'] + '/manifest.json')
        write_json(spec['iops_paths']['spec'], spec)
        try:
            row = run_service(spec, args.service_user)
            verify_source(source)
        finally:
            verify_baseline(clone)
        rows.append(row)
        if spec['steps'] == by_backend[backend]['matrix_spec']['steps']:
            validate_work([by_backend[backend], row])
            validate_predictions([by_backend[backend], row], 1e-6)
        validate_work(rows)
        validate_predictions(rows, 1e-6)
        fsutil.rm(spec['runtime_prepared_dir'])
    summary = summarize(rows, args.repetitions, 20261109)
    summary['iops_config'] = spec['iops_config']
    summary['iops_rate_summary'] = dict(
        estimator='Median of each trial count divided by its own measured interval',
        backends=summarize_rates(rows))
    write_json(str(Path(args.output_dir).resolve() / 'summary.json'), summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='command', required=True)
    cell = modes.add_parser('cell')
    cell.add_argument('--spec', required=True)
    matrix = modes.add_parser('run')
    matrix.add_argument('--template-results', required=True)
    matrix.add_argument('--output-dir', required=True)
    matrix.add_argument('--expected-native-sha256', required=True)
    matrix.add_argument('--expected-source-commit')
    matrix.add_argument('--build-provenance')
    matrix.add_argument('--python')
    matrix.add_argument('--steps', type=int)
    matrix.add_argument('--repetitions', type=int, default=5)
    matrix.add_argument('--pilot', action='store_true')
    matrix.add_argument('--resume', action='store_true')
    matrix.add_argument('--trace-mode', choices=('on', 'off'), default='on')
    matrix.add_argument('--phase-markers', choices=('0', '1'), default='1')
    matrix.add_argument('--service-user', default='dmawhirter')
    matrix.add_argument('--collector-cpus', default='0-4,10-14')
    matrix.add_argument('--device')
    matrix.add_argument('--request-capacity', type=int, default=8192)
    matrix.add_argument('--drain-seconds', type=float, default=10.)
    matrix.add_argument('--startup-timeout', type=float, default=180.)
    args = parser.parse_args()
    if args.command == 'cell':
        run_gated_cell(read_json(args.spec))
    else:
        run(args)


if __name__ == '__main__':
    main()
