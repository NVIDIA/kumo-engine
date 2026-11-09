# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run matched fit-to-prediction trials in fresh, bounded Linux cgroups.

Run from the repository root with the candidate environment and user systemd
bus accessible. No global cache eviction or changes to system policy occur.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from native_io_isolation import clone_prepared, verify_baseline
from native_io_resources import (
    cgroup_path,
    evict_files,
    group_snapshot,
    owned_files,
    warm_files,
)

from io_utils import fsutil

# Reuse the task-aware, near-tie-safe ranking correctness contract.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'test'))
from wheel_gpu_smoke import compare_predictions  # noqa: E402

BACKENDS = ('mmap', 'buffered', 'direct')
TASKS = ('binary', 'regression', 'temporal_link')
REGIMES = ('resident', 'constrained')


def source_provenance(runner: str) -> dict:
    files = [str(Path(__file__).resolve()), runner,
             str(Path(__file__).with_name('native_io_resources.py').resolve()),
             str(Path(__file__).with_name('native_io_isolation.py').resolve())]
    hashes = {path: hashlib.sha256(fsutil.read_binary(path)).hexdigest()
              for path in files}
    git_sha = subprocess.run(['git', 'rev-parse', 'HEAD'], check=True,
                             capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(['git', 'diff', '--binary', 'HEAD'], check=True,
                          capture_output=True).stdout
    return dict(git_sha=git_sha, tracked_diff_sha256=hashlib.sha256(diff).hexdigest(),
                file_sha256=hashes)


def read_json(path: str) -> dict:
    return json.loads(fsutil.read_text(path))


def write_json(path: str, value: dict) -> None:
    fsutil.write_text(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


def trial_order(repetitions: int, seed: int) -> list[tuple[int, str]]:
    """Rotate randomized backend permutations to balance ordinal positions."""
    rng = random.Random(seed)
    permutations = list(itertools.permutations(BACKENDS))
    result = []
    while len(result) < repetitions * len(BACKENDS):
        rng.shuffle(permutations)
        for order in permutations:
            repetition = len(result) // len(BACKENDS)
            if repetition >= repetitions:
                break
            result.extend((repetition, backend) for backend in order)
    return result


def confidence_interval(values: list[float], seed: int) -> list[float]:
    """Percentile bootstrap over paired repetition-level speedups."""
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('Speedups must be finite and positive')
    rng = random.Random(seed)
    medians = sorted(statistics.median(rng.choices(values, k=len(values)))
                     for _ in range(10000))
    return [medians[249], medians[9749]]


def validate_work(rows: list[dict]) -> None:
    if not rows:
        raise ValueError('No completed trials')
    reference = rows[0]
    for row in rows:
        if row['status'] != 'PASS':
            raise ValueError('Incomplete or failed trial')
        logical = row['work'].get('logical_seed_sha256')
        if (not isinstance(logical, str) or len(logical) != 64 or
                any(character not in '0123456789abcdef' for character in logical)):
            raise ValueError('Completed trial has no valid logical training stream hash')
        if logical != reference['work'].get('logical_seed_sha256'):
            raise ValueError('Backends completed different logical training streams')
        fields = ('optimizer_steps', 'train_batches', 'predict_rows',
                  'predict_batches')
        if any(row['work'][key] != reference['work'][key] for key in fields):
            raise ValueError('Backends completed different work')
        if row['config_sha256'] != reference['config_sha256']:
            raise ValueError('Backends used different model/workload plans')
        if (row['provenance']['native_sha256'] !=
                reference['provenance']['native_sha256']):
            raise ValueError('Backends loaded different native candidate builds')
        for phase in ('fit', 'predict', 'e2e'):
            duration = row['phase_seconds'][phase]
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f'Invalid duration for {phase}')


def validate_predictions(rows: list[dict], atol: float) -> None:
    reference = rows[0]['predictions']
    for row in rows:
        predicted = row['predictions']
        if predicted.get('classes') != reference.get('classes'):
            raise ValueError('Backends returned different top-k class identities')
        compare_predictions(reference['values'], predicted['values'],
                            task=row['task'], atol=atol,
                            before_decisions=reference['decisions'],
                            after_decisions=predicted['decisions'])


def training_evidence(row: dict, spec: dict) -> dict:
    interval = row['work']['training_interval']
    before, after = (interval[f'resources_{name}'] for name in ('before', 'after'))
    process_before = before['observer_process']
    process_after = after['observer_process']
    if process_before['pid'] != process_after['pid']:
        raise ValueError('Training I/O snapshots came from different processes')
    physical = (process_after['io']['read_bytes'] -
                process_before['io']['read_bytes'])
    scan = (after['memory_stat'].get('pgscan', 0) -
            before['memory_stat'].get('pgscan', 0))
    evictions = (interval['native_after']['evictions'] -
                 interval['native_before']['evictions'])
    ratio = row['provenance']['index_bytes'] / spec['memory_bytes']
    return dict(
        interval_seconds=interval['seconds'],
        training_pid=process_after['pid'], pid_physical_read_bytes=physical,
        physical_read_scope='training process; excludes unobserved child I/O',
        cgroup_pgscan=scan, native_cache_evictions=evictions,
        index_to_memory_ratio=ratio,
        has_pressure_evidence=(spec['workers'] == 0 and ratio > 2 and
                               physical > 0 and (scan > 0 or evictions > 0)),
    )


def validate_native(row: dict, spec: dict) -> None:
    if row['config']['loader_mode'] != spec['loader_mode']:
        raise ValueError('Runner changed the configured loader mode')
    environment = row['provenance']['environment']
    if environment['DG_DIRECT_CACHE_GB'] != str(spec['direct_cache_gib']):
        raise ValueError('Runner changed the configured native cache budget')
    after = row['native_io']['fit']['after']
    if after['configured_backend'] != spec['backend']:
        raise ValueError('Runner used a different native backend')
    if spec['backend'] != 'mmap':
        if not after['runtime_initialized'] or after['io_ops'] <= 0:
            raise ValueError('Native backend completed no actual I/O operations')
        if after['runtime_rings'] != spec['io_rings']:
            raise ValueError('Native runtime used a different ring count')
    if spec['backend'] == 'direct' and spec['direct_cache_gib']:
        expected = spec['direct_cache_gib'] * 1024**3 // 4096
        if not after['cache_initialized'] or after['cache_capacity_pages'] != expected:
            raise ValueError('Effective native cache capacity differs from requested')
    elif after['cache_capacity_pages'] != 0:
        raise ValueError('Unexpected native cache in an uncached backend')


def reuse_trial(row: dict, spec: dict, expected_native: str) -> None:
    if row.get('status') != 'PASS' or 'matrix_spec' not in row:
        raise ValueError('Existing trial is incomplete; retain it and use a new output')
    previous = row['matrix_spec']
    fields = ('task', 'backend', 'regime', 'repeat', 'prepared_dir', 'memory_bytes',
              'steps', 'batch_size', 'workers', 'threads', 'torch_threads',
              'model_seed', 'python', 'runner', 'direct_cache_gib', 'io_rings',
              'io_qd', 'cpus', 'loader_mode', 'cleanup_work', 'build_provenance_sha256',
              'prepared_manifest_sha256')
    if any(previous[key] != spec[key] for key in fields):
        raise ValueError('Existing trial parameters differ from requested parameters')
    if previous['source_provenance'] != spec['source_provenance']:
        raise ValueError('Existing trial source differs from current source')
    if not expected_native or row['provenance']['native_sha256'] != expected_native:
        raise ValueError('Resume requires the matching --expected-native-sha256')
    verify_baseline({'isolation': row['provenance']['isolation']})
    validate_native(row, spec)


def stop_scope(unit: str) -> None:
    """Stop only our uniquely named unit, including any surviving descendants."""
    try:
        subprocess.run(['systemctl', '--user', 'stop', f'{unit}.scope'],
                       capture_output=True, timeout=30, check=False)
    except subprocess.TimeoutExpired:
        subprocess.run(['systemctl', '--user', 'kill', '--kill-whom=all',
                        '--signal=KILL', f'{unit}.scope'], capture_output=True,
                       timeout=10, check=False)
    state = subprocess.run(['systemctl', '--user', 'is-active', f'{unit}.scope'],
                           capture_output=True, text=True, timeout=10,
                           check=False).stdout.strip()
    if state not in ('inactive', 'failed', 'unknown'):
        raise RuntimeError(f'Benchmark scope did not stop: {unit} ({state})')


def runner_command(spec: dict) -> list[str]:
    command = [spec['python'], spec['runner'], '--task', spec['task'],
               '--backend', spec['backend'], '--prepared-dir',
               spec['runtime_prepared_dir'], '--output', spec['result'],
               '--steps', str(spec['steps']), '--batch-size', str(spec['batch_size']),
               '--workers', str(spec['workers']), '--threads',
               str(spec['torch_threads']), '--rayon-threads', str(spec['threads']),
               '--seed', str(spec['model_seed']), '--repeat', str(spec['repeat']),
               '--regime', spec['regime'], '--loader-mode', spec['loader_mode']]
    if spec['cleanup_work']:
        command.append('--cleanup-work')
    if spec['build_provenance'] is not None:
        command.extend(['--build-provenance', spec['build_provenance']])
    return command


def summarize(rows: list[dict], repetitions: int, seed: int,
              prediction_atol: float = 1e-6) -> dict:
    if not rows:
        raise ValueError('No completed trials')
    summary = {}
    for task in TASKS:
        for regime in REGIMES:
            selected = [row for row in rows
                        if row['task'] == task and row['regime'] == regime]
            if not selected:
                continue
            validate_work(selected)
            validate_predictions(selected, prediction_atol)
            cells = {(row['repeat'], row['backend']): row for row in selected}
            if len(cells) != len(selected):
                raise ValueError('Duplicate trial')
            expected = set(itertools.product(range(repetitions), BACKENDS))
            if set(cells) != expected:
                raise ValueError(f'Incomplete matrix for {task}/{regime}')
            group = {}
            phases_to_report = ['fit', 'predict', 'e2e']
            if all('training' in row['phase_seconds'] for row in selected):
                phases_to_report.append('training')
            for backend in BACKENDS:
                backend_rows = [cells[rep, backend]
                                for rep in range(repetitions)]
                phases = {}
                for phase in phases_to_report:
                    values = [row['phase_seconds'][phase]
                              for row in backend_rows]
                    speedups = [cells[rep, 'mmap']['phase_seconds'][phase] /
                                cells[rep, backend]['phase_seconds'][phase]
                                for rep in range(repetitions)]
                    phases[phase] = {
                        'seconds': values,
                        'median_seconds': statistics.median(values),
                        'paired_speedups': speedups,
                        'median_speedup': statistics.median(speedups),
                        'speedup_bootstrap_95': confidence_interval(
                            speedups, seed),
                    }
                group[backend] = phases
            summary[f'{task}/{regime}'] = group
    return dict(status='PASS', repetitions=repetitions,
                prediction_atol=prediction_atol,
                speedup_definition='mmap / candidate',
                uncertainty='paired percentile bootstrap; exploratory at n=5',
                groups=summary, trials=rows)


def run_cell(spec: dict) -> None:
    """This controller and its child share the fresh systemd scope."""
    prepared = spec['runtime_prepared_dir']
    manifest = read_json(f'{prepared}/manifest.json')
    if (manifest['isolation']['source_manifest_sha256'] !=
            spec['prepared_manifest_sha256']):
        raise ValueError('Private trial copy came from a different baseline')
    files = owned_files(manifest, prepared)
    group = cgroup_path()
    before = group_snapshot(group)
    if (before['memory_max'] != str(spec['memory_bytes']) or
            before['swap_max'] != '0'):
        raise RuntimeError('Requested memory and swap bounds were not applied')
    warm = warm_files(files) if spec['regime'] == 'resident' else None
    before = group_snapshot(group)
    peaks = dict(anon=0, file=0, memory_current=0)
    stop = threading.Event()
    errors: list[str] = []

    def monitor() -> None:
        while not stop.is_set():
            try:
                snapshot = group_snapshot(group)
                peaks['anon'] = max(peaks['anon'],
                                    snapshot['memory_stat'].get('anon', 0))
                peaks['file'] = max(peaks['file'],
                                    snapshot['memory_stat'].get('file', 0))
                peaks['memory_current'] = max(peaks['memory_current'],
                                              snapshot['memory_current'])
            except (OSError, ValueError) as error:
                errors.append(str(error))
                return
            stop.wait(.25)

    observer = threading.Thread(target=monitor, daemon=True)
    observer.start()
    command = runner_command(spec)
    started = time.monotonic()
    failure = None
    try:
        with fsutil.get_binary_handle(spec['log'], 'wb') as stream:
            completed = subprocess.run(command, stdout=stream,
                                       stderr=subprocess.STDOUT,
                                       timeout=spec['timeout'], check=False)
    except subprocess.TimeoutExpired as error:
        failure = str(error)
    finally:
        stop.set()
        observer.join()
    after = group_snapshot(group)
    resources = dict(
        cgroup=group, before=before, after=after, sampled_peaks=peaks,
        sampling_interval_seconds=.25, monitor_errors=errors,
        process_wall_seconds=time.monotonic() - started,
        resident_warmup=warm, eviction=spec['eviction'],
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        physical_read_bytes=(after['io'].get('rbytes', 0) -
                             before['io'].get('rbytes', 0)
                             if after['io_available'] else None),
        cpu_usage_usec=(after['cpu'].get('usage_usec', 0) -
                        before['cpu'].get('usage_usec', 0)),
    )
    write_json(spec['resources'], resources)
    if failure is not None:
        raise RuntimeError(f'Trial timed out: {failure}')
    if completed.returncode:
        raise RuntimeError(f'Trial exited {completed.returncode}; {spec["log"]}')
    if errors:
        raise RuntimeError(f'Cgroup monitor failed: {errors}')
    for event in ('oom', 'oom_kill'):
        if (after['memory_events'].get(event, 0) >
                before['memory_events'].get(event, 0)):
            raise RuntimeError(f'Trial had a cgroup {event} event')
    result = read_json(spec['result'])
    if result['provenance']['isolation'] != manifest['isolation']:
        raise ValueError('Runner did not report the selected private trial copy')
    validate_native(result, spec)
    if not spec['pilot'] and 'training_interval' not in result['work']:
        raise RuntimeError('Formal trial has no actual training interval evidence')
    if 'training_interval' in result['work']:
        result['training_evidence'] = training_evidence(result, spec)
        result['phase_seconds']['training'] = result['training_evidence'][
            'interval_seconds']
        if (not spec['pilot'] and result['phase_seconds']['training'] <
                spec['min_training_seconds']):
            raise RuntimeError('Training interval too short; calibrate more steps')
        if (not spec['pilot'] and spec['regime'] == 'constrained' and
                not result['training_evidence']['has_pressure_evidence']):
            raise RuntimeError('No actual training pressure; recalibrate the workload')
    result.update(regime=spec['regime'], repeat=spec['repeat'],
                  resources=resources, matrix_spec=spec)
    write_json(spec['result'], result)


def run_matrix(args: argparse.Namespace) -> None:
    if args.repetitions < 5 and not args.pilot:
        raise ValueError('Use at least five repetitions or label --pilot')
    if not args.pilot and args.build_provenance is None:
        raise ValueError('Formal trials require --build-provenance')
    output = str(Path(args.output_dir).resolve())
    fsutil.makedirs(output, exist_ok=True)
    script = str(Path(__file__).resolve())
    runner = str(Path(args.runner).resolve())
    source = source_provenance(runner)
    tasks = args.tasks.split(',')
    regimes = args.regimes.split(',')
    if set(tasks) - set(TASKS) or set(regimes) - set(REGIMES):
        raise ValueError('Unknown task or regime')
    rows = []
    for task in tasks:
        for regime in regimes:
            prepared = str(Path(args.prepared_root, task).resolve())
            manifest = read_json(f'{prepared}/manifest.json')
            owned_files(manifest, prepared)
            pinned = manifest.get('pristine_owned_files')
            if not args.pilot and pinned is None:
                raise ValueError('Formal baseline is missing its pristine file oracle')
            expected_sha = (None if pinned is None else
                            {entry['path']: entry['sha256'] for entry in pinned})
            if expected_sha is not None:
                expected_paths = {str(Path(path).relative_to(prepared))
                                  for path in manifest['owned_files']}
                if set(expected_sha) != expected_paths or len(pinned) != len(expected_sha):
                    raise ValueError('Pristine oracle does not cover every owned file')
            memory_gib = (args.resident_memory_gib if regime == 'resident'
                          else args.constrained_memory_gib)
            memory_bytes = memory_gib * 1024**3
            if (regime == 'constrained' and
                    manifest['index_bytes'] <= 2 * memory_bytes):
                raise ValueError('Constrained index must exceed twice its budget')
            for repeat, backend in trial_order(args.repetitions, args.order_seed):
                name = f'{task}-{regime}-{repeat}-{backend}'
                result = f'{output}/{name}.json'
                spec = dict(
                    task=task, regime=regime, repeat=repeat, backend=backend,
                    prepared_dir=prepared, memory_bytes=memory_bytes,
                    runtime_prepared_dir=f'{output}/{name}-prepared',
                    steps=args.steps, batch_size=args.batch_size,
                    workers=args.workers, threads=args.threads,
                    torch_threads=args.torch_threads,
                    loader_mode=args.loader_mode,
                    model_seed=args.model_seed, python=args.python,
                    runner=runner, result=result, log=f'{output}/{name}.log',
                    resources=f'{output}/{name}-resources.json',
                    eviction=None, timeout=args.timeout,
                    direct_cache_gib=args.direct_cache_gib,
                    io_rings=args.io_rings, io_qd=args.io_qd, cpus=args.cpus,
                    source_provenance=source,
                    pilot=args.pilot, min_training_seconds=args.min_training_seconds,
                    cleanup_work=not args.keep_work,
                    build_provenance=args.build_provenance,
                    build_provenance_sha256=(None if args.build_provenance is None
                                             else hashlib.sha256(fsutil.read_binary(
                                                 args.build_provenance)).hexdigest()),
                    prepared_manifest_sha256=hashlib.sha256(
                        fsutil.read_binary(f'{prepared}/manifest.json')).hexdigest(),
                )
                if fsutil.exists(result):
                    if not args.resume:
                        raise FileExistsError(f'Refusing to overwrite trial: {result}')
                    row = read_json(result)
                    reuse_trial(row, spec, args.expected_native_sha256)
                    print(f'REUSE {name}', flush=True)
                    rows.append(row)
                    continue
                clone_manifest = clone_prepared(
                    prepared, spec['runtime_prepared_dir'], expected_sha256=expected_sha)
                clone_files = owned_files(clone_manifest, spec['runtime_prepared_dir'])
                spec['eviction'] = evict_files(clone_files)
                spec['runtime_manifest_sha256'] = hashlib.sha256(
                    fsutil.read_binary(f'{spec["runtime_prepared_dir"]}/manifest.json')
                ).hexdigest()
                spec_file = f'{output}/{name}-spec.json'
                write_json(spec_file, spec)
                environment = {
                    **os.environ, 'DG_EDGE_BACKEND': backend,
                    'DG_DIRECT_CACHE_GB': str(args.direct_cache_gib),
                    'DG_IO_RINGS': str(args.io_rings),
                    'DG_IO_QD': str(args.io_qd),
                    'RAYON_NUM_THREADS': str(args.threads),
                    'OMP_NUM_THREADS': str(args.torch_threads),
                    'OPENBLAS_NUM_THREADS': str(args.torch_threads),
                    'MKL_NUM_THREADS': str(args.torch_threads),
                    'PYTHONHASHSEED': str(args.model_seed),
                }
                unit = 'dg-native-bench-' + uuid.uuid4().hex
                command = [
                    'systemd-run', '--user', '--scope', '--quiet',
                    f'--unit={unit}', '-p', f'MemoryMax={memory_bytes}',
                    '-p', 'MemorySwapMax=0', '-p', 'IOAccounting=yes',
                    '-p', f'RuntimeMaxSec={args.timeout + 600}',
                    '--', 'taskset', '-c', args.cpus,
                    args.python, script, 'cell', '--spec', spec_file,
                ]
                print(f'RUN {name} MemoryMax={memory_gib}GiB', flush=True)
                try:
                    subprocess.run(command, env=environment, check=True,
                                   timeout=args.timeout + 600)
                finally:
                    stop_scope(unit)
                    verify_baseline(clone_manifest)
                row = read_json(result)
                rows.append(row)
                selected = [trial for trial in rows if trial['task'] == task
                            and trial['regime'] == regime]
                validate_work(selected)
                validate_predictions(selected, args.prediction_atol)
                write_json(f'{output}/summary.json',
                           summarize_partial(rows, args))
                fsutil.rm(spec['runtime_prepared_dir'])
    summary = summarize(rows, args.repetitions, args.order_seed,
                        args.prediction_atol)
    summary.update(pilot=args.pilot, arguments=vars(args), source_provenance=source)
    write_json(f'{output}/summary.json', summary)


def summarize_partial(rows: list[dict], args: argparse.Namespace) -> dict:
    return dict(status='RUNNING', pilot=args.pilot, completed=len(rows),
                arguments=vars(args), trials=rows,
                driver_sha256=hashlib.sha256(
                    fsutil.read_binary(str(Path(__file__)))).hexdigest())


def aggregate(args: argparse.Namespace) -> None:
    rows = []
    for path in fsutil.listdir(args.results_dir, suffix='.json'):
        if fsutil.isdir(path) or not path.endswith('.json'):
            continue
        row = read_json(path)
        if 'matrix_spec' in row and 'phase_seconds' in row:
            rows.append(row)
    summary = summarize(rows, args.repetitions, args.order_seed,
                        args.prediction_atol)
    summary['aggregation_driver_sha256'] = hashlib.sha256(
        fsutil.read_binary(str(Path(__file__)))).hexdigest()
    if not fsutil.write_text_if_absent(
            args.output, json.dumps(summary, indent=2, allow_nan=False) + '\n'):
        raise FileExistsError(args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    cell = commands.add_parser('cell')
    cell.add_argument('--spec', required=True)
    combined = commands.add_parser('aggregate')
    combined.add_argument('--results-dir', required=True)
    combined.add_argument('--output', required=True)
    combined.add_argument('--repetitions', type=int, default=5)
    combined.add_argument('--order-seed', type=int, default=20261109)
    combined.add_argument('--prediction-atol', type=float, default=1e-6)
    matrix = commands.add_parser('run')
    matrix.add_argument('--prepared-root', required=True)
    matrix.add_argument('--output-dir', required=True)
    matrix.add_argument('--runner', required=True)
    matrix.add_argument('--build-provenance')
    matrix.add_argument('--keep-work', action='store_true')
    matrix.add_argument('--python', default=sys.executable)
    matrix.add_argument('--tasks', default=','.join(TASKS))
    matrix.add_argument('--regimes', default=','.join(REGIMES))
    matrix.add_argument('--repetitions', type=int, default=5)
    matrix.add_argument('--pilot', action='store_true')
    matrix.add_argument('--resume', action='store_true')
    matrix.add_argument('--expected-native-sha256')
    matrix.add_argument('--order-seed', type=int, default=20261109)
    matrix.add_argument('--model-seed', type=int, default=0)
    matrix.add_argument('--steps', type=int, default=128)
    matrix.add_argument('--batch-size', type=int, default=256)
    matrix.add_argument('--workers', type=int, default=0)
    matrix.add_argument('--loader-mode', choices=('auto', 'standalone'),
                        default='standalone')
    matrix.add_argument('--threads', type=int, default=4)
    matrix.add_argument('--torch-threads', type=int, default=4)
    matrix.add_argument('--cpus', default='5-9,15-19')
    matrix.add_argument('--resident-memory-gib', type=int, default=32)
    matrix.add_argument('--constrained-memory-gib', type=int, default=8)
    matrix.add_argument('--direct-cache-gib', type=int, default=2)
    matrix.add_argument('--io-rings', type=int, default=2)
    matrix.add_argument('--io-qd', type=int, default=32)
    matrix.add_argument('--timeout', type=int, default=3600)
    matrix.add_argument('--prediction-atol', type=float, default=1e-6)
    matrix.add_argument('--min-training-seconds', type=float, default=30.)
    args = parser.parse_args()
    if args.command == 'cell':
        run_cell(read_json(args.spec))
    elif args.command == 'aggregate':
        aggregate(args)
    else:
        for name in ('repetitions', 'steps', 'batch_size', 'threads',
                     'torch_threads', 'resident_memory_gib',
                     'constrained_memory_gib', 'io_rings', 'io_qd', 'timeout'):
            if vars(args)[name] < 1:
                parser.error(f'--{name.replace("_", "-")} must be positive')
        if args.workers < 0 or args.direct_cache_gib < 0:
            parser.error('Workers and native cache size must be nonnegative')
        if not math.isfinite(args.prediction_atol) or args.prediction_atol < 0:
            parser.error('Prediction tolerance must be finite and nonnegative')
        if not 1 <= args.io_rings <= 64 or not 1 <= args.io_qd <= 4096:
            parser.error('Native ring or queue depth would be clamped')
        run_matrix(args)


if __name__ == '__main__':
    main()
