# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finite, read-only fio sweeps; an observed path plateau is not an SSD rating.

Planning is inert. Execution requires --execute and a clean, pinned checkout.
fio stdout and observer samples remain in memory until the workload exits.
Kernel device samples include other host users and have their own clock windows.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict
from typing import TypeAlias

from io_utils import fsutil

Scalar: TypeAlias = str | int | float | bool | None
Json: TypeAlias = Scalar | list['Json'] | dict[str, 'Json']
Record: TypeAlias = dict[str, Json]

FIO = '/usr/bin/fio'
FIXTURE_BYTES = 8 * 1024**3
FIXTURE_SHA256 = 'f6e39fcf6fdc5fe4b98b4ec2c43d21760b3a7abf14df7812498ee7ba338f8d14'
PERF_CPUS = '5-9,15-19'
OBSERVER_CPUS = '0-4,10-14'
GEOMETRIES = ((1, 1), (1, 4), (1, 16), (1, 64), (1, 256), (4, 64), (4, 256))
PROTOCOL: Record = {
    'version': 1, 'repetitions': 5, 'block_bytes': [4096, 8192],
    'ramp_seconds': 10, 'measure_seconds': 30, 'sample_seconds': 1,
    'material_gain': 0.05, 'relative_mad_limit': 0.05, 'issuer_cpu_percent': 90,
    'extension': '8x256 for a block size if 4x256 gains >5% over 4x64 or has an issuer >=90% CPU',
    'batch_extension': 'Under the same trigger, 4x256 submit16/reap32; five reps, distinct from unbatched fio',
    'ceiling_scope': 'Observed read-only filesystem/device path; not an SSD firmware or flash rating',
}


def record(value: Json) -> Record:
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object')
    return value


def number(value: Json) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected finite number')
    return float(value)


def integer(value: Json) -> int:
    if type(value) is not int:
        raise ValueError('Expected integer')
    return value


def digest(path: str) -> str:
    result = hashlib.sha256()
    with fsutil.get_binary_handle(path, 'rb') as handle:
        while chunk := handle.read(8 * 1024**2):
            result.update(chunk)
    return result.hexdigest()


def save(path: str, value: Record) -> None:
    fsutil.write_text(path, json.dumps(value, indent=2, allow_nan=False) + '\n')
    # Finish only our artifact writes before a subsequent timed trial. Closing
    # a buffered file alone can defer observer writeback into that next trial.
    with fsutil.get_binary_handle(path, 'rb') as handle:
        os.fsync(handle.fileno())
    # fsutil has no directory-fd durability interface for these local artifacts.
    directory = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@dataclasses.dataclass(frozen=True)
class TrialSpec:
    block_bytes: int
    jobs: int
    depth: int
    repeat: int
    ramp_seconds: int = 10
    measure_seconds: int = 30
    submit_batch: int = 1
    complete_batch: int = 1

    def __post_init__(self) -> None:
        if (any(type(value) is not int for value in dataclasses.asdict(self).values())
                or self.block_bytes not in (4096, 8192) or (self.jobs, self.depth) not in (*GEOMETRIES, (8, 256))
                or self.repeat not in range(5)
                or self.ramp_seconds != 10 or self.measure_seconds != 30):
            raise ValueError('Trial is outside the finite protocol')
        if (self.submit_batch, self.complete_batch) != (1, 1):
            if (self.jobs, self.depth, self.submit_batch, self.complete_batch) != (4, 256, 16, 32):
                raise ValueError('Unsupported optimized issuer configuration')

    @property
    def name(self) -> str:
        return (f'b{self.block_bytes}-j{self.jobs}-q{self.depth}'
                f'-s{self.submit_batch}-c{self.complete_batch}-r{self.repeat}')


def trial_plan() -> list[TrialSpec]:
    """Rotate and reverse configuration order; every coordinate appears once."""
    configurations = [(block, jobs, depth) for block in (4096, 8192) for jobs, depth in GEOMETRIES]
    result = []
    for repeat in range(5):
        order = configurations[repeat:] + configurations[:repeat]
        if repeat % 2:
            order = list(reversed(order))
        result.extend(TrialSpec(*configuration, repeat) for configuration in order)
    return result


def fixture_info(path: str) -> Record:
    # POSIX identity/allocation are required here; fsutil has no inode interface.
    info = os.stat(path, follow_symlinks=False)
    if not os.path.isabs(path) or not os.path.isfile(path) or os.path.islink(path):
        raise ValueError('Fixture must be an existing absolute regular file, not a symlink')
    if info.st_size != FIXTURE_BYTES or info.st_blocks * 512 < info.st_size or info.st_uid != os.getuid():
        raise ValueError('Expected owned, non-sparse 8GiB fixture')
    return {'path': path, 'device': info.st_dev, 'inode': info.st_ino, 'size': info.st_size,
            'allocated_bytes': info.st_blocks * 512, 'uid': info.st_uid,
            'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns}


def fio_command(spec: TrialSpec, fixture: str) -> list[str]:
    return [FIO, '--name=ceiling-read', '--filename=' + fixture, '--readonly', '--allow_file_create=0',
            '--ioengine=io_uring', '--direct=1', '--rw=randread', '--size=8G',
            f'--bs={spec.block_bytes}', f'--numjobs={spec.jobs}', f'--iodepth={spec.depth}',
            '--cpus_allowed=' + PERF_CPUS, '--cpus_allowed_policy=split',
            '--time_based=1', f'--ramp_time={spec.ramp_seconds}s', f'--runtime={spec.measure_seconds}s',
            '--randrepeat=1', f'--randseed={20261117 + spec.repeat}', '--norandommap=1',
            '--verify=0', '--invalidate=0', '--clocksource=clock_gettime',
            f'--iodepth_batch_submit={spec.submit_batch}', '--iodepth_batch_complete_min=1',
            f'--iodepth_batch_complete_max={spec.complete_batch}',
            '--group_reporting=0', '--output-format=json+', '--status-interval=1',
            '--exitall_on_error=1', '--eta=never']


def parse_reports(text: str) -> list[Record]:
    decoder = json.JSONDecoder()
    reports = []
    remaining = text.strip()
    while remaining:
        value, end = decoder.raw_decode(remaining)
        reports.append(record(value))
        remaining = remaining[end:].lstrip()
    if not reports:
        raise ValueError('fio emitted no JSON reports')
    return reports


def read_counts(read: Record, block_bytes: int, depth: int, ramp_seconds: int) -> Record:
    """fio 3.36 counts issues and completions on opposite sides of ramp reset."""
    issued = integer(read['total_ios'])
    size = integer(read['io_bytes'])
    latency = integer(record(read['clat_ns'])['N'])
    bound = depth if ramp_seconds else 0
    if issued <= 0 or size <= 0 or size % block_bytes or latency <= 0:
        raise ValueError('Invalid fixed-size read counters')
    completed = size // block_bytes
    # ioengines.c counts issues; io_u.c counts completed bytes/latencies.
    # time.c resets ramp statistics without draining at most one queue depth.
    if (not 0 <= completed - issued <= bound or abs(latency - issued) > bound
            or abs(completed - latency) > bound):
        raise ValueError('Read counter difference exceeds ramp queue capacity')
    bins = record(record(read['clat_ns'])['bins'])
    if any(integer(value) < 0 for value in bins.values()) or sum(integer(value) for value in bins.values()) != latency:
        raise ValueError('Completion histogram/sample count mismatch')
    return {'issued_reads': issued, 'completed_reads': completed, 'latency_samples': latency,
            'completed_minus_issued': completed - issued, 'latency_minus_issued': latency - issued,
            'completed_minus_latency': completed - latency, 'ramp_boundary_bound_reads': bound,
            'completed_read_bytes': size}


def parse_fio(report: Record, spec: TrialSpec) -> Record:
    jobs = report.get('jobs')
    if not isinstance(jobs, list) or len(jobs) != spec.jobs:
        raise ValueError('fio job count differs from protocol')
    iops = bandwidth = outstanding = 0.0
    stability = []
    cpu, runtimes, per_job = [], [], []
    for job_value in jobs:
        job = record(job_value)
        read = record(job['read'])
        options = record(job['job options'])
        if integer(job['error']) or integer(read['short_ios']) or integer(read['drop_ios']):
            raise ValueError('fio reported failed, short or dropped I/O')
        if options['rw'] != 'randread' or options['direct'] != '1' or options['ioengine'] != 'io_uring':
            raise ValueError('Unexpected fio access path')
        if int(str(options['bs'])) != spec.block_bytes or int(str(options['iodepth'])) != spec.depth:
            raise ValueError('fio geometry changed')
        seconds = number(read['runtime']) / 1000
        counts = read_counts(read, spec.block_bytes, spec.depth, spec.ramp_seconds)
        count = integer(counts['issued_reads'])
        size = integer(counts['completed_read_bytes'])
        if seconds < spec.measure_seconds * 0.98:
            raise ValueError('Incomplete timed read window')
        for operation in ('write', 'trim'):
            values = record(job[operation])
            if integer(values['total_ios']) or integer(values['io_bytes']):
                raise ValueError('Read-only trial performed writes or discards')
        rate = number(read['iops'])
        if not math.isclose(rate, count / seconds, rel_tol=1e-5):
            raise ValueError('fio rate/count/window mismatch')
        if not math.isclose(number(read['bw_bytes']), size / seconds, rel_tol=1e-5):
            raise ValueError('fio bandwidth/count/window mismatch')
        clat = record(read['clat_ns'])
        if number(clat['mean']) <= 0:
            raise ValueError('Incomplete completion-latency observations')
        usage = number(job['usr_cpu']) + number(job['sys_cpu'])
        if not 0 <= usage <= 105:
            raise ValueError('Invalid fio per-job CPU usage')
        cpu.append(usage)
        if integer(read['bw_samples']) < 20 or number(read['bw_mean']) <= 0:
            raise ValueError('Insufficient measured-window bandwidth observations')
        variation = number(read['bw_dev']) / number(read['bw_mean'])
        if variation < 0:
            raise ValueError('Invalid bandwidth dispersion')
        stability.append(variation)
        runtimes.append(seconds)
        iops += rate
        bandwidth += size / seconds
        outstanding += integer(counts['latency_samples']) / seconds * number(clat['mean']) / 1e9
        per_job.append({'iops': rate, 'seconds': seconds, 'cpu_percent': usage,
                        'completion_latency_ns': clat, 'submission_latency_ns': read['slat_ns'],
                        'depth_buckets': job['iodepth_level'],
                        'bandwidth_samples': read['bw_samples'], 'bandwidth_mean_kib': read['bw_mean'],
                        'bandwidth_stddev_kib': read['bw_dev'], 'read_counts': counts})
    if max(runtimes) - min(runtimes) > 1:
        raise ValueError('Job windows are not sufficiently aligned')
    return {'application_read_iops': iops, 'read_bytes_per_second': bandwidth,
            'estimated_application_outstanding': outstanding,
            'read_counter_semantics': 'fio total_ios/iops count issued reads; io_bytes count completed bytes; ramp reset can retain at most one per-job queue depth of boundary reads; counters are preserved separately',
            'outstanding_semantics': 'Sum of per-job latency-sample rate * mean fio completion latency; approximate across the bounded ramp boundary; not device queue occupancy',
            'issuer_cpu_percent': cpu, 'issuer_max_cpu_percent': max(cpu), 'per_job': per_job,
            'max_job_bandwidth_cv': max(stability),
            'physical_iops_semantics': 'fio counts application reads; kernel device samples are separately device-wide'}


def aggregate(rows: list[Record]) -> list[Record]:
    grouped: dict[tuple[int, ...], list[Record]] = defaultdict(list)
    identities = set()
    for row in rows:
        spec = record(row['spec'])
        identity = tuple(integer(spec[key]) for key in ('block_bytes', 'jobs', 'depth',
                                                        'submit_batch', 'complete_batch', 'repeat'))
        if identity in identities:
            raise ValueError('Duplicate trial coordinate')
        identities.add(identity)
        grouped[identity[:-1]].append(record(row['metrics']) | {'repeat': integer(spec['repeat'])})
    summaries = []
    for (block, jobs, depth, submit, complete), samples in sorted(grouped.items()):
        if len(samples) != 5:
            raise ValueError('Every configuration requires exactly five repetitions')
        rates = [number(sample['application_read_iops']) for sample in samples]
        center = statistics.median(rates)
        relative_mad = statistics.median(abs(rate - center) for rate in rates) / center
        summaries.append({'block_bytes': block, 'jobs': jobs, 'depth': depth,
                          'submit_batch': submit, 'complete_batch': complete,
                          'median_application_read_iops': center,
                          'min_application_read_iops': min(rates), 'max_application_read_iops': max(rates),
                          'median_read_mib_per_second': statistics.median(
                              number(sample['read_bytes_per_second']) / 1024**2 for sample in samples),
                          'relative_mad': relative_mad,
                          'max_trial_bandwidth_cv': max(number(sample['max_job_bandwidth_cv']) for sample in samples),
                          'per_trial_read_counts': [
                              {'repeat': sample['repeat'],
                               'per_job': [record(job)['read_counts'] for job in sample['per_job']]}
                              for sample in sorted(samples, key=lambda sample: integer(sample['repeat']))],
                          'median_max_issuer_cpu_percent': statistics.median(
                              number(sample['issuer_max_cpu_percent']) for sample in samples)})
    return summaries


def extension_blocks(summaries: list[Record]) -> list[int]:
    result = []
    for block in (4096, 8192):
        group = {(integer(row['jobs']), integer(row['depth'])): row
                 for row in summaries if row['block_bytes'] == block and row['submit_batch'] == 1}
        if not all(key in group for key in GEOMETRIES):
            raise ValueError('Adaptive decision requires the complete base matrix')
        high, low = group[(4, 256)], group[(4, 64)]
        gain = number(high['median_application_read_iops']) / number(low['median_application_read_iops']) - 1
        if gain > 0.05 or number(high['median_max_issuer_cpu_percent']) >= 90:
            result.append(block)
    return result


def classify_ceiling(summaries: list[Record]) -> list[Record]:
    result = []
    for block in (4096, 8192):
        group = {(integer(row['jobs']), integer(row['depth'])): row
                 for row in summaries if row['block_bytes'] == block and row['submit_batch'] == 1}
        single, parallel, high = group[(1, 256)], group[(4, 64)], group[(4, 256)]
        issuer_gain = number(parallel['median_application_read_iops']) / number(single['median_application_read_iops']) - 1
        previous = high if (8, 256) in group else parallel
        last = group.get((8, 256), high)
        last_gain = number(last['median_application_read_iops']) / number(previous['median_application_read_iops']) - 1
        gain_range = [number(last['min_application_read_iops']) / number(previous['max_application_read_iops']) - 1,
                      number(last['max_application_read_iops']) / number(previous['min_application_read_iops']) - 1]
        unstable = any(number(row['relative_mad']) > 0.05 or number(row['max_trial_bandwidth_cv']) > 0.10
                       for row in group.values())
        cpu_busy = number(last['median_max_issuer_cpu_percent']) >= 90
        category = ('INCONCLUSIVE_VARIABILITY' if unstable else 'ISSUER_BUSY' if cpu_busy else
                    'THROUGHPUT_STILL_GROWING' if last_gain > 0.05 else
                    'SCALING_REGRESSION' if last_gain < -0.05 else 'STORAGE_PATH_PLATEAU')
        optimized = [row for row in summaries if row['block_bytes'] == block and row['submit_batch'] == 16]
        result.append({'block_bytes': block, 'classification': category,
                       'same_nominal_depth_issuer_gain': issuer_gain, 'last_scaling_gain': last_gain,
                       'last_gain_trial_range_not_ci': gain_range,
                       'five_percent_decision_weak': gain_range[0] <= 0.05 <= gain_range[1],
                       'optimized_issuer_gain': (number(optimized[0]['median_application_read_iops'])
                                                 / number(high['median_application_read_iops']) - 1
                                                 if optimized else None),
                       'single_issuer_limitation_supported': issuer_gain > 0.05,
                       'hardware_ceiling_proven': False,
                       'threshold_qualification': 'Five-repeat 5% decisions are protocol triggers, not significance tests; inspect all raw variation',
                       'qualification': 'A stable path plateau alone cannot separate controller/media limits from kernel-path limits'})
    return result


def command_output(command: list[str], timeout: int = 20) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout).stdout


def source_info(repo: str, expected: str) -> Record:
    head = command_output(['git', '-C', repo, 'rev-parse', 'HEAD']).strip()
    dirty = command_output(['git', '-C', repo, 'status', '--porcelain']).strip()
    if head != expected or dirty:
        raise ValueError('Execution requires the exact clean committed source')
    return {'head': head, 'dirty': False, 'driver_sha256': digest(os.path.abspath(__file__)),
            'fio_sha256': digest(FIO), 'fio_version': command_output([FIO, '--version']).strip(),
            'kernel': os.uname().release, 'protocol': PROTOCOL}


def thermal_info() -> Record:
    return record(json.loads(command_output(['sudo', '-n', 'nvme', 'smart-log', '/dev/nvme0', '-o', 'json'])))


def check_thermal(before: Record, after: Record) -> None:
    for key in ('critical_warning', 'media_errors', 'thm_temp1_trans_count', 'thm_temp2_trans_count'):
        if integer(after[key]) != integer(before[key]) or (key == 'critical_warning' and integer(after[key])):
            raise ValueError(f'NVMe health/thermal gate changed: {key}')


def kernel_snapshot() -> Record:
    return {'monotonic_ns': time.monotonic_ns(), 'realtime_ns': time.time_ns(),
            'device_stat': fsutil.read_text('/sys/block/nvme0n1/stat'),
            'cpu_stat': fsutil.read_text('/proc/stat'),
            'io_pressure': fsutil.read_text('/proc/pressure/io'),
            'cpu_pressure': fsutil.read_text('/proc/pressure/cpu'),
            'memory_pressure': fsutil.read_text('/proc/pressure/memory')}


def host_info() -> Record:
    paths = ['/proc/sys/kernel/random/boot_id', '/proc/meminfo', '/proc/sys/kernel/perf_event_paranoid',
             '/sys/block/nvme0n1/device/model', '/sys/block/nvme0n1/device/firmware_rev',
             '/sys/block/nvme0n1/device/device/current_link_speed',
             '/sys/block/nvme0n1/device/device/current_link_width',
             '/sys/block/nvme0n1/device/queue_count', '/sys/block/nvme0n1/device/sqsize',
             '/sys/block/nvme0n1/queue/max_sectors_kb', '/sys/block/nvme0n1/queue/scheduler']
    return {'kernel': os.uname().release, 'machine': os.uname().machine,
            'files': {path: fsutil.read_text(path).strip() for path in paths},
            'performance_cpus': PERF_CPUS, 'observer_cpus': OBSERVER_CPUS,
            'idle_reference': kernel_snapshot(), 'smart': thermal_info()}


def cgroup_info() -> Record:
    relative = fsutil.read_text('/proc/self/cgroup').strip().split('::', 1)[1]
    path = '/sys/fs/cgroup' + relative
    ancestors = []
    current = path
    while current.startswith('/sys/fs/cgroup'):
        values: Record = {'path': current}
        for name in ('memory.max', 'memory.swap.max', 'cpu.max', 'cpuset.cpus.effective', 'io.max'):
            if fsutil.exists(current + '/' + name):
                values[name] = fsutil.read_text(current + '/' + name).strip()
        ancestors.append(values)
        memory = values.get('memory.max', 'max')
        if memory != 'max' and int(str(memory)) < 64 * 1024**3:
            raise ValueError('Ancestor memory limit is below 64GiB')
        quota = str(values.get('cpu.max', 'max')).split()[0]
        if quota != 'max':
            raise ValueError('CPU quota would confound the ceiling measurement')
        if values.get('io.max', ''):
            raise ValueError('Ancestor I/O limits must be reviewed before a ceiling trial')
        if current == '/sys/fs/cgroup':
            break
        current = os.path.dirname(current)
    if fsutil.read_text(path + '/memory.max').strip() != str(64 * 1024**3):
        raise ValueError('Expected 64GiB trial cgroup')
    if fsutil.read_text(path + '/memory.swap.max').strip() != '0':
        raise ValueError('Expected disabled cgroup swap')
    return {'path': path, 'id': os.stat(path).st_ino, 'ancestors': ancestors,
            'memory_events_before': fsutil.read_text(path + '/memory.events')}


def trial_snapshot(group_path: str) -> Record:
    snapshot = kernel_snapshot()
    snapshot['cgroup_cpu'] = fsutil.read_text(group_path + '/cpu.stat')
    snapshot['cgroup_io'] = fsutil.read_text(group_path + '/io.stat')
    snapshot['cgroup_memory'] = fsutil.read_text(group_path + '/memory.current').strip()
    processes = []
    for raw_pid in fsutil.read_text(group_path + '/cgroup.procs').split():
        pid = int(raw_pid)
        try:
            processes.append({'pid': pid, 'affinity': sorted(os.sched_getaffinity(pid)),
                              'stat': fsutil.read_text(f'/proc/{pid}/stat'),
                              'io': fsutil.read_text(f'/proc/{pid}/io')})
        except FileNotFoundError:
            # A completed fio child can disappear between cgroup.procs and stat.
            continue
    snapshot['processes'] = processes
    return snapshot


def execute_child(spec: TrialSpec, fixture: str) -> Record:
    os.sched_setaffinity(0, {0})
    group = cgroup_info()
    before = fixture_info(fixture)
    command = fio_command(spec, fixture)
    samples: list[Record] = []
    errors: list[str] = []
    stop = threading.Event()

    def sample() -> None:
        # Observer userspace stays on an efficiency CPU, including the caller.
        os.sched_setaffinity(0, {0})
        while not stop.is_set():
            try:
                samples.append(trial_snapshot(str(group['path'])))
            except (OSError, ValueError) as error:
                errors.append(str(error))
                return
            stop.wait(1)

    observer = threading.Thread(target=sample, daemon=True)
    observer.start()
    process = None
    stdout, stderr, execution_error = '', '', None
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        stdout, stderr = process.communicate(timeout=spec.ramp_seconds + spec.measure_seconds + 60)
    except BaseException as error:
        execution_error = f'{type(error).__name__}: {error}'
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process is not None:
            try:
                # communicate retains earlier buffered output after a timeout.
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired as drain_error:
                raw_stdout, raw_stderr = drain_error.output or b'', drain_error.stderr or b''
                stdout = raw_stdout.decode(errors='replace') if isinstance(raw_stdout, bytes) else raw_stdout
                stderr = raw_stderr.decode(errors='replace') if isinstance(raw_stderr, bytes) else raw_stderr
                execution_error += '; failed to drain killed fio pipes'
    finally:
        stop.set()
        observer.join(timeout=5)
    samples.append(trial_snapshot(str(group['path'])))
    after = fixture_info(fixture)
    group_path = str(group['path'])
    events = fsutil.read_text(group_path + '/memory.events')
    row: Record = {'spec': dataclasses.asdict(spec), 'fio_command': command,
                   'fio_pid': process.pid if process is not None else None,
                   'fio_returncode': process.returncode if process is not None else None,
                   'execution_error': execution_error,
                   'observer_pid': os.getpid(), 'fio_stdout': stdout, 'fio_stderr': stderr,
                   'kernel_samples': samples, 'fixture_before': before, 'fixture_after': after,
                   'cgroup': group, 'memory_events_after': events, 'sampling_errors': errors,
                   'window_semantics': 'fio rates exclude its ramp; raw host/device samples span process setup+ramp+measurement, not an identical window'}
    try:
        if execution_error or process is None or errors or observer.is_alive() or process.returncode:
            raise ValueError(f'fio/observer failed: {execution_error}, {errors}')
        if before != after:
            raise ValueError('Read-only fixture identity changed')
        if any(int(value) for key, value in (line.split() for line in events.splitlines()) if key in ('oom', 'oom_kill')):
            raise ValueError('Trial cgroup OOM')
        reports = parse_reports(stdout)
        row.update({'fio_reports': reports, 'metrics': parse_fio(reports[-1], spec), 'status': 'PASS'})
    except ValueError as error:
        row.update({'status': 'FAIL', 'error': str(error)})
    return row


def crc_command(fixture: str) -> list[str]:
    return [FIO, '--name=fixture-crc', '--filename=' + fixture, '--readonly', '--allow_file_create=0',
            '--ioengine=io_uring', '--direct=1', '--rw=read', '--bs=4096', '--size=8G',
            '--iodepth=256', '--verify=crc32c', '--verify_only=1', '--verify_fatal=1',
            '--verify_state_save=0', '--output-format=json', '--eta=never']


def service_command(spec: TrialSpec, fixture: str, unit: str, repo: str) -> list[str]:
    return ['sudo', '-n', 'systemd-run', '--quiet', '--wait', '--pipe', '--collect',
            '--service-type=exec', '--unit=' + unit, '--uid=' + str(os.getuid()),
            '-p', 'MemoryMax=64G', '-p', 'MemorySwapMax=0', '-p', 'IOAccounting=yes',
            '-p', 'KillMode=control-group', '-p', 'RuntimeMaxSec=150',
            '-p', 'WorkingDirectory=' + repo,
            '--', '/usr/bin/env', 'PYTHONPATH=' + repo, sys.executable, os.path.abspath(__file__),
            '_child', '--execute', '--fixture', fixture,
            '--spec', json.dumps(dataclasses.asdict(spec), separators=(',', ':'))]


def run_trial(spec: TrialSpec, fixture: str, repo: str) -> Record:
    unit = 'dg424-ceiling-' + uuid.uuid4().hex + '.service'
    command = service_command(spec, fixture, unit, repo)
    before = thermal_info()
    row: Record = {'status': 'FAIL', 'spec': dataclasses.asdict(spec)}
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=170)
        row.update({'service_stdout': completed.stdout, 'service_stderr': completed.stderr,
                    'service_returncode': completed.returncode})
        if completed.returncode:
            raise RuntimeError(f'Service failed: {completed.returncode}: {completed.stderr}')
        row.update(record(json.loads(completed.stdout)))
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        row.update({'status': 'FAIL', 'error': str(error)})
    finally:
        try:
            stopped = subprocess.run(['sudo', '-n', 'systemctl', 'stop', unit],
                                     capture_output=True, text=True, timeout=20)
            active = subprocess.run(['sudo', '-n', 'systemctl', 'is-active', unit],
                                    capture_output=True, text=True, timeout=20)
            row['cleanup'] = {'stop_returncode': stopped.returncode, 'active_returncode': active.returncode,
                              'active_stdout': active.stdout, 'active_stderr': active.stderr}
            if active.returncode not in (3, 4) or active.stdout.strip() not in ('inactive', 'failed', 'unknown', ''):
                raise RuntimeError('Owned fio service did not become inactive')
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            row.update({'status': 'FAIL', 'cleanup_error': str(error)})
    row.update({'service_unit': unit, 'service_command': command, 'thermal_before': before})
    try:
        after = thermal_info()
        row['thermal_after'] = after
        check_thermal(before, after)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        row.update({'status': 'FAIL', 'thermal_error': str(error)})
    return row


def accept_trial(spec: TrialSpec, args: argparse.Namespace, source: Record) -> Record:
    if source_info(args.repo_root, args.expected_source) != source:
        raise ValueError('Pinned source/fio changed')
    row = run_trial(spec, args.fixture, args.repo_root)
    row['source'] = source
    save(args.output + '/' + spec.name + '.json', row)
    if row.get('status') != 'PASS':
        raise RuntimeError('Trial failed; complete captured diagnostics saved before stopping the sweep')
    return row


def run_sweep(args: argparse.Namespace) -> None:
    if not args.execute or not args.expected_source:
        raise ValueError('Actual fio/CRC/services require --execute and --expected-source')
    os.sched_setaffinity(0, {0})
    source = source_info(args.repo_root, args.expected_source)
    info = fixture_info(args.fixture)
    host = host_info()
    if fsutil.exists(args.output):
        raise ValueError('Output must be a new directory; automatic resume is intentionally unsupported')
    fsutil.makedirs(args.output)
    # Setup reads are outside all timed windows; timed fio never verifies CRC.
    crc = subprocess.run(crc_command(args.fixture), capture_output=True, text=True, timeout=180)
    if crc.returncode:
        raise ValueError('Existing fixture CRC validation failed: ' + crc.stderr)
    crc_result = record(json.loads(crc.stdout))
    crc_jobs = crc_result['jobs']
    if not isinstance(crc_jobs, list) or len(crc_jobs) != 1:
        raise ValueError('Incomplete CRC fixture report')
    job = record(crc_jobs[0])
    if integer(job['error']) or integer(record(job['read'])['io_bytes']) != FIXTURE_BYTES:
        raise ValueError('CRC preflight did not validate the complete existing fixture')
    fixture = {'identity': info, 'sha256': digest(args.fixture), 'crc_command': crc_command(args.fixture),
               'crc_stdout': crc.stdout, 'crc_stderr': crc.stderr}
    if fixture['sha256'] != args.fixture_sha256:
        raise ValueError('Fixture SHA differs from the explicitly pinned existing fixture')
    if fixture_info(args.fixture) != info:
        raise ValueError('Fixture changed during preflight')
    save(args.output + '/plan.json', {'source': source, 'fixture': fixture, 'host': host,
                                      'base_trials': [dataclasses.asdict(spec) for spec in trial_plan()]})
    rows = []
    specs = trial_plan()
    for spec in specs:
        rows.append(accept_trial(spec, args, source))
    summaries = aggregate(rows)
    extensions = extension_blocks(summaries)
    save(args.output + '/adaptive-decision.json', {'summaries': summaries, 'extension_blocks': extensions})
    for repeat in range(5):
        for block in extensions:
            for jobs, submit, complete in ((8, 1, 1), (4, 16, 32)):
                spec = TrialSpec(block, jobs, 256, repeat, submit_batch=submit, complete_batch=complete)
                rows.append(accept_trial(spec, args, source))
    if fixture_info(args.fixture) != info or source_info(args.repo_root, args.expected_source) != source:
        raise ValueError('Fixture/source changed during sweep')
    summaries = aggregate(rows)
    result: Record = {
        'status': 'COMPLETED_READ_ONLY_SWEEP', 'trials': len(rows), 'source': source,
        'summaries': summaries, 'classification': classify_ceiling(summaries),
        'limits': ['O_DIRECT bypasses OS page cache, not controller caches',
                   '8GiB/4KiB/8KiB uniform-random fixture is not a model trace',
                   'Device-wide counters include unrelated host traffic',
                   'No per-request BPF observer was active during fio'],
    }
    save(args.output + '/summary.json', result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'run', '_child'))
    parser.add_argument('--fixture', default='/tmp/dg424-iops-overhead-owned-8g.bin')
    parser.add_argument('--fixture-sha256', default=FIXTURE_SHA256)
    parser.add_argument('--repo-root', default=os.getcwd())
    parser.add_argument('--output')
    parser.add_argument('--expected-source')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--spec')
    args = parser.parse_args()
    if args.mode == 'plan':
        print(json.dumps({'protocol': PROTOCOL, 'trials': [dataclasses.asdict(spec) for spec in trial_plan()]}, indent=2))
    elif args.mode == '_child':
        if not args.spec or not args.execute:
            raise ValueError('Internal child requires a service-provided spec')
        print(json.dumps(execute_child(TrialSpec(**json.loads(args.spec)), args.fixture), allow_nan=False))
    elif not args.output:
        parser.error('--output is required for run')
    else:
        run_sweep(args)


if __name__ == '__main__':
    main()
