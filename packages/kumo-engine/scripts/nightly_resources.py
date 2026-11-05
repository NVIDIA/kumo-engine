# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admit compact case requirements and account for an isolated case process."""

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

CGROUP_ROOT = Path('/sys/fs/cgroup')
MINIMUM_FIELDS = frozenset(('cpus', 'ram_bytes', 'scratch_bytes',
                            'gpu_memory_bytes', 'gpus', 'time_seconds'))
MEMORY_METHOD = 'cgroup-v2-memory.peak'


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def resource_requirements(measurement, headroom=1.25):
    """Derive proposed minima; retain full measurements outside runtime records."""
    if not isinstance(measurement, dict):
        raise ValueError('Resource proposals need a measured record')
    peaks = measurement.get('peaks', {})
    accounting = measurement.get('memory_accounting', {})
    if (not isinstance(peaks, dict) or not isinstance(accounting, dict)
            or measurement.get('format') != 'kumo-engine-nightly-resources-v1'
            or measurement.get('status') != 'MEASURED'
            or measurement.get('errors') != [] or not _positive(headroom)
            or headroom <= 1 or accounting.get('method') != MEMORY_METHOD
            or accounting.get('dedicated') is not True
            or not isinstance(accounting.get('path'), str)
            or not accounting['path']
            or not isinstance(measurement.get('gpu_class'), str)
            or not measurement['gpu_class']
            or type(measurement.get('gpus')) is not int
            or measurement['gpus'] < 1 or set(peaks)
            != {'cpu_cores', 'ram_bytes', 'scratch_bytes', 'gpu_memory_bytes'}
            or not all(_positive(value) for value in peaks.values())
            or not _positive(measurement.get('seconds'))
            or any(not isinstance(measurement.get(name), str)
                   or not re.fullmatch('[0-9a-f]{64}', measurement[name])
                   for name in ('wheel_sha256', 'input_manifest_sha256',
                                'harness_sha256'))):
        raise ValueError(
            'Resource proposals need complete dedicated-cgroup evidence')
    minimum = dict(
        cpus=math.ceil(peaks['cpu_cores'] * headroom),
        ram_bytes=math.ceil(peaks['ram_bytes'] * headroom),
        scratch_bytes=math.ceil(peaks['scratch_bytes'] * headroom),
        gpu_memory_bytes=math.ceil(peaks['gpu_memory_bytes'] * headroom),
        gpus=measurement['gpus'],
        time_seconds=math.ceil(measurement['seconds'] * headroom))
    return dict(minimum=minimum, measurement=measurement, headroom=headroom)


def admit_resources(minimum, capacity):
    """Check reviewed numeric minima without requiring archived provenance inline."""
    if (not isinstance(minimum, dict) or set(minimum) != MINIMUM_FIELDS or any(
            type(value) is not int or value <= 0
            for value in minimum.values())):
        raise ValueError('Provide all six positive numeric resource minima')
    for name, required in minimum.items():
        actual = capacity.get(name)
        unlimited_time = name == 'time_seconds' and actual == math.inf
        if (type(actual) not in (int, float)
                or not math.isfinite(actual) and not unlimited_time
                or actual < required):
            raise ValueError(
                f'Insufficient {name}: {actual!r}, requires {required!r}')
    return minimum


def _current_cgroup():
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            relative = PurePosixPath(line[3:])
            if not relative.is_absolute() or '..' in relative.parts:
                break
            return (CGROUP_ROOT /
                    str(relative).lstrip('/')).resolve(strict=True)
    raise ValueError('A visible cgroup-v2 membership is required')


def _scope_path(raw):
    path = Path(raw)
    if (not path.is_absolute() or path.resolve(strict=True) != path
            or not path.is_relative_to(CGROUP_ROOT.resolve(strict=True))
            or not re.fullmatch(r'kumo-nightly-[0-9a-f]{32}', path.name)):
        raise ValueError('Use the fresh owned nightly cgroup')
    for name in ('cgroup.procs', 'cgroup.events', 'memory.peak', 'cpu.stat'):
        item = path / name
        if item.is_symlink() or not item.is_file():
            raise ValueError(f'Dedicated cgroup counter is unavailable: {name}')
    return path


def _counter(path):
    value = path.read_text().strip()
    if not re.fullmatch(r'[0-9]+', value):
        raise ValueError(f'Invalid cgroup counter: {path.name}')
    return int(value)


def cgroup_memory_peak(scope):
    return _counter(_scope_path(scope) / 'memory.peak')


def _cpu_seconds(scope):
    fields = dict(line.split()
                  for line in (scope / 'cpu.stat').read_text().splitlines())
    if not re.fullmatch(r'[0-9]+', fields.get('usage_usec', '')):
        raise ValueError('Invalid cgroup CPU counter')
    return int(fields['usage_usec']) / 1000000


def _disk_bytes(output, scratch):
    size = 0
    for directory in (output, scratch):
        for path in directory.rglob('*'):
            try:
                if not path.is_symlink() and path.is_file():
                    size += path.stat().st_blocks * 512
            except FileNotFoundError:
                pass
    return size


def _checkpoint(output, result):
    temporary = output / '.resources.json.tmp'
    temporary.write_text(json.dumps(result, allow_nan=False) + '\n')
    temporary.replace(output / 'resources.json')


def finalize_resources(measurement, output, scratch, cgroup, seconds, *,
                       completed, error=None):
    """Retain final counters after worker exit, including interrupted observations."""
    scope = _scope_path(cgroup)
    if _current_cgroup() == scope:
        raise ValueError(
            'Finalize resources from the supervisor outside the case')
    output, scratch = Path(output).resolve(), Path(scratch).resolve()
    if output.is_relative_to(scratch) or scratch.is_relative_to(output):
        raise ValueError('Case scratch and evidence must have separate roots')
    recovery_error = None
    if measurement is None:
        try:
            measurement = json.loads((output / 'resources.json').read_text())
        except (OSError, ValueError) as read_error:
            recovery_error = f'Worker resource record unavailable: {read_error}'
    if not isinstance(measurement, dict):
        measurement = dict(
            format='kumo-engine-nightly-resources-v1', status='INCOMPLETE',
            errors=[], gpu_class=None, gpus=None,
            peaks=dict(cpu_cores=0., ram_bytes=0, scratch_bytes=0,
                       gpu_memory_bytes=0))
    result = measurement
    errors = result.setdefault('errors', [])
    if recovery_error:
        errors.append(recovery_error)
    if error is not None or not completed:
        errors.append(f'case: {error or "Worker did not complete"}')
    if result.get('status') != 'MEASURED':
        errors.append('Worker resource observation was not finalized')
    accounting = dict(method=MEMORY_METHOD, dedicated=True, path=str(scope))
    if result.get('memory_accounting', accounting) != accounting:
        errors.append('Worker resource accounting scope differs')
    result['memory_accounting'] = accounting
    peaks = result['peaks']

    def cpu():
        used, now = _cpu_seconds(scope), time.monotonic()
        previous = result.get('last_sample')
        if previous is not None:
            delta = used - previous['cpu_seconds']
            elapsed = now - previous['monotonic']
            if delta < 0 or elapsed < 0:
                raise ValueError('Final cgroup CPU accounting decreased')
            if elapsed > 0:
                peaks['cpu_cores'] = max(peaks['cpu_cores'], delta / elapsed)
        result['cpu_seconds'] = used
        result['last_sample'] = dict(cpu_seconds=used, monotonic=now)

    def ram():
        peaks['ram_bytes'] = max(peaks['ram_bytes'], cgroup_memory_peak(scope))

    def disk():
        peaks['scratch_bytes'] = max(peaks['scratch_bytes'],
                                     _disk_bytes(output, scratch))

    for name, observe in (('cpu', cpu), ('ram', ram), ('scratch', disk)):
        try:
            observe()
        except Exception as counter_error:
            errors.append(
                f'{name}: {type(counter_error).__name__}: {counter_error}')
    if _populated(scope):
        errors.append('Case descendants remain active during final accounting')
    if not _positive(seconds):
        errors.append('Complete case duration is invalid')
    result['seconds'] = seconds
    result['status'] = 'INCOMPLETE' if errors else 'MEASURED'
    try:
        _checkpoint(output, result)
        disk()
        _checkpoint(output, result)
    except Exception as write_error:
        errors.append(f'evidence: {type(write_error).__name__}: {write_error}')
        result['status'] = 'INCOMPLETE'
    return result


def _create_scope(root):
    scope = root / ('kumo-nightly-' + uuid.uuid4().hex)
    scope.mkdir(mode=0o700)
    return scope


def _remove_scope(scope):
    scope.rmdir()


def _populated(scope):
    fields = dict(line.split()
                  for line in (scope /
                               'cgroup.events').read_text().splitlines())
    if fields.get('populated') not in ('0', '1'):
        raise ValueError('Cgroup population status is unavailable')
    return fields['populated'] == '1'


@contextmanager
def dedicated_cgroup(root):
    """Own a fresh child scope; keep the caller outside it until final accounting."""
    root = Path(root)
    if (not root.is_absolute() or root.resolve(strict=True) != root
            or not root.is_relative_to(CGROUP_ROOT.resolve(strict=True))):
        raise ValueError('Provide a delegated cgroup-v2 directory')
    if 'memory' not in (root / 'cgroup.subtree_control').read_text().split():
        raise ValueError('The case caller needs delegated memory accounting')
    scope = _create_scope(root)
    try:
        _scope_path(scope)
        if (_current_cgroup() == scope or _populated(scope)
                or (scope / 'cgroup.procs').read_text().strip()
                or cgroup_memory_peak(scope) != 0):
            raise ValueError('A case needs a fresh empty accounting scope')
        yield scope
    finally:
        error = sys.exc_info()[1]
        try:
            if _populated(scope):
                (scope / 'cgroup.kill').write_text('1')
                deadline = time.monotonic() + 3
                while _populated(scope) and time.monotonic() < deadline:
                    time.sleep(.01)
                if _populated(scope):
                    raise RuntimeError('Case processes survived cgroup cleanup')
            _remove_scope(scope)
        except Exception as cleanup_error:
            if error is None:
                raise
            error.add_note(f'Cgroup cleanup failed: {cleanup_error}')


def join_cgroup(raw):
    """Join before Engine import; a second call only verifies existing membership."""
    scope = _scope_path(raw)
    if _current_cgroup() != scope:
        if ((scope / 'cgroup.procs').read_text().strip()
                or cgroup_memory_peak(scope) != 0 or _populated(scope)):
            raise ValueError('Join only a fresh empty nightly cgroup')
        (scope / 'cgroup.procs').write_text(str(os.getpid()))
    if (_current_cgroup() != scope or str(os.getpid())
            not in (scope / 'cgroup.procs').read_text().split()):
        raise ValueError('Case cgroup membership did not change')
    return scope


def _joined_command(command, scope):
    return [
        sys.executable, '-B',
        str(Path(__file__).resolve()), '--join-cgroup',
        str(scope), '--', *command
    ]


def run_with_deadline(command, cgroup, seconds, *, stdout=None, stderr=None,
                      env=None):
    """Join a dedicated scope, exec the worker, and hard-kill its group on timeout."""
    scope = _scope_path(cgroup)
    if not _positive(seconds) or not command or isinstance(
            command, (str, bytes)):
        raise ValueError('Provide a command and finite positive case deadline')
    if _current_cgroup() == scope:
        raise ValueError(
            'The case supervisor must remain outside its worker cgroup')
    command = [os.fspath(part) for part in command]
    environment = dict(os.environ if env is None else env)
    environment['NIGHTLY_CASE_CGROUP'] = str(scope)
    process = subprocess.Popen(_joined_command(command, scope), stdout=stdout,
                               stderr=stderr, env=environment,
                               start_new_session=True)
    try:
        code = process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                'Case process survived its hard deadline') from error
        raise subprocess.TimeoutExpired(command, seconds) from None
    return subprocess.CompletedProcess(command, code)


def _gpu_memory_info():
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError('Resource accounting requires one visible CUDA GPU')
    selected = os.environ.get('ONE_GPU_UUID')
    if selected:
        actual = 'GPU-' + str(
            torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
        if (actual != selected
                or os.environ.get('CUDA_VISIBLE_DEVICES') != selected):
            raise ValueError(
                'CUDA resource accounting differs from the selected GPU')
    free, total = torch.cuda.mem_get_info(0)
    if not 0 <= free <= total or total <= 0:
        raise ValueError('CUDA memory accounting is invalid')
    return free, total


def available_capacity(scratch, runtime, time_seconds):
    import psutil

    if not _positive(time_seconds):
        raise ValueError('Provide a finite positive case time budget')
    cpus = len(os.sched_getaffinity(0))
    memory = psutil.virtual_memory().available
    current, root = _current_cgroup(), CGROUP_ROOT.resolve(strict=True)
    while True:
        cpu = current / 'cpu.max'
        if cpu.is_file():
            quota, period = cpu.read_text().split()
            if quota != 'max':
                cpus = min(cpus, int(quota) / int(period))
        limit = current / 'memory.max'
        if limit.is_file() and (value := limit.read_text().strip()) != 'max':
            memory = min(
                memory, max(0,
                            int(value) - _counter(current / 'memory.current')))
        if current == root:
            break
        if not current.is_relative_to(root):
            raise ValueError('Cgroup capacity scope escaped its visible mount')
        current = current.parent
    import shutil
    return dict(cpus=cpus, ram_bytes=memory,
                scratch_bytes=shutil.disk_usage(scratch).free,
                gpu_memory_bytes=_gpu_memory_info()[0],
                gpus=runtime['gpu_count'], time_seconds=time_seconds)


@contextmanager
def track_resources(output, scratch, runtime, cgroup):
    """Observe a joined case; never substitute RSS for its cgroup RAM peak."""
    scope = _scope_path(cgroup)
    if _current_cgroup() != scope:
        raise ValueError('Observe only the worker in its dedicated cgroup')
    output, scratch = Path(output).resolve(), Path(scratch).resolve()
    if output.is_relative_to(scratch) or scratch.is_relative_to(output):
        raise ValueError('Case scratch and evidence must have separate roots')
    peaks = dict(cpu_cores=0., ram_bytes=0, scratch_bytes=0, gpu_memory_bytes=0)
    result = dict(
        format='kumo-engine-nightly-resources-v1', status='INCOMPLETE',
        peaks=peaks, errors=[], gpu_class=runtime['gpu'],
        gpus=runtime['gpu_count'], cpu_seconds=0., sample_interval_seconds=1,
        memory_accounting=dict(method=MEMORY_METHOD, dedicated=True,
                               path=str(scope)))
    done = threading.Event()
    started = previous_time = time.monotonic()
    previous_cpu = None

    def cpu():
        nonlocal previous_cpu, previous_time
        used = _cpu_seconds(scope)
        now = time.monotonic()
        if previous_cpu is not None:
            if used < previous_cpu:
                raise ValueError('Cgroup CPU accounting decreased')
            peaks['cpu_cores'] = max(peaks['cpu_cores'], (used - previous_cpu) /
                                     max(now - previous_time, 1e-6))
        result['cpu_seconds'] = used
        result['last_sample'] = dict(cpu_seconds=used, monotonic=now)
        previous_cpu, previous_time = used, now

    def ram():
        peaks['ram_bytes'] = max(peaks['ram_bytes'], cgroup_memory_peak(scope))

    def disk():
        peaks['scratch_bytes'] = max(peaks['scratch_bytes'],
                                     _disk_bytes(output, scratch))

    def gpu():
        free, total = _gpu_memory_info()
        peaks['gpu_memory_bytes'] = max(peaks['gpu_memory_bytes'], total - free)

    def sample():
        for name, observe in (('cpu', cpu), ('ram', ram), ('scratch', disk),
                              ('gpu', gpu)):
            try:
                observe()
            except Exception as error:
                message = f'{name}: {type(error).__name__}: {error}'
                if message not in result['errors']:
                    result['errors'].append(message)
        result['gpu_class'] = runtime['gpu']
        result['gpus'] = runtime['gpu_count']
        try:
            _checkpoint(output, result)
        except Exception as error:
            result['errors'].append(
                f'evidence: {type(error).__name__}: {error}')

    def observe():
        while not done.wait(1):
            sample()

    sample()
    worker = threading.Thread(target=observe, daemon=True)
    worker.start()
    try:
        yield result
    except BaseException as error:
        result['errors'].append(f'case: {type(error).__name__}: {error}')
        raise
    finally:
        done.set()
        worker.join()
        sample()
        result['seconds'] = time.monotonic() - started
        result['gpu_class'] = runtime['gpu']
        result['gpus'] = runtime['gpu_count']
        result['status'] = 'INCOMPLETE' if result['errors'] else 'MEASURED'
        try:
            _checkpoint(output, result)
        except Exception as error:
            result['errors'].append(
                f'evidence: {type(error).__name__}: {error}')
            result['status'] = 'INCOMPLETE'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--join-cgroup', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        parser.error('Provide the worker command')
    join_cgroup(arguments.join_cgroup)
    os.execvpe(command[0], command, os.environ)
